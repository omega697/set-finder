import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from segment_anything import sam_model_registry, SamPredictor
import time
from pathlib import Path
import sys
import argparse

# Import standardized CV logic
from cv_library import unwarp, apply_white_balance, get_hable_lut
from model_downloader import ensure_checkpoint

"""
BULK LABELER
------------
Processes video files to generate training data for Stage 1 (Quad Finding).
Uses SAM to auto-detect potential cards and verifies them with the Card Filter.
Standardized to RGB after applying the Hable tonemap LUT for App parity.
"""

def main():
    parser = argparse.ArgumentParser(description="Bulk label video frames using SAM and MobileNetV2.")
    parser.add_argument("--checkpoint", default="ml/tools/temp/sam_vit_b_01ec64.pth", help="Path to SAM checkpoint.")
    parser.add_argument("--model_type", default="vit_b", help="SAM model type.")
    parser.add_argument("--device", default="cpu", help="Device to run SAM on (cpu or cuda).")
    parser.add_argument("--video_dir", default="ml/raw_data/quad_finding_training", help="Directory containing videos.")
    parser.add_argument("--video", help="Process a single video file instead of a directory.")
    parser.add_argument("--filter_model", default="ml/models/card_filter_v14.keras", help="Path to card filter model.")
    parser.add_argument("--output_dir", default="ml/dataset/yolo_pose", help="Directory to save labels.")
    parser.add_argument("--debug_dir", default="ml/tools/debug_output", help="Directory to save debug images.")
    parser.add_argument("--log_dir", default="ml/tools/logs", help="Directory to save logs.")
    parser.add_argument("--save_debug", action="store_true", help="Save debug images with quad overlays.")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold for accepting cards.")
    parser.add_argument("--step", type=int, default=10, help="Process every Nth frame.")
    parser.add_argument("--max_frames", type=int, help="Limit number of frames per video.")
    parser.add_argument("--log_to_file", action="store_true", help="Redirect output to a log file.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    if args.log_to_file:
        log_file = os.path.join(args.log_dir, "bulk_label.log")
        print(f"Logging to {log_file}")
        f = open(log_file, 'w', buffering=1)
        sys.stdout = f
        sys.stderr = f

    if args.video:
        VIDEOS = [Path(args.video)]
    else:
        VIDEOS = sorted(list(Path(args.video_dir).glob("*.mp4")))
    
    if not VIDEOS:
        print(f"No videos found.")
        sys.exit(0)

    # --- MODELS ---
    print("Initializing SAM Predictor...")
    checkpoint_path = ensure_checkpoint(args.checkpoint, args.model_type)
    sam = sam_model_registry[args.model_type](checkpoint=checkpoint_path)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    print("Loading Card Filter...")
    card_filter = tf.keras.models.load_model(args.filter_model)
    
    # --- LUT ---
    hable_lut = get_hable_lut()

    def fit_outer_quad(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt).reshape(-1, 2).astype(float)
        
        def intersect_lines(p1, p2, p3, p4):
            x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
            denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
            if denom == 0: return None
            ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
            return np.array([x1 + ua*(x2-x1), y1 + ua*(y2-y1)])

        while len(hull) > 4:
            best_idx = -1; min_added_area = float('inf'); best_new_pt = None
            for i in range(len(hull)):
                p_m1, p_0, p_p1, p_p2 = hull[i-1], hull[i], hull[(i+1)%len(hull)], hull[(i+2)%len(hull)]
                new_pt = intersect_lines(p_m1, p_0, p_p1, p_p2)
                if new_pt is not None:
                    area = 0.5 * abs(p_0[0]*(p_p1[1]-new_pt[1]) + p_p1[0]*(new_pt[1]-p_0[1]) + new_pt[0]*(p_0[1]-p_p1[1]))
                    if area < min_added_area:
                        min_added_area = area; best_idx = i; best_new_pt = new_pt
            if best_idx == -1: break
            next_idx = (best_idx + 1) % len(hull)
            if next_idx > best_idx:
                hull = np.delete(hull, [best_idx, next_idx], axis=0); hull = np.insert(hull, best_idx, best_new_pt, axis=0)
            else:
                hull = np.delete(hull, [best_idx, next_idx], axis=0); hull = np.append(hull, [best_new_pt], axis=0)
        return hull.astype(int)

    def get_mask_metrics(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        if intersection == 0: return 0, 0
        union = np.logical_or(mask1, mask2).sum()
        area1 = np.sum(mask1)
        return (intersection / union), (intersection / area1)

    def fit_area_regression(candidates):
        high_conf = [c for c in candidates if c['prob'] > 0.85]
        if len(high_conf) < 3: return None
        median_area = np.median([c['area'] for c in high_conf])
        anchors = [c for c in high_conf if 0.4 * median_area < c['area'] < 2.5 * median_area]
        kept_anchors = []
        for a in sorted(anchors, key=lambda x: x['prob'], reverse=True):
            center = np.mean(a['quad'], axis=0)
            if not any(np.linalg.norm(center - np.mean(ka['quad'], axis=0)) < 100 for ka in kept_anchors):
                kept_anchors.append(a)
        if len(kept_anchors) < 3: return None
        X_reg = np.array([[np.mean(a['quad'], axis=0)[0], np.mean(a['quad'], axis=0)[1], 1.0] for a in kept_anchors])
        y_reg = np.array([a['area'] for a in kept_anchors])
        beta, residuals, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
        y_mean = np.mean(y_reg)
        ss_tot = np.sum((y_reg - y_mean)**2)
        ss_res = residuals[0] if len(residuals) > 0 else np.sum((np.dot(X_reg, beta) - y_reg)**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return {'beta': beta, 'r2': r2, 'median': median_area}

    def auto_detect(frame_rgb):
        h, w = frame_rgb.shape[:2]
        predictor.set_image(frame_rgb) # SAM expects RGB
        margin = 10 
        grid_x = np.linspace(margin, w - margin, 24).astype(int)
        grid_y = np.linspace(margin, h - margin, 16).astype(int)
        grid_points = [(gx, gy) for gy in grid_y for gx in grid_x]
        raw_candidates = [] 
        for gx, gy in grid_points:
            masks, _, _ = predictor.predict(point_coords=np.array([[gx, gy]]), point_labels=np.array([1]), multimask_output=True)
            for mask in masks:
                area = np.sum(mask)
                if not (0.002 * h * w < area < 0.15 * h * w): continue
                quad = fit_outer_quad(mask.astype(np.uint8)*255)
                if quad is not None:
                    # unwarp and apply_white_balance handle RGB
                    chip_rgb = unwarp(frame_rgb, quad)
                    chip_balanced_rgb = apply_white_balance(chip_rgb)
                    # MobileNetV2 preprocessing still expects RGB
                    crop_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(cv2.resize(chip_balanced_rgb, (224, 224)).astype(np.float32))
                    prob = card_filter.predict(np.expand_dims(crop_preprocessed, 0), verbose=0)[0][0]
                    if prob > 0.005:
                        raw_candidates.append({'prob': prob, 'mask': mask, 'quad': quad.tolist(), 'area': area, 'chip': chip_balanced_rgb})

        model = fit_area_regression(raw_candidates)
        use_geom = model is not None and model['r2'] > 0.6
        for cand in raw_candidates:
            if use_geom:
                center = np.mean(cand['quad'], axis=0)
                beta = model['beta']; expected = center[0] * beta[0] + center[1] * beta[1] + beta[2]
                cand['residual'] = abs(cand['area'] - expected) / max(1, expected)
            elif model:
                cand['residual'] = abs(cand['area'] - model['median']) / max(1, model['median'])
            else:
                cand['residual'] = 0.0

        candidates = sorted(raw_candidates, key=lambda x: (x['prob'] > args.threshold, x['prob'] > 0.98, x['prob'] > 0.85, -x['residual'], -x['area'], x['prob']), reverse=True)
        found_quads, rejected_quads, final_masks = [], [], []
        for cand in candidates:
            if use_geom and cand['residual'] > 0.40 and cand['prob'] > args.threshold:
                rejected_quads.append((cand['quad'], cand['prob'], cand.get('chip')))
                continue
            if any(get_mask_metrics(cand['mask'], fm)[0] > 0.1 or get_mask_metrics(cand['mask'], fm)[1] > 0.1 for fm in final_masks):
                continue
            if cand['prob'] > args.threshold:
                found_quads.append(cand['quad']); final_masks.append(cand['mask'])
            else:
                rejected_quads.append((cand['quad'], cand['prob'], cand.get('chip')))
        return found_quads, rejected_quads, grid_points

    def save_labels(frame_rgb, quads, filename):
        # Convert back to BGR for imwrite
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_path = os.path.join(args.output_dir, filename); cv2.imwrite(img_path, frame_bgr)
        label_path = os.path.join(args.output_dir, filename.replace('.jpg', '.txt'))
        h, w = frame_rgb.shape[:2]
        with open(label_path, 'w') as f_out:
            for q in quads:
                q_arr = np.array(q)
                x_min, y_min = np.min(q_arr, axis=0); x_max, y_max = np.max(q_arr, axis=0)
                bw, bh = (x_max - x_min), (y_max - y_min); cx, cy = x_min + bw/2, y_min + bh/2
                pts_str = " ".join([f"{pt[0]/w:.6f} {pt[1]/h:.6f} 2" for pt in q])
                f_out.write(f"0 {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f} {pts_str}\n")

    def save_debug_frame(frame_rgb, found, rejected, grid_points, filename):
        # Standardize for drawing
        canvas_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        chip_dir = os.path.join(args.debug_dir, "chips", filename.replace(".jpg", ""))
        os.makedirs(chip_dir, exist_ok=True)
        for i, (quad, prob, chip_rgb) in enumerate(rejected):
            if chip_rgb is not None:
                cv2.imwrite(os.path.join(chip_dir, f"rej_prob_{prob:.4f}_idx_{i}.jpg"), cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2BGR))
        for gx, gy in grid_points: cv2.circle(canvas_bgr, (gx, gy), 2, (255, 0, 0), -1)
        for quad, prob, _ in rejected:
            pts = np.array(quad, np.int32).reshape((-1, 1, 2)); cv2.polylines(canvas_bgr, [pts], True, (0, 0, 255), 2)
            cv2.putText(canvas_bgr, f"{prob:.2f}", (quad[0][0], quad[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
        for quad in found:
            pts = np.array(quad, np.int32).reshape((-1, 1, 2)); cv2.polylines(canvas_bgr, [pts], True, (0, 255, 0), 4)
        cv2.imwrite(os.path.join(args.debug_dir, filename), canvas_bgr)

    for video_path in VIDEOS:
        print(f"Processing video: {str(video_path)}")
        cap = cv2.VideoCapture(str(video_path)); video_name = video_path.stem; frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret: break
            if args.max_frames is not None and frame_idx >= args.max_frames: break
            if frame_idx % args.step == 0:
                print(f"  Frame {frame_idx}...")
                # 1. Normalize HDR (BGR)
                frame_fixed_bgr = cv2.LUT(frame_bgr, hable_lut)
                # 2. Standardize RGB for Logic
                frame_rgb = cv2.cvtColor(frame_fixed_bgr, cv2.COLOR_BGR2RGB)
                
                found, rejected, grid_points = auto_detect(frame_rgb)
                if found or rejected:
                    filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                    if found and not args.save_debug: save_labels(frame_rgb, found, filename)
                    if args.save_debug: save_debug_frame(frame_rgb, found, rejected, grid_points, filename)
                    print(f"    Found {len(found)} cards, rejected {len(rejected)} candidates.")
            frame_idx += 1
        cap.release()
    print("Bulk processing complete!")

if __name__ == "__main__":
    main()
