import cv2
import numpy as np
import os
import argparse
from pathlib import Path

def rectify(pts):
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 2), dtype=np.float32)
    add = pts.sum(1)
    new_pts[0] = pts[np.argmin(add)]
    new_pts[2] = pts[np.argmax(add)]
    diff = np.diff(pts, axis=1)
    new_pts[1] = pts[np.argmin(diff)]
    new_pts[3] = pts[np.argmax(diff)]
    return new_pts

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interWidth = max(0, xB - xA); interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

class SetChipExtractor:
    def __init__(self, target_width=144, target_height=224):
        self.target_width = target_width
        self.target_height = target_height

    def find_candidates(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        frame_area = img.shape[0] * img.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frame_area / 500 or area > frame_area / 2: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx): candidates.append(approx)
        candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
        unique = []
        for cand in candidates:
            box = cv2.boundingRect(cand); is_dup = False
            for u in unique:
                if get_iou(box, cv2.boundingRect(u)) > 0.7: is_dup = True; break
            if not is_dup: unique.append(cand)
        return unique

    def unwarp(self, img, contour):
        pts = contour.reshape(4, 2).astype(np.float32); rect = rectify(pts)
        (tl, tr, br, bl) = rect
        width = np.linalg.norm(tr - tl); height = np.linalg.norm(bl - tl)
        if width > height: rect = np.array([bl, tl, tr, br], dtype="float32")
        dst = np.array([[0, 0], [self.target_width - 1, 0], [self.target_width - 1, self.target_height - 1], [0, self.target_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (self.target_width, self.target_height))

    def apply_white_balance(self, img):
        # Mandatory white-balance normalization using LAB color space shifting
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # We assume the background (which is most of the chip) should be neutral gray/white
        # Shift a and b channels to center them at 128
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        a = cv2.add(a, int(128 - a_mean))
        b = cv2.add(b, int(128 - b_mean))
        
        balanced = cv2.merge([l, a, b])
        return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

    def process_video(self, video_path, output_dir, interval_ms=500, start_frame=0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error: Could not open {video_path}"); return
        os.makedirs(output_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * (interval_ms / 1000.0))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"Resuming from frame {start_frame}...")

        frame_idx = start_frame
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % frame_interval == 0:
                scale = 1000.0 / max(frame.shape[:2])
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                candidates = self.find_candidates(small)
                print(f"Frame {frame_idx}: Found {len(candidates)} candidates")
                for i, quad in enumerate(candidates):
                    quad_orig = (quad / scale).astype(np.int32)
                    chip = self.unwarp(frame, quad_orig)
                    chip = self.apply_white_balance(chip)
                    # Unique filename using frame index and sub-index
                    filename = f"chip_{Path(video_path).stem}_f{frame_idx:06d}_{i:02d}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), chip)
                    saved_count += 1
            frame_idx += 1
        cap.release()
        print(f"Finished. Saved {saved_count} chips to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="chips")
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    extractor = SetChipExtractor()
    extractor.process_video(args.input, args.output, args.interval, args.start)
