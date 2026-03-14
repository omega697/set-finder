import cv2
import numpy as np
import os
import argparse
from pathlib import Path

# Import shared standardized logic
from cv_library import unwarp, apply_white_balance, get_iou

class SetChipExtractor:
    def __init__(self, target_width=144, target_height=224):
        self.target_width = target_width
        self.target_height = target_height

    def find_candidates(self, img_rgb):
        """
        Finds candidate quadrilaterals using multi-scale thresholding.
        Two passes (small/large block sizes) ensure we catch both distant 
        cards (windowsill) and close-up cards.
        """
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        blur = cv2.GaussianBlur(l_channel, (5, 5), 0)

        all_candidates = []
        frame_area = img_rgb.shape[0] * img_rgb.shape[1]

        # Two passes: 31 for small/distant, 91 for large/close
        # C=1 is more sensitive than C=2
        for b_size in [31, 91]:
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b_size, 1)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < frame_area / 2000 or area > frame_area / 2: continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx): 
                    all_candidates.append(approx)

        # Deduplication using IoU
        all_candidates = sorted(all_candidates, key=cv2.contourArea, reverse=True)
        unique = []
        for cand in all_candidates:
            box = cv2.boundingRect(cand); is_dup = False
            for u in unique:
                if get_iou(box, cv2.boundingRect(u)) > 0.7: is_dup = True; break
            if not is_dup: unique.append(cand)
        return unique

        return unique

    def process_video(self, video_path, output_dir, interval_ms=500, start_frame=0):
        """
        Processes a video file and extracts card chips at regular intervals.
        Designed to work with pre-processed SDR videos (no local LUT needed).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            print(f"Error: Could not open {video_path}")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * (interval_ms / 1000.0))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"Resuming from frame {start_frame}...")

        frame_idx = start_frame
        saved_count = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                # Convert to RGB immediately for all logic
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                scale = 1000.0 / max(frame_rgb.shape[:2])
                small_rgb = cv2.resize(frame_rgb, (0, 0), fx=scale, fy=scale)
                candidates = self.find_candidates(small_rgb)
                
                print(f"Frame {frame_idx}: Found {len(candidates)} candidates")
                for i, quad in enumerate(candidates):
                    quad_orig = (quad / scale).astype(np.int32)
                    # Use shared unwarp and white balance (Expects RGB)
                    chip_rgb = unwarp(frame_rgb, quad_orig, self.target_width, self.target_height)
                    chip_rgb = apply_white_balance(chip_rgb)
                    
                    filename = f"chip_{Path(video_path).stem}_f{frame_idx:06d}_{i:02d}.jpg"
                    # Convert to BGR for final write
                    cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2BGR))
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
