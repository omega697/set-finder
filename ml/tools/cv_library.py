import cv2
import numpy as np

"""
CV LIBRARY - Centralized Image Pipeline (Honest SDR)
---------------------------------------------------
Logic for unwarping and color balancing. Designed to work 
with standard SDR frames (either from the phone's ISP or 
pre-processed SDR videos).

Standard Flow:
1. Load (RGB/BGR)
2. Logic (Unwarp, White Balance, Deduplicate)
3. Save (BGR -> imwrite)
"""

def rectify(pts):
    """
    RADIAL SORTING + TL ANCHOR.
    Robustly orders corners as [TL, TR, BR, BL] regardless of rotation.
    Matches Android ChipUnwarper.kt.
    """
    pts = pts.reshape((4, 2))
    # 1. Radial Sort (Clockwise)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts_clockwise = pts[np.argsort(angles)]
    
    # 2. Anchor Top-Left (Min sum of X+Y)
    sums = pts_clockwise[:, 0] + pts_clockwise[:, 1]
    tl_idx = np.argmin(sums)
    
    # 3. Shift array so TL is index 0
    return np.roll(pts_clockwise, -tl_idx, axis=0)

def unwarp(img_rgb, contour, target_width=144, target_height=224):
    """
    Extracts and straightens a card. Input/Output is RGB.
    Matches Android ChipUnwarper.kt parity.
    """
    pts = contour.reshape(4, 2).astype(np.float32)
    rect = rectify(pts)
    tl, tr, br, bl = rect
    
    # Rotation check using aspect ratio
    width_a = np.linalg.norm(tr - tl); width_b = np.linalg.norm(br - bl)
    height_a = np.linalg.norm(tl - bl); height_b = np.linalg.norm(tr - br)
    avg_w = (width_a + width_b) / 2.0
    avg_h = (height_a + height_b) / 2.0

    # If width > height, rotate SOURCE mapping to keep output Portrait
    if avg_w > avg_h:
        # Rotate 90 deg clockwise
        src = np.array([bl, tl, tr, br], dtype="float32")
    else:
        src = rect

    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_rgb, M, (target_width, target_height))

def apply_white_balance(img_rgb):
    """
    Standardizes card colors using LAB color space shifting.
    Matches Android OpenCVWhiteBalancer.kt.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    a_median = np.median(a)
    b_median = np.median(b)
    a_new = np.clip(a.astype(np.float16) + (128 - a_median), 0, 255).astype(np.uint8)
    b_new = np.clip(b.astype(np.float16) + (128 - b_median), 0, 255).astype(np.uint8)
    balanced = cv2.merge([l, a_new, b_new])
    return cv2.cvtColor(balanced, cv2.COLOR_LAB2RGB)

def get_iou(boxA, boxB):
    """Intersection over Union for bounding box deduplication."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interWidth = max(0, xB - xA); interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)
