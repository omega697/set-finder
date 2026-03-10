# Set Finder - Project Status & TODO (March 10, 2026)

## 🛠️ Pipeline Progress (Current State)

### Stage 1: Quad Finding (Candidate Generation)
- [x] **High-Precision Implementation:** Combined `RETR_EXTERNAL` boundaries with margin-based `isWhiteCard` validation.
- [x] **Geometric Precision:** Implemented deduplication using polygon IoU via `intersectConvexConvex`.
- [x] **Centralized Ground Truth:** Created regression suite for 15 scenes in `QuadFindingGroundTruth.kt`.
- [!] **Status:** **63.93% baseline recall** (spatial). Optimization needed for tight shots and edge-touching cards.

### Stage 2: Chip Extraction (Normalization)
- [x] **Verified Parity:** 100% match with Python `chip_extractor.py` (Histogram Correlation > 0.99).
- [x] **Perspective Correction:** Produces standard 200x300 RGB chips for identification.
- [x] **Integrated:** Linked to `CardFinder` metadata for strategy auditing.

### Stage 3: Card Filtering & Identification
- [x] **100% Accuracy:** Achieved perfect verification on 162-card identification suite.
- [x] **Stable Output Mapping:** Implemented `CardModelMapper` to own stable trait indices, eliminating fragility.
- [x] **Zero-Crash Conversion:** Developed robust TFLite export script with `training=False` wrapper.

## 📋 Next Major Focus (Upcoming)
1.  **Stage 1 Optimization:** Improve quad finder recall to handle cards touching frame edges.
2.  **v14 Model Training:** Currently training (15 epochs) with optimized 'squiggle-safe' augmentation and OpenCV-based white balancing.
3.  **ML Lifecycle Dashboard:** Automated interface for video upload, labeling, bootstrapping, training, and conversion.

## 🗺️ Future Roadmap
- **Stage 1 Polish:** Finalize multi-context block size tuning to resolve the remaining windowsill bottleneck.
- **ML Lifecycle Dashboard:** Web-based interface for video upload, labeling, bootstrapping, training, and conversion.
- **Set Statistics:** Track most frequent cards and sets in user history.
- **Real-Time Performance:** Optimize Stage 1 for 60FPS tracking on mid-range hardware.
