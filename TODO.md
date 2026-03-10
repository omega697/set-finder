# Set Finder - Project Status & TODO (March 10, 2026)

## 🛠️ Pipeline Progress (Current State)

### Stage 1: Quad Finding (Candidate Generation)
- [x] **High-Precision Implementation:** Combined `RETR_EXTERNAL` boundaries with margin-based `isWhiteCard` validation.
- [x] **Geometric Precision:** Implemented deduplication using polygon IoU via `intersectConvexConvex`.
- [x] **Centralized Ground Truth:** Created regression suite for 15 scenes in `QuadFindingGroundTruth.kt`.
- [!] **Status:** ~70.5% baseline recall. Functional but needs further work on both **recall** (capturing high-clutter scenes like the windowsill) and **accuracy/precision** (minimizing false positives on non-card surfaces).

### Stage 2: Chip Extraction (Normalization)
- [x] **Verified Parity:** 100% match with Python `chip_extractor.py` (Histogram Correlation > 0.99).
- [x] **Perspective Correction:** Produces standard 200x300 RGB chips for identification.
- [x] **Integrated:** Linked to `CardFinder` metadata for strategy auditing.

### Stage 3: Card Filtering & Identification
- [x] **Verified Accuracy:** >98% accuracy on standardized card chips.
- [x] **Functional Terminology:** Logic updated to use `empty`, `shaded`, `solid`.
- [x] **Robust Attributes:** Extraction of Color, Shape, Number, and Filling is verified.

## 📋 Next Major Focus (Upcoming)
1.  **ML Model Improvement:** Refine the training dataset and architecture to handle diverse lighting and "off-white" card conditions.
2.  **TFLite Conversion:** Optimize the conversion pipeline to ensure high fidelity and performance on mobile hardware.
3.  **Output Mapping:** Finalize named TFLite outputs to eliminate fragile index-based mapping.

## 🗺️ Future Roadmap
- **Stage 1 Polish:** Finalize multi-context block size tuning to resolve the remaining windowsill bottleneck.
- **ML Lifecycle Dashboard:** Web-based interface for video upload, labeling, bootstrapping, training, and conversion.
- **Set Statistics:** Track most frequent cards and sets in user history.
- **Real-Time Performance:** Optimize Stage 1 for 60FPS tracking on mid-range hardware.
