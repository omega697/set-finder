# Set Finder - Project Status & TODO (March 8, 2026)

## 🛑 Current Blockers
- **None:** UI Scaling/Offset and Overlapping bugs fixed.

## 🚀 Model Status
- **v13 Training:** In progress (Background PID 2768608). 
    - *Progress:* Card Filter Epoch 1/5. 
    - *Fix:* Resolved multi-output metric mismatch in `train.py`.
- **v12 Alignment:**
    - **TFLite Mapping:** Fixed v12 indices (Col=0, Shp=1, Cnt=2, Pat=3).
    - **CardFinder Alignment:** Removed aspect-ratio filter to match `chip_extractor.py`.

## 🛠️ Implemented Features
- **Scanner UI Overhaul:**
    - Controls moved to bottom (Buttons + Sensitivity Slider).
    - Camera view moved to top with `FIT_CENTER` scaling.
    - Added Label Visibility toggle.
    - Added App Icon Logo to Welcome Screen.
- **Persistence & History:**
    - `SettingsManager`: Persisted highlight colors, sensitivity, and label settings.
    - `HistoryPersistence`: Saved unique found sets with card images (latest 50).
    - `HistoryScreen`: New screen to browse found sets.
- **UI/Theme Improvements:**
    - Full Dark Mode support with refined color palettes.
    - Material 2 theme alignment across all screens.
    - Drag-and-drop reordering for highlight colors in Settings.
- **CV Refinement:**
    - Relaxed max area constraint in `CardFinder` to 80% of frame for close-ups.
    - Fixed overlapping card detections using IoU and containment checks.

## 📋 Next Steps
1.  **Verify Pipeline:** Run `PipelineAlignmentTest` to confirm `FIT_CENTER` mapping and detection accuracy.
2.  **v13 Deployment:** Once v13 training finishes, convert to TFLite and verify dynamic name-based mapping in Android.
3.  **UI Testing:** Verify drag-and-drop reordering on device and ensure history saving doesn't impact FPS.

## 🗺️ Future Roadmap
- **Full-Featured Web App for ML Pipeline:**
    - Expand `labeler.py` into a comprehensive end-to-end dashboard.
    - Features: Video upload, chip extraction, manual/bulk labeling, bootstrapping, card rescue, training (Keras), and TFLite conversion.
    - Goal: Allow a non-technical user to manage the entire ML lifecycle from a browser.
