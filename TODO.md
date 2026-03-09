# Set Finder - Project Status & TODO (March 8, 2026)

## 🛑 Current Blockers
- **Identification Accuracy:** Reverted to v12 models, but pattern identification is failing (e.g., SHADED seen as EMPTY).
- **Output Mapping:** TFLite models have generic output names (`StatefulPartitionedCall_1:0-3`), making manual mapping fragile.

## 🛠️ Refactoring & Modularity (NEW)
- **`SetDetector` Component:** Extracted stateless CV/ML logic from `SetAnalyzer` for isolated testing.
- **`FrameProcessor` Interface:** Abstracted OpenCV calls (`resize`, `rotate`) to allow Robolectric JVM testing without native libraries.
- **Dependency Injection:** `SetAnalyzer` now takes mocks, enabling 100% JVM verification of orchestration logic.
- **Bug Fixes:**
    - Fixed infinite recursion in `ImageProxy.toMat()`.
    - Fixed parameter order bug in `Utils.bitmapToMat`.
    - Restored critical coordinate scaling: Detection (1000px) -> Unwarping (Full-Res).

## 📋 Verified Components
- [x] **`CardUnwarper`:** Confirmed to produce 144x224 RGB chips with healthy brightness (~130.0).
- [x] **`SetSolver`:** Verified set-solving logic via Robolectric unit tests.
- [x] **`SetAnalyzer` (Orchestration):** Verified high-level logic (detect -> track -> solve) via Robolectric.

## 🚀 Model Status
- **v12 Revert:** Currently active, but accuracy is poor (1/3 sets found in integration test).
- **v13 Expert:** Highly accurate in Python, but failing in Android due to mapping/preprocessing nuances.
- **Preprocessing Findings:**
    - `chip_extractor.py` uses BGR for white balance.
    - Android uses RGB for everything else; parity requires careful conversion steps.

## 📋 Next Steps
1.  **Definitive v12 Mapping:** Fix the SHADED vs EMPTY swap by logging raw indices for all 19 test chips.
2.  **Restore Integration Pass:** Get `SetDetectorTest` (instrumentation) passing 3/3 sets on v12.
3.  **v13 Re-Integration:** Once v12 is stable, re-apply v13 expert model with verified BGR white-balance steps.
4.  **Named TFLite Outputs:** (Critical) Update training script to force consistent output names to stop the "index mapping" circle.

## 🗺️ Future Roadmap
- **ML Lifecycle Dashboard:** Web-based interface for video upload, labeling, bootstrapping, training, and conversion.
- **Set Statistics:** Track most frequent cards/sets in history.
