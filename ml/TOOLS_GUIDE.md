# ML Tools & Pipeline Guide

This guide explains how to set up the ML environment, manage the dataset, and train models for the Set Finder app.

## 🛠️ Environment Setup

The ML pipeline requires Python 3.10+ and several dependencies (TensorFlow, OpenCV, Keras, etc.).

1.  **Create Virtual Environment:**
    ```bash
    cd ml
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset Management

The project uses a combination of internal (Git-tracked) and external (Git-ignored) data.

### Fetching External Data
To bootstrap the dataset with high-quality labelled cards from the [tomwhite/set-game](https://github.com/tomwhite/set-game) repository:
```bash
python tools/fetch_external_data.py
```
This script downloads ~1,000 images, maps them to our naming convention, and saves them with an `ext_` prefix. These files are automatically ignored by Git.

### Labeling New Data
If you have new photos or videos of cards, follow this workflow:

1.  **Extract Chips:**
    Use the extractor to find quads in raw videos and save them as warped 144x224 chips.
    ```bash
    # Extract from a video file
    # --input: path to video
    # --output: target directory (defaults to 'chips')
    # --interval: process one frame every N frames (defaults to 500)
    python tools/chip_extractor.py --input raw_data/video.mp4 --output raw_data/chips/ --interval 100
    ```

2.  **Label Chips:**
    Launch the custom UI to sort chips into the `dataset/` directory.
    ```bash
    python tools/labeler.py
    ```

3.  **Bootstrapping (Optional):**
    If you already have a trained model, you can "rescue" unlabelled chips by moving them to the `predictions/` directory based on model traits for faster verification:
    ```bash
    python tools/rescue_pile.py --source raw_data/chips/ --filter_model card_filter_v13.keras --expert_model attribute_expert_v13.keras --predictions predictions/
    ```

## 🚀 Training Pipeline

### 1. Train Models
To train both the **Card Filter** (Is it a card?) and **Attribute Expert** (What are its traits?) models:
```bash
python tools/train.py --epochs 10 --filter-model card_filter_v13.keras --expert-model attribute_expert_v13.keras
```
*Note: Training uses squiggle-safe augmentation (rotation only, no flipping) and includes brightness/contrast variations for lighting robustness.*

### 2. Convert to TFLite
The Android app requires models in `.tflite` format. Our converter uses a specialized process to ensure models are reliable and easy to use:

*   **Stripping "Training Wheels":** The script creates an inference-only wrapper (`training=False`). This bypasses layers like `RandomRotation` and `Dropout` that are only needed during training, which prevents conversion errors and runtime crashes in the app.
*   **Stable Output Order:** The converter preserves the order of outputs defined in the Keras model. For the Attribute Expert, this order is: 
    - **Index 0:** Count
    - **Index 1:** Shape
    - **Index 2:** Color
    - **Index 3:** Pattern
*   **Android Mapping:** The `CardModelMapper` class in the Android app is configured to match this stable order.

```bash
python tools/convert_to_tflite.py --input card_filter_v13.keras --output card_filter_v13.tflite
python tools/convert_to_tflite.py --input attribute_expert_v13.keras --output attribute_expert_v13.tflite
```

## 🛠️ Utility Tools

- **`rescue_pile.py`**: Moves unlabelled chips from `raw_data` into the `dataset` structure based on model predictions (useful for bootstrapping).
- **`chip_extractor.py`**: Extracts card chips from video files or folders of images.
- **`verify_tflite.py`**: Validates that a `.tflite` model produces expected outputs for a given sample image.

## 📁 Ignored Files
The following are excluded from Git but can be recreated using the steps above:
- `ml/venv/`: Recreate via `venv` and `requirements.txt`.
- `ml/*.keras`, `ml/*.tflite`: Recreate via `train.py` and `convert_to_tflite.py`.
- `ml/dataset/**/ext_*`: Recreate via `fetch_external_data.py`.
- `ml/predictions/`, `ml/raw_data/`: Intermediate pipeline artifacts.
