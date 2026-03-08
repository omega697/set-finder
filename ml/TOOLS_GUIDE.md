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
Use the custom UI to label chips extracted from your own camera frames:
```bash
python tools/labeler.py
```

## 🚀 Training Pipeline

### 1. Train Models
To train both the **Card Filter** (Is it a card?) and **Attribute Expert** (What are its traits?) models:
```bash
python tools/train.py --epochs 10 --filter-model card_filter_v13.keras --expert-model attribute_expert_v13.keras
```
*Note: Training will automatically use available GPUs.*

### 2. Convert to TFLite
The Android app requires models in `.tflite` format:
```bash
python tools/convert_to_tflite.py --keras-model card_filter_v13.keras --tflite-model card_filter_v13.tflite
python tools/convert_to_tflite.py --keras-model attribute_expert_v13.keras --tflite-model attribute_expert_v13.tflite
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
