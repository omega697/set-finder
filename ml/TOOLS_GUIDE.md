# ML Tools & Pipeline Guide

This guide explains how to set up the ML environment, manage the dataset, and train models for the Set Finder app.

## 🛠️ Environment Setup

The ML pipeline requires Python 3.10+ and several dependencies (TensorFlow, OpenCV, Keras, PyTorch for SAM, etc.).

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
To bootstrap the dataset with high-quality labelled cards from external sources:
```bash
python tools/temp/fetch_external_data.py
```
This script downloads ~1,000 images, maps them to our naming convention, and saves them with an `ext_` prefix.

### Labeling New Data
We use Segment Anything (SAM) for efficient auto-labeling of video frames.

1.  **Auto-Labeling (Batch):**
    Processes every Nth frame of target videos and uses SAM to isolate cards, which are then verified by a classifier.
    ```bash
    python tools/bulk_label.py
    ```
    *Note: Outputs logs to `ml/tools/logs/bulk_label.log`.*

2.  **Interactive Labeling (Web UI):**
    Launches a Flask-based UI for manual sorting and verification of chips.
    ```bash
    python tools/labeler.py
    ```

## 🚀 Training Pipeline

### 1. Train Models
To train both the **Card Filter** and **Attribute Expert** models:
```bash
python tools/train.py --epochs 15 --oversample 4 --log tools/logs/train_v14.log
```
*Note: Models are saved to `ml/models/` by default.*

### 2. Convert to TFLite
The Android app requires models in `.tflite` format. Use the generalized conversion script:
```bash
python tools/convert_to_tflite.py --input ml/models/card_filter_v14.keras
python tools/convert_to_tflite.py --input ml/models/attribute_expert_v14.keras
```
This script automatically handles:
-   **Inference-only wrapping:** Bypasses training-only layers (Dropout, RandomRotation).
-   **Optimizations:** Enables standard DEFAULT optimizations for mobile.
-   **Naming:** Defaults to the same base name with a `.tflite` extension.

### 3. Verify Mapping
If model heads shift during retraining, verify the output indices before updating the Android app:
```bash
python tools/debug_tflite_mapping.py --model ml/models/attribute_expert_v14.tflite --images ../app/src/androidTest/assets/chips/cards/
```

## 🛠️ Core Tools (`ml/tools/`)

-   **`bulk_label.py`**: Headless batch auto-labeler using SAM.
-   **`train.py`**: Main training entry point for both models.
-   **`labeler.py`**: Web-based manual labeling interface.
-   **`convert_to_tflite.py`**: Generalized Keras to TFLite converter.
-   **`debug_tflite_mapping.py`**: Tool to disambiguate model output heads.
-   **`refiner.py`**: Library for sub-pixel quad refinement.
-   **`dataset_cleaner.py`**: Utility to prune low-confidence or duplicate data.

*Note: Ad-hoc debug scripts and one-off utilities are located in `ml/tools/temp/` and are ignored by Git.*

## 📁 Project Structure
-   `ml/models/`: Source `.keras` and exported `.tflite` models.
-   `ml/dataset/`: Sorted training data (verified cards and non-cards).
-   `ml/tools/logs/`: Standard output for long-running scripts (ignored).
-   `ml/tools/temp/`: Experimental or ad-hoc scripts (ignored).
