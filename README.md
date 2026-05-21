# Attendance System Using Face Recognition

This project implements a face-recognition-based attendance workflow using OpenCV, HOG features, and multiple classifiers (LBPH, SVM, XGBoost), with a stacked voting strategy for final identity prediction.

## Features

- Face detection and preprocessing pipeline
- Data cleaning notebook to crop/prepare training faces
- Data augmentation and feature extraction with HOG
- Multiple recognition models:
  - LBPH (OpenCV)
  - SVM (scikit-learn)
  - XGBoost
- Stacked recognizer (`StackedFaceRecognizer`) that combines model outputs
- Testing notebook for model evaluation metrics
- Attendance notebook with interactive image upload UI

## Repository Structure

```text
Dataset/
  training/
    Training_RAW/        # Raw training images
    Cleaned_Training/    # Cleaned face crops used for training
  Snapshots/             # Optional test/input snapshots
Notebooks/
  Data_Cleaning.ipynb    # Cleans raw images into cropped faces
  Training.ipynb         # Trains LBPH, SVM, and XGBoost models
  Testing.ipynb          # Evaluates model performance
  Attendance.ipynb       # Runs attendance prediction workflow
  utils.py               # Shared utilities and stacked recognizer logic
README.md
```

## How It Works

1. **Clean dataset** using `Notebooks/Data_Cleaning.ipynb`
2. **Train models** using `Notebooks/Training.ipynb`
3. **Evaluate models** using `Notebooks/Testing.ipynb`
4. **Run recognition/attendance flow** using `Notebooks/Attendance.ipynb`

## Requirements

Use Python 3.10+ (3.11 recommended) and install:

- `numpy`
- `opencv-contrib-python` (required for `cv2.face.LBPHFaceRecognizer_create`)
- `scikit-learn`
- `scikit-image`
- `xgboost`
- `joblib`
- `pillow`
- `matplotlib`
- `seaborn`
- `ipywidgets`
- `jupyter`

Install with:

```bash
pip install numpy opencv-contrib-python scikit-learn scikit-image xgboost joblib pillow matplotlib seaborn ipywidgets jupyter
```

## Dataset Naming Convention

Training files are expected to follow:

```text
user.<ID>.<index>.<ext>
```

Example: `user.3.14.jpg`

The code extracts the numeric person ID from the filename.

## Model Artifacts

Training notebooks save model files under `models/` (create this folder if missing):

- `models/trainer.yml` (LBPH)
- `models/svm_face_model.pkl` (SVM)
- `models/xgb_face_model.pkl` (XGBoost)
- `models/label_encoder.pkl` (Label encoder for XGBoost)

## Notes

- Some notebook paths are currently Windows-style (`..\\Dataset\\...`). If you run on Linux/macOS, adjust paths to Unix-style (`../Dataset/...`) where needed.
- For attendance display names, update the `id_to_name` mapping in `Attendance.ipynb` to match your dataset IDs.
