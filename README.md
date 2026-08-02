# Attendance System Using Face Recognition

Welcome to the **Attendance System Using Face Recognition** project. This repository contains the code and resources for a robust and high-accuracy face recognition-based attendance system, utilizing a stacked ensemble of machine learning models to effectively identify faces in real-time.

## 📊 Dataset Details

The initial dataset for this project was custom-collected from **6 friends**. The raw images are stored in the `Dataset/` directory. Each individual's facial images are grouped and labeled to train the face recognition models, ensuring that the system can reliably distinguish between these 6 specific classes (people) under various conditions.

## 🧠 Approach & Models

To achieve high accuracy and robustness, this project doesn't rely on a single model. Instead, it uses a **Stacked Face Recognizer** that combines three distinct approaches:
1. **LBPH (Local Binary Patterns Histograms)**: Provided by OpenCV, it looks at local features of the raw images.
2. **SVM (Support Vector Machine)**: Trained on extracted HOG (Histogram of Oriented Gradients) features to find the optimal hyperplane separating different faces.
3. **XGBoost**: A powerful gradient boosting classifier also trained on HOG features for highly accurate probabilistic predictions.

A custom consensus algorithm (Tournament/Voting logic) combines the predictions of these models to minimize false positives and confirm identity, falling back to a Restricted KNN tie-breaker if needed.

---

## 📁 Project Structure

The main logic resides in the `Notebooks/` directory which separates the different stages of the pipeline:

```text
Attendance-System-Using-Face-Recognition/
├── Dataset/                   # Contains the raw face images from 6 friends
├── Notebooks/
│   ├── Data_Cleaning.ipynb    # Preprocessing and augmentation step
│   ├── Training.ipynb         # Model training step
│   ├── Testing.ipynb          # Model evaluation and metrics
│   ├── Attendance.ipynb       # Main script for real-time face recognition
│   └── utils.py               # Core utilities, HOG extraction, & Ensemble logic
└── README.md
```

---

## 🔄 Workflow and Flowchart

The system operates in four distinct phases:

### 1. Data Cleaning & Augmentation (`Data_Cleaning.ipynb`)
- Reads the raw images from the `Dataset/` folder.
- Converts images to grayscale and resizes them to a standard 200x200 pixel resolution.
- Applies **Bilateral Filtering** to smooth the image while preserving edges, and **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to improve contrast in different lighting conditions.
- Uses **Data Augmentation** techniques (flipping, rotating, brightness/darkness adjustments) to artificially increase the size of the dataset, making the model more robust to varied real-world scenarios.

### 2. Feature Extraction & Model Training (`Training.ipynb` & `utils.py`)
- **Feature Extraction**: Extracts HOG features (using `skimage.feature.hog`) which effectively capture the shape and structure of the faces.
- **Training**: 
  - The LBPH model is trained directly on the augmented grayscale images.
  - The SVM and XGBoost models are trained on the extracted HOG feature vectors.
- After training, the models (and a LabelEncoder) are saved to disk via `joblib` and OpenCV's built-in save functions for later use.

### 3. Model Testing & Evaluation (`Testing.ipynb`)
- Loads the trained models and runs them against a test split of the data.
- Outputs detailed evaluation metrics for each model, including:
  - Overall Accuracy
  - Precision, Sensitivity (Recall), and Specificity per class
  - Confusion Matrices generated using `seaborn` and `matplotlib` to visually inspect true positives vs. false positives.

### 4. Real-time Attendance (`Attendance.ipynb`)
- The main application interface. It continuously captures frames (likely from a webcam).
- Uses a face detector to locate bounding boxes around faces in the current frame.
- Extracts Regions of Interest (ROIs) for the detected faces.
- Passes the ROIs to the `StackedFaceRecognizer` class (defined in `utils.py`).
- **The Ensemble Logic**:
  - Predicts the identity using LBPH, SVM, and XGBoost.
  - Filters out weak candidates based on strict confidence and distance thresholds.
  - Uses a voting mechanism to determine the final identity. If there's a tie, a restricted KNN model acts as a tie-breaker.
- Displays the recognized name and confidence score on the frame in real-time, effectively marking the presence (attendance) of the individual.

## 🛠️ Requirements & Setup

Make sure you have the following main libraries installed (you can use `pip install -r requirements.txt` if available):
- `opencv-python` and `opencv-contrib-python`
- `numpy`
- `scikit-learn`
- `scikit-image`
- `xgboost`
- `matplotlib`, `seaborn` (for testing visualizations)
- `Pillow`
- `joblib`
