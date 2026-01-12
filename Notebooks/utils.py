import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import joblib

# --- CONFIGURATION ---
RADIUS = 1
NEIGHBORS = 8
GRID_X = 8
GRID_Y = 8

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def extract_hog_features(image):
    """
    Extracts Histogram of Oriented Gradients (HOG) features.
    """
    features = hog(image, 
                   orientations=9, 
                   pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), 
                   block_norm='L2-Hys', 
                   visualize=False)
    return features

def load_and_augment_data(path, augment=True):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    # Lists for LBPH (Raw Images)
    lbph_faces = []
    lbph_ids = []
    
    # Lists for SVM & KNN (HOG Feature Vectors)
    hog_features = []
    hog_labels = []
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    print(f"Processing {len(image_paths)} source images...")
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, 'uint8')
            
            # CAUTION: Ensure filenames are "User.ID.jpg"
            user_id = int(os.path.split(img_path)[-1].split(".")[1])
            
            # Base Preprocessing (200x200 standard)
            face_resized = cv2.resize(img_np, (200, 200), interpolation=cv2.INTER_CUBIC)
            face_smooth = cv2.bilateralFilter(face_resized, 5, 75, 75)
            enhanced = clahe.apply(face_smooth)
            
            # --- AUGMENTATION STACK ---
            aug_imgs = [enhanced]
            if augment:
                aug_imgs.append(cv2.flip(enhanced, 1))                        # Flip
                aug_imgs.append(rotate_image(enhanced, -10))                  # Rotate Left
                aug_imgs.append(rotate_image(enhanced, 10))                   # Rotate Right
                aug_imgs.append(cv2.convertScaleAbs(enhanced, alpha=1, beta=-40)) # Darker
                aug_imgs.append(cv2.convertScaleAbs(enhanced, alpha=1, beta=40))  # Brighter
            
            # Add to datasets
            for face in aug_imgs:
                # 1. For LBPH: Add the raw image
                lbph_faces.append(face)
                lbph_ids.append(user_id)
                
                # 2. For SVM & KNN: Extract HOG features
                feat = extract_hog_features(face)
                hog_features.append(feat)
                hog_labels.append(user_id)
                
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            
    return np.array(lbph_faces), np.array(lbph_ids), np.array(hog_features), np.array(hog_labels)

def train_lbph(faces, ids, save_path=None):
    """Trains and optionally saves an LBPH model."""
    print("Training LBPH Model (OpenCV)...")
    lbph = cv2.face.LBPHFaceRecognizer_create(radius=RADIUS, neighbors=NEIGHBORS, grid_x=GRID_X, grid_y=GRID_Y)
    lbph.train(faces, ids)
    if save_path:
        lbph.save(save_path)
        print(f"✅ Saved LBPH model to '{save_path}'")
    return lbph

def train_svm(features, labels, save_path=None):
    """Trains and optionally saves an SVM model."""
    print("Training SVM Model (HOG)...")
    svm = SVC(kernel='linear', C=10.0, gamma='scale', probability=True, random_state=42)
    svm.fit(features, labels)
    if save_path:
        joblib.dump(svm, save_path)
        print(f"✅ Saved SVM model to '{save_path}'")
    return svm

def train_xgboost(features, labels, save_path_model=None, save_path_le=None):
    """Trains and optionally saves an XGBoost model and LabelEncoder."""
    print("Training XGBoost Model (HOG)...")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        n_estimators=200,
        max_depth=6,
        learning_rate=0.5,
        subsample=0.8,
        colsample_bytree=1,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(features, labels_encoded)
    
    if save_path_model:
        joblib.dump(xgb_model, save_path_model)
        print(f"✅ Saved XGBoost model to '{save_path_model}'")
        
    if save_path_le:
        joblib.dump(le, save_path_le)
        print(f"✅ Saved LabelEncoder to '{save_path_le}'")
        
    return xgb_model, le

def print_metrics(model_name, y_true, y_pred):
    print(f"\n--- {model_name} EVALUATION ---")
    
    # Overall Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy:  {acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(list(set(y_true) | set(y_pred)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(f"\nDetailed Metrics per Class for {model_name}:")
    print(f"{'Class':<10} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Accuracy':<10} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 90)
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    for i, cls_label in enumerate(classes):
        tp_val = TP[i]
        tn_val = TN[i]
        fp_val = FP[i]
        fn_val = FN[i]
        
        sensitivity = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
        specificity = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0
        precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
        accuracy_cls = (tp_val + tn_val) / (tp_val + tn_val + fp_val + fn_val)
        
        print(f"{cls_label:<10} {tp_val:<6} {tn_val:<6} {fp_val:<6} {fn_val:<6} {accuracy_cls:<10.4f} {precision:<10.4f} {sensitivity:<12.4f} {specificity:<12.4f}")
    
    print("-" * 90)

class StackedFaceRecognizer:
    def __init__(self, models_dict, train_feats, train_lbls, id_map=None):
        self.lbph = models_dict.get('LBPH')
        self.svm = models_dict.get('SVM')
        self.xgb = models_dict.get('XGB')
        self.le = models_dict.get('LE')
        self.X_train = np.array(train_feats)
        self.y_train = np.array(train_lbls)
        self.id_map = id_map if id_map else {}
        
        # Internal memory to prevent duplicate counting
        self.session_history = {'SVM': set(), 'XGB': set(), 'LBPH': set()}

    def _reset_session(self):
        """Clears the short-term memory for a new frame/image."""
        self.session_history['SVM'].clear()
        self.session_history['XGB'].clear()
        self.session_history['LBPH'].clear()

    def _extract_hog(self, image):
        return hog(image, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                   
    def _filter_best_candidates(self, all_faces):
        """
        INTERNAL TOURNAMENT:
        Filters duplicate predictions for the same person ID.
        Ensures LBPH keeps the Lowest distance, while SVM/XGB keep Highest confidence.
        """
        for model_name in ['SVM', 'XGB', 'LBPH']:
            # 1. Gather all candidates
            candidates = []
            for idx, face in enumerate(all_faces):
                p = face['preds'].get(model_name)
                if p: candidates.append((idx, p['id'], p['conf']))
            
            # 2. Group by ID (Normalized to String to prevent int/str duplications)
            grouped = {}
            for c in candidates:
                pid_key = str(c[1])  # CRITICAL FIX: "5" and 5 are now the same key
                if pid_key not in grouped: grouped[pid_key] = []
                grouped[pid_key].append(c)
                
            # 3. Filter
            for pid, items in grouped.items():
                if len(items) > 1:
                    # LBPH: Lower distance is better (False means Ascending sort)
                    # SVM/XGB: Higher probability is better (True means Descending sort)
                    is_higher_better = (model_name != 'LBPH')
                    
                    # Sort candidates
                    items.sort(key=lambda x: x[2], reverse=is_higher_better)
                    
                    # Winner is items[0]. Remove the losers.
                    for loser in items[1:]:
                        idx_rem = loser[0]
                        all_faces[idx_rem]['preds'][model_name] = None
                        
        return all_faces

    def _predict_single(self, face_img, preds):
        """INTERNAL: Calculates the final vote for ONE face."""
        
        # --- CONFIGURATION ---
        SVM_THRESHOLD = 0.4 
        CONF_THRESHOLD = 0.5   # 50% minimum for SVM/XGB
        LBPH_THRESHOLD = 75    # 75 maximum distance for LBPH
        # ---------------------

        svm_p = preds.get('SVM')
        xgb_p = preds.get('XGB')
        lbph_p = preds.get('LBPH')
        
        valid_votes = []
        valid_confs = []
        
        # 1. Collect Valid Votes (STRICT ENTRY REQUIREMENTS)
        if (svm_p and 
            svm_p['id'] not in self.session_history['SVM'] and 
            svm_p['conf'] >= SVM_THRESHOLD):
            valid_votes.append(svm_p['id'])
            valid_confs.append(svm_p['conf'])
            self.session_history['SVM'].add(svm_p['id'])
            
        if (xgb_p and 
            xgb_p['id'] not in self.session_history['XGB'] and 
            xgb_p['conf'] >= CONF_THRESHOLD):
            valid_votes.append(xgb_p['id'])
            valid_confs.append(xgb_p['conf'])
            self.session_history['XGB'].add(xgb_p['id'])

        if (lbph_p and 
            lbph_p['id'] not in self.session_history['LBPH'] and 
            lbph_p['conf'] <= LBPH_THRESHOLD):
            valid_votes.append(lbph_p['id'])
            valid_confs.append(1.0 if lbph_p['conf'] < 50 else 0.6) 
            self.session_history['LBPH'].add(lbph_p['id'])
        
        # If everyone was filtered out, give up
        if not valid_votes: 
            return None, 0.0, "Weak/No Votes"
        
        # 2. Consensus Logic
        top_vote, count = Counter(valid_votes).most_common(1)[0]
        
        if count >= 2: 
            return top_vote, 1.0, f"Voting ({count}/3)"
        
        if len(set(valid_votes)) == 1:
            idx = valid_votes.index(top_vote)
            return top_vote, valid_confs[idx], "Single Vote"

        # 3. Restricted KNN (Tie Breaker)
        unique_cands = list(set(valid_votes))
        hog_vec = self._extract_hog(face_img)
        mask = np.isin(self.y_train, unique_cands)
        
        if np.sum(mask) == 0: 
            return None, 0.0, "Err"
        
        n_neighbors = min(5, len(self.X_train[mask]))
        mini_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', algorithm='brute')
        mini_knn.fit(self.X_train[mask], self.y_train[mask])
        
        final_id = mini_knn.predict([hog_vec])[0]
        final_conf = np.max(mini_knn.predict_proba([hog_vec])[0])
        
        return final_id, final_conf, "Restricted KNN"
        
    def process_frame(self, all_faces):
        """
        🛑 MAIN EXTERNAL CALL
        1. Resets session history.
        2. Filters duplicates across the whole frame (The Tournament).
        3. Predicts each face using the clean data.
        """
        self._reset_session()
        
        # Step A: Filter Duplicates
        cleaned_faces = self._filter_best_candidates(all_faces)
        
        final_results = []
        
        # Step B: Stack Predictions
        for face in cleaned_faces:
            pid, conf, method = self._predict_single(face['roi'], face['preds'])
            
            name = "Unknown"
            score_txt = ""
            
            if pid:
                name = self.id_map.get(pid, "Unknown")
                score_txt = f"{int(conf*100)}%"
            
            final_results.append({
                'bbox': face['bbox'], 
                'name': name, 
                'score': conf, 
                'score_txt': score_txt, 
                'info': method, 
                'roi': face['img_roi'],
                'raw_stats': face['raw_stats'],
                'raw_preds': face['unfiltered'] # Passed for GUI recovery if needed
            })
            
        return final_results
