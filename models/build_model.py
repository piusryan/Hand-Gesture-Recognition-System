import os
import math
import cv2
import mediapipe as mp
import numpy as np
import joblib
from glob import glob
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

DATASET_DIR = "dataset"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

def normalize_landmarks(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    d = np.linalg.norm(pts, axis=1)
    maxd = np.max(d) if np.max(d) != 0 else 1.0
    pts /= maxd
    v = pts[5][:2]
    angle = math.atan2(v[1], v[0])
    c, s = math.cos(-angle), math.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2].dot(R.T)
    return pts

def canonicalize_handedness(pts, label):
    if label and label.lower().startswith("right"):
        pts[:, 0] = -pts[:, 0]
    return pts

def finger_indices():
    return {
        "thumb": [1,2,3,4],
        "index": [5,6,7,8],
        "middle": [9,10,11,12],
        "ring": [13,14,15,16],
        "pinky": [17,18,19,20],
    }

def angle_at(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def make_features(pts):
    f = []
    f.extend(pts.flatten().tolist())
    tips = [4,8,12,16,20]
    wrist = pts[0]
    for t in tips:
        f.append(np.linalg.norm(pts[t] - wrist))
    fi = finger_indices()
    for inds in fi.values():
        a,b,c,d = [pts[i] for i in inds]
        f.append(angle_at(a,b,c))
        f.append(angle_at(b,c,d))
    tip_pairs = [(4,8),(4,12),(4,16),(4,20),(8,12),(8,16),(8,20),(12,16),(12,20),(16,20)]
    for i,j in tip_pairs:
        f.append(np.linalg.norm(pts[i]-pts[j]))
    return np.array(f, dtype=np.float32)

def augment_pts(pts, n=8, jitter_scale=0.012, max_rot_deg=12):
    out = [pts]
    for _ in range(n):
        noise = np.random.normal(scale=jitter_scale, size=pts.shape)
        a = math.radians(np.random.uniform(-max_rot_deg, max_rot_deg))
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        aug = pts.copy() + noise
        aug[:, :2] = aug[:, :2].dot(R.T)
        out.append(aug)
    return out

print("Scanning dataset and extracting features...")
X = []
y = []
img_paths_for_cnn = defaultdict(list)

for gesture_folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, gesture_folder)
    if not os.path.isdir(folder_path):
        continue
    image_files = glob(os.path.join(folder_path, "*"))
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        land = res.multi_hand_landmarks[0].landmark
        handed = None
        if res.multi_handedness:
            handed = res.multi_handedness[0].classification[0].label
        pts = normalize_landmarks(land)
        pts = canonicalize_handedness(pts, handed)
        for a in augment_pts(pts):
            X.append(make_features(a))
            y.append(gesture_folder)
        img_paths_for_cnn[gesture_folder].append(img_path)

X = np.array(X)
y = np.array(y)
print("Samples:", X.shape[0])

if len(X) < 20:
    print("WARNING: very small dataset. Add more images.")

le = LabelEncoder()
y_enc = le.fit_transform(y)
np.save(os.path.join(OUT_DIR, "label_classes.npy"), le.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

USE_PCA = True
PCA_COMPONENTS = 64
if USE_PCA:
    pca = PCA(n_components=min(PCA_COMPONENTS, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, os.path.join(OUT_DIR, "pca.pkl"))
else:
    X_pca = X_scaled

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_enc, test_size=0.15, stratify=y_enc, random_state=42)

print("Training classifiers...")
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_gs = GridSearchCV(svm, {'C':[1,5,10], 'gamma':['scale','auto']}, cv=cv, n_jobs=-1, verbose=0)
svm_gs.fit(X_train, y_train)
svm_best = svm_gs.best_estimator_

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=600, random_state=42)
mlp.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_gs = GridSearchCV(knn, {'n_neighbors':[3,5,7,9]}, cv=cv, n_jobs=-1, verbose=0)
knn_gs.fit(X_train, y_train)
knn_best = knn_gs.best_estimator_

svm_cal = CalibratedClassifierCV(svm_best, method='sigmoid', cv=cv)
svm_cal.fit(X_train, y_train)
voting = VotingClassifier(estimators=[('svm', svm_cal), ('rf', rf), ('mlp', mlp), ('knn', knn_best)], voting='soft', n_jobs=-1)
voting.fit(X_train, y_train)

for name, model in [("SVM", svm_best), ("RF", rf), ("MLP", mlp), ("KNN", knn_best), ("Voting", voting)]:
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} acc: {acc:.4f}")
print(classification_report(y_test, voting.predict(X_test), target_names=le.classes_))

joblib.dump(voting, os.path.join(OUT_DIR, "voting_ensemble.pkl"))
joblib.dump(svm_cal, os.path.join(OUT_DIR, "svm.pkl"))
joblib.dump(rf, os.path.join(OUT_DIR, "rf.pkl"))
joblib.dump(mlp, os.path.join(OUT_DIR, "mlp.pkl"))
joblib.dump(knn_best, os.path.join(OUT_DIR, "knn.pkl"))

embeddings = X
gesture_names = y
np.save(os.path.join(OUT_DIR, "gesture_embeddings.npy"), {"embeddings": embeddings, "names": gesture_names})

print("Training finished. Models saved to", OUT_DIR)
