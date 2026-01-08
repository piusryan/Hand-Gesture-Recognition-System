# train_all_models_no_torchvision.py
import os
import cv2
import math
import random
import joblib
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict

import mediapipe as mp
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===== Config =====
DATASET_DIR = "dataset"       # dataset/<gesture>/*.jpg
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ===== Mediapipe =====
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

# ===== Helpers (normalize + augment landmarks) =====
def normalize_landmarks(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    d = np.linalg.norm(pts, axis=1)
    maxd = np.max(d) if np.max(d) != 0 else 1.0
    pts /= maxd
    # align index finger MCP (5)
    v = pts[5][:2]
    angle = math.atan2(v[1], v[0])
    c, s = math.cos(-angle), math.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    pts[:, :2] = pts[:, :2].dot(R.T)
    return pts.flatten()

def augment_landmark_vector(vec, n=6, jitter_scale=0.01):
    out = []
    for _ in range(n):
        noise = np.random.normal(scale=jitter_scale, size=vec.shape)
        out.append(vec + noise)
    return out

# ===== Image augmentation helpers (cv2-based) - used for CNN =====
def random_resize_crop(img, size=(128,128), scale=(0.7,1.0)):
    h, w = img.shape[:2]
    s = random.uniform(*scale)
    new_h, new_w = int(h*s), int(w*s)
    img_rs = cv2.resize(img, (new_w, new_h))
    # pad/crop to target
    if new_h < size[1] or new_w < size[0]:
        # pad
        top = max((size[1]-new_h)//2, 0)
        left = max((size[0]-new_w)//2, 0)
        pad_img = np.zeros((size[1], size[0], 3), dtype=img.dtype)
        pad_img[top:top+new_h, left:left+new_w] = img_rs[:size[1], :size[0]]
        return pad_img
    else:
        # crop random
        y = random.randint(0, new_h - size[1])
        x = random.randint(0, new_w - size[0])
        return img_rs[y:y+size[1], x:x+size[0]]

def random_rotate(img, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def color_jitter(img, brightness=0.2, contrast=0.2):
    img = img.astype(np.float32) / 255.0
    b = random.uniform(-brightness, brightness)
    c = random.uniform(1-contrast, 1+contrast)
    img = np.clip((img + b) * c, 0, 1)
    return (img * 255).astype(np.uint8)

def random_horizontal_flip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, 1)
    return img

def apply_image_augs(img, size=(128,128)):
    img = random_resize_crop(img, size=size, scale=(0.8,1.0))
    img = random_rotate(img, max_angle=12)
    img = color_jitter(img, brightness=0.15, contrast=0.15)
    img = random_horizontal_flip(img, p=0.5)
    # final resize to exact
    img = cv2.resize(img, size)
    # convert to CHW float tensor later
    return img

# ===== Build datasets: extract landmarks & collect image paths =====
print("Scanning dataset and extracting landmarks...")
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
        res = hands_detector.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        land = res.multi_hand_landmarks[0].landmark
        emb = normalize_landmarks(land)
        X.append(emb)
        y.append(gesture_folder)
        img_paths_for_cnn[gesture_folder].append(img_path)

X = np.array(X)
y = np.array(y)
print("Found", len(X), "usable images with detected hands.")

if len(X) < 20:
    print("WARNING: very small dataset. Add more images.")

# ===== Augment landmark vectors =====
print("Augmenting landmark vectors...")
X_aug = []
y_aug = []
for emb, label in zip(X, y):
    X_aug.append(emb); y_aug.append(label)
    extra = augment_landmark_vector(emb, n=8, jitter_scale=0.012)
    for a in extra:
        X_aug.append(a); y_aug.append(label)

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)
print("After augmentation:", X_aug.shape[0], "samples")

# ===== Encode labels & scale =====
le = LabelEncoder()
y_enc = le.fit_transform(y_aug)
np.save(os.path.join(OUT_DIR, "label_classes.npy"), le.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_aug)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

# ===== PCA (optional) =====
USE_PCA = True
PCA_COMPONENTS = 64
if USE_PCA:
    pca = PCA(n_components=min(PCA_COMPONENTS, X_scaled.shape[1]), random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, os.path.join(OUT_DIR, "pca.pkl"))
    print("Saved PCA components:", pca.n_components_)
else:
    X_pca = X_scaled

# ===== Train-test split =====
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_enc, test_size=0.15, random_state=RANDOM_SEED, stratify=y_enc)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ===== Classical ML models =====
print("Training classical models (SVM, RF, MLP)...")

svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)
svm_params = {'C':[1,5], 'gamma':['scale','auto']}
svm_gs = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1, verbose=0)
svm_gs.fit(X_train, y_train)
svm_best = svm_gs.best_estimator_
print("SVM best params:", svm_gs.best_params_)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
rf.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=RANDOM_SEED)
mlp.fit(X_train, y_train)

for name, model in [("SVM", svm_best), ("RF", rf), ("MLP", mlp)]:
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name} accuracy: {acc:.4f}")

voting = VotingClassifier(estimators=[('svm', svm_best), ('rf', rf), ('mlp', mlp)], voting='soft', n_jobs=-1)
voting.fit(X_train, y_train)
vote_pred = voting.predict(X_test)
print("Voting accuracy:", accuracy_score(y_test, vote_pred))
print(classification_report(y_test, vote_pred, target_names=le.classes_))

joblib.dump(voting, os.path.join(OUT_DIR, "voting_ensemble.pkl"))
joblib.dump(svm_best, os.path.join(OUT_DIR, "svm.pkl"))
joblib.dump(rf, os.path.join(OUT_DIR, "rf.pkl"))
joblib.dump(mlp, os.path.join(OUT_DIR, "mlp.pkl"))

# ===== PyTorch MLP on landmark vectors =====
print("Training PyTorch MLP on landmark vectors...")

class LandmarksDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_for_torch = X_pca  # input dim after PCA (or scaled if PCA disabled)
Xtr, Xval, ytr, yval = train_test_split(X_for_torch, y_enc, test_size=0.15, random_state=RANDOM_SEED, stratify=y_enc)
train_ds = LandmarksDataset(Xtr, ytr)
val_ds = LandmarksDataset(Xval, yval)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_torch = SimpleMLP(X_for_torch.shape[1], len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_torch.parameters(), lr=1e-3)

best_val = 0.0
for epoch in range(1, 51):
    model_torch.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        logits = model_torch(bx)
        loss = criterion(logits, by)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # validate
    model_torch.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model_torch(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(by.numpy())
    acc = accuracy_score(all_trues, all_preds)
    if acc > best_val:
        best_val = acc
        torch.save(model_torch.state_dict(), os.path.join(OUT_DIR, "torch_mlp_best.pth"))
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} val_acc = {acc:.4f} (best {best_val:.4f})")
print("PyTorch MLP best val acc:", best_val)

# ===== Optional PyTorch CNN using OpenCV-based augmentations =====
print("Preparing CNN dataset (if enough images exist)...")
cnn_img_paths = []
cnn_labels = []
for label, paths in img_paths_for_cnn.items():
    for p in paths:
        cnn_img_paths.append(p)
        cnn_labels.append(label)

if len(cnn_img_paths) < 50:
    print("Not enough images for CNN, skipping CNN training.")
else:
    print("Training small CNN on images (no torchvision)...")
    cnn_y = le.transform(cnn_labels)
    class ImgDataset(Dataset):
        def __init__(self, paths, labels, augment=True, size=(128,128)):
            self.paths = paths
            self.labels = labels
            self.augment = augment
            self.size = size
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            p = self.paths[idx]
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.augment:
                img = apply_image_augs(img, size=self.size)
            else:
                img = cv2.resize(img, self.size)
            # convert to CHW float tensor normalized [0,1]
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2,0,1))
            return torch.tensor(img, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    train_paths, val_paths, train_y, val_y = train_test_split(cnn_img_paths, cnn_y, test_size=0.15, random_state=RANDOM_SEED, stratify=cnn_y)
    train_ds = ImgDataset(train_paths, train_y, augment=True)
    val_ds = ImgDataset(val_paths, val_y, augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    class SmallCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*4*4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes)
            )
        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = SmallCNN(len(le.classes_)).to(device)
    opt = optim.Adam(cnn.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, 31):
        cnn.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = cnn(bx)
            loss = crit(logits, by)
            opt.zero_grad(); loss.backward(); opt.step()
        # validate
        cnn.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                logits = cnn(bx.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(by.numpy())
        acc = accuracy_score(all_trues, all_preds)
        if acc > best_val:
            best_val = acc
            torch.save(cnn.state_dict(), os.path.join(OUT_DIR, "cnn_best.pth"))
        if epoch % 5 == 0:
            print(f"[CNN] epoch {epoch} val_acc={acc:.4f} best={best_val:.4f}")
    print("CNN best val acc:", best_val)

print("Training finished. Models saved to", OUT_DIR)
