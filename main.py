import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import pairwise_distances
import joblib
import os
import math
import win32com.client
import time

MODEL_DIR = "models"
classifier = None
scaler = None
pca = None
label_classes = None

if os.path.exists(os.path.join(MODEL_DIR, "voting_ensemble.pkl")):
    try:
        classifier = joblib.load(os.path.join(MODEL_DIR, "voting_ensemble.pkl"))
        if os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
            scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        if os.path.exists(os.path.join(MODEL_DIR, "pca.pkl")):
            pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
        if os.path.exists(os.path.join(MODEL_DIR, "label_classes.npy")):
            label_classes = np.load(os.path.join(MODEL_DIR, "label_classes.npy"))
    except Exception:
        classifier = None

embeddings_db = None
names_db = None
if classifier is None and os.path.exists(os.path.join(MODEL_DIR, "gesture_embeddings.npy")):
    model = np.load(os.path.join(MODEL_DIR, "gesture_embeddings.npy"), allow_pickle=True).item()
    embeddings_db = model["embeddings"]
    names_db = model["names"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6)

mp_draw = mp.solutions.drawing_utils

# Initialize Windows SAPI Speech Engine (No errors)
speaker = win32com.client.Dispatch("SAPI.SpVoice")

last_gestures = {}
speech_delay = 3.0  # seconds

def speak_gesture(gesture, hand_id):
    current_time = time.time()
    if hand_id not in last_gestures or (current_time - last_gestures[hand_id]) > speech_delay:
        speaker.Speak(gesture)
        last_gestures[hand_id] = current_time

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

def extract_live_embeddings(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return []
    data = []
    handed_list = []
    if result.multi_handedness:
        handed_list = [h.classification[0].label for h in result.multi_handedness]
    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        lm = hand_landmarks.landmark
        pts = normalize_landmarks(lm)
        handed = handed_list[idx] if idx < len(handed_list) else None
        pts = canonicalize_handedness(pts, handed)
        emb = make_features(pts)
        data.append((emb, hand_landmarks, idx))
    return data

# Open webcam with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    hands_data = extract_live_embeddings(frame)
    
    for embedding, handLms, hand_id in hands_data:
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        if classifier is not None:
            x = np.array([embedding], dtype=np.float32)
            if scaler is not None:
                x = scaler.transform(x)
            if pca is not None:
                x = pca.transform(x)
            pred_idx = int(classifier.predict(x)[0])
            predicted_gesture = label_classes[pred_idx] if label_classes is not None else str(pred_idx)
            if hasattr(classifier, "predict_proba"):
                prob = classifier.predict_proba(x)[0][pred_idx]
                confidence_pct = int(prob * 100)
            else:
                confidence_pct = 0
        else:
            dist = pairwise_distances([embedding], embeddings_db)[0]
            idx = np.argmin(dist)
            predicted_gesture = names_db[idx]
            max_dist = np.max(dist) if np.max(dist) != 0 else 1
            confidence = 1 - (dist[idx] / max_dist)
            confidence_pct = int(confidence * 100)

        x = int(handLms.landmark[0].x * frame.shape[1])
        y = int(handLms.landmark[0].y * frame.shape[0]) - 20

        display_text = f"{predicted_gesture}  ({confidence_pct}%)"

        cv2.putText(frame, display_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Speak output
        #speak_gesture(predicted_gesture, hand_id)

    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    if cv2.getWindowProperty("Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
