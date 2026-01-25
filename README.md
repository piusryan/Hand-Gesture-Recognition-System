# Speech Recognition for Sign Language
**Project Report / Documentation**

---

## Abstract

Communication is a fundamental aspect of human interaction. For individuals with hearing or speech impairments, Sign Language is the primary mode of communication. However, the majority of the population does not understand sign language, creating a significant communication gap. This project, **Speech Recognition for Sign Language**, aims to bridge this gap by developing a real-time system that translates hand gestures into text and speech. Leveraging Computer Vision (OpenCV) and Machine Learning (MediaPipe, Scikit-learn), the system detects hand landmarks, extracts geometric features, and classifies gestures with high accuracy using a Voting Ensemble model.

---

## 1. Introduction

### 1.1 Problem Statement
Sign language users often face challenges communicating with non-signers in daily life, educational institutions, and workplaces. There is a need for an assistive technology that can interpret sign language gestures and convert them into understandable speech or text in real-time without requiring expensive hardware like sensory gloves.

### 1.2 Objectives
- To develop a real-time computer vision-based system for hand gesture recognition.
- To implement robust feature extraction techniques using hand landmark geometry.
- To train a machine learning model capable of classifying various sign language gestures.
- To provide audio-visual feedback (Text-to-Speech) to facilitate two-way communication.

### 1.3 Scope
The project currently focuses on static hand gestures (e.g., alphabets, numbers, and simple words). It utilizes a standard webcam, making it accessible and cost-effective. The system is designed to be lightweight and runnable on standard personal computers.

---

## 2. System Analysis

### 2.1 Proposed System
The proposed system uses a camera-based approach. It captures video frames, detects hands using MediaPipe, and extracts a skeletal representation. Unlike deep learning approaches that rely solely on raw pixel data (CNNs), this system uses **feature engineering** (angles, distances) fed into a classical machine learning ensemble. This results in faster inference times and lower computational requirements while maintaining high accuracy for static signs.

### 2.2 Methodology
1.  **Input**: Video stream from a webcam.
2.  **Preprocessing**: Frame flipping, RGB conversion.
3.  **Detection**: Identifying hand landmarks (21 points) using MediaPipe.
4.  **Feature Extraction**: Computing normalized vectors, joint angles, and fingertip distances.
5.  **Classification**: predicting the gesture class using a Voting Classifier (SVM + Random Forest + MLP).
6.  **Output**: Displaying the label on screen and converting text to speech.

---

## 3. Implementation Details

### 3.1 Tech Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV (`cv2`)
- **Hand Tracking**: MediaPipe (`mediapipe`)
- **Machine Learning**: Scikit-learn (`sklearn`), NumPy
- **Model Persistence**: Joblib
- **Text-to-Speech**: PyWin32 (`win32com.client`)

### 3.2 Algorithm: Feature Extraction
To ensure the model is robust to hand position and scale, we perform the following geometric transformations:
- **Normalization**: All landmark coordinates are relative to the wrist (point 0).
- **Scale Invariance**: Coordinates are divided by the maximum distance from the wrist.
- **Rotation Alignment**: The hand is virtually rotated so the wrist-to-index-finger vector aligns with the X-axis.
- **Feature Vector**: A combination of:
    - Normalized (x, y, z) coordinates.
    - Euclidean distances between fingertips and wrist.
    - Angles between finger joints (calculated using dot product of vectors).
    - Distances between adjacent fingertips.

### 3.3 Algorithm: Classification
A **Voting Classifier** is employed to improve robustness. It aggregates the predictions of multiple estimators:
- **Support Vector Machine (SVM)**: Effective for high-dimensional spaces.
- **Random Forest**: Handles non-linear data well and reduces overfitting.
- **Multi-layer Perceptron (MLP)**: Captures complex patterns in the feature set.
The final prediction is determined by the majority vote (hard voting) or highest probability (soft voting).

---

## 4. System Requirements

### 4.1 Hardware Requirements
- **Processor**: Intel Core i5 or equivalent (recommended for smooth real-time performance).
- **RAM**: 4GB or higher.
- **Camera**: Standard USB Webcam or integrated laptop camera.
- **Storage**: Minimum 500MB free space.

### 4.2 Software Requirements
- **OS**: Windows 10/11 (required for PyWin32 TTS), macOS/Linux (without PyWin32).
- **Python Environment**: Python 3.8+.
- **Libraries**: `opencv-python`, `mediapipe`, `scikit-learn`, `numpy`, `joblib`, `pywin32`.

---

## 5. Installation & User Manual

### 5.1 Setup
1.  **Clone/Download** the repository to your local machine.
2.  **Install Dependencies**:
    Open a terminal in the project directory and run:
    ```bash
    pip install opencv-python mediapipe numpy scikit-learn joblib pywin32
    ```

### 5.2 Directory Structure
```
imgyoada/
│
├── main.py                 # Application Entry Point
├── models/                 # Trained Models & Training Scripts
│   ├── voting_ensemble.pkl # The meta-model
│   ├── train_all_models.py # Script for training
│   └── ...
├── dataset/                # Training Data (Images)
└── README.md               # Project Documentation
```

### 5.3 Execution
1.  Run the application:
    ```bash
    python main.py
    ```
2.  **Operation**:
    - Ensure your webcam is connected.
    - Place your hand in front of the camera.
    - The system will draw the hand skeleton and display the predicted text.
    - (Optional) Enable sound in `main.py` to hear the prediction.
3.  **Termination**: Press `ESC` to close the application.

---

## 6. Conclusion and Future Scope

### 6.1 Conclusion
This project successfully demonstrates a lightweight, real-time sign language recognition system. By combining efficient landmark detection with geometric feature engineering, we achieved a responsive system capable of running on standard hardware without the need for heavy GPUs.

### 6.2 Future Scope
- **Dynamic Gestures**: extending the system to recognize moving gestures (e.g., "J" or "Z") using LSTM or temporal analysis.
- **Sentence Construction**: Logic to string together words into full sentences.
- **Mobile App**: Porting the model to Android/iOS for greater accessibility.
- **Expanded Vocabulary**: Training on a larger dataset to cover the full dictionary of sign language.

---

**Developed for NLP Semester Project**