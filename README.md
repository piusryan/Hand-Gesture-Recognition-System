## 🤖 Hand Gesture Recognition System
Python Version: 3.11.0

### 📋 Project Description
This is a real-time hand gesture recognition system that uses computer vision and machine learning to detect and classify hand gestures from webcam input. The system combines MediaPipe for hand landmark detection with multiple machine learning models for gesture classification, including ensemble methods for improved accuracy.

🎓 Academic Context: This project is my MSc mini project on NLP topic, exploring the intersection of computer vision and natural user interfaces for human-computer interaction.

### 🎯 Features
- Real-time gesture recognition from webcam feed
- Multiple ML models (SVM, Random Forest, MLP, CNN, Voting Ensemble)
- Hand landmark extraction using MediaPipe
- Data augmentation for improved model performance
- Confidence scoring for predictions
- Text-to-speech capability (optional)
- Cross-platform compatibility (Windows-focused)
### 🏗️ Architecture
The system uses a multi-stage pipeline:

1. Hand Detection : MediaPipe Hands extracts 21 key landmarks per hand
2. Feature Engineering : Normalized landmark coordinates + geometric features (angles, distances)
3. Model Ensemble : Voting classifier combining SVM, Random Forest, and MLP
4. Real-time Processing : Live webcam feed with overlay predictions
### 📊 Dataset
- Gesture Classes : Single gesture class ("1" - appears to be pointing/number one gesture)
- Training Images : 300+ images per gesture class
- Data Augmentation : Landmark jittering and image transformations
- Preprocessing : Hand normalization, rotation alignment, handedness canonicalization
### 🤖 Models Used
1. Voting Ensemble (Primary model)
   
   - Combines: SVM + Random Forest + MLP
   - Soft voting for probability-based predictions
2. Support Vector Machine (SVM)
   
   - RBF kernel with probability estimation
   - Grid search optimized (C=[1,5], gamma=['scale','auto'])
3. Random Forest
   
   - 200 estimators
   - Balanced class weights
4. Multi-Layer Perceptron (MLP)
   
   - Architecture: (256, 128) hidden layers
   - Max iterations: 500
5. Convolutional Neural Network (CNN)
   
   - PyTorch implementation
   - Architecture: Conv(32→64→128) + FC(256) + Dropout
   - Input: 128×128 RGB images
6. K-Nearest Neighbors (Fallback)
   
   - Used for embedding-based matching when ensemble unavailable
### 📦 Dependencies
```
# Core Computer Vision
opencv-python
mediapipe
numpy

# Machine Learning
scikit-learn
joblib
torch  # PyTorch for CNN

# Utilities
tqdm
glob2
```
### 🔧 Environment Setup
```
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe 
numpy scikit-learn joblib torch tqdm
```
### 🚀 Usage
```
# Train models (optional - 
pre-trained models included)
python models/train_all_models.py

# Run real-time recognition
python main.py
```
### 📁 Project Structure
```
imgyoada/
├── main.py                    # 
Real-time recognition application
├── models/
│   ├── train_all_models.py   # 
Model training pipeline
│   ├── build_model.py        # 
Alternative training script
│   ├── voting_ensemble.pkl   # 
Primary ensemble model
│   ├── svm.pkl              # 
Support Vector Machine
│   ├── rf.pkl               # 
Random Forest
│   ├── mlp.pkl              # 
Multi-Layer Perceptron
│   ├── knn.pkl              # 
K-Nearest Neighbors
│   ├── scaler.pkl           # 
Feature scaler
│   ├── pca.pkl              # PCA 
transformation
│   └── label_classes.npy    # 
Gesture class labels
├── dataset/
│   └── 1/                   # 
Gesture training images
└── .venv/                   # 
Python virtual environment
```
### 🎮 Key Features
- Landmark-based features : 63D normalized coordinates + geometric features
- Real-time processing : 30+ FPS on modern hardware
- Confidence scoring : Percentage-based prediction confidence
- Fallback mechanism : KNN matching when primary models unavailable
- Handedness handling : Automatic left/right hand canonicalization
### 🔍 Technical Details
- Feature Dimensions : 63 (landmarks) + 5 (tip distances) + 10 (angles) + 10 (tip-tip distances) = 88D
- PCA Reduction : Optional 64 components
- Input Resolution : 800×600 webcam capture
- Detection Confidence : 0.6 minimum threshold
- Speech Delay : 3-second cooldown between announcements
### 📝 Notes
- Currently trained on single gesture class ("1" - pointing gesture)
- Windows SAPI integration for text-to-speech (commented out by default)
- Models are pre-trained and ready for immediate use
- Dataset can be expanded by adding more gesture folders to dataset/ directory
### 🤝 Contributing
To add new gestures:

1. Create a new folder in dataset/ with gesture name
2. Add 300+ training images of the gesture
3. Run python models/train_all_models.py to retrain
4. Updated models will be automatically saved
🎓 Academic Project : MSc Mini Project on NLP Topic - Exploring Natural User Interfaces through Computer Vision and Machine Learning Status : Ready for deployment with pre-trained models
