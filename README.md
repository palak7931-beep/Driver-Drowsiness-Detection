# 🚗 Driver Drowsiness Detection

A real-time computer vision system for detecting driver drowsiness using facial landmarks, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and temporal analysis.

The project also includes a classical machine-learning experiment using HOG features and an SVM classifier for open-eye and closed-eye classification.

---

## 📌 Project Overview

Driver drowsiness is a serious safety concern because reduced alertness can affect concentration, visual attention, and reaction time.

This project implements a webcam-based system that continuously analyzes facial features to identify visual indicators of drowsiness.

The system:

- Captures live video using a webcam
- Detects facial landmarks using MediaPipe Face Mesh
- Calculates Eye Aspect Ratio (EAR)
- Calculates Mouth Aspect Ratio (MAR)
- Detects prolonged eye closure
- Detects sustained mouth opening/yawning
- Calculates a drowsiness score
- Generates an audible alarm
- Logs session data into CSV files

A separate HOG + SVM experiment is also performed on the MRL Eye Dataset.

---

## 🎯 Objectives

- Develop a real-time driver drowsiness detection system.
- Process webcam video using OpenCV.
- Detect facial landmarks using MediaPipe.
- Analyze eye closure using EAR.
- Analyze mouth opening using MAR.
- Apply consecutive-frame conditions to reduce false alerts.
- Generate an audible warning when drowsiness is detected.
- Store session measurements for analysis.
- Evaluate eye-state classification using HOG and SVM.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| OpenCV | Image and video processing |
| MediaPipe Face Mesh | Facial landmark detection |
| NumPy | Numerical computation |
| scikit-image | HOG feature extraction |
| scikit-learn | SVM classification and evaluation |
| Matplotlib | Result visualization |
| Pygame | Audio alarm |
| Visual Studio Code | Development environment |
| Git & GitHub | Version control |

---

## 🧠 Real-Time Detection Methodology

### 1. Webcam Acquisition

OpenCV captures frames continuously from the webcam.

### 2. Facial Landmark Detection

MediaPipe Face Mesh detects facial landmarks from each frame.

### 3. Eye Aspect Ratio

The Eye Aspect Ratio (EAR) is calculated from selected eye landmarks.

The project uses:

```text
EAR threshold = 0.22
