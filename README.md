<div align="center">
  <h1>🤖 Gesture-Controlled Mime Robot</h1>
  <p><i>A cutting-edge hackathon project bridging human gestures with real-time robot mimicry.</i></p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/ROS_2-Humble%2FFoxy-22314E.svg?logo=ros" alt="ROS 2">
    <img src="https://img.shields.io/badge/MediaPipe-Pose-orange.svg" alt="MediaPipe">
    <img src="https://img.shields.io/badge/Unity-3D-lightgrey.svg?logo=unity" alt="Unity">
    <img src="https://img.shields.io/badge/Machine_Learning-GAIL%20%7C%20KNN%20%7C%20LSTM-green.svg" alt="ML">
  </p>
</div>

---

## 📖 Overview

The **Gesture-Controlled Mime Robot** is an end-to-end framework built during a hackathon to bridge the gap between human kinetic motion and robot execution. By utilizing everyday webcams, the system extracts human poses in real-time, classifies specific interactive gestures, and retargets this motion to both a **virtual Unity simulation** and a **physical/simulated robot via ROS 2**.

Whether you want the robot to say "Hello", do the "Disco", or learn via Imitation Learning, this pipeline supports high-frequency, low-latency pose streaming.

---

## ✨ Key Features

- **Real-Time Pose Tracking**: Utilizes Google's MediaPipe for highly accurate skeletal landmark extraction without the need for specialized tracking suits.
- **Machine Learning Gesture Recognition**: Employs trained classifiers (KNN, LSTM) to quickly identify specific user gestures (`Clap`, `Disco`, `Hello`, `Wakanda`, `Zombie`).
- **Generative Adversarial Imitation Learning (GAIL)**: Incorporates state-of-the-art imitation learning architectures to train robots (such as OP3 models) based on expert human trajectories.
- **Digital Twin Visualization**: A complete Unity 3D environment that subscribes to the pose data and visually acts as a mirror/avatar.
- **ROS 2 Integration**: Docker-supported ROS 2 interface utilizing `ros2_control` for safe, reliable transmission of joint trajectories to hardware or robotic simulations.

---

## 🏗️ Architecture & Repository Structure

The project is logically divided into four core modules:

```text
Gesture_Robot/
├── mediapipefiles_k/                       # Machine Learning & Core Perception
│   ├── dataset_builder.py                  # Extracts skeletal data into CSVs
│   ├── train_gail.py / train_lstm.py       # Trains policies & classifiers
│   └── live_demo.py                        # Standalone webcam demonstration
│
├── pose_mimicing-main/                     # Real-time Inference & Logic
│   └── src/                                # Model handling, utils, and bridges
│
├── Gesture-detection-Unity-mimicry-main/   # Digital Twin
│   ├── mediapipe_sender.py                 # UDP/network streaming to Unity
│   └── Assets/                             # 3D assets & scripts for the engine
│
└── gesture-pose-interface/                 # Robot Control Interface
    ├── docker-compose.yml                  # Containerized ROS 2 environment
    ├── perception_pipeline.launch.py       # Main launch file for ROS node
    └── ros2_control/                       # Hardware abstraction layer
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**: Essential for MediaPipe and ML inference.
- **Unity Hub & Engine**: To run the virtual digital twin.
- **Docker & Docker Compose**: (Recommended) For running the ROS 2 interface without local host pollution.
- **Webcam**: Required for live gesture detection.

### 1. Model Tracking & Machine Learning
We use MediaPipe to track skeletons and map them dynamically.
1. Navigate to the pipeline code (either `pose_mimicing-main` or `mediapipefiles_k`).
2. Run standard inferences:
   ```bash
   python live_demo.py
   ```
   *(Note: Pre-trained `.pkl` and `.zip` models are intentionally kept out of source control. Place your datasets/weights back in the directory if pulled freshly from git).*

### 2. Digital Twin Visualization (Unity)
To see your virtual avatar mirroring your actions:
1. Open the inner `Gesture-detection-Unity-mimicry-main` via Unity Hub.
2. In a separate terminal, run the pose broadcaster:
   ```bash
   python mediapipe_sender.py
   ```
3. Hit `Play` in the Unity Editor.

### 3. Deploying to the Robot (ROS 2)
We encapsulate the robotics logic within Docker for maximum repeatability:
1. Change into the hardware interface directory:
   ```bash
   cd gesture-pose-interface
   ```
2. Build and run the perception stack:
   ```bash
   docker-compose up --build
   ```
3. This spin-ups a ROS 2 node capable of talking directly with `ros2_control` interfaces.

---

## ⚠️ Repository Notes

**Large Files Ignored**:  
Because this is a hackathon project processing thousands of frames, datasets (`*.csv`) and network weights (`*.pkl`, `*.pth`, `*.zip`) can exceed standard VCS limits. We proudly use a strict `.gitignore` to keep this repository lightweight. You will need to regenerate the datasets locally using `dataset_builder.py` or fetch the model weights from the release page.

---

## 🔮 Future Work
- Extend gesture libraries with Dynamic Time Warping (DTW) for complex multi-stage actions.
- Improve Docker container size optimizations.
- Native VR headset integration for viewpoint matching.

