
# 🤟 ASL Sign Language Recognition System
### Real-Time Hand Gesture Recognition Using Deep Learning & Computer Vision

<p align="center">
  <b>An AI-powered system that recognizes American Sign Language (ASL) hand gestures through a webcam.</b>
</p>

---

## 📌 Project Overview

The ASL Sign Language Recognition System is a computer vision and deep learning project designed to recognize American Sign Language hand gestures in real time.

The system captures hand gestures through a webcam, detects and tracks the hand using MediaPipe, and classifies gestures using a trained MobileNetV2 deep learning model.

The main objective is to explore how Artificial Intelligence can support communication by recognizing hand gestures.

## 🎯 Project Objectives

- Recognize ASL hand gestures using deep learning.
- Perform real-time gesture classification through a webcam.
- Apply computer vision techniques for hand detection.
- Explore AI-based solutions for accessible communication.

## ✨ Key Features

- 📷 Real-time webcam-based gesture recognition.
- ✋ Hand detection and tracking using MediaPipe.
- 🧠 Deep learning-based classification using MobileNetV2.
- 🔤 Recognition of 34 gesture classes.
- ⚡ Image preprocessing and model prediction.
- 🖥️ Live prediction display.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning model |
| MobileNetV2 | Gesture classification |
| OpenCV | Webcam and image processing |
| MediaPipe | Hand detection and tracking |
| NumPy | Numerical operations |

## 🧠 Model Architecture

This project uses MobileNetV2, a convolutional neural network architecture, for image classification.

**Model details:**
- Architecture: MobileNetV2
- Input image size: 224 × 224
- Output: 34 gesture classes
- Framework: TensorFlow / Keras

## 🔄 System Workflow

1. Webcam captures the user's hand gesture.
2. MediaPipe detects and tracks the hand.
3. The input image is processed and resized.
4. The trained MobileNetV2 model predicts the gesture.
5. The predicted class is displayed on the screen.

## 📊 Dataset Information

The project dataset was prepared for 34 gesture classes.

- Total gesture classes: 34
- Images per class: 300
- Total images: 10,200
- Image resolution: 640 × 480

The dataset was organized into gesture-specific folders for model training and evaluation.

The full dataset is not included in this repository if its size is too large.

## 🚀 Future Enhancements

- Support for continuous sign language recognition.
- Recognition of complete words and sentences.
- Improved prediction accuracy under different lighting conditions.
- Support for additional sign language gestures.
- Development of a user-friendly web or mobile application.

## 🎓 Applications

- Accessible communication support.
- Sign language learning systems.
- AI-based gesture recognition.
- Human-computer interaction.
- Assistive technology research.

## 👩‍💻 Author

**Harika Rodda**

B.Sc. Honours Computer Science

Interested in Artificial Intelligence, Machine Learning, Deep Learning and Computer Vision.

## ⭐ Acknowledgement

This project was developed as part of my learning journey in Artificial Intelligence, Deep Learning and Computer Vision.

---

<p align="center">
  <b>Built with Python, Deep Learning and Computer Vision ❤️</b>
</p>
