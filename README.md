# Real-Time Facial Expression Recognition using CNN

This project implements a **real-time facial expression recognition system** using a **Convolutional Neural Network (CNN)** trained on grayscale facial images. The system captures facial expressions via a webcam, detects faces using **Haar Cascades**, and classifies emotions into seven distinct categories.

---

## 📌 Overview

Facial expressions are key indicators of human emotions. This project focuses on identifying emotions using **deep learning**, specifically a CNN trained on **48×48 grayscale facial images**, and integrates OpenCV for real-time webcam-based inference.

---

## 🔧 Model Architecture

The CNN model is built using **Keras with TensorFlow backend** and follows a Sequential architecture.

### Architecture Details:
- **Input Shape:** `(48, 48, 1)`
- **Convolutional Layers:**  
  - 4 Conv2D layers  
  - Filters ranging from **32 to 128**
- **Batch Normalization:** Applied after each convolution layer
- **Activation Functions:**  
  - ReLU for hidden layers  
  - Softmax for output layer
- **Pooling:** MaxPooling layers for spatial reduction
- **Dropout:**  
  - 0.25 and 0.5 to prevent overfitting
- **Dense Layers:**  
  - Fully connected layer with **250 units**  
  - Output layer with **7 units**
- **Loss Function:** `categorical_crossentropy`

---

## 🧠 Emotion Classes

The model predicts the following seven emotions:

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## 📈 Model Performance

- **Training Accuracy:** ~95%  
- **Validation Accuracy:** ~65%  

> Validation accuracy may vary depending on dataset balance and preprocessing.

---

## 🎥 Real-Time Detection (`webcam_test.py`)

The `webcam_test.py` script enables real-time facial expression recognition using a webcam.

### Workflow:

#### 1️⃣ Model Loading
- Loads CNN architecture from `Facial Expression Recognition.json`
- Loads trained weights from `fer.h5`

#### 2️⃣ Face Detection
- Uses `haarcascade_frontalface_default.xml` to detect faces

#### 3️⃣ Preprocessing
- Converts frames to grayscale  
- Crops detected face regions  
- Resizes faces to **48×48 pixels**  
- Normalizes pixel values  

#### 4️⃣ Prediction
- Passes processed face images into the CNN
- Predicts the most probable emotion

#### 5️⃣ Visualization
- Draws bounding boxes around detected faces
- Displays predicted emotion labels on the video feed
- Press **`q`** to exit the webcam window

---

## 📁 Project Structure

