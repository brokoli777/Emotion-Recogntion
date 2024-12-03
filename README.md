# Emotion Recognition using Convolutional Neural Networks (CNN)

## Overview

This project implements an **Emotion Recognition** system using Convolutional Neural Networks (CNNs) to classify facial expressions into seven categories: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`. The model was trained on the **FER2013 dataset** and uses Python libraries like TensorFlow/Keras for deep learning and OpenCV for real-time application.

---

## Features

- **Dataset**: Uses FER2013 dataset (grayscale 48x48 images).
- **Two CNN Architectures**:
  - **Model 1**: High-capacity CNN designed for larger datasets and complex tasks.
  - **Model 2**: Lightweight alternative optimized for smaller datasets.
- **Real-Time Emotion Recognition**: Combines trained CNN with OpenCV and Haar Cascade for face detection and emotion recognition in webcam footage.
- **Performance Metrics**:
  - Model 1 achieved 64% test accuracy.
  - Model 2 achieved 58% test accuracy.

---

## References

The project is inspired by several references and existing implementations, including:
1. [Kaggle: Facial Expression Recognition FER Challenge](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge)
2. [Facial Emotion Recognition in Real-Time](https://learnopencv.com/facial-emotion-recognition/)
3. [Introduction to Convolutional Neural Networks](https://www.geeksforgeeks.org/introduction-convolution-neural-network/)
4. [Microsoft FERPlus](https://github.com/microsoft/FERPlus): FER+ annotations for the FER dataset.

---

## Applications

The emotion recognition system has practical uses in fields like:
- **Customer Service**: Evaluating customer satisfaction and conducting product surveys.
- **Healthcare**: Monitoring emotional states for mental health diagnosis and treatment.
- **Security**: Detecting unusual behavior in surveillance systems.
- **Entertainment**: Adapting gaming or AR environments to the player's emotions.

---

## Project Details

### Dataset
- **FER2013**: A publicly available dataset containing 48x48 grayscale images split into training and testing sets.
- Preprocessing steps:
  - Image normalization to values in [0, 1].
  - One-hot encoding for emotion labels.

### Model Architectures

#### Model 1
- **Features**:
  - Four convolutional blocks with filters increasing from 32 to 128.
  - Fully connected layer with 1024 neurons.
  - Dropout rates between 0.2 and 0.5 for regularization.
  - Optimized for larger, complex datasets.
- **Performance**:
  - Test Accuracy: 64%
  - Test Loss: 1.08

#### Model 2
- **Features**:
  - Three convolutional blocks with a Global Average Pooling layer.
  - Fully connected layer with 64 neurons.
  - Lightweight architecture for smaller datasets.
- **Performance**:
  - Test Accuracy: 58%
  - Test Loss: 1.08

### Training Details
- Optimizer: **Adam** with default (Model 1) and custom (Model 2) learning rates.
- Loss Function: **Categorical Cross-Entropy**.
- Metrics: **Accuracy**.
- Callbacks:
  - **Early Stopping**: Monitors validation loss to avoid overfitting.
  - **ReduceLROnPlateau**: Dynamically adjusts the learning rate.

### Real-Time Emotion Recognition
A Python notebook (`video.ipynb`) demonstrates real-time emotion recognition:
- Detects faces using OpenCV and Haar Cascade.
- Passes cropped faces to the trained CNN model for emotion inference.
- Displays bounding boxes and predicted emotions on webcam footage.

---

## Results & Insights

### Comparison of Models

| Metric          | Model 1 | Model 2 |
|------------------|---------|---------|
| Test Accuracy    | 64%     | 58%     |
| Test Loss        | 1.08    | 1.08    |
| Training Speed   | Slower  | Faster  |
| Overfitting Risk | Higher  | Lower   |

- **Model 1** excels at handling large datasets with high complexity but risks overfitting.
- **Model 2** is a simpler alternative for faster training and better generalization on smaller datasets.
  ![image](https://github.com/user-attachments/assets/a677a52c-6ce7-4e27-961d-bdb9647536a9)


### Real-Time Demo
- Detects emotions in real-time using the webcam.
- Achieves accurate predictions for a range of expressions under good lighting conditions.

---

## Future Improvements
- Incorporate diverse datasets for better generalization.
- Add multimodal inputs (audio/text) for enhanced emotion detection.
- Explore transfer learning with pre-trained models for higher accuracy.
- Fine-tune models for real-time performance optimization.

---

## Prerequisites

### Installation
- Python 3.8+
- Required Libraries:
  ```bash
  pip install tensorflow pandas numpy matplotlib seaborn kagglehub scikit-learn opencv-python
  ```

---

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/emotion-recognition.git
   cd emotion-recognition
   ```

2. **Download the Dataset**:
   The dataset is downloaded automatically using KaggleHub. Ensure your Kaggle API key is configured.

3. **Train the Models**:
   ```bash
   python train_model.py
   ```

4. **Run the Real-Time Demo**:
   Open `video.ipynb` and follow the instructions to test emotion recognition using your webcam.

---

## Contributors

- **Bregwin Jogi**
- **Harshil Patel**
- **Nonthachai Plodthong**
- **Matt Hyland**
