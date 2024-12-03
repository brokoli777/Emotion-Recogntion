# Emotion Recognition using Convolutional Neural Networks (CNN)

## Overview
This project implements emotion recognition from facial expressions using the **FER2013 dataset**. It leverages Convolutional Neural Networks (CNNs) for feature extraction and classification into seven emotion categories: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`. The project is built using TensorFlow and Keras and includes multiple preprocessing, training, and evaluation steps.

---

## Features
- **Dataset Handling**: Automatic download and setup of the FER2013 dataset using KaggleHub.
- **Data Preprocessing**: Includes data normalization, one-hot encoding of labels, and feature validation.
- **Model Architectures**: 
  - **Model 1**: A deep CNN with multiple convolutional, pooling, and dropout layers for feature extraction and generalization.
  - **Model 2**: A simpler CNN architecture leveraging global average pooling for better performance on smaller datasets.
- **Model Training**: Includes callbacks for early stopping and dynamic learning rate adjustment.
- **Visualization**: Comparison of training and validation accuracy/loss for both models.
- **Prediction**: Real-time testing of model performance on random samples from the test set.

---

## Prerequisites
- Python 3.8+
- Libraries:
  - TensorFlow/Keras
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - KaggleHub
  - Scikit-learn

Install the dependencies using:
```bash
pip install tensorflow pandas numpy matplotlib seaborn kagglehub scikit-learn
```

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/emotion-recognition.git
cd emotion-recognition
```

### 2. Download Dataset
The FER2013 dataset is downloaded automatically using KaggleHub. Ensure you have your Kaggle API key configured.

### 3. Run the Script
```bash
python Training.ipynb
```

This will:
1. Preprocess the dataset.
2. Train two CNN models.
3. Plot training/validation accuracy and loss for both models.
4. Test the models on a random sample.

---

## Key Results
- **Model Comparison**:
  - Training/Validation Accuracy and Loss for both models are plotted side-by-side for comparison.
- **Prediction Example**:
  - A random test sample is visualized along with its actual and predicted labels.

---

## Project Structure
```
emotion-recognition/
│
├── dataset/                # Contains the FER2013 dataset (downloaded at runtime)
├── Training.ipynb          # Main script for preprocessing, training, and evaluation
├── video.ipynb             # Detect emotion from webcam
└── README.md               # Project documentation (this file)
```

---

## Models
1. **Model 1**: Deep CNN architecture with:
   - Multiple convolutional and pooling layers
   - Dense fully connected layer with dropout
   - Optimized for detailed feature extraction

2. **Model 2**: Lightweight CNN with:
   - Global average pooling
   - Fewer convolutional layers
   - Faster training with comparable accuracy

---

## Training Hyperparameters
- Batch Size: `128`
- Epochs: `40`
- Optimizer: `Adam`
- Loss Function: `Categorical Cross-Entropy`

---

## Visualization
Plots include:
1. **Accuracy Comparison**: Training vs. Validation accuracy for both models.
2. **Loss Comparison**: Training vs. Validation loss for both models.
![image](https://github.com/user-attachments/assets/3e6ae029-a3d3-45eb-96f6-6968dc58c315)

---

## Contributors
- **Harshil Patel**
- **Bregwin Jogi**
- **Nonthachai Plodthong**
- **Matt Hyland**
