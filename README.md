# Breast Ultrasound Image Classification using CNN, Class Weights, SMOTE, and Depthwise Separable CNN

## Course
Deep Learning for Medical Images

## Author
**Name:** A Smaran Reddy  
**Roll Number:** CS23B1011  

---

# Project Overview

Breast cancer is one of the leading causes of cancer-related deaths worldwide. Early detection plays a critical role in improving survival rates. Ultrasound imaging is widely used as a non-invasive method to detect abnormalities in breast tissue.

This project develops deep learning models to classify breast ultrasound images into three categories:

- **Benign**
- **Malignant**
- **Normal**

A key challenge in the dataset is **class imbalance**, where benign samples significantly outnumber malignant and normal samples.

To address this issue, two modeling approaches were explored.

---

# Approaches

## 1. Standard CNN (Baseline Notebook)

Three experimental approaches were evaluated:

1. Simple CNN  
2. CNN with Class Weights  
3. CNN with SMOTE  

---

## 2. Depthwise Separable CNN (DWS Notebook)

A more advanced and efficient architecture using:

- MobileNetV2 (Transfer Learning)  
- Depthwise Separable Convolutions  
- Fine-Tuning  
- Data Augmentation  

Three experiments were conducted:

1. Transfer Learning Baseline  
2. Transfer Learning + Class Weights  
3. Augmented Training (No SMOTE)  

---

# Dataset

The project uses the **BUSI (Breast Ultrasound Images) dataset** available on Kaggle.

Dataset Source:  
https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images

The dataset contains three classes:

- Benign  
- Malignant  
- Normal  

The dataset was split using stratified sampling into training, validation, and test sets.

---

# Model Architectures

## Standard CNN

- Conv2D layers with increasing filters  
- MaxPooling layers  
- Flatten layer  
- Dense layer with dropout  
- Softmax output  

---

## Depthwise Separable CNN (DWS)

- MobileNetV2 backbone (pretrained on ImageNet)  
- Depthwise separable convolutions  
- Global Average Pooling  
- Dense layer with Batch Normalization  
- Dropout for regularization  
- Softmax output layer  

---

# Experiments

## Standard CNN

### Experiment A — Simple CNN
- Trained on original dataset  
- Observed bias toward majority class  

---

### Experiment B — CNN with Class Weights
- Applied class weights to address imbalance  
- Slight improvement in minority class handling  

---

### Experiment C — CNN with SMOTE
- Applied SMOTE on flattened images  
- Improved balance but introduced unrealistic samples  

---

## Depthwise Separable CNN (DWS)

### Experiment A — Transfer Learning Baseline
- Used MobileNetV2 with frozen layers  
- Faster convergence compared to standard CNN  

---

### Experiment B — Transfer Learning + Class Weights
- Combined transfer learning with weighted loss  
- Better handling of imbalance  

---

### Experiment C — Augmented Training (No SMOTE)
- Used strong data augmentation  
- Avoided synthetic image generation  
- Provided more stable and realistic training  

---

# Key Findings

- Class imbalance significantly impacts performance  
- Class weights alone provide limited improvement  
- SMOTE can improve balance but is **not ideal for image data** as it distorts spatial features  
- Transfer learning significantly improves learning efficiency  
- Depthwise separable CNNs are:
  - More computationally efficient  
  - Better suited for small medical datasets  
- Data augmentation is a more reliable alternative to synthetic oversampling for images  

---

# Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- Scikit-learn  
- Matplotlib  
- Kaggle  

---

# How to Run the Project

## 1. Install dependencies

```bash
pip install tensorflow opencv-python scikit-learn matplotlib
