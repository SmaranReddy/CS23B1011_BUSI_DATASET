# Breast Ultrasound Image Classification using CNN, Class Weights, and SMOTE

## Course
Deep Learning for Medical Images

## Author
**Name:** A Smaran Reddy  
**Roll Number:** CS23B1011  

---

# Project Overview

Breast cancer is one of the leading causes of cancer-related deaths worldwide. Early detection plays a critical role in improving survival rates. Ultrasound imaging is widely used as a non-invasive method to detect abnormalities in breast tissue.

This project develops a **Convolutional Neural Network (CNN)** to classify breast ultrasound images into three categories:

- **Benign**
- **Malignant**
- **Normal**

A key challenge in the dataset is **class imbalance**, where benign samples significantly outnumber malignant and normal samples.

To address this issue, three experimental approaches were evaluated:

1. Simple CNN
2. CNN with Class Weights
3. CNN with SMOTE (Synthetic Minority Oversampling Technique)

---

# Dataset

The project uses the **BUSI (Breast Ultrasound Images) dataset** available on Kaggle.

Dataset Source:  
https://www.kaggle.com/datasets/subhajournal/busi-breast-ultrasound-images

### Dataset Characteristics

| Class | Train | Validation | Description |
|------|------|------|------|
| Benign | 713 | 178 | Non-cancerous tumor |
| Malignant | 337 | 84 | Cancerous tumor |
| Normal | 213 | 53 | No abnormality |

The dataset is split using an **80/20 train-validation split**.

---

# Model Architecture

The CNN architecture used in all experiments consists of:

- Conv2D (32 filters)
- MaxPooling
- Conv2D (64 filters)
- MaxPooling
- Conv2D (128 filters)
- MaxPooling
- Flatten Layer
- Dense Layer (256 neurons)
- Dropout (0.5)
- Softmax Output Layer

Optimizer: **Adam**  
Loss Function: **Categorical Crossentropy**

---

# Experiments

## Experiment A — Simple CNN

The CNN model was trained on the original dataset without addressing class imbalance.

Result:
- Accuracy: **82%**

However, the model showed poor performance on minority classes due to the imbalanced dataset.

---

## Experiment B — CNN with Class Weights

Class weights were applied during training to penalize misclassification of minority classes.

Result:
- Accuracy: **81%**

This approach slightly adjusted the loss function but did not significantly improve minority class detection.

---

## Experiment C — CNN with SMOTE

SMOTE (Synthetic Minority Oversampling Technique) was applied to generate additional synthetic samples for minority classes, creating a more balanced dataset.

Result:
- Accuracy: **93%**

This approach significantly improved model performance.

---

# Performance Comparison

| Experiment | Precision | Recall | F1 Score |
|---|---|---|---|
| Simple CNN | 0.072 | 0.269 | 0.114 |
| CNN + Class Weights | 0.072 | 0.269 | 0.114 |
| CNN + SMOTE | **0.890** | **0.882** | **0.882** |

The results show that **SMOTE dramatically improves classification performance** by balancing the dataset.

---

# Key Findings

- Class imbalance severely impacts CNN performance.
- Class weights alone are not sufficient to address imbalance.
- SMOTE significantly improves precision, recall, and F1-score.
- Balanced datasets allow the model to learn better representations for minority classes.

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- KaggleHub
- Google Colab

---

# How to Run the Project

### 1. Install required libraries

```
pip install tensorflow opencv-python scikit-learn imbalanced-learn kagglehub
```

### 2. Run the notebook

```
BUSI_dataset.ipynb
```

### 3. The notebook performs

- Dataset download
- Image preprocessing
- CNN training
- SMOTE oversampling
- Model evaluation

---

# Conclusion

This project demonstrates the importance of addressing **class imbalance in medical imaging datasets**. While baseline CNN models struggle with minority classes, applying SMOTE significantly improves model performance and leads to more reliable predictions.

---

# Acknowledgements

BUSI Dataset provided by Kaggle.
