# Transfer Learning with ResNet for Fashion Image Classification

## Overview
This project focuses on high-accuracy fashion image classification using **Transfer Learning** with **ResNet**. It adapts the pre-trained model to the Fashion MNIST dataset, achieving high performance.

---

## Objective
To build and train a robust deep learning model that accurately classifies 10 apparel categories using ResNet and provides reliable prediction.

---

## Key Features
- **Transfer Learning (ResNet):** Uses a pre-trained ResNet base for superior feature extraction.
- **High Accuracy:** Achieved through expanded training data (500 images/class) and extended epochs (10).
- **Data Augmentation:** Utilizes image transformations to enhance model generalization.

---

## Technologies Used
| Technology | Purpose |
|-------------|----------|
| **Python** | Programming Language |
| **TensorFlow / Keras** | Deep Learning Framework (ResNet implementation) |
| **NumPy** | Data handling |
| **OpenCV (`cv2`)** | Image preprocessing for ResNet input |
| **Google Colab** | GPU Training Environment |

---

## Dataset
The model is trained on an **expanded subset** of the **Fashion MNIST** dataset, classifying images into 10 categories (e.g., T-shirts, Trousers, Coats, Bags).

---

## How to Run the Project
1. Open the notebook in Google Colab with GPU enabled.
2. Run the cells sequentially from Phase 1 to Phase 4.
3. The final cell will prompt for an image upload for live prediction.
