# 👚 Transfer learning with ResNet for fashion image classification

## 💡 Overview

This project implements a high-performance fashion apparel classification system using **Transfer Learning** with the powerful **ResNet50** Convolutional Neural Network (CNN) architecture. By utilizing a model pre-trained on the vast ImageNet dataset, we achieve **high classification accuracy** on fashion items from the **Fashion MNIST** dataset with efficient training.

A key enhancement is the inclusion of **confidence-based input validation** to detect ambiguity.

---

## 🎯 Project Objective

To build a robust and highly accurate deep learning model for classifying 10 categories of fashion apparel, primarily focusing on:
1.  Leveraging the pre-extracted features of **ResNet50** for superior performance.
2.  Implementing a mechanism to provide a **warning message** when an image is classified with low confidence (likely multiple items or ambiguity).

---

## ✨ Key Features

* **Transfer Learning with ResNet50**: Uses the state-of-the-art ResNet50 base model, freezing its weights and training only a new classification head.
* **Enhanced Accuracy**: Achieved by training on an **expanded dataset subset** (500 images per class) and increasing the training epochs to 10.
* **Confidence-Based Input Check**: Implements a confidence threshold (e.g., 80%) to warn the user that an image might contain **multiple fashion items** if the prediction is not strong.
* **Data Preprocessing and Augmentation**: Essential techniques used to format low-resolution Fashion MNIST images for ResNet and to improve model generalization.
* **Model Evaluation**: Tracking accuracy and loss metrics throughout training.

---

## 🧩 Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python** | Programming Language |
| **TensorFlow / Keras** | Deep Learning Framework; used for implementing ResNet50 and training. |
| **NumPy, Pandas** | Data preparation and handling. |
| **OpenCV (`cv2`)** | Image manipulation (converting grayscale $28 \times 28$ to $224 \times 224$ RGB). |
| **Matplotlib** | Data Visualization and image display. |
| **Google Colab** | Primary environment for GPU-accelerated model training. |

---

## 📂 Dataset

The project utilizes an expanded subset of the **Fashion MNIST** dataset, which consists of $28 \times 28$ grayscale images of the following 10 fashion product categories:

* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

---

## ⚙️ How to Run the Project

The entire workflow is contained within the Jupyter Notebook (`Untitled2.ipynb`), which is optimized for **Google Colab**.

1.  **Clone this repository**
    ```bash
    git clone [Your Repository URL Here]
    ```
2.  **Open the Notebook**
    Upload and open `Untitled2.ipynb` in Google Colab.
3.  **Enable GPU**
    Ensure the runtime type is set to **GPU** (`Runtime` -> `Change runtime type`).
4.  **Execute Cells**
    Run all cells sequentially. The training process (Phase 2) will run for 10 epochs.
5.  **Interactive Prediction**
    The final cell (Phase 4) will prompt you to upload a test image and will display the classified item, its confidence, and the **multi-item warning** if confidence is low.

***
