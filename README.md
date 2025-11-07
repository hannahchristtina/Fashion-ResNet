# Transfer Learning with ResNet for Fashion Image Classification

## Overview
This project focuses on high-accuracy fashion apparel classification using **Transfer Learning** with the **ResNet50** CNN architecture. It adapts the pre-trained model to the Fashion MNIST dataset, achieving high performance while including a critical **confidence-based warning system** for potential multi-item input.

---

## 🎯 Project Objective
To build and train a robust deep learning model that accurately classifies 10 apparel categories using ResNet50 and provides reliable feedback on ambiguous or multi-item images.

---

## ✨ Key Features
- **Transfer Learning (ResNet50):** Uses a pre-trained ResNet50 base for superior feature extraction.
- **High Accuracy:** Achieved through expanded training data (500 images/class) and extended epochs (10).
- **Multi-Item Input Detection:** Provides a warning if prediction confidence is below **80%**, suggesting ambiguous or multiple items in the image.
- **Data Augmentation:** Utilizes image transformations to enhance model generalization.

---

## 🧩 Technologies Used
| Technology | Purpose |
|-------------|----------|
| **Python** | Programming Language |
| **TensorFlow / Keras** | Deep Learning Framework (ResNet50 implementation) |
| **NumPy** | Data handling |
| **OpenCV (`cv2`)** | Image preprocessing for ResNet input |
| **Google Colab** | GPU Training Environment |

---

## 📂 Dataset
The model is trained on an **expanded subset** of the **Fashion MNIST** dataset, classifying images into 10 categories (e.g., T-shirts, Trousers, Coats, Bags).

---

## ⚙️ How to Run the Project
1. **Clone this repository**:
    ```bash
    git clone [YOUR-GITHUB-REPO-LINK]
    ```

2. **Run in Colab:** Open the `Untitled2.ipynb` notebook and ensure the runtime is set to **GPU**.
3. **Execute:** Run all cells sequentially. The final cell will prompt you to upload an image for live prediction.
