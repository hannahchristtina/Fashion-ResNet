# 👕 Transfer Learning with ResNet for Fashion Image Classification

## 💡 Overview
This project focuses on high-performance apparel classification by implementing **Transfer Learning** using the pre-trained **ResNet50** Convolutional Neural Network (CNN). We adapt this state-of-the-art architecture to the Fashion MNIST dataset, aiming for high accuracy and robust real-world usage.

The key innovation is the inclusion of a **confidence-based warning system** that suggests potential multi-item input when the model is uncertain, providing immediate feedback to the user.

---

## 🎯 Project Objective
To build and train a deep learning model that efficiently and accurately classifies fashion apparel into 10 categories, leveraging the feature extraction power of ResNet50 while ensuring reliable input processing through a low-confidence detection mechanism.

---

## ✨ Key Features
- **Transfer Learning with ResNet50:** Utilizes a ResNet base pre-trained on ImageNet, freezing its layers and training only a new classification head.
- **Enhanced Accuracy:** Achieved by training on an **expanded dataset subset** (500 images per class) and increasing training epochs.
- **Multi-Item Input Detection:** Implements a custom function to check the prediction confidence against an **80% threshold**. If confidence is low, a specific message is delivered suggesting the image may contain multiple items.
- **Data Augmentation:** Uses real-time image transformations (rotation, shifting, flipping) to improve model generalization.
- **Interactive Prediction:** Includes a script for easy, live prediction by uploading an image in the Colab environment.

---

## 🧩 Technologies Used
| Technology | Purpose |
|-------------|----------|
| **Python** | Programming Language |
| **TensorFlow / Keras** | Deep Learning Framework; used for implementing ResNet50 and training the classification model. |
| **NumPy, Pandas** | Data handling and numerical operations. |
| **OpenCV (`cv2`)** | Preprocessing: Converting grayscale Fashion MNIST images to 3-channel RGB and resizing to $224 \times 224$ for ResNet. |
| **Matplotlib** | Data Visualization and image display during prediction. |
| **Google Colab** | GPU-accelerated training environment. |

---

## 📂 Dataset
The model is trained on an **expanded subset** of the **Fashion MNIST** dataset.
It classifies images into 10 distinct apparel categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## ⚙️ How to Run the Project

The project is designed as a single, sequential Jupyter Notebook (`Untitled2.ipynb`) to be executed in Google Colab.

1.  **Clone this repository** (or download the notebook file):
    ```bash
    git clone [YOUR-GITHUB-REPO-LINK]
    ```

2.  **Open in Colab:** Upload and open the notebook file in Google Colab.

3.  **Enable GPU:** Go to `Runtime` -> `Change runtime type` and select **T4 GPU** for faster training.

4.  **Execute Cells:** Run all cells sequentially. The code will automatically download the Fashion MNIST data, preprocess it, train the ResNet model (Phase 1 & 2), and save it (Phase 3).

5.  **Interactive Prediction:** Execute the final cell (Phase 4). It will display a file chooser, allowing you to upload any image for instant classification and confidence feedback.
