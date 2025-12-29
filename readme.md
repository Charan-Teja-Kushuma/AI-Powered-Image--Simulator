# AI-Powered Image Similarity Search and Recommendation System 🔍

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview

This project is an **AI-powered visual search engine** capable of finding visually similar images from a large database without using any text tags or metadata.

It leverages **Deep Metric Learning** using a **Triplet Network** architecture. The core model is built on a pre-trained **MobileNetV2** backbone, which learns to map images into a 128-dimensional embedding space where similar items (e.g., two different watches) are clustered closely together.

### 🚀 Key Features
* **Visual Similarity Search:** Upload an image to find the top *k* most visually similar items from the database.
* **Deep Learning Model:** Uses a Siamese-style Triplet Network trained with **Triplet Loss**.
* **Transfer Learning:** Utilizes **MobileNetV2** (pre-trained on ImageNet) for robust feature extraction.
* **Efficient Retrieval:** Implements **k-Nearest Neighbors (k-NN)** for fast embedding search.
* **Custom Dataset Support:** Designed to handle custom, folder-structured image datasets.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation:** NumPy
* **Search Algorithm:** Scikit-learn (`NearestNeighbors`)
* **Visualization:** Matplotlib

---

## 📂 Dataset Structure

The system expects a dataset organized by class folders. For example:

```text
MY_DATASET/
├── watch/
│   ├── watch_01.jpg
│   ├── watch_02.jpg
│   └── ...
├── laptop/
│   ├── laptop_01.jpg
│   └── ...
└── jacket/
    ├── jacket_01.jpg
    └── ...
Note: The code automatically handles (Anchor, Positive, Negative) triplet generation from this directory structure.

🏗️ Model Architecture
The system uses a Triplet Network architecture:
1.Input: Three images are fed into the network:
Anchor (A): Reference image.
Positive (P): Same class as Anchor.
Negative (N): Different class from Anchor.
2.Base Network (Shared Weights): Each image passes through the same MobileNetV2 backbone (frozen) + Custom Head (GlobalAveragePooling -> Dense -> L2 Normalization).
3.Embedding: The network outputs a 128-dimensional vector for each image.
4.Loss Function: The model minimizes Triplet Loss:
L(A, P, N) = \max(d(A, P) - d(A, N) + \alpha, 0)
Where $d$ is the Euclidean distance and $\alpha$ is the margin.

💻 Installation & Usage
1. Clone the Repository git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
2. Install Dependencies pip install tensorflow numpy opencv-python matplotlib scikit-learn
3. Run the Project
The project is currently structured as a Jupyter Notebook / Google Colab file.
1.Open the .ipynb file in Jupyter Notebook or upload it to Google Colab.
2.Upload your dataset zip file (e.g., MY_DATASET.zip) to the environment.
3.Run the cells sequentially to:
Unzip and preprocess data.
Train the Triplet Network.
Generate embeddings.
Visualize search results.

📊 Results
The model was trained for 10 epochs on a custom dataset of 1000+ images.
Training Loss: Converged from 0.22 to ~0.02, indicating successful metric learning.
Qualitative Search Accuracy:
Query: Watch
Retrieved 5/5 relevant "Watch" images with varying styles.
Query: Laptop
Retrieved 5/5 relevant "Laptop" images.

🔮 Future Improvements
[ ] Fast Search: Integrate Faiss (Facebook AI Similarity Search) for scaling to millions of images.
[ ] API Deployment: Wrap the inference logic in a FastAPI or Flask server.
[ ] Frontend: Build a simple React/Streamlit UI for users to upload query images.

