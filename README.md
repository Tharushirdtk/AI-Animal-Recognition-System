# 🐾 AI Animal Recognition System

An end-to-end **AI/ML image classification project** that detects and identifies animals from images using **Deep Learning (CNN + Transfer Learning)**. This project covers the full ML product lifecycle: dataset preparation, model training, backend API development, and frontend integration.

---

## 📌 Project Overview

This system allows users to upload an image of an animal and receive:

* The **predicted animal name (in English)**
* The **confidence score**
* Probability distribution across all supported animals

The model is trained on the **Animals-10 dataset** and deployed via a **Flask REST API**, making it easy to integrate with any frontend.

---

## 🎯 Objectives

* Build a real-world AI product, not just a model
* Learn image preprocessing and CNN-based classification
* Apply **transfer learning using MobileNetV2**
* Serve ML predictions through a backend API
* Handle real-world issues like label mismatches and confidence uncertainty

---

## 🧠 Supported Animal Classes

The original dataset labels are in Italian, but the system outputs **English labels**:

| Dataset Label | Output Label |
| ------------- | ------------ |
| cane          | dog          |
| gatto         | cat          |
| cavallo       | horse        |
| farfalla      | butterfly    |
| elefante      | elephant     |
| mucca         | cow          |
| pecora        | sheep        |
| pollo         | chicken      |
| scoiattolo    | squirrel     |
| ragno         | spider       |

---

## 🏗️ Project Architecture

```
AI-Animal-Recognition/
│
├── dataset/
│   ├── train/
│   │   ├── cane/
│   │   ├── gatto/
│   │   └── ...
│   └── test/
│       ├── cane/
│       ├── gatto/
│       └── ...
│
├── backend/
│   ├── app.py
│   ├── animal_model.h5
│   └── class_labels.json
│
├── train_model.py
└── README.md
```

---

## ⚙️ Technologies Used

### 🔹 Machine Learning

* Python
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* NumPy, Pillow, SciPy

### 🔹 Backend

* Flask
* Flask-CORS
* REST API

### 🔹 Dataset

* **Animals-10 Dataset** (Kaggle)

---

## 📂 Dataset Preparation

1. Download the *Animals-10* dataset from Kaggle
2. Dataset contains raw images grouped by animal class
3. Images are split into training and testing sets using an automated script

Split ratio:

* **80% Training**
* **20% Testing**

---

## 🏋️ Model Training

The model is trained using **MobileNetV2** with transfer learning:

* Pre-trained on ImageNet
* Base layers frozen
* Custom classification head added
* Softmax activation for multi-class output

### Training highlights:

* Image resizing: `224 x 224`
* Normalization: `1/255`
* Data augmentation (rotation, zoom, flip)

During training:

* Model learns patterns, not memorization
* Overfitting is avoided
* Accuracy typically reaches **80–90%**

The trained model is saved as:

```
animal_model.h5
```

Class labels are saved separately as:

```
class_labels.json
```

---

## 🚀 Backend API

The Flask backend exposes a prediction endpoint.

### 🔹 Endpoint

```
POST /predict
```

### 🔹 Request

* Form-data
* Key: `file`
* Value: image file (jpg/png)

### 🔹 Response (Example)

```json
{
  "prediction": "horse",
  "confidence": 0.87,
  "all_probabilities": {
    "horse": 0.87,
    "cow": 0.09,
    "sheep": 0.02
  }
}
```

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow flask flask-cors numpy pillow scipy
```

### 3️⃣ Train the Model

```bash
python train_model.py
```

This generates:

* `animal_model.h5`
* `class_labels.json`

### 4️⃣ Run Backend Server

```bash
cd backend
python app.py
```

Server runs at:

```
http://127.0.0.1:5000
```

---

## ❗ Important Notes

* The model does **not guarantee 100% accuracy**
* Predictions are probabilistic
* Some animals look visually similar
* Confidence score indicates certainty level

This behavior reflects **real-world AI systems**.

---

## 📈 Future Improvements

* Show top-3 predictions in UI
* Add confidence threshold handling
* Fine-tune base CNN layers
* Improve dataset balance
* Deploy using Docker or cloud platform

---

## 🎓 Learning Outcomes

Through this project, we learned:

* How to build an end-to-end AI product
* Practical image classification with CNNs
* Transfer learning advantages
* Model deployment using Flask
* Handling real ML uncertainty in predictions

---

## ✅ Conclusion

This project successfully demonstrates a **complete AI product pipeline**, from raw data to a deployed prediction service. While not perfect, the system behaves realistically and meets industry expectations for ML-based applications.

---

✨ *Built as part of an AI/ML Course Assignment*
