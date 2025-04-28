# 🧠 Prediction of Brain Tumor with CNN

## 🔍 Overview
This project applies **Convolutional Neural Networks (CNNs)** to classify brain tumor images and predict tumor presence. The model is trained on medical imaging datasets to assist in automated diagnosis.

## 🚀 Features
- **Deep Learning Model**: Uses CNN for accurate tumor classification.
- **Data Preprocessing**: Handles image resizing, normalization, and augmentation.
- **Model Serialization**:
  - Save model in **H5 (`.h5`)** format for TensorFlow/Keras compatibility.
  - Save model in **Pickle (`.pkl`)** format for Scikit-learn use cases.
- **Web Deployment**: Can be integrated with Flask or Streamlit for real-world applications.

## 🏗️ Tech Stack
- **Python** (NumPy, Pandas)
- **Deep Learning** (TensorFlow, Keras, OpenCV)
- **Machine Learning** (Scikit-learn, Pickle)
- **Data Visualization** (Matplotlib, Seaborn)

## 📂 Project Structure
```
📦 Prediction-Brain-Tumor
├── 📁 data
│   ├── brain_tumor_images/   # Dataset directory
├── 📁 models
│   ├── brain_tumor.h5   # Trained model (H5 format)
│   ├── brain_tumor.pkl   # Trained model (Pickle format)
├── 📁 flask_app
│   ├── app.py   # Flask application for deployment
├── training_script.ipynb   # Jupyter Notebook for training
├── requirements.txt   # Dependencies
├── README.md   # Documentation
```

## 🧠 Model Training & Saving
### 🔹 **Saving the Model in H5 Format**
```python
model.save("/content/drive/MyDrive/data/brain_tumor.h5")
```

### 🔹 **Saving the Model in Pickle Format**
```python
import pickle

with open("/content/drive/MyDrive/data/brain_tumor.pkl", "wb") as f:
    pickle.dump(model, f)
```

### 🔹 **Loading the Pickle Model**
```python
with open("/content/drive/MyDrive/data/brain_tumor.pkl", "rb") as f:
    loaded_model = pickle.load(f)
```

## ⚙️ Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/iseptianto/prediction-of-brain-tumor-with-cnn.git
   ```
2. Navigate to the project folder:
   ```bash
   cd prediction-of-brain-tumor-with-cnn
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```bash
   jupyter notebook training_script.ipynb
   ```


## 📌 Future Improvements
- Implement **Transfer Learning** using models like **ResNet** or **EfficientNet**.
- Integrate **Grad-CAM** for visual explanations of model decisions.
- Enhance **real-time inference** using optimized model deployment strategies.

---
