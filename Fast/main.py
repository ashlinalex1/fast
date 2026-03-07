from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import joblib
from utils import preprocess_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Medical AI API")

# Configure CORS so the Next.js frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev. In prod, specify the localhost/domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load Models
# =========================

xray_model = tf.keras.models.load_model("models/xray_model.h5", compile=False)
mri_model = tf.keras.models.load_model("models/mri_model.h5", compile=False)
ultrasound_model = tf.keras.models.load_model("models/ultrasound_model.h5", compile=False)

symptom_model = joblib.load("models/symptom_model.pkl")

# =========================
# Class Labels
# =========================

xray_classes = ["covid","normal","pneumonia","tuberculosis"]

mri_classes = ["glioma","meningioma","notumor","pituitary"]

ultrasound_classes = ["benign","malignant","normal"]

# =========================
# X-ray Prediction
# =========================

@app.post("/predict/xray")

async def predict_xray(file: UploadFile = File(...)):

    contents = await file.read()
    img = preprocess_image(contents)

    prediction = xray_model.predict(img)

    idx = np.argmax(prediction)

    return {
        "prediction": xray_classes[idx],
        "confidence": float(prediction[0][idx])
    }

# =========================
# MRI Prediction
# =========================

@app.post("/predict/mri")

async def predict_mri(file: UploadFile = File(...)):

    contents = await file.read()
    img = preprocess_image(contents)
    prediction = mri_model.predict(img)

    idx = np.argmax(prediction)

    return {
        "prediction": mri_classes[idx],
        "confidence": float(prediction[0][idx])
    }

# =========================
# Ultrasound Prediction
# =========================

@app.post("/predict/ultrasound")

async def predict_ultrasound(file: UploadFile = File(...)):

    contents = await file.read()
    img = preprocess_image(contents)

    prediction = ultrasound_model.predict(img)

    idx = np.argmax(prediction)

    return {
        "prediction": ultrasound_classes[idx],
        "confidence": float(prediction[0][idx])
    }

# =========================
# Symptom Prediction
# =========================

@app.post("/predict/symptoms")

async def predict_symptoms(symptoms: dict):

    features = np.array([symptoms["vector"]])

    prediction = symptom_model.predict(features)

    return {
        "prediction": prediction[0]
    }