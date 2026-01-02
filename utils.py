import cv2
import numpy as np
import tensorflow as tf
import joblib

# Load models
disease_model = tf.keras.models.load_model("sugarcane_disease_model.h5")
yield_model = joblib.load("yield_model.pkl")

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_disease(image):
    img = preprocess_image(image)
    pred = disease_model.predict(img)
    return np.argmax(pred)

def predict_yield(data):
    return yield_model.predict([data])[0]
