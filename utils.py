import cv2
import numpy as np
import tensorflow as tf
import joblib

# Load models
disease_model = tf.keras.models.load_model("trained_model.h5")
yield_model = joblib.load("yield_model.pkl")

def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

classes = [
    "Healthy",
    "Red Rot",
    "Smut",
    "Mosaic Virus",
    "Leaf Scald",
    "Top Shoot Borer",
    "Gumming",
    "Grassy Shoot",
    "Ratoon Stunting",
    "Yellow Leaf",
    "Other Disease"
]

def predict_disease(image):
    img = preprocess_image(image)
    pred = disease_model.predict(img)
    class_index = np.argmax(pred)
    return classes[class_index]  # Return disease name instead of number


def predict_yield(data):
    return yield_model.predict([data])[0]
