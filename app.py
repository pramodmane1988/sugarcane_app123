import streamlit as st
import cv2
import numpy as np
from utils import predict_disease, predict_yield

st.title("🌾 Smart Sugarcane Monitoring System")

# Image Upload
st.header("Disease Detection")
uploaded_file = st.file_uploader("Upload Sugarcane Leaf Image", type=["jpg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    disease = predict_disease(image)
    st.success(f"Predicted Disease Class: {disease}")

# Yield Prediction
st.header("Yield Prediction")

soil_moisture = st.number_input("Soil Moisture")
pH = st.number_input("Soil pH")
N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
disease_severity = st.number_input("Disease Severity (%)")

if st.button("Predict Yield"):
    data = [soil_moisture, pH, N, P, K, temp, humidity, disease_severity]
    y = predict_yield(data)
    st.success(f"Predicted Sugarcane Yield: {y:.2f} tons/hectare")
