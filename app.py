# app.py

import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load model
model = pickle.load(open("knn_digit_model.pkl", "rb"))

st.title("🧠 Digit Recognition using KNN")

uploaded_file = st.file_uploader("Upload a digit image (0-9)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    image = image.resize((8, 8))  # resize to match training data
    
    img_array = np.array(image)
    
    # Normalize (important!)
    img_array = 16 - (img_array / 16)
    
    img_flat = img_array.flatten().reshape(1, -1)

    prediction = model.predict(img_flat)

    st.image(image, caption="Processed Image", width=150)
    st.write("### Predicted Digit:", prediction[0])