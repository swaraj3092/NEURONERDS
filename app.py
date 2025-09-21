import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# ----------------------------
# Model Load
# ----------------------------
model = tf.keras.models.load_model("models/animal_classifier.keras")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="NeuroNerds Animal Classifier", page_icon="🐾", layout="centered")

# ----------------------------
# Login Page (unchanged)
# ----------------------------
if "user" not in st.session_state:
    st.title("🔐 Welcome to NeuroNerds")
    st.markdown(
        """
        <div style="text-align:center;">
            <h3 style="color:#4CAF50;">Login to Continue</h3>
            <p>Please sign in with your Google account to use the animal classifier.</p>
            <a href="#" target="_self">
                <button style="
                    background-color:#4285F4;
                    color:white;
                    font-size:18px;
                    padding:10px 20px;
                    border:none;
                    border-radius:8px;
                    cursor:pointer;
                ">
                    Continue with Google 🚀
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# ----------------------------
# Classifier UI (after login)
# ----------------------------
st.title("🐾 Animal Type Classifier")
st.write("Upload an image of an animal and let the AI classify it!")

uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔎 Analyzing image... Please wait..."):
        predictions = model(img_array, training=False)
        predicted_class = int(np.argmax(predictions[0]))  # ✅ Fix np.int64 issue

        # Load class names
        with open("models/class_names.json", "r") as f:
            class_names = json.load(f)

    st.success(f"✅ Prediction: **{class_names[predicted_class]}**")
