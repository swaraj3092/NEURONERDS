import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from tensorflow.keras.preprocessing import image
import json

# ----------------------------
# Load TFSMLayer model
# ----------------------------
model = TFSMLayer("models/animal_classifier_savedmodel", call_endpoint="serving_default")

# Load class names
with open("models/model.json", "r") as f:
    classes = json.load(f)

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide")

# ----------------------------
# CSS Styling
# ----------------------------
st.markdown("""
<style>
body { background: linear-gradient(135deg,#f5f7fa,#c3cfe2); font-family: 'Arial', sans-serif; }
h1,h2,h3 { color: #2c3e50; }
.stButton>button { background-color:#ff6f61; color:white; font-weight:bold; border-radius:10px; padding:10px 24px; }
.stButton>button:hover { background-color:#ff3b2f; transition:0.3s; }
.progress-bar { background-color:#2ecc71; height:18px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Login Page
# ----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🔒 BPA Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")
    
    if login_btn:
        if username == "bpa" and password == "batch":  # simple authentication
            st.session_state.logged_in = True
            st.success("Login Successful! Redirecting...")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Try again.")

# ----------------------------
# Main App
# ----------------------------
else:
    st.markdown("<h1 style='text-align:center;'>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload an image to see the AI prediction instantly!</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        # Display image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)

        with st.spinner("Analyzing... 🔍"):
            # Call TFSMLayer
            prediction_output = model(tf.constant(img_array, dtype=tf.float32))

            # Safely extract prediction
            if isinstance(prediction_output, tf.Tensor):
                prediction = prediction_output.numpy()[0]
            elif isinstance(prediction_output, (list, tuple)) and isinstance(prediction_output[0], tf.Tensor):
                prediction = prediction_output[0].numpy()[0]
            elif isinstance(prediction_output, dict):
                key = list(prediction_output.keys())[0]
                prediction = prediction_output[key].numpy()[0]
            else:
                st.error(f"Cannot handle prediction output of type {type(prediction_output)}")
                prediction = None

            if prediction is not None:
                # Top 3 predictions
                top3_idx = prediction.argsort()[-3:][::-1]

                st.markdown("<h2>Top Predictions:</h2>", unsafe_allow_html=True)
                for i in top3_idx:
                    st.markdown(f"**{classes[i]}:** {prediction[i]*100:.2f}%")
                    st.progress(int(prediction[i]*100))
