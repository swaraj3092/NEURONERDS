import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json

# ----------------------------
# Load TFSMLayer model
# ----------------------------
model = TFSMLayer("models/animal_classifier_savedmodel", call_endpoint="serving_default")

# ----------------------------
# Load class names from JSON
# ----------------------------
with open("models/model.json", "r") as f:
    classes_dict = json.load(f)

classes = [classes_dict[str(k)] for k in range(len(classes_dict))]

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide")

# ----------------------------
# Modern CSS Styling
# ----------------------------
st.markdown("""
<style>
/* Background Gradient */
body {
    background: linear-gradient(120deg, #f6d365, #fda085);
    font-family: 'Arial', sans-serif;
}

/* Headers */
h1, h2, h3 {
    color: #2c3e50;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

/* Buttons */
.stButton>button {
    background-color: #ff6f61;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 28px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #ff3b2f;
    transform: scale(1.05);
}

/* Prediction Cards */
.pred-card {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0px 8px 16px rgba(0,0,0,0.2);
    text-align: center;
    transition: transform 0.3s ease;
}
.pred-card:hover {
    transform: translateY(-5px);
}

/* Camera Input Styling */
.css-1r6slb0 input {
    border-radius: 12px;
}

/* Progress bars */
.progress-bar {
    background-color:#2ecc71; 
    height:18px; 
    border-radius:10px;
    transition: width 1s ease-in-out;
}
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
        if username == "bpa" and password == "batch":
            st.session_state.logged_in = True
            st.success("Login Successful! You can now use the app below.")
        else:
            st.error("Invalid credentials. Try again.")

# ----------------------------
# Main App
# ----------------------------
if st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Choose an input method to see AI prediction instantly!</p>", unsafe_allow_html=True)

    # Input choice
    input_method = st.radio("Select input method:", ["Upload Image", "Use Camera"])

    uploaded_file = None
    camera_file = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    else:
        camera_file = st.camera_input("Capture an image using your camera")

    input_file = uploaded_file if uploaded_file else camera_file

    if input_file:
        # Display image
        img = Image.open(input_file).convert("RGB")
        st.image(img, caption="Input Image", use_column_width=True)

        # Preprocess
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing... 🔍"):
            prediction_output = model(tf.constant(img_array, dtype=tf.float32))

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

                st.markdown("<h2 style='text-align:center;'>Top Predictions:</h2>", unsafe_allow_html=True)

                # Display in columns with animated cards
                cols = st.columns(3)
                for col, i in zip(cols, top3_idx):
                    with col:
                        st.markdown(f"""
                        <div class="pred-card">
                        <h3>{classes[i]}</h3>
                        <p>{prediction[i]*100:.2f}% confidence</p>
                        <div class="progress-bar" style="width:{int(prediction[i]*100)}%"></div>
                        </div>
                        """, unsafe_allow_html=True)

                # Optional: show full predictions
                show_all = st.checkbox("Show all class predictions")
                if show_all:
                    st.markdown("<h2>All Class Predictions:</h2>", unsafe_allow_html=True)
                    sorted_idx = np.argsort(prediction)[::-1]
                    for i in sorted_idx:
                        st.markdown(f"**{classes[i]}:** {prediction[i]*100:.4f}%")
