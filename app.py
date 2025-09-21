import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

# ----------------------------
# Load model & classes
# ----------------------------
model = load_model("models/animal_classifier.keras")
with open("models/model.json", "r") as f:
    classes = json.load(f)

# ----------------------------
# Page config
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
# Session state for login
# ----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'rerun_needed' not in st.session_state:
    st.session_state.rerun_needed = False

# ----------------------------
# Login Function
# ----------------------------
def login():
    st.markdown("<h1 style='text-align:center;'>🔒 BPA Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")
    
    if login_btn:
        if username == "bpa" and password == "batch":
            st.session_state.logged_in = True
            st.success("Login Successful! Redirecting...")
            st.session_state.rerun_needed = True  # flag for safe rerun
        else:
            st.error("Invalid credentials. Try again.")

# ----------------------------
# Run login or main app
# ----------------------------
if not st.session_state.logged_in:
    login()
    if st.session_state.rerun_needed:
        st.session_state.rerun_needed = False
        st.experimental_rerun()
else:
    # ----------------------------
    # Main App
    # ----------------------------
    st.markdown("<h1 style='text-align:center;'>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload an image to see the AI prediction instantly!</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")  # ensure RGB
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        img = img.resize((224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)/255.0
        
        with st.spinner("Analyzing... 🔍"):
            prediction = model.predict(img_array)[0]
            top3_idx = prediction.argsort()[-3:][::-1]
            
        st.markdown("<h2>Top Predictions:</h2>", unsafe_allow_html=True)
        for i in top3_idx:
            st.markdown(f"{classes[i]}: {prediction[i]*100:.2f}%")
            st.progress(int(prediction[i]*100))
