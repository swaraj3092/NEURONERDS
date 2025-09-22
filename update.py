import os
import warnings
import sqlite3
import hashlib
from datetime import datetime
import io

# ------------------ SUPPRESS WARNINGS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ------------------ LIBRARIES ------------------
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json
import requests
import urllib.parse
import base64

# ---------------------------- DB FUNCTIONS ----------------------------
DB_FILE = "new/users.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_user(email, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hash_password(password)))
    result = c.fetchone()
    conn.close()
    return result is not None

def add_user(email, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def reset_password(email, new_password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET password=? WHERE email=?", (hash_password(new_password), email))
    conn.commit()
    conn.close()

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide", page_icon="cow.png")

# ---------------------------- STYLING ----------------------------
st.markdown("""
<style>
/* ---------- Body and Background ---------- */
body {background: linear-gradient(135deg, #1e3c72, #2a5298); font-family: 'Segoe UI', sans-serif; color: #f0f2f6;}
.main .block-container {background-color: rgba(255,255,255,0.05); padding: 2rem; border-radius: 15px; backdrop-filter: blur(10px); box-shadow: 0 10px 30px rgba(0,0,0,0.5);}
h1,h2,h3 {color:#fff; text-align:center; animation: fadeInDown 1s ease-out;}
.stButton>button {background:linear-gradient(90deg,#ff758c,#ff7eb3); color:white; font-weight:bold; border-radius:12px; padding:12px 28px; border:none; cursor:pointer; transition:all 0.3s ease;}
.stButton>button:hover {transform: scale(1.05); box-shadow:0 8px 20px rgba(255,255,255,0.4);}
div[role="tablist"] button {background: linear-gradient(90deg,#00c6ff,#0072ff); color:white; font-weight:bold; border-radius:10px; margin:0 5px; transition:all 0.3s ease;}
div[role="tablist"] button:focus {outline:none;}
div[role="tablist"] button:hover {transform:scale(1.05);}
.stImage img {border-radius:12px; border:3px solid #00c6ff; transition: transform 0.3s ease;}
.stImage img:hover {transform: scale(1.05);}
@keyframes fadeInDown {from {opacity:0; transform:translateY(-20px);} to {opacity:1; transform:translateY(0);}}
</style>
""", unsafe_allow_html=True)

# ---------------------------- LOAD MODEL ----------------------------
@st.cache_resource
def load_model():
    return TFSMLayer("models/animal_classifier_savedmodel", call_endpoint="serving_default")

@st.cache_data
def load_classes():
    try:
        with open("models/model.json", "r") as f:
            classes_dict = json.load(f)
        return [classes_dict[str(k)] for k in range(len(classes_dict))]
    except FileNotFoundError:
        st.error("Class names file not found. Please ensure 'models/model.json' exists.")
        return []

model = load_model()
classes = load_classes()

# ---------------------------- SESSION STATE ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = "User"
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------- LOGIN PAGE ----------------------------
if not st.session_state.logged_in:
    st.write("")  # spacing
    st.image("cow.png", width=120)
    st.markdown("<h2 style='text-align:center; color:#f0f2f6;'>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ccc;'>Sign in to continue</p>", unsafe_allow_html=True)

    email = st.text_input("Email", placeholder="user@example.com")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Login"):
            if check_user(email, password):
                st.session_state.logged_in = True
                st.session_state.user_name = email
                st.success(f"Welcome {email}!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials.")
    with col2:
        if st.button("Forgot password?"):
            new_pass = st.text_input("Enter new password for reset", type="password", key="reset_pass")
            if st.button("Reset Password", key="reset_btn"):
                reset_password(email, new_pass)
                st.success("Password reset successful!")

    st.markdown("<p style='text-align:center;'>Need an account? <a href='#'>Sign up below</a></p>", unsafe_allow_html=True)

    st.markdown("### Register New Account")
    new_email = st.text_input("New Email", key="reg_email")
    new_password = st.text_input("New Password", type="password", key="reg_pass")
    if st.button("Register", key="reg_btn"):
        if add_user(new_email, new_password):
            st.success("Account created successfully! You can now login.")
        else:
            st.error("Email already exists. Try a different one.")

# ---------------------------- MAIN APP ----------------------------
if st.session_state.logged_in:
    st.markdown(f"<h2>Welcome, {st.session_state.user_name}!</h2>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🖼️ Classifier", "📊 History"])

    with tab1:
        st.image("cow.png", width=80)
        input_method = st.radio("Select input method:", ["📁 Upload Image", "📸 Use Camera"])
        input_file = None
        if input_method == "📁 Upload Image":
            input_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        elif input_method == "📸 Use Camera":
            input_file = st.camera_input("Capture an image")

        if input_file:
            img = Image.open(input_file).convert("RGB")
            st.image(img, use_column_width=True)
            img_array = np.array(img.resize((128,128)), dtype=np.float32)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            with st.spinner("Analyzing... 🔍"):
                try:
                    pred = model(tf.constant(img_array, dtype=tf.float32))
                    if isinstance(pred, dict):
                        if "dense_1" in pred: pred = pred["dense_1"]
                        else: pred = list(pred.values())[0]
                    if hasattr(pred, "numpy"): pred = pred.numpy()
                    pred = np.array(pred[0])
                    if pred.size == 0: st.error("Model returned empty prediction.")
                    else:
                        top3 = np.argsort(pred)[-3:][::-1]
                        cols = st.columns(3)
                        for col, i in zip(cols, top3):
                            with col: st.metric(label=classes[int(i)], value=f"{pred[i]*100:.2f}%")

                        if st.checkbox("Show all predictions"):
                            st.markdown("---")
                            left_col, right_col = st.columns(2)
                            sorted_idx = np.argsort(pred)[::-1]
                            half = len(sorted_idx)//2
                            for i in sorted_idx[:half]: left_col.markdown(f"**{classes[int(i)]}:** {pred[i]*100:.4f}%")
                            for i in sorted_idx[half:]: right_col.markdown(f"**{classes[int(i)]}:** {pred[i]*100:.4f}%")

                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        st.session_state.history.append({
                            "image": image_bytes,
                            "predictions": pred.tolist(),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with tab2:
        st.markdown("<h2>📊 Prediction History</h2>", unsafe_allow_html=True)
        if len(st.session_state.history) == 0:
            st.info("No history yet.")
        else:
            for entry in reversed(st.session_state.history):
                image_base64 = base64.b64encode(entry["image"]).decode("utf-8")
                top3_idx = np.argsort(entry["predictions"])[-3:][::-1]
                top3_text = ", ".join([f"{classes[int(i)]}: {entry['predictions'][i]*100:.2f}%" for i in top3_idx])
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding:15px; border-radius:15px; display:flex; align-items:center; margin-bottom:15px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);'>
                    <img src="data:image/png;base64,{image_base64}" width="80" style="border-radius:12px; margin-right:15px;">
                    <div><b>Time:</b> {entry['timestamp']}<br><b>Top 3 Predictions:</b> {top3_text}</div>
                </div>
                """, unsafe_allow_html=True)
