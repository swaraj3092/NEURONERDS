import os
import warnings
import sqlite3
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json
import requests
import urllib.parse
import base64

# ------------------ SUPPRESS WARNINGS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ------------------ DATABASE INIT ------------------
DB_PATH = "new/users.db"  # Your db inside 'new' folder

def init_db():
    os.makedirs("new", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    # Insert default user if not exists
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", ("bpa", "batch123"))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

init_db()

# ------------------ PAGE STYLING ------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    font-family: 'Segoe UI', sans-serif;
    color: #f0f2f6;
}
.main .block-container {
    background-color: rgba(255,255,255,0.05);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
h1,h2,h3 { color:#fff; text-align:center; animation: fadeInDown 1s ease-out; }
.stButton>button {
    background: linear-gradient(90deg,#ff758c,#ff7eb3);
    color:white;font-weight:bold;border-radius:12px;padding:12px 28px;border:none;
    cursor:pointer;transition: all 0.3s ease;
}
.stButton>button:hover { transform:scale(1.05); box-shadow:0 8px 20px rgba(255,255,255,0.4); }
div[role="tablist"] button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;font-weight:bold;border-radius:10px;margin:0 5px;transition: all 0.3s ease;
}
div[role="tablist"] button:focus { outline:none; }
div[role="tablist"] button:hover { transform: scale(1.05); }
.stImage img { border-radius:12px;border:3px solid #00c6ff;transition:transform 0.3s ease;}
.stImage img:hover { transform: scale(1.05); }
@keyframes fadeInDown { from {opacity:0; transform:translateY(-20px);} to {opacity:1; transform:translateY(0);} }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="🐾 Animal Classifier", layout="wide", page_icon="cow.png")

# ------------------ LOAD MODEL & CLASSES ------------------
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
        st.error("Class names file not found.")
        return []

model = load_model()
classes = load_classes()

# ------------------ SESSION STATE ------------------
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_name" not in st.session_state: st.session_state.user_name = "User"
if "history" not in st.session_state: st.session_state.history = []

# ------------------ HELPER DB FUNCTIONS ------------------
def verify_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    user = c.fetchone()
    conn.close()
    return user is not None

def register_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?,?)", (email,password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def reset_password(email, new_password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    if c.fetchone():
        c.execute("UPDATE users SET password=? WHERE email=?", (new_password,email))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

# ------------------ LOGIN / REGISTER UI ------------------
tab_login, tab_register, tab_forgot = st.tabs(["Login","Register","Forgot Password"])

with tab_login:
    st.image("cow.png", width=120)
    email = st.text_input("Email", placeholder="user@example.com", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login", key="login_btn"):
        if verify_user(email,password):
            st.session_state.logged_in = True
            st.session_state.user_name = email
            st.success(f"Logged in as {email}")
        else:
            st.error("Invalid credentials.")

with tab_register:
    st.image("cow.png", width=120)
    reg_email = st.text_input("Gmail (must end with @gmail.com)", placeholder="user@gmail.com", key="reg_email")
    reg_pass = st.text_input("Password", type="password", key="reg_pass")
    if st.button("Register", key="reg_btn"):
        if not reg_email.endswith("@gmail.com"):
            st.error("Email must be a Gmail address ending with @gmail.com")
        elif register_user(reg_email, reg_pass):
            st.success("Registration successful! You can now login.")
        else:
            st.error("Email already exists.")

with tab_forgot:
    st.image("cow.png", width=120)
    f_email = st.text_input("Registered Email", placeholder="user@gmail.com", key="forgot_email")
    new_pass = st.text_input("New Password", type="password", key="forgot_pass")
    if st.button("Reset Password", key="forgot_btn"):
        if reset_password(f_email,new_pass):
            st.success("Password updated successfully!")
        else:
            st.error("Email not found.")

# ------------------ MAIN APP ------------------
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
        input_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"]) if input_method=="📁 Upload Image" else st.camera_input("Capture an image")
        if input_file:
            img = Image.open(input_file).convert("RGB")
            st.image(img, use_column_width=True)
            img_array = np.array(img.resize((128,128)), dtype=np.float32)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            with st.spinner("Analyzing... 🔍"):
                try:
                    pred = model(tf.constant(img_array,dtype=tf.float32))
                    if isinstance(pred, dict):
                        pred = list(pred.values())[0]
                    if hasattr(pred,"numpy"): pred = pred.numpy()
                    pred = np.array(pred[0])
                    top3 = np.argsort(pred)[-3:][::-1]
                    cols = st.columns(3)
                    for col,i in zip(cols, top3):
                        col.metric(label=classes[int(i)], value=f"{pred[i]*100:.2f}%")
                    # Save history
                    import io
                    from datetime import datetime
                    buffer = io.BytesIO(); img.save(buffer, format="PNG")
                    st.session_state.history.append({
                        "image": buffer.getvalue(),
                        "predictions": pred.tolist(),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with tab2:
        st.markdown("<h2>📊 Prediction History</h2>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("No history yet.")
        else:
            for entry in reversed(st.session_state.history):
                img_b64 = base64.b64encode(entry["image"]).decode("utf-8")
                top3_idx = np.argsort(entry["predictions"])[-3:][::-1]
                top3_text = ", ".join([f"{classes[int(i)]}: {entry['predictions'][i]*100:.2f}%" for i in top3_idx])
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding:15px; border-radius:15px;
                            display:flex; align-items:center; margin-bottom:15px; box-shadow:0 5px 15px rgba(0,0,0,0.3);'>
                    <img src="data:image/png;base64,{img_b64}" width="80" style="border-radius:12px; margin-right:15px;">
                    <div><b>Time:</b> {entry['timestamp']}<br><b>Top 3 Predictions:</b> {top3_text}</div>
                </div>
                """, unsafe_allow_html=True)
