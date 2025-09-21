import os
import warnings
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json
import requests
import urllib.parse

# ------------------ SUPPRESS WARNINGS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide", page_icon="cow.png")

# ---------------------------- Load Model & Classes ----------------------------
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

# ---------------------------- Google OAuth Config ----------------------------
CLIENT_ID = "44089178154-3tfm5sc60qmnc8t5d2p92innn10t3pu3.apps.googleusercontent.com"
CLIENT_SECRET = "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"
REDIRECT_URI = "https://neuronerds.streamlit.app/"
SCOPES = "openid email profile"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USER_INFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"

# ---------------------------- Session State ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = "User"

# ---------------------------- GOOGLE LOGIN HANDLER ----------------------------
if "code" in st.query_params and not st.session_state.logged_in:
    try:
        code = st.query_params["code"][0]
        data = {
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        token_resp = requests.post(TOKEN_URI, data=data).json()
        access_token = token_resp.get("access_token")
        if access_token and not st.session_state.logged_in:
        user_info = requests.get(
            USER_INFO_URI,
            params={"alt": "json"},
            headers={"Authorization": f"Bearer {access_token}"}
        ).json()
        st.session_state.logged_in = True
        st.session_state.user_name = user_info.get("name", "User")
        # Clear the code from query params to avoid rerun loop
        st.experimental_set_query_params()
        st.experimental_rerun()

        else:
            st.error("Failed to login. Please try again.")
    except Exception as e:
        st.error(f"An error occurred during authentication: {e}")

# ---------------------------- Login Page ----------------------------
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align:center;'>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
    
    # Google login button (same tab)
    auth_params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent"
    }
    auth_url = f"{AUTH_URI}?{urllib.parse.urlencode(auth_params)}"
    
    st.markdown(f"""
    <form action="{auth_url}" method="get">
        <button type="submit" style="
            width:100%; padding:12px; font-weight:bold; border-radius:12px;
            background-color:#4285F4; color:white; border:none; cursor:pointer;
            ">Continue with Google 🚀</button>
    </form>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)
    
    # Demo login
    email = st.text_input("Email", placeholder="user@example.com")
    password = st.text_input("Password", type="password")
    if st.button("Login Demo"):
        if email=="user" and password=="demo123":
            st.session_state.logged_in = True
            st.session_state.user_name = "Demo User"
            st.experimental_rerun()
        else:
            st.error("Invalid demo credentials.")

# ---------------------------- Main App ----------------------------
else:
    st.markdown(f"<h2>Welcome, {st.session_state.user_name}!</h2>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    
    input_method = st.radio("Select input method:", ["📁 Upload Image", "📸 Use Camera"])
    input_file = None
    if input_method=="📁 Upload Image":
        input_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    elif input_method=="📸 Use Camera":
        input_file = st.camera_input("Capture an image")
    
    if input_file:
        img = Image.open(input_file).convert("RGB")
        st.image(img, use_container_width=True)
        img_array = np.array(img.resize((128,128)), dtype=np.float32)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        with st.spinner("Analyzing... 🔍"):
            try:
                pred = model(tf.constant(img_array, dtype=tf.float32))
                if isinstance(pred, dict):
                    pred = list(pred.values())[0].numpy()[0]
                
                top3 = np.argsort(pred)[-3:][::-1]
                
                cols = st.columns(3)
                for col, i in zip(cols, top3):
                    st.metric(label=classes[int(i)], value=f"{pred[i]*100:.2f}%")
            except Exception as e:
                st.error(f"Error: {e}")
