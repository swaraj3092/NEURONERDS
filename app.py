import os
import warnings

# ------------------ SUPPRESS WARNINGS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
warnings.filterwarnings("ignore")  # Ignore Python warnings

# Then import other libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json
import requests
import urllib.parse


# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide", page_icon="cow.png")

# ---------------------------- Caching Functions ----------------------------
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

# ---------------------------- GOOGLE LOGIN HANDLER (Corrected) ----------------------------
# The login handling is now correctly placed at the top and uses st.query_params
if "code" in st.query_params:
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
        if access_token:
            user_info = requests.get(
                USER_INFO_URI,
                params={"alt": "json"},
                headers={"Authorization": f"Bearer {access_token}"}
            ).json()
            st.session_state.logged_in = True
            st.session_state.user_name = user_info.get("name","User")
            st.rerun()
        else:
            st.error("Failed to login. Please try again.")
    except Exception as e:
        st.error(f"An error occurred during authentication: {e}")

# ---------------------------- LOGIN PAGE ----------------------------
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.image("cow.png", width=120)
        st.markdown("<h2>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #ccc;'>Sign in to continue</p>", unsafe_allow_html=True)

        auth_params = {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": SCOPES,
            "access_type": "offline",
            "prompt": "consent"
        }
        auth_url = f"{AUTH_URI}?{urllib.parse.urlencode(auth_params)}"
        st.markdown(
            f'''
            <a href="{auth_url}" target="_blank">
                <button style="
                    width:100%; padding:12px; font-weight:bold; border-radius:12px;
                    background-color:#4285F4; color:white; border:none; cursor:pointer;
                    ">Continue with Google 🚀</button>
            </a>
            ''', unsafe_allow_html=True
        )

        st.markdown('<div class="or-separator">OR</div>', unsafe_allow_html=True)
        email = st.text_input("Email", placeholder="user@example.com")
        password = st.text_input("Password", type="password")
        if st.button("Login Demo"):
            if email=="user" and password=="demo123":
                st.session_state.logged_in = True
                st.session_state.user_name = "Demo User"
                st.rerun()
            else:
                st.error("Invalid demo credentials.")

        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.markdown("<p style='text-align:left;'><a href='#'>Forgot password?</a></p>", unsafe_allow_html=True)
        with col_link2:
            st.markdown("<p style='text-align:right;'>Need an account? <a href='#'>Sign up</a></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- MAIN APP ----------------------------
else:
    st.markdown(f"<h2>Welcome, {st.session_state.get('user_name', 'User')}!</h2>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

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
                if isinstance(pred, dict) and "dense_1" in pred:
                    pred = pred["dense_1"].numpy()[0]
                else:
                    pred = pred.numpy()[0]
                
                top3 = np.argsort(pred)[-3:][::-1]
                
                cols = st.columns(3)
                for col,i in zip(cols,top3):
                    with col:
                        st.metric(label=classes[int(i)],value=f"{pred[i]*100:.2f}%")
                if st.checkbox("Show all predictions"):
                    st.markdown("---")
                    left_col,right_col=st.columns(2)
                    sorted_idx = np.argsort(pred)[::-1]
                    half = len(sorted_idx)//2
                    for i in sorted_idx[:half]:
                        left_col.markdown(f"**{classes[int(i)]}:** {pred[i]*100:.4f}%")
                    for i in sorted_idx[half:]:
                        right_col.markdown(f"**{classes[int(i)]}:** {pred[i]*100:.4f}%")
            except Exception as e:
                st.error(f"Error: {e}")
