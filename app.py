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
CLIENT_ID = st.secrets.google_oauth.client_id
CLIENT_SECRET = st.secrets.google_oauth.client_secret
REDIRECT_URI = st.secrets.google_oauth.redirect_uri
SCOPES = "openid email profile"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USER_INFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"

# ---------------------------- Login Flow ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🐾 Animal Classifier - Login")
    
    # Handle Google OAuth redirect
    if "code" in st.experimental_get_query_params():
        code = st.experimental_get_query_params()["code"][0]
        # Exchange code for token
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
            user_info = requests.get(USER_INFO_URI, params={"alt": "json"}, headers={"Authorization": f"Bearer {access_token}"}).json()
            st.session_state.logged_in = True
            st.session_state.user_name = user_info.get("name", "User")
            st.experimental_rerun()
        else:
            st.error("Failed to login. Please try again.")

    # Button to redirect directly to Google login
    auth_params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent"
    }
    auth_url = f"{AUTH_URI}?{urllib.parse.urlencode(auth_params)}"
    
    if st.button("Continue with Google", use_container_width=True):
        st.markdown(f'<meta http-equiv="refresh" content="0; URL={auth_url}">', unsafe_allow_html=True)

    st.markdown("---")
    st.info("Or use the demo login form below:")
    email = st.text_input("Email", placeholder="user@example.com")
    password = st.text_input("Password", type="password")
    if st.button("Login Demo"):
        if email == "user" and password == "demo123":
            st.session_state.logged_in = True
            st.session_state.user_name = "Demo User"
            st.experimental_rerun()
        else:
            st.error("Invalid demo credentials.")

# ---------------------------- Main App ----------------------------
if st.session_state.get("logged_in"):
    st.markdown(f"<h2>Welcome, {st.session_state.user_name}!</h2>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    input_method = st.radio("Select input method:", ["📁 Upload Image", "📸 Use Camera"])
    input_file = None
    if input_method == "📁 Upload Image":
        input_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    elif input_method == "📸 Use Camera":
        input_file = st.camera_input("Capture an image")

    if input_file:
        img = Image.open(input_file).convert("RGB")
        st.image(img, use_column_width=True)
        img_array = np.array(img.resize((128, 128)), dtype=np.float32)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        with st.spinner("Analyzing... 🔍"):
            try:
                pred = model(tf.constant(img_array, dtype=tf.float32))
                if isinstance(pred, dict) and "dense_1" in pred:
                    pred = pred["dense_1"].numpy()[0]
                top3 = np.argsort(pred)[-3:][::-1]
                cols = st.columns(3)
                for col, i in zip(cols, top3):
                    col.metric(label=classes[i], value=f"{pred[i]*100:.2f}%")
            except Exception as e:
                st.error(f"Prediction error: {e}")
