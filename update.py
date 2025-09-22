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
import base64
from datetime import datetime
import io

# ------------------ FIREBASE IMPORTS ------------------
import firebase_admin
from firebase_admin import credentials, firestore, auth, exceptions

# ------------------ SUPPRESS WARNINGS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ------------------ FIREBASE SETUP ------------------
# These are provided by the canvas environment
firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
initial_auth_token = os.environ.get('__initial_auth_token', None)
app_id = os.environ.get('__app_id', 'default-app-id')

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred, name=app_id)

db = firestore.client(app=firebase_admin.get_app(name=app_id))
firebase_auth = auth

def add_user(email, password):
    try:
        user = firebase_auth.create_user(email=email, password=password)
        db.collection("users").document(user.uid).set({"email": email})
        return True, ""
    except exceptions.FirebaseError as e:
        return False, str(e)

def verify_user(email, password):
    try:
        user = firebase_auth.get_user_by_email(email)
        # Firebase does not expose a direct password verification method for security.
        # For a full-stack app, you would handle this on a secure backend.
        # As a workaround for this environment, we check if the user exists.
        # The user will need to log in with a valid password on the client side.
        # The provided Google Auth is a better approach for this environment.
        return True
    except exceptions.FirebaseError:
        return False

# ------------------ STYLING ------------------
st.markdown("""
<style>
body { background: linear-gradient(135deg, #1e3c72, #2a5298); font-family: 'Segoe UI', sans-serif; color: #f0f2f6;}
.main .block-container { background-color: rgba(255,255,255,0.05); padding: 2rem; border-radius: 15px; backdrop-filter: blur(10px); box-shadow: 0 10px 30px rgba(0,0,0,0.5);}
h1,h2,h3 { color: #ffffff; text-align:center; animation: fadeInDown 1s ease-out;}
.stButton>button { background: linear-gradient(90deg,#ff758c,#ff7eb3); color:white; font-weight:bold; border-radius:12px; padding:12px 28px; border:none; cursor:pointer; transition: all 0.3s ease;}
.stButton>button:hover { transform: scale(1.05); box-shadow:0 8px 20px rgba(255,255,255,0.4);}
div[role="tablist"] button { background: linear-gradient(90deg,#00c6ff,#0072ff); color:white; font-weight:bold; border-radius:10px; margin:0 5px; transition: all 0.3s ease;}
div[role="tablist"] button:focus { outline:none;}
div[role="tablist"] button:hover { transform: scale(1.05);}
.stImage img { border-radius:12px; border:3px solid #00c6ff; transition: transform 0.3s ease;}
.stImage img:hover { transform: scale(1.05);}
@keyframes fadeInDown { from {opacity:0; transform:translateY(-20px);} to {opacity:1; transform:translateY(0);} }
</style>
""", unsafe_allow_html=True)

# ------------------ PAGE CONFIG ------------------
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
        st.error("Class names file not found. Please ensure 'models/model.json' exists.")
        return []

model = load_model()
classes = load_classes()

# ------------------ GOOGLE OAUTH CONFIG ------------------
CLIENT_ID = "44089178154-3tfm5sc60qmnc8t5d2p92innn10t3pu3.apps.googleusercontent.com"
CLIENT_SECRET = "GOCSPX-oJkYZxFa-4s4t8VHrBIhAgsi"
REDIRECT_URI = "https://neuronerds.streamlit.app/"
SCOPES = "openid email profile"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USER_INFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"

# ------------------ SESSION STATE ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = "User"
if "history" not in st.session_state:
    st.session_state.history = []
if "user_uid" not in st.session_state:
    st.session_state.user_uid = None

# ------------------ GOOGLE LOGIN HANDLER ------------------
if "code" in st.query_params:
    code_list = st.query_params["code"]
    if code_list:
        code = code_list[0]
        try:
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
                st.session_state.user_name = user_info.get("name", "User")
                st.experimental_set_query_params()
                st.rerun()
        except Exception as e:
            st.error(f"An error occurred during authentication: {e}")

# ------------------ LOGIN / REGISTER PAGE ------------------
if not st.session_state.logged_in:
    st.image("cow.png", width=120)
    st.markdown("<h2 style='text-align:center; color:#f0f2f6;'>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ccc;'>Sign in to continue</p>", unsafe_allow_html=True)

    auth_params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent"
    }
    auth_url = f"{AUTH_URI}?{urllib.parse.urlencode(auth_params)}"
    st.markdown(f'<div style="display:flex; justify-content:center; margin:20px 0;"><a href="{auth_url}"><button style="width:250px;padding:12px;font-weight:bold;border-radius:12px;background-color:#4285F4;color:white;border:none;cursor:pointer;">Continue with Google 🚀</button></a></div>', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email", placeholder="user@example.com", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            try:
                user = firebase_auth.sign_in_with_email_and_password(email, password)
                st.session_state.logged_in = True
                st.session_state.user_name = email
                st.session_state.user_uid = user.uid
                st.rerun()
            except exceptions.FirebaseError:
                st.error("Invalid credentials.")

    with tab_register:
        new_email = st.text_input("Email (must be @gmail.com)", placeholder="example@gmail.com", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register", key="reg_btn"):
            if not new_email.endswith("@gmail.com"):
                st.error("Email must end with @gmail.com")
            else:
                success, error_msg = add_user(new_email, new_password)
                if success:
                    st.success("Registration successful! You can now login.")
                else:
                    st.error(f"Registration failed: {error_msg}")

# ------------------ MAIN APP ------------------
if st.session_state.logged_in:
    st.markdown(f"<h2>Welcome, {st.session_state.get('user_name','User')}!</h2>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🖼️ Classifier", "📊 History"])

    def load_history():
        if st.session_state.user_uid:
            history_ref = db.collection("users").document(st.session_state.user_uid).collection("history")
            docs = history_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).
