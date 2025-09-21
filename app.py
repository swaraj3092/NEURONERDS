import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide", page_icon="cow.png")

# ----------------------------
# Caching Functions
# ----------------------------
@st.cache_resource
def load_model():
    """Loads the TensorFlow SavedModel into a TFSMLayer."""
    return TFSMLayer("models/animal_classifier_savedmodel", call_endpoint="serving_default")

@st.cache_data
def load_classes():
    """Loads class names from a JSON file."""
    try:
        with open("models/model.json", "r") as f:
            classes_dict = json.load(f)
        return [classes_dict[str(k)] for k in range(len(classes_dict))]
    except FileNotFoundError:
        st.error("Class names file not found. Please ensure 'models/model.json' exists.")
        return []

# ----------------------------
# Load assets
# ----------------------------
model = load_model()
classes = load_classes()

# ----------------------------
# CSS Styling
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #1a1a2e;
    color: #f0f2f6;
    font-family: 'Arial', sans-serif;
}
/* Center and circular style for cow image */
div[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
div[data-testid="stImage"] img {
    border-radius: 50% !important;
    border: 3px solid #3b5998 !important;
    object-fit: cover !important;
    margin: auto !important;
    display: block !important;
}
/* Google button */
.google-btn {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 8px;
    cursor: pointer;
    background-color: white;
    color: #1e1e2f;
    font-weight: bold;
    transition: background-color 0.2s;
}
.google-btn:hover {
    background-color: #f1f1f1;
}
.google-btn img {
    margin-right: 10px;
}
/* OR Separator */
.or-separator {
    display: flex;
    align-items: center;
    text-align: center;
    margin: 20px 0;
    color: #ccc;
}
.or-separator::before,
.or-separator::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #444;
}
.or-separator:not(:empty)::before {
    margin-right: .25em;
}
.or-separator:not(:empty)::after {
    margin-left: .25em;
}
/* Input fields */
.stTextInput > div > div > input {
    border-radius: 8px;
    background-color: #2a2a3e;
    color: #f0f2f6;
    border: 1px solid #444;
}
a {
    color: #87CEEB;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Login Page
# ----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Cow logo (now perfectly centered & circular)
    st.image("cow.png", width=100)

    # Title and subtext
    st.markdown("<h2 style='text-align:center;'>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ccc;'>Sign in to continue</p>", unsafe_allow_html=True)

    # Google Button
    st.markdown(
        '<div class="google-btn">'
        '<img src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_Google_g_darkmode_2020.svg" width="20">'
        ' Continue with Google</div>',
        unsafe_allow_html=True
    )

    # OR Separator
    st.markdown('<div class="or-separator">OR</div>', unsafe_allow_html=True)

    # Login Form
    email = st.text_input("Email", placeholder="you@example.com", label_visibility="collapsed")
    password = st.text_input("Password", type="password", label_visibility="collapsed")

    login_btn = st.button("Sign in", use_container_width=True)

    # Links
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        st.markdown("<p style='text-align: left;'><a href='#'>Forgot password?</a></p>", unsafe_allow_html=True)
    with col_link2:
        st.markdown("<p style='text-align: right;'>Need an account? <a href='#'>Sign up</a></p>", unsafe_allow_html=True)

    # Login validation
    if login_btn:
        if email == "user" and password == "demo123":
            st.session_state.logged_in = True
            st.toast("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")

else:
    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    st.markdown("<p>Choose an input method to see AI prediction instantly!</p>", unsafe_allow_html=True)
    # main app code continues...
