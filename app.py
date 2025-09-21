import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json
from io import BytesIO

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
# Modern Light CSS Styling
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #f0f2f6; /* Light gray background */
    color: #1e1e2f; /* Dark text for contrast */
    font-family: 'Arial', sans-serif;
}
h1, h2, h3 {
    color: #1e1e2f; /* Dark header text */
    text-align: center;
    text-shadow: none;
}
.stButton>button {
    background-color: #3b5998; /* A shade of blue for buttons */
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 28px;
    transition: all 0.3s ease;
    border: none;
}
.stButton>button:hover {
    background-color: #314a79;
    transform: scale(1.02);
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
}
/* Style for the Google Button */
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
    text-decoration: none;
}
.google-btn:hover {
    background-color: #f1f1f1;
}
.google-btn img {
    margin-right: 10px;
}
.login-container {
    background: #ffffff; /* White background for the card */
    border-radius: 20px;
    padding: 40px;
    margin: 50px auto;
    width: 400px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
    text-align: center;
}
/* CSS for the image: makes it circular, small, and centered */
.stImage {
    display: flex; /* Use flexbox to center content */
    justify-content: center; /* Center horizontally */
    align-items: center; /* Center vertically (if height allows) */
    margin-bottom: 20px;
}

.stImage > img {
    border-radius: 50%; /* Makes the image circular */
    object-fit: cover; /* Ensures the image covers the circular area */
    border: 3px solid #3b5998; /* Optional: adds a border around the circle */
}

.or-separator {
    display: flex;
    align-items: center;
    text-align: center;
    margin: 20px 0;
}
.or-separator::before,
.or-separator::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #ccc;
}
.or-separator:not(:empty)::before {
    margin-right: .25em;
}
.or-separator:not(:empty)::after {
    margin-left: .25em;
}
.stTextInput > div > div > input {
    border-radius: 8px;
}
.stTextInput > label {
    font-weight: normal; /* Normal font weight for field labels */
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Login Page
# ----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Login UI
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Logo: circular, smaller, and centered.
        # Ensure cow.png is in the same directory as your script.
        st.image("cow.png", width=100, use_column_width=False, output_format="PNG")
        
        # Welcome Text
        st.markdown("<h2>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #555;'>Sign in to continue</p>", unsafe_allow_html=True)

        # Google Button (Styling)
        st.markdown('<div class="google-btn"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_Google_g_darkmode_2020.svg" width="20"> Continue with Google</div>', unsafe_allow_html=True)
        
        # OR Separator
        st.markdown('<div class="or-separator">OR</div>', unsafe_allow_html=True)
        
        # Login Form Fields
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        
        # Login Button
        login_btn = st.button("Sign in", use_container_width=True)
        
        # Links
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='text-align: left;'><a href='#'>Forgot password?</a></p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p style='text-align: right;'>Need an account? <a href='#'>Sign up</a></p>", unsafe_allow_html=True)

        # Login validation logic
        if login_btn:
            if email == "user" and password == "demo123":
                st.session_state.logged_in = True
                st.toast("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
else:
# ----------------------------
# Main App (Hidden until logged in)
#
