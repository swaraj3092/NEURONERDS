import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json
import requests
from google_auth_oauthlib.flow import Flow
import os

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

# ---------------------------- Google OAuth ----------------------------
CLIENT_ID = "44089178154-3tfm5sc60qmnc8t5d2p92innn10t3pu3.apps.googleusercontent.com"
CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
REDIRECT_URI = "https://neuronerds.streamlit.app/"
SCOPES = ["https://www.googleapis.com/auth/userinfo.profile",
          "https://www.googleapis.com/auth/userinfo.email"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "auth_initiated" not in st.session_state:
    st.session_state.auth_initiated = False

# ---------------------------- CSS Styling ----------------------------
st.markdown("""
<style>
/* Overall page background */
body {
    background-color: #1a1a2e;
    color: #f0f2f6;
    font-family: 'Arial', sans-serif;
}
/* Main content background if logged in */
.main .block-container {
    background-color: #f0f2f6;
    color: #1e1e2f;
    padding: 2rem;
    border-radius: 10px;
}
h1, h2, h3 {
    color: #f0f2f6;
    text-align: center;
}
.stButton>button {
    background-color: #3b5998;
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
    background: #1a1a2e;
    border-radius: 20px;
    padding: 40px;
    margin: 50px auto;
    width: 400px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    text-align: center;
    color: #f0f2f6;
}
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
.stTextInput > div > div > input {
    border-radius: 8px;
    background-color: #2a2a3e;
    color: #f0f2f6;
    border: 1px solid #444;
}
.stTextInput > label {
    font-weight: normal;
    color: #f0f2f6;
}
a {
    color: #87CEEB;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
/* Force center for st.image */
div[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
/* Make the logo circular with border */
div[data-testid="stImage"] img {
    border-radius: 50% !important;
    border: 3px solid #3b5998;
    object-fit: cover;
    margin: auto !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- LOGIN ----------------------------
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Cow logo
        st.image("cow.png", width=120)

        st.markdown("<h2>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #ccc;'>Sign in to continue</p>", unsafe_allow_html=True)

        # Handle redirect with code
        if "code" in st.query_params:
            if not st.session_state.auth_initiated:
                st.session_state.auth_initiated = True
                code = st.query_params["code"]
                
                # Check if 'flow' exists in session state before using it
                if "flow" in st.session_state:
                    flow = st.session_state.flow
                    flow.fetch_token(code=code)
                    credentials = flow.credentials
                    user_info = requests.get(
                        "https://www.googleapis.com/oauth2/v1/userinfo",
                        params={"alt": "json"},
                        headers={"Authorization": f"Bearer {credentials.token}"}
                    ).json()
                    st.session_state.logged_in = True
                    st.session_state.user_name = user_info.get("name", "User")
                    st.rerun()
                else:
                    st.error("Authentication flow not found. Please try logging in again.")

        # Google login button
        google_btn = st.button("Continue with Google", use_container_width=True)
        if google_btn:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": CLIENT_ID,
                        "client_secret": CLIENT_SECRET,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [REDIRECT_URI]
                    }
                },
                scopes=SCOPES,
                redirect_uri=REDIRECT_URI
            )
            auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
            st.session_state.flow = flow
            st.session_state.auth_initiated = False # Reset for new auth attempt
            st.query_params.auth_url = auth_url # Use the new method
            st.markdown(f"[Click here to login with Google]({auth_url})")

        st.markdown('<div class="or-separator">OR</div>', unsafe_allow_html=True)

        # Standard Login Form
        email = st.text_input("Email", placeholder="you@example.com", label_visibility="collapsed")
        password = st.text_input("Password", type="password", label_visibility="collapsed")
        login_btn = st.button("Sign in", use_container_width=True)

        if login_btn:
            if email == "user" and password == "demo123":
                st.session_state.logged_in = True
                st.toast("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Try again.")

        # Links
        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.markdown("<p style='text-align: left;'><a href='#'>Forgot password?</a></p>", unsafe_allow_html=True)
        with col_link2:
            st.markdown("<p style='text-align: right;'>Need an account? <a href='#'>Sign up</a></p>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
# ---------------------------- MAIN APP ----------------------------
    st.markdown(f"<h2>Welcome, {st.session_state.get('user_name', 'User')}!</h2>", unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    st.markdown("<p>Choose an input method to see AI prediction instantly!</p>", unsafe_allow_html=True)

    # Removed tab2 and tab3
    # The main classifier logic is now in a single, default view
    input_method = st.radio("Select input method:", ["📁 Upload Image", "📸 Use Camera"])
    input_file = None
    if input_method == "📁 Upload Image":
        input_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    elif input_method == "📸 Use Camera":
        input_file = st.camera_input("Capture an image using your camera")

    if input_file:
        img = Image.open(input_file).convert("RGB")
        st.image(img, caption="Input Image", use_container_width=True)
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized, dtype=np.float32)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        with st.spinner("Analyzing... 🔍"):
            try:
                prediction_output = model(tf.constant(img_array, dtype=tf.float32))
                if isinstance(prediction_output, dict) and "dense_1" in prediction_output:
                    prediction = prediction_output["dense_1"].numpy()[0]
                else:
                    st.error("Unexpected model output format.")
                    prediction = None
            except Exception as e:
                st.error(f"Error: {e}")
                prediction = None
        if prediction is not None:
            st.subheader("Top Predictions")
            top3_idx = np.argsort(prediction)[-3:][::-1]
            cols = st.columns(3)
            for col, i in zip(cols, top3_idx):
                with col:
                    st.metric(label=classes[i], value=f"{prediction[i]*100:.2f}%")
            if st.checkbox("Show all predictions"):
                st.markdown("---")
                left_col, right_col = st.columns(2)
                sorted_idx = np.argsort(prediction)[::-1]
                half = len(sorted_idx)//2
                for i in sorted_idx[:half]:
                    left_col.markdown(f"**{classes[i]}:** {prediction[i]*100:.4f}%")
                for i in sorted_idx[half:]:
                    right_col.markdown(f"**{classes[i]}:** {prediction[i]*100:.4f}%")
