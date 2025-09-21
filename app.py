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
CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"  # keep this safe
REDIRECT_URI = "https://neuronerds.streamlit.app/"  # your app URL
SCOPES = ["https://www.googleapis.com/auth/userinfo.profile",
          "https://www.googleapis.com/auth/userinfo.email"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------------------- LOGIN ----------------------------
if not st.session_state.logged_in:
    st.title("🐾 Animal Classifier - Login")

    if st.button("Continue with Google"):
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
        st.experimental_set_query_params(auth_url=auth_url)
        st.markdown(f"[Click here to login with Google]({auth_url})")

    # Handle redirect with code
    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        code = query_params["code"][0]
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
        st.experimental_rerun()

# ---------------------------- MAIN APP ----------------------------
if st.session_state.get("logged_in", False):
    st.markdown(f"<h2>Welcome, {st.session_state.user_name}!</h2>", unsafe_allow_html=True)
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    st.markdown("<p>Choose an input method to see AI prediction instantly!</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Classifier", "📄 Model Info", "💡 About App"])

    with tab1:
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

    with tab2:
        st.markdown("<h2>📄 Model Information</h2>", unsafe_allow_html=True)
        st.info("CNN model trained on custom animal dataset.")
        st.markdown("""
        | Metric | Value |
        | :--- | :--- |
        | **Accuracy** | 92.5% |
        | **Precision** | 91.2% |
        | **Recall** | 90.8% |
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<h2>💡 About this App</h2>", unsafe_allow_html=True)
        st.info("""
        Web app demo of an animal classifier with Streamlit & TensorFlow.
        Allows image upload or camera capture for real-time prediction.
        **Developers:** BPA Batch 2024
        """)
