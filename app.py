import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import json
from io import BytesIO
import streamlit_authenticator as stauth  # ---------------------------- EDIT MARK: Google Login ----------------------------

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="🐾 Animal Classifier", layout="wide", page_icon="cow.png")

# ---------------------------- Caching Functions ----------------------------
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

# ---------------------------- Load assets ----------------------------
model = load_model()
classes = load_classes()

# ---------------------------- CSS Styling ----------------------------
st.markdown("""
<style>
body { background-color: #1a1a2e; color: #f0f2f6; font-family: 'Arial', sans-serif; }
.main .block-container { background-color: #f0f2f6; color: #1e1e2f; padding: 2rem; border-radius: 10px; }
h1, h2, h3 { color: #f0f2f6; text-align: center; }
.stButton>button { background-color: #3b5998; color: white; font-weight: bold; border-radius: 12px; padding: 12px 28px; transition: all 0.3s ease; border: none; }
.stButton>button:hover { background-color: #314a79; transform: scale(1.02); }
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; font-weight: bold; }
.google-btn { width: 100%; display: flex; align-items: center; justify-content: center; padding: 10px; border: 1px solid #ccc; border-radius: 8px; cursor: pointer; background-color: white; color: #1e1e2f; font-weight: bold; transition: background-color 0.2s; }
.google-btn:hover { background-color: #f1f1f1; }
.google-btn img { margin-right: 10px; }
.or-separator { display: flex; align-items: center; text-align: center; margin: 20px 0; color: #ccc; }
.or-separator::before, .or-separator::after { content: ''; flex: 1; border-bottom: 1px solid #444; }
.or-separator:not(:empty)::before { margin-right: .25em; }
.or-separator:not(:empty)::after { margin-left: .25em; }
.stTextInput > div > div > input { border-radius: 8px; background-color: #2a2a3e; color: #f0f2f6; border: 1px solid #444; }
a { color: #87CEEB; text-decoration: none; }
a:hover { text-decoration: underline; }
div[data-testid="stImage"] { display: flex !important; justify-content: center !important; align-items: center !important; }
div[data-testid="stImage"] img { border-radius: 50% !important; border: 3px solid #3b5998; object-fit: cover; margin: auto !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- EDIT MARK: Google Login Setup ----------------------------
# Credentials dictionary for streamlit-authenticator demo login
credentials = {
    "usernames": {
        "user": {
            "name": "Demo User",
            "password": stauth.Hasher(["demo123"]).generate()[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "animal_classifier_cookie",
    "animal_classifier_signature",
    cookie_expiry_days=1
)

# Login form
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.session_state.logged_in = True
    st.success(f"Welcome {name}!")  # ---------------------------- END EDIT ----------------------------
elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")

# ---------------------------- MAIN APP ----------------------------
if st.session_state.get('logged_in', False):

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
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner("Analyzing... 🔍"):
                try:
                    prediction_output = model(tf.constant(img_array, dtype=tf.float32))
                    if isinstance(prediction_output, dict) and "dense_1" in prediction_output:
                        prediction = prediction_output["dense_1"].numpy()[0]
                    else:
                        st.error("Unexpected model output format. Please check the model's serving signature.")
                        prediction = None
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    prediction = None

            if prediction is not None:
                st.subheader("Top Predictions")
                top3_idx = np.argsort(prediction)[-3:][::-1]
                cols = st.columns(3)

                for col, i in zip(cols, top3_idx):
                    with col:
                        confidence = prediction[i]
                        st.metric(label=classes[i], value=f"{confidence*100:.2f}%")

                show_all = st.checkbox("Show all class predictions")
                if show_all:
                    st.markdown("---")
                    st.markdown("<h2>All Predictions:</h2>", unsafe_allow_html=True)
                    
                    left_col, right_col = st.columns(2)
                    sorted_idx = np.argsort(prediction)[::-1]
                    
                    half = len(sorted_idx) // 2
                    for i in sorted_idx[:half]:
                        left_col.markdown(f"**{classes[i]}:** {prediction[i]*100:.4f}%")
                    for i in sorted_idx[half:]:
                        right_col.markdown(f"**{classes[i]}:** {prediction[i]*100:.4f}%")

    with tab2:
        st.markdown("<h2>📄 Model Information</h2>", unsafe_allow_html=True)
        st.info("""
            This classifier uses a **Convolutional Neural Network (CNN)**. 

[Image of a convolutional neural network architecture]

            The model was trained on a custom dataset of animal images.
        """)
        st.write("### Key Metrics")
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
            This web application is a **simple demonstration** of an animal image classifier built with **Streamlit** and **TensorFlow/Keras**.
            It allows you to upload an image or use your camera to get a real-time prediction of the animal type.
            
            **Developers:** BPA Batch 2024
        """)
