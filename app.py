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
# The page_icon now correctly references the local PNG file.
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
# Modern Dark CSS Styling
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #1e1e2f;
    color: #f5f5f5;
    font-family: 'Arial', sans-serif;
}
h1, h2, h3 {
    color: #ffffff;
    text-align: center;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
}
.stButton>button {
    background-color: #ff6f61;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 28px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #ff3b2f;
    transform: scale(1.05);
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
}
.pred-card {
    background: #2e2e3f;
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0px 8px 16px rgba(0,0,0,0.5);
    text-align: center;
    transition: transform 0.3s ease;
}
.pred-card:hover {
    transform: translateY(-5px);
}
.stMetric {
    background: #2e2e3f;
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
}
.css-1r6slb0 input {
    border-radius: 12px;
}
.stRadio > label {
    font-weight: bold;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Login Page
# ----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login_form():
    """Displays the login form in a sidebar."""
    with st.sidebar:
        st.markdown("<h1>🔒 BPA Login</h1>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if username == "user" and password == "demo123":
                st.session_state.logged_in = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Try again.")

if not st.session_state.logged_in:
    login_form()
    st.image("https://images.unsplash.com/photo-1547493706-03c623b0a7c4", caption="Image of various animals", use_container_width=True)
    st.markdown("<h3>Welcome! Please log in on the left to use the Animal Classifier.</h3>", unsafe_allow_html=True)
else:
# ----------------------------
# Main App
# ----------------------------
    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)
    st.markdown("<p>Choose an input method to see AI prediction instantly!</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Classifier", "📄 Model Info", "💡 About App"])

    with tab1:
        # Input choice with icons
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
                    
                    # Create two columns for a more compact display
                    left_col, right_col = st.columns(2)
                    sorted_idx = np.argsort(prediction)[::-1]
                    
                    # Distribute predictions evenly
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
