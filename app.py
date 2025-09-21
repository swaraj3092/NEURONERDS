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
/* Overall page background */
body {
    background-color: #1a1a2e; /* Dark background for the entire page */
    color: #f0f2f6; /* Light text for contrast */
    font-family: 'Arial', sans-serif;
}
/* Main content background if logged in */
.main .block-container {
    background-color: #f0f2f6; /* Light background for the main app content */
    color: #1e1e2f;
    padding: 2rem; /* Add some padding */
    border-radius: 10px; /* Rounded corners for the main app container */
}
h1, h2, h3 {
    color: #f0f2f6; /* Light header text for login page */
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
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 auto !important;
    width: 100% !important;
    text-align: center;
    color: #f0f2f6;
}



/* Specific CSS to center the image and make it circular */
.stImage img {
    border-radius: 50%;
    border: 3px solid #3b5998;
    object-fit: cover;
}

.stImage img {
    border-radius: 50%; /* Makes the image circular */
    object-fit: cover; /* Ensures the image covers the circular area */
    border: 3px solid #3b5998; /* Optional: adds a border around the circle */
    display: block; /* Important for centering the image itself within its flex container */
}

.or-separator {
    display: flex;
    align-items: center;
    text-align: center;
    margin: 20px 0;
    color: #ccc; /* Lighter color for OR text on dark background */
}
.or-separator::before,
.or-separator::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #444; /* Darker border for separator lines */
}
.or-separator:not(:empty)::before {
    margin-right: .25em;
}
.or-separator:not(:empty)::after {
    margin-left: .25em;
}
.stTextInput > div > div > input {
    border-radius: 8px;
    background-color: #2a2a3e; /* Darker input background */
    color: #f0f2f6; /* Light text in input */
    border: 1px solid #444; /* Darker border */
}
.stTextInput > label {
    font-weight: normal; /* Normal font weight for field labels */
    color: #f0f2f6; /* Light label text */
}
a {
    color: #87CEEB; /* Lighter blue for links */
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
    # This is the LOGIN UI which will be displayed first
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Logo first, with some top padding
        st.image("cow.png", width=100)
        
        # Title and sub-text
        st.markdown("<h2>Welcome to Animal Classifier</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #ccc;'>Sign in to continue</p>", unsafe_allow_html=True)

        # Google Button (Styling)
        st.markdown('<div class="google-btn"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_Google_g_darkmode_2020.svg" width="20"> Continue with Google</div>', unsafe_allow_html=True)
        
        # OR Separator
        st.markdown('<div class="or-separator">OR</div>', unsafe_allow_html=True)
        
        # Login Form Fields
        email = st.text_input("Email", placeholder="you@example.com", label_visibility="collapsed")
        password = st.text_input("Password", type="password", label_visibility="collapsed")
        
        # Login Button
        login_btn = st.button("Sign in", use_container_width=True)
        
        # Links
        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.markdown("<p style='text-align: left;'><a href='#'>Forgot password?</a></p>", unsafe_allow_html=True)
        with col_link2:
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
# This is the MAIN APP, displayed only AFTER login
# ----------------------------
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
