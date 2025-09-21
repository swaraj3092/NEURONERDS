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
import base64



st.markdown("""
<style>
/* ---------- Body and Background ---------- */
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    font-family: 'Segoe UI', sans-serif;
    color: #f0f2f6;
}

/* ---------- Main Container ---------- */
.main .block-container {
    background-color: rgba(255,255,255,0.05);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* ---------- Headings ---------- */
h1, h2, h3 {
    color: #ffffff;
    text-align: center;
    animation: fadeInDown 1s ease-out;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(90deg, #ff758c, #ff7eb3);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 28px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(255, 255, 255, 0.4);
}

/* ---------- Tabs Styling ---------- */
div[role="tablist"] button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    margin: 0 5px;
    transition: all 0.3s ease;
}
div[role="tablist"] button:focus {
    outline: none;
}
div[role="tablist"] button:hover {
    transform: scale(1.05);
}

/* ---------- History Cards ---------- */
.stImage img {
    border-radius: 12px;
    border: 3px solid #00c6ff;
    transition: transform 0.3s ease;
}
.stImage img:hover {
    transform: scale(1.05);
}

/* ---------- Animations ---------- */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)



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
REDIRECT_URI = "https://neuronerds.streamlit.app"   # ✅ Updated without trailing slash

SCOPES = "openid email profile"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USER_INFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"

# ---------------------------- Session State ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = "User"
if "history" not in st.session_state:
    st.session_state.history = []  # stores uploaded images, predictions, timestamps


# ---------------------------- GOOGLE LOGIN HANDLER ----------------------------
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
                st.rerun()
        except Exception as e:
            st.error(f"An error occurred during authentication: {e}")





# ---------------------------- LOGIN PAGE ----------------------------
if not st.session_state.logged_in:
    st.write("")  # spacing at top

    # Centered small cow logo
    st.image("cow.png", width=120, use_container_width=False)

    # Centered headings
    st.markdown(
        "<h2 style='text-align:center; color:#f0f2f6; margin-top:20px;'>Welcome to Animal Classifier</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#ccc;'>Sign in to continue</p>",
        unsafe_allow_html=True
    )

    # Google login button (centered)
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
        <div style="display:flex; justify-content:center; margin:20px 0;">
            <a href="{auth_url}">
                <button style="
                    width:250px; padding:12px; font-weight:bold; border-radius:12px;
                    background-color:#4285F4; color:white; border:none; cursor:pointer;
                    display:block; margin:auto;
                    ">Continue with Google 🚀</button>
            </a>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # OR separator
    st.markdown('<div class="or-separator" style="margin:10px 0;"></div>', unsafe_allow_html=True)

    # Manual login with email/password
    email = st.text_input("Email", placeholder="user@example.com")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email == "user" and password == "demo123":  # ✅ password updated
            st.session_state.logged_in = True
            st.session_state.user_name = email
            st.rerun()
        else:
            st.error("Invalid credentials.")

    # Links (centered using columns)
    col_link1, col_link2 = st.columns([1,1])
    with col_link1:
        st.markdown("<p style='text-align:center;'><a href='#'>Forgot password?</a></p>", unsafe_allow_html=True)
    with col_link2:
        st.markdown("<p style='text-align:center;'>Need an account? <a href='#'>Sign up</a></p>", unsafe_allow_html=True)




if st.session_state.logged_in:
    st.markdown(f"<h2>Welcome, {st.session_state.get('user_name', 'User')}!</h2>", unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("<h1>🐾 Animal Type Classifier 🐾</h1>", unsafe_allow_html=True)

    # -------------------- Tabs --------------------
    tab1, tab2 = st.tabs(["🖼️ Classifier", "📊 History"])

    # -------------------- Tab 1: Classifier --------------------
    with tab1:
        input_method = st.radio("Select input method:", ["📁 Upload Image", "📸 Use Camera"])
        input_file = None

        if input_method == "📁 Upload Image":
            input_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        elif input_method == "📸 Use Camera":
            input_file = st.camera_input("Capture an image")

        if input_file:
            img = Image.open(input_file).convert("RGB")
            st.image(img, use_container_width=True)

            # Preprocess image
            img_array = np.array(img.resize((128,128)), dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner("Analyzing... 🔍"):
                try:
                    # Prediction
                    pred = model(tf.constant(img_array, dtype=tf.float32))
                    if isinstance(pred, dict):
                        if "dense_1" in pred:
                            pred = pred["dense_1"]
                        else:
                            pred = list(pred.values())[0]
                    if hasattr(pred, "numpy"):
                        pred = pred.numpy()
                    pred = np.array(pred[0])

                    if pred.size == 0:
                        st.error("Model returned empty prediction.")
                    else:
                        # Top 3 predictions
                        top3 = np.argsort(pred)[-3:][::-1]
                        cols = st.columns(3)
                        for col, i in zip(cols, top3):
                            with col:
                                st.metric(label=classes[int(i)], value=f"{pred[i]*100:.2f}%")

                        # Optionally show all predictions
                        if st.checkbox("Show all predictions"):
                            st.markdown("---")
                            left_col, right_col = st.columns(2)
                            sorted_idx = np.argsort(pred)[::-1]
                            half = len(sorted_idx)//2
                            for i in sorted_idx[:half]:
                                left_col.markdown(f"**{classes[int(i)]}:** {pred[i]*100:.4f}%")
                            for i in sorted_idx[half:]:
                                right_col.markdown(f"**{classes[int(i)]}:** {pred[i]*100:.4f}%")

                        # Save to history
                        from datetime import datetime
                        import io
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        st.session_state.history.append({
                            "image": image_bytes,
                            "predictions": pred.tolist(),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    # -------------------- Tab 2: History --------------------
    with tab2:
        st.markdown("<h2>📊 Prediction History</h2>", unsafe_allow_html=True)

        if len(st.session_state.history) == 0:
            st.info("No history yet.")
        else:
            for entry in reversed(st.session_state.history):
    # Convert image bytes to base64
    image_base64 = base64.b64encode(entry["image"]).decode("utf-8")
    
    # Get top 3 predictions
    top3_idx = np.argsort(entry["predictions"])[-3:][::-1]
    top3_text = ", ".join([f"{classes[int(i)]}: {entry['predictions'][i]*100:.2f}%" for i in top3_idx])
    
    # Display card
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); padding:15px; border-radius:15px; 
                display:flex; align-items:center; margin-bottom:15px; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.3); transition: transform 0.3s ease;'>
        <img src="data:image/png;base64,{image_base64}" width="80" style="border-radius:12px; margin-right:15px;">
        <div>
            <b>Time:</b> {entry['timestamp']}<br>
            <b>Top 3:</b> {top3_text}
        </div>
    </div>
    """, unsafe_allow_html=True)



