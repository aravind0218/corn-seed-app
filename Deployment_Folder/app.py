import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AgriScan: Corn Quality AI",
    page_icon="🌽",
    layout="centered"
)

# --- FRONTEND: CUSTOM CSS (ONLY UI CHANGES) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    .stApp { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }

    /* Top main title on gradient background */
    h1 {
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        font-weight: 700;
        border-bottom: none;
        padding-bottom: 0;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 0.2rem;
    }

    /* Subheading under title */
    .agri-subtitle {
        text-align: center;
        color: #e5e7eb;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* Main card that holds uploader + text */
    .main-card {
        background: #f9fafb;                 /* light gray so dark text is visible */
        border-radius: 22px;
        padding: 2rem 2.5rem;
        box-shadow: 0 18px 40px rgba(15,23,42,0.25);
        border: 1px solid #e5e7eb;
        margin-top: 1.5rem;
    }

    .main-card h2, .main-card h3, .main-card p, .main-card label {
        color: #111827 !important;           /* dark text inside card */
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 10px 25px rgba(22,163,74,0.45);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        transform: translateY(-1px);
        box-shadow: 0 16px 30px rgba(22,163,74,0.6);
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #22c55e;
        border-radius: 16px;
        padding: 1.8rem;
        background-color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(249,250,251,0.96);
    }
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
        font-family: 'Poppins', sans-serif;
    }

    /* Image styling */
    div[data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(15,23,42,0.35);
    }
    </style>
""", unsafe_allow_html=True)

# --- UTILITY: LOAD ASSETS (BACKEND UNCHANGED) ---
@st.cache_resource
def load_assets():
    # 1. Find the folder where this app.py is running
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the full paths to the files
    model_path = os.path.join(base_dir, 'corn_model.h5')
    json_path = os.path.join(base_dir, 'classes.json')

    # 3. Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"🚨 Critical Error: Could not load model from {model_path}")
        st.error(f"Details: {e}")
        return None, None
        
    # 4. Load Class Mappings
    try:
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        # Invert dictionary to map ID -> Class Name
        label_map = {v: k for k, v in class_indices.items()}
    except Exception as e:
        st.error(f"🚨 Critical Error: Could not load classes.json from {json_path}")
        label_map = None
        
    return model, label_map

model, label_map = load_assets()

# --- PREDICTION ENGINE (BACKEND UNCHANGED) ---
def process_and_predict(image_data, model):
    # Resize to match training input
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Reshape for Batch (1, 224, 224, 3)
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction = model.predict(img_reshape)
    return prediction

# --- FRONTEND UI (SAME LOGIC, BETTER LOOK) ---
st.title("🌽 AgriScan Pro")
st.markdown("<p class='agri-subtitle'>Automated Corn Seed Quality Control</p>", unsafe_allow_html=True)
st.write("Determine if your batch is **High**, **Medium**, or **Low** quality.")

# Check if model loaded correctly
if model is None:
    st.warning("⚠️ Please ensure 'corn_model.h5' and 'classes.json' are in the same folder as this app.py file.")
    st.stop()

# Wrap main area in card
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Sidebar: Input Selection (unchanged logic)
with st.sidebar:
    st.header("Input Mode")
    mode = st.radio("Choose source:", ["📁 Upload Image", "📷 Capture Image"])
    st.info("Supported formats: JPG, PNG")

file_input = None

if mode == "📁 Upload Image":
    file_input = st.file_uploader("Upload seed image...", type=["jpg", "jpeg", "png"])
elif mode == "📷 Capture Image":
    file_input = st.camera_input("Take a photo of the seed")

# --- EXECUTION LOGIC (UNCHANGED) ---
if file_input is not None:
    # Display Input
    image = Image.open(file_input)
    st.image(image, caption="Input Specimen", width=300)
    
    if st.button("Analyze Quality"):
        with st.spinner("Processing neural network..."):
            preds = process_and_predict(image, model)
            
            # Get highest probability
            result_idx = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            # Map raw label to Quality Grade
            if label_map:
                raw_label = label_map[result_idx]  # e.g., 'healthy', 'broken'
            else:
                raw_label = str(result_idx)
            
            # Quality Logic Mapping (Customize these keywords based on your folder names!)
            quality_grade = ""
            color = ""
            
            # Convert label to lowercase to be safe
            raw_label_clean = raw_label.lower()

            if "healthy" in raw_label_clean:
                quality_grade = "HIGH QUALITY"
                color = "green"
            elif "discolored" in raw_label_clean or "silkcut" in raw_label_clean:
                quality_grade = "MEDIUM QUALITY"
                color = "orange"
            else:  # broken or anything else
                quality_grade = "LOW QUALITY"
                color = "red"

            # Display Results
            st.markdown("---")
            st.markdown(f"### Grade: :{color}[{quality_grade}]")
            st.caption(f"Detected Class: {raw_label.title()} | Confidence: {confidence:.2f}%")
            
            # Progress bar for visual confidence
            st.progress(int(confidence))
            
            if quality_grade == "HIGH QUALITY":
                st.success("✅ APPROVED: Suitable for premium export.")
            elif quality_grade == "MEDIUM QUALITY":
                st.warning("⚠️ ATTENTION: Check for fungal infection or moisture damage.")
            else:
                st.error("❌ REJECTED: Seed integrity compromised.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#e5e7eb; margin-top:2rem;'>Powered by TensorFlow & Streamlit 🌽</p>",
    unsafe_allow_html=True
)
