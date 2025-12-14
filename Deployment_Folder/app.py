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

# --- FRONTEND: CLASSY THEME (ONLY UI) ---
st.markdown("""
    <style>
    /* Elegant font */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at top left, #f4f5fb 0, #e7edf8 28%, #dfe7fb 55%, #cfd9f6 75%, #c1c4f0 100%);
        font-family: 'Manrope', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #0f172a;
    }

    /* Top title */
    h1 {
        color: #0b1020;
        text-align: center;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.15rem;
    }

    .agri-tagline {
        text-align: center;
        color: #4b5563;
        font-size: 0.98rem;
        margin-bottom: 1.4rem;
    }

    /* Main glass card */
    .main-card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(18px);
        border-radius: 24px;
        padding: 2.2rem 2.6rem;
        box-shadow: 0 20px 55px rgba(15,23,42,0.16);
        border: 1px solid rgba(148,163,184,0.5);
    }

    .main-card h2, .main-card h3, .main-card p, .main-card label {
        color: #111827 !important;
    }

    /* Headings inside card */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.9rem;
    }

    /* Buttons – subtle, not neon */
    div.stButton > button {
        background: linear-gradient(135deg, #059669, #047857);
        color: #f9fafb;
        border-radius: 999px;
        padding: 0.6rem 1.1rem;
        font-weight: 600;
        border: 0;
        box-shadow: 0 12px 30px rgba(5,150,105,0.32);
        transition: all 0.18s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #047857, #059669);
        transform: translateY(-1px);
        box-shadow: 0 18px 38px rgba(5,150,105,0.45);
    }

    /* File uploader – minimal */
    div[data-testid="stFileUploader"] {
        border: 1px dashed #9ca3af;
        border-radius: 18px;
        padding: 1.6rem;
        background: #f9fafb;
        color: #111827 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9fafb 0%, #eef2ff 100%);
        border-right: 1px solid rgba(148,163,184,0.4);
    }
    section[data-testid="stSidebar"] * {
        font-family: 'Manrope', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: #111827 !important;
    }

    /* Sidebar header */
    .sidebar-header {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }

    /* Image preview */
    div[data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 14px 38px rgba(15,23,42,0.35);
    }

    /* Result text colors – softer but clear */
    .grade-text-high {
        color: #047857;
    }
    .grade-text-medium {
        color: #d97706;
    }
    .grade-text-low {
        color: #b91c1c;
    }

    /* Footer */
    .agri-footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- BACKEND: LOAD ASSETS (UNCHANGED) ---
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

# --- BACKEND: PREDICTION ENGINE (UNCHANGED) ---
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

# --- TOP TEXT ---
st.title("AgriScan Pro")
st.markdown("<p class='agri-tagline'>Precision corn seed quality assessment for farms, labs and exporters.</p>", unsafe_allow_html=True)
st.write("Determine if your batch is **High**, **Medium**, or **Low** quality.")

# Check if model loaded correctly
if model is None:
    st.warning("⚠️ Please ensure 'corn_model.h5' and 'classes.json' are in the same folder as this app.py file.")
    st.stop()

# --- MAIN CARD CONTAINER ---
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Sidebar (same logic)
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Input mode</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section-title'>Specimen preview</div>", unsafe_allow_html=True)
    image = Image.open(file_input)
    st.image(image, caption="Input Specimen", width=320)
    
    if st.button("Analyze quality"):
        with st.spinner("Running neural network on image..."):
            preds = process_and_predict(image, model)
            
            # Get highest probability
            result_idx = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            # Map raw label to Quality Grade
            if label_map:
                raw_label = label_map[result_idx]  # e.g., 'healthy', 'broken'
            else:
                raw_label = str(result_idx)
            
            # Quality Logic Mapping (unchanged rules)
            quality_grade = ""
            css_class = ""
            
            raw_label_clean = raw_label.lower()

            if "healthy" in raw_label_clean:
                quality_grade = "HIGH QUALITY"
                css_class = "grade-text-high"
            elif "discolored" in raw_label_clean or "silkcut" in raw_label_clean:
                quality_grade = "MEDIUM QUALITY"
                css_class = "grade-text-medium"
            else:
                quality_grade = "LOW QUALITY"
                css_class = "grade-text-low"

            st.markdown("---")
            st.markdown(
                f"<h3 class='{css_class}'>Grade: {quality_grade}</h3>",
                unsafe_allow_html=True
            )
            st.caption(f"Detected Class: {raw_label.title()} | Confidence: {confidence:.2f}%")
            
            st.progress(int(confidence))
            
            if quality_grade == "HIGH QUALITY":
                st.success("Approved: Suitable for premium export.")
            elif quality_grade == "MEDIUM QUALITY":
                st.warning("Attention: Check for fungal infection or moisture damage.")
            else:
                st.error("Rejected: Seed integrity compromised.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p class='agri-footer'>AgriScan Pro · TensorFlow · Streamlit</p>", unsafe_allow_html=True)
