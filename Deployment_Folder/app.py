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

# --- FRONTEND: SHADED LIGHT-GREEN BACKGROUND ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Shaded light-green app background */
.stApp {
    background: radial-gradient(circle at top left, #e9f7ec 0, #d6efdf 35%, #c5e4d2 65%, #b3d8c4 100%);
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #102418;
}

/* Global text */
html, body, [class*="css"] {
    color: #102418;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Headings use Playfair */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', serif;
}

/* Main title */
h1 {
    text-align: center;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.agri-tagline {
    text-align: center;
    color: #425448;
    font-size: 0.95rem;
    margin-bottom: 1.1rem;
}

/* Card sitting on green background */
.main-card {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    border: 1px solid #c2d9c8;
    box-shadow: 0 18px 40px rgba(8, 24, 14, 0.16);
}

/* Section label */
.section-label {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #5f6f63;
    font-weight: 600;
    margin-bottom: 0.7rem;
}

/* Analyze button */
div.stButton > button {
    background: linear-gradient(135deg, #1a7f4b, #155e38);
    color: #f9fafb;
    border-radius: 999px;
    padding: 0.55rem 1.4rem;
    border: none;
    font-weight: 600;
    box-shadow: 0 12px 26px rgba(21, 98, 58, 0.45);
    transition: all 0.18s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #155e38, #1a7f4b);
    transform: translateY(-1px);
    box-shadow: 0 16px 32px rgba(21, 98, 58, 0.55);
}

/* File uploader container */
div[data-testid="stFileUploader"] {
    border-radius: 14px;
    padding: 1.4rem;
    background: #f3faf4;
    border: 1px dashed #c2d9c8;
}

/* Browse files button text: bold white */
div[data-testid="stFileUploader"] button {
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* Sidebar with its own light-green shading */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e9f7ec 0%, #d2eede 100%);
    border-right: 1px solid #c2d9c8;
}
section[data-testid="stSidebar"] * {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: #183325 !important;
}
.sidebar-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #5f6f63;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

/* Image preview */
div[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.35);
}

/* Quality pill */
.quality-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.quality-high {
    background: #1ea463;
    color: #f9fafb;
}
.quality-medium {
    background: #f4b000;
    color: #3b2a08;
}
.quality-low {
    background: #d64242;
    color: #f9fafb;
}

/* Footer */
.agri-footer {
    text-align: center;
    color: #5f6f63;
    font-size: 0.85rem;
    margin-top: 2rem;
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

# --- FRONTEND UI (ONLY TEXT POSITION CHANGED) ---

st.title("AgriScan Pro")
st.markdown(
    "<p class='agri-tagline'>Upload images of corn seeds and get instant quality classification using advanced AI technology. Categorize seeds as HIGH, MEDIUM, or LOW quality with confidence scores.</p>",
    unsafe_allow_html=True
)

# Put the white card and the “Determine…” line together
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.write("Determine if your batch is **High**, **Medium**, or **Low** quality.")

# Check if model loaded correctly
if model is None:
    st.warning("⚠️ Please ensure 'corn_model.h5' and 'classes.json' are in the same folder as this app.py file.")
    st.stop()

# Sidebar: Input Selection (unchanged backend logic)
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Input mode</div>", unsafe_allow_html=True)
    mode = st.radio("Choose source:", ["📁 Upload Image", "📷 Capture Image"])
    st.info("Supported formats: JPG, PNG")

file_input = None

if mode == "📁 Upload Image":
    file_input = st.file_uploader("Upload seed image...", type=["jpg", "jpeg", "png"])
elif mode == "📷 Capture Image":
    file_input = st.camera_input("Take a photo of the seed")

# --- EXECUTION LOGIC (BACKEND UNCHANGED) ---
if file_input is not None:
    # Display Input
    st.markdown("<div class='section-label'>Specimen preview</div>", unsafe_allow_html=True)
    image = Image.open(file_input)
    st.image(image, caption="Input Specimen", width=320)
    
    if st.button("Analyze quality"):
        with st.spinner("Processing image with the neural network..."):
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
            pill_class = ""
            raw_label_clean = raw_label.lower()

            if "healthy" in raw_label_clean:
                quality_grade = "HIGH QUALITY"
                pill_class = "quality-high"
            elif "discolored" in raw_label_clean or "silkcut" in raw_label_clean:
                quality_grade = "MEDIUM QUALITY"
                pill_class = "quality-medium"
            else:  # broken or anything else
                quality_grade = "LOW QUALITY"
                pill_class = "quality-low"

            st.markdown("---")
            st.markdown(
                f"<div class='quality-pill {pill_class}'>{quality_grade}</div>",
                unsafe_allow_html=True
            )
            st.caption(f"Detected Class: {raw_label.title()} | Confidence: {confidence:.2f}%")
            
            st.progress(int(confidence))
            
            if quality_grade == "HIGH QUALITY":
                st.success("✅ APPROVED: Suitable for premium export.")
            elif quality_grade == "MEDIUM QUALITY":
                st.warning("⚠️ ATTENTION: Check for fungal infection or moisture damage.")
            else:
                st.error("❌ REJECTED: Seed integrity compromised.")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<p class='agri-footer'>Automated Corn Seed Quality Control</p>", unsafe_allow_html=True)
