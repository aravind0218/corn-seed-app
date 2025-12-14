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

# --- FRONTEND: PLAYFAIR + INTER, GREEN THEME ---
st.markdown("""
<style>
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* CSS variables approximating your Tailwind theme */
:root {
    --background: hsl(120, 20%, 98%);
    --foreground: hsl(140, 30%, 15%);

    --card: hsl(0, 0%, 100%);
    --card-foreground: hsl(140, 30%, 15%);

    --primary: hsl(142, 55%, 35%);
    --primary-foreground: hsl(0, 0%, 100%);

    --muted: hsl(120, 15%, 92%);
    --muted-foreground: hsl(140, 15%, 45%);

    --accent: hsl(45, 90%, 55%);
    --accent-foreground: hsl(40, 50%, 15%);

    --border: hsl(140, 20%, 88%);

    --quality-high: hsl(142, 70%, 45%);
    --quality-medium: hsl(45, 90%, 50%);
    --quality-low: hsl(0, 70%, 55%);

    --gradient-start: hsl(142, 55%, 35%);
    --gradient-end: hsl(160, 50%, 45%);
}

/* Global */
.stApp {
    background: radial-gradient(circle at top left, var(--background) 0, #e6f4ea 45%, #d4f1e1 100%);
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--foreground);
}

/* Typography */
html, body, [class*="css"]  {
    color: var(--foreground);
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', serif;
}

/* Main title */
h1 {
    text-align: center;
    letter-spacing: 0.04em;
    font-weight: 600;
    color: var(--foreground);
    margin-bottom: 0.25rem;
}

.agri-tagline {
    text-align: center;
    color: var(--muted-foreground);
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}

/* Utility classes */
.gradient-primary {
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
}

.text-gradient {
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.glass-effect {
    backdrop-filter: blur(14px);
    background: rgba(255, 255, 255, 0.88);
}

/* Main card */
.main-card {
    border-radius: 18px;
    padding: 2rem 2.4rem;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    border: 1px solid var(--border);
}

/* Section title */
.section-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted-foreground);
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d);
    color: var(--primary-foreground);
    border-radius: 999px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    border: none;
    box-shadow: 0 12px 28px rgba(22, 163, 74, 0.35);
    transition: all 0.18s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #15803d, #16a34a);
    transform: translateY(-1px);
    box-shadow: 0 18px 34px rgba(22, 163, 74, 0.48);
}

/* File uploader */
div[data-testid="stFileUploader"] {
    border-radius: 16px;
    padding: 1.5rem;
    background: #f9fafb;
    border: 1px dashed var(--border);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: hsl(120, 20%, 97%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: hsl(140, 30%, 20%) !important;
}
.sidebar-header {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted-foreground);
    font-weight: 600;
    margin-bottom: 0.6rem;
}

/* Image */
div[data-testid="stImage"] img {
    border-radius: 16px;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.35);
}

/* Quality badges */
.quality-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.quality-high {
    background: var(--quality-high);
    color: #f9fafb;
}
.quality-medium {
    background: var(--quality-medium);
    color: var(--accent-foreground);
}
.quality-low {
    background: var(--quality-low);
    color: #f9fafb;
}

/* Footer */
.agri-footer {
    text-align: center;
    color: var(--muted-foreground);
    font-size: 0.85rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --- BACKEND: LOAD ASSETS (UNCHANGED) ---
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'corn_model.h5')
    json_path = os.path.join(base_dir, 'classes.json')

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"🚨 Critical Error: Could not load model from {model_path}")
        st.error(f"Details: {e}")
        return None, None
        
    try:
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        label_map = {v: k for k, v in class_indices.items()}
    except Exception as e:
        st.error(f"🚨 Critical Error: Could not load classes.json from {json_path}")
        label_map = None
        
    return model, label_map

model, label_map = load_assets()

# --- BACKEND: PREDICTION ENGINE (UNCHANGED) ---
def process_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# --- TOP CONTENT ---
st.markdown("<h1 class='text-gradient'>AgriScan Pro</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='agri-tagline'>Nature-inspired AI for accurate corn seed quality grading.</p>",
    unsafe_allow_html=True
)
st.write("Determine if your batch is **High**, **Medium**, or **Low** quality.")

if model is None:
    st.warning("⚠️ Please ensure 'corn_model.h5' and 'classes.json' are in the same folder as this app.py file.")
    st.stop()

# --- MAIN CARD ---
st.markdown("<div class='main-card glass-effect'>", unsafe_allow_html=True)

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
    st.markdown("<div class='section-title'>Specimen preview</div>", unsafe_allow_html=True)
    image = Image.open(file_input)
    st.image(image, caption="Input Specimen", width=320)
    
    if st.button("Analyze quality"):
        with st.spinner("Processing image with neural network..."):
            preds = process_and_predict(image, model)
            result_idx = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            if label_map:
                raw_label = label_map[result_idx]
            else:
                raw_label = str(result_idx)
            
            raw_label_clean = raw_label.lower()

            if "healthy" in raw_label_clean:
                quality_grade = "HIGH QUALITY"
                badge_class = "quality-high"
            elif "discolored" in raw_label_clean or "silkcut" in raw_label_clean:
                quality_grade = "MEDIUM QUALITY"
                badge_class = "quality-medium"
            else:
                quality_grade = "LOW QUALITY"
                badge_class = "quality-low"

            st.markdown("---")
            st.markdown(
                f"<div class='quality-badge {badge_class}'>{quality_grade}</div>",
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
st.markdown("<p class='agri-footer'>AgriScan Pro · Playfair Display & Inter · Nature Green Theme</p>", unsafe_allow_html=True)
