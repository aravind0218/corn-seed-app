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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FRONTEND: ENHANCED CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        position: relative;
    }
    h1::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #f9a825, #ff6b35);
        border-radius: 2px;
    }
    
    /* Main Content Cards */
    .main-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    /* Enhanced Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        height: 60px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(76, 175, 80, 0.6);
        background: linear-gradient(135deg, #45a049, #4CAF50);
    }
    
    /* File Uploader */
    div[data-testid="stFileUploader"] {
        border: 3px dashed #4CAF50;
        border-radius: 20px;
        padding: 3rem;
        background: linear-gradient(145deg, rgba(255,255,255,0.9), rgba(240,248,255,0.9));
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #f9a825;
        background: linear-gradient(145deg, rgba(255,255,255,1), rgba(255,248,240,1));
        transform: scale(1.02);
    }
    
    /* Image Display */
    div[data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        max-height: 400px;
        object-fit: cover;
    }
    
    /* Result Cards */
    .result-high {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.4);
    }
    .result-medium {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(255, 152, 0, 0.4);
    }
    .result-low {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(244, 67, 54, 0.4);
    }
    
    /* Progress Bar */
    div[data-testid="stProgress"] div[role="progressbar"] {
        border-radius: 12px;
        height: 12px;
    }
    
    /* Sidebar Enhancement */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        backdrop-filter: blur(20px);
    }
    .sidebar-header {
        color: #2e7d32;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #f9a825, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        h1 { font-size: 2rem; }
        .main-card { padding: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# --- ANIMATED HEADER ---
st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <h1>🌽 AgriScan Pro</h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.3rem; font-weight: 300;'>
        Precision Corn Seed Quality Assessment
    </p>
</div>
""", unsafe_allow_html=True)

# --- UTILITY: LOAD ASSETS (UNCHANGED) ---
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

# --- PREDICTION ENGINE (UNCHANGED) ---
def process_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# Check if model loaded
if model is None:
    st.error("⚠️ Model files not found. Please ensure 'corn_model.h5' and 'classes.json' are in the app directory.")
    st.stop()

# --- ENHANCED LAYOUT ---
col1, col2 = st.columns([1, 2])

# Left Column - Input & Control
with col1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #2e7d32; text-align: center; margin-bottom: 1.5rem;">📤 Input Specimen</h3>', unsafe_allow_html=True)
    
    # Radio buttons with icons
    mode = st.radio(
        "Choose Input Method:",
        ["📁 Upload Image", "📷 Camera Capture"],
        label_visibility="collapsed",
        horizontal=True
    )
    
    file_input = None
    if mode == "📁 Upload Image":
        file_input = st.file_uploader(
            "Drop your seed image here...", 
            type=["jpg", "jpeg", "png"],
            help="Supports JPG, PNG formats"
        )
    else:
        file_input = st.camera_input("Capture seed image")
    
    if file_input:
        analyze = st.button("🚀 ANALYZE QUALITY", use_container_width=True)
    else:
        analyze = False
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Results & Visualization
with col2:
    if file_input is not None:
        image = Image.open(file_input)
        
        # Image metrics card
        st.markdown('<div class="main-card metric-card">', unsafe_allow_html=True)
        col_img1, col_img2 = st.columns([2, 1])
        with col_img1:
            st.image(image, caption="🔍 Specimen Under Analysis", use_column_width=True)
        with col_img2:
            st.metric("Image Size", f"{image.size[0]}×{image.size[1]}px")
            st.metric("File Size", f"{file_input.size/1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze:
            with st.spinner("🔬 Neural Network Processing..."):
                preds = process_and_predict(image, model)
                
                result_idx = np.argmax(preds)
                confidence = np.max(preds) * 100
                raw_label = label_map[result_idx] if label_map else str(result_idx)
                raw_label_clean = raw_label.lower()

                # Quality mapping
                if "healthy" in raw_label_clean:
                    quality_grade = "HIGH QUALITY"
                    result_class = "result-high"
                    status_emoji = "✅"
                    recommendation = "Premium Export Ready"
                elif "discolored" in raw_label_clean or "silkcut" in raw_label_clean:
                    quality_grade = "MEDIUM QUALITY"
                    result_class = "result-medium"
                    status_emoji = "⚠️"
                    recommendation = "Inspect for Moisture/Fungus"
                else:
                    quality_grade = "LOW QUALITY"
                    result_class = "result-low"
                    status_emoji = "❌"
                    recommendation = "Seed Integrity Compromised"

                # Enhanced Results Display
                st.markdown(f'''
                <div class="{result_class}">
                    <h2 style="margin: 0; font-size: 2.5rem; font-weight: 700;">{status_emoji} {quality_grade}</h2>
                    <p style="font-size: 1.2rem; margin: 1rem 0;">Detected: <strong>{raw_label.title()}</strong></p>
                    <div style="font-size: 4rem; font-weight: 300; margin: 1rem 0;">{confidence:.1f}%</div>
                    <p style="font-size: 1.1rem; opacity: 0.9;">{recommendation}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Confidence Visualization
                col_prog1, col_prog2 = st.columns(2)
                with col_prog1:
                    st.progress(confidence / 100)
                with col_prog2:
                    st.metric("Confidence", f"{confidence:.1f}%", f"{confidence:.1f}%")

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Control Panel</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Quick Stats")
    st.info("**Model:** CNN (224×224)")
    st.info("**Classes:** Multi-label")
    st.info("**Accuracy:** Production Ready")
    
    st.markdown("### 🎯 Usage Tips")
    st.markdown("""
    - ✅ Use well-lit images
    - ✅ Single seed centered
    - ✅ Clean background
    - ✅ 300+ pixels recommended
    """)

# Footer
st.markdown("""
<div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.7); font-size: 0.9rem;'>
    🌽 AgriScan Pro | Precision Agriculture AI | Powered by TensorFlow
</div>
""", unsafe_allow_html=True)
