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
    layout="wide"
)

# --- FRONTEND: NEW CUSTOM CSS ONLY ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
.stApp { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Poppins', sans-serif;
}
h1 {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 3rem !important;
    text-align: center !important;
    text-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    margin-bottom: 1rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4CAF50, #45a049) !important;
    color: white !important;
    border-radius: 16px !important;
    padding: 16px 32px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 8px 24px rgba(76,175,80,0.4) !important;
    width: 100% !important;
    height: 60px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(76,175,80,0.6) !important;
}
[data-testid="stFileUploader"] {
    border: 3px dashed #4CAF50 !important;
    border-radius: 20px !important;
    padding: 3rem !important;
    background: rgba(255,255,255,0.9) !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# --- UTILITY: LOAD ASSETS (EXACT SAME) ---
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

# --- PREDICTION ENGINE (EXACT SAME) ---
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

# --- FRONTEND UI (IMPROVED BUT SAME LOGIC) ---
st.title("🌽 AgriScan Pro")
st.markdown("### Automated Corn Seed Quality Control")
st.markdown("**Determine if your batch is HIGH, MEDIUM, or LOW quality.**")

# Check if model loaded correctly (EXACT SAME)
if model is None:
    st.warning("⚠️ Please ensure 'corn_model.h5' and 'classes.json' are in the same folder as this app.py file.")
    st.stop()

# Sidebar: Input Selection (EXACT SAME LOGIC)
with st.sidebar:
    st.header("🔧 Input Mode")
    mode = st.radio("Choose source:", ["📁 Upload Image", "📷 Capture Image"])
    st.info("Supported formats: JPG, PNG")

file_input = None

if mode == "📁 Upload Image":
    file_input = st.file_uploader("Upload seed image...", type=["jpg", "jpeg", "png"])
elif mode == "📷 Capture Image":
    file_input = st.camera_input("Take a photo of the seed")

# --- EXECUTION LOGIC (100% IDENTICAL) ---
if file_input is not None:
    # Display Input
    image = Image.open(file_input)
    st.image(image, caption="Input Specimen", width=400)
    
    if st.button("🚀 Analyze Quality"):
        with st.spinner("Processing neural network..."):
            preds = process_and_predict(image, model)
            
            # Get highest probability
            result_idx = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            # Map raw label to Quality Grade
            if label_map:
                raw_label = label_map[result_idx] # e.g., 'healthy', 'broken'
            else:
                raw_label = str(result_idx)
            
            # Quality Logic Mapping (EXACT SAME)
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
            else: # broken or anything else
                quality_grade = "LOW QUALITY"
                color = "red"

            # Display Results (ENHANCED VISUALS ONLY)
            st.markdown("---")
            st.markdown(f"### 🎯 **Grade: :{color}[{quality_grade}]** ✨")
            st.caption(f"🔍 Detected Class: {raw_label.title()} | Confidence: {confidence:.2f}%")
            
            # Progress bar for visual confidence
            st.progress(int(confidence))
            
            if quality_grade == "HIGH QUALITY":
                st.success("✅ **APPROVED**: Suitable for premium export.")
            elif quality_grade == "MEDIUM QUALITY":
                st.warning("⚠️ **ATTENTION**: Check for fungal infection or moisture damage.")
            else:
                st.error("❌ **REJECTED**: Seed integrity compromised.")

st.markdown("---")
st.markdown("*Powered by TensorFlow & Streamlit 🌽*")
