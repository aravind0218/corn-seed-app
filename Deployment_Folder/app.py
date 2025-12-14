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

# --- FRONTEND: CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfdf5; }
    h1 {
        color: #2e7d32;
        font-family: 'Helvetica', sans-serif;
        text-align: center;
        border-bottom: 3px solid #f9a825;
        padding-bottom: 10px;
    }
    div.stButton > button {
        background-color: #f9a825;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #c17900;
        color: white;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #2e7d32;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY: LOAD ASSETS (FIXED) ---
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

# --- PREDICTION ENGINE ---
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

# --- FRONTEND UI ---
st.title("🌽 AgriScan Remote")
st.markdown("### Automated Corn Seed Quality Control")
st.write("Determine if your batch is **High**, **Medium**, or **Low** quality.")

# Check if model loaded correctly
if model is None:
    st.warning("⚠️ Please ensure 'corn_model.h5' and 'classes.json' are in the same folder as this app.py file.")
    st.stop()

# Sidebar: Input Selection
with st.sidebar:
    st.header("Input Mode")
    mode = st.radio("Choose source:", ["📁 Upload Image", "📷 Capture Image"])
    st.info("Supported formats: JPG, PNG")

file_input = None

if mode == "📁 Upload Image":
    file_input = st.file_uploader("Upload seed image...", type=["jpg", "jpeg", "png"])
elif mode == "📷 Capture Image":
    file_input = st.camera_input("Take a photo of the seed")

# --- EXECUTION LOGIC ---
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
                raw_label = label_map[result_idx] # e.g., 'healthy', 'broken'
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
            else: # broken or anything else
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
