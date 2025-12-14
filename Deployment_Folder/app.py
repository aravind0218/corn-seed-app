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

# --- UTILITY: LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load Model
    try:
        model = tf.keras.models.load_model('corn_model.h5')
    except:
        return None, None
        
    # Load Class Indices
    try:
        with open('classes.json', 'r') as f:
            class_indices = json.load(f)
        # Invert dictionary to map ID -> Class Name
        label_map = {v: k for k, v in class_indices.items()}
    except:
        label_map = None
        
    return model, label_map

model, label_map = load_assets()

# --- PREDICTION ENGINE ---
def process_and_predict(image_data, model):
    # 1. Preprocess Image to match Training (224x224, Normalized)
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Reshape for Batch (1, 224, 224, 3)
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # 2. Inference
    prediction = model.predict(img_reshape)
    return prediction

# --- FRONTEND ---
st.title("🌽 AgriScan Remote")
st.markdown("### Automated Corn Seed Quality Control")
st.write("Determine if your batch is **High**, **Medium**, or **Low** quality.")

# Check if model loaded correctly
if model is None:
    st.error("🚨 System Error: Model not found. Please upload 'corn_model.h5' and 'classes.json'.")
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
            raw_label = label_map[result_idx] # e.g., 'healthy', 'broken'
            
            # Quality Logic Mapping
            quality_grade = ""
            color = ""
            
            if "healthy" in raw_label.lower():
                quality_grade = "HIGH QUALITY"
                color = "green"
            elif "discolored" in raw_label.lower() or "silkcut" in raw_label.lower():
                quality_grade = "MEDIUM QUALITY"
                color = "orange"
            else: # broken
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