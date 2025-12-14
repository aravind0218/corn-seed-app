import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AgriScan AI",
    page_icon="🌽",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING (CSS) ---
# This forces the "Agri-Tech" theme regardless of your system's Dark Mode
st.markdown("""
    <style>
    /* IMPORT GOOGLE FONT */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* GLOBAL TEXT & BACKGROUND */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* MAIN BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* TEXT COLORS (Force Dark Text) */
    h1, h2, h3, p, div, label, span {
        color: #2c3e50 !important;
    }

    /* CARD CONTAINER STYLE */
    .css-1r6slb0, .stFileUploader {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    /* HEADER STYLE */
    h1 {
        color: #2E7D32 !important; /* Forest Green */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    
    /* SUBHEADER */
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }

    /* BUTTON STYLE */
    div.stButton > button {
        background: linear-gradient(to right, #2E7D32, #43A047);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.3s;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        background: linear-gradient(to right, #1B5E20, #2E7D32);
    }

    /* SIDEBAR STYLE */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* UPLOAD BOX STYLING */
    [data-testid="stFileUploader"] section {
        background-color: #f8f9fa;
        border: 2px dashed #2E7D32;
    }
    
    /* SUCCESS BOX */
    div.stAlert {
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC: LOAD ASSETS ---
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'corn_model.h5')
    json_path = os.path.join(base_dir, 'classes.json')

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return None, None
        
    try:
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        label_map = {v: k for k, v in class_indices.items()}
    except Exception as e:
        label_map = None
        
    return model, label_map

model, label_map = load_assets()

# --- 4. LOGIC: PREDICTION ---
def process_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# --- 5. UI LAYOUT ---

# Top Header Area
st.markdown("<h1>🌽 AgriScan Remote</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Artificial Intelligence Seed Quality Control System</p>", unsafe_allow_html=True)

# Main Container
if model is None:
    st.error("🚨 System Error: Model files not found. Please check deployment.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=50)
    st.markdown("### Control Panel")
    mode = st.radio("Input Source", ["Upload Image", "Camera Capture"])
    st.markdown("---")
    st.markdown("**System Status:** \n🟢 Online \n⚡ TensorFlow CPU")

file_input = None

# Input Section
st.write("") # Spacer
if mode == "Upload Image":
    file_input = st.file_uploader("Drop your corn seed image here", type=["jpg", "jpeg", "png"])
else:
    file_input = st.camera_input("Capture Specimen")

# Results Section
if file_input is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### Input Specimen")
        image = Image.open(file_input)
        st.image(image, use_column_width=True, caption="Analyzed Image")
    
    with col2:
        st.markdown("##### Analysis Results")
        if st.button("Run Diagnostics"):
            with st.spinner("Processing neural network..."):
                preds = process_and_predict(image, model)
                result_idx = np.argmax(preds)
                confidence = np.max(preds) * 100
                
                if label_map:
                    raw_label = label_map[result_idx].lower()
                else:
                    raw_label = str(result_idx)

                # Quality Logic
                if "healthy" in raw_label:
                    grade = "EXPORT QUALITY"
                    color = "#2E7D32" # Green
                    msg = "Seed is healthy and viable."
                    icon = "✅"
                elif "discolored" in raw_label or "silkcut" in raw_label:
                    grade = "PROCESSING QUALITY"
                    color = "#F9A825" # Yellow
                    msg = "Minor discoloration detected."
                    icon = "⚠️"
                else:
                    grade = "REJECTED"
                    color = "#C62828" # Red
                    msg = "Seed is broken or damaged."
                    icon = "❌"

                # Custom Result Card
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                    <h3 style="color: {color} !important; margin:0;">{icon} {grade}</h3>
                    <p style="margin-top: 5px; font-weight: bold;">Confidence: {confidence:.2f}%</p>
                    <p style="font-size: 14px;">{msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(int(confidence))
