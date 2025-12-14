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
        background: linear-gradient(135deg, #fcfdf5 0%, #e8f5e9 100%);
    }

    /* TEXT COLORS (Force Dark Text) */
    h1, h2, h3, p, div, label, span {
        color: #2c3e50 !important;
    }

    /* CARD CONTAINER STYLE */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px dashed #2E7D32;
    }

    /* HEADER STYLE */
    h1 {
        color: #2E7D32 !important; /* Forest Green */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
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
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(to right, #1B5E20, #2E7D32);
        color: white !important;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC: LOAD ASSETS (Robust Pathing) ---
@st.cache_resource
def load_assets():
    # 1. Find the folder where this app.py is running
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the full paths to the files
    # Note: We look in the SAME folder as app.py
    model_path = os.path.join(base_dir, 'corn_model.h5')
    json_path = os.path.join(base_dir, 'classes.json')

    # 3. Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return None, None
        
    # 4. Load Class Mappings
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

st.markdown("<h1>🌽 AgriScan Remote</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Automated Corn Seed Quality Control</p>", unsafe_allow_html=True)

if model is None:
    st.error("🚨 System Error: 'corn_model.h5' is MISSING from the server.")
    st.info("The file exists on your PC, but it was not uploaded to GitHub.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Control Panel")
    mode = st.radio("Input Source", ["Upload Image", "Camera Capture"])
    st.info("System Online: TensorFlow CPU")

file_input = None

# Input Section
st.write("") 
if mode == "Upload Image":
    file_input = st.file_uploader("Drop your corn seed image here", type=["jpg", "jpeg", "png"])
else:
    file_input = st.camera_input("Capture Specimen")

# Results Section
if file_input is not None:
    # Display Input
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### Input Specimen")
        image = Image.open(file_input)
        st.image(image, use_column_width=True)
    
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
                    grade = "HIGH QUALITY"
                    color = "#2E7D32" # Green
                    msg = "Seed is healthy and suitable for export."
                    icon = "✅"
                elif "discolored" in raw_label or "silkcut" in raw_label:
                    grade = "MEDIUM QUALITY"
                    color = "#F9A825" # Yellow
                    msg = "Minor discoloration detected."
                    icon = "⚠️"
                else:
                    grade = "LOW QUALITY"
                    color = "#C62828" # Red
                    msg = "Seed is broken or damaged."
                    icon = "❌"

                # Pro Result Card
                st.markdown(f"""
                <div style="background-color: {color}15; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <h3 style="color: {color} !important; margin:0;">{icon} {grade}</h3>
                    <p style="margin-top: 5px; font-weight: bold; color: #333 !important;">Confidence: {confidence:.2f}%</p>
                    <p style="font-size: 14px; color: #555 !important;">{msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(int(confidence))
