import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import os
import pandas as pd
import altair as alt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriScan Dashboard",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'counts' not in st.session_state:
    st.session_state['counts'] = {'High': 0, 'Medium': 0, 'Low': 0}

# Default to Dark Mode
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True 

# --- 3. THEME TOGGLE & STYLING ---

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state['dark_mode'], key="theme_toggle")
    st.session_state['dark_mode'] = dark_mode
    st.markdown("---")

# --- LIGHT MODE CSS ---
LIGHT_MODE_CSS = """
<style>
/* CSS for light mode goes here */
</style>
"""

# --- DARK MODE CSS ---
DARK_MODE_CSS = """
<style>
/* CSS for dark mode goes here */
</style>
"""

if st.session_state['dark_mode']:
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# --- 4. BACKEND LOGIC: ASSET LOADING ---
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

def process_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# --- 5. DASHBOARD UI LAYOUT ---

st.title("🌽 AgriScan Live Dashboard")
st.markdown("<p class='small-muted'>Real-time quality control analytics for corn seed batches.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
total_scans = sum(st.session_state['counts'].values())

with col1:
    st.metric("Total Scanned", f"{total_scans}", delta=None)
with col2:
    st.metric("High Quality", f"{st.session_state['counts']['High']}", delta="Export Ready")
with col3:
    st.metric("Medium Quality", f"{st.session_state['counts']['Medium']}", delta="- Processing", delta_color="off")
with col4:
    st.metric("Low Quality", f"{st.session_state['counts']['Low']}", delta="- Rejected", delta_color="inverse")

st.markdown("---")

main_col_1, main_col_2 = st.columns([1, 2])

with main_col_1:
    st.subheader("🔍 Seed Scanner")
    if model is None:
        st.error("🚨 Model not found. Please upload 'corn_model.h5' and 'classes.json'.")
        st.stop()

    mode = st.radio("Input Mode", ["Upload", "Camera"], horizontal=True)
    file_input = None
    if mode == "Upload":
        file_input = st.file_uploader("Upload Image", type=["jpg", "png"], label_visibility="collapsed")
    else:
        file_input = st.camera_input("Capture")

    if file_input:
        image = Image.open(file_input)
        st.image(image, caption="Current Specimen", use_container_width=True)
        
        if st.button("Run Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                preds = process_and_predict(image, model)
                result_idx = np.argmax(preds)
                confidence = np.max(preds) * 100
                
                # Validation: Confidence threshold to detect non-seed photos
                if confidence < 70.0:
                    st.error("🚨 Incorrect photo detected! Please upload a clear seed photo to analyze.")
                else:
                    if label_map:
                        raw_label = label_map[result_idx].lower()
                    else:
                        raw_label = str(result_idx)

                    if "healthy" in raw_label:
                        grade = "High"
                    elif "discolored" in raw_label or "silkcut" in raw_label:
                        grade = "Medium"
                    else:
                        grade = "Low"

                    # Update session state history and counts
                    st.session_state['counts'][grade] += 1
                    st.session_state['history'].append({
                        "Grade": grade,
                        "Confidence": f"{confidence:.1f}%",
                        "Time": pd.Timestamp.now().strftime("%H:%M:%S")
                    })
                    
                    st.success(f"**Result: {grade} Quality** ({confidence:.1f}%)")

with main_col_2:
    st.subheader("📊 Batch Analytics")
    
    chart_data = pd.DataFrame([
        {"Quality": "High", "Count": st.session_state['counts']['High'], "Color": "#2ecc71"},
        {"Quality": "Medium", "Count": st.session_state['counts']['Medium'], "Color": "#f1c40f"},
        {"Quality": "Low", "Count": st.session_state['counts']['Low'], "Color": "#e74c3c"}
    ])

    if total_scans > 0:
        c = alt.Chart(chart_data).mark_bar().encode(
            x='Quality',
            y='Count',
            color=alt.Color('Quality', scale=alt.Scale(domain=['High', 'Medium', 'Low'], range=['#2ecc71', '#f1c40f', '#e74c3c'])),
            tooltip=['Quality', 'Count']
        ).properties(height=300)
        
        st.altair_chart(c, use_container_width=True)
        
        st.write("**Recent Scans Log:**")
        if len(st.session_state['history']) > 0:
            df_hist = pd.DataFrame(st.session_state['history'])
            st.dataframe(df_hist.tail(5), use_container_width=True)
    else:
        st.info("Waiting for data... Scan a seed to see analytics.")
