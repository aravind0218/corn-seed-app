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
    layout="wide", # Uses the full screen width for a dashboard feel
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE (The "Brain" of the Dashboard) ---
# This remembers data while you are using the app
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'counts' not in st.session_state:
    st.session_state['counts'] = {'High': 0, 'Medium': 0, 'Low': 0}

# --- 3. CUSTOM DASHBOARD STYLING ---
st.markdown("""
<style>
    /* Dashboard Background */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica', sans-serif;
        color: #2c3e50;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #2E7D32;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. BACKEND LOGIC (Same as before) ---
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'corn_model.h5')
    json_path = os.path.join(base_dir, 'classes.json')

    try:
        model = tf.keras.models.load_model(model_path)
    except:
        return None, None
        
    try:
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        label_map = {v: k for k, v in class_indices.items()}
    except:
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

# Top Header
st.title("🌽 AgriScan Live Dashboard")
st.markdown("Real-time Quality Control Analytics")
st.markdown("---")

# Row 1: The Metrics (KPIs)
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

# Row 2: Main Interface (Scanner + Charts)
main_col_1, main_col_2 = st.columns([1, 2])

with main_col_1:
    st.subheader("🔍 Seed Scanner")
    
    # Check Model
    if model is None:
        st.error("🚨 Model not found. Please upload 'corn_model.h5'.")
        st.stop()

    # Input
    mode = st.radio("Input Mode", ["Upload", "Camera"], horizontal=True)
    file_input = None
    
    if mode == "Upload":
        file_input = st.file_uploader("Upload Image", type=["jpg", "png"], label_visibility="collapsed")
    else:
        file_input = st.camera_input("Capture")

    if file_input:
        image = Image.open(file_input)
        st.image(image, caption="Current Specimen", use_column_width=True)
        
        if st.button("Run Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                preds = process_and_predict(image, model)
                result_idx = np.argmax(preds)
                confidence = np.max(preds) * 100
                
                if label_map:
                    raw_label = label_map[result_idx].lower()
                else:
                    raw_label = str(result_idx)

                # Determine Grade
                if "healthy" in raw_label:
                    grade = "High"
                    color = "green"
                elif "discolored" in raw_label or "silkcut" in raw_label:
                    grade = "Medium"
                    color = "orange"
                else:
                    grade = "Low"
                    color = "red"

                # Update Stats (Session State)
                st.session_state['counts'][grade] += 1
                st.session_state['history'].append({
                    "Grade": grade,
                    "Confidence": f"{confidence:.1f}%",
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                
                # Show Result
                st.success(f"**Result: {grade} Quality** ({confidence:.1f}%)")

with main_col_2:
    st.subheader("📊 Batch Analytics")
    
    # 1. Create Dataframe from Counts
    chart_data = pd.DataFrame([
        {"Quality": "High", "Count": st.session_state['counts']['High'], "Color": "#2ecc71"},
        {"Quality": "Medium", "Count": st.session_state['counts']['Medium'], "Color": "#f1c40f"},
        {"Quality": "Low", "Count": st.session_state['counts']['Low'], "Color": "#e74c3c"}
    ])

    # 2. Altair Bar Chart
    if total_scans > 0:
        c = alt.Chart(chart_data).mark_bar().encode(
            x='Quality',
            y='Count',
            color=alt.Color('Quality', scale=alt.Scale(domain=['High', 'Medium', 'Low'], range=['#2ecc71', '#f1c40f', '#e74c3c'])),
            tooltip=['Quality', 'Count']
        ).properties(height=300)
        
        st.altair_chart(c, use_container_width=True)
        
        # 3. Recent History Table
        st.write("**Recent Scans Log:**")
        if len(st.session_state['history']) > 0:
            # Show last 5 entries
            df_hist = pd.DataFrame(st.session_state['history'])
            st.dataframe(df_hist.tail(5), use_container_width=True)
    else:
        st.info("Waiting for data... Scan a seed to see analytics.")
