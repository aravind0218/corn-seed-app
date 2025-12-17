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

# --- 2. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'counts' not in st.session_state:
    st.session_state['counts'] = {'High': 0, 'Medium': 0, 'Low': 0}

# --- Set Default to True (Dark Mode) ---
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
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #e6f7ed 50%, #d1fae5 100%);
        color: #022c22;
    }
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, span, div {
        color: #022c22 !important;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    div[data-testid="stToggle"] div[role="switch"] {
        border: 2px solid #4ade80 !important;
    }
    div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background-color: #4ade80 !important;
    }
    div[data-testid="stToggle"] div[role="switch"][aria-checked="false"] {
        background-color: #dcfce7 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #d1fae5;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fefce8 0%, #fef9c3 100%); 
        border-right: 2px solid #eab308;
    }
    h1, h2, h3 {
        font-family: "Playfair Display", "Georgia", serif;
        color: #14532d !important;
        font-weight: 800;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: #ffffff !important;
        border-radius: 10px;
        font-weight: 700;
    }
</style>
"""

# --- DARK MODE CSS ---
DARK_MODE_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    html, body, [class*="css"] {
        color: #e2e8f0 !important;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid rgba(34, 197, 94, 0.4);
    }
    h1, h2, h3 {
        font-family: "Playfair Display", "Georgia", serif;
        color: #f1f5f9 !important;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: #ffffff !important;
        border-radius: 8px;
    }
</style>
"""

if st.session_state['dark_mode']:
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# --- 4. BACKEND LOGIC ---
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
st.title("🌽 AgriScan Live Dashboard")
st.markdown("<p class='small-muted'>Real-time quality control analytics for corn seed batches.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
total_scans = sum(st.session_state['counts'].values())

with col1:
    st.metric("Total Scanned", f"{total_scans}")
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
        st.error("🚨 Model not found.")
        st.stop()

    mode = st.radio("Input Mode", ["Upload", "Camera"], horizontal=True)
    file_input = st.file_uploader("Upload Image", type=["jpg", "png"], label_visibility="collapsed") if mode == "Upload" else st.camera_input("Capture")

    if file_input:
        image = Image.open(file_input)
        st.image(image, caption="Current Specimen", use_container_width=True)
        
        if st.button("Run Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                preds = process_and_predict(image, model)
                result_idx = np.argmax(preds)
                confidence = np.max(preds) * 100
                
                if confidence < 70.0:
                    st.error("🚨 Incorrect photo detected! Please upload a clear seed photo to analyze.")
                else:
                    raw_label = label_map[result_idx].lower() if label_map else str(result_idx)
                    grade = "High" if "healthy" in raw_label else "Medium" if ("discolored" in raw_label or "silkcut" in raw_label) else "Low"

                    st.session_state['counts'][grade] += 1
                    st.session_state['history'].append({"Grade": grade, "Confidence": f"{confidence:.1f}%", "Time": pd.Timestamp.now().strftime("%H:%M:%S")})
                    st.success(f"**Result: {grade} Quality** ({confidence:.1f}%)")import streamlit as st
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
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #e6f7ed 50%, #d1fae5 100%);
        color: #022c22;
    }
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, span, div {
        color: #022c22 !important;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    div[data-testid="stToggle"] div[role="switch"] {
        border: 2px solid #4ade80 !important;
    }
    div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background-color: #4ade80 !important;
    }
    div[data-testid="stToggle"] div[role="switch"][aria-checked="false"] {
        background-color: #dcfce7 !important;
    }
    div[data-testid="stToggle"] div[role="switch"] span {
        background-color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #d1fae5;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fefce8 0%, #fef9c3 100%); 
        border-right: 2px solid #eab308;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important; 
    }
    h1, h2, h3 {
        font-family: "Playfair Display", "Georgia", serif;
        color: #14532d !important;
        font-weight: 800;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: #ffffff !important;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(22, 163, 74, 0.3);
    }
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
        background-color: #ffffff; 
        border: 2px dashed #16a34a; 
    }
</style>
"""

# --- DARK MODE CSS ---
DARK_MODE_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    html, body, [class*="css"] {
        color: #e2e8f0 !important;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    div[data-testid="stToggle"] div[role="switch"] {
        border: 2px solid #4ade80 !important;
    }
    div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background-color: #4ade80 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid rgba(34, 197, 94, 0.4);
    }
    h1, h2, h3 {
        font-family: "Playfair Display", "Georgia", serif;
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
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
            # Fixed the parameter name to use_container_width
            st.dataframe(df_hist.tail(5), use_container_width=True)
    else:
        st.info("Waiting for data... Scan a seed to see analytics.")

with main_col_2:
    st.subheader("📊 Batch Analytics")
    if total_scans > 0:
        chart_data = pd.DataFrame([{"Quality": "High", "Count": st.session_state['counts']['High']}, {"Quality": "Medium", "Count": st.session_state['counts']['Medium']}, {"Quality": "Low", "Count": st.session_state['counts']['Low']}])
        c = alt.Chart(chart_data).mark_bar().encode(x='Quality', y='Count', color=alt.Color('Quality', scale=alt.Scale(domain=['High', 'Medium', 'Low'], range=['#2ecc71', '#f1c40f', '#e74c3c'])))
        st.altair_chart(c, use_container_width=True)
        st.write("**Recent Scans Log:**")
        df_hist = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_hist.tail(5), use_container_width=True) # FIXED LINE
    else:
        st.info("Waiting for data... Scan a seed to see analytics.")
