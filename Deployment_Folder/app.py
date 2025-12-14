import streamlit as st
import os

st.title("📂 File System Debugger")

# 1. Where am I running?
current_dir = os.getcwd()
st.write(f"**Current Working Directory:** `{current_dir}`")

# 2. What files are here?
st.write("**Files in Current Directory:**")
files = os.listdir(current_dir)
st.code(files)

# 3. Search for the model
found = False
for root, dirs, files in os.walk(current_dir):
    if "corn_model.h5" in files:
        st.success(f"✅ FOUND MODEL AT: `{os.path.join(root, 'corn_model.h5')}`")
        found = True
        break

if not found:
    st.error("❌ Model file NOT found in any subfolder.")
