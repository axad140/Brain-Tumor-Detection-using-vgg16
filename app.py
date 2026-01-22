import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image, ImageOps

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    .result-card {
        padding: 20px; 
        border-radius: 10px; 
        text-align: center; 
        color: white; 
        font-size: 26px; 
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .safe { background-color: #2ecc71; }
    .danger { background-color: #e74c3c; }
    .debug-text { font-size: 12px; color: #555; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR: SETTINGS & SYMPTOMS ---
with st.sidebar:
    st.title("⚙️ Calibration Settings")
    
    # Toggle for Preprocessing
    proc_mode = st.radio(
        "Preprocessing Mode:",
        ["Rescale (1./255)", "VGG Standard"],
        index=0,
        help="Try switching this if predictions are random or low confidence."
    )
    
    # Toggle for Class Swapping
    invert_result = st.checkbox(
        "Swap Yes/No Logic", 
        value=False,
        help="Check this if the model predicts 'Tumor' on healthy images."
    )

    st.markdown("---")
    st.header("Patient Symptoms")
    s1 = st.checkbox("Severe Headaches")
    s2 = st.checkbox("Nausea & Vomiting")
    s3 = st.checkbox("Vision Problems")
    s4 = st.checkbox("Seizures")
    
    if sum([s1, s2, s3, s4]) >= 2:
        st.warning("⚠️ High Risk Clinical Symptoms")

# --- 4. LOAD MODEL ---
@st.cache_resource
def get_model():
    # Ensure this matches your file name exactly
    model = load_model('vgg16_brain_tumor.h5')
    return model

# --- 5. MAIN APPLICATION ---
st.title("🧠 Brain Tumor Detection System")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload MRI Scan", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Patient MRI", use_container_width=True)

with col2:
    if uploaded_file:
        st.write("### Analysis Report")
        
        try:
            model = get_model()
            
            # --- IMAGE PREPROCESSING ---
            # 1. Resize to 224x224
            img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            
            # 2. Convert to RGB
            if img_resized.mode != "RGB":
                img_resized = img_resized.convert("RGB")
                
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            # 3. APPLY CHOSEN PREPROCESSING (From Sidebar)
            if proc_mode == "Rescale (1./255)":
                # Most common for custom training
                img_array = img_array / 255.0
            else:
                # VGG16 Standard (subtracts mean RGB)
                img_array = preprocess_input(img_array)

            # --- PREDICTION ---
            if st.button("Run Diagnostics"):
                with st.spinner("Analyzing neural patterns..."):
                    prediction = model.predict(img_array)
                    
                    # --- RESULT LOGIC ---
                    class_idx = np.argmax(prediction)
                    confidence = float(np.max(prediction))
                    
                    # Default Assumption: 0=No, 1=Yes (Alphabetical)
                    if class_idx == 1:
                        is_tumor = True
                    else:
                        is_tumor = False
                    
                    # Apply "Swap" from Sidebar if checked
                    if invert_result:
                        is_tumor = not is_tumor

                    # --- DISPLAY ---
                    if is_tumor:
                        result_text = "Tumor Detected (Positive)"
                        css_class = "danger"
                        st.markdown(f'<div class="result-card {css_class}">{result_text}</div>', unsafe_allow_html=True)
                        st.error("Action: Immediate consultation recommended.")
                    else:
                        result_text = "No Tumor Detected (Negative)"
                        css_class = "safe"
                        st.markdown(f'<div class="result-card {css_class}">{result_text}</div>', unsafe_allow_html=True)
                        st.success("Action: Routine checkup recommended.")
                    
                    st.progress(int(confidence * 100))
                    st.write(f"**AI Confidence:** {confidence*100:.2f}%")
                    
                    # Debug Info (Helps you see what is happening)
                    st.markdown("---")
                    st.markdown(f"""
                    <div class="debug-text">
                    <b>Debug Data:</b><br>
                    Raw Output: {prediction}<br>
                    Predicted Index: {class_idx}<br>
                    Processing Mode: {proc_mode}
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Ensure 'vgg16_brain_tumor.h5' is in the folder.")