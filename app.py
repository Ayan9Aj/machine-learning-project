import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification App")
st.write("Upload a Brain MRI scan image to classify whether a tumor is detected.")

# Load Trained Model
@st.cache_resource
def load_tumor_model():
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    return model

model = load_tumor_model()

# Image Upload Widget
uploaded_file = st.file_uploader("Choose an MRI Scan Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image (Fixed Warning)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", width="stretch")
    
    with st.spinner("Analyzing MRI Scan..."):
        # Preprocessing matching training pipeline
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized)
        
        # Handle Grayscale / Single-channel Images
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
        img_tensor = img_array.reshape(1, 128, 128, 3)
        
        # Run Prediction
        prediction = model.predict(img_tensor)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx] * 100

    st.divider()

    # Display Results
    if class_idx == 0:
        st.error(f"**Prediction:** Tumor Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.success(f"**Prediction:** No Tumor Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")