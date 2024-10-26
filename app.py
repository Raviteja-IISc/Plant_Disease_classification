import streamlit as st
import tensorflow as tf
from PIL import Image
import time
from utils import load_model, predict


# Page config
st.set_page_config(page_title="Plant Disease Detection App",
                   page_icon="app-images/logo-02.png")

# Page title
st.title("Plant Disease Detection")
st.image("app-images/logo-01.png")
st.write("\n\n")

# Load the TFLite model and labels
interpreter = load_model("plant_model_5Classes.tflite")

class_names = ['Healthy', 'Powdery', 'Rust', 'Slug', 'Spot']

# Streamlit app
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Make prediction when button is clicked
    if st.button("Classify"):
        start_time = time.time()
        predicted_class_name, probability = predict(image, class_names, interpreter)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        st.success(f"Predicted Class: {predicted_class_name} with Confidence {probability:.2f}"
                   f" in {inference_time:.2f} ms")
