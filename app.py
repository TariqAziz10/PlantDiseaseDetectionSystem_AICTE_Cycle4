
import streamlit as st
st.set_page_config(page_title='Plant Disease Detection', layout='wide')

import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import base64


st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #505081;
        color: #ffffff;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #272757;
        color: #ffffff;
        padding: 20px;
    }
    /* Sidebar header text */
    .sidebar-header {
        text-align: center;
        margin-bottom: 20px;
    }
    /* Dark themed buttons */
    div.stButton > button {
        background-color: #0F0E47;
        color: #ffffff;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #2B2B2B;
    }
    /* Ensure navigation selectbox in the sidebar shows pointer on hover */
    [data-testid="stSidebar"] select {
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)


def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None


def model_predict(image_path):
    model = tf.keras.models.load_model('plant_disease_cnn_model.keras')
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, H, W, C)
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction


def home_page():
    st.markdown("""
        <h1 style="text-align: center; color: #0F0E47;">🌱 Plant Disease Prediction System 🌱</h1>
        <p style="text-align: center; font-size: 1.2em;">Upload an image of a plant leaf and let our AI detect any disease.</p>
    """, unsafe_allow_html=True)

def predict_page():
    st.markdown("<h2 style='color: #0F0E47;'>📸 Upload Plant Leaf Image</h2>", unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        save_path = os.path.join(os.getcwd(), "uploaded_image.jpg")
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())
        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing Image..."):
                result_index = model_predict(save_path)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
                prediction = class_name[result_index]
                st.success(f"✅ Prediction: {prediction}")
                st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #333333; color: #ffffff;">
                        <h3>Prediction Successful!</h3>
                        <p style="font-size: 1.2em;"><strong>{prediction}</strong></p>
                    </div>
                """, unsafe_allow_html=True)

def about_page():
    st.markdown("""
        <h2 style="color: #1e88e5;">📚 About This Model</h2>
        <p>Our Plant Disease Detection application utilizes advanced deep learning techniques, specifically Convolutional Neural Networks (CNNs), to accurately identify various plant diseases from leaf images</p>
        <hr>
        <h4>How Our Model Works</h4>
        <ul>
            <li><b>Image Input:</b> Users upload a clear image of a plant leaf using the app</li>
            <li><b>Feature Extraction:</b> The CNN processes the image to detect unique features, such as color variations, shapes, and textures, that may indicate a specific disease</li>
            <li><b>Classification:</b> Based on these features, the model compares them to its learned knowledge from a vast dataset of healthy and diseased leaf images to determine the presence and type of disease</li>
        </ul>
        <h4>Benefits of Using Our App</h4>
        <ul>
            <li><b>Early Detection:</b> Identifying diseases early helps in taking prompt actions to protect crops</li>
            <li><b>Educational:</b> Users can learn about various plant diseases and their symptoms through the app's informative sections</li>
        </ul>
        <hr>
        <p>By leveraging CNNs, our app provides an efficient and accessible tool for farmers, gardeners, and plant enthusiasts to monitor plant health and address issues proactively</p>
        
    """, unsafe_allow_html=True)


img_base64 = get_base64_of_bin_file("Disease.png")
if img_base64:
    img_html = f'<img src="data:image/png;base64,{img_base64}" width="180" alt="Disease Detection Logo">'
else:
    img_html = "<p>Image not found.</p>"

with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-header">
            {img_html}
            <h2 style="color: #ffffff;">🌿 Plant Disease Detection</h2>
            <p>Identify plant diseases for sustainable agriculture.</p>
        </div>
    """, unsafe_allow_html=True)
    page = st.selectbox("Navigation", ["Home", "Predict Disease", "About Model"])


if page == "Home":
    home_page()
elif page == "Predict Disease":
    predict_page()
elif page == "About Model":
    about_page()
