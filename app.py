import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="RawanAlwadeya/BrainTumorClassifier",
        filename="BrainTumorClassifier.h5"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalize the uploaded image for InceptionV3.
    """
    IMG_SIZE = (299, 299)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detection"])


if page == "Home":
    st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Transfer Learning with InceptionV3</h3>", unsafe_allow_html=True)

    st.write(
        """
        **Brain tumors** occur when abnormal cells form within the brain, potentially causing symptoms such as
        headaches, seizures, or neurological deficits.  
        
        Early detection through imaging like **MRI scans** is critical for treatment planning and improved outcomes.

        This app leverages **Transfer Learning** with **InceptionV3** to classify MRI brain scans into:
        - **⚠️ Brain Tumor Detected**
        - **✅ Healthy Brain (No Tumor Detected)**
        """
    )

    st.image("BrainTumor.jpg",
             caption="MRI Brain Scan Example",
             use_container_width=True)

    st.info("👉 Go to the **Detection** page from the left sidebar to upload an MRI image and get predictions.")


elif page == "Detection":
    st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection</h1>", unsafe_allow_html=True)
    st.write(
        "Upload a brain MRI image below, and the model will predict whether "
        "it shows **signs of a Brain Tumor** or indicates a **Healthy Brain**."
    )

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array, verbose=0)[0][0]
        

        if prediction < 0.5:
            st.error(
                "⚠️ **The model predicts: Brain Tumor likely detected.** "
                "Please consult a qualified healthcare professional for further evaluation."
            )
        else:
            st.success(
                "✅ **The model predicts: Likely Healthy Brain (No Tumor detected).** "
                "For medical certainty, always consult a licensed medical professional."
            )    
