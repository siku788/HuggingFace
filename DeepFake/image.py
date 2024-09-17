import streamlit as st
from transformers import pipeline
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the pipeline
@st.cache_resource
def load_model():
    logger.info("Loading the model...")
    return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

pipe = load_model()

# Streamlit app
st.title("Image Deepfake Detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    logger.info(f"Image uploaded: {uploaded_file.name}")
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform classification
    logger.info("Performing classification...")
    results = pipe(image)

    # Determine if the image is real or fake
    top_result = results[0]
    is_fake = top_result['label'] == 'fake'
    
    # Display results
    st.subheader("Classification Results:")
    for result in results:
        st.write(f"Label: {result['label']}, Score: {result['score']:.4f}")
    
    # Display tag
    tag_color = "red" if is_fake else "green"
    tag_text = "FAKE" if is_fake else "REAL"
    st.markdown(f"<h2 style='color: {tag_color};'>Image is: {tag_text}</h2>", unsafe_allow_html=True)

    logger.info(f"Classification complete. Result: {tag_text}")

# Display logs in the Streamlit app
with st.expander("View Logs"):
    st.text(f"Model loaded: {pipe is not None}")
    if uploaded_file:
        st.text(f"Image processed: {uploaded_file.name}")
        st.text(f"Classification result: {tag_text}")