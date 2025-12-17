
import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import io

from src.pipeline import PigeonPipeline
import src.config as config
from src.keypoints import visualize_keypoints

# Set page config
st.set_page_config(
    page_title="PigeonPhoto Automator",
    page_icon="🐦",
    layout="wide"
)

def main():
    st.title("🐦 PigeonPhoto Automator")
    st.markdown("Transform raw racing pigeon photographs into 'Golden Prince' standard portraits.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Text Overlay")
        bird_name = st.text_input("Bird Name", "My Champion")
        ring_number = st.text_input("Ring Number", "2024-123456")
        
        st.subheader("Processing")
        # Initialize pipeline (cached to avoid reloading model)
        @st.cache_resource
        def get_pipeline():
            return PigeonPipeline()
        
        try:
            pipeline = get_pipeline()
            st.success("System initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return

    # Main content area
    col1, col2 = st.columns(2)
    
    body_file = None
    eye_file = None
    
    with col1:
        st.subheader("1. Upload Body Image")
        body_file = st.file_uploader("Choose body image", type=['jpg', 'jpeg', 'png'], key="body")
        if body_file:
            st.image(body_file, caption="Body Image", use_container_width=True)
            
    with col2:
        st.subheader("2. Upload Eye Image")
        eye_file = st.file_uploader("Choose eye macro", type=['jpg', 'jpeg', 'png'], key="eye")
        if eye_file:
            st.image(eye_file, caption="Eye Macro", use_container_width=True)
            
    if body_file and eye_file:
        st.divider()
        st.subheader("3. Process & Result")
        
        if st.button("Process Images", type="primary", use_container_width=True):
            with st.spinner("Processing... This may take up to 30 seconds."):
                try:
                    # Save uploaded files temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_body:
                        tmp_body.write(body_file.getvalue())
                        body_path = tmp_body.name
                        
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_eye:
                        tmp_eye.write(eye_file.getvalue())
                        eye_path = tmp_eye.name
                        
                    # Process
                    result = pipeline.process(
                        body_image_path=body_path,
                        eye_image_path=eye_path,
                        name=bird_name,
                        ring_number=ring_number
                    )
                    
                    # Cleanup temp files
                    Path(body_path).unlink()
                    Path(eye_path).unlink()
                    
                    # Convert BGR to RGB for display
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    
                    # Display result
                    st.image(result_rgb, caption="Golden Prince Portrait", use_container_width=True)
                    
                    # Download button
                    # Convert to byte array for download
                    is_success, buffer = cv2.imencode(".jpg", result)
                    if is_success:
                        byte_io = io.BytesIO(buffer)
                        st.download_button(
                            label="Download Portrait",
                            data=byte_io,
                            file_name=f"{bird_name}_{ring_number}.jpg",
                            mime="image/jpeg"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()
