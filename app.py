"""
Streamlit web application for PigeonPhoto_Automator.
Provides UI for batch upload, review, and manual keypoint adjustment.
"""

import streamlit as st
import sys
import os

# Set environment variables to prevent OpenGL/GUI library loading (for headless servers)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# Prevent OpenGL from being loaded
os.environ['OPENCV_IO_ENABLE_OPENGL'] = '0'
# Force headless mode
# For headless environments, do NOT set DISPLAY to ':0' as it causes X server connection attempts
# The QT_QPA_PLATFORM='offscreen' setting above is sufficient for headless mode.
# Leave DISPLAY unset or empty for truly headless operation.
if 'DISPLAY' in os.environ and os.environ['DISPLAY'] == ':0':
    os.environ['DISPLAY'] = ''  # Override problematic ':0' setting

# Try importing cv2 with better error handling
try:
    import cv2
    # Diagnostic: Check OpenCV version and build info
    cv_version = cv2.__version__
    # Try to get build info to see if it's headless
    try:
        build_info = cv2.getBuildInformation()
        is_headless = 'GUI' not in build_info or 'NO' in build_info.split('GUI:')[1].split('\n')[0] if 'GUI:' in build_info else True
    except:
        is_headless = None
    
    # Log diagnostic info (only in debug mode)
    if os.environ.get('STREAMLIT_DEBUG', ''):
        st.info(f"OpenCV version: {cv_version}")
        if is_headless is not None:
            st.info(f"Headless build: {is_headless}")
except ImportError as e:
    st.error(f"Failed to import OpenCV: {e}")
    st.error("This usually means opencv-python-headless is not installed correctly.")
    st.error("Please check that requirements.txt includes 'opencv-python-headless>=4.8.0'")
    st.error(f"Python version: {sys.version}")
    st.error(f"Python path: {sys.executable}")
    st.stop()
except Exception as e:
    # Handle libGL.so.1 and other system library errors
    error_msg = str(e)
    if 'libGL' in error_msg or 'libGL.so' in error_msg:
        st.error("OpenCV is trying to load OpenGL libraries (libGL.so.1).")
        st.error("This means the regular 'opencv-python' package is installed instead of 'opencv-python-headless'.")
        st.error("Solution: Ensure requirements.txt uses 'opencv-python-headless' and explicitly excludes 'opencv-python'.")
        st.error(f"Full error: {e}")
    else:
        st.error(f"Unexpected error importing OpenCV: {e}")
    st.stop()

import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, Tuple, Optional

try:
    from src.pipeline import PigeonPipeline
    from src.keypoints import KeypointDetector, visualize_keypoints
    import src.config as config
except ImportError as e:
    st.error(f"Failed to import project modules: {e}")
    st.error("Please ensure all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PigeonPhoto Automator",
    page_icon="🕊️",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'keypoints' not in st.session_state:
    st.session_state.keypoints = None
if 'body_image' not in st.session_state:
    st.session_state.body_image = None
if 'eye_image' not in st.session_state:
    st.session_state.eye_image = None
if 'result_image' not in st.session_state:
    st.session_state.result_image = None


def initialize_pipeline():
    """Initialize the pipeline if not already done."""
    if st.session_state.pipeline is None:
        with st.spinner("Loading pipeline..."):
            try:
                st.session_state.pipeline = PigeonPipeline()
                st.success("Pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
                return False
    return True


def main():
    """Main application."""
    st.title("🕊️ PigeonPhoto Automator")
    st.markdown("Transform raw racing pigeon photographs into 'Golden Prince' standard portraits")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Bird information
        bird_name = st.text_input("Bird Name", value="Bird")
        ring_number = st.text_input("Ring Number", value="0000")
        
        # Text overlay settings
        st.subheader("Text Overlay")
        text_position = st.selectbox(
            "Text Position",
            ["bottom-right", "bottom-left", "top-right", "top-left"],
            index=0
        )
        text_offset_x = st.slider("Offset X", 0, 200, 50)
        text_offset_y = st.slider("Offset Y", 0, 200, 50)
        
        # Processing options
        st.subheader("Processing Options")
        use_manual_keypoints = st.checkbox("Use Manual Keypoint Adjustment", value=False)
        show_keypoints = st.checkbox("Show Keypoints on Preview", value=True)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Upload", "Review & Adjust", "Result"])
    
    # Tab 1: Upload
    with tab1:
        st.header("Upload Images")
        st.markdown("Upload a pair of images: Body shot and Eye macro")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Body Image")
            body_file = st.file_uploader(
                "Upload Body Image (Body_Raw.jpg)",
                type=['jpg', 'jpeg', 'png'],
                key="body_upload"
            )
            
            if body_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(body_file.read())
                    body_path = tmp_file.name
                
                # Load and display
                body_image = cv2.imread(body_path)
                if body_image is not None:
                    body_image_rgb = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
                    st.image(body_image_rgb, caption="Body Image", use_container_width=True)
                    st.session_state.body_image = body_image
                    st.session_state.body_path = body_path
        
        with col2:
            st.subheader("Eye Macro Image")
            eye_file = st.file_uploader(
                "Upload Eye Macro (Eye_Macro.jpg)",
                type=['jpg', 'jpeg', 'png'],
                key="eye_upload"
            )
            
            if eye_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(eye_file.read())
                    eye_path = tmp_file.name
                
                # Load and display
                eye_image = cv2.imread(eye_path)
                if eye_image is not None:
                    eye_image_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
                    st.image(eye_image_rgb, caption="Eye Macro", use_container_width=True)
                    st.session_state.eye_image = eye_image
                    st.session_state.eye_path = eye_path
        
        # Process button
        if st.button("Process Images", type="primary"):
            if 'body_path' in st.session_state and 'eye_path' in st.session_state:
                if initialize_pipeline():
                    with st.spinner("Processing images..."):
                        try:
                            # Detect keypoints
                            detector = KeypointDetector()
                            keypoints = detector.detect(st.session_state.body_image)
                            keypoints = detector.interpolate_missing_keypoints(keypoints)
                            
                            # Validate
                            is_valid, missing = detector.validate_keypoints(keypoints)
                            if not is_valid:
                                st.warning(f"Missing critical keypoints: {missing}. "
                                         f"You can adjust them manually in the Review tab.")
                            
                            st.session_state.keypoints = keypoints
                            
                            # Process
                            result = st.session_state.pipeline.process(
                                st.session_state.body_path,
                                st.session_state.eye_path,
                                name=bird_name,
                                ring_number=ring_number,
                                keypoints=keypoints
                            )
                            
                            st.session_state.result_image = result
                            st.success("Processing complete! Check the Result tab.")
                            
                        except Exception as e:
                            st.error(f"Processing failed: {e}")
                            st.exception(e)
            else:
                st.error("Please upload both body and eye images first.")
    
    # Tab 2: Review & Adjust
    with tab2:
        st.header("Review & Manual Keypoint Adjustment")
        st.markdown("Review detected keypoints and adjust if needed")
        
        if st.session_state.body_image is not None:
            if st.session_state.keypoints is None:
                st.info("Please process images first in the Upload tab.")
            else:
                # Display image with keypoints
                if show_keypoints:
                    vis_image = visualize_keypoints(
                        st.session_state.body_image.copy(),
                        st.session_state.keypoints
                    )
                    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    st.image(vis_image_rgb, caption="Detected Keypoints", use_container_width=True)
                
                # Manual adjustment (simplified - in production, use interactive canvas)
                st.subheader("Manual Keypoint Adjustment")
                st.info("In a full implementation, this would allow dragging keypoints on the image.")
                
                # Display keypoint coordinates
                with st.expander("Keypoint Coordinates"):
                    for kp_name, (x, y, conf) in st.session_state.keypoints.items():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            new_x = st.number_input(f"{kp_name} X", value=x, key=f"{kp_name}_x")
                        with col2:
                            new_y = st.number_input(f"{kp_name} Y", value=y, key=f"{kp_name}_y")
                        with col3:
                            st.metric("Confidence", f"{conf:.2f}")
                        
                        # Update keypoint if changed
                        if new_x != x or new_y != y:
                            st.session_state.keypoints[kp_name] = (new_x, new_y, conf)
                
                if st.button("Re-process with Adjusted Keypoints"):
                    if initialize_pipeline():
                        with st.spinner("Re-processing..."):
                            try:
                                result = st.session_state.pipeline.process(
                                    st.session_state.body_path,
                                    st.session_state.eye_path,
                                    name=bird_name,
                                    ring_number=ring_number,
                                    keypoints=st.session_state.keypoints
                                )
                                st.session_state.result_image = result
                                st.success("Re-processing complete!")
                            except Exception as e:
                                st.error(f"Re-processing failed: {e}")
        else:
            st.info("Please upload images in the Upload tab first.")
    
    # Tab 3: Result
    with tab3:
        st.header("Final Result")
        
        if st.session_state.result_image is not None:
            result_rgb = cv2.cvtColor(st.session_state.result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Golden Prince Portrait", use_container_width=True)
            
            # Download button
            output_filename = f"{bird_name}_{ring_number}_golden_prince.jpg"
            
            # Convert to bytes for download
            is_success, buffer = cv2.imencode(".jpg", st.session_state.result_image,
                                              [cv2.IMWRITE_JPEG_QUALITY, config.OUTPUT_QUALITY])
            if is_success:
                st.download_button(
                    label="Download Result",
                    data=buffer.tobytes(),
                    file_name=output_filename,
                    mime="image/jpeg"
                )
        else:
            st.info("No result available. Please process images in the Upload tab.")


if __name__ == "__main__":
    main()

