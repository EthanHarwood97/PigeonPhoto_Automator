# PigeonPhoto Automator

Automated web-based pipeline to transform raw racing pigeon photographs into "Golden Prince" standard portraits using precise geometric transformation and surgical component assembly.

## Overview

This project implements a sophisticated computer vision pipeline that:
- Segments pigeons from background using deep learning
- Detects 9 anatomical keypoints using YOLOv8-Pose
- Applies surgical component assembly (Head, Body, Feet) with different transformation strategies
- Harmonizes lighting and blends components seamlessly
- Enhances eye detail with zoom bubble effect
- Produces professional "Golden Prince" standard portraits

## Key Features

- **Surgical Component Assembly**: Head (rigid rotation), Body (TPS warping), Feet (translation)
- **Seamless Blending**: Poisson blending for invisible seams
- **Exposure Matching**: Automatic gamma correction for consistent lighting
- **Eye Enhancement**: Automatic pupil detection and zoom bubble overlay
- **Interactive UI**: Streamlit web interface with manual keypoint adjustment
- **Batch Processing**: Process multiple bird pairs efficiently

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for YOLOv8 inference, but not required)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd PigeonPhoto_Automator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure directory structure:
```bash
# Directories should be created automatically, but verify:
# - assets/templates/ (contains Pigeon Template.jpg)
# - assets/overlays/ (for glass_reflection.png)
# - assets/fonts/ (for TrajanPro.ttf)
# - models/ (for pigeon_pose_v1.pt)
```

## Project Structure

```
PigeonPhoto_Automator/
├── assets/
│   ├── templates/
│   │   └── Pigeon Template.jpg      # Background template (3000x2000px)
│   ├── overlays/
│   │   └── glass_reflection.png     # Eye enhancement overlay (optional)
│   └── fonts/
│       └── TrajanPro.ttf            # Text overlay font (optional)
├── models/
│   └── pigeon_pose_v1.pt            # Trained YOLOv8-Pose model
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configuration and ideal skeleton
│   ├── segmentation.py             # Background removal (rembg)
│   ├── keypoints.py                 # YOLOv8-Pose keypoint detection
│   ├── geometry.py                  # TPS warping, rotation, translation
│   ├── harmonization.py             # Poisson blending, exposure matching
│   ├── compositor.py                # Eye enhancement, final assembly
│   └── pipeline.py                  # Main orchestrator
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Usage

### Web Application (Recommended)

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload images:
   - Upload a body image (Body_Raw.jpg)
   - Upload an eye macro image (Eye_Macro.jpg)
   - Enter bird name and ring number
   - Click "Process Images"

4. Review and adjust:
   - View detected keypoints
   - Manually adjust keypoint coordinates if needed
   - Re-process with adjusted keypoints

5. Download result:
   - View final "Golden Prince" portrait
   - Download high-quality JPG

### Programmatic Usage

```python
from src.pipeline import PigeonPipeline

# Initialize pipeline
pipeline = PigeonPipeline()

# Process images
result = pipeline.process(
    body_image_path="path/to/Body_Raw.jpg",
    eye_image_path="path/to/Eye_Macro.jpg",
    name="Champion",
    ring_number="2024-001"
)

# Save result
pipeline.save_output(result, "output/golden_prince.jpg")
```

## Model Training

The YOLOv8-Pose model needs to be trained on your dataset:

1. **Data Preparation**:
   - Collect 300+ diverse pigeon images
   - Label 9 keypoints using Roboflow or CVAT:
     - Head: Beak Tip, Eye Center, Skull Back
     - Body: Neck Base, Shoulder, Wing Tip, Tail Tip
     - Legs: Left Leg Joint, Right Leg Joint

2. **Training**:
   ```python
   from ultralytics import YOLO
   
   # Load base model
   model = YOLO("yolov8n-pose.pt")
   
   # Train on your dataset
   model.train(
       data="path/to/dataset.yaml",
       epochs=100,
       imgsz=640,
       batch=16
   )
   
   # Save trained model
   model.save("models/pigeon_pose_v1.pt")
   ```

3. **Place trained model** in `models/pigeon_pose_v1.pt`

## Configuration

Edit `src/config.py` to customize:
- Ideal skeleton coordinates
- Template paths
- Processing parameters
- Output settings
- Text overlay defaults

## Technical Details

### Keypoint Detection
- Uses YOLOv8-Pose (Ultralytics)
- Detects 9 anatomical points
- Confidence threshold: 0.5 (configurable)
- Automatic interpolation for missing non-critical keypoints

### Geometric Transformations
- **Head**: Rigid rotation around eye center (26.57° target angle)
- **Body**: Thin Plate Spline (TPS) warping using scipy.interpolate.Rbf
- **Feet**: Simple translation to target position

### Harmonization
- **Poisson Blending**: cv2.seamlessClone with MIXED_CLONE flag
- **Exposure Matching**: Gamma correction based on luminance matching
- **Color Grading**: S-curve contrast adjustment

### Eye Enhancement
- Automatic pupil detection (HoughCircles or contour analysis)
- Circular crop with soft-edged mask
- Glass reflection overlay (optional)
- Zoom bubble compositing

## Troubleshooting

### Model Not Found
If `pigeon_pose_v1.pt` is not available, the system will use the default YOLOv8n-pose model. This will need fine-tuning for best results.

### Missing Keypoints
- Check keypoint confidence scores
- Use manual adjustment in the Review tab
- Ensure images have good lighting and bird is clearly visible

### Segmentation Issues
- If rembg fails, the system automatically falls back to threshold-based segmentation
- Ensure images have sufficient contrast between bird and background

### Memory Issues
- Reduce image sizes if processing large batches
- Use GPU acceleration for YOLOv8 inference
- Process images sequentially rather than in parallel

## Performance

- **Target Processing Time**: < 30 seconds per bird
- **Bottlenecks**: YOLO inference (~2-5s), TPS warping (~5-10s), Poisson blending (~3-5s)
- **Optimization**: GPU acceleration recommended for production use

## Output Specifications

- **Resolution**: 3000x2000px (matches template)
- **Format**: JPG (quality=95) or PNG
- **Color Space**: sRGB
- **DPI**: 300 (print quality)

## Known Limitations

1. **Model Dependency**: Requires trained YOLOv8-Pose model for accurate keypoint detection
2. **Template Requirement**: Background template must be provided
3. **Eye Matching**: Eye macro should ideally be from the same bird as body image
4. **Manual Adjustment**: Full interactive keypoint dragging requires additional UI development

## Future Enhancements

- [ ] Interactive keypoint dragging on canvas
- [ ] Batch processing with progress bar
- [ ] Multiple template options
- [ ] Export to different formats (PNG with alpha, TIFF)
- [ ] API endpoint for programmatic access
- [ ] GPU acceleration optimization
- [ ] Real-time preview during processing

## License

[Specify your license here]

## Credits

- **Computer Vision**: OpenCV, scikit-image
- **Segmentation**: rembg (U2-Net)
- **Pose Estimation**: Ultralytics YOLOv8
- **Web Framework**: Streamlit
- **Scientific Computing**: NumPy, SciPy

## Support

For issues, questions, or contributions, please [create an issue](link-to-issues) or contact the development team.

---

**Version**: 2.0  
**Last Updated**: December 2024

