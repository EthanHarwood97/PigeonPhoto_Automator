"""
Configuration module for PigeonPhoto_Automator
Contains ideal skeleton coordinates, template paths, and system settings.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Asset paths
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
OVERLAYS_DIR = ASSETS_DIR / "overlays"
FONTS_DIR = ASSETS_DIR / "fonts"
MODELS_DIR = PROJECT_ROOT / "models"

# Template file
TEMPLATE_FILE = TEMPLATES_DIR / "Pigeon Template.jpg"
TEMPLATE_DIMENSIONS = (3000, 2000)  # (width, height) in pixels

# Overlay files
GLASS_REFLECTION_FILE = OVERLAYS_DIR / "glass_reflection.png"
SHADOW_FILE = OVERLAYS_DIR / "shadow.png"

# Font file
FONT_FILE = FONTS_DIR / "TrajanPro.ttf"

# Model file
YOLO_MODEL_FILE = MODELS_DIR / "pigeon_pose_v1.pt"

# Ideal Skeleton (Target Coordinates)
# Coordinates are in (x, y) format with top-left origin (0, 0)
# Canvas size: 3000x2000px
IDEAL_SKELETON: Dict[str, List[int]] = {
    "beak_tip": [2200, 400],
    "eye_center": [2100, 450],
    "skull_back": [2000, 500],
    "neck_base": [1900, 800],
    "shoulder": [1600, 900],
    "wing_tip": [1400, 1000],
    "tail_tip": [300, 1400],
    "left_leg_joint": [1600, 1600],
    "right_leg_joint": [1700, 1600],
    "feet_center": [1650, 1600]  # Computed midpoint
}

# Keypoint groups for organization
KEYPOINT_GROUPS = {
    "head": ["beak_tip", "eye_center", "skull_back"],
    "body": ["neck_base", "shoulder", "wing_tip", "tail_tip"],
    "legs": ["left_leg_joint", "right_leg_joint"]
}

# Critical keypoints (must be present for processing)
CRITICAL_KEYPOINTS = ["eye_center", "neck_base", "shoulder"]

# Keypoint detection settings
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
KEYPOINT_INTERPOLATION_ENABLED = True

# Head rotation settings
HEAD_ROTATION_TARGET_ANGLE = 26.57  # degrees above horizontal (calculated from ideal skeleton)
HEAD_ROTATION_PIVOT = "eye_center"  # Pivot point for rotation

# Feet shadow settings
SHADOW_BLUR_KERNEL = (15, 15)
SHADOW_BLUR_SIGMA = 5
SHADOW_COLOR = (40, 40, 40)  # RGB dark gray
SHADOW_OPACITY = 0.6
SHADOW_OFFSET_Y = 15  # pixels below feet_center
SHADOW_WIDTH_MULTIPLIER = 1.5
SHADOW_HEIGHT_MULTIPLIER = 0.5

# Eye enhancement settings
EYE_ZOOM_BUBBLE_SIZE = (400, 400)  # (width, height) in pixels
EYE_ZOOM_BUBBLE_POSITION = None  # To be determined from template analysis
GLASS_REFLECTION_OPACITY = 0.4  # Default opacity for glass reflection overlay
PUPIL_DETECTION_METHOD = "hough"  # "hough" or "contour"

# Harmonization settings
POISSON_BLEND_MODE = "MIXED_CLONE"  # cv2.MIXED_CLONE
EXPOSURE_MATCHING_METHOD = "gamma"  # "gamma" or "histogram"
COLOR_GRADING_METHOD = "sigmoid"  # "sigmoid" or "clahe"

# Output settings
OUTPUT_RESOLUTION = TEMPLATE_DIMENSIONS  # (3000, 2000)
OUTPUT_FORMAT = "jpg"
OUTPUT_QUALITY = 95  # JPG quality (1-100)
OUTPUT_DPI = 300
OUTPUT_COLOR_SPACE = "sRGB"

# Text overlay defaults (UI configurable)
TEXT_DEFAULT_POSITION = "bottom-right"  # or "bottom-left", "top-right", "top-left"
TEXT_DEFAULT_OFFSET = (50, 50)  # (x, y) offset from corner
TEXT_NAME_SIZE = 48  # points
TEXT_RING_SIZE = 36  # points
TEXT_DEFAULT_COLOR = (255, 255, 255)  # White
TEXT_STROKE_COLOR = (0, 0, 0)  # Black outline
TEXT_STROKE_WIDTH = 2
TEXT_SHADOW_ENABLED = True
TEXT_SHADOW_OFFSET = (2, 2)
TEXT_SHADOW_BLUR = 4
TEXT_SHADOW_OPACITY = 0.7

# Input validation
MIN_BODY_SIZE = (1000, 1000)
MIN_EYE_SIZE = (500, 500)
MAX_IMAGE_SIZE = (10000, 10000)
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png"]

# Batch processing
BATCH_PROCESSING_WORKERS = 2  # For parallel processing in production
EXPECTED_PROCESSING_TIME = 30  # seconds per bird (target)

def get_ideal_skeleton() -> Dict[str, List[int]]:
    """Get the ideal skeleton coordinates."""
    return IDEAL_SKELETON.copy()

def get_keypoint_coordinate(keypoint_name: str) -> Tuple[int, int]:
    """Get a specific keypoint coordinate from ideal skeleton."""
    if keypoint_name not in IDEAL_SKELETON:
        raise ValueError(f"Unknown keypoint: {keypoint_name}")
    coords = IDEAL_SKELETON[keypoint_name]
    return (coords[0], coords[1])

def ensure_directories():
    """Ensure all required directories exist."""
    for directory in [TEMPLATES_DIR, OVERLAYS_DIR, FONTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Ensure directories exist on import
ensure_directories()

