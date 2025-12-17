"""
Main pipeline orchestrator.
Ties together all modules to process a bird from input to output.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

import src.config as config
from src.segmentation import segment_bird
from src.keypoints import KeypointDetector
from src.geometry import (
    calculate_head_rotation_angle, rotate_head_rigid, translate_feet,
    tps_warp_body, generate_feet_shadow, extract_head_region, extract_feet_region
)
from src.harmonization import (
    match_exposure_gamma, blend_head_to_body, apply_color_grading
)
from src.compositor import (
    enhance_eye, composite_final_image, add_text_overlay
)


class PigeonPipeline:
    """Main pipeline for processing pigeon images."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to YOLOv8-Pose model
        """
        self.keypoint_detector = KeypointDetector(model_path)
        self.template = None
        self.glass_reflection = None
        self._load_assets()
    
    def _load_assets(self):
        """Load template and overlay assets."""
        # Load template
        if config.TEMPLATE_FILE.exists():
            self.template = cv2.imread(str(config.TEMPLATE_FILE))
            if self.template is not None:
                # Resize to target dimensions if needed
                if self.template.shape[:2] != config.OUTPUT_RESOLUTION[::-1]:
                    self.template = cv2.resize(self.template, config.OUTPUT_RESOLUTION,
                                             interpolation=cv2.INTER_LANCZOS4)
        else:
            # Create blank template if not found
            self.template = np.zeros((config.OUTPUT_RESOLUTION[1], 
                                     config.OUTPUT_RESOLUTION[0], 3), dtype=np.uint8)
            self.template.fill(255)  # White background
        
        # Load glass reflection overlay if available
        if config.GLASS_REFLECTION_FILE.exists():
            self.glass_reflection = cv2.imread(str(config.GLASS_REFLECTION_FILE), cv2.IMREAD_UNCHANGED)
        else:
            self.glass_reflection = None
    
    def process(self, body_image_path: str, eye_image_path: str,
               name: str = "Bird", ring_number: str = "0000",
               keypoints: Optional[Dict[str, Tuple[int, int, float]]] = None) -> np.ndarray:
        """
        Process a bird image pair through the full pipeline.
        
        Args:
            body_image_path: Path to body image
            eye_image_path: Path to eye macro image
            name: Bird name for text overlay
            ring_number: Ring number for text overlay
            keypoints: Optional pre-detected keypoints (if None, will detect)
        
        Returns:
            Final processed image (BGR)
        """
        # Load images
        body_image = cv2.imread(body_image_path)
        eye_image = cv2.imread(eye_image_path)
        
        if body_image is None:
            raise ValueError(f"Could not load body image: {body_image_path}")
        if eye_image is None:
            raise ValueError(f"Could not load eye image: {eye_image_path}")
        
        # Step 1: Segment bird from background
        body_segmented, body_mask = segment_bird(body_image)
        
        # Step 2: Detect keypoints
        if keypoints is None:
            keypoints = self.keypoint_detector.detect(body_image)
            keypoints = self.keypoint_detector.interpolate_missing_keypoints(keypoints)
        
        # Validate critical keypoints
        is_valid, missing = self.keypoint_detector.validate_keypoints(keypoints)
        if not is_valid:
            raise ValueError(f"Missing critical keypoints: {missing}")
        
        # Step 3: Extract components
        # Head
        head_image, head_mask, head_offset = extract_head_region(
            body_segmented, body_mask, keypoints
        )
        
        # Feet
        feet_image, feet_mask, feet_offset = extract_feet_region(
            body_segmented, body_mask, keypoints
        )
        
        # Body (everything except head and feet)
        # Create body mask by removing head and feet regions
        body_only_mask = body_mask.copy()
        # Remove head region
        eye = keypoints.get("eye_center", (0, 0, 0.0))
        neck = keypoints.get("neck_base", (0, 0, 0.0))
        if eye[2] > 0 and neck[2] > 0:
            dx = eye[0] - neck[0]
            dy = eye[1] - neck[1]
            radius = int(np.sqrt(dx*dx + dy*dy) * 1.2)
            cv2.circle(body_only_mask, (eye[0], eye[1]), radius, 0, -1)
        
        # Remove feet region (approximate)
        feet_center = keypoints.get("feet_center", (0, 0, 0.0))
        if feet_center[2] > 0:
            cv2.ellipse(body_only_mask, (feet_center[0], feet_center[1]), 
                       (150, 200), 0, 0, 360, 0, -1)
        
        body_only = body_segmented.copy()
        body_only[:, :, 3] = cv2.bitwise_and(body_only[:, :, 3], body_only_mask)
        
        # Step 4: Apply geometric transformations
        # Head rotation
        rotation_angle = calculate_head_rotation_angle(keypoints)
        head_rotated, head_mask_rotated, new_eye_center = rotate_head_rigid(
            head_image, head_mask, keypoints, rotation_angle
        )
        
        # Body warping (TPS)
        body_warped, body_mask_warped = tps_warp_body(
            body_only, body_only_mask, keypoints
        )
        
        # Feet translation
        feet_translated, feet_mask_translated, new_feet_center = translate_feet(
            feet_image, feet_mask, keypoints
        )
        
        # Generate feet shadow
        feet_width = int(np.linalg.norm(np.array(keypoints.get("left_leg_joint", (0, 0))[:2]) - 
                                       np.array(keypoints.get("right_leg_joint", (0, 0))[:2])))
        if feet_width == 0:
            feet_width = 200  # Default
        feet_shadow = generate_feet_shadow(new_feet_center, feet_width)
        
        # Step 5: Harmonization
        # Match eye exposure to body
        eye_matched = match_exposure_gamma(eye_image, body_image)
        
        # Blend head to body using Poisson blending
        neck_base = config.IDEAL_SKELETON["neck_base"]
        body_blended = blend_head_to_body(
            head_rotated, body_warped, head_mask_rotated,
            (new_eye_center[0], new_eye_center[1]), neck_base
        )
        
        # Step 6: Eye enhancement
        enhanced_eye = enhance_eye(eye_matched, self.glass_reflection)
        
        # Step 7: Final composition
        # Determine eye position (use config or default)
        eye_position = config.EYE_ZOOM_BUBBLE_POSITION
        if eye_position is None:
            # Default position: top-left area
            eye_position = (400, 300)
        
        final_image = composite_final_image(
            self.template, body_blended, head_rotated, feet_translated,
            feet_shadow, enhanced_eye, eye_position
        )
        
        # Step 8: Color grading
        final_image = apply_color_grading(final_image, config.COLOR_GRADING_METHOD)
        
        # Step 9: Text overlay
        final_image = add_text_overlay(
            final_image, name, ring_number,
            position=config.TEXT_DEFAULT_POSITION,
            offset=config.TEXT_DEFAULT_OFFSET,
            font_size_name=config.TEXT_NAME_SIZE,
            font_size_ring=config.TEXT_RING_SIZE,
            text_color=config.TEXT_DEFAULT_COLOR,
            stroke_color=config.TEXT_STROKE_COLOR,
            stroke_width=config.TEXT_STROKE_WIDTH,
            shadow=config.TEXT_SHADOW_ENABLED
        )
        
        return final_image
    
    def save_output(self, image: np.ndarray, output_path: str):
        """
        Save the final output image.
        
        Args:
            image: Final processed image (BGR)
            output_path: Path to save output
        """
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(output_path), image,
                       [cv2.IMWRITE_JPEG_QUALITY, config.OUTPUT_QUALITY])
        else:
            cv2.imwrite(str(output_path), image)
