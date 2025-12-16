"""
Keypoint detection module using YOLOv8-Pose.
Detects 9 anatomical keypoints on the pigeon.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from pathlib import Path
import src.config as config


class KeypointDetector:
    """Detector for pigeon anatomical keypoints using YOLOv8-Pose."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the keypoint detector.
        
        Args:
            model_path: Path to trained YOLOv8-Pose model. If None, uses default from config.
        """
        if model_path is None:
            model_path = config.YOLO_MODEL_FILE
        
        self.model_path = model_path
        self.model = None
        self.confidence_threshold = config.KEYPOINT_CONFIDENCE_THRESHOLD
        
        # Keypoint mapping: YOLO output indices to our keypoint names
        # This mapping should match the training dataset
        self.keypoint_mapping = {
            0: "beak_tip",
            1: "eye_center",
            2: "skull_back",
            3: "neck_base",
            4: "shoulder",
            5: "wing_tip",
            6: "tail_tip",
            7: "left_leg_joint",
            8: "right_leg_joint"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8-Pose model."""
        try:
            if self.model_path.exists():
                self.model = YOLO(str(self.model_path))
                print(f"Loaded model from {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}")
                print("Using default YOLOv8-Pose model (will need fine-tuning)")
                # Load default pose model as fallback
                self.model = YOLO("yolov8n-pose.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default YOLOv8-Pose model")
            self.model = YOLO("yolov8n-pose.pt")
    
    def detect(self, image: np.ndarray) -> Dict[str, Tuple[int, int, float]]:
        """
        Detect keypoints in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
        
        Returns:
            Dictionary mapping keypoint names to (x, y, confidence) tuples.
            Missing keypoints will have confidence 0.0.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(image, verbose=False)
        
        # Initialize result dictionary
        keypoints = {name: (0, 0, 0.0) for name in config.IDEAL_SKELETON.keys() 
                    if name != "feet_center"}  # feet_center is computed
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints
            
            if keypoints_data.xy.numel() > 0:
                # Extract keypoints (shape: [num_detections, num_keypoints, 3])
                # Last dimension: [x, y, confidence]
                kpts = keypoints_data.xy.cpu().numpy()
                confs = keypoints_data.conf.cpu().numpy() if hasattr(keypoints_data, 'conf') else None
                
                # Use first detection (assuming single bird per image)
                if len(kpts) > 0:
                    detection_kpts = kpts[0]  # Shape: [num_keypoints, 3]
                    
                    # Map YOLO keypoints to our keypoint names
                    for idx, (x, y, conf) in enumerate(detection_kpts):
                        if idx in self.keypoint_mapping:
                            keypoint_name = self.keypoint_mapping[idx]
                            confidence = float(conf) if confs is None else float(confs[0][idx])
                            
                            if confidence >= self.confidence_threshold:
                                keypoints[keypoint_name] = (int(x), int(y), confidence)
        
        # Compute feet_center from leg joints
        left_leg = keypoints.get("left_leg_joint", (0, 0, 0.0))
        right_leg = keypoints.get("right_leg_joint", (0, 0, 0.0))
        
        if left_leg[2] > 0 and right_leg[2] > 0:
            feet_x = (left_leg[0] + right_leg[0]) // 2
            feet_y = (left_leg[1] + right_leg[1]) // 2
            feet_conf = (left_leg[2] + right_leg[2]) / 2
            keypoints["feet_center"] = (feet_x, feet_y, feet_conf)
        elif left_leg[2] > 0:
            keypoints["feet_center"] = left_leg
        elif right_leg[2] > 0:
            keypoints["feet_center"] = right_leg
        else:
            keypoints["feet_center"] = (0, 0, 0.0)
        
        return keypoints
    
    def validate_keypoints(self, keypoints: Dict[str, Tuple[int, int, float]]) -> Tuple[bool, List[str]]:
        """
        Validate that critical keypoints are present.
        
        Args:
            keypoints: Dictionary of detected keypoints
        
        Returns:
            Tuple of (is_valid, missing_keypoints)
        """
        missing = []
        
        for kp_name in config.CRITICAL_KEYPOINTS:
            if kp_name not in keypoints or keypoints[kp_name][2] < self.confidence_threshold:
                missing.append(kp_name)
        
        return len(missing) == 0, missing
    
    def interpolate_missing_keypoints(self, keypoints: Dict[str, Tuple[int, int, float]]) -> Dict[str, Tuple[int, int, float]]:
        """
        Interpolate missing non-critical keypoints from available ones.
        
        Args:
            keypoints: Dictionary of detected keypoints
        
        Returns:
            Dictionary with interpolated keypoints added
        """
        if not config.KEYPOINT_INTERPOLATION_ENABLED:
            return keypoints
        
        # Interpolate skull_back from beak_tip and eye_center
        if keypoints.get("skull_back", (0, 0, 0.0))[2] == 0.0:
            beak = keypoints.get("beak_tip")
            eye = keypoints.get("eye_center")
            if beak and eye and beak[2] > 0 and eye[2] > 0:
                # Skull back is behind the eye, opposite direction from beak
                dx = eye[0] - beak[0]
                dy = eye[1] - beak[1]
                skull_x = int(eye[0] - dx * 0.5)
                skull_y = int(eye[1] - dy * 0.5)
                keypoints["skull_back"] = (skull_x, skull_y, 0.5)  # Lower confidence for interpolated
        
        # Interpolate wing_tip from shoulder and tail_tip
        if keypoints.get("wing_tip", (0, 0, 0.0))[2] == 0.0:
            shoulder = keypoints.get("shoulder")
            tail = keypoints.get("tail_tip")
            if shoulder and tail and shoulder[2] > 0 and tail[2] > 0:
                # Wing tip is between shoulder and tail, slightly outward
                mid_x = (shoulder[0] + tail[0]) // 2
                mid_y = (shoulder[1] + tail[1]) // 2
                # Offset outward (to the right typically)
                wing_x = int(mid_x + 100)
                wing_y = int(mid_y - 50)
                keypoints["wing_tip"] = (wing_x, wing_y, 0.5)
        
        return keypoints


def visualize_keypoints(image: np.ndarray, keypoints: Dict[str, Tuple[int, int, float]], 
                        show_confidence: bool = True) -> np.ndarray:
    """
    Visualize detected keypoints on the image.
    
    Args:
        image: Input image
        keypoints: Dictionary of keypoints
        show_confidence: Whether to show confidence scores
    
    Returns:
        Image with keypoints drawn
    """
    vis_image = image.copy()
    
    # Color mapping for different keypoint groups
    colors = {
        "head": (0, 255, 255),  # Cyan
        "body": (255, 0, 0),    # Blue
        "legs": (0, 255, 0)     # Green
    }
    
    for group_name, kp_names in config.KEYPOINT_GROUPS.items():
        color = colors.get(group_name, (255, 255, 255))
        
        for kp_name in kp_names:
            if kp_name in keypoints:
                x, y, conf = keypoints[kp_name]
                if conf > 0:
                    # Draw circle
                    cv2.circle(vis_image, (x, y), 5, color, -1)
                    cv2.circle(vis_image, (x, y), 8, color, 2)
                    
                    # Draw label
                    label = kp_name.replace("_", " ").title()
                    if show_confidence:
                        label += f" ({conf:.2f})"
                    cv2.putText(vis_image, label, (x + 10, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw feet_center if available
    if "feet_center" in keypoints:
        x, y, conf = keypoints["feet_center"]
        if conf > 0:
            cv2.circle(vis_image, (x, y), 5, (255, 255, 0), -1)  # Yellow
            cv2.circle(vis_image, (x, y), 8, (255, 255, 0), 2)
    
    return vis_image

