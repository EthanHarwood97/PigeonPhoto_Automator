"""
Geometry transformation module.
Implements TPS warping, rigid rotation, and translation for the surgical component assembly.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.interpolate import Rbf
import src.config as config


def calculate_head_rotation_angle(keypoints: Dict[str, Tuple[int, int, float]]) -> float:
    """
    Calculate the rotation angle needed for the head.
    
    Args:
        keypoints: Dictionary of detected keypoints
    
    Returns:
        Rotation angle in degrees
    """
    beak = keypoints.get("beak_tip", (0, 0, 0.0))
    eye = keypoints.get("eye_center", (0, 0, 0.0))
    
    if beak[2] == 0.0 or eye[2] == 0.0:
        return 0.0
    
    # Calculate angle from eye to beak in source image
    dx = beak[0] - eye[0]
    dy = beak[1] - eye[1]
    source_angle = np.degrees(np.arctan2(dy, dx))
    
    # Target angle from ideal skeleton
    ideal_beak = config.IDEAL_SKELETON["beak_tip"]
    ideal_eye = config.IDEAL_SKELETON["eye_center"]
    target_dx = ideal_beak[0] - ideal_eye[0]
    target_dy = ideal_beak[1] - ideal_eye[1]
    target_angle = np.degrees(np.arctan2(target_dy, target_dx))
    
    # Calculate rotation needed
    rotation_angle = target_angle - source_angle
    
    return rotation_angle


def rotate_head_rigid(image: np.ndarray, mask: np.ndarray, 
                      keypoints: Dict[str, Tuple[int, int, float]],
                      rotation_angle: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Apply rigid rotation to the head layer.
    
    Args:
        image: Head region image (RGBA)
        mask: Head region mask
        keypoints: Dictionary of keypoints
        rotation_angle: Rotation angle in degrees
    
    Returns:
        Tuple of (rotated_image, rotated_mask, new_eye_center)
    """
    eye = keypoints.get("eye_center", (0, 0, 0.0))
    if eye[2] == 0.0:
        return image, mask, (0, 0)
    
    eye_center = (eye[0], eye[1])
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, rotation_angle, 1.0)
    
    # Calculate new image size to accommodate rotation
    h, w = image.shape[:2]
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_w / 2) - eye_center[0]
    rotation_matrix[1, 2] += (new_h / 2) - eye_center[1]
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
    
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (new_w, new_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    
    # Calculate new eye center position
    eye_point = np.array([eye_center[0], eye_center[1], 1])
    new_eye_point = rotation_matrix @ eye_point
    new_eye_center = (int(new_eye_point[0]), int(new_eye_point[1]))
    
    return rotated_image, rotated_mask, new_eye_center


def extract_head_region(image: np.ndarray, mask: np.ndarray,
                       keypoints: Dict[str, Tuple[int, int, float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Extract the head region using a circular mask.
    
    Args:
        image: Full image (RGBA)
        mask: Full mask
        keypoints: Dictionary of keypoints
    
    Returns:
        Tuple of (head_image, head_mask, offset) where offset is (x, y) of top-left corner
    """
    eye = keypoints.get("eye_center", (0, 0, 0.0))
    neck = keypoints.get("neck_base", (0, 0, 0.0))
    
    if eye[2] == 0.0 or neck[2] == 0.0:
        # Fallback: use fixed radius
        radius = 200
        center = (image.shape[1] // 2, image.shape[0] // 2)
    else:
        # Calculate radius as distance from eye to neck
        dx = eye[0] - neck[0]
        dy = eye[1] - neck[1]
        radius = int(np.sqrt(dx*dx + dy*dy) * 1.2)  # Add 20% padding
        center = (eye[0], eye[1])
    
    # Create circular mask
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Extract region with padding
    x_min = max(0, center[0] - radius)
    x_max = min(w, center[0] + radius)
    y_min = max(0, center[1] - radius)
    y_max = min(h, center[1] + radius)
    
    head_image = image[y_min:y_max, x_min:x_max].copy()
    head_mask = mask[y_min:y_max, x_min:x_max].copy()
    
    # Apply circular mask
    circle_mask_region = mask_circle[y_min:y_max, x_min:x_max]
    head_mask = cv2.bitwise_and(head_mask, (circle_mask_region * 255).astype(np.uint8))
    
    # Update alpha channel
    if head_image.shape[2] == 4:
        head_image[:, :, 3] = head_mask
    
    offset = (x_min, y_min)
    
    return head_image, head_mask, offset


def translate_feet(image: np.ndarray, mask: np.ndarray,
                  keypoints: Dict[str, Tuple[int, int, float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Translate the feet to the target position.
    
    Args:
        image: Feet region image (RGBA)
        mask: Feet region mask
        keypoints: Dictionary of keypoints
    
    Returns:
        Tuple of (translated_image, translated_mask, new_feet_center)
    """
    feet_source = keypoints.get("feet_center", (0, 0, 0.0))
    feet_target = config.IDEAL_SKELETON["feet_center"]
    
    if feet_source[2] == 0.0:
        return image, mask, (0, 0)
    
    # Calculate translation vector
    tx = feet_target[0] - feet_source[0]
    ty = feet_target[1] - feet_source[1]
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Calculate new image size
    h, w = image.shape[:2]
    new_w = w + abs(tx)
    new_h = h + abs(ty)
    
    # Adjust translation matrix
    if tx < 0:
        translation_matrix[0, 2] -= tx
    if ty < 0:
        translation_matrix[1, 2] -= ty
    
    # Apply translation
    translated_image = cv2.warpAffine(image, translation_matrix, (new_w, new_h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0, 0))
    
    translated_mask = cv2.warpAffine(mask, translation_matrix, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    
    new_feet_center = (feet_target[0], feet_target[1])
    
    return translated_image, translated_mask, new_feet_center


def extract_feet_region(image: np.ndarray, mask: np.ndarray,
                       keypoints: Dict[str, Tuple[int, int, float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Extract the feet/legs region based on leg joint keypoints.
    
    Args:
        image: Full image (RGBA)
        mask: Full mask
        keypoints: Dictionary of keypoints
    
    Returns:
        Tuple of (feet_image, feet_mask, offset)
    """
    left_leg = keypoints.get("left_leg_joint", (0, 0, 0.0))
    right_leg = keypoints.get("right_leg_joint", (0, 0, 0.0))
    
    if left_leg[2] == 0.0 and right_leg[2] == 0.0:
        # Fallback: use feet_center
        feet_center = keypoints.get("feet_center", (0, 0, 0.0))
        if feet_center[2] == 0.0:
            return image, mask, (0, 0)
        center_x, center_y = feet_center[0], feet_center[1]
        width, height = 300, 400
    else:
        # Calculate bounding box
        if left_leg[2] > 0 and right_leg[2] > 0:
            x_coords = [left_leg[0], right_leg[0]]
            y_coords = [left_leg[1], right_leg[1]]
        elif left_leg[2] > 0:
            x_coords = [left_leg[0]]
            y_coords = [left_leg[1]]
        else:
            x_coords = [right_leg[0]]
            y_coords = [right_leg[1]]
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        width = int((max(x_coords) - min(x_coords)) * 2.5) if len(x_coords) > 1 else 300
        height = int((max(y_coords) - min(y_coords)) * 2.5) if len(y_coords) > 1 else 400
    
    # Extract region with padding
    h, w = image.shape[:2]
    x_min = max(0, center_x - width // 2)
    x_max = min(w, center_x + width // 2)
    y_min = max(0, center_y - height // 2)
    y_max = min(h, center_y + height // 2)
    
    feet_image = image[y_min:y_max, x_min:x_max].copy()
    feet_mask = mask[y_min:y_max, x_min:x_max].copy()
    
    # Update alpha channel
    if feet_image.shape[2] == 4:
        feet_image[:, :, 3] = feet_mask
    
    offset = (x_min, y_min)
    
    return feet_image, feet_mask, offset


def tps_warp_body(image: np.ndarray, mask: np.ndarray,
                 source_keypoints: Dict[str, Tuple[int, int, float]],
                 target_keypoints: Optional[Dict[str, Tuple[int, int, float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Thin Plate Spline (TPS) warping to the body region.
    
    Args:
        image: Body region image (RGBA)
        mask: Body region mask
        source_keypoints: Detected keypoints in source image
        target_keypoints: Target keypoints (defaults to ideal skeleton)
    
    Returns:
        Tuple of (warped_image, warped_mask)
    """
    if target_keypoints is None:
        target_keypoints = {k: (v[0], v[1]) for k, v in config.IDEAL_SKELETON.items()}
    
    # Prepare control points for TPS
    # Use body keypoints: neck_base, shoulder, wing_tip, tail_tip
    body_keypoints = ["neck_base", "shoulder", "wing_tip", "tail_tip"]
    
    source_points = []
    target_points = []
    
    for kp_name in body_keypoints:
        if kp_name in source_keypoints and source_keypoints[kp_name][2] > 0:
            src_kp = source_keypoints[kp_name]
            tgt_kp = target_keypoints.get(kp_name, (0, 0))
            source_points.append([src_kp[0], src_kp[1]])
            target_points.append([tgt_kp[0], tgt_kp[1]])
    
    if len(source_points) < 3:
        # Not enough points for TPS, return original
        return image, mask
    
    source_points = np.array(source_points, dtype=np.float32)
    target_points = np.array(target_points, dtype=np.float32)
    
    # Create coordinate grids
    h, w = image.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Create RBF interpolators for x and y displacements
    # Using thin-plate spline function
    rbf_x = Rbf(source_points[:, 0], source_points[:, 1], 
                target_points[:, 0] - source_points[:, 0],
                function='thin-plate', smooth=0)
    rbf_y = Rbf(source_points[:, 0], source_points[:, 1],
                target_points[:, 1] - source_points[:, 1],
                function='thin-plate', smooth=0)
    
    # Calculate displacement fields
    dx = rbf_x(x_coords, y_coords)
    dy = rbf_y(x_coords, y_coords)
    
    # Create mapping
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)
    
    # Apply warping
    warped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    
    warped_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    
    return warped_image, warped_mask


def generate_feet_shadow(feet_center: Tuple[int, int], feet_width: int = 200) -> np.ndarray:
    """
    Generate a contact shadow for the feet.
    
    Args:
        feet_center: Center position of feet (x, y)
        feet_width: Width of feet region
    
    Returns:
        Shadow image (RGBA) with alpha channel
    """
    # Calculate shadow dimensions
    shadow_w = int(feet_width * config.SHADOW_WIDTH_MULTIPLIER)
    shadow_h = int(feet_width * config.SHADOW_HEIGHT_MULTIPLIER)
    
    # Create ellipse mask
    shadow = np.zeros((shadow_h, shadow_w, 4), dtype=np.uint8)
    center = (shadow_w // 2, shadow_h // 2)
    axes = (shadow_w // 2, shadow_h // 2)
    
    cv2.ellipse(shadow, center, axes, 0, 0, 360, config.SHADOW_COLOR + (255,), -1)
    
    # Apply Gaussian blur
    shadow[:, :, 3] = cv2.GaussianBlur(shadow[:, :, 3], 
                                       config.SHADOW_BLUR_KERNEL,
                                       config.SHADOW_BLUR_SIGMA)
    
    # Apply opacity
    shadow[:, :, 3] = (shadow[:, :, 3] * config.SHADOW_OPACITY).astype(np.uint8)
    
    return shadow

