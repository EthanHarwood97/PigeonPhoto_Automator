"""
Image harmonization module.
Handles Poisson blending, exposure matching, and color grading.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from skimage import exposure
import src.config as config


def poisson_blend(source: np.ndarray, destination: np.ndarray, mask: np.ndarray,
                 center: Tuple[int, int], blend_mode: str = "MIXED_CLONE") -> np.ndarray:
    """
    Blend source image into destination using Poisson blending.
    
    Args:
        source: Source image (BGR or BGRA)
        destination: Destination image (BGR or BGRA)
        mask: Binary mask indicating region to blend (255 for blend area, 0 for background)
        center: Center point of the blend region (x, y)
        blend_mode: Blending mode ("NORMAL_CLONE", "MIXED_CLONE", or "MONOCHROME_TRANSFER")
    
    Returns:
        Blended image (BGR)
    """
    # Convert to BGR if needed
    if len(source.shape) == 4 or source.shape[2] == 4:
        source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
    if len(destination.shape) == 4 or destination.shape[2] == 4:
        destination = cv2.cvtColor(destination, cv2.COLOR_BGRA2BGR)
    
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Normalize mask to 0-255
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    
    # Select blend mode
    if blend_mode == "MIXED_CLONE":
        flags = cv2.MIXED_CLONE
    elif blend_mode == "MONOCHROME_TRANSFER":
        flags = cv2.MONOCHROME_TRANSFER
    else:
        flags = cv2.NORMAL_CLONE
    
    # Apply Poisson blending
    result = cv2.seamlessClone(source, destination, mask, center, flags)
    
    return result


def match_exposure_gamma(source: np.ndarray, target: np.ndarray,
                        region_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Match exposure of source image to target using gamma correction.
    
    Args:
        source: Source image to adjust (BGR)
        target: Target image for reference (BGR)
        region_mask: Optional mask indicating region to use for target luminance calculation
    
    Returns:
        Exposure-matched source image (BGR)
    """
    # Convert to LAB color space for luminance matching
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    # Extract L channel (luminance)
    source_l = source_lab[:, :, 0].astype(np.float32)
    target_l = target_lab[:, :, 0].astype(np.float32)
    
    # Calculate mean luminance
    if region_mask is not None:
        # Use masked region for target
        if len(region_mask.shape) == 3:
            region_mask = cv2.cvtColor(region_mask, cv2.COLOR_BGR2GRAY)
        if region_mask.max() <= 1.0:
            region_mask = (region_mask * 255).astype(np.uint8)
        
        mask_bool = region_mask > 127
        target_mean = np.mean(target_l[mask_bool]) if np.any(mask_bool) else np.mean(target_l)
    else:
        target_mean = np.mean(target_l)
    
    source_mean = np.mean(source_l)
    
    # Avoid division by zero
    if source_mean == 0:
        return source
    
    # Calculate gamma
    # gamma = log(L_target / 255) / log(L_source / 255)
    gamma = np.log(target_mean / 255.0) / np.log(source_mean / 255.0)
    
    # Clamp gamma to reasonable range
    gamma = np.clip(gamma, 0.1, 3.0)
    
    # Apply gamma correction using skimage
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    adjusted = exposure.adjust_gamma(source_rgb, gamma)
    adjusted_bgr = cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR)
    
    return adjusted_bgr


def match_exposure_histogram(source: np.ndarray, target: np.ndarray,
                            region_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Match exposure of source image to target using histogram matching.
    
    Args:
        source: Source image to adjust (BGR)
        target: Target image for reference (BGR)
        region_mask: Optional mask indicating region to use for target histogram
    
    Returns:
        Exposure-matched source image (BGR)
    """
    # Convert to RGB for skimage
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
    # Apply mask if provided
    if region_mask is not None:
        if len(region_mask.shape) == 3:
            region_mask = cv2.cvtColor(region_mask, cv2.COLOR_BGR2GRAY)
        if region_mask.max() <= 1.0:
            region_mask = (region_mask * 255).astype(np.uint8)
        
        mask_bool = region_mask > 127
        # Use masked region for reference
        target_masked = target_rgb.copy()
        target_masked[~mask_bool] = 0
    else:
        target_masked = target_rgb
    
    # Apply histogram matching
    matched = exposure.match_histograms(source_rgb, target_masked, multichannel=True)
    
    # Convert back to BGR
    matched_bgr = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return matched_bgr


def apply_color_grading(image: np.ndarray, method: str = "sigmoid") -> np.ndarray:
    """
    Apply global color grading to unify black levels and enhance contrast.
    
    Args:
        image: Input image (BGR)
        method: Grading method ("sigmoid" or "clahe")
    
    Returns:
        Color-graded image (BGR)
    """
    if method == "sigmoid":
        # Apply S-curve using sigmoid adjustment
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        adjusted = exposure.adjust_sigmoid(image_rgb, cutoff=0.5, gain=10, inv=False)
        adjusted_bgr = cv2.cvtColor(adjusted.astype(np.uint8), cv2.COLOR_RGB2BGR)
        return adjusted_bgr
    
    elif method == "clahe":
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        adjusted_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return adjusted_bgr
    
    else:
        return image


def create_head_neck_blend_mask(head_center: Tuple[int, int], neck_base: Tuple[int, int],
                               radius: int) -> np.ndarray:
    """
    Create a gradient mask for blending head and neck.
    
    Args:
        head_center: Center of head region (x, y)
        neck_base: Neck base position (x, y)
        radius: Radius of the blend region
    
    Returns:
        Gradient mask (0-255) for blending
    """
    # Calculate mask size (enough to cover both points)
    x_min = min(head_center[0], neck_base[0]) - radius
    x_max = max(head_center[0], neck_base[0]) + radius
    y_min = min(head_center[1], neck_base[1]) - radius
    y_max = max(head_center[1], neck_base[1]) + radius
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Create distance-based gradient mask
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x = (head_center[0] + neck_base[0]) // 2 - x_min
    center_y = (head_center[1] + neck_base[1]) // 2 - y_min
    
    y_coords, x_coords = np.ogrid[:height, :width]
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create smooth gradient (1.0 at center, 0.0 at edges)
    mask = np.clip(255 * (1.0 - distances / radius), 0, 255).astype(np.uint8)
    
    # Apply Gaussian blur for smoother transition
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    
    return mask


def blend_head_to_body(head_image: np.ndarray, body_image: np.ndarray,
                      head_mask: np.ndarray, head_position: Tuple[int, int],
                      neck_base: Tuple[int, int]) -> np.ndarray:
    """
    Blend the rotated head onto the warped body using Poisson blending.
    
    Args:
        head_image: Rotated head image (RGBA)
        body_image: Warped body image (BGR or BGRA)
        head_mask: Head mask
        head_position: Position where head should be placed (x, y)
        neck_base: Neck base position for blending center
    
    Returns:
        Blended image (BGR)
    """
    # Convert head to BGR
    if head_image.shape[2] == 4:
        head_bgr = cv2.cvtColor(head_image, cv2.COLOR_BGRA2BGR)
    else:
        head_bgr = head_image
    
    # Convert body to BGR
    if len(body_image.shape) == 4 or body_image.shape[2] == 4:
        body_bgr = cv2.cvtColor(body_image, cv2.COLOR_BGRA2BGR)
    else:
        body_bgr = body_image
    
    # Create composite canvas
    canvas = body_bgr.copy()
    
    # Calculate head region bounds
    h_h, w_h = head_image.shape[:2]
    h_b, w_b = canvas.shape[:2]
    
    # Calculate paste region
    x_start = max(0, head_position[0] - w_h // 2)
    y_start = max(0, head_position[1] - h_h // 2)
    x_end = min(w_b, x_start + w_h)
    y_end = min(h_b, y_start + h_h)
    
    # Adjust head image if needed
    head_x_start = max(0, w_h // 2 - head_position[0])
    head_y_start = max(0, h_h // 2 - head_position[1])
    head_x_end = head_x_start + (x_end - x_start)
    head_y_end = head_y_start + (y_end - y_start)
    
    head_region = head_bgr[head_y_start:head_y_end, head_x_start:head_x_end]
    mask_region = head_mask[head_y_start:head_y_end, head_x_start:head_x_end]
    
    # Create blend center (at neck base)
    blend_center = (neck_base[0], neck_base[1])
    
    # Adjust blend center relative to canvas
    blend_center = (blend_center[0] - x_start, blend_center[1] - y_start)
    
    # Extract destination region
    dest_region = canvas[y_start:y_end, x_start:x_end]
    
    # Apply Poisson blending
    if dest_region.shape == head_region.shape and mask_region.shape[:2] == head_region.shape[:2]:
        blended_region = poisson_blend(head_region, dest_region, mask_region,
                                      blend_center, config.POISSON_BLEND_MODE)
        canvas[y_start:y_end, x_start:x_end] = blended_region
    
    return canvas

