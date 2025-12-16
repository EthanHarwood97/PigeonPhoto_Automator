"""
Segmentation module for isolating the bird from background.
Uses rembg (U2-Net) for automatic background removal.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from rembg import remove
from PIL import Image
import io


def segment_bird(image: np.ndarray, use_rembg: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment the bird from the background.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        use_rembg: Whether to use rembg (True) or fallback method (False)
    
    Returns:
        Tuple of (segmented_image, mask) where:
        - segmented_image: Image with background removed (RGBA format)
        - mask: Binary mask (255 for bird, 0 for background)
    """
    if use_rembg:
        try:
            # Convert BGR to RGB for rembg
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Remove background using rembg
            output = remove(pil_image)
            
            # Convert back to numpy array
            output_array = np.array(output)
            
            # Extract alpha channel as mask
            if output_array.shape[2] == 4:
                mask = output_array[:, :, 3]
                segmented = output_array
            else:
                # If no alpha channel, create one
                mask = np.ones((output_array.shape[0], output_array.shape[1]), dtype=np.uint8) * 255
                segmented = np.dstack([output_array, mask])
            
            return segmented, mask
            
        except Exception as e:
            print(f"Warning: rembg failed ({e}), using fallback method")
            return _segment_fallback(image)
    else:
        return _segment_fallback(image)


def _segment_fallback(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback segmentation method using threshold-based approach.
    Less accurate than rembg but more robust.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (assumed to be the bird)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [largest_contour], 255)
    
    # Apply mask to original image
    segmented = image.copy()
    alpha_channel = mask
    segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2BGRA)
    segmented[:, :, 3] = alpha_channel
    
    return segmented, mask


def refine_mask(mask: np.ndarray, erode_iterations: int = 2, dilate_iterations: int = 2) -> np.ndarray:
    """
    Refine the segmentation mask by removing noise and smoothing edges.
    
    Args:
        mask: Binary mask
        erode_iterations: Number of erosion iterations
        dilate_iterations: Number of dilation iterations
    
    Returns:
        Refined binary mask
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # Erode to remove small noise
    refined = cv2.erode(mask, kernel, iterations=erode_iterations)
    
    # Dilate to restore size
    refined = cv2.dilate(refined, kernel, iterations=dilate_iterations)
    
    # Smooth edges with Gaussian blur
    refined = cv2.GaussianBlur(refined, (5, 5), 0)
    _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)
    
    return refined


def extract_region(image: np.ndarray, mask: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Extract a specific region from the image using the mask.
    
    Args:
        image: Input image (BGR or RGBA)
        mask: Binary mask
        bbox: Optional bounding box (x, y, width, height) to crop region
    
    Returns:
        Extracted region with alpha channel
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Add alpha channel if not present
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Apply mask to alpha channel
    result = image.copy()
    result[:, :, 3] = cv2.bitwise_and(result[:, :, 3], mask)
    
    if bbox:
        x, y, w, h = bbox
        result = result[y:y+h, x:x+w]
    
    return result

