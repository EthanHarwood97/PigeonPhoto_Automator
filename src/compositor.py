"""
Compositor module for final assembly.
Handles eye enhancement, component assembly, and text overlay.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import src.config as config
from pathlib import Path


def detect_pupil_center(eye_image: np.ndarray, method: str = "hough") -> Tuple[int, int]:
    """
    Detect the center of the pupil in the eye macro image.
    
    Args:
        eye_image: Eye macro image (BGR or grayscale)
        method: Detection method ("hough" or "contour")
    
    Returns:
        Pupil center coordinates (x, y)
    """
    # Convert to grayscale if needed
    if len(eye_image.shape) == 3:
        gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_image
    
    h, w = gray.shape
    center = (w // 2, h // 2)  # Default to image center
    
    if method == "hough":
        # Use HoughCircles to detect pupil
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(h, w) // 4,
            param1=50,
            param2=30,
            minRadius=min(h, w) // 20,
            maxRadius=min(h, w) // 4
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Use the first (largest) circle
            center = (int(circles[0][0][0]), int(circles[0][0][1]))
    
    elif method == "contour":
        # Use contour analysis to find darkest circular region
        # Apply threshold to find dark regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the most circular contour
            best_contour = None
            best_circularity = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Too small
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > best_circularity:
                    best_circularity = circularity
                    best_contour = contour
            
            if best_contour is not None:
                # Get center of contour
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    return center


def enhance_eye(eye_image: np.ndarray, glass_reflection: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Enhance the eye macro image with cropping, masking, and glass reflection overlay.
    
    Args:
        eye_image: Eye macro image (BGR)
        glass_reflection: Optional glass reflection overlay (RGBA)
    
    Returns:
        Enhanced eye image (RGBA)
    """
    # Detect pupil center
    pupil_center = detect_pupil_center(eye_image, config.PUPIL_DETECTION_METHOD)
    
    h, w = eye_image.shape[:2]
    
    # Calculate crop size (square around pupil)
    # Use 2x the maximum of pupil radius estimate or eye width/2
    pupil_radius_estimate = min(h, w) // 4
    eye_width_estimate = w // 2
    crop_size = int(2 * max(pupil_radius_estimate, eye_width_estimate))
    
    # Ensure crop doesn't exceed image bounds
    crop_size = min(crop_size, min(h, w))
    
    # Calculate crop bounds
    x_start = max(0, pupil_center[0] - crop_size // 2)
    y_start = max(0, pupil_center[1] - crop_size // 2)
    x_end = min(w, x_start + crop_size)
    y_end = min(h, y_start + crop_size)
    
    # Adjust if needed
    if x_end - x_start < crop_size:
        x_start = max(0, x_end - crop_size)
    if y_end - y_start < crop_size:
        y_start = max(0, y_end - crop_size)
    
    # Crop image
    cropped = eye_image[y_start:y_end, x_start:x_end].copy()
    
    # Resize to target zoom bubble size if needed
    target_size = config.EYE_ZOOM_BUBBLE_SIZE
    if cropped.shape[0] != target_size[1] or cropped.shape[1] != target_size[0]:
        cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Create circular mask with soft edges
    h_crop, w_crop = cropped.shape[:2]
    center_crop = (w_crop // 2, h_crop // 2)
    radius = min(w_crop, h_crop) // 2 - 10  # Leave some padding
    
    y_coords, x_coords = np.ogrid[:h_crop, :w_crop]
    mask = (x_coords - center_crop[0])**2 + (y_coords - center_crop[1])**2 <= radius**2
    
    # Create soft-edged mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_uint8 = cv2.GaussianBlur(mask_uint8, (15, 15), 5)
    
    # Convert to RGBA
    if len(cropped.shape) == 2:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    cropped_rgba[:, :, 3] = mask_uint8
    
    # Apply glass reflection overlay if provided
    if glass_reflection is not None:
        if glass_reflection.shape[:2] != cropped_rgba.shape[:2]:
            glass_reflection = cv2.resize(glass_reflection, 
                                         (cropped_rgba.shape[1], cropped_rgba.shape[0]),
                                         interpolation=cv2.INTER_LANCZOS4)
        
        # Blend with opacity
        alpha = config.GLASS_REFLECTION_OPACITY
        cropped_rgba = cv2.addWeighted(cropped_rgba, 1.0 - alpha, 
                                       glass_reflection, alpha, 0)
    
    return cropped_rgba


def composite_eye_zoom_bubble(canvas: np.ndarray, enhanced_eye: np.ndarray,
                              position: Tuple[int, int]) -> np.ndarray:
    """
    Composite the enhanced eye into the zoom bubble area of the canvas.
    
    Args:
        canvas: Main canvas image (BGR or BGRA)
        enhanced_eye: Enhanced eye image (RGBA)
        position: Position to place eye (x, y) - center of zoom bubble
    
    Returns:
        Canvas with eye composited (BGR)
    """
    # Convert canvas to BGRA if needed
    if len(canvas.shape) == 3 and canvas.shape[2] == 3:
        canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    else:
        canvas_rgba = canvas.copy()
    
    h_eye, w_eye = enhanced_eye.shape[:2]
    h_canvas, w_canvas = canvas_rgba.shape[:2]
    
    # Calculate paste region
    x_start = max(0, position[0] - w_eye // 2)
    y_start = max(0, position[1] - h_eye // 2)
    x_end = min(w_canvas, x_start + w_eye)
    y_end = min(h_canvas, y_start + h_eye)
    
    # Adjust eye image if needed
    eye_x_start = max(0, w_eye // 2 - position[0])
    eye_y_start = max(0, h_eye // 2 - position[1])
    eye_x_end = eye_x_start + (x_end - x_start)
    eye_y_end = eye_y_start + (y_end - y_start)
    
    eye_region = enhanced_eye[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
    canvas_region = canvas_rgba[y_start:y_end, x_start:x_end]
    
    # Alpha blend
    alpha = eye_region[:, :, 3:4] / 255.0
    canvas_region = (canvas_region * (1 - alpha) + eye_region[:, :, :3] * alpha).astype(np.uint8)
    
    canvas_rgba[y_start:y_end, x_start:x_end] = canvas_region
    
    # Convert back to BGR
    result = cv2.cvtColor(canvas_rgba, cv2.COLOR_BGRA2BGR)
    
    return result


def add_text_overlay(image: np.ndarray, name: str, ring_number: str,
                    position: str = "bottom-right",
                    offset: Tuple[int, int] = (50, 50),
                    font_size_name: int = 48,
                    font_size_ring: int = 36,
                    text_color: Tuple[int, int, int] = (255, 255, 255),
                    stroke_color: Tuple[int, int, int] = (0, 0, 0),
                    stroke_width: int = 2,
                    shadow: bool = True) -> np.ndarray:
    """
    Add text overlay (name and ring number) to the image.
    
    Args:
        image: Input image (BGR)
        name: Bird name
        ring_number: Ring number
        position: Text position ("bottom-right", "bottom-left", "top-right", "top-left")
        offset: Offset from corner (x, y) in pixels
        font_size_name: Font size for name in points
        font_size_ring: Font size for ring number in points
        text_color: Text color (RGB)
        stroke_color: Stroke/outline color (RGB)
        stroke_width: Stroke width in pixels
        shadow: Whether to add drop shadow
    
    Returns:
        Image with text overlay (BGR)
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Load font
    try:
        if config.FONT_FILE.exists():
            font_name = ImageFont.truetype(str(config.FONT_FILE), font_size_name)
            font_ring = ImageFont.truetype(str(config.FONT_FILE), font_size_ring)
        else:
            # Use default font
            font_name = ImageFont.load_default()
            font_ring = ImageFont.load_default()
    except:
        font_name = ImageFont.load_default()
        font_ring = ImageFont.load_default()
    
    # Get text dimensions
    bbox_name = draw.textbbox((0, 0), name, font=font_name)
    bbox_ring = draw.textbbox((0, 0), ring_number, font=font_ring)
    
    text_width = max(bbox_name[2] - bbox_name[0], bbox_ring[2] - bbox_ring[0])
    text_height = (bbox_name[3] - bbox_name[1]) + (bbox_ring[3] - bbox_ring[1]) + 10  # 10px spacing
    
    # Calculate position
    img_width, img_height = pil_image.size
    
    if position == "bottom-right":
        x = img_width - text_width - offset[0]
        y = img_height - text_height - offset[1]
    elif position == "bottom-left":
        x = offset[0]
        y = img_height - text_height - offset[1]
    elif position == "top-right":
        x = img_width - text_width - offset[0]
        y = offset[1]
    else:  # top-left
        x = offset[0]
        y = offset[1]
    
    # Draw text with shadow if enabled
    if shadow:
        shadow_offset = config.TEXT_SHADOW_OFFSET
        shadow_color = (0, 0, 0, int(255 * config.TEXT_SHADOW_OPACITY))
        
        # Draw shadow for name
        draw.text((x + shadow_offset[0], y + shadow_offset[1]), name,
                 font=font_name, fill=shadow_color)
        # Draw shadow for ring number
        name_height = bbox_name[3] - bbox_name[1]
        draw.text((x + shadow_offset[0], y + name_height + 10 + shadow_offset[1]), 
                 ring_number, font=font_ring, fill=shadow_color)
    
    # Draw text with stroke
    # Name
    draw.text((x, y), name, font=font_name, fill=text_color,
             stroke_width=stroke_width, stroke_fill=stroke_color)
    # Ring number
    name_height = bbox_name[3] - bbox_name[1]
    draw.text((x, y + name_height + 10), ring_number, font=font_ring, fill=text_color,
             stroke_width=stroke_width, stroke_fill=stroke_color)
    
    # Convert back to BGR
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return result


def composite_final_image(template: np.ndarray, body: np.ndarray, head: np.ndarray,
                         feet: np.ndarray, feet_shadow: np.ndarray,
                         enhanced_eye: Optional[np.ndarray] = None,
                         eye_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Composite all components into the final image.
    
    Args:
        template: Background template (BGR)
        body: Warped body image (BGR or BGRA)
        head: Rotated head image (BGR or BGRA)
        feet: Translated feet image (BGR or BGRA)
        feet_shadow: Feet shadow image (RGBA)
        enhanced_eye: Optional enhanced eye image (RGBA)
        eye_position: Optional position for eye zoom bubble
    
    Returns:
        Final composite image (BGR)
    """
    # Start with template
    canvas = template.copy()
    
    # Ensure canvas is correct size
    if canvas.shape[:2] != config.OUTPUT_RESOLUTION[::-1]:  # (height, width)
        canvas = cv2.resize(canvas, config.OUTPUT_RESOLUTION, interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to RGBA for compositing
    canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    
    # Composite feet shadow first (lowest layer)
    if feet_shadow is not None and feet_shadow.shape[2] == 4:
        h_shadow, w_shadow = feet_shadow.shape[:2]
        # Position shadow at feet_center
        feet_center = config.IDEAL_SKELETON["feet_center"]
        x_start = max(0, feet_center[0] - w_shadow // 2)
        y_start = max(0, feet_center[1] - h_shadow // 2 + config.SHADOW_OFFSET_Y)
        x_end = min(canvas_rgba.shape[1], x_start + w_shadow)
        y_end = min(canvas_rgba.shape[0], y_start + h_shadow)
        
        if x_end > x_start and y_end > y_start:
            shadow_region = feet_shadow[0:y_end-y_start, 0:x_end-x_start]
            canvas_region = canvas_rgba[y_start:y_end, x_start:x_end]
            
            alpha = shadow_region[:, :, 3:4] / 255.0
            canvas_region = (canvas_region * (1 - alpha) + shadow_region[:, :, :3] * alpha).astype(np.uint8)
            canvas_rgba[y_start:y_end, x_start:x_end] = canvas_region
    
    # Composite body
    if body is not None:
        if len(body.shape) == 4 or body.shape[2] == 4:
            body_rgba = body
        else:
            body_rgba = cv2.cvtColor(body, cv2.COLOR_BGR2BGRA)
        
        # Simple alpha composite (body should already be positioned correctly)
        h_body, w_body = body_rgba.shape[:2]
        if h_body <= canvas_rgba.shape[0] and w_body <= canvas_rgba.shape[1]:
            alpha = body_rgba[:, :, 3:4] / 255.0
            canvas_region = canvas_rgba[0:h_body, 0:w_body]
            canvas_region = (canvas_region * (1 - alpha) + body_rgba[:, :, :3] * alpha).astype(np.uint8)
            canvas_rgba[0:h_body, 0:w_body] = canvas_region
    
    # Composite head (already blended with body via Poisson, but may need final composite)
    # Head should already be composited in harmonization step
    
    # Composite feet
    if feet is not None:
        if len(feet.shape) == 4 or feet.shape[2] == 4:
            feet_rgba = feet
        else:
            feet_rgba = cv2.cvtColor(feet, cv2.COLOR_BGR2BGRA)
        
        # Position at feet_center
        feet_center = config.IDEAL_SKELETON["feet_center"]
        h_feet, w_feet = feet_rgba.shape[:2]
        x_start = max(0, feet_center[0] - w_feet // 2)
        y_start = max(0, feet_center[1] - h_feet // 2)
        x_end = min(canvas_rgba.shape[1], x_start + w_feet)
        y_end = min(canvas_rgba.shape[0], y_start + h_feet)
        
        if x_end > x_start and y_end > y_start:
            feet_region = feet_rgba[0:y_end-y_start, 0:x_end-x_start]
            canvas_region = canvas_rgba[y_start:y_end, x_start:x_end]
            
            alpha = feet_region[:, :, 3:4] / 255.0
            canvas_region = (canvas_region * (1 - alpha) + feet_region[:, :, :3] * alpha).astype(np.uint8)
            canvas_rgba[y_start:y_end, x_start:x_end] = canvas_region
    
    # Composite enhanced eye if provided
    if enhanced_eye is not None and eye_position is not None:
        canvas_rgba = cv2.cvtColor(canvas_rgba, cv2.COLOR_BGRA2BGR)
        canvas_rgba = composite_eye_zoom_bubble(canvas_rgba, enhanced_eye, eye_position)
    else:
        canvas_rgba = cv2.cvtColor(canvas_rgba, cv2.COLOR_BGRA2BGR)
    
    return canvas_rgba

