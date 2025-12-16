# PROJECT: PigeonPhoto_Automator (Master Specification v2)

**Version:** 2.0 (Final)
**Document Type:** Master Implementation Guide
**Intended Audience:** AI Engineers / Senior Python Developers / LLM Coding Agents

---

## 1. Executive Summary
**Objective:** Build an automated web-based pipeline to transform raw racing pigeon photographs into "Golden Prince" standard portraits.
**Core Challenge:** Pigeons act as "fingerprints"; their feather patterns cannot be hallucinated by Generative AI. The solution requires precise **geometric transformation** of original pixels.
**Core Method:** "Surgical Component Assembly." The bird is not warped globally. Instead, it is segmented into three distinct components—**Head (Rigid)**, **Feet (Independent)**, and **Body (Flexible)**—which are re-assembled onto a template using Thin Plate Spline (TPS) warping and Poisson blending.

**Input Requirements:**
1.  `Body_Raw.jpg`: Full body shot of the bird.
   - Format: JPG, PNG (RGB or RGBA)
   - Recommended: Minimum 2000px on longest side
   - Content: Bird should be clearly visible with minimal occlusion
   
2.  `Eye_Macro.jpg`: High-resolution macro shot of the eye.
   - Format: JPG, PNG (RGB)
   - Recommended: Minimum 1000x1000px
   - Content: Eye should be centered and in focus
   - **Note:** Should ideally be from the same bird as Body_Raw.jpg

---

## 2. Technical Stack
* **Language:** Python 3.9+
* **Core Computer Vision:** `OpenCV` (cv2), `NumPy`, `scikit-image`
* **Segmentation:** `rembg` (U2-Net) or `Segment Anything Model (SAM)`
* **Pose Estimation:** `YOLOv8-Pose` (Ultralytics) - Custom fine-tuned on client dataset.
* **Warping Algorithms:** `scipy.interpolate.Rbf` with `function='thin-plate'` for TPS warping
* **Harmonization:** `cv2.seamlessClone` (Poisson Blending), `skimage.exposure`
* **Frontend:** Streamlit (for MVP) or React/FastAPI (Production)

---

## 3. System Architecture & Logic Pipeline

### Module A: Inputs & Configuration
**1. The "Ideal Skeleton" (Target Coordinates):**
A JSON definition of the perfect "Golden Prince" posture on a 3000x2000px canvas.
```json
{
  "beak_tip": [2200, 400],
  "eye_center": [2100, 450],
  "skull_back": [2000, 500],
  "neck_base": [1900, 800],
  "shoulder": [1600, 900],
  "wing_tip": [1400, 1000],
  "tail_tip": [300, 1400],
  "left_leg_joint": [1600, 1600],
  "right_leg_joint": [1700, 1600],
  "feet_center": [1650, 1600]
}
```

**2. The Keypoint Detector (Custom Model):** A YOLOv8-Pose model trained to detect 9 specific anatomical points:

- **Head Group:** Beak Tip, Eye Center, Skull Back.
- **Body Group:** Neck Base, Shoulder, Wing Tip, Tail Tip.
- **Leg Group:** Left Leg Joint, Right Leg Joint.

**Note:** The `feet_center` is a computed midpoint for positioning, not a detected keypoint.

**3. Background Template Configuration:**
- **Template File:** `Pigeon Template.jpg` (located in project root or `/assets/templates/`)
- **Expected Dimensions:** 3000x2000px
- **Zoom Bubble Configuration:** (To be extracted from template or defined in config JSON)
  - **Position:** [x, y] coordinates of bubble center
  - **Size:** Width and height in pixels
  - **Note:** The zoom bubble area should be visually identified on the template and coordinates added to configuration

### Module B: The "Surgical Disassembly" (Geometry Engine)
Critically important: Do not apply one global warp. Split the bird into three layers with unique physics.

**Layer 1: The Head (Rigid Transformation)**

- **Action:** Isolate the head using a circular mask (radius = distance from Eye Center to Neck Base).
- **Logic:** 
  - Calculate the angle of the Beak-to-Eye vector in the raw image: `θ_raw = atan2(eye_y - beak_y, eye_x - beak_x)`
  - Calculate the target angle from the ideal template: `θ_target = atan2(450 - 400, 2100 - 2200) = atan2(-50, -100) ≈ -153.43°` (or `26.57°` above horizontal)
  - Compute rotation angle: `Δθ = θ_target - θ_raw`
  - Apply rigid rotation around Eye Center pivot point using `cv2.getRotationMatrix2D()` and `cv2.warpAffine()`
- **Constraint:** Never warp the head pixels. The eye shape must remain a perfect circle. Use only rotation + translation (affine transformation), no non-linear warping.

**Layer 2: The Feet (Translation)**

- **Action:** Cut out the feet/legs based on the Leg Joint keypoints (Left Leg Joint, Right Leg Joint).
- **Logic:** 
  - Create a bounding box around both leg joints with padding
  - Extract feet region using segmentation mask
  - Calculate translation vector: `T = feet_center_target - feet_center_source`
  - Apply translation using `cv2.warpAffine()` with identity rotation matrix
- **Enhancement:** Generate a "Contact Shadow" (blurred black oval) underneath the feet to ground them within the template 3D space.
  - Create elliptical mask: width = 1.5 × feet_width, height = 0.5 × feet_width
  - Position: centered below feet_center, offset downward by 10-20px
  - Apply `cv2.GaussianBlur()` with kernel size (15, 15) and sigma=5
  - Use semi-transparent dark gray (RGB: 40, 40, 40) with 60% opacity
  - Blend using `cv2.addWeighted()` with alpha=0.6

**Layer 3: The Body & Neck (The Bridge)**

- **Action:** Contains the neck feathers, chest, wing, and tail.
- **Logic:** Apply Thin Plate Spline (TPS) warping using `scipy.interpolate.Rbf(function='thin-plate')`.
- **Control Points:**
  - **Source points:** All detected keypoints from Body_Raw.jpg (Neck Base, Shoulder, Wing Tip, Tail Tip)
  - **Target points:** Corresponding ideal template coordinates
  - **Anchors:** Lock the Shoulder and Wing Tip keypoints (keep them relatively stable with higher weight)
- **Stretch:** Pull the Neck Base pixels upwards to meet the new location of the Rotated Head.
- **Fill:** The warping will naturally stretch the iridescent neck feathers to fill the gap.

### Module C: Image Harmonization (Lighting & Blending)
Required to handle inconsistent lighting (flash vs. ambient) and different bird colors.

**1. Head-to-Body Stitching (The Invisible Seam)**

- **Problem:** The Rotated Head and Stretched Neck will have a visible cut line.
- **Solution:** Use Poisson Blending (`cv2.seamlessClone` with `MIXED_CLONE` flag).
- **Implementation:**
  - Create a mask for the head-neck boundary region (circular gradient mask)
  - Use `cv2.seamlessClone(head_layer, body_layer, mask, center, cv2.MIXED_CLONE)`
  - The `center` parameter should be at the neck base connection point
- **Why:** This algorithm forces the neck colors to "bleed" into the head's edge, matching local gradients and hiding the seam, even if the head is "Cool" light and the body is "Warm" light.

**2. Eye Exposure Matching**

- **Problem:** The macro eye photo is often flash-lit (bright); the body is ambient-lit (dark).
- **Solution:** 
  - Calculate the Mean Luminance (L channel in LAB color space) of pixels around the bird's face region (circular area around eye_center)
  - Calculate target luminance from the face region: `L_target = mean(L_face_region)`
  - Calculate source luminance from Eye_Macro: `L_source = mean(L_eye_macro)`
  - Compute gamma correction: `gamma = log(L_target / 255) / log(L_source / 255)`
  - Apply gamma correction using `skimage.exposure.adjust_gamma(Eye_Macro, gamma)`
  - **Alternative:** Use histogram matching (`skimage.exposure.match_histograms`) for more sophisticated matching

**3. Global Color Grading**

- **Action:** Apply a subtle S-Curve contrast adjustment to the final composite (Bird + Background) to unify black levels.
- **Implementation:** Use `skimage.exposure.adjust_sigmoid()` or `cv2.createCLAHE()` for adaptive contrast enhancement.

### Module D: Eye Enhancement (The "Zoom Bubble")
**Action:** Process the high-res Eye_Macro.jpg.

**Steps:**

1. **Crop:** Extract a perfect square around the pupil.
   - **Automatic pupil detection:** Use `cv2.HoughCircles()` on grayscale image with appropriate parameters, or apply edge detection (`cv2.Canny`) + contour analysis to find the darkest circular region (pupil)
   - **Fallback:** If automatic detection fails, use center of image as pupil center
   - Create square crop with side length = 2 × max(pupil_radius, eye_width/2)
   - Center crop on detected pupil center

2. **Mask:** Apply a soft-edged circular mask.
   - Create circular mask with soft edges using Gaussian blur on binary mask
   - Apply mask to cropped eye image for smooth blending

3. **Overlay:** Superimpose a transparent PNG "Glass Reflection" layer (simulating a studio softbox reflection) to make the eye pop.
   - Load `glass_reflection.png` from `/assets/overlays/`
   - Resize to match eye crop dimensions
   - Blend using `cv2.addWeighted()` with appropriate opacity (typically 0.3-0.5)

4. **Placement:** Composite this result into the specific "Zoom Bubble" area of the background template.
   - Use alpha blending: `result = background * (1 - alpha) + eye_overlay * alpha`
   - **Note:** The "Zoom Bubble" coordinates and dimensions need to be defined in the template configuration JSON.

---

## 4. Development Phases & Checklist
Phase 1: Data Preparation (Client Side)
[ ] Select 300 diverse "Before" images.

[ ] Upload to Roboflow/CVAT.

[ ] Label the 9 Keypoints defined in Module A.

[ ] Rule: Ensure consistency (e.g., "Shoulder" is always the top curve of the wing).

Phase 2: Core AI (Developer Side)
[ ] Train YOLOv8-pose on the dataset. Target: >95% keypoint accuracy.

[ ] Build segmentation.py using rembg to generate clean alpha masks.

Phase 3: The Geometry Script
[ ] Build geometry.py.

[ ] Implement scipy.interpolate.Rbf for TPS Warping.

[ ] Implement Rigid Rotation logic for the head.

[ ] Unit Test: Input one bird -> Output a bird with "Long Neck" and "Rotated Head."

Phase 4: Compositing & Harmonization
[ ] Build blender.py.

[ ] Implement cv2.seamlessClone for the neck connection.

[ ] Implement the "Feet Shadow" generator.

[ ] Implement the "Text Overlay" (Pillow library) for Name/Ring Number.

Phase 5: Interface (Web App)
[ ] Batch Upload: Drag & drop box for pairs of Body + Eye.

[ ] Review Canvas: A screen showing the AI's result.

[ ] Manual Override (Critical): If the AI fails, allow the user to drag the "Keypoints" to fix the skeleton and re-run the Warp button immediately.

[ ] Download: Export high-res JPG.

---

## 5. Suggested File Structure
Direct the LLM/Developer to use this exact structure.

```

/pigeon-automator
│
├── /assets
│   ├── /templates
│   │   └── Pigeon Template.jpg (3000x2000px background)
│   ├── /overlays (glass_reflection.png, shadow.png)
│   └── /fonts (TrajanPro.ttf)
│
├── /models
│   └── pigeon_pose_v1.pt (The trained YOLO model)
│
├── /src
│   ├── segmentation.py (Rembg logic)
│   ├── keypoints.py (Inference logic)
│   ├── geometry.py (TPS Warping & Rigid Rotation logic)
│   ├── harmonization.py (Poisson blending & exposure matching)
│   ├── compositor.py (Final assembly & text)
│   └── config.py (Template paths, ideal skeleton JSON, zoom bubble coordinates)
│
├── app.py (Streamlit/React entry point)
├── requirements.txt
└── config.json (Optional: Template configuration, zoom bubble coordinates, default UI settings)
```

---

## 6. Technical Clarifications & Open Questions

### 6.1 TPS Implementation Details
**RESOLVED:**
- **Implementation:** Use `scipy.interpolate.Rbf(function='thin-plate')` for TPS warping
- **Rationale:** Provides TPS-like behavior with simpler API than custom TPS implementation
- **Alternative:** If more precise TPS needed, can implement custom TPS using `scipy.spatial.distance` (not recommended for MVP)

### 6.2 Background Template
**RESOLVED:**
- **Template Path:** `E:\Users\ethan\Documents\PigeonPhoto_Automator\Pigeon Template.jpg` (or relative path: `Pigeon Template.jpg` in project root)
- **Implementation:** Load template at runtime, verify dimensions match expected 3000x2000px
- **Zoom Bubble:** Coordinates and dimensions should be extracted from template analysis or defined in configuration JSON (see Section 6.2.1 below)
- **Note:** The template should be placed in `/assets/templates/` directory for deployment

### 6.3 Eye Macro Processing
**RESOLVED:**
- **Pupil Detection:** Automatic detection using `cv2.HoughCircles()` or contour analysis (see Module D for details)
  - **Fallback:** If detection fails, use image center as pupil center
- **Zoom Bubble Size:** 
  - **Default:** 400x400px square (user-configurable in UI)
  - **Position:** To be determined from template analysis or defined in config JSON
- **Glass Reflection Overlay:**
  - **File Format:** PNG with alpha channel (transparent background)
  - **Opacity:** 0.3-0.5 (user-adjustable, default: 0.4)
  - **Blend Mode:** Normal (alpha blending via `cv2.addWeighted()`)
  - **File Location:** `/assets/overlays/glass_reflection.png`

### 6.4 Text Overlay Specifications
**RESOLVED:**
- **UI Configurable:** Text position, size, color, and style will be adjustable in the web app UI
- **Default Recommendations:**
  - **Position:** Bottom-right corner, with configurable offset (default: 50px from edges)
  - **Font:** TrajanPro.ttf (or system fallback: Arial/Helvetica)
  - **Default Size:** 48pt for Name, 36pt for Ring Number
  - **Default Color:** White (#FFFFFF) with black stroke/outline for readability
  - **Shadow:** Optional drop shadow (offset: 2px, blur: 4px, opacity: 0.7)
  - **Format:** Name on first line, Ring Number on second line (or side-by-side, user configurable)
- **Implementation:** Use PIL/Pillow `ImageDraw.text()` with `stroke_width` parameter for outline

### 6.5 Error Handling & Edge Cases
**RESOLVED with Default Recommendations:**
- **Keypoint Detection Threshold:** Confidence threshold = 0.5 (configurable)
  - If confidence < 0.5: Flag for manual review, allow user to drag keypoints in UI
  - If keypoint missing: Use interpolation from neighboring keypoints or fail gracefully with error message
- **Missing Keypoints:** 
  - **Critical keypoints** (Eye Center, Neck Base, Shoulder): Must be present, fail if missing
  - **Optional keypoints** (Skull Back, Wing Tip): Interpolate from available points
  - **User Override:** Always allow manual keypoint adjustment in UI (Phase 5 requirement)
- **Segmentation Failure:** 
  - If rembg fails: Fall back to threshold-based segmentation or edge detection
  - If still fails: Return error with diagnostic message, allow user to upload new image
- **Input Validation:**
  - **Formats:** JPG, JPEG, PNG (RGB or RGBA)
  - **Min Size:** 1000x1000px for Body_Raw, 500x500px for Eye_Macro
  - **Max Size:** 10000x10000px (resize if larger to prevent memory issues)
  - **Aspect Ratio:** No strict limits, but warn if extreme (e.g., > 3:1 or < 1:3)
- **Bird Matching:** 
  - No automatic validation (user responsibility)
  - Optional: Add UI checkbox "Same bird?" for user confirmation

### 6.6 Output Specifications
**RESOLVED with Default Recommendations:**
- **Output Resolution:** Match template dimensions (3000x2000px) - fixed to maintain consistency
- **Output Format:** 
  - **Primary:** JPG with quality=95 (high quality, smaller file size)
  - **Optional:** PNG with alpha channel (if transparency needed, user selectable in UI)
- **Color Space:** sRGB (standard for web/print compatibility)
- **DPI/PPI:** 300 DPI (standard print quality) - embedded in EXIF metadata
- **File Naming:** `{bird_name}_{ring_number}_golden_prince.jpg` (or user-configurable pattern)

### 6.7 Coordinate System
**RESOLVED:**
- **Format:** (x, y) where x = horizontal (column), y = vertical (row)
- **Origin:** Top-left corner is (0, 0) - standard image coordinate system
- **Units:** Pixels (absolute pixel coordinates)
- **Note:** OpenCV uses (x, y) format with top-left origin, which matches this specification

### 6.8 Head Rotation Logic
**RESOLVED:**
- **Exact Target Angle:** Calculated from ideal skeleton coordinates:
  - Beak-to-Eye vector: (2200-2100, 400-450) = (100, -50)
  - Angle: `atan2(-50, 100) ≈ -26.57°` (or `26.57°` above horizontal)
  - **Note:** The "~30°" approximation is close; exact value is `26.57°`
- **Rotation Pivot:** Eye Center (as specified in Module B, Layer 1)
- **Canvas Bounds Handling:**
  - Before rotation: Check if rotated head would exceed canvas bounds
  - If bounds exceeded: Scale down head layer proportionally (maintain aspect ratio) or extend canvas
  - **Recommended:** Extend canvas with transparent padding, then crop to final template size

### 6.9 Feet Shadow Generation
**RESOLVED:**
- **Shadow Parameters (Default):**
  - **Shape:** Elliptical (oval)
  - **Dimensions:** Width = 1.5 × feet_width, Height = 0.5 × feet_width
  - **Position:** Centered below `feet_center`, offset downward by 15px
  - **Blur:** Gaussian blur with kernel size (15, 15) and sigma=5
  - **Color:** Semi-transparent dark gray (RGB: 40, 40, 40)
  - **Opacity:** 60% (alpha=0.6)
  - **Direction:** Downward (vertical offset only, no horizontal offset)
- **Implementation:** See Module B, Layer 2 for detailed implementation

### 6.10 Batch Processing
**RESOLVED with Default Recommendations:**
- **File Pairing:**
  - **Primary Method:** Filename matching convention (e.g., `bird001_body.jpg` + `bird001_eye.jpg`)
  - **Alternative:** Manual pairing in UI (drag-and-drop matching interface)
  - **Metadata:** Optional EXIF tag matching (if available)
- **Processing Mode:**
  - **MVP (Streamlit):** Sequential processing (one bird at a time)
  - **Production (FastAPI):** Parallel processing with worker pool (configurable: 2-4 concurrent workers)
- **Performance Targets:**
  - **Target:** < 30 seconds per bird on standard hardware (CPU: 4+ cores, 8GB RAM)
  - **Bottlenecks:** YOLO inference (~2-5s), TPS warping (~5-10s), Poisson blending (~3-5s)
  - **Optimization:** GPU acceleration for YOLO (if available), caching of template/overlays

---

## 7. Known Risks & Mitigations
The "Frankenstein Neck" Seam: Addressed via Poisson Blending (Module C).

Floating Feet: Addressed via Shadow Generation and "Perch" translation (Module B, Layer 2).

Lighting Mismatch: Addressed via Gamma Correction/Luminance Matching (Module C).

---

## 8. Specification Status & Resolved Items

### ✅ Resolved Technical Questions
All major technical clarifications have been addressed:

1. **Background Template:** Path specified (`Pigeon Template.jpg`), dimensions confirmed (3000x2000px)
2. **Text Overlay:** UI-configurable with sensible defaults provided
3. **Error Handling:** Default thresholds and fallback strategies defined
4. **Output Specifications:** Fixed resolution, format, and quality settings specified
5. **Eye Detection:** Automatic detection method specified with fallback
6. **Head Rotation:** Exact angle calculated (26.57°) from ideal skeleton coordinates
7. **Shadow Parameters:** Detailed specifications provided for feet shadow generation
8. **Coordinate System:** Standardized to (x, y) with top-left origin
9. **TPS Implementation:** Method confirmed (`scipy.interpolate.Rbf`)
10. **Batch Processing:** Pairing strategy and performance targets defined

### 📋 Remaining Items for Implementation
- **Zoom Bubble Coordinates:** Need to be extracted from template image or manually defined
- **Glass Reflection Overlay:** Asset file needs to be created/obtained
- **Font File:** TrajanPro.ttf needs to be obtained or alternative specified
- **YOLO Model Training:** Dataset preparation and model training (Phase 1-2)

### 🎯 Ready for Development
The specification is now **implementation-ready** with all critical technical details resolved. Developers can proceed with Phase 1 (Data Preparation) and Phase 2 (Core AI) with confidence.