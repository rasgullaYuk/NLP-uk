import cv2
import numpy as np

def preprocess_image(image_path, output_path):
    """
    Tier 0 - Image Preprocessing (OpenCV)
    Applies adaptive thresholding, morphological operations, and basic deskewing.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # 1. Adaptive Thresholding (Binarization & Lighting Correction)
    binarized = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 2. Morphological Operations (Noise Reduction)
    kernel = np.ones((1, 1), np.uint8)
    denoised = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # 3. Deskewing (Orientation Detection & Correction)
    coords = np.column_stack(np.where(denoised > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = denoised.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save processed image
    cv2.imwrite(output_path, deskewed)
    print(f"Preprocessed image saved to {output_path}")
    
    return output_path

# Example Usage
# preprocess_image("raw_clinical_doc.jpg", "processed_doc.jpg")
