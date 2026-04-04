import cv2
import numpy as np
import logging
import os

logger = logging.getLogger("tier0.preprocessing")


def preprocess_image(image_path, output_path):
    """
    Tier 0 - Image Preprocessing (OpenCV)
    Applies adaptive thresholding, morphological operations, and deskewing.

    Fixed vs original:
    - Deskew no longer crashes when coords is empty (blank/white pages)
    - Skips trivial rotations (<0.3 degrees) to avoid sub-pixel smearing
    - Logs each step for traceability
    - Returns output_path so callers can chain it
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    logger.info("Preprocessing: %s", os.path.basename(image_path))

    # 1. Adaptive Thresholding (Binarization & Lighting Correction)
    binarized = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,   # larger block handles uneven lighting better
        C=10
    )

    # 2. Morphological Operations (Noise Reduction)
    kernel = np.ones((2, 2), np.uint8)
    denoised = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # 3. Deskewing (Orientation Detection & Correction)
    deskewed = _deskew(denoised)

    cv2.imwrite(output_path, deskewed)
    print(f"  Preprocessed -> {os.path.basename(output_path)}")
    logger.info("Saved preprocessed image: %s", output_path)

    return output_path


def _deskew(img):
    """
    Detect and correct page skew safely.

    BUG FIX from original: np.where on a blank image returns empty coords,
    and cv2.minAreaRect crashes on an empty array. We guard against this.
    """
    coords = np.column_stack(np.where(img > 0))

    # Guard: if too few foreground pixels, nothing to deskew
    if coords is None or len(coords) < 100:
        logger.debug("Skipping deskew — not enough foreground pixels.")
        return img

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip trivial corrections to avoid sub-pixel smearing
    if abs(angle) < 0.3:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    logger.debug("Deskewed by %.2f degrees", angle)
    return deskewed


def preprocess_batch(image_paths):
    """
    Run preprocess_image on a list of image paths (output of document_handler).

    For each path like:  temp_pages/discharge_summary1page1_original.jpg
    Writes cleaned file: temp_pages/discharge_summary1page1_CLEANED.jpg

    Returns:
        success : list of {"original": path, "cleaned": path}
        failed  : list of {"original": path, "error": str}
    """
    success = []
    failed = []

    for img_path in image_paths:
        try:
            cleaned_path = img_path.replace("_original.jpg", "_CLEANED.jpg")
            preprocess_image(img_path, cleaned_path)
            success.append({
                "original": img_path,
                "cleaned": cleaned_path,
            })
        except Exception as e:
            print(f"  FAILED to preprocess {os.path.basename(img_path)}: {e}")
            logger.error("Preprocessing failed for %s: %s", img_path, e)
            failed.append({
                "original": img_path,
                "error": str(e),
            })

    return success, failed


def get_tier1_payload(success_pages):
    """
    Flatten preprocessing results into the format Tier 1 (Textract) expects.

    Returns:
    [
        {
            "image_path" : "temp_pages/discharge_summary1page1_CLEANED.jpg",
            "doc_name"   : "discharge_summary1",
            "page"       : 1,
        },
        ...
    ]
    """
    payload = []
    for item in success_pages:
        cleaned = item["cleaned"]
        filename = os.path.basename(cleaned)
        try:
            name_part = filename.replace("_CLEANED.jpg", "")
            page_idx = name_part.rfind("page")
            doc_name = name_part[:page_idx]
            page_num = int(name_part[page_idx + 4:])
        except Exception:
            doc_name = filename
            page_num = 0

        payload.append({
            "image_path": cleaned,
            "doc_name": doc_name,
            "page": page_num,
        })
    return payload