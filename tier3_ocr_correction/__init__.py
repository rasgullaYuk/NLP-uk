"""
__init__.py — tier3_ocr_correction package
==========================================
Exposes the public API surface of Tier 3.

Usage
-----
    from tier3_ocr_correction import process_low_confidence_regions
    from PIL import Image

    result = process_low_confidence_regions(
        low_confidence_regions=[
            {
                "text":        "Metflrmin 500 mg",
                "confidence":  0.52,
                "bbox":        [120, 300, 480, 340],
                "page_number": 1,
            }
        ],
        page_image=Image.open("page_1_CLEANED.png"),
        surrounding_context_text="Patient is a diabetic on oral hypoglycaemics.",
        confidence_threshold=0.80,   # optional override
    )

    print(result["status"])          # "SUCCESS" or "REVIEW_REQUIRED"
    print(result["corrected_regions"])
    print(result["audit_log"])
"""

from tier3_processor import process_low_confidence_regions  # noqa: F401
from config import ReasonCode, DEFAULT_OCR_CONFIDENCE_THRESHOLD  # noqa: F401

__all__ = [
    "process_low_confidence_regions",
    "ReasonCode",
    "DEFAULT_OCR_CONFIDENCE_THRESHOLD",
]
