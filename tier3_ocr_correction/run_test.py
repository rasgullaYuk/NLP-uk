"""this file is for primary module testing only using sample pdfs provided in aws s3"""


from tier3_processor import process_low_confidence_regions
from pdf2image import convert_from_path
print("RUNNING TEST...")
# -------- STEP 1: LOAD PDF --------
pdf_path = "sample.pdf"   # CHANGE THIS to your file name

images = convert_from_path(pdf_path)

# Use page 3 (where meds are)
page_image = images[2]

# -------- STEP 2: FAKE OCR OUTPUT --------
low_conf_regions = [
    {
        "text": "Administer ONE drop into the left eye",
        "confidence": 0.55,
        "bbox": [100, 250, 500, 300],
        "page_number": 3,
    }
]

context = "Patient should receive TWO drops in left eye daily."
# -------- STEP 3: CONTEXT --------
#context = "Patient medication includes tobramycin dexamethasone eye drops administered to left eye multiple times daily."

# -------- STEP 4: RUN YOUR MODULE --------
result = process_low_confidence_regions(
    low_confidence_regions=low_conf_regions,
    page_image=page_image,
    surrounding_context_text=context,
)

# -------- STEP 5: PRINT OUTPUT --------
import json
print("\n========= RESULT =========\n")
print(json.dumps(result, indent=2))
print("Done")