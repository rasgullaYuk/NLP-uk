from document_handler import prepare_document
from preprocessing import preprocess_image
import os

def run_universal_preprocessing(input_file):
    print("--- STARTING TIER 0: UNIVERSAL PREPROCESSING ---")
    
    # Check if the file actually exists on your computer
    if not os.path.exists(input_file):
        print(f"ERROR: I cannot find '{input_file}'. Please check the name!")
        return

    # Phase 1: The Receptionist handles the file formats
    # It doesn't matter if it's PDF, TIFF, or JPEG!
    page_images = prepare_document(input_file)
    
    if not page_images:
        print("Failed to prepare images. Pipeline stopped.")
        return

    # Phase 2: OpenCV Cleaning (Magic Glasses)
    print("\n[Phase 2: Cleaning Images with OpenCV]")
    for img_path in page_images:
        # Create a new filename for the cleaned version
        cleaned_path = img_path.replace(".jpg", "_CLEANED.jpg").replace(".jpeg", "_CLEANED.jpeg").replace(".png", "_CLEANED.png").replace(".tiff", "_CLEANED.tiff")
        
        preprocess_image(img_path, cleaned_path)

    print("\n--- TIER 0 COMPLETE! ---")
    print("Go look in your 'temp_pages' folder to see the cleaned results!")

if __name__ == "__main__":
    # You can change this filename to test different files!
    # Try "sample.pdf", then try "scan.tiff", then try "photo.jpeg"
    my_file = "sample_clinical_doc.pdf" 
    run_universal_preprocessing(my_file)