import json
import os
import glob
from hipaa_compliance import (
    build_phi_detection_summary,
    create_secure_client,
    detect_phi_entities,
)

def process_documents_with_textract(input_dir="temp_pages", output_dir="textract_outputs"):
    """
    Tier 1: Document Extraction Layer
    Reads cleaned images, sends them to Textract, and saves the structured JSON.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the Textract client (it automatically uses your new keys!)
    textract_client = create_secure_client('textract', region_name='us-east-1')

    # Find all the CLEANED images in your temp folder
    cleaned_images = glob.glob(os.path.join(input_dir, "*_CLEANED.*"))
    
    if not cleaned_images:
        print(f"No cleaned images found in {input_dir}. Please run Tier 0 first!")
        return

    print(f"Found {len(cleaned_images)} pages to process. Starting extraction...")

    # Medical Queries tailored for SNOMED mapping
    queries = [
        {"Text": "What are the patient's primary diagnoses?", "Alias": "DIAGNOSIS"},
        {"Text": "What medications is the patient currently taking?", "Alias": "MEDICATIONS"},
        {"Text": "What are the key clinical findings or symptoms?", "Alias": "FINDINGS"}
    ]

    for img_path in cleaned_images:
        print(f"\nProcessing: {img_path}")
        
        with open(img_path, 'rb') as document:
            image_bytes = document.read()

        try:
            # Call Textract
            response = textract_client.analyze_document(
                Document={'Bytes': image_bytes},
                FeatureTypes=["QUERIES", "TABLES", "FORMS"],
                QueriesConfig={'Queries': queries}
            )

            # Detect PHI in extracted clinical text and attach compliance metadata
            text_lines = [
                block.get("Text", "")
                for block in response.get("Blocks", [])
                if block.get("BlockType") == "LINE" and block.get("Text")
            ]
            extracted_text = "\n".join(text_lines)
            phi_entities = detect_phi_entities(extracted_text)
            response["PhiDetection"] = build_phi_detection_summary(phi_entities)

            # Create a nice filename for the output
            base_name = os.path.basename(img_path).split('.')[0]
            output_file = os.path.join(output_dir, f"{base_name}_textract.json")
            
            # Save the results
            with open(output_file, 'w') as f:
                json.dump(response, f, indent=4)
                
            print(
                f"SUCCESS: Saved extracted data to {output_file} | "
                f"PHI flagged: {response['PhiDetection']['entity_count']}"
            )

        except Exception as e:
            print(f"FAILED to process {img_path}. Error: {e}")

if __name__ == "__main__":
    process_documents_with_textract()
