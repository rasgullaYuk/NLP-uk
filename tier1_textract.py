import boto3
import json
import os
import glob

def process_documents_with_textract(input_dir="temp_pages", output_dir="textract_outputs"):
    """
    Tier 1: Document Extraction Layer
    Reads cleaned images, sends them to Textract, and saves the structured JSON.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the Textract client (it automatically uses your new keys!)
    textract_client = boto3.client('textract', region_name='us-east-1')

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

            # Create a nice filename for the output
            base_name = os.path.basename(img_path).split('.')[0]
            output_file = os.path.join(output_dir, f"{base_name}_textract.json")
            
            # Save the results
            with open(output_file, 'w') as f:
                json.dump(response, f, indent=4)
                
            print(f"SUCCESS: Saved extracted data to {output_file}")

        except Exception as e:
            print(f"FAILED to process {img_path}. Error: {e}")

if __name__ == "__main__":
    process_documents_with_textract()