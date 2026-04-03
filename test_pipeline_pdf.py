import time
import os
import statistics
from document_handler import convert_pdf_to_images
from preprocessing import preprocess_image
from textract_extraction import run_textract_extraction
from sqs_setup import setup_pipeline_queues
from sqs_messaging import send_to_sqs

def run_pdf_test():
    print("--- STARTING MULTI-PAGE PDF PIPELINE TEST ---")
    
    queue_urls = setup_pipeline_queues()
    
    # Place a multi-page PDF in your folder and name it "sample_clinical_doc.pdf"
    input_pdf = "sample_clinical_doc.pdf" 
    
    # 1. Convert PDF to Images
    print("\n[Extracting Pages from PDF]")
    try:
        page_images = convert_pdf_to_images(input_pdf)
    except Exception as e:
        print(e)
        return

    all_extracted_text = []
    all_confidence_scores = []

    # 2. Process Each Page
    for i, img_path in enumerate(page_images):
        print(f"\n--- Processing Page {i + 1} ---")
        
        # Tier 0: Preprocessing
        processed_img_path = img_path.replace(".jpg", "_processed.jpg")
        preprocess_image(img_path, processed_img_path)
        
        # Tier 1: Textract
        page_result = run_textract_extraction(processed_img_path)
        
        all_extracted_text.append(f"--- PAGE {i + 1} ---\n{page_result['text']}")
        if page_result['average_confidence'] > 0:
            all_confidence_scores.append(page_result['average_confidence'])

    # 3. Aggregate Results
    combined_text = "\n\n".join(all_extracted_text)
    doc_average_confidence = statistics.mean(all_confidence_scores) if all_confidence_scores else 0.0
    
    print("\n=========================================")
    print(f"DOCUMENT EXTRACTION COMPLETE")
    print(f"Total Pages Processed: {len(page_images)}")
    print(f"Document Average Confidence: {doc_average_confidence:.2f}%")
    print("=========================================\n")

    # 4. Routing Logic
    payload = {
        "document_id": "pdf_doc_001",
        "text": combined_text,
        "average_confidence": doc_average_confidence
    }

    if doc_average_confidence >= 90.0:
        print("High confidence (>= 90%). Routing to Track A SQS Queue...")
        target_queue_url = queue_urls['TrackA_Entity_SNOMED_Queue']
    else:
        print("Low confidence (< 90%). Escalating to Tier 2 (LayoutLMv3) SQS Queue...")
        target_queue_url = queue_urls['Tier2_LayoutLM_Queue']
        
    # Send the combined 5-page document payload to the queue
    send_to_sqs(target_queue_url, payload)

if __name__ == "__main__":
    run_pdf_test()