import json
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from cloudwatch_monitoring import CloudWatchMonitoringManager, infer_document_type
from hipaa_compliance import (
    build_phi_detection_summary,
    create_secure_client,
    detect_phi_entities,
)

TIER1_PAGE_TARGET_SECONDS = float(os.getenv("TIER1_PAGE_TARGET_SECONDS", "5"))


def _process_single_page(textract_client, monitor, img_path, output_dir, queries):
    page_start = time.perf_counter()
    base_name = os.path.basename(img_path).split('.')[0]
    try:
        with open(img_path, 'rb') as document:
            image_bytes = document.read()

        response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=["QUERIES", "TABLES", "FORMS"],
            QueriesConfig={'Queries': queries}
        )

        text_lines = [
            block.get("Text", "")
            for block in response.get("Blocks", [])
            if block.get("BlockType") == "LINE" and block.get("Text")
        ]
        extracted_text = "\n".join(text_lines)
        phi_entities = detect_phi_entities(extracted_text)
        response["PhiDetection"] = build_phi_detection_summary(phi_entities)

        output_file = os.path.join(output_dir, f"{base_name}_textract.json")
        with open(output_file, 'w') as f:
            json.dump(response, f, indent=4)

        latency = time.perf_counter() - page_start
        if monitor:
            try:
                monitor.publish_extraction_result(
                    document_id=base_name,
                    success=True,
                    latency_seconds=latency,
                    document_type=infer_document_type(base_name),
                )
            except Exception:
                pass
        return {
            "page": img_path,
            "status": "SUCCESS",
            "output_file": output_file,
            "latency_seconds": round(latency, 4),
            "target_met": latency <= TIER1_PAGE_TARGET_SECONDS,
        }
    except Exception as e:
        latency = time.perf_counter() - page_start
        if monitor:
            try:
                monitor.publish_extraction_result(
                    document_id=base_name,
                    success=False,
                    latency_seconds=latency,
                    document_type=infer_document_type(base_name),
                )
            except Exception:
                pass
        return {
            "page": img_path,
            "status": "FAILED",
            "error": str(e),
            "latency_seconds": round(latency, 4),
            "target_met": False,
        }


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
    try:
        monitor = CloudWatchMonitoringManager()
    except Exception:
        monitor = None

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

    max_workers = max(1, min(len(cleaned_images), int(os.getenv("TIER1_MAX_WORKERS", "4"))))
    page_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_single_page, textract_client, monitor, img_path, output_dir, queries)
            for img_path in cleaned_images
        ]
        for future in as_completed(futures):
            result = future.result()
            page_results.append(result)
            if result["status"] == "SUCCESS":
                print(f"SUCCESS: Saved extracted data to {result['output_file']}")
            else:
                print(f"FAILED to process {result['page']}. Error: {result.get('error', 'unknown')}")

    metrics_path = os.path.join(output_dir, "tier1_latency_profile.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_pages": len(cleaned_images),
                "max_workers": max_workers,
                "target_seconds_per_page": TIER1_PAGE_TARGET_SECONDS,
                "pages": page_results,
            },
            handle,
            indent=2,
        )
    print(f"Latency profile saved: {metrics_path}")

if __name__ == "__main__":
    process_documents_with_textract()
