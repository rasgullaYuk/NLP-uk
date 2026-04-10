import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from cloudwatch_monitoring import CloudWatchMonitoringManager, infer_document_type
from cost_optimization import (
    RequestDeduplicator,
    content_hash,
    resolve_batch_window,
    split_into_batches,
)
from hipaa_compliance import (
    build_phi_detection_summary,
    create_secure_client,
    detect_phi_entities,
)

TIER1_PAGE_TARGET_SECONDS = float(os.getenv("TIER1_PAGE_TARGET_SECONDS", "5"))


def _process_single_page(textract_client, monitor, img_path, output_dir, queries):
    page_start = time.perf_counter()
    base_name = os.path.basename(img_path).split(".")[0]
    try:
        with open(img_path, "rb") as document:
            image_bytes = document.read()

        response = textract_client.analyze_document(
            Document={"Bytes": image_bytes},
            FeatureTypes=["QUERIES", "TABLES", "FORMS"],
            QueriesConfig={"Queries": queries},
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
        with open(output_file, "w", encoding="utf-8") as f:
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    textract_client = create_secure_client("textract", region_name="us-east-1")
    try:
        monitor = CloudWatchMonitoringManager()
    except Exception:
        monitor = None

    cleaned_images = glob.glob(os.path.join(input_dir, "*_CLEANED.*"))
    if not cleaned_images:
        print(f"No cleaned images found in {input_dir}. Please run Tier 0 first!")
        return

    print(f"Found {len(cleaned_images)} pages to process. Starting extraction...")
    batch_window = resolve_batch_window()
    deduplicator = RequestDeduplicator(
        fallback_file=os.path.join(output_dir, ".request_dedup_cache.json")
    )
    print(
        f"Processing mode: {batch_window.mode} | "
        f"batch_size={batch_window.max_batch_size} wait_seconds={batch_window.wait_seconds}"
    )

    queries = [
        {"Text": "What are the patient's primary diagnoses?", "Alias": "DIAGNOSIS"},
        {"Text": "What medications is the patient currently taking?", "Alias": "MEDICATIONS"},
        {"Text": "What are the key clinical findings or symptoms?", "Alias": "FINDINGS"},
    ]

    max_workers = max(1, int(os.getenv("TIER1_MAX_WORKERS", "4")))
    page_results = []
    for batch in split_into_batches(cleaned_images, batch_window.max_batch_size):
        unique_batch = []
        for img_path in batch:
            with open(img_path, "rb") as document:
                image_bytes = document.read()
            req_hash = content_hash(image_bytes)
            request_key = f"tier1_textract:{req_hash}"
            if deduplicator.is_duplicate(request_key):
                page_results.append(
                    {
                        "page": img_path,
                        "status": "SKIPPED_DUPLICATE",
                        "latency_seconds": 0.0,
                        "target_met": True,
                    }
                )
                print(f"SKIPPED duplicate request: {img_path}")
                continue
            unique_batch.append(img_path)

        if unique_batch:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_batch))) as executor:
                futures = [
                    executor.submit(
                        _process_single_page,
                        textract_client,
                        monitor,
                        img_path,
                        output_dir,
                        queries,
                    )
                    for img_path in unique_batch
                ]
                for future in as_completed(futures):
                    result = future.result()
                    page_results.append(result)
                    if result["status"] == "SUCCESS":
                        print(f"SUCCESS: Saved extracted data to {result['output_file']}")
                    else:
                        print(f"FAILED to process {result['page']}. Error: {result.get('error', 'unknown')}")

        if batch_window.mode == "scheduled" and batch_window.wait_seconds > 0:
            time.sleep(batch_window.wait_seconds)

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
