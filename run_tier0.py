"""
run_tier0.py - Tier 0 Entry Point
"""
import os, sys, json, time, logging, argparse
from document_handler import prepare_batch
from preprocessing import preprocess_batch, get_tier1_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("tier0.runner")

def run_universal_preprocessing(input_path, output_dir="temp_pages"):
    print("--- STARTING TIER 0: UNIVERSAL PREPROCESSING ---")
    t_start = time.time()

    if not os.path.exists(input_path):
        print(f"ERROR: Cannot find '{input_path}'. Please check the path!")
        return None

    print("\n[Phase 1: Document Handler - Extracting Pages]")
    all_image_paths, batch_results = prepare_batch(input_path, output_dir)

    if not all_image_paths:
        print("No images extracted. Pipeline stopped.")
        return None

    print(f"\n[Phase 2: OpenCV Preprocessing - Cleaning {len(all_image_paths)} page(s)]")
    success_pages, failed_pages = preprocess_batch(all_image_paths)

    elapsed = round(time.time() - t_start, 3)
    total_pages = len(all_image_paths)
    total_success = len(success_pages)
    total_failed = len(failed_pages)

    status = "SUCCESS"
    if total_failed > 0 and total_success == 0:
        status = "FAILED"
    elif total_failed > 0:
        status = "PARTIAL"

    tier1_payload = get_tier1_payload(success_pages)
    report = {
        "tier": 0,
        "status": status,
        "summary": {
            "total_pages": total_pages,
            "pages_succeeded": total_success,
            "pages_failed": total_failed,
            "elapsed_seconds": elapsed,
        },
        "documents": batch_results,
        "preprocessing_failures": failed_pages,
        "tier1_payload": tier1_payload,
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "tier0_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- TIER 0 COMPLETE! ---")
    print(f"Status  : {status}")
    print(f"Pages   : {total_success}/{total_pages} succeeded")
    print(f"Time    : {elapsed}s")
    print(f"Report  : {report_path}")
    print(f"Images  : look in '{output_dir}' folder!")

    if total_failed > 0:
        print(f"\nWARNING: {total_failed} page(s) failed:")
        for f in failed_pages:
            print(f"  - {f['original']}: {f['error']}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tier 0 - Preprocess clinical documents")
    parser.add_argument("input_path", nargs="?", default="sample_clinical_doc.pdf",
                        help="Path to a single document OR a directory")
    parser.add_argument("--output-dir", default="temp_pages")
    args = parser.parse_args()

    result = run_universal_preprocessing(args.input_path, args.output_dir)
    if result is None or result["status"] == "FAILED":
        sys.exit(1)
    elif result["status"] == "PARTIAL":
        sys.exit(1)
    else:
        sys.exit(0)