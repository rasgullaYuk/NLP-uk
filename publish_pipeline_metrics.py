"""
Publish CloudWatch custom metrics from pipeline output artifacts.

This script supports batch publishing for:
- Tier 1 extraction metrics
- Track A SNOMED mapping success
- Track B LLM latency/confidence
- Queue depth snapshots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

from cloudwatch_monitoring import (
    DEFAULT_QUEUE_NAMES,
    CloudWatchMonitoringManager,
    infer_document_type,
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def publish_textract_metrics(
    monitor: CloudWatchMonitoringManager,
    textract_dir: str = "textract_outputs",
) -> int:
    count = 0
    for path in glob.glob(os.path.join(textract_dir, "*_textract.json")):
        document_id = os.path.basename(path).replace("_textract.json", "")
        document_type = infer_document_type(document_id)
        try:
            payload = _load_json(path)
            blocks = payload.get("Blocks", [])
            line_confidences = [
                float(block.get("Confidence", 0.0))
                for block in blocks
                if block.get("BlockType") == "LINE" and block.get("Confidence") is not None
            ]
            avg_confidence = sum(line_confidences) / len(line_confidences) if line_confidences else 0.0
            # Use inverse confidence as coarse extraction error proxy for dashboard trend.
            extraction_error_proxy = max(0.0, min(100.0, 100.0 - avg_confidence))
            monitor.publish_extraction_result(
                document_id=document_id,
                success=True,
                latency_seconds=0.0,
                document_type=document_type,
            )
            monitor.put_metric(
                metric_name="ExtractionConfidenceAvg",
                value=avg_confidence,
                unit="Percent",
                dimensions=[
                    {"Name": "Stage", "Value": "Tier1Textract"},
                    {"Name": "DocumentType", "Value": document_type},
                ],
            )
            monitor.put_metric(
                metric_name="ExtractionErrorRate",
                value=extraction_error_proxy,
                unit="Percent",
                dimensions=[
                    {"Name": "Stage", "Value": "Tier1Textract"},
                    {"Name": "DocumentType", "Value": document_type},
                ],
            )
            count += 1
        except Exception:
            monitor.publish_extraction_result(
                document_id=document_id,
                success=False,
                latency_seconds=0.0,
                document_type=document_type,
            )
            count += 1
    return count


def publish_track_a_metrics(
    monitor: CloudWatchMonitoringManager,
    track_a_dir: str = "track_a_outputs",
) -> int:
    count = 0
    for path in glob.glob(os.path.join(track_a_dir, "*_snomed.json")):
        document_id = os.path.basename(path).replace("_snomed.json", "")
        document_type = infer_document_type(document_id)
        try:
            payload = _load_json(path)

            if "categorized_entities" in payload:
                entities = []
                for category_entities in payload.get("categorized_entities", {}).values():
                    entities.extend(category_entities)
                total_entities = len(entities)
                mapped_entities = len(
                    [entry for entry in entities if entry.get("snomed_code") not in {"", "NOT_MAPPED"}]
                )
                fallback_count = len(
                    [entry for entry in entities if entry.get("source") == "semantic_fallback"]
                )
                latency_seconds = float(payload.get("processing_time_seconds", 0.0))
            else:
                entities = payload.get("Entities", [])
                total_entities = len(entities)
                mapped_entities = len(
                    [entry for entry in entities if entry.get("SNOMEDCTConcepts")]
                )
                fallback_count = 0
                latency_seconds = 0.0

            monitor.publish_snomed_mapping_result(
                document_id=document_id,
                total_entities=total_entities,
                mapped_entities=mapped_entities,
                fallback_count=fallback_count,
                latency_seconds=latency_seconds,
                document_type=document_type,
            )
            count += 1
        except Exception:
            monitor.put_metric(
                metric_name="SNOMEDMappingSuccessRate",
                value=0.0,
                unit="Percent",
                dimensions=[
                    {"Name": "Stage", "Value": "TrackASNOMED"},
                    {"Name": "DocumentType", "Value": document_type},
                ],
            )
            count += 1
    return count


def publish_track_b_metrics(
    monitor: CloudWatchMonitoringManager,
    track_b_dir: str = "track_b_outputs",
) -> int:
    count = 0
    for path in glob.glob(os.path.join(track_b_dir, "*_summary.json")):
        document_id = os.path.basename(path).replace("_summary.json", "")
        document_type = infer_document_type(document_id)
        try:
            payload = _load_json(path)
            role = payload.get("role", "unknown")
            latency_ms = float(payload.get("generation_time_ms", 0.0))
            confidence_score = float(payload.get("confidence_score", 0.0))
            monitor.publish_llm_latency(
                document_id=document_id,
                role=role,
                latency_ms=latency_ms,
                confidence_score=confidence_score,
                document_type=document_type,
            )
            count += 1
        except Exception:
            monitor.publish_llm_latency(
                document_id=document_id,
                role="unknown",
                latency_ms=0.0,
                confidence_score=0.0,
                document_type=document_type,
            )
            count += 1
    return count


def publish_queue_depth_metrics(
    monitor: CloudWatchMonitoringManager, queue_names: List[str]
) -> Dict[str, int]:
    return monitor.publish_queue_depths(queue_names=queue_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish NLP-uk CloudWatch custom metrics")
    parser.add_argument("--textract-dir", default="textract_outputs")
    parser.add_argument("--track-a-dir", default="track_a_outputs")
    parser.add_argument("--track-b-dir", default="track_b_outputs")
    parser.add_argument(
        "--queue-names",
        default=",".join(DEFAULT_QUEUE_NAMES),
        help="Comma-separated SQS queue names",
    )
    args = parser.parse_args()

    queue_names = [name.strip() for name in args.queue_names.split(",") if name.strip()]
    monitor = CloudWatchMonitoringManager()

    summary = {
        "textract_documents": publish_textract_metrics(monitor, textract_dir=args.textract_dir),
        "track_a_documents": publish_track_a_metrics(monitor, track_a_dir=args.track_a_dir),
        "track_b_documents": publish_track_b_metrics(monitor, track_b_dir=args.track_b_dir),
        "queue_depths": publish_queue_depth_metrics(monitor, queue_names=queue_names),
        "cost_recommendations": monitor.cost_optimization_recommendations(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
