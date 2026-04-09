"""
Unified confidence score aggregation Lambda (Task 8).

Aggregates confidence from:
  1) Textract
  2) Comprehend Medical
  3) FAISS semantic retrieval
  4) LLM logprobs/confidence

Then routes documents:
  - final_score >= 0.85 -> high-confidence queue (bypass review)
  - final_score <  0.85 -> human-review queue
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Tuple

from audit_dynamodb import get_audit_logger
from hipaa_compliance import create_secure_client

DEFAULT_THRESHOLD = 0.85
DEFAULT_WEIGHTS = {
    "textract": 0.25,
    "comprehend": 0.25,
    "faiss": 0.25,
    "llm_logprobs": 0.25,
}

HIGH_CONFIDENCE_QUEUE = "Confidence_High_Bypass_Queue"
LOW_CONFIDENCE_QUEUE = "Confidence_Low_Review_Queue"


def _to_unit_interval(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default

    # Accept either 0-1 or 0-100 inputs.
    if f > 1.0:
        f = f / 100.0
    return max(0.0, min(1.0, f))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_textract_confidence_from_file(textract_json_path: str) -> float:
    data = _load_json(textract_json_path)
    confidences = [
        float(block.get("Confidence", 0.0))
        for block in data.get("Blocks", [])
        if block.get("BlockType") in {"LINE", "WORD"} and block.get("Confidence") is not None
    ]
    if not confidences:
        return 0.0
    return _to_unit_interval(sum(confidences) / len(confidences))


def _extract_track_a_scores(track_a_json_path: str) -> Tuple[float, float]:
    """
    Returns:
      comprehend_confidence, faiss_similarity_confidence
    """
    data = _load_json(track_a_json_path)
    categories = data.get("categorized_entities", {})

    comprehend_scores = []
    faiss_scores = []

    for bucket_entities in categories.values():
        for entity in bucket_entities:
            score = _to_unit_interval(entity.get("confidence", 0.0))
            source = entity.get("source", "")
            if source == "comprehend_medical":
                comprehend_scores.append(score)
            elif source == "semantic_fallback":
                faiss_scores.append(score)

    comprehend_conf = (
        sum(comprehend_scores) / len(comprehend_scores)
        if comprehend_scores else _to_unit_interval(data.get("unified_confidence_score", 0.0))
    )
    faiss_conf = sum(faiss_scores) / len(faiss_scores) if faiss_scores else 0.0
    return comprehend_conf, faiss_conf


def _extract_track_b_llm_confidence(track_b_json_path: str) -> float:
    data = _load_json(track_b_json_path)
    # Track B uses confidence_score as model confidence output.
    return _to_unit_interval(data.get("confidence_score", 0.0))


def collect_component_scores(event: Dict[str, Any]) -> Dict[str, float]:
    textract = _to_unit_interval(event.get("textract_confidence"))
    comprehend = _to_unit_interval(event.get("comprehend_confidence"))
    faiss = _to_unit_interval(event.get("faiss_similarity"))
    llm = _to_unit_interval(event.get("llm_logprobs_confidence", event.get("llm_confidence")))

    textract_path = event.get("textract_json_path")
    track_a_path = event.get("track_a_output_path")
    track_b_path = event.get("track_b_output_path")

    if textract == 0.0 and textract_path and os.path.exists(textract_path):
        textract = _extract_textract_confidence_from_file(textract_path)

    if track_a_path and os.path.exists(track_a_path):
        track_a_comprehend, track_a_faiss = _extract_track_a_scores(track_a_path)
        if comprehend == 0.0:
            comprehend = track_a_comprehend
        if faiss == 0.0:
            faiss = track_a_faiss

    if llm == 0.0 and track_b_path and os.path.exists(track_b_path):
        llm = _extract_track_b_llm_confidence(track_b_path)

    return {
        "textract": textract,
        "comprehend": comprehend,
        "faiss": faiss,
        "llm_logprobs": llm,
    }


def resolve_weights(event: Dict[str, Any]) -> Dict[str, float]:
    provided = event.get("weights", {})
    weights = {
        "textract": _to_unit_interval(provided.get("textract", os.getenv("CONF_WEIGHT_TEXTRACT", DEFAULT_WEIGHTS["textract"])), default=DEFAULT_WEIGHTS["textract"]),
        "comprehend": _to_unit_interval(provided.get("comprehend", os.getenv("CONF_WEIGHT_COMPREHEND", DEFAULT_WEIGHTS["comprehend"])), default=DEFAULT_WEIGHTS["comprehend"]),
        "faiss": _to_unit_interval(provided.get("faiss", os.getenv("CONF_WEIGHT_FAISS", DEFAULT_WEIGHTS["faiss"])), default=DEFAULT_WEIGHTS["faiss"]),
        "llm_logprobs": _to_unit_interval(provided.get("llm_logprobs", os.getenv("CONF_WEIGHT_LLM_LOGPROBS", DEFAULT_WEIGHTS["llm_logprobs"])), default=DEFAULT_WEIGHTS["llm_logprobs"]),
    }
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in weights.items()}


def calculate_weighted_score(component_scores: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, float]:
    calc_start = time.perf_counter()
    final_score = (
        (weights["textract"] * component_scores["textract"])
        + (weights["comprehend"] * component_scores["comprehend"])
        + (weights["faiss"] * component_scores["faiss"])
        + (weights["llm_logprobs"] * component_scores["llm_logprobs"])
    )
    latency_ms = (time.perf_counter() - calc_start) * 1000
    return round(final_score, 6), latency_ms


def route_document(
    document_id: str,
    final_score: float,
    threshold: float,
    component_scores: Dict[str, float],
    weights: Dict[str, float],
    calculation_latency_ms: float,
) -> Dict[str, Any]:
    route = "bypass_database" if final_score >= threshold else "human_review"
    queue_name = (
        os.getenv("HIGH_CONFIDENCE_QUEUE_NAME", HIGH_CONFIDENCE_QUEUE)
        if route == "bypass_database"
        else os.getenv("LOW_CONFIDENCE_QUEUE_NAME", LOW_CONFIDENCE_QUEUE)
    )

    sqs_client = create_secure_client("sqs", region_name=os.getenv("AWS_REGION", "us-east-1"))
    try:
        queue_url = sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]
    except Exception:
        queue_url = sqs_client.create_queue(QueueName=queue_name)["QueueUrl"]

    routing_payload = {
        "document_id": document_id,
        "final_confidence_score": final_score,
        "threshold": threshold,
        "route": route,
        "component_scores": component_scores,
        "weights": weights,
        "calculation_latency_ms": round(calculation_latency_ms, 4),
        "calculation_latency_sla_met": calculation_latency_ms < 100.0,
        "routed_at": datetime.utcnow().isoformat() + "Z",
    }
    sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(routing_payload))
    return {
        "route": route,
        "queue_name": queue_name,
        "queue_url": queue_url,
        "routing_payload": routing_payload,
    }


def log_routing_audit(
    document_id: str,
    route_result: Dict[str, Any],
):
    audit_logger = get_audit_logger()
    payload = route_result["routing_payload"]
    audit_logger.log_change(
        document_id=document_id,
        user_id="SYSTEM",
        change_type="CONFIDENCE_AGGREGATION_ROUTE",
        before_state={"status": "post_generation_pending_route"},
        after_state={"status": payload["route"]},
        metadata={
            "final_confidence_score": payload["final_confidence_score"],
            "threshold": payload["threshold"],
            "component_scores": payload["component_scores"],
            "weights": payload["weights"],
            "calculation_latency_ms": payload["calculation_latency_ms"],
            "calculation_latency_sla_met": payload["calculation_latency_sla_met"],
            "queue_name": route_result["queue_name"],
        },
    )


def lambda_handler(event, context):
    _ = context
    document_id = event.get("document_id", f"doc_{int(time.time())}")
    threshold = _to_unit_interval(event.get("threshold", os.getenv("FINAL_CONFIDENCE_THRESHOLD", DEFAULT_THRESHOLD)), default=DEFAULT_THRESHOLD)

    try:
        component_scores = collect_component_scores(event)
        weights = resolve_weights(event)
        final_score, calculation_latency_ms = calculate_weighted_score(component_scores, weights)

        route_result = route_document(
            document_id=document_id,
            final_score=final_score,
            threshold=threshold,
            component_scores=component_scores,
            weights=weights,
            calculation_latency_ms=calculation_latency_ms,
        )
        log_routing_audit(document_id, route_result)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "document_id": document_id,
                    "final_confidence_score": final_score,
                    "threshold": threshold,
                    "component_scores": component_scores,
                    "weights": weights,
                    "route": route_result["route"],
                    "queue_name": route_result["queue_name"],
                    "calculation_latency_ms": round(calculation_latency_ms, 4),
                    "calculation_latency_sla_met": calculation_latency_ms < 100.0,
                }
            ),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "document_id": document_id,
                    "error": str(exc),
                    "message": "Unified confidence aggregation failed",
                }
            ),
        }


if __name__ == "__main__":
    test_event = {
        "document_id": "demo_doc_001",
        "textract_confidence": 0.93,
        "comprehend_confidence": 0.89,
        "faiss_similarity": 0.81,
        "llm_logprobs_confidence": 0.88,
    }
    print(lambda_handler(test_event, None))
