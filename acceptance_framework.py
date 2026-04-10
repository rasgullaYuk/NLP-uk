"""
Acceptance test helpers for validating SRS criteria.
"""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List

from hipaa_compliance import scrub_text_for_logs
from lambda_confidence_aggregator import (
    DEFAULT_THRESHOLD,
    calculate_weighted_score,
    resolve_weights,
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def text_similarity_ratio(extracted_text: str, ground_truth_text: str) -> float:
    return SequenceMatcher(
        None,
        _normalize_text(extracted_text),
        _normalize_text(ground_truth_text),
    ).ratio()


def evaluate_text_accuracy(cases: List[Dict[str, str]]) -> Dict[str, Any]:
    if not cases:
        return {"accuracy_percent": 0.0, "passed": False, "threshold": 98.0, "case_scores": []}

    case_scores = []
    for case in cases:
        ratio = text_similarity_ratio(case.get("extracted", ""), case.get("ground_truth", ""))
        case_scores.append(
            {
                "document_id": case.get("document_id", "unknown"),
                "score_percent": round(ratio * 100, 3),
            }
        )

    overall = sum(item["score_percent"] for item in case_scores) / len(case_scores)
    threshold = 98.0
    return {
        "accuracy_percent": round(overall, 3),
        "passed": overall >= threshold,
        "threshold": threshold,
        "case_scores": case_scores,
    }


def evaluate_snomed_mapping_accuracy(cases: List[Dict[str, str]]) -> Dict[str, Any]:
    if not cases:
        return {"accuracy_percent": 0.0, "passed": False, "threshold": 95.0, "matched": 0, "total": 0}

    matched = 0
    for case in cases:
        if str(case.get("predicted_code", "")).strip() == str(case.get("expected_code", "")).strip():
            matched += 1

    accuracy = (matched / len(cases)) * 100.0
    threshold = 95.0
    return {
        "accuracy_percent": round(accuracy, 3),
        "passed": accuracy >= threshold,
        "threshold": threshold,
        "matched": matched,
        "total": len(cases),
    }


def evaluate_confidence_routing(
    events: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    results = []
    for event in events:
        weights = resolve_weights(event)
        score, _ = calculate_weighted_score(
            {
                "textract": float(event["textract"]),
                "comprehend": float(event["comprehend"]),
                "faiss": float(event["faiss"]),
                "llm_logprobs": float(event["llm_logprobs"]),
            },
            weights,
        )
        expected = event["expected_route"]
        actual = "bypass_database" if score >= threshold else "human_review"
        results.append(
            {
                "document_id": event.get("document_id", "unknown"),
                "score": round(score, 6),
                "expected_route": expected,
                "actual_route": actual,
                "passed": expected == actual,
            }
        )

    passed = all(item["passed"] for item in results) if results else False
    return {"threshold": threshold, "passed": passed, "results": results}


def benchmark_runtime(workload: Callable[[], Any], max_seconds: float = 60.0) -> Dict[str, Any]:
    started = time.perf_counter()
    workload()
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": round(elapsed, 3),
        "max_seconds": float(max_seconds),
        "passed": elapsed < max_seconds,
    }


def verify_phi_masking(sample_text: str) -> Dict[str, Any]:
    masked = scrub_text_for_logs(sample_text)
    markers = ["REDACTED", "PHI_"]
    passed = any(marker in masked for marker in markers) and masked != sample_text
    return {"masked_text": masked, "passed": passed}


def verify_encryption_posture(s3_result: Dict[str, Any], dynamodb_result: Dict[str, Any]) -> Dict[str, Any]:
    s3_ok = bool(s3_result.get("encrypted"))
    ddb_ok = bool(dynamodb_result.get("encrypted"))
    return {"s3_encrypted": s3_ok, "dynamodb_encrypted": ddb_ok, "passed": s3_ok and ddb_ok}


def verify_ui_editability_contract(app_source_path: str) -> Dict[str, Any]:
    with open(app_source_path, "r", encoding="utf-8") as handle:
        source = handle.read()

    required_tokens = [
        "_summary_edit",
        "_actions_edit",
        "_code",
        "_description",
        "_action_text_",
        "_action_assignee_",
    ]
    found = {token: (token in source) for token in required_tokens}
    all_present = all(found.values())
    return {"all_present": all_present, "tokens": found, "passed": all_present}


def save_acceptance_report(path: str, report_payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)


def load_acceptance_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
