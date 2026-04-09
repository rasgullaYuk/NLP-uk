"""
Utility helpers for the Streamlit clinician review interface.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from lambda_confidence_aggregator import (
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    calculate_weighted_score,
    collect_component_scores,
    resolve_weights,
)

SUMMARY_ROLES = ("clinician", "patient", "pharmacist")
DEFAULT_MEDIUM_THRESHOLD = 0.60
ACTION_PRIORITY_OPTIONS = ("High", "Medium", "Low")

_DOC_SUFFIXES = (
    "_clinician_summary",
    "_patient_summary",
    "_pharmacist_summary",
    "_summary",
    "_snomed",
    "_textract",
    "_confidence_aggregation",
    "_confidence",
)


def normalize_score(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def infer_document_id(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    stem_lower = stem.lower()
    for suffix in _DOC_SUFFIXES:
        if stem_lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def infer_summary_role_from_filename(file_name: str) -> str:
    lowered = os.path.basename(file_name).lower()
    if "_patient_summary" in lowered:
        return "patient"
    if "_pharmacist_summary" in lowered:
        return "pharmacist"
    return "clinician"


def _safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def discover_document_assets(
    summary_dir: str,
    snomed_dir: str,
    textract_dir: str,
    confidence_dir: str = "confidence_outputs",
) -> Dict[str, Dict[str, Any]]:
    assets: Dict[str, Dict[str, Any]] = {}

    def _ensure_doc(doc_id: str) -> Dict[str, Any]:
        return assets.setdefault(
            doc_id,
            {
                "summary_txt": {},
                "summary_json": {},
                "snomed_json": None,
                "textract_json": None,
                "confidence_json": None,
            },
        )

    search_dirs = [summary_dir, snomed_dir, textract_dir, confidence_dir]
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue

        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if not os.path.isfile(full_path):
                continue

            lowered = file_name.lower()
            doc_id = infer_document_id(file_name)
            doc_assets = _ensure_doc(doc_id)

            if lowered.endswith(".txt") and "_summary" in lowered:
                role = infer_summary_role_from_filename(file_name)
                doc_assets["summary_txt"][role] = full_path
            elif lowered.endswith(".json") and "_summary" in lowered:
                role = infer_summary_role_from_filename(file_name)
                doc_assets["summary_json"][role] = full_path
            elif lowered.endswith("_snomed.json"):
                doc_assets["snomed_json"] = full_path
            elif lowered.endswith("_textract.json"):
                doc_assets["textract_json"] = full_path
            elif lowered.endswith(".json") and "confidence" in lowered:
                doc_assets["confidence_json"] = full_path

    return dict(sorted(assets.items()))


def load_role_summary(doc_assets: Dict[str, Any], role: str) -> Dict[str, Any]:
    output = {
        "summary": "",
        "key_points": [],
        "medications": [],
        "diagnoses": [],
        "follow_up_actions": [],
        "confidence_score": 0.0,
        "path_txt": None,
        "path_json": None,
    }
    summary_json_paths = doc_assets.get("summary_json", {})
    summary_txt_paths = doc_assets.get("summary_txt", {})

    role_json_path = summary_json_paths.get(role)
    role_txt_path = summary_txt_paths.get(role)

    if role == "clinician":
        role_json_path = role_json_path or summary_json_paths.get("clinician")
        role_txt_path = role_txt_path or summary_txt_paths.get("clinician")

    if role_json_path and os.path.exists(role_json_path):
        try:
            payload = _safe_load_json(role_json_path)
            output["summary"] = payload.get("summary", output["summary"])
            output["key_points"] = payload.get("key_points", output["key_points"])
            output["medications"] = payload.get("medications", output["medications"])
            output["diagnoses"] = payload.get("diagnoses", output["diagnoses"])
            output["follow_up_actions"] = payload.get(
                "follow_up_actions", output["follow_up_actions"]
            )
            output["confidence_score"] = normalize_score(
                payload.get("confidence_score", output["confidence_score"])
            )
            output["path_json"] = role_json_path
        except (OSError, json.JSONDecodeError):
            pass

    if role_txt_path and os.path.exists(role_txt_path):
        try:
            txt_content = _safe_load_text(role_txt_path)
            if not output["summary"]:
                output["summary"] = txt_content.strip()
            output["path_txt"] = role_txt_path
        except OSError:
            pass

    return output


def load_all_role_summaries(doc_assets: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {role: load_role_summary(doc_assets, role) for role in SUMMARY_ROLES}


def extract_textract_text(textract_json_path: Optional[str]) -> str:
    if not textract_json_path or not os.path.exists(textract_json_path):
        return ""

    try:
        payload = _safe_load_json(textract_json_path)
    except (OSError, json.JSONDecodeError):
        try:
            return _safe_load_text(textract_json_path).strip()
        except OSError:
            return ""

    if isinstance(payload, dict):
        blocks = payload.get("Blocks", [])
        lines = [
            block.get("Text", "")
            for block in blocks
            if block.get("BlockType") == "LINE" and block.get("Text")
        ]
        if lines:
            return "\n".join(lines)
        if payload.get("summary"):
            return str(payload["summary"])
        return json.dumps(payload, indent=2)

    if isinstance(payload, list):
        return "\n".join(str(item) for item in payload if item)

    return str(payload)


def load_snomed_entities(snomed_json_path: Optional[str]) -> List[Dict[str, Any]]:
    if not snomed_json_path or not os.path.exists(snomed_json_path):
        return []

    try:
        payload = _safe_load_json(snomed_json_path)
    except (OSError, json.JSONDecodeError):
        return []

    entities: List[Dict[str, Any]] = []

    if "categorized_entities" in payload:
        categorized = payload.get("categorized_entities", {})
        for category, category_entities in categorized.items():
            for index, entity in enumerate(category_entities):
                entities.append(
                    {
                        "entity_id": f"{category}_{index}",
                        "text": entity.get("text", ""),
                        "category": category,
                        "snomed_code": entity.get("snomed_code", ""),
                        "description": entity.get("description", ""),
                        "confidence": normalize_score(entity.get("confidence", 0.0)),
                        "source": entity.get("source", ""),
                    }
                )
        return entities

    for index, entity in enumerate(payload.get("Entities", [])):
        concepts = entity.get("SNOMEDCTConcepts", [])
        top_concept = concepts[0] if concepts else {}
        entities.append(
            {
                "entity_id": f"entity_{index}",
                "text": entity.get("Text", ""),
                "category": entity.get("Category", "Uncategorized"),
                "snomed_code": top_concept.get("Code", ""),
                "description": top_concept.get("Description", ""),
                "confidence": normalize_score(
                    entity.get("confidence", entity.get("Score", top_concept.get("Score", 0.0)))
                ),
                "source": "comprehend_medical",
            }
        )
    return entities


def parse_actions_from_text(action_text: str) -> List[str]:
    actions: List[str] = []
    for raw_line in action_text.splitlines():
        cleaned = raw_line.strip().lstrip("-").strip()
        if cleaned:
            actions.append(cleaned)
    return actions


def format_actions_for_text(actions: List[Any]) -> str:
    normalized: List[str] = []
    for action in actions or []:
        if isinstance(action, dict):
            normalized.append(str(action.get("action", action.get("text", ""))).strip())
        else:
            normalized.append(str(action).strip())
    return "\n".join([line for line in normalized if line])


def _normalize_due_date(value: Any, default_due_date: date) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10]).isoformat()
        except ValueError:
            return default_due_date.isoformat()
    return default_due_date.isoformat()


def normalize_action_items(
    actions: List[Any],
    default_assignee: str = "",
    default_due_date: Optional[date] = None,
) -> List[Dict[str, str]]:
    due_date_default = default_due_date or (date.today() + timedelta(days=7))
    normalized_items: List[Dict[str, str]] = []

    for action in actions or []:
        if isinstance(action, dict):
            action_text = str(
                action.get("action_text")
                or action.get("action")
                or action.get("text")
                or action.get("instruction")
                or ""
            ).strip()
            priority = str(action.get("priority", "Medium")).strip().title()
            assignee = str(action.get("assignee", default_assignee)).strip()
            snomed_code = str(action.get("snomed_code", "")).strip()
            action_due_date = _normalize_due_date(action.get("due_date"), due_date_default)
        else:
            action_text = str(action).strip()
            priority = "Medium"
            assignee = default_assignee
            snomed_code = ""
            action_due_date = due_date_default.isoformat()

        if priority not in ACTION_PRIORITY_OPTIONS:
            priority = "Medium"

        if action_text:
            normalized_items.append(
                {
                    "action_text": action_text,
                    "due_date": action_due_date,
                    "priority": priority,
                    "assignee": assignee,
                    "snomed_code": snomed_code,
                }
            )

    return normalized_items


def serialize_action_items(
    actions: List[Dict[str, Any]],
    default_due_date: Optional[date] = None,
) -> List[Dict[str, str]]:
    due_date_default = default_due_date or (date.today() + timedelta(days=7))
    serialized: List[Dict[str, str]] = []

    for action in actions or []:
        action_text = str(action.get("action_text", "")).strip()
        assignee = str(action.get("assignee", "")).strip()
        snomed_code = str(action.get("snomed_code", "")).strip()
        if not action_text and not assignee and not snomed_code:
            continue

        priority = str(action.get("priority", "Medium")).strip().title()
        if priority not in ACTION_PRIORITY_OPTIONS:
            priority = "Medium"

        serialized.append(
            {
                "action_text": action_text,
                "due_date": _normalize_due_date(action.get("due_date"), due_date_default),
                "priority": priority,
                "assignee": assignee,
                "snomed_code": snomed_code,
            }
        )

    return serialized


def confidence_band(
    score: float,
    threshold: float = DEFAULT_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
) -> str:
    normalized_score = normalize_score(score)
    if normalized_score >= threshold:
        return "high"
    if normalized_score >= medium_threshold:
        return "medium"
    return "low"


def confidence_visual(
    score: float,
    threshold: float = DEFAULT_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
) -> Dict[str, str]:
    band = confidence_band(score, threshold=threshold, medium_threshold=medium_threshold)
    if band == "high":
        return {
            "band": "high",
            "icon": "✅",
            "label": "HIGH",
            "color": "#166534",
            "background": "#dcfce7",
        }
    if band == "medium":
        return {
            "band": "medium",
            "icon": "🟡",
            "label": "MEDIUM",
            "color": "#92400e",
            "background": "#fef9c3",
        }
    return {
        "band": "low",
        "icon": "🔴",
        "label": "LOW",
        "color": "#991b1b",
        "background": "#fee2e2",
    }


def recommendation_text(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    return (
        "Ready for automatic processing"
        if normalize_score(score) >= normalize_score(threshold)
        else "Manual clinician review required"
    )


def _extract_confidence_payload(confidence_json_path: Optional[str]) -> Dict[str, Any]:
    if not confidence_json_path or not os.path.exists(confidence_json_path):
        return {}
    try:
        payload = _safe_load_json(confidence_json_path)
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(payload, dict) and isinstance(payload.get("body"), str):
        try:
            body_payload = json.loads(payload["body"])
        except json.JSONDecodeError:
            body_payload = {}
        if isinstance(body_payload, dict):
            return body_payload
    return payload if isinstance(payload, dict) else {}


def compute_confidence_bundle(
    doc_assets: Dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    parsed_payload = _extract_confidence_payload(doc_assets.get("confidence_json"))

    parsed_threshold = normalize_score(parsed_payload.get("threshold"), default=threshold)
    if parsed_threshold > 0:
        threshold = parsed_threshold

    parsed_weights = parsed_payload.get("weights", {}) or {}
    normalized_weights = resolve_weights({"weights": parsed_weights}) if parsed_weights else dict(DEFAULT_WEIGHTS)

    component_scores = parsed_payload.get("component_scores") or {}
    if component_scores:
        component_scores = {
            "textract": normalize_score(component_scores.get("textract")),
            "comprehend": normalize_score(component_scores.get("comprehend")),
            "faiss": normalize_score(component_scores.get("faiss")),
            "llm_logprobs": normalize_score(component_scores.get("llm_logprobs")),
        }
    else:
        collect_event = {
            "textract_json_path": doc_assets.get("textract_json"),
            "track_a_output_path": doc_assets.get("snomed_json"),
            "track_b_output_path": doc_assets.get("summary_json", {}).get("clinician"),
        }
        component_scores = collect_component_scores(collect_event)

    raw_final_score = parsed_payload.get("final_confidence_score")
    if raw_final_score is None:
        raw_final_score = parsed_payload.get("unified_confidence_score")

    final_score = None
    if raw_final_score is not None:
        final_score = normalize_score(raw_final_score)
    calculation_latency_ms = parsed_payload.get("calculation_latency_ms", 0.0)
    calculation_latency_sla_met = parsed_payload.get("calculation_latency_sla_met", True)

    if final_score is None:
        final_score, calculation_latency_ms = calculate_weighted_score(
            component_scores, normalized_weights
        )
        calculation_latency_sla_met = calculation_latency_ms < 100.0

    route = parsed_payload.get("route")
    if not route:
        route = "bypass_database" if final_score >= threshold else "human_review"

    return {
        "unified_confidence_score": normalize_score(final_score, default=0.0),
        "threshold": normalize_score(threshold, default=DEFAULT_THRESHOLD),
        "component_scores": {
            "textract": normalize_score(component_scores.get("textract")),
            "comprehend": normalize_score(component_scores.get("comprehend")),
            "faiss": normalize_score(component_scores.get("faiss")),
            "llm_logprobs": normalize_score(component_scores.get("llm_logprobs")),
        },
        "weights": normalized_weights,
        "route": route,
        "recommendation": recommendation_text(
            normalize_score(final_score, default=0.0), threshold=threshold
        ),
        "calculation_latency_ms": float(calculation_latency_ms),
        "calculation_latency_sla_met": bool(calculation_latency_sla_met),
    }
