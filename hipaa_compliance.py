"""
Shared PHI + HIPAA compliance utilities for the clinical NLP pipeline.
"""

from __future__ import annotations

import hashlib
import re
import ssl
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

DEFAULT_REGION = "us-east-1"
COMPREHEND_MEDICAL_MAX_CHARS = 18000


def _secure_config() -> Config:
    return Config(
        retries={"mode": "standard", "max_attempts": 5},
        connect_timeout=10,
        read_timeout=60,
    )


def assert_runtime_tls12() -> None:
    if hasattr(ssl, "HAS_TLSv1_2") and not ssl.HAS_TLSv1_2:
        raise RuntimeError("Python/OpenSSL runtime does not support TLS 1.2+")


def _assert_https_endpoint(endpoint_url: Optional[str], service_name: str) -> None:
    if endpoint_url and not endpoint_url.lower().startswith("https://"):
        raise RuntimeError(f"{service_name} endpoint must use HTTPS/TLS: {endpoint_url}")


def create_secure_client(service_name: str, region_name: str = DEFAULT_REGION, **kwargs: Any):
    assert_runtime_tls12()
    _assert_https_endpoint(kwargs.get("endpoint_url"), service_name)
    client = boto3.client(service_name, region_name=region_name, config=_secure_config(), **kwargs)
    endpoint_url = getattr(client.meta, "endpoint_url", "")
    _assert_https_endpoint(endpoint_url, service_name)
    return client


def create_secure_resource(service_name: str, region_name: str = DEFAULT_REGION, **kwargs: Any):
    assert_runtime_tls12()
    _assert_https_endpoint(kwargs.get("endpoint_url"), service_name)
    resource = boto3.resource(service_name, region_name=region_name, config=_secure_config(), **kwargs)
    endpoint_url = getattr(resource.meta.client.meta, "endpoint_url", "")
    _assert_https_endpoint(endpoint_url, service_name)
    return resource


def _chunk_text(text: str, chunk_size: int = COMPREHEND_MEDICAL_MAX_CHARS) -> List[tuple[int, str]]:
    chunks: List[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((start, text[start:end]))
        start = end
    return chunks


def detect_phi_entities(text: str, comprehend_medical_client=None) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    client = comprehend_medical_client or create_secure_client("comprehendmedical")
    entities: List[Dict[str, Any]] = []

    for base_offset, chunk in _chunk_text(text):
        if not chunk.strip():
            continue
        response = client.detect_phi(Text=chunk)
        for entity in response.get("Entities", []):
            begin = int(entity.get("BeginOffset", 0)) + base_offset
            end = int(entity.get("EndOffset", 0)) + base_offset
            raw_text = text[begin:end] if 0 <= begin < end <= len(text) else entity.get("Text", "")
            entities.append(
                {
                    "type": entity.get("Type", "UNKNOWN"),
                    "category": entity.get("Category", "PROTECTED_HEALTH_INFORMATION"),
                    "score": round(float(entity.get("Score", 0.0)), 4),
                    "begin_offset": begin,
                    "end_offset": end,
                    "text": raw_text,
                }
            )
    return entities


def pseudonymize_value(value: str, prefix: str = "PHI") -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"[{prefix}_{digest}]"


def mask_text_by_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    if not text:
        return text
    if not entities:
        return mask_text_with_patterns(text)

    masked = text
    valid = [
        e for e in entities
        if isinstance(e.get("begin_offset"), int)
        and isinstance(e.get("end_offset"), int)
        and 0 <= e["begin_offset"] < e["end_offset"] <= len(text)
    ]
    for entity in sorted(valid, key=lambda x: x["begin_offset"], reverse=True):
        start = entity["begin_offset"]
        end = entity["end_offset"]
        entity_type = str(entity.get("type", "PHI")).upper()
        original = text[start:end]
        replacement = pseudonymize_value(original, prefix=entity_type)
        masked = masked[:start] + replacement + masked[end:]

    return mask_text_with_patterns(masked)


def mask_text_with_patterns(text: str) -> str:
    if not text:
        return text
    patterns = [
        (r"\b(MRN[:\s-]*\d{4,12})\b", "[MRN_REDACTED]"),
        (r"\b(?:DOB|Date of Birth)[:\s-]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DOB_REDACTED]"),
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE_REDACTED]"),
        (r"\b(?:Mr|Mrs|Ms|Miss|Dr|Patient)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", "[NAME_REDACTED]"),
        (r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\s(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Boulevard|Blvd)\b", "[ADDRESS_REDACTED]"),
    ]
    masked = text
    for pattern, token in patterns:
        masked = re.sub(pattern, token, masked, flags=re.IGNORECASE)
    return masked


def scrub_text_for_logs(text: str, phi_entities: Optional[List[Dict[str, Any]]] = None) -> str:
    if phi_entities:
        return mask_text_by_entities(text, phi_entities)
    return mask_text_with_patterns(text)


def scrub_json_value(value: Any, phi_entities: Optional[List[Dict[str, Any]]] = None) -> Any:
    if isinstance(value, str):
        return scrub_text_for_logs(value, phi_entities)
    if isinstance(value, list):
        return [scrub_json_value(v, phi_entities) for v in value]
    if isinstance(value, dict):
        return {k: scrub_json_value(v, phi_entities) for k, v in value.items()}
    return value


def sanitize_phi_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe_entities = []
    for entity in entities:
        text_value = entity.get("text", "")
        safe_entities.append(
            {
                "type": entity.get("type"),
                "category": entity.get("category"),
                "score": entity.get("score"),
                "begin_offset": entity.get("begin_offset"),
                "end_offset": entity.get("end_offset"),
                "pseudonym": pseudonymize_value(text_value, prefix=str(entity.get("type", "PHI")).upper()),
            }
        )
    return safe_entities


def build_phi_detection_summary(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for entity in entities:
        key = str(entity.get("type", "UNKNOWN"))
        counts[key] = counts.get(key, 0) + 1
    return {
        "phi_detected": bool(entities),
        "entity_count": len(entities),
        "counts_by_type": counts,
        "entities": sanitize_phi_entities(entities),
        "policy": "mask/pseudonymize in logs, intermediate storage, and non-clinician outputs",
    }


def verify_s3_encryption(bucket_name: str, s3_client=None) -> Dict[str, Any]:
    client = s3_client or create_secure_client("s3")
    result = {"bucket": bucket_name, "encrypted": False, "algorithm": None}
    response = client.get_bucket_encryption(Bucket=bucket_name)
    rules = response.get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
    if rules:
        rule = rules[0].get("ApplyServerSideEncryptionByDefault", {})
        result["encrypted"] = True
        result["algorithm"] = rule.get("SSEAlgorithm")
    return result


def verify_dynamodb_sse(table_name: str, dynamodb_client=None) -> Dict[str, Any]:
    client = dynamodb_client or create_secure_client("dynamodb")
    response = client.describe_table(TableName=table_name)
    sse = response.get("Table", {}).get("SSEDescription", {})
    return {
        "table": table_name,
        "encrypted": sse.get("Status") in {"ENABLED", "ENABLING", "UPDATING"},
        "status": sse.get("Status"),
        "kms_master_key_arn": sse.get("KMSMasterKeyArn"),
    }
