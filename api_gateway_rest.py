"""
API Gateway Lambda proxy handler for clinical document REST endpoints.
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from audit_dynamodb import get_audit_logger
from hipaa_compliance import create_secure_client, scrub_json_value

UPLOAD_BUCKET = os.getenv("API_UPLOAD_BUCKET", "")
API_STATE_DIR = os.getenv("API_STATE_DIR", "api_state")
ALLOWED_EXTENSIONS = {".pdf", ".tiff", ".tif", ".jpeg", ".jpg"}
DOC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]{3,128}$")
AUTH_API_KEYS = [item.strip() for item in os.getenv("API_ALLOWED_KEYS", "").split(",") if item.strip()]
AUTH_BEARER_TOKENS = [
    item.strip() for item in os.getenv("API_ALLOWED_BEARER_TOKENS", "").split(",") if item.strip()
]


def _cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": os.getenv("API_CORS_ALLOW_ORIGIN", "*"),
        "Access-Control-Allow-Headers": "Content-Type,Authorization,x-api-key",
        "Access-Control-Allow-Methods": "OPTIONS,GET,POST,PUT",
    }


def _response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {**_cors_headers(), "Content-Type": "application/json"},
        "body": json.dumps(scrub_json_value(body)),
    }


def _auth_ok(headers: Dict[str, str]) -> bool:
    headers_lower = {k.lower(): v for k, v in (headers or {}).items()}
    api_key = headers_lower.get("x-api-key", "").strip()
    auth_header = headers_lower.get("authorization", "").strip()
    token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""

    if AUTH_API_KEYS and api_key in AUTH_API_KEYS:
        return True
    if AUTH_BEARER_TOKENS and token in AUTH_BEARER_TOKENS:
        return True
    return False


def _parse_path(path: str) -> Tuple[str, Optional[str], Optional[str]]:
    cleaned = (path or "").strip()
    if cleaned == "/documents/upload":
        return "upload", None, None

    match = re.match(r"^/documents/([^/]+)/([^/]+)$", cleaned)
    if match:
        return "documents", match.group(1), match.group(2)

    match = re.match(r"^/audit/([^/]+)$", cleaned)
    if match:
        return "audit", match.group(1), None
    return "unknown", None, None


def _safe_doc_id(doc_id: str) -> bool:
    return bool(doc_id and DOC_ID_PATTERN.match(doc_id))


def _ensure_state_dir() -> None:
    os.makedirs(API_STATE_DIR, exist_ok=True)


def _doc_state_path(doc_id: str) -> str:
    _ensure_state_dir()
    return os.path.join(API_STATE_DIR, f"{doc_id}.json")


def _save_doc_state(doc_id: str, payload: Dict[str, Any]) -> None:
    path = _doc_state_path(doc_id)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_doc_state(doc_id: str) -> Dict[str, Any]:
    path = _doc_state_path(doc_id)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _find_json_by_suffix(directory: str, suffix: str, doc_id: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    for name in os.listdir(directory):
        if name.startswith(doc_id) and name.endswith(suffix):
            return os.path.join(directory, name)
    return None


def _find_summary_json(doc_id: str) -> Optional[str]:
    directory = "track_b_outputs"
    if not os.path.isdir(directory):
        return None
    for name in os.listdir(directory):
        if name.startswith(doc_id) and name.endswith("_summary.json"):
            return os.path.join(directory, name)
    return None


def _load_json_file(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _extract_textract_text(textract_payload: Dict[str, Any]) -> str:
    lines = []
    for block in textract_payload.get("Blocks", []):
        if block.get("BlockType") == "LINE" and block.get("Text"):
            lines.append(str(block.get("Text")))
    return "\n".join(lines)


def _upload_binary(doc_id: str, filename: str, content_type: str, payload_bytes: bytes) -> str:
    if not UPLOAD_BUCKET:
        raise RuntimeError("API_UPLOAD_BUCKET is not configured")
    key = f"uploads/{doc_id}/{filename}"
    s3_client = create_secure_client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    s3_client.put_object(
        Bucket=UPLOAD_BUCKET,
        Key=key,
        Body=payload_bytes,
        ContentType=content_type,
        ServerSideEncryption="AES256",
        Metadata={"doc_id": doc_id},
    )
    return f"s3://{UPLOAD_BUCKET}/{key}"


def _handle_upload(event: Dict[str, Any]) -> Dict[str, Any]:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    filename = headers.get("x-filename", f"document-{uuid.uuid4().hex[:8]}.pdf")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _response(
            400,
            {
                "error": "unsupported_file_type",
                "message": "Only PDF, TIFF, and JPEG files are accepted",
            },
        )

    body = event.get("body") or ""
    is_base64 = bool(event.get("isBase64Encoded"))
    if not body:
        return _response(400, {"error": "empty_body", "message": "Request body is required"})

    try:
        payload_bytes = base64.b64decode(body) if is_base64 else body.encode("utf-8")
    except Exception:
        return _response(400, {"error": "invalid_body", "message": "Invalid encoded body"})

    if not payload_bytes:
        return _response(400, {"error": "empty_payload", "message": "Uploaded file content is empty"})

    doc_id = f"doc-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    content_type = headers.get("content-type", "application/octet-stream")

    try:
        location = _upload_binary(doc_id, filename, content_type, payload_bytes)
    except Exception as exc:
        return _response(500, {"error": "upload_failed", "message": str(exc)})

    state = {
        "doc_id": doc_id,
        "filename": filename,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "status": "uploaded",
        "storage_uri": location,
    }
    _save_doc_state(doc_id, state)
    return _response(201, state)


def _handle_status(doc_id: str) -> Dict[str, Any]:
    state = _load_doc_state(doc_id)
    if not state:
        return _response(404, {"error": "not_found", "message": "Document not found"})
    return _response(200, state)


def _handle_extraction(doc_id: str) -> Dict[str, Any]:
    textract_path = _find_json_by_suffix("textract_outputs", "_textract.json", doc_id)
    payload = _load_json_file(textract_path)
    if not payload:
        return _response(404, {"error": "not_found", "message": "Extraction output not found"})
    return _response(
        200,
        {
            "doc_id": doc_id,
            "extracted_text": _extract_textract_text(payload),
            "phi_detection": payload.get("PhiDetection", {}),
        },
    )


def _handle_snomed(doc_id: str) -> Dict[str, Any]:
    snomed_path = _find_json_by_suffix("track_a_outputs", "_snomed.json", doc_id)
    payload = _load_json_file(snomed_path)
    if not payload:
        return _response(404, {"error": "not_found", "message": "SNOMED mapping not found"})
    return _response(
        200,
        {
            "doc_id": doc_id,
            "categorized_entities": payload.get("categorized_entities", {}),
            "unified_confidence_score": payload.get("unified_confidence_score", 0.0),
        },
    )


def _handle_summary(doc_id: str) -> Dict[str, Any]:
    summary_path = _find_summary_json(doc_id)
    payload = _load_json_file(summary_path)
    if not payload:
        return _response(404, {"error": "not_found", "message": "Summary output not found"})
    return _response(
        200,
        {
            "doc_id": doc_id,
            "role": payload.get("role"),
            "summary": payload.get("summary", ""),
            "key_points": payload.get("key_points", []),
            "follow_up_actions": payload.get("follow_up_actions", []),
            "confidence_score": payload.get("confidence_score", 0.0),
        },
    )


def _handle_approve(doc_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
    state = _load_doc_state(doc_id)
    if not state:
        state = {"doc_id": doc_id, "status": "pending_review"}
    before_state = dict(state)

    body = event.get("body") or "{}"
    try:
        payload = json.loads(body) if isinstance(body, str) else body
    except Exception:
        payload = {}

    approved_by = payload.get("approved_by", "api-user")
    state.update(
        {
            "status": "approved",
            "approved": True,
            "approved_by": approved_by,
            "approved_at": datetime.utcnow().isoformat() + "Z",
        }
    )
    _save_doc_state(doc_id, state)

    try:
        logger = get_audit_logger()
        logger.log_change(
            document_id=doc_id,
            user_id=approved_by,
            change_type="API_DOCUMENT_APPROVE",
            before_state=before_state,
            after_state=state,
            metadata={"source": "api_gateway", "endpoint": "PUT /documents/{doc_id}/approve"},
        )
    except Exception:
        pass

    return _response(200, {"doc_id": doc_id, "status": state["status"], "approved_at": state["approved_at"]})


def _handle_audit(doc_id: str) -> Dict[str, Any]:
    try:
        logger = get_audit_logger()
        trail = logger.get_audit_trail_by_document(doc_id, limit=200)
        return _response(200, {"doc_id": doc_id, "audit_entries": trail, "count": len(trail)})
    except Exception as exc:
        return _response(500, {"error": "audit_query_failed", "message": str(exc)})


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    _ = context
    method = str(event.get("httpMethod", "")).upper()
    path = event.get("path", "")
    headers = event.get("headers") or {}

    if method == "OPTIONS":
        return _response(200, {"ok": True})

    if not _auth_ok(headers):
        return _response(401, {"error": "unauthorized", "message": "Valid API key or OAuth bearer token is required"})

    route_group, doc_id, action = _parse_path(path)
    if route_group == "unknown":
        return _response(404, {"error": "not_found", "message": "Unknown endpoint"})

    if route_group == "upload":
        if method != "POST":
            return _response(405, {"error": "method_not_allowed"})
        return _handle_upload(event)

    if route_group == "audit":
        if method != "GET":
            return _response(405, {"error": "method_not_allowed"})
        if not _safe_doc_id(str(doc_id)):
            return _response(400, {"error": "invalid_doc_id"})
        return _handle_audit(str(doc_id))

    if not _safe_doc_id(str(doc_id)):
        return _response(400, {"error": "invalid_doc_id"})

    if method == "GET" and action == "status":
        return _handle_status(str(doc_id))
    if method == "GET" and action == "extraction":
        return _handle_extraction(str(doc_id))
    if method == "GET" and action == "snomed":
        return _handle_snomed(str(doc_id))
    if method == "GET" and action == "summary":
        return _handle_summary(str(doc_id))
    if method == "PUT" and action == "approve":
        return _handle_approve(str(doc_id), event)

    return _response(404, {"error": "not_found", "message": "Endpoint not implemented"})

