"""
EMIS export integration with retry queue and audit logging.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional

from hipaa_compliance import create_secure_client, scrub_json_value

EMIS_FORMAT = os.getenv("EMIS_EXPORT_FORMAT", "proprietary_json").strip().lower()
EMIS_TRANSPORT = os.getenv("EMIS_TRANSPORT", "api").strip().lower()  # api | s3_file_drop
EMIS_API_BASE_URL = os.getenv("EMIS_API_BASE_URL", "").strip()
EMIS_API_PATH = os.getenv("EMIS_API_PATH", "/documents/import").strip()
EMIS_API_TOKEN = os.getenv("EMIS_API_TOKEN", "").strip()
EMIS_FILE_DROP_BUCKET = os.getenv("EMIS_FILE_DROP_BUCKET", "").strip()
EMIS_RETRY_QUEUE_NAME = os.getenv("EMIS_RETRY_QUEUE_NAME", "EMIS_Export_Retry_Queue")
EMIS_MAX_ATTEMPTS = int(os.getenv("EMIS_MAX_EXPORT_ATTEMPTS", "3"))


def _build_export_payload(document_id: str, validated_payload: Dict[str, Any]) -> Dict[str, Any]:
    envelope = {
        "document_id": document_id,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "format": EMIS_FORMAT,
        "payload": validated_payload,
    }
    if EMIS_FORMAT == "hl7_fhir":
        envelope["resourceType"] = "Bundle"
        envelope["type"] = "collection"
    return envelope


def _log_event(
    audit_logger,
    document_id: str,
    user_id: str,
    change_type: str,
    status: str,
    metadata: Dict[str, Any],
) -> None:
    if not audit_logger:
        return
    audit_logger.log_change(
        document_id=document_id,
        user_id=user_id,
        change_type=change_type,
        before_state={"status": "pending"},
        after_state={"status": status},
        metadata=scrub_json_value(metadata),
    )


def _send_via_api(export_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not EMIS_API_BASE_URL.lower().startswith("https://"):
        raise RuntimeError("EMIS API base URL must use HTTPS/TLS (TLS 1.2+)")

    target = EMIS_API_BASE_URL.rstrip("/") + "/" + EMIS_API_PATH.lstrip("/")
    request = urllib.request.Request(
        target,
        data=json.dumps(export_payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            **({"Authorization": f"Bearer {EMIS_API_TOKEN}"} if EMIS_API_TOKEN else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310 - enforced https check
        raw = response.read() if response.readable() else b""
        body = raw.decode("utf-8") if raw else ""
        return {"status_code": int(getattr(response, "status", 200)), "response_body": body}


def _send_via_file_drop(export_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not EMIS_FILE_DROP_BUCKET:
        raise RuntimeError("EMIS_FILE_DROP_BUCKET is required for file transfer mode")
    s3_client = create_secure_client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    key = f"emis_exports/{export_payload['document_id']}_{int(time.time())}.json"
    s3_client.put_object(
        Bucket=EMIS_FILE_DROP_BUCKET,
        Key=key,
        Body=json.dumps(export_payload).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return {"status_code": 200, "response_body": f"s3://{EMIS_FILE_DROP_BUCKET}/{key}"}


def _send_to_emis(export_payload: Dict[str, Any]) -> Dict[str, Any]:
    if EMIS_TRANSPORT == "s3_file_drop":
        return _send_via_file_drop(export_payload)
    return _send_via_api(export_payload)


def _queue_for_retry(document_id: str, export_payload: Dict[str, Any], attempts: int, error: str) -> None:
    sqs = create_secure_client("sqs", region_name=os.getenv("AWS_REGION", "us-east-1"))
    try:
        queue_url = sqs.get_queue_url(QueueName=EMIS_RETRY_QUEUE_NAME)["QueueUrl"]
    except Exception:
        queue_url = sqs.create_queue(QueueName=EMIS_RETRY_QUEUE_NAME)["QueueUrl"]

    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(
            {
                "document_id": document_id,
                "attempts": attempts,
                "error": str(error),
                "export_payload": export_payload,
                "queued_at": datetime.utcnow().isoformat() + "Z",
            }
        ),
    )


def export_to_emis(
    document_id: str,
    validated_payload: Dict[str, Any],
    user_id: str,
    audit_logger=None,
) -> Dict[str, Any]:
    export_payload = _build_export_payload(document_id, validated_payload)
    last_error: Optional[str] = None
    for attempt in range(1, EMIS_MAX_ATTEMPTS + 1):
        try:
            response = _send_to_emis(export_payload)
            _log_event(
                audit_logger=audit_logger,
                document_id=document_id,
                user_id=user_id,
                change_type="EMIS_EXPORT_EVENT",
                status="SUCCESS",
                metadata={
                    "attempt": attempt,
                    "transport": EMIS_TRANSPORT,
                    "format": EMIS_FORMAT,
                    "emis_response": response,
                },
            )
            return {"success": True, "attempts": attempt, "emis_response": response}
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, OSError) as exc:
            last_error = str(exc)
            _log_event(
                audit_logger=audit_logger,
                document_id=document_id,
                user_id=user_id,
                change_type="EMIS_EXPORT_EVENT",
                status="FAILED_ATTEMPT",
                metadata={
                    "attempt": attempt,
                    "transport": EMIS_TRANSPORT,
                    "format": EMIS_FORMAT,
                    "error": last_error,
                },
            )
            if attempt < EMIS_MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 8))

    _queue_for_retry(
        document_id=document_id,
        export_payload=export_payload,
        attempts=EMIS_MAX_ATTEMPTS,
        error=last_error or "unknown_error",
    )
    _log_event(
        audit_logger=audit_logger,
        document_id=document_id,
        user_id=user_id,
        change_type="EMIS_EXPORT_EVENT",
        status="QUEUED_FOR_RETRY",
        metadata={
            "transport": EMIS_TRANSPORT,
            "format": EMIS_FORMAT,
            "retry_queue": EMIS_RETRY_QUEUE_NAME,
            "error": last_error,
        },
    )
    return {
        "success": False,
        "attempts": EMIS_MAX_ATTEMPTS,
        "error": last_error,
        "queued_for_retry": True,
        "retry_queue": EMIS_RETRY_QUEUE_NAME,
    }


def process_retry_message(message_body: Dict[str, Any], audit_logger=None, user_id: str = "SYSTEM") -> Dict[str, Any]:
    document_id = message_body.get("document_id", "unknown")
    export_payload = message_body.get("export_payload", {})
    prior_attempts = int(message_body.get("attempts", 0))
    try:
        response = _send_to_emis(export_payload)
        _log_event(
            audit_logger=audit_logger,
            document_id=document_id,
            user_id=user_id,
            change_type="EMIS_EXPORT_RETRY_EVENT",
            status="SUCCESS",
            metadata={"prior_attempts": prior_attempts, "emis_response": response},
        )
        return {"success": True, "emis_response": response}
    except Exception as exc:
        _log_event(
            audit_logger=audit_logger,
            document_id=document_id,
            user_id=user_id,
            change_type="EMIS_EXPORT_RETRY_EVENT",
            status="FAILED_UNRESOLVED",
            metadata={"prior_attempts": prior_attempts, "error": str(exc)},
        )
        return {"success": False, "error": str(exc), "unresolved": True}
