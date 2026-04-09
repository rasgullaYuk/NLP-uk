"""
test_dynamodb.py — Integration Tests for DynamoDB Tables
=========================================================
Tests real DynamoDB tables (requires live AWS credentials + ACTIVE tables).

Run AFTER create_tables.py has completed successfully:
    python test_dynamodb.py

What is tested per table:
  1. Insert a sample item
  2. Read it back by primary key (point-read)
  3. Query via each GSI with the expected key
  4. Delete the test item (clean-up)

All test items use IDs prefixed with "TEST_" so they are easy to identify
and clean up if a test run is interrupted.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
import logging
from decimal import Decimal
from datetime import datetime, timezone

from botocore.exceptions import ClientError
import boto3
try:
    from hipaa_compliance import create_secure_client, create_secure_resource
except ImportError:
    def create_secure_client(service_name: str, region_name: str, **kwargs):
        return boto3.client(service_name, region_name=region_name, **kwargs)

    def create_secure_resource(service_name: str, region_name: str, **kwargs):
        return boto3.resource(service_name, region_name=region_name, **kwargs)

from config import AWS_REGION, TABLE_NAMES, TTL_ATTRIBUTE
from ttl_config import compute_ttl_expiry

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# ── Shared client and resource ─────────────────────────────────────────────────
_CLIENT   = create_secure_client("dynamodb", region_name=AWS_REGION)
_RESOURCE = create_secure_resource("dynamodb", region_name=AWS_REGION)

# ── Utility ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _test_id(prefix: str) -> str:
    """Generate a unique test ID with a TEST_ prefix."""
    return f"TEST_{prefix}_{uuid.uuid4().hex[:8]}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Documents Table
# ═══════════════════════════════════════════════════════════════════════════════

def test_documents_table() -> bool:
    """
    1. Insert a document item.
    2. Read it back by document_id (PK).
    3. Query upload_date-index GSI.
    4. Delete the item.
    """
    table = _RESOURCE.Table(TABLE_NAMES["documents"])
    doc_id     = _test_id("DOC")
    upload_date = _now_iso()

    logger.info("[Documents] Testing insert...")
    try:
        table.put_item(Item={
            "document_id":    doc_id,
            "upload_date":    upload_date,
            "status":         "TIER3_COMPLETE",
            "source_file":    "test/sample_clinical_doc.pdf",
            "extracted_text": "Patient: John Doe. Diagnosis: hypertension.",
        })
    except ClientError as exc:
        logger.error("[Documents] put_item failed: %s", exc)
        return False

    # Point-read by PK
    logger.info("[Documents] Reading back by document_id...")
    response = table.get_item(Key={"document_id": doc_id})
    item = response.get("Item")
    if not item or item["document_id"] != doc_id:
        logger.error("[Documents] get_item returned unexpected item: %s", item)
        return False
    logger.info("[Documents] Point-read OK: status=%s", item["status"])

    # GSI query: upload_date-index
    logger.info("[Documents] Querying upload_date-index GSI...")
    gsi_response = table.query(
        IndexName="upload_date-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("upload_date").eq(upload_date),
    )
    if gsi_response.get("Count", 0) < 1:
        logger.error("[Documents] GSI query returned 0 items.")
        return False
    logger.info("[Documents] GSI query OK: %d item(s) found.", gsi_response["Count"])

    # Clean up
    table.delete_item(Key={"document_id": doc_id})
    logger.info("[Documents] Test item deleted. PASS")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Audit_Logs Table
# ═══════════════════════════════════════════════════════════════════════════════

def test_audit_logs_table() -> bool:
    """
    1. Insert two audit log items for the same document.
    2. Read one back by log_id (PK).
    3. Query document_id-timestamp-index GSI (both items, sorted by time).
    4. Query user_id-index GSI.
    5. Delete both items.
    """
    table = _RESOURCE.Table(TABLE_NAMES["audit_logs"])
    doc_id  = _test_id("DOC")
    user_id = _test_id("USR")

    items_to_delete = []

    for i, action in enumerate(["OCR_CORRECTION_ACCEPTED", "MANUAL_REVIEW_SUBMITTED"]):
        log_id    = _test_id(f"LOG{i}")
        timestamp = _now_iso()
        time.sleep(0.01)  # ensure distinct timestamps for sort-key ordering

        logger.info("[AuditLogs] Inserting log: %s...", action)
        try:
            table.put_item(Item={
                "log_id":       log_id,
                "document_id":  doc_id,
                "user_id":      user_id,
                "timestamp":    timestamp,
                "action":       action,
                "before_state": json.dumps({"text": "Ibuprofn 200mg"}),
                "after_state":  json.dumps({"text": "Ibuprofen 200mg", "reason": "ACCEPTED"}),
            })
        except ClientError as exc:
            logger.error("[AuditLogs] put_item failed: %s", exc)
            return False

        # Point-read
        response = table.get_item(Key={"log_id": log_id})
        item = response.get("Item")
        if not item or item["log_id"] != log_id:
            logger.error("[AuditLogs] get_item mismatch for log_id=%s", log_id)
            return False

        items_to_delete.append(log_id)

    # GSI 1: document_id-timestamp-index (both items, sorted by timestamp)
    logger.info("[AuditLogs] Querying document_id-timestamp-index GSI...")
    gsi1 = table.query(
        IndexName="document_id-timestamp-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("document_id").eq(doc_id),
    )
    if gsi1.get("Count", 0) < 2:
        logger.error(
            "[AuditLogs] document_id-timestamp-index returned %d items (expected ≥2).",
            gsi1.get("Count", 0),
        )
        return False
    logger.info("[AuditLogs] GSI-1 query OK: %d item(s) sorted by timestamp.", gsi1["Count"])

    # GSI 2: user_id-index
    logger.info("[AuditLogs] Querying user_id-index GSI...")
    gsi2 = table.query(
        IndexName="user_id-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id),
    )
    if gsi2.get("Count", 0) < 1:
        logger.error("[AuditLogs] user_id-index returned 0 items.")
        return False
    logger.info("[AuditLogs] GSI-2 query OK: %d item(s) for user.", gsi2["Count"])

    # Clean up
    for log_id in items_to_delete:
        table.delete_item(Key={"log_id": log_id})
    logger.info("[AuditLogs] Test items deleted. PASS")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Test: User_Sessions Table
# ═══════════════════════════════════════════════════════════════════════════════

def test_user_sessions_table() -> bool:
    """
    1. Insert a session item with ttl_expiry set.
    2. Read it back by session_id.
    3. Query user_id-index GSI.
    4. Verify ttl_expiry is a future epoch integer.
    5. Delete the item.
    """
    table      = _RESOURCE.Table(TABLE_NAMES["user_sessions"])
    session_id = _test_id("SES")
    user_id    = _test_id("USR")

    login_ts  = _now_iso()
    logout_epoch = int(time.time())  # simulated logout = now
    ttl_expiry   = compute_ttl_expiry(logout_epoch)

    logger.info("[UserSessions] Inserting session...")
    try:
        table.put_item(Item={
            "session_id":        session_id,
            "user_id":           user_id,
            "login_timestamp":   login_ts,
            "logout_timestamp":  _now_iso(),
            "actions_count":     Decimal("7"),
            TTL_ATTRIBUTE:       Decimal(str(ttl_expiry)),  # must be Number for DynamoDB TTL
        })
    except ClientError as exc:
        logger.error("[UserSessions] put_item failed: %s", exc)
        return False

    # Point-read
    response = table.get_item(Key={"session_id": session_id})
    item = response.get("Item")
    if not item:
        logger.error("[UserSessions] get_item returned nothing.")
        return False

    stored_ttl = int(item.get(TTL_ATTRIBUTE, 0))
    if stored_ttl <= int(time.time()):
        logger.error("[UserSessions] ttl_expiry %d is not in the future.", stored_ttl)
        return False
    logger.info("[UserSessions] ttl_expiry OK: %d (future epoch).", stored_ttl)

    # GSI: user_id-index
    logger.info("[UserSessions] Querying user_id-index GSI...")
    gsi = table.query(
        IndexName="user_id-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id),
    )
    if gsi.get("Count", 0) < 1:
        logger.error("[UserSessions] user_id-index returned 0 items.")
        return False
    logger.info("[UserSessions] GSI query OK: %d session(s) found.", gsi["Count"])

    # Clean up
    table.delete_item(Key={"session_id": session_id})
    logger.info("[UserSessions] Test item deleted. PASS")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Feedback_Loop Table
# ═══════════════════════════════════════════════════════════════════════════════

def test_feedback_loop_table() -> bool:
    """
    1. Insert a feedback item with user correction data.
    2. Read it back by feedback_id.
    3. Query document_id-index GSI.
    4. Delete the item.
    """
    table       = _RESOURCE.Table(TABLE_NAMES["feedback_loop"])
    feedback_id = _test_id("FB")
    doc_id      = _test_id("DOC")

    corrections = [
        {
            "region_id":  "region_0",
            "original":   "Paracetmol 500mg",
            "corrected":  "Paracetamol 500mg",
            "accepted":   True,
            "reason":     "ACCEPTED",
        }
    ]

    logger.info("[FeedbackLoop] Inserting feedback item...")
    try:
        table.put_item(Item={
            "feedback_id":      feedback_id,
            "document_id":      doc_id,
            "user_corrections": json.dumps(corrections),
            "model_version":    "anthropic.claude-3-haiku-20240307-v1:0",
            "created_date":     _now_iso(),
        })
    except ClientError as exc:
        logger.error("[FeedbackLoop] put_item failed: %s", exc)
        return False

    # Point-read
    response = table.get_item(Key={"feedback_id": feedback_id})
    item = response.get("Item")
    if not item or item["feedback_id"] != feedback_id:
        logger.error("[FeedbackLoop] get_item mismatch.")
        return False
    logger.info("[FeedbackLoop] Point-read OK: model_version=%s", item["model_version"])

    # GSI: document_id-index
    logger.info("[FeedbackLoop] Querying document_id-index GSI...")
    gsi = table.query(
        IndexName="document_id-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("document_id").eq(doc_id),
    )
    if gsi.get("Count", 0) < 1:
        logger.error("[FeedbackLoop] document_id-index returned 0 items.")
        return False
    logger.info("[FeedbackLoop] GSI query OK: %d item(s) found.", gsi["Count"])

    # Clean up
    table.delete_item(Key={"feedback_id": feedback_id})
    logger.info("[FeedbackLoop] Test item deleted. PASS")
    return True


# ── Run all tests ──────────────────────────────────────────────────────────────

def run_all_tests() -> bool:
    """
    Execute all four table tests and report results.

    Returns:
        True if all tests pass, False if any fail.
    """
    tests = [
        ("Documents",     test_documents_table),
        ("Audit_Logs",    test_audit_logs_table),
        ("User_Sessions", test_user_sessions_table),
        ("Feedback_Loop", test_feedback_loop_table),
    ]

    results: dict[str, bool] = {}

    logger.info("=== DynamoDB Integration Tests — start ===")

    for name, fn in tests:
        logger.info("--- %s ---", name)
        try:
            results[name] = fn()
        except Exception as exc:
            logger.error("[%s] Unexpected exception: %s", name, exc, exc_info=True)
            results[name] = False

    logger.info("=== Results ===")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-20s %s", name, status)
        if not passed:
            all_passed = False

    logger.info(
        "=== DynamoDB Integration Tests — %s ===",
        "ALL PASSED" if all_passed else "SOME FAILED",
    )
    return all_passed


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
