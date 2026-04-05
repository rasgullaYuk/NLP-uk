"""
table_definitions.py — DynamoDB Table Schema Definitions
=========================================================
Each table is defined as a dict that can be passed directly to
boto3's dynamodb.create_table(**definition).

Schema decisions
----------------
* ONLY key attributes appear in AttributeDefinitions — DynamoDB is schemaless
  for non-key fields; adding them here would cause an error.
* All GSIs use KEYS_ONLY projection to minimise storage cost while still
  supporting fast index lookups.  Callers that need full item data can use
  the GSI key to perform a cheap point-read on the base table.
* PAY_PER_REQUEST billing removes the need to pre-provision read/write units.
* SSESpecification Enabled=True turns on AWS-managed encryption at rest
  (AES-256 via AWS KMS), as required for healthcare data.
"""

from config import BILLING_MODE, TABLE_NAMES

# ── Shared SSE (encryption at rest) ───────────────────────────────────────────
_SSE = {"Enabled": True}

# ── 1. Documents Table ─────────────────────────────────────────────────────────
#
# Stores the raw output of the Textract / OCR pipeline per document.
#
#  PK: document_id  (UUID string)
#
#  Non-key attributes (stored in item, not declared here):
#    upload_date   – ISO 8601 timestamp  e.g. "2026-04-05T07:00:00Z"
#    status        – pipeline stage      e.g. "TIER3_COMPLETE"
#    source_file   – S3 key or filename
#    extracted_text – raw OCR text (compressed / chunked if > 400 KB)
#
#  GSI: upload_date-index
#    Allows efficient time-range queries (e.g. "all docs uploaded today").
#    upload_date is the PK so the index covers the full date space.
DOCUMENTS_TABLE = {
    "TableName": TABLE_NAMES["documents"],
    "BillingMode": BILLING_MODE,
    "SSESpecification": _SSE,
    "AttributeDefinitions": [
        {"AttributeName": "document_id", "AttributeType": "S"},
        {"AttributeName": "upload_date", "AttributeType": "S"},
    ],
    "KeySchema": [
        {"AttributeName": "document_id", "KeyType": "HASH"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "upload_date-index",
            "KeySchema": [
                {"AttributeName": "upload_date", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "KEYS_ONLY"},
        },
    ],
}

# ── 2. Audit_Logs Table ────────────────────────────────────────────────────────
#
# Immutable audit trail — never update, only append.  Written by Tier 3
# after every correction decision (ACCEPTED / REVIEW_REQUIRED / NO_CHANGE).
#
#  PK: log_id  (UUID — globally unique per event)
#
#  Non-key attributes:
#    document_id   – FK to Documents table
#    user_id       – who triggered the action (or "SYSTEM" for pipeline ops)
#    timestamp     – ISO 8601 UTC
#    action        – e.g. "OCR_CORRECTION_ACCEPTED", "MANUAL_REVIEW_SUBMITTED"
#    before_state  – serialised JSON of original span
#    after_state   – serialised JSON of corrected span
#
#  GSI 1: document_id-timestamp-index
#    Supports "give me the full audit history for document X, ordered by time".
#    document_id = PK, timestamp = SK → DynamoDB returns results sorted by time.
#
#  GSI 2: user_id-index
#    Supports "show all actions taken by user Y" (clinician review dashboards).
AUDIT_LOGS_TABLE = {
    "TableName": TABLE_NAMES["audit_logs"],
    "BillingMode": BILLING_MODE,
    "SSESpecification": _SSE,
    "AttributeDefinitions": [
        {"AttributeName": "log_id",      "AttributeType": "S"},
        {"AttributeName": "document_id", "AttributeType": "S"},
        {"AttributeName": "timestamp",   "AttributeType": "S"},
        {"AttributeName": "user_id",     "AttributeType": "S"},
    ],
    "KeySchema": [
        {"AttributeName": "log_id", "KeyType": "HASH"},
    ],
    "GlobalSecondaryIndexes": [
        {
            # Fast audit-trail retrieval per document with chronological ordering
            "IndexName": "document_id-timestamp-index",
            "KeySchema": [
                {"AttributeName": "document_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp",   "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "KEYS_ONLY"},
        },
        {
            # Clinician / admin activity lookup
            "IndexName": "user_id-index",
            "KeySchema": [
                {"AttributeName": "user_id", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "KEYS_ONLY"},
        },
    ],
}

# ── 3. User_Sessions Table ─────────────────────────────────────────────────────
#
# Tracks active and recent clinician sessions.
# TTL (ttl_expiry) auto-deletes stale records 30 days after logout —
# DynamoDB evaluates TTL in the background, so expired items may persist
# briefly; callers should still check logout_timestamp for hard guarantees.
#
#  PK: session_id  (UUID)
#
#  Non-key attributes:
#    user_id           – FK to identity provider
#    login_timestamp   – ISO 8601
#    logout_timestamp  – ISO 8601 (populated on logout / session close)
#    actions_count     – counter of audit actions taken in this session
#    ttl_expiry        – Unix epoch int, set to logout_timestamp + 30 days
#                        (populated on logout; TTL is enabled in ttl_config.py)
#
#  GSI: user_id-index
#    Supports "show all sessions for user Y" without a full table scan.
USER_SESSIONS_TABLE = {
    "TableName": TABLE_NAMES["user_sessions"],
    "BillingMode": BILLING_MODE,
    "SSESpecification": _SSE,
    "AttributeDefinitions": [
        {"AttributeName": "session_id", "AttributeType": "S"},
        {"AttributeName": "user_id",    "AttributeType": "S"},
    ],
    "KeySchema": [
        {"AttributeName": "session_id", "KeyType": "HASH"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "user_id-index",
            "KeySchema": [
                {"AttributeName": "user_id", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "KEYS_ONLY"},
        },
    ],
    # Note: TTL is NOT set here — it is enabled separately via ttl_config.py
    # after the table is ACTIVE, using UpdateTimeToLive.
}

# ── 4. Feedback_Loop Table ─────────────────────────────────────────────────────
#
# Stores clinician corrections and validation decisions used to retrain
# or fine-tune the LLM correction model over time.
#
#  PK: feedback_id  (UUID)
#
#  Non-key attributes:
#    document_id      – FK to Documents table
#    user_corrections – JSON list of {region_id, original, corrected, accepted}
#    model_version    – e.g. "anthropic.claude-3-haiku-20240307-v1:0"
#    created_date     – ISO 8601
#
#  GSI: document_id-index
#    Supports "retrieve all feedback for document X" to build training sets.
FEEDBACK_LOOP_TABLE = {
    "TableName": TABLE_NAMES["feedback_loop"],
    "BillingMode": BILLING_MODE,
    "SSESpecification": _SSE,
    "AttributeDefinitions": [
        {"AttributeName": "feedback_id",  "AttributeType": "S"},
        {"AttributeName": "document_id",  "AttributeType": "S"},
    ],
    "KeySchema": [
        {"AttributeName": "feedback_id", "KeyType": "HASH"},
    ],
    "GlobalSecondaryIndexes": [
        {
            "IndexName": "document_id-index",
            "KeySchema": [
                {"AttributeName": "document_id", "KeyType": "HASH"},
            ],
            "Projection": {"ProjectionType": "KEYS_ONLY"},
        },
    ],
}

# ── All table definitions in creation order ────────────────────────────────────
# create_tables.py iterates this list.
ALL_TABLE_DEFINITIONS = [
    DOCUMENTS_TABLE,
    AUDIT_LOGS_TABLE,
    USER_SESSIONS_TABLE,
    FEEDBACK_LOOP_TABLE,
]
