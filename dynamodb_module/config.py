"""
config.py — DynamoDB Module Configuration
==========================================
All AWS settings and table name constants live here.
Import from this file everywhere else — never hardcode these values.
"""

# ── AWS ────────────────────────────────────────────────────────────────────────
AWS_REGION = "us-east-1"

# ── Table names ────────────────────────────────────────────────────────────────
# Prefix all tables with the project name for easy IAM scoping and cost tagging.
TABLE_NAMES = {
    "documents":     "ClinicalDocs_Documents",
    "audit_logs":    "ClinicalDocs_AuditLogs",
    "user_sessions": "ClinicalDocs_UserSessions",
    "feedback_loop": "ClinicalDocs_FeedbackLoop",
}

# ── TTL ────────────────────────────────────────────────────────────────────────
# Only User_Sessions uses TTL.
# DynamoDB expects the TTL attribute to contain a Unix epoch integer (seconds).
TTL_ATTRIBUTE   = "ttl_expiry"      # name of the TTL attribute in User_Sessions
TTL_WINDOW_DAYS = 30                # session records expire 30 days after logout

# ── Billing ────────────────────────────────────────────────────────────────────
# PAY_PER_REQUEST: no capacity planning, scales automatically, better for
# variable OCR pipeline workloads (burst during batch uploads, idle otherwise).
BILLING_MODE = "PAY_PER_REQUEST"

# ── Retry / waiter ────────────────────────────────────────────────────────────
TABLE_CREATION_TIMEOUT_SECONDS = 120   # max wait for table to become ACTIVE
RETRY_MAX_ATTEMPTS             = 3
RETRY_BASE_DELAY_SECONDS       = 2.0   # delay = base * 2^attempt (exponential)
