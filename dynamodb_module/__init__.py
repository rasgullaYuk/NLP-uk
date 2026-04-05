"""
dynamodb_module — ClinicalDocs DynamoDB Persistence Layer
==========================================================
Public API surface of this package.
"""

from .create_tables      import create_all_tables
from .ttl_config         import enable_ttl, verify_ttl, compute_ttl_expiry
from .table_definitions  import (
    DOCUMENTS_TABLE,
    AUDIT_LOGS_TABLE,
    USER_SESSIONS_TABLE,
    FEEDBACK_LOOP_TABLE,
    ALL_TABLE_DEFINITIONS,
)

__all__ = [
    "create_all_tables",
    "enable_ttl",
    "verify_ttl",
    "compute_ttl_expiry",
    "DOCUMENTS_TABLE",
    "AUDIT_LOGS_TABLE",
    "USER_SESSIONS_TABLE",
    "FEEDBACK_LOOP_TABLE",
    "ALL_TABLE_DEFINITIONS",
]
