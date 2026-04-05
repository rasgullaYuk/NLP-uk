"""
create_tables.py — DynamoDB Table Provisioner
=============================================
Creates all four ClinicalDocs DynamoDB tables.

Run once during infrastructure setup:
    python create_tables.py

Behaviour:
- Skips tables that already exist (idempotent).
- Waits for each table to reach ACTIVE status before proceeding.
- Enables TTL on User_Sessions after it becomes ACTIVE.
- Retries transient AWS errors with exponential backoff.
- Exits non-zero if any table fails to become ACTIVE.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from config import (
    AWS_REGION,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_ATTEMPTS,
    TABLE_CREATION_TIMEOUT_SECONDS,
    TABLE_NAMES,
)
from table_definitions import ALL_TABLE_DEFINITIONS
from ttl_config import enable_ttl, verify_ttl

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# ── Helper: check if table exists ─────────────────────────────────────────────

def _table_exists(client: Any, table_name: str) -> bool:
    """
    Return True if the DynamoDB table already exists (in any status),
    False if it does not exist.

    Args:
        client:     boto3 DynamoDB client.
        table_name: Table name to check.
    """
    try:
        client.describe_table(TableName=table_name)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ResourceNotFoundException":
            return False
        raise


# ── Helper: wait for ACTIVE ────────────────────────────────────────────────────

def _wait_for_active(client: Any, table_name: str) -> bool:
    """
    Poll until the table status is ACTIVE or the timeout is exceeded.

    Args:
        client:     boto3 DynamoDB client.
        table_name: Table to wait on.

    Returns:
        True if ACTIVE, False if timed out or an error occurred.
    """
    deadline = time.time() + TABLE_CREATION_TIMEOUT_SECONDS
    poll_interval = 5  # seconds between status checks

    logger.info("Waiting for table '%s' to become ACTIVE...", table_name)
    while time.time() < deadline:
        try:
            response = client.describe_table(TableName=table_name)
            status   = response["Table"]["TableStatus"]

            if status == "ACTIVE":
                logger.info("Table '%s' is ACTIVE.", table_name)
                return True

            logger.debug("Table '%s' status: %s — waiting...", table_name, status)

        except ClientError as exc:
            logger.warning(
                "describe_table error for '%s': %s", table_name, exc
            )

        time.sleep(poll_interval)

    logger.error(
        "Table '%s' did not become ACTIVE within %ds.",
        table_name, TABLE_CREATION_TIMEOUT_SECONDS,
    )
    return False


# ── Helper: create a single table with retry ──────────────────────────────────

def _create_table_with_retry(client: Any, definition: dict) -> bool:
    """
    Create a single DynamoDB table, retrying on transient errors.

    Args:
        client:     boto3 DynamoDB client.
        definition: Table definition dict (from table_definitions.py).

    Returns:
        True if the table was created (or already existed), False on failure.
    """
    table_name = definition["TableName"]

    # ── Existence check ────────────────────────────────────────────────────────
    if _table_exists(client, table_name):
        logger.info("Table '%s' already exists — skipping creation.", table_name)
        return True

    # ── Create with exponential backoff ────────────────────────────────────────
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            logger.info(
                "Creating table '%s' (attempt %d/%d)...",
                table_name, attempt, RETRY_MAX_ATTEMPTS,
            )
            client.create_table(**definition)
            logger.info("create_table request accepted for '%s'.", table_name)
            return True

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]

            if error_code == "ResourceInUseException":
                # Table was created by a concurrent process — treat as success.
                logger.info(
                    "Table '%s' already being created (ResourceInUseException).",
                    table_name,
                )
                return True

            if error_code in ("ProvisionedThroughputExceededException",
                              "RequestLimitExceeded",
                              "ServiceUnavailableException"):
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error creating '%s': %s. Retrying in %.1fs.",
                    table_name, error_code, delay,
                )
                time.sleep(delay)
                continue

            # Non-retryable error
            logger.error(
                "Failed to create table '%s': %s", table_name, exc
            )
            return False

    logger.error(
        "Exhausted %d attempts for table '%s'.", RETRY_MAX_ATTEMPTS, table_name
    )
    return False


# ── Main provisioner ───────────────────────────────────────────────────────────

def create_all_tables() -> bool:
    """
    Create all four DynamoDB tables and configure TTL on User_Sessions.

    Returns:
        True if all tables are ACTIVE and TTL is configured, False otherwise.
    """
    client  = boto3.client("dynamodb", region_name=AWS_REGION)
    success = True

    logger.info("=== DynamoDB Table Provisioner — start ===")
    logger.info("Region : %s", AWS_REGION)
    logger.info("Tables : %s", list(TABLE_NAMES.values()))

    for definition in ALL_TABLE_DEFINITIONS:
        table_name = definition["TableName"]

        # Step 1: Create (or confirm existence)
        created = _create_table_with_retry(client, definition)
        if not created:
            logger.error("FAILED to initiate creation of '%s'.", table_name)
            success = False
            continue

        # Step 2: Wait for ACTIVE
        active = _wait_for_active(client, table_name)
        if not active:
            logger.error("Table '%s' never became ACTIVE.", table_name)
            success = False
            continue

        logger.info("Table '%s' is ready.", table_name)

    # Step 3: Enable TTL on User_Sessions (only after it is ACTIVE)
    if success or _table_exists(client, TABLE_NAMES["user_sessions"]):
        logger.info("Configuring TTL on User_Sessions...")
        ttl_ok = enable_ttl(client=client)
        if ttl_ok:
            ttl_status = verify_ttl(client=client)
            logger.info("TTL configuration: %s", ttl_status)
        else:
            logger.error("TTL configuration failed.")
            success = False

    logger.info(
        "=== DynamoDB Table Provisioner — %s ===",
        "COMPLETE" if success else "COMPLETED WITH ERRORS",
    )
    return success


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok = create_all_tables()
    sys.exit(0 if ok else 1)
