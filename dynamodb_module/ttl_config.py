"""
ttl_config.py — Enable TTL on User_Sessions Table
==================================================
DynamoDB TTL must be enabled AFTER the table reaches ACTIVE status.
This module is called by create_tables.py once the table is ready.

TTL attribute: ttl_expiry (Unix epoch integer, seconds)
Items are automatically deleted ~30 days after the stored epoch passes.
DynamoDB TTL deletion is eventually consistent — callers should treat
logout_timestamp as the authoritative session-end marker.
"""

from __future__ import annotations

import logging
import time

from botocore.exceptions import ClientError
try:
    from hipaa_compliance import create_secure_client
except ImportError:
    import boto3

    def create_secure_client(service_name: str, region_name: str, **kwargs):
        return boto3.client(service_name, region_name=region_name, **kwargs)

from config import AWS_REGION, TABLE_NAMES, TTL_ATTRIBUTE

logger = logging.getLogger(__name__)


def enable_ttl(client=None) -> bool:
    """
    Enable TTL on the User_Sessions table using the ttl_expiry attribute.

    This is idempotent — calling it on a table where TTL is already enabled
    with the same attribute is a no-op (AWS returns a validation error that
    we catch and treat as success).

    Args:
        client: Optional pre-built boto3 DynamoDB client.  If None, a new
                client is created using the configured AWS_REGION.

    Returns:
        True  if TTL was enabled (or was already enabled correctly).
        False if an unrecoverable error occurred.
    """
    if client is None:
        client = create_secure_client("dynamodb", region_name=AWS_REGION)

    table_name = TABLE_NAMES["user_sessions"]

    try:
        client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={
                "Enabled":       True,
                "AttributeName": TTL_ATTRIBUTE,
            },
        )
        logger.info(
            "TTL enabled on '%s' using attribute '%s'.", table_name, TTL_ATTRIBUTE
        )
        return True

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]

        # ValidationException is raised when TTL is already enabled with the
        # same attribute — treat this as a success (idempotent).
        if error_code == "ValidationException":
            message = exc.response["Error"].get("Message", "")
            if "TimeToLive is already enabled" in message:
                logger.info(
                    "TTL already enabled on '%s' (attribute: '%s'). Skipping.",
                    table_name, TTL_ATTRIBUTE,
                )
                return True
            logger.error("Unexpected ValidationException enabling TTL: %s", exc)
        else:
            logger.error("Failed to enable TTL on '%s': %s", table_name, exc)

        return False


def verify_ttl(client=None) -> dict | None:
    """
    Describe the current TTL configuration for the User_Sessions table.

    Args:
        client: Optional pre-built boto3 DynamoDB client.

    Returns:
        The TimeToLiveDescription dict from AWS, or None on error.
        Example: {"TimeToLiveStatus": "ENABLED", "AttributeName": "ttl_expiry"}
    """
    if client is None:
        client = create_secure_client("dynamodb", region_name=AWS_REGION)

    table_name = TABLE_NAMES["user_sessions"]

    try:
        response = client.describe_time_to_live(TableName=table_name)
        ttl_desc = response.get("TimeToLiveDescription", {})
        logger.info("TTL status for '%s': %s", table_name, ttl_desc)
        return ttl_desc

    except ClientError as exc:
        logger.error("Could not describe TTL for '%s': %s", table_name, exc)
        return None


def compute_ttl_expiry(logout_epoch_seconds: int, window_days: int = 30) -> int:
    """
    Compute the ttl_expiry Unix epoch value to store on a User_Sessions item.

    Args:
        logout_epoch_seconds: The session logout time as a Unix epoch integer.
        window_days:          Number of days after logout before the item expires.

    Returns:
        Unix epoch integer for the TTL attribute.

    Example:
        import time
        ttl = compute_ttl_expiry(int(time.time()))  # 30 days from now
    """
    return logout_epoch_seconds + (window_days * 86_400)
