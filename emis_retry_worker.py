"""
Worker for processing queued EMIS export retries.
"""

from __future__ import annotations

import json
import os
import time

from audit_dynamodb import get_audit_logger
from emis_export_integration import EMIS_RETRY_QUEUE_NAME, process_retry_message
from hipaa_compliance import create_secure_client


def process_retry_queue(poll_seconds: int = 5) -> None:
    sqs = create_secure_client("sqs", region_name=os.getenv("AWS_REGION", "us-east-1"))
    audit_logger = get_audit_logger()
    try:
        queue_url = sqs.get_queue_url(QueueName=EMIS_RETRY_QUEUE_NAME)["QueueUrl"]
    except Exception:
        queue_url = sqs.create_queue(QueueName=EMIS_RETRY_QUEUE_NAME)["QueueUrl"]

    print(f"Listening for retry messages on {EMIS_RETRY_QUEUE_NAME} ...")
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=poll_seconds,
            VisibilityTimeout=60,
        )
        messages = response.get("Messages", [])
        if not messages:
            continue

        for message in messages:
            body = json.loads(message["Body"])
            result = process_retry_message(body, audit_logger=audit_logger, user_id="EMIS_RETRY_WORKER")
            if result.get("success"):
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
            else:
                attempts = int(body.get("attempts", 0)) + 1
                if attempts >= int(os.getenv("EMIS_MAX_RETRY_QUEUE_ATTEMPTS", "6")):
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
                else:
                    # Leave message for redrive; slight pause to avoid hot-looping.
                    time.sleep(1)


if __name__ == "__main__":
    process_retry_queue()
