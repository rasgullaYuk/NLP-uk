import json
import time
import logging
from hipaa_compliance import create_secure_client, scrub_json_value

logger = logging.getLogger("track_a.sqs")

AWS_REGION = "us-east-1"

sqs_client = create_secure_client("sqs", region_name=AWS_REGION)


def send_to_sqs(queue_url, payload, max_retries=3):
    """
    Sends a JSON payload to a specific SQS queue.
    Enhanced: exponential backoff retry on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            safe_payload = scrub_json_value(payload)
            response = sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(safe_payload)
            )
            logger.info("Message sent. MessageId: %s", response["MessageId"])
            print(f"Message successfully sent. MessageId: {response['MessageId']}")
            return response["MessageId"]
        except Exception as e:
            wait = 2 ** attempt  
            logger.warning("Send attempt %d/%d failed: %s. Retrying in %ds...", attempt, max_retries, e, wait)
            if attempt < max_retries:
                time.sleep(wait)
            else:
                logger.error("All %d send attempts failed for queue %s: %s", max_retries, queue_url, e)
                print(f"Failed to send message to {queue_url} after {max_retries} attempts: {e}")
                return None


def send_to_dlq(dlq_url, original_payload, error_reason, attempt_count):
    """
    Sends a failed message to the Dead Letter Queue with full diagnostic context.
    Called after all retries are exhausted — ensures zero message loss.
    """
    dlq_payload = {
        "original_payload": scrub_json_value(original_payload),
        "error_reason": str(error_reason),
        "attempt_count": attempt_count,
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queue": "TrackA_Medical_Queue",
    }
    try:
        response = sqs_client.send_message(
            QueueUrl=dlq_url,
            MessageBody=json.dumps(dlq_payload)
        )
        logger.warning("Message sent to DLQ. DLQ MessageId: %s", response["MessageId"])
        print(f"  [DLQ] Failed message captured. DLQ MessageId: {response['MessageId']}")
        return response["MessageId"]
    except Exception as e:
        logger.critical("CRITICAL: Could not send to DLQ either! Payload: %s | Error: %s", dlq_payload, e)
        print(f"  [DLQ] CRITICAL: Could not send to DLQ: {e}")
        return None


def receive_from_sqs(queue_url, max_messages=1):
    """
    Polls the SQS queue for new messages to process.
    Preserved from original.
    """
    try:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=5,
            VisibilityTimeout=300
        )
        messages = response.get("Messages", [])
        if not messages:
            print("No messages in queue.")
            return []
        return messages
    except Exception as e:
        logger.error("Error receiving messages from %s: %s", queue_url, e)
        print(f"Error receiving messages from {queue_url}: {e}")
        return []


def delete_from_sqs(queue_url, receipt_handle):
    """
    Deletes a message from the queue after successful processing.
    Preserved from original.
    """
    try:
        sqs_client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
        logger.info("Message deleted from queue.")
        print("Message deleted successfully.")
    except Exception as e:
        logger.error("Failed to delete message: %s", e)
        print(f"Failed to delete message: {e}")
