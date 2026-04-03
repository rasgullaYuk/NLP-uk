import boto3
import json

sqs_client = boto3.client('sqs', region_name='us-east-1')

def send_to_sqs(queue_url, payload):
    """
    Sends a JSON payload to a specific SQS queue.
    """
    try:
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(payload)
        )
        print(f"Message successfully sent. MessageId: {response['MessageId']}")
        return response['MessageId']
    except Exception as e:
        print(f"Failed to send message to {queue_url}: {e}")
        return None

def receive_from_sqs(queue_url, max_messages=1):
    """
    Polls the SQS queue for new messages to process.
    """
    try:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=5, # Short polling
            VisibilityTimeout=300
        )
        
        messages = response.get('Messages', [])
        if not messages:
            print("No messages in queue.")
            return []
            
        return messages
    except Exception as e:
        print(f"Error receiving messages from {queue_url}: {e}")
        return []

def delete_from_sqs(queue_url, receipt_handle):
    """
    Deletes a message from the queue after successful processing to prevent re-processing.
    """
    try:
        sqs_client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
        print("Message deleted successfully.")
    except Exception as e:
        print(f"Failed to delete message: {e}")