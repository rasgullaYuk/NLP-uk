from hipaa_compliance import create_secure_client

sqs_client = create_secure_client('sqs', region_name='us-east-1')

def setup_pipeline_queues():
    """
    Creates/retrieves all SQS queues needed for the medical document pipeline.

    Returns:
        dict: Queue name to URL mapping
    """
    queue_names = [
        'TrackA_Entity_SNOMED_Queue',
        'TrackB_Summary_Queue',
        'Tier2_LayoutLM_Queue',
        'Tier3_Escalation_Queue'
    ]

    queue_urls = {}

    for queue_name in queue_names:
        try:
            response = sqs_client.create_queue(QueueName=queue_name)
            queue_urls[queue_name] = response['QueueUrl']
            print(f"Queue ready: {queue_name}")
        except Exception as e:
            print(f"Error creating queue {queue_name}: {e}")
            # Try to get existing queue
            try:
                response = sqs_client.get_queue_url(QueueName=queue_name)
                queue_urls[queue_name] = response['QueueUrl']
                print(f"Found existing queue: {queue_name}")
            except Exception as e2:
                print(f"Could not find or create queue {queue_name}: {e2}")

    return queue_urls


def get_queue_url(queue_name):
    """
    Gets the URL for a specific queue.

    Args:
        queue_name: Name of the SQS queue

    Returns:
        str: Queue URL or None if not found
    """
    try:
        response = sqs_client.get_queue_url(QueueName=queue_name)
        return response['QueueUrl']
    except Exception as e:
        print(f"Could not find queue {queue_name}: {e}")
        return None


if __name__ == "__main__":
    print("Setting up pipeline queues...")
    urls = setup_pipeline_queues()
    print("\nQueue URLs:")
    for name, url in urls.items():
        print(f"  {name}: {url}")
