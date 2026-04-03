import boto3
import json

def setup_sns_fanout(track_a_queue_url, track_b_queue_url):
    """
    Creates an SNS topic and subscribes Track A and Track B SQS queues to it
    for parallel processing.
    """
    sns_client = boto3.client('sns', region_name='us-east-1')
    sqs_client = boto3.client('sqs', region_name='us-east-1')
    
    print("Creating SNS Topic...")
    topic_response = sns_client.create_topic(Name='Clinical_Doc_Parallel_Processing')
    topic_arn = topic_response['TopicArn']
    print(f"Topic created: {topic_arn}")
    
    # Helper function to get Queue ARN and setup policy
    def subscribe_queue_to_topic(queue_url):
        # Get Queue ARN
        attributes = sqs_client.get_queue_attributes(
            QueueUrl=queue_url, AttributeNames=['QueueArn']
        )
        queue_arn = attributes['Attributes']['QueueArn']
        
        # Subscribe Queue to SNS Topic
        sns_client.subscribe(
            TopicArn=topic_arn, Protocol='sqs', Endpoint=queue_arn
        )
        
        # Grant SNS permission to write to this SQS queue
        policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sns.amazonaws.com"},
                "Action": "sqs:SendMessage",
                "Resource": queue_arn,
                "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}}
            }]
        }
        sqs_client.set_queue_attributes(
            QueueUrl=queue_url, Attributes={'Policy': json.dumps(policy)}
        )
        print(f"Subscribed {queue_arn} to Topic.")

    print("\nSubscribing Queues to SNS Topic...")
    subscribe_queue_to_topic(track_a_queue_url)
    subscribe_queue_to_topic(track_b_queue_url)
    
    return topic_arn

# Example usage (you will pass your actual URLs generated from sqs_setup.py)
# topic_arn = setup_sns_fanout(urls['TrackA_Entity_SNOMED_Queue'], urls['TrackB_Chunking_Embedding_Queue'])