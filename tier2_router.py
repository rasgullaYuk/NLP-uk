import boto3
import json
import os
import glob

def setup_queues_and_route_data(input_dir="textract_outputs"):
    """
    Central Processing Router
    Creates SQS queues and routes Textract JSON data to them for processing.
    """
    # Initialize the SQS client using your keys
    sqs = boto3.client('sqs', region_name='us-east-1')
    
    print("--- [Step 1: Setting up Conveyor Belts (SQS Queues)] ---")
    
    try:
        # Create Queue for Track A (Medical Mapping)
        track_a_response = sqs.create_queue(QueueName='TrackA_Medical_Queue')
        track_a_url = track_a_response['QueueUrl']
        print(f"Created/Found Track A Queue: {track_a_url}")
        
        # Create Queue for Track B (Summarization)
        track_b_response = sqs.create_queue(QueueName='TrackB_Summary_Queue')
        track_b_url = track_b_response['QueueUrl']
        print(f"Created/Found Track B Queue: {track_b_url}")
        
    except Exception as e:
        print(f"Error creating queues: {e}")
        return

    print("\n--- [Step 2: Routing Extracted Data] ---")
    
    # Find all the JSON files Textract just made
    json_files = glob.glob(os.path.join(input_dir, "*_textract.json"))
    
    if not json_files:
        print("No Textract JSON files found! Please run tier1_textract.py first.")
        return

    for file_path in json_files:
        print(f"Routing document: {file_path}")
        
        # Open and read the Textract data
        with open(file_path, 'r') as f:
            document_data = json.load(f)
            
        # Convert the data to a string message
        message_body = json.dumps({"source_file": file_path, "data": "Data loaded from file"})
        
        # Send the message to BOTH tracks (Fan-out routing)
        try:
            sqs.send_message(QueueUrl=track_a_url, MessageBody=message_body)
            sqs.send_message(QueueUrl=track_b_url, MessageBody=message_body)
            print("  -> Successfully routed to Track A and Track B!")
        except Exception as e:
            print(f"  -> Error routing message: {e}")

if __name__ == "__main__":
    setup_queues_and_route_data()