import boto3
import json
import os

def process_track_a_queue(output_dir="track_a_outputs"):
    """
    Track A: Medical Entity & SNOMED Mapping
    Pulls messages continuously until the SQS queue is empty.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sqs = boto3.client('sqs', region_name='us-east-1')
    comprehend_medical = boto3.client('comprehendmedical', region_name='us-east-1')

    try:
        queue_url = sqs.get_queue_url(QueueName='TrackA_Medical_Queue')['QueueUrl']
    except Exception as e:
        print("Could not find TrackA_Medical_Queue. Did you run tier2_router.py?")
        return

    print("--- [Track A: Starting Continuous Batch Processing] ---")
    
    # The 'while True' loop keeps the machine running until the queue is empty
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=5
        )

        # If AWS returns no messages, the conveyor belt is empty! Break the loop.
        if 'Messages' not in response:
            print("\n--- The queue is empty! All documents processed. ---")
            break

        for message in response['Messages']:
            body = json.loads(message['Body'])
            file_path = body['source_file']
            
            print(f"\nProcessing document: {file_path}")

            with open(file_path, 'r') as f:
                textract_data = json.load(f)

            raw_text_lines = []
            for block in textract_data.get('Blocks', []):
                if block['BlockType'] == 'LINE':
                    raw_text_lines.append(block['Text'])
            
            full_text = " ".join(raw_text_lines)
            text_to_analyze = full_text[:9500] 

            if not text_to_analyze.strip():
                print("No text found. Deleting message.")
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message['ReceiptHandle'])
                continue

            print("Sending text to AWS Comprehend Medical...")
            
            try:
                snomed_response = comprehend_medical.infer_snomedct(Text=text_to_analyze)
                
                base_name = os.path.basename(file_path).replace('_textract.json', '')
                output_file = os.path.join(output_dir, f"{base_name}_snomed.json")
                
                with open(output_file, 'w') as f:
                    json.dump(snomed_response, f, indent=4)
                    
                print(f"SUCCESS: Saved to {output_file}")
                
                # Delete the message so we don't process it twice
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
                print("Message removed from queue.")

            except Exception as e:
                print(f"FAILED to process. Error: {e}")
                return # Stop the machine if there is an AWS error

if __name__ == "__main__":
    process_track_a_queue()