import boto3
import json
import os
import glob

def calculate_document_confidence(textract_data):
    """
    Calculates average confidence from Textract blocks.

    Args:
        textract_data: Parsed Textract JSON

    Returns:
        float: Average confidence score (0-100)
    """
    confidences = []
    for block in textract_data.get('Blocks', []):
        if block.get('BlockType') in ['LINE', 'WORD']:
            confidence = block.get('Confidence', 0)
            if confidence > 0:
                confidences.append(confidence)

    return sum(confidences) / len(confidences) if confidences else 0.0


def find_image_for_textract(textract_path, image_dir="temp_pages"):
    """
    Finds the corresponding image file for a Textract JSON.

    Args:
        textract_path: Path to Textract JSON file
        image_dir: Directory containing images

    Returns:
        str: Path to image file or None
    """
    base_name = os.path.basename(textract_path).replace('_textract.json', '')

    # Try common image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.tiff']:
        image_path = os.path.join(image_dir, f"{base_name}{ext}")
        if os.path.exists(image_path):
            return image_path

    return None


def setup_queues_and_route_data(input_dir="textract_outputs", confidence_threshold=90.0):
    """
    Central Processing Router
    Creates SQS queues and routes Textract JSON data based on confidence.

    - High confidence (>= 90%): Direct to Track A and Track B
    - Low confidence (< 90%): Route to Tier 2 LayoutLMv3 for refinement
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

        # Create Queue for Tier 2 LayoutLMv3 (Low confidence documents)
        tier2_response = sqs.create_queue(QueueName='Tier2_LayoutLM_Queue')
        tier2_url = tier2_response['QueueUrl']
        print(f"Created/Found Tier 2 Queue: {tier2_url}")

    except Exception as e:
        print(f"Error creating queues: {e}")
        return

    print("\n--- [Step 2: Routing Extracted Data] ---")
    print(f"Confidence threshold: {confidence_threshold}%")

    # Find all the JSON files Textract just made
    json_files = glob.glob(os.path.join(input_dir, "*_textract.json"))

    if not json_files:
        print("No Textract JSON files found! Please run tier1_textract.py first.")
        return

    high_confidence_count = 0
    low_confidence_count = 0

    for file_path in json_files:
        print(f"\nRouting document: {file_path}")

        # Open and read the Textract data
        with open(file_path, 'r') as f:
            document_data = json.load(f)

        # Calculate confidence
        avg_confidence = calculate_document_confidence(document_data)
        print(f"  Average confidence: {avg_confidence:.2f}%")

        # Find corresponding image
        image_path = find_image_for_textract(file_path)

        # Create message with file paths for Tier 2
        base_name = os.path.basename(file_path).replace('_textract.json', '')

        if avg_confidence >= confidence_threshold:
            # High confidence - route directly to Track A and B
            message_body = json.dumps({
                "document_id": base_name,
                "source_file": file_path,
                "data": "Data loaded from file",
                "average_confidence": avg_confidence
            })

            try:
                sqs.send_message(QueueUrl=track_a_url, MessageBody=message_body)
                sqs.send_message(QueueUrl=track_b_url, MessageBody=message_body)
                print(f"  -> HIGH CONFIDENCE: Routed to Track A and Track B")
                high_confidence_count += 1
            except Exception as e:
                print(f"  -> Error routing message: {e}")
        else:
            # Low confidence - route to Tier 2 for LayoutLMv3 refinement
            message_body = json.dumps({
                "document_id": base_name,
                "textract_json_path": file_path,
                "image_path": image_path,
                "average_confidence": avg_confidence,
                "low_confidence_flag": True
            })

            try:
                sqs.send_message(QueueUrl=tier2_url, MessageBody=message_body)
                print(f"  -> LOW CONFIDENCE: Routed to Tier 2 (LayoutLMv3)")
                low_confidence_count += 1
            except Exception as e:
                print(f"  -> Error routing message: {e}")

    print("\n--- [Routing Summary] ---")
    print(f"High confidence (>= {confidence_threshold}%): {high_confidence_count} documents -> Track A/B")
    print(f"Low confidence (< {confidence_threshold}%): {low_confidence_count} documents -> Tier 2")

if __name__ == "__main__":
    setup_queues_and_route_data()