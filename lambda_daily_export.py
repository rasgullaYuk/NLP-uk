"""
AWS Lambda Function for Daily DynamoDB to S3 Export

This Lambda function is triggered daily by CloudWatch Events/EventBridge
to export clinician feedback from DynamoDB to the S3 data lake.

Deployment:
1. Package this file with s3_data_lake.py and dependencies
2. Create Lambda function with Python 3.9+ runtime
3. Set IAM role with DynamoDB read and S3 write permissions
4. Create EventBridge rule: cron(0 2 * * ? *) for 2 AM UTC daily

Environment Variables:
- DATA_LAKE_BUCKET: S3 bucket name (default: clinical-nlp-data-lake)
- DYNAMODB_TABLE: DynamoDB table name (default: ClinicalDocumentAuditLog)
- ANONYMIZE_PII: Whether to anonymize PII (default: true)
"""

import json
import os
from datetime import datetime, timedelta

# Import from s3_data_lake module
from s3_data_lake import (
    S3DataLake,
    DynamoDBToS3Exporter,
    DATA_LAKE_BUCKET,
    DYNAMODB_AUDIT_TABLE
)


def lambda_handler(event, context):
    """
    Lambda handler for daily export.

    Args:
        event: Lambda event (can contain 'export_date' for manual runs)
        context: Lambda context

    Returns:
        dict: Response with export results
    """
    print(f"Lambda triggered at {datetime.utcnow().isoformat()}")
    print(f"Event: {json.dumps(event)}")

    # Get configuration from environment or defaults
    bucket_name = os.environ.get('DATA_LAKE_BUCKET', DATA_LAKE_BUCKET)
    anonymize = os.environ.get('ANONYMIZE_PII', 'true').lower() == 'true'

    # Allow manual export date override
    export_date = event.get('export_date')
    if not export_date:
        # Default to yesterday
        export_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Exporting data for: {export_date}")
    print(f"Anonymize PII: {anonymize}")
    print(f"Target bucket: {bucket_name}")

    try:
        # Initialize data lake and exporter
        data_lake = S3DataLake(bucket_name=bucket_name)
        exporter = DynamoDBToS3Exporter(data_lake)

        # Run the export
        results = exporter.export_daily_feedback(
            export_date=export_date,
            anonymize=anonymize
        )

        print(f"Export completed: {json.dumps(results, indent=2)}")

        # Return success response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Export completed successfully',
                'export_date': export_date,
                'results': results
            })
        }

    except Exception as e:
        error_msg = f"Export failed: {str(e)}"
        print(error_msg)

        # Return error response
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Export failed',
                'error': str(e),
                'export_date': export_date
            })
        }


def create_eventbridge_rule():
    """
    Helper to create the EventBridge rule for daily scheduling.

    Run this once to set up the schedule:
        python -c "from lambda_daily_export import create_eventbridge_rule; create_eventbridge_rule()"
    """
    import boto3

    events_client = boto3.client('events', region_name='us-east-1')
    lambda_client = boto3.client('lambda', region_name='us-east-1')

    rule_name = 'DailyFeedbackExportRule'
    lambda_function_name = 'clinical-nlp-daily-export'

    try:
        # Create the rule (2 AM UTC daily)
        events_client.put_rule(
            Name=rule_name,
            ScheduleExpression='cron(0 2 * * ? *)',
            State='ENABLED',
            Description='Daily export of clinician feedback to S3 data lake'
        )
        print(f"Created EventBridge rule: {rule_name}")

        # Get Lambda ARN
        lambda_arn = lambda_client.get_function(
            FunctionName=lambda_function_name
        )['Configuration']['FunctionArn']

        # Add Lambda as target
        events_client.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': 'DailyExportLambda',
                    'Arn': lambda_arn
                }
            ]
        )
        print(f"Added Lambda target: {lambda_function_name}")

        # Add permission for EventBridge to invoke Lambda
        try:
            lambda_client.add_permission(
                FunctionName=lambda_function_name,
                StatementId='EventBridgeInvoke',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=f"arn:aws:events:us-east-1:*:rule/{rule_name}"
            )
            print("Added Lambda invoke permission")
        except lambda_client.exceptions.ResourceConflictException:
            print("Permission already exists")

        print("\nEventBridge rule setup complete!")
        print(f"Schedule: Daily at 2:00 AM UTC")

    except Exception as e:
        print(f"Error setting up EventBridge rule: {e}")


def get_lambda_deployment_package_instructions():
    """Returns instructions for deploying this Lambda function."""
    return """
    ============================================================
    LAMBDA DEPLOYMENT INSTRUCTIONS
    ============================================================

    1. CREATE DEPLOYMENT PACKAGE:
       --------------------------
       mkdir lambda_package
       cd lambda_package
       pip install boto3 pyarrow -t .
       cp ../s3_data_lake.py .
       cp ../lambda_daily_export.py .
       zip -r ../lambda_deployment.zip .

    2. CREATE IAM ROLE:
       ----------------
       Create role 'clinical-nlp-lambda-role' with policies:
       - AWSLambdaBasicExecutionRole
       - Custom policy:
         {
           "Version": "2012-10-17",
           "Statement": [
             {
               "Effect": "Allow",
               "Action": [
                 "dynamodb:Scan",
                 "dynamodb:Query"
               ],
               "Resource": "arn:aws:dynamodb:*:*:table/ClinicalDocumentAuditLog*"
             },
             {
               "Effect": "Allow",
               "Action": [
                 "s3:PutObject",
                 "s3:GetObject",
                 "s3:ListBucket"
               ],
               "Resource": [
                 "arn:aws:s3:::clinical-nlp-data-lake",
                 "arn:aws:s3:::clinical-nlp-data-lake/*"
               ]
             }
           ]
         }

    3. CREATE LAMBDA FUNCTION:
       -----------------------
       aws lambda create-function \\
         --function-name clinical-nlp-daily-export \\
         --runtime python3.9 \\
         --handler lambda_daily_export.lambda_handler \\
         --role arn:aws:iam::YOUR_ACCOUNT:role/clinical-nlp-lambda-role \\
         --zip-file fileb://lambda_deployment.zip \\
         --timeout 300 \\
         --memory-size 512

    4. SET ENVIRONMENT VARIABLES:
       --------------------------
       aws lambda update-function-configuration \\
         --function-name clinical-nlp-daily-export \\
         --environment "Variables={DATA_LAKE_BUCKET=clinical-nlp-data-lake,ANONYMIZE_PII=true}"

    5. CREATE EVENTBRIDGE SCHEDULE:
       ----------------------------
       Run: python -c "from lambda_daily_export import create_eventbridge_rule; create_eventbridge_rule()"

       Or manually:
       aws events put-rule \\
         --name DailyFeedbackExportRule \\
         --schedule-expression "cron(0 2 * * ? *)"

    6. TEST MANUALLY:
       ---------------
       aws lambda invoke \\
         --function-name clinical-nlp-daily-export \\
         --payload '{"export_date": "2024-01-15"}' \\
         output.json

    ============================================================
    """


if __name__ == "__main__":
    # Print deployment instructions
    print(get_lambda_deployment_package_instructions())

    # Test locally (simulates Lambda invocation)
    print("\n" + "=" * 60)
    print("LOCAL TEST RUN")
    print("=" * 60)

    test_event = {}  # Empty event uses yesterday's date
    test_context = None

    result = lambda_handler(test_event, test_context)
    print(f"\nResult: {json.dumps(result, indent=2)}")
