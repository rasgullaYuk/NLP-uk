"""
Initializes runtime resources after infrastructure deployment.
"""

from __future__ import annotations

import argparse

from audit_dynamodb import AuditLogger
from s3_data_lake import S3DataLake
from sqs_setup import setup_pipeline_queues


def initialize_environment() -> None:
    print("Initializing SQS queues...")
    setup_pipeline_queues()

    print("Ensuring DynamoDB audit table exists...")
    _ = AuditLogger()

    print("Ensuring S3 data lake bucket exists...")
    data_lake = S3DataLake()
    data_lake.create_bucket()

    print("Environment initialization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize deployed NLP-uk environment resources.")
    _ = parser.parse_args()
    initialize_environment()
