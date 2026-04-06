"""
DynamoDB Audit Logging Module

Provides audit logging capabilities for the Clinician Review Dashboard.
Captures before/after states for every user edit, tracks user identity,
timestamps, and changes to SNOMED codes, summaries, and actions.
"""

import boto3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError

# DynamoDB Configuration
AUDIT_TABLE_NAME = "ClinicalDocumentAuditLog"
AWS_REGION = "us-east-1"


class AuditLogger:
    """
    DynamoDB-backed audit logger for clinical document edits.

    Schema:
    - audit_id (PK): Unique identifier for the audit entry
    - document_id (GSI): Document being edited
    - user_id (GSI): User making the edit
    - timestamp: ISO 8601 timestamp
    - change_type: Type of change (SUMMARY_EDIT, SNOMED_STATUS, APPROVE_ALL, FLAG_REVIEW, etc.)
    - before_state: State before the change
    - after_state: State after the change
    - metadata: Additional context
    """

    _instance = None
    _dynamodb = None
    _table = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._dynamodb is None:
            self._dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
            self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Creates the audit table if it doesn't exist."""
        try:
            self._table = self._dynamodb.Table(AUDIT_TABLE_NAME)
            self._table.load()
            print(f"Audit table '{AUDIT_TABLE_NAME}' ready.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                print(f"Creating audit table '{AUDIT_TABLE_NAME}'...")
                self._create_table()
            else:
                raise

    def _create_table(self):
        """Creates the DynamoDB audit table with proper schema."""
        try:
            self._table = self._dynamodb.create_table(
                TableName=AUDIT_TABLE_NAME,
                KeySchema=[
                    {'AttributeName': 'audit_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'audit_id', 'AttributeType': 'S'},
                    {'AttributeName': 'document_id', 'AttributeType': 'S'},
                    {'AttributeName': 'user_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                ],
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'document-index',
                        'KeySchema': [
                            {'AttributeName': 'document_id', 'KeyType': 'HASH'},
                            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'},
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    },
                    {
                        'IndexName': 'user-index',
                        'KeySchema': [
                            {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                        ],
                        'Projection': {'ProjectionType': 'ALL'},
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )

            # Wait for table to be created
            self._table.meta.client.get_waiter('table_exists').wait(
                TableName=AUDIT_TABLE_NAME
            )
            print(f"Audit table '{AUDIT_TABLE_NAME}' created successfully.")

        except ClientError as e:
            print(f"Error creating audit table: {e}")
            raise

    def log_change(self,
                   document_id: str,
                   user_id: str,
                   change_type: str,
                   before_state: Any,
                   after_state: Any,
                   metadata: Optional[Dict] = None) -> str:
        """
        Logs a change to the audit table.

        Args:
            document_id: ID of the document being edited
            user_id: ID of the user making the edit
            change_type: Type of change (SUMMARY_EDIT, SNOMED_STATUS, etc.)
            before_state: State before the change
            after_state: State after the change
            metadata: Additional context (optional)

        Returns:
            str: The audit_id of the logged entry

        Raises:
            Exception: If the write fails (zero data loss guarantee)
        """
        audit_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + 'Z'

        item = {
            'audit_id': audit_id,
            'document_id': document_id,
            'user_id': user_id,
            'timestamp': timestamp,
            'change_type': change_type,
            'before_state': json.dumps(before_state) if before_state else None,
            'after_state': json.dumps(after_state) if after_state else None,
            'metadata': json.dumps(metadata) if metadata else None
        }

        try:
            self._table.put_item(Item=item)
            print(f"Audit logged: {change_type} for {document_id} by {user_id}")
            return audit_id
        except ClientError as e:
            print(f"CRITICAL: Failed to log audit entry: {e}")
            raise  # Zero data loss - must succeed before UI confirms

    def log_summary_edit(self,
                         document_id: str,
                         user_id: str,
                         before_summary: str,
                         after_summary: str) -> str:
        """Logs a summary text edit."""
        return self.log_change(
            document_id=document_id,
            user_id=user_id,
            change_type='SUMMARY_EDIT',
            before_state={'summary': before_summary},
            after_state={'summary': after_summary},
            metadata={'field': 'clinical_summary'}
        )

    def log_snomed_status_change(self,
                                  document_id: str,
                                  user_id: str,
                                  entity_text: str,
                                  snomed_code: str,
                                  before_status: str,
                                  after_status: str) -> str:
        """Logs a SNOMED code status change."""
        return self.log_change(
            document_id=document_id,
            user_id=user_id,
            change_type='SNOMED_STATUS',
            before_state={'status': before_status},
            after_state={'status': after_status},
            metadata={
                'entity_text': entity_text,
                'snomed_code': snomed_code
            }
        )

    def log_approve_all(self,
                        document_id: str,
                        user_id: str,
                        entities_approved: int) -> str:
        """Logs an 'Approve All' action."""
        return self.log_change(
            document_id=document_id,
            user_id=user_id,
            change_type='APPROVE_ALL',
            before_state={'status': 'pending'},
            after_state={'status': 'approved', 'exported_to': 'EMIS'},
            metadata={'entities_approved': entities_approved}
        )

    def log_flag_for_review(self,
                            document_id: str,
                            user_id: str,
                            reason: Optional[str] = None) -> str:
        """Logs a 'Flag for Specialist Review' action."""
        return self.log_change(
            document_id=document_id,
            user_id=user_id,
            change_type='FLAG_REVIEW',
            before_state={'status': 'pending'},
            after_state={'status': 'flagged_for_review'},
            metadata={'reason': reason}
        )

    def get_audit_trail_by_document(self,
                                    document_id: str,
                                    limit: int = 100) -> List[Dict]:
        """
        Retrieves audit trail for a specific document.

        Args:
            document_id: Document ID to query
            limit: Maximum number of entries to return

        Returns:
            List of audit entries sorted by timestamp (newest first)
        """
        try:
            response = self._table.query(
                IndexName='document-index',
                KeyConditionExpression='document_id = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id},
                ScanIndexForward=False,  # Newest first
                Limit=limit
            )
            return self._parse_audit_entries(response.get('Items', []))
        except ClientError as e:
            print(f"Error querying audit trail: {e}")
            return []

    def get_audit_trail_by_user(self,
                                user_id: str,
                                limit: int = 100) -> List[Dict]:
        """
        Retrieves audit trail for a specific user.

        Args:
            user_id: User ID to query
            limit: Maximum number of entries to return

        Returns:
            List of audit entries sorted by timestamp (newest first)
        """
        try:
            response = self._table.query(
                IndexName='user-index',
                KeyConditionExpression='user_id = :user_id',
                ExpressionAttributeValues={':user_id': user_id},
                ScanIndexForward=False,
                Limit=limit
            )
            return self._parse_audit_entries(response.get('Items', []))
        except ClientError as e:
            print(f"Error querying audit trail: {e}")
            return []

    def get_audit_trail_by_date_range(self,
                                       start_date: str,
                                       end_date: str,
                                       document_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieves audit entries within a date range.

        Args:
            start_date: Start date (ISO 8601 format)
            end_date: End date (ISO 8601 format)
            document_id: Optional document ID filter

        Returns:
            List of audit entries within the date range
        """
        try:
            if document_id:
                response = self._table.query(
                    IndexName='document-index',
                    KeyConditionExpression='document_id = :doc_id AND #ts BETWEEN :start AND :end',
                    ExpressionAttributeNames={'#ts': 'timestamp'},
                    ExpressionAttributeValues={
                        ':doc_id': document_id,
                        ':start': start_date,
                        ':end': end_date
                    }
                )
            else:
                # Full table scan with filter (less efficient but required for date-only queries)
                response = self._table.scan(
                    FilterExpression='#ts BETWEEN :start AND :end',
                    ExpressionAttributeNames={'#ts': 'timestamp'},
                    ExpressionAttributeValues={
                        ':start': start_date,
                        ':end': end_date
                    }
                )
            return self._parse_audit_entries(response.get('Items', []))
        except ClientError as e:
            print(f"Error querying audit trail by date: {e}")
            return []

    def _parse_audit_entries(self, items: List[Dict]) -> List[Dict]:
        """Parses DynamoDB items into readable audit entries."""
        entries = []
        for item in items:
            entry = {
                'audit_id': item.get('audit_id'),
                'document_id': item.get('document_id'),
                'user_id': item.get('user_id'),
                'timestamp': item.get('timestamp'),
                'change_type': item.get('change_type'),
                'before_state': json.loads(item['before_state']) if item.get('before_state') else None,
                'after_state': json.loads(item['after_state']) if item.get('after_state') else None,
                'metadata': json.loads(item['metadata']) if item.get('metadata') else None
            }
            entries.append(entry)
        return entries

    def export_audit_trail_to_json(self,
                                   document_id: Optional[str] = None,
                                   user_id: Optional[str] = None,
                                   output_path: Optional[str] = None) -> str:
        """
        Exports audit trail to JSON format.

        Args:
            document_id: Filter by document (optional)
            user_id: Filter by user (optional)
            output_path: File path to save JSON (optional)

        Returns:
            JSON string of the audit trail
        """
        if document_id:
            entries = self.get_audit_trail_by_document(document_id)
        elif user_id:
            entries = self.get_audit_trail_by_user(user_id)
        else:
            # Get all entries (scan)
            try:
                response = self._table.scan()
                entries = self._parse_audit_entries(response.get('Items', []))
            except ClientError as e:
                print(f"Error scanning audit table: {e}")
                entries = []

        export_data = {
            'export_timestamp': datetime.utcnow().isoformat() + 'Z',
            'filter': {
                'document_id': document_id,
                'user_id': user_id
            },
            'total_entries': len(entries),
            'audit_entries': entries
        }

        json_str = json.dumps(export_data, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            print(f"Audit trail exported to {output_path}")

        return json_str


def get_current_user() -> str:
    """
    Gets the current user ID for audit logging.

    In a production environment, this would integrate with
    your authentication system (e.g., AWS Cognito, OAuth).

    Returns:
        str: User ID
    """
    import os
    import getpass

    # Try environment variable first (for SSO/OAuth integration)
    user_id = os.environ.get('CLINICAL_USER_ID')

    if not user_id:
        # Fall back to system username
        user_id = getpass.getuser()

    return user_id


# Singleton instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Returns the singleton AuditLogger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


if __name__ == "__main__":
    # Test the audit logger
    print("Testing DynamoDB Audit Logger...")

    logger = get_audit_logger()
    user = get_current_user()

    # Test logging a summary edit
    audit_id = logger.log_summary_edit(
        document_id="test_doc_001",
        user_id=user,
        before_summary="Original summary text",
        after_summary="Updated summary text with corrections"
    )
    print(f"Logged summary edit: {audit_id}")

    # Test logging a SNOMED status change
    audit_id = logger.log_snomed_status_change(
        document_id="test_doc_001",
        user_id=user,
        entity_text="Hypertension",
        snomed_code="38341003",
        before_status="Pending Review",
        after_status="Approved"
    )
    print(f"Logged SNOMED status change: {audit_id}")

    # Test retrieving audit trail
    trail = logger.get_audit_trail_by_document("test_doc_001")
    print(f"\nAudit trail for test_doc_001:")
    for entry in trail:
        print(f"  - {entry['timestamp']}: {entry['change_type']} by {entry['user_id']}")

    # Test export
    json_export = logger.export_audit_trail_to_json(document_id="test_doc_001")
    print(f"\nExported JSON:\n{json_export}")
