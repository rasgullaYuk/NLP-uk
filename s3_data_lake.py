"""
S3 Data Lake & Continuous Learning Loop

Sets up Amazon S3 data lake for storing validated clinician edits and corrections.
Exports DynamoDB feedback into training datasets for model fine-tuning.

Features:
- S3 bucket with versioning and lifecycle policies
- Daily export from DynamoDB to S3 (Parquet format)
- PII detection and anonymization
- Dataset organization by document type and category
- SageMaker integration for model fine-tuning
"""

import boto3
import json
import os
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError

# Configuration
AWS_REGION = "us-east-1"
DATA_LAKE_BUCKET = "clinical-nlp-data-lake"
DYNAMODB_AUDIT_TABLE = "ClinicalDocumentAuditLog"

# Dataset paths in S3
DATASET_PATHS = {
    "ner": "datasets/ner/",
    "layout": "datasets/layout/",
    "summarization": "datasets/summarization/",
    "feedback": "feedback/raw/",
    "exports": "exports/"
}


class S3DataLake:
    """
    Manages the S3 data lake for clinical NLP training data.
    """

    def __init__(self, bucket_name: str = DATA_LAKE_BUCKET):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        self.s3_resource = boto3.resource('s3', region_name=AWS_REGION)

    def create_bucket(self) -> bool:
        """
        Creates the S3 bucket with versioning and lifecycle policies.

        Returns:
            bool: True if bucket created/exists, False on error
        """
        try:
            # Check if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket '{self.bucket_name}' already exists.")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if AWS_REGION == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                        )
                    print(f"Bucket '{self.bucket_name}' created successfully.")

                    # Enable versioning
                    self._enable_versioning()

                    # Set lifecycle policy
                    self._set_lifecycle_policy()

                    # Create folder structure
                    self._create_folder_structure()

                    return True
                except ClientError as create_error:
                    print(f"Error creating bucket: {create_error}")
                    return False
            else:
                print(f"Error checking bucket: {e}")
                return False

    def _enable_versioning(self):
        """Enables versioning on the bucket."""
        try:
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            print("Versioning enabled.")
        except ClientError as e:
            print(f"Error enabling versioning: {e}")

    def _set_lifecycle_policy(self):
        """Sets lifecycle policy for data retention."""
        lifecycle_policy = {
            'Rules': [
                {
                    'ID': 'MoveToGlacierAfter90Days',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'exports/'},
                    'Transitions': [
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                },
                {
                    'ID': 'DeleteOldVersionsAfter365Days',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': ''},
                    'NoncurrentVersionExpiration': {
                        'NoncurrentDays': 365
                    }
                },
                {
                    'ID': 'AbortIncompleteMultipartUploads',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': ''},
                    'AbortIncompleteMultipartUpload': {
                        'DaysAfterInitiation': 7
                    }
                }
            ]
        }

        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_policy
            )
            print("Lifecycle policy set.")
        except ClientError as e:
            print(f"Error setting lifecycle policy: {e}")

    def _create_folder_structure(self):
        """Creates the folder structure in S3."""
        for path in DATASET_PATHS.values():
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=path,
                    Body=''
                )
            except ClientError as e:
                print(f"Error creating folder {path}: {e}")

        print("Folder structure created.")

    def upload_dataset(self,
                       data: bytes,
                       dataset_type: str,
                       filename: str,
                       metadata: Optional[Dict] = None) -> str:
        """
        Uploads a dataset file to the appropriate S3 path.

        Args:
            data: File content as bytes
            dataset_type: Type of dataset (ner, layout, summarization, feedback)
            filename: Name of the file
            metadata: Optional metadata dict

        Returns:
            str: S3 URI of uploaded file
        """
        if dataset_type not in DATASET_PATHS:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        key = f"{DATASET_PATHS[dataset_type]}{filename}"

        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                **extra_args
            )

            s3_uri = f"s3://{self.bucket_name}/{key}"
            print(f"Uploaded: {s3_uri}")
            return s3_uri

        except ClientError as e:
            print(f"Error uploading {filename}: {e}")
            raise

    def list_datasets(self, dataset_type: str) -> List[Dict]:
        """
        Lists all datasets of a given type.

        Args:
            dataset_type: Type of dataset to list

        Returns:
            List of dataset metadata dicts
        """
        if dataset_type not in DATASET_PATHS:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        prefix = DATASET_PATHS[dataset_type]
        datasets = []

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'] != prefix:  # Skip folder itself
                        datasets.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            's3_uri': f"s3://{self.bucket_name}/{obj['Key']}"
                        })
        except ClientError as e:
            print(f"Error listing datasets: {e}")

        return datasets


class PIIAnonymizer:
    """
    Detects and anonymizes Personally Identifiable Information (PII).
    """

    # Common PII patterns
    PATTERNS = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'nhs_number': r'\d{3}[-\s]?\d{3}[-\s]?\d{4}',
        'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'postcode': r'[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}',
        'mrn': r'MRN[:\s]?\d{6,10}',
    }

    # Name patterns (basic - would need NER for better detection)
    NAME_PREFIXES = ['Mr', 'Mrs', 'Ms', 'Miss', 'Dr', 'Prof', 'Patient']

    @classmethod
    def anonymize(cls, text: str) -> tuple:
        """
        Anonymizes PII in text.

        Args:
            text: Input text to anonymize

        Returns:
            tuple: (anonymized_text, list of detected PII types)
        """
        if not text:
            return text, []

        anonymized = text
        detected_pii = []

        # Replace pattern-based PII
        for pii_type, pattern in cls.PATTERNS.items():
            matches = re.findall(pattern, anonymized, re.IGNORECASE)
            if matches:
                detected_pii.append(pii_type)
                anonymized = re.sub(
                    pattern,
                    f'[{pii_type.upper()}_REDACTED]',
                    anonymized,
                    flags=re.IGNORECASE
                )

        # Basic name detection (after prefixes)
        for prefix in cls.NAME_PREFIXES:
            pattern = rf'\b{prefix}\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?'
            if re.search(pattern, anonymized):
                detected_pii.append('name')
                anonymized = re.sub(pattern, f'{prefix}. [NAME_REDACTED]', anonymized)

        return anonymized, list(set(detected_pii))

    @classmethod
    def hash_identifier(cls, identifier: str) -> str:
        """
        Creates a consistent hash for an identifier (for linking records).

        Args:
            identifier: The identifier to hash

        Returns:
            str: SHA-256 hash of the identifier
        """
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


class DynamoDBToS3Exporter:
    """
    Exports DynamoDB audit data to S3 in Parquet format.
    """

    def __init__(self, data_lake: S3DataLake):
        self.data_lake = data_lake
        self.dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
        self.table = self.dynamodb.Table(DYNAMODB_AUDIT_TABLE)
        self.anonymizer = PIIAnonymizer()

    def export_daily_feedback(self,
                              export_date: Optional[str] = None,
                              anonymize: bool = True) -> Dict:
        """
        Exports a day's worth of feedback from DynamoDB to S3.

        Args:
            export_date: Date to export (YYYY-MM-DD), defaults to yesterday
            anonymize: Whether to anonymize PII

        Returns:
            dict: Export summary
        """
        if export_date is None:
            export_date = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

        start_timestamp = f"{export_date}T00:00:00Z"
        end_timestamp = f"{export_date}T23:59:59Z"

        print(f"Exporting feedback for {export_date}...")

        # Scan DynamoDB for the date range
        items = self._scan_date_range(start_timestamp, end_timestamp)

        if not items:
            print(f"No data found for {export_date}")
            return {'date': export_date, 'records': 0, 'status': 'no_data'}

        # Process and categorize data
        processed_data = self._process_feedback(items, anonymize)

        # Quality checks
        quality_report = self._run_quality_checks(processed_data)

        if not quality_report['passed']:
            print(f"Quality checks failed: {quality_report['issues']}")
            return {
                'date': export_date,
                'records': len(items),
                'status': 'quality_check_failed',
                'issues': quality_report['issues']
            }

        # Export to Parquet and upload to S3
        export_results = self._export_to_parquet(processed_data, export_date)

        # Log the export
        self._log_export(export_date, export_results)

        return {
            'date': export_date,
            'records': len(items),
            'status': 'success',
            'exports': export_results,
            'quality_report': quality_report
        }

    def _scan_date_range(self, start: str, end: str) -> List[Dict]:
        """Scans DynamoDB for records in a date range."""
        items = []

        try:
            response = self.table.scan(
                FilterExpression='#ts BETWEEN :start AND :end',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':start': start,
                    ':end': end
                }
            )
            items.extend(response.get('Items', []))

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    FilterExpression='#ts BETWEEN :start AND :end',
                    ExpressionAttributeNames={'#ts': 'timestamp'},
                    ExpressionAttributeValues={
                        ':start': start,
                        ':end': end
                    },
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))

        except ClientError as e:
            print(f"Error scanning DynamoDB: {e}")

        return items

    def _process_feedback(self, items: List[Dict], anonymize: bool) -> Dict:
        """
        Processes and categorizes feedback data.

        Returns:
            dict: Categorized data for each model type
        """
        categorized = {
            'ner': [],           # SNOMED status changes
            'layout': [],        # Layout corrections (future)
            'summarization': [], # Summary edits
            'general': []        # Other feedback
        }

        for item in items:
            change_type = item.get('change_type', '')
            before_state = json.loads(item.get('before_state', '{}') or '{}')
            after_state = json.loads(item.get('after_state', '{}') or '{}')
            metadata = json.loads(item.get('metadata', '{}') or '{}')

            # Anonymize if requested
            if anonymize:
                if 'summary' in before_state:
                    before_state['summary'], _ = self.anonymizer.anonymize(before_state['summary'])
                if 'summary' in after_state:
                    after_state['summary'], _ = self.anonymizer.anonymize(after_state['summary'])

                # Hash user_id for privacy
                item['user_id'] = self.anonymizer.hash_identifier(item.get('user_id', ''))

            record = {
                'audit_id': item.get('audit_id'),
                'document_id': item.get('document_id'),
                'user_id': item.get('user_id'),
                'timestamp': item.get('timestamp'),
                'change_type': change_type,
                'before_state': before_state,
                'after_state': after_state,
                'metadata': metadata
            }

            # Categorize by type
            if change_type == 'SNOMED_STATUS':
                categorized['ner'].append(record)
            elif change_type == 'SUMMARY_EDIT':
                categorized['summarization'].append(record)
            else:
                categorized['general'].append(record)

        return categorized

    def _run_quality_checks(self, data: Dict) -> Dict:
        """
        Runs data quality checks before export.

        Returns:
            dict: Quality check report
        """
        issues = []

        total_records = sum(len(v) for v in data.values())

        if total_records == 0:
            issues.append("No records to export")

        # Check for required fields
        for category, records in data.items():
            for record in records:
                if not record.get('audit_id'):
                    issues.append(f"Missing audit_id in {category}")
                if not record.get('timestamp'):
                    issues.append(f"Missing timestamp in {category}")

        # Check for residual PII (basic check)
        pii_patterns = ['@', 'NHS', 'MRN']
        for category, records in data.items():
            for record in records:
                record_str = json.dumps(record)
                for pattern in pii_patterns:
                    if pattern in record_str and 'REDACTED' not in record_str:
                        issues.append(f"Possible PII detected in {category}: {pattern}")

        return {
            'passed': len(issues) == 0,
            'total_records': total_records,
            'issues': issues[:10]  # Limit to first 10 issues
        }

    def _export_to_parquet(self, data: Dict, export_date: str) -> Dict:
        """
        Exports data to Parquet format and uploads to S3.

        Returns:
            dict: Export results per category
        """
        results = {}

        for category, records in data.items():
            if not records:
                continue

            # Convert to JSON (Parquet requires pyarrow, fall back to JSON)
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                import io

                # Create PyArrow table
                table = pa.Table.from_pylist(records)

                # Write to buffer
                buffer = io.BytesIO()
                pq.write_table(table, buffer)
                buffer.seek(0)

                filename = f"{export_date}_{category}.parquet"
                file_data = buffer.getvalue()

            except ImportError:
                # Fall back to JSON if PyArrow not installed
                print("PyArrow not installed, using JSON format")
                filename = f"{export_date}_{category}.json"
                file_data = json.dumps(records, indent=2).encode('utf-8')

            # Determine dataset type
            if category in ['ner', 'layout', 'summarization']:
                dataset_type = category
            else:
                dataset_type = 'feedback'

            # Upload to S3
            try:
                s3_uri = self.data_lake.upload_dataset(
                    data=file_data,
                    dataset_type=dataset_type,
                    filename=filename,
                    metadata={
                        'export_date': export_date,
                        'record_count': str(len(records)),
                        'category': category
                    }
                )
                results[category] = {
                    'records': len(records),
                    's3_uri': s3_uri,
                    'status': 'success'
                }
            except Exception as e:
                results[category] = {
                    'records': len(records),
                    'status': 'failed',
                    'error': str(e)
                }

        return results

    def _log_export(self, export_date: str, results: Dict):
        """Logs the export to S3 for audit trail."""
        log_entry = {
            'export_date': export_date,
            'export_timestamp': datetime.utcnow().isoformat() + 'Z',
            'results': results
        }

        log_filename = f"export_log_{export_date}.json"
        log_data = json.dumps(log_entry, indent=2).encode('utf-8')

        try:
            self.data_lake.s3_client.put_object(
                Bucket=self.data_lake.bucket_name,
                Key=f"{DATASET_PATHS['exports']}logs/{log_filename}",
                Body=log_data
            )
        except ClientError as e:
            print(f"Error logging export: {e}")


class SageMakerIntegration:
    """
    Provides helpers for SageMaker integration and model fine-tuning.
    """

    def __init__(self, data_lake: S3DataLake):
        self.data_lake = data_lake
        self.sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)

    def get_training_data_uri(self, model_type: str) -> str:
        """
        Gets the S3 URI for training data of a specific model type.

        Args:
            model_type: ner, layout, or summarization

        Returns:
            str: S3 URI prefix for training data
        """
        if model_type not in ['ner', 'layout', 'summarization']:
            raise ValueError(f"Invalid model type: {model_type}")

        return f"s3://{self.data_lake.bucket_name}/{DATASET_PATHS[model_type]}"

    def create_training_manifest(self, model_type: str) -> str:
        """
        Creates a training manifest file for SageMaker.

        Args:
            model_type: Type of model to create manifest for

        Returns:
            str: S3 URI of the manifest file
        """
        datasets = self.data_lake.list_datasets(model_type)

        manifest_entries = []
        for dataset in datasets:
            manifest_entries.append({
                'source': dataset['s3_uri'],
                'size': dataset['size'],
                'last_modified': dataset['last_modified']
            })

        manifest = {
            'model_type': model_type,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'dataset_count': len(manifest_entries),
            'datasets': manifest_entries
        }

        manifest_filename = f"{model_type}_training_manifest.json"
        manifest_data = json.dumps(manifest, indent=2).encode('utf-8')

        s3_uri = self.data_lake.upload_dataset(
            data=manifest_data,
            dataset_type=model_type,
            filename=manifest_filename,
            metadata={'type': 'manifest'}
        )

        return s3_uri

    def get_sagemaker_config(self, model_type: str) -> Dict:
        """
        Returns SageMaker training configuration for a model type.

        Args:
            model_type: ner, layout, or summarization

        Returns:
            dict: SageMaker training job configuration
        """
        training_uri = self.get_training_data_uri(model_type)

        config = {
            'TrainingJobName': f'clinical-{model_type}-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            'HyperParameters': {},
            'InputDataConfig': [
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': training_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f"s3://{self.data_lake.bucket_name}/models/{model_type}/"
            },
            'ResourceConfig': {
                'InstanceType': 'ml.p3.2xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 50
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400
            }
        }

        # Model-specific configurations
        if model_type == 'ner':
            config['HyperParameters'] = {
                'epochs': '10',
                'learning_rate': '2e-5',
                'batch_size': '16'
            }
        elif model_type == 'summarization':
            config['HyperParameters'] = {
                'epochs': '5',
                'learning_rate': '3e-5',
                'batch_size': '8',
                'max_length': '512'
            }

        return config


def setup_data_lake() -> S3DataLake:
    """
    Sets up the S3 data lake with all required configurations.

    Returns:
        S3DataLake: Configured data lake instance
    """
    data_lake = S3DataLake()
    data_lake.create_bucket()
    return data_lake


def run_daily_export(anonymize: bool = True) -> Dict:
    """
    Runs the daily export from DynamoDB to S3.

    Args:
        anonymize: Whether to anonymize PII (default True)

    Returns:
        dict: Export results
    """
    data_lake = S3DataLake()
    exporter = DynamoDBToS3Exporter(data_lake)
    return exporter.export_daily_feedback(anonymize=anonymize)


if __name__ == "__main__":
    print("=" * 60)
    print("S3 Data Lake Setup & Export")
    print("=" * 60)

    # Setup data lake
    print("\n1. Setting up S3 Data Lake...")
    data_lake = setup_data_lake()

    # Test PII anonymization
    print("\n2. Testing PII Anonymization...")
    test_text = "Patient Mr. John Smith (MRN: 12345678) was seen on 01/15/2024. Email: john.smith@email.com"
    anonymized, pii_types = PIIAnonymizer.anonymize(test_text)
    print(f"   Original: {test_text}")
    print(f"   Anonymized: {anonymized}")
    print(f"   PII detected: {pii_types}")

    # Run daily export
    print("\n3. Running Daily Export...")
    results = run_daily_export()
    print(f"   Export results: {json.dumps(results, indent=2)}")

    # SageMaker integration
    print("\n4. SageMaker Integration...")
    sm = SageMakerIntegration(data_lake)
    print(f"   NER training data: {sm.get_training_data_uri('ner')}")
    print(f"   Summarization training data: {sm.get_training_data_uri('summarization')}")

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
