# AWS Service Configuration Guide

## Core services

1. **Textract**: OCR and query extraction.
2. **Comprehend Medical**: PHI and medical concept detection.
3. **Bedrock**: Summarization and OCR correction models.
4. **SQS**: Pipeline routing queues and retry queues.
5. **DynamoDB**: Audit trail persistence.
6. **S3**: Data lake, exports, and optional EMIS file-drop transport.
7. **CloudWatch/SNS**: Metrics, alarms, and notifications.
8. **API Gateway + Lambda**: REST API endpoint layer.
9. **ECS/ALB**: Production service runtime.

## Security baseline

- Enforce TLS 1.2+ through secure client factory (`hipaa_compliance.py`).
- Enable S3 server-side encryption.
- Enable DynamoDB SSE + PITR.
- Use least-privilege IAM policy attachments.
- Store secrets in AWS Secrets Manager; load via centralized config.

## Required environment variables

- `AWS_REGION`
- `FINAL_CONFIDENCE_THRESHOLD`
- `EMIS_TRANSPORT`
- `EMIS_API_BASE_URL`
- `EMIS_API_TOKEN`
- `EMIS_FILE_DROP_BUCKET`
- `API_ALLOWED_KEYS` or `API_ALLOWED_BEARER_TOKENS`

## Recommended IAM scopes

- Textract/ComprehendMedical/Bedrock invoke actions.
- SQS send/receive/delete on app queues.
- DynamoDB read/write on audit table and indexes.
- S3 read/write scoped to project buckets and prefixes.
- CloudWatch metric publish + log stream writes.
