# HIPAA Compliance Checklist (SRS 5.2)

This document records PHI handling and HIPAA safeguard posture for the NLP pipeline components.

## PHI handling policy

1. **Detection**: Extracted clinical text is scanned with Amazon Comprehend Medical `DetectPHI`.
2. **Flagging**: PHI metadata (`phi_detected`, entity counts, PHI type breakdown, pseudonymized entities) is attached to pipeline outputs.
3. **Masking/pseudonymization**:
   - Logs and queue payloads are scrubbed using deterministic pseudonyms/tokens.
   - Intermediate/non-clinician outputs are masked.
   - Clinician-facing outputs keep clinically useful content; PHI metadata is still flagged.
4. **No raw PHI in diagnostics**: Debug/log output uses scrubbed text only.

## In-transit encryption (TLS 1.2+)

| Component | Control | Status |
|---|---|---|
| Shared AWS client/resource creation | `hipaa_compliance.create_secure_client/resource` validates HTTPS endpoint and runtime TLS 1.2 support | ✅ |
| Tier 1 Textract + PHI | Secure client wrappers | ✅ |
| Tier 2 Router + SNS/SQS | Secure client wrappers | ✅ |
| Track A (SQS + Comprehend Medical) | Secure client wrappers | ✅ |
| Track B (SQS + Bedrock + PHI) | Secure client wrappers | ✅ |
| Tier 3 Bedrock corrections | Secure client wrappers | ✅ |
| DynamoDB module + audit logger | Secure client/resource wrappers | ✅ |
| Lambda/EventBridge helpers | Secure client wrappers | ✅ |

## Encryption at rest

| Data store | Control | Status |
|---|---|---|
| DynamoDB (`dynamodb_module` tables) | `SSESpecification.Enabled=True` in table definitions | ✅ |
| DynamoDB (`ClinicalDocumentAuditLog`) | `SSESpecification.Enabled=True` during table creation | ✅ |
| S3 Data Lake bucket | Bucket default encryption (`AES256`) enabled during setup | ✅ |
| S3 object writes | `ServerSideEncryption=AES256` on `put_object` paths | ✅ |

## IAM least-privilege role artifacts

The following policy templates scope service permissions to required actions/resources:

- `iam/tier0-tier1-role-policy.json`
- `iam/router-role-policy.json`
- `iam/track-a-role-policy.json`
- `iam/track-b-role-policy.json`
- `iam/clinician-dashboard-role-policy.json`
- `iam/datalake-export-role-policy.json`

## Component posture summary

| Pipeline component | PHI detection | PHI masking/scrubbing | TLS 1.2+ | At-rest encryption | IAM policy template |
|---|---|---|---|---|---|
| Tier 1 (Textract extraction) | ✅ | N/A (flags attached) | ✅ | Output files local; downstream protected | ✅ |
| Tier 2 Router (SQS/SNS) | N/A | Queue payloads scrubbed where applicable | ✅ | AWS-managed queue storage | ✅ |
| Track A (SNOMED mapping) | ✅ | Logs scrubbed; PHI metadata pseudonymized | ✅ | File outputs + DynamoDB protections | ✅ |
| Track B (Summaries) | ✅ | Non-clinician summaries masked; clinician summary retained | ✅ | File outputs + S3/DynamoDB protections | ✅ |
| Tier 3 OCR correction | N/A | Audit payloads can be scrubbed before persistence | ✅ | DynamoDB SSE | ✅ |
| Audit Logger | N/A | Before/after states scrubbed on write | ✅ | DynamoDB SSE | ✅ |
| Data Lake Export | ✅ (summary field masking flow) | Export payloads anonymized/scrubbed | ✅ | S3 SSE + bucket encryption | ✅ |
| Daily Export Lambda | Inherited from exporter | Inherited from exporter | ✅ | S3 SSE + DynamoDB SSE | ✅ |

## Sign-off

- **Security owner**: ______________________
- **Clinical safety owner**: ______________________
- **Date**: ______________________
- **Status**: ✅ Ready for sign-off
