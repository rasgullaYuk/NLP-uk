# Troubleshooting Guide

## Common issues

### 1. API returns `401 unauthorized`

**Symptoms**
- REST requests fail with auth error.

**Checks**
- Ensure `x-api-key` is in `API_ALLOWED_KEYS`, or bearer token is in `API_ALLOWED_BEARER_TOKENS`.
- Confirm API Gateway usage plan key is active.

### 2. EMIS export fails and queues for retry

**Symptoms**
- UI shows “queued for retry”.

**Checks**
- Verify `EMIS_API_BASE_URL` uses HTTPS.
- Validate `EMIS_API_TOKEN` and endpoint path.
- Check SQS retry queue backlog and `emis_retry_worker.py` process.

### 3. Audit log entries missing

**Symptoms**
- No review/export events in audit trail.

**Checks**
- Confirm DynamoDB table exists and IAM includes `PutItem/Query`.
- Verify `get_audit_logger()` initializes successfully.

### 4. Low confidence routes unexpectedly

**Symptoms**
- Docs route to human review more often than expected.

**Checks**
- Inspect component scores from confidence payload.
- Validate configured threshold (`FINAL_CONFIDENCE_THRESHOLD`).
- Check normalization of percent vs 0-1 confidence inputs.

### 5. Secrets Manager access denied

**Symptoms**
- Config loading fails when `load_secrets=True`.

**Checks**
- Ensure IAM has `secretsmanager:GetSecretValue` on listed secret names.
- Confirm region and secret names in runtime config.

## Quick diagnostics commands

```bash
pytest -q -o addopts='' test_pipeline_integration.py
pytest -q -o addopts='' test_api_gateway_rest.py
pytest -q -o addopts='' test_emis_export_integration.py
```
