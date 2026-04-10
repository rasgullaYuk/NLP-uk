# EMIS Platform Export Integration

Implements SRS Section 7 Step 11: exporting clinician-approved validated JSON to EMIS.

## Flow

1. Clinician clicks **Approve All & Export to EMIS** in `app.py`.
2. App builds validated structured payload and calls `export_to_emis(...)`.
3. Export module transmits to EMIS using configured transport:
   - API (`EMIS_TRANSPORT=api`)
   - File transfer via encrypted S3 drop (`EMIS_TRANSPORT=s3_file_drop`)
4. Every export attempt/event is logged to DynamoDB audit trail.
5. Failures retry automatically (`EMIS_MAX_EXPORT_ATTEMPTS`), then queue to `EMIS_Export_Retry_Queue`.
6. `emis_retry_worker.py` processes queued retries and logs unresolved failures.

## Configuration

- `EMIS_EXPORT_FORMAT` = `hl7_fhir` or `proprietary_json`
- `EMIS_TRANSPORT` = `api` or `s3_file_drop`
- `EMIS_API_BASE_URL` (must be `https://`)
- `EMIS_API_PATH`
- `EMIS_API_TOKEN`
- `EMIS_FILE_DROP_BUCKET`
- `EMIS_RETRY_QUEUE_NAME`
- `EMIS_MAX_EXPORT_ATTEMPTS`
- `EMIS_MAX_RETRY_QUEUE_ATTEMPTS`

## TLS and PHI

- API mode enforces HTTPS endpoint check before transmission.
- AWS clients use secure client factory with TLS/runtime checks.
- Export events are scrubbed before logging.

## Audit trail events

- `EMIS_EXPORT_EVENT` with statuses:
  - `SUCCESS`
  - `FAILED_ATTEMPT`
  - `QUEUED_FOR_RETRY`
- `EMIS_EXPORT_RETRY_EVENT` with statuses:
  - `SUCCESS`
  - `FAILED_UNRESOLVED`

## Testing

Run:

```bash
pytest -q -o addopts='' test_emis_export_integration.py
```
