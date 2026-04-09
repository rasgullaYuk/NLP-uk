# Integration Testing (Task 17)

This suite validates end-to-end stage handoffs from ingestion to confidence routing.

## Test file

- `test_pipeline_integration.py`

## Covered flows

1. PDF/image preprocessing to Tier 1 extraction output
2. Low-confidence routing from Tier 2 to Tier 3 correction handling
3. Track A SNOMED mapping with semantic fallback path
4. Track B RAG summarization pipeline path
5. Unified confidence routing decision path
6. DynamoDB audit logging write path
7. Retry behavior on transient failures
8. Mocked performance benchmark assertion for routing path

## Run locally

```bash
pytest -q test_pipeline_integration.py
```

## Optional staging mode

Set `STAGING_AWS_TESTS=1` to enable staging smoke tests:

```bash
STAGING_AWS_TESTS=1 pytest -q test_pipeline_integration.py
```

By default, staging tests are skipped to keep CI deterministic.
