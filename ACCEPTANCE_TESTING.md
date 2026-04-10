# Acceptance Testing (Task 18)

This suite validates SRS acceptance criteria across extraction quality, editability, auditability, routing, performance, and security.

## Test assets

- Framework: `acceptance_framework.py`
- Dataset: `acceptance_data/srs_acceptance_dataset.json`
- Tests: `test_srs_acceptance.py`

## Covered SRS criteria

1. **Text extraction accuracy**: validates `>=98%` similarity on the acceptance dataset.
2. **Full editability**: verifies UI editability contract for summary/SNOMED/action fields in `app.py`.
3. **Audit trail capture**: verifies before/after states are recorded for edit events.
4. **Confidence routing**: validates routing behavior at threshold `0.85`.
5. **Performance**: validates end-to-end acceptance workload runtime `<60s`.
6. **Security**: validates PHI masking behavior and encryption posture assertions.
7. **SNOMED mapping**: validates `>=95%` mapping accuracy on acceptance dataset.

## Run locally

```bash
pytest -q test_srs_acceptance.py
```

## Reporting

Use `acceptance_framework.save_acceptance_report(...)` to persist structured acceptance results in JSON for sign-off workflows.
