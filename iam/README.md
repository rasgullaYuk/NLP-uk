# IAM least-privilege policy templates

These JSON policies are intended as attachable inline/managed policy templates for each pipeline service role:

- `tier0-tier1-role-policy.json` — Tier 0/Tier 1 extraction + PHI detection
- `router-role-policy.json` — Tier 2 routing (SQS/SNS fanout and queue wiring)
- `track-a-role-policy.json` — Track A SNOMED/PHI processing and DLQ
- `track-b-role-policy.json` — Track B summarization and PHI masking
- `clinician-dashboard-role-policy.json` — clinician review dashboard audit access
- `datalake-export-role-policy.json` — data lake export and S3 encryption operations

Replace account/resource ARNs as required before deployment.
