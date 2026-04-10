# Deployment and Operations Runbook

## Deployment prerequisites

- Docker available
- Terraform >= 1.5
- AWS credentials with required infra and runtime privileges
- GitHub Actions secret: `AWS_GITHUB_ROLE_ARN`

## Staging deployment steps

1. Build container images and push to ECR.
2. Run Terraform:
   ```bash
   cd infra/terraform
   terraform init
   terraform validate
   terraform plan -var-file=staging.tfvars
   terraform apply -var-file=staging.tfvars
   ```
3. Initialize runtime resources:
   ```bash
   python scripts/run_migrations.py
   ```
4. Validate health:
   - ALB endpoint
   - SQS queue connectivity
   - DynamoDB audit writes
   - CloudWatch alarms and dashboards

## Local/dev run

```bash
docker compose up --build
```

## Rollback procedure

1. Redeploy previous known-good image tags.
2. Re-run workflow with prior immutable artifact references.
3. If infra rollback is needed, apply Terraform from prior commit.
4. Restore data from backup/PITR where required.

## Operational checks

- Queue depth and DLQ trends
- Latency and error-rate alarms
- Export retry queue backlog
- Audit logging throughput and failures
