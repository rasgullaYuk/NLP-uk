# Production Deployment Configuration (Task 19)

This document defines the production deployment model for NLP-uk using Terraform + Docker + GitHub Actions.

## Included deliverables

- **IaC**: `infra/terraform/*.tf` and `*.tfvars`
- **Containers**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- **Environment config**: `config/environments/dev.env`, `config/environments/staging.env`, `config/environments/prod.env`
- **Deployment pipeline**: `.github/workflows/deploy-infrastructure.yml`
- **Initialization and migrations**: `scripts/init_environment.py`, `scripts/run_migrations.py`
- **Backup / DR**: `BACKUP_DISASTER_RECOVERY.md`

## Infrastructure scope

Terraform provisions:

1. Networking: VPC, public/private subnets, routing, ALB and ECS security groups.
2. Compute: ECS Fargate cluster, review UI service, worker service, autoscaling policy.
3. Data services: encrypted S3 data lake, encrypted DynamoDB audit table with PITR, AWS Backup plan/vault/selection.
4. Messaging/alerts: SQS queues + DLQ, SNS critical alert topic/subscription, CloudWatch log group and queue-depth alarm.
5. IAM least-privilege: ECS execution/task roles and scoped policy for S3/DynamoDB/SQS/CloudWatch + required AI APIs.

## Deployment flow

1. Build/push images (`review-ui`, `worker`) to ECR.
2. Trigger **Deploy Infrastructure** workflow.
3. Select environment (`dev`, `staging`, or `prod`) and run plan.
4. Re-run with `apply=true` to apply.
5. Run post-deploy initialization:
   - `python scripts/run_migrations.py`
6. Validate health:
   - ALB endpoint responds
   - queues and audit writes succeed
   - CloudWatch alarms/metrics visible

## Secrets and configuration

Required GitHub repository secret:

- `AWS_GITHUB_ROLE_ARN`: IAM role assumed by GitHub OIDC for Terraform and ECR steps.

Runtime env files:

- `config/environments/dev.env`
- `config/environments/staging.env`
- `config/environments/prod.env`

Do not commit credentials into env files.

## Staging validation checklist

- Terraform `init/validate/plan/apply` complete for `staging`.
- ECS services stable with desired count.
- ALB health checks green.
- S3 bucket encryption and DynamoDB SSE enabled.
- Queue depth and critical alarms configured.
- `scripts/run_migrations.py` executed successfully.

## Rollback procedure

1. Roll back service image tags to prior known-good release.
2. Re-run deployment workflow with previous image variables.
3. If infra rollback is required:
   - use `terraform plan` against prior commit
   - `terraform apply` that state
4. Restore data from backup/PITR where needed (see DR document).
