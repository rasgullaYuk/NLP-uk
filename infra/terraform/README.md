# Terraform Deployment

## Prerequisites

- Terraform >= 1.5
- AWS credentials/role with permissions to create ECS, ALB, IAM, S3, DynamoDB, SQS, SNS, CloudWatch, Backup resources

## Quickstart

```bash
cd infra/terraform
terraform init
terraform validate
terraform plan -var-file=staging.tfvars
terraform apply -var-file=staging.tfvars
```

## Environments

- `dev.tfvars`
- `staging.tfvars`
- `prod.tfvars`

## Post-deploy

```bash
python scripts/run_migrations.py
```
