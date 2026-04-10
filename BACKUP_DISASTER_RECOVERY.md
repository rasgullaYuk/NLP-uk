# Backup and Disaster Recovery Plan

## Objectives

- Protect audit and training data against accidental deletion/corruption.
- Meet operational recovery requirements for production incidents.

## Backup strategy

1. **DynamoDB**
   - Point-in-time recovery enabled in Terraform.
   - AWS Backup daily job configured for 30-day retention.

2. **S3 Data Lake**
   - Bucket versioning enabled.
   - Server-side encryption enabled.
   - Lifecycle policy configured for retention and archival behavior.

3. **Infrastructure state**
   - Terraform state should be stored in a remote backend (S3 + DynamoDB lock table) in production.
   - Protect backend bucket with versioning and restricted IAM.

## Disaster scenarios and response

1. **Application regression**
   - Roll back ECS service images to prior stable tags.
   - Verify service health and queue processing.

2. **Data corruption in DynamoDB**
   - Use PITR to restore table to a timestamp before incident.
   - Repoint services or replay required audit records.

3. **S3 object deletion/corruption**
   - Restore object versions from S3 version history.
   - Validate downstream dataset integrity.

4. **Region/service outage**
   - Rehydrate infrastructure in failover region using Terraform variables.
   - Restore data from backups and latest artifacts.

## Recovery validation

Run a quarterly DR drill:

- Restore a sample DynamoDB table from PITR.
- Restore versioned S3 objects.
- Deploy stack in non-prod environment from Terraform.
- Execute smoke tests for ingestion, routing, and review UI.
