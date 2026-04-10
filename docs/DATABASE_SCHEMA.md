# Database Schema

## DynamoDB tables

### `ClinicalDocumentAuditLog`

Stores all review, decision, and export events.

Primary key:
- `audit_id` (HASH)

GSIs:
- `document-index` (`document_id` HASH + `timestamp` RANGE)
- `user-index` (`user_id` HASH + `timestamp` RANGE)

Core attributes:
- `document_id`
- `user_id`
- `timestamp`
- `change_type`
- `before_state` (JSON string)
- `after_state` (JSON string)
- `metadata` (JSON string)

### `...-audit-log` (Terraform-managed environment tables)

Equivalent environment-specific table naming through IaC.

## ER diagram (logical)

```mermaid
erDiagram
    DOCUMENT ||--o{ AUDIT_ENTRY : has
    USER ||--o{ AUDIT_ENTRY : performs
    DOCUMENT ||--o{ EXPORT_EVENT : emits

    DOCUMENT {
      string document_id PK
      string source_uri
      string status
    }

    USER {
      string user_id PK
      string role
    }

    AUDIT_ENTRY {
      string audit_id PK
      string document_id
      string user_id
      string timestamp
      string change_type
      string before_state_json
      string after_state_json
      string metadata_json
    }

    EXPORT_EVENT {
      string event_id PK
      string document_id
      string status
      string transport
      string response_summary
      string timestamp
    }
```
