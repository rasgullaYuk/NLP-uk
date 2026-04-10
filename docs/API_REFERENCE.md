# API Reference

## REST endpoints

Primary OpenAPI source: `openapi/documents_api.json`

### Authentication

- API Key header: `x-api-key`
- Bearer header: `Authorization: Bearer <token>`

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| POST | `/documents/upload` | Upload PDF/TIFF/JPEG |
| GET | `/documents/{doc_id}/status` | Processing status |
| GET | `/documents/{doc_id}/extraction` | Extracted text |
| GET | `/documents/{doc_id}/snomed` | SNOMED mappings |
| GET | `/documents/{doc_id}/summary` | Clinical summary |
| PUT | `/documents/{doc_id}/approve` | Approve and export intent |
| GET | `/audit/{doc_id}` | Audit trail for document |

## Example request

```bash
curl -X GET "https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/documents/doc-123/status" \
  -H "x-api-key: <key>"
```

## Example response

```json
{
  "doc_id": "doc-123",
  "status": "approved",
  "approved_by": "clinician-a",
  "approved_at": "2026-04-10T06:45:00Z"
}
```

## Error model

```json
{
  "error": "not_found",
  "message": "Document not found"
}
```

## CORS

- `Access-Control-Allow-Origin`
- `Access-Control-Allow-Headers`
- `Access-Control-Allow-Methods`
