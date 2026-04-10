# REST API Endpoints via API Gateway (Task 20)

## Endpoints

- `POST /documents/upload` — upload PDF/TIFF/JPEG (binary/base64 API Gateway proxy body)
- `GET /documents/{doc_id}/status` — processing status
- `GET /documents/{doc_id}/extraction` — extracted text
- `GET /documents/{doc_id}/snomed` — SNOMED mapping
- `GET /documents/{doc_id}/summary` — summary output
- `PUT /documents/{doc_id}/approve` — approve document
- `GET /audit/{doc_id}` — audit trail retrieval

Implementation:

- Handler: `api_gateway_rest.py`
- Provisioning helper: `api_gateway_setup.py`
- OpenAPI spec: `openapi/documents_api.json`

## Security

- Supports API key (`x-api-key`) and OAuth bearer token (`Authorization: Bearer ...`).
- Configure allowed credentials via:
  - `API_ALLOWED_KEYS`
  - `API_ALLOWED_BEARER_TOKENS`
- CORS headers are returned on all responses.

## Rate limiting

- API Gateway usage plan + throttling is provisioned in `api_gateway_setup.py`.
- Defaults:
  - rate limit: `20 req/sec`
  - burst: `40`
  - quota: `10,000/day`

## Error handling

- Standard JSON response body:
  - `error` code
  - `message`
- HTTP status codes: `400`, `401`, `404`, `405`, `500`.

## Setup

```bash
python api_gateway_setup.py --api-name nlp-uk-clinical-api --stage prod --openapi openapi\documents_api.json
```

The setup command returns:

- REST API ID
- invoke URL
- usage plan ID
- API key ID and generated API key value
