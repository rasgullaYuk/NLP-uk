# Module Reference

## Tier and Track modules

### `document_handler.py`
- Handles PDF/image intake.
- Splits PDFs into per-page images.
- Normalizes document naming.

### `preprocessing.py`
- Applies image cleanup and deskewing.
- Produces `_CLEANED` page assets for Textract.

### `tier1_textract.py`
- Extracts OCR + forms/tables/queries via Textract.
- Adds PHI detection metadata.
- Publishes extraction monitoring metrics.

### `tier2_router.py`
- Computes extraction confidence.
- Routes high-confidence docs to Track A/B and low-confidence docs to Tier 2.

### `tier2_layoutlmv3_refinement.py`
- Applies LayoutLMv3 refinement for low-confidence OCR regions.

### `tier3_ocr_correction/*`
- Vision-LLM correction and hallucination guard.
- Produces review-required signals for ambiguous corrections.

### `track_a_snomed.py`
- Clinical entity extraction + SNOMED mapping.
- Semantic fallback with SapBERT + FAISS + reranking.

### `track_b_summarization.py`
- RAG-based role-specific summarization.
- Prompt version tracking and validation integration.

### `track_b_validation.py`
- Rule/schema validation and hallucination checks.
- OCR deviation guard.

### `lambda_confidence_aggregator.py`
- Weighted confidence aggregation.
- Routes to bypass vs human review at threshold (default 0.85).

## UI and review

### `app.py`
- Streamlit review UI.
- Editable summaries/SNOMED/actions.
- Approval/flag/audit export controls.
- EMIS export trigger on approval.

### `review_interface_utils.py`
- Asset discovery/load helpers.
- Confidence visuals and action normalization.

### `audit_dynamodb.py`
- DynamoDB audit logger with document/user/time queries.

## Integration modules

### `api_gateway_rest.py`
- Lambda proxy REST API endpoints for document and audit access.
- API key/OAuth-style auth + CORS handling.

### `emis_export_integration.py`
- Final EMIS transmission step with retry + queue fallback.
- API and secure file-drop transport options.

### `emis_retry_worker.py`
- Retry queue consumer for unresolved EMIS export attempts.

## Ops/config modules

### `centralized_config.py`
- Environment-aware config loader.
- Schema validation + Secrets Manager merge.
- Feature flag and model parameter accessors.

### `infra/terraform/*`
- Production IaC for network, ECS, IAM, data, messaging, monitoring, backup.

## Code example

```python
from centralized_config import load_runtime_config, is_feature_enabled
from emis_export_integration import export_to_emis

cfg = load_runtime_config(environment="prod", load_secrets=True)
if is_feature_enabled(cfg, "enable_validation_layer"):
    result = export_to_emis(
        document_id="doc-123",
        validated_payload={"document_id": "doc-123", "summaries": {}},
        user_id="clinician-a",
        audit_logger=None,
    )
```
