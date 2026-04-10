# Architecture Overview

## High-level architecture

```mermaid
flowchart LR
    A[Document Upload] --> B[Tier 0\nDocument Handler + Preprocessing]
    B --> C[Tier 1\nAWS Textract]
    C --> D[Router\nConfidence-based]
    D -->|High confidence| E[Track A SNOMED Mapping]
    D -->|High confidence| F[Track B Summarization]
    D -->|Low confidence| G[Tier 2 LayoutLMv3]
    G --> H[Tier 3 OCR Correction]
    H --> E
    H --> F
    E --> I[Unified Confidence Aggregator]
    F --> I
    I --> J[Review Interface\nRVCE-14/RVCE-15]
    J --> K[DynamoDB Audit Trail\nRVCE-13]
    J --> L[EMIS Export Integration]
    L --> M[EMIS API / Secure File Transfer]
```

## Core architectural decisions

- **Pipeline-first design**: independent stages for OCR, clinical mapping, summarization, validation, and routing.
- **Human-in-the-loop**: clinician review is mandatory for low-confidence outcomes.
- **Security-by-default**: secure AWS clients enforce TLS checks and encrypted-at-rest controls.
- **Traceability**: all edits and exports are logged in DynamoDB audit trail.

## Runtime boundaries

- **UI/Review boundary**: `app.py` and `review_interface_utils.py`.
- **Processing boundary**: Tier/Track modules (`tier1_textract.py`, `tier2_router.py`, `track_a_snomed.py`, `track_b_summarization.py`).
- **Integration boundary**: API + export (`api_gateway_rest.py`, `emis_export_integration.py`).
- **Ops boundary**: Terraform + CI/CD (`infra/terraform/*`, `.github/workflows/*`).
