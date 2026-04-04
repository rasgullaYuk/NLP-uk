# Tier 3 — Vision-LLM OCR Correction Module

Part of the **Clinical Document Processing Pipeline**. Triggered only when
Textract (Tier 1) returns regions with OCR confidence below threshold.

---

## Pipeline Position

```
Tier 0  →  preprocessing.py       (OpenCV image cleaning)
Tier 1  →  tier1_textract.py      (AWS Textract OCR)
Tier 2  →  tier2_router.py        (SQS fan-out routing)
Tier 3  →  tier3_ocr_correction/  ← THIS MODULE
  ├── Track A  →  track_a_snomed.py   (Comprehend Medical / SNOMED CT)
  └── Track B  →  track_b_summary.py  (Clinical summarisation)
```

---

## Files

| File | Role |
|------|------|
| `config.py` | All tunable constants (thresholds, model ID, region, timeouts) |
| `bedrock_client.py` | AWS Bedrock call with image encoding, retries, timeout |
| `hallucination_detector.py` | Levenshtein + Jaccard dual-signal safety check |
| `span_merger.py` | Merges corrections back into the region list |
| `audit_logger.py` | JSON-ready structured audit log builder |
| `tier3_processor.py` | Orchestrator — main entry point |
| `test_tier3.py` | Unit tests (no AWS credentials needed) |

---

## Quick Start

```python
from tier3_ocr_correction import process_low_confidence_regions
from PIL import Image

low_conf_regions = [
    {
        "text":        "Metflrmin 500 mg",   # OCR misread
        "confidence":  0.52,
        "bbox":        [120, 300, 480, 340],
        "page_number": 1,
    }
]

page_image = Image.open("temp_pages/page_1_CLEANED.png")
context    = "Patient is a Type 2 diabetic on oral hypoglycaemics."

result = process_low_confidence_regions(
    low_confidence_regions=low_conf_regions,
    page_image=page_image,
    surrounding_context_text=context,
    confidence_threshold=0.80,  # optional, default = 0.80
)

print(result["status"])           # "SUCCESS" or "REVIEW_REQUIRED"
print(result["corrected_regions"])
print(result["audit_log"])
```

---

## Routing States & Reason Codes

| Status | Reason Code | Meaning |
|--------|-------------|---------|
| `SUCCESS` | `ACCEPTED` | Correction safe — applied to span |
| `SUCCESS` | `NO_CHANGE` | Model agrees text is correct — no merge needed |
| `SUCCESS` | `SKIPPED` | Region above confidence threshold, not processed |
| `REVIEW_REQUIRED` | `HIGH_DEVIATION` | Correction deviates > 30% or token sim < 50% (hallucination) |
| `REVIEW_REQUIRED` | `LOW_LLM_CONFIDENCE` | LLM confidence < 60% despite acceptable deviation |
| `REVIEW_REQUIRED` | `TIMEOUT` | Bedrock call exceeded 10 s per-region limit |

---

## Key Safety Rules

1. **Hallucination gate** (combined): `deviation > 0.30` **OR** `token_similarity < 0.50`
2. **LLM confidence gate**: `llm_confidence < 0.60` → `REVIEW_REQUIRED`
3. **NO_CHANGE gate**: `deviation < 0.05` or exact match → skip merge, minimal log
4. **Context window**: surrounding context capped at ~250 tokens (sliding window)
5. **Timeout**: hard 10 s socket timeout per Bedrock call → `TIMEOUT` reason code
6. **Idempotency**: duplicate regions (same text + bbox + page) skipped in a single run
7. **No file I/O** in audit_logger — caller owns persistence

---

## Configuration (`config.py`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Pinned model — no aliasing |
| `DEFAULT_OCR_CONFIDENCE_THRESHOLD` | `0.80` | Trigger threshold |
| `HALLUCINATION_DEVIATION_THRESHOLD` | `0.30` | Max char-level deviation |
| `HALLUCINATION_TOKEN_SIM_THRESHOLD` | `0.50` | Min Jaccard token similarity |
| `LLM_CONFIDENCE_GATE` | `0.60` | Min acceptable LLM confidence |
| `NO_CHANGE_DEVIATION_THRESHOLD` | `0.05` | Trivial-edit boundary |
| `CONTEXT_MAX_TOKENS` | `250` | Context sliding window |
| `BEDROCK_CALL_TIMEOUT_SECONDS` | `10` | Per-call hard timeout |
| `MAX_RETRIES` | `3` | Exponential backoff attempts |

---

## Running Tests

```bash
cd tier3_ocr_correction
python -m pytest test_tier3.py -v
```

No AWS credentials are needed — Bedrock is fully mocked in tests.

**Requires:** `pytest`, `Pillow`

```bash
pip install pytest Pillow
```
