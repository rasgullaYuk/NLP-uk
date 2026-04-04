"""
audit_logger.py — Tier 3 OCR Correction Module
===============================================
Builds JSON-serialisable audit log entries for every region processed
by the Tier 3 pipeline, regardless of outcome.

Every entry captures:
  - What was seen  (original OCR text + bbox)
  - What was done  (corrected text, model used, LLM confidence)
  - Why            (deviation score, token similarity, reason code)
  - When           (ISO 8601 UTC timestamp)
  - Outcome        (status: ACCEPTED / HALLUCINATED / NO_CHANGE / TIMEOUT / SKIPPED)

No file I/O is performed here — the caller decides where logs are stored.
"""

from __future__ import annotations
import datetime
from typing import Any

from config import BEDROCK_MODEL_ID, ReasonCode


def audit_logging(
    *,
    original_text:       str,
    corrected_text:      str | None,
    ocr_confidence:      float,
    llm_confidence:      float | None,
    deviation_score:     float | None,
    token_similarity:    float | None,
    levenshtein_distance: int | None,
    status:              str,
    reason_code:         str,
    reasoning:           str | None,
    bbox:                list[int | float],
    page_number:         int,
    model_id:            str = BEDROCK_MODEL_ID,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a single structured audit log entry.

    All arguments are keyword-only to prevent accidental positional mis-ordering.

    Args:
        original_text:        Raw OCR text from Textract for this region.
        corrected_text:       LLM-proposed correction (None if model was not called).
        ocr_confidence:       Original Textract confidence score (0–1).
        llm_confidence:       Confidence returned by the LLM (0–1), or None.
        deviation_score:      Character-level deviation ratio (0–1), or None.
        token_similarity:     Jaccard token similarity (0–1), or None.
        levenshtein_distance: Raw Levenshtein edit distance, or None.
        status:               Final status string (e.g. "ACCEPTED", "REVIEW_REQUIRED").
        reason_code:          Machine-readable reason (see config.ReasonCode).
        reasoning:            Human-readable explanation from the LLM, or None.
        bbox:                 Bounding box [x1, y1, x2, y2] of the region.
        page_number:          1-indexed page number.
        model_id:             Bedrock model identifier used for this call.
        extra:                Optional dict of any additional fields to merge in.

    Returns:
        A dict that is directly JSON-serialisable (all values are primitives).
    """
    entry: dict[str, Any] = {
        # ── Identity ──────────────────────────────────────────────────────────
        "timestamp":           datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "page_number":         page_number,
        "bbox":                bbox,

        # ── Source text ───────────────────────────────────────────────────────
        "original_text":       original_text,
        "corrected_text":      corrected_text,
        "ocr_confidence":      round(ocr_confidence, 4),

        # ── Model metadata ────────────────────────────────────────────────────
        "model_id":            model_id,
        "llm_confidence":      round(llm_confidence, 4) if llm_confidence is not None else None,
        "reasoning":           reasoning,

        # ── Similarity metrics ────────────────────────────────────────────────
        "deviation_score":     round(deviation_score, 4) if deviation_score is not None else None,
        "token_similarity":    round(token_similarity, 4) if token_similarity is not None else None,
        "levenshtein_distance": levenshtein_distance,

        # ── Outcome ───────────────────────────────────────────────────────────
        "status":              status,
        "reason_code":         reason_code,
    }

    # Merge any caller-supplied extra fields (e.g. idempotency key, run_id).
    if extra:
        entry.update(extra)

    return entry


def build_audit_log_for_skipped_region(
    region: dict[str, Any],
    reason_code: str = ReasonCode.SKIPPED,
    note: str = "",
) -> dict[str, Any]:
    """
    Convenience wrapper: create a minimal audit entry for regions that were
    skipped entirely (e.g. high-confidence regions, deduplication hits).

    Args:
        region:      The original region dict from the input.
        reason_code: Why it was skipped (default: SKIPPED).
        note:        Optional human-readable note.

    Returns:
        A JSON-serialisable audit dict.
    """
    return audit_logging(
        original_text=region.get("text", ""),
        corrected_text=None,
        ocr_confidence=region.get("confidence", -1.0),
        llm_confidence=None,
        deviation_score=None,
        token_similarity=None,
        levenshtein_distance=None,
        status="SKIPPED",
        reason_code=reason_code,
        reasoning=note or "Region skipped — did not require model processing.",
        bbox=region.get("bbox", []),
        page_number=region.get("page_number", -1),
    )
