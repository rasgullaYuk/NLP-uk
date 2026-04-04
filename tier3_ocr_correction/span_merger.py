"""
span_merger.py — Tier 3 OCR Correction Module
=============================================
Replaces low-confidence OCR spans in the region list with LLM-accepted
corrections, while fully preserving document structure (bbox, page_number,
and all original metadata fields).

Public API
----------
    merge_spans(original_regions, corrected_regions)
        → list[dict]   (merged region objects)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from config import ReasonCode

logger = logging.getLogger(__name__)


def merge_spans(
    original_regions:  list[dict[str, Any]],
    corrected_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Replace original low-confidence text spans with accepted LLM corrections.

    Matching is performed by list index — the corrected_regions list must
    correspond 1-to-1 with original_regions (same length, same order).
    Only spans whose corrected region carries `correction_applied: True`
    are replaced; all others retain their original text.

    The function performs a deep-copy of the original regions before merging
    to guarantee idempotency — calling merge_spans twice on the same inputs
    produces the same result and does not mutate the caller's data.

    Args:
        original_regions:  Source region dicts from the pipeline input.
                           Each dict has at minimum: text, confidence, bbox,
                           page_number.
        corrected_regions: Processed region dicts produced by tier3_processor.
                           Each dict has the same fields plus:
                               - corrected_text       (str)
                               - correction_applied   (bool)
                               - reason_code          (str)
                               - llm_confidence       (float | None)
                               - deviation_score      (float | None)

    Returns:
        A new list of region dicts where accepted corrections have been applied.
        Each dict carries an additional `correction_applied` bool field so
        downstream consumers can detect which spans were changed.

    Raises:
        ValueError: If original_regions and corrected_regions have different lengths.
    """
    if len(original_regions) != len(corrected_regions):
        raise ValueError(
            f"original_regions ({len(original_regions)}) and corrected_regions "
            f"({len(corrected_regions)}) must have the same length."
        )

    merged: list[dict[str, Any]] = []

    for idx, (orig, corr) in enumerate(zip(original_regions, corrected_regions)):
        # Deep-copy the original to avoid mutating the caller's data
        merged_region = deepcopy(orig)

        # Carry forward all metadata from the corrected region dict
        merged_region["correction_applied"]  = corr.get("correction_applied", False)
        merged_region["reason_code"]         = corr.get("reason_code", ReasonCode.SKIPPED)
        merged_region["llm_confidence"]      = corr.get("llm_confidence")
        merged_region["deviation_score"]     = corr.get("deviation_score")
        merged_region["token_similarity"]    = corr.get("token_similarity")
        merged_region["levenshtein_distance"] = corr.get("levenshtein_distance")
        merged_region["llm_reasoning"]       = corr.get("reasoning")
        # Carry blended confidence (set by tier3_processor as final_confidence)
        if "confidence" in corr:
            merged_region["confidence"]      = corr["confidence"]

        if corr.get("correction_applied"):
            corrected_text = corr.get("corrected_text", orig.get("text", ""))
            original_text  = orig.get("text", "")

            merged_region["original_text"] = original_text   # preserve original
            merged_region["text"]          = corrected_text  # replace with correction

            logger.debug(
                "Region %d [page %d]: replaced '%s' → '%s'",
                idx,
                orig.get("page_number", -1),
                original_text[:60],
                corrected_text[:60],
            )
        else:
            # Keep original text, but record it for audit trail completeness
            merged_region["original_text"] = orig.get("text", "")

        merged.append(merged_region)

    accepted_count = sum(1 for r in merged if r.get("correction_applied"))
    logger.info(
        "merge_spans: %d/%d regions had corrections applied.",
        accepted_count, len(merged),
    )

    return merged
