"""
tier3_processor.py — Tier 3 OCR Correction Module
==================================================
Main orchestrator for the Tier 3 Vision-LLM correction stage.

Entry point
-----------
    process_low_confidence_regions(
        low_confidence_regions,
        page_image,
        surrounding_context_text,
        confidence_threshold=DEFAULT_OCR_CONFIDENCE_THRESHOLD,
    )

Pipeline (per region)
---------------------
    1. Idempotency check   — skip if region was already processed (same text + bbox)
    2. Confidence filter   — only process regions below confidence_threshold
    3. bedrock_call()      — call Claude Sonnet vision for correction proposal
    4. hallucination_detection() — dual-signal deviation check
    5. LLM confidence gate — if llm_confidence < LLM_CONFIDENCE_GATE → REVIEW_REQUIRED
    6. NO_CHANGE gate      — if deviation trivially small → accept as-is, minimal log
    7. Routing             — ACCEPTED / HALLUCINATED → REVIEW_REQUIRED / TIMEOUT → REVIEW_REQUIRED
    8. audit_logging()     — record full entry for every region
    9. merge_spans()       — apply accepted corrections to output

Output
------
    {
        "status":            "SUCCESS" | "REVIEW_REQUIRED",
        "corrected_regions": [ ... ],
        "audit_log":         [ ... ]
    }
"""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from audit_logger    import audit_logging, build_audit_log_for_skipped_region
from bedrock_client  import bedrock_call
from config          import (
    DEFAULT_OCR_CONFIDENCE_THRESHOLD,
    LLM_CONFIDENCE_GATE,
    ReasonCode,
)
from hallucination_detector import hallucination_detection, has_dosage_change
from span_merger     import merge_spans

logger = logging.getLogger(__name__)


# ── Idempotency registry (in-process, per-call cache) ─────────────────────────

def _region_fingerprint(region: dict[str, Any]) -> str:
    """
    Produce a lightweight fingerprint for a region to detect duplicates.

    Fingerprint = (text, page_number, bbox_tuple)

    Args:
        region: A single low-confidence region dict.

    Returns:
        A string key unique to this region's content + position.
    """
    text    = (region.get("text") or "").strip()
    page    = region.get("page_number", -1)
    bbox    = tuple(int(v) for v in (region.get("bbox") or []))
    return f"{page}|{bbox}|{text}"


# ── Main orchestrator ─────────────────────────────────────────────────────────

def process_low_confidence_regions(
    low_confidence_regions:  list[dict[str, Any]],
    page_image:              Image.Image,
    surrounding_context_text: str,
    confidence_threshold:    float = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """
    Run the Tier 3 OCR correction pipeline over a list of low-confidence regions.

    Args:
        low_confidence_regions:   List of region dicts from Textract / Tier 2.
                                  Each must contain:
                                    - text         (str)
                                    - confidence   (float, 0–1)
                                    - bbox         ([x1, y1, x2, y2])
                                    - page_number  (int)
        page_image:               Full-page PIL Image corresponding to this batch.
                                  Tier 3 internally crops each region's bbox.
        surrounding_context_text: Clinical context from the same page (raw string).
                                  Will be token-limited before sending to the model.
        confidence_threshold:     Regions with confidence >= this value are skipped
                                  (default from config: 0.80).

    Returns:
        {
            "status":            "SUCCESS" | "REVIEW_REQUIRED",
            "corrected_regions": list of region dicts (merged, annotated),
            "audit_log":         list of JSON-serialisable audit entries
        }
    """
    audit_log:          list[dict[str, Any]] = []
    processed_regions:  list[dict[str, Any]] = []
    overall_review_required = False

    # Idempotency cache — tracks fingerprints seen this call
    seen_fingerprints: set[str] = set()

    for idx, region in enumerate(low_confidence_regions):
        ocr_text      = (region.get("text") or "").strip()
        ocr_confidence = float(region.get("confidence", 0.0))
        bbox           = region.get("bbox", [])
        page_number    = region.get("page_number", -1)

        log_prefix = f"[Region {idx} | page {page_number}]"

        # ── 1. Skip high-confidence regions ───────────────────────────────────
        if ocr_confidence >= confidence_threshold:
            logger.debug("%s Skipped — confidence %.2f >= threshold %.2f",
                         log_prefix, ocr_confidence, confidence_threshold)
            audit_log.append(build_audit_log_for_skipped_region(
                region,
                reason_code=ReasonCode.SKIPPED,
                note=f"OCR confidence {ocr_confidence:.2f} >= threshold {confidence_threshold:.2f}.",
            ))
            # Include unmodified region in output
            r = dict(region)
            r["correction_applied"] = False
            r["reason_code"]        = ReasonCode.SKIPPED
            r["original_text"]      = ocr_text
            processed_regions.append(r)
            continue

        # ── 2. Idempotency check ───────────────────────────────────────────────
        fingerprint = _region_fingerprint(region)
        if fingerprint in seen_fingerprints:
            logger.info("%s Skipped — duplicate region (idempotency).", log_prefix)
            audit_log.append(build_audit_log_for_skipped_region(
                region,
                reason_code=ReasonCode.SKIPPED,
                note="Duplicate region detected — skipped for idempotency.",
            ))
            r = dict(region)
            r["correction_applied"] = False
            r["reason_code"]        = ReasonCode.SKIPPED
            r["original_text"]      = ocr_text
            processed_regions.append(r)
            continue
        seen_fingerprints.add(fingerprint)

        logger.info(
            "%s Processing — OCR confidence=%.2f, text='%s'",
            log_prefix, ocr_confidence, ocr_text[:80],
        )

        # ── 3. Call Bedrock ────────────────────────────────────────────────────
        try:
            llm_result = bedrock_call(
                ocr_text=ocr_text,
                full_page_image=page_image,
                bbox=bbox,
                surrounding_context=surrounding_context_text,
                region_index=idx,
            )
        except TimeoutError as exc:
            logger.warning("%s Bedrock timeout: %s", log_prefix, exc)
            overall_review_required = True

            r = dict(region)
            r.update({
                "correction_applied":  False,
                "corrected_text":      None,
                "reason_code":         ReasonCode.TIMEOUT,
                "original_text":       ocr_text,
                "llm_confidence":      None,
                "deviation_score":     None,
                "token_similarity":    None,
                "levenshtein_distance": None,
                "reasoning":           str(exc),
            })
            processed_regions.append(r)

            audit_log.append(audit_logging(
                original_text=ocr_text,
                corrected_text=None,
                ocr_confidence=ocr_confidence,
                llm_confidence=None,
                deviation_score=None,
                token_similarity=None,
                levenshtein_distance=None,
                status="REVIEW_REQUIRED",
                reason_code=ReasonCode.TIMEOUT,
                reasoning=str(exc),
                bbox=bbox,
                page_number=page_number,
            ))
            continue

        except Exception as exc:
            logger.error("%s Bedrock call failed: %s", log_prefix, exc)
            overall_review_required = True

            r = dict(region)
            r.update({
                "correction_applied":  False,
                "corrected_text":      None,
                "reason_code":         ReasonCode.REVIEW_REQUIRED,
                "original_text":       ocr_text,
                "llm_confidence":      None,
                "deviation_score":     None,
                "token_similarity":    None,
                "levenshtein_distance": None,
                "reasoning":           f"Bedrock error: {exc}",
            })
            processed_regions.append(r)

            audit_log.append(audit_logging(
                original_text=ocr_text,
                corrected_text=None,
                ocr_confidence=ocr_confidence,
                llm_confidence=None,
                deviation_score=None,
                token_similarity=None,
                levenshtein_distance=None,
                status="REVIEW_REQUIRED",
                reason_code=ReasonCode.REVIEW_REQUIRED,
                reasoning=f"Bedrock error: {exc}",
                bbox=bbox,
                page_number=page_number,
            ))
            continue

        # Unpack LLM result
        corrected_text = llm_result["corrected_text"]
        llm_confidence = llm_result["confidence"]
        reasoning      = llm_result["reasoning"]

        # -- Compute similarity metrics -----------------------------------------
        hal_result       = hallucination_detection(ocr_text, corrected_text)
        deviation_score  = hal_result["deviation_score"]
        token_similarity = hal_result["token_similarity"]
        lev_distance     = hal_result["levenshtein_distance"]
        is_hallucinated  = hal_result["is_hallucinated"]

        # -- Routing: 6-step priority chain ------------------------------------
        #
        # Priority order (clinical safety first):
        #   1. Bedrock failure      -> handled above (TIMEOUT / REVIEW_REQUIRED)
        #   2. DOSAGE CHECK         -> REVIEW_REQUIRED / DOSAGE_MISMATCH
        #   3. EXACT MATCH          -> NO_CHANGE
        #   4. LOW LLM CONFIDENCE   -> REVIEW_REQUIRED / LOW_LLM_CONFIDENCE
        #   5. HALLUCINATION        -> REVIEW_REQUIRED / HIGH_DEVIATION
        #   6. ACCEPT SAFE FIXES    -> ACCEPTED
        #
        # A single region dict and audit entry are built after the chain.

        if has_dosage_change(ocr_text, corrected_text):
            # 2. DOSAGE CHECK -- highest post-Bedrock priority
            # Quantity/unit changes are never applied automatically.
            logger.warning(
                "%s DOSAGE_MISMATCH -- dosage tokens differ between OCR and correction.",
                log_prefix,
            )
            status             = "REVIEW_REQUIRED"
            reason_code        = ReasonCode.DOSAGE_MISMATCH
            correction_applied = False
            overall_review_required = True

        elif ocr_text == corrected_text:
            # 3. EXACT MATCH -> NO_CHANGE
            # Model agrees the text is already correct; no merge needed.
            logger.info("%s NO_CHANGE -- model returned identical text.", log_prefix)
            status             = "NO_CHANGE"
            reason_code        = ReasonCode.NO_CHANGE
            correction_applied = False

        elif llm_confidence < LLM_CONFIDENCE_GATE:
            # 4. LOW LLM CONFIDENCE -> REVIEW_REQUIRED
            # Model is uncertain about its own correction; escalate.
            logger.warning(
                "%s LOW_LLM_CONFIDENCE -- llm_confidence=%.2f < gate=%.2f",
                log_prefix, llm_confidence, LLM_CONFIDENCE_GATE,
            )
            status             = "REVIEW_REQUIRED"
            reason_code        = ReasonCode.LOW_LLM_CONFIDENCE
            correction_applied = False
            overall_review_required = True

        elif is_hallucinated:
            # 5. HALLUCINATION -> REVIEW_REQUIRED / HIGH_DEVIATION
            # deviation > 0.30 OR token_similarity < 0.30
            # (fast-accept lane cleared deviation < 0.15 inside hallucination_detector)
            logger.warning(
                "%s HALLUCINATED -- deviation=%.4f, token_sim=%.4f",
                log_prefix, deviation_score, token_similarity,
            )
            status             = "REVIEW_REQUIRED"
            reason_code        = ReasonCode.HIGH_DEVIATION
            correction_applied = False
            overall_review_required = True

        else:
            # 6. ACCEPT SAFE CORRECTIONS
            logger.info(
                "%s ACCEPTED -- deviation=%.4f, token_sim=%.4f, llm_conf=%.2f",
                log_prefix, deviation_score, token_similarity, llm_confidence,
            )
            status             = "ACCEPTED"
            reason_code        = ReasonCode.ACCEPTED
            correction_applied = True

        # -- Build region entry (shared across all branches) -------------------
        # text: expose corrected value when accept, original otherwise.
        # confidence: blend OCR + LLM when correction applied (higher combined
        #             certainty); keep raw OCR confidence for non-corrections.
        final_text = corrected_text if correction_applied else ocr_text
        final_confidence = (
            (ocr_confidence + llm_confidence) / 2
            if correction_applied
            else ocr_confidence
        )

        r = dict(region)
        r.update({
            "text":                 final_text,
            "confidence":           round(final_confidence, 4),
            "correction_applied":   correction_applied,
            "corrected_text":       corrected_text,
            "original_text":        ocr_text,
            "reason_code":          reason_code,
            "llm_confidence":       llm_confidence,
            "deviation_score":      deviation_score,
            "token_similarity":     token_similarity,
            "levenshtein_distance": lev_distance,
            "reasoning":            reasoning,
        })
        processed_regions.append(r)

        audit_log.append(audit_logging(
            original_text=ocr_text,
            corrected_text=corrected_text,
            ocr_confidence=ocr_confidence,
            llm_confidence=llm_confidence,
            deviation_score=deviation_score,
            token_similarity=token_similarity,
            levenshtein_distance=lev_distance,
            status=status,
            reason_code=reason_code,
            reasoning=reasoning,
            bbox=bbox,
            page_number=page_number,
        ))

    # ── 9. Merge accepted spans back into the region list ─────────────────────
    merged_regions = merge_spans(low_confidence_regions, processed_regions)

    overall_status = "REVIEW_REQUIRED" if overall_review_required else "SUCCESS"

    logger.info(
        "Tier 3 complete — status=%s, regions=%d, audit_entries=%d",
        overall_status, len(merged_regions), len(audit_log),
    )

    return {
        "status":            overall_status,
        "corrected_regions": merged_regions,
        "audit_log":         audit_log,
    }
