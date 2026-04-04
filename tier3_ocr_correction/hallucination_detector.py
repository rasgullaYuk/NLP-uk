"""
hallucination_detector.py — Tier 3 OCR Correction Module
=========================================================
Detects hallucinations by comparing the LLM-corrected text against the
original OCR text using two complementary signals:

  1. Character-level Levenshtein deviation ratio
  2. Token-level Jaccard similarity

Decision rules (updated for clinical safety + typo-fix accuracy):

  NO_CHANGE       if  original_text == corrected_text  (exact match only)

  ACCEPTED        if  deviation < SMALL_DEVIATION_THRESHOLD (< 0.15)
                      — fast-accept lane for single-char OCR typos;
                      overrides token-similarity gate.

  HALLUCINATED    if  deviation > HALLUCINATION_DEVIATION_THRESHOLD (> 0.30)
                  OR  token_similarity < HALLUCINATION_TOKEN_SIM_THRESHOLD (< 0.30)

  ACCEPTED        otherwise

Dosage safety ("ONE drop" vs "TWO drops") is checked separately via
has_dosage_change() and enforced in tier3_processor.py BEFORE this function.

No third-party NLP libraries required — pure Python only.
"""

from __future__ import annotations
import string
from config import (
    DOSAGE_TOKENS,
    HALLUCINATION_DEVIATION_THRESHOLD,
    HALLUCINATION_TOKEN_SIM_THRESHOLD,
    SMALL_DEVIATION_THRESHOLD,
    ReasonCode,
)


# ── Levenshtein distance (pure Python) ────────────────────────────────────────

def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Uses a memory-optimised two-row DP approach — O(min(m, n)) space.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Integer edit distance (number of single-char insertions, deletions,
        or substitutions needed to transform s1 into s2).
    """
    # Ensure s1 is the shorter string to minimise memory.
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)

    # Base cases
    if m == 0:
        return n
    if n == 0:
        return m

    prev_row = list(range(m + 1))

    for j in range(1, n + 1):
        curr_row = [j] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                curr_row[i - 1] + 1,        # insertion
                prev_row[i] + 1,            # deletion
                prev_row[i - 1] + cost,     # substitution
            )
        prev_row = curr_row

    return prev_row[m]


# ── Token-level Jaccard similarity ────────────────────────────────────────────

def _token_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute word-level Jaccard similarity between two strings.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Case-insensitive. Punctuation is stripped from token boundaries.

    Args:
        text_a: First string.
        text_b: Second string.

    Returns:
        Float in [0, 1]. 1.0 = identical token sets, 0.0 = no shared tokens.
    """
    import string

    def tokenise(text: str) -> set:
        # Lower-case, remove punctuation, split on whitespace
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return set(text.split())

    tokens_a = tokenise(text_a)
    tokens_b = tokenise(text_b)

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    if not union:
        # Both strings are empty or contain only punctuation — treat as identical.
        return 1.0

    return len(intersection) / len(union)


# ── Dosage safety check ───────────────────────────────────────────────────

def has_dosage_change(original_text: str, corrected_text: str) -> bool:
    """
    Determine whether the correction alters any dosage-critical token.

    Extracts the set of DOSAGE_TOKENS present in each text (case-insensitive)
    and returns True if those sets differ.  A difference means quantity or
    unit information has changed (e.g. "ONE drop" → "TWO drops"), which must
    be escalated to manual review regardless of deviation score.

    Args:
        original_text:  Raw OCR string.
        corrected_text: LLM-proposed correction.

    Returns:
        True  if any dosage token is added, removed, or changed.
        False if dosage token sets are identical (safe to process normally).
    """
    def _dosage_tokens_in(text: str) -> frozenset:
        # Lowercase, strip punctuation, extract words that are in DOSAGE_TOKENS.
        cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
        return frozenset(w for w in cleaned.split() if w in DOSAGE_TOKENS)

    orig_dosage = _dosage_tokens_in(original_text)
    corr_dosage = _dosage_tokens_in(corrected_text)

    return orig_dosage != corr_dosage


# ── Public API ────────────────────────────────────────────────────────────────

def hallucination_detection(original_text: str, corrected_text: str) -> dict:
    """
    Compare original OCR text against LLM-corrected text and determine
    whether the correction is safe to accept.

    Decision logic (in priority order)
    ------------------------------------
    1. Identical text                          → NO_CHANGE
    2. deviation < SMALL_DEVIATION_THRESHOLD   → ACCEPTED  (fast-accept lane)
    3. deviation > HALLUCINATION_DEVIATION_THRESHOLD
       OR token_similarity < HALLUCINATION_TOKEN_SIM_THRESHOLD
                                               → HALLUCINATED
    4. Otherwise                               → ACCEPTED

    Note: dosage-token safety is checked *before* this function is called
    in tier3_processor.py and is not part of this function's responsibility.

    Args:
        original_text:  The raw OCR string from Textract.
        corrected_text: The correction proposed by the LLM.

    Returns:
        dict with keys:
            - deviation_score       (float 0–1, higher = more different)
            - levenshtein_distance  (int)
            - token_similarity      (float 0–1, higher = more similar)
            - is_hallucinated       (bool)
            - is_no_change          (bool)
            - reason_code           (str, one of ReasonCode constants)
    """
    original_text  = (original_text  or "").strip()
    corrected_text = (corrected_text or "").strip()

    # ── 1. Exact match → NO_CHANGE ───────────────────────────────────────────
    if original_text == corrected_text:
        return {
            "deviation_score":      0.0,
            "levenshtein_distance": 0,
            "token_similarity":     1.0,
            "is_hallucinated":      False,
            "is_no_change":         True,
            "reason_code":          ReasonCode.NO_CHANGE,
        }

    # ── Compute metrics ───────────────────────────────────────────────────────
    max_len = max(len(original_text), len(corrected_text), 1)
    lev_dist        = _levenshtein_distance(original_text, corrected_text)
    deviation_score = lev_dist / max_len          # 0.0 = identical, 1.0 = fully different
    token_similarity = _token_jaccard_similarity(original_text, corrected_text)

    # ── 2. Small deviation → fast-accept (overrides token-similarity gate) ─────
    # Handles single-char/few-char OCR typo fixes (e.g. "Paracetmol" → "Paracetamol",
    # "tobramycn" → "tobramycin") where corrected token differs from original token
    # and Jaccard would otherwise fire.
    if deviation_score < SMALL_DEVIATION_THRESHOLD:
        return {
            "deviation_score":      round(deviation_score, 4),
            "levenshtein_distance": lev_dist,
            "token_similarity":     round(token_similarity, 4),
            "is_hallucinated":      False,
            "is_no_change":         False,   # text DID change — report as ACCEPTED, not NO_CHANGE
            "reason_code":          ReasonCode.ACCEPTED,
        }

    # ── 3. Combined hallucination gate ───────────────────────────────────────
    is_hallucinated = (
        deviation_score   > HALLUCINATION_DEVIATION_THRESHOLD
        or token_similarity < HALLUCINATION_TOKEN_SIM_THRESHOLD
    )

    reason_code = ReasonCode.HIGH_DEVIATION if is_hallucinated else ReasonCode.ACCEPTED

    return {
        "deviation_score":      round(deviation_score, 4),
        "levenshtein_distance": lev_dist,
        "token_similarity":     round(token_similarity, 4),
        "is_hallucinated":      is_hallucinated,
        "is_no_change":         False,
        "reason_code":          reason_code,
    }
