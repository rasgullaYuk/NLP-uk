"""
config.py — Tier 3 OCR Correction Module Configuration
=======================================================
All tunable constants live here. Do NOT hardcode these values in other modules.
Per-document overrides can be passed at call time (e.g., confidence_threshold).
"""

# ── AWS ────────────────────────────────────────────────────────────────────────
AWS_REGION = "us-east-1"

# Pinned exact model version — NO aliasing (e.g., "sonnet" shorthand) allowed.
# Update this string if the model version is superseded.
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# ── Thresholds ─────────────────────────────────────────────────────────────────

# Textract OCR confidence below this value triggers Tier 3 (0–1 scale).
# Configurable per document via process_low_confidence_regions(confidence_threshold=...).
DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.80

# If deviation between original and corrected text exceeds this, mark HALLUCINATED.
HALLUCINATION_DEVIATION_THRESHOLD = 0.30

# Secondary hallucination gate — if token Jaccard similarity drops below this,
# also mark HALLUCINATED (combined rule: deviation > 0.30 OR token_sim < 0.30).
# Lowered from 0.50 → 0.30 to reduce false positives on single-word typo fixes
# (e.g. "tobramycn" → "tobramycin" where corrected token is not in original set).
HALLUCINATION_TOKEN_SIM_THRESHOLD = 0.30

# If the LLM returns a confidence below this, route to REVIEW_REQUIRED
# regardless of deviation score.
LLM_CONFIDENCE_GATE = 0.60

# If deviation is below this value the correction is ALWAYS accepted
# (overrides the token-similarity gate) — covers single-char OCR typos.
# Must be < HALLUCINATION_DEVIATION_THRESHOLD.
SMALL_DEVIATION_THRESHOLD = 0.15

# ── Dosage safety ──────────────────────────────────────────────────────────────
# Tokens that carry dosage/quantity meaning.  Any change in these tokens between
# original and corrected text forces DOSAGE_MISMATCH → REVIEW_REQUIRED.
DOSAGE_TOKENS: frozenset = frozenset([
    "one", "two", "three", "four", "five",
    "mg", "ml", "mcg", "iu", "units",
    "%",
    "drop", "drops",
    "tablet", "tablets", "tab", "tabs",
    "capsule", "capsules", "cap", "caps",
    "patch", "patches",
    "sachet", "sachets",
    "ampoule", "ampoules",
])

# ── Context Window ─────────────────────────────────────────────────────────────
# Approximate max token count for surrounding_context_text sent to the model.
# Enforced by a simple word-count sliding window (~1.3 words/token heuristic).
CONTEXT_MAX_TOKENS = 250
WORDS_PER_TOKEN_ESTIMATE = 0.75   # conservative: 1 token ≈ 0.75 words

# ── Networking ─────────────────────────────────────────────────────────────────
# Maximum seconds to wait for a single Bedrock API call before timing out.
BEDROCK_CALL_TIMEOUT_SECONDS = 10

# Exponential backoff settings for retryable Bedrock errors.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.5    # delay = base * 2^attempt

# ── Claude API ─────────────────────────────────────────────────────────────────
# Max tokens the model is allowed to generate per correction response.
MAX_RESPONSE_TOKENS = 512

# ── Reason Codes ───────────────────────────────────────────────────────────────
# Standardised reason code strings used throughout the module.
class ReasonCode:
    NO_CHANGE           = "NO_CHANGE"
    ACCEPTED            = "ACCEPTED"
    HIGH_DEVIATION      = "HIGH_DEVIATION"
    LOW_LLM_CONFIDENCE  = "LOW_LLM_CONFIDENCE"
    DOSAGE_MISMATCH     = "DOSAGE_MISMATCH"   # dosage tokens differ between original & corrected
    TIMEOUT             = "TIMEOUT"
    REVIEW_REQUIRED     = "REVIEW_REQUIRED"
    SKIPPED             = "SKIPPED"
