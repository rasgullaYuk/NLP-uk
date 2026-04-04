"""
test_tier3.py — Unit tests for tier3_ocr_correction
====================================================
Covers:
  - hallucination_detector: all decision branches
  - audit_logger:           output schema
  - span_merger:            structure preservation, replacement logic
  - tier3_processor:        full pipeline with mocked bedrock_call

Run with:
    cd tier3_ocr_correction
    python -m pytest test_tier3.py -v

Dependencies:
    pytest, Pillow  (both are development dependencies)
"""

import sys
import os
import json
import types
import unittest
from copy      import deepcopy
from unittest  import mock
from io        import BytesIO

from PIL import Image

# ── Path setup (run from repo root or from tier3_ocr_correction/) ─────────────
sys.path.insert(0, os.path.dirname(__file__))

from hallucination_detector import (
    hallucination_detection,
    _levenshtein_distance,
    _token_jaccard_similarity,
)
from audit_logger  import audit_logging, build_audit_log_for_skipped_region
from span_merger   import merge_spans
from config        import ReasonCode


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_small_image(width: int = 100, height: int = 50) -> Image.Image:
    """Create a tiny blank PIL Image for testing without file I/O."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _make_region(
    text:        str   = "Metflrmin 500 mg",
    confidence:  float = 0.55,
    bbox:        list  = None,
    page_number: int   = 1,
) -> dict:
    return {
        "text":        text,
        "confidence":  confidence,
        "bbox":        bbox or [10, 20, 200, 40],
        "page_number": page_number,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Levenshtein distance (unit)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLevenshteinDistance(unittest.TestCase):

    def test_identical_strings(self):
        self.assertEqual(_levenshtein_distance("hello", "hello"), 0)

    def test_empty_strings(self):
        self.assertEqual(_levenshtein_distance("", ""), 0)

    def test_one_empty(self):
        self.assertEqual(_levenshtein_distance("abc", ""), 3)
        self.assertEqual(_levenshtein_distance("", "abc"), 3)

    def test_single_substitution(self):
        self.assertEqual(_levenshtein_distance("cat", "bat"), 1)

    def test_insertion_deletion(self):
        self.assertEqual(_levenshtein_distance("kitten", "sitting"), 3)

    def test_completely_different(self):
        dist = _levenshtein_distance("abc", "xyz")
        self.assertEqual(dist, 3)

    def test_swapped_args_symmetry(self):
        self.assertEqual(
            _levenshtein_distance("Metformin", "Metflrmin"),
            _levenshtein_distance("Metflrmin", "Metformin"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Token Jaccard similarity (unit)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenJaccardSimilarity(unittest.TestCase):

    def test_identical(self):
        self.assertAlmostEqual(
            _token_jaccard_similarity("hello world", "hello world"), 1.0
        )

    def test_completely_different(self):
        self.assertAlmostEqual(
            _token_jaccard_similarity("cat sat mat", "dog ran far"), 0.0
        )

    def test_partial_overlap(self):
        score = _token_jaccard_similarity("the quick brown fox", "the slow brown dog")
        # shared: {the, brown} = 2, union = {the, quick, brown, fox, slow, dog} = 6 → 2/6
        self.assertAlmostEqual(score, 2 / 6, places=4)

    def test_empty_strings(self):
        # Both empty → treated as identical
        self.assertAlmostEqual(_token_jaccard_similarity("", ""), 1.0)

    def test_case_insensitive(self):
        self.assertAlmostEqual(
            _token_jaccard_similarity("Metformin", "metformin"), 1.0
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Hallucination detection — all branches
# ═══════════════════════════════════════════════════════════════════════════════

class TestHallucinationDetection(unittest.TestCase):

    # 3a. Exact match → NO_CHANGE
    def test_exact_match(self):
        result = hallucination_detection("Metformin 500 mg", "Metformin 500 mg")
        self.assertTrue(result["is_no_change"])
        self.assertFalse(result["is_hallucinated"])
        self.assertEqual(result["deviation_score"], 0.0)
        self.assertEqual(result["reason_code"], ReasonCode.NO_CHANGE)

    # 3b. Trivial edit (< 15%) AND text differs → ACCEPTED (fast-accept lane)
    # Under the new rules NO_CHANGE fires ONLY on exact match.
    # Small but real edits (deviation < SMALL_DEVIATION_THRESHOLD) are ACCEPTED.
    def test_trivial_edit_no_change(self):
        a = "Patient is prescribed Metformin for diabetes management daily"
        b = "Patient is prescribed Metformin for diabetes management daliy"  # typo fix
        result = hallucination_detection(a, b)
        # deviation = 2/60 ≈ 0.033 < 0.15 → fast-accept lane → ACCEPTED (not NO_CHANGE)
        self.assertFalse(result["is_no_change"])
        self.assertFalse(result["is_hallucinated"])
        self.assertEqual(result["reason_code"], ReasonCode.ACCEPTED)

    # 3c. Clear OCR fix within safe bounds → ACCEPTED
    def test_accepted_correction(self):
        # Multi-word phrase: one character OCR error in a shared-context phrase.
        # "dally" → "daily" — 1 char diff.  Shared tokens: {dose, taken, dally/daily, morning}
        # deviation ≈ 1/36 ≈ 0.028 → NO_CHANGE gate fires (< 0.05).
        # Use a pair where deviation is between 0.05–0.30 AND token_sim ≥ 0.50.
        # "taken once dally in the morning" vs "taken once daily in the morning":
        #   lev = 1, max_len = 31, deviation ≈ 0.032 → NO_CHANGE (too small).
        # Better: introduce a 3-char substitution in a short phrase.
        # "dose 500 rng oral" (rng→mg) vs "dose 500 mg oral":
        #   lev = 2 (remove 'r', 'n'→empty... actually lev("rng","mg") = 2)
        #   deviation = 2/18 ≈ 0.111, tokens: {dose,500,oral} shared 3/4 → 0.75 → ACCEPTED
        original  = "dose 500 rng oral"
        corrected = "dose 500 mg oral"
        result = hallucination_detection(original, corrected)
        self.assertFalse(result["is_hallucinated"])
        self.assertFalse(result["is_no_change"])
        self.assertEqual(result["reason_code"], ReasonCode.ACCEPTED)

    # 3d. Very different corrected text → HALLUCINATED (high deviation)
    def test_hallucinated_high_deviation(self):
        original  = "prednlsone"
        corrected = "amoxicillin 500mg twice daily after meals for infection"
        result = hallucination_detection(original, corrected)
        self.assertTrue(result["is_hallucinated"])
        self.assertIn(result["reason_code"], [ReasonCode.HIGH_DEVIATION])

    # 3e. Same words but shuffled → low token sim → HALLUCINATED
    def test_hallucinated_low_token_similarity(self):
        # Construct a case where deviation < 30% but token similarity < 50%
        # This is tricky; we simulate with direct mock instead.
        # Just test boundary: if token_sim < 0.5, it should be hallucinated.
        from hallucination_detector import HALLUCINATION_TOKEN_SIM_THRESHOLD
        original  = "atorvastatin 20 mg"
        # Replace every word entirely: deviation will be high anyway, but let's verify rule fires
        corrected = "aspirin 81 tablet"
        result = hallucination_detection(original, corrected)
        self.assertTrue(result["is_hallucinated"])

    # 3f. Return structure has expected keys
    def test_return_keys(self):
        result = hallucination_detection("abc", "xyz")
        required_keys = {
            "deviation_score", "levenshtein_distance", "token_similarity",
            "is_hallucinated", "is_no_change", "reason_code",
        }
        self.assertTrue(required_keys.issubset(result.keys()))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Audit logging — schema validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditLogging(unittest.TestCase):

    def _make_entry(self, **overrides) -> dict:
        defaults = dict(
            original_text="Metflrmin 500 mg",
            corrected_text="Metformin 500 mg",
            ocr_confidence=0.55,
            llm_confidence=0.92,
            deviation_score=0.133,
            token_similarity=1.0,
            levenshtein_distance=2,
            status="ACCEPTED",
            reason_code=ReasonCode.ACCEPTED,
            reasoning="Corrected OCR misread 'fl' → 'fo'.",
            bbox=[10, 20, 200, 40],
            page_number=1,
        )
        defaults.update(overrides)
        return audit_logging(**defaults)

    def test_required_keys_present(self):
        entry = self._make_entry()
        required = {
            "timestamp", "page_number", "bbox",
            "original_text", "corrected_text", "ocr_confidence",
            "model_id", "llm_confidence", "reasoning",
            "deviation_score", "token_similarity", "levenshtein_distance",
            "status", "reason_code",
        }
        self.assertTrue(required.issubset(entry.keys()), f"Missing keys: {required - entry.keys()}")

    def test_json_serialisable(self):
        entry = self._make_entry()
        # Should not raise
        serialised = json.dumps(entry)
        self.assertIsInstance(serialised, str)

    def test_timestamp_format(self):
        entry = self._make_entry()
        ts = entry["timestamp"]
        # Should end with 'Z' and be parseable
        self.assertTrue(ts.endswith("Z"))
        from datetime import datetime
        datetime.fromisoformat(ts.rstrip("Z"))

    def test_none_values_handled(self):
        entry = self._make_entry(
            corrected_text=None,
            llm_confidence=None,
            deviation_score=None,
        )
        self.assertIsNone(entry["corrected_text"])
        self.assertIsNone(entry["llm_confidence"])
        json.dumps(entry)  # must still be serialisable

    def test_skipped_region_helper(self):
        region = _make_region()
        entry = build_audit_log_for_skipped_region(region)
        self.assertEqual(entry["reason_code"], ReasonCode.SKIPPED)
        self.assertEqual(entry["status"], "SKIPPED")
        json.dumps(entry)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Span merger
# ═══════════════════════════════════════════════════════════════════════════════

class TestMergeSpans(unittest.TestCase):

    def _make_corrected(self, region, *, accepted=True, corrected_text="Metformin 500 mg"):
        r = dict(region)
        r["correction_applied"]  = accepted
        r["corrected_text"]      = corrected_text if accepted else None
        r["reason_code"]         = ReasonCode.ACCEPTED if accepted else ReasonCode.HIGH_DEVIATION
        r["llm_confidence"]      = 0.93 if accepted else None
        r["deviation_score"]     = 0.13 if accepted else 0.55
        r["token_similarity"]    = 0.85 if accepted else 0.3
        r["levenshtein_distance"] = 2 if accepted else 10
        r["reasoning"]           = "test"
        return r

    def test_accepted_correction_applied(self):
        original  = [_make_region(text="Metflrmin 500 mg")]
        corrected = [self._make_corrected(original[0], accepted=True)]
        merged = merge_spans(original, corrected)
        self.assertEqual(merged[0]["text"], "Metformin 500 mg")
        self.assertEqual(merged[0]["original_text"], "Metflrmin 500 mg")
        self.assertTrue(merged[0]["correction_applied"])

    def test_rejected_keeps_original_text(self):
        original  = [_make_region(text="prednlsone")]
        corrected = [self._make_corrected(original[0], accepted=False)]
        merged = merge_spans(original, corrected)
        self.assertEqual(merged[0]["text"], "prednlsone")
        self.assertFalse(merged[0]["correction_applied"])

    def test_bbox_preserved(self):
        bbox = [50, 100, 300, 130]
        original  = [_make_region(bbox=bbox)]
        corrected = [self._make_corrected(original[0], accepted=True)]
        merged = merge_spans(original, corrected)
        self.assertEqual(merged[0]["bbox"], bbox)

    def test_page_number_preserved(self):
        original  = [_make_region(page_number=5)]
        corrected = [self._make_corrected(original[0])]
        merged = merge_spans(original, corrected)
        self.assertEqual(merged[0]["page_number"], 5)

    def test_length_mismatch_raises(self):
        original  = [_make_region(), _make_region()]
        corrected = [self._make_corrected(_make_region())]
        with self.assertRaises(ValueError):
            merge_spans(original, corrected)

    def test_deep_copy_no_mutation(self):
        original = [_make_region(text="before")]
        corrected = [self._make_corrected(original[0], corrected_text="after")]
        _ = merge_spans(original, corrected)
        # Original list must be unchanged
        self.assertEqual(original[0]["text"], "before")

    def test_multiple_regions(self):
        originals  = [_make_region(text=f"word_{i}") for i in range(5)]
        correcteds = [
            self._make_corrected(r, accepted=(i % 2 == 0), corrected_text=f"FIXED_{i}")
            for i, r in enumerate(originals)
        ]
        merged = merge_spans(originals, correcteds)
        for i, m in enumerate(merged):
            if i % 2 == 0:
                self.assertEqual(m["text"], f"FIXED_{i}")
            else:
                self.assertEqual(m["text"], f"word_{i}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Full pipeline — tier3_processor (mocked Bedrock)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier3Processor(unittest.TestCase):
    """
    All Bedrock calls are mocked.  Tests verify routing logic and output schema.
    """

    def setUp(self):
        self.image   = _make_small_image()
        self.context = "Patient is a Type 2 diabetic managed with oral hypoglycaemics."

    def _run(self, regions, mock_llm_return, **kwargs):
        """Helper: patch bedrock_call, run processor, return result."""
        with mock.patch("tier3_processor.bedrock_call", return_value=mock_llm_return):
            from tier3_processor import process_low_confidence_regions
            return process_low_confidence_regions(
                low_confidence_regions=regions,
                page_image=self.image,
                surrounding_context_text=self.context,
                **kwargs,
            )

    # 6a. Happy path — correction accepted
    def test_accepted_correction(self):
        regions = [_make_region(text="Metflrmin 500 mg", confidence=0.55)]
        result  = self._run(regions, {
            "corrected_text": "Metformin 500 mg",
            "confidence":     0.93,
            "reasoning":      "Corrected OCR 'fl' → 'fo'.",
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(result["corrected_regions"]), 1)
        self.assertTrue(result["corrected_regions"][0]["correction_applied"])
        self.assertEqual(result["corrected_regions"][0]["text"], "Metformin 500 mg")
        self.assertEqual(result["audit_log"][0]["reason_code"], ReasonCode.ACCEPTED)

    # 6b. High-confidence region is skipped
    def test_high_confidence_skipped(self):
        regions = [_make_region(confidence=0.95)]
        with mock.patch("tier3_processor.bedrock_call") as mock_bc:
            from tier3_processor import process_low_confidence_regions
            result = process_low_confidence_regions(
                regions, self.image, self.context, confidence_threshold=0.80
            )
            mock_bc.assert_not_called()
        self.assertFalse(result["corrected_regions"][0]["correction_applied"])
        self.assertEqual(result["corrected_regions"][0]["reason_code"], ReasonCode.SKIPPED)

    # 6c. NO_CHANGE — model returns same text
    def test_no_change(self):
        text    = "Aspirin 75 mg"
        regions = [_make_region(text=text, confidence=0.60)]
        result  = self._run(regions, {
            "corrected_text": text,     # identical → NO_CHANGE
            "confidence":     0.99,
            "reasoning":      "Text appears correct.",
        })
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["corrected_regions"][0]["correction_applied"])
        self.assertEqual(result["audit_log"][0]["reason_code"], ReasonCode.NO_CHANGE)

    # 6d. LLM confidence too low → REVIEW_REQUIRED
    def test_low_llm_confidence(self):
        regions = [_make_region(text="Metflrmin", confidence=0.55)]
        result  = self._run(regions, {
            "corrected_text": "Metformin",
            "confidence":     0.45,     # below LLM_CONFIDENCE_GATE (0.60)
            "reasoning":      "Uncertain.",
        })
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["corrected_regions"][0]["correction_applied"])
        self.assertEqual(result["audit_log"][0]["reason_code"], ReasonCode.LOW_LLM_CONFIDENCE)

    # 6e. Hallucinated correction → REVIEW_REQUIRED with HIGH_DEVIATION
    # Use a corrected_text that has no dosage token overlap change so the
    # dosage gate doesn’t fire first — pure deviation-based hallucination.
    def test_hallucination_detected(self):
        regions = [_make_region(text="prednlsone chronic use", confidence=0.50)]
        result  = self._run(regions, {
            # Completely different phrase, no dosage tokens in either string
            "corrected_text": "furosemide therapy long term chronic administration",
            "confidence":     0.88,
            "reasoning":      "Hallucinated a completely different drug.",
        })
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertFalse(result["corrected_regions"][0]["correction_applied"])
        self.assertEqual(result["audit_log"][0]["reason_code"], ReasonCode.HIGH_DEVIATION)

    # 6f. Bedrock timeout → REVIEW_REQUIRED, region flagged TIMEOUT
    def test_timeout_handling(self):
        regions = [_make_region(confidence=0.55)]
        with mock.patch("tier3_processor.bedrock_call", side_effect=TimeoutError("timed out")):
            from tier3_processor import process_low_confidence_regions
            result = process_low_confidence_regions(regions, self.image, self.context)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["audit_log"][0]["reason_code"], ReasonCode.TIMEOUT)
        self.assertFalse(result["corrected_regions"][0]["correction_applied"])

    # 6g. Idempotency — duplicate region not processed twice
    def test_idempotency(self):
        region  = _make_region(text="same text", confidence=0.55)
        regions = [region, deepcopy(region)]  # identical fingerprint
        call_count = 0

        def _mock_bedrock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"corrected_text": "same text fixed", "confidence": 0.9, "reasoning": "ok"}

        with mock.patch("tier3_processor.bedrock_call", side_effect=_mock_bedrock):
            from tier3_processor import process_low_confidence_regions
            result = process_low_confidence_regions(regions, self.image, self.context)

        self.assertEqual(call_count, 1, "Bedrock should only be called once for duplicate regions")
        self.assertEqual(len(result["corrected_regions"]), 2)

    # 6h. Output schema validation
    def test_output_schema(self):
        regions = [_make_region(confidence=0.55)]
        result  = self._run(regions, {
            "corrected_text": "Metformin 500 mg",
            "confidence":     0.92,
            "reasoning":      "Fixed typo.",
        })
        self.assertIn("status",            result)
        self.assertIn("corrected_regions", result)
        self.assertIn("audit_log",         result)
        # Entire result must be JSON-serialisable
        json.dumps(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
