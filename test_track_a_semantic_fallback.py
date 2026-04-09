import unittest
from unittest.mock import patch

import numpy as np

import track_a_snomed as track_a


class TestTrackASemanticFallback(unittest.TestCase):
    def test_threshold_is_075(self):
        self.assertEqual(track_a.COMPREHEND_CONF_THRESHOLD, 0.75)

    def test_sliding_window_returns_context_50_to_100_words(self):
        words = [f"w{i}" for i in range(250)]
        words[125] = "hypertension"
        full_text = " ".join(words)

        context = track_a._get_sliding_window(full_text, "hypertension", window_words=75)
        context_words = context.split()

        self.assertGreaterEqual(len(context_words), 50)
        self.assertLessEqual(len(context_words), 100)
        self.assertIn("hypertension", context_words)

    @patch("track_a_snomed.semantic_snomed_fallback")
    def test_map_entity_uses_fallback_below_threshold(self, mock_fallback):
        mock_fallback.return_value = {
            "snomed_code": "123456",
            "description": "Mock SNOMED",
            "confidence": 0.91,
            "source": "semantic_fallback",
        }
        entity = {
            "Text": "hypertension",
            "Score": 0.74,
            "SNOMEDCTConcepts": [{"Code": "111", "Description": "Low confidence code"}],
        }

        result, confidence = track_a.map_entity_to_snomed(entity, "patient has hypertension")
        self.assertEqual(result["source"], "semantic_fallback")
        self.assertEqual(confidence, 0.91)
        mock_fallback.assert_called_once()

    @patch("track_a_snomed.semantic_snomed_fallback")
    def test_map_entity_uses_comprehend_at_or_above_threshold(self, mock_fallback):
        entity = {
            "Text": "hypertension",
            "Score": 0.92,
            "SNOMEDCTConcepts": [{"Code": "38341003", "Description": "Hypertensive disorder"}],
        }

        result, confidence = track_a.map_entity_to_snomed(entity, "patient has hypertension")
        self.assertEqual(result["source"], "comprehend_medical")
        self.assertEqual(result["snomed_code"], "38341003")
        self.assertEqual(confidence, 0.92)
        mock_fallback.assert_not_called()

    @patch("track_a_snomed._cross_encoder_rerank")
    @patch("track_a_snomed._search_faiss")
    @patch("track_a_snomed._get_sapbert_embedding")
    @patch("track_a_snomed._get_sliding_window")
    def test_semantic_fallback_uses_context_embedding_and_top10(
        self,
        mock_window,
        mock_embedding,
        mock_search,
        mock_rerank,
    ):
        mock_window.return_value = "context around term with nearby diagnosis and medication"
        mock_embedding.return_value = np.ones(768, dtype=np.float32)
        candidates = [
            {
                "snomed_code": f"{1000 + i}",
                "description": f"candidate-{i}",
                "similarity": 0.8 - (i * 0.01),
                "retrieval_confidence": 0.9 - (i * 0.01),
            }
            for i in range(10)
        ]
        reranked = []
        for i, c in enumerate(candidates):
            row = dict(c)
            row["cross_encoder_score"] = 4.0 - i
            row["cross_encoder_confidence"] = 0.95 - (i * 0.02)
            reranked.append(row)
        mock_search.return_value = candidates
        mock_rerank.return_value = reranked

        result = track_a.semantic_snomed_fallback("hypertension", "very long clinical narrative")

        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "semantic_fallback")
        self.assertEqual(result["snomed_code"], "1000")
        self.assertEqual(len(result["all_candidates"]), 10)
        mock_embedding.assert_called_once()
        embedding_input = mock_embedding.call_args.args[0]
        self.assertIn("[SEP]", embedding_input)


if __name__ == "__main__":
    unittest.main()
