import builtins
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import track_a_snomed as snomed


class TestTrackASnomedUnit(unittest.TestCase):
    def test_normalize_and_sliding_window(self):
        self.assertEqual(snomed._normalize_token("ABC-123!"), "abc123")
        text = " ".join([f"word{i}" for i in range(200)]) + " hypertension " + " ".join(
            [f"tail{i}" for i in range(200)]
        )
        window = snomed._get_sliding_window(text, "hypertension", window_words=60)
        self.assertGreaterEqual(len(window.split()), snomed.SLIDING_WINDOW_MIN_WORDS)

    def test_sapbert_embedding_importerror_returns_zero(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in {"transformers", "torch"}:
                raise ImportError("forced")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            emb = snomed._get_sapbert_embedding("test context")
        self.assertEqual(len(emb), 768)

    def test_search_faiss_missing_index_returns_empty(self):
        fake_emb = MagicMock()
        fake_emb.reshape.return_value = fake_emb
        fake_emb.astype.return_value = fake_emb
        with patch("track_a_snomed.os.path.exists", return_value=False):
            out = snomed._search_faiss(fake_emb)
            self.assertEqual(out, [])

    def test_cross_encoder_rerank_no_candidates(self):
        with patch("builtins.__import__", side_effect=ImportError("forced")):
            ranked = snomed._cross_encoder_rerank("ctx", [])
        self.assertEqual(ranked, [])

    @patch("track_a_snomed._cross_encoder_rerank")
    @patch("track_a_snomed._search_faiss")
    @patch("track_a_snomed._get_sapbert_embedding")
    def test_semantic_fallback_happy_path(self, mock_embed, mock_search, mock_rerank):
        mock_embed.return_value = MagicMock()
        mock_search.return_value = [{"snomed_code": "123", "description": "Dx", "retrieval_confidence": 0.9}]
        mock_rerank.return_value = [{"snomed_code": "123", "description": "Dx", "cross_encoder_confidence": 0.8}]
        result = snomed.semantic_snomed_fallback("dx term", "full text content")
        self.assertEqual(result["source"], "semantic_fallback")
        self.assertEqual(result["snomed_code"], "123")

    def test_categorize_entities_and_aggregate_confidence(self):
        entities = [
            {
                "Text": "Hypertension",
                "Category": "MEDICAL_CONDITION",
                "Type": "DX_NAME",
                "confidence": 0.8,
                "snomed_result": {
                    "snomed_code": "38341003",
                    "description": "Hypertension",
                    "source": "comprehend_medical",
                },
            },
            {
                "Text": "Metformin",
                "Category": "MEDICATION",
                "Type": "GENERIC_NAME",
                "confidence": 0.6,
                "snomed_result": {
                    "snomed_code": "860975",
                    "description": "Metformin",
                    "source": "semantic_fallback",
                },
            },
        ]
        categorized = snomed.categorize_entities(entities, full_text="x")
        self.assertEqual(len(categorized["Diagnosis"]), 1)
        self.assertEqual(len(categorized["Medication"]), 1)
        score = snomed.aggregate_confidence(entities)
        self.assertGreater(score, 0.0)

    def test_map_entity_to_snomed_uses_comprehend_when_confident(self):
        entity = {
            "Text": "Hypertension",
            "Score": 0.95,
            "SNOMEDCTConcepts": [{"Code": "38341003", "Description": "Hypertension"}],
        }
        result, conf = snomed.map_entity_to_snomed(entity, "context")
        self.assertEqual(result["source"], "comprehend_medical")
        self.assertEqual(result["snomed_code"], "38341003")
        self.assertAlmostEqual(conf, 0.95, places=3)

    @patch("track_a_snomed.semantic_snomed_fallback")
    def test_map_entity_to_snomed_fallback_used(self, mock_fallback):
        mock_fallback.return_value = {
            "snomed_code": "999",
            "description": "Fallback",
            "confidence": 0.77,
            "source": "semantic_fallback",
        }
        entity = {"Text": "Unknown term", "Score": 0.2, "SNOMEDCTConcepts": []}
        result, conf = snomed.map_entity_to_snomed(entity, "context")
        self.assertEqual(result["source"], "semantic_fallback")
        self.assertAlmostEqual(conf, 0.77, places=3)

    @patch("track_a_snomed.semantic_snomed_fallback", return_value=None)
    def test_map_entity_to_snomed_failed_case(self, _mock_fallback):
        entity = {"Text": "Unknown term", "Score": 0.1, "SNOMEDCTConcepts": []}
        result, conf = snomed.map_entity_to_snomed(entity, "context")
        self.assertEqual(result["source"], "failed")
        self.assertEqual(conf, 0.0)

    @patch("track_a_snomed.semantic_snomed_fallback")
    def test_map_entity_to_snomed_uses_cache(self, mock_fallback):
        snomed._MAP_ENTITY_CACHE.clear()
        mock_fallback.return_value = {
            "snomed_code": "999",
            "description": "Fallback",
            "confidence": 0.77,
            "source": "semantic_fallback",
        }
        entity = {"Text": "Cache term", "Score": 0.2, "SNOMEDCTConcepts": []}
        first = snomed.map_entity_to_snomed(entity, "context")
        second = snomed.map_entity_to_snomed(entity, "context")
        self.assertEqual(first, second)
        mock_fallback.assert_called_once()

    @patch("track_a_snomed.cloudwatch_monitor", None)
    @patch("track_a_snomed.snomed_cache")
    @patch("track_a_snomed.comprehend_medical")
    @patch("track_a_snomed.detect_phi_entities")
    def test_process_document_success(self, mock_phi, mock_cm, mock_cache):
        mock_phi.return_value = []
        mock_cache.get.return_value = None
        mock_cm.infer_snomedct.return_value = {
            "Entities": [
                {
                    "Text": "Hypertension",
                    "Score": 0.92,
                    "SNOMEDCTConcepts": [{"Code": "38341003", "Description": "Hypertension"}],
                    "Category": "MEDICAL_CONDITION",
                    "Type": "DX_NAME",
                }
            ],
            "ModelVersion": "1",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            p = os.path.join(temp_dir, "d_textract.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"Blocks": [{"BlockType": "LINE", "Text": "Patient has hypertension"}]}, f)
            result = snomed.process_document(p)
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["total_entities"], 1)
            self.assertFalse(result["comprehend_response_cached"])

    @patch("track_a_snomed.cloudwatch_monitor", None)
    @patch("track_a_snomed.snomed_cache")
    @patch("track_a_snomed.comprehend_medical")
    @patch("track_a_snomed.detect_phi_entities")
    def test_process_document_uses_cached_comprehend(self, mock_phi, mock_cm, mock_cache):
        mock_phi.return_value = []
        mock_cache.get.return_value = {
            "Entities": [],
            "ModelVersion": "cache-v1",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            p = os.path.join(temp_dir, "d_textract.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"Blocks": [{"BlockType": "LINE", "Text": "Patient has hypertension"}]}, f)
            result = snomed.process_document(p)
            self.assertTrue(result["comprehend_response_cached"])
            mock_cm.infer_snomedct.assert_not_called()

    @patch("track_a_snomed.process_document")
    @patch("track_a_snomed.time.sleep")
    def test_process_with_retry_fail_then_success(self, _sleep, mock_process):
        mock_process.side_effect = [RuntimeError("boom"), {"status": "SUCCESS"}]
        result, err = snomed.process_with_retry("f.json", max_retries=2)
        self.assertIsNone(err)
        self.assertEqual(result["status"], "SUCCESS")

    @patch("track_a_snomed.process_document", side_effect=RuntimeError("always"))
    @patch("track_a_snomed.time.sleep")
    def test_process_with_retry_all_fail(self, _sleep, _proc):
        result, err = snomed.process_with_retry("f.json", max_retries=2)
        self.assertIsNone(result)
        self.assertIn("always", err)

    @patch("track_a_snomed.process_with_retry")
    @patch("track_a_snomed.sqs_client")
    def test_process_track_a_queue_success(self, mock_sqs, mock_retry):
        mock_retry.return_value = (
            {
                "total_entities": 1,
                "fallback_count": 0,
                "unified_confidence_score": 0.9,
                "processing_time_seconds": 0.1,
                "categorized_entities": {"Problems_Issues": [], "Diagnosis": [], "Medication": []},
            },
            None,
        )
        mock_sqs.get_queue_url.side_effect = [
            {"QueueUrl": "track-a-url"},
            {"QueueUrl": "dlq-url"},
        ]
        mock_sqs.receive_message.side_effect = [
            {"Messages": [{"Body": json.dumps({"source_file": "a_textract.json"}), "ReceiptHandle": "rh"}]},
            {},
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            snomed.process_track_a_queue(output_dir=out_dir)
            self.assertTrue(any(name.endswith("_snomed.json") for name in os.listdir(out_dir)))

    @patch("track_a_snomed.process_with_retry", return_value=(None, "bad"))
    @patch("track_a_snomed.sqs_client")
    def test_process_track_a_queue_failure_without_dlq(self, mock_sqs, _mock_retry):
        mock_sqs.get_queue_url.side_effect = [
            {"QueueUrl": "track-a-url"},
            RuntimeError("dlq missing"),
        ]
        mock_sqs.receive_message.side_effect = [
            {"Messages": [{"Body": json.dumps({"source_file": "a_textract.json"}), "ReceiptHandle": "rh"}]},
            {},
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            snomed.process_track_a_queue(output_dir=out_dir)


if __name__ == "__main__":
    unittest.main()
