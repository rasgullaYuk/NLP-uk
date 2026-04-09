import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import audit_dynamodb
import lambda_confidence_aggregator as agg
import preprocessing
import tier1_textract
import tier2_router
import track_a_snomed
import track_b_summarization as tb
from tier3_ocr_correction import tier3_processor


class _FakeBody:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")


class _FakeSQS:
    def __init__(self):
        self.urls = {}
        self.messages = []

    def create_queue(self, QueueName):
        url = f"https://queue/{QueueName}"
        self.urls[QueueName] = url
        return {"QueueUrl": url}

    def get_queue_url(self, QueueName):
        if QueueName in self.urls:
            return {"QueueUrl": self.urls[QueueName]}
        raise RuntimeError("missing queue")

    def send_message(self, QueueUrl, MessageBody):
        self.messages.append((QueueUrl, json.loads(MessageBody)))
        return {"MessageId": "m1"}


class TestPipelineIntegration(unittest.TestCase):
    def test_pdf_like_preprocess_to_tier1_extraction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_path = os.path.join(temp_dir, "doc_page1_CLEANED.jpg")
            with open(cleaned_path, "wb") as f:
                f.write(b"img")
            success = [{"original": os.path.join(temp_dir, "doc_page1_original.jpg"), "cleaned": cleaned_path}]
            payload = preprocessing.get_tier1_payload(success)
            self.assertEqual(payload[0]["page"], 1)

            fake_textract = MagicMock()
            fake_textract.analyze_document.return_value = {"Blocks": [{"BlockType": "LINE", "Text": "Patient"}]}
            with patch("tier1_textract.create_secure_client", return_value=fake_textract), \
                 patch("tier1_textract.detect_phi_entities", return_value=[]), \
                 patch("tier1_textract.CloudWatchMonitoringManager", side_effect=RuntimeError("off")):
                tier1_textract.process_documents_with_textract(input_dir=temp_dir, output_dir=temp_dir)
                out_files = [x for x in os.listdir(temp_dir) if x.endswith("_textract.json")]
                self.assertGreaterEqual(len(out_files), 1)

    @patch("tier3_ocr_correction.tier3_processor.write_audit_batch_to_dynamodb")
    @patch("tier3_ocr_correction.tier3_processor.merge_spans")
    @patch("tier3_ocr_correction.tier3_processor.hallucination_detection")
    @patch("tier3_ocr_correction.tier3_processor.bedrock_call")
    def test_low_confidence_tier2_to_tier3_flow(
        self, mock_bedrock, mock_hallucination, mock_merge, _mock_write
    ):
        fake_sqs = _FakeSQS()
        with tempfile.TemporaryDirectory() as temp_dir:
            textract_path = os.path.join(temp_dir, "doc_textract.json")
            with open(textract_path, "w", encoding="utf-8") as f:
                json.dump({"Blocks": [{"BlockType": "LINE", "Confidence": 40.0, "Text": "blurry text"}]}, f)
            with open(os.path.join(temp_dir, "doc.jpg"), "wb") as f:
                f.write(b"img")
            with patch("tier2_router.create_secure_client", return_value=fake_sqs):
                tier2_router.setup_queues_and_route_data(input_dir=temp_dir, confidence_threshold=90.0)
            self.assertTrue(any("Tier2_LayoutLM_Queue" in x[0] for x in fake_sqs.messages))

        mock_bedrock.return_value = {"corrected_text": "corrected", "confidence": 0.95, "reasoning": "ok"}
        mock_hallucination.return_value = {"is_hallucinated": False, "deviation_score": 0.1, "token_similarity": 0.9, "levenshtein_distance": 1}
        mock_merge.side_effect = lambda _orig, processed: processed
        out = tier3_processor.process_low_confidence_regions(
            low_confidence_regions=[{"text": "blurry", "confidence": 0.2, "bbox": [0, 0, 10, 10], "page_number": 1}],
            page_image=Image.new("RGB", (20, 20), "white"),
            surrounding_context_text="context",
            document_id="doc1",
        )
        self.assertIn(out["status"], {"SUCCESS", "REVIEW_REQUIRED"})
        self.assertEqual(len(out["audit_log"]), 1)

    @patch("track_a_snomed.cloudwatch_monitor", None)
    @patch("track_a_snomed.detect_phi_entities", return_value=[])
    @patch("track_a_snomed.comprehend_medical")
    @patch("track_a_snomed.semantic_snomed_fallback")
    def test_track_a_snomed_fallback_integration(self, mock_fallback, mock_cm, _mock_phi):
        mock_cm.infer_snomedct.return_value = {
            "Entities": [{"Text": "UnknownDx", "Score": 0.2, "SNOMEDCTConcepts": [], "Category": "MEDICAL_CONDITION", "Type": "DX_NAME"}],
            "ModelVersion": "1",
        }
        mock_fallback.return_value = {
            "snomed_code": "9999",
            "description": "Fallback Dx",
            "confidence": 0.8,
            "source": "semantic_fallback",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            p = os.path.join(temp_dir, "doc_textract.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"Blocks": [{"BlockType": "LINE", "Text": "UnknownDx"}]}, f)
            out = track_a_snomed.process_document(p)
            self.assertEqual(out["fallback_count"], 1)
            self.assertGreater(out["unified_confidence_score"], 0)

    @patch("track_b_summarization.TrackBPipeline._save_results")
    @patch("track_b_summarization.FAISSIndex")
    @patch("track_b_summarization.TitanEmbeddings")
    @patch("track_b_summarization.DocumentChunker")
    @patch("track_b_summarization.ClaudeSummarizer")
    @patch("track_b_summarization.SummaryValidator")
    def test_track_b_rag_summarization_integration(
        self, mock_validator_cls, mock_summarizer_cls, mock_chunker_cls, mock_embeddings_cls, mock_index_cls, _mock_save
    ):
        chunk = tb.DocumentChunk("c1", "chunk", tb.DocumentType.CLINICAL_NOTE, "full", 0, 5, {})
        mock_chunker_cls.return_value.chunk_document.return_value = [chunk]
        mock_embeddings_cls.return_value.embed_batch.return_value = MagicMock()
        mock_embeddings_cls.return_value.embed_text.return_value = MagicMock()
        mock_index_cls.return_value.search.return_value = [(chunk, 0.9)]
        mock_summarizer_cls.return_value.generate_summary.return_value = {
            "summary": "ok",
            "key_points": [],
            "medications": [],
            "diagnoses": [],
            "follow_up_actions": [],
            "confidence_score": 0.8,
            "prompt_tracking": {"selected_versions": {"medical_summarization": "v1"}},
        }
        mock_validator = MagicMock()
        mock_validator.validate.return_value = (True, [])
        mock_validator.get_last_report.return_value = {
            "corrected_output": mock_summarizer_cls.return_value.generate_summary.return_value,
            "validation_confidence_score": 0.9,
            "audit_log": [],
            "auto_corrections": [],
        }
        mock_validator.calculate_hallucination_score.return_value = 0.1
        mock_validator.compute_ocr_deviation_guard.return_value = {
            "flagged_for_review": False,
            "deviation_score": 0.0,
            "critical_term_results": [],
            "audit_checks": [],
        }
        mock_validator_cls.return_value = mock_validator
        results = tb.TrackBPipeline().process_document("doc text", "doc1")
        self.assertTrue(results["clinician"].validation_passed)
        self.assertIn("selected_versions", results["clinician"].prompt_tracking)

    def test_confidence_routing_integration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            textract = os.path.join(temp_dir, "textract.json")
            track_a = os.path.join(temp_dir, "track_a.json")
            track_b = os.path.join(temp_dir, "track_b.json")
            with open(textract, "w", encoding="utf-8") as f:
                json.dump({"Blocks": [{"BlockType": "LINE", "Confidence": 98.0}]}, f)
            with open(track_a, "w", encoding="utf-8") as f:
                json.dump({"categorized_entities": {}, "unified_confidence_score": 0.9}, f)
            with open(track_b, "w", encoding="utf-8") as f:
                json.dump({"confidence_score": 0.9}, f)

            with patch("lambda_confidence_aggregator.create_secure_client", return_value=_FakeSQS()), \
                 patch("lambda_confidence_aggregator.get_audit_logger", return_value=MagicMock()), \
                 patch("lambda_confidence_aggregator._get_monitor", return_value=None):
                response = agg.lambda_handler(
                    {
                        "document_id": "doc1",
                        "textract_json_path": textract,
                        "track_a_output_path": track_a,
                        "track_b_output_path": track_b,
                    },
                    None,
                )
            body = json.loads(response["body"])
            self.assertIn(body["route"], {"bypass_database", "human_review"})

    def test_dynamodb_audit_logging_integration(self):
        fake_table = MagicMock()
        fake_table.load.return_value = None
        fake_resource = MagicMock()
        fake_resource.Table.return_value = fake_table
        with patch("audit_dynamodb.create_secure_resource", return_value=fake_resource):
            logger = audit_dynamodb.AuditLogger()
            audit_id = logger.log_change(
                document_id="doc1",
                user_id="tester",
                change_type="INTEGRATION_TEST",
                before_state={"status": "a"},
                after_state={"status": "b"},
                metadata={"source": "integration"},
            )
            self.assertIsInstance(audit_id, str)
            fake_table.put_item.assert_called_once()

    @patch("track_a_snomed.time.sleep")
    @patch("track_a_snomed.process_document", side_effect=RuntimeError("transient"))
    def test_error_and_retry_logic(self, _mock_process, _mock_sleep):
        result, err = track_a_snomed.process_with_retry("missing.json", max_retries=2)
        self.assertIsNone(result)
        self.assertIn("transient", err)

    def test_pipeline_performance_benchmark_mocked(self):
        start = time.perf_counter()
        with patch("lambda_confidence_aggregator.route_document") as mock_route, \
             patch("lambda_confidence_aggregator.log_routing_audit"), \
             patch("lambda_confidence_aggregator._get_monitor", return_value=None):
            mock_route.return_value = {
                "route": "bypass_database",
                "queue_name": "q",
                "queue_url": "u",
                "routing_payload": {
                    "route": "bypass_database",
                    "final_confidence_score": 0.9,
                    "threshold": 0.85,
                    "component_scores": {"textract": 0.9, "comprehend": 0.9, "faiss": 0.9, "llm_logprobs": 0.9},
                    "weights": agg.DEFAULT_WEIGHTS,
                    "calculation_latency_ms": 1.0,
                    "calculation_latency_sla_met": True,
                },
            }
            agg.lambda_handler(
                {
                    "document_id": "bench",
                    "textract_confidence": 0.9,
                    "comprehend_confidence": 0.9,
                    "faiss_similarity": 0.9,
                    "llm_logprobs_confidence": 0.9,
                },
                None,
            )
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)


@unittest.skipUnless(os.getenv("STAGING_AWS_TESTS") == "1", "Set STAGING_AWS_TESTS=1 to run against staging AWS")
class TestStagingAWSSmoke(unittest.TestCase):
    @patch("lambda_confidence_aggregator.route_document")
    def test_staging_mode_smoke(self, mock_route):
        mock_route.return_value = {
            "route": "bypass_database",
            "queue_name": "q",
            "queue_url": "u",
            "routing_payload": {
                "route": "bypass_database",
                "final_confidence_score": 0.9,
                "threshold": 0.85,
                "component_scores": {},
                "weights": agg.DEFAULT_WEIGHTS,
                "calculation_latency_ms": 1.0,
                "calculation_latency_sla_met": True,
            },
        }
        response = agg.lambda_handler({"document_id": "staging-smoke", "textract_confidence": 0.95}, None)
        self.assertEqual(response["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()
