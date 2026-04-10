import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import track_b_summarization as tb


class _FakeBody:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode("utf-8")


class TestTrackBSummarizationUnit(unittest.TestCase):
    def test_faiss_index_search_save_load_with_fake_faiss(self):
        class FakeIndex:
            def __init__(self):
                self.ntotal = 0
                self._vectors = []

            def add(self, embeddings):
                self._vectors.extend(list(embeddings))
                self.ntotal = len(self._vectors)

            def search(self, query, k):
                _ = query
                valid = list(range(min(k, self.ntotal)))
                padded = valid + ([-1] * max(0, k - len(valid)))
                return np.array([[0.9] * k], dtype=np.float32), np.array([padded], dtype=np.int64)

        class FakeFaissModule:
            IndexFlatIP = lambda self, dim: FakeIndex()

            @staticmethod
            def write_index(index, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("ok")

            @staticmethod
            def read_index(path):
                _ = path
                idx = FakeIndex()
                idx.ntotal = 1
                return idx

        fake_module = FakeFaissModule()
        with patch.dict(sys.modules, {"faiss": fake_module}):
            index = tb.FAISSIndex(embedding_dim=3)
            chunk = tb.DocumentChunk("c1", "hello", tb.DocumentType.UNKNOWN, "full", 0, 5, {})
            index.add_chunks([chunk], np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
            results = index.search(np.array([0.1, 0.2, 0.3], dtype=np.float32), k=1)
            self.assertEqual(len(results), 1)
            with tempfile.TemporaryDirectory() as tmp:
                base = os.path.join(tmp, "idx")
                index.save(base)
                index.load(base)
                self.assertIsInstance(index.chunks, list)

    def test_document_chunker_detect_and_chunk(self):
        chunker = tb.DocumentChunker(chunk_size=40, chunk_overlap=5)
        txt = "Discharge summary\nChief complaint\nChest pain.\nPlan\nContinue meds."
        doc_type = chunker.detect_document_type(txt)
        self.assertEqual(doc_type, tb.DocumentType.DISCHARGE_SUMMARY)
        chunks = chunker.chunk_document(txt * 5, "docA")
        self.assertGreaterEqual(len(chunks), 1)

    @patch("track_b_summarization.create_secure_client")
    def test_titan_embeddings_embed_text_and_batch(self, mock_client):
        fake_runtime = MagicMock()
        fake_runtime.invoke_model.return_value = {"body": _FakeBody(json.dumps({"embedding": [0.1, 0.2, 0.3]}))}
        mock_client.return_value = fake_runtime
        emb = tb.TitanEmbeddings()
        one = emb.embed_text("hello")
        many = emb.embed_batch(["a", "b"])
        self.assertEqual(one.shape[0], 3)
        self.assertEqual(many.shape[0], 2)

    @patch("track_b_summarization.create_secure_client")
    def test_titan_embeddings_cache_reduces_calls(self, mock_client):
        fake_runtime = MagicMock()
        fake_runtime.invoke_model.return_value = {"body": _FakeBody(json.dumps({"embedding": [0.1, 0.2, 0.3]}))}
        mock_client.return_value = fake_runtime
        emb = tb.TitanEmbeddings()
        emb.embed_batch(["repeat", "repeat", "repeat"])
        self.assertEqual(fake_runtime.invoke_model.call_count, 1)

    @patch("track_b_summarization.create_secure_client")
    @patch("track_b_summarization.BedrockPromptManager")
    def test_claude_generate_summary_includes_prompt_tracking(
        self, mock_prompt_manager_cls, mock_secure_client
    ):
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.compose_track_b_prompt.return_value = (
            "prompt text",
            {"selected_versions": {"medical_summarization": "v1"}},
        )
        mock_prompt_manager_cls.return_value = mock_prompt_manager

        fake_runtime = MagicMock()
        fake_runtime.invoke_model.return_value = {
            "body": _FakeBody(
                json.dumps(
                    {
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "summary": "ok",
                                        "key_points": [],
                                        "medications": [],
                                        "diagnoses": [],
                                        "follow_up_actions": [],
                                        "confidence_score": 0.8,
                                    }
                                )
                            }
                        ]
                    }
                )
            )
        }
        mock_secure_client.return_value = fake_runtime

        summarizer = tb.ClaudeSummarizer()
        result = summarizer.generate_summary(
            document_text="Patient has hypertension",
            document_id="doc1",
            role=tb.SummaryRole.CLINICIAN,
            retrieved_context=["ctx1"],
            document_type=tb.DocumentType.CLINICAL_NOTE,
        )
        self.assertIn("prompt_tracking", result)
        self.assertEqual(
            result["prompt_tracking"]["selected_versions"]["medical_summarization"], "v1"
        )

    def test_summary_validator_uses_engine_report(self):
        validator = tb.SummaryValidator()
        fake_report = {
            "validation_passed": True,
            "errors": [],
            "hallucination_score": 0.11,
            "corrected_output": {"summary": "ok"},
            "audit_log": [],
            "auto_corrections": [],
            "validation_confidence_score": 0.9,
        }
        validator.engine = MagicMock()
        validator.engine.validate.return_value = fake_report

        ok, errors = validator.validate({"summary": "x"}, "source", tb.DocumentType.UNKNOWN)
        self.assertTrue(ok)
        self.assertEqual(errors, [])
        self.assertAlmostEqual(validator.calculate_hallucination_score({}, ""), 0.11)
        self.assertIsInstance(validator.get_last_report(), dict)

    @patch("track_b_summarization.create_secure_client")
    @patch("track_b_summarization.BedrockPromptManager")
    def test_claude_generate_summary_json_fallback(self, mock_prompt_manager_cls, mock_secure_client):
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.compose_track_b_prompt.return_value = ("p", {"selected_versions": {}})
        mock_prompt_manager_cls.return_value = mock_prompt_manager
        fake_runtime = MagicMock()
        fake_runtime.invoke_model.return_value = {"body": _FakeBody(json.dumps({"content": [{"text": "not-json"}]}))}
        mock_secure_client.return_value = fake_runtime
        out = tb.ClaudeSummarizer().generate_summary(
            document_text="x",
            document_id="d",
            role=tb.SummaryRole.PATIENT,
            retrieved_context=[],
            document_type=tb.DocumentType.UNKNOWN,
        )
        self.assertIn("summary", out)
        self.assertIn("prompt_tracking", out)

    def test_save_results_masks_non_clinician(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("track_b_summarization.SUMMARY_OUTPUT_DIR", tmp):
                with patch("track_b_summarization.create_secure_client") as mock_client, \
                     patch("track_b_summarization.FAISSIndex"), \
                     patch("track_b_summarization.TitanEmbeddings"), \
                     patch("track_b_summarization.DocumentChunker"), \
                     patch("track_b_summarization.ClaudeSummarizer"), \
                     patch("track_b_summarization.SummaryValidator"):
                    mock_client.return_value = MagicMock()
                    pipeline = tb.TrackBPipeline()
                sample = tb.SummaryOutput(
                    document_id="doc1",
                    role=tb.SummaryRole.PATIENT,
                    summary="John Doe has diabetes",
                    key_points=["John Doe details"],
                    medications=[{"name": "Metformin", "dosage": "500 mg", "frequency": "daily", "instructions": ""}],
                    diagnoses=["diabetes"],
                    follow_up_actions=["Follow up with John Doe"],
                    confidence_score=0.8,
                    validation_passed=True,
                    validation_errors=[],
                    validation_confidence_score=0.9,
                    hallucination_score=0.1,
                    ocr_deviation_flag=False,
                    ocr_deviation_score=0.0,
                    ocr_deviation_details=[],
                    validation_audit_log=[],
                    auto_corrections=[],
                    generation_time_ms=10,
                    prompt_tracking={},
                )
                pipeline._save_results("doc1", {"patient": sample}, phi_entities=[{"Text": "John Doe", "Type": "NAME"}])
                self.assertTrue(os.path.exists(os.path.join(tmp, "doc1_patient_summary.json")))

    def test_process_track_b_queue_single_message(self):
        sqs_messaging = types.SimpleNamespace()
        sqs_setup = types.SimpleNamespace()
        audit_dynamodb = types.SimpleNamespace()
        sqs_messaging.receive_from_sqs = MagicMock(side_effect=[[{"Body": json.dumps({"document_id": "d1", "text": "t", "source_file": None, "layout_refined_file": None}), "ReceiptHandle": "rh"}], KeyboardInterrupt()])
        sqs_messaging.delete_from_sqs = MagicMock()
        sqs_setup.get_queue_url = MagicMock(return_value="queue-url")
        audit_logger = MagicMock()
        audit_dynamodb.get_audit_logger = MagicMock(return_value=audit_logger)

        fake_sqs = MagicMock()
        fake_sqs.get_queue_url.return_value = {"QueueUrl": "review-url"}
        fake_pipeline = MagicMock()
        fake_pipeline.process_document.return_value = {
            "clinician": tb.SummaryOutput(
                document_id="d1",
                role=tb.SummaryRole.CLINICIAN,
                summary="s",
                key_points=[],
                medications=[],
                diagnoses=[],
                follow_up_actions=[],
                confidence_score=0.8,
                validation_passed=True,
                validation_errors=[],
                validation_confidence_score=0.9,
                hallucination_score=0.1,
                ocr_deviation_flag=True,
                ocr_deviation_score=0.7,
                ocr_deviation_details=[],
                validation_audit_log=[],
                auto_corrections=[],
                generation_time_ms=5,
                prompt_tracking={},
            )
        }
        with patch.dict(sys.modules, {"sqs_messaging": sqs_messaging, "sqs_setup": sqs_setup, "audit_dynamodb": audit_dynamodb}), \
             patch("track_b_summarization.create_secure_client", return_value=fake_sqs), \
             patch("track_b_summarization.TrackBPipeline", return_value=fake_pipeline), \
             patch("track_b_summarization.detect_phi_entities", return_value=[]):
            with self.assertRaises(KeyboardInterrupt):
                tb.process_track_b_queue("TrackB_Summary_Queue")
            fake_sqs.send_message.assert_called()

    @patch("track_b_summarization.TrackBPipeline._save_results")
    @patch("track_b_summarization.FAISSIndex")
    @patch("track_b_summarization.TitanEmbeddings")
    @patch("track_b_summarization.DocumentChunker")
    @patch("track_b_summarization.ClaudeSummarizer")
    @patch("track_b_summarization.SummaryValidator")
    def test_pipeline_process_document_calls_rag_and_validation(
        self,
        mock_validator_cls,
        mock_summarizer_cls,
        mock_chunker_cls,
        mock_embeddings_cls,
        mock_index_cls,
        _mock_save,
    ):
        chunk = tb.DocumentChunk(
            chunk_id="c1",
            text="sample chunk text",
            document_type=tb.DocumentType.CLINICAL_NOTE,
            section="full_document",
            start_pos=0,
            end_pos=10,
            metadata={},
        )

        mock_chunker = MagicMock()
        mock_chunker.chunk_document.return_value = [chunk]
        mock_chunker_cls.return_value = mock_chunker

        mock_embeddings = MagicMock()
        mock_embeddings.embed_batch.return_value = MagicMock()
        mock_embeddings.embed_text.return_value = MagicMock()
        mock_embeddings_cls.return_value = mock_embeddings

        mock_index = MagicMock()
        mock_index.search.return_value = [(chunk, 0.9)]
        mock_index_cls.return_value = mock_index

        mock_summarizer = MagicMock()
        mock_summarizer.generate_summary.return_value = {
            "summary": "Patient summary",
            "key_points": [],
            "medications": [],
            "diagnoses": [],
            "follow_up_actions": [],
            "confidence_score": 0.8,
            "prompt_tracking": {"selected_versions": {"medical_summarization": "v1"}},
        }
        mock_summarizer_cls.return_value = mock_summarizer

        mock_validator = MagicMock()
        mock_validator.validate.return_value = (True, [])
        mock_validator.get_last_report.return_value = {
            "corrected_output": mock_summarizer.generate_summary.return_value,
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

        pipeline = tb.TrackBPipeline()
        output = pipeline.process_document("sample doc", "doc-123")
        self.assertIn("clinician", output)
        self.assertTrue(output["clinician"].validation_passed)
        mock_summarizer.generate_summary.assert_called()

    @patch("track_b_summarization.TrackBPipeline._save_results")
    @patch("track_b_summarization.FAISSIndex")
    @patch("track_b_summarization.TitanEmbeddings")
    @patch("track_b_summarization.DocumentChunker")
    @patch("track_b_summarization.ClaudeSummarizer")
    @patch("track_b_summarization.SummaryValidator")
    def test_pipeline_handles_generation_exception(
        self,
        mock_validator_cls,
        mock_summarizer_cls,
        mock_chunker_cls,
        mock_embeddings_cls,
        mock_index_cls,
        _mock_save,
    ):
        chunk = tb.DocumentChunk("c1", "text", tb.DocumentType.UNKNOWN, "full", 0, 4, {})
        mock_chunker_cls.return_value.chunk_document.return_value = [chunk]
        mock_embeddings_cls.return_value.embed_batch.return_value = np.array([[0.1, 0.2]])
        mock_embeddings_cls.return_value.embed_text.return_value = np.array([0.1, 0.2])
        mock_index_cls.return_value.search.return_value = [(chunk, 0.9)]
        mock_summarizer_cls.return_value.generate_summary.side_effect = RuntimeError("bedrock down")
        mock_validator_cls.return_value = MagicMock()

        pipeline = tb.TrackBPipeline()
        output = pipeline.process_document("sample doc", "doc-err", roles=[tb.SummaryRole.CLINICIAN])
        self.assertFalse(output["clinician"].validation_passed)


if __name__ == "__main__":
    unittest.main()
