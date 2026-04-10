import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import tier1_textract


class TestTier1Textract(unittest.TestCase):
    @patch("tier1_textract.infer_document_type", return_value="unknown")
    @patch("tier1_textract.CloudWatchMonitoringManager")
    @patch("tier1_textract.build_phi_detection_summary", return_value={"entity_count": 1})
    @patch("tier1_textract.detect_phi_entities", return_value=[{"Text": "John"}])
    @patch("tier1_textract.create_secure_client")
    def test_process_documents_success(
        self,
        mock_client,
        _mock_detect,
        _mock_phi_summary,
        mock_monitor_cls,
        _mock_doc_type,
    ):
        textract_client = MagicMock()
        textract_client.analyze_document.return_value = {
            "Blocks": [{"BlockType": "LINE", "Text": "Patient has fever", "Confidence": 99.0}]
        }
        mock_client.return_value = textract_client
        monitor = MagicMock()
        mock_monitor_cls.return_value = monitor

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "temp_pages")
            output_dir = os.path.join(temp_dir, "textract_outputs")
            os.makedirs(input_dir, exist_ok=True)

            image_path = os.path.join(input_dir, "doc_page_1_CLEANED.jpg")
            with open(image_path, "wb") as f:
                f.write(b"fake-image-bytes")

            tier1_textract.process_documents_with_textract(input_dir=input_dir, output_dir=output_dir)

            output_file = os.path.join(output_dir, "doc_page_1_CLEANED_textract.json")
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "tier1_latency_profile.json")))
            with open(output_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertIn("PhiDetection", saved)
            monitor.publish_extraction_result.assert_called()

    @patch("tier1_textract.infer_document_type", return_value="unknown")
    @patch("tier1_textract.CloudWatchMonitoringManager")
    @patch("tier1_textract.create_secure_client")
    def test_process_documents_handles_textract_error(
        self, mock_client, mock_monitor_cls, _mock_doc_type
    ):
        textract_client = MagicMock()
        textract_client.analyze_document.side_effect = RuntimeError("Textract failed")
        mock_client.return_value = textract_client
        monitor = MagicMock()
        mock_monitor_cls.return_value = monitor

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "temp_pages")
            output_dir = os.path.join(temp_dir, "textract_outputs")
            os.makedirs(input_dir, exist_ok=True)

            image_path = os.path.join(input_dir, "doc_error_CLEANED.jpg")
            with open(image_path, "wb") as f:
                f.write(b"fake-image-bytes")

            tier1_textract.process_documents_with_textract(input_dir=input_dir, output_dir=output_dir)
            monitor.publish_extraction_result.assert_called()
            kwargs = monitor.publish_extraction_result.call_args.kwargs
            self.assertFalse(kwargs["success"])


if __name__ == "__main__":
    unittest.main()
