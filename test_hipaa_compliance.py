import unittest
from unittest.mock import MagicMock, patch

from hipaa_compliance import (
    build_phi_detection_summary,
    create_secure_client,
    detect_phi_entities,
    mask_text_by_entities,
    scrub_json_value,
)


class TestHipaaCompliance(unittest.TestCase):
    def test_mask_text_by_entities_masks_detected_spans(self):
        text = "Patient John Doe MRN: 123456 visited on 01/10/2024."
        entities = [
            {"type": "NAME", "begin_offset": 8, "end_offset": 16},
            {"type": "ID", "begin_offset": 22, "end_offset": 33},
        ]
        masked = mask_text_by_entities(text, entities)
        self.assertNotIn("John Doe", masked)
        self.assertNotIn("MRN: 123456", masked)
        self.assertIn("[DATE_REDACTED]", masked)

    def test_phi_summary_counts(self):
        entities = [
            {"type": "NAME", "text": "John Doe"},
            {"type": "NAME", "text": "Jane Doe"},
            {"type": "DATE", "text": "01/01/1990"},
        ]
        summary = build_phi_detection_summary(entities)
        self.assertTrue(summary["phi_detected"])
        self.assertEqual(summary["entity_count"], 3)
        self.assertEqual(summary["counts_by_type"]["NAME"], 2)

    def test_detect_phi_entities_aggregates_chunks(self):
        long_text = "A" * 19050
        mock_client = MagicMock()
        mock_client.detect_phi.side_effect = [
            {"Entities": [{"Type": "NAME", "BeginOffset": 5, "EndOffset": 10, "Score": 0.99}]},
            {"Entities": [{"Type": "DATE", "BeginOffset": 3, "EndOffset": 8, "Score": 0.98}]},
        ]
        entities = detect_phi_entities(long_text, comprehend_medical_client=mock_client)
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[1]["begin_offset"], 18003)

    def test_create_secure_client_rejects_http_endpoint(self):
        with self.assertRaises(RuntimeError):
            create_secure_client("s3", endpoint_url="http://localhost:4566")

    def test_scrub_json_value_masks_nested_strings(self):
        data = {"summary": "Patient John Doe lives at 12 Main Street", "list": ["DOB 01/10/1999"]}
        scrubbed = scrub_json_value(data)
        self.assertNotIn("John Doe", scrubbed["summary"])
        self.assertIn("REDACTED", scrubbed["list"][0])

    @patch("hipaa_compliance.boto3.client")
    def test_create_secure_client_accepts_https(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.meta.endpoint_url = "https://s3.us-east-1.amazonaws.com"
        mock_client_fn.return_value = mock_client
        client = create_secure_client("s3")
        self.assertEqual(client, mock_client)


if __name__ == "__main__":
    unittest.main()
