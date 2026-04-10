import json
import unittest
from unittest.mock import MagicMock, patch

import emis_export_integration as emis


class TestEmisExportIntegration(unittest.TestCase):
    def setUp(self):
        emis.EMIS_MAX_ATTEMPTS = 2
        emis.EMIS_TRANSPORT = "api"
        emis.EMIS_FORMAT = "proprietary_json"
        emis.EMIS_API_BASE_URL = "https://emis.example.com"
        emis.EMIS_API_PATH = "/documents/import"
        emis.EMIS_API_TOKEN = "token"
        emis.EMIS_RETRY_QUEUE_NAME = "retry-q"

    def test_export_success_logs_event(self):
        audit_logger = MagicMock()
        with patch("emis_export_integration._send_to_emis", return_value={"status_code": 200, "response_body": "ok"}):
            result = emis.export_to_emis(
                document_id="doc1",
                validated_payload={"field": "value"},
                user_id="u1",
                audit_logger=audit_logger,
            )
        self.assertTrue(result["success"])
        self.assertEqual(audit_logger.log_change.call_count, 1)

    def test_export_failure_queues_retry(self):
        audit_logger = MagicMock()
        fake_sqs = MagicMock()
        fake_sqs.get_queue_url.return_value = {"QueueUrl": "https://q/retry"}
        with patch("emis_export_integration._send_to_emis", side_effect=RuntimeError("down")), \
             patch("emis_export_integration.create_secure_client", return_value=fake_sqs), \
             patch("emis_export_integration.time.sleep", return_value=None):
            result = emis.export_to_emis(
                document_id="doc2",
                validated_payload={"field": "value"},
                user_id="u2",
                audit_logger=audit_logger,
            )
        self.assertFalse(result["success"])
        self.assertTrue(result["queued_for_retry"])
        fake_sqs.send_message.assert_called_once()
        self.assertGreaterEqual(audit_logger.log_change.call_count, 3)

    def test_tls_enforcement_on_non_https(self):
        emis.EMIS_API_BASE_URL = "http://insecure.example.com"
        with self.assertRaises(RuntimeError):
            emis._send_via_api({"document_id": "doc", "payload": {}})

    def test_file_drop_transport(self):
        emis.EMIS_TRANSPORT = "s3_file_drop"
        emis.EMIS_FILE_DROP_BUCKET = "bucket-x"
        fake_s3 = MagicMock()
        with patch("emis_export_integration.create_secure_client", return_value=fake_s3):
            result = emis._send_to_emis({"document_id": "doc3", "payload": {}})
        self.assertEqual(result["status_code"], 200)
        fake_s3.put_object.assert_called_once()

    def test_retry_message_success(self):
        audit_logger = MagicMock()
        with patch("emis_export_integration._send_to_emis", return_value={"status_code": 200, "response_body": "ok"}):
            result = emis.process_retry_message(
                {"document_id": "doc4", "attempts": 2, "export_payload": {"document_id": "doc4"}},
                audit_logger=audit_logger,
            )
        self.assertTrue(result["success"])
        audit_logger.log_change.assert_called_once()


if __name__ == "__main__":
    unittest.main()
