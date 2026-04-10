import base64
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import api_gateway_rest as api


class TestApiGatewayRest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.orig_state_dir = api.API_STATE_DIR
        api.API_STATE_DIR = self.temp_dir.name
        api.AUTH_API_KEYS = ["test-key"]
        api.AUTH_BEARER_TOKENS = ["token-1"]
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        api.API_STATE_DIR = self.orig_state_dir

    def _event(self, method, path, body=None, headers=None, is_base64=False):
        return {
            "httpMethod": method,
            "path": path,
            "headers": headers or {"x-api-key": "test-key"},
            "body": body,
            "isBase64Encoded": is_base64,
        }

    def test_unauthorized_request(self):
        event = self._event("GET", "/documents/doc1/status", headers={"x-api-key": "invalid"})
        response = api.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 401)

    def test_upload_success(self):
        api.UPLOAD_BUCKET = "bucket-a"
        fake_s3 = MagicMock()
        with patch("api_gateway_rest.create_secure_client", return_value=fake_s3):
            payload = base64.b64encode(b"pdf-bytes").decode("utf-8")
            event = self._event(
                "POST",
                "/documents/upload",
                body=payload,
                is_base64=True,
                headers={"x-api-key": "test-key", "x-filename": "sample.pdf", "content-type": "application/pdf"},
            )
            response = api.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 201)
        body = json.loads(response["body"])
        self.assertTrue(body["doc_id"].startswith("doc-"))
        fake_s3.put_object.assert_called_once()

    def test_status_and_approve(self):
        doc_id = "doc-abc-123"
        api._save_doc_state(doc_id, {"doc_id": doc_id, "status": "uploaded"})
        status_response = api.lambda_handler(self._event("GET", f"/documents/{doc_id}/status"), None)
        self.assertEqual(status_response["statusCode"], 200)

        fake_logger = MagicMock()
        approve_event = self._event(
            "PUT",
            f"/documents/{doc_id}/approve",
            body=json.dumps({"approved_by": "user-a"}),
        )
        with patch("api_gateway_rest.get_audit_logger", return_value=fake_logger):
            approve_response = api.lambda_handler(approve_event, None)
        self.assertEqual(approve_response["statusCode"], 200)
        fake_logger.log_change.assert_called_once()

    def test_extraction_snomed_summary_and_audit(self):
        doc_id = "docxyz"
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as work_dir:
            os.chdir(work_dir)
            try:
                os.makedirs("textract_outputs", exist_ok=True)
                os.makedirs("track_a_outputs", exist_ok=True)
                os.makedirs("track_b_outputs", exist_ok=True)

                with open(os.path.join("textract_outputs", f"{doc_id}_textract.json"), "w", encoding="utf-8") as f:
                    json.dump({"Blocks": [{"BlockType": "LINE", "Text": "hello world"}]}, f)
                with open(os.path.join("track_a_outputs", f"{doc_id}_snomed.json"), "w", encoding="utf-8") as f:
                    json.dump({"categorized_entities": {"Diagnosis": []}, "unified_confidence_score": 0.9}, f)
                with open(os.path.join("track_b_outputs", f"{doc_id}_clinician_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"role": "clinician", "summary": "summary text", "confidence_score": 0.8}, f)

                r1 = api.lambda_handler(self._event("GET", f"/documents/{doc_id}/extraction"), None)
                r2 = api.lambda_handler(self._event("GET", f"/documents/{doc_id}/snomed"), None)
                r3 = api.lambda_handler(self._event("GET", f"/documents/{doc_id}/summary"), None)
                self.assertEqual(r1["statusCode"], 200)
                self.assertEqual(r2["statusCode"], 200)
                self.assertEqual(r3["statusCode"], 200)

                fake_logger = MagicMock()
                fake_logger.get_audit_trail_by_document.return_value = [{"change_type": "X"}]
                with patch("api_gateway_rest.get_audit_logger", return_value=fake_logger):
                    r4 = api.lambda_handler(self._event("GET", f"/audit/{doc_id}"), None)
                self.assertEqual(r4["statusCode"], 200)
            finally:
                os.chdir(cwd)

    def test_options_cors(self):
        event = self._event("OPTIONS", "/documents/upload", headers={})
        response = api.lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertIn("Access-Control-Allow-Origin", response["headers"])


if __name__ == "__main__":
    unittest.main()
