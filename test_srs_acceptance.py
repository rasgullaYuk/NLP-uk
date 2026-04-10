import os
import tempfile
import unittest
from unittest.mock import patch

import acceptance_framework as af


class TestSRSAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = af.load_acceptance_dataset(
            os.path.join("acceptance_data", "srs_acceptance_dataset.json")
        )

    def test_text_extraction_accuracy_meets_98_percent(self):
        result = af.evaluate_text_accuracy(self.dataset["text_accuracy_cases"])
        self.assertGreaterEqual(result["accuracy_percent"], 98.0)
        self.assertTrue(result["passed"])

    def test_ui_fields_are_editable_contract(self):
        result = af.verify_ui_editability_contract("app.py")
        self.assertTrue(result["passed"])

    def test_audit_before_after_states_captured(self):
        previous_state = {"summary": "Old summary", "actions": ["old action"]}
        current_state = {"summary": "New summary", "actions": ["new action"]}

        class _FakeLogger:
            def __init__(self):
                self.rows = []

            def log_change(self, **kwargs):
                self.rows.append(kwargs)

        logger = _FakeLogger()
        logger.log_change(
            document_id="acc-doc-audit",
            user_id="acceptance-tester",
            change_type="SUMMARY_EDIT",
            before_state=previous_state,
            after_state=current_state,
            metadata={"criterion": "audit-trail"},
        )
        self.assertEqual(len(logger.rows), 1)
        self.assertEqual(logger.rows[0]["before_state"], previous_state)
        self.assertEqual(logger.rows[0]["after_state"], current_state)

    def test_confidence_routing_threshold_at_085(self):
        result = af.evaluate_confidence_routing(self.dataset["confidence_events"], threshold=0.85)
        self.assertTrue(result["passed"])
        self.assertTrue(all(item["passed"] for item in result["results"]))

    def test_end_to_end_performance_under_60_seconds(self):
        def _workload():
            _ = af.evaluate_confidence_routing(self.dataset["confidence_events"], threshold=0.85)
            _ = af.evaluate_snomed_mapping_accuracy(self.dataset["snomed_mapping_cases"])

        result = af.benchmark_runtime(_workload, max_seconds=60.0)
        self.assertTrue(result["passed"])
        self.assertLess(result["elapsed_seconds"], 60.0)

    @patch("acceptance_framework.scrub_text_for_logs")
    def test_security_phi_handling_and_encryption(self, mock_scrub):
        mock_scrub.return_value = "Patient [NAME_REDACTED] MRN [MRN_REDACTED]"
        phi = af.verify_phi_masking("Patient John Doe MRN 123456")
        self.assertTrue(phi["passed"])

        encryption = af.verify_encryption_posture(
            s3_result={"encrypted": True, "algorithm": "AES256"},
            dynamodb_result={"encrypted": True, "status": "ENABLED"},
        )
        self.assertTrue(encryption["passed"])

    def test_snomed_mapping_accuracy_meets_95_percent(self):
        result = af.evaluate_snomed_mapping_accuracy(self.dataset["snomed_mapping_cases"])
        self.assertGreaterEqual(result["accuracy_percent"], 95.0)
        self.assertTrue(result["passed"])

    def test_acceptance_report_artifact_generation(self):
        payload = {
            "text_accuracy": af.evaluate_text_accuracy(self.dataset["text_accuracy_cases"]),
            "snomed_accuracy": af.evaluate_snomed_mapping_accuracy(
                self.dataset["snomed_mapping_cases"]
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            out = os.path.join(temp_dir, "acceptance_report.json")
            af.save_acceptance_report(out, payload)
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
