import unittest

from track_b_validation import MedicalValidationEngine


class TestTrackBValidation(unittest.TestCase):
    def setUp(self):
        self.engine = MedicalValidationEngine()
        self.source_text = (
            "Patient has hypertension and type 2 diabetes mellitus. "
            "Current medications include Metformin 1000 mg and Lisinopril 20 mg daily. "
            "Follow-up in two weeks."
        )

    def test_schema_and_rule_validation_passes_for_valid_payload(self):
        payload = {
            "summary": "Patient diagnosed with hypertension and type 2 diabetes mellitus.",
            "key_points": ["Continue medications", "Follow-up in two weeks"],
            "medications": [
                {"name": "Metformin", "dosage": "1000 mg", "frequency": "twice daily", "instructions": "with food"},
                {"name": "Lisinopril", "dosage": "20 mg", "frequency": "daily", "instructions": "morning"},
            ],
            "diagnoses": ["hypertension", "type 2 diabetes mellitus"],
            "follow_up_actions": ["Primary care follow-up in two weeks"],
            "confidence_score": 0.88,
        }
        report = self.engine.validate(payload, self.source_text)
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["errors"], [])
        self.assertGreaterEqual(report["validation_confidence_score"], 0.7)

    def test_auto_correct_for_string_medication_and_confidence(self):
        payload = {
            "summary": "Patient is on metformin.",
            "key_points": [],
            "medications": ["Metformin"],
            "diagnoses": ["hypertension"],
            "follow_up_actions": [],
            "confidence_score": 1.7,
        }
        report = self.engine.validate(payload, self.source_text)
        corrected = report["corrected_output"]
        self.assertIsInstance(corrected["medications"][0], dict)
        self.assertEqual(corrected["confidence_score"], 1.0)
        self.assertTrue(any("Clamped confidence_score" in msg for msg in report["auto_corrections"]))

    def test_dosage_validation_catches_invalid_format(self):
        payload = {
            "summary": "Medication list reviewed.",
            "key_points": [],
            "medications": [{"name": "Metformin", "dosage": "1000", "frequency": "daily", "instructions": ""}],
            "diagnoses": ["hypertension"],
            "follow_up_actions": [],
            "confidence_score": 0.8,
        }
        report = self.engine.validate(payload, self.source_text)
        self.assertFalse(report["validation_passed"])
        self.assertTrue(any("Invalid dosage format" in e for e in report["errors"]))

    def test_hallucination_scoring_flags_unsupported_content(self):
        payload = {
            "summary": "Patient has rare zebra syndrome and probable alien pathogen.",
            "key_points": ["Likely extraterrestrial etiology"],
            "medications": [{"name": "Xenodrug", "dosage": "10 mg", "frequency": "daily", "instructions": ""}],
            "diagnoses": ["zebra syndrome"],
            "follow_up_actions": ["Space medicine consult"],
            "confidence_score": 0.6,
        }
        report = self.engine.validate(payload, self.source_text)
        self.assertGreater(report["hallucination_score"], 0.25)


if __name__ == "__main__":
    unittest.main()
