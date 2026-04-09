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

    def test_ocr_deviation_guard_flags_terms_deviating_from_both_ocr_sources(self):
        summary_output = {
            "summary": "Patient has zebra syndrome and was prescribed Xenodrug 999 mg with alien catheterization.",
            "key_points": ["Xenodrug 999 mg daily"],
            "medications": [{"name": "Xenodrug", "dosage": "999 mg", "frequency": "daily", "instructions": ""}],
            "diagnoses": ["zebra syndrome"],
            "follow_up_actions": ["urgent consult"],
            "confidence_score": 0.7,
        }
        textract_text = "Patient has hypertension and takes Metformin 1000 mg."
        layout_text = "Diagnosis: hypertension. Medication: lisinopril 20 mg."
        guard = self.engine.compute_ocr_deviation_guard(
            summary_output=summary_output,
            textract_source_text=textract_text,
            layoutlm_source_text=layout_text,
            deviation_threshold=0.30,
        )
        self.assertTrue(guard["flagged_for_review"])
        self.assertGreater(guard["deviation_score"], 0.30)

    def test_ocr_deviation_guard_passes_when_terms_match_sources(self):
        summary_output = {
            "summary": "Patient with hypertension on Metformin 1000 mg.",
            "key_points": ["Continue metformin"],
            "medications": [{"name": "Metformin", "dosage": "1000 mg", "frequency": "daily", "instructions": ""}],
            "diagnoses": ["hypertension"],
            "follow_up_actions": ["follow-up"],
            "confidence_score": 0.9,
        }
        source = "Patient diagnosed with hypertension. Current medications: Metformin 1000 mg."
        guard = self.engine.compute_ocr_deviation_guard(
            summary_output=summary_output,
            textract_source_text=source,
            layoutlm_source_text=source,
            deviation_threshold=0.30,
        )
        self.assertFalse(guard["flagged_for_review"])
        self.assertLessEqual(guard["deviation_score"], 0.30)


if __name__ == "__main__":
    unittest.main()
