import json
import os
import tempfile
import unittest

from review_interface_utils import (
    ACTION_PRIORITY_OPTIONS,
    compute_confidence_bundle,
    confidence_band,
    discover_document_assets,
    infer_document_id,
    infer_summary_role_from_filename,
    load_snomed_entities,
    normalize_action_items,
    serialize_action_items,
)


class TestReviewInterfaceUtils(unittest.TestCase):
    def test_document_id_and_role_inference(self):
        self.assertEqual(infer_document_id("page_1_clinician_summary.json"), "page_1")
        self.assertEqual(infer_document_id("page_1_snomed.json"), "page_1")
        self.assertEqual(infer_document_id("page_1_textract.json"), "page_1")
        self.assertEqual(infer_summary_role_from_filename("doc_patient_summary.txt"), "patient")
        self.assertEqual(
            infer_summary_role_from_filename("doc_pharmacist_summary.json"),
            "pharmacist",
        )
        self.assertEqual(infer_summary_role_from_filename("doc_summary.txt"), "clinician")

    def test_discover_document_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_dir = os.path.join(temp_dir, "track_b_outputs")
            snomed_dir = os.path.join(temp_dir, "track_a_outputs")
            textract_dir = os.path.join(temp_dir, "textract_outputs")
            confidence_dir = os.path.join(temp_dir, "confidence_outputs")
            os.makedirs(summary_dir, exist_ok=True)
            os.makedirs(snomed_dir, exist_ok=True)
            os.makedirs(textract_dir, exist_ok=True)
            os.makedirs(confidence_dir, exist_ok=True)

            with open(
                os.path.join(summary_dir, "doc_001_clinician_summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump({"summary": "Example"}, handle)

            with open(
                os.path.join(summary_dir, "doc_001_patient_summary.txt"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("Patient summary")

            with open(
                os.path.join(snomed_dir, "doc_001_snomed.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump({"Entities": []}, handle)

            with open(
                os.path.join(textract_dir, "doc_001_textract.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump({"Blocks": []}, handle)

            with open(
                os.path.join(confidence_dir, "doc_001_confidence.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump({"final_confidence_score": 0.91}, handle)

            assets = discover_document_assets(
                summary_dir=summary_dir,
                snomed_dir=snomed_dir,
                textract_dir=textract_dir,
                confidence_dir=confidence_dir,
            )
            self.assertIn("doc_001", assets)
            self.assertTrue(assets["doc_001"]["summary_json"]["clinician"].endswith(".json"))
            self.assertTrue(assets["doc_001"]["summary_txt"]["patient"].endswith(".txt"))
            self.assertTrue(assets["doc_001"]["snomed_json"].endswith("_snomed.json"))
            self.assertTrue(assets["doc_001"]["textract_json"].endswith("_textract.json"))
            self.assertTrue(assets["doc_001"]["confidence_json"].endswith("_confidence.json"))

    def test_load_snomed_entities_handles_both_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            entities_path = os.path.join(temp_dir, "legacy_snomed.json")
            categorized_path = os.path.join(temp_dir, "categorized_snomed.json")

            with open(entities_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "Entities": [
                            {
                                "Text": "Diabetes",
                                "Category": "MEDICAL_CONDITION",
                                "Score": 0.88,
                                "SNOMEDCTConcepts": [
                                    {
                                        "Code": "44054006",
                                        "Description": "Diabetes mellitus",
                                        "Score": 0.80,
                                    }
                                ],
                            }
                        ]
                    },
                    handle,
                )

            with open(categorized_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "categorized_entities": {
                            "Diagnosis": [
                                {
                                    "text": "Hypertension",
                                    "snomed_code": "38341003",
                                    "description": "Hypertensive disorder",
                                    "confidence": 0.93,
                                    "source": "comprehend_medical",
                                }
                            ]
                        }
                    },
                    handle,
                )

            legacy_entities = load_snomed_entities(entities_path)
            categorized_entities = load_snomed_entities(categorized_path)

            self.assertEqual(len(legacy_entities), 1)
            self.assertEqual(legacy_entities[0]["snomed_code"], "44054006")
            self.assertEqual(len(categorized_entities), 1)
            self.assertEqual(categorized_entities[0]["category"], "Diagnosis")

    def test_compute_confidence_bundle_from_stage_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            textract_path = os.path.join(temp_dir, "doc_textract.json")
            track_a_path = os.path.join(temp_dir, "doc_snomed.json")
            track_b_path = os.path.join(temp_dir, "doc_clinician_summary.json")

            with open(textract_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "Blocks": [
                            {"BlockType": "LINE", "Confidence": 90.0, "Text": "A"},
                            {"BlockType": "LINE", "Confidence": 80.0, "Text": "B"},
                        ]
                    },
                    handle,
                )

            with open(track_a_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "categorized_entities": {
                            "Diagnosis": [
                                {
                                    "text": "Hypertension",
                                    "confidence": 0.9,
                                    "source": "comprehend_medical",
                                }
                            ],
                            "Problems/Issues": [
                                {
                                    "text": "Chest pain",
                                    "confidence": 0.7,
                                    "source": "semantic_fallback",
                                }
                            ],
                        }
                    },
                    handle,
                )

            with open(track_b_path, "w", encoding="utf-8") as handle:
                json.dump({"confidence_score": 0.8}, handle)

            bundle = compute_confidence_bundle(
                {
                    "textract_json": textract_path,
                    "snomed_json": track_a_path,
                    "summary_json": {"clinician": track_b_path},
                    "confidence_json": None,
                }
            )

            self.assertAlmostEqual(bundle["component_scores"]["textract"], 0.85, places=2)
            self.assertAlmostEqual(bundle["component_scores"]["comprehend"], 0.90, places=2)
            self.assertAlmostEqual(bundle["component_scores"]["faiss"], 0.70, places=2)
            self.assertAlmostEqual(bundle["component_scores"]["llm_logprobs"], 0.80, places=2)
            self.assertAlmostEqual(bundle["unified_confidence_score"], 0.8125, places=3)
            self.assertEqual(bundle["route"], "human_review")

    def test_confidence_band_thresholds(self):
        self.assertEqual(confidence_band(0.9), "high")
        self.assertEqual(confidence_band(0.7), "medium")
        self.assertEqual(confidence_band(0.4), "low")

    def test_action_item_normalization_and_serialization(self):
        raw_actions = [
            "Schedule blood pressure follow-up",
            {
                "action_text": "Review medication adherence",
                "priority": "high",
                "assignee": "Clinician",
                "snomed_code": "38341003",
                "due_date": "2026-04-15",
            },
            {
                "text": "Educate patient on diet",
                "priority": "urgent",  # invalid -> defaults to Medium
                "assignee": "Nurse",
                "snomed_code": "27113001",
                "due_date": "bad-date",
            },
            {"action_text": "", "assignee": "", "snomed_code": ""},  # dropped on serialize
        ]

        normalized = normalize_action_items(raw_actions, default_assignee="System")
        self.assertEqual(len(normalized), 3)
        self.assertIn(normalized[0]["priority"], ACTION_PRIORITY_OPTIONS)
        self.assertEqual(normalized[1]["priority"], "High")
        self.assertEqual(normalized[2]["priority"], "Medium")

        serialized = serialize_action_items(normalized + [{"action_text": "", "assignee": "", "snomed_code": ""}])
        self.assertEqual(len(serialized), 3)
        self.assertEqual(serialized[1]["snomed_code"], "38341003")
        self.assertTrue(serialized[0]["due_date"])


if __name__ == "__main__":
    unittest.main()
