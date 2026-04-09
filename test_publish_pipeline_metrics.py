import json
import os
import tempfile
import unittest

from publish_pipeline_metrics import (
    publish_textract_metrics,
    publish_track_a_metrics,
    publish_track_b_metrics,
)


class FakeMonitor:
    def __init__(self):
        self.extraction_calls = []
        self.metric_calls = []
        self.track_a_calls = []
        self.track_b_calls = []

    def publish_extraction_result(self, **kwargs):
        self.extraction_calls.append(kwargs)

    def put_metric(self, **kwargs):
        self.metric_calls.append(kwargs)

    def publish_snomed_mapping_result(self, **kwargs):
        self.track_a_calls.append(kwargs)

    def publish_llm_latency(self, **kwargs):
        self.track_b_calls.append(kwargs)


class TestPublishPipelineMetrics(unittest.TestCase):
    def test_publishers_process_output_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            textract_dir = os.path.join(temp_dir, "textract")
            track_a_dir = os.path.join(temp_dir, "track_a")
            track_b_dir = os.path.join(temp_dir, "track_b")
            os.makedirs(textract_dir, exist_ok=True)
            os.makedirs(track_a_dir, exist_ok=True)
            os.makedirs(track_b_dir, exist_ok=True)

            with open(
                os.path.join(textract_dir, "doc_001_textract.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "Blocks": [
                            {"BlockType": "LINE", "Confidence": 95.0},
                            {"BlockType": "LINE", "Confidence": 85.0},
                        ]
                    },
                    handle,
                )

            with open(
                os.path.join(track_a_dir, "doc_001_snomed.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "categorized_entities": {
                            "Diagnosis": [
                                {"snomed_code": "38341003", "source": "comprehend_medical"},
                                {"snomed_code": "NOT_MAPPED", "source": "semantic_fallback"},
                            ]
                        },
                        "processing_time_seconds": 1.3,
                    },
                    handle,
                )

            with open(
                os.path.join(track_b_dir, "doc_001_clinician_summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "role": "clinician",
                        "generation_time_ms": 920.0,
                        "confidence_score": 0.84,
                    },
                    handle,
                )

            monitor = FakeMonitor()
            textract_count = publish_textract_metrics(monitor, textract_dir=textract_dir)
            track_a_count = publish_track_a_metrics(monitor, track_a_dir=track_a_dir)
            track_b_count = publish_track_b_metrics(monitor, track_b_dir=track_b_dir)

            self.assertEqual(textract_count, 1)
            self.assertEqual(track_a_count, 1)
            self.assertEqual(track_b_count, 1)
            self.assertEqual(len(monitor.extraction_calls), 1)
            self.assertGreaterEqual(len(monitor.metric_calls), 2)  # confidence + error proxy
            self.assertEqual(len(monitor.track_a_calls), 1)
            self.assertEqual(len(monitor.track_b_calls), 1)


if __name__ == "__main__":
    unittest.main()
