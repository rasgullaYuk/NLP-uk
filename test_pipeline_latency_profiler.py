import json
import os
import tempfile
import unittest

from pipeline_latency_profiler import LatencyProfiler


class TestLatencyProfiler(unittest.TestCase):
    def test_record_and_summary(self):
        profiler = LatencyProfiler()
        profiler.record("tier1_textract_per_page", 3.1)
        profiler.record("tier1_textract_per_page", 4.2)
        summary = profiler.summary()
        self.assertIn("tier1_textract_per_page", summary)
        self.assertTrue(summary["tier1_textract_per_page"]["target_met"])

    def test_start_stop_and_write_report(self):
        profiler = LatencyProfiler()
        profiler.start("pipeline_end_to_end_per_document")
        profiler.stop("pipeline_end_to_end_per_document")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "latency.json")
            profiler.write_report(out, run_label="unit-test")
            with open(out, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["run_label"], "unit-test")


if __name__ == "__main__":
    unittest.main()
