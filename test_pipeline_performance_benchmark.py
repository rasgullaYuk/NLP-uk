import os
import tempfile
import unittest

from pipeline_performance_benchmark import benchmark_from_profiles


class TestPipelinePerformanceBenchmark(unittest.TestCase):
    def test_benchmark_from_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "p1.json")
            p2 = os.path.join(tmp, "p2.json")
            with open(p1, "w", encoding="utf-8") as f:
                f.write(
                    '{"tier1_pages":[{"latency_seconds":4.0}],"tier2_pages":[{"latency_seconds":6.0}],'
                    '"tier3_regions":[{"latency_seconds":8.0}],"track_a_seconds":9.0,'
                    '"track_b_seconds":12.0,"pipeline_total_seconds":49.0}'
                )
            with open(p2, "w", encoding="utf-8") as f:
                f.write(
                    '{"tier1_pages":[{"latency_seconds":4.1}],"tier2_pages":[{"latency_seconds":6.1}],'
                    '"tier3_regions":[{"latency_seconds":8.1}],"track_a_seconds":9.1,'
                    '"track_b_seconds":12.1,"pipeline_total_seconds":50.0}'
                )
            report = benchmark_from_profiles([p1, p2], run_label="test-run")
            self.assertEqual(report["run_label"], "test-run")
            self.assertEqual(report["documents_profiled"], 2)
            self.assertIn("pipeline_end_to_end_per_document", report["summary"])


if __name__ == "__main__":
    unittest.main()
