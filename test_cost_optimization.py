import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cost_optimization as co


class TestCostOptimization(unittest.TestCase):
    def test_content_hash_is_stable(self):
        payload = b"abc123"
        self.assertEqual(co.content_hash(payload), co.content_hash(payload))

    def test_request_deduplicator_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            dedup = co.RequestDeduplicator(
                table_name="missing",
                ttl_seconds=60,
                fallback_file=f"{tmp}/dedup.json",
            )
            key = "k1"
            self.assertFalse(dedup.is_duplicate(key))
            self.assertTrue(dedup.is_duplicate(key))

    def test_snomed_cache_memory_backend(self):
        cache = co.SnomedMappingCache(table_name="missing")
        cache.put("k1", {"Entities": [1]})
        self.assertEqual(cache.get("k1"), {"Entities": [1]})

    def test_split_into_batches(self):
        batches = list(co.split_into_batches([1, 2, 3, 4, 5], 2))
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

    @patch("cost_optimization.create_secure_client")
    def test_tag_resource(self, mock_client):
        fake = MagicMock()
        mock_client.return_value = fake
        co.tag_resource("arn:aws:s3:::example", {"env": "prod"})
        fake.tag_resources.assert_called_once()

    def test_estimate_cost_savings(self):
        result = co.estimate_cost_savings(100.0, 68.0)
        self.assertGreaterEqual(result["savings_percent"], 30.0)


if __name__ == "__main__":
    unittest.main()
