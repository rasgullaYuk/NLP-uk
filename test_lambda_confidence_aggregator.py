import json
import unittest
from unittest.mock import patch

import lambda_confidence_aggregator as agg


class TestLambdaConfidenceAggregator(unittest.TestCase):
    def test_calculate_weighted_score_default_weights(self):
        components = {
            "textract": 0.9,
            "comprehend": 0.8,
            "faiss": 0.7,
            "llm_logprobs": 0.6,
        }
        score, latency_ms = agg.calculate_weighted_score(components, agg.DEFAULT_WEIGHTS)
        self.assertAlmostEqual(score, 0.75, places=6)
        self.assertLess(latency_ms, 100.0)

    def test_resolve_weights_normalizes_custom_values(self):
        weights = agg.resolve_weights({"weights": {"textract": 0.5, "comprehend": 0.2, "faiss": 0.2, "llm_logprobs": 0.1}})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertAlmostEqual(weights["textract"], 0.5, places=6)

    def test_collect_component_scores_accepts_percentage_inputs(self):
        scores = agg.collect_component_scores(
            {
                "textract_confidence": 93.0,
                "comprehend_confidence": 82.0,
                "faiss_similarity": 70.0,
                "llm_logprobs_confidence": 88.0,
            }
        )
        self.assertAlmostEqual(scores["textract"], 0.93, places=6)
        self.assertAlmostEqual(scores["llm_logprobs"], 0.88, places=6)

    @patch("lambda_confidence_aggregator.log_routing_audit")
    @patch("lambda_confidence_aggregator.route_document")
    def test_lambda_handler_routes_high_confidence(self, mock_route, mock_audit):
        mock_route.return_value = {
            "route": "bypass_database",
            "queue_name": "Confidence_High_Bypass_Queue",
            "queue_url": "https://example/high",
            "routing_payload": {
                "final_confidence_score": 0.9,
                "threshold": 0.85,
                "component_scores": {"textract": 0.9, "comprehend": 0.9, "faiss": 0.9, "llm_logprobs": 0.9},
                "weights": agg.DEFAULT_WEIGHTS,
                "calculation_latency_ms": 0.05,
                "calculation_latency_sla_met": True,
                "route": "bypass_database",
            },
        }
        response = agg.lambda_handler(
            {
                "document_id": "doc_high",
                "textract_confidence": 0.9,
                "comprehend_confidence": 0.9,
                "faiss_similarity": 0.9,
                "llm_logprobs_confidence": 0.9,
            },
            None,
        )
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["route"], "bypass_database")
        mock_route.assert_called_once()
        mock_audit.assert_called_once()

    @patch("lambda_confidence_aggregator.log_routing_audit")
    @patch("lambda_confidence_aggregator.route_document")
    def test_lambda_handler_routes_low_confidence(self, mock_route, mock_audit):
        mock_route.return_value = {
            "route": "human_review",
            "queue_name": "Confidence_Low_Review_Queue",
            "queue_url": "https://example/low",
            "routing_payload": {
                "final_confidence_score": 0.61,
                "threshold": 0.85,
                "component_scores": {"textract": 0.6, "comprehend": 0.6, "faiss": 0.6, "llm_logprobs": 0.65},
                "weights": agg.DEFAULT_WEIGHTS,
                "calculation_latency_ms": 0.03,
                "calculation_latency_sla_met": True,
                "route": "human_review",
            },
        }
        response = agg.lambda_handler(
            {
                "document_id": "doc_low",
                "textract_confidence": 0.6,
                "comprehend_confidence": 0.6,
                "faiss_similarity": 0.6,
                "llm_logprobs_confidence": 0.65,
            },
            None,
        )
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["route"], "human_review")
        mock_route.assert_called_once()
        mock_audit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
