"""
CloudWatch monitoring utilities for NLP-uk pipeline services.

Provides:
- Custom metric publishing helpers for pipeline stages.
- Log group creation/retention setup.
- Dashboard provisioning.
- Alarm provisioning with SNS notifications.
- Queue depth collection helpers.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from hipaa_compliance import create_secure_client

DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_NAMESPACE = os.environ.get("CLOUDWATCH_NAMESPACE", "NLP-UK/Pipeline")
DEFAULT_DASHBOARD_NAME = os.environ.get(
    "CLOUDWATCH_DASHBOARD_NAME", "NLP-UK-Pipeline-Health"
)

DEFAULT_LOG_GROUPS = [
    "/nlp-uk/tier1-textract",
    "/nlp-uk/track-a-snomed",
    "/nlp-uk/track-b-summarization",
    "/nlp-uk/lambda-confidence-aggregator",
    "/nlp-uk/review-interface",
]

DEFAULT_QUEUE_NAMES = [
    "TrackA_Entity_SNOMED_Queue",
    "TrackB_Summary_Queue",
    "Tier2_LayoutLM_Queue",
    "Tier3_Escalation_Queue",
    "Confidence_High_Bypass_Queue",
    "Confidence_Low_Review_Queue",
]


def infer_document_type(document_id: str) -> str:
    value = (document_id or "").lower()
    if "discharge" in value:
        return "discharge_summary"
    if "prescription" in value:
        return "prescription"
    if "lab" in value:
        return "lab_report"
    if "radiology" in value:
        return "radiology_report"
    if "note" in value:
        return "clinical_note"
    return "unknown"


class CloudWatchMonitoringManager:
    def __init__(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        region_name: str = DEFAULT_REGION,
        cloudwatch_client: Any = None,
        logs_client: Any = None,
        sqs_client: Any = None,
        sns_client: Any = None,
    ):
        self.namespace = namespace
        self.region_name = region_name
        self.cloudwatch = cloudwatch_client or create_secure_client(
            "cloudwatch", region_name=region_name
        )
        self.logs = logs_client or create_secure_client("logs", region_name=region_name)
        self.sqs = sqs_client or create_secure_client("sqs", region_name=region_name)
        self.sns = sns_client or create_secure_client("sns", region_name=region_name)

    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "Count",
        dimensions: Optional[List[Dict[str, str]]] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        metric_data = {
            "MetricName": metric_name,
            "Timestamp": datetime.utcnow(),
            "Value": float(value),
            "Unit": unit,
        }
        if dimensions:
            metric_data["Dimensions"] = dimensions
        self.cloudwatch.put_metric_data(
            Namespace=namespace or self.namespace,
            MetricData=[metric_data],
        )
        return metric_data

    def put_metrics_batch(
        self, metric_data: List[Dict[str, Any]], namespace: Optional[str] = None
    ) -> None:
        if not metric_data:
            return
        self.cloudwatch.put_metric_data(
            Namespace=namespace or self.namespace,
            MetricData=metric_data,
        )

    def publish_extraction_result(
        self,
        document_id: str,
        success: bool,
        latency_seconds: float,
        document_type: Optional[str] = None,
    ) -> None:
        doc_type = document_type or infer_document_type(document_id)
        dimensions = [
            {"Name": "Stage", "Value": "Tier1Textract"},
            {"Name": "DocumentType", "Value": doc_type},
        ]
        metrics = [
            {
                "MetricName": "StageLatencySeconds",
                "Timestamp": datetime.utcnow(),
                "Value": float(latency_seconds),
                "Unit": "Seconds",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "ExtractionSuccessCount",
                "Timestamp": datetime.utcnow(),
                "Value": 1.0 if success else 0.0,
                "Unit": "Count",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "ExtractionErrorRate",
                "Timestamp": datetime.utcnow(),
                "Value": 0.0 if success else 100.0,
                "Unit": "Percent",
                "Dimensions": dimensions,
            },
        ]
        self.put_metrics_batch(metrics)

    def publish_snomed_mapping_result(
        self,
        document_id: str,
        total_entities: int,
        mapped_entities: int,
        fallback_count: int,
        latency_seconds: float,
        document_type: Optional[str] = None,
    ) -> None:
        doc_type = document_type or infer_document_type(document_id)
        dimensions = [
            {"Name": "Stage", "Value": "TrackASNOMED"},
            {"Name": "DocumentType", "Value": doc_type},
        ]
        success_rate = (
            (float(mapped_entities) / float(total_entities) * 100.0)
            if total_entities > 0
            else 0.0
        )
        metrics = [
            {
                "MetricName": "StageLatencySeconds",
                "Timestamp": datetime.utcnow(),
                "Value": float(latency_seconds),
                "Unit": "Seconds",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "SNOMEDMappingSuccessRate",
                "Timestamp": datetime.utcnow(),
                "Value": success_rate,
                "Unit": "Percent",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "SNOMEDFallbackCount",
                "Timestamp": datetime.utcnow(),
                "Value": float(fallback_count),
                "Unit": "Count",
                "Dimensions": dimensions,
            },
        ]
        self.put_metrics_batch(metrics)

    def publish_llm_latency(
        self,
        document_id: str,
        role: str,
        latency_ms: float,
        confidence_score: Optional[float] = None,
        document_type: Optional[str] = None,
    ) -> None:
        doc_type = document_type or infer_document_type(document_id)
        dimensions = [
            {"Name": "Stage", "Value": "TrackBSummarization"},
            {"Name": "Role", "Value": role},
            {"Name": "DocumentType", "Value": doc_type},
        ]
        metrics = [
            {
                "MetricName": "LLMInferenceLatencyMs",
                "Timestamp": datetime.utcnow(),
                "Value": float(latency_ms),
                "Unit": "Milliseconds",
                "Dimensions": dimensions,
            }
        ]
        if confidence_score is not None:
            metrics.append(
                {
                    "MetricName": "LLMConfidenceScore",
                    "Timestamp": datetime.utcnow(),
                    "Value": float(confidence_score),
                    "Unit": "None",
                    "Dimensions": dimensions,
                }
            )
        self.put_metrics_batch(metrics)

    def publish_confidence_routing(
        self,
        document_id: str,
        final_confidence: float,
        route: str,
        latency_ms: float,
        document_type: Optional[str] = None,
    ) -> None:
        doc_type = document_type or infer_document_type(document_id)
        dimensions = [
            {"Name": "Stage", "Value": "ConfidenceAggregation"},
            {"Name": "Route", "Value": route},
            {"Name": "DocumentType", "Value": doc_type},
        ]
        metrics = [
            {
                "MetricName": "UnifiedConfidenceScore",
                "Timestamp": datetime.utcnow(),
                "Value": float(final_confidence),
                "Unit": "None",
                "Dimensions": dimensions,
            },
            {
                "MetricName": "AggregationLatencyMs",
                "Timestamp": datetime.utcnow(),
                "Value": float(latency_ms),
                "Unit": "Milliseconds",
                "Dimensions": dimensions,
            },
        ]
        self.put_metrics_batch(metrics)

    def get_queue_depth(self, queue_name: str) -> int:
        queue_url = self.sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]
        attrs = self.sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=["ApproximateNumberOfMessages"],
        )
        return int(attrs.get("Attributes", {}).get("ApproximateNumberOfMessages", "0"))

    def publish_queue_depth(self, queue_name: str) -> int:
        depth = self.get_queue_depth(queue_name)
        self.put_metric(
            "QueueDepth",
            depth,
            unit="Count",
            dimensions=[{"Name": "QueueName", "Value": queue_name}],
        )
        return depth

    def ensure_log_group(self, log_group_name: str, retention_days: int = 30) -> None:
        existing = self.logs.describe_log_groups(logGroupNamePrefix=log_group_name).get(
            "logGroups", []
        )
        exists = any(group.get("logGroupName") == log_group_name for group in existing)
        if not exists:
            self.logs.create_log_group(logGroupName=log_group_name)
        self.logs.put_retention_policy(
            logGroupName=log_group_name,
            retentionInDays=retention_days,
        )

    def ensure_log_groups(self, log_groups: List[str], retention_days: int = 30) -> None:
        for name in log_groups:
            self.ensure_log_group(name, retention_days=retention_days)

    def ensure_alert_topic(self, topic_name: str, email: Optional[str] = None) -> str:
        topic_arn = self.sns.create_topic(Name=topic_name)["TopicArn"]
        if email:
            self.sns.subscribe(
                TopicArn=topic_arn,
                Protocol="email",
                Endpoint=email,
                ReturnSubscriptionArn=True,
            )
        return topic_arn

    def create_dashboard(
        self,
        dashboard_name: str = DEFAULT_DASHBOARD_NAME,
        queue_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        queue_names = queue_names or DEFAULT_QUEUE_NAMES
        queue_metrics = [
            [self.namespace, "QueueDepth", "QueueName", queue_names[0], {"stat": "Average"}]
        ]
        for queue in queue_names[1:]:
            queue_metrics.append([".", "QueueDepth", "QueueName", queue, {"stat": "Average"}])

        body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Pipeline Stage Duration (Seconds)",
                        "view": "timeSeries",
                        "region": self.region_name,
                        "metrics": [
                            [
                                self.namespace,
                                "StageLatencySeconds",
                                "Stage",
                                "Tier1Textract",
                                {"stat": "Average"},
                            ],
                            [".", "StageLatencySeconds", "Stage", "TrackASNOMED", {"stat": "Average"}],
                        ],
                    },
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Error Rates and Mapping Success",
                        "view": "timeSeries",
                        "region": self.region_name,
                        "metrics": [
                            [
                                self.namespace,
                                "ExtractionErrorRate",
                                "Stage",
                                "Tier1Textract",
                                {"stat": "Average"},
                            ],
                            [
                                ".",
                                "SNOMEDMappingSuccessRate",
                                "Stage",
                                "TrackASNOMED",
                                {"stat": "Average"},
                            ],
                        ],
                    },
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "LLM and Confidence Metrics",
                        "view": "timeSeries",
                        "region": self.region_name,
                        "metrics": [
                            [self.namespace, "LLMInferenceLatencyMs", {"stat": "Average"}],
                            [".", "UnifiedConfidenceScore", {"stat": "Average"}],
                        ],
                    },
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Queue Depth",
                        "view": "timeSeries",
                        "region": self.region_name,
                        "metrics": queue_metrics,
                    },
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 12,
                    "width": 24,
                    "height": 6,
                    "properties": {
                        "title": "Estimated AWS Charges (USD)",
                        "view": "timeSeries",
                        "region": "us-east-1",
                        "metrics": [
                            [
                                "AWS/Billing",
                                "EstimatedCharges",
                                "Currency",
                                "USD",
                                {"stat": "Maximum"},
                            ]
                        ],
                    },
                },
            ]
        }
        response = self.cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(body),
        )
        return {"dashboard_name": dashboard_name, "response": response}

    def put_alarm(
        self,
        alarm_name: str,
        metric_name: str,
        threshold: float,
        comparison_operator: str,
        evaluation_periods: int,
        period_seconds: int,
        statistic: str = "Average",
        unit: str = "Count",
        dimensions: Optional[List[Dict[str, str]]] = None,
        namespace: Optional[str] = None,
        alarm_actions: Optional[List[str]] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {
            "AlarmName": alarm_name,
            "MetricName": metric_name,
            "Namespace": namespace or self.namespace,
            "ComparisonOperator": comparison_operator,
            "Threshold": float(threshold),
            "EvaluationPeriods": int(evaluation_periods),
            "Period": int(period_seconds),
            "Statistic": statistic,
            "Unit": unit,
            "TreatMissingData": "notBreaching",
        }
        if dimensions:
            kwargs["Dimensions"] = dimensions
        if alarm_actions:
            kwargs["AlarmActions"] = alarm_actions
        self.cloudwatch.put_metric_alarm(**kwargs)

    def configure_default_alarms(
        self, topic_arn: Optional[str] = None, queue_names: Optional[List[str]] = None
    ) -> None:
        queue_names = queue_names or DEFAULT_QUEUE_NAMES
        for queue_name in queue_names:
            self.put_alarm(
                alarm_name=f"NLPUK-QueueDepth-{queue_name}-gt-100",
                metric_name="QueueDepth",
                threshold=100.0,
                comparison_operator="GreaterThanThreshold",
                evaluation_periods=1,
                period_seconds=60,
                statistic="Maximum",
                unit="Count",
                dimensions=[{"Name": "QueueName", "Value": queue_name}],
                alarm_actions=[topic_arn] if topic_arn else None,
            )

        self.put_alarm(
            alarm_name="NLPUK-StageLatency-gt-30s",
            metric_name="StageLatencySeconds",
            threshold=30.0,
            comparison_operator="GreaterThanThreshold",
            evaluation_periods=2,
            period_seconds=60,
            statistic="Average",
            unit="Seconds",
            alarm_actions=[topic_arn] if topic_arn else None,
        )
        self.put_alarm(
            alarm_name="NLPUK-ExtractionErrorRate-gt-5pct",
            metric_name="ExtractionErrorRate",
            threshold=5.0,
            comparison_operator="GreaterThanThreshold",
            evaluation_periods=2,
            period_seconds=300,
            statistic="Average",
            unit="Percent",
            dimensions=[{"Name": "Stage", "Value": "Tier1Textract"}],
            alarm_actions=[topic_arn] if topic_arn else None,
        )

    def publish_queue_depths(self, queue_names: Optional[List[str]] = None) -> Dict[str, int]:
        queue_names = queue_names or DEFAULT_QUEUE_NAMES
        result: Dict[str, int] = {}
        for queue_name in queue_names:
            result[queue_name] = self.publish_queue_depth(queue_name)
        return result

    @staticmethod
    def cost_optimization_recommendations() -> List[str]:
        return [
            "Enable log retention to avoid indefinite CloudWatch Logs storage growth.",
            "Use metric dimensions selectively to limit high-cardinality custom metrics costs.",
            "Use anomaly alarms only on critical metrics to reduce alarm noise and spend.",
            "Review Bedrock token usage and prompt size to reduce LLM invocation cost.",
            "Archive old outputs to S3 lifecycle tiers for lower storage cost.",
        ]

    def setup_monitoring_stack(
        self,
        alert_email: Optional[str] = None,
        queue_names: Optional[List[str]] = None,
        dashboard_name: str = DEFAULT_DASHBOARD_NAME,
    ) -> Dict[str, Any]:
        self.ensure_log_groups(DEFAULT_LOG_GROUPS, retention_days=30)
        topic_arn = self.ensure_alert_topic("NLPUK_Critical_Alerts", email=alert_email)
        self.configure_default_alarms(topic_arn=topic_arn, queue_names=queue_names)
        dashboard = self.create_dashboard(dashboard_name=dashboard_name, queue_names=queue_names)
        return {
            "namespace": self.namespace,
            "dashboard": dashboard["dashboard_name"],
            "alert_topic_arn": topic_arn,
            "log_groups": DEFAULT_LOG_GROUPS,
            "recommendations": self.cost_optimization_recommendations(),
        }
