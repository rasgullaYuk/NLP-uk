import unittest

from cloudwatch_monitoring import CloudWatchMonitoringManager, infer_document_type


class FakeCloudWatch:
    def __init__(self):
        self.metric_calls = []
        self.dashboard_calls = []
        self.alarm_calls = []

    def put_metric_data(self, **kwargs):
        self.metric_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def put_dashboard(self, **kwargs):
        self.dashboard_calls.append(kwargs)
        return {"DashboardValidationMessages": []}

    def put_metric_alarm(self, **kwargs):
        self.alarm_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


class FakeLogs:
    def __init__(self):
        self.groups = set()
        self.retention = {}

    def describe_log_groups(self, logGroupNamePrefix):
        return {
            "logGroups": [
                {"logGroupName": name}
                for name in self.groups
                if name.startswith(logGroupNamePrefix)
            ]
        }

    def create_log_group(self, logGroupName):
        self.groups.add(logGroupName)

    def put_retention_policy(self, logGroupName, retentionInDays):
        self.retention[logGroupName] = retentionInDays


class FakeSQS:
    def __init__(self):
        self.depths = {"TrackA_Entity_SNOMED_Queue": 7, "TrackB_Summary_Queue": 2}

    def get_queue_url(self, QueueName):
        return {"QueueUrl": f"https://example.com/{QueueName}"}

    def get_queue_attributes(self, QueueUrl, AttributeNames):
        _ = AttributeNames
        queue_name = QueueUrl.rsplit("/", 1)[-1]
        return {
            "Attributes": {
                "ApproximateNumberOfMessages": str(self.depths.get(queue_name, 0))
            }
        }


class FakeSNS:
    def __init__(self):
        self.subscriptions = []

    def create_topic(self, Name):
        return {"TopicArn": f"arn:aws:sns:us-east-1:123456789012:{Name}"}

    def subscribe(self, **kwargs):
        self.subscriptions.append(kwargs)
        return {"SubscriptionArn": "pending confirmation"}


class TestCloudWatchMonitoring(unittest.TestCase):
    def setUp(self):
        self.cloudwatch = FakeCloudWatch()
        self.logs = FakeLogs()
        self.sqs = FakeSQS()
        self.sns = FakeSNS()
        self.manager = CloudWatchMonitoringManager(
            cloudwatch_client=self.cloudwatch,
            logs_client=self.logs,
            sqs_client=self.sqs,
            sns_client=self.sns,
        )

    def test_document_type_inference(self):
        self.assertEqual(infer_document_type("discharge_doc_001"), "discharge_summary")
        self.assertEqual(infer_document_type("prescription_page_2"), "prescription")
        self.assertEqual(infer_document_type("unknown_doc"), "unknown")

    def test_publish_extraction_metrics(self):
        self.manager.publish_extraction_result(
            document_id="discharge_doc_1",
            success=True,
            latency_seconds=2.5,
        )
        self.assertEqual(len(self.cloudwatch.metric_calls), 1)
        metric_data = self.cloudwatch.metric_calls[0]["MetricData"]
        self.assertEqual(len(metric_data), 3)
        names = {item["MetricName"] for item in metric_data}
        self.assertIn("StageLatencySeconds", names)
        self.assertIn("ExtractionErrorRate", names)

    def test_publish_snomed_metrics(self):
        self.manager.publish_snomed_mapping_result(
            document_id="doc_001",
            total_entities=10,
            mapped_entities=8,
            fallback_count=2,
            latency_seconds=1.2,
        )
        metric_data = self.cloudwatch.metric_calls[-1]["MetricData"]
        success_metric = [m for m in metric_data if m["MetricName"] == "SNOMEDMappingSuccessRate"][0]
        self.assertAlmostEqual(success_metric["Value"], 80.0, places=3)

    def test_queue_depth_metric(self):
        depth = self.manager.publish_queue_depth("TrackA_Entity_SNOMED_Queue")
        self.assertEqual(depth, 7)
        self.assertEqual(self.cloudwatch.metric_calls[-1]["MetricData"][0]["MetricName"], "QueueDepth")

    def test_dashboard_and_alarms(self):
        self.manager.create_dashboard(queue_names=["TrackA_Entity_SNOMED_Queue"])
        self.assertEqual(len(self.cloudwatch.dashboard_calls), 1)

        self.manager.configure_default_alarms(
            topic_arn="arn:aws:sns:us-east-1:123456789012:topic",
            queue_names=["TrackA_Entity_SNOMED_Queue", "TrackB_Summary_Queue"],
        )
        # 2 queue alarms + latency alarm + extraction error alarm
        self.assertEqual(len(self.cloudwatch.alarm_calls), 4)

    def test_setup_monitoring_stack(self):
        summary = self.manager.setup_monitoring_stack(
            alert_email="alerts@example.com",
            queue_names=["TrackA_Entity_SNOMED_Queue"],
            dashboard_name="test-dashboard",
        )
        self.assertEqual(summary["dashboard"], "test-dashboard")
        self.assertTrue(summary["alert_topic_arn"].startswith("arn:aws:sns:"))
        self.assertGreater(len(self.logs.retention), 0)
        self.assertEqual(len(self.sns.subscriptions), 1)


if __name__ == "__main__":
    unittest.main()
