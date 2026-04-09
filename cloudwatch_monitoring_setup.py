"""
Provision CloudWatch monitoring resources for NLP-uk pipeline.

Creates:
- Log groups and retention policies.
- SNS topic + email subscription for critical alarms.
- CloudWatch dashboard for stage health, throughput, errors, queue depth, and cost.
- Alarms for queue depth, stage latency, and extraction error rate.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from cloudwatch_monitoring import (
    DEFAULT_DASHBOARD_NAME,
    DEFAULT_QUEUE_NAMES,
    CloudWatchMonitoringManager,
)


def _parse_queue_names(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_QUEUE_NAMES)
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup CloudWatch monitoring for NLP-uk")
    parser.add_argument("--alert-email", default=None, help="Email endpoint for SNS alarm notifications")
    parser.add_argument(
        "--queue-names",
        default="",
        help="Comma-separated queue names. Defaults to pipeline queues.",
    )
    parser.add_argument(
        "--dashboard-name",
        default=DEFAULT_DASHBOARD_NAME,
        help="CloudWatch dashboard name",
    )
    args = parser.parse_args()

    monitor = CloudWatchMonitoringManager()
    summary = monitor.setup_monitoring_stack(
        alert_email=args.alert_email,
        queue_names=_parse_queue_names(args.queue_names),
        dashboard_name=args.dashboard_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
