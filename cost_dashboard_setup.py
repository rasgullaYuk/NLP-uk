import json
from pathlib import Path

from hipaa_compliance import create_secure_client


def create_cost_dashboard(
    dashboard_name: str = "ClinicalNLP-CostOptimization",
    dashboard_json_path: str = "cost/cost_monitoring_dashboard.json",
):
    cloudwatch = create_secure_client("cloudwatch")
    body = Path(dashboard_json_path).read_text(encoding="utf-8")
    response = cloudwatch.put_dashboard(
        DashboardName=dashboard_name,
        DashboardBody=body,
    )
    return response


if __name__ == "__main__":
    result = create_cost_dashboard()
    print(json.dumps(result, indent=2, default=str))
