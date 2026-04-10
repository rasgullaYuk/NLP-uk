"""
Provisioning helper for API Gateway + usage plan + API key throttling.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any

from hipaa_compliance import create_secure_client


def _load_openapi(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def setup_api_gateway(
    api_name: str,
    stage_name: str,
    openapi_path: str,
    rate_limit_per_sec: float = 20.0,
    burst_limit: int = 40,
    quota_limit_per_day: int = 10000,
) -> Dict[str, Any]:
    client = create_secure_client("apigateway")
    spec = _load_openapi(openapi_path)
    response = client.import_rest_api(parameters={"endpointConfigurationTypes": "REGIONAL"}, body=json.dumps(spec))
    rest_api_id = response["id"]

    client.create_deployment(restApiId=rest_api_id, stageName=stage_name, description="Initial deployment")
    usage_plan = client.create_usage_plan(
        name=f"{api_name}-{stage_name}-usage-plan",
        description="Rate limiting and quota for clinical API",
        throttle={"rateLimit": rate_limit_per_sec, "burstLimit": burst_limit},
        quota={"limit": quota_limit_per_day, "period": "DAY"},
        apiStages=[{"apiId": rest_api_id, "stage": stage_name}],
    )
    api_key = client.create_api_key(name=f"{api_name}-{stage_name}-key", enabled=True, generateDistinctId=True)
    client.create_usage_plan_key(
        usagePlanId=usage_plan["id"],
        keyId=api_key["id"],
        keyType="API_KEY",
    )
    return {
        "rest_api_id": rest_api_id,
        "stage_name": stage_name,
        "invoke_url": f"https://{rest_api_id}.execute-api.us-east-1.amazonaws.com/{stage_name}",
        "usage_plan_id": usage_plan["id"],
        "api_key_id": api_key["id"],
        "api_key_value": api_key["value"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision API Gateway REST API for NLP-uk")
    parser.add_argument("--api-name", default="nlp-uk-clinical-api")
    parser.add_argument("--stage", default="prod")
    parser.add_argument("--openapi", default="openapi\\documents_api.json")
    parser.add_argument("--rate-limit", type=float, default=20.0)
    parser.add_argument("--burst-limit", type=int, default=40)
    parser.add_argument("--quota-day", type=int, default=10000)
    args = parser.parse_args()

    result = setup_api_gateway(
        api_name=args.api_name,
        stage_name=args.stage,
        openapi_path=args.openapi,
        rate_limit_per_sec=args.rate_limit,
        burst_limit=args.burst_limit,
        quota_limit_per_day=args.quota_day,
    )
    print(json.dumps(result, indent=2))
