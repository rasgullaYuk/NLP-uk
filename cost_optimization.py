import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from hipaa_compliance import create_secure_client, create_secure_resource


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class RequestDeduplicator:
    def __init__(
        self,
        table_name: str = "ClinicalDocs_RequestDedup",
        ttl_seconds: int = 60 * 60 * 24,
        fallback_file: str = "temp_pages/.request_dedup_cache.json",
    ):
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self.fallback_file = fallback_file
        self._table = None
        self._cache = {}
        try:
            dynamodb = create_secure_resource("dynamodb")
            self._table = dynamodb.Table(table_name)
            self._table.load()
        except Exception:
            self._table = None
            self._load_fallback()

    def _load_fallback(self):
        if os.path.exists(self.fallback_file):
            try:
                with open(self.fallback_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_fallback(self):
        os.makedirs(os.path.dirname(self.fallback_file) or ".", exist_ok=True)
        with open(self.fallback_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    def is_duplicate(self, request_key: str) -> bool:
        now = int(time.time())
        if self._table is not None:
            item = self._table.get_item(Key={"request_key": request_key}).get("Item")
            if item and int(item.get("expires_at", now + 1)) > now:
                return True
            self._table.put_item(
                Item={
                    "request_key": request_key,
                    "seen_at": _utc_now(),
                    "expires_at": now + self.ttl_seconds,
                }
            )
            return False

        expires_at = int(self._cache.get(request_key, 0))
        if expires_at > now:
            return True
        self._cache[request_key] = now + self.ttl_seconds
        self._save_fallback()
        return False


class SnomedMappingCache:
    def __init__(
        self,
        table_name: str = "ClinicalDocs_SnomedCache",
        ttl_seconds: int = 60 * 60 * 24 * 7,
    ):
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self._table = None
        self._mem = {}
        try:
            dynamodb = create_secure_resource("dynamodb")
            self._table = dynamodb.Table(table_name)
            self._table.load()
        except Exception:
            self._table = None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = int(time.time())
        if self._table is not None:
            item = self._table.get_item(Key={"cache_key": key}).get("Item")
            if not item:
                return None
            if int(item.get("expires_at", now + 1)) <= now:
                return None
            payload = item.get("payload_json")
            return json.loads(payload) if payload else None
        entry = self._mem.get(key)
        if not entry or entry["expires_at"] <= now:
            return None
        return entry["payload"]

    def put(self, key: str, payload: Dict[str, Any]) -> None:
        expires_at = int(time.time()) + self.ttl_seconds
        if self._table is not None:
            self._table.put_item(
                Item={
                    "cache_key": key,
                    "payload_json": json.dumps(payload),
                    "created_at": _utc_now(),
                    "expires_at": expires_at,
                }
            )
            return
        self._mem[key] = {"payload": payload, "expires_at": expires_at}


@dataclass
class BatchWindow:
    mode: str
    max_batch_size: int
    wait_seconds: int


def resolve_batch_window() -> BatchWindow:
    mode = os.getenv("PIPELINE_PROCESSING_MODE", "realtime").lower()
    max_batch_size = int(os.getenv("TEXTRACT_BATCH_SIZE", "10"))
    wait_seconds = int(os.getenv("TEXTRACT_BATCH_WAIT_SECONDS", "30"))
    return BatchWindow(mode=mode, max_batch_size=max_batch_size, wait_seconds=wait_seconds)


def split_into_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), max(1, batch_size)):
        yield items[i : i + max(1, batch_size)]


def tag_resource(resource_arn: str, tags: Dict[str, str]) -> None:
    tagging = create_secure_client("resourcegroupstaggingapi")
    tagging.tag_resources(
        ResourceARNList=[resource_arn],
        Tags=tags,
    )


def build_cost_monitoring_dashboard_payload(region: str = "us-east-1") -> Dict[str, Any]:
    return {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Textract API Calls",
                    "region": region,
                    "metrics": [["AWS/Usage", "CallCount", "Type", "API", "Resource", "Textract"]],
                    "stat": "Sum",
                    "period": 300,
                },
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "title": "Bedrock Invocation Count",
                    "region": region,
                    "metrics": [["AWS/Bedrock", "InvocationCount", "ModelId", "ALL"]],
                    "stat": "Sum",
                    "period": 300,
                },
            },
        ]
    }


def estimate_cost_savings(baseline_cost: float, optimized_cost: float) -> Dict[str, float]:
    if baseline_cost <= 0:
        return {"baseline_cost": baseline_cost, "optimized_cost": optimized_cost, "savings_percent": 0.0}
    savings_percent = ((baseline_cost - optimized_cost) / baseline_cost) * 100.0
    return {
        "baseline_cost": round(baseline_cost, 2),
        "optimized_cost": round(optimized_cost, 2),
        "savings_percent": round(savings_percent, 2),
    }
