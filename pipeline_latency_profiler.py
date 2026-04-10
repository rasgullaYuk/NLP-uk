import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StageBudget:
    stage: str
    target_seconds: float


DEFAULT_STAGE_BUDGETS: List[StageBudget] = [
    StageBudget("tier1_textract_per_page", 5.0),
    StageBudget("tier2_layoutlmv3_per_page", 8.0),
    StageBudget("tier3_vision_llm_per_region", 10.0),
    StageBudget("track_a_snomed_per_document", 10.0),
    StageBudget("track_b_summary_per_document", 15.0),
    StageBudget("pipeline_end_to_end_per_document", 60.0),
]


class LatencyProfiler:
    def __init__(self, stage_budgets: Optional[List[StageBudget]] = None):
        budgets = stage_budgets or DEFAULT_STAGE_BUDGETS
        self.stage_budgets = {b.stage: b.target_seconds for b in budgets}
        self._start_times: Dict[str, float] = {}
        self.measurements: Dict[str, List[float]] = {}

    def start(self, stage: str) -> None:
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        if stage not in self._start_times:
            raise KeyError(f"Stage '{stage}' was not started")
        duration = time.perf_counter() - self._start_times.pop(stage)
        self.record(stage, duration)
        return duration

    def record(self, stage: str, duration_seconds: float) -> None:
        self.measurements.setdefault(stage, []).append(float(duration_seconds))

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for stage, values in self.measurements.items():
            if not values:
                continue
            avg = sum(values) / len(values)
            p95 = sorted(values)[int(max(0, (len(values) - 1) * 0.95))]
            target = self.stage_budgets.get(stage)
            result[stage] = {
                "count": float(len(values)),
                "avg_seconds": round(avg, 4),
                "p95_seconds": round(p95, 4),
                "max_seconds": round(max(values), 4),
                "target_seconds": target if target is not None else -1.0,
                "target_met": bool(target is None or max(values) <= target),
            }
        return result

    def to_report_payload(self, run_label: str) -> Dict:
        return {
            "run_label": run_label,
            "generated_at_epoch": int(time.time()),
            "summary": self.summary(),
        }

    def write_report(self, output_path: str, run_label: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        payload = self.to_report_payload(run_label=run_label)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path
