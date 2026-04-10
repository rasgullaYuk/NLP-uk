import argparse
import json
import os
import statistics
from typing import Dict, List

from pipeline_latency_profiler import LatencyProfiler


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def benchmark_from_profiles(profile_paths: List[str], run_label: str = "optimized-run") -> Dict:
    profiler = LatencyProfiler()
    pipeline_totals: List[float] = []

    for path in profile_paths:
        with open(path, "r", encoding="utf-8") as handle:
            profile = json.load(handle)

        tier1_pages = profile.get("tier1_pages", [])
        for page in tier1_pages:
            profiler.record("tier1_textract_per_page", _safe_float(page.get("latency_seconds")))

        tier2_pages = profile.get("tier2_pages", [])
        for page in tier2_pages:
            profiler.record("tier2_layoutlmv3_per_page", _safe_float(page.get("latency_seconds")))

        tier3_regions = profile.get("tier3_regions", [])
        for region in tier3_regions:
            profiler.record("tier3_vision_llm_per_region", _safe_float(region.get("latency_seconds")))

        profiler.record("track_a_snomed_per_document", _safe_float(profile.get("track_a_seconds")))
        profiler.record("track_b_summary_per_document", _safe_float(profile.get("track_b_seconds")))
        total = _safe_float(profile.get("pipeline_total_seconds"))
        profiler.record("pipeline_end_to_end_per_document", total)
        pipeline_totals.append(total)

    payload = profiler.to_report_payload(run_label=run_label)
    payload["documents_profiled"] = len(profile_paths)
    payload["pipeline_total_avg_seconds"] = round(statistics.mean(pipeline_totals), 4) if pipeline_totals else 0.0
    payload["pipeline_total_max_seconds"] = round(max(pipeline_totals), 4) if pipeline_totals else 0.0
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run benchmark from stage profile JSON files.")
    parser.add_argument("--profiles", nargs="+", required=True, help="Input profile JSON paths")
    parser.add_argument("--output", default="benchmarks/latency_optimization_report.json")
    parser.add_argument("--label", default="optimized-run")
    args = parser.parse_args()

    report = benchmark_from_profiles(args.profiles, run_label=args.label)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote benchmark report: {args.output}")


if __name__ == "__main__":
    main()
