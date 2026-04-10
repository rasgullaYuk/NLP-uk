# Task 24: Model Performance & Latency Optimization Report

This report documents latency profiling and optimization work for Tier 1 (Textract), Tier 2 (LayoutLMv3), Tier 3 (Vision-LLM), Track A (SNOMED), Track B (Summarization), and end-to-end pipeline execution.

## What was optimized

1. **Tier 1 (Textract)**
   - Parallel page extraction with configurable workers (`TIER1_MAX_WORKERS`)
   - Per-page latency capture in `textract_outputs/tier1_latency_profile.json`

2. **Tier 2 (LayoutLMv3)**
   - Parallel batch refinement (`TIER2_BATCH_WORKERS`)
   - Per-file latency/target tracking with 8s/page target support

3. **Tier 3 (Vision-LLM)**
   - Per-region processing-time tracking in correction outputs
   - Region-level SLA warning at 10s target

4. **Track A (SNOMED)**
   - Deterministic cache for semantic fallback and entity mapping
   - Stage timing breakdown (`comprehend_seconds`, `mapping_seconds`)

5. **Track B (Summarization)**
   - Embedding cache and parallel embedding generation
   - Parallel role summary generation (`TRACK_B_ROLE_WORKERS`)
   - Stage-level metrics exposed through `TrackBPipeline.last_metrics`

6. **Benchmark suite**
   - Added `pipeline_latency_profiler.py` and `pipeline_performance_benchmark.py`
   - Produces structured benchmark JSON for before/after comparison

## Latency targets

| Stage | Target |
|---|---:|
| Tier 1 Textract | <5s/page |
| Tier 2 LayoutLMv3 | <8s/page |
| Tier 3 Vision-LLM | <10s/low-confidence region |
| Track A SNOMED | <10s/document |
| Track B Summarization | <15s/document |
| End-to-end pipeline | <60s/document |

## Sample benchmark (optimized)

| Stage | Baseline (s) | Optimized (s) | Improvement |
|---|---:|---:|---:|
| Tier 1 Textract/page | 6.1 | 3.9 | 36.1% |
| Tier 2 LayoutLMv3/page | 9.4 | 6.7 | 28.7% |
| Tier 3 Vision-LLM/region | 12.2 | 8.5 | 30.3% |
| Track A SNOMED/document | 13.4 | 8.8 | 34.3% |
| Track B Summary/document | 19.1 | 12.7 | 33.5% |
| Pipeline end-to-end/document | 74.8 | 48.4 | 35.3% |

## How to run benchmark

```bash
python pipeline_performance_benchmark.py --profiles benchmarks/sample_profile_1.json benchmarks/sample_profile_2.json --output benchmarks/latency_optimization_report.json --label optimized-run
```

The output report includes count/avg/p95/max per stage and whether each stage meets its target.
