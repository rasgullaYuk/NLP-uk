# CloudWatch Monitoring & Alerting (Task 15)

This project includes CloudWatch monitoring utilities for pipeline observability and alerting.

## Components

1. `cloudwatch_monitoring.py`
   - Custom metric publishing helpers
   - Log group provisioning
   - Queue depth collection
   - Dashboard generation
   - Alarm generation
   - SNS alert topic setup

2. `cloudwatch_monitoring_setup.py`
   - One-command setup for dashboard, alarms, log groups, and alerts

3. `publish_pipeline_metrics.py`
   - Batch metric publisher from output artifacts (`textract_outputs`, `track_a_outputs`, `track_b_outputs`)

## Metrics

### Stage and quality metrics

- `StageLatencySeconds`
- `ExtractionSuccessCount`
- `ExtractionErrorRate`
- `ExtractionConfidenceAvg`
- `SNOMEDMappingSuccessRate`
- `SNOMEDFallbackCount`
- `LLMInferenceLatencyMs`
- `LLMConfidenceScore`
- `UnifiedConfidenceScore`
- `AggregationLatencyMs`

### Queue metrics

- `QueueDepth` by queue name

## Dashboard

Dashboard includes:

- Stage duration trend
- Error rate + SNOMED mapping success
- LLM latency + unified confidence
- Queue depth for pipeline queues
- Estimated AWS charges (`AWS/Billing`)

## Alarm Policies

Default alarms configured:

1. Queue depth alarm per queue: `QueueDepth > 100`
2. Stage latency alarm: `StageLatencySeconds > 30`
3. Extraction error alarm: `ExtractionErrorRate > 5%`

All alarms can notify an SNS topic with optional email subscription.

## Setup

```bash
python cloudwatch_monitoring_setup.py --alert-email your-email@example.com
```

Optional:

```bash
python cloudwatch_monitoring_setup.py --queue-names TrackA_Entity_SNOMED_Queue,TrackB_Summary_Queue --dashboard-name NLP-UK-Pipeline-Health
```

## Publish Runtime/Batch Metrics

```bash
python publish_pipeline_metrics.py
```

## Cost Tracking and Optimization Recommendations

The monitoring module includes built-in recommendations:

- Enable log retention policies.
- Limit high-cardinality metric dimensions.
- Scope alarms to critical metrics only.
- Control Bedrock prompt/token sizes.
- Apply S3 lifecycle policies for historical artifacts.
