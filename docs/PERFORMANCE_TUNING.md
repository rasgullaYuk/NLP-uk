# Performance Tuning Recommendations

## Evidence-backed hotspots

Based on current pipeline design and test instrumentation:

1. OCR extraction latency (Textract call time)
2. SNOMED fallback path (embedding + FAISS + rerank)
3. Summarization model inference latency
4. Queue backpressure when confidence routes spike
5. Export retries under EMIS endpoint instability

## Recommendations

### 1. Optimize OCR and routing throughput
- Batch intake processing where possible.
- Keep image preprocessing deterministic and avoid excessive intermediate writes.
- Scale worker count by queue depth.

### 2. Reduce SNOMED fallback cost
- Ensure FAISS index stays resident in memory when possible.
- Cache embeddings for repeated terms within a document.
- Trigger fallback only below confidence threshold (already implemented at 0.75).

### 3. Improve summarization latency
- Constrain prompt/context size with chunking and top-k retrieval.
- Keep model temperature low and token budgets bounded.
- Track p50/p95 model latency in CloudWatch.

### 4. Control retry storms
- Use exponential backoff for EMIS retries.
- Cap retry attempts and move unresolved messages to monitored queue/dead-letter path.
- Alert on sustained retry queue growth.

### 5. Monitor and tune by document type
- Publish per-document-type latency and success rates.
- Maintain dashboards for extraction, mapping, summarization, and export stages.

## Suggested target SLOs

- End-to-end processing: < 60s per document (acceptance target).
- Export request attempt latency: < 5s p95.
- Retry queue backlog: near-zero steady state.
