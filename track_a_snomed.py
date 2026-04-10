import json
import os
import time
import logging
import math
import re
import hashlib
from cloudwatch_monitoring import CloudWatchMonitoringManager, infer_document_type
from hipaa_compliance import (
    build_phi_detection_summary,
    create_secure_client,
    detect_phi_entities,
    scrub_text_for_logs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("track_a.snomed")

AWS_REGION          = "us-east-1"          # RVCE_AIML account region
MAX_COMPREHEND_CHARS = 9500                 # Comprehend Medical hard limit
COMPREHEND_CONF_THRESHOLD = 0.75           # Below this → trigger fallback
FAISS_TOP_K         = 10                   # Candidates to retrieve from FAISS
MAX_RETRIES         = 3                    # Retry attempts before DLQ
PERF_TARGET_SECONDS = 15                   # SRS acceptance criterion
SLIDING_WINDOW_MIN_WORDS = 50              # SRS: 50-100 words
SLIDING_WINDOW_MAX_WORDS = 100             # SRS: 50-100 words
FAISS_INDEX_PATH    = "faiss_index/snomed.index"
FAISS_META_PATH     = "faiss_index/snomed_meta.json"
FAISS_MIN_CODES     = int(os.environ.get("SNOMED_FAISS_MIN_CODES", "50000"))
TRACK_A_TARGET_SECONDS = float(os.environ.get("TRACK_A_TARGET_SECONDS", "10"))

CATEGORY_MAP = {
    "MEDICAL_CONDITION": "Problems_Issues",
    "DX_NAME":           "Diagnosis",
    "PROBLEM":           "Problems_Issues",
    "DIAGNOSIS":         "Diagnosis",
    "SIGN":              "Problems_Issues",
    "SYMPTOM":           "Problems_Issues",
    "MEDICATION":        "Medication",
    "GENERIC_NAME":      "Medication",
    "BRAND_NAME":        "Medication",
    "PROCEDURE":         "Procedures",
    "TEST":              "Investigations",
    "TEST_NAME":         "Investigations",
}

sqs_client          = create_secure_client("sqs",              region_name=AWS_REGION)
comprehend_medical  = create_secure_client("comprehendmedical", region_name=AWS_REGION)
try:
    cloudwatch_monitor = CloudWatchMonitoringManager()
except Exception:
    cloudwatch_monitor = None

_SEMANTIC_FALLBACK_CACHE = {}
_MAP_ENTITY_CACHE = {}



def _normalize_token(token):
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def _get_sliding_window(full_text, term, window_words=75):
    """
    Extract 50-100 word window around a clinical term to provide context.
    Per SRS: "sliding window of surrounding text (50-100 words)".
    """
    words = full_text.split()
    if not words:
        return ""

    target_words = max(SLIDING_WINDOW_MIN_WORDS, min(SLIDING_WINDOW_MAX_WORDS, int(window_words)))
    term_words = [_normalize_token(w) for w in term.split() if _normalize_token(w)]
    if not term_words:
        return " ".join(words[:target_words])

    for i, word in enumerate(words):
        if _normalize_token(word).startswith(term_words[0]):
            start = max(0, i - target_words // 2)
            end = min(len(words), start + target_words)

            # Ensure lower bound window length when possible.
            if end - start < SLIDING_WINDOW_MIN_WORDS and len(words) > SLIDING_WINDOW_MIN_WORDS:
                deficit = SLIDING_WINDOW_MIN_WORDS - (end - start)
                start = max(0, start - deficit)
                end = min(len(words), max(end, start + SLIDING_WINDOW_MIN_WORDS))
            return " ".join(words[start:end])
    return " ".join(words[:target_words])


def _get_sapbert_embedding(context_text):
    """
    Generate a normalized SapBERT embedding for clinical context text.

    SapBERT is loaded lazily on first call so startup time stays fast.
    Per SRS: "uses SapBERT embeddings to retrieve top 10 SNOMED candidates
    from the FAISS vector database".
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import numpy as np

        if not hasattr(_get_sapbert_embedding, "_model"):
            logger.info("Loading SapBERT model (first call — one-time load)...")
            _get_sapbert_embedding._tokenizer = AutoTokenizer.from_pretrained(
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            )
            _get_sapbert_embedding._model = AutoModel.from_pretrained(
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            )
            _get_sapbert_embedding._model.eval()
            logger.info("SapBERT model loaded.")

        tokenizer = _get_sapbert_embedding._tokenizer
        model     = _get_sapbert_embedding._model

        inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embedding = embedding.astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    except ImportError:
        logger.warning("SapBERT dependencies not installed. Returning zero embedding.")
        import numpy as np
        return np.zeros(768, dtype=np.float32)


def _search_faiss(embedding, top_k=FAISS_TOP_K):
    """
    Search the pre-built FAISS index for top-k SNOMED CT candidate codes.
    Per SRS: "retrieve top 10 potential SNOMED CT candidates from FAISS".

    Expects:
        faiss_index/snomed.index   — FAISS flat index
        faiss_index/snomed_meta.json — list of {code, description} dicts
    """
    try:
        import faiss
        import numpy as np

        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_META_PATH):
            logger.warning("FAISS index not found at %s. Skipping semantic search.", FAISS_INDEX_PATH)
            return []

        if not hasattr(_search_faiss, "_index"):
            logger.info("Loading FAISS index...")
            _search_faiss._index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                _search_faiss._meta = json.load(f)
            entry_count = min(_search_faiss._index.ntotal, len(_search_faiss._meta))
            logger.info("FAISS index loaded (%d entries).", entry_count)
            if entry_count < FAISS_MIN_CODES:
                logger.warning(
                    "FAISS index entries (%d) are below expected minimum (%d).",
                    entry_count, FAISS_MIN_CODES
                )

        index = _search_faiss._index
        meta  = _search_faiss._meta

        query = embedding.reshape(1, -1).astype(np.float32)
        if np.linalg.norm(query) == 0:
            logger.warning("SapBERT embedding norm is zero. Skipping FAISS retrieval.")
            return []
        faiss.normalize_L2(query)
        distances, indices = index.search(query, top_k)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(meta):
                continue
            cosine_similarity = max(-1.0, min(1.0, float(dist)))
            candidates.append({
                "snomed_code":   meta[idx]["code"],
                "description":   meta[idx]["description"],
                "faiss_distance": float(1 - cosine_similarity),
                "similarity":    cosine_similarity,
                "retrieval_confidence": round((cosine_similarity + 1) / 2, 4),
            })
        return candidates

    except ImportError:
        logger.warning("FAISS not installed. Skipping semantic search.")
        return []
    except Exception as e:
        logger.error("FAISS search failed: %s", e)
        return []


def _cross_encoder_rerank(clinical_context, candidates):
    """
    Re-rank SNOMED candidates using a Cross-Encoder model.
    Per SRS: "Cross-Encoder model re-ranks candidates to select the most
    contextually relevant code".
    """
    try:
        from sentence_transformers import CrossEncoder

        if not hasattr(_cross_encoder_rerank, "_model"):
            logger.info("Loading Cross-Encoder model (first call)...")
            _cross_encoder_rerank._model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("Cross-Encoder loaded.")

        model = _cross_encoder_rerank._model
        pairs = [(clinical_context, c["description"]) for c in candidates]

        if not pairs:
            return candidates

        scores = model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["cross_encoder_score"] = float(score)
            candidate["cross_encoder_confidence"] = float(1 / (1 + math.exp(-float(score))))

        reranked = sorted(candidates, key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
        return reranked

    except ImportError:
        logger.warning("sentence-transformers not installed. Returning candidates without re-ranking.")
        return candidates
    except Exception as e:
        logger.error("Cross-Encoder re-ranking failed: %s", e)
        return candidates


def semantic_snomed_fallback(term, full_text):
    """
    Full semantic fallback pipeline per SRS Section 3.2:
      1. Extract sliding window context around the term
      2. Generate SapBERT embedding
      3. Retrieve top-10 FAISS candidates
      4. Cross-Encoder re-ranking
      5. Return best match with confidence

    Returns a dict with snomed_code, description, confidence, source='semantic_fallback'
    or None if fallback also fails.
    """
    context = _get_sliding_window(full_text, term)
    cache_key = hashlib.sha256(f"{term}|{context}".encode("utf-8")).hexdigest()
    if cache_key in _SEMANTIC_FALLBACK_CACHE:
        return _SEMANTIC_FALLBACK_CACHE[cache_key]

    logger.info("  [Fallback] Running semantic search for term: '%s'", term)

    embedding = _get_sapbert_embedding(f"{term} [SEP] {context}")

    candidates = _search_faiss(embedding)
    if not candidates:
        logger.warning("  [Fallback] No FAISS candidates found for '%s'.", term)
        return None

    reranked = _cross_encoder_rerank(context, candidates)

    best = reranked[0]
    confidence = best.get("cross_encoder_confidence")
    if confidence is None:
        confidence = best.get("retrieval_confidence", 0.0)

    logger.info(
        "  [Fallback] Best match for '%s': %s (%s) score=%.3f",
        term, best["snomed_code"], best["description"],
        best.get("cross_encoder_score", best.get("similarity", 0))
    )
    output = {
        "snomed_code":   best["snomed_code"],
        "description":   best["description"],
        "confidence":    round(confidence, 4),
        "source":        "semantic_fallback",
        "all_candidates": reranked[:FAISS_TOP_K],
        "context_window": context,
    }
    _SEMANTIC_FALLBACK_CACHE[cache_key] = output
    return output


# ===========================================================================
# Entity Categorization (SRS: Problems/Issues, Medication, Diagnosis)
# ===========================================================================

def categorize_entities(snomed_entities, full_text):
    """
    Organise extracted SNOMED entities into clinical categories per SRS:
      - Problems_Issues
      - Diagnosis
      - Medication
      - Procedures
      - Investigations

    Each entity gets a category, snomed code, confidence, and the source
    (comprehend_medical or semantic_fallback).
    """
    categories = {
        "Problems_Issues":  [],
        "Diagnosis":        [],
        "Medication":       [],
        "Procedures":       [],
        "Investigations":   [],
        "Uncategorized":    [],
    }

    for entity in snomed_entities:
        text    = entity.get("Text", "")
        ent_cat = entity.get("Category", "")
        ent_type = entity.get("Type", "")
        conf    = entity.get("confidence", 0)
        snomed  = entity.get("snomed_result", {})

        bucket = (
            CATEGORY_MAP.get(ent_type)
            or CATEGORY_MAP.get(ent_cat)
            or "Uncategorized"
        )

        record = {
            "text":        text,
            "snomed_code": snomed.get("snomed_code", "NOT_MAPPED"),
            "description": snomed.get("description", ""),
            "confidence":  round(conf, 4),
            "source":      snomed.get("source", "comprehend_medical"),
        }

        categories[bucket].append(record)

    return categories


# ===========================================================================
# Confidence Aggregation
# ===========================================================================

def aggregate_confidence(snomed_entities):
    """
    Compute a unified confidence score across all entities.
    Weighted average of per-entity confidence (higher confidence entities
    matter more).
    """
    if not snomed_entities:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for entity in snomed_entities:
        conf = entity.get("confidence", 0)
        weighted_sum += conf * conf   
        total_weight += conf

    if total_weight == 0:
        return 0.0

    return round(weighted_sum / total_weight, 4)


# ===========================================================================
# Per-entity SNOMED mapping with fallback
# ===========================================================================

def map_entity_to_snomed(entity, full_text):
    """
    Map a single Comprehend Medical entity to a SNOMED code.

    Strategy:
      1. Use the SNOMED concept already returned by InferSNOMEDCT if
         confidence >= threshold.
      2. If confidence < threshold → semantic fallback.
      3. If no concepts returned at all → semantic fallback.
    """
    text = entity.get("Text", "")
    concepts = entity.get("SNOMEDCTConcepts", [])
    comprehend_conf = entity.get("Score", 0)
    concept_key = concepts[0].get("Code", "") if concepts else ""
    cache_key = hashlib.sha256(f"{text}|{round(comprehend_conf,4)}|{concept_key}".encode("utf-8")).hexdigest()
    if cache_key in _MAP_ENTITY_CACHE:
        return _MAP_ENTITY_CACHE[cache_key]

    if concepts and comprehend_conf >= COMPREHEND_CONF_THRESHOLD:
        top = concepts[0]
        output = {
            "snomed_code": top.get("Code", "NOT_MAPPED"),
            "description": top.get("Description", ""),
            "confidence":  round(comprehend_conf, 4),
            "source":      "comprehend_medical",
        }, comprehend_conf
        _MAP_ENTITY_CACHE[cache_key] = output
        return output

    logger.info(
        "  Confidence %.2f < %.2f for '%s' — triggering semantic fallback.",
        comprehend_conf, COMPREHEND_CONF_THRESHOLD, text
    )
    fallback_result = semantic_snomed_fallback(text, full_text)
    if fallback_result:
        output = (fallback_result, fallback_result["confidence"])
        _MAP_ENTITY_CACHE[cache_key] = output
        return output

    output = ({
        "snomed_code": "NOT_MAPPED",
        "description": "",
        "confidence":  0.0,
        "source":      "failed",
    }, 0.0)
    _MAP_ENTITY_CACHE[cache_key] = output
    return output


# ===========================================================================
# Core document processor
# ===========================================================================

def process_document(file_path):
    """
    Process a single Textract JSON output through Track A.

    Returns a result dict with:
      - categorized_entities (Problems, Medication, Diagnosis, ...)
      - unified_confidence_score
      - processing_time_seconds
      - status
    """
    t_start = time.time()
    logger.info("Processing document: %s", file_path)

    try:
        with open(file_path, "r") as f:
            textract_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not load Textract file '{file_path}': {e}")

    raw_text_lines = [
        block["Text"]
        for block in textract_data.get("Blocks", [])
        if block.get("BlockType") == "LINE" and "Text" in block
    ]
    full_text = " ".join(raw_text_lines)
    phi_entities = detect_phi_entities(full_text, comprehend_medical_client=comprehend_medical)
    phi_summary = build_phi_detection_summary(phi_entities)
    text_to_analyze = full_text[:MAX_COMPREHEND_CHARS]

    if not text_to_analyze.strip():
        raise ValueError("No text found in Textract output — cannot process.")

    stage_timings = {}
    infer_start = time.perf_counter()
    logger.info("Calling Amazon Comprehend Medical InferSNOMEDCT (PHI flagged=%d).", phi_summary["entity_count"])
    logger.debug("Masked clinical text preview: %s", scrub_text_for_logs(full_text, phi_entities)[:240])
    print("  Sending text to AWS Comprehend Medical...")
    snomed_response = comprehend_medical.infer_snomedct(Text=text_to_analyze)
    stage_timings["comprehend_seconds"] = round(time.perf_counter() - infer_start, 4)

    entities = snomed_response.get("Entities", [])
    logger.info("Comprehend returned %d entities.", len(entities))
    print(f"  Comprehend returned {len(entities)} entities.")

    mapping_start = time.perf_counter()
    enriched_entities = []
    fallback_count = 0
    for entity in entities:
        snomed_result, conf = map_entity_to_snomed(entity, full_text)
        if snomed_result.get("source") == "semantic_fallback":
            fallback_count += 1
        enriched_entities.append({
            **entity,
            "confidence":    conf,
            "snomed_result": snomed_result,
        })

    logger.info(
        "Mapping complete. %d entities: %d from Comprehend, %d from semantic fallback.",
        len(enriched_entities),
        len(enriched_entities) - fallback_count,
        fallback_count,
    )
    stage_timings["mapping_seconds"] = round(time.perf_counter() - mapping_start, 4)

    categories = categorize_entities(enriched_entities, full_text)

    unified_confidence = aggregate_confidence(enriched_entities)
    mapped_entities = len(
        [
            entity
            for entity in enriched_entities
            if entity.get("snomed_result", {}).get("snomed_code") not in {"", "NOT_MAPPED"}
        ]
    )

    elapsed = round(time.time() - t_start, 3)

    if elapsed > TRACK_A_TARGET_SECONDS:
        logger.warning(
            "Performance target missed: %.2fs > %.2fs for '%s'.",
            elapsed, TRACK_A_TARGET_SECONDS, file_path
        )
    else:
        logger.info("Performance OK: %.2fs < %ss.", elapsed, TRACK_A_TARGET_SECONDS)

    if cloudwatch_monitor:
        try:
            cloudwatch_monitor.publish_snomed_mapping_result(
                document_id=os.path.basename(file_path).replace("_textract.json", ""),
                total_entities=len(enriched_entities),
                mapped_entities=mapped_entities,
                fallback_count=fallback_count,
                latency_seconds=elapsed,
                document_type=infer_document_type(file_path),
            )
        except Exception as monitor_error:
            logger.debug("CloudWatch metric publish failed: %s", monitor_error)

    return {
        "source_file":             file_path,
        "status":                  "SUCCESS",
        "total_entities":          len(enriched_entities),
        "fallback_count":          fallback_count,
        "categorized_entities":    categories,
        "unified_confidence_score": unified_confidence,
        "processing_time_seconds": elapsed,
        "stage_timings_seconds": stage_timings,
        "latency_target_seconds": TRACK_A_TARGET_SECONDS,
        "latency_target_met": elapsed <= TRACK_A_TARGET_SECONDS,
        "comprehend_model_version": snomed_response.get("ModelVersion", "unknown"),
        "phi_detection":           phi_summary,
    }


# ===========================================================================
# Retry logic with exponential backoff
# ===========================================================================

def process_with_retry(file_path, max_retries=MAX_RETRIES):
    """
    Attempt to process a document up to max_retries times.
    Uses exponential backoff between attempts.

    Returns (result_dict, None) on success.
    Returns (None, error_str) after all retries exhausted.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Attempt %d/%d for: %s", attempt, max_retries, file_path)
            result = process_document(file_path)
            return result, None
        except Exception as e:
            last_error = e
            wait = 2 ** attempt  
            logger.warning(
                "Attempt %d/%d failed for '%s': %s. Retrying in %ds...",
                attempt, max_retries, file_path, e, wait
            )
            print(f"  ATTEMPT {attempt}/{max_retries} FAILED: {e}. Retrying in {wait}s...")
            if attempt < max_retries:
                time.sleep(wait)

    logger.error("All %d attempts failed for '%s': %s", max_retries, file_path, last_error)
    return None, str(last_error)


# ===========================================================================
# Main queue processor
# ===========================================================================

def process_track_a_queue(output_dir="track_a_outputs"):
    """
    Track A: Medical Entity & SNOMED Mapping
    Pulls messages continuously until the SQS queue is empty.

    Enhanced from original:
    - Per-message retry with exponential backoff (3 attempts)
    - DLQ capture after all retries exhausted (zero message loss)
    - Continues processing next message even if one fails
    - Outputs categorized JSON with confidence scores
    - Performance tracked per document
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        queue_url = sqs_client.get_queue_url(
            QueueName="TrackA_Medical_Queue"
        )["QueueUrl"]
    except Exception as e:
        print(f"Could not find TrackA_Medical_Queue. Did you run tier2_router.py? Error: {e}")
        logger.critical("Could not find TrackA_Medical_Queue: %s", e)
        return

    try:
        dlq_url = sqs_client.get_queue_url(
            QueueName="TrackA_Medical_DLQ"
        )["QueueUrl"]
        logger.info("DLQ found: %s", dlq_url)
    except Exception:
        dlq_url = None
        logger.warning("TrackA_Medical_DLQ not found. Failed messages will only be logged.")
        print("  WARNING: DLQ not configured. Create 'TrackA_Medical_DLQ' in SQS for zero message loss.")

    print("--- [Track A: Starting Continuous Batch Processing] ---")

    total_processed = 0
    total_failed    = 0
    total_dlq       = 0

    while True:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=5,
            VisibilityTimeout=300,
        )

        if "Messages" not in response:
            print("\n--- Queue is empty! All documents processed. ---")
            break

        for message in response["Messages"]:
            body          = json.loads(message["Body"])
            file_path     = body.get("source_file", "")
            receipt_handle = message["ReceiptHandle"]

            print(f"\nProcessing document: {file_path}")
            logger.info("Received message for: %s", file_path)

            result, error = process_with_retry(file_path, max_retries=MAX_RETRIES)

            if result:
                base_name   = os.path.basename(file_path).replace("_textract.json", "")
                output_file = os.path.join(output_dir, f"{base_name}_snomed.json")

                with open(output_file, "w") as f:
                    json.dump(result, f, indent=4)

                print(f"  SUCCESS: Saved to {output_file}")
                print(f"  Entities: {result['total_entities']} | "
                      f"Fallbacks: {result['fallback_count']} | "
                      f"Confidence: {result['unified_confidence_score']} | "
                      f"Time: {result['processing_time_seconds']}s")

                cats = result["categorized_entities"]
                print(f"  Categories → Problems: {len(cats['Problems_Issues'])} | "
                      f"Diagnosis: {len(cats['Diagnosis'])} | "
                      f"Medication: {len(cats['Medication'])}")

                logger.info(
                    "SUCCESS: %s | entities=%d | fallbacks=%d | confidence=%.3f | time=%.2fs",
                    file_path,
                    result["total_entities"],
                    result["fallback_count"],
                    result["unified_confidence_score"],
                    result["processing_time_seconds"],
                )

                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                )
                logger.info("Message deleted from queue.")
                total_processed += 1

            else:
                total_failed += 1
                logger.error(
                    "FAILED after %d retries: %s | Error: %s",
                    MAX_RETRIES, file_path, error
                )
                print(f"  FAILED after {MAX_RETRIES} retries: {error}")

                if dlq_url:
                    from sqs_messaging import send_to_dlq
                    send_to_dlq(
                        dlq_url=dlq_url,
                        original_payload=body,
                        error_reason=error,
                        attempt_count=MAX_RETRIES,
                    )
                    total_dlq += 1
                    sqs_client.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=receipt_handle
                    )
                else:
                    logger.warning(
                        "No DLQ configured — message left in queue for retry after visibility timeout."
                    )
                    print("  Message left in queue (no DLQ configured).")

    print("\n====== TRACK A COMPLETE ======")
    print(f"  Processed successfully : {total_processed}")
    print(f"  Failed (sent to DLQ)   : {total_dlq}")
    print(f"  Failed (no DLQ)        : {total_failed - total_dlq}")
    logger.info(
        "Track A complete. Processed: %d | DLQ: %d | Failed: %d",
        total_processed, total_dlq, total_failed - total_dlq
    )



if __name__ == "__main__":
    process_track_a_queue()
