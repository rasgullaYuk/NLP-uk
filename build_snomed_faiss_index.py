"""
Build SNOMED FAISS index for Track A semantic fallback.

Expected output:
  - faiss_index/snomed.index
  - faiss_index/snomed_meta.json

The builder enforces a minimum number of SNOMED concepts (default 50,000)
to satisfy semantic fallback coverage requirements.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger("track_a.faiss_builder")

DEFAULT_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_MIN_CODES = 50000
DEFAULT_BATCH_SIZE = 64


def _first_value(record: Dict, candidates: List[str]) -> str:
    for key in candidates:
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    return ""


def load_snomed_records(input_path: str) -> List[Dict[str, str]]:
    """
    Load SNOMED records from JSON/JSONL/CSV/TSV with flexible field names.
    """
    ext = os.path.splitext(input_path)[1].lower()
    records: List[Dict[str, str]] = []

    if ext in {".json", ".jsonl"}:
        with open(input_path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                parsed = json.load(f)
                data = parsed.get("records", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records or {'records': [...]} format.")
        iterable = data
    elif ext in {".csv", ".tsv"}:
        delimiter = "\t" if ext == ".tsv" else ","
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            iterable = list(csv.DictReader(f, delimiter=delimiter))
    else:
        raise ValueError(f"Unsupported input format: {ext}. Use .json, .jsonl, .csv, or .tsv")

    for row in iterable:
        code = _first_value(row, ["code", "Code", "snomed_code", "SNOMED_CODE", "conceptId", "concept_id"])
        description = _first_value(
            row,
            ["description", "Description", "term", "Term", "fsn", "FSN", "preferredTerm", "preferred_term"],
        )
        if not code or not description:
            continue
        records.append({"code": code, "description": description})

    # De-duplicate by code.
    dedup = {}
    for rec in records:
        dedup[rec["code"]] = rec
    return list(dedup.values())


def build_faiss_index(
    records: List[Dict[str, str]],
    output_dir: str = "faiss_index",
    model_name: str = DEFAULT_SAPBERT_MODEL,
    min_codes: int = DEFAULT_MIN_CODES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, str]:
    """
    Generate SapBERT embeddings, create cosine-similarity FAISS index, and persist artifacts.
    """
    if len(records) < min_codes:
        raise ValueError(
            f"SNOMED record count {len(records)} is below required minimum {min_codes}."
        )

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    import faiss

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading SapBERT model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    descriptions = [r["description"] for r in records]
    all_embeddings = []

    for start in range(0, len(descriptions), batch_size):
        batch = descriptions[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(batch_embeddings)
        all_embeddings.append(batch_embeddings)
        if (start // batch_size) % 200 == 0:
            logger.info("Embedded %d/%d SNOMED terms...", min(start + batch_size, len(descriptions)), len(descriptions))

    matrix = np.vstack(all_embeddings).astype(np.float32)
    dimension = matrix.shape[1]

    logger.info("Creating FAISS IndexFlatIP (cosine over normalized vectors), dim=%d", dimension)
    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)

    index_path = os.path.join(output_dir, "snomed.index")
    meta_path = os.path.join(output_dir, "snomed_meta.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("FAISS index created with %d SNOMED codes.", index.ntotal)
    return {
        "index_path": index_path,
        "meta_path": meta_path,
        "count": str(index.ntotal),
        "dimension": str(dimension),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for SNOMED semantic fallback.")
    parser.add_argument("--input", required=True, help="Path to SNOMED source file (.json/.jsonl/.csv/.tsv)")
    parser.add_argument("--output-dir", default="faiss_index", help="Directory for output index/meta")
    parser.add_argument("--model", default=DEFAULT_SAPBERT_MODEL, help="SapBERT model id")
    parser.add_argument("--min-codes", type=int, default=DEFAULT_MIN_CODES, help="Minimum required SNOMED code count")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    args = parse_args()
    records = load_snomed_records(args.input)
    logger.info("Loaded %d SNOMED records from %s", len(records), args.input)
    result = build_faiss_index(
        records=records,
        output_dir=args.output_dir,
        model_name=args.model,
        min_codes=args.min_codes,
        batch_size=args.batch_size,
    )
    logger.info("Build complete: %s", result)


if __name__ == "__main__":
    main()
