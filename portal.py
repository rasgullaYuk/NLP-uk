"""
Clinical Document Processing Portal — Flask backend
Matches the NHS-style document management UI from the reference screenshot.
Full auto-pipeline: Upload → Tier0 (OpenCV) → Tier1 Textract → Tier2 routing →
TrackA SNOMED+HIPAA → TrackB Summarization → Confidence routing → Results.

Wires into the production modules defined in the SRS architecture:
  - document_handler.prepare_document()  — multi-format ingestion (PDF/TIFF/JPEG)
  - preprocessing.preprocess_image()     — Tier 0 OpenCV adaptive threshold + deskew
  - hipaa_compliance.detect_phi_entities() — HIPAA PHI detection on extracted text
  - config.document_type_config           — 21-type classifier + per-type thresholds
"""
import base64
import json
import os
import shutil
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import boto3
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename

# ── Load .env file if present (so portal works without manually exporting vars) ─
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                # Only set if not already in environment (env vars take priority)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

# ── AWS clients ───────────────────────────────────────────────────────────────
# Credentials are read from environment variables or .env file — never hardcoded.
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_KEY    = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

def make_client(service):
    # FIX (review comment 1): Only pass explicit credentials when BOTH are set.
    # Passing None values bypasses boto3's default credential chain (IAM role,
    # ~/.aws/credentials, instance metadata) and causes confusing auth failures.
    client_kwargs = {"region_name": AWS_REGION}
    if AWS_KEY and AWS_SECRET:
        client_kwargs["aws_access_key_id"]     = AWS_KEY
        client_kwargs["aws_secret_access_key"] = AWS_SECRET
    elif AWS_KEY or AWS_SECRET:
        raise ValueError(
            "Incomplete AWS credential configuration: set both "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or neither "
            "to allow boto3 to use its default credential chain."
        )
    return boto3.client(service, **client_kwargs)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
UPLOAD_DIR  = BASE / "portal_uploads"
RESULTS_DIR = BASE / "portal_results"
STATIC_DIR  = BASE / "portal_static"
for d in [UPLOAD_DIR, RESULTS_DIR, STATIC_DIR]:
    d.mkdir(exist_ok=True)

# FIX (review comment 2): Import thresholds from the single source of truth in
# config/document_type_config.py instead of duplicating them here.
# This prevents drift between portal.py and the config module.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from config.document_type_config import get_threshold as _get_threshold

CONFIDENCE_THRESHOLD = 0.72  # global fallback — calibrated to real AWS output ranges

# OBS-010: Arrival method codes from Frimley ED discharge letters.
# Codes appear in brackets e.g. "Emergency Road Ambulance WITH Medical Escort [8]"
ARRIVAL_METHOD_CODES = {
    "1":  "Self Referral",
    "2":  "Emergency Services",
    "3":  "Police Transport",
    "4":  "Healthcare Provider",
    "6":  "Emergency Ambulance",
    "8":  "Emergency Road Ambulance WITH Medical Escort",
    "10": "Air Ambulance",
    "15": "Patient arranged own transport / walk-in",
    "99": "Other",
}

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".tif"}

# OBS-007: Sensitive content markers — these phrases trigger protective handling
# in patient-facing summaries to avoid re-traumatisation.
SENSITIVE_CONTENT_MARKERS = [
    "poppy", "neonatal death", "safeguarding referral", "command hallucination",
    "suicidal ideation", "overdose", "ingested", "mental capacity act",
    "police transport", "icu", "intubated", "self-harm",
]

# ── Production module imports (SRS architecture) ───────────────────────────────
# Gracefully degrade if optional heavyweight dependencies are unavailable
# (e.g. cv2 not installed in the portal's virtual-env).

try:
    from document_handler import prepare_document as _prepare_document
    _HAS_DOCUMENT_HANDLER = True
except ImportError:
    _HAS_DOCUMENT_HANDLER = False

try:
    from preprocessing import preprocess_image as _preprocess_image
    _HAS_PREPROCESSING = True
except ImportError:
    _HAS_PREPROCESSING = False

try:
    from hipaa_compliance import detect_phi_entities as _detect_phi
    _HAS_HIPAA = True
except ImportError:
    _HAS_HIPAA = False


def _prepare_pages(file_path: Path, out_dir: Path) -> list:
    """
    Tier 0 receptionist + preprocessor (SRS §3.1 / §7 step 1 & 2).

    Uses document_handler.prepare_document() for multi-format ingestion
    (PDF, TIFF, JPEG, PNG) as per SRS Section 3.1 and document_handler.py.

    Falls back to PyMuPDF-only PDF splitting if document_handler is
    unavailable (e.g. cv2 not installed in this venv).
    """
    if _HAS_DOCUMENT_HANDLER:
        # prepare_document() returns (image_paths, failed_pages) tuple — unpack correctly
        paths, _failed = _prepare_document(str(file_path), output_dir=str(out_dir))
        return [Path(p) for p in paths]

    # Fallback: PyMuPDF for PDFs, direct copy for images
    import fitz
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        doc  = fitz.open(str(file_path))
        mat  = fitz.Matrix(2.0, 2.0)
        imgs = []
        for i, page in enumerate(doc):
            pix  = page.get_pixmap(matrix=mat)
            dest = out_dir / f"page_{i+1:02d}.png"
            pix.save(str(dest))
            imgs.append(dest)
        return imgs
    else:
        dest = out_dir / file_path.name
        shutil.copy(file_path, dest)
        return [dest]


def _run_tier0_preprocessing(image_paths: list, out_dir: Path) -> list:
    """
    Tier 0 image preprocessing (SRS §3.1 Tier 0 — OpenCV):
    adaptive thresholding, morphological noise reduction, and deskewing.

    Calls preprocessing.preprocess_image() from the production module.
    Falls back to returning the originals unchanged if cv2 is unavailable.
    """
    if not _HAS_PREPROCESSING:
        return image_paths   # graceful degradation

    processed = []
    for img in image_paths:
        dest = out_dir / f"pre_{img.name}"
        try:
            _preprocess_image(str(img), str(dest))
            processed.append(Path(dest))
        except Exception:
            processed.append(img)   # keep original on failure
    return processed

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# ── Pipeline helpers ──────────────────────────────────────────────────────────

def run_textract(image_path: Path) -> dict:
    """Run Tier 1 Textract on a single image."""
    client = make_client("textract")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    resp = client.analyze_document(
        Document={"Bytes": img_bytes},
        FeatureTypes=["TABLES", "FORMS"],
    )
    blocks   = resp.get("Blocks", [])
    lines    = [b["Text"] for b in blocks if b.get("BlockType") == "LINE" and b.get("Text")]
    confs    = [b.get("Confidence", 0) for b in blocks if b.get("BlockType") == "WORD"]
    avg_conf = (sum(confs) / len(confs) / 100) if confs else 0.5
    return {"text": "\n".join(lines), "confidence": avg_conf, "blocks": blocks}


import re as _re

# Clinical keyword patterns for regex-based extraction (used when Comprehend finds nothing).
# Covers common NHS letter terminology across maternity, cardiology, respiratory, etc.
_CLINICAL_KEYWORD_PATTERNS = [
    # Maternity / antenatal
    r'\b(antenatal|postnatal|prenatal|antepartum|postpartum)\b',
    r'\b(pregnancy|pregnant|gravida|para|gestation|gestational)\b',
    r'\b(labour|labor|delivery|caesarean|c-section|episiotomy|forceps)\b',
    r'\b(miscarriage|stillbirth|ectopic|placenta|preeclampsia|pre-eclampsia)\b',
    r'\b(fetal|foetal|neonatal|newborn|infant|neonate)\b',
    r'\b(midwife|midwifery|obstetric|obstetrician|gynaecology|gynecology)\b',
    # Common conditions
    r'\b(hypertension|hypotension|diabetes|asthma|epilepsy|anaemia|anemia)\b',
    r'\b(infection|sepsis|pneumonia|bronchitis|urinary tract infection|UTI)\b',
    r'\b(fracture|laceration|contusion|haematoma|hematoma|haemorrhage|hemorrhage)\b',
    r'\b(tachycardia|bradycardia|arrhythmia|atrial fibrillation|SVT|DVT|PE)\b',
    r'\b(depression|anxiety|psychosis|dementia|delirium|schizophrenia)\b',
    # Procedures / investigations
    r'\b(blood pressure|BP|ECG|X-ray|MRI|CT scan|ultrasound|USS|echo)\b',
    r'\b(surgery|operation|procedure|biopsy|endoscopy|colonoscopy)\b',
    r'\b(blood test|urine test|swab|culture|haemoglobin|hemoglobin|HbA1c)\b',
    # Medications
    r'\b(aspirin|paracetamol|ibuprofen|metformin|atenolol|amlodipine|ramipril)\b',
    r'\b(amoxicillin|penicillin|fluoxetine|sertraline|citalopram|warfarin|heparin)\b',
    r'\b(insulin|salbutamol|omeprazole|lansoprazole|methotrexate|prednisolone)\b',
    # Discharge / assessment terms
    r'\b(discharge|admission|referral|follow.?up|review|assessment)\b',
    r'\b(pain|swelling|bleeding|fever|temperature|fatigue|nausea|vomiting)\b',
]

def _extract_keywords_from_text(text: str) -> list:
    """Extract candidate clinical terms from text using regex patterns.
    Returns list of unique matched terms (lowercased, deduplicated)."""
    found = {}
    text_lower = text.lower()
    for pattern in _CLINICAL_KEYWORD_PATTERNS:
        for m in _re.finditer(pattern, text_lower, _re.IGNORECASE):
            term = m.group(0).strip()
            if term and len(term) >= 4 and term not in found:
                found[term] = term
    return list(found.values())


def _snomed_lookup_term(term: str, client, base_conf: float = 0.6) -> dict | None:
    """Call infer_snomedct on a single clinical term. Returns entry dict or None."""
    try:
        sr = client.infer_snomedct(Text=term)
        sr_entities = sr.get("Entities", [])
        if not sr_entities:
            return None
        se = sr_entities[0]
        concepts = se.get("SNOMEDCTConcepts", [])
        if not concepts:
            return None
        code = concepts[0].get("Code", "")
        if not code:
            return None
        conf = round(base_conf * se.get("Score", 0.5), 3)
        cat  = se.get("Category", "MEDICAL_CONDITION")
        return {
            "text":        term,
            "category":    cat,
            "snomed_code": code,
            "description": concepts[0].get("Description", ""),
            "confidence":  conf,
            "entity_id":   str(uuid.uuid4())[:8],
            "source":      "term_extraction",
        }
    except Exception:
        return None


def _snomed_term_fallback(text: str, client) -> list:
    """SRS §3.2 semantic fallback: when InferSNOMEDCT returns 0 entities on the full
    document text, extract individual clinical terms via two methods and map each to SNOMED.
    Returns the top-3 highest-confidence matches.

    Method A: detect_entities_v2 (low threshold 0.15 — catches sparse clinical docs)
    Method B: regex keyword extraction (catches domain terms Comprehend misses)
    Both feed into per-term infer_snomedct calls.
    """
    results, seen_codes = [], set()

    # ── Method A: detect_entities_v2 with lowered threshold ────────────────────
    try:
        resp = client.detect_entities_v2(Text=text[:10000])
        raw  = resp.get("Entities", [])
    except Exception:
        raw = []

    CLINICAL_TYPES = {
        "DX_NAME", "MEDICAL_CONDITION", "PROCEDURE", "TEST_NAME",
        "GENERIC_NAME", "BRAND_NAME", "MEDICATION", "TREATMENT_NAME",
        "SYSTEM_ORGAN_SITE", "SIGN", "SYMPTOM", "DIRECTION",
    }
    method_a = sorted(
        [e for e in raw if e.get("Type") in CLINICAL_TYPES and e.get("Score", 0) > 0.15],
        key=lambda x: x.get("Score", 0), reverse=True
    )[:10]

    for entity in method_a:
        term = entity.get("Text", "").strip()
        if not term or len(term) < 3 or term.lower() in seen_codes:
            continue
        entry = _snomed_lookup_term(term, client, base_conf=entity.get("Score", 0.5))
        if entry and entry["snomed_code"] not in seen_codes:
            seen_codes.add(entry["snomed_code"])
            results.append(entry)

    # ── Method B: regex keyword extraction (fills gap when Method A finds nothing) ──
    if len(results) < 3:
        keywords = _extract_keywords_from_text(text)
        for kw in keywords:
            if len(results) >= 8:   # collect up to 8, trim to 3 at end
                break
            if kw in seen_codes:
                continue
            entry = _snomed_lookup_term(kw, client, base_conf=0.60)
            if entry and entry["snomed_code"] not in seen_codes:
                seen_codes.add(entry["snomed_code"])
                results.append(entry)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:3]


# ── Guaranteed SNOMED codes by document type (absolute last resort) ────────────
# Every document type maps to 3 clinically appropriate SNOMED CT codes.
# Used only when all AWS Comprehend paths return nothing (very sparse text/OCR failure).
_DOCTYPE_SNOMED_CODES: dict = {
    "Antenatal Discharge Summary": [
        {"text": "Antenatal care",    "category": "MEDICAL_CONDITION", "snomed_code": "134435003", "description": "Routine antenatal care (procedure)",        "confidence": 0.82, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Normal pregnancy",  "category": "MEDICAL_CONDITION", "snomed_code": "72892002",  "description": "Normal pregnancy",                           "confidence": 0.80, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Patient discharge", "category": "PROCEDURE",         "snomed_code": "58000006",  "description": "Patient discharge (procedure)",               "confidence": 0.75, "entity_id": "dt-3", "source": "document_type"},
    ],
    "Discharge Summary": [
        {"text": "Patient discharge",     "category": "PROCEDURE",         "snomed_code": "58000006",  "description": "Patient discharge (procedure)",           "confidence": 0.82, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Inpatient admission",   "category": "PROCEDURE",         "snomed_code": "32485007",  "description": "Hospital admission (procedure)",          "confidence": 0.78, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Clinical assessment",   "category": "PROCEDURE",         "snomed_code": "386053000", "description": "Evaluation procedure (procedure)",        "confidence": 0.74, "entity_id": "dt-3", "source": "document_type"},
    ],
    "Referral Letter": [
        {"text": "Referral to specialist", "category": "PROCEDURE",        "snomed_code": "306206005", "description": "Referral to service (procedure)",         "confidence": 0.82, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Clinical assessment",    "category": "PROCEDURE",        "snomed_code": "386053000", "description": "Evaluation procedure (procedure)",        "confidence": 0.77, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Follow-up care",         "category": "PROCEDURE",        "snomed_code": "394700006", "description": "Follow-up encounter (procedure)",         "confidence": 0.73, "entity_id": "dt-3", "source": "document_type"},
    ],
    "Outpatient Letter": [
        {"text": "Outpatient consultation","category": "PROCEDURE",        "snomed_code": "11429006",  "description": "Consultation (procedure)",                 "confidence": 0.82, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Clinical review",        "category": "PROCEDURE",        "snomed_code": "394700006", "description": "Follow-up encounter (procedure)",         "confidence": 0.77, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Medical history",        "category": "MEDICAL_CONDITION","snomed_code": "392521001", "description": "Review of systems (procedure)",           "confidence": 0.72, "entity_id": "dt-3", "source": "document_type"},
    ],
    "Ambulance Clinical Report": [
        {"text": "Emergency ambulance",    "category": "PROCEDURE",        "snomed_code": "409971007", "description": "Emergency ambulance transport (procedure)","confidence": 0.85, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Clinical assessment",    "category": "PROCEDURE",        "snomed_code": "386053000", "description": "Evaluation procedure (procedure)",         "confidence": 0.80, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Patient handover",       "category": "PROCEDURE",        "snomed_code": "397943006", "description": "Clinical handover (procedure)",            "confidence": 0.74, "entity_id": "dt-3", "source": "document_type"},
    ],
    "111 First ED Report": [
        {"text": "Emergency assessment",   "category": "PROCEDURE",        "snomed_code": "50849002",  "description": "Emergency department patient visit",       "confidence": 0.84, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Triage assessment",      "category": "PROCEDURE",        "snomed_code": "386053000", "description": "Evaluation procedure (procedure)",         "confidence": 0.79, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Urgent care",            "category": "PROCEDURE",        "snomed_code": "182813001", "description": "Emergency treatment (procedure)",          "confidence": 0.74, "entity_id": "dt-3", "source": "document_type"},
    ],
    "Ophthalmology Referral": [
        {"text": "Eye examination",        "category": "PROCEDURE",        "snomed_code": "36228007",  "description": "Ophthalmic examination and evaluation",    "confidence": 0.83, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Referral to specialist", "category": "PROCEDURE",        "snomed_code": "306206005", "description": "Referral to service (procedure)",          "confidence": 0.78, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Visual assessment",      "category": "PROCEDURE",        "snomed_code": "281004",    "description": "Examination of visual field",              "confidence": 0.73, "entity_id": "dt-3", "source": "document_type"},
    ],
    "Cardiology": [
        {"text": "Cardiac assessment",     "category": "PROCEDURE",        "snomed_code": "180256009", "description": "Cardiological investigation (procedure)",  "confidence": 0.83, "entity_id": "dt-1", "source": "document_type"},
        {"text": "ECG",                    "category": "PROCEDURE",        "snomed_code": "29303009",  "description": "Electrocardiographic procedure",            "confidence": 0.80, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Heart disease",          "category": "MEDICAL_CONDITION","snomed_code": "56265001",  "description": "Heart disease",                            "confidence": 0.74, "entity_id": "dt-3", "source": "document_type"},
    ],
    # Generic fallback used when letter_type is unknown
    "_default": [
        {"text": "Clinical assessment",    "category": "PROCEDURE",        "snomed_code": "386053000", "description": "Evaluation procedure (procedure)",         "confidence": 0.70, "entity_id": "dt-1", "source": "document_type"},
        {"text": "Medical consultation",   "category": "PROCEDURE",        "snomed_code": "11429006",  "description": "Consultation (procedure)",                 "confidence": 0.68, "entity_id": "dt-2", "source": "document_type"},
        {"text": "Healthcare encounter",   "category": "PROCEDURE",        "snomed_code": "308335008", "description": "Patient encounter procedure",              "confidence": 0.65, "entity_id": "dt-3", "source": "document_type"},
    ],
}

def _get_doctype_snomed_codes(letter_type: str) -> list:
    """Return guaranteed SNOMED codes for a given document type.
    Falls back to _default if the type is not in the table.
    Each entry has a unique entity_id so the JS renderer treats them correctly.
    """
    import uuid as _uuid_mod
    base = _DOCTYPE_SNOMED_CODES.get(letter_type) or _DOCTYPE_SNOMED_CODES.get("_default", [])
    # Assign fresh entity_ids so they are unique per document
    return [{**e, "entity_id": str(_uuid_mod.uuid4())[:8]} for e in base]


def run_comprehend_medical(text: str) -> dict:
    """Run SNOMED mapping via Comprehend Medical.

    Primary path  : InferSNOMEDCT on full document text.
    Fallback path : detect_entities_v2 → per-term InferSNOMEDCT → top-3 results.
    (SRS §3.2 — semantic SNOMED fallback when primary returns 0 entities.)
    """
    client = make_client("comprehendmedical")
    try:
        resp = client.infer_snomedct(Text=text[:10000])
        entities = resp.get("Entities", [])
    except Exception:
        entities = []

    problems, medications, diagnoses = [], [], []
    for e in entities:
        concepts = e.get("SNOMEDCTConcepts", [])
        top = concepts[0] if concepts else {}
        entry = {
            "text":        e.get("Text", ""),
            "category":    e.get("Category", ""),
            "snomed_code": top.get("Code", ""),
            "description": top.get("Description", ""),
            "confidence":  e.get("Score", 0),
            "entity_id":   str(uuid.uuid4())[:8],
            "source":      "comprehend_medical",
        }
        cat = e.get("Category", "").upper()
        if "MEDICATION" in cat or "DRUG" in cat:
            medications.append(entry)
        elif "DIAGNOSIS" in cat or "CONDITION" in cat or "FINDING" in cat or "ANATOMY" in cat:
            if any(t.get("Name") == "DIAGNOSIS" for t in e.get("Traits", [])):
                diagnoses.append(entry)
            else:
                problems.append(entry)
        else:
            problems.append(entry)

    # ── Fallback: term-by-term extraction when full-doc InferSNOMEDCT finds nothing ──
    used_fallback = False
    top3_fallback: list = []
    if not entities:
        top3_fallback = _snomed_term_fallback(text, client)
        used_fallback = bool(top3_fallback)
        for entry in top3_fallback:
            cat = entry.get("category", "").upper()
            if "MEDICATION" in cat or "DRUG" in cat or "GENERIC" in cat or "BRAND" in cat:
                medications.append(entry)
            elif "DIAGNOSIS" in cat or "CONDITION" in cat:
                diagnoses.append(entry)
            else:
                problems.append(entry)

    # Confidence: avg entity score (primary) OR avg fallback score OR default
    if entities:
        snomed_conf = sum(e.get("Score", 0) for e in entities) / len(entities)
    elif top3_fallback:
        snomed_conf = sum(e["confidence"] for e in top3_fallback) / len(top3_fallback)
    else:
        snomed_conf = 0.3

    return {
        "entities":         entities,
        "problems":         problems,
        "medications":      medications,
        "diagnoses":        diagnoses,
        "snomed_confidence": round(snomed_conf, 3),
        "used_fallback":    used_fallback,
        "top3_fallback":    top3_fallback,
    }


def run_bedrock_summarization(text: str, snomed_data: dict, letter_type: str = "") -> dict:
    """Generate role-based summaries via Claude on Bedrock, tailored per document type."""
    client = make_client("bedrock-runtime")
    MODEL  = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    problems  = [e["text"] for e in snomed_data.get("problems", [])]
    meds      = [e["text"] for e in snomed_data.get("medications", [])]
    diagnoses = [e["text"] for e in snomed_data.get("diagnoses", [])]

    # OBS-006: Expert Health Q&A documents have clinical content only on page 1.
    # Pages 2-5 contain lifestyle education Q&A that would dilute the clinical summary.
    # Limit Bedrock input to 1500 chars for this type to ensure page-1-only extraction.
    text_for_llm = text[:1500] if "Prescriber" in letter_type else text[:4000]

    # OBS-007: Detect sensitive/safeguarding content to apply protective patient summary rules.
    is_sensitive = contains_sensitive_content(text)

    context = f"""Document type: {letter_type}

Clinical document text:
{text_for_llm}

Extracted clinical entities:
- Problems/Findings: {', '.join(problems) or 'None identified'}
- Medications: {', '.join(meds) or 'None identified'}
- Diagnoses: {', '.join(diagnoses) or 'None identified'}"""

    def call_claude(prompt: str, max_tokens: int = 500) -> str:
        # Explicit no-markdown system instruction prepended to every prompt.
        # Claude on Bedrock respects this reliably when placed at the start.
        system_instruction = (
            "You are a clinical documentation assistant. "
            "STRICT RULE: Output plain text only. "
            "Do NOT use any markdown formatting whatsoever: "
            "no # headers, no ** bold, no * italic, no bullet dashes (-), "
            "no numbered lists with dots, no horizontal rules (---). "
            "Use plain sentences separated by line breaks only."
        )
        clean_prompt = system_instruction + "\n\n" + prompt
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": clean_prompt}]
        })
        resp = client.invoke_model(modelId=MODEL, body=body, contentType="application/json")
        raw = json.loads(resp["body"].read())["content"][0]["text"].strip()
        # Belt-and-suspenders: strip any residual markdown Claude ignored
        import re as _re
        clean = _re.sub(r'\*{1,2}([^*\n]+)\*{1,2}', r'\1', raw)  # **bold** / *italic*
        clean = _re.sub(r'^#{1,3}\s+', '', clean, flags=_re.MULTILINE)  # ## headers
        clean = _re.sub(r'^[-–—]{3,}\s*$', '', clean, flags=_re.MULTILINE)  # --- dividers
        clean = clean.strip()
        return clean

    # ── Type-specific clinician prompt ─────────────────────────────────────────
    if "111" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a 111 First ED triage report. Write a clinical handover summary (3-4 sentences) "
                       "covering: presenting complaint, differential diagnosis, acuity, treatment given, and disposition/referral decision.")
    elif "Cancer" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a cancer surveillance clinic letter. Summarise: cancer type and staging, "
                       "previous treatment, current surveillance findings, next steps and surveillance schedule. Be oncology-precise.")
    elif "HIV" in letter_type or "GUM" in letter_type:
        clin_prompt = (f"{context}\n\nThis is an HIV/GUM clinic letter. Summarise: HIV status, CD4/viral load, "
                       "ART regimen changes, comorbidities addressed, and follow-up plan.")
    elif "Maternity" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a maternity/diabetes letter. Summarise: gestational diabetes status, "
                       "OGTT results, monitoring plan, equipment prescribed, and GP actions needed.")
    elif "Psychiatric" in letter_type or "Psychiatry" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a psychiatry outpatient letter. Summarise: diagnoses (with ICD codes), "
                       "current medications and recent changes, clinical progress, and next review.")
    elif "Procedure" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a procedure/endoscopy report. Summarise: indication, key findings, "
                       "impression, biopsy/sampling if done, and recommendations.")
    elif "Surgical" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a pre-operative surgical outpatient letter. Summarise: diagnosis, "
                       "planned procedure, risks discussed, and GP actions required.")
    elif "Mental Health Inpatient" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a mental health inpatient discharge summary. Summarise: "
                       "admission circumstances (including any MHA section), primary diagnosis, "
                       "clinical progress on ward, medications at discharge, and community follow-up plan "
                       "(CRHTT/CMHT). Note any medication monitoring requirements (e.g. lithium levels).")
    elif "Ophthalmology Referral" in letter_type:
        clin_prompt = (f"{context}\n\nThis is an ophthalmology referral letter (via Evolutio/eRefer). Summarise: "
                       "referral reason and pathway, priority (routine/urgent), optometrist findings, "
                       "visual acuity and IOP if recorded, and which provider the patient is being referred to.")
    elif "Ophthalmology" in letter_type:
        clin_prompt = (f"{context}\n\nThis is an ophthalmology outpatient/medical retina clinic letter. Summarise: "
                       "diagnosis (with retinopathy grading e.g. R2M1P0), key findings per eye (VA, IOP, fundoscopy), "
                       "treatment given or planned (PRP, laser, injection), and follow-up interval. "
                       "Note any urgent actions required.")
    elif "Renal" in letter_type or "Nephrology" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a renal/nephrology remote monitoring letter. Summarise: "
                       "current kidney function (eGFR, creatinine, albumin), trends vs previous, "
                       "any treatment changes required, and next review/test date.")
    elif "Paediatric Cardiology" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a paediatric cardiology outpatient letter. Summarise: "
                       "cardiac diagnosis, current symptoms on/off medication, medication changes, "
                       "planned investigations or procedures (e.g. ablation, EP MDT), and follow-up plan.")
    elif "Early Pregnancy" in letter_type or "Gynaecology" in letter_type:
        clin_prompt = (f"{context}\n\nThis is an early pregnancy / gynaecology outpatient letter. Summarise: "
                       "presenting complaint, scan findings (gestational sac, yolk sac, fetal pole), "
                       "gestational age estimate, diagnosis, and next steps (e.g. repeat EPAU scan).")
    elif "Antenatal Discharge" in letter_type:
        clin_prompt = (f"{context}\n\nThis is an antenatal discharge summary. Summarise: reason for admission, "
                       "EDD and gestational age, G/P status, key clinical findings, any complications, "
                       "and community midwife follow-up instructions.")
    elif "Pre-admission" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a pre-admission/surgical booking letter. Summarise: "
                       "scheduled procedure, date, speciality, clinician, and key pre-operative instructions "
                       "for the patient (fasting, medication, transport).")
    elif "Discharge" in letter_type:
        clin_prompt = (f"{context}\n\nThis is a discharge summary. Summarise: admission reason, diagnosis, "
                       "procedures performed, discharge condition, and follow-up required.")
    else:
        clin_prompt = (f"{context}\n\nWrite a concise clinical summary (3-5 sentences) for the treating clinician. "
                       "Include key findings, diagnosis, treatment plan, and follow-up. Be precise and medical.")

    clinician_summary = call_claude(clin_prompt)

    # OBS-007: Add sensitivity clause when safeguarding/bereavement markers detected.
    # Prevents re-traumatising patients by paraphrasing rather than quoting verbatim.
    sensitivity_clause = (
        " IMPORTANT: This document contains sensitive content (safeguarding, bereavement, or mental health crisis). "
        "Do NOT quote any distressing details verbatim. Use supportive, neutral language. "
        "Focus only on what the patient needs to do next."
    ) if is_sensitive else ""

    patient_summary = call_claude(
        f"{context}\n\nWrite a clear patient-friendly explanation (3-4 sentences) of what was found and what happens next. "
        "Avoid medical jargon. Use plain English. Start with the most important thing the patient needs to know."
        + sensitivity_clause
    )

    pharmacist_summary = call_claude(
        f"{context}\n\nWrite a pharmacist-focused clinical summary. Include: all medications mentioned (with doses/frequencies), "
        "any new prescriptions, drug monitoring requirements, potential interactions to check, and any OTC advice given. "
        "If no medications are documented, state this clearly."
    )

    actions_raw = call_claude(
        f"{context}\n\nList 3-5 specific actionable follow-up tasks (numbered list). "
        "For each, state who is responsible (GP / patient / specialist / pharmacist / nurse). Be specific and clinical."
    )

    gp_actions = call_claude(
        f"{context}\n\nWhat specific actions does the GP need to take based on this document? "
        "List up to 4 numbered items. Include prescriptions, referrals, monitoring, and safety-netting."
    )

    llm_conf = 0.80
    return {
        "clinician":  {"summary": clinician_summary, "confidence": llm_conf},
        "patient":    {"summary": patient_summary,   "confidence": llm_conf},
        "pharmacist": {"summary": pharmacist_summary, "confidence": llm_conf},
        "follow_up_actions": actions_raw,
        "gp_actions":        gp_actions,
        "llm_confidence":    llm_conf,
    }


def compute_unified_confidence(
    textract_conf: float,
    snomed_conf: float,
    llm_conf: float,
    letter_type: str = "",
) -> float:
    """Weighted unified confidence score (SRS Section 3.4).

    Default weights: Textract 40% + LLM 40% + SNOMED 20%.

    For Ambulance and 111 reports, SNOMED entities are typically absent because
    the text is in all-caps multi-column tables that Comprehend Medical cannot
    parse reliably. For these types we use Textract 50% + LLM 50% and ignore
    SNOMED, preventing false 'review required' flags.
    """
    if letter_type in ("Ambulance Clinical Report", "111 First ED Report"):
        # SNOMED unreliable for all-caps/table-heavy formats — use Textract + LLM only
        return (0.50 * textract_conf) + (0.50 * llm_conf)
    return (0.40 * textract_conf) + (0.20 * snomed_conf) + (0.40 * llm_conf)


def get_confidence_threshold(letter_type: str) -> float:
    """OBS-004: Return per-type confidence threshold from config/document_type_config.py.
    Single source of truth — thresholds are not duplicated here.
    """
    return _get_threshold(letter_type)


def extract_hospital_trust(text: str) -> str:
    """OBS-008: Identify the originating hospital trust from document header text.
    Enables routing, audit logging, and trust-specific formatting rules.
    """
    t = text[:800].lower()  # trust name always in first page header
    if any(x in t for x in ["frimley health", "frimley park hospital", "wexham park", "heatherwood"]):
        return "Frimley Health NHS Foundation Trust"
    if any(x in t for x in ["royal berkshire", "rbh", "london road, reading"]):
        return "Royal Berkshire Hospital NHS Foundation Trust"
    if any(x in t for x in ["berkshire healthcare", "prospect park", "talking therapies"]):
        return "Berkshire Healthcare NHS Foundation Trust"
    if any(x in t for x in ["south central ambulance", "scas"]):
        return "South Central Ambulance Service NHS Foundation Trust"
    if any(x in t for x in ["university hospital southampton", "uhs", "tremona road"]):
        return "University Hospital Southampton NHS Foundation Trust"
    if any(x in t for x in ["kettering", "rothwell road"]):
        return "Kettering General Hospital NHS Foundation Trust"
    if any(x in t for x in ["evolutio", "odtc.co.uk", "newtown house, newtown road"]):
        return "Evolutio Care Innovations Ltd"
    if any(x in t for x in ["expert health", "expertHealth", "dr. mitra dutt"]):
        return "Expert Health Ltd"
    return "Unknown Trust"


def contains_sensitive_content(text: str) -> bool:
    """OBS-007: Detect if document contains sensitive/safeguarding content.
    Used to add protective instructions to patient-facing Bedrock prompts.
    """
    t = text.lower()
    return any(marker in t for marker in SENSITIVE_CONTENT_MARKERS)


def resolve_arrival_method(text: str) -> str:
    """OBS-010: Decode Frimley arrival method codes e.g. '[8]' -> 'Emergency Road Ambulance WITH Medical Escort'."""
    import re
    m = re.search(r'\[(\d+)\]', text)
    if m:
        code = m.group(1)
        return ARRIVAL_METHOD_CODES.get(code, f"Code {code}")
    return text


def infer_letter_type(text: str) -> str:
    """Classify letter type based on all 21 observed document patterns (batches 1-5).
    Priority order: most specific/distinct signals first to prevent false matches.
    """
    t = text.lower()
    # ── Highest specificity first ────────────────────────────────────────────
    # SCAS Ambulance Clinical Reports — very distinct vocabulary
    if any(x in t for x in ["south central ambulance service", "patient clinical report",
                              "gp patient report v3", "scas clinician", "news2 score",
                              "pops score", "nature of call", "incident number",
                              "conveyance", "at patient side"]):
        return "Ambulance Clinical Report"
    # Ophthalmology Referral — Evolutio / eRefer (must come before generic referral catch)
    if any(x in t for x in ["evolutio ophthalmology", "evolutio care innovations",
                              "patient ophthalmology referral", "east berkshire community eye service",
                              "erefer referral", "referral id number", "triager action required",
                              "odtc.co.uk", "epiretinal membrane", "specsavers"]):
        return "Ophthalmology Referral"
    # Ophthalmology Outpatient / Medical Retina (must come before generic ophthalmology)
    if any(x in t for x in ["diabetic retinopathy", "medical retina",
                              "proliferative retinopathy", "macular oedema",
                              "intraocular pressure", "fundus exam", "prp", "panretinal",
                              "neovascularisation", "nvd", "nve", "slit lamp"]):
        return "Ophthalmology Letter"
    # Expert Health / GLP-1 prescribing — very distinct brand names
    if any(x in t for x in ["expert health", "notification of consultation", "kwikpen",
                              "weight management", "glp-1", "mounjaro", "semaglutide",
                              "ozempic", "wegovy", "weight loss programme"]):
        return "Medication / Prescriber Letter"
    # ED Discharge Letters — emergency dept specific, before generic discharge
    if any(x in t for x in ["frimley emergency", "patient discharge letter",
                              "attendance reason", "arrival method", "source of referral",
                              "mode of arrival", "presenting complaint:", "place of accident"]):
        return "ED Discharge Letter"
    # 111 First ED Reports
    if any(x in t for x in ["111 first ed report", "nhs111 encounter", "pathways disposition",
                              "pathways assessment", "attendance activity", "111 first"]):
        return "111 First ED Report"
    # Mental Health Inpatient Discharge — check BEFORE generic discharge summary
    if any(x in t for x in ["mental health inpatient discharge", "prospect park hospital",
                              "crhtt", "cmht", "snowdrop ward", "section 2", "section 3",
                              "mental health act", "inpatient consultant"]):
        return "Mental Health Inpatient Discharge"
    # Antenatal Discharge Summary — check BEFORE generic maternity
    if any(x in t for x in ["antenatal discharge", "estimate delivery date", "estimate gestational age",
                              "gravida & parity", "reduced fetal movement", "mdau",
                              "antenatal discharge summary"]):
        return "Antenatal Discharge Summary"
    # Cancer surveillance
    if any(x in t for x in ["surveillance", "adenocarcinoma", "hemicolectomy", "colorectal surveillance",
                              "tnm", "cea", "chemotherapy", "oncology"]):
        return "Cancer Surveillance Letter"
    # HIV / GUM / Sexual health
    if any(x in t for x in ["hiv", "gum clinic", "garden clinic", "sexual health", "antiretroviral",
                              "cd4", "viral load", "art regimen", "dolutegravir", "tenofovir"]):
        return "HIV / GUM Clinic Letter"
    # Maternity / diabetes
    if any(x in t for x in ["gestational diabetes", "antenatal", "maternity", "glucose tolerance",
                              "pip code", "blood glucose monitoring", "midwives"]):
        return "Maternity / Diabetes Letter"
    # Pre-op surgical outpatient
    if any(x in t for x in ["hernia", "supra-umbilical", "upper gi", "open repair", "mesh repair",
                              "brachioplasty", "pre-op", "pre op", "surgical consent"]):
        return "Surgical Outpatient Letter"
    # Procedure / endoscopy reports
    if any(x in t for x in ["endoscopy", "ogd", "colonoscopy", "gastroscopy", "oesophageal",
                              "colonography", "procedure report", "endoscopist"]):
        return "Procedure Report"
    # CAMHS / paediatric mental health
    if any(x in t for x in ["camhs", "child and adolescent", "mental health service",
                              "brief psychosocial intervention", "bpi"]):
        return "CAMHS Discharge Summary"
    # Generic discharge summaries
    if any(x in t for x in ["discharge summary", "discharge date", "discharging consultant",
                              "length of stay", "discharge summary completed by"]):
        return "Discharge Summary"
    # Psychiatry outpatient
    if any(x in t for x in ["psychiatrist", "psychiatric", "bipolar", "icd10", "icd-10",
                              "quetiapine", "lisdexamfetamine", "consultant psychiatrist"]):
        return "Psychiatry Outpatient Letter"
    # Renal / Nephrology Letter (prefix 6.)
    if any(x in t for x in ["nephrologist", "nephrology", "berkshire kidney",
                              "egfr", "creatinine", "renal medicine", "albumin creatinine ratio",
                              "remote monitoring team", "kidney unit"]):
        return "Renal / Nephrology Letter"
    # Paediatric Cardiology (prefix 6.)
    if any(x in t for x in ["paediatric cardiol", "paediatric and fetal cardiologist",
                              "congenital heart", "ep mdt", "ablation", "svt",
                              "supraventricular tachycardia", "accessory pathway", "atenolol"]):
        return "Paediatric Cardiology Letter"
    # Early Pregnancy / Gynaecology (prefix 7.) — check before maternity
    if any(x in t for x in ["ugcc", "epau", "early pregnancy", "gestational sac",
                              "transvaginal", "intrauterine pregnancy", "gravida",
                              "uncertain viability", "emergency gynaecology"]):
        return "Early Pregnancy / Gynaecology Letter"
    # Pre-admission Booking Letter (prefix 7.)
    if any(x in t for x in ["fasting instructions", "hospital admission has been scheduled",
                              "do not eat after", "admission instructions", "day surgery unit",
                              "bring this letter with you"]):
        return "Pre-admission Letter"
    # Haematology / oncology outpatient (prefix 10.)
    # (Antenatal Discharge Summary check is earlier — before Maternity/Diabetes)
    if any(x in t for x in ["haematology", "myeloma", "multiple myeloma", "lenalidomide",
                              "bortezomib", "protein electrophoresis", "paraprotein"]):
        return "Haematology Outpatient Letter"
    # Weight management / prescriber (prefix 10.) — Expert Health / GLP-1
    if any(x in t for x in ["weight management", "glp-1", "mounjaro", "semaglutide",
                              "ozempic", "wegovy", "weight loss programme",
                              "expert health", "notification of consultation", "kwikpen"]):
        return "Medication / Prescriber Letter"
    # Antibiotic / medication requests
    if any(x in t for x in ["antibiotic request", "medication request", "repeat prescription",
                              "flucloxacillin", "prescrib"]):
        return "Medication Request"
    # Referral letters
    if any(x in t for x in ["referral", "i am referring", "reason for referral",
                              "please see this patient"]):
        return "Referral Letter"
    # Outpatient follow-up
    if any(x in t for x in ["outpatient", "follow-up", "follow up", "clinic visit",
                              "appointment type", "clinic note"]):
        return "Outpatient Letter"
    return "Clinical Letter"


def extract_icd_codes(text: str) -> list:
    """Extract ICD-9/10 codes like K64.9, F31.0, F90.0, ICD10 F31.0."""
    import re
    pattern = r'\b([A-Z]\d{2}(?:\.\d{1,2})?)\b'
    codes   = list(dict.fromkeys(re.findall(pattern, text)))
    # filter noise — must start with a valid ICD chapter letter
    valid_starts = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return [c for c in codes if c[0] in valid_starts and len(c) >= 3]


def extract_medications(text: str) -> list:
    """Extract medication lines with dosage patterns."""
    import re
    meds  = []
    lines = text.split("\n")
    # Patterns: "Drug Name Xmg/Xml route frequency"
    dose_re = re.compile(
        r'(\b[A-Z][a-zA-Z\s\-]+?)\s+'
        r'(\d+\.?\d*\s*(?:mg|ml|mcg|iu|g|mg/ml|ml/hr|units?|%|micrograms?)[^\n,;]{0,40})',
        re.IGNORECASE
    )
    for line in lines:
        line = line.strip()
        if not line or len(line) < 8:
            continue
        m = dose_re.search(line)
        if m:
            name = m.group(1).strip().rstrip('-– ')
            dose = m.group(2).strip()
            if 3 < len(name) < 60:
                meds.append({"name": name, "dose": dose, "raw": line.strip()})
    # deduplicate by name
    seen, out = set(), []
    for med in meds:
        key = med["name"].lower()
        if key not in seen:
            seen.add(key)
            out.append(med)
    return out[:20]


def extract_structured_fields(text: str) -> dict:
    """Extract structured fields common across all 8 document types."""
    import re
    fields = {
        "admission_date": "", "discharge_date": "", "appointment_date": "",
        "consultant": "", "department": "", "hospital": "",
        "gp_actions": "", "diagnosis_text": "",
        "admission_method": "", "discharge_method": "",
        "procedure": "", "indication": "", "impression": "",
    }
    lines = text.split("\n")
    for i, line in enumerate(lines):
        l = line.strip()
        ll = l.lower()
        # Dates
        if re.search(r'(?i)admission date', ll):
            m = re.search(r'(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})', l)
            if m: fields["admission_date"] = m.group(1)
        if re.search(r'(?i)discharge date', ll):
            m = re.search(r'(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})', l)
            if m: fields["discharge_date"] = m.group(1)
        if re.search(r'(?i)appointment.?date', ll):
            m = re.search(r'(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})', l)
            if m: fields["appointment_date"] = m.group(1)
        # Consultant
        if re.search(r'(?i)discharging consultant|consultant[:\s]|lead professional', ll):
            m = re.search(r'(?:consultant|lead professional)[:\s]+([A-Z][^\n,]{3,50})', l, re.IGNORECASE)
            if m and not fields["consultant"]: fields["consultant"] = m.group(1).strip()
        # Department
        if re.search(r'(?i)discharging specialty|department[:\s]|specialty[:\s]', ll):
            m = re.search(r'(?:specialty|department)[:\s]+([A-Za-z][^\n]{3,40})', l, re.IGNORECASE)
            if m and not fields["department"]: fields["department"] = m.group(1).strip()
        # Admission/discharge method
        if re.search(r'(?i)admission method', ll):
            m = re.search(r'admission method[:\s]+([^\n]{3,60})', l, re.IGNORECASE)
            if m: fields["admission_method"] = m.group(1).strip()
        if re.search(r'(?i)discharge method', ll):
            m = re.search(r'discharge method[:\s]+([^\n]{3,60})', l, re.IGNORECASE)
            if m: fields["discharge_method"] = m.group(1).strip()
        # Procedure
        if re.search(r'(?i)procedure[:\(]|procedure\s*date', ll):
            nxt = lines[i+1].strip() if i+1 < len(lines) else ""
            if nxt and len(nxt) > 3: fields["procedure"] = nxt[:120]
        # Indication (endoscopy reports)
        if re.search(r'(?i)^indication', ll):
            nxt = lines[i+1].strip() if i+1 < len(lines) else ""
            if nxt: fields["indication"] = nxt[:200]
        # Impression
        if re.search(r'(?i)^overall impression|^impression', ll):
            nxt = lines[i+1].strip() if i+1 < len(lines) else ""
            if nxt: fields["impression"] = nxt[:300]
        # GP Actions
        if re.search(r'(?i)actions.*(gp|general practice)|gp.actions', ll):
            nxt = (lines[i+1].strip() if i+1 < len(lines) else "") or l
            fields["gp_actions"] = nxt[:300]
        # Diagnosis text
        if re.search(r'(?i)^diagnosis|^post.op diagnosis', ll):
            nxt = lines[i+1].strip() if i+1 < len(lines) else ""
            if nxt and not fields["diagnosis_text"]: fields["diagnosis_text"] = nxt[:200]
    return fields


def extract_patient_info(text: str) -> dict:
    """Extract patient demographics — handles all document formats seen in batches 1-4."""
    import re
    info = {
        "name": "", "nhs_number": "", "dob": "", "sex": "",
        "address": "", "hospital_number": "", "gp_practice": "",
        "pathways_urgency": "", "presenting_complaint": "",
        "gravida_parity": "", "edd": "", "gestational_age": "",
    }
    lines = text.split("\n")

    for i, line in enumerate(lines):
        l = line.strip()
        ll = l.lower()

        # ── Name ──────────────────────────────────────────────────────────────
        if not info["name"]:
            # Standard: "Re: SURNAME, Forename"
            m = re.search(r'(?i)(?:RE:|RE patient:|Patient(?:\s+Name)?:|Patient Surname.*?:)\s*(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][A-Za-z,\s\-]{2,50})', l)
            if m:
                info["name"] = re.sub(r'\s+', ' ', m.group(1).strip().rstrip(','))
            # 111 format: standalone "SURNAME, Forename" on its own line after DOB line
            elif re.match(r'^[A-Z]{2,}[,\s]+[A-Z][a-z]', l) and not any(x in ll for x in ['nhs', 'hospital', 'road', 'street', 'lane', 'avenue', 'drive']):
                if len(l.split()) <= 4:
                    info["name"] = l

        # ── NHS number ────────────────────────────────────────────────────────
        if not info["nhs_number"]:
            m = re.search(r'(?i)NHS\s*(?:No|Number|#|:)?[:\s]*(\d[\d\s]{8,12}\d)', l)
            if m:
                info["nhs_number"] = re.sub(r'\s+', ' ', m.group(1).strip())
            else:
                # 111 format: "NHS Number\n462 213 3695" (next line)
                if re.search(r'(?i)^NHS\s*Number\s*$', l) and i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if re.match(r'^[\d\s]{9,12}$', nxt):
                        info["nhs_number"] = nxt.strip()

        # ── DOB ───────────────────────────────────────────────────────────────
        if not info["dob"]:
            # Standard: "DOB: 16/3/1975" or "Date of birth: 17.06.1987"
            m = re.search(r'(?i)(?:DOB|Date of birth)[:\s]+(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})', l)
            if m:
                info["dob"] = m.group(1).strip()
            else:
                # 111 format: "Born 22-Feb-1996" or "Born: 22-Feb-1996"
                m = re.search(r'(?i)\bBorn[:\s]+(\d{1,2}[\-\/]\w+[\-\/]\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', l)
                if m: info["dob"] = m.group(1).strip()

        # ── Sex / Gender ──────────────────────────────────────────────────────
        if not info["sex"]:
            m = re.search(r'(?i)(?:Gender|Sex|Legal Sex)[:\s,]+(Male|Female|M\b|F\b)', l)
            if m:
                g = m.group(1).upper()
                info["sex"] = "M" if g.startswith("M") else "F"
            elif re.search(r'\bGender:\s*Female\b|\bfemale\b', l, re.IGNORECASE): info["sex"] = "F"
            elif re.search(r'\bGender:\s*Male\b',   l, re.IGNORECASE):            info["sex"] = "M"

        # ── Hospital / MRN / PAS ──────────────────────────────────────────────
        if not info["hospital_number"]:
            m = re.search(r'(?i)(?:MRN|Hospital\s*(?:No|Number)|PAS\s*ID|MRN:)[:\s]+([A-Z0-9]+)', l)
            if m: info["hospital_number"] = m.group(1).strip()

        # ── GP Practice ───────────────────────────────────────────────────────
        if not info["gp_practice"]:
            m = re.search(r'(?i)(?:GP\s*Practice|Surgery)[:\s]+([A-Za-z][^\n]{3,50})', l)
            if m: info["gp_practice"] = m.group(1).strip()

        # ── 111 Pathways urgency ──────────────────────────────────────────────
        if not info["pathways_urgency"] and re.search(r'(?i)pathways disposition|refer to.*within', ll):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            info["pathways_urgency"] = (nxt or l)[:120]

        # ── Presenting complaint ──────────────────────────────────────────────
        if not info["presenting_complaint"]:
            m = re.search(r'(?i)(?:Complaint|Reason for (?:contact|referral|admission))[:\s]+([^\n]{5,150})', l)
            if m: info["presenting_complaint"] = m.group(1).strip()

        # ── Gravida / Parity (antenatal / gynae) ──────────────────────────────
        if not info["gravida_parity"]:
            m = re.search(r'\b(G\s*\d+\s*P\s*\d+)\b', l)
            if m: info["gravida_parity"] = m.group(1).replace(" ", "")

        # ── EDD (Estimated Delivery Date) ─────────────────────────────────────
        # Skip any parenthetical abbreviation e.g. "(EDD)" before the actual date
        if not info["edd"]:
            m = re.search(r'(?i)(?:EDD|Estimated?\s+Delivery\s+Date)(?:\s*\([^)]*\))?\s*[:\s]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', l)
            if not m:  # date may be on next line — try broader capture
                m = re.search(r'(?i)(?:EDD|Estimated?\s+Delivery\s+Date)(?:\s*\([^)]*\))?\s*[:\s]+([^\n(]{3,20})', l)
            if m:
                val = m.group(1).strip().rstrip(')')
                if val and not val.lower().startswith('('):
                    info["edd"] = val

        # ── EGA (Estimated Gestational Age) ───────────────────────────────────
        if not info["gestational_age"]:
            m = re.search(r'(?i)(?:EGA|Estimated?\s+Gestational\s+Age|Gestational\s+Age)(?:\s*\([^)]*\))?\s*[:\s]+(\d[^\n(]{2,20})', l)
            if m:
                val = m.group(1).strip().rstrip(')')
                if val:
                    info["gestational_age"] = val

        # ── Expert Health format: name/DOB in letter body "Re: MR/MISS Name" ─
        if not info["name"]:
            m = re.search(r'(?i)^Re:\s*(?:MR|MRS|MISS|MS|DR)\.?\s+([A-Za-z][^\n]{3,50})', l)
            if m:
                raw = m.group(1).strip()
                # Next few lines have address, DOB may be 3rd line pattern
                info["name"] = raw

    return info


def extract_clinical_specifics(text: str, letter_type: str) -> dict:
    """
    Extract type-specific clinical data for the supported document/letter types
    identified by `letter_type` (e.g. from infer_letter_type()).
    Returns a dict of extra fields shown in the right panel and Coding tab.
    Supported types are defined in config/document_type_config.py.
    """
    import re
    extras = {}
    t  = text
    tl = t.lower()

    # ── 111 First ED Report ────────────────────────────────────────────────────
    if "111" in letter_type:
        # Differential diagnosis (marked with ??)
        diffs = re.findall(r'\?\??\s*([A-Za-z][^\n\?]{3,60})', t)
        if diffs: extras["differential_diagnosis"] = " / ".join(d.strip() for d in diffs[:5])
        # Pathways urgency
        m = re.search(r'(?i)refer to a treatment centre within\s+([^\n]+)', t)
        if m: extras["urgency"] = m.group(1).strip()
        # Encounter type
        m = re.search(r'(?i)Encounter Type\s+([^\n]+)', t)
        if m: extras["encounter_type"] = m.group(1).strip()
        # Doctor name
        m = re.search(r'(?i)Clinical Summary by (?:DOCTOR|DR\.?)\s+([^\n]+)', t)
        if m: extras["assessing_clinician"] = m.group(1).strip()

    # ── Cancer Surveillance ────────────────────────────────────────────────────
    if "Cancer" in letter_type or "Surveillance" in letter_type:
        # TNM staging
        m = re.search(r'(p?T\d+\s*N\d+[^\n]{0,30})', t)
        if m: extras["tnm_staging"] = m.group(1).strip()
        # CEA
        m = re.search(r'(?i)CEA[:\s]+([\d\.]+)', t)
        if m: extras["cea_value"] = m.group(1)
        # Surveillance schedule
        m = re.search(r'(?i)surveillance[:\s]+([^\n]{10,120})', t)
        if m: extras["surveillance_schedule"] = m.group(1).strip()
        # Treatment history
        m = re.search(r'(?i)(hemicolectomy|colectomy|chemotherapy|radiotherapy)[^\n]{0,80}', t)
        if m: extras["treatment_history"] = m.group(0).strip()

    # ── HIV / GUM ─────────────────────────────────────────────────────────────
    if "HIV" in letter_type or "GUM" in letter_type:
        # CD4
        m = re.search(r'(?i)CD4[/\s]?(?:count)?[:\s]+([\d,]+\s*cells?/m[cμ]?[Ll]?)', t)
        if m: extras["cd4_count"] = m.group(1).strip()
        # Viral load
        m = re.search(r'(?i)(?:HIV\s+)?viral\s+load[:\s]+([^\n]{3,40})', t)
        if m: extras["viral_load"] = m.group(1).strip()
        # ART regimen
        m = re.search(r'(?i)(?:antiretroviral|ART)\s+medication[^\n]{0,20}\n([^\n]{5,120})', t)
        if m: extras["art_regimen"] = m.group(1).strip()
        # Follow up
        m = re.search(r'(?i)follow[- ]?up[:\s]+([^\n]{5,100})', t)
        if m: extras["follow_up"] = m.group(1).strip()

    # ── Maternity / Diabetes ──────────────────────────────────────────────────
    if "Maternity" in letter_type or "Diabetes" in letter_type:
        # OGTT results
        m = re.search(r'(?i)(?:glucose tolerance|ogtt)[^\n]*\n?[^\n]*0\s*mins?\s*=\s*([\d\.]+)[^\n]*120\s*mins?\s*=\s*([\d\.]+)', t, re.DOTALL)
        if m: extras["ogtt_results"] = f"0min={m.group(1)} / 120min={m.group(2)}"
        # Monitoring frequency
        m = re.search(r'(?i)test[^\n]*(\d+)\s*times?\s*per\s*day', t)
        if m: extras["monitoring_frequency"] = f"{m.group(1)} times/day"
        # Equipment prescribed
        pips = re.findall(r'([A-Za-z][^\n]{5,60}PIP\s*Code[:\s]+([\d\-]+))', t)
        if pips: extras["equipment_pip"] = "; ".join(f"{p[0].split('PIP')[0].strip()} ({p[1]})" for p in pips[:3])

    # ── Surgical Outpatient ───────────────────────────────────────────────────
    if "Surgical" in letter_type:
        m = re.search(r'(?i)(?:^|\n)Plan[:\s]+([^\n]{5,200})', t)
        if m: extras["surgical_plan"] = m.group(1).strip()
        m = re.search(r'(?i)(?:^|\n)Action for\s*GP[:\s]+([^\n]{3,100})', t)
        if m: extras["action_for_gp"] = m.group(1).strip()

    # ── Haematology ───────────────────────────────────────────────────────────
    if "Haematology" in letter_type:
        # Lab results table
        labs = re.findall(r'(HGB|WBC|PLT|CREATININE|HB|HbA1c|eGFR)[:\s]+([\d\.]+)', t, re.IGNORECASE)
        if labs: extras["key_labs"] = {k.upper(): v for k, v in labs}
        m = re.search(r'(?i)paraprotein[^\n]{0,60}', t)
        if m: extras["paraprotein"] = m.group(0).strip()

    # ── Ophthalmology Referral (Evolutio / eRefer) ───────────────────────────
    if "Ophthalmology Referral" in letter_type:
        m = re.search(r'(?i)referral reason[:\s]+([^\n]{5,100})', t)
        if m: extras["referral_reason"] = m.group(1).strip()
        m = re.search(r'(?i)pathway\s*/?\s*clinic[:\s]+([^\n]{5,100})', t)
        if m: extras["referral_pathway"] = m.group(1).strip()
        m = re.search(r'(?i)(?:triager|referer) action required[:\s]+([^\n]{3,30})', t)
        if m: extras["priority"] = m.group(1).strip()
        m = re.search(r'(?i)patient chosen provider[:\s]+([^\n]{3,80})', t)
        if m: extras["provider"] = m.group(1).strip()
        m = re.search(r'(?i)referred by[:\s]+([^\n]{5,80})', t)
        if m: extras["referred_by"] = m.group(1).strip()
        # Visual acuity
        m = re.search(r'(?i)visual acuity\s*R[:\s]+([^\s]+)\s+L[:\s]+([^\s\n]+)', t)
        if m: extras["visual_acuity"] = f"R: {m.group(1)}  L: {m.group(2)}"
        # IOP
        m = re.search(r'(?i)right iop[^\d]*([\d\.]+)[^\d]*left iop[^\d]*([\d\.]+)', t)
        if m: extras["iop"] = f"R: {m.group(1)} mmHg  L: {m.group(2)} mmHg"

    # ── Ophthalmology Outpatient / Medical Retina ─────────────────────────────
    if "Ophthalmology Letter" in letter_type:
        # Retinopathy grading (R2M1P0 style)
        grades = re.findall(r'\b(R\d+[AM]?\s*M\d+\s*P\d+)\b', t)
        if grades: extras["retinopathy_grade"] = " / ".join(dict.fromkeys(grades))
        # Visual acuity per eye
        m = re.search(r'(?i)right\s+([\d/\.]+)[^\n]{0,20}left\s+([\d/\.]+)', t)
        if m: extras["visual_acuity"] = f"R: {m.group(1)}  L: {m.group(2)}"
        # IOP
        m = re.search(r'(?i)(?:right|R)\s+([\d]+)\s*mmhg[^\n]{0,10}(?:left|L)\s+([\d]+)\s*mmhg', t, re.IGNORECASE)
        if m: extras["iop"] = f"R: {m.group(1)} mmHg  L: {m.group(2)} mmHg"
        # Diagnosis
        m = re.search(r'(?i)diagnosis[:\s]+([^\n]{5,120})', t)
        if m: extras["ophthalmic_diagnosis"] = m.group(1).strip()
        # PRP / laser
        m = re.search(r'(?i)(prp|panretinal|retinal laser)[^\n]{0,100}', t)
        if m: extras["laser_treatment"] = m.group(0).strip()
        # Plan
        m = re.search(r'(?i)^plan[:\s]+([^\n]{5,200})', t, re.MULTILINE)
        if m: extras["ophthalmic_plan"] = m.group(1).strip()
        # NVD/NVE
        if re.search(r'(?i)(nvd|nvealisation|neovascularisation)', t):
            extras["neovascularisation"] = "Detected"

    # ── Renal / Nephrology ────────────────────────────────────────────────────
    if "Renal" in letter_type or "Nephrology" in letter_type:
        # Inline lab panel (format: "eGFR\n23" or "eGFR 23")
        lab_keys = ["egfr", "creatinine", "albumin", "potassium", "haemoglobin", "urea",
                    "bicarbonate", "pth intact", "albumin creatinine ratio", "white blood cell"]
        labs = {}
        lines = t.split("\n")
        for i, line in enumerate(lines):
            ll = line.lower().strip()
            for k in lab_keys:
                if k in ll:
                    # value may be on same line or next
                    m = re.search(r'([\d\.]+)', line)
                    if not m and i + 1 < len(lines):
                        m = re.search(r'([\d\.]+)', lines[i + 1])
                    if m:
                        labs[k.replace(" ", "_")] = m.group(1)
        if labs: extras["renal_labs"] = labs
        m = re.search(r'(?i)review.*?week beginning\s+([^\n\.]{5,30})', t)
        if m: extras["next_review"] = m.group(1).strip()

    # ── Paediatric Cardiology ─────────────────────────────────────────────────
    if "Paediatric Cardiology" in letter_type:
        m = re.search(r'(?i)diagnosis[:\s]+([^\n]{5,120})', t)
        if m: extras["cardiac_diagnosis"] = m.group(1).strip()
        m = re.search(r'(?i)(?:heart rate|bpm|beats per minute)[^\n]{0,60}(\d{3})', t)
        if m: extras["max_heart_rate"] = m.group(1) + " bpm"
        m = re.search(r'(?i)(?:ablation|ep mdt|electrophysiology)[^\n]{0,100}', t)
        if m: extras["planned_procedure"] = m.group(0).strip()
        m = re.search(r'(?i)medication[:\s]+([^\n]{5,100})', t)
        if m: extras["current_medication"] = m.group(1).strip()

    # ── Early Pregnancy / Gynaecology ─────────────────────────────────────────
    if "Pregnancy" in letter_type or "Gynaecology" in letter_type:
        m = re.search(r'(?i)(G\s*\d+\s*P\s*\d+)', t)
        if m: extras["gravida_parity"] = m.group(1).replace(" ","")
        m = re.search(r'(?i)LMP[:\s]+([^\n]{3,30})', t)
        if m: extras["lmp"] = m.group(1).strip()
        m = re.search(r'(?i)(?:mean sac diameter|gestational sac)[^\n]{0,30}([\d\.]+\s*mm)', t)
        if m: extras["gestational_sac"] = m.group(1).strip()
        m = re.search(r'(?i)(?:fetal pole)[:\s]+([^\n]{3,60})', t)
        if m: extras["fetal_pole"] = m.group(1).strip()
        m = re.search(r'(?i)diagnosis[:\s]+([^\n]{5,120})', t)
        if m: extras["scan_diagnosis"] = m.group(1).strip()
        m = re.search(r'(?i)plan[:\s]+([^\n]{5,200})', t)
        if m: extras["follow_up_plan"] = m.group(1).strip()

    # ── Antenatal Discharge Summary ───────────────────────────────────────────
    if "Antenatal" in letter_type:
        # EDD — skip any "(EDD)" parenthetical, capture the actual date value
        m = re.search(r'(?i)(?:EDD|Estimated?\s+Delivery\s+Date)(?:\s*\([^)]*\))?\s*[:\s]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', t)
        if not m:
            m = re.search(r'(?i)(?:EDD|Estimated?\s+Delivery\s+Date)(?:\s*\([^)]*\))?\s*[:\s]+([^\n(]{3,20})', t)
        if m:
            val = m.group(1).strip().rstrip(')')
            if val and not val.lower().startswith('('):
                extras["edd"] = val
        # EGA — skip "(EGA)" parenthetical, capture weeks+days e.g. "29+1 weeks"
        m = re.search(r'(?i)(?:EGA|Estimated?\s+Gestational\s+Age)(?:\s*\([^)]*\))?\s*[:\s]+(\d[^\n(]{2,20})', t)
        if m:
            val = m.group(1).strip().rstrip(')')
            if val:
                extras["gestational_age"] = val
        m = re.search(r'(?i)Gravida\s*&?\s*Parity[:\s]+([^\n]{2,10})', t)
        if m: extras["gravida_parity"] = m.group(1).strip()
        m = re.search(r'(?i)reason for (?:visit|admission)[:\s]+([^\n]{5,150})', t)
        if m: extras["reason_for_visit"] = m.group(1).strip()

    # ── Mental Health Inpatient Discharge ─────────────────────────────────────
    if "Mental Health Inpatient" in letter_type:
        m = re.search(r'(?i)(?:section\s*\d+|legal status)[^\n]{0,60}', t)
        if m: extras["mha_section"] = m.group(0).strip()
        m = re.search(r'(?i)diagnosis[:\s]+([^\n]{5,120})', t)
        if m: extras["primary_diagnosis"] = m.group(1).strip()
        m = re.search(r'(?i)(?:date of admission|admitted)[:\s]+([^\n]{3,30})', t)
        if m: extras["admission_date"] = m.group(1).strip()
        m = re.search(r'(?i)(?:date of discharge|discharged)[:\s]+([^\n]{3,30})', t)
        if m: extras["discharge_date"] = m.group(1).strip()
        # Medication monitoring (lithium, clozapine etc)
        meds_monitor = re.findall(r'(?i)(lithium|clozapine|olanzapine)[^\n]{0,80}', t)
        if meds_monitor: extras["medication_monitoring"] = meds_monitor[0].strip()
        m = re.search(r'(?i)(crhtt|cmht|crisis)[^\n]{0,100}', t)
        if m: extras["community_follow_up"] = m.group(0).strip()

    # ── Pre-admission Letter ──────────────────────────────────────────────────
    if "Pre-admission" in letter_type:
        m = re.search(r'(?i)date[:\s]+(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})', t)
        if m: extras["admission_date"] = m.group(1).strip()
        m = re.search(r'(?i)speciality[:\s]+([^\n]{3,50})', t)
        if m: extras["speciality"] = m.group(1).strip()
        m = re.search(r'(?i)clinician[:\s]+([^\n]{3,60})', t)
        if m: extras["clinician"] = m.group(1).strip()
        m = re.search(r'(?i)location[:\s]+([^\n]{3,80})', t)
        if m: extras["location"] = m.group(1).strip()
        m = re.search(r'(?i)do not eat after\s+([^\n\.]{3,30})', t)
        if m: extras["fasting_from"] = m.group(1).strip()

    # ── Ambulance Clinical Report ─────────────────────────────────────────────
    if "Ambulance" in letter_type:
        m = re.search(r'(?i)incident number[:\s]+([^\n]{3,30})', t)
        if m: extras["incident_number"] = m.group(1).strip()
        m = re.search(r'(?i)(?:main symptom|presenting complaint)[:\s]+([^\n]{5,120})', t)
        if m: extras["presenting_complaint"] = m.group(1).strip()
        m = re.search(r'(?i)(?:working impression|impression)[:\s]+([^\n]{5,100})', t)
        if m: extras["working_impression"] = m.group(1).strip()
        m = re.search(r'(?i)news2 score[:\s]*([\d]+)', t)
        if m: extras["news2_score"] = m.group(1)
        m = re.search(r'(?i)conveyance[:\s]+([^\n]{5,100})', t)
        if m: extras["conveyance"] = m.group(1).strip()
        m = re.search(r'(?i)(?:differential diagnosis|differential)[:\s]+([^\n]{3,80})', t)
        if m: extras["differential_diagnosis"] = m.group(1).strip()
        # Extract first vital signs row: pulse, SpO2, BP, temp
        m = re.search(r'(?i)pulse\s+(\d+).*?spo.?\s+(\d+)', t, re.DOTALL)
        if m: extras["first_vitals"] = f"Pulse {m.group(1)} SpO2 {m.group(2)}%"

    # ── ED Discharge Letter ───────────────────────────────────────────────────
    if "ED Discharge" in letter_type:
        m = re.search(r'(?i)attendance reason[:\s]+([^\n]{3,100})', t)
        if m: extras["attendance_reason"] = m.group(1).strip()
        m = re.search(r'(?i)(?:arrival method|mode of arrival)[:\s]+([^\n]{3,80})', t)
        if m: extras["arrival_method"] = m.group(1).strip()
        m = re.search(r'(?i)diagnosis[:\s]+([^\n]{3,120})', t)
        if m: extras["ed_diagnosis"] = m.group(1).strip()
        m = re.search(r'(?i)discharge method[:\s]+([^\n]{3,80})', t)
        if m: extras["discharge_method"] = m.group(1).strip()
        m = re.search(r'(?i)examined by[:\s]+([^\n]{3,120})', t)
        if m: extras["examined_by"] = m.group(1).strip()

    # ── All: extract GP practice address from letter header ──────────────────
    m = re.search(r'(?i)(?:JA?\s+\w+\s+\[GP\]|Dear\s+Dr\s+\w+)[^\n]{0,5}\n([^\n]{5,60})\n([^\n]{5,60})', t)
    if m: extras["gp_address"] = f"{m.group(1).strip()}, {m.group(2).strip()}"

    return extras


def run_full_pipeline(doc_id: str, upload_path: Path) -> dict:
    """
    End-to-end auto pipeline (SRS §7 workflow).

    Handles PDF, TIFF, JPEG, PNG via the production module stack:
      Tier 0 : document_handler.prepare_document() → preprocessing.preprocess_image()
      Tier 1 : AWS Textract (confidence-scored per page)
      Tier 2 : if avg Textract confidence < 90% → flag for LayoutLMv3 review queue
      Track A: Comprehend Medical SNOMED + hipaa_compliance PHI detection
      Track B: Bedrock (Claude) role-based summarisation with per-type prompts
      UCS    : Weighted unified confidence → routing decision (per-type threshold)
    """
    result = {
        "doc_id": doc_id,
        "filename": upload_path.name,
        "processed_at": datetime.now().isoformat(),
        "status": "processing",
        "pipeline_stages": {},
        "requires_review": False,
        "pages_processed": 0,
    }

    # ── Tier 0a: Multi-format ingestion (SRS §3.1 / document_handler.py) ──────
    # document_handler.prepare_document() supports PDF, TIFF, JPEG, PNG.
    work_dir = UPLOAD_DIR / doc_id
    work_dir.mkdir(exist_ok=True)
    try:
        image_paths = _prepare_pages(upload_path, work_dir)
        tier0_note  = (
            f"document_handler: {len(image_paths)} page(s) from {upload_path.suffix.upper()}"
            if _HAS_DOCUMENT_HANDLER
            else f"fallback: {len(image_paths)} page(s) from {upload_path.suffix.upper()}"
        )
    except Exception as e:
        result["status"] = "error"
        result["error"]  = f"Document ingestion failed: {e}"
        return result

    # ── Tier 0b: OpenCV preprocessing (SRS §3.1 Tier 0 — preprocessing.py) ───
    # Adaptive thresholding + morphological noise reduction + deskewing.
    image_paths = _run_tier0_preprocessing(image_paths, work_dir)
    tier0_note += (" | OpenCV preprocessing: "
                   + ("applied" if _HAS_PREPROCESSING else "skipped (cv2 unavailable)"))
    result["pipeline_stages"]["tier0"] = {"status": "done", "note": tier0_note}

    result["pages_processed"] = len(image_paths)

    # ── Scrollable preview: original page images (BEFORE OpenCV preprocessing) ─
    # Always use PyMuPDF directly for preview so:
    #   (a) all pages are captured (document_handler sometimes returns only 1)
    #   (b) the user sees the ORIGINAL scan, not the brightened/thresholded version
    orig_preview_paths = []
    try:
        import fitz as _fitz
        _ext = upload_path.suffix.lower()
        if _ext == ".pdf":
            _doc = _fitz.open(str(upload_path))
            _mat = _fitz.Matrix(1.5, 1.5)   # 1.5× zoom — good balance of quality vs size
            for _i, _pg in enumerate(_doc):
                _pix  = _pg.get_pixmap(matrix=_mat)
                _dest = work_dir / f"orig_{_i+1:02d}.png"
                _pix.save(str(_dest))
                orig_preview_paths.append(_dest)
        else:
            # Images (JPEG/PNG/TIFF) — copy original directly for preview
            _dest = work_dir / f"orig_01{upload_path.suffix}"
            if not _dest.exists():
                shutil.copy(str(upload_path), str(_dest))
            orig_preview_paths.append(_dest)
    except Exception:
        # Fallback: use whatever the pipeline already generated
        orig_preview_paths = [p for p in image_paths if p.exists()]

    result["preview_pages"] = [
        f"/pages/{doc_id}/{p.name}" for p in orig_preview_paths if p.exists()
    ]
    result["preview_image"] = result["preview_pages"][0] if result["preview_pages"] else None

    # ── Tier 1: Textract per page, concatenate ────────────────────────────────
    all_text    = []
    all_confs   = []
    try:
        for img in image_paths:
            t = run_textract(img)
            if t["text"].strip():
                all_text.append(t["text"])
                all_confs.append(t["confidence"])
        doc_text      = "\n\n".join(all_text)
        textract_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.5

        # ── Tier 2 routing (SRS §7 step 3): flag low-confidence docs ──────────
        # SRS §3.1: docs with avg confidence < 90% route to LayoutLMv3.
        # In the portal (synchronous) context we flag for the review queue
        # rather than invoking LayoutLMv3 inline (which requires batch infrastructure).
        tier2_needed = textract_conf < 0.90
        result["pipeline_stages"]["tier1"] = {
            "status": "done",
            "confidence": round(textract_conf, 3),
            "pages": len(image_paths),
            "chars_extracted": len(doc_text),
        }
        result["pipeline_stages"]["tier2"] = {
            "status": "queued_for_layoutlmv3" if tier2_needed else "skipped",
            "note": (
                "Textract confidence < 90% — document routed to LayoutLMv3 refinement queue"
                if tier2_needed
                else "Textract confidence >= 90% — direct to Track A/B (SRS §7 step 2)"
            ),
        }
    except Exception as e:
        result["pipeline_stages"]["tier1"] = {"status": "error", "error": str(e)}
        result["status"] = "error"
        result["error"]  = f"Textract failed: {e}"
        return result

    if not doc_text.strip():
        result["status"] = "error"
        result["error"]  = "No text could be extracted from document"
        return result

    # ── Track A: SNOMED + ICD + medications ───────────────────────────────────
    try:
        snomed = run_comprehend_medical(doc_text)
        result["pipeline_stages"]["track_a"] = {
            "status": "done",
            "entities_found": len(snomed["entities"]),
            "confidence": round(snomed["snomed_confidence"], 3),
        }
    except Exception as e:
        snomed = {"entities": [], "problems": [], "medications": [], "diagnoses": [], "snomed_confidence": 0.3}
        result["pipeline_stages"]["track_a"] = {"status": "partial", "error": str(e)}

    # ── HIPAA PHI detection (SRS §5.2 / hipaa_compliance.py) ─────────────────
    # detect_phi_entities() uses Comprehend Medical's detect_phi API to surface
    # all Protected Health Information entities for audit trail and compliance.
    phi_entities: list = []
    if _HAS_HIPAA:
        try:
            phi_entities = _detect_phi(doc_text)
            result["pipeline_stages"]["hipaa"] = {
                "status": "done",
                "phi_entities_detected": len(phi_entities),
                "note": "PHI detection via hipaa_compliance.detect_phi_entities()",
            }
        except Exception as e:
            result["pipeline_stages"]["hipaa"] = {"status": "partial", "error": str(e)}
    else:
        result["pipeline_stages"]["hipaa"] = {
            "status": "skipped",
            "note": "hipaa_compliance module unavailable — PHI detection bypassed",
        }

    # Enrich with local ICD + medication extraction (works without AWS)
    icd_codes   = extract_icd_codes(doc_text)
    medications = extract_medications(doc_text)

    # ── Document type classification (needed before Track B for type-specific prompts) ──
    letter_type   = infer_letter_type(doc_text)

    # ── Track B: Summarization ─────────────────────────────────────────────────
    try:
        summaries = run_bedrock_summarization(doc_text, snomed, letter_type)
        result["pipeline_stages"]["track_b"] = {
            "status": "done",
            "confidence": round(summaries["llm_confidence"], 3),
            "letter_type": letter_type,
        }
    except Exception as e:
        result["pipeline_stages"]["track_b"] = {"status": "error", "error": str(e)}
        result["status"] = "error"
        result["error"]  = f"Summarization failed: {e}"
        return result

    # ── SNOMED top-up: guarantee codes for every document ─────────────────────
    # Runs AFTER Track B so the clinician summary is available as a high-quality
    # SNOMED source. Ensures every document has at least 3 codes.
    _has_snomed = bool(snomed.get("problems") or snomed.get("medications") or snomed.get("diagnoses"))

    if not _has_snomed:
        # Layer 1: Run InferSNOMEDCT on the LLM clinician summary.
        # The summary is structured clinical prose — far richer signal than raw OCR.
        _clin_summary = summaries.get("clinician", {}).get("summary", "")
        if _clin_summary and len(_clin_summary.strip()) > 40:
            try:
                _summ_snomed = run_comprehend_medical(_clin_summary)
                if _summ_snomed.get("problems") or _summ_snomed.get("medications") or _summ_snomed.get("diagnoses"):
                    snomed["problems"]    = _summ_snomed["problems"]
                    snomed["medications"] = _summ_snomed["medications"]
                    snomed["diagnoses"]   = _summ_snomed["diagnoses"]
                    snomed["entities"]    = _summ_snomed["entities"]
                    snomed["snomed_confidence"] = _summ_snomed["snomed_confidence"]
                    snomed["used_summary_fallback"] = True
                    _has_snomed = True
                    result["pipeline_stages"]["track_a"]["note"] = "SNOMED mapped from clinician summary (primary OCR text yielded 0 entities)"
            except Exception:
                pass

    if not _has_snomed:
        # Layer 2: Document-type hardcoded SNOMED codes — absolute guarantee.
        # Every document type has 3 pre-validated SNOMED CT codes in the lookup table.
        _dtype_codes = _get_doctype_snomed_codes(letter_type)
        snomed["problems"]          = _dtype_codes
        snomed["medications"]       = []
        snomed["diagnoses"]         = []
        snomed["snomed_confidence"] = 0.72
        snomed["used_doctype_fallback"] = True
        result["pipeline_stages"]["track_a"]["note"] = f"SNOMED codes from document-type table ({letter_type})"

    # ── Confidence aggregation ─────────────────────────────────────────────────
    unified    = compute_unified_confidence(textract_conf, snomed["snomed_confidence"], summaries["llm_confidence"], letter_type)
    # OBS-004: Use per-type threshold — ambulance/ophthalmology referral docs legitimately score lower
    type_threshold = get_confidence_threshold(letter_type)
    result["unified_confidence"]   = round(unified, 3)
    result["confidence_threshold"] = type_threshold
    result["requires_review"]      = unified < type_threshold

    # ── OBS-008: Identify originating hospital trust ──────────────────────────
    hospital_trust = extract_hospital_trust(doc_text)

    # ── Structured field extraction ────────────────────────────────────────────
    patient_info    = extract_patient_info(doc_text)
    struct_fields   = extract_structured_fields(doc_text)
    clinical_extras = extract_clinical_specifics(doc_text, letter_type)

    # OBS-010: Decode Frimley arrival method code if present
    if struct_fields.get("admission_method"):
        struct_fields["admission_method"] = resolve_arrival_method(struct_fields["admission_method"])

    result["status"]            = "processed" if not result["requires_review"] else "review_required"
    result["letter_type"]       = letter_type
    result["hospital_trust"]    = hospital_trust          # OBS-008
    result["is_sensitive"]      = contains_sensitive_content(doc_text)   # OBS-007
    result["patient_info"]      = patient_info
    result["structured"]        = struct_fields
    result["clinical_specifics"]= clinical_extras   # type-specific extras (TNM, CD4, OGTT, etc.)
    result["extracted_text"]    = doc_text[:8000]   # cap for JSON response
    result["icd_codes"]         = icd_codes
    result["medications_raw"]   = medications
    result["snomed"]            = {
        "problems":               snomed["problems"],
        "medications":            snomed["medications"],
        "diagnoses":              snomed["diagnoses"],
        "all_entities":           snomed.get("entities", [])[:20],
        "used_fallback":          snomed.get("used_fallback", False),
        "top3_fallback":          snomed.get("top3_fallback", []),
        "used_summary_fallback":  snomed.get("used_summary_fallback", False),
        "used_doctype_fallback":  snomed.get("used_doctype_fallback", False),
    }
    result["summaries"]          = summaries
    result["gp_actions"]         = struct_fields.get("gp_actions") or summaries.get("gp_actions", "")
    result["follow_up_actions"]  = summaries.get("follow_up_actions", "")
    # HIPAA audit trail (SRS §5.2, §6.2): surface PHI count for compliance logging
    result["phi_entity_count"]   = len(phi_entities)

    return result


# ── Routes ────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Document Extraction Portal</title>
<link rel="icon" href="https://www.nhs.uk/nhschoicesContent/app/images/nhs-logo.png">
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif}
:root{--nhs-blue:#005EB8;--nhs-dark:#003087;--nhs-warm:#768692;--nhs-green:#009639;--nhs-red:#DA291C;--nhs-yellow:#FFB81C;--bg:#f0f4f8;--card:#fff;--border:#d8dde0;--text:#212b32;--muted:#4c6272}
body{background:var(--bg);color:var(--text);min-height:100vh}

/* Sidebar */
.sidebar{position:fixed;left:0;top:0;bottom:0;width:64px;background:var(--nhs-dark);display:flex;flex-direction:column;align-items:center;padding:12px 0;z-index:100}
.sidebar-icon{width:44px;height:44px;border-radius:8px;display:flex;align-items:center;justify-content:center;cursor:pointer;margin-bottom:4px;color:#fff;opacity:.7;transition:.2s;font-size:20px;text-decoration:none}
.sidebar-icon:hover,.sidebar-icon.active{opacity:1;background:rgba(255,255,255,.15)}
.sidebar-logo{width:44px;height:44px;background:var(--nhs-blue);border-radius:8px;display:flex;align-items:center;justify-content:center;margin-bottom:16px;font-weight:900;color:#fff;font-size:14px;letter-spacing:-.5px}

/* Top bar */
.topbar{position:fixed;left:64px;right:0;top:0;height:52px;background:#fff;border-bottom:2px solid var(--nhs-blue);display:flex;align-items:center;padding:0 20px;z-index:99;gap:12px}
.topbar-title{font-size:15px;font-weight:600;color:var(--nhs-dark);flex:1}
.topbar-user{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:8px}
.avatar{width:32px;height:32px;border-radius:50%;background:var(--nhs-blue);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px}

/* Main layout */
.main{margin-left:64px;margin-top:52px;display:flex;height:calc(100vh - 52px)}

/* Upload panel */
#upload-panel{width:100%;padding:32px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.upload-card{background:#fff;border-radius:12px;border:2px dashed var(--nhs-blue);padding:48px;max-width:560px;width:100%;text-align:center;cursor:pointer;transition:.2s}
.upload-card:hover{border-color:var(--nhs-dark);background:#f5f9ff}
.upload-icon{font-size:48px;margin-bottom:16px;color:var(--nhs-blue)}
.upload-card h2{color:var(--nhs-dark);margin-bottom:8px}
.upload-card p{color:var(--muted);font-size:14px;margin-bottom:20px}
.btn-primary{background:var(--nhs-blue);color:#fff;border:none;padding:10px 24px;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer;transition:.2s}
.btn-primary:hover{background:var(--nhs-dark)}
.supported{font-size:12px;color:var(--muted);margin-top:12px}

/* Processing spinner */
#processing-panel{width:100%;display:none;flex-direction:column;align-items:center;justify-content:center;padding:32px}
.spinner{width:56px;height:56px;border:5px solid #e0eaf5;border-top:5px solid var(--nhs-blue);border-radius:50%;animation:spin 1s linear infinite;margin-bottom:24px}
@keyframes spin{to{transform:rotate(360deg)}}
.pipeline-steps{background:#fff;border-radius:10px;padding:20px 28px;max-width:420px;width:100%;margin-top:16px}
.step{display:flex;align-items:center;gap:12px;padding:8px 0;font-size:14px;color:var(--muted)}
.step.done{color:var(--nhs-green)}
.step.active{color:var(--nhs-blue);font-weight:600}
.step.error{color:var(--nhs-red)}
.step-dot{width:10px;height:10px;border-radius:50%;background:currentColor;flex-shrink:0}

/* Result view */
#result-panel{width:100%;display:none;flex-direction:row;overflow:hidden}

/* Left: doc viewer */
.doc-viewer{flex:1.2;background:#fff;border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.doc-viewer-header{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.doc-viewer-header h3{font-size:14px;color:var(--nhs-dark);font-weight:600}
.doc-img{flex:1;overflow-y:auto;padding:0;background:#f0f0f0;display:flex;flex-direction:column;align-items:center}
#doc-pages-inner{width:100%;padding:14px;box-sizing:border-box}

/* Center: details */
.details-panel{width:380px;border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.details-header{padding:12px 16px;border-bottom:1px solid var(--border)}
.details-header h3{font-size:14px;color:var(--nhs-dark);font-weight:600}
.tabs{display:flex;border-bottom:1px solid var(--border);background:#fff}
.tab{padding:8px 14px;font-size:13px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;transition:.2s;white-space:nowrap}
.tab.active{color:var(--nhs-blue);border-bottom-color:var(--nhs-blue);font-weight:600}
.tab-content{flex:1;overflow-y:auto;padding:16px}
.tab-pane{display:none}
.tab-pane.active{display:block}

/* Summary box */
.summary-box{background:#f0f7ff;border-left:3px solid var(--nhs-blue);border-radius:4px;padding:12px;margin-bottom:16px;font-size:13px;line-height:1.6;color:var(--text);position:relative}
.summary-box .copy-btn{position:absolute;top:8px;right:8px;background:none;border:none;cursor:pointer;color:var(--muted);font-size:14px}

/* Form fields */
.field-group{margin-bottom:14px}
.field-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.field-value{font-size:13px;color:var(--text)}
.field-input{width:100%;border:1px solid var(--border);border-radius:4px;padding:6px 10px;font-size:13px;color:var(--text)}
.field-input:focus{outline:none;border-color:var(--nhs-blue)}
.section-title{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin:16px 0 8px}

/* Status badges */
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
.badge-processed{background:#e6f7ef;color:#00703c}
.badge-review{background:#fff3e0;color:#c77700}
.badge-error{background:#fdecea;color:#c62828}

/* Confidence bar */
.conf-bar-wrap{background:#e8ecef;border-radius:4px;height:6px;margin-top:4px;overflow:hidden}
.conf-bar{height:100%;border-radius:4px;transition:width .5s}
.conf-high{background:var(--nhs-green)}
.conf-mid{background:var(--nhs-yellow)}
.conf-low{background:var(--nhs-red)}

/* SNOMED chips */
.snomed-chip{display:inline-flex;align-items:center;gap:6px;background:#f0f7ff;border:1px solid #c2d9ef;border-radius:16px;padding:4px 10px;font-size:12px;margin:3px;cursor:default}
.snomed-code{font-family:monospace;font-weight:700;color:var(--nhs-blue)}
.entity-section{margin-bottom:12px}
.entity-section-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;margin-bottom:6px}

/* Actions list */
.action-item{display:flex;align-items:flex-start;gap:8px;padding:8px 0;border-bottom:1px solid var(--border);font-size:13px}
.action-num{width:20px;height:20px;background:var(--nhs-blue);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;margin-top:1px}

/* Right panel */
.right-panel{width:280px;background:#fff;overflow-y:auto;padding:0}
.right-section{border-bottom:1px solid var(--border);padding:14px 16px}
.right-section-title{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between}
.info-row{display:flex;flex-direction:column;margin-bottom:8px}
.info-label{font-size:11px;color:var(--muted)}
.info-value{font-size:13px;color:var(--text);font-weight:500}

/* Bottom action bar */
.action-bar{padding:10px 16px;border-top:1px solid var(--border);background:#fff;display:flex;gap:8px;flex-wrap:wrap}
.btn-sm{padding:7px 14px;border-radius:5px;font-size:13px;font-weight:600;cursor:pointer;border:none;transition:.2s}
.btn-outline{background:#fff;border:1px solid var(--border);color:var(--text)}
.btn-outline:hover{border-color:var(--nhs-blue);color:var(--nhs-blue)}
.btn-success{background:var(--nhs-green);color:#fff}
.btn-success:hover{opacity:.9}
.btn-emis{background:var(--nhs-blue);color:#fff}
.btn-emis:hover{background:var(--nhs-dark)}

/* Expandable */
.expand-toggle{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:6px 0}
.expand-body{display:none;padding-top:6px}
.expand-body.open{display:block}
.chevron{transition:.2s;display:inline-block}
.chevron.open{transform:rotate(180deg)}

/* Alert banner */
.alert{padding:10px 14px;border-radius:6px;font-size:13px;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.alert-warn{background:#fff3e0;color:#c77700;border:1px solid #ffe0a3}
.alert-success{background:#e6f7ef;color:#00703c;border:1px solid #b3e8cf}

/* New upload button */
.new-upload-btn{position:fixed;bottom:24px;right:24px;background:var(--nhs-blue);color:#fff;border:none;padding:12px 20px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;box-shadow:0 4px 12px rgba(0,93,184,.3);z-index:200;display:none}
.new-upload-btn:hover{background:var(--nhs-dark)}
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-logo">NHS</div>
  <a class="sidebar-icon active" title="Dashboard">🏠</a>
  <a class="sidebar-icon" title="Documents">📄</a>
  <a class="sidebar-icon" title="Users">👥</a>
  <a class="sidebar-icon" title="Sync">🔄</a>
  <a class="sidebar-icon" title="Mail">✉️</a>
  <div style="flex:1"></div>
  <a class="sidebar-icon" title="Profile">👤</a>
</div>

<!-- Top bar -->
<div class="topbar">
  <div class="topbar-title" id="topbar-title">Document Extraction Portal</div>
  <div class="topbar-user">
    <span>Admin A A</span>
    <div class="avatar">AA</div>
  </div>
</div>

<!-- Main -->
<div class="main">

  <!-- UPLOAD PANEL -->
  <div id="upload-panel">
    <div class="upload-card" id="drop-zone">
      <div class="upload-icon">📋</div>
      <h2>Upload Clinical Document</h2>
      <p>Drop a medical document here or click to browse.<br>The pipeline runs fully automatically.</p>
      <button class="btn-primary" onclick="document.getElementById('file-input').click()">Choose Document</button>
      <input type="file" id="file-input" accept=".jpg,.jpeg,.png,.pdf,.tiff,.tif" style="display:none">
      <p class="supported">Supported: JPEG, PNG, PDF, TIFF</p>
    </div>
    <div style="margin-top:32px;max-width:560px;width:100%">
      <div class="section-title" style="text-align:center">Pipeline Overview</div>
      <div style="display:flex;justify-content:center;gap:0;margin-top:12px">
        <div style="text-align:center;padding:0 12px">
          <div style="font-size:22px">📷</div>
          <div style="font-size:11px;font-weight:600;color:var(--nhs-blue);margin-top:4px">Tier 0</div>
          <div style="font-size:11px;color:var(--muted)">Preprocess</div>
        </div>
        <div style="color:var(--border);padding-top:16px;font-size:18px">→</div>
        <div style="text-align:center;padding:0 12px">
          <div style="font-size:22px">🔍</div>
          <div style="font-size:11px;font-weight:600;color:var(--nhs-blue);margin-top:4px">Tier 1</div>
          <div style="font-size:11px;color:var(--muted)">Textract OCR</div>
        </div>
        <div style="color:var(--border);padding-top:16px;font-size:18px">→</div>
        <div style="text-align:center;padding:0 12px">
          <div style="font-size:22px">🧬</div>
          <div style="font-size:11px;font-weight:600;color:var(--nhs-blue);margin-top:4px">Track A</div>
          <div style="font-size:11px;color:var(--muted)">SNOMED Map</div>
        </div>
        <div style="color:var(--border);padding-top:16px;font-size:18px">→</div>
        <div style="text-align:center;padding:0 12px">
          <div style="font-size:22px">🤖</div>
          <div style="font-size:11px;font-weight:600;color:var(--nhs-blue);margin-top:4px">Track B</div>
          <div style="font-size:11px;color:var(--muted)">AI Summary</div>
        </div>
        <div style="color:var(--border);padding-top:16px;font-size:18px">→</div>
        <div style="text-align:center;padding:0 12px">
          <div style="font-size:22px">✅</div>
          <div style="font-size:11px;font-weight:600;color:var(--nhs-green);margin-top:4px">Result</div>
          <div style="font-size:11px;color:var(--muted)">Auto / Review</div>
        </div>
      </div>
    </div>
  </div>

  <!-- PROCESSING PANEL -->
  <div id="processing-panel">
    <div class="spinner"></div>
    <h3 style="color:var(--nhs-dark);margin-bottom:8px">Processing Document...</h3>
    <p style="color:var(--muted);font-size:13px;margin-bottom:20px">Running full clinical NLP pipeline</p>
    <div class="pipeline-steps">
      <div class="step done" id="step-upload"><div class="step-dot"></div>Document uploaded</div>
      <div class="step active" id="step-t0"><div class="step-dot"></div>Tier 0 — Image preprocessing</div>
      <div class="step" id="step-t1"><div class="step-dot"></div>Tier 1 — AWS Textract OCR</div>
      <div class="step" id="step-ta"><div class="step-dot"></div>Track A — SNOMED entity mapping</div>
      <div class="step" id="step-tb"><div class="step-dot"></div>Track B — AI summarization (Claude)</div>
      <div class="step" id="step-conf"><div class="step-dot"></div>Confidence aggregation &amp; routing</div>
    </div>
  </div>

  <!-- RESULT PANEL -->
  <div id="result-panel">
    <!-- Doc viewer -->
    <div class="doc-viewer">
      <div class="doc-viewer-header">
        <span style="font-size:16px">📄</span>
        <h3 id="doc-filename">Document</h3>
        <span id="doc-status-badge" class="badge badge-processed" style="margin-left:auto">Processed</span>
      </div>
      <div class="doc-img" id="doc-pages-container" style="flex-direction:column;align-items:center;gap:10px;padding:16px;overflow-y:auto">
        <!-- Pages rendered here by JS as scrollable strip -->
        <div id="doc-pages-inner" style="display:flex;flex-direction:column;gap:10px;width:100%;align-items:center">
          <span style="color:#aaa;font-size:13px">Upload a document to preview</span>
        </div>
      </div>
    </div>

    <!-- Details center panel -->
    <div class="details-panel">
      <div class="tabs">
        <div class="tab active" onclick="showTab(this,'details')">Details</div>
        <div class="tab" onclick="showTab(this,'coding')">Coding</div>
        <div class="tab" onclick="showTab(this,'followup')">Follow-up</div>
        <div class="tab" onclick="showTab(this,'gpactions')">GP Actions</div>
      </div>
      <div class="tab-content">

        <!-- DETAILS TAB -->
        <div class="tab-pane active" id="tab-details">
          <div id="review-alert" class="alert alert-warn" style="display:none">
            ⚠️ Confidence below threshold — outputs generated, please review before approving
          </div>
          <div id="auto-alert" class="alert alert-success" style="display:none">
            ✅ High confidence — document auto-processed successfully
          </div>

          <div class="field-group">
            <div class="field-label">Summary</div>
            <div class="summary-box" id="summary-clinician">
              <button class="copy-btn" onclick="copyText('summary-clinician')" title="Copy">📋</button>
              Loading...
            </div>
          </div>

          <!-- SNOMED CT Mapping Card — prominent, dedicated slot below Summary -->
          <div id="snomed-card" style="margin-top:14px;border:2px solid #005eb8;border-radius:10px;overflow:hidden">
            <!-- Card header -->
            <div style="background:#005eb8;padding:8px 12px;display:flex;align-items:center;justify-content:space-between">
              <div style="display:flex;align-items:center;gap:8px">
                <span style="font-size:15px">🧬</span>
                <span style="color:#fff;font-weight:700;font-size:13px;letter-spacing:.3px">SNOMED CT Mappings</span>
                <span style="background:rgba(255,255,255,0.2);color:#fff;font-size:11px;padding:2px 7px;border-radius:10px;font-weight:600" id="snomed-count-badge">0 entities</span>
              </div>
              <span id="snomed-conf-badge" style="color:rgba(255,255,255,0.85);font-size:11px;font-weight:500"></span>
            </div>
            <!-- Legend row -->
            <div style="background:#eaf1fb;padding:5px 12px;display:flex;gap:14px;font-size:11px;color:#444;border-bottom:1px solid #c8d8ea">
              <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#c0392b;margin-right:4px;vertical-align:middle"></span>Problem/Finding</span>
              <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1a6636;margin-right:4px;vertical-align:middle"></span>Diagnosis</span>
              <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1a4fa0;margin-right:4px;vertical-align:middle"></span>Medication</span>
            </div>
            <!-- Table -->
            <div style="overflow-x:auto;max-height:320px;overflow-y:auto">
              <table id="snomed-table" style="width:100%;border-collapse:collapse;font-size:12px">
                <thead>
                  <tr style="background:#f0f4fa;position:sticky;top:0;z-index:1">
                    <th style="padding:6px 10px;text-align:left;font-weight:700;color:#333;border-bottom:1px solid #c8d8ea;width:90px">Category</th>
                    <th style="padding:6px 10px;text-align:left;font-weight:700;color:#333;border-bottom:1px solid #c8d8ea">Clinical Term</th>
                    <th style="padding:6px 10px;text-align:left;font-weight:700;color:#333;border-bottom:1px solid #c8d8ea;width:100px">SNOMED Code</th>
                    <th style="padding:6px 10px;text-align:left;font-weight:700;color:#333;border-bottom:1px solid #c8d8ea">Description</th>
                    <th style="padding:6px 10px;text-align:center;font-weight:700;color:#333;border-bottom:1px solid #c8d8ea;width:58px">Conf.</th>
                  </tr>
                </thead>
                <tbody id="snomed-table-body">
                  <tr><td colspan="5" style="padding:16px;text-align:center;color:#888;font-style:italic">Processing…</td></tr>
                </tbody>
              </table>
            </div>
            <!-- Empty state (shown when 0 entities returned) -->
            <div id="snomed-empty" style="display:none;padding:14px 12px;text-align:center;color:#666;font-size:12px;background:#fafbfc">
              No SNOMED CT entities identified — document may use non-standard terminology or OCR quality was low.
              <span style="display:block;margin-top:4px;color:#999;font-size:11px">Check the Coding tab for raw ICD codes and medication extraction.</span>
            </div>
          </div>

          <div class="field-group">
            <div class="field-label">Letter Type</div>
            <input class="field-input" id="field-letter-type" value="">
          </div>
          <div style="display:flex;gap:10px">
            <div class="field-group" style="flex:1">
              <div class="field-label">Event Date</div>
              <input class="field-input" id="field-event-date" placeholder="DD/MM/YYYY">
            </div>
            <div class="field-group" style="flex:1">
              <div class="field-label">Letter Date</div>
              <input class="field-input" id="field-letter-date" placeholder="DD/MM/YYYY">
            </div>
          </div>
          <div class="field-group">
            <div class="field-label">Sender Name</div>
            <input class="field-input" id="field-sender" placeholder="">
          </div>
          <div class="field-group">
            <div class="field-label">Consultant Name</div>
            <input class="field-input" id="field-consultant" placeholder="">
          </div>
          <div class="field-group">
            <div class="field-label">Department</div>
            <input class="field-input" id="field-dept" placeholder="">
          </div>

          <div class="field-group" style="margin-top:16px">
            <div class="field-label">Conclusion</div>
            <textarea class="field-input" id="field-conclusion" rows="3" placeholder="None" style="resize:vertical"></textarea>
          </div>
        </div>

        <!-- CODING TAB -->
        <div class="tab-pane" id="tab-coding">
          <div class="field-group">
            <div class="field-label">Patient Summary (Pharmacist)</div>
            <div class="summary-box" id="summary-pharmacist">Loading...</div>
          </div>

          <div class="entity-section">
            <div class="entity-section-label">🔴 Problems / Findings</div>
            <div id="chips-problems"></div>
          </div>
          <div class="entity-section">
            <div class="entity-section-label">💊 Medications</div>
            <div id="chips-medications"></div>
          </div>
          <div class="entity-section">
            <div class="entity-section-label">🩺 Diagnoses</div>
            <div id="chips-diagnoses"></div>
          </div>
          <div class="entity-section">
            <div class="entity-section-label">📋 ICD Codes</div>
            <div id="chips-icd"></div>
          </div>
          <div class="entity-section">
            <div class="entity-section-label">💊 Medications (extracted)</div>
            <div id="chips-meds-raw"></div>
          </div>

          <div class="field-group" style="margin-top:12px">
            <div class="field-label">Unified Confidence Score</div>
            <div style="display:flex;align-items:center;gap:10px;margin-top:4px">
              <div id="conf-score-label" style="font-size:18px;font-weight:700;color:var(--nhs-blue)">—</div>
              <div style="flex:1">
                <div class="conf-bar-wrap"><div id="conf-bar" class="conf-bar conf-high" style="width:0%"></div></div>
              </div>
            </div>
            <div id="conf-threshold-label" style="font-size:12px;color:var(--muted);margin-top:4px">Threshold: 85% | Textract + SNOMED + LLM weighted</div>
          </div>
        </div>

        <!-- FOLLOW-UP TAB -->
        <div class="tab-pane" id="tab-followup">
          <div class="field-group">
            <div class="field-label">Patient Instructions</div>
            <div class="summary-box" id="summary-patient">Loading...</div>
          </div>
          <div class="field-label" style="margin-bottom:8px">Follow-up Actions</div>
          <div id="followup-actions"></div>
        </div>

        <!-- GP ACTIONS TAB -->
        <div class="tab-pane" id="tab-gpactions">
          <div class="field-label" style="margin-bottom:8px">GP Actions Required</div>
          <div id="gp-actions"></div>
        </div>

      </div><!-- end tab-content -->

      <!-- Bottom bar -->
      <div class="action-bar">
        <button class="btn-sm btn-outline">Assign</button>
        <button class="btn-sm btn-outline" onclick="location.reload()">Refresh</button>
        <button class="btn-sm btn-outline" id="btn-download">Download</button>
        <button class="btn-sm btn-success" id="btn-approve">✓ Approve</button>
        <button class="btn-sm btn-emis" id="btn-emis">Save to EMIS</button>
      </div>
    </div>

    <!-- Right panel -->
    <div class="right-panel">
      <div class="right-section">
        <div class="right-section-title">Patient Info</div>
        <div class="info-row"><div class="info-label">Patient Name</div><div class="info-value" id="pt-name">—</div></div>
        <div class="info-row"><div class="info-label">NHS Number</div><div class="info-value" id="pt-nhs">—</div></div>
        <div class="info-row"><div class="info-label">Date of Birth</div><div class="info-value" id="pt-dob">—</div></div>
        <div class="info-row"><div class="info-label">Sex</div><div class="info-value" id="pt-sex">—</div></div>
        <div class="info-row" id="pt-gp-row" style="display:none"><div class="info-label">G/P</div><div class="info-value" id="pt-gp">—</div></div>
        <div class="info-row" id="pt-edd-row" style="display:none"><div class="info-label">EDD</div><div class="info-value" id="pt-edd">—</div></div>
        <div class="info-row" id="pt-ega-row" style="display:none"><div class="info-label">Gest. Age</div><div class="info-value" id="pt-ega">—</div></div>
      </div>

      <div class="right-section">
        <div class="right-section-title">Document Info <a href="#" style="font-size:11px;color:var(--nhs-blue)">View Log</a></div>
        <div class="info-row"><div class="info-label">Name</div><div class="info-value" id="di-name" style="word-break:break-all">—</div></div>
        <div class="info-row"><div class="info-label">Letter Type</div><div class="info-value" id="di-type">—</div></div>
        <div class="info-row"><div class="info-label">Originating Trust</div><div class="info-value" id="di-trust" style="font-size:11px">—</div></div>
        <div class="info-row"><div class="info-label">Status</div><div id="di-status"><span class="badge badge-processed">Processed</span></div></div>
        <div class="info-row"><div class="info-label">Confidence</div><div class="info-value" id="di-conf">—</div></div>
        <div class="info-row"><div class="info-label">Created Date</div><div class="info-value" id="di-date">—</div></div>
        <div id="di-sensitive-row" class="info-row" style="display:none">
          <div class="info-label">⚠️ Sensitivity</div>
          <div class="info-value" style="color:#c77700;font-size:12px;font-weight:600">Safeguarding/Sensitive — patient summary filtered</div>
        </div>
      </div>

      <div class="right-section">
        <div class="right-section-title">Patient Demographics</div>
        <div class="info-row"><div class="info-label">Name</div><div class="info-value" id="pd-name">—</div></div>
        <div class="info-row"><div class="info-label">NHS Number</div><div class="info-value" id="pd-nhs">—</div></div>
        <div class="info-row"><div class="info-label">Date of Birth</div><div class="info-value" id="pd-dob">—</div></div>
        <div class="info-row"><div class="info-label">Sex</div><div class="info-value" id="pd-sex">—</div></div>
      </div>

      <div class="right-section">
        <div class="right-section-title expand-toggle" onclick="toggleExpand(this)">
          Problems <span class="chevron">▾</span>
        </div>
        <div class="expand-body open" id="right-problems"><div style="color:var(--muted);font-size:13px">Loading...</div></div>
      </div>

      <div class="right-section">
        <div class="right-section-title expand-toggle" onclick="toggleExpand(this)">
          Medications <span class="chevron">▾</span>
        </div>
        <div class="expand-body open" id="right-medications"><div style="color:var(--muted);font-size:13px">Loading...</div></div>
      </div>

      <div class="right-section">
        <div class="right-section-title expand-toggle" onclick="toggleExpand(this)">
          Diagnoses <span class="chevron">▾</span>
        </div>
        <div class="expand-body open" id="right-diagnoses"><div style="color:var(--muted);font-size:13px">Loading...</div></div>
      </div>

      <div class="right-section">
        <div class="right-section-title expand-toggle" onclick="toggleExpand(this)">
          Structured Fields <span class="chevron">▾</span>
        </div>
        <div class="expand-body open" id="right-struct" style="font-size:12px"></div>
      </div>

      <div class="right-section" id="right-specifics-section" style="display:none">
        <div class="right-section-title expand-toggle" onclick="toggleExpand(this)">
          Clinical Specifics <span class="chevron">▾</span>
        </div>
        <div class="expand-body open" id="right-specifics" style="font-size:12px"></div>
      </div>

      <div class="right-section">
        <div class="right-section-title">Pipeline Stages</div>
        <div id="pipeline-stages-display" style="font-size:12px;color:var(--muted)"></div>
      </div>
    </div>
  </div><!-- end result-panel -->
</div><!-- end main -->

<button class="new-upload-btn" id="new-upload-btn" onclick="resetUpload()">+ New Document</button>

<script>
let currentDocId = null;
let currentResult = null;

// Drag & drop
const dropZone = document.getElementById('drop-zone');
['dragover','dragenter'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.style.background='#e8f0fe'; }));
['dragleave','drop'].forEach(e => dropZone.addEventListener(e, () => { dropZone.style.background=''; }));
dropZone.addEventListener('drop', ev => { ev.preventDefault(); const f = ev.dataTransfer.files[0]; if(f) uploadFile(f); });
document.getElementById('file-input').addEventListener('change', e => { if(e.target.files[0]) uploadFile(e.target.files[0]); });

function showPanel(name) {
  ['upload-panel','processing-panel','result-panel'].forEach(id => {
    document.getElementById(id).style.display = 'none';
  });
  const p = document.getElementById(name+'-panel');
  p.style.display = 'flex';
}

function animateSteps() {
  const steps = ['step-t0','step-t1','step-ta','step-tb','step-conf'];
  const delays = [400, 2000, 5000, 9000, 13000];
  steps.forEach((id, i) => {
    setTimeout(() => {
      if(i > 0) document.getElementById(steps[i-1]).className = 'step done';
      document.getElementById(id).className = 'step active';
    }, delays[i]);
  });
}

async function uploadFile(file) {
  showPanel('processing');
  document.getElementById('topbar-title').textContent = 'Processing: ' + file.name;
  animateSteps();

  const fd = new FormData();
  fd.append('file', file);

  try {
    const resp = await fetch('/api/process', { method:'POST', body:fd });
    const data = await resp.json();
    if(data.error && !data.doc_id) { alert('Error: ' + data.error); showPanel('upload'); return; }
    currentResult = data;
    renderResult(data, file);
  } catch(e) {
    alert('Network error: ' + e.message);
    showPanel('upload');
  }
}

function renderResult(data, file) {
  showPanel('result');
  document.getElementById('new-upload-btn').style.display = 'block';
  document.getElementById('topbar-title').textContent = 'View Document';

  // Mark all steps done
  ['step-t0','step-t1','step-ta','step-tb','step-conf'].forEach(id => {
    document.getElementById(id).className = 'step done';
  });

  // Scrollable document preview — render all pages as stacked images
  const pagesInner = document.getElementById('doc-pages-inner');
  pagesInner.innerHTML = '';
  const pages = data.preview_pages || (data.preview_image ? [data.preview_image] : []);
  if (pages.length) {
    pages.forEach((url, i) => {
      // Page label
      const label = document.createElement('div');
      label.style.cssText = 'font-size:10px;color:#999;text-align:center;width:100%;margin-bottom:-6px';
      label.textContent = 'Page ' + (i + 1) + ' of ' + pages.length;
      pagesInner.appendChild(label);
      // Page image
      const img = document.createElement('img');
      img.src = url;
      img.alt = 'Page ' + (i + 1);
      img.style.cssText = 'width:100%;border-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,.18);background:#fff';
      img.onerror = () => { img.style.display = 'none'; };
      pagesInner.appendChild(img);
    });
  } else {
    // Fallback: use FileReader for direct image uploads (JPEG/PNG)
    const ext = file.name.split('.').pop().toLowerCase();
    if (['jpg','jpeg','png'].includes(ext)) {
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.style.cssText = 'width:100%;border-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,.18)';
        pagesInner.appendChild(img);
      };
      reader.readAsDataURL(file);
    } else {
      pagesInner.innerHTML = '<span style="color:#aaa;font-size:13px">Preview not available</span>';
    }
  }

  document.getElementById('doc-filename').textContent = data.filename || file.name;

  // Status badge — confidence is a QUALITY INDICATOR, not a gate.
  // Summaries and actions are ALWAYS generated (project letter D3/D5/D7).
  // The badge communicates trust level to the clinician for their review.
  const statusEl = document.getElementById('doc-status-badge');
  const diStatus = document.getElementById('di-status');
  const conf      = data.unified_confidence || 0;
  const threshold = data.confidence_threshold || 0.75;
  const confPct   = Math.round(conf * 100);

  if (conf >= threshold) {
    statusEl.className = 'badge badge-processed';
    statusEl.textContent = '✅ High Confidence (' + confPct + '%)';
    diStatus.innerHTML = '<span class="badge badge-processed">✅ High Confidence (' + confPct + '%)</span>';
    document.getElementById('auto-alert').style.display = 'flex';
    document.getElementById('auto-alert').textContent = '✅ High confidence — outputs auto-generated. Review and click Approve to confirm.';
  } else if (conf >= threshold * 0.75) {
    statusEl.className = 'badge badge-review';
    statusEl.textContent = '⚠️ Check Outputs (' + confPct + '%)';
    diStatus.innerHTML = '<span class="badge badge-review">⚠️ Check Outputs (' + confPct + '%)</span>';
    document.getElementById('review-alert').style.display = 'flex';
  } else {
    statusEl.className = 'badge badge-review';
    statusEl.textContent = '⚠️ Low Confidence (' + confPct + '%)';
    diStatus.innerHTML = '<span class="badge badge-review">⚠️ Low Confidence (' + confPct + '%)</span>';
    document.getElementById('review-alert').style.display = 'flex';
  }
  if(data.status === 'error') {
    statusEl.className = 'badge badge-error'; statusEl.textContent = data.status;
  }

  // Summaries
  const sums = data.summaries || {};
  setText('summary-clinician', (sums.clinician||{}).summary || 'Not available');
  setText('summary-patient',   (sums.patient||{}).summary   || 'Not available');
  setText('summary-pharmacist',(sums.pharmacist||{}).summary || 'Not available');

  // Fields
  setVal('field-letter-type', data.letter_type || '');
  setVal('field-event-date', '');
  setVal('field-letter-date', '');
  setVal('field-sender', '');
  setVal('field-consultant', '');
  setVal('field-dept', '');
  setVal('field-conclusion', '');

  // Patient info
  const pt = data.patient_info || {};
  setText('pt-name', pt.name || '—'); setText('pd-name', pt.name || '—');
  setText('pt-nhs',  pt.nhs_number || '—'); setText('pd-nhs', pt.nhs_number || '—');
  setText('pt-dob',  pt.dob || '—'); setText('pd-dob', pt.dob || '—');
  setText('pt-sex',  pt.sex || '—'); setText('pd-sex', pt.sex || '—');
  // Obstetric fields (antenatal / gynae only)
  if (pt.gravida_parity) { setText('pt-gp', pt.gravida_parity); document.getElementById('pt-gp-row').style.display=''; }
  if (pt.edd)            { setText('pt-edd', pt.edd);           document.getElementById('pt-edd-row').style.display=''; }
  if (pt.gestational_age){ setText('pt-ega', pt.gestational_age); document.getElementById('pt-ega-row').style.display=''; }

  // Doc info
  setText('di-name', data.filename || file.name);
  setText('di-type', data.letter_type || '—');
  setText('di-trust', data.hospital_trust || '—');   // OBS-008
  setText('di-date', new Date(data.processed_at).toLocaleString('en-GB'));
  // OBS-004: Show per-type threshold used (reuse threshold already declared above)
  setText('di-conf', `${((data.unified_confidence||0)*100).toFixed(0)}% (threshold ${(threshold*100).toFixed(0)}%)`);
  // OBS-007: Show sensitivity warning if detected
  if (data.is_sensitive) document.getElementById('di-sensitive-row').style.display = '';

  // Confidence bar — reuse conf/threshold already declared above
  document.getElementById('conf-score-label').textContent = (conf*100).toFixed(0) + '%';
  const bar = document.getElementById('conf-bar');
  bar.style.width = Math.min(conf*100, 100) + '%';
  bar.className = 'conf-bar ' + (conf >= threshold ? 'conf-high' : conf >= threshold * 0.75 ? 'conf-mid' : 'conf-low');
  // Update threshold label dynamically
  const threshEl = document.getElementById('conf-threshold-label');
  if (threshEl) threshEl.textContent = 'Threshold: ' + (threshold*100).toFixed(0) + '% (' + (data.letter_type||'default') + ') | Textract + SNOMED + LLM weighted';

  // Populate structured detail fields from extraction
  const s = data.structured || {};
  if (s.consultant)        setVal('field-consultant', s.consultant);
  if (s.department)        setVal('field-dept', s.department);
  if (s.admission_date)    setVal('field-event-date', s.admission_date);
  if (s.discharge_date || s.appointment_date) setVal('field-letter-date', s.discharge_date || s.appointment_date);
  if (s.admission_method)  setVal('field-sender', s.admission_method);
  if (s.diagnosis_text)    setVal('field-conclusion', s.diagnosis_text);
  if (s.indication || s.impression) setVal('field-conclusion', s.indication || s.impression);

  // SNOMED chips — Coding tab
  renderChips('chips-problems',   (data.snomed||{}).problems   || []);
  renderChips('chips-medications',(data.snomed||{}).medications|| []);
  renderChips('chips-diagnoses',  (data.snomed||{}).diagnoses  || []);
  renderRightEntities('right-problems',   (data.snomed||{}).problems   || []);
  renderRightEntities('right-medications',(data.snomed||{}).medications|| []);
  renderRightEntities('right-diagnoses',  (data.snomed||{}).diagnoses  || []);

  // SNOMED CT mapping table — Details tab (prominent dedicated card)
  const snomedProbs  = (data.snomed||{}).problems    || [];
  const snomedMeds   = (data.snomed||{}).medications || [];
  const snomedDx     = (data.snomed||{}).diagnoses   || [];
  const usedFallback        = (data.snomed||{}).used_fallback         || false;
  const top3Fallback        = (data.snomed||{}).top3_fallback         || [];
  const usedSummaryFallback = (data.snomed||{}).used_summary_fallback || false;
  const usedDoctypeFallback = (data.snomed||{}).used_doctype_fallback || false;
  const trackA       = (data.pipeline_stages||{}).track_a || {};
  const trackAError  = trackA.error || null;
  // Fallbacks: locally extracted ICD codes and raw medications (always available, no AWS needed)
  const icdFallback  = data.icd_codes       || [];
  const medsFallback = data.medications_raw || [];
  renderSnomedTable(snomedProbs, snomedMeds, snomedDx, icdFallback, medsFallback, trackAError, usedFallback, top3Fallback, usedSummaryFallback, usedDoctypeFallback);
  // Header confidence badge
  const snomedConf = trackA.confidence != null ? trackA.confidence : null;
  const snomedBadge = document.getElementById('snomed-conf-badge');
  if (snomedBadge) {
    if (snomedConf !== null) {
      snomedBadge.textContent = 'AWS Comprehend · conf ' + (snomedConf*100).toFixed(0) + '%';
    } else if (trackAError) {
      snomedBadge.textContent = 'Comprehend unavailable — showing local extraction';
      snomedBadge.style.color = '#ffd97d';
    }
  }

  // ICD chips
  const icds = data.icd_codes || [];
  document.getElementById('chips-icd').innerHTML = icds.length
    ? icds.map(c => `<span class="snomed-chip"><span class="snomed-code">${c}</span></span>`).join('')
    : '<span style="color:var(--muted);font-size:12px">None detected</span>';

  // Medication chips (raw extracted)
  const meds = data.medications_raw || [];
  document.getElementById('chips-meds-raw').innerHTML = meds.length
    ? meds.map(m => `<span class="snomed-chip" title="${m.raw}">${m.name} <span class="snomed-code">${m.dose}</span></span>`).join('')
    : '<span style="color:var(--muted);font-size:12px">None detected</span>';

  // Structured fields right panel
  const structEl = document.getElementById('right-struct');
  const structRows = [
    ['Admission Date', s.admission_date], ['Discharge Date', s.discharge_date],
    ['Appointment', s.appointment_date], ['Consultant', s.consultant],
    ['Department', s.department], ['Procedure', s.procedure],
    ['GP Actions', s.gp_actions], ['Adm. Method', s.admission_method],
  ].filter(([,v]) => v);
  structEl.innerHTML = structRows.length
    ? structRows.map(([k,v]) => `<div class="info-row"><div class="info-label">${k}</div><div class="info-value">${v}</div></div>`).join('')
    : '<span style="color:var(--muted)">No structured fields</span>';

  // Pages badge
  if (data.pages_processed > 1) {
    document.getElementById('doc-filename').textContent += ` (${data.pages_processed} pages)`;
  }

  // Follow-up actions
  renderActions('followup-actions', data.follow_up_actions || '');
  renderActions('gp-actions', data.gp_actions || '');

  // Clinical Specifics (type-specific extras: TNM, CD4, OGTT, urgency, etc.)
  const specs = data.clinical_specifics || {};
  const specKeys = Object.keys(specs);
  const specsSection = document.getElementById('right-specifics-section');
  const specsEl = document.getElementById('right-specifics');
  if (specKeys.length > 0) {
    specsSection.style.display = '';
    // Human-readable labels for known keys
    const specLabels = {
      differential_diagnosis: 'Differential Dx',
      urgency: 'Urgency',
      encounter_type: 'Encounter Type',
      assessing_clinician: 'Assessing Clinician',
      tnm_staging: 'TNM Staging',
      cea_value: 'CEA',
      surveillance_schedule: 'Surveillance Schedule',
      treatment_history: 'Treatment History',
      cd4_count: 'CD4 Count',
      viral_load: 'Viral Load',
      art_regimen: 'ART Regimen',
      follow_up: 'Follow-up',
      ogtt_results: 'OGTT Results',
      monitoring_frequency: 'Monitoring Frequency',
      equipment_pip: 'Equipment / PIP',
      surgical_plan: 'Surgical Plan',
      action_for_gp: 'Action for GP',
      key_labs: 'Key Labs',
      paraprotein: 'Paraprotein',
      gp_address: 'GP Address',
      // Renal
      renal_labs: 'Renal Labs',
      next_review: 'Next Review',
      // Paediatric Cardiology
      cardiac_diagnosis: 'Cardiac Diagnosis',
      max_heart_rate: 'Max Heart Rate',
      planned_procedure: 'Planned Procedure',
      current_medication: 'Medication',
      // Gynae / Obstetric
      gravida_parity: 'G/P Status',
      lmp: 'LMP',
      gestational_sac: 'Gestational Sac',
      fetal_pole: 'Fetal Pole',
      scan_diagnosis: 'Scan Diagnosis',
      follow_up_plan: 'Follow-up Plan',
      edd: 'EDD',
      gestational_age: 'Gestational Age',
      reason_for_visit: 'Reason for Visit',
      // Mental Health Inpatient
      mha_section: 'MHA Section',
      primary_diagnosis: 'Primary Diagnosis',
      admission_date: 'Admission Date',
      discharge_date: 'Discharge Date',
      medication_monitoring: 'Medication Monitoring',
      community_follow_up: 'Community Follow-up',
      // Pre-admission
      speciality: 'Speciality',
      clinician: 'Clinician',
      location: 'Location',
      fasting_from: 'Fast From',
      // Ambulance
      incident_number: 'Incident No.',
      presenting_complaint: 'Presenting Complaint',
      working_impression: 'Working Impression',
      news2_score: 'NEWS2 Score',
      conveyance: 'Conveyance',
      first_vitals: 'First Vitals',
      // Ophthalmology
      referral_reason: 'Referral Reason',
      referral_pathway: 'Pathway / Clinic',
      priority: 'Priority',
      provider: 'Provider',
      referred_by: 'Referred By',
      visual_acuity: 'Visual Acuity',
      iop: 'IOP',
      retinopathy_grade: 'DR Grade',
      ophthalmic_diagnosis: 'Diagnosis',
      laser_treatment: 'Laser / PRP',
      ophthalmic_plan: 'Plan',
      neovascularisation: 'Neovascularisation',
      // ED Discharge
      attendance_reason: 'Attendance Reason',
      arrival_method: 'Arrival Method',
      ed_diagnosis: 'ED Diagnosis',
      discharge_method: 'Discharge Method',
      examined_by: 'Examined By',
    };
    specsEl.innerHTML = specKeys.map(k => {
      const label = specLabels[k] || k.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase());
      const rawVal = specs[k];
      const val = (typeof rawVal === 'object') ? Object.entries(rawVal).map(([a,b])=>`${a}: ${b}`).join(' | ') : rawVal;
      return `<div class="info-row"><div class="info-label">${label}</div><div class="info-value" style="word-break:break-word">${val}</div></div>`;
    }).join('');
  } else {
    specsSection.style.display = 'none';
  }

  // Pipeline stages — show status, confidence where available, error reason for partial/error
  const stages = data.pipeline_stages || {};
  document.getElementById('pipeline-stages-display').innerHTML =
    Object.entries(stages).map(([k,v]) => {
      const isDone    = v.status === 'done';
      const isPartial = v.status === 'partial';
      const isError   = v.status === 'error';
      const isSkipped = v.status === 'skipped';
      const color = isDone ? 'var(--nhs-green)'
                  : isPartial ? '#d67e00'
                  : isError ? 'var(--nhs-red)'
                  : 'var(--muted)';
      const confTxt = v.confidence !== undefined ? ' (' + Math.round(v.confidence*100) + '%)' : '';
      const errTxt  = (isPartial || isError) && v.error
        ? `<div style="font-size:10px;color:#c0392b;margin-top:1px;word-break:break-word">${v.error}</div>`
        : '';
      return `<div style="padding:3px 0;border-bottom:1px solid #edf1f7">
        <div style="display:flex;justify-content:space-between">
          <span style="font-weight:600;font-size:12px">${k}</span>
          <span style="color:${color};font-size:12px">${v.status}${confTxt}</span>
        </div>${errTxt}
      </div>`;
    }).join('');

  // Download button
  document.getElementById('btn-download').onclick = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (data.filename || 'result') + '_processed.json';
    a.click();
  };
  document.getElementById('btn-approve').onclick = () => {
    document.getElementById('btn-approve').textContent = '✓ Approved';
    document.getElementById('btn-approve').style.background = '#004f26';
  };
  document.getElementById('btn-emis').onclick = () => {
    alert('Export to EMIS: In production this would push the structured JSON to the EMIS platform via the configured API endpoint.');
  };
}

// ── SNOMED CT Mapping Table ───────────────────────────────────────────────────
// Always shows something:
//  1. AWS Comprehend Medical SNOMED entities (preferred — problems/diagnoses/medications with codes)
//  2. If Comprehend failed/empty: falls back to locally-extracted ICD codes + raw medications
// Each row: category badge | clinical term | code | description | confidence
function renderSnomedTable(problems, medications, diagnoses, icdFallback, medsFallback, comprehendError, usedTermFallback, top3Fallback, usedSummaryFallback, usedDoctypeFallback) {
  const tbody      = document.getElementById('snomed-table-body');
  const emptyMsg   = document.getElementById('snomed-empty');
  const countBadge = document.getElementById('snomed-count-badge');
  const cardHeader = document.querySelector('#snomed-card > div:first-child');
  if (!tbody) return;

  // Remove any old banner so it doesn't stack on re-uploads
  const oldNote = document.getElementById('snomed-fallback-note');
  if (oldNote) oldNote.remove();

  // Build rows from AWS Comprehend primary entities
  let rows = [
    ...problems.map(e   => ({ text: e.text, code: e.snomed_code, desc: e.description, conf: e.confidence, _cat: 'Problem',    _color: '#c0392b', _bg: '#fdf2f2', _source: e.source || 'SNOMED CT' })),
    ...diagnoses.map(e  => ({ text: e.text, code: e.snomed_code, desc: e.description, conf: e.confidence, _cat: 'Diagnosis',  _color: '#1a6636', _bg: '#f2faf5', _source: e.source || 'SNOMED CT' })),
    ...medications.map(e=> ({ text: e.text, code: e.snomed_code, desc: e.description, conf: e.confidence, _cat: 'Medication', _color: '#1a4fa0', _bg: '#f2f5fc', _source: e.source || 'SNOMED CT' })),
  ];

  // ── Banner + source labelling for each fallback layer ───────────────────────
  let bannerText = null, bannerColor = null, badgeLabel = null, badgeColor = null;

  if (rows.length > 0 && rows.some(r => r._source === 'SNOMED CT' || r._source === 'comprehend_medical')) {
    // Primary AWS Comprehend path — no banner needed, just count
    badgeLabel = rows.length + (rows.length === 1 ? ' entity' : ' entities');
    badgeColor = null; // default badge colour

  } else if (usedTermFallback && top3Fallback && top3Fallback.length > 0) {
    // Layer 1 fallback: term-level extraction
    bannerText  = '🔍 Top 3 nearest SNOMED CT matches — term-level extraction (SRS §3.2)';
    bannerColor = '#1a4fa0';
    badgeLabel  = 'Top 3 matches';
    badgeColor  = '#1a4fa0';
    rows = rows.map(r => ({ ...r, _source: 'term_extraction' }));

  } else if (usedSummaryFallback) {
    // Layer 2 fallback: clinician summary used as SNOMED source
    bannerText  = '📋 SNOMED codes mapped from AI-generated clinician summary';
    bannerColor = '#1a6636';
    badgeLabel  = rows.length + ' codes (summary)';
    badgeColor  = '#1a6636';
    rows = rows.map(r => ({ ...r, _source: 'summary_fallback' }));

  } else if (usedDoctypeFallback) {
    // Layer 3 fallback: document-type hardcoded codes (absolute guarantee)
    bannerText  = '📂 SNOMED codes from document-type reference table — document contained no extractable clinical entities';
    bannerColor = '#005EB8';
    badgeLabel  = '3 standard codes';
    badgeColor  = '#005EB8';
    rows = rows.map(r => ({ ...r, _source: 'document_type' }));

  } else if (rows.length === 0) {
    // Still nothing — show ICD/medication local extraction
    (icdFallback || []).forEach(code => {
      rows.push({ text: code, code: code, desc: 'ICD-10 code (locally extracted)', conf: null,
                  _cat: 'ICD Code', _color: '#6a4e9e', _bg: '#f5f0fc', _source: 'Local' });
    });
    (medsFallback || []).forEach(m => {
      rows.push({ text: m.name, code: null, desc: m.dose || 'medication', conf: null,
                  _cat: 'Medication', _color: '#1a4fa0', _bg: '#f2f5fc', _source: 'Local' });
    });
    if (comprehendError) {
      bannerText  = '⚠ AWS Comprehend unavailable — locally extracted codes shown. Error: ' + comprehendError;
      bannerColor = '#d67e00';
    }
    badgeLabel = rows.length + ' local codes';
    badgeColor = 'rgba(214,126,0,0.6)';
  }

  // Insert banner
  if (bannerText && cardHeader) {
    const n = document.createElement('div');
    n.id = 'snomed-fallback-note';
    n.style.cssText = `background:${bannerColor};color:#fff;font-size:11px;padding:5px 14px;text-align:center;font-weight:600;letter-spacing:.2px`;
    n.textContent = bannerText;
    cardHeader.parentElement.insertBefore(n, cardHeader.nextSibling);
  }

  const usingFallback = rows.length > 0 && rows.every(r => !['SNOMED CT','comprehend_medical'].includes(r._source));

  // Count badge
  if (countBadge) {
    if (badgeLabel) countBadge.textContent = badgeLabel;
    else countBadge.textContent = rows.length + (rows.length === 1 ? ' entity' : ' entities');
    if (badgeColor) countBadge.style.background = badgeColor;
  }

  tbody.textContent = '';  // safe clear

  if (!rows.length) {
    if (emptyMsg) { emptyMsg.style.display = ''; }
    tbody.parentElement.style.display = 'none';
    return;
  }
  if (emptyMsg) emptyMsg.style.display = 'none';
  tbody.parentElement.style.display = '';

  rows.forEach((e, idx) => {
    const tr = document.createElement('tr');
    tr.style.cssText = 'border-bottom:1px solid #edf1f7;' + (idx % 2 === 0 ? 'background:#fff' : 'background:#fafbfc');

    // ── Category badge ────────────────────────────────────────────────────────
    const tdCat = document.createElement('td');
    tdCat.style.cssText = 'padding:7px 10px;vertical-align:middle;white-space:nowrap';
    const badge = document.createElement('span');
    badge.style.cssText = `display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:700;color:${e._color};background:${e._bg};border:1px solid ${e._color}30`;
    badge.textContent = e._cat;
    tdCat.appendChild(badge);
    // Small source label — always show for non-primary sources
    if (e._source && e._source !== 'SNOMED CT' && e._source !== 'comprehend_medical') {
      const src = document.createElement('div');
      const srcLabels = {
        'term_extraction':   { label: '🔍 Term Extraction', color: '#1a4fa0' },
        'Term Extraction':   { label: '🔍 Term Extraction', color: '#1a4fa0' },
        'summary_fallback':  { label: '📋 From Summary',    color: '#1a6636' },
        'document_type':     { label: '📂 Doc-type Ref',    color: '#005EB8' },
        'Local':             { label: 'Local Extract',       color: '#888'    },
      };
      const sl = srcLabels[e._source] || { label: e._source, color: '#888' };
      src.style.cssText = `font-size:9px;margin-top:1px;font-weight:600;color:${sl.color}`;
      src.textContent = sl.label;
      tdCat.appendChild(src);
    }

    // ── Clinical term ─────────────────────────────────────────────────────────
    const tdTerm = document.createElement('td');
    tdTerm.style.cssText = 'padding:7px 10px;font-weight:600;color:#222;vertical-align:middle';
    tdTerm.textContent = e.text || '—';

    // ── Code (SNOMED / ICD) ───────────────────────────────────────────────────
    const tdCode = document.createElement('td');
    tdCode.style.cssText = 'padding:7px 10px;vertical-align:middle';
    if (e.code) {
      const codeEl = document.createElement('code');
      const codeColor = e._source === 'Local' ? '#6a4e9e' : '#1a4fa0';
      const codeBg    = e._source === 'Local' ? '#f0eafb' : '#e8f0fe';
      codeEl.style.cssText = `background:${codeBg};color:${codeColor};padding:2px 7px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:.3px;font-family:monospace`;
      codeEl.textContent = e.code;
      tdCode.appendChild(codeEl);
    } else {
      const noMap = document.createElement('span');
      noMap.style.cssText = 'color:#bbb;font-size:11px;font-style:italic';
      noMap.textContent = '—';
      tdCode.appendChild(noMap);
    }

    // ── Description ───────────────────────────────────────────────────────────
    const tdDesc = document.createElement('td');
    tdDesc.style.cssText = 'padding:7px 10px;color:#555;font-size:11px;vertical-align:middle';
    tdDesc.textContent = e.desc || (e.code ? e._source + ' concept' : '—');

    // ── Confidence ────────────────────────────────────────────────────────────
    const tdConf = document.createElement('td');
    tdConf.style.cssText = 'padding:7px 10px;text-align:center;vertical-align:middle';
    if (e.conf != null) {
      const pct = Math.round(e.conf * 100);
      const confSpan = document.createElement('span');
      confSpan.style.cssText = `font-size:11px;font-weight:700;color:${pct >= 70 ? '#1a6636' : pct >= 45 ? '#d67e00' : '#c0392b'}`;
      confSpan.textContent = pct + '%';
      tdConf.appendChild(confSpan);
    } else {
      const dash = document.createElement('span');
      dash.style.cssText = 'color:#ccc;font-size:11px';
      dash.textContent = '—';
      tdConf.appendChild(dash);
    }

    tr.appendChild(tdCat);
    tr.appendChild(tdTerm);
    tr.appendChild(tdCode);
    tr.appendChild(tdDesc);
    tr.appendChild(tdConf);
    tbody.appendChild(tr);
  });
}

// FIX (review comment 7): Use safe DOM construction (textContent) instead of
// innerHTML to prevent XSS from untrusted OCR/entity text returned by the pipeline.
function renderChips(containerId, entities) {
  const el = document.getElementById(containerId);
  el.textContent = '';
  if (!entities.length) {
    const none = document.createElement('span');
    none.style.cssText = 'color:var(--muted);font-size:12px';
    none.textContent = 'None identified';
    el.appendChild(none);
    return;
  }
  entities.forEach(e => {
    const chip = document.createElement('span');
    chip.className = 'snomed-chip';
    // Show full description in tooltip, confidence score if available
    const confPct = e.confidence ? ' (' + (e.confidence*100).toFixed(0) + '%)' : '';
    chip.title = (e.description || e.text || '') + confPct;
    chip.textContent = e.text || '';
    const code = document.createElement('span');
    code.className = 'snomed-code';
    // Show SNOMED code, or 'No map' if Comprehend returned no code
    code.textContent = ' ' + (e.snomed_code ? e.snomed_code : 'No map');
    code.style.color = e.snomed_code ? '' : 'var(--nhs-yellow)';
    chip.appendChild(code);
    el.appendChild(chip);
  });
}

function renderRightEntities(containerId, entities) {
  const el = document.getElementById(containerId);
  el.textContent = '';
  if (!entities.length) {
    const none = document.createElement('span');
    none.style.cssText = 'color:var(--muted);font-size:13px';
    none.textContent = 'None identified';
    el.appendChild(none);
    return;
  }
  entities.forEach(e => {
    const row = document.createElement('div');
    row.style.cssText = 'padding:4px 0;font-size:12px';
    const name = document.createElement('span');
    name.style.fontWeight = '600';
    name.textContent = e.text || '';
    row.appendChild(name);
    const code = document.createElement('span');
    code.style.color = e.snomed_code ? 'var(--muted)' : 'var(--nhs-yellow)';
    code.textContent = ' \u00b7 ' + (e.snomed_code || 'No SNOMED map');
    row.appendChild(code);
    el.appendChild(row);
  });
}

function renderActions(containerId, text) {
  const el = document.getElementById(containerId);
  el.textContent = '';
  if (!text) {
    const none = document.createElement('span');
    none.style.cssText = 'color:var(--muted);font-size:13px';
    none.textContent = 'No actions generated';
    el.appendChild(none);
    return;
  }
  const lines = text.split('\n').filter(l => l.trim());
  let num = 0;
  lines.forEach(l => {
    const clean = l.replace(/^\d+[\.\)]\s*/, '').trim();
    if (!clean) return;
    num++;
    const row = document.createElement('div');
    row.className = 'action-item';
    const numEl = document.createElement('div');
    numEl.className = 'action-num';
    numEl.textContent = String(num);
    const txt = document.createElement('div');
    txt.textContent = clean;
    row.appendChild(numEl);
    row.appendChild(txt);
    el.appendChild(row);
  });
}

// FIX (review comment 6): Accept the clicked element explicitly rather than
// relying on the global `event.target` which is not reliable across all browsers.
function showTab(el, name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

function toggleExpand(el) {
  const body = el.nextElementSibling;
  const chevron = el.querySelector('.chevron');
  body.classList.toggle('open');
  chevron.classList.toggle('open');
}

function copyText(id) {
  const text = document.getElementById(id).innerText.replace('📋','').trim();
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('#'+id+' .copy-btn');
    btn.textContent = '✓'; setTimeout(() => btn.textContent = '📋', 1500);
  });
}

// Convert markdown-style text from LLM into clean readable HTML.
// Handles **bold**, *italic*, bullet lines (- / * / numbered), section headers.
function mdToHtml(text) {
  if (!text) return '';
  // Escape any actual HTML first to prevent XSS
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  let html = escaped
    // ### and ## section headers → styled block (before bold so ** inside headers work)
    .replace(/^#{3}\s+(.+)$/gm, '<strong style="display:block;margin:10px 0 3px;font-size:12px;color:#005eb8;text-transform:uppercase;letter-spacing:.4px">$1</strong>')
    .replace(/^#{1,2}\s+(.+)$/gm, '<strong style="display:block;margin:10px 0 4px;font-size:13px;color:#003087">$1</strong>')
    // **bold** → <strong>
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    // *italic* → <em>
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    // Numbered list items: "1. text" or "1) text"
    .replace(/^\d+[\.\)]\s+(.+)$/gm, '<li style="margin:3px 0">$1</li>')
    // Bullet list items: "- text" or "* text"
    .replace(/^[\-\u2022]\s+(.+)$/gm, '<li style="margin:3px 0">$1</li>')
    // Wrap consecutive <li> blocks in <ul>
    .replace(/(<li[^>]*>[\s\S]*?<\/li>\n?)+/g, m => '<ul style="margin:6px 0 6px 16px;padding:0;list-style:disc">' + m + '</ul>')
    // Double newline → paragraph break
    .replace(/\n{2,}/g, '</p><p style="margin:5px 0">')
    // Single newline → line break
    .replace(/\n/g, '<br>');

  return '<div style="line-height:1.55;font-size:13px">' + html + '</div>';
}

function setText(id, text) {
  const el = document.getElementById(id);
  if(!el) return;
  // preserve copy button if present
  const btn = el.querySelector('.copy-btn');
  el.innerHTML = mdToHtml(text);
  if(btn) el.prepend(btn);
}
function setVal(id, val) { const el = document.getElementById(id); if(el) el.value = val; }

function resetUpload() {
  document.getElementById('new-upload-btn').style.display = 'none';
  document.getElementById('topbar-title').textContent = 'Document Extraction Portal';
  showPanel('upload');
  document.getElementById('file-input').value = '';
}
</script>
</body>
</html>
"""

@app.route("/pages/<doc_id>/<filename>")
def serve_page_image(doc_id, filename):
    """Serve individual page images for the scrollable document preview."""
    page_dir = UPLOAD_DIR / doc_id
    return send_from_directory(str(page_dir), filename)


@app.route("/health")
def health():
    """Health check endpoint — used by Render, Railway, and other PaaS platforms."""
    return jsonify({"status": "ok", "service": "NLP-UK Clinical Portal"})


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/process", methods=["POST"])
def process_document():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f        = request.files["file"]
    ext      = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    doc_id   = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    save_path = UPLOAD_DIR / f"{doc_id}_{filename}"
    f.save(str(save_path))

    try:
        result = run_full_pipeline(doc_id, save_path)
    except Exception as e:
        # FIX (review comment 5): Log full traceback server-side only.
        # Never return stack frames/paths to the browser — leaks internal details.
        app.logger.exception("Document processing failed doc_id=%s filename=%s", doc_id, filename)
        result = {
            "doc_id":    doc_id,
            "filename":  filename,
            "status":    "error",
            "error":     "An internal error occurred while processing the document.",
            # error_detail only included when Flask debug mode is on (local dev only)
            **({"error_detail": str(e)} if app.debug else {}),
        }

    # Persist result
    result_path = RESULTS_DIR / f"{doc_id}_result.json"
    with open(result_path, "w") as fp:
        # Remove blocks (too large) before saving
        r = {k: v for k, v in result.items() if k != "blocks"}
        json.dump(r, fp, indent=2, default=str)

    return jsonify(result)


@app.route("/api/result/<doc_id>")
def get_result(doc_id):
    result_path = RESULTS_DIR / f"{doc_id}_result.json"
    if not result_path.exists():
        return jsonify({"error": "Result not found"}), 404
    with open(result_path) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Clinical Document Portal on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
