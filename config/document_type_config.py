"""
document_type_config.py
------------------------
Central registry of all 21 document types identified from the 40-document
clinical dataset (Batches 1–5, April 2026).

This module is the single source of truth for:
  - Document type names (used by infer_letter_type, extract_clinical_specifics,
    run_bedrock_summarization, and the frontend)
  - Prefix categories (filing system used by the originating GP practice)
  - Per-type confidence thresholds (OBS-004)
  - Key classifier signals per type (mirrors infer_letter_type logic)

Import this from portal.py, track_b_summarization.py, or any module that
needs to reason about document types rather than hardcode strings.

Insights source: /insights/observations.json, /docs/data_insights.md
"""

from __future__ import annotations
from typing import Dict, List

# ── Document type registry ───────────────────────────────────────────────────
# Each entry: { prefix, confidence_threshold, classifier_signals, clinical_domain }

DOCUMENT_TYPES: Dict[str, dict] = {

    # ── Prefix 1: Inpatient ──────────────────────────────────────────────────
    "Discharge Summary": {
        "prefix": "1",
        "confidence_threshold": 0.85,
        "domain": "Acute inpatient",
        "signals": ["discharge summary", "discharge date", "discharging consultant",
                    "length of stay", "discharge summary completed by"],
        "page_strategy": "all",
    },
    "CAMHS Discharge Summary": {
        "prefix": "1",
        "confidence_threshold": 0.85,
        "domain": "Paediatric mental health",
        "signals": ["camhs", "child and adolescent", "mental health service",
                    "brief psychosocial intervention", "bpi"],
        "page_strategy": "all",
    },

    # ── Prefix 2: Specialist outpatient ─────────────────────────────────────
    "Cancer Surveillance Letter": {
        "prefix": "2",
        "confidence_threshold": 0.85,
        "domain": "Oncology",
        "signals": ["surveillance", "adenocarcinoma", "hemicolectomy",
                    "colorectal surveillance", "tnm", "cea", "chemotherapy", "oncology"],
        "page_strategy": "all",
    },
    "HIV / GUM Clinic Letter": {
        "prefix": "2",
        "confidence_threshold": 0.85,
        "domain": "Sexual health / HIV",
        "signals": ["hiv", "gum clinic", "garden clinic", "sexual health",
                    "antiretroviral", "cd4", "viral load", "art regimen",
                    "dolutegravir", "tenofovir"],
        "page_strategy": "all",
    },
    "Maternity / Diabetes Letter": {
        "prefix": "2",
        "confidence_threshold": 0.85,
        "domain": "Obstetric endocrinology",
        "signals": ["gestational diabetes", "antenatal", "maternity", "glucose tolerance",
                    "pip code", "blood glucose monitoring", "midwives"],
        "page_strategy": "all",
    },
    "Surgical Outpatient Letter": {
        "prefix": "2",
        "confidence_threshold": 0.85,
        "domain": "Surgery pre-operative",
        "signals": ["hernia", "supra-umbilical", "upper gi", "open repair", "mesh repair",
                    "brachioplasty", "pre-op", "pre op", "surgical consent"],
        "page_strategy": "all",
    },

    # ── Prefix 3: 111 / Triage ───────────────────────────────────────────────
    "111 First ED Report": {
        "prefix": "3",
        "confidence_threshold": 0.78,
        "domain": "NHS 111 triage / urgent care",
        "signals": ["111 first ed report", "nhs111 encounter", "pathways disposition",
                    "pathways assessment", "attendance activity", "111 first"],
        "page_strategy": "all",
    },

    # ── Prefix 4: Emergency Department ───────────────────────────────────────
    "ED Discharge Letter": {
        "prefix": "4",
        "confidence_threshold": 0.80,
        "domain": "Emergency medicine",
        "signals": ["frimley emergency", "patient discharge letter", "attendance reason",
                    "arrival method", "source of referral", "mode of arrival",
                    "presenting complaint:", "place of accident"],
        "page_strategy": "all",
        "trust_variants": ["Frimley Health", "Royal Berkshire Hospital", "Kettering General"],
    },

    # ── Prefix 5: Ambulance ──────────────────────────────────────────────────
    "Ambulance Clinical Report": {
        "prefix": "5",
        "confidence_threshold": 0.72,  # Lower: multi-column tables, all-caps text (OBS-004)
        "domain": "Prehospital / ambulance",
        "signals": ["south central ambulance service", "patient clinical report",
                    "gp patient report v3", "scas clinician", "news2 score",
                    "pops score", "nature of call", "incident number",
                    "conveyance", "at patient side"],
        "page_strategy": "all",  # CRITICAL: page 6 contains clinical conclusion (OBS-005)
        "versions": ["GP Report for Information v4.7.1 (2 pages)", "GP Patient Report v3.62 (6-9 pages)"],
    },

    # ── Prefix 6: Specialist outpatient (complex) ────────────────────────────
    "Renal / Nephrology Letter": {
        "prefix": "6",
        "confidence_threshold": 0.85,
        "domain": "Nephrology / CKD monitoring",
        "signals": ["nephrologist", "nephrology", "berkshire kidney", "egfr",
                    "creatinine", "renal medicine", "albumin creatinine ratio",
                    "remote monitoring team", "kidney unit"],
        "page_strategy": "all",
    },
    "Paediatric Cardiology Letter": {
        "prefix": "6",
        "confidence_threshold": 0.85,
        "domain": "Paediatric cardiology",
        "signals": ["paediatric cardiol", "paediatric and fetal cardiologist",
                    "congenital heart", "ep mdt", "ablation", "svt",
                    "supraventricular tachycardia", "accessory pathway", "atenolol"],
        "page_strategy": "all",
    },
    "Medication / Prescriber Letter": {
        "prefix": "6/9",
        "confidence_threshold": 0.80,
        "domain": "Online prescribing / weight management",
        "signals": ["expert health", "notification of consultation", "kwikpen",
                    "weight management", "glp-1", "mounjaro", "semaglutide",
                    "ozempic", "wegovy", "weight loss programme"],
        "page_strategy": "page1_clinical",  # OBS-006: pages 2+ are lifestyle Q&A noise
    },

    # ── Prefix 7: Complex outpatient ─────────────────────────────────────────
    "Mental Health Inpatient Discharge": {
        "prefix": "7",
        "confidence_threshold": 0.85,
        "domain": "Psychiatry inpatient",
        "signals": ["mental health inpatient discharge", "prospect park hospital",
                    "crhtt", "cmht", "snowdrop ward", "section 2", "section 3",
                    "mental health act", "inpatient consultant"],
        "page_strategy": "all",
        "sensitive": True,  # OBS-007: contains overdose/MH crisis details
    },
    "Antenatal Discharge Summary": {
        "prefix": "7",
        "confidence_threshold": 0.85,
        "domain": "Obstetrics",
        "signals": ["antenatal discharge", "estimate delivery date", "estimate gestational age",
                    "gravida & parity", "reduced fetal movement", "mdau",
                    "antenatal discharge summary"],
        "page_strategy": "all",
        "sensitive": True,  # OBS-007: may contain bereavement markers (Poppy, neonatal death)
    },
    "Early Pregnancy / Gynaecology Letter": {
        "prefix": "7",
        "confidence_threshold": 0.85,
        "domain": "Early pregnancy / gynaecology",
        "signals": ["ugcc", "epau", "early pregnancy", "gestational sac", "transvaginal",
                    "intrauterine pregnancy", "gravida", "uncertain viability",
                    "emergency gynaecology"],
        "page_strategy": "all",
    },
    "Pre-admission Letter": {
        "prefix": "7",
        "confidence_threshold": 0.85,
        "domain": "Surgical booking",
        "signals": ["fasting instructions", "hospital admission has been scheduled",
                    "do not eat after", "admission instructions", "day surgery unit",
                    "bring this letter with you"],
        "page_strategy": "all",
    },

    # ── Prefix 8: Ophthalmology outpatient ───────────────────────────────────
    "Ophthalmology Letter": {
        "prefix": "8",
        "confidence_threshold": 0.82,
        "domain": "Ophthalmology / medical retina",
        "signals": ["diabetic retinopathy", "medical retina", "ophthalmology",
                    "proliferative retinopathy", "macular oedema", "visual acuity",
                    "iop", "fundus exam", "prp", "panretinal", "neovascularisation",
                    "nvd", "nve", "slit lamp", "ophthalmic"],
        "page_strategy": "all",
        "dr_grading": "R0-R3A, M0-M1, P0-P1",  # NHS Diabetic Retinopathy grading
    },

    # ── Prefix 9: Ophthalmology referral + mixed ──────────────────────────────
    "Ophthalmology Referral": {
        "prefix": "9",
        "confidence_threshold": 0.75,  # Lower: complex prescription tables (OBS-004)
        "domain": "Community eye referral (Evolutio / eRefer)",
        "signals": ["evolutio ophthalmology", "evolutio care innovations",
                    "patient ophthalmology referral", "east berkshire community eye service",
                    "erefer referral", "referral id number", "triager action required",
                    "odtc.co.uk"],
        "page_strategy": "all",
    },

    # ── Prefix 10: General outpatient ─────────────────────────────────────────
    "Psychiatry Outpatient Letter": {
        "prefix": "10",
        "confidence_threshold": 0.85,
        "domain": "Outpatient psychiatry",
        "signals": ["psychiatrist", "psychiatric", "bipolar", "icd10", "icd-10",
                    "quetiapine", "lisdexamfetamine", "consultant psychiatrist"],
        "page_strategy": "all",
    },
    "Haematology Outpatient Letter": {
        "prefix": "10",
        "confidence_threshold": 0.85,
        "domain": "Haematology / oncology",
        "signals": ["haematology", "myeloma", "multiple myeloma", "lenalidomide",
                    "bortezomib", "protein electrophoresis", "paraprotein"],
        "page_strategy": "all",
    },
    "Procedure Report": {
        "prefix": "10",
        "confidence_threshold": 0.85,
        "domain": "Endoscopy / procedural",
        "signals": ["endoscopy", "ogd", "colonoscopy", "gastroscopy", "oesophageal",
                    "colonography", "procedure report", "endoscopist"],
        "page_strategy": "all",
    },
}

# Ordered list of type names for the classifier (priority matters — specific before general)
CLASSIFICATION_ORDER: List[str] = [
    "Ambulance Clinical Report",
    "Ophthalmology Referral",
    "ED Discharge Letter",
    "111 First ED Report",
    "Mental Health Inpatient Discharge",
    "Antenatal Discharge Summary",
    "Cancer Surveillance Letter",
    "HIV / GUM Clinic Letter",
    "Early Pregnancy / Gynaecology Letter",
    "Pre-admission Letter",
    "Maternity / Diabetes Letter",
    "Surgical Outpatient Letter",
    "Procedure Report",
    "CAMHS Discharge Summary",
    "Discharge Summary",
    "Ophthalmology Letter",
    "Renal / Nephrology Letter",
    "Paediatric Cardiology Letter",
    "Psychiatry Outpatient Letter",
    "Haematology Outpatient Letter",
    "Medication / Prescriber Letter",
]

# ── Convenience accessors ────────────────────────────────────────────────────

def get_threshold(letter_type: str) -> float:
    """Return the calibrated confidence threshold for a document type."""
    cfg = DOCUMENT_TYPES.get(letter_type, {})
    return cfg.get("confidence_threshold", 0.85)


def get_page_strategy(letter_type: str) -> str:
    """Return the page processing strategy for a document type.
    'all'             — concatenate all pages (default)
    'page1_clinical'  — use all pages for OCR but limit LLM input to page 1
    """
    cfg = DOCUMENT_TYPES.get(letter_type, {})
    return cfg.get("page_strategy", "all")


def is_sensitive_type(letter_type: str) -> bool:
    """Return True if this document type is flagged as inherently sensitive."""
    cfg = DOCUMENT_TYPES.get(letter_type, {})
    return cfg.get("sensitive", False)


def all_type_names() -> List[str]:
    """Return all 21 registered document type names."""
    return list(DOCUMENT_TYPES.keys())
