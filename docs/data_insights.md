# Dataset Insights & Pipeline Calibration Guide

**Generated:** 2026-04-12  
**Dataset:** 40 clinical PDF documents (Batches 1–5)  
**Analyst:** NLP-UK Pipeline Development  
**Machine-readable version:** [`/insights/observations.json`](../insights/observations.json)

---

## 🔍 1. Corpus Overview

All 40 documents flow into a **single GP practice** — Easthampstead Surgery, Bracknell (ODS: K81087).  
In a production multi-practice deployment, trust → practice routing logic will be required.

**Originating organisations (8 distinct senders):**
| Organisation | Trust Type | Document Types |
|---|---|---|
| Frimley Health NHS FT | Acute | ED discharge, Surgical, Antenatal, Ophthalmology, Haematology |
| Royal Berkshire Hospital NHS FT | Acute | ED discharge, Discharge summary, Ophthalmology, Cardiology |
| Berkshire Healthcare NHS FT | Mental Health | MH Inpatient discharge, Talking Therapies |
| SCAS NHS FT | Ambulance | Ambulance Clinical Reports (v4.7.1, v3.62) |
| University Hospital Southampton | Acute | Paediatric Cardiology |
| Kettering General Hospital | Acute | ED Discharge letter (variant format) |
| Evolutio Care Innovations Ltd | Private (eye referrals) | Ophthalmology Referral (eRefer) |
| Expert Health Ltd | Private (online prescriber) | GLP-1 weight management consultations |

> ⚠️ **ALL 40 documents are scanned PDFs with no embedded text.** AWS Textract OCR is mandatory — pdfplumber returns empty strings for every document.

---

## 📊 2. Document Type Taxonomy (20 Types)

| Prefix | Type | Volume Est. | Key Clinical Domain |
|---|---|---|---|
| 1 | Discharge Summary | Medium | Acute inpatient |
| 1 | CAMHS Discharge Summary | Low | Paediatric mental health |
| 2 | Cancer Surveillance Letter | Low | Oncology |
| 2 | HIV / GUM Clinic Letter | Low | Sexual health |
| 2 | Maternity / Diabetes Letter | Low | Obstetric diabetes |
| 2 | Surgical Outpatient Letter | Low | Pre-operative |
| 3 | 111 First ED Report | Medium | Triage / urgent care |
| 4 | ED Discharge Letter | High | Emergency medicine |
| 5 | Ambulance Clinical Report | Medium | Prehospital care |
| 6 | Renal / Nephrology Letter | Low | CKD monitoring |
| 6 | Paediatric Cardiology Letter | Low | Cardiac (SVT, congenital) |
| 6 | Medication / Prescriber Letter | Medium | GLP-1, weight management |
| 7 | Mental Health Inpatient Discharge | Medium | Psychiatry inpatient |
| 7 | Antenatal Discharge Summary | Low | Obstetrics |
| 7 | Early Pregnancy / Gynaecology | Low | EPAU / early pregnancy |
| 7 | Pre-admission Letter | Low | Surgical booking |
| 8 | Ophthalmology Letter | Medium | Medical retina / diabetic eye |
| 9 | Ophthalmology Referral | Medium | Community eye referral |
| 10 | Psychiatry Outpatient Letter | Low | Outpatient psychiatry |
| 10 | Haematology Outpatient Letter | Low | Myeloma / haematological |

---

## 🩺 3. Key Clinical Patterns

### 3.1 Diabetic Disease Cluster (Dominant)
All 4 ophthalmology documents in the dataset relate to **diabetic retinopathy** — the highest-volume specialist condition in this practice population. Expected to drive repeat clinic letters.

- **DR grading system:** R0–R3A (retinopathy severity) + M0–M1 (maculopathy) + P0–P1 (PRP status)  
  - Example: `R2M1P0` = Moderate DR + Maculopathy + No PRP yet
- **IOP format:** `Right 20 mmHg, Left 17 mmHg`
- **VA formats:** Snellen (`6/24`), decimal (`0.64`), complog (`6/8.7`)
- **Treatment terms:** PRP, NVD laser, NVE laser, intravitreal injection

### 3.2 GLP-1 Weight Management (Growing Trend)
3 Expert Health documents seen (Mounjaro, Wegovy). Online prescribing services produce a **Q&A transcript format** (20+ questions across 5-7 pages). Only page 1 contains clinical data — pages 2+ are lifestyle education content.

### 3.3 Emergency/Urgent Care (~30% of volume)
Three distinct ED letter formats:
- **Frimley Health format:** `Re: SURNAME, Forename` header + arrival method codes in `[brackets]`
- **RBH format:** `Patient Discharge Letter` + MRN as primary ID + full medication table
- **Kettering format:** Free-text `Clinical Summary` narrative + structured diagnosis block

### 3.4 Ambulance Reports (Multi-page, Multi-section)
Two SCAS report versions. The detailed v3.62 spans 6–9 pages:
- **Page 1:** Demographics, GP, incident, crew, consent
- **Page 2:** Main symptom, PMH, medications, NEWS2/POPS
- **Page 3:** Social history  
- **Page 4:** Vital signs time-series table  
- **Page 5:** GCS, airway, circulation assessment  
- **Page 6:** Full body examination + clinical impression + plan + **conveyance decision**

> ⚠️ Page 6 contains the clinical conclusion. Processing must include ALL pages.

### 3.5 Mental Health Sensitivity
MH documents contain high-sensitivity content requiring special handling:
- Serious overdose narratives
- Command hallucinations / suicidal ideation details
- Safeguarding referrals
- Bereavement markers (`Poppy` = neonatal loss)

> **Rule:** These must NOT appear verbatim in patient-facing summaries.

---

## 📋 4. Field Extraction Patterns

### NHS Number Formats (5 variants)
```
456 956 5328         # spaced 3-3-4 (most common — Frimley, RBH, SCAS)
NHS No: 456 956 5328 # labelled inline
NHS Number\n462 213 3695  # label on one line, number on next (111 reports)
476 028 2297, SURNAME, Forename  # SCAS footer format
```

### Patient Name Formats (7 variants)
```
Re: SURNAME, Forename           # Frimley standard
Name: SURNAME, Forename         # RBH Patient Discharge Letter
Patient Name: Forename SURNAME  # Frimley outpatient letters
Re: MR/MISS Forename SURNAME    # Expert Health
Patient Forename / Surname      # SCAS v3.62 separate fields
Dear Mr Forename SURNAME        # RBH outpatient body
SURNAME, Forename (standalone)  # 111 First ED Report
```

### Date of Birth Formats
```
16/03/1975    # DD/MM/YYYY (most common)
16 Jan 1996   # DD MMM YYYY (Kettering ED)
23-Jun-1981   # DD-MMM-YYYY (SCAS)
Born 22-Feb-1996  # 111 First ED Report
```

### Hospital Number Formats (by trust)
```
MRN: 11115790      # Frimley (7-digit)
MRN Number: 5216236 # RBH (7-digit)
Hospital Number: 10848071  # Frimley outpatient (8-digit)
Hospital no. 005203351     # RBH ophthalmology (9-digit)
Referral ID Number: 1928141 # Evolutio system
```

---

## ⚡ 5. OCR Confidence by Document Type

| Document Type | Expected Confidence | Reason |
|---|---|---|
| Frimley outpatient letters | High (0.90+) | Clean laser print, standard layout |
| RBH discharge letters | High (0.88+) | Clean print, structured fields |
| Expert Health | High (0.88+) | Digital-origin scan |
| Kettering ED letter | Medium (0.82+) | Mixed font sizes |
| Evolutio ophthalmology referral | Medium (0.75+) | Complex prescription tables |
| SCAS ambulance reports | Medium (0.72+) | Multi-column tables, all-caps text |
| 111 First ED reports | Medium (0.78+) | Dense text, small font |

**Current global threshold:** 0.85 — causes false "review required" flags for ambulance and ophthalmology referral documents.

**Recommended:** Per-type thresholds (see `CONFIDENCE_THRESHOLDS` dict in `portal.py`).

---

## 🔒 6. Safeguarding & Sensitive Data Rules

The following fields must be detected and protected:

| Indicator | Action |
|---|---|
| `Poppy` bereavement marker | Do not include in patient summary |
| Neonatal death in PMH | Summarise clinically, not verbatim |
| Safeguarding referral submitted | Flag for GP review, exclude from patient summary |
| MCA S5/S6 invoked | Retain in clinician summary only |
| Police referral as source | Neutral language in patient summary |
| Overdose with ICU admission | Clinical summary only — paraphrase for patient |
| Command hallucinations / suicidal ideation | Paraphrase in patient summary |

---

## 🔧 7. Pending Code Improvements (OBS-004 through OBS-010)

| ID | Status | Description |
|---|---|---|
| OBS-004 | ✅ Done | Per-type confidence thresholds in `CONFIDENCE_THRESHOLDS` dict |
| OBS-005 | ✅ Done | All pages concatenated via `pdf_to_images()` |
| OBS-006 | ✅ Done | Expert Health: limit Bedrock input to page 1 clinical content |
| OBS-007 | ✅ Done | Sensitivity clause added to patient summary prompt |
| OBS-008 | ✅ Done | `extract_hospital_trust()` helper function |
| OBS-009 | ✅ Done | DR grade extraction in `extract_clinical_specifics()` |
| OBS-010 | ✅ Done | `ARRIVAL_METHOD_CODES` lookup dict |

---

## 📈 8. Preprocessing Recommendations

Based on OCR quality observations:

1. **Deskewing:** Already implemented in `preprocessing.py` — effective for most scans
2. **Resolution:** 2x zoom factor in `pdf_to_images()` is appropriate — produces ~150-200 DPI equivalent
3. **Multi-column tables:** Textract TABLES feature type is enabled — captures vital signs, medication tables
4. **Noise filtering:** Frimley footer text (Ministry of Defence, registered office) and Evolutio footer should be stripped before LLM summarisation to save tokens
5. **Expert Health Q&A pages:** Should be included in concatenated text but Bedrock input should be capped to avoid lifestyle education content dominating the summary

---

*This document is auto-generated from dataset analysis and updated with each new batch of documents. Do not edit manually — update `insights/observations.json` instead.*
