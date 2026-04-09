"""
Post-generation validation and rule-based checks for Track B summaries.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib import parse, request


@dataclass
class ValidationCheck:
    check_name: str
    passed: bool
    severity: str
    message: str
    details: Dict[str, Any]
    auto_corrected: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "auto_corrected": self.auto_corrected,
        }


class DrugReferenceDB:
    """
    Lightweight drug reference database with optional OpenFDA fallback.
    """

    def __init__(self, db_path: str = "medical_drug_db.json"):
        self.db_path = db_path
        self.drugs = self._load_local_db()

    def _load_local_db(self) -> Dict[str, Dict[str, Any]]:
        default = {
            "aspirin": {"aliases": ["asa"], "min_dose_mg": 25, "max_dose_mg": 1000},
            "metformin": {"aliases": [], "min_dose_mg": 250, "max_dose_mg": 3000},
            "lisinopril": {"aliases": [], "min_dose_mg": 2.5, "max_dose_mg": 80},
            "atorvastatin": {"aliases": [], "min_dose_mg": 5, "max_dose_mg": 80},
            "clopidogrel": {"aliases": [], "min_dose_mg": 37.5, "max_dose_mg": 300},
            "metoprolol": {"aliases": [], "min_dose_mg": 12.5, "max_dose_mg": 400},
            "amoxicillin": {"aliases": [], "min_dose_mg": 125, "max_dose_mg": 3000},
            "ibuprofen": {"aliases": [], "min_dose_mg": 100, "max_dose_mg": 3200},
            "paracetamol": {"aliases": ["acetaminophen"], "min_dose_mg": 125, "max_dose_mg": 4000},
            "insulin": {"aliases": [], "min_dose_units": 1, "max_dose_units": 300},
        }
        if not os.path.exists(self.db_path):
            return default
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                return {k.lower(): v for k, v in file_data.items()}
        except Exception:
            return default

    def lookup(self, medication_name: str) -> Tuple[bool, Dict[str, Any]]:
        candidate = medication_name.strip().lower()
        if not candidate:
            return False, {}
        if candidate in self.drugs:
            return True, self.drugs[candidate]
        for drug_name, metadata in self.drugs.items():
            aliases = [a.lower() for a in metadata.get("aliases", [])]
            if candidate in aliases:
                return True, {"canonical": drug_name, **metadata}

        # Optional online fallback: OpenFDA exact brand/generic search
        online = self._query_openfda(candidate)
        if online:
            return True, online
        return False, {}

    def _query_openfda(self, medication_name: str) -> Dict[str, Any]:
        query = parse.quote(f'openfda.generic_name:"{medication_name}"')
        url = f"https://api.fda.gov/drug/label.json?search={query}&limit=1"
        try:
            with request.urlopen(url, timeout=2.5) as response:
                data = json.loads(response.read().decode("utf-8"))
                results = data.get("results", [])
                if not results:
                    return {}
                first = results[0].get("openfda", {})
                generic = (first.get("generic_name") or [medication_name])[0].lower()
                return {"canonical": generic, "source": "openfda"}
        except Exception:
            return {}


class MedicalValidationEngine:
    DOSAGE_PATTERN = re.compile(
        r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|units|iu)\s*$",
        re.IGNORECASE,
    )
    DOSAGE_LIMITS = {
        "mg": (0.1, 5000),
        "mcg": (1, 10000),
        "g": (0.01, 10),
        "ml": (0.1, 500),
        "units": (1, 1000),
        "iu": (1, 1000),
    }
    SNOMED_DIAGNOSIS_MAP = {
        "hypertension": "38341003",
        "type 2 diabetes mellitus": "44054006",
        "diabetes": "73211009",
        "asthma": "195967001",
        "pneumonia": "233604007",
        "myocardial infarction": "22298006",
        "nstemi": "401303003",
        "hyperlipidemia": "55822004",
    }
    REQUIRED_SCHEMA = {
        "summary": str,
        "key_points": list,
        "medications": list,
        "diagnoses": list,
        "follow_up_actions": list,
        "confidence_score": (int, float),
    }

    def __init__(self, drug_db: DrugReferenceDB | None = None):
        self.drug_db = drug_db or DrugReferenceDB()
        self.last_report: Dict[str, Any] = {}

    def validate(self, summary_output: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        checks: List[ValidationCheck] = []
        corrected_output = self._auto_correct_common_patterns(summary_output, checks)

        checks.extend(self._validate_schema(corrected_output))
        checks.extend(self._validate_medications(corrected_output.get("medications", []), source_text))
        checks.extend(self._validate_dosages(corrected_output.get("medications", [])))
        checks.extend(self._validate_diagnoses(corrected_output.get("diagnoses", []), source_text))

        hallucination_score, hallu_checks = self._hallucination_checks(corrected_output, source_text)
        checks.extend(hallu_checks)

        failed_errors = [c for c in checks if not c.passed and c.severity == "error"]
        failed_warnings = [c for c in checks if not c.passed and c.severity == "warning"]
        passed = len(failed_errors) == 0
        confidence = self._validation_confidence(checks, hallucination_score)

        self.last_report = {
            "validation_passed": passed,
            "errors": [c.message for c in failed_errors],
            "warnings": [c.message for c in failed_warnings],
            "validation_confidence_score": round(confidence, 4),
            "hallucination_score": round(hallucination_score, 4),
            "audit_log": [c.as_dict() for c in checks],
            "auto_corrections": [c.message for c in checks if c.auto_corrected],
            "corrected_output": corrected_output,
        }
        return self.last_report

    def _auto_correct_common_patterns(self, output: Dict[str, Any], checks: List[ValidationCheck]) -> Dict[str, Any]:
        corrected = dict(output)
        if not isinstance(corrected.get("medications", []), list):
            corrected["medications"] = []
            checks.append(ValidationCheck("auto_correct_medications_type", False, "error", "medications must be a list", {"field": "medications"}))
            return corrected

        normalized_meds = []
        for med in corrected.get("medications", []):
            if isinstance(med, str):
                normalized_meds.append({"name": med.strip(), "dosage": "", "frequency": "", "instructions": ""})
                checks.append(
                    ValidationCheck(
                        "auto_correct_medication_string",
                        True,
                        "info",
                        f"Converted medication string to object: {med}",
                        {"original": med},
                        auto_corrected=True,
                    )
                )
                continue
            med_obj = dict(med)
            if "name" in med_obj:
                med_obj["name"] = str(med_obj["name"]).strip()
            if "dosage" in med_obj and isinstance(med_obj["dosage"], str):
                med_obj["dosage"] = re.sub(r"\s+", " ", med_obj["dosage"]).strip()
                med_obj["dosage"] = re.sub(r"(?i)\b(mg|mcg|ml|iu)\b", lambda m: m.group(1).lower(), med_obj["dosage"])
            normalized_meds.append(med_obj)
        corrected["medications"] = normalized_meds

        score = corrected.get("confidence_score", 0.5)
        if not isinstance(score, (int, float)):
            corrected["confidence_score"] = 0.5
            checks.append(
                ValidationCheck(
                    "auto_correct_confidence_type",
                    True,
                    "info",
                    "Invalid confidence_score type, reset to 0.5",
                    {"original_type": str(type(score))},
                    auto_corrected=True,
                )
            )
        else:
            clamped = max(0.0, min(1.0, float(score)))
            if clamped != score:
                checks.append(
                    ValidationCheck(
                        "auto_correct_confidence_range",
                        True,
                        "info",
                        "Clamped confidence_score to [0, 1]",
                        {"original": score, "corrected": clamped},
                        auto_corrected=True,
                    )
                )
            corrected["confidence_score"] = clamped
        return corrected

    def _validate_schema(self, output: Dict[str, Any]) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        for field, expected_type in self.REQUIRED_SCHEMA.items():
            if field not in output:
                checks.append(ValidationCheck("schema_required_field", False, "error", f"Missing required field: {field}", {"field": field}))
                continue
            if not isinstance(output[field], expected_type):
                checks.append(
                    ValidationCheck(
                        "schema_type_check",
                        False,
                        "error",
                        f"Field '{field}' has wrong type",
                        {"field": field, "expected": str(expected_type), "actual": str(type(output[field]))},
                    )
                )

        for idx, med in enumerate(output.get("medications", [])):
            if not isinstance(med, dict):
                checks.append(ValidationCheck("schema_medication_item", False, "error", "Medication item must be object", {"index": idx}))
                continue
            for med_field in ("name", "dosage", "frequency", "instructions"):
                if med_field not in med:
                    checks.append(ValidationCheck("schema_medication_field", False, "warning", f"Medication missing field '{med_field}'", {"index": idx, "field": med_field}))
        return checks

    def _validate_medications(self, medications: List[Dict[str, Any]], source_text: str) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        source_lower = source_text.lower()
        for idx, med in enumerate(medications):
            name = str(med.get("name", "")).strip()
            if not name:
                checks.append(ValidationCheck("medication_name_present", False, "error", "Medication name is empty", {"index": idx}))
                continue
            found, metadata = self.drug_db.lookup(name)
            checks.append(
                ValidationCheck(
                    "medication_drug_db_lookup",
                    found,
                    "error" if not found else "info",
                    f"Medication '{name}' {'found' if found else 'not found'} in reference DB",
                    {"index": idx, "metadata": metadata},
                )
            )
            in_source = name.lower() in source_lower
            if not in_source:
                token_overlap = any(part in source_lower for part in name.lower().split() if len(part) > 3)
                checks.append(
                    ValidationCheck(
                        "medication_source_crossref",
                        token_overlap,
                        "warning" if not token_overlap else "info",
                        f"Medication '{name}' {'not found' if not token_overlap else 'partially matched'} in source text",
                        {"index": idx},
                    )
                )
        return checks

    def _validate_dosages(self, medications: List[Dict[str, Any]]) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        for idx, med in enumerate(medications):
            dosage = str(med.get("dosage", "")).strip()
            if not dosage:
                checks.append(ValidationCheck("dosage_present", False, "warning", f"Dosage missing for medication '{med.get('name', '')}'", {"index": idx}))
                continue
            match = self.DOSAGE_PATTERN.match(dosage)
            if not match:
                checks.append(ValidationCheck("dosage_format_check", False, "error", f"Invalid dosage format: {dosage}", {"index": idx, "dosage": dosage}))
                continue

            value = float(match.group("value"))
            unit = match.group("unit").lower()
            min_allowed, max_allowed = self.DOSAGE_LIMITS.get(unit, (None, None))
            range_pass = min_allowed is not None and min_allowed <= value <= max_allowed
            checks.append(
                ValidationCheck(
                    "dosage_range_check",
                    range_pass,
                    "error" if not range_pass else "info",
                    f"Dosage {dosage} {'within' if range_pass else 'outside'} expected range",
                    {"index": idx, "min": min_allowed, "max": max_allowed},
                )
            )
        return checks

    def _validate_diagnoses(self, diagnoses: List[Any], source_text: str) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        source_lower = source_text.lower()

        for idx, diagnosis in enumerate(diagnoses):
            dx_text = str(diagnosis).strip().lower()
            if not dx_text:
                checks.append(ValidationCheck("diagnosis_present", False, "error", "Diagnosis entry is empty", {"index": idx}))
                continue

            # Support string like "Hypertension (38341003)"
            snomed_match = re.search(r"\b(\d{6,18})\b", dx_text)
            mapped_snomed = self.SNOMED_DIAGNOSIS_MAP.get(dx_text)
            if snomed_match:
                checks.append(
                    ValidationCheck(
                        "diagnosis_snomed_format",
                        True,
                        "info",
                        f"Diagnosis includes SNOMED code {snomed_match.group(1)}",
                        {"index": idx, "snomed_code": snomed_match.group(1)},
                    )
                )
            elif mapped_snomed:
                checks.append(
                    ValidationCheck(
                        "diagnosis_snomed_map",
                        True,
                        "info",
                        f"Diagnosis '{diagnosis}' mapped to SNOMED {mapped_snomed}",
                        {"index": idx, "snomed_code": mapped_snomed},
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        "diagnosis_snomed_map",
                        False,
                        "warning",
                        f"Diagnosis '{diagnosis}' not mapped to known SNOMED reference set",
                        {"index": idx},
                    )
                )

            in_source = dx_text in source_lower or any(w in source_lower for w in dx_text.split() if len(w) > 4)
            checks.append(
                ValidationCheck(
                    "diagnosis_source_crossref",
                    in_source,
                    "error" if not in_source else "info",
                    f"Diagnosis '{diagnosis}' {'not found' if not in_source else 'found'} in source text",
                    {"index": idx},
                )
            )
        return checks

    def _hallucination_checks(self, output: Dict[str, Any], source_text: str) -> Tuple[float, List[ValidationCheck]]:
        checks: List[ValidationCheck] = []
        source_tokens = self._tokenize(source_text)
        candidates = [output.get("summary", "")]
        candidates.extend(output.get("key_points", []))
        candidates.extend([str(x) for x in output.get("diagnoses", [])])
        candidates.extend([str(m.get("name", "")) for m in output.get("medications", []) if isinstance(m, dict)])
        candidate_tokens = self._tokenize(" ".join(candidates))

        if not candidate_tokens:
            return 0.0, checks

        unsupported = [t for t in candidate_tokens if t not in source_tokens and len(t) > 5]
        hallucination_score = len(unsupported) / max(len(candidate_tokens), 1)
        checks.append(
            ValidationCheck(
                "hallucination_token_overlap",
                hallucination_score < 0.25,
                "error" if hallucination_score >= 0.25 else "info",
                f"Hallucination token score={hallucination_score:.3f}",
                {"unsupported_token_count": len(unsupported), "sample_tokens": unsupported[:15]},
            )
        )

        speculative_patterns = [
            r"(?i)\b(probably|possibly|might have|likely has)\b",
            r"(?i)\b(as an ai|i believe|i think)\b",
        ]
        summary_text = output.get("summary", "")
        for pattern in speculative_patterns:
            hit = re.search(pattern, summary_text) is not None
            checks.append(
                ValidationCheck(
                    "hallucination_speculative_language",
                    not hit,
                    "warning" if hit else "info",
                    f"Speculative language {'detected' if hit else 'not detected'}",
                    {"pattern": pattern},
                )
            )
        return hallucination_score, checks

    def _validation_confidence(self, checks: List[ValidationCheck], hallucination_score: float) -> float:
        if not checks:
            return 0.5
        weighted_total = 0.0
        weighted_pass = 0.0
        for check in checks:
            weight = 1.5 if check.severity == "error" else (1.0 if check.severity == "warning" else 0.5)
            weighted_total += weight
            if check.passed:
                weighted_pass += weight
        base = weighted_pass / max(weighted_total, 1e-6)
        adjusted = (base * 0.7) + ((1 - hallucination_score) * 0.3)
        return max(0.0, min(1.0, adjusted))

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text.lower()))
