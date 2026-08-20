"""Evidence sufficiency: relevant ≠ answerable."""
from __future__ import annotations

import re

from .models import Evidence

_INTENT_REQUIRED = {
    "dosage": [r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)\b", r"\bdose\b", r"\bdosage\b"],
    "treatment": [r"\btreat", r"\btherap", r"\bmedicat", r"\binhibitor", r"\bblocker", r"\binsulin"],
    "diagnosis": [r"\bdiagnos", r"\bmmhg\b", r"\bscreen", r"\btest", r"\bmeasure"],
    "symptoms": [r"\bsymptom", r"\bsign", r"\bpain", r"\bthirst", r"\bfatigue", r"\bwheez"],
    "prevention": [r"\bprevent", r"\blifestyle", r"\bdiet", r"\bexercise", r"\bsalt"],
}


def assess_sufficiency(query: str, evidence: list[Evidence], intents: list[str] | None = None) -> dict:
    corpus = " ".join(e.chunk.text.lower() for e in evidence)
    intents = intents or []
    missing: list[str] = []
    for intent in intents:
        pats = _INTENT_REQUIRED.get(intent)
        if not pats:
            continue
        if not any(re.search(p, corpus) for p in pats):
            missing.append(intent)
    # dose question without numeric dose in evidence
    if re.search(r"\b(?:dose|dosage|how much|mg)\b", (query or "").lower()):
        if not re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|mmhg)\b", corpus):
            missing.append("numeric_dose")
    q = (query or "").lower()
    if re.search(r"step\s*1.*step\s*2.*step\s*3|steps?\s*1\s*,\s*2\s*,?\s*(?:and\s+)?3", q):
        for step, pat in (("step1", r"step\s*1"), ("step3", r"step\s*3")):
            if not re.search(pat, corpus):
                missing.append(step)
    if "treatment" in intents or re.search(r"\brecommend\w*\b|\bfirst[- ]line\b|\bstep\s*[123]\b", q):
        if not re.search(r"\bstep\s*[123]\b|\boffer\b|\brecommend\w*\b|\binhibitor\b|\bblocker\b|\bcombine\b|\badd\b|\btreat\w*\b", corpus):
            missing.append("direct_treatment_recommendation")
    if re.search(r"\bnot controlled after step\s*1\b|\buncontrolled after step\s*1\b", q):
        if not re.search(r"\bstep\s*2\b|\bif blood pressure is not controlled\b", corpus):
            missing.append("step2_recommendation")
    if re.search(r"\bunder\s*55\b|\byounger than (?:age )?55\b|\bbelow 55\b", q):
        if not re.search(r"\bunder\s*55\b|\byounger than 55\b", corpus):
            missing.append("under55_population")
    if re.search(r"\bafrican\b|\bcaribbean\b", q):
        if not re.search(r"\bafrican\b|\bcaribbean\b", corpus):
            missing.append("ethnicity_population")
    if re.search(r"\bthreshold\b|\bdefine\w*\b", q) and "blood pressure" in q:
        if not re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\s*mmhg\b|\b\d{2,3}\s*mmhg\b", corpus):
            missing.append("blood_pressure_threshold")
    requested_source = None
    original = query or ""
    if re.search(r"\bNICE\b", original):
        requested_source = "nice"
    elif re.search(r"\bWHO\b", original):
        requested_source = "who"
    if requested_source and not any(
        requested_source in e.chunk.organization.lower() or requested_source in e.chunk.source_id.lower()
        for e in evidence
    ):
        missing.append("requested_source")
    missing = list(dict.fromkeys(missing))
    sufficient = not missing and bool(evidence)
    return {
        "sufficient": sufficient,
        "missing_aspects": missing,
        "status": "SUFFICIENT" if sufficient else "INSUFFICIENT_EVIDENCE",
        "policy": "abstain_if_requested_aspects_missing",
    }