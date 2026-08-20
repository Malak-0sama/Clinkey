"""Evidence-support scoring: similarity ≠ support.

This is a deterministic medical-support heuristic (not a calibrated NLI model).
It fails closed on topic mismatch, polarity conflict, and missing required
aspects (e.g. dose). Cross-encoder NLI is used only if installed, and only
on the already-bounded candidate set.
"""
from __future__ import annotations

import re

from ..query.analyzer import QueryAnalyzer
from .entity import find_query_entities
from .models import Evidence

_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "what", "how", "recommended", "management", "treatment", "treat", "exact",
}

_NEG = re.compile(r"\b(?:do not|not recommended|contraindicated|should not)\b", re.I)
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)\b", re.I)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?\b)", re.I)
_RATIO = re.compile(
    r"\b(?:odds ratio|relative risk|risk ratio|hazard ratio|rr|or|hr)\s*(?:of|=|:)?\s*\d+(?:\.\d+)?\b",
    re.I,
)
_COUNT = re.compile(
    r"\b\d[\d,.]*\s*(?:hundred|thousand|million|billion|patients?|people|adults?|cases?|deaths?)\b",
    re.I,
)

_NLI = None


def _nli():
    global _NLI
    if _NLI is False:
        return None
    if _NLI is not None:
        return _NLI
    try:
        from sentence_transformers import CrossEncoder

        _NLI = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        return _NLI
    except Exception:  # noqa: BLE001
        _NLI = False
        return None


def distinctive_terms(query: str) -> set[str]:
    toks = set(re.findall(r"[a-z0-9]+", (query or "").lower())) - _STOP
    return {t for t in toks if len(t) >= 5}


def topic_compatible(query: str, evidence: Evidence, analyzer: QueryAnalyzer | None = None) -> bool:
    return support_score(query, evidence, analyzer) >= 0.35


def support_score(query: str, evidence: Evidence, analyzer: QueryAnalyzer | None = None) -> float:
    """Return support in [0, 1]. 0 = no support / wrong topic."""
    analyzer = analyzer or QueryAnalyzer()
    analysis = analyzer.analyze(query)
    text = evidence.chunk.text
    tl = text.lower()
    ql = (query or "").lower()

    requested_sources = {
        r"\bNICE\b|\bNational Institute for Health and Care Excellence\b": "nice",
        r"\bWHO\b|\bWorld Health Organization\b": "who",
        r"\bCDC\b|\bCenters for Disease Control\b": "cdc",
        r"\bUSPSTF\b": "uspstf",
        r"\bNHS\b": "nhs",
        r"\bFDA\b|\bFood and Drug Administration\b": "fda",
        r"\bEMA\b|\bEuropean Medicines Agency\b": "ema",
        r"\bESC\b|\bEuropean Society of Cardiology\b": "esc",
        r"\bAHA\b|\bAmerican Heart Association\b": "aha",
    }
    organization = evidence.chunk.organization.lower()
    source_id = evidence.chunk.source_id.lower()
    for pattern, expected in requested_sources.items():
        if re.search(pattern, query or "") and expected not in organization and expected not in source_id:
            return 0.0

    if re.search(r"\bpercent(?:age)?\b|\bproportion\b", ql) and not _PERCENT.search(text):
        return 0.20
    if re.search(r"\bodds ratio\b|\brelative risk\b|\brisk ratio\b|\bhazard ratio\b", ql) and not _RATIO.search(text):
        return 0.20
    if re.search(r"\bexact (?:number|count)\b|\bhow many (?:patients|people|adults|cases|deaths)\b", ql) and not _COUNT.search(text):
        return 0.20

    rare = distinctive_terms(query)
    ents = find_query_entities(query)
    topic_syn = {
        "hypertension": ("ace", "arb", "ccb", "calcium-channel", "blood pressure", "antihypertens", "hypertension"),
        "diabetes": ("insulin", "glucose", "metformin", "hba1c", "diabetes"),
        "infectious_disease": ("malaria", "influenza", "hiv", "infection"),
    }

    topic_ok = False
    if ents and any(e in tl for e in ents):
        topic_ok = True
    specific_topics = [
        topic for topic in analysis.topics
        if topic not in {"medication", "lifestyle"}
    ]
    specific_topic_hit = False
    if analysis.topics:
        for topic in analysis.topics:
            hit = (
                topic.replace("_", " ") in tl
                or topic in tl
                or any(synonym in tl for synonym in topic_syn.get(topic, ()))
            )
            if hit:
                topic_ok = True
                if topic in specific_topics:
                    specific_topic_hit = True
    if specific_topics and analysis.intents and not specific_topic_hit:
        return 0.0
    if rare and any(term in tl for term in rare):
        topic_ok = True
    if not topic_ok and (rare or analysis.topics):
        return 0.0

    # polarity: query asks for recommend, evidence only forbids
    if re.search(r"\brecommend|management|treat", ql) and _NEG.search(text) and not re.search(
        r"\brecommend(?!ed not)", tl
    ):
        if not re.search(r"\bnot\b", ql):
            return 0.15

    score = 0.4 if topic_ok else 0.0
    if analysis.keywords and any(k in tl for k in analysis.keywords):
        score += 0.15
    if analysis.intents:
        intent_ok = {
            "dosage": bool(_DOSE.search(text) or "dose" in tl),
            "treatment": bool(re.search(r"\bstep\s*[123]\b|\boffer\b|\brecommend\w*\b|\btreat\w*\b|\btherap\w*\b|\binhibitor\b|\bblocker\b|\bmanage\w*\b|\bcombine\b|\badd\b", tl)),
            "symptoms": bool(re.search(r"symptom|sign|pain|thirst", tl)),
            "diagnosis": bool(re.search(r"diagnos|mmhg|screen|measure|threshold", tl)),
            "definition": bool(re.search(r"\bis a\b|\bcondition\b|\bdefined\b|\bcalled\b|\bmmhg\b", tl)),
            "prevention": bool(re.search(r"prevent|avoid|lifestyle|diet|exercise", tl)),
            "causes": bool(re.search(r"cause|risk factor|result", tl)),
            "complications": bool(re.search(r"complication|damage|stroke|failure|heart attack", tl)),
            "risk": bool(re.search(r"risk|factor|likelihood", tl)),
            "comparison": bool(re.search(r"compare|difference|versus|than", tl)),
            "side_effects": bool(re.search(r"side effect|adverse|reaction", tl)),
        }
        matched_intents = [i for i in analysis.intents if intent_ok.get(i, False)]
        if matched_intents:
            score += 0.2
        elif "dosage" in analysis.intents or re.search(r"\bdose|dosage|how much\b", ql):
            if not _DOSE.search(text):
                return 0.2
        else:
            return min(score, 0.25)
    if re.search(r"\bdose|dosage|how much\b", ql) and not _DOSE.search(text):
        return min(score, 0.25)

    if re.search(r"\bthreshold\b|\bdefine\w*\b", ql):
        if not re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\s*mmhg\b|\b\d{2,3}\s*mmhg\b|\bdefined\b|\bcondition\b", tl):
            return min(score, 0.20)

    if re.search(r"\bunder\s*55\b|\byounger than (?:age )?55\b|\bbelow 55\b", ql):
        if not re.search(r"\bunder\s*55\b|\byounger than 55\b|\badults? under 55\b", tl):
            return min(score, 0.20)

    if re.search(r"\bafrican\b|\bcaribbean\b", ql):
        if not re.search(r"\bafrican\b|\bcaribbean\b", tl):
            return min(score, 0.20)

    if re.search(r"\bnot controlled after step\s*1\b|\buncontrolled after step\s*1\b", ql):
        if not re.search(r"\bstep\s*2\b|\bif blood pressure is not controlled\b", tl):
            return min(score, 0.15)

    if re.search(r"\bstep\s*2\b", ql) and not re.search(r"step\s*2|calcium-channel|\bccb\b", tl):
        return min(score, 0.30)
    if re.search(r"\bstep\s*3\b", ql) and not re.search(r"step\s*3|thiazide", tl):
        return min(score, 0.30)
    if re.search(r"\bstep\s*1\b|first[- ]line|initial (?:drug|medication|medicine|therapy|treatment)|under 55|under-55|younger than (?:age )?55", ql):
        if not re.search(r"step\s*1|ace inhibitor|\barb\b|under 55|calcium-channel", tl):
            return min(score, 0.30)

    section = evidence.chunk.section_title.lower()
    if "treatment" in analysis.intents and re.search(r"treat|recommend|management", section):
        score += 0.1
    if evidence.chunk.organization.upper() in {"NICE", "WHO", "CDC", "USPSTF"}:
        score += 0.05

    nli = _nli()
    if nli is not None:
        try:
            # NLI: premise=evidence, hypothesis=query-as-statement
            logits = nli.predict([(text[:512], query)])
            # deberta-nli typically [contradiction, entailment, neutral]
            import numpy as np

            arr = np.asarray(logits).reshape(-1)
            if arr.size >= 3:
                entail = float(arr[1])
                score = max(score, 0.5 + 0.5 * (entail > 0))
        except Exception:  # noqa: BLE001
            pass

    return max(0.0, min(1.0, score))


def filter_supported(query: str, pool: list[Evidence], min_support: float = 0.35) -> list[Evidence]:
    out = []
    for e in pool:
        e.support_score = support_score(query, e)
        if e.support_score >= min_support:
            out.append(e)
    return out