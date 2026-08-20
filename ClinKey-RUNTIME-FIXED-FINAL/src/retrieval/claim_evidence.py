from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..query.analyzer import QueryAnalyzer
from .models import Evidence
from .support import distinctive_terms, support_score

PERCENTAGE_OUTCOME = "NUMERICAL_OUTCOME_PERCENTAGE"
MORTALITY_RATE = "MORTALITY_RATE"
RISK_EFFECT = "RISK_EFFECT"
RATIO_EFFECT = "ODDS_OR_RELATIVE_RISK"
THRESHOLD = "EXACT_THRESHOLD"
DOSAGE = "EXACT_DOSAGE"
NUMERICAL_OUTCOME = "NUMERICAL_OUTCOME"
TREATMENT_EFFECT = "TREATMENT_EFFECT"
TREATMENT_RECOMMENDATION = "TREATMENT_RECOMMENDATION"
GENERAL = "GENERAL_FACT"

_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?\b)", re.I)
_RATE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:per|/)\s*\d[\d,]*\b", re.I)
_COUNT = re.compile(
    r"\b\d[\d,.]*\s*(?:hundred|thousand|million|billion|patients?|people|adults?|cases?|deaths?)\b",
    re.I,
)
_RATIO = re.compile(
    r"\b(?:odds ratio|relative risk|risk ratio|hazard ratio|rr|or|hr)\s*(?:of|=|:)?\s*\d+(?:\.\d+)?\b",
    re.I,
)
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)(?:\s*/\s*(?:day|dose|kg))?\b", re.I)
_BP = re.compile(r"\b\d{2,3}\s*/\s*\d{2,3}\s*mmhg\b|\b\d{2,3}\s*mmhg\b", re.I)
_TREATMENT = re.compile(
    r"\bstep\s*[123]\b|\boffer\b|\brecommend\w*\b|\btreat\w*\b|\btherap\w*\b|"
    r"\bmedicat\w*\b|\binhibitor\b|\bblocker\b|\bcombine\b|\badd\b|\bmanage\w*\b",
    re.I,
)

_TOPIC_ALIASES = {
    "hypertension": ("hypertension", "blood pressure", "antihypertens", "ace inhibitor", "arb", "calcium-channel", "ccb"),
    "diabetes": ("diabetes", "diabetic", "blood glucose", "blood sugar", "insulin", "metformin", "hba1c"),
    "cardiovascular": ("cardiovascular", "heart", "cardiac", "coronary", "stroke"),
    "cancer": ("cancer", "tumor", "tumour", "malignan", "carcinoma", "oncology"),
    "obesity": ("obesity", "obese", "overweight", "body mass", "bmi"),
    "vaccination": ("vaccine", "vaccination", "immunization", "immunisation"),
    "tobacco": ("tobacco", "smoking", "nicotine", "cigarette", "vaping"),
    "infectious_disease": ("infection", "infectious", "virus", "bacteria", "influenza", "malaria", "hiv"),
    "anemia": ("anemia", "anaemia", "haemoglobin", "hemoglobin", "iron deficiency"),
    "asthma": ("asthma", "wheez", "inhaler"),
    "respiratory": ("copd", "bronchitis", "emphysema", "respiratory", "lung disease"),
    "neurology": ("neurolog", "dementia", "alzheimer", "parkinson", "epilepsy", "seizure", "migraine", "narcolepsy"),
    "mental_health": ("depression", "anxiety", "mental health", "bipolar"),
}

_SOURCE_PATTERNS = (
    (r"\bNICE\b|\bNational Institute for Health and Care Excellence\b", "nice"),
    (r"\bWHO\b|\bWorld Health Organization\b", "who"),
    (r"\bCDC\b|\bCenters for Disease Control\b", "cdc"),
    (r"\bUSPSTF\b", "uspstf"),
    (r"\bFDA\b|\bFood and Drug Administration\b", "fda"),
    (r"\bEMA\b|\bEuropean Medicines Agency\b", "ema"),
    (r"\bNHS\b", "nhs"),
    (r"\bNIH\b|\bNational Institutes of Health\b", "nih"),
    (r"\bESC\b|\bEuropean Society of Cardiology\b", "esc"),
    (r"\bAHA\b|\bAmerican Heart Association\b", "aha"),
    (r"\bADA\b|\bAmerican Diabetes Association\b", "ada"),
    (r"\bIDSA\b|\bInfectious Diseases Society of America\b", "idsa"),
)

_ENTITY_STOP = {
    "exact", "percentage", "percent", "patients", "patient", "adults", "adult",
    "achieve", "according", "guideline", "guidance", "treatment", "treated",
    "recommended", "recommendation", "control", "controlled", "uncontrolled",
    "after", "before", "step", "effect", "effects", "effectiveness", "outcome",
    "outcomes", "mortality", "death", "deaths", "reduction", "relative", "risk",
    "ratio", "threshold", "cutoff", "number", "value", "source", "clinical",
    "medication", "therapy", "management", "blood", "pressure", "dose", "dosage",
    "optimal", "reported", "provide", "given", "used",
    "nice", "world", "health", "organization", "fda", "cdc", "nhs", "nih",
}


@dataclass
class ClaimRequirement:
    claim_text: str
    fact_type: str
    requested_source: str = ""
    topics: list[str] = field(default_factory=list)
    entity_terms: list[str] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    outcome_concepts: list[str] = field(default_factory=list)
    atomic: bool = False

    def to_dict(self) -> dict:
        return {
            "claim": self.claim_text,
            "required_fact_type": self.fact_type,
            "requested_source": self.requested_source,
            "topics": list(self.topics),
            "entity_terms": list(self.entity_terms),
            "steps": list(self.steps),
            "outcome_concepts": list(self.outcome_concepts),
            "atomic": self.atomic,
        }


@dataclass
class EvidenceAssessment:
    evidence: Evidence
    relevant_topic: bool
    relevant_intent: bool
    relevant_source: bool
    terminology_match: bool
    section_relevant: bool
    directly_supports_claim: bool
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ClaimEvidenceResult:
    requirement: ClaimRequirement
    supporting_evidence: list[Evidence]
    assessments: list[EvidenceAssessment]

    @property
    def supported(self) -> bool:
        return bool(self.supporting_evidence)


def extract_claim_requirement(query: str, intents: list[str] | None = None) -> ClaimRequirement:
    original = query or ""
    q = original.lower()
    analysis = QueryAnalyzer().analyze(original)
    intents = intents or analysis.intents
    requested_source = ""
    for pattern, source in _SOURCE_PATTERNS:
        if re.search(pattern, original):
            requested_source = source
            break
    topics = [topic for topic in analysis.topics if topic not in {"medication", "lifestyle"}]
    entity_terms = []
    if not topics:
        entity_terms = sorted(
            term for term in distinctive_terms(original)
            if term not in _ENTITY_STOP and not term.isdigit()
        )
    steps = sorted({int(step) for step in re.findall(r"\bstep\s*([123])\b", q)})
    if re.search(
        r"\b(?:not controlled|uncontrolled)\b.*\bafter step\s*1\b|"
        r"\bafter step\s*1\b.*\b(?:not controlled|uncontrolled|fails?|failed)\b",
        q,
    ):
        steps = [2]
    outcome_concepts = []
    concept_patterns = (
        ("blood_pressure_control", r"\b(?:achiev\w*|attain\w*|reach\w*)?\s*(?:blood pressure|bp)?\s*control\w*\b|\bcontrolled blood pressure\b"),
        ("mortality", r"\bmortal\w*\b|\bdeath\w*\b"),
        ("reduction", r"\breduc\w*\b|\blower\w*\b|\bdecreas\w*\b"),
        ("risk", r"\brisk\b"),
        ("response", r"\brespond\w*\b|\bresponse\b"),
        ("survival", r"\bsurviv\w*\b"),
        ("unawareness", r"\bunaware\b|\bundiagnos\w*\b"),
        ("adverse_event", r"\badverse event\w*\b|\bside effect\w*\b"),
        ("treatment_effect", r"\btreatment effect\w*\b|\beffectiveness\b|\bclinical outcome\w*\b"),
        ("prevalence", r"\bprevalence\b|\b(?:what|which) percentage of (?:adults|people|patients) (?:have|are living with|are affected by)\b"),
    )
    for name, pattern in concept_patterns:
        if re.search(pattern, q):
            outcome_concepts.append(name)
    asks_exact = bool(re.search(r"\bexact\b|\bhow many\b|\bhow much\b|\bwhat (?:is|was) the (?:number|value|rate)\b", q))
    asks_percentage = bool(re.search(r"\bpercent(?:age)?\b|\bproportion\b", q))
    asks_mortality_rate = bool(re.search(r"\bmortal\w*\b|\bdeath rate\b", q) and (asks_exact or asks_percentage or "rate" in q))
    asks_ratio = bool(re.search(r"\bodds ratio\b|\brelative risk\b|\brisk ratio\b|\bhazard ratio\b", q))
    asks_risk_effect = bool(re.search(r"\brisk reduction\b|\breduc\w* (?:the )?risk\b", q))
    asks_threshold = bool(re.search(r"\bthreshold\b|\bcut[- ]?off\b|\bwhat .*\breading\b.*\bconsidered\b", q))
    asks_dosage = bool(re.search(r"\bdose\b|\bdosage\b|\bhow much\b.*\b(?:drug|medication|medicine)\b", q))
    asks_count = bool(re.search(r"\bexact (?:number|count)\b|\bhow many (?:patients|people|adults|cases|deaths)\b", q))
    asks_treatment_effect = bool(re.search(r"\btreatment effect\w*\b|\beffectiveness\b|\bclinical outcome\w*\b|\bachiev\w*\b|\bresponse rate\b", q))
    if asks_mortality_rate:
        fact_type = MORTALITY_RATE
    elif asks_ratio:
        fact_type = RATIO_EFFECT
    elif asks_risk_effect and (asks_exact or asks_percentage):
        fact_type = RISK_EFFECT
    elif asks_percentage:
        fact_type = PERCENTAGE_OUTCOME
    elif asks_threshold:
        fact_type = THRESHOLD
    elif asks_dosage:
        fact_type = DOSAGE
    elif asks_count or asks_exact:
        fact_type = NUMERICAL_OUTCOME
    elif asks_treatment_effect:
        fact_type = TREATMENT_EFFECT
    elif "treatment" in intents or re.search(r"\brecommend\w*\b|\bstep\s*[123]\b|\bfirst[- ]line\b", q):
        fact_type = TREATMENT_RECOMMENDATION
    else:
        fact_type = GENERAL
    atomic = fact_type in {
        PERCENTAGE_OUTCOME, MORTALITY_RATE, RISK_EFFECT, RATIO_EFFECT,
        THRESHOLD, DOSAGE, NUMERICAL_OUTCOME, TREATMENT_EFFECT,
    } or bool(steps) or bool(requested_source and fact_type != GENERAL)
    return ClaimRequirement(
        claim_text=original,
        fact_type=fact_type,
        requested_source=requested_source,
        topics=topics,
        entity_terms=entity_terms,
        steps=steps,
        outcome_concepts=outcome_concepts,
        atomic=atomic,
    )


def validate_claim_evidence(
    query: str,
    evidence: list[Evidence],
    intents: list[str] | None = None,
) -> ClaimEvidenceResult:
    requirement = extract_claim_requirement(query, intents)
    assessments = [_assess(requirement, item, query) for item in evidence]
    supporting = [item.evidence for item in assessments if item.directly_supports_claim]
    supporting.sort(key=lambda item: (-item.clinical_relevance_score, -item.rerank_score))
    for index, item in enumerate(supporting):
        item.rank = index + 1
        item.relevance_score = max(item.relevance_score, item.clinical_relevance_score)
    return ClaimEvidenceResult(requirement, supporting, assessments)


def _assess(requirement: ClaimRequirement, evidence: Evidence, query: str) -> EvidenceAssessment:
    chunk = evidence.chunk
    text = chunk.text or ""
    corpus = " ".join((text, chunk.section_title, chunk.document_name, chunk.organization, chunk.source_id)).lower()
    topic_relevant = _topic_relevant(requirement, corpus)
    source_relevant = not requirement.requested_source or requirement.requested_source in corpus
    intent_relevant = _intent_relevant(requirement.fact_type, text, chunk.section_title)
    terminology_match = _terminology_match(requirement, corpus)
    section_relevant = _section_relevant(requirement.fact_type, chunk.section_title, text)
    direct = False
    if topic_relevant and source_relevant and intent_relevant and terminology_match:
        direct = _direct_support(requirement, evidence, query)
    score = (
        0.30 * float(topic_relevant)
        + 0.15 * float(intent_relevant)
        + 0.10 * float(source_relevant)
        + 0.10 * float(terminology_match)
        + 0.10 * float(section_relevant)
        + 0.25 * float(direct)
    )
    reasons = []
    if not topic_relevant:
        reasons.append("topic_mismatch")
    if not source_relevant:
        reasons.append("source_mismatch")
    if not intent_relevant:
        reasons.append("intent_mismatch")
    if not terminology_match:
        reasons.append("terminology_mismatch")
    if topic_relevant and source_relevant and not direct:
        reasons.append("requested_fact_not_supported")
    evidence.claim_evaluated = True
    evidence.relevant_topic = topic_relevant
    evidence.relevant_intent = intent_relevant
    evidence.relevant_source = source_relevant
    evidence.directly_supports_claim = direct
    evidence.required_fact_type = requirement.fact_type
    evidence.claim_atomic = requirement.atomic
    evidence.clinical_relevance_score = round(score, 4)
    evidence.claim_support_reasons = reasons
    return EvidenceAssessment(
        evidence=evidence,
        relevant_topic=topic_relevant,
        relevant_intent=intent_relevant,
        relevant_source=source_relevant,
        terminology_match=terminology_match,
        section_relevant=section_relevant,
        directly_supports_claim=direct,
        score=score,
        reasons=reasons,
    )


def _topic_relevant(requirement: ClaimRequirement, corpus: str) -> bool:
    if requirement.topics:
        return any(
            any(alias in corpus for alias in _TOPIC_ALIASES.get(topic, (topic.replace("_", " "),)))
            for topic in requirement.topics
        )
    if requirement.entity_terms:
        return any(term in corpus for term in requirement.entity_terms)
    return True


def _terminology_match(requirement: ClaimRequirement, corpus: str) -> bool:
    if requirement.steps and not all(re.search(rf"\bstep\s*{step}\b", corpus) for step in requirement.steps):
        return False
    concept_patterns = {
        "blood_pressure_control": r"\bcontrol\w*\b|\btarget\b",
        "mortality": r"\bmortal\w*\b|\bdeath\w*\b",
        "reduction": r"\breduc\w*\b|\blower\w*\b|\bdecreas\w*\b",
        "risk": r"\brisk\b",
        "response": r"\brespond\w*\b|\bresponse\b",
        "survival": r"\bsurviv\w*\b",
        "unawareness": r"\bunaware\b|\bundiagnos\w*\b",
        "adverse_event": r"\badverse event\w*\b|\bside effect\w*\b",
        "treatment_effect": r"\beffective\w*\b|\bresult\w*\b|\boutcome\w*\b|\bachiev\w*\b|\bimprov\w*\b|\breduc\w*\b|\blower\w*\b",
        "prevalence": r"\bprevalence\b|\b(?:have|has|living with|affected by)\b",
    }
    return all(re.search(concept_patterns[name], corpus) for name in requirement.outcome_concepts)


def _intent_relevant(fact_type: str, text: str, section: str) -> bool:
    corpus = f"{section} {text}".lower()
    if fact_type in {PERCENTAGE_OUTCOME, MORTALITY_RATE, RISK_EFFECT, RATIO_EFFECT, NUMERICAL_OUTCOME, TREATMENT_EFFECT}:
        return bool(re.search(r"outcome|result|effect|control|risk|mortal|death|surviv|respond|unaware|burden|preval", corpus))
    if fact_type == THRESHOLD:
        return bool(re.search(r"threshold|diagnos|target|reading|mmhg|defined", corpus))
    if fact_type == DOSAGE:
        return bool(re.search(r"dose|dosage|medicat|drug|treat|mg|mcg", corpus))
    if fact_type == TREATMENT_RECOMMENDATION:
        return bool(_TREATMENT.search(corpus))
    return True


def _section_relevant(fact_type: str, section: str, text: str) -> bool:
    corpus = f"{section} {text[:160]}".lower()
    if fact_type == TREATMENT_RECOMMENDATION:
        return bool(re.search(r"treat|recommend|management|step", corpus))
    if fact_type in {PERCENTAGE_OUTCOME, MORTALITY_RATE, RISK_EFFECT, RATIO_EFFECT, NUMERICAL_OUTCOME, TREATMENT_EFFECT}:
        return bool(re.search(r"outcome|result|effect|control|burden|risk|mortal|death|surviv|preval", corpus))
    if fact_type == THRESHOLD:
        return bool(re.search(r"diagnos|threshold|target|mmhg|overview", corpus))
    if fact_type == DOSAGE:
        return bool(re.search(r"dose|treat|medicat|drug", corpus))
    return True


def _direct_support(requirement: ClaimRequirement, evidence: Evidence, query: str) -> bool:
    text = evidence.chunk.text or ""
    if requirement.fact_type == TREATMENT_RECOMMENDATION:
        if requirement.steps and not all(re.search(rf"\bstep\s*{step}\b", text, re.I) for step in requirement.steps):
            return False
        return bool(_TREATMENT.search(text) and support_score(query, evidence) >= 0.35)
    if requirement.fact_type == THRESHOLD:
        values = _numeric_sentences(text, _BP if re.search(r"blood pressure|hypertension|\bbp\b", query, re.I) else None)
        return any(
            _sentence_matches(requirement, sentence)
            and re.search(
                r"\bthreshold\b|\bdiagnos\w*\b|\bdefined\b|\bconsidered\b|"
                r"\btarget\b|\b(?:or )?higher\b|\bgreater\b|\bbelow\b|\babove\b",
                sentence,
                re.I,
            )
            for sentence in values
        )
    if requirement.fact_type == DOSAGE:
        return any(_sentence_matches(requirement, sentence) for sentence in _numeric_sentences(text, _DOSE))
    if requirement.fact_type == RATIO_EFFECT:
        return any(_sentence_matches(requirement, sentence) for sentence in _numeric_sentences(text, _RATIO))
    if requirement.fact_type in {PERCENTAGE_OUTCOME, RISK_EFFECT}:
        return any(_sentence_matches(requirement, sentence) for sentence in _numeric_sentences(text, _PERCENT))
    if requirement.fact_type == MORTALITY_RATE:
        sentences = _numeric_sentences(text, _PERCENT) + _numeric_sentences(text, _RATE)
        return any(_sentence_matches(requirement, sentence) for sentence in dict.fromkeys(sentences))
    if requirement.fact_type == NUMERICAL_OUTCOME:
        sentences = _numeric_sentences(text, _PERCENT) + _numeric_sentences(text, _RATE) + _numeric_sentences(text, _COUNT)
        return any(_sentence_matches(requirement, sentence) for sentence in dict.fromkeys(sentences))
    if requirement.fact_type == TREATMENT_EFFECT:
        return bool(_TREATMENT.search(text) and _sentence_matches(requirement, text))
    return support_score(query, evidence) >= 0.35


def _numeric_sentences(text: str, required_pattern: re.Pattern | None) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if required_pattern is not None:
        return [sentence for sentence in sentences if required_pattern.search(sentence)]
    return [sentence for sentence in sentences if re.search(r"\d", sentence)]


def _sentence_matches(requirement: ClaimRequirement, sentence: str) -> bool:
    lower = sentence.lower()
    if requirement.steps and not all(re.search(rf"\bstep\s*{step}\b", lower) for step in requirement.steps):
        return False
    if requirement.topics and not _topic_relevant(requirement, lower):
        return False
    if requirement.entity_terms and not any(
        term in lower for term in requirement.entity_terms
    ):
        return False
    concept_patterns = {
        "blood_pressure_control": r"\bcontrol\w*\b|\btarget\b",
        "mortality": r"\bmortal\w*\b|\bdeath\w*\b",
        "reduction": r"\breduc\w*\b|\blower\w*\b|\bdecreas\w*\b",
        "risk": r"\brisk\b",
        "response": r"\brespond\w*\b|\bresponse\b",
        "survival": r"\bsurviv\w*\b",
        "unawareness": r"\bunaware\b|\bundiagnos\w*\b",
        "adverse_event": r"\badverse event\w*\b|\bside effect\w*\b",
        "treatment_effect": r"\beffective\w*\b|\bresult\w*\b|\boutcome\w*\b|\bachiev\w*\b|\bimprov\w*\b|\breduc\w*\b|\blower\w*\b",
        "prevalence": r"\bprevalence\b|\b(?:have|has|living with|affected by)\b",
    }
    return all(re.search(concept_patterns[name], lower) for name in requirement.outcome_concepts)
