"""Query analysis for the canonical English query.

Extracts medical topics, question intent, and keywords used by the source
router, retrieval expansion, and answer generation. Expanded intent coverage
so the extractive answerer can match the question type (causes vs symptoms
vs complications vs treatment, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOPIC_KEYWORDS = {
    "hypertension": ["blood pressure", "hypertension", "hypertensive", "bp", "systolic", "diastolic"],
    "diabetes": ["diabetes", "diabetic", "blood sugar", "blood glucose", "glucose", "insulin", "a1c", "hba1c"],
    "cardiovascular": ["cardiovascular", "heart", "cardiac", "stroke", "cholesterol", "statin", "coronary", "heart disease", "heart attack"],
    "cancer": ["cancer", "tumor", "tumour", "malignan", "oncology", "carcinoma"],
    "obesity": ["obesity", "obese", "overweight", "body mass", "bmi", "weight loss"],
    "vaccination": ["vaccine", "vaccination", "immunization", "immunisation", "flu shot", "influenza"],
    "tobacco": ["tobacco", "smoking", "nicotine", "cigarette", "vaping"],
    "lifestyle": ["physical activity", "exercise", "diet", "nutrition", "sedentary"],
    "medication": ["medication", "medicine", "drug", "dose", "dosage", "prescription", "antibiotic", "statin", "metformin", "aspirin", "ibuprofen"],
    "infectious_disease": ["infection", "infectious", "virus", "bacteria", "flu", "pneumonia", "tuberculosis", "influenza", "malaria", "hiv", "hepatitis", "measles", "covid", "coronavirus", "meningitis", "sepsis"],
    "anemia": ["anemia", "anaemia", "haemoglobin", "hemoglobin", "iron deficiency"],
    "asthma": ["asthma", "wheez", "inhaler"],
    "respiratory": ["copd", "chronic obstructive", "bronchitis", "emphysema", "lung disease", "respiratory"],
    "neurology": ["dementia", "alzheimer", "parkinson", "epilepsy", "seizure", "migraine", "neurolog"],
    "mental_health": ["depression", "anxiety", "anxious", "stress", "mental health", "depressive", "bipolar"],
}

# Question-intent patterns. Ordered — first match wins for the primary intent.
_INTENT_PATTERNS = [
    ("symptoms", r"\bsymptom|\bsigns?\b|\bwhat (?:are|are) the (?:symptoms|signs)\b|\bsymptoms?\b"),
    ("causes", r"\bcause[sd]?\b|\bwhy does|\bwhat leads to|\betiology"),
    ("complications", r"\bcomplication|\bconsequence|\beffects of\b|\bdanger\w*|\bwhat can happen\b|\bserious\b|\bdeadly|\bfatal\b|\blethal"),
    ("treatment", r"\btreat|\bmedic\w+|\bdrug|\btherap|\bmanag\w+|\brecommend\w*|\bpharmacolog\w*|\bantihypertensive\w*|\bfirst[- ]line\b|\bstep\s*(?:[123]|one|two|three)\b|\bnot controlled\b|\buncontrolled\b|\bwhat to do\b|\bhow do you (?:treat|manage)|\bmedication"),
    ("diagnosis", r"\bdiagnos|\bscreen\w*|\bmeasure\w*|\btest\w*|\bhow (?:is|are).*diagnos|\bdetect"),
    ("prevention", r"\bprevent|\bavoid|\breduce (?:the )?risk\b|\bhow (?:can|to) (?:i )?(?:prevent|avoid)|\blower(?:ing)?\b"),
    ("risk", r"\brisk\b|\blikelihood\b|\bchance\b|\bpredispos"),
    ("definition", r"\bwhat is\b|\bdefine\b|\bdefinition\b|\bmeaning\b|\bexplain\b|\bdescribe\b"),
    ("comparison", r"\bcompare\b|\bdifference\b|\bversus\b|\bvs\b|\bbetter than\b|\bwhich is (?:better|worse)"),
    ("side_effects", r"\bside effects?\b|\badverse\b|\breaction\b"),
    ("dosage", r"\bdosage\b|\bdose\b|\bhow much\b|\bhow many\b|\bmg\b|\bfrequency\b"),
]


# Intent -> retrieval expansion terms (bridge synonym gaps for the
# lexical/hash retrieval used in demo mode). Deliberately avoids generic
# symptom words (headache, pain, fever, feel) that would dilute retrieval
# toward unrelated topics.
_INTENT_EXPANSION = {
    "symptoms": [],
    "causes": ["cause", "risk factor"],
    "complications": ["damage", "heart attack", "stroke", "kidney failure", "serious"],
    "treatment": ["treatment", "medication", "therapy"],
    "diagnosis": ["diagnose", "measure", "test", "screen"],
    "prevention": ["prevent", "lifestyle", "diet", "exercise"],
    "risk": ["risk", "factor"],
    "definition": ["condition", "disease"],
    "comparison": ["compare", "difference"],
    "side_effects": ["side effect", "adverse"],
    "dosage": ["dose", "dosage", "mg"],
}


@dataclass
class QueryAnalysis:
    english_query: str
    topics: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    keywords: list = field(default_factory=list)

    @property
    def primary_intent(self) -> str:
        return self.intents[0] if self.intents else ""

    def to_dict(self) -> dict:
        return {
            "topics": self.topics,
            "intents": self.intents,
            "keywords": self.keywords,
        }


class QueryAnalyzer:
    def analyze(self, english_query: str) -> QueryAnalysis:
        ql = english_query.lower()
        topics = [t for t, kws in _TOPIC_KEYWORDS.items() if any(k in ql for k in kws)]
        intents = [i for i, pat in _INTENT_PATTERNS if re.search(pat, ql)]
        keywords = [kw for kws in _TOPIC_KEYWORDS.values() for kw in kws if kw in ql]
        return QueryAnalysis(
            english_query=english_query,
            topics=topics,
            intents=intents,
            keywords=sorted(set(keywords)),
        )

    @staticmethod
    def expansion_terms(intents: list) -> list[str]:
        terms = []
        for i in intents:
            terms.extend(_INTENT_EXPANSION.get(i, []))
        return terms

    def normalize_for_retrieval(self, english_query: str, analysis: QueryAnalysis | None = None) -> str:
        analysis = analysis or self.analyze(english_query)
        normalized = " ".join((english_query or "").strip().split())
        replacements = (
            (r"\bstep\s*one\b|\bfirst step\b", "step 1"),
            (r"\bstep\s*two\b|\bsecond step\b", "step 2"),
            (r"\bstep\s*three\b|\bthird step\b", "step 3"),
            (r"\bhigh bp\b|\bhigh blood pressure\b", "hypertension blood pressure"),
            (r"\bantihypertensive(?:s)?\b", "hypertension pharmacological treatment"),
            (r"\binitial(?:ly)? (?:drug|medication|medicine|therapy|treatment)\b", "step 1 pharmacological treatment"),
            (r"\bfirst[- ]line (?:drug|medication|medicine|therapy|treatment)\b", "step 1 pharmacological treatment"),
            (r"\byounger than (?:age )?55(?: years)?\b", "under 55 years"),
            (r"\bbelow 55(?: years)?\b", "under 55 years"),
            (r"\bnot controlled after step\s*1\b", "not controlled step 2"),
            (r"\buncontrolled after step\s*1\b", "uncontrolled step 2"),
            (r"\bafter step\s*1 (?:fails|failed|does not control|did not control)\b", "step 2 not controlled"),
            (r"\bwhen step\s*1 (?:fails|failed|does not control|did not control)\b", "step 2 not controlled"),
            (r"\bwhat comes next (?:after|when) step\s*1\b", "step 2 treatment"),
            (r"\bcut[- ]?off\b", "diagnostic threshold"),
            (r"\bdefinition threshold\b", "diagnostic threshold definition"),
        )
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        additions = []
        for term in self.expansion_terms(analysis.intents):
            if term.lower() not in normalized.lower():
                additions.append(term)
        if "treatment" in analysis.intents and re.search(r"\bstep\s*1\b|\binitial\b|\bfirst[- ]line\b", normalized, re.I):
            additions.extend(["step 1", "pharmacological treatment"])
        if "treatment" in analysis.intents and re.search(r"\bstep\s*2\b|\bnot controlled\b|\buncontrolled\b", normalized, re.I):
            additions.extend(["step 2", "blood pressure not controlled"])
        if re.search(r"\bthreshold\b|\bdefine\w*\b", normalized, re.I):
            additions.extend(["definition", "diagnostic threshold"])
        unique = []
        for item in additions:
            if item.lower() not in normalized.lower() and item.lower() not in [x.lower() for x in unique]:
                unique.append(item)
        return normalized + ((" " + " ".join(unique)) if unique else "")

    def retrieval_variants(self, english_query: str, analysis: QueryAnalysis | None = None) -> list[str]:
        analysis = analysis or self.analyze(english_query)
        normalized = self.normalize_for_retrieval(english_query, analysis)
        variants = [normalized]
        source = ""
        if re.search(r"\bNICE\b|\bNational Institute for Health and Care Excellence\b", english_query):
            source = "NICE guideline"
        elif re.search(r"\bWHO\b|\bWorld Health Organization\b", english_query):
            source = "WHO fact sheet"
        elif re.search(r"\bFDA\b|\bFood and Drug Administration\b", english_query):
            source = "FDA"
        elif re.search(r"\bEMA\b|\bEuropean Medicines Agency\b", english_query):
            source = "EMA"
        elif re.search(r"\bESC\b|\bEuropean Society of Cardiology\b", english_query):
            source = "ESC guideline"
        elif re.search(r"\bAHA\b|\bAmerican Heart Association\b", english_query):
            source = "AHA guideline"
        topic = " ".join(analysis.topics[:2])
        if "treatment" in analysis.intents:
            if re.search(r"\bstep\s*2\b|\bnot controlled\b|\buncontrolled\b", normalized, re.I):
                variants.append(f"{source} {topic} step 2 treatment blood pressure not controlled".strip())
            elif re.search(r"\bstep\s*1\b|\binitial\b|\bfirst[- ]line\b|\bunder 55\b", normalized, re.I):
                population = "under 55 black African African-Caribbean family origin" if re.search(r"african|caribbean", normalized, re.I) else "under 55"
                variants.append(f"{source} {topic} step 1 pharmacological treatment {population}".strip())
        if re.search(r"\bthreshold\b|\bdefine\w*\b", normalized, re.I):
            variants.append(f"{source} {topic} blood pressure diagnostic threshold definition".strip())
        compact = " ".join([source, topic, *self.expansion_terms(analysis.intents)]).strip()
        if compact:
            variants.append(compact)
        stop = {
            "what", "which", "does", "this", "that", "with", "from", "have",
            "produce", "compared", "general", "purpose", "according", "recommended",
        }
        lexical_terms = [
            token for token in re.findall(r"[a-z0-9]+", english_query.lower())
            if len(token) >= 4 and token not in stop
        ]
        if lexical_terms:
            variants.append(" ".join([source, topic, *lexical_terms[:10]]).strip())
        out = []
        for item in variants:
            item = " ".join(item.split())
            if item and item.lower() not in [x.lower() for x in out]:
                out.append(item)
        return out[:3]

    @staticmethod
    def clarification_request(english_query: str, analysis: QueryAnalysis) -> str:
        ql = (english_query or "").lower()
        treatment = "treatment" in analysis.intents
        patient_specific = bool(re.search(r"\bthis patient\b|\bthe patient\b|\bmy patient\b|\bfor me\b", ql))
        if not treatment or not patient_specific:
            return ""
        missing = []
        if not analysis.topics:
            missing.append("the diagnosis or clinical condition")
        if not re.search(r"\b\d{1,3}\s*(?:years?|yo)\b|\bunder\s*\d+\b|\bover\s*\d+\b|\bolder\b|\byounger\b", ql):
            missing.append("the patient's age")
        if not re.search(r"\bcontrolled\b|\buncontrolled\b|\bsever\w*\b|\bstage\b|\bstep\s*[123]\b|\bcurrent\b|\balready\b", ql):
            missing.append("severity and current treatment")
        if not missing:
            return ""
        if len(missing) == 1:
            details = missing[0]
        else:
            details = ", ".join(missing[:-1]) + ", and " + missing[-1]
        return f"Please provide {details} so I can identify the relevant guideline recommendation without assuming missing clinical context."