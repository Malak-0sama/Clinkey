"""Grounded English answer generation.

Evidence is the ONLY source of truth. Gemini synthesizes when available;
otherwise a deterministic extractive answer is built directly from the
retrieved chunks (never from model memory). Citation markers [n] in the
answer are renumbered to match the final citation list (1..k).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.client import LLMUnavailable, get_llm
from ..retrieval.models import Evidence
from .prompts import SYSTEM_GROUNDED, build_user_prompt, format_evidence_xml

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _sig_matches(signal: str, text: str) -> bool:
    """Match a signal against text.

    Single-word signals are matched on word boundaries (so "symptom" does not
    double-count inside "symptoms"); multi-word phrases are matched as
    substrings (so "blood pressure" matches "high blood pressure").
    """
    if " " in signal:
        return signal in text
    return re.search(r"\b" + re.escape(signal) + r"\b", text) is not None

# Topic -> signal words (synonyms) used to keep answers on-topic.
_TOPIC_SIGNALS = {
    "hypertension": ["hypertension", "hypertensive", "blood pressure", "high blood pressure"],
    "diabetes": ["diabetes", "diabetic", "blood sugar", "blood glucose", "glucose", "insulin"],
    "cardiovascular": ["heart", "cardiac", "stroke", "cholesterol", "cardiovascular", "coronary", "heart attack"],
    "cancer": ["cancer", "tumor", "tumour", "malignant", "carcinoma", "oncology"],
    "obesity": ["obesity", "obese", "overweight", "body mass", "bmi"],
    "vaccination": ["vaccine", "vaccination", "immunization", "immunisation"],
    "tobacco": ["tobacco", "smoking", "nicotine", "cigarette"],
    "lifestyle": ["physical activity", "exercise", "diet", "nutrition"],
    "medication": ["medication", "medicine", "drug", "dose", "dosage", "prescription"],
    "infectious_disease": ["infection", "infectious", "virus", "bacteria", "pneumonia", "influenza", "flu", "malaria", "hiv", "hepatitis", "measles", "covid", "tuberculosis"],
    "anemia": ["anemia", "anaemia", "hemoglobin", "haemoglobin", "iron"],
    "asthma": ["asthma", "wheezing", "inhaler"],
    "respiratory": ["copd", "chronic obstructive", "bronchitis", "emphysema", "lung disease"],
    "neurology": ["dementia", "alzheimer", "parkinson", "epilepsy", "migraine", "seizure"],
    "mental_health": ["depression", "anxiety", "mental health", "depressive"],
}

# Unambiguous disease markers used ONLY for the wrong-topic penalty. These
# exclude shared anatomical/event terms (heart, stroke, kidney, ...) that
# legitimately appear across multiple topics and intents.
_DISEASE_MARKERS = {
    "hypertension": ["hypertension", "hypertensive", "blood pressure"],
    "diabetes": ["diabetes", "diabetic", "blood sugar", "blood glucose"],
    "cancer": ["cancer", "tumor", "tumour", "malignant", "carcinoma"],
    "asthma": ["asthma"],
    "obesity": ["obesity", "obese", "overweight"],
    "anemia": ["anemia", "anaemia", "hemoglobin", "haemoglobin"],
    "tobacco": ["tobacco", "smoking", "nicotine"],
    "vaccination": ["vaccine", "vaccination", "immunization"],
    "infectious_disease": ["infection", "infectious", "virus", "bacteria", "pneumonia", "tuberculosis", "influenza", "flu", "malaria", "hiv", "hepatitis", "measles", "covid", "coronavirus", "meningitis", "sepsis"],
    "respiratory": ["copd", "chronic obstructive", "bronchitis", "emphysema"],
    "neurology": ["dementia", "alzheimer", "parkinson", "epilepsy", "migraine"],
    "mental_health": ["depression", "anxiety", "depressive"],
}


# Intent -> signal words used to score sentences for the extractive answerer.
_INTENT_SIGNALS = {
    "symptoms": {"symptom", "symptoms", "sign", "signs", "feel", "headache", "pain", "fatigue", "thirst", "vision", "urination", "fever"},
    "causes": {"cause", "causes", "caused", "because", "due", "risk factor", "results", "leads", "associated", "trigger"},
    "complications": {"complication", "complications", "damage", "lead", "increase", "risk", "heart", "stroke", "kidney", "blind", "failure", "death", "serious", "danger", "consequence"},
    "treatment": {"treat", "treatment", "medication", "medicine", "drug", "therapy", "manage", "management", "recommend", "prescribe", "dose", "inhibitor", "blocker", "insulin"},
    "diagnosis": {"diagnos", "diagnosis", "measure", "measured", "test", "testing", "screen", "screening", "detect", "confirmed", "reading", "mmhg", "monitor"},
    "prevention": {"prevent", "prevention", "avoid", "reduce", "lower", "lifestyle", "diet", "exercise", "salt", "smoking", "alcohol", "healthy", "weight"},
    "risk": {"risk", "factor", "factors", "increase", "likelihood", "associated", "higher"},
    "definition": {"is a", "is an", "refers", "defined", "condition", "means", "disease", "chronic"},
    "comparison": {"difference", "compare", "versus", "higher", "lower", "both", "unlike", "whereas"},
    "side_effects": {"side effect", "side effects", "adverse", "reaction", "nausea", "dizziness", "rash", "stomach"},
    "dosage": {"dose", "dosage", "mg", "milligram", "daily", "twice", "frequency", "amount", "tablet"},
}


@dataclass
class GeneratedAnswer:
    text: str
    cited_evidence: list = field(default_factory=list)  # Evidence in citation order
    provider: str = ""
    intent: str = ""


class GroundedGenerator:
    def generate(self, english_query: str, evidence: list[Evidence], intent: str = "", topics: list | None = None) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(text="", cited_evidence=[], provider="none", intent=intent)
        try:
            llm = get_llm()
            if llm.available:
                return self._gemini(llm, english_query, evidence, intent)
        except (LLMUnavailable, Exception):  # noqa: BLE001
            pass
        return self._extractive(english_query, evidence, intent, topics)

    # ---- Gemini synthesis ----
    @staticmethod
    def _gemini(llm, query: str, evidence: list[Evidence], intent: str) -> GeneratedAnswer:
        prompt = build_user_prompt(query, format_evidence_xml(evidence))
        text = llm.generate(prompt, system=SYSTEM_GROUNDED)

        # collect cited evidence (original numbering) then renumber to 1..k
        nums = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", text) if 1 <= int(n) <= len(evidence)})
        cited = [evidence[n - 1] for n in nums]
        if not cited:
            cited = list(evidence[:5])
        remap = {old: new for new, old in enumerate(nums, start=1)}
        if remap:
            text = re.sub(r"\[(\d{1,2})\]", lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0), text)
        return GeneratedAnswer(text=text, cited_evidence=cited, provider="gemini", intent=intent)

    # ---- extractive fallback ----
    def _extractive(self, query: str, evidence: list[Evidence], intent: str, topics: list | None = None) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(text="", cited_evidence=[], provider="extractive", intent=intent)
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        intent_signals = _INTENT_SIGNALS.get(intent, set())

        # Topic signals: synonyms for the detected topic(s). Weighted heavily
        # so answers stay on-topic (hypertension vs diabetes, etc.).
        topic_signals: set[str] = set()
        for t in (topics or []):
            topic_signals.update(_TOPIC_SIGNALS.get(t, []))

        # Wrong-topic signals: disease terms belonging to OTHER topics. A
        # sentence about a different disease is penalized so it does not
        # displace the correct evidence.
        wrong_topic_signals: set[str] = set()
        topic_set = set(topics or [])
        for t, sigs in _DISEASE_MARKERS.items():
            if t not in topic_set:
                wrong_topic_signals.update(sigs)

        # Score every sentence across ALL retrieved chunks, then pick the top
        # most-relevant ones (topic match >> intent match > query overlap).
        scored: list[tuple[float, Evidence, str]] = []
        for e in evidence:
            for s in _SENT_RE.split(e.chunk.text):
                s = s.strip()
                if len(s) < 20 or len(s) > 400:
                    continue
                slow = s.lower()
                st = set(re.findall(r"[a-z0-9]+", slow))
                overlap = len(st & q_tokens)
                # Binary topic match: a sentence is either on-topic or not
                # (avoids double-counting synonyms like "hypertension" +
                # "blood pressure" in the same sentence).
                topic_hits = 1 if any(_sig_matches(sig, slow) for sig in topic_signals) else 0
                intent_hits = sum(1 for sig in intent_signals if _sig_matches(sig, slow))
                wrong_hits = sum(1 for sig in wrong_topic_signals if _sig_matches(sig, slow))
                score = overlap * 1.0 + intent_hits * 2.0 + topic_hits * 4.0 - wrong_hits * 6.0
                if score > 0:
                    scored.append((score, e, s))

        scored.sort(key=lambda x: -x[0])

        # Keep extractive sentences on the asked step when the query names one.
        step_m = re.search(r"\bstep\s*([123])\b", query.lower())
        if step_m:
            want = f"step {step_m.group(1)}"
            focused = [(sc, e, s) for sc, e, s in scored if want in s.lower()]
            if focused:
                scored = focused

        lines: list[str] = []
        used: list[Evidence] = []
        seen_text: set[str] = set()
        for _, e, s in scored:
            if len(used) >= 5:
                break
            key = s.lower()[:120]
            if key in seen_text:
                continue
            seen_text.add(key)
            if e not in used:
                used.append(e)
            lines.append(f"• {s} [{used.index(e) + 1}]")

        if not lines:
            # fallback: lead sentences verbatim
            for e in evidence:
                sents = [s.strip() for s in _SENT_RE.split(e.chunk.text) if s.strip()]
                if sents:
                    if e not in used:
                        used.append(e)
                    lines.append(f"• {sents[0]} [{used.index(e) + 1}]")
                if len(lines) >= 5:
                    break

        if not lines:
            return GeneratedAnswer(text="", cited_evidence=[], provider="extractive", intent=intent)

        text = "Based on the retrieved official guidelines:\n\n" + "\n".join(lines)
        return GeneratedAnswer(text=text, cited_evidence=used, provider="extractive", intent=intent)