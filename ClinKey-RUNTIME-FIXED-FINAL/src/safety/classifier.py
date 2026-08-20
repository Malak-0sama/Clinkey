"""Safety classifier operating on the canonical ENGLISH query.

Primary: Gemini structured classification (when available).
Fallback: deterministic rule-based classifier (never silently assumes safe).
"""
from __future__ import annotations

import re

from ..llm.client import LLMUnavailable, get_llm
from .models import (
    CAUTION_NOTICE,
    EMERGENCY_RESPONSE,
    REFUSAL_RESPONSE,
    SafetyClass,
    SafetyResult,
)

# Emergencies that are ALWAYS urgent, regardless of phrasing.
_ALWAYS_EMERGENCY = [
    r"\bsuicid\w+\b",
    r"\bkill (?:myself|themselves)\b",
    r"\bself[- ]?harm\b",
    r"\boverdose\b",
    r"\bnot breathing\b",
    r"\bunconscious\b",
    r"\b(?:can'?t|cannot|struggl\w+ to) breathe\b",
    r"\bsevere allergic reaction\b",
    r"\banaphylax\w+\b",
    r"\bsevere bleeding\b",
]

# Symptom terms that are emergencies ONLY in a first-person / imminent context.
# A factual question like "What causes a heart attack?" must NOT be refused.
_CONTEXT_EMERGENCY = [
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bstroke\b",
    r"\bseizure\b",
]

_URGENCY_MARKERS = [
    r"\bi\b\s+(?:'m|am|am having|am experiencing|have|feel)",
    r"\bmy\b\s+(?:chest|heart|breath)",
    r"\bright now\b",
    r"\bcurrently\b",
    r"\bhappening to me\b",
]

# Personal diagnosis requests only — factual questions like
# "How is hypertension diagnosed?" must remain ALLOWED.
_DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b",
    r"\bdo you think i have\b",
    r"\bwhat'?s wrong with me\b",
    r"\bam i (?:having|suffering|dying)\b",
    r"\bis this (?:cancer|a tumor|serious)\b",
    r"\btell me (?:what|if) i (?:have|am)\b",
    r"\bdiagnos(?:e|ing|is)\b[^.!?]{0,40}\b(?:me|my|myself|i)\b",
    r"\b(?:should|could|can) you diagnose\b",
]

# creative / non-informational requests are out of scope for a medical evidence tool
_NON_QUERY_PATTERNS = [
    r"\b(?:write|compose|make)\b.*\b(?:poem|song|story|joke|essay|rap|haiku)\b",
    r"\b(?:tell me a|make me a)\b.*\b(?:joke|story|poem)\b",
]

_PATIENT_PATTERNS = [
    r"\bi\b\s+(?:have|am|take|feel|suffer)",
    r"\bmy\b\s+(?:blood pressure|sugar|glucose|heart|symptom)",
    r"\bme\b",
]

# Very light off-topic guard: questions with zero medical signal.
# Includes follow-up intents (prevent, complications, causes, diagnosed,
# dangerous, side effects, etc.) so terse follow-ups are not refused.
_MEDICAL_SIGNAL = re.compile(
    r"\b(?:symptom|treatment|disease|diabetes|hypertension|blood pressure|bp|cancer|"
    r"infection|vaccin|medic\w+|drug|dos\w+|fever|pain|heart|kidney|liver|lung|"
    r"pregnan\w+|cholesterol|glucose|insulin|therapy|guideline|dose|antibiotic|"
    r"health|doctor|nurse|patient|chronic|acute|virus|bacteria|"
    r"pneumonia|tuberculosis|influenza|flu|malaria|hepatitis|meningitis|sepsis|"
    r"asthma|anemia|anaemia|depression|anxiety|arthritis|osteoporosis|epilepsy|"
    r"migraine|stroke|heart attack|heart disease|coronary|"
    r"prevent\w*|avoid\w*|complication\w*|cause\w*|diagnos\w*|screen\w*|test\w*|"
    r"dangerous|serious|fatal|deadly|side effect\w*|adverse|risk\w*|"
    r"treat\w*|manag\w*|cur\w*|heal\w*|protect\w*|health\w*|diet\w*|exercise\w*|compare\w*|difference\w*)\b",
    re.IGNORECASE,
)


class SafetyClassifier:
    def __init__(self):
        self.enabled = True

    def classify(self, english_query: str) -> SafetyResult:
        q = english_query.strip()
        if not q:
            return SafetyResult(SafetyClass.ALLOWED, "empty query")

        # 1) deterministic checks (fast, always-on)
        rule = self._rule_classify(q)
        if rule.classification in (SafetyClass.EMERGENCY, SafetyClass.REFUSE):
            return rule

        # 2) Gemini refinement
        try:
            llm = get_llm()
            if llm.available:
                gem = self._gemini_classify(llm, q)
                if gem is not None:
                    return gem
        except (LLMUnavailable, Exception):  # noqa: BLE001
            pass

        return rule

    # ---- deterministic rules ----
    def _rule_classify(self, q: str) -> SafetyResult:
        ql = q.lower()

        # always-urgent emergencies
        for pat in _ALWAYS_EMERGENCY:
            if re.search(pat, ql):
                return SafetyResult(SafetyClass.EMERGENCY, "possible emergency", EMERGENCY_RESPONSE, "rules")

        # context-sensitive emergencies: only if the user is describing their
        # own imminent situation, not asking a factual question.
        has_urgency = any(re.search(m, ql) for m in _URGENCY_MARKERS)
        if has_urgency:
            for pat in _CONTEXT_EMERGENCY:
                if re.search(pat, ql):
                    return SafetyResult(SafetyClass.EMERGENCY, "possible emergency", EMERGENCY_RESPONSE, "rules")

        for pat in _DIAGNOSIS_PATTERNS:
            if re.search(pat, ql):
                return SafetyResult(SafetyClass.REFUSE, "diagnosis request", REFUSAL_RESPONSE, "rules")

        # creative / non-informational requests -> out of scope
        if any(re.search(p, ql) for p in _NON_QUERY_PATTERNS):
            return SafetyResult(
                SafetyClass.REFUSE,
                "out of scope (creative request)",
                "I'm a medical evidence assistant and can only answer health and "
                "medical questions grounded in official guidelines. Could you rephrase "
                "your question about a health topic?",
                "rules",
            )

        # patient-specific -> caution (still answer, but flag)
        if any(re.search(p, ql) for p in _PATIENT_PATTERNS) and _MEDICAL_SIGNAL.search(ql):
            return SafetyResult(SafetyClass.CAUTION, "patient-specific phrasing", CAUTION_NOTICE, "rules")

        # no medical signal at all -> out of scope
        if not _MEDICAL_SIGNAL.search(ql):
            return SafetyResult(
                SafetyClass.REFUSE,
                "out of scope",
                "I'm a medical evidence assistant and can only answer health and "
                "medical questions grounded in official guidelines. Could you rephrase "
                "your question about a health topic?",
                "rules",
            )

        return SafetyResult(SafetyClass.ALLOWED, "", "", "rules")

    # ---- Gemini ----
    @staticmethod
    def _gemini_classify(llm, q: str) -> SafetyResult | None:
        system = (
            "You are a medical AI safety classifier. Classify the query into exactly one "
            "of: ALLOWED (general guideline-based medical information), CAUTION "
            "(patient-specific, requires careful wording), REFUSE (diagnosis request, "
            "unsafe request, or out-of-scope), EMERGENCY (immediate danger: chest pain, "
            "suicide, overdose, difficulty breathing, severe bleeding, etc). "
            "Respond with JSON only: {\"classification\": \"...\", \"reason\": \"...\"}."
        )
        out = llm.generate_json(f"Query:\n{q}", system=system)
        if isinstance(out, dict) and "classification" in out:
            cls = str(out["classification"]).upper()
            mapping = {
                "ALLOWED": SafetyClass.ALLOWED,
                "CAUTION": SafetyClass.CAUTION,
                "REFUSE": SafetyClass.REFUSE,
                "EMERGENCY": SafetyClass.EMERGENCY,
            }
            sc = mapping.get(cls)
            if sc is None:
                return None
            reason = str(out.get("reason", ""))
            response = ""
            if sc == SafetyClass.EMERGENCY:
                response = EMERGENCY_RESPONSE
            elif sc == SafetyClass.REFUSE:
                response = REFUSAL_RESPONSE
            elif sc == SafetyClass.CAUTION:
                response = CAUTION_NOTICE
            return SafetyResult(sc, reason, response, "gemini")
        return None