"""Numerical, unit, negation, and recommendation-strength safety checks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..retrieval.models import Evidence

_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(mg|mcg|µg|g|ml|mL|L|mmHg|mmhg|%|iu|IU)\b"
)
_NEG = re.compile(
    r"\b(?:do not|don't|not recommended|contraindicated|should not|must not|never)\b",
    re.I,
)
_STRONG = re.compile(r"\b(?:strongly recommend|must|is recommended|should)\b", re.I)
_WEAK = re.compile(r"\b(?:may be considered|may consider|consider|offer|can be)\b", re.I)


@dataclass
class MedicalSafetyResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    status: str = "OK"


class MedicalSafetyValidator:
    def validate(self, answer: str, evidence: list[Evidence]) -> MedicalSafetyResult:
        issues: list[str] = []
        corpus = " ".join(e.chunk.text for e in evidence)
        corpus_l = corpus.lower()
        ans = answer or ""

        # numbers in the answer must appear in evidence (except citation indices)
        ans_wo_cite = re.sub(r"\[\d{1,2}\]", "", ans)
        for n in set(_NUM.findall(ans_wo_cite)):
            if n not in corpus and n not in {"1", "2", "3", "4", "5"}:
                issues.append(f"unsupported_number:{n}")

        for m in _UNIT.finditer(ans_wo_cite):
            token = m.group(0).replace(" ", "").lower()
            compact = re.sub(r"\s+", "", corpus.lower())
            if token not in compact and m.group(0).lower() not in corpus_l:
                issues.append(f"unsupported_unit:{m.group(0)}")

        # polarity: answer asserts recommend while evidence only negates
        if _STRONG.search(ans) and _NEG.search(corpus) and not _STRONG.search(corpus):
            issues.append("polarity_upgrade_or_reversal")
        if re.search(r"\brecommend(?:ed|s)?\b", ans, re.I) and re.search(
            r"\bdo not recommend|not recommended|contraindicated\b", corpus, re.I
        ):
            if not re.search(r"\bdo not recommend|not recommended\b", ans, re.I):
                issues.append("negation_reversal")

        # strength upgrade: evidence weak, answer strong
        if _WEAK.search(corpus) and _STRONG.search(ans) and not _STRONG.search(corpus):
            issues.append("recommendation_strength_upgrade")

        ok = not issues
        return MedicalSafetyResult(
            ok=ok,
            issues=issues,
            status="OK" if ok else "VALIDATION_FAILURE",
        )