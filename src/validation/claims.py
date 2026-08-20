"""Claim validation — compare generated claims against retrieved evidence.

Hybrid: deterministic lexical-support check, optionally reinforced by the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..retrieval.models import Evidence

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_MEDICAL_SIGNAL = re.compile(
    r"\b(?:symptom|treatment|disease|diabetes|hypertension|blood pressure|cancer|"
    r"infection|vaccin|medic\w+|drug|dos\w+|fever|pain|heart|kidney|cholesterol|"
    r"glucose|insulin|chronic|acute|risk|complication|mmHg|mg|gram)\b",
    re.IGNORECASE,
)


@dataclass
class ClaimValidationResult:
    ok: bool
    unsupported_sentences: list = field(default_factory=list)
    provider: str = "deterministic"


class ClaimValidator:
    def validate(self, answer_text: str, evidence: list[Evidence]) -> ClaimValidationResult:
        if not answer_text.strip():
            return ClaimValidationResult(ok=False, unsupported_sentences=["empty answer"])

        corpus = " ".join(e.chunk.text.lower() for e in evidence)
        corpus_tokens = set(re.findall(r"[a-z0-9]+", corpus))

        unsupported = []
        for sent in _SENT_RE.split(answer_text):
            s = sent.strip()
            if not s or not _MEDICAL_SIGNAL.search(s):
                continue
            # Explicitly grouped to match the original (precedence-derived)
            # intent: skip bracket-only fragments, and skip short
            # non-bullet fragments that are too small to be a real claim.
            if s.startswith("[") or (
                not s.startswith("•") and len(s) < 15
            ):
                continue
            sent_tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
            substantive = {t for t in sent_tokens if t not in _STOPWORDS}
            if not substantive:
                continue
            overlap = len(substantive & corpus_tokens) / max(1, len(substantive))
            # a claim is supported if a meaningful share of its terms appear in evidence
            if overlap < 0.25:
                unsupported.append(s)

        ok = len(unsupported) == 0
        return ClaimValidationResult(ok=ok, unsupported_sentences=unsupported[:5])


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in",
    "on", "for", "and", "or", "with", "as", "by", "at", "it", "its", "this",
    "that", "these", "those", "from", "can", "may", "should", "would", "could",
    "you", "your", "i", "my", "not", "no", "if", "then", "than", "also", "has",
    "have", "had", "do", "does", "did", "based", "retrieved", "guidelines",
}