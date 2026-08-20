"""Detect contradictory recommendations across retrieved evidence."""
from __future__ import annotations

import re

from .medical import _NEG, _STRONG
from ..retrieval.models import Evidence

# A negated chunk and a recommending chunk are only in conflict if they are
# actually talking about the same thing (e.g. both about "Drug X"). Without
# this, any negation anywhere ("hypertension is often asymptomatic" - no
# negation words) paired with any unrelated recommendation elsewhere in the
# evidence set was being flagged as a contradiction. We require a minimum
# share of shared, meaningful vocabulary between the two chunks.
_CONFLICT_TOPIC_THRESHOLD = 0.3
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "not", "no", "this", "that", "these", "those", "and", "or",
    "with", "for", "in", "on", "of", "to", "as", "at", "it", "its", "will",
    "shall", "if", "then", "than", "also", "has", "have", "had",
}
_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")


def _topic_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def detect_conflicts(evidence: list[Evidence]) -> list[dict]:
    pos, neg = [], []
    for e in evidence:
        t = e.chunk.text
        if _NEG.search(t):
            neg.append(e)
        elif _STRONG.search(t) or re.search(r"\brecommend", t, re.I):
            pos.append(e)

    conflicts: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for n in neg:
        n_tokens = _topic_tokens(n.chunk.text)
        if not n_tokens:
            continue
        for p in pos:
            p_tokens = _topic_tokens(p.chunk.text)
            if not p_tokens:
                continue
            overlap = len(n_tokens & p_tokens) / max(1, min(len(n_tokens), len(p_tokens)))
            if overlap < _CONFLICT_TOPIC_THRESHOLD:
                # Negation and recommendation don't share enough subject
                # matter to be a real contradiction (e.g. asymptomatic
                # hypertension vs. an unrelated screening recommendation).
                continue
            key = (p.chunk.chunk_id, n.chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append({
                "type": "recommend_vs_not",
                "positive": [p.chunk.chunk_id],
                "negative": [n.chunk.chunk_id],
            })
    return conflicts