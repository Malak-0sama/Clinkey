"""Exact / near-exact clinical entity retrieval."""
from __future__ import annotations

import re

from ..ingestion.chunker import Chunk

_ENTITIES = [
    "dka", "hhs", "copd", "ckd", "ace inhibitor", "ace inhibitors", "arb",
    "hba1c", "egfr", "bnp", "abpm", "hbpm", "mmhg", "insulin", "metformin",
    "hypertension", "diabetes", "asthma", "depression", "obesity",
    "lisinopril", "amlodipine", "atorvastatin", "warfarin",
]


def find_query_entities(query: str) -> list[str]:
    ql = (query or "").lower()
    hits = [e for e in _ENTITIES if e in ql]
    # guideline / rec ids
    hits.extend(re.findall(r"\b(?:ng|cg|who|nice)[- ]?\d+\b", ql))
    return sorted(set(hits))


def entity_score(query: str, chunk: Chunk) -> float:
    ents = find_query_entities(query)
    if not ents:
        return 0.0
    text = chunk.text.lower()
    hits = sum(1 for e in ents if e in text)
    return hits / len(ents)