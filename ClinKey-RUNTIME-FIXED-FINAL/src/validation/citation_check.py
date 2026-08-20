"""Citation markers must point at real evidence items."""
from __future__ import annotations

import re

from ..retrieval.models import Evidence


def validate_citations(answer: str, evidence: list[Evidence]) -> dict:
    n = len(evidence)
    nums = [int(x) for x in re.findall(r"\[(\d{1,2})\]", answer or "")]
    bad = [i for i in nums if i < 1 or i > n]
    return {
        "ok": not bad,
        "invalid": bad,
        "status": "OK" if not bad else "CITATION_FAILURE",
    }