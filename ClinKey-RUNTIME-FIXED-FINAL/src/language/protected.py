"""Protect clinical tokens during translation (citations, numbers, units, names)."""
from __future__ import annotations

import re

_CITE = re.compile(r"\[\d{1,2}\]")
_NUMUNIT = re.compile(
    r"\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:mg|mcg|g|mL|ml|L|mmHg|mmhg|%|IU|iu)?\b"
)
_TERMS = [
    "ACE inhibitor", "ACE inhibitors", "ARB", "HbA1c", "eGFR", "BNP",
    "ABPM", "HBPM", "COPD", "CKD", "DKA", "HHS", "WHO", "NICE", "CDC",
    "hypertension", "diabetes", "insulin", "metformin",
]


class TermProtector:
    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        table: dict[str, str] = {}
        i = 0

        def _sub(m: re.Match) -> str:
            nonlocal i
            key = f"⟦P{i}⟧"
            table[key] = m.group(0)
            i += 1
            return key

        out = _CITE.sub(_sub, text)
        out = _NUMUNIT.sub(_sub, out)
        for term in sorted(_TERMS, key=len, reverse=True):
            out = re.sub(re.escape(term), _sub, out, flags=re.I)
        return out, table

    def unmask(self, text: str, table: dict[str, str]) -> str:
        out = text
        for k, v in table.items():
            out = out.replace(k, v)
        return out