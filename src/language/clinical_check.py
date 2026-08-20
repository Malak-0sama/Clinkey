"""Validate a translated clinical answer against the English source."""
from __future__ import annotations

import re

from .protected import _CITE, _NUMUNIT


def validate_translation(english: str, localized: str) -> dict:
    issues: list[str] = []
    en_cites = _CITE.findall(english or "")
    loc_cites = _CITE.findall(localized or "")
    if set(en_cites) - set(loc_cites):
        issues.append("citations_lost")
    en_nu = {m.group(0).replace(" ", "").lower() for m in _NUMUNIT.finditer(english or "")}
    loc_nu = {m.group(0).replace(" ", "").lower() for m in _NUMUNIT.finditer(localized or "")}
    # allow numbers without units if the digit sequence survives
    en_nums = set(re.findall(r"\d+(?:\.\d+)?", english or ""))
    loc_nums = set(re.findall(r"\d+(?:\.\d+)?", localized or ""))
    if en_nums - loc_nums:
        issues.append("numbers_lost")
    if any(u in (english or "").lower() for u in ("mg", "mmhg", "ml")):
        if "g" in loc_nu and "mg" not in " ".join(loc_nu) and "mg" in (english or "").lower():
            issues.append("unit_changed")
    return {"ok": not issues, "issues": issues}