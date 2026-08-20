"""Localization validation — verify the translated answer preserved meaning."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LocalizationResult:
    ok: bool
    issues: list = field(default_factory=list)


class LocalizationValidator:
    def validate(self, english_answer: str, localized_answer: str, target_language: str) -> LocalizationResult:
        issues: list[str] = []

        if not localized_answer or not localized_answer.strip():
            issues.append("localized answer is empty")
            return LocalizationResult(ok=False, issues=issues)

        # citations must be preserved (numbers in brackets)
        en_cites = set(re.findall(r"\[(\d{1,2})\]", english_answer))
        loc_cites = set(re.findall(r"\[(\d{1,2})\]", localized_answer))
        if en_cites and not en_cites.issubset(loc_cites):
            issues.append("citation numbers were not preserved")

        # numbers (dosages / units / values) must be preserved
        en_nums = set(re.findall(r"\d+(?:\.\d+)?", english_answer))
        loc_nums = set(re.findall(r"\d+(?:\.\d+)?", localized_answer))
        missing = en_nums - loc_nums
        if missing and len(missing) > max(1, len(en_nums) * 0.3):
            issues.append("numbers were not preserved")

        # length sanity (avoid gross truncation or explosion)
        if localized_answer and english_answer:
            ratio = len(localized_answer) / max(1, len(english_answer))
            if ratio < 0.3 or ratio > 3.5:
                issues.append("translation length deviates suspiciously")

        return LocalizationResult(ok=len(issues) == 0, issues=issues)