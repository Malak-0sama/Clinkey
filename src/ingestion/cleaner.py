"""Text cleaning for extracted guideline content.

Removes repeated headers/footers, extraction artifacts, excess whitespace
and duplicated lines, while preserving page/section structure.
"""
from __future__ import annotations

import re


class TextCleaner:
    def clean(self, text: str) -> str:
        if not text:
            return ""
        # normalize line endings + unicode whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\u00a0\u200b\u2007]", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        # drop page-break artifacts
        text = re.sub(r"\f", "\n", text)
        # remove lines that are pure page numbers / headers
        lines = []
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                lines.append("")
                continue
            if re.fullmatch(r"\d{1,4}", s):  # standalone page number
                continue
            if re.fullmatch(r"Page \d+ of \d+", s, re.IGNORECASE):
                continue
            if len(s) < 60 and s.count(" ") < 2 and s.isupper() is False and s.endswith(".com"):
                continue
            lines.append(s)
        text = "\n".join(lines)
        # collapse 3+ blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # remove duplicated consecutive lines (headers/footers repeated)
        seen: set[str] = set()
        deduped = []
        for line in text.split("\n"):
            key = line.strip().lower()
            if line.strip() and key in seen:
                continue
            if line.strip():
                seen.add(key)
            deduped.append(line)
        return "\n".join(deduped).strip()