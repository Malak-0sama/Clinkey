"""Reject navigation/boilerplate chunks before they are embedded."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config.settings import get_settings
from .chunker import Chunk

_NAV_LINE = re.compile(
    r"^(home|menu|read|read more|back to top|next|previous|search|contact|"
    r"login|sign in|subscribe|contents|table of contents|view all|"
    r"previous page|next page|skip to|cookie|privacy|terms)$",
    re.I,
)
_BREADCRUMB = re.compile(r".+\s*[>»/]\s*.+\s*[>»/]\s*.+")
_BOILER = re.compile(
    r"copyright|all rights reserved|cookie (policy|banner|consent)|"
    r"privacy policy|terms of (use|service)|subscribe to (our )?newsletter|"
    r"follow us on|this website uses cookies",
    re.I,
)
_CLINICAL = re.compile(
    r"\b(?:patient|treatment|diagnosis|dose|dosage|recommended|therapy|"
    r"risk|symptom|disease|condition|medication|monitor|contraindicat|"
    r"mmhg|mg\b|guideline)\b",
    re.I,
)


@dataclass
class QualityResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0


class ChunkQualityFilter:
    def __init__(self):
        s = get_settings()
        self.nav_ratio_max = float(s.get("ingestion_nav_ratio_max", 0.55))
        self.min_chars = int(s.get("ingestion_min_chunk_chars", 40))

    def evaluate(self, chunk: Chunk) -> QualityResult:
        text = (chunk.text or "").strip()
        reasons: list[str] = []
        if len(text) < self.min_chars:
            return QualityResult(False, ["too_short"], 0.0)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return QualityResult(False, ["empty"], 0.0)

        nav_n = 0
        for ln in lines:
            if _NAV_LINE.match(ln) or _BREADCRUMB.match(ln):
                nav_n += 1
        nav_ratio = nav_n / max(1, len(lines))
        if nav_ratio >= self.nav_ratio_max and not _CLINICAL.search(text):
            reasons.append("navigation_heavy")
        if _BOILER.search(text) and len(text) < 280 and not _CLINICAL.search(text):
            reasons.append("boilerplate")

        # whole chunk is a single nav token
        if len(lines) <= 2 and all(_NAV_LINE.match(ln) or len(ln) < 18 for ln in lines):
            if not _CLINICAL.search(text):
                reasons.append("nav_fragment")

        accepted = not reasons
        score = max(0.0, 1.0 - nav_ratio)
        return QualityResult(accepted, reasons, score)

    def filter_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], dict]:
        kept, rejected = [], 0
        counts: dict[str, int] = {}
        for c in chunks:
            r = self.evaluate(c)
            if r.accepted:
                kept.append(c)
            else:
                rejected += 1
                for reason in r.reasons:
                    counts[reason] = counts.get(reason, 0) + 1
        metrics = {
            "chunks_total": len(chunks),
            "chunks_accepted": len(kept),
            "chunks_rejected": rejected,
            "rejection_counts": counts,
        }
        return kept, metrics