"""Reranker.

Production default is the calibrated *heuristic* scorer (range ~[0, 1]).

CrossEncoder is NEVER auto-enabled. It loads only when BOTH:
  retrieval.reranker_backend == cross_encoder
  retrieval.cross_encoder_calibrated == true

Uncalibrated CE logits must not enter the production path.
"""
from __future__ import annotations

import re

from ..config.settings import get_settings
from .models import Evidence
from .support import support_score

_AUTHORITY = {
    "WHO": 1.0,
    "NICE": 1.0,
    "CDC": 0.95,
    "USPSTF": 0.95,
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Reranker:
    def __init__(self):
        self.settings = get_settings()
        self.backend = str(self.settings.get("retrieval_reranker_backend", "heuristic"))
        self._cross = None
        calibrated = bool(self.settings.get("retrieval_cross_encoder_calibrated", False))
        if self.backend == "cross_encoder" and calibrated:
            try:
                from sentence_transformers import CrossEncoder

                model = str(self.settings.get(
                    "retrieval_cross_encoder_model",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ))
                self._cross = CrossEncoder(model)
            except Exception:  # noqa: BLE001
                self._cross = None
                self.backend = "heuristic"

    def rerank(self, query: str, pool: list[Evidence], top_n: int | None = None) -> list[Evidence]:
        if self._cross is not None and pool:
            pairs = [(query, e.chunk.text) for e in pool]
            scores = self._cross.predict(pairs)
            for e, s in zip(pool, scores):
                e.rerank_score = float(s)
                e.raw_rerank_score = float(s)
        else:
            for e in pool:
                e.rerank_score = self._heuristic(query, e)
                e.raw_rerank_score = e.rerank_score

        ranked = sorted(pool, key=lambda e: -e.rerank_score)
        for i, e in enumerate(ranked):
            e.rank = i + 1
        if top_n is None:
            return ranked
        return ranked[:top_n]

    def uses_cross_encoder(self) -> bool:
        return self._cross is not None

    @staticmethod
    def _heuristic(query: str, e: Evidence) -> float:
        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        chunk_tokens = set(_TOKEN_RE.findall(e.chunk.text.lower()))
        if not q_tokens:
            return e.combined_score
        overlap = len(q_tokens & chunk_tokens) / max(1, len(q_tokens))
        section_tokens = set(_TOKEN_RE.findall(e.chunk.section_title.lower()))
        section_hit = 1.0 if (section_tokens and q_tokens & section_tokens) else 0.0
        authority = _AUTHORITY.get(e.chunk.organization.upper(), 0.8)
        directness = support_score(query, e)
        return (
            0.35 * directness
            + 0.20 * e.combined_score
            + 0.20 * overlap
            + 0.15 * section_hit
            + 0.10 * authority
        )