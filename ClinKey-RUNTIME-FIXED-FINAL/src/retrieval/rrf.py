"""Reciprocal Rank Fusion."""
from __future__ import annotations


def rrf_score(ranks: list[int], k: int = 60) -> float:
    """RRF(d) = Σ 1 / (k + r_m(d)) for each ranking that contains d.

    ``ranks`` are 1-based. Missing modalities should be omitted, not passed as 0.
    """
    return sum(1.0 / (k + r) for r in ranks if r and r > 0)


def rrf_normalize(score: float, n_modalities: int = 2, k: int = 60) -> float:
    """Map RRF into [0, 1] using the theoretical max (all ranks = 1)."""
    peak = n_modalities / (k + 1.0)
    if peak <= 0:
        return 0.0
    return max(0.0, min(1.0, score / peak))