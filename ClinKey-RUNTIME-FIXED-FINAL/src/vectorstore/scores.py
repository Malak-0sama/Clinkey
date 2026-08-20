"""Canonical similarity contract for all vector-store adapters.

Returned scores MUST be in [0, 1]:
  1.0 = identical
  0.0 = orthogonal or opposite
"""
from __future__ import annotations


def chroma_distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine *distance* d to similarity S in [0, 1].

    Cosine distance is typically 1 - cosine_similarity (0 identical, 2 opposite).
    We clamp d into [0, 1] then invert, so opposite/orthogonal both map to 0.
    """
    d = max(0.0, min(1.0, float(distance)))
    return 1.0 - d


def cosine_to_unit_similarity(cosine: float) -> float:
    """Map raw cosine in [-1, 1] onto the [0, 1] contract (negatives → 0)."""
    return max(0.0, min(1.0, float(cosine)))