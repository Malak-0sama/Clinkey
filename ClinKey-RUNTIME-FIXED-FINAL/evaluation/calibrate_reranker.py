"""Empirical heuristic/CE rerank calibration. Prints real score stats."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.chunker import Chunk
from src.retrieval.models import Evidence
from src.retrieval.reranker import Reranker


def main() -> None:
    data = json.loads((Path(__file__).parent / "datasets" / "rerank_calibration.json").read_text())
    rr = Reranker()
    pos, neg = [], []
    print("reranker_cross_encoder=", rr.uses_cross_encoder())
    for p in data["pairs"]:
        ev = Evidence(chunk=Chunk(
            chunk_id=p["id"], source_id="x", document_name="d", organization="WHO",
            page_number=1, section_title="s", source_url="", publication_date="",
            version="1", text=p["evidence"],
        ))
        scored = rr.rerank(p["query"], [ev], 1)[0]
        row = (p["id"], p["label"], scored.rerank_score)
        print(row)
        (pos if p["label"] == "positive" else neg).append(scored.rerank_score)

    def stats(xs, name):
        xs = sorted(xs)
        print(f"{name} n={len(xs)} min={min(xs):.4f} max={max(xs):.4f} "
              f"mean={statistics.mean(xs):.4f} median={statistics.median(xs):.4f}")

    stats(pos, "POSITIVE")
    stats(neg, "NEGATIVE")
    # conservative: above 75th percentile of negatives, below min positive if possible
    neg_sorted = sorted(neg)
    p75 = neg_sorted[int(0.75 * (len(neg_sorted) - 1))]
    suggested = max(p75, 0.30)
    print("suggested_heuristic_threshold", round(suggested, 3))


if __name__ == "__main__":
    main()