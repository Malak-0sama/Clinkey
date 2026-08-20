"""Reproducible retrieval + gate evaluation on the bundled corpus.

Usage:
    python evaluation/rag_evaluator.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.pipeline import ClinKeyPipeline  # noqa: E402


def _load():
    path = Path(__file__).parent / "datasets" / "retrieval_calibration.json"
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def evaluate_thresholds(rows: list[tuple[dict, object]], thresholds: list[float]) -> list[dict]:
    out = []
    for t in thresholds:
        tp = fp = tn = fn = 0
        for q, r in rows:
            gold = bool(q["expected_answerable"])
            top = (r.confidence_signals or {}).get("top") or r.confidence_score or 0.0
            pred = top >= t
            if gold and pred:
                tp += 1
            elif (not gold) and pred:
                fp += 1
            elif (not gold) and (not pred):
                tn += 1
            else:
                fn += 1
        n = max(1, tp + fp + tn + fn)
        far = fp / max(1, fp + tn)
        frr = fn / max(1, fn + tp)
        out.append({
            "threshold": t,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "false_acceptance": far,
            "false_rejection": frr,
            "abstention_rate": (tn + fn) / n,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return out


def precision_at_k(rows: list[tuple[dict, object]], k: int) -> dict:
    """Chunk-level Precision@K = (# gold-relevant chunks in top K) / K,
    averaged over labeled queries (those with a non-empty gold set)."""
    scores = []
    for q, r in rows:
        gold = set(q.get("relevant_chunk_ids") or [])
        if not gold:
            continue
        top_ids = [e.get("chunk_id") for e in r.evidence[:k]]
        hits = sum(1 for cid in top_ids if cid in gold)
        scores.append(hits / k)
    n = len(scores)
    avg = sum(scores) / n if n else 0.0
    return {"k": k, "precision_at_k": avg, "labeled_queries": n}


def recall_at_k(rows: list[tuple[dict, object]], k: int) -> dict:
    """Retrieval Recall@K = (# gold-relevant chunks retrieved in top K)
    / (total # gold-relevant chunks for that query), averaged over
    labeled queries (those with a non-empty gold set).

    This is distinct from the threshold-level "recall" in
    evaluate_thresholds(), which measures answerability accept/reject
    rate, NOT chunk-level retrieval recall. Per-query gold sets here are
    small (mostly single-chunk), so Recall@K is typically binary
    (0.0 or 1.0) per query; the reported figure is the mean across the
    labeled set, i.e. the fraction of queries for which top-K retrieval
    surfaced at least their full gold set.
    """
    scores = []
    for q, r in rows:
        gold = set(q.get("relevant_chunk_ids") or [])
        if not gold:
            continue
        top_ids = {e.get("chunk_id") for e in r.evidence[:k]}
        hits = len(gold & top_ids)
        scores.append(hits / len(gold))
    n = len(scores)
    avg = sum(scores) / n if n else 0.0
    return {"k": k, "recall_at_k": avg, "labeled_queries": n}


def mrr(rows: list[tuple[dict, object]]) -> dict:
    """Mean Reciprocal Rank over labeled queries.

    For each labeled query, reciprocal_rank = 1 / rank_of_first_relevant
    result in the full returned evidence order (0 if no gold chunk was
    retrieved at all). Averaged across labeled queries.
    """
    reciprocal_ranks = []
    per_query = []
    for q, r in rows:
        gold = set(q.get("relevant_chunk_ids") or [])
        if not gold:
            continue
        ranked_ids = [e.get("chunk_id") for e in r.evidence]
        rr = 0.0
        rank_found = None
        for i, cid in enumerate(ranked_ids, start=1):
            if cid in gold:
                rr = 1.0 / i
                rank_found = i
                break
        reciprocal_ranks.append(rr)
        per_query.append({"id": q["id"], "rank": rank_found, "reciprocal_rank": rr})
    n = len(reciprocal_ranks)
    avg = sum(reciprocal_ranks) / n if n else 0.0
    return {"mrr": avg, "labeled_queries": n, "per_query": per_query}


def main() -> None:
    questions = _load()
    pipeline = ClinKeyPipeline()
    print("Providers:", pipeline.providers_info())
    rows = []
    for q in questions:
        r = pipeline.process(q["query"], session_id=f"eval-{q['id']}")
        ids = [e.get("chunk_id") for e in r.evidence]
        gold = set(q.get("relevant_chunk_ids") or [])
        hit = bool(gold & set(ids)) if gold else None
        rows.append((q, r))
        print(
            f"{q['id']} answerable={q['expected_answerable']} "
            f"conf={r.confidence} score={r.confidence_score} "
            f"signals={r.confidence_signals} abstain={r.abstained} "
            f"gen={r.generation_called} gold_hit={hit} "
            f"top_chunks={ids[:3]}"
        )

    metrics = evaluate_thresholds(rows, [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.40])
    print("\nThreshold sweep (on recorded top relevance):")
    for m in metrics:
        print(m)

    # Prefer FAR == 0, then highest recall
    safe = [m for m in metrics if m["false_acceptance"] == 0]
    pick = max(safe, key=lambda m: m["recall"]) if safe else min(metrics, key=lambda m: m["false_acceptance"])
    print("\nSuggested conservative threshold:", pick)

    p3 = precision_at_k(rows, 3)
    p5 = precision_at_k(rows, 5)
    r3 = recall_at_k(rows, 3)
    r5 = recall_at_k(rows, 5)
    r10 = recall_at_k(rows, 10)
    m = mrr(rows)
    print("\n" + "=" * 40)
    print("RETRIEVAL QUALITY — gold relevant_chunk_ids")
    print(f"Precision@3: {p3['precision_at_k']:.3f}  (labeled queries: {p3['labeled_queries']})")
    print(f"Precision@5: {p5['precision_at_k']:.3f}  (labeled queries: {p5['labeled_queries']})")
    print(f"Recall@3:    {r3['recall_at_k']:.3f}  (labeled queries: {r3['labeled_queries']})")
    print(f"Recall@5:    {r5['recall_at_k']:.3f}  (labeled queries: {r5['labeled_queries']})")
    print(f"Recall@10:   {r10['recall_at_k']:.3f}  (labeled queries: {r10['labeled_queries']})")
    print(f"MRR:         {m['mrr']:.3f}  (labeled queries: {m['labeled_queries']})")
    print("=" * 40)
    return {
        "precision_at_3": p3, "precision_at_5": p5,
        "recall_at_3": r3, "recall_at_5": r5, "recall_at_10": r10,
        "mrr": m,
    }


if __name__ == "__main__":
    main()