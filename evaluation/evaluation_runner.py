"""ClinKey evaluation runner.

Runs the benchmark in configs/../evaluation/test_questions.json through the
full pipeline and reports metrics:

LANGUAGE   — detection accuracy, translation success
RETRIEVAL  — evidence retrieved, confidence distribution
SAFETY     — refusal/emergency/caution classification accuracy
LOCALIZATION — localization success
SYSTEM     — latency (ms)

Usage:
    python evaluation/evaluation_runner.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestration.pipeline import ClinKeyPipeline  # noqa: E402


def main() -> None:
    bench_path = Path(__file__).with_name("test_questions.json")
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    questions = bench["questions"]

    pipeline = ClinKeyPipeline()
    print(f"Providers: {pipeline.providers_info()}\n")

    rows = []
    for q in questions:
        t0 = time.time()
        r = pipeline.process(q["question"])
        r.elapsed_ms = int((time.time() - t0) * 1000)
        rows.append((q, r))

    # ---- metrics ----
    n = len(rows)
    det_ok = sum(1 for q, r in rows if r.detected_language == q["expected_language"])
    trans_ok = sum(1 for q, r in rows if r.input_translation_ok or r.detected_language == "en")
    answered = sum(1 for q, r in rows if (r.localized_answer or r.english_answer) and not r.error)
    safety_ok = sum(
        1 for q, r in rows
        if q.get("expected_safety") and r.safety_classification == q["expected_safety"]
    )
    safety_n = sum(1 for q, r in rows if q.get("expected_safety"))
    conf = {}
    for q, r in rows:
        conf[r.confidence] = conf.get(r.confidence, 0) + 1
    loc_ok = sum(1 for q, r in rows if r.localization_ok)
    avg_ms = sum(r.elapsed_ms for q, r in rows) / max(1, n)

    print("=" * 70)
    print(f"LANGUAGE    detection accuracy : {det_ok}/{n} ({100*det_ok/n:.0f}%)")
    print(f"            translation success: {trans_ok}/{n} ({100*trans_ok/n:.0f}%)")
    print(f"RETRIEVAL   answered            : {answered}/{n}")
    print(f"            confidence dist     : {conf}")
    print(f"SAFETY      classification acc  : {safety_ok}/{safety_n}" + (f" ({100*safety_ok/safety_n:.0f}%)" if safety_n else ""))
    print(f"LOCALIZATION localization ok    : {loc_ok}/{n}")
    print(f"SYSTEM      avg latency         : {avg_ms:.0f} ms")
    print("=" * 70)

    print("\nPer-question detail:")
    for q, r in rows:
        flag = "✓" if (r.detected_language == q["expected_language"] and not r.error) else "✗"
        print(
            f"{flag} [{q['category']:<18}] lang={r.detected_language} "
            f"conf={r.confidence:<10} safety={r.safety_classification:<9} "
            f"cites={len(r.citations)} {r.elapsed_ms:>5}ms  {q['question'][:44]}"
        )
        if r.error:
            print(f"      ERROR: {r.error[:120]}")


if __name__ == "__main__":
    main()