# ClinKey RAG + Translation — production audit

## Executive summary

The prior stack mixed uncalibrated scores, rebuilt BM25 from `all_chunks()` every query, used lexical overlap as “grounding,” and could translate clinical numbers/citations unsafely. This pass adds **source-policy filtering**, **evidence sufficiency**, **medical safety validation** (numbers, units, negation, recommendation strength), **citation integrity**, **contradiction flags**, **mixed AR/EN detection**, and **protected translation**. Hash embeddings still **fail closed in production** unless `CLINKEY_ALLOW_HASH=1` or `embeddings.provider=hash`.

## Verdict

**NOT PRODUCTION READY**

Blocking:

- No sentence-transformers / BGE in this environment (dense retrieval is hash-only when explicitly allowed).
- BM25 still built per query from in-memory chunks (not a 1M-scale index).
- No calibrated probabilities; qualitative evidence status remains the honest user signal.
- Live HTML ingest can still pollute the store.
- Gemini generation/translation paths not exercised here (no API key).

## Tests (actual, 2026-08-18)
