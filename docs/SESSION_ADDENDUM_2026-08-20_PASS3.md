# Session Addendum — Pass 3 (runtime bug fix)

## What was actually wrong (reproduced, not assumed)

Ran the 6 diagnostic queries from the bug report directly against
`ClinKeyPipeline.process()` before touching any code. Confirmed:

- Query 1 ("According to the guidelines, what is the first-line
  pharmacological treatment...") — **abstained**, `reason: missing_rare_terms`.
  `NICE_HTN_003` WAS retrieved (semantic 0.779) but the gate rejected it.
- Query 2 (age/family-origin/tolerance question) — **abstained**, same
  `missing_rare_terms` reason.
- Queries 3, 4, 5, 6 (step 2, step 3, pancreatic cancer, emergency) — already
  behaved correctly before any fix.

## Root cause (found in `src/retrieval/confidence.py::_rare_terms_supported`)

Any query token ≥8 characters not in a small hardcoded `generic` set had to
appear **literally** in the top-5 retrieved chunk text, or the whole answer
was rejected — regardless of how strong the retrieval scores were. `"guidance"`
was already in that generic set; `"guidelines"` (the same meaning, different
inflection) was not — a plain oversight. Query 2 additionally tripped on
`"selected"` and `"tolerated"`, generic care-pathway verbs, not clinical
entities.

This is exactly the "meta-word" bug described in the report, and it's the
real, sole cause of both abstentions — confirmed by testing before/after.

## Fix (smallest safe change — one function, one file)

Expanded the `generic` exclusion set in `_rare_terms_supported()` with the
missing inflection and a few equivalent generic/procedural words
(`guideline`, `guidelines`, `medication`, `medications`, `selected`,
`initial`, `tolerated`, `prescribed`, `administered`, `patient`, `patients`).
This is a category-level fix (words describing *how a question is framed*,
not specific clinical entities) — **not** a per-question hardcode: it doesn't
reference any of the 6 test questions, drugs, or expected answers, and the
gate still requires genuine rare clinical terms (e.g. "pancreatic",
"malaria") to be textually supported.

Nothing else was touched: retrieval, reranking, embeddings, the hard
evidence gate's other checks, claim validation, citation validation, and
safety classification are all unmodified.

## Verified after the fix

All 6 diagnostic queries re-run live against the pipeline:

| # | Query | Before | After |
|---|---|---|---|
| 1 | Step 1, "according to the guidelines" | abstained (missing_rare_terms) | **answers correctly**, cites NICE_HTN_003, mentions ACE/ARB |
| 2 | Age/family-origin/tolerance | abstained (missing_rare_terms) | **no longer abstains** — see honest caveat below |
| 3 | Step 2 | already correct | unchanged, correct |
| 4 | Step 3 | already correct | unchanged, correct |
| 5 | Pancreatic cancer | already correct (abstain) | unchanged, correct abstain |
| 6 | Emergency dosing | already correct (EMERGENCY, no dose) | unchanged, correct |

**Honest caveat on Query 2:** the gate bug is fixed and it now generates an
answer, but under this sandbox's hash-embedding fallback, retrieval surfaces
WHO general-hypertension chunks rather than the ideal `NICE_HTN_003`/`004`
chunks that most directly answer the age/family-origin distinction. Two
things could not be verified from this environment and are stated
explicitly rather than assumed:

1. **Real BGE embeddings could not be tested here.** This sandbox's network
   is blocked from `huggingface.co` (403) and does not have enough free
   disk space to install `torch`/`sentence-transformers` (attempted; failed
   with `OSError: No space left on device`). Hash-embedding semantic scores
   for this long compound query were weak (0.2–0.28) across the board, which
   is consistent with — but not proof of — a hash-embedding ranking weakness
   rather than an architecture defect. This should be re-verified against
   the real BGE stack the report confirms is working locally.
2. **The bundled demo corpus has no chunk discussing "if the initial
   treatment is not tolerated."** `NICE_HTN_003`/`004` cover step 1–3 drug
   selection by age/family origin, but nothing in the corpus addresses
   switching/tolerance. Even with perfect retrieval, the *tolerance* half of
   this compound question cannot be honestly grounded from this corpus — the
   system correctly has no evidence for that part, not a bug. Forcing a full
   answer to that part would mean generating from general knowledge, which
   the architecture (correctly) forbids.

## Regression discrepancy noted (not silently accepted)

The bug report stated the baseline was "72 passed, 2 failed" with
`test_step2_uses_nice_not_guess` and `test_step3_thiazide` failing. Running
the actual test suite in the uploaded repository, **before any change**,
showed **74 passed, 0 failed** — those two tests were already passing, and
Queries 3/4 already worked correctly when reproduced live. This is stated
plainly rather than fabricating a "fix" for a failure that wasn't present in
this repository state. (Two *different* prior-session addenda in this same
repo — `SESSION_ADDENDUM_2026-08-20.md` and `..._PASS2.md` — independently
document a 2-failure state caused by `langdetect` not being installed in
*those* sandboxes, which is a different, already-understood, dependency
issue, not this bug.)

## Full test suite (this pass)
