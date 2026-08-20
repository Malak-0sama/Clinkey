# ClinKey — Final Agenda Compliance Report

## 1. Executive Summary

ClinKey was already a working, end-to-end RAG system implementing the
agenda's core theme (safe, evidence-grounded, traceable clinical guidance)
before this pass. This pass focused on the **real, verifiable gaps** found
during targeted inspection — it did not rewrite or replace working
architecture. All changes are additive or corrective, not structural.

**Status: COMPLETE.** All items in the requirement matrix below are COMPLETE
or FIXED. There are no PARTIAL, MISSING, or BLOCKED items remaining — the
one PARTIAL item from the previous report (architectural diagram) is closed
in this pass (§5b). §9 documents real, non-blocking implementation notes
only, not unresolved requirements.

## 2. Final Compliance Status

| Area (from Agenda / rubric) | Status | Notes |
|---|---|---|
| Ingestion (official public guideline PDFs, page-preserving extraction) | COMPLETE | Pre-existing, unchanged |
| Section-aware chunking, metadata schema (doc/page/section/chunk_id/url) | COMPLETE | Pre-existing, unchanged |
| Embeddings (BGE/sentence-transformers primary) | COMPLETE | Verified this pass with a stub model — primary path unaffected |
| Embeddings fallback (must not crash without sentence-transformers) | FIXED (prior pass) | Verified again this pass |
| Retrieval pipeline (Top-K tuning, hybrid, reranking) | COMPLETE | Pre-existing, unchanged |
| Evidence Panel UI (chunks, scores, metadata before generation) | COMPLETE | Verified wired to real `pipeline.process()` output, not mocked |
| Guardrail workflow (risk classification, confidence thresholds, unsupported-claim detection) | COMPLETE | Pre-existing; false-positive contradiction bug fixed (prior pass) |
| Grounded generation + citation mechanics | FIXED (prior pass) | Citation misattribution bug fixed and verified |
| Confidence tiers (High/Medium/Low/Insufficient) | COMPLETE | Pre-existing, unchanged |
| Retrieval Quality — Precision@K reported | FIXED (this + prior pass) | Now computed over **20 gold-labeled queries** (was 7) |
| Evaluation dataset ≥20 questions with real gold labels | FIXED (this pass) | 29 total queries, 20 with verified `relevant_chunk_ids` |
| Evaluation Depth — ≥2 quantitative metrics | COMPLETE | Precision@K (new) + threshold sweep precision/recall/FAR/FRR (pre-existing) |
| System architecture — 7-stage modular pipeline | COMPLETE | Pre-existing, unchanged |
| Safety & UX — disclaimers, confidence indicators, refusal logic | COMPLETE | Pre-existing, unchanged |
| Streamlit UI functional (no dead controls) | VERIFIED | See §4 — all 5 interactive controls trace to real handlers, none removed as none were dead |
| Session history (new/rename/delete) actually persists | VERIFIED (this pass) | Real code-level runtime check, see §4 |
| Live demo scenarios (success / multi-step / safe refusal) | COMPLETE | Verified via direct `pipeline.process()` calls in prior pass |
| Architectural diagram for judging | COMPLETE | `docs/architecture.mmd` (Mermaid, maintainable source of truth) + `docs/architecture.png` (rendered) — both accurately mirror the real 14-step flow in `src/orchestration/pipeline.py`, see §5b |

## 3. Streamlit Functionality Audit

The app is a single-page Streamlit chat interface (not multi-page), with
exactly 5 interactive control types. Each was traced from control → handler
→ backend call:

| Control | Handler | Backend call | Status |
|---|---|---|---|
| Example-question buttons | sets `pending_question`, reruns | → `pipeline.process()` on rerun | Real, verified |
| Chat input | reads question | → `pipeline.process(question, session_id=..., context_query=...)` | Real, verified (already exercised across 8 scenarios in the prior pass) |
| "+ New Chat" | resets session_id/messages | lazy-persisted via `HistoryRepository.add_message` on first message | Real, verified |
| "Save name" (rename) | → `repo.rename_session()` | `HistoryRepository` → JSON file | Real, verified this pass with fresh-instance reload |
| "Delete" | → `repo.delete_session()` | `HistoryRepository` → JSON file | Real, verified this pass with fresh-instance reload |

**No dead or fake controls were found.** No controls were removed, because
none were disconnected — there was nothing to fix or remove here beyond what
was already handled in prior passes (startup crash guard, evidence/citation
rendering already fed from real pipeline output).

**Limitation on UI verification method:** this environment does not have a
browser/UI automation tool available to me. I did **not** perform an actual
click-through in a rendered browser. What I *did* verify:
- The app boots and serves HTTP 200 without sentence-transformers installed (prior pass).
- Every control's Python handler was traced to confirm it calls real,
  non-mocked backend code (this pass, by direct source inspection).
- The session create/rename/delete flow was verified by calling
  `HistoryRepository` directly through the exact same methods the UI calls,
  across fresh instances (simulating an app restart/rerun) to confirm disk
  persistence, not just in-memory state.

This is real runtime/integration verification, but it is not equivalent to
a browser click-test. I'm stating that distinction explicitly rather than
implying more than was actually done.

## 4. RAG Preservation Verification

All of the following were verified to still pass after this pass's changes
(29-query dataset expansion, no RAG code touched):

- Navigation/boilerplate filtering — `test_hackathon_regression.test_nav_rejected_clinical_kept`, `test_rag_reliability.TestQuality.test_nav_rejected`
- Bounded candidate retrieval — `test_rag_final.TestCandidateK`
- Reranking / calibration threshold — `test_rag_final.TestRerankerBackend`, `TestCalibrationThreshold`
- Evidence sufficiency / no-dose abstention — `test_rag_reliability.TestSufficiencyDose.test_lisinopril_no_dose`, `test_hackathon_regression.test_lisinopril_dose_abstain`
- Topic/evidence support validation — `test_rag_reliability.TestSupport` (femoral vs malaria, HTN vs fracture, step2 compatibility)
- Hard generation gate / no-best-bad-fallback — `test_rag_final.TestPipelineNoGeneration.test_femoral_malaria_no_generate`
- Citation traceability / mismatch detection — `test_rag_reliability.TestCitationMismatch.test_wrong_source_cite`
- Contradiction detection (real conflicts still caught) — `test_rag_reliability.TestContradiction.test_conflict`
- **Femoral-fracture → malaria regression** — 4 independent tests, all passing: `test_rag_final.test_femoral_malaria_no_generate`, `test_hackathon_regression.test_femoral_abstain`, `test_hackathon_regression.test_malaria_not_support_fracture`, `test_rag_reliability.test_femoral_vs_malaria`

No RAG module (retrieval, reranking, confidence, support/sufficiency,
generation, citation validation) was modified in this pass.

## 5. Evaluation Dataset — What Changed and Why

## 5b. Architecture Diagram (this pass)

`docs/architecture.mmd` (Mermaid source, human-maintainable) and
`docs/architecture.png` (rendered image) were added. The diagram was built
directly from the numbered stage comments and actual control flow in
`src/orchestration/pipeline.py::ClinKeyPipeline._run` — every node name,
branch, and abstention path (translation failure, safety refusal, retrieval
exception, hard evidence gate, unsupported claim, contradiction/CONFLICT,
citation/medical check failure) corresponds to a real `return` branch in
that method. No fictional components were added; nothing was invented.

`docs/architecture.dot` (Graphviz source) is also included — it's the
intermediate file used to render the PNG. Mermaid's own renderer
(`mmdc`) requires downloading a headless-Chrome binary from a host outside
this sandbox's allowed network egress list, so it could not render safely.
Graphviz was already installed and required no network access, so it was
used to render an accurate PNG of the same content instead. `docs/architecture.mmd`
remains the primary, maintainable source — regenerate `architecture.dot`/`.png`
from it (or re-attempt `mmdc` in an environment with broader network access)
if the pipeline's stage order changes.


**Before:** 13 queries in `retrieval_calibration.json`, only 7 with gold
`relevant_chunk_ids`.

**After:** 29 queries, 20 with gold `relevant_chunk_ids`.

Every new gold label was derived by reading the actual chunk text in
`data/demo_knowledge/guidelines.json` (46 real chunks across 11 documents)
and hand-verifying the chunk genuinely answers the query — none were
invented. All 20 gold chunk IDs were cross-checked programmatically against
the corpus to confirm they exist (`invalid gold ids: []`).

The new queries include:
- 13 new direct-evidence questions across hypertension, diabetes, asthma,
  depression, obesity, and influenza (previously only hypertension/diabetes/
  asthma/obesity had gold coverage)
- 1 cross-organization case reused from the existing set style (WHO vs CDC
  phrasing of the same fact)
- 3 new **distractor** cases — questions about a real topic in the corpus
  (hypertension drug class, diabetes) where the *specific* fact asked
  (exact drug dosage, gestational diabetes, pregnancy safety) is genuinely
  absent, so the correct behavior is abstention, not a wrong grounded answer

Resulting Precision@K (measured this pass, 20 labeled queries):