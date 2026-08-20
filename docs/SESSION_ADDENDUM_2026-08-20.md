# Session Addendum — 2026-08-20

This addendum documents work done in this session on top of
`ClinKey-FINAL-AGENDA-COMPLIANT-v2`. It follows the anti-fabrication rules
in the compliance-pass instructions: only claims that were actually
verified in this session are marked done.

## Environment constraint (read this first)

This session ran in a sandboxed environment whose outbound network is
limited to an allowlist that does **not** include
`generativelanguage.googleapis.com` (Gemini) or `huggingface.co`
(sentence-transformers model weights). No `GEMINI_API_KEY` was configured.

Consequence: all pipeline runs and test runs in this session used the
codebase's own **hash-embedding dev fallback**
(`CLINKEY_ALLOW_HASH=1`, `providers: {llm: demo, embeddings: hash,
vectorstore: memory}`), not the production Gemini + sentence-transformers
stack. Numbers below are real (not fabricated) but were produced in that
fallback mode. **They must be re-run against the live stack (real
`GEMINI_API_KEY`, `sentence-transformers` installed) before being quoted
as production metrics.** The commands to reproduce are given below so
that re-run is a one-step check, not new engineering work.

## 1. Test suite — verified
