# ClinKey — Medical Evidence RAG

A multilingual, evidence-grounded medical RAG assistant built as a production-quality,
hackathon-ready Python project. Ask a health question in any supported language — ClinKey
detects the language, grounds the answer in official clinical guidelines, and returns a cited
answer back in your language.

> ClinKey provides general, guideline-based information. It does not diagnose, treat, or
> replace professional medical advice.

---

## Key features

- **Multilingual end-to-end** — ask in English, Arabic, French, Spanish, German, Italian,
  Portuguese, Turkish, Hindi, Urdu or Chinese. ClinKey detects the language, runs the full RAG
  pipeline in a canonical English representation, then translates the validated answer back.
- **Evidence-grounded, not generative** — the language model is an evidence synthesizer, never
  the medical authority. Every substantive claim is tied to a retrieved chunk with a citation.
- **Safety layer** — classifies queries as `ALLOWED`, `CAUTION`, `REFUSE`, or `EMERGENCY`
  (and handles emergencies/refusals in the user's own language).
- **Full RAG stack** — hybrid retrieval (semantic + BM25), reranking, retrieval-confidence
  estimation, claim validation, citation generation, localization validation.
- **Light and dark themes** built from the ClinKey brand palette extracted from the logo
  (navy `#003060` → blue `#0070B0` → cyan `#0080C0`).
- **Zero-config demo mode** — runs fully offline with bundled evidence. Add a Gemini key to
  enable live language-model translation, routing, generation and validation.
- **Official clinical sources** — WHO, CDC, NICE, USPSTF, NIH, NCBI (PubMed/PMC), IDSA, AHA,
  ESC, ADA, MedlinePlus, FDA, EMA, NHS, Cochrane, Health Canada and the Australian Department
  of Health. The approved catalog (`configs/approved_sources.json`) holds 40 validated
  guideline sources; the backend downloads and ingests them on demand.
- **Local chat history** — sessions are saved on disk and can be searched or deleted from the sidebar.

---

## Architecture
