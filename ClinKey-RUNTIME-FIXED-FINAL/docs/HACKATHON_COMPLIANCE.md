# ClinKey — hackathon final compliance

## Verdict

**READY FOR HACKATHON SUBMISSION**

(Not a claim of hospital-grade production. Safety gates work; BGE/CE/Gemini are optional upgrades.)

## This pass (minimum fixes)

| File | Change | Why |
|---|---|---|
| `src/retrieval/confidence.py` | Skip 0.70 cosine floor on **hash** embeddings; topic synonyms; rare-term aliases | Hash scale ≠ BGE; NICE step Qs were wrongly abstaining |
| `src/retrieval/support.py` | Step 1/2/3 / first-line need matching step language | Cross-topic / wrong-step chunks |
| `src/generation/generator.py` | Reuse citation numbers; keep only asked-step sentences | `[3]` on 2 chunks; step 3 leaking into step 2 |
| `tests/test_hackathon_regression.py` | NICE under-55 / step 2 / step 3; dose/fracture/appendicitis/Parkinson abstain; AR detect | Required cases |

## Tests (actual)
