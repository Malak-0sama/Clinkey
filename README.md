# ClinKey

ClinKey is a medical RAG assistant. You ask a health question in whatever language you want, it finds the relevant bits of official clinical guidelines, and it gives you an answer back in your language with citations.

The whole point is that it doesn't make things up. The LLM here is a summarizer, not a doctor. If a claim isn't backed by a retrieved chunk from an approved source, it doesn't go in the answer.

> ClinKey provides general, guideline-based information. It does not diagnose, treat, or replace professional medical advice.

---

## What it actually does

- **Multilingual, both directions.** English, Arabic, French, Spanish, German, Italian, Portuguese, Turkish, Hindi, Urdu, Chinese. It detects the language you wrote in, does all the retrieval and reasoning in English, then translates the validated answer back. Emergencies and refusals get localized too, which matters more than it sounds.
- **Cited or it didn't happen.** Every substantive sentence is tied to a chunk with a source reference. No citation, no claim.
- **Safety routing.** Queries get classified as `ALLOWED`, `CAUTION`, `REFUSE`, or `EMERGENCY` before retrieval even starts.
- **The full stack, not a wrapper.** Hybrid retrieval (dense + BM25), reranking, confidence estimation, claim validation, citation generation, and localization validation on the way back out.
- **Offline by default.** The repo ships with bundled evidence so you can clone it and run it with no API key. Add a Gemini key when you want live translation and generation.
- **Light and dark themes**, built from the brand palette: `#003060` → `#0070B0` → `#0080C0`.
- **Chat history on disk.** Searchable, deletable, stays local.

---

## Languages

| Language | Code |
|---|---|
| English | `en` |
| Arabic | `ar` |
| French | `fr` |
| Spanish | `es` |
| German | `de` |
| Italian | `it` |
| Portuguese | `pt` |
| Turkish | `tr` |
| Hindi | `hi` |
| Urdu | `ur` |
| Chinese | `zh` |

Detection is automatic. You don't pick a language, you just type.

---

## How a query flows through the system

```
user query (any language)
        |
        v
  language detection
        |
        v
  safety classification  ->  ALLOWED / CAUTION / REFUSE / EMERGENCY
        |
        v
  translate to English (canonical form)
        |
        v
  hybrid retrieval (dense + BM25)
        |
        v
  rerank + confidence check
        |
        v
  generate from retrieved chunks only
        |
        v
  validate each claim against evidence
        |
        v
  attach citations
        |
        v
  translate back + re-validate localization
        |
        v
  answer, in your language, with sources
```

If retrieval confidence comes back low, the pipeline is supposed to say so rather than guess. That's the part that separates this from a chatbot with a vector store bolted on.

---

## Running it

You need Python 3.10 or newer. pip works, `uv` works too.

```bash
git clone https://github.com/Malak-0sama/Clinkey.git
cd Clinkey
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Demo mode (no API key, runs offline):**

```bash
streamlit run app/main.py
```

**Live mode:**

```bash
export GEMINI_API_KEY="your-key-here"
streamlit run app/main.py
```

Live mode turns on the real translation, routing, generation, and validation calls. Demo mode uses the bundled evidence in `data/demo_knowledge/` and skips anything that needs the network.

---

## Config

| Variable | Required | What it does |
|---|---|---|
| `GEMINI_API_KEY` | No | Enables live mode. Without it, ClinKey runs entirely offline. |
| `CLINKEY_THEME` | No | `light` or `dark`. |
| `CLINKEY_DATA_DIR` | No | Where chat history and ingested evidence live. Defaults to `data/`. |

The source catalog is at `configs/approved_sources.json`. It currently lists 40 validated guideline sources from WHO, CDC, NICE, USPSTF, NIH, NCBI (PubMed/PMC), IDSA, AHA, ESC, ADA, MedlinePlus, FDA, EMA, NHS, Cochrane, Health Canada, and Australia's Department of Health. The backend fetches and ingests them on demand. Add or remove entries in that file to change what ClinKey is allowed to cite.

---

## Layout

```
Clinkey/
├── .streamlit/              # Streamlit config (theme, server settings)
├── app/                     # Streamlit UI and entrypoint
├── configs/                 # Source catalog and pipeline config
│   └── approved_sources.json
├── data/
│   └── demo_knowledge/      # Bundled offline evidence
├── docs/                    # Architecture notes
└── README.md
```

---

## Safety

Every query gets classified before anything else happens.

- `ALLOWED` — normal evidence-grounded answer with citations.
- `CAUTION` — answer, plus a nudge to check with a clinician.
- `REFUSE` — declines. Covers things like diagnosis requests and prescription dosing.
- `EMERGENCY` — escalates with emergency guidance, written in the language the user actually used.

That last point is deliberate. If someone is describing chest pain in Urdu at 3am, an English-only emergency message is worse than useless.

---


