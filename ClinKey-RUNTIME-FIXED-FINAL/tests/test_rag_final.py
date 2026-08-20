"""Final-phase E2E and contract tests."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CLINKEY_ALLOW_HASH", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import get_settings
from src.ingestion.chunker import Chunk
from src.ingestion.downloader import Downloader
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import Evidence
from src.retrieval.reranker import Reranker
from src.retrieval.sufficiency import assess_sufficiency
from src.retrieval.support import support_score
from src.validation.claim_support import ClaimCitationValidator


def _ch(text, cid="c", sid="who_hypertension"):
    return Chunk(
        chunk_id=cid, source_id=sid, document_name="d", organization="WHO",
        page_number=1, section_title="s", source_url="https://www.who.int/x",
        publication_date="2023", version="1", text=text,
    )


class TestRerankerBackend(unittest.TestCase):
    def test_default_is_heuristic_not_ce(self):
        rr = Reranker()
        self.assertFalse(rr.uses_cross_encoder())
        self.assertEqual(rr.backend, "heuristic")


class TestCalibrationThreshold(unittest.TestCase):
    def test_threshold_separates_labeled_pairs(self):
        thr = float(get_settings().get("retrieval_heuristic_min_rerank"))
        self.assertGreaterEqual(thr, 0.10)
        self.assertLessEqual(thr, 0.20)
        rr = Reranker()
        pos = Evidence(chunk=_ch("Step 2: combine an ACE inhibitor or ARB with a calcium-channel blocker."))
        neg = Evidence(chunk=_ch("Antimalarial treatment should be started promptly for falciparum malaria."))
        ps = rr.rerank("What is step 2 hypertension treatment?", [pos], 1)[0].rerank_score
        ns = rr.rerank("What is the recommended management of a femoral fracture?", [neg], 1)[0].rerank_score
        self.assertGreaterEqual(ps, thr)
        self.assertLess(ns, thr)

    def test_all_below_threshold_empty(self):
        from src.vectorstore.memory import MemoryStore
        import numpy as np
        store = MemoryStore()
        store.add([_ch("Antimalarial treatment should be started.", "m")], np.ones((1, 2), dtype="float32"))

        class E:
            def embed_query(self, q):
                return np.array([1.0, 0.0], dtype="float32")

        out = HybridRetriever(store, E()).retrieve(
            "What is the recommended management of a femoral fracture?",
            trusted_only=False,
        )
        self.assertEqual(out.evidence, [])
        self.assertFalse(out.generation_allowed)


class TestCandidateK(unittest.TestCase):
    def test_candidate_k_bounds_rerank_input(self):
        from src.vectorstore.memory import MemoryStore
        import numpy as np
        store = MemoryStore()
        chunks = [_ch(f"hypertension guideline chunk number {i} ACE inhibitor", f"c{i}") for i in range(20)]
        store.add(chunks, np.eye(20, 8, dtype="float32"))
        seen = {}

        class E:
            def embed_query(self, q):
                v = np.zeros(8, dtype="float32"); v[0] = 1; return v

        orig = Reranker.rerank

        def wrap(self, query, pool, top_n=None):
            seen["n"] = len(pool)
            return orig(self, query, pool, top_n)

        with patch.object(Reranker, "rerank", wrap):
            HybridRetriever(store, E()).retrieve("hypertension ACE", trusted_only=False)
        self.assertLessEqual(seen.get("n", 99), int(get_settings().get("retrieval_candidate_k", 15)))


class TestSupportNotSimilarity(unittest.TestCase):
    def test_same_disease_not_enough_for_dose(self):
        ev = Evidence(chunk=_ch("ACE inhibitors are recommended."))
        self.assertLess(support_score("What is the exact dose of lisinopril?", ev), 0.35)


class TestPartialPolicy(unittest.TestCase):
    def test_missing_steps_insufficient(self):
        ev = Evidence(chunk=_ch("Step 2: combine ACE inhibitor with CCB."))
        r = assess_sufficiency("What are treatment steps 1, 2, and 3?", [ev], ["treatment"])
        self.assertFalse(r["sufficient"])
        self.assertTrue(any("step" in a for a in r["missing_aspects"]))


class TestHTMLClean(unittest.TestCase):
    def test_nav_stripped(self):
        html = b"<html><nav>Home Menu</nav><main><p>ACE inhibitors are recommended.</p></main><footer>cookie</footer></html>"
        text = Downloader._html_to_text(html)
        self.assertIn("ACE", text)
        self.assertNotIn("Home Menu", text)


class TestPipelineNoGeneration(unittest.TestCase):
    def test_femoral_malaria_no_generate(self):
        from src.orchestration.pipeline import ClinKeyPipeline
        from src.generation.generator import GroundedGenerator
        called = {"n": 0}
        real = GroundedGenerator.generate

        def spy(self, *a, **k):
            called["n"] += 1
            return real(self, *a, **k)

        p = ClinKeyPipeline()
        # inject malaria-only store
        from src.vectorstore.memory import MemoryStore
        import numpy as np
        store = MemoryStore()
        store.add([_ch("Antimalarial treatment should be started promptly for falciparum malaria.", "mal", "who_malaria")],
                  np.ones((1, 4), dtype="float32"))
        p.knowledge.store = store
        p.retriever.store = store
        p.retriever.semantic.store = store
        with patch.object(GroundedGenerator, "generate", spy):
            r = p.process("What is the recommended management of a femoral fracture?", session_id="e2e-ff")
        self.assertTrue(r.abstained)
        self.assertNotEqual(r.confidence, "HIGH")
        self.assertEqual(called["n"], 0)
        self.assertFalse(r.generation_called)
        self.assertNotIn("malaria", (r.localized_answer or "").lower())


class TestCitationMismatch(unittest.TestCase):
    def test_wrong_cite(self):
        nice = Evidence(chunk=_ch("Step 2 combines ACEi/ARB with CCB.", "n", "nice_hypertension_ng136"))
        who = Evidence(chunk=_ch("An estimated 1.28 billion adults have hypertension.", "w"))
        r = ClaimCitationValidator().validate("Step 2 combines ACEi/ARB with CCB. [2]", [nice, who])
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()