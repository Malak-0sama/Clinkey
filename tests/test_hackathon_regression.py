"""Hackathon regression: NICE HTN when evidence exists; abstain otherwise."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CLINKEY_ALLOW_HASH", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.quality import ChunkQualityFilter
from src.ingestion.chunker import Chunk
from src.language.detector import LanguageDetector
from src.orchestration.pipeline import ClinKeyPipeline
from src.retrieval.models import Evidence
from src.retrieval.support import support_score


def _ch(text: str) -> Chunk:
    return Chunk(
        chunk_id="x", source_id="s", document_name="d", organization="NICE",
        page_number=1, section_title="t", source_url="https://www.nice.org.uk/x",
        publication_date="2019", version="1", text=text,
    )


class TestHackathonPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = ClinKeyPipeline()

    def test_step2_uses_nice_not_guess(self):
        r = self.p.process("What is step 2 hypertension treatment?", session_id="hk-s2")
        text = (r.localized_answer or "").lower()
        if r.abstained:
            self.fail("step 2 is in bundled NICE evidence and should answer, got abstain: " + text[:120])
        self.assertTrue("ace" in text or "arb" in text or "calcium" in text)
        self.assertNotIn("lisinopril 10", text)

    def test_step3_thiazide(self):
        r = self.p.process("What is step 3 hypertension treatment?", session_id="hk-s3")
        text = (r.localized_answer or "").lower()
        if r.abstained:
            self.fail("step 3 is in bundled NICE evidence: " + text[:120])
        self.assertIn("thiazide", text)

    def test_under55_ace_or_arb(self):
        r = self.p.process(
            "According to NICE guidance, what is the first-line pharmacological "
            "treatment for hypertension in adults under 55 years?",
            session_id="hk-u55",
        )
        text = (r.localized_answer or "").lower()
        if r.abstained:
            self.fail("NICE step 1 under-55 is in corpus: " + text[:120])
        self.assertTrue("ace" in text or "arb" in text)

    def test_lisinopril_dose_abstain(self):
        r = self.p.process("What exact dose of lisinopril is recommended at step 2?", session_id="hk-dose")
        self.assertTrue(r.abstained)
        self.assertFalse(r.generation_called)

    def test_femoral_abstain(self):
        r = self.p.process("What is the recommended treatment for a femoral fracture?", session_id="hk-ff")
        self.assertTrue(r.abstained)
        self.assertNotIn("malaria", (r.localized_answer or "").lower())

    def test_appendicitis_abstain(self):
        r = self.p.process("What is the recommended treatment for acute appendicitis?", session_id="hk-ap")
        self.assertTrue(r.abstained)

    def test_parkinson_abstain(self):
        r = self.p.process("What is the first-line treatment for Parkinson's disease?", session_id="hk-pk")
        self.assertTrue(r.abstained)

    def test_english_stays_english(self):
        r = self.p.process("What are the symptoms of hypertension?", session_id="hk-en")
        self.assertEqual(r.detected_language, "en")

    def test_guidelines_meta_word_not_required(self):
        """Regression: 'According to the guidelines...' must not hard-reject
        correct, on-topic NICE evidence just because the meta-word
        'guidelines' (a reference to the source type, not clinical content)
        doesn't appear verbatim in the retrieved chunk text. See
        ConfidenceEstimator._rare_terms_supported.
        """
        r = self.p.process(
            "According to the guidelines, what is the first-line pharmacological "
            "treatment for hypertension in adults under 55 years?",
            session_id="hk-meta",
        )
        text = (r.localized_answer or "").lower()
        if r.abstained:
            self.fail("meta-word 'guidelines' incorrectly caused abstention: " + text[:160])
        self.assertTrue("ace" in text or "arb" in text)
        self.assertEqual(r.confidence_signals.get("reason"), None)

    def test_pancreatic_cancer_abstains(self):
        r = self.p.process("What is the recommended treatment for pancreatic cancer?", session_id="hk-panc")
        self.assertTrue(r.abstained)
        self.assertFalse(r.generation_called)

    def test_arabic_detected(self):
        d = LanguageDetector()
        self.assertEqual(d.detect("ما هو العلاج في الخطوة الثانية لارتفاع ضغط الدم؟").language_code, "ar")


class TestSupportAndQuality(unittest.TestCase):
    def test_malaria_not_support_fracture(self):
        ev = Evidence(chunk=_ch("Antimalarial treatment should be started for falciparum malaria."))
        self.assertLess(support_score("What is the recommended treatment for a femoral fracture?", ev), 0.35)

    def test_nav_rejected_clinical_kept(self):
        qf = ChunkQualityFilter()
        self.assertFalse(qf.evaluate(_ch("Home\nRead more\nBack to top")).accepted)
        self.assertTrue(qf.evaluate(_ch(
            "Step 2: combine an ACE inhibitor or ARB with a calcium-channel blocker."
        )).accepted)


if __name__ == "__main__":
    unittest.main()