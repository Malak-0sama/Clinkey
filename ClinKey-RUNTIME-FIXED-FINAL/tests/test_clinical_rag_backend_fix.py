from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("CLINKEY_ALLOW_HASH", "1")

from src.ingestion.chunker import Chunk
from src.orchestration.pipeline import ClinKeyPipeline
from src.query.analyzer import QueryAnalyzer
from src.retrieval.models import Evidence
from src.retrieval.support import support_score
from src.sources.router import SourceRouter


EXACT_STEP1 = (
    "According to the NICE hypertension guideline, what is the recommended "
    "Step 1 pharmacological treatment for adults under 55 years who are not "
    "of Black African or African-Caribbean family origin?"
)

PARAPHRASED_STEP1 = (
    "In the NICE guidance, which antihypertensive drug class is recommended "
    "as the initial medication for a person younger than 55 years who is not "
    "of Black African or African-Caribbean family origin?"
)

STEP2 = (
    "According to the NICE hypertension guideline, what is recommended if "
    "blood pressure is not controlled after Step 1?"
)

WHO_THRESHOLD = (
    "What blood pressure threshold does the WHO hypertension fact sheet use "
    "to define hypertension?"
)

AMBIGUOUS = "What is the recommended treatment for this patient?"

INSUFFICIENT = (
    "What exact mortality reduction percentage does this Clinical RAG "
    "produce compared with a general-purpose LLM?"
)

OUT_OF_SCOPE = "What is the capital of France?"


class TestClinicalRagTargetCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = ClinKeyPipeline()
        cls.pipeline.knowledge.ensure_ready()

    def process(self, query: str, session: str):
        status = {"source_id": "offline", "status": "not_available", "chunks_added": 0}
        with patch.object(
            self.pipeline.knowledge.pipeline,
            "ingest_source",
            return_value=status,
        ):
            return self.pipeline.process(query, session_id=session)

    def test_a_exact_nice_step1(self):
        result = self.process(EXACT_STEP1, "target-a")
        self.assertFalse(result.abstained)
        self.assertNotEqual(result.confidence, "INSUFFICIENT_EVIDENCE")
        self.assertIn("NICE_HTN_003", {e["chunk_id"] for e in result.evidence})
        answer = (result.localized_answer or result.english_answer).lower()
        self.assertTrue("ace" in answer or "angiotensin-converting" in answer)
        self.assertIn("arb", answer)

    def test_b_paraphrased_nice_step1(self):
        result = self.process(PARAPHRASED_STEP1, "target-b")
        self.assertFalse(result.abstained)
        self.assertNotEqual(result.confidence, "INSUFFICIENT_EVIDENCE")
        self.assertIn("NICE_HTN_003", {e["chunk_id"] for e in result.evidence})
        answer = (result.localized_answer or result.english_answer).lower()
        self.assertTrue("ace" in answer or "angiotensin-converting" in answer)
        self.assertIn("arb", answer)

    def test_c_nice_step2(self):
        result = self.process(STEP2, "target-c")
        self.assertFalse(result.abstained)
        self.assertIn("NICE_HTN_004", {e["chunk_id"] for e in result.evidence})
        answer = (result.localized_answer or result.english_answer).lower()
        self.assertIn("step 2", answer)
        self.assertTrue("calcium-channel" in answer or "ccb" in answer)
        self.assertNotIn("step 3", answer)

    def test_d_who_threshold(self):
        result = self.process(WHO_THRESHOLD, "target-d")
        self.assertFalse(result.abstained)
        self.assertIn("WHO_HTN_001", {e["chunk_id"] for e in result.evidence})
        self.assertIn("140/90", result.localized_answer or result.english_answer)

    def test_e_ambiguous_requests_clarification(self):
        result = self.process(AMBIGUOUS, "target-e")
        self.assertTrue(result.abstained)
        self.assertEqual(result.confidence, "CLARIFICATION_REQUIRED")
        self.assertEqual(result.safety_classification, "ALLOWED")
        self.assertEqual(result.evidence, [])
        self.assertFalse(result.generation_called)
        self.assertIn("provide", result.localized_answer.lower())
        self.assertIn("diagnosis", result.localized_answer.lower())

    def test_f_insufficient_after_retry(self):
        result = self.process(INSUFFICIENT, "target-f")
        self.assertTrue(result.abstained)
        self.assertEqual(result.confidence, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.confidence_signals.get("retrieval_attempts"), 2)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.citations, [])
        self.assertFalse(result.generation_called)
        self.assertNotRegex(result.localized_answer, r"\b\d+(?:\.\d+)?\s*%")

    def test_g_out_of_scope_preserves_refusal(self):
        result = self.process(OUT_OF_SCOPE, "target-g")
        self.assertEqual(result.safety_classification, "REFUSE")
        self.assertEqual(result.evidence, [])
        self.assertFalse(result.generation_called)


class TestClinicalQueryProcessing(unittest.TestCase):
    def setUp(self):
        self.analyzer = QueryAnalyzer()

    def test_step1_paraphrase_normalization(self):
        analysis = self.analyzer.analyze(PARAPHRASED_STEP1)
        normalized = self.analyzer.normalize_for_retrieval(
            PARAPHRASED_STEP1, analysis
        ).lower()
        self.assertIn("step 1", normalized)
        self.assertIn("under 55", normalized)
        self.assertIn("pharmacological treatment", normalized)

    def test_step2_normalization(self):
        analysis = self.analyzer.analyze(STEP2)
        normalized = self.analyzer.normalize_for_retrieval(STEP2, analysis).lower()
        self.assertIn("step 2", normalized)
        self.assertIn("not controlled", normalized)

    def test_worded_step2_paraphrase_normalization(self):
        query = "What comes next when step one did not control hypertension?"
        analysis = self.analyzer.analyze(query)
        normalized = self.analyzer.normalize_for_retrieval(query, analysis).lower()
        self.assertIn("step 2", normalized)
        self.assertIn("not controlled", normalized)

    def test_cutoff_threshold_normalization(self):
        query = "What cutoff does the WHO use for high blood pressure?"
        analysis = self.analyzer.analyze(query)
        normalized = self.analyzer.normalize_for_retrieval(query, analysis).lower()
        self.assertIn("diagnostic threshold", normalized)
        self.assertIn("hypertension", normalized)

    def test_controlled_variant_limit(self):
        analysis = self.analyzer.analyze(PARAPHRASED_STEP1)
        variants = self.analyzer.retrieval_variants(
            PARAPHRASED_STEP1, analysis
        )
        self.assertGreaterEqual(len(variants), 2)
        self.assertLessEqual(len(variants), 3)
        self.assertEqual(len(variants), len({v.lower() for v in variants}))

    def test_ambiguity_not_triggered_for_specific_question(self):
        analysis = self.analyzer.analyze(EXACT_STEP1)
        self.assertEqual(
            self.analyzer.clarification_request(EXACT_STEP1, analysis),
            "",
        )

    def test_ambiguity_triggered_for_unspecified_patient(self):
        analysis = self.analyzer.analyze(AMBIGUOUS)
        clarification = self.analyzer.clarification_request(AMBIGUOUS, analysis)
        self.assertIn("diagnosis", clarification.lower())
        self.assertIn("age", clarification.lower())


class TestSourceAwareRoutingAndDirectness(unittest.TestCase):
    def test_nice_source_priority(self):
        selected = SourceRouter().route(EXACT_STEP1, 5)
        self.assertEqual(selected[0].source_id, "nice_hypertension_ng136")

    def test_generic_step2_prioritizes_nice(self):
        selected = SourceRouter().route("What is step 2 hypertension treatment?", 5)
        self.assertEqual(selected[0].source_id, "nice_hypertension_ng136")

    def test_who_source_priority(self):
        selected = SourceRouter().route(WHO_THRESHOLD, 5)
        self.assertEqual(selected[0].source_id, "who_hypertension")

    def test_symptom_definition_is_weak_for_step2_treatment(self):
        chunk = Chunk(
            chunk_id="symptom",
            source_id="nhlbi_high_blood_pressure",
            document_name="High Blood Pressure",
            organization="NIH (NHLBI)",
            page_number=1,
            section_title="Symptoms",
            source_url="https://www.nhlbi.nih.gov/health/high-blood-pressure",
            publication_date="2022",
            version="1",
            text="High blood pressure often has no symptoms and is found by measurement.",
        )
        score = support_score(STEP2, Evidence(chunk=chunk))
        self.assertLess(score, 0.35)

    def test_nice_step2_is_direct_for_step2_treatment(self):
        chunk = Chunk(
            chunk_id="step2",
            source_id="nice_hypertension_ng136",
            document_name="Hypertension NG136",
            organization="NICE",
            page_number=25,
            section_title="Treatment",
            source_url="https://www.nice.org.uk/guidance/ng136",
            publication_date="2019",
            version="NG136",
            text=(
                "Step 2: if blood pressure is not controlled, combine an ACE "
                "inhibitor or ARB with a calcium-channel blocker."
            ),
        )
        score = support_score(STEP2, Evidence(chunk=chunk))
        self.assertGreaterEqual(score, 0.55)


if __name__ == "__main__":
    unittest.main()
