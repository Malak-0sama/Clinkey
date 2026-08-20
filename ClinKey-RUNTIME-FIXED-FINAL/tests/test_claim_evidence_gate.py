from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("CLINKEY_ALLOW_HASH", "1")

from src.ingestion.chunker import Chunk
from src.orchestration.pipeline import ClinKeyPipeline
from src.retrieval.claim_evidence import (
    PERCENTAGE_OUTCOME,
    validate_claim_evidence,
)
from src.retrieval.models import Evidence

STEP2_TREATMENT = "What is the NICE Step 2 treatment for uncontrolled hypertension?"
STEP2_PERCENTAGE = (
    "What exact percentage of patients achieve blood pressure control after "
    "Step 2 treatment according to the NICE hypertension guideline?"
)
STEP2_PERCENTAGE_SHORT = (
    "What exact percentage of patients achieve BP control after Step 2?"
)
AMBIGUOUS = "What is the recommended treatment for this patient?"
OUT_OF_SCOPE = "What is the capital of France?"


def evidence(
    chunk_id: str,
    text: str,
    organization: str = "NICE",
    source_id: str = "nice_hypertension_ng136",
    document: str = "Hypertension NG136",
    section: str = "Treatment",
) -> Evidence:
    item = Evidence(
        chunk=Chunk(
            chunk_id=chunk_id,
            source_id=source_id,
            document_name=document,
            organization=organization,
            page_number=1,
            section_title=section,
            source_url="https://example.org/source",
            publication_date="2024",
            version="1",
            text=text,
        ),
        support_score=0.9,
        rerank_score=0.9,
        relevance_score=0.9,
    )
    return item


class TestClaimEvidencePipeline(unittest.TestCase):
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

    def test_a_step2_treatment_supported_high(self):
        result = self.process(STEP2_TREATMENT, "claim-a")
        self.assertFalse(result.abstained)
        self.assertEqual(result.confidence, "HIGH")
        self.assertTrue(result.generation_called)
        self.assertEqual(
            [item["chunk_id"] for item in result.evidence],
            ["NICE_HTN_004"],
        )
        self.assertTrue(all(item["directly_supports_claim"] for item in result.evidence))

    def test_b_unsupported_step2_percentage_abstains(self):
        result = self.process(STEP2_PERCENTAGE, "claim-b")
        self.assertTrue(result.abstained)
        self.assertEqual(result.confidence, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.confidence_signals.get("reason"), "unsupported_requested_claim")
        self.assertEqual(result.confidence_signals.get("retrieval_attempts"), 2)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.citations, [])
        self.assertFalse(result.generation_called)
        self.assertEqual(
            result.confidence_signals["claim_requirement"]["required_fact_type"],
            PERCENTAGE_OUTCOME,
        )

    def test_b_short_bp_percentage_query_abstains(self):
        result = self.process(STEP2_PERCENTAGE_SHORT, "claim-b-short")
        self.assertEqual(result.safety_classification, "ALLOWED")
        self.assertTrue(result.abstained)
        self.assertEqual(result.confidence, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.confidence_signals.get("reason"), "unsupported_requested_claim")
        self.assertEqual(result.confidence_signals.get("retrieval_attempts"), 2)
        self.assertFalse(result.generation_called)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.citations, [])

    def test_c_fda_narcolepsy_does_not_route_hypertension_sources(self):
        result = self.process(
            "What is the FDA treatment for narcolepsy?",
            "claim-c-route",
        )
        self.assertEqual(result.safety_classification, "ALLOWED")
        self.assertTrue(result.selected_source_ids)
        self.assertTrue(
            all("fda" in source_id.lower() for source_id in result.selected_source_ids)
        )
        self.assertFalse(
            any("hypertension" in source_id.lower() for source_id in result.selected_source_ids)
        )

    def test_d_ambiguous_patient_requires_clarification(self):
        result = self.process(AMBIGUOUS, "claim-d")
        self.assertEqual(result.confidence, "CLARIFICATION_REQUIRED")
        self.assertTrue(result.abstained)
        self.assertFalse(result.generation_called)
        self.assertEqual(result.evidence, [])

    def test_e_capital_of_france_refused(self):
        result = self.process(OUT_OF_SCOPE, "claim-e")
        self.assertEqual(result.safety_classification, "REFUSE")
        self.assertFalse(result.generation_called)
        self.assertEqual(result.evidence, [])


class TestClinicalRelevanceFilter(unittest.TestCase):
    def test_topic_relevance_is_not_claim_support(self):
        item = evidence(
            "nice-step2",
            "Step 2: if blood pressure is not controlled, combine an ACE "
            "inhibitor or ARB with a calcium-channel blocker.",
        )
        result = validate_claim_evidence(STEP2_PERCENTAGE, [item], ["treatment"])
        assessment = result.assessments[0]
        self.assertTrue(assessment.relevant_topic)
        self.assertFalse(assessment.directly_supports_claim)
        self.assertEqual(result.supporting_evidence, [])

    def test_explicit_percentage_outcome_is_direct_support(self):
        item = evidence(
            "nice-step2-statistic",
            "At Step 2, 68% of patients with hypertension achieved blood "
            "pressure control after treatment.",
        )
        result = validate_claim_evidence(STEP2_PERCENTAGE, [item], ["treatment"])
        self.assertTrue(result.supported)
        self.assertTrue(result.assessments[0].directly_supports_claim)

    def test_c_fda_narcolepsy_only_relevant_to_narcolepsy(self):
        item = evidence(
            "fda-narcolepsy",
            "FDA approves modafinil for treatment of excessive sleepiness "
            "associated with narcolepsy.",
            organization="FDA",
            source_id="fda_drugs",
            document="FDA Narcolepsy",
        )
        hypertension = validate_claim_evidence(STEP2_TREATMENT, [item], ["treatment"])
        self.assertFalse(hypertension.assessments[0].relevant_topic)
        self.assertFalse(hypertension.assessments[0].directly_supports_claim)
        self.assertEqual(hypertension.supporting_evidence, [])
        narcolepsy = validate_claim_evidence(
            "What is the FDA treatment for narcolepsy?",
            [item],
            ["treatment"],
        )
        self.assertTrue(narcolepsy.assessments[0].relevant_topic)
        self.assertTrue(narcolepsy.assessments[0].relevant_source)
        self.assertTrue(narcolepsy.assessments[0].directly_supports_claim)
        self.assertEqual(
            [support.chunk.chunk_id for support in narcolepsy.supporting_evidence],
            ["fda-narcolepsy"],
        )

    def test_protected_fact_types_reject_treatment_only_evidence(self):
        treatment_only = (
            "Step 2: if blood pressure is not controlled, combine an ACE "
            "inhibitor or ARB with a calcium-channel blocker."
        )
        queries = [
            "What exact mortality rate follows Step 2 treatment for hypertension?",
            "What exact percentage risk reduction does Step 2 treatment provide for hypertension?",
            "What odds ratio is reported for Step 2 hypertension treatment?",
            "What relative risk is reported for Step 2 hypertension treatment?",
            "What exact blood pressure threshold is reported by NICE?",
            "What exact dose in mg is used at Step 2 for hypertension?",
            "How many patients achieve blood pressure control after Step 2?",
            "What is the treatment effectiveness of Step 2 for hypertension?",
        ]
        for index, query in enumerate(queries):
            with self.subTest(query=query):
                item = evidence(f"treatment-only-{index}", treatment_only)
                result = validate_claim_evidence(query, [item], ["treatment"])
                self.assertFalse(result.assessments[0].directly_supports_claim)
                self.assertEqual(result.supporting_evidence, [])

    def test_filter_removes_higher_scored_irrelevant_chunk(self):
        fda = evidence(
            "fda-narcolepsy",
            "FDA approves modafinil for treatment of narcolepsy.",
            organization="FDA",
            source_id="fda_drugs",
            document="FDA Narcolepsy",
        )
        fda.rerank_score = 0.99
        nice = evidence(
            "nice-step2",
            "Step 2: if blood pressure is not controlled, combine an ACE "
            "inhibitor or ARB with a calcium-channel blocker.",
        )
        nice.rerank_score = 0.59
        result = validate_claim_evidence(STEP2_TREATMENT, [fda, nice], ["treatment"])
        self.assertEqual(
            [support.chunk.chunk_id for support in result.supporting_evidence],
            ["nice-step2"],
        )
        self.assertFalse(result.assessments[0].directly_supports_claim)
        self.assertTrue(result.assessments[1].directly_supports_claim)


if __name__ == "__main__":
    unittest.main()
