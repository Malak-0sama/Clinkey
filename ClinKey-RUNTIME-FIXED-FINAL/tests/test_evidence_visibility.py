from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("CLINKEY_ALLOW_HASH", "1")

from src.orchestration.pipeline import ClinKeyPipeline


class TestEvidenceVisibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = ClinKeyPipeline()

    def run_state(self, confidence: str, safety: str):
        def fake_run(original_query, result, *args):
            result.confidence = confidence
            result.safety_classification = safety
            result.evidence = [{"chunk_id": "secret-chunk", "excerpt": "hidden"}]
            result.citations = [{"number": 1, "chunk_id": "secret-chunk"}]
            result.localized_answer = "response"

        with patch.object(self.pipeline, "_run", side_effect=fake_run):
            return self.pipeline.process("medical question", session_id="visibility")

    def test_insufficient_evidence_hides_chunks(self):
        result = self.run_state("INSUFFICIENT_EVIDENCE", "ALLOWED")
        self.assertEqual(result.evidence, [])

    def test_insufficient_evidence_hides_citations(self):
        result = self.run_state("INSUFFICIENT_EVIDENCE", "ALLOWED")
        self.assertEqual(result.citations, [])

    def test_insufficient_alias_hides_chunks(self):
        result = self.run_state("INSUFFICIENT", "ALLOWED")
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.citations, [])

    def test_insufficient_spacing_and_case_hides_chunks(self):
        result = self.run_state(" insufficient evidence ", "ALLOWED")
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.citations, [])

    def test_emergency_hides_chunks(self):
        result = self.run_state("HIGH", "EMERGENCY")
        self.assertEqual(result.evidence, [])

    def test_emergency_hides_citations(self):
        result = self.run_state("HIGH", "EMERGENCY")
        self.assertEqual(result.citations, [])

    def test_allowed_high_preserves_evidence(self):
        result = self.run_state("HIGH", "ALLOWED")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(len(result.citations), 1)

    def test_allowed_medium_preserves_evidence(self):
        result = self.run_state("MEDIUM", "ALLOWED")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(len(result.citations), 1)

    def test_serialized_result_hides_evidence(self):
        result = self.run_state("INSUFFICIENT_EVIDENCE", "ALLOWED")
        serialized = result.to_dict()
        self.assertEqual(serialized["evidence"], [])
        self.assertEqual(serialized["citations"], [])

    def test_log_receives_no_hidden_evidence(self):
        captured = []
        with patch("src.orchestration.pipeline.log_decision", side_effect=captured.append):
            result = self.run_state("INSUFFICIENT_EVIDENCE", "ALLOWED")
        self.assertEqual(result.evidence, [])
        self.assertEqual(captured[0]["retrieval_count"], 0)
        self.assertEqual(captured[0]["selected_chunk_ids"], [])

    def test_real_emergency_has_no_evidence(self):
        result = self.pipeline.process("I am having chest pain right now", session_id="emergency")
        self.assertEqual(result.safety_classification, "EMERGENCY")
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.citations, [])

    def test_real_insufficient_query_has_no_evidence(self):
        result = self.pipeline.process(
            "What is the optimal dosage of ranitidine for a newborn?",
            session_id="insufficient",
        )
        self.assertEqual(result.confidence, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.citations, [])


if __name__ == "__main__":
    unittest.main()
