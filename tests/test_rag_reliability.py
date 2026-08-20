"""Regression tests for RAG reliability (quality, support, citations)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CLINKEY_ALLOW_HASH", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.chunker import Chunk
from src.ingestion.quality import ChunkQualityFilter
from src.retrieval.models import Evidence
from src.retrieval.sufficiency import assess_sufficiency
from src.retrieval.support import topic_compatible
from src.validation.claim_support import ClaimCitationValidator
from src.validation.contradiction import detect_conflicts


def _chunk(text: str, cid="c", **kw) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_id=kw.get("source_id", "who_hypertension"),
        document_name="d",
        organization=kw.get("org", "WHO"),
        page_number=1,
        section_title="s",
        source_url="https://www.who.int/x",
        publication_date="2023-01-01",
        version="1.0",
        text=text,
    )


class TestQuality(unittest.TestCase):
    def test_nav_rejected(self):
        f = ChunkQualityFilter()
        r = f.evaluate(_chunk("Home\nMenu\nRead\nBack to top\nWorld Cardio Agenda"))
        self.assertFalse(r.accepted)

    def test_clinical_kept(self):
        f = ChunkQualityFilter()
        r = f.evaluate(_chunk(
            "ACE inhibitors are recommended as step 1 treatment for hypertension in adults under 55."
        ))
        self.assertTrue(r.accepted)


class TestSupport(unittest.TestCase):
    def test_femoral_vs_malaria(self):
        ev = Evidence(chunk=_chunk("Antimalarial treatment should be started promptly for falciparum malaria."))
        self.assertFalse(topic_compatible("What is the recommended management of a femoral fracture?", ev))

    def test_htn_vs_fracture(self):
        ev = Evidence(chunk=_chunk("Offer an ACE inhibitor or ARB to adults under 55 with hypertension."))
        self.assertFalse(topic_compatible("What is the treatment of femoral fracture?", ev))

    def test_step2_compatible(self):
        ev = Evidence(chunk=_chunk("Step 2: combine an ACE inhibitor or ARB with a calcium-channel blocker."))
        self.assertTrue(topic_compatible("What is step 2 hypertension treatment?", ev))


class TestSufficiencyDose(unittest.TestCase):
    def test_lisinopril_no_dose(self):
        ev = Evidence(chunk=_chunk("ACE inhibitors are recommended."))
        out = assess_sufficiency("What is the exact dose of lisinopril?", [ev], ["dosage"])
        self.assertFalse(out["sufficient"])


class TestCitationMismatch(unittest.TestCase):
    def test_wrong_source_cite(self):
        nice = Evidence(chunk=_chunk("Step 2 combines ACEi/ARB with CCB.", "nice", source_id="nice_hypertension_ng136", org="NICE"))
        who = Evidence(chunk=_chunk("An estimated 1.28 billion adults have hypertension.", "who", source_id="who_hypertension", org="WHO"))
        ans = "Step 2 combines ACEi/ARB with CCB. [2]"
        r = ClaimCitationValidator().validate(ans, [nice, who])
        self.assertFalse(r["ok"])


class TestContradiction(unittest.TestCase):
    def test_conflict(self):
        a = Evidence(chunk=_chunk("Recommend ACE inhibitors as first line.", "a"))
        b = Evidence(chunk=_chunk("Do not recommend ACE inhibitors in this group.", "b"))
        self.assertTrue(detect_conflicts([a, b]))


if __name__ == "__main__":
    unittest.main()