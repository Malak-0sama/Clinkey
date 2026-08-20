"""Core unit tests for ClinKey (run: python -m unittest discover tests)."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CLINKEY_ALLOW_HASH", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.language.detector import LanguageDetector
from src.language.translator import TranslationService
from src.safety.classifier import SafetyClassifier
from src.sources.catalog import SourceCatalog
from src.sources.validator import SourceValidator
from src.sources.models import SourceSelection
from src.ingestion.chunker import SectionAwareChunker
from src.ingestion.cleaner import TextCleaner
from src.retrieval.bm25 import BM25


class TestLanguageDetector(unittest.TestCase):
    def setUp(self):
        self.d = LanguageDetector()

    def test_english(self):
        r = self.d.detect("What are the symptoms of hypertension?")
        self.assertEqual(r.language_code, "en")

    def test_arabic(self):
        r = self.d.detect("ما هي أعراض ارتفاع ضغط الدم؟")
        self.assertEqual(r.language_code, "ar")

    def test_french(self):
        r = self.d.detect("Quels sont les symptômes de l'hypertension ?")
        self.assertEqual(r.language_code, "fr")


class TestTranslator(unittest.TestCase):
    def setUp(self):
        self.t = TranslationService()

    def test_identity_english(self):
        r = self.t.translate_to_english("hello", "en")
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "hello")


class TestSafety(unittest.TestCase):
    def setUp(self):
        self.s = SafetyClassifier()

    def test_emergency(self):
        r = self.s.classify("I am having chest pain right now")
        self.assertEqual(r.classification.value, "EMERGENCY")

    def test_refuse_offtopic(self):
        r = self.s.classify("What is the weather in Cairo?")
        self.assertEqual(r.classification.value, "REFUSE")

    def test_allowed(self):
        r = self.s.classify("What are the symptoms of hypertension?")
        self.assertEqual(r.classification.value, "ALLOWED")


class TestCatalogAndValidator(unittest.TestCase):
    def setUp(self):
        self.cat = SourceCatalog()
        self.val = SourceValidator()

    def test_catalog_loaded(self):
        self.assertGreater(len(self.cat.allowed()), 0)

    def test_validate_known(self):
        sid = self.cat.allowed()[0].source_id
        res = self.val.validate([SourceSelection(source_id=sid, relevance="high", reason="t")])
        self.assertEqual(len(res.valid), 1)
        self.assertEqual(len(res.rejected), 0)

    def test_reject_unknown(self):
        res = self.val.validate([SourceSelection(source_id="not_real", relevance="high", reason="t")])
        self.assertEqual(len(res.valid), 0)
        self.assertEqual(len(res.rejected), 1)


class TestChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = SectionAwareChunker()

    def test_chunks_nonempty(self):
        from src.sources.catalog import SourceCatalog

        src = SourceCatalog().allowed()[0]
        text = (
            "Overview\nHypertension is high blood pressure. It is common but serious and "
            "can cause heart disease and stroke if it is not treated properly over time. "
            "Many people do not know they have the condition because it often has no symptoms.\n\n"
            "Symptoms\nMost people have no symptoms at all. Some people may have headaches, "
            "blurred vision, chest pain, or other symptoms when blood pressure is very high. "
            "The best way to know is to measure blood pressure regularly with a validated device."
        )
        chunks = self.chunker.chunk(src, text)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertTrue(c.text.strip())
            self.assertTrue(c.chunk_id)
            self.assertEqual(c.source_id, src.source_id)


class TestCleaner(unittest.TestCase):
    def test_removes_page_numbers(self):
        c = TextCleaner()
        out = c.clean("hello world\n12\nPage 1 of 5\n\nhello world")
        self.assertNotIn("Page 1 of 5", out)


class TestBM25(unittest.TestCase):
    def test_relevant_doc_scores_higher(self):
        corpus = [
            "hypertension is high blood pressure and can cause heart disease",
            "the cat sat on the mat and looked at the bird",
        ]
        bm = BM25(corpus)
        scores = bm.scores("high blood pressure hypertension")
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()


class TestQuestionUnderstanding(unittest.TestCase):
    """Regression tests for the question-understanding fixes."""

    def test_short_english_not_misdetected(self):
        from src.language.detector import LanguageDetector
        d = LanguageDetector()
        for q in ["Is hypertension dangerous?", "What causes diabetes?",
                  "How is it diagnosed?", "And what about treatment?",
                  "How can I prevent it?", "What are the complications?"]:
            self.assertEqual(d.detect(q).language_code, "en", q)

    def test_non_english_still_detected(self):
        from src.language.detector import LanguageDetector
        d = LanguageDetector()
        self.assertEqual(d.detect("ما هي أعراض ارتفاع ضغط الدم؟").language_code, "ar")
        self.assertEqual(d.detect("Quels sont les symptômes de l'hypertension ?").language_code, "fr")

    def test_intent_detection(self):
        from src.query.analyzer import QueryAnalyzer
        a = QueryAnalyzer()
        self.assertIn("causes", a.analyze("What causes diabetes?").intents)
        self.assertIn("complications", a.analyze("What are the complications?").intents)
        self.assertIn("prevention", a.analyze("How can I prevent it?").intents)
        self.assertIn("diagnosis", a.analyze("How is it diagnosed?").intents)

    def test_followup_gets_context(self):
        from src.orchestration.pipeline import ClinKeyPipeline
        p = ClinKeyPipeline()
        p.process("What are the symptoms of hypertension?", session_id="ctx-test")
        r = p.process("And what about treatment?", session_id="ctx-test",
                      context_query="What are the symptoms of hypertension?")
        self.assertIn("hypertension", r.topics)
        self.assertTrue(r.localized_answer or r.english_answer)

    def test_safety_allows_followups(self):
        from src.safety.classifier import SafetyClassifier
        s = SafetyClassifier()
        self.assertEqual(s.classify("What are the complications?").classification.value, "ALLOWED")
        self.assertEqual(s.classify("How can I prevent it?").classification.value, "ALLOWED")
        self.assertEqual(s.classify("What is the weather?").classification.value, "REFUSE")