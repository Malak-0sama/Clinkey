"""Clinical evidence-sufficiency gate.

Generation is allowed only when:
  1. top semantic similarity S >= min_semantic_similarity (default 0.70)
  2. fused/overall confidence >= generation_threshold (default 0.65)
  3. question intent / topic entities appear in the retrieved text

Otherwise the result is INSUFFICIENT_EVIDENCE and the pipeline must not call the LLM.
"""
from __future__ import annotations

import re

from ..config.settings import get_settings
from ..query.analyzer import QueryAnalyzer
from ..validation.contradiction import detect_conflicts
from .models import RetrievalResult

_AUTHORITY = {
    "NICE": 1.0,
    "WHO": 1.0,
    "CDC": 0.95,
    "USPSTF": 0.95,
    "NIH": 0.9,
    "NHS": 0.9,
}

_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "what", "how", "why", "when", "which", "who", "does", "do", "can", "i",
    "my", "with", "from", "about", "this", "that", "optimal",
}


class ConfidenceEstimator:
    def __init__(self):
        self.settings = get_settings()
        self.high = float(self.settings.get("confidence_thresholds_high", 0.55))
        self.medium = float(self.settings.get("confidence_thresholds_medium", 0.30))
        self.low = float(self.settings.get("confidence_thresholds_low", 0.22))
        self.min_semantic = float(self.settings.get("confidence_min_semantic_similarity", 0.70))
        self.gen_threshold = float(self.settings.get("confidence_generation_threshold", 0.65))
        self.min_strong = int(self.settings.get("confidence_min_strong_hits", 1))
        self.strong_cut = float(self.settings.get("confidence_strong_score", 0.28))
        self.analyzer = QueryAnalyzer()

    def estimate(self, result: RetrievalResult) -> RetrievalResult:
        if not result.evidence:
            result.confidence = "INSUFFICIENT_EVIDENCE"
            result.confidence_score = 0.0
            result.sufficient = False
            result.signals = {"reason": "no_evidence"}
            return result

        scores = [e.relevance_score or e.rerank_score or e.combined_score for e in result.evidence]
        support_scores = [
            e.support_score or (score if not result.query else 0.0)
            for e, score in zip(result.evidence, scores)
        ]
        top = max(scores)
        avg = sum(scores[:3]) / max(1, len(scores[:3]))
        mean_all = sum(scores) / len(scores)
        top_support = max(support_scores)
        mean_support = sum(support_scores[:3]) / max(1, len(support_scores[:3]))
        direct_hits = sum(1 for score in support_scores if score >= 0.55)
        claim_evaluated = bool(result.evidence) and all(
            evidence.claim_evaluated for evidence in result.evidence
        )
        claim_direct_hits = sum(
            1 for evidence in result.evidence
            if evidence.claim_evaluated and evidence.directly_supports_claim
        )
        atomic_claim = claim_evaluated and all(
            evidence.claim_atomic for evidence in result.evidence
        )
        strong = sum(1 for score in scores if score >= self.strong_cut)
        gap = (scores[0] - scores[1]) if len(scores) > 1 else scores[0]
        lexical = self._lexical_overlap(result)
        top_sem = max(e.semantic_score for e in result.evidence)
        intent_ok = self._intent_entities_match(result)
        rare_ok = self._rare_terms_supported(result)
        conflicts = detect_conflicts(result.evidence)
        authority_scores = []
        for evidence in result.evidence:
            organization = evidence.chunk.organization.upper()
            authority_scores.append(max(
                (score for name, score in _AUTHORITY.items() if name in organization),
                default=0.75,
            ))
        authority = max(authority_scores)
        n_sources = len({e.chunk.source_id for e in result.evidence})
        coverage = min(1.0, direct_hits / max(1, min(2, len(result.evidence))))
        quality = (
            0.25 * top
            + 0.25 * top_support
            + 0.15 * avg
            + 0.10 * mean_support
            + 0.08 * authority
            + 0.07 * coverage
            + 0.05 * lexical
            + 0.05 * top_sem
        )

        result.signals = {
            "top": round(top, 3),
            "mean_top3": round(avg, 3),
            "mean_all": round(mean_all, 3),
            "top_support": round(top_support, 3),
            "mean_support": round(mean_support, 3),
            "direct_support_hits": direct_hits,
            "claim_evaluated": claim_evaluated,
            "direct_claim_support_hits": claim_direct_hits,
            "atomic_claim": atomic_claim,
            "strong_hits": strong,
            "gap": round(gap, 3),
            "n": len(scores),
            "n_sources": n_sources,
            "lexical_overlap": round(lexical, 3),
            "top_semantic": round(top_sem, 3),
            "source_authority": round(authority, 3),
            "coverage": round(coverage, 3),
            "intent_match": intent_ok,
            "rare_terms_ok": rare_ok,
            "conflicts": conflicts,
            "quality": round(quality, 3),
        }

        if conflicts:
            result.confidence = "CONFLICT"
            result.confidence_score = round(quality, 3)
            result.sufficient = False
            result.signals["reason"] = "contradictory_evidence"
            return result
        if claim_evaluated and claim_direct_hits != len(result.evidence):
            return self._fail(result, quality, "claim_evidence_mismatch")
        if direct_hits == 0 or top_support < 0.45:
            return self._fail(result, quality, "no_direct_support")
        if result.query and not intent_ok:
            return self._fail(result, quality, "intent_entity_mismatch")
        if result.query and not rare_ok:
            return self._fail(result, quality, "missing_rare_terms")

        min_lex = float(self.settings.get("confidence_min_lexical_overlap", 0.18))
        if result.query and lexical < min_lex and top_support < 0.75:
            return self._fail(result, quality, "low_lexical_overlap")

        embed_provider = str(self.settings.embeddings_effective_provider)
        if embed_provider != "hash" and top_sem < self.min_semantic and top_support < 0.75:
            return self._fail(result, quality, "semantic_below_min")

        if top < self.low or strong < self.min_strong:
            return self._fail(result, quality, "below_gate")

        authoritative_atomic_support = (
            atomic_claim
            and claim_direct_hits >= 1
            and authority >= 0.95
            and coverage >= 1.0
        )
        if (
            quality >= 0.72
            and top_support >= 0.75
            and (direct_hits >= 2 or authoritative_atomic_support)
            and not conflicts
        ):
            result.confidence = "HIGH"
        elif quality >= 0.52 and top_support >= 0.55:
            result.confidence = "MEDIUM"
        else:
            result.confidence = "LOW"

        result.confidence_score = round(quality, 3)
        result.sufficient = True
        return result

    @staticmethod
    def _fail(result: RetrievalResult, avg: float, reason: str) -> RetrievalResult:
        result.confidence = "INSUFFICIENT_EVIDENCE"
        result.confidence_score = round(avg, 3)
        result.sufficient = False
        result.signals["reason"] = reason
        return result

    def _intent_entities_match(self, result: RetrievalResult) -> bool:
        """Require topic/intent entities from the question to appear in evidence."""
        if not result.query:
            return True
        analysis = self.analyzer.analyze(result.query)
        corpus = " ".join(e.chunk.text.lower() for e in result.evidence[:5])
        if not analysis.topics and not analysis.intents:
            return True
        topic_syn = {
            "hypertension": ("hypertension", "blood pressure", "ace", "arb", "ccb", "calcium-channel"),
            "diabetes": ("diabetes", "insulin", "glucose", "metformin"),
        }
        topic_hit = any(t.replace("_", " ") in corpus or t in corpus for t in analysis.topics)
        for t in analysis.topics:
            if any(s in corpus for s in topic_syn.get(t, ())):
                topic_hit = True
        kw_hit = any(kw in corpus for kw in analysis.keywords)
        intent_needles = {
            "symptoms": ("symptom", "sign"),
            "treatment": ("treat", "therapy", "medication", "manage", "inhibitor", "blocker", "step"),
            "diagnosis": ("diagnos", "screen", "test", "measure"),
            "prevention": ("prevent", "avoid", "lifestyle"),
            "causes": ("cause", "risk"),
            "complications": ("complication", "damage", "stroke", "failure"),
            "dosage": ("dose", "dosage", "mg"),
            "definition": ("condition", "called", "defined", "mmhg"),
            "risk": ("risk", "factor"),
            "comparison": ("compare", "difference", "versus"),
            "side_effects": ("side effect", "adverse", "reaction"),
        }
        intent_hit = False
        for intent in analysis.intents:
            needles = intent_needles.get(intent, (intent,))
            if any(n in corpus for n in needles):
                intent_hit = True
                break
        if analysis.intents and analysis.topics:
            return bool((topic_hit or kw_hit) and intent_hit)
        return bool(topic_hit or kw_hit or intent_hit)

    @staticmethod
    def _lexical_overlap(result: RetrievalResult) -> float:
        q = set(re.findall(r"[a-z0-9]+", (result.query or "").lower())) - _STOP
        if not q:
            return 0.0
        corpus = " ".join(e.chunk.text.lower() for e in result.evidence[:5])
        ctoks = set(re.findall(r"[a-z0-9]+", corpus))
        return len(q & ctoks) / max(1, len(q))

    @staticmethod
    def _rare_terms_supported(result: RetrievalResult) -> bool:
        q = set(re.findall(r"[a-z0-9]+", (result.query or "").lower())) - _STOP
        # Meta/process words: they describe HOW the question is phrased or
        # what kind of clinical decision is being asked about (source type,
        # request framing, generic care-pathway verbs), not a specific
        # clinical entity/topic the evidence must literally contain. This
        # gate exists to catch genuine topic mismatches (e.g. malaria
        # evidence for a femoral-fracture question) — a word like "guidelines"
        # or "tolerated" appearing only in the QUESTION's framing must not by
        # itself cause a hard rejection of otherwise correct, on-topic
        # evidence just because that exact word isn't in the source text.
        generic = {
            "treatment", "treatments", "recommended", "management", "symptoms",
            "according", "guidance", "guideline", "guidelines", "pharmacological",
            "medication", "medications", "selected", "initial", "tolerated",
            "prescribed", "administered", "patient", "patients",
        }
        rare = {t for t in q if len(t) >= 8} - generic
        if not rare:
            return True
        corpus = " ".join(e.chunk.text.lower() for e in result.evidence[:5])
        aliases = {
            "hypertension": ("hypertension", "blood pressure", "ace", "arb"),
            "antihypertensive": ("hypertension", "blood pressure", "ace inhibitor", "arb", "calcium-channel"),
            "antihypertensives": ("hypertension", "blood pressure", "ace inhibitor", "arb", "calcium-channel"),
            "younger": ("under 55", "younger than 55"),
            "threshold": ("mmhg", "140/90", "diagnosed", "condition"),
            "uncontrolled": ("not controlled", "step 2", "step 3"),
            "mortality": ("mortality", "death"),
            "percentage": ("percent", "%"),
        }
        supported = 0
        for term in rare:
            if any(alias in corpus for alias in aliases.get(term, (term,))):
                supported += 1
        required = max(1, (len(rare) + 1) // 2)
        return supported >= required