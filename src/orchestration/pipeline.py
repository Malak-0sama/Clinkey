"""Orchestration layer — the central controller.

Implements the full ClinKey flow:

    user query -> language detect -> translate to English -> safety
    -> query analysis -> source routing -> validation -> acquisition/ingest
    -> hybrid retrieval -> rerank -> confidence -> grounded generation
    -> claim validation -> citations -> response localization -> result

Translation happens at exactly TWO controlled points (input and output);
the RAG core is language-independent and operates on canonical English.
"""
from __future__ import annotations

import time

from ..citations.builder import CitationBuilder
from ..config.settings import get_settings
from ..embeddings.embedder import Embedder
from ..generation.generator import GroundedGenerator
from ..language.translator import TranslationService
from ..query.analyzer import QueryAnalyzer
from ..retrieval.confidence import ConfidenceEstimator
from ..retrieval.errors import UnauthorizedRetrievalError
from ..retrieval.hybrid import HybridRetriever
from ..safety.classifier import SafetyClassifier
from ..safety.models import SafetyClass
from ..sources.router import SourceRouter
from ..sources.validator import SourceValidator
from ..generation.prompts import ABSTENTION_TEXT
from ..vectorstore import get_vectorstore
from .knowledge import KnowledgeBase
from .models import PipelineResult
from .rag_log import log_decision

_INSUFFICIENT_EVIDENCE = ABSTENTION_TEXT

_TRANSLATION_ERROR = (
    "I couldn't reliably translate your question, so I'm not able to answer it "
    "accurately. Please try rephrasing it or asking again in the same language."
)


class ClinKeyPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.translation = TranslationService()
        self.safety = SafetyClassifier()
        self.analyzer = QueryAnalyzer()
        self.router = SourceRouter()
        self.validator = SourceValidator()
        self.knowledge = KnowledgeBase()
        self.embedder = Embedder()
        self.retriever = HybridRetriever(self.knowledge.store, self.embedder)
        self.confidence = ConfidenceEstimator()
        self.generator = GroundedGenerator()
        self.citation_builder = CitationBuilder()
        self._result: PipelineResult | None = None
        self._session_topics = {}  # session_id -> last known topic string

    def providers_info(self) -> dict:
        return {
            "llm": self.settings.llm_effective_provider,
            "embeddings": self.embedder.provider,
            "vectorstore": self.settings.vectorstore_effective_provider,
            "chunks": self.knowledge.chunk_count,
        }

    def process(
        self,
        original_query: str,
        session_id: str = "",
        context_query: str = "",
        *,
        user_id: str = "",
        tenant_id: str = "",
    ) -> PipelineResult:
        """Run the RAG pipeline.

        ``user_id`` / ``tenant_id`` MUST come from the authenticated backend
        session (AuthService), never from an untrusted frontend field.
        """
        start = time.time()
        result = PipelineResult(original_query=original_query, session_id=session_id, user_id=user_id)
        self._result = result
        try:
            if not (original_query or "").strip():
                result.abstained = True
                result.localized_answer = _INSUFFICIENT_EVIDENCE
                result.confidence = "INSUFFICIENT_EVIDENCE"
            else:
                self._run(original_query, result, context_query, session_id, user_id, tenant_id)
        except Exception as exc:  # noqa: BLE001 — never leak stack traces to UI
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            result.abstained = True
            if not result.localized_answer:
                result.localized_answer = _INSUFFICIENT_EVIDENCE
        confidence_state = str(result.confidence or "").strip().upper().replace(" ", "_")
        safety_state = str(result.safety_classification or "").strip().upper()
        if (
            confidence_state in {"INSUFFICIENT", "INSUFFICIENT_EVIDENCE"}
            or safety_state == SafetyClass.EMERGENCY.value
        ):
            result.evidence = []
            result.citations = []
        result.elapsed_ms = int((time.time() - start) * 1000)
        result.providers = self.providers_info()
        log_decision({
            "request_id": result.request_id,
            "user_id": user_id,
            "retrieval_count": len(result.evidence),
            "top_score": (result.confidence_signals or {}).get("top"),
            "confidence": result.confidence,
            "threshold": self.settings.get("confidence_thresholds_low"),
            "selected_chunk_ids": [e.get("chunk_id") for e in (result.evidence or [])[:8]],
            "abstained": result.abstained,
            "generation_called": result.generation_called,
            "elapsed_ms": result.elapsed_ms,
            "error": result.error,
        })
        return result

    # ---- internals ----
    def _run(
        self,
        original_query: str,
        r: PipelineResult,
        context_query: str = "",
        session_id: str = "",
        user_id: str = "",
        tenant_id: str = "",
    ) -> None:
        # 1) language detection
        det = self.translation.detect_language(original_query)
        r.detected_language = det.language_code
        r.language_name = det.language_name
        r.language_confidence = det.confidence

        # 2) translate to English (input)
        tr = self.translation.translate_to_english(original_query, det.language_code)
        if not tr.ok:
            r.error = "input translation failed"
            r.localized_answer = self._localize(_TRANSLATION_ERROR, det.language_code)
            r.target_language = det.language_code
            return
        r.input_translation_ok = True
        r.input_translation_provider = tr.provider
        r.canonical_english_query = tr.text

        # 3) safety classification on canonical English
        safety = self.safety.classify(r.canonical_english_query)
        r.safety_classification = safety.classification.value
        r.safety_reason = safety.reason

        if safety.requires_refusal:
            r.localized_answer = self._localize(safety.english_safe_response, det.language_code)
            r.target_language = det.language_code
            return

        # 4) query analysis (with conversation context for follow-ups)
        analysis = self.analyzer.analyze(r.canonical_english_query)

        # Resolve context: explicit context_query first, then session memory.
        context_analysis = None
        ctx_src = context_query or self._session_topics.get(session_id, "")
        if ctx_src and ctx_src.strip():
            ctx = self.analyzer.analyze(ctx_src)
            if ctx.topics or ctx.keywords:
                context_analysis = ctx
        # If the explicit context had no topic (e.g. an intermediate
        # follow-up), fall back to the remembered session topic.
        if context_analysis is None and session_id:
            mem = self._session_topics.get(session_id, "")
            if mem and mem != ctx_src:
                mctx = self.analyzer.analyze(mem)
                if mctx.topics or mctx.keywords:
                    context_analysis = mctx

        # Follow-up resolution: carry the previous topic forward when the
        # current query is terse and has no topic of its own.
        if not analysis.topics and not analysis.keywords and context_analysis is not None:
            analysis.topics = list(context_analysis.topics)
            analysis.keywords = list(context_analysis.keywords)
        elif context_analysis is not None:
            for t in context_analysis.topics:
                if t not in analysis.topics:
                    analysis.topics.append(t)

        # Remember this turn's topic for the next follow-up.
        if analysis.topics and session_id:
            self._session_topics[session_id] = " ".join(analysis.topics)

        r.topics = analysis.topics
        r.intents = analysis.intents

        clarification = self.analyzer.clarification_request(
            r.canonical_english_query, analysis
        )
        if clarification:
            r.abstained = True
            r.generation_called = False
            r.confidence = "CLARIFICATION_REQUIRED"
            r.confidence_signals = {"reason": "missing_clinical_context"}
            r.localized_answer = self._localize(clarification, det.language_code)
            r.target_language = det.language_code
            return

        retrieval_query = self.analyzer.normalize_for_retrieval(
            r.canonical_english_query, analysis
        )
        retrieval_variants = self.analyzer.retrieval_variants(
            r.canonical_english_query, analysis
        )

        selections = self.router.route(
            retrieval_query, self.settings.sources_max_selected
        )
        r.selected_source_ids = [s.source_id for s in selections]

        validation = self.validator.validate(selections)
        r.validated_sources = [s.source_id for s in validation.valid]
        r.acquisition = self.knowledge.ensure_sources(validation.valid)

        trusted_only = bool(self.settings.get("grounding_trusted_only", True))
        try:
            retrieval = self.retriever.retrieve(
                retrieval_query,
                owner_id=user_id,
                tenant_id=tenant_id,
                trusted_only=trusted_only,
                allowed_source_ids=r.validated_sources or None,
                support_query=r.canonical_english_query,
            )
            retrieval = self._assess_retrieval(
                retrieval, r.canonical_english_query, r.intents
            )
            retrieval.signals["retrieval_attempts"] = 1

            if not retrieval.sufficient or retrieval.confidence == "INSUFFICIENT_EVIDENCE":
                retry_queries = [
                    query for query in retrieval_variants
                    if query.lower() != retrieval_query.lower()
                ]
                if retry_queries:
                    retry_selections = self.router.route(
                        " ".join(retry_queries), self.settings.sources_max_selected
                    )
                    retry_validation = self.validator.validate(retry_selections)
                    known_sources = {source.source_id for source in validation.valid}
                    new_sources = [
                        source for source in retry_validation.valid
                        if source.source_id not in known_sources
                    ]
                    if new_sources:
                        r.acquisition.extend(self.knowledge.ensure_sources(new_sources))
                    for selection in retry_selections:
                        if selection.source_id not in r.selected_source_ids:
                            r.selected_source_ids.append(selection.source_id)
                    for source in retry_validation.valid:
                        if source.source_id not in r.validated_sources:
                            r.validated_sources.append(source.source_id)
                    retry = self.retriever.retrieve_many(
                        [retrieval_query, *retry_queries],
                        support_query=r.canonical_english_query,
                        owner_id=user_id,
                        tenant_id=tenant_id,
                        trusted_only=trusted_only,
                        allowed_source_ids=r.validated_sources or None,
                    )
                    retry = self._assess_retrieval(
                        retry, r.canonical_english_query, r.intents
                    )
                    retry.signals["retrieval_attempts"] = 2
                    retry.signals["retrieval_variants"] = retry_queries
                    retrieval = retry
        except UnauthorizedRetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001
            r.error = f"retrieval_failed: {exc}"[:200]
            r.abstained = True
            r.localized_answer = self._localize(_INSUFFICIENT_EVIDENCE, det.language_code)
            r.target_language = det.language_code
            return

        r.confidence = retrieval.confidence
        r.confidence_score = retrieval.confidence_score
        r.confidence_signals = retrieval.signals
        r.evidence = [e.to_dict() for e in retrieval.evidence]

        # 9) HARD EVIDENCE GATE — never call generation without sufficient evidence
        abstain_on = bool(self.settings.get("grounding_abstention_enabled", True))
        if retrieval.confidence == "CONFLICT":
            r.abstained = True
            r.generation_called = False
            r.localized_answer = self._localize(
                "Authoritative sources disagree on this recommendation. "
                "I will not pick a side. Please consult a clinician and the "
                "primary guidelines.",
                det.language_code,
            )
            r.target_language = det.language_code
            return
        if abstain_on and (not retrieval.sufficient or retrieval.confidence == "INSUFFICIENT_EVIDENCE"):
            r.abstained = True
            r.generation_called = False
            r.localized_answer = self._localize(_INSUFFICIENT_EVIDENCE, det.language_code)
            r.target_language = det.language_code
            return

        intent = analysis.intents[0] if analysis.intents else ""
        generated = self.generator.generate(
            retrieval_query, retrieval.evidence, intent, topics=analysis.topics
        )
        r.generation_called = True
        r.english_answer = generated.text

        if not (r.english_answer or "").strip():
            r.abstained = True
            r.localized_answer = self._localize(_INSUFFICIENT_EVIDENCE, det.language_code)
            r.target_language = det.language_code
            return

        if bool(self.settings.get("grounding_validation_enabled", True)):
            claim_result = self.claim_validator().validate(r.english_answer, retrieval.evidence)
            r.claim_ok = claim_result.ok
            r.unsupported_claims = claim_result.unsupported_sentences
            if not claim_result.ok:
                r.abstained = True
                r.english_answer = _INSUFFICIENT_EVIDENCE
                r.citations = []
                r.localized_answer = self._localize(_INSUFFICIENT_EVIDENCE, det.language_code)
                r.target_language = det.language_code
                return

        from ..validation.citation_check import validate_citations
        from ..validation.contradiction import detect_conflicts
        from ..validation.medical import MedicalSafetyValidator

        med = MedicalSafetyValidator().validate(r.english_answer, retrieval.evidence)
        cites_ok = validate_citations(r.english_answer, generated.cited_evidence or retrieval.evidence)
        conflicts = detect_conflicts(retrieval.evidence)
        if conflicts:
            r.confidence_signals = {**(r.confidence_signals or {}), "conflicts": conflicts}
            r.confidence = "CONFLICT"
            r.abstained = True
            r.error = "CONFLICT"
            r.localized_answer = self._localize(
                "Authoritative sources disagree on this recommendation. "
                "I will not pick a side. Please consult a clinician and the "
                "primary guidelines.",
                det.language_code,
            )
            r.target_language = det.language_code
            return
        from ..validation.claim_support import ClaimCitationValidator
        cmap = ClaimCitationValidator().validate(
            r.english_answer, generated.cited_evidence or retrieval.evidence
        )
        if not cmap["ok"]:
            r.abstained = True
            r.error = "CITATION_FAILURE"
            r.localized_answer = self._localize(_INSUFFICIENT_EVIDENCE, det.language_code)
            r.target_language = det.language_code
            return
        if not med.ok or not cites_ok["ok"]:
            r.abstained = True
            r.error = (med.status if not med.ok else cites_ok["status"])
            r.english_answer = _INSUFFICIENT_EVIDENCE
            r.citations = []
            r.localized_answer = self._localize(_INSUFFICIENT_EVIDENCE, det.language_code)
            r.target_language = det.language_code
            return

        citations = self.citation_builder.build(generated.cited_evidence)
        r.citations = [c.to_dict() for c in citations]
        r.localized_answer = self._localize(r.english_answer, det.language_code)
        r.target_language = det.language_code

    # ---- helpers ----
    def _assess_retrieval(
        self,
        retrieval,
        query: str,
        intents: list[str],
    ):
        from ..retrieval.claim_evidence import validate_claim_evidence
        from ..retrieval.sufficiency import assess_sufficiency

        retrieval.query = query
        candidate_count = len(retrieval.evidence)
        claim_validation = validate_claim_evidence(query, retrieval.evidence, intents)
        retrieval.evidence = claim_validation.supporting_evidence
        sufficiency = assess_sufficiency(query, retrieval.evidence, intents)
        retrieval = self.confidence.estimate(retrieval)
        missing_aspects = list(sufficiency["missing_aspects"])
        if not claim_validation.supported:
            missing_aspects.append(
                f"required_fact_type:{claim_validation.requirement.fact_type}"
            )
        missing_aspects = list(dict.fromkeys(missing_aspects))
        retrieval.signals = {
            **(retrieval.signals or {}),
            "coverage_status": sufficiency["status"],
            "missing_aspects": missing_aspects,
            "claim_requirement": claim_validation.requirement.to_dict(),
            "claim_validation_supported": claim_validation.supported,
            "clinical_filter_candidates": candidate_count,
            "clinical_filter_kept": len(retrieval.evidence),
            "clinical_filter_removed": candidate_count - len(retrieval.evidence),
            "topic_relevant_candidates": sum(
                1 for item in claim_validation.assessments if item.relevant_topic
            ),
            "direct_claim_support_candidates": sum(
                1 for item in claim_validation.assessments
                if item.directly_supports_claim
            ),
        }
        if not claim_validation.supported:
            retrieval.sufficient = False
            retrieval.confidence = "INSUFFICIENT_EVIDENCE"
            retrieval.generation_allowed = False
            retrieval.signals["reason"] = "unsupported_requested_claim"
        elif not sufficiency["sufficient"]:
            retrieval.sufficient = False
            retrieval.confidence = "INSUFFICIENT_EVIDENCE"
            retrieval.generation_allowed = False
            retrieval.signals["reason"] = "incomplete_evidence_coverage"
        return retrieval

    def _localize(self, english_text: str, target_language: str) -> str:
        if target_language == "en":
            return english_text
        tr = self.translation.translate_from_english(english_text, target_language)
        if tr.ok:
            # lightweight localization validation
            from ..validation.localization import LocalizationValidator

            validator = LocalizationValidator()
            check = validator.validate(english_text, tr.text, target_language)
            if self._result is not None:
                self._result.localization_ok = check.ok
                self._result.localization_issues = check.issues
            if check.ok:
                return tr.text
            # retry once (gemini already tried; attempt google bridge)
            return tr.text
        # fallback: return validated English with a clear notice
        if self._result is not None:
            self._result.localization_ok = False
            self._result.localization_issues = ["could not localize — returning English"]
        return english_text

    _claim_validator = None

    def claim_validator(self):
        if self._claim_validator is None:
            from ..validation.claims import ClaimValidator

            self._claim_validator = ClaimValidator()
        return self._claim_validator


_pipeline: ClinKeyPipeline | None = None


def get_pipeline() -> ClinKeyPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ClinKeyPipeline()
    return _pipeline