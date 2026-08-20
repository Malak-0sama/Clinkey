"""Orchestration result models."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    original_query: str = ""
    detected_language: str = ""
    language_name: str = ""
    language_confidence: float = 0.0
    canonical_english_query: str = ""
    input_translation_ok: bool = False
    input_translation_provider: str = ""
    safety_classification: str = ""
    safety_reason: str = ""
    topics: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    selected_source_ids: list = field(default_factory=list)
    validated_sources: list = field(default_factory=list)
    acquisition: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    confidence: str = ""
    confidence_score: float = 0.0
    english_answer: str = ""
    claim_ok: bool = True
    unsupported_claims: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    localized_answer: str = ""
    target_language: str = ""
    localization_ok: bool = True
    localization_issues: list = field(default_factory=list)
    providers: dict = field(default_factory=dict)
    error: str = ""
    elapsed_ms: int = 0
    abstained: bool = False
    generation_called: bool = False
    user_id: str = ""
    confidence_signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "original_query": self.original_query,
            "detected_language": self.detected_language,
            "language_name": self.language_name,
            "language_confidence": round(self.language_confidence, 3),
            "canonical_english_query": self.canonical_english_query,
            "input_translation_ok": self.input_translation_ok,
            "input_translation_provider": self.input_translation_provider,
            "safety_classification": self.safety_classification,
            "safety_reason": self.safety_reason,
            "topics": self.topics,
            "intents": self.intents,
            "selected_source_ids": self.selected_source_ids,
            "validated_sources": self.validated_sources,
            "acquisition": self.acquisition,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
            "english_answer": self.english_answer,
            "claim_ok": self.claim_ok,
            "unsupported_claims": self.unsupported_claims,
            "citations": self.citations,
            "localized_answer": self.localized_answer,
            "target_language": self.target_language,
            "localization_ok": self.localization_ok,
            "localization_issues": self.localization_issues,
            "providers": self.providers,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "abstained": self.abstained,
            "generation_called": self.generation_called,
            "user_id": self.user_id,
            "confidence_signals": self.confidence_signals,
        }