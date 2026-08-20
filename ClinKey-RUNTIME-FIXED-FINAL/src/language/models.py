"""Language-layer data models."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LanguageDetectionResult:
    language_code: str
    language_name: str
    confidence: float
    is_supported: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "language_code": self.language_code,
            "language_name": self.language_name,
            "confidence": round(self.confidence, 3),
            "is_supported": self.is_supported,
            "note": self.note,
        }


@dataclass
class TranslationResult:
    ok: bool
    text: str
    source_language: str = ""
    target_language: str = ""
    provider: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "text": self.text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "provider": self.provider,
            "error": self.error,
        }


@dataclass
class CanonicalQuery:
    """The English canonical query used by the RAG core."""
    original_text: str
    english_text: str
    detected_language: str
    detection_confidence: float


@dataclass
class LocalizedAnswer:
    """The validated English answer localized back to the user's language."""
    english_text: str
    localized_text: str
    target_language: str
    citations: list = field(default_factory=list)
    localization_ok: bool = True