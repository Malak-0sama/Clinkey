"""Conversation history models."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Message:
    role: str                       # "user" | "assistant"
    message_id: str = field(default_factory=_uid)
    original_text: str = ""
    detected_language: str = ""
    canonical_english_text: str = ""
    english_answer: str = ""
    localized_answer: str = ""
    target_language: str = ""
    citations: list = field(default_factory=list)
    confidence: str = ""
    safety_classification: str = ""
    topics: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "original_text": self.original_text,
            "detected_language": self.detected_language,
            "canonical_english_text": self.canonical_english_text,
            "english_answer": self.english_answer,
            "localized_answer": self.localized_answer,
            "target_language": self.target_language,
            "citations": self.citations,
            "confidence": self.confidence,
            "safety_classification": self.safety_classification,
            "topics": self.topics,
            "intents": self.intents,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        msg = cls(
            role=d.get("role", ""),
            message_id=d.get("message_id", _uid()),
            original_text=d.get("original_text", ""),
            detected_language=d.get("detected_language", ""),
            canonical_english_text=d.get("canonical_english_text", ""),
            english_answer=d.get("english_answer", ""),
            localized_answer=d.get("localized_answer", ""),
            target_language=d.get("target_language", ""),
            citations=d.get("citations") or [],
            confidence=d.get("confidence", ""),
            safety_classification=d.get("safety_classification", ""),
            topics=d.get("topics") or [],
            intents=d.get("intents") or [],
            timestamp=d.get("timestamp", _now()),
        )
        return msg


@dataclass
class Session:
    session_id: str = field(default_factory=_uid)
    title: str = "New chat"
    created_at: str = field(default_factory=_now)
    messages: list = field(default_factory=list)
    user_id: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls(session_id=d.get("session_id", _uid()), title=d.get("title", "New chat"),
                 created_at=d.get("created_at", _now()), user_id=d.get("user_id", ""))
        s.messages = [Message.from_dict(m) for m in d.get("messages", [])]
        return s