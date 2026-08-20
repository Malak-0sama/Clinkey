"""Safety-layer data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SafetyClass(str, Enum):
    ALLOWED = "ALLOWED"                      # general guideline-based info
    CAUTION = "CAUTION"                      # patient-specific — careful wording
    REFUSE = "REFUSE"                        # unsupported / unsafe requests
    EMERGENCY = "EMERGENCY"                  # immediate danger — redirect to care


@dataclass
class SafetyResult:
    classification: SafetyClass
    reason: str = ""
    english_safe_response: str = ""
    provider: str = ""

    @property
    def requires_refusal(self) -> bool:
        return self.classification in (SafetyClass.REFUSE, SafetyClass.EMERGENCY)

    def to_dict(self) -> dict:
        return {
            "classification": self.classification.value,
            "reason": self.reason,
            "provider": self.provider,
        }


EMERGENCY_RESPONSE = (
    "This sounds like a medical emergency. Please stop using this assistant and "
    "seek immediate medical help: contact your local emergency number, or go to "
    "the nearest emergency department. If you are with someone who may be having "
    "a heart attack, stroke, severe allergic reaction, or difficulty breathing, "
    "call for emergency assistance right away."
)

REFUSAL_RESPONSE = (
    "I can only provide general, evidence-grounded information from official "
    "medical guidelines, and I cannot diagnose individual patients or give "
    "personal medical advice. Please consult a qualified healthcare professional "
    "for anything specific to you or a loved one."
)

CAUTION_NOTICE = (
    "Note: this information is general and not a substitute for personalized "
    "medical advice. Please discuss your own situation with a healthcare provider."
)