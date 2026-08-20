"""Language detection with a layered, robustness-first strategy:

1. Unicode-script heuristic — reliably identifies non-Latin scripts
   (Arabic, Hebrew, Cyrillic, Devanagari, Greek, CJK...) for which English
   is impossible.
2. English-signal heuristic — for Latin-script text, strong English
   function words / medical terms force English even for short queries
   (where statistical detectors like langdetect are unreliable).
3. langdetect — statistical refinement when the above are inconclusive.

The medical domain is English-heavy ("hypertension", "symptoms",
"treatment", ...), so short English questions must never be mis-labelled
as another Latin-script language.
"""
from __future__ import annotations

import re

from ..config.settings import get_settings
from .models import LanguageDetectionResult

# langdetect codes -> our supported codes (they mostly match; guard aliases)
_ALIASES = {
    "iw": "he",
    "ji": "yi",
    "zh-cn": "zh",
    "zh-tw": "zh",
}

# Unicode-script -> language hint.
_SCRIPT_TO_LANG = {
    "arabic": "ar",
    "hebrew": "he",
    "cyrillic": "ru",
    "devanagari": "hi",
    "greek": "el",
    "hiragana": "ja",
    "katakana": "ja",
    "hangul": "ko",
    "han": "zh",
}

_SCRIPT_RANGES = {
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "hebrew": [(0x0590, 0x05FF)],
    "cyrillic": [(0x0400, 0x04FF)],
    "devanagari": [(0x0900, 0x097F)],
    "greek": [(0x0370, 0x03FF)],
    "hiragana": [(0x3040, 0x309F)],
    "katakana": [(0x30A0, 0x30FF)],
    "hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "han": [(0x4E00, 0x9FFF)],
}

# Distinctly-English function words / medical terms. If enough of these are
# present in Latin-script text, it is English regardless of langdetect's
# short-text guess.
_ENGLISH_FUNCTION = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have",
    "has", "had", "will", "would", "can", "could", "should", "may", "might",
    "what", "how", "why", "when", "where", "which", "who", "it", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "my", "your", "our",
    "their", "about", "from", "as", "by", "at", "if", "then", "than", "also",
    "not", "no", "yes", "some", "any", "all", "there", "here",
}

_ENGLISH_MEDICAL = {
    "symptoms", "symptom", "treatment", "treat", "causes", "cause", "caused",
    "diagnosis", "diagnosed", "diagnose", "prevention", "prevent", "risk",
    "risks", "complications", "side effects", "dosage", "dose", "medication",
    "medicine", "drug", "disease", "diseases", "condition", "blood pressure",
    "hypertension", "diabetes", "diabetic", "cancer", "heart", "stroke",
    "infection", "vaccine", "fever", "pain", "chronic", "acute", "dangerous",
    "serious", "safe", "safety", "pregnant", "pregnancy", "children", "adults",
    "health", "healthy", "diet", "exercise", "manage", "management", "test",
    "tests", "blood", "sugar", "glucose", "cholesterol", "screening", "signs",
    "warning", "explain", "explaination", "describe", "difference", "compare",
    "comparison", "cure", "cures", "benefits", "effects", "help", "helps",
}


class LanguageDetector:
    def __init__(self):
        self.settings = get_settings()
        self.supported = {lang["code"] for lang in self.settings.language_supported}
        self.names = {lang["code"]: lang["name"] for lang in self.settings.language_supported}
        self.min_conf = self.settings.language_detection_min_confidence

    def detect(self, text: str) -> LanguageDetectionResult:
        text = (text or "").strip()
        if not text:
            return LanguageDetectionResult("en", "English", 0.0, note="empty input")

        # ---- 1) script heuristic (always available) ----
        script = self._detect_script(text)
        script_lang = _SCRIPT_TO_LANG.get(script)

        # Non-Latin script -> that language is authoritative (English impossible).
        if script_lang is not None:
            latin = len(re.findall(r"[A-Za-z]{3,}", text))
            mixed = latin >= 1 and script_lang == "ar"
            code, confidence = self._langdetect(text)
            if mixed:
                return LanguageDetectionResult(
                    "ar", self.names.get("ar", "Arabic"), 0.8,
                    is_supported=True, note="mixed-ar-en",
                )
            if code in self.supported and code != "en":
                return LanguageDetectionResult(code, self.names[code], confidence)
            return LanguageDetectionResult(
                script_lang, self.names.get(script_lang, script_lang.title()),
                0.85, is_supported=script_lang in self.supported,
                note="script-based detection",
            )

        # ---- 2) Latin script -> English-signal heuristic first ----
        # French/Spanish diacritics: do not force English because of "hypertension"
        if self._has_foreign_diacritics(text) and not self._looks_english(text[:80]):
            code, confidence = self._langdetect(text)
            if code in self.supported:
                return LanguageDetectionResult(code, self.names[code], confidence)
        if self._looks_english(text):
            return LanguageDetectionResult("en", "English", 0.95, note="english-signal")

        # ---- 3) langdetect statistical ----
        code, confidence = self._langdetect(text)
        if code in self.supported:
            return LanguageDetectionResult(code, self.names[code], confidence)

        # ---- 4) low-confidence Latin fallback ----
        return LanguageDetectionResult(
            "en", "English", 0.4, is_supported=True, note="low-confidence fallback"
        )

    # ---- helpers ----
    def _looks_english(self, text: str) -> bool:
        """Heuristic: Latin-script text with enough English-only signals."""
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        if not tokens:
            return False
        function_hits = tokens & _ENGLISH_FUNCTION
        medical_hits = tokens & _ENGLISH_MEDICAL
        # strong single-word English function signals
        if function_hits & {"the", "and", "what", "how", "why", "does", "is", "are"}:
            return True
        # otherwise require a combination
        if len(function_hits) >= 2:
            return True
        if len(medical_hits) >= 1 and len(function_hits) >= 1:
            return True
        # short standalone English medical query
        if len(tokens) <= 3 and medical_hits and not self._has_foreign_diacritics(text):
            return True
        return False

    @staticmethod
    def _has_foreign_diacritics(text: str) -> bool:
        return any(ch in text for ch in "àâäéèêëîïôöùûüçñáíóú¿¡")

    def _langdetect(self, text: str) -> tuple[str | None, float]:
        try:
            from langdetect import DetectorFactory, detect_langs

            DetectorFactory.seed = 0
            raw = detect_langs(text)
            if not raw:
                return None, 0.0
            # prefer the best *supported* guess
            for cand in raw:
                c = _ALIASES.get(cand.lang, cand.lang)
                if c in self.supported:
                    return c, float(cand.prob)
            return None, 0.0
        except Exception:  # noqa: BLE001
            return None, 0.0

    @staticmethod
    def _detect_script(text: str) -> str | None:
        best_script, best_count = None, 0
        for script, ranges in _SCRIPT_RANGES.items():
            count = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in ranges))
            if count > best_count:
                best_script, best_count = script, count
        return best_script if best_count > 0 else None