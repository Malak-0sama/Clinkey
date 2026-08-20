"""Translation service abstraction.

Two controlled translation points only:
  INPUT : original user query  -> canonical English query
  OUTPUT: validated English answer -> original user language

Providers are tried in order: Gemini (if available) -> Google Translate
bridge (deep-translator). On failure the caller receives ok=False and may
apply the configured English fallback.
"""
from __future__ import annotations

from ..config.settings import get_settings
from ..llm.client import LLMUnavailable, get_llm
from .detector import LanguageDetector
from .models import LanguageDetectionResult, TranslationResult
from .protected import TermProtector
from .clinical_check import validate_translation

# BCP-47 codes accepted by deep-translator (matches our supported list)
_GT_CODES = {"en", "ar", "fr", "es", "de", "it", "pt", "tr", "hi", "ur", "zh"}

# Map our internal codes to the exact codes expected by the translation
# providers (deep-translator / Google and MyMemory both expect zh-CN).
_PROVIDER_CODE_MAP = {"he": "iw", "zh": "zh-CN"}


class TranslationService:
    def __init__(self):
        self.settings = get_settings()
        self.detector = LanguageDetector()
        self.protector = TermProtector()

    def translate_from_english(self, text: str, target_language: str) -> TranslationResult:
        if target_language == "en":
            return TranslationResult(True, text, "en", "en", provider="identity")
        masked, table = self.protector.mask(text)
        result = self._translate(masked, "en", target_language)
        if result.ok:
            result.text = self.protector.unmask(result.text, table)
            check = validate_translation(text, result.text)
            if not check["ok"]:
                return TranslationResult(
                    False, text, "en", target_language,
                    provider=result.provider,
                    error="translation_validation:" + ",".join(check["issues"]),
                )
        return result

    # ---- detection ----
    def detect_language(self, text: str) -> LanguageDetectionResult:
        return self.detector.detect(text)

    # ---- input translation ----
    def translate_to_english(self, text: str, source_language: str) -> TranslationResult:
        if source_language == "en":
            return TranslationResult(True, text, "en", "en", provider="identity")
        result = self._translate(text, source_language, "en")
        # Fallback: if the source is actually English (mis-detected) and
        # translation failed/returned empty, keep the original text rather
        # than dropping the user's question.
        if not result.ok or not result.text.strip():
            if self._is_english_text(text):
                return TranslationResult(True, text, source_language, "en", provider="identity-fallback")
        return result

    # ---- internal ----
    def _translate(self, text: str, src: str, tgt: str) -> TranslationResult:
        if not text or not text.strip():
            return TranslationResult(False, "", src, tgt, error="empty input")

        # 1) Gemini
        try:
            llm = get_llm()
            if llm.available:
                result = self._gemini_translate(llm, text, src, tgt)
                if result.ok:
                    return result
        except (LLMUnavailable, Exception):  # noqa: BLE001
            pass

        # 2) Google Translate bridge (with retries)
        if src in _GT_CODES and tgt in _GT_CODES:
            out = self._google_translate(text, src, tgt)
            if out:
                return TranslationResult(True, out, src, tgt, provider="google")

        # 3) MyMemory free translation API (fallback)
        try:
            out = self._mymemory_translate(text, src, tgt)
            if out:
                return TranslationResult(True, out, src, tgt, provider="mymemory")
        except Exception as exc:  # noqa: BLE001
            return TranslationResult(False, text, src, tgt, provider="mymemory", error=str(exc)[:160])

        return TranslationResult(False, text, src, tgt, error="no provider available")

    @staticmethod
    def _is_english_text(text: str) -> bool:
        """Best-effort: does this Latin-script text look like English?"""
        import re

        tokens = set(re.findall(r"[a-z]+", text.lower()))
        english_signals = {
            "the", "and", "what", "how", "why", "does", "is", "are", "about",
            "symptoms", "treatment", "causes", "prevent", "diagnosed", "risk",
            "blood", "pressure", "hypertension", "diabetes", "dangerous",
            "complications", "side", "effects", "explain", "compare", "difference",
        }
        return bool(tokens & english_signals)

    @staticmethod
    def _google_translate(text: str, src: str, tgt: str, attempts: int = 3) -> str:
        import time

        from deep_translator import GoogleTranslator

        src = _PROVIDER_CODE_MAP.get(src, src)
        tgt = _PROVIDER_CODE_MAP.get(tgt, tgt)

        last_err: Exception | None = None
        for i in range(attempts):
            try:
                out = GoogleTranslator(source=src, target=tgt).translate(text)
                if out and out.strip() and out.strip() != text:
                    return out.strip()
                last_err = ValueError("empty or unchanged output")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            if i < attempts - 1:
                time.sleep(0.6 * (i + 1))
        if last_err:
            raise last_err
        return ""

    @staticmethod
    def _mymemory_translate(text: str, src: str, tgt: str) -> str:
        # MyMemory expects ISO codes; map ours where needed
        s = _PROVIDER_CODE_MAP.get(src, src)
        t = _PROVIDER_CODE_MAP.get(tgt, tgt)
        if s == t:
            return text
        import requests

        resp = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": f"{s}|{t}"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        out = (data.get("responseData") or {}).get("translatedText", "")
        out = (out or "").replace("MYMEMORY WARNING:", "").strip()
        if out and out != text:
            return out
        return ""

    @staticmethod
    def _gemini_translate(llm, text: str, src: str, tgt: str) -> TranslationResult:
        system = (
            "You are a precise medical translation engine. Translate the user text "
            "into the target language while strictly preserving: medical entities, "
            "diseases, symptoms, medications, dosages, units, numbers, negation, "
            "uncertainty, severity, and question intent. Translate semantically, not "
            "word-for-word. Map lay terms to standard medical terminology when it "
            "preserves meaning. Output ONLY the translation, no commentary."
        )
        prompt = f"Source language: {src}\nTarget language: {tgt}\n\nText:\n{text}"
        out = llm.generate(prompt, system=system, temperature=0.0)
        if out and out.strip():
            return TranslationResult(True, out.strip(), src, tgt, provider="gemini")
        return TranslationResult(False, text, src, tgt, provider="gemini", error="empty output")