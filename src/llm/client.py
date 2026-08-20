"""LLM abstraction layer.

Gemini is used when GEMINI_API_KEY is present; otherwise callers receive
an LLMUnavailable signal and fall back to deterministic behavior. This
keeps the RAG core independent of any single model provider.
"""
from __future__ import annotations

from ..config.settings import Settings, get_settings


class LLMUnavailable(Exception):
    """Raised when the configured LLM cannot be reached."""


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_effective_provider
        self.model = None
        if self.provider == "gemini":
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.settings.gemini_api_key)
                self.model = genai.GenerativeModel(self.settings.get("llm_model", "gemini-1.5-flash"))
            except Exception:
                self.model = None

    @property
    def available(self) -> bool:
        return self.provider == "gemini" and self.model is not None

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        """Generate free-form text. Raises LLMUnavailable in demo mode."""
        if not self.available:
            raise LLMUnavailable("LLM is not available (demo mode)")
        import google.generativeai as genai

        temp = self.settings.get("llm_generation_temperature", 0.2) if temperature is None else temperature
        parts = []
        if system:
            parts.append(f"System instructions:\n{system}")
        parts.append(prompt)
        try:
            resp = self.model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(temperature=temp),
            )
            return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"LLM call failed: {exc}") from exc

    def generate_json(self, prompt: str, system: str = "") -> dict | list:
        """Generate structured JSON. Returns parsed object; raises on failure."""
        import json

        text = self.generate(prompt, system=system)
        text = text.strip()
        # tolerate markdown code fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 and text.startswith("["):
            start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            raise LLMUnavailable("LLM returned non-JSON output")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"LLM returned malformed JSON: {exc}") from exc


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client