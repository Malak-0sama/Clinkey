"""Embedding layer.

Production must use a real semantic model. Hash embeddings are allowed only
when explicitly configured (tests / offline fixtures), never as a silent fallback.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from ..config.settings import get_settings
from .errors import EmbeddingConfigError

_HASH_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    def __init__(self):
        self.settings = get_settings()
        requested = self.settings.get("embeddings_provider", "auto")
        self.model_name = str(self.settings.get("embeddings_model", "BAAI/bge-small-en-v1.5"))
        self.provider = self.settings.embeddings_effective_provider
        self._model = None
        allow_hash = bool(self.settings.get("embeddings_allow_hash_fallback", False))
        if os.environ.get("CLINKEY_ALLOW_HASH", "").lower() in {"1", "true", "yes"}:
            allow_hash = True
        if requested == "hash":
            allow_hash = True
            self.provider = "hash"

        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001
                if not allow_hash:
                    raise EmbeddingConfigError(
                        "Production embeddings require sentence-transformers "
                        f"model '{self.model_name}'. Install it or set "
                        "embeddings.allow_hash_fallback=true / CLINKEY_ALLOW_HASH=1 "
                        "for local fixtures only."
                    ) from exc
                self._model = None
                self.provider = "hash"
        elif self.provider == "hash" and not allow_hash and requested == "auto":
            raise EmbeddingConfigError(
                "No semantic embedding model is installed. Production must not "
                "silently use hash embeddings. Install sentence-transformers or "
                "explicitly set embeddings.provider=hash for development."
            )

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension()) if self._model else _HASH_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        return self._hash_embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @staticmethod
    def _hash_embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), _HASH_DIM), dtype="float32")
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            if not tokens:
                continue
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                out[i, h % _HASH_DIM] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out