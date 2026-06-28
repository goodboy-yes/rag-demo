from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_DIMENSION = 512


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        ...


class FastEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is not installed. Run `pip install -e \".[dev]\"` first."
            ) from exc

        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize FastEmbed model '{model_name}': {exc}") from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        try:
            vectors = list(self._model.embed(list(texts)))
        except Exception as exc:
            raise RuntimeError(f"Failed to embed texts with model '{self.model_name}': {exc}") from exc
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)

        return np.asarray(vectors, dtype=np.float32)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D matrix.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return (vectors / safe_norms).astype(np.float32)
