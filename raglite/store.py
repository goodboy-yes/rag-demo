from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_documents
from .documents import DocumentLoadError, load_documents
from .embeddings import DEFAULT_MODEL, Embedder, FastEmbedder, normalize_rows


INDEX_FILE = "index.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


class IndexError(ValueError):
    """Raised when an index cannot be read or queried."""


@dataclass(frozen=True)
class Manifest:
    model_name: str
    dimension: int
    chunk_size: int
    overlap: int
    document_count: int
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk: Chunk


def build_index(
    input_path: Path | str,
    *,
    index_path: Path | str = ".raglite",
    chunk_size: int = 500,
    overlap: int = 80,
    model_name: str = DEFAULT_MODEL,
    embedder: Embedder | None = None,
) -> Manifest:
    documents = load_documents(input_path)
    if not documents:
        raise DocumentLoadError(f"No supported non-empty documents found in: {input_path}")

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise DocumentLoadError(f"No non-empty chunks produced from: {input_path}")

    active_embedder = embedder or FastEmbedder(model_name=model_name)
    vectors = active_embedder.embed([chunk.text for chunk in chunks])
    vectors = normalize_rows(vectors)

    if vectors.shape[0] != len(chunks):
        raise IndexError(f"Embedder returned {vectors.shape[0]} vectors for {len(chunks)} chunks.")

    manifest = Manifest(
        model_name=getattr(active_embedder, "model_name", model_name),
        dimension=int(vectors.shape[1]),
        chunk_size=chunk_size,
        overlap=overlap,
        document_count=len(documents),
        chunk_count=len(chunks),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    output_dir = Path(index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / INDEX_FILE, vectors)
    _write_chunks(output_dir / CHUNKS_FILE, chunks)
    _write_manifest(output_dir / MANIFEST_FILE, manifest)

    return manifest


def search_index(
    query: str,
    *,
    index_path: Path | str = ".raglite",
    top_k: int = 5,
    model_name: str | None = None,
    embedder: Embedder | None = None,
) -> list[SearchResult]:
    if not query.strip():
        raise IndexError("Query must not be empty.")
    if top_k <= 0:
        raise IndexError("top_k must be greater than 0.")

    vectors, chunks, manifest = load_index(index_path)
    active_model_name = model_name or manifest.model_name
    active_embedder = embedder or FastEmbedder(model_name=active_model_name)
    query_vector = normalize_rows(active_embedder.embed([query]))

    if query_vector.shape[1] != manifest.dimension:
        raise IndexError(
            f"Query vector dimension {query_vector.shape[1]} does not match index dimension {manifest.dimension}."
        )

    scores = vectors @ query_vector[0]
    limit = min(top_k, len(chunks))
    ranked_indices = np.argsort(scores)[::-1][:limit]

    return [
        SearchResult(score=float(scores[index]), chunk=chunks[int(index)])
        for index in ranked_indices
    ]


def inspect_index(index_path: Path | str = ".raglite") -> Manifest:
    _, _, manifest = load_index(index_path)
    return manifest


def load_index(index_path: Path | str = ".raglite") -> tuple[np.ndarray, list[Chunk], Manifest]:
    index_dir = Path(index_path)
    vector_path = index_dir / INDEX_FILE
    chunk_path = index_dir / CHUNKS_FILE
    manifest_path = index_dir / MANIFEST_FILE

    missing = [path.name for path in (vector_path, chunk_path, manifest_path) if not path.exists()]
    if missing:
        raise IndexError(f"Index is incomplete at {index_dir}. Missing: {', '.join(missing)}")

    vectors = np.load(vector_path)
    if vectors.ndim != 2:
        raise IndexError(f"{INDEX_FILE} must contain a 2D matrix.")

    chunks = _read_chunks(chunk_path)
    manifest = _read_manifest(manifest_path)

    if len(chunks) != manifest.chunk_count:
        raise IndexError(
            f"Manifest says {manifest.chunk_count} chunks, but {CHUNKS_FILE} has {len(chunks)} rows."
        )
    if vectors.shape[0] != manifest.chunk_count:
        raise IndexError(
            f"Manifest says {manifest.chunk_count} chunks, but {INDEX_FILE} has {vectors.shape[0]} rows."
        )
    if vectors.shape[1] != manifest.dimension:
        raise IndexError(
            f"Manifest says dimension {manifest.dimension}, but {INDEX_FILE} has dimension {vectors.shape[1]}."
        )

    return vectors.astype(np.float32), chunks, manifest


def _write_manifest(path: Path, manifest: Manifest) -> None:
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_manifest(path: Path) -> Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(**payload)


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def _read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload: dict[str, Any] = json.loads(line)
        try:
            chunks.append(Chunk(**payload))
        except TypeError as exc:
            raise IndexError(f"{path}:{line_number} is not a valid chunk row.") from exc
    return chunks
