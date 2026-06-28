from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .documents import Document


@dataclass(frozen=True)
class Chunk:
    id: int
    text: str
    source: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    validate_chunk_options(chunk_size, overlap)
    chunks: list[Chunk] = []
    next_id = 0

    for document in documents:
        document_chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, (text, start, end) in enumerate(document_chunks):
            chunks.append(
                Chunk(
                    id=next_id,
                    text=text,
                    source=document.source,
                    chunk_index=chunk_index,
                    char_start=start,
                    char_end=end,
                    metadata=dict(document.metadata),
                )
            )
            next_id += 1

    return chunks


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 80) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries."""
    validate_chunk_options(chunk_size, overlap)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = _choose_boundary(normalized, start, hard_end, chunk_size)
        raw_chunk = normalized[start:end]
        chunk = raw_chunk.strip()

        if chunk:
            leading_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            trailing_trim = len(raw_chunk.rstrip())
            chunks.append((chunk, start + leading_trim, start + trailing_trim))

        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)
        while next_start < text_length and normalized[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def validate_chunk_options(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def _choose_boundary(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    if hard_end >= len(text):
        return len(text)

    min_soft_end = start + max(1, int(chunk_size * 0.55))
    window = text[start:hard_end]

    for boundary in ("\n\n", "\n"):
        idx = window.rfind(boundary)
        if idx != -1 and start + idx + len(boundary) >= min_soft_end:
            return start + idx + len(boundary)

    punctuation_positions = [
        window.rfind(mark)
        for mark in ("。", "！", "？", ".", "!", "?", "；", ";")
    ]
    idx = max(punctuation_positions)
    if idx != -1 and start + idx + 1 >= min_soft_end:
        return start + idx + 1

    return hard_end
