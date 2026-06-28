from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_EXTENSIONS = {".md", ".txt", ".jsonl"}


class DocumentLoadError(ValueError):
    """Raised when an input document cannot be loaded as text."""


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_documents(path: Path | str) -> list[Document]:
    root = Path(path)
    if not root.exists():
        raise DocumentLoadError(f"Input path does not exist: {root}")

    files = _iter_supported_files(root)
    documents: list[Document] = []
    for file_path in files:
        documents.extend(_load_file(file_path))

    return documents


def _iter_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [path]

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_file(path: Path) -> Iterable[Document]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        return [Document(text=text, source=str(path), metadata={"file_type": suffix.lstrip(".")})]

    if suffix == ".jsonl":
        return _load_jsonl(path)

    raise DocumentLoadError(f"Unsupported file extension: {suffix}")


def _load_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(f"{path}:{line_number} is not valid JSON: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise DocumentLoadError(f"{path}:{line_number} must be a JSON object.")

        text = payload.get("text", payload.get("content"))
        if not isinstance(text, str) or not text.strip():
            raise DocumentLoadError(f"{path}:{line_number} must contain a non-empty text/content field.")

        metadata = {key: value for key, value in payload.items() if key not in {"text", "content"}}
        metadata.update({"file_type": "jsonl", "line": line_number})
        documents.append(Document(text=text, source=str(path), metadata=metadata))

    return documents
