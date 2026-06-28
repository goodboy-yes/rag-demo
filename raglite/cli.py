from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .documents import DocumentLoadError
from .embeddings import DEFAULT_MODEL
from .store import IndexError, build_index, inspect_index, search_index


app = typer.Typer(help="A lightweight local vector retrieval demo.")
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="A file or directory containing md/txt/jsonl documents."),
    index: Path = typer.Option(Path(".raglite"), "--index", "-i", help="Index output directory."),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Maximum characters per chunk."),
    overlap: int = typer.Option(80, "--overlap", help="Character overlap between adjacent chunks."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="FastEmbed model name."),
) -> None:
    """Load documents, embed chunks, and rebuild the local vector index."""
    try:
        manifest = build_index(
            path,
            index_path=index,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model,
        )
    except (DocumentLoadError, IndexError, ValueError, RuntimeError) as exc:
        _fail(str(exc))

    table = Table(title="Index Built")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("index", str(index))
    table.add_row("model", manifest.model_name)
    table.add_row("documents", str(manifest.document_count))
    table.add_row("chunks", str(manifest.chunk_count))
    table.add_row("dimension", str(manifest.dimension))
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    index: Path = typer.Option(Path(".raglite"), "--index", "-i", help="Index directory."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return."),
    model: str | None = typer.Option(None, "--model", help="Override the model recorded in manifest."),
) -> None:
    """Search the vector index and print the top matching chunks."""
    try:
        results = search_index(query, index_path=index, top_k=top_k, model_name=model)
    except (IndexError, RuntimeError, ValueError) as exc:
        _fail(str(exc))

    table = Table(title=f"Top {len(results)} Results")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Chunk", justify="right")
    table.add_column("Preview")

    for rank, result in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{result.score:.4f}",
            result.chunk.source,
            str(result.chunk.chunk_index),
            _preview(result.chunk.text),
        )

    console.print(table)


@app.command(name="inspect")
def inspect_command(
    index: Path = typer.Option(Path(".raglite"), "--index", "-i", help="Index directory."),
) -> None:
    """Show index metadata."""
    try:
        manifest = inspect_index(index)
    except IndexError as exc:
        _fail(str(exc))

    table = Table(title="Index Info")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("index", str(index))
    table.add_row("model", manifest.model_name)
    table.add_row("dimension", str(manifest.dimension))
    table.add_row("documents", str(manifest.document_count))
    table.add_row("chunks", str(manifest.chunk_count))
    table.add_row("chunk_size", str(manifest.chunk_size))
    table.add_row("overlap", str(manifest.overlap))
    table.add_row("created_at", manifest.created_at)
    console.print(table)


def _preview(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1)
