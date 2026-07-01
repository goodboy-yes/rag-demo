from __future__ import annotations

# 标准库导入
from pathlib import Path

# 第三方库：typer 用于构建命令行接口，rich 用于美化终端输出（彩色文本和表格）
import typer
from rich.console import Console
from rich.table import Table

# 内部模块导入
from .documents import DocumentLoadError  # 文档加载异常
from .embeddings import DEFAULT_MODEL  # 默认的嵌入模型名称
from .store import IndexError, build_index, inspect_index, search_index  # 索引相关操作


# 创建 typer 应用实例，作为整个 CLI 的入口点
app = typer.Typer(help="A lightweight local vector retrieval demo.")
# 创建 rich 控制台实例，用于打印彩色文本和表格
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
        # 调用底层构建索引的函数：
        # - 读取 path 下的文档
        # - 将文档切分为指定大小的 chunk（带 overlap 重叠）
        # - 使用指定 model 生成向量嵌入
        # - 将索引写入 index 指定的目录
        manifest = build_index(
            path,
            index_path=index,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model,
        )
    except (DocumentLoadError, IndexError, ValueError, RuntimeError) as exc:
        # 捕获构建过程中可能出现的各类错误，统一交给 _fail 处理（打印红色错误信息并以非零状态退出）
        _fail(str(exc))

    # 构建成功，使用 rich 表格输出索引的元数据摘要
    table = Table(title="Index Built")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("index", str(index))                # 索引存储路径
    table.add_row("model", manifest.model_name)       # 使用的嵌入模型
    table.add_row("documents", str(manifest.document_count))  # 索引包含的文档数量
    table.add_row("chunks", str(manifest.chunk_count))        # 切分出的文本块数量
    table.add_row("dimension", str(manifest.dimension))        # 向量维度
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
        # 调用底层搜索函数：
        # - 将 query 通过嵌入模型转为向量
        # - 在 index 目录中检索最相似的 top_k 个文本块
        # - 如果指定了 model，则覆盖 manifest 中记录的模型（用于跨模型兼容场景）
        results = search_index(query, index_path=index, top_k=top_k, model_name=model)
    except (IndexError, RuntimeError, ValueError) as exc:
        # 捕获搜索过程中可能出现的错误（索引不存在、模型不匹配等），统一交给 _fail 处理
        _fail(str(exc))

    # 构建结果表格，标题中显示实际返回的结果数（可能小于请求的 top_k，例如索引为空时）
    table = Table(title=f"Top {len(results)} Results")
    table.add_column("#", justify="right")              # 排名序号（右对齐）
    table.add_column("Score", justify="right")          # 相似度分数（右对齐，保留 4 位小数）
    table.add_column("Source")                          # 文档来源
    table.add_column("Chunk", justify="right")          # 文本块在原文档中的序号
    table.add_column("Preview")                         # 文本块内容的预览（前 120 个字符）

    # 遍历结果，enumerate 从 1 开始计数以便作为排名序号
    for rank, result in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{result.score:.4f}",          # 相似度分数保留 4 位小数
            result.chunk.source,            # 文档来源路径
            str(result.chunk.chunk_index),  # 文本块在原文档中的索引
            _preview(result.chunk.text),    # 调用辅助函数生成内容预览
        )

    console.print(table)


@app.command(name="inspect")
def inspect_command(
    index: Path = typer.Option(Path(".raglite"), "--index", "-i", help="Index directory."),
) -> None:
    """Show index metadata."""
    try:
        # 读取 index 目录中的 manifest 文件，获取索引元数据（不执行任何搜索或重建）
        manifest = inspect_index(index)
    except IndexError as exc:
        # 当索引不存在或已损坏时抛出此异常，统一交给 _fail 处理
        _fail(str(exc))

    # 输出索引的详细元数据表格
    table = Table(title="Index Info")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("index", str(index))                  # 索引路径
    table.add_row("model", manifest.model_name)         # 嵌入模型名称
    table.add_row("dimension", str(manifest.dimension)) # 向量维度
    table.add_row("documents", str(manifest.document_count))  # 文档数量
    table.add_row("chunks", str(manifest.chunk_count))        # 文本块数量
    table.add_row("chunk_size", str(manifest.chunk_size))     # 切块大小
    table.add_row("overlap", str(manifest.overlap))           # 切块重叠长度
    table.add_row("created_at", manifest.created_at)         # 索引创建时间
    console.print(table)


def _preview(text: str, limit: int = 120) -> str:
    """辅助函数：生成文本的简短预览，用于在表格中显示文本块内容。

    Args:
        text: 原始文本（可能包含多余的空白字符和换行符）。
        limit: 预览的最大长度（字符数），默认为 120。

    Returns:
        规范化后的单行文本；若超过长度限制则截断并附加省略号。
    """
    # 使用 split() 拆分后用单空格连接，去除所有连续的空白字符（包括换行符、制表符等）
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    # 截断到 limit - 3 个字符以预留空间给 "..." 省略号
    return normalized[: limit - 3] + "..."


def _fail(message: str) -> None:
    """辅助函数：统一的错误处理流程。

    使用 rich 以红色高亮显示错误信息，然后以退出码 1 终止 CLI 进程。
    使用 typer.Exit 而非直接 sys.exit 是为了与 typer 框架更好地集成。

    Args:
        message: 要显示给用户的错误描述。
    """
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1)
