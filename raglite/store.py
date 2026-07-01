from __future__ import annotations

import json  # 用于 manifest/chunks 序列化为 JSON 文本
# asdict 用于把 dataclass 实例转为 dict；dataclass 用于定义 Manifest/SearchResult
from dataclasses import asdict, dataclass
# datetime + timezone 用于生成 manifest 中的 UTC 时间戳
from datetime import datetime, timezone
from pathlib import Path
# typing.Any 用于类型化反序列化时的 payload dict
from typing import Any

# numpy：向量矩阵的存储与矩阵乘法核心依赖
import numpy as np

# 内部模块依赖
from .chunking import Chunk, chunk_documents  # 切片阶段
from .documents import DocumentLoadError, load_documents  # 文档加载
from .embeddings import DEFAULT_MODEL, Embedder, FastEmbedder, normalize_rows  # embedding


# 索引目录中三个关键文件的统一命名，集中管理便于排查
INDEX_FILE = "index.npy"  # 二进制向量矩阵（numpy 原生格式）
CHUNKS_FILE = "chunks.jsonl"  # 每个 chunk 一行 JSON（与 manifest 中 chunk_count 对齐）
MANIFEST_FILE = "manifest.json"  # 索引元数据（模型、维度、统计信息等）


class IndexError(ValueError):
    """索引读取或查询过程中发生的错误。

    继承 ValueError 是为了语义上区分于系统级错误（IOError 等）。
    使用 dataclass 风格的统一异常类型，方便上层（如 CLI）通过单一 except 子句捕获。
    """


@dataclass(frozen=True)
class Manifest:
    """索引的元数据（manifest），序列化后写入 manifest.json。

    Attributes:
        model_name: 索引构建时使用的 embedding 模型名称。
        dimension: 向量维度（与每条 chunk 对应的向量长度一致）。
        chunk_size: 构建索引时使用的目标切片字符数。
        overlap: 构建索引时使用的切片重叠字符数。
        document_count: 实际参与切片的非空文档数。
        chunk_count: 切分出的 chunk 数量，等于 INDEX_FILE 的行数。
        created_at: 索引构建时间（ISO-8601 格式，UTC 时区）。
    """
    model_name: str
    dimension: int
    chunk_size: int
    overlap: int
    document_count: int
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class SearchResult:
    """单条检索命中的结果。

    Attributes:
        score: 该 chunk 与 query 的相似度（已经是向量内积形式，归一化后等价于余弦相似度）。
        chunk: 命中的 chunk 本身，含文本与定位信息。
    """
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
    """构建本地向量索引的完整流水线：文档加载 → 切片 → embedding → 落盘。

    Args:
        input_path: 单文件或目录路径，作为语料源。
        index_path: 索引输出目录（不存在会自动创建）。
        chunk_size/overlap: 切片参数，会原样写入 manifest。
        model_name: embedding 模型名称，传入 FastEmbedder。
        embedder: 自定义 Embedder（如测试时传入 mock），为 None 时使用 FastEmbedder。

    Returns:
        构建完毕的 Manifest，供调用方展示或断言使用。

    Raises:
        DocumentLoadError: 没有加载到任何文档，或者全部文档都被切为空时抛出。
        IndexError: embedding 产出的向量数与 chunk 数不匹配时抛出。
        RuntimeError: embedding 模型初始化或编码失败时抛出（来自 FastEmbedder）。
    """
    # 1. 加载所有支持的文档
    documents = load_documents(input_path)
    if not documents:
        # 立刻失败：避免下游 embedding 在空输入上浪费资源
        raise DocumentLoadError(f"No supported non-empty documents found in: {input_path}")

    # 2. 把每篇文档切成多个 chunk
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise DocumentLoadError(f"No non-empty chunks produced from: {input_path}")

    # 3. 选择 embedder：优先使用调用方注入的（便于测试），否则实例化 FastEmbedder
    active_embedder = embedder or FastEmbedder(model_name=model_name)
    vectors = active_embedder.embed([chunk.text for chunk in chunks])
    # 4. L2 归一化：使后续检索只需一次矩阵乘法即可得到余弦相似度
    vectors = normalize_rows(vectors)

    # 防御性校验：embedder 必须为每个 chunk 都产出一个向量
    if vectors.shape[0] != len(chunks):
        raise IndexError(f"Embedder returned {vectors.shape[0]} vectors for {len(chunks)} chunks.")

    # 5. 构造 manifest 对象
    # getattr(..., 'model_name', model_name)：兼容自定义 embedder 不一定有 model_name 属性的情况
    manifest = Manifest(
        model_name=getattr(active_embedder, "model_name", model_name),
        dimension=int(vectors.shape[1]),
        chunk_size=chunk_size,
        overlap=overlap,
        document_count=len(documents),
        chunk_count=len(chunks),
        # UTC ISO-8601 时间戳，便于跨时区/跨系统比对索引版本
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # 6. 落盘：mkdir -p 等价操作（parents=True 会创建中间目录，exist_ok=True 避免已存在时报错）
    output_dir = Path(index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 向量矩阵以 numpy 原生格式保存（高效、可直接 np.load 恢复）
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
    """在本地索引上执行一次相似度检索。

    Args:
        query: 用户查询的原始文本。
        index_path: 索引目录路径。
        top_k: 期望返回的结果数；实际返回数可能更少（若索引中 chunk 不足）。
        model_name: 覆盖 manifest 中记录的模型名称；若为 None 则沿用 manifest 的模型。
            用于跨模型兼容场景（例如 manifest 来自其他环境构建，但当前只有另一款模型）。
        embedder: 自定义 Embedder（测试用）。

    Returns:
        按相似度降序排列的 SearchResult 列表。

    Raises:
        IndexError: query 为空、top_k <= 0、索引不完整、维度不匹配时抛出。
        RuntimeError: 模型调用失败时抛出。
    """
    # 参数前置校验：避免走到底层 embedder 才报错
    if not query.strip():
        raise IndexError("Query must not be empty.")
    if top_k <= 0:
        raise IndexError("top_k must be greater than 0.")

    # 加载磁盘上的索引（三件套：向量矩阵、chunks 列表、manifest）
    vectors, chunks, manifest = load_index(index_path)
    # 选择模型：调用方显式指定 > manifest 中的默认
    active_model_name = model_name or manifest.model_name
    active_embedder = embedder or FastEmbedder(model_name=active_model_name)
    # 单条 query 也走和批量 embedding 相同的归一化流程，保证维度与索引一致、相似度等价于余弦
    query_vector = normalize_rows(active_embedder.embed([query]))

    # 关键校验：query 向量的维度必须与索引一致，否则矩阵乘法会触发 numpy 广播错误或给出无意义结果
    if query_vector.shape[1] != manifest.dimension:
        raise IndexError(
            f"Query vector dimension {query_vector.shape[1]} does not match index dimension {manifest.dimension}."
        )

    # 检索核心：矩阵乘法 = 所有候选 chunk 与 query 的点积
    # 等价于余弦相似度的批量计算，复杂度 O(n*d)
    scores = vectors @ query_vector[0]
    # 实际可返回数不能超过 chunk 总数
    limit = min(top_k, len(chunks))
    # argsort 默认升序，[::-1] 反转为降序，再切片取前 limit 个；这里的 index 就是 vector 矩阵的行号
    ranked_indices = np.argsort(scores)[::-1][:limit]

    return [
        # float() 把 numpy 标量转为 Python 原生 float，避免 dataclass 在某些场景下被 numpy 类的不可序列化
        SearchResult(score=float(scores[index]), chunk=chunks[int(index)])
        for index in ranked_indices
    ]


def inspect_index(index_path: Path | str = ".raglite") -> Manifest:
    """仅读取并返回索引的 manifest（元数据），不执行检索。

    适用于 `raglite inspect` 这类"看一眼索引状况"的场景，避免大向量的全量加载。

    Args:
        index_path: 索引目录路径。

    Returns:
        Manifest 实例，描述该索引的元信息。
    """
    _, _, manifest = load_index(index_path)
    return manifest


def load_index(index_path: Path | str = ".raglite") -> tuple[np.ndarray, list[Chunk], Manifest]:
    """从磁盘加载完整的索引（三件套），并执行若干一致性校验。

    校验项：
    - 三个文件必须都存在（否则索引不完整）；
    - 向量矩阵必须是二维；
    - chunks 行数 == manifest.chunk_count；
    - 向量行数 == manifest.chunk_count；
    - 向量维度 == manifest.dimension。

    Args:
        index_path: 索引目录路径。

    Returns:
        (vectors, chunks, manifest) 三元组：
        - vectors：形状 (n_chunks, dimension) 的 float32 矩阵；
        - chunks：与之顺序对应的 Chunk 列表；
        - manifest：元数据。
    """
    index_dir = Path(index_path)
    vector_path = index_dir / INDEX_FILE
    chunk_path = index_dir / CHUNKS_FILE
    manifest_path = index_dir / MANIFEST_FILE

    # 一次性列出所有缺失文件，提示用户明确知道如何修复
    missing = [path.name for path in (vector_path, chunk_path, manifest_path) if not path.exists()]
    if missing:
        raise IndexError(f"Index is incomplete at {index_dir}. Missing: {', '.join(missing)}")

    # 加载向量矩阵；mmap_mode 在 npy 不太大时无需关心，这里使用默认 load
    vectors = np.load(vector_path)
    # 必须二维：检索的核心逻辑依赖矩阵乘法，1D/3D 都会触发下游错误
    if vectors.ndim != 2:
        raise IndexError(f"{INDEX_FILE} must contain a 2D matrix.")

    chunks = _read_chunks(chunk_path)
    manifest = _read_manifest(manifest_path)

    # 三向交叉校验：三者必须保持一致，否则属于损坏索引
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

    # 显式转为 float32：与 build_index 写盘时的 dtype 对齐，保证后续矩阵乘法的内存效率
    return vectors.astype(np.float32), chunks, manifest


def _write_manifest(path: Path, manifest: Manifest) -> None:
    """将 Manifest 序列化为 JSON 写入 manifest.json。

    使用 ensure_ascii=False 以保留中文 / Unicode 字符的可读性；
    indent=2 让文件可直接被人阅读和 git diff 友好对照。
    """
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_manifest(path: Path) -> Manifest:
    """读取 manifest.json 并反序列化为 Manifest 实例。

    Raises:
        json.JSONDecodeError: JSON 损坏时抛出。
        TypeError: 字段缺失时抛出（由 ** 解包触发）。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(**payload)


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    """将 chunks 列表按 JSONL 格式逐行写入 chunks.jsonl。

    JSONL 相对于单一大 JSON 数组的优势：
    - 写入/读取都可以流式进行，内存友好；
    - 任意一行损坏不会影响其他行的解析；
    - 文本编辑器可以直接打开看。

    ensure_ascii=False 同样是为了保留非 ASCII 字符。
    """
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def _read_chunks(path: Path) -> list[Chunk]:
    """从 chunks.jsonl 逐行读取并重建 Chunk 列表。

    Raises:
        IndexError: 任意一行的字段不能构造 Chunk 时（被包装并附加行号）。
        json.JSONDecodeError: 单行 JSON 损坏时抛出。
    """
    chunks: list[Chunk] = []
    # enumerate 从 1 开始，方便报告具体行号
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        # 跳过空行（与写入端策略一致）
        if not line.strip():
            continue
        payload: dict[str, Any] = json.loads(line)
        try:
            # ** 解包时如果多了/少了字段会触发 TypeError，统一转为 IndexError 暴露给上层
            chunks.append(Chunk(**payload))
        except TypeError as exc:
            raise IndexError(f"{path}:{line_number} is not a valid chunk row.") from exc
    return chunks
