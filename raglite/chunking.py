from __future__ import annotations

# dataclass 用于定义 Chunk 数据结构；field 提供可变性默认值
from dataclasses import dataclass, field
# Iterable 接受任意可迭代的文档集合（如 list、generator）
from typing import Any, Iterable

# Document 是上一阶段加载到的统一文档对象
from .documents import Document


@dataclass(frozen=True)
class Chunk:
    """切片阶段产出的最小检索单元。

    Attributes:
        id: 跨整个语料库唯一的全局 chunk 编号，用于在向量矩阵中定位行（与 vectors[i] 一一对应）。
        text: 切片后的文本内容（已去除两端空白）。
        source: 该 chunk 所属的原始文档路径，便于检索结果直接定位回源文件。
        chunk_index: 该 chunk 在所属文档内的切片序号（从 0 开始），用来在某个文件中排序。
        char_start: chunk 起始字符在原始文档文本中的偏移（左闭区间）。
        char_end: chunk 结束字符在原始文档文本中的偏移（左闭右闭区间），可用于精确高亮。
        metadata: 从 Document 复制来的元数据，可携带 file_type、来源行号、id 等附加信息。
    """
    id: int
    text: str
    source: str
    chunk_index: int
    char_start: int
    char_end: int
    # metadata 是 dict，使用 default_factory 避免 dataclass 共享同一可变对象
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """把一组 Document 切片为带全局编号的 Chunk 列表。

    Args:
        documents: 已加载好的文档（任意可迭代对象，允许多次迭代因为内部用 list 消费）。
        chunk_size: 每个 chunk 的最大字符数（不包括重叠部分）。
        overlap: 相邻 chunk 之间的重叠字符数，使跨边界语义不丢失。

    Returns:
        全局唯一编号的 Chunk 列表，顺序与文档遍历顺序一致。
    """
    # 先对参数做防御性校验，让错误尽早暴露
    validate_chunk_options(chunk_size, overlap)
    chunks: list[Chunk] = []
    # 全局唯一 id 自增游标
    next_id = 0

    for document in documents:
        # 把单篇文档切成若干 (text, start, end) 三元组
        document_chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        # enumerate 给出文档内的 chunk_index，从 0 起步
        for chunk_index, (text, start, end) in enumerate(document_chunks):
            chunks.append(
                Chunk(
                    id=next_id,
                    text=text,
                    # source 取自 Document，检索结果可以追溯到原始文件
                    source=document.source,
                    # 文档内的相对位置
                    chunk_index=chunk_index,
                    # 字符级偏移，便于高亮显示
                    char_start=start,
                    char_end=end,
                    # metadata 必须显式拷贝 dict：原 Document 是 frozen dataclass，
                    # 但 dict 自身仍是可变对象，多个 chunk 共用同一份 dict 在未来被修改时会出错
                    metadata=dict(document.metadata),
                )
            )
            next_id += 1

    return chunks


def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 80) -> list[tuple[str, int, int]]:
    """将单段连续文本切分为若干带重叠的 chunk。

    切分时优先在语义边界处断开（段落 > 句子 > 硬切），保证每个 chunk 都是相对完整的语义单元，
    有利于后续 embedding 模型产出更稳定的向量。

    Args:
        text: 任意文本字符串。
        chunk_size: 单个 chunk 的目标最大字符数。
        overlap: 相邻 chunk 之间的字符重叠量，用于让边界处的语义在两侧 chunk 中都能被检索到。

    Returns:
        每个元素为 (text, char_start, char_end) 三元组的列表：
        - text：去除了两端空白的 chunk 内容；
        - char_start / char_end：对应文本在 `text` 中的字符偏移（去除两端空白后重新计算的偏移）。
    """
    # 参数校验（函数被独立调用时也能直接复用）
    validate_chunk_options(chunk_size, overlap)
    # 统一换行符再 strip：避免 Windows 文件中 \r\n 干扰 chunk 边界判断
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_length = len(normalized)

    # 主循环：每次产出一个 chunk，然后回退 overlap 个字符继续切分
    while start < text_length:
        # 硬切点的最大位置（不能超过文本末尾）
        hard_end = min(start + chunk_size, text_length)
        # 优先在段落/句子边界断开；如果找不到合适的软边界，再使用硬切点
        end = _choose_boundary(normalized, start, hard_end, chunk_size)
        raw_chunk = normalized[start:end]
        chunk = raw_chunk.strip()

        if chunk:
            # strip() 会去掉两端空白，但 char_start/char_end 需要指向原始文本中的偏移，
            # 否则检索回显时会出现"字符偏移对不上原文"的问题。
            leading_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            # 末尾：rstrip() 等价于去除末尾空白，用 raw_chunk 减去末尾空白长度就是"原始文本中真正有效范围的结束偏移"。
            trailing_trim = len(raw_chunk.rstrip())
            chunks.append((chunk, start + leading_trim, start + trailing_trim))

        # 如果已经切到文本末尾，循环结束
        if end >= text_length:
            break

        # 计算下一个 chunk 的起始位置：
        # - 至少回退 overlap 个字符（保证有重叠）
        # - 但不能 <= 当前 start，否则会无限循环，所以下界是 start + 1
        next_start = max(end - overlap, start + 1)
        # 跳过重叠区起始位置的空白字符，避免在词中间断开（同时让 chunk 开头更干净）
        while next_start < text_length and normalized[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def validate_chunk_options(chunk_size: int, overlap: int) -> None:
    """校验切块参数合法性。

    Raises:
        ValueError: chunk_size <= 0 或 overlap 超出允许范围时抛出。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0.")
    if overlap >= chunk_size:
        # overlap 必须 < chunk_size，否则下次起始位置会落回当前 chunk 内部，导致死循环或退化
        raise ValueError("overlap must be smaller than chunk_size.")


def _choose_boundary(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    """在硬切点范围内找一个更"自然"的语义边界。

    优先级：
    1. 段落边界（\\n\\n）；
    2. 行边界（\\n）；
    3. 句子边界（中文/英文常见句末标点）；
    4. 实在找不到则回退到硬切点（按字符数硬切）。

    选用"软边界"前的最低要求是：必须达到 chunk_size * 0.55，避免一味追求语义完整而让 chunk 过短。

    Args:
        text: 整个归一化后的文本。
        start: 当前 chunk 在文本中的起始偏移。
        hard_end: 当前 chunk 在文本中的最远结束偏移（硬切点）。
        chunk_size: 目标 chunk 大小，用于计算软边界下限。

    Returns:
        选定的 chunk 结束偏移（绝对偏移）。
    """
    # 已经到达文本末尾：直接返回文本长度
    if hard_end >= len(text):
        return len(text)

    # 软边界的下限：必须达到 chunk_size 的 55% 以上，否则该软边界"太靠前"会切出极短的 chunk
    # max(1, ...) 防止 chunk_size 极小时下限退化为 0
    min_soft_end = start + max(1, int(chunk_size * 0.55))
    window = text[start:hard_end]

    # 优先尝试段落边界（\n\n），失败则退到行边界（\n）
    for boundary in ("\n\n", "\n"):
        idx = window.rfind(boundary)
        if idx != -1 and start + idx + len(boundary) >= min_soft_end:
            # 找到 >= 下限的边界，使用 `rfind` 保证切点在窗口内尽量靠后（即更接近 hard_end）
            return start + idx + len(boundary)

    # 句子边界搜索：中英文常见句末标点；rfind 在窗口内找最靠后的标记
    punctuation_positions = [
        window.rfind(mark)
        for mark in ("。", "！", "？", ".", "!", "?", "；", ";")
    ]
    idx = max(punctuation_positions)
    if idx != -1 and start + idx + 1 >= min_soft_end:
        # +1 是为了把标点符号本身也包含进 chunk（让句子保持完整）
        return start + idx + 1

    # 没有任何合适的软边界，回退到硬切点（按字符数硬切）
    return hard_end
