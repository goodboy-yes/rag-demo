from __future__ import annotations

# 类型标注：Protocol 用于定义结构化类型（鸭子类型），Sequence 用于接受任意可索引序列
from typing import Protocol, Sequence

import numpy as np


# 默认使用的 embedding 模型
# BAAI/bge-small-zh-v1.5：智源研究院开源的中文向量模型，体积小、速度快
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
# 默认向量维度，必须与所选模型实际输出的维度一致，用于一些需要知道维度的场景（如在未实际运行模型时占位）
DEFAULT_DIMENSION = 512


class Embedder(Protocol):
    """Embedding 模型的抽象接口（Protocol 形式）。

    通过定义 Protocol，本项目允许不同后端的 embedding 实现（如 FastEmbedder、
    或未来可能添加的 OpenAI/Sentence-Transformers 等），只要具备 `model_name`
    属性和 `embed()` 方法即可在任何期望 Embedder 的地方被使用。
    """
    # 当前加载的模型名称（用于在 manifest 中记录，确保检索时模型一致）
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """将一组文本编码为向量矩阵。

        Args:
            texts: 待编码的文本序列（如若干个 chunk 或一个 query）。

        Returns:
            形状为 (len(texts), dimension) 的 numpy 数组，每行为一段文本对应的向量。
        """
        ...


class FastEmbedder:
    """基于 fastembed 库的 Embedder 实现。

    fastembed 是一个轻量级、纯 Python/CPU 友好的 embedding 库，
    与原始 sentence-transformers 相比，无需 torch，启动更快、体积更小。
    """
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # 记录实际使用的模型名称
        self.model_name = model_name

        # fastembed 是可选依赖，仅在真正使用 embeddings 时才检查导入
        # 这样在 build_index 或 search 之前其他模块（如 load_documents）可以正常工作
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            # 将底层 ImportError 包装为更友好的 RuntimeError，引导用户安装依赖
            raise RuntimeError(
                "fastembed is not installed. Run `pip install -e \".[dev]\"` first."
            ) from exc

        # 实例化 fastembed 的 TextEmbedding 类；
        # 首次运行时会自动下载模型权重到本地缓存（默认 ~/.cache/fastembed）
        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:
            # 模型名称拼错或网络拉取失败都可能触发，统一包装为 RuntimeError
            raise RuntimeError(f"Failed to initialize FastEmbed model '{model_name}': {exc}") from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """调用 fastembed 模型对文本进行批量编码。

        Args:
            texts: 待编码的文本序列；为空时直接返回空矩阵，避免触发底层库报错。

        Returns:
            形状为 (n, dimension) 的 float32 向量矩阵。
        """
        # 空输入短路：避免调用底层库时传入空列表导致意外行为
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        try:
            # fastembed 的 embed 是一个惰性生成器（generator），需要 list() 一次性求值
            # 内部会按模型的 batch size 自动分批推理
            vectors = list(self._model.embed(list(texts)))
        except Exception as exc:
            # 编码过程中可能因为 OOM、模型损坏等原因失败，统一包装为 RuntimeError
            raise RuntimeError(f"Failed to embed texts with model '{self.model_name}': {exc}") from exc
        if not vectors:
            # fastembed 在某些边界情况下可能返回空列表，做一次防御性检查
            return np.empty((0, 0), dtype=np.float32)

        # 将 list[ndarray] 转换为单个二维 ndarray，并固定 dtype 为 float32
        # （fastembed 默认输出 float32，但显式声明可避免后续 numpy 操作时类型不匹配）
        return np.asarray(vectors, dtype=np.float32)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """对向量矩阵的每一行做 L2 归一化，使每行的模长为 1。

    L2 归一化后，两个向量的点积等价于它们的余弦相似度，
    因此检索时只需用一次矩阵乘法（vectors @ query_vector）即可得到所有候选的相似度分数，
    避免再单独计算余弦，大幅提升检索效率。

    Args:
        vectors: 形状为 (n, dimension) 的二维 numpy 数组。

    Returns:
        与输入同形状的 float32 数组，每一行已经 L2 归一化。

    Raises:
        ValueError: 当输入不是二维矩阵时抛出（如一维向量或三维张量）。
    """
    # 确保输入是二维矩阵（行=样本，列=特征维度），否则后续 axis=1 没有意义
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D matrix.")

    # axis=1 表示按行计算 L2 范数；keepdims=True 保留维度，便于后续广播除法
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 防止零向量除以 0 导致 NaN：将范数为 0 的位置替换为 1.0（这样 0/1 = 0，保留零向量）
    safe_norms = np.where(norms == 0, 1.0, norms)
    # 逐元素除法得到归一化结果，并显式转为 float32 以便后续 .npy 存储
    return (vectors / safe_norms).astype(np.float32)
