"""轻量级本地向量检索 Demo 包。

主要能力：
- 从本地文件系统加载 .md / .txt / .jsonl 文档；
- 对文档进行带重叠的语义切片（优先在段落/句子边界处断开）；
- 使用 fastembed 提供的本地 embedding 模型将切片编码为向量；
- 将向量矩阵、切片元数据、索引 manifest 三件套落盘；
- 基于余弦相似度在已构建索引上做 top-k 检索。

典型用法（命令行）：
    python -m raglite ingest path/to/docs --index .raglite
    python -m raglite search "你的问题" --index .raglite
    python -m raglite inspect --index .raglite
"""

# 包版本号：遵循语义化版本约定，使用方可通过 importlib.metadata / pkg_resources 读取
# 主版本号.次版本号.补丁号：每次不兼容改动递增主版本
__version__ = "0.1.0"
