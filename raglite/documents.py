from __future__ import annotations

# 标准库导入
import json
# dataclass 用于简洁地定义数据类；field 用于自定义字段默认值（元数据中可变默认值需要用 default_factory）
from dataclasses import dataclass, field
from pathlib import Path
# typing.Any 用于 metadata 这种任意类型；Iterable 用于在辅助函数中接受可迭代对象
from typing import Any, Iterable


# 当前支持的文档文件后缀，统一用小写存储以便与 path.suffix.lower() 直接比对
# .md  / .txt：单文档视为单个 Document；.jsonl：每行一个 JSON 记录，每条记录作为一个 Document
SUPPORTED_EXTENSIONS = {".md", ".txt", ".jsonl"}


class DocumentLoadError(ValueError):
    """文档加载相关错误的统一异常类型。

    继承自 ValueError 而不是自定义基类，方便调用方用 catch-all 方式捕获，
    同时也能让 IDE/类型检查器将其识别为值类错误而非系统错误。
    """


@dataclass(frozen=True)
class Document:
    """统一的文档数据类。

    Attributes:
        text: 文档正文（已解码为字符串的纯文本）。
        source: 文档来源路径字符串，用于在检索结果中定位原始文件。
        metadata: 与该文档关联的自由形式元数据（如文件类型、原始行号等），
                  切片阶段会原样拷贝到每个 Chunk 上，便于检索后过滤或展示。
    """
    text: str
    source: str
    # 使用 default_factory 而不是直接 = {} 是因为可变默认值在 dataclass 中需要工厂函数，
    # 否则多个实例会共享同一个字典引用（在 frozen 场景下虽然不可变，但 dataclass 仍要求按惯例书写）
    metadata: dict[str, Any] = field(default_factory=dict)


def load_documents(path: Path | str) -> list[Document]:
    """从文件或目录加载全部受支持格式的文档。

    Args:
        path: 单个文件路径，或包含若干文档的目录路径（递归扫描）。

    Returns:
        加载得到的 Document 列表；空文件或全部空白内容会被跳过。

    Raises:
        DocumentLoadError: 路径不存在、扩展名不受支持，或文件内容无法解析时抛出。
    """
    # 兼容 str 和 Path 两种入参，统一转为 Path 便于后续操作
    root = Path(path)
    if not root.exists():
        raise DocumentLoadError(f"Input path does not exist: {root}")

    # 收集所有受支持的文件路径（目录情况下递归扫描，单文件情况下只校验一次）
    files = _iter_supported_files(root)
    documents: list[Document] = []
    for file_path in files:
        # _load_file 可能返回多个 Document（jsonl 一行一个），用 extend 而不是 append
        documents.extend(_load_file(file_path))

    return documents


def _iter_supported_files(path: Path) -> list[Path]:
    """枚举某个路径下所有受支持的文件。

    - 若 path 是文件：检查后缀受支持后返回包含它自身的列表（长度为 1）。
    - 若 path 是目录：递归遍历（rglob），按文件名排序，保证处理顺序稳定（可重现构建索引）。

    Raises:
        DocumentLoadError: 当 path 是文件但后缀不受支持时立即抛出。
    """
    if path.is_file():
        # 单文件场景：必须明确受支持，否则报错（让用户尽早发现错误，而不是默默跳过）
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [path]

    # 目录场景：rglob("*") 递归查找所有条目；按文件名字典序排序以便稳定输出
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_file(path: Path) -> Iterable[Document]:
    """根据文件后缀分发到不同的加载器。

    .md / .txt 视为纯文本，整个文件作为一个 Document；
    .jsonl 则按行解析，每条非空记录作为一个 Document。

    Raises:
        DocumentLoadError: 后缀不支持（理论上 _iter_supported_files 已过滤，这里是防御性兜底）。
    """
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        # 强制 UTF-8 编码：现代语料基本都为 UTF-8，避免依赖系统 locale
        text = path.read_text(encoding="utf-8")
        # 完全空白的文件视为无用文档直接跳过（避免后续 chunk_text 失败或产生空 chunk）
        if not text.strip():
            return []
        # file_type 写入 metadata，方便检索后按文件类型过滤
        return [Document(text=text, source=str(path), metadata={"file_type": suffix.lstrip(".")})]

    if suffix == ".jsonl":
        # jsonl 单独处理：可能返回多个 Document
        return _load_jsonl(path)

    # 实际流程中此分支不会执行，_iter_supported_files 已校验过；保留作为防御
    raise DocumentLoadError(f"Unsupported file extension: {suffix}")


def _load_jsonl(path: Path) -> list[Document]:
    """解析 JSONL（每行一个 JSON 对象）文件。

    期望每行是一个 JSON object，必须包含 `text` 或 `content` 字段作为正文。
    其余字段会被原样拷贝到 metadata，便于携带诸如 id、tags、source_url 等附加信息。

    Raises:
        DocumentLoadError: 单行 JSON 解析失败，或正文字段缺失/非字符串时抛出，
                           并附带文件路径与行号以便用户定位问题。
    """
    documents: list[Document] = []
    # enumerate 从 1 开始，与编辑器和常见错误信息中的行号习惯一致
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        # 跳过空行（jsonl 中空行通常视为分隔符，不应报错）
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            # 抛出时附加 文件路径:行号 信息，便于用户在编辑器中跳转
            raise DocumentLoadError(f"{path}:{line_number} is not valid JSON: {exc.msg}") from exc

        # 必须是 JSON object（dict），数组或基本类型没有 metadata 字段的概念
        if not isinstance(payload, dict):
            raise DocumentLoadError(f"{path}:{line_number} must be a JSON object.")

        # 兼容两种正文字段命名：text 或 content（不同数据源常用不同字段名）
        text = payload.get("text", payload.get("content"))
        # 必须存在且为非空字符串，避免后续切片产生空块
        if not isinstance(text, str) or not text.strip():
            raise DocumentLoadError(f"{path}:{line_number} must contain a non-empty text/content field.")

        # 其余字段一律当作 metadata；过滤掉正文字段防止冗余
        metadata = {key: value for key, value in payload.items() if key not in {"text", "content"}}
        # 补充框架自有的元信息：file_type 与原始行号（行号在检索时可用于回溯到 jsonl 的具体记录）
        metadata.update({"file_type": "jsonl", "line": line_number})
        documents.append(Document(text=text, source=str(path), metadata=metadata))

    return documents
