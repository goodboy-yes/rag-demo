import json

import pytest

from raglite.documents import DocumentLoadError, load_documents


def test_loads_text_and_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("# 向量检索\n\n把文本转换成向量。", encoding="utf-8")
    (tmp_path / "b.txt").write_text("普通文本资料", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("ignored", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert {doc.metadata["file_type"] for doc in documents} == {"md", "txt"}


def test_loads_jsonl_text_and_metadata(tmp_path):
    path = tmp_path / "items.jsonl"
    rows = [
        {"text": "第一条知识", "title": "A"},
        {"content": "第二条知识", "score": 3},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    documents = load_documents(path)

    assert [doc.text for doc in documents] == ["第一条知识", "第二条知识"]
    assert documents[0].metadata["title"] == "A"
    assert documents[1].metadata["line"] == 2


def test_jsonl_requires_text_or_content(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps({"title": "missing text"}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="text/content"):
        load_documents(path)
