import json

import numpy as np
import pytest

from raglite.store import IndexError, build_index, inspect_index, load_index, search_index


class FakeEmbedder:
    model_name = "fake-embedder"

    def embed(self, texts):
        vectors = []
        for text in texts:
            if "向量" in text or "检索" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "咖啡" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_build_and_search_index_with_fake_embedder(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("向量检索会把问题和文档放到同一个语义空间。", encoding="utf-8")
    (docs / "coffee.txt").write_text("咖啡豆经过研磨和萃取后会产生香气。", encoding="utf-8")

    index = tmp_path / ".raglite"
    manifest = build_index(docs, index_path=index, embedder=FakeEmbedder(), chunk_size=80, overlap=10)
    results = search_index("怎么做向量检索", index_path=index, embedder=FakeEmbedder())

    assert manifest.chunk_count == 2
    assert results[0].score == pytest.approx(1.0)
    assert "向量检索" in results[0].chunk.text


def test_inspect_and_load_index_validate_manifest(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("向量检索基础", encoding="utf-8")
    index = tmp_path / ".raglite"

    build_index(docs, index_path=index, embedder=FakeEmbedder(), chunk_size=80, overlap=10)
    manifest = inspect_index(index)

    assert manifest.model_name == "fake-embedder"
    assert manifest.dimension == 3
    assert len(load_index(index)[1]) == 1


def test_load_index_reports_manifest_mismatch(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rag.md").write_text("向量检索基础", encoding="utf-8")
    index = tmp_path / ".raglite"
    build_index(docs, index_path=index, embedder=FakeEmbedder(), chunk_size=80, overlap=10)

    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_count"] = 999
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(IndexError, match="Manifest says"):
        load_index(index)
