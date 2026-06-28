import pytest

from raglite.chunking import chunk_text


def test_chunk_text_uses_overlap_for_long_text():
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert [chunk[0] for chunk in chunks] == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_chunk_text_prefers_sentence_boundary():
    text = "第一句很短。第二句也很短。第三句会进入下一个块。"

    chunks = chunk_text(text, chunk_size=13, overlap=2)

    assert chunks[0][0].endswith("。")
    assert chunks[1][0]


def test_chunk_options_require_overlap_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", chunk_size=10, overlap=10)
