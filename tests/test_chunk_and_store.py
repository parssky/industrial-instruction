"""Offline chunking and vector-store tests using the hash embedder."""

import pytest

from industrial_instruction.chunk.chunker import chunk_documents, get_chunker
from industrial_instruction.config import ChunkConfig, Config, EmbedConfig, StoreConfig
from industrial_instruction.embed.registry import get_embedder
from industrial_instruction.schemas import Document

MARKDOWN = """# Pump maintenance

The pump requires inspection every 500 hours of operation.

## Lubrication

Use ISO VG 46 oil. Replace the oil every 2000 hours under continuous duty.

## Fault codes

| Code | Meaning |
| --- | --- |
| E01 | Phase imbalance |
| E02 | Overtemperature |
"""


def make_document() -> Document:
    return Document(
        id="doc1",
        source_path="pump.pdf",
        title="Pump manual",
        markdown=MARKDOWN,
    )


def make_chunks(min_chars: int = 20):
    return get_chunker(ChunkConfig(min_chars=min_chars)).split(make_document())


def test_heading_chunker_splits_on_headings():
    chunks = make_chunks()
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc1" for c in chunks)
    assert any("Lubrication" in " > ".join(c.heading_path) for c in chunks)
    # the breadcrumb is part of the text so retrieval sees it
    assert any(c.text.startswith("## Pump maintenance > Lubrication") for c in chunks)


def test_short_sections_are_merged_not_dropped():
    chunks = make_chunks(min_chars=10_000)
    text = "\n".join(c.text for c in chunks)
    for fact in ("500 hours", "ISO VG 46", "E02 | Overtemperature"):
        assert fact in text


@pytest.mark.parametrize("overlap", [0, 50])
def test_fixed_chunker_does_not_cut_words(overlap):
    words = [f"word{i}" for i in range(400)]
    doc = Document(id="d", source_path="x.pdf", markdown=" ".join(words))
    chunker = get_chunker(
        ChunkConfig(strategy="fixed", max_chars=200, overlap=overlap, min_chars=1)
    )
    chunks = chunker.split(doc)
    seen = set()
    for chunk in chunks:
        tokens = chunk.text.split()
        assert all(t in words for t in tokens), "a word was cut in half"
        seen.update(tokens)
    assert seen == set(words)


def test_chunk_documents_writes_jsonl(tmp_path):
    from industrial_instruction.chunk.chunker import load_chunks

    config = Config.from_dict({"paths": {"root": str(tmp_path)}})
    report = chunk_documents(config, documents=[make_document()])
    assert report.stage == "chunk"
    assert report.outputs == len(load_chunks(config.paths.resolve("chunks"))) >= 1


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = get_embedder(EmbedConfig(backend="hash", dimension=64))
    a = embedder.encode_documents(["phase imbalance fault"])
    b = embedder.encode_documents(["phase imbalance fault"])
    assert a.shape == (1, 64)
    assert (a == b).all()
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5


def test_faiss_roundtrip_and_retrieval(tmp_path):
    pytest.importorskip("faiss", reason="faiss-cpu not installed")
    from industrial_instruction.store.faiss_store import FaissStore

    embed_config = EmbedConfig(backend="hash", dimension=128)
    chunks = make_chunks()

    store = FaissStore(
        embedder=get_embedder(embed_config),
        store_config=StoreConfig(),
        directory=tmp_path / "index",
    )
    assert store.add_chunks(chunks, show_progress=False) == len(chunks)
    store.save()

    reopened = FaissStore(
        embedder=get_embedder(embed_config),
        store_config=StoreConfig(),
        directory=tmp_path / "index",
    ).load()
    assert reopened.size == len(chunks)

    hits = reopened.search("which oil grade should be used", k=2)
    assert hits
    assert hits[0].chunk.doc_id == "doc1"


def test_store_rejects_dimension_mismatch(tmp_path):
    pytest.importorskip("faiss", reason="faiss-cpu not installed")
    from industrial_instruction.store.faiss_store import FaissStore

    chunks = make_chunks()
    store = FaissStore(
        embedder=get_embedder(EmbedConfig(backend="hash", dimension=128)),
        directory=tmp_path / "index",
    )
    store.add_chunks(chunks, show_progress=False)
    store.save()

    mismatched = FaissStore(
        embedder=get_embedder(EmbedConfig(backend="hash", dimension=64)),
        directory=tmp_path / "index",
    )
    with pytest.raises(RuntimeError):
        mismatched.load()
