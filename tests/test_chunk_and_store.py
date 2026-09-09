"""Offline chunking and vector-store tests using the hash embedder."""

import pytest

from industrial_instruction.chunk.chunker import chunk_documents
from industrial_instruction.config import Config, EmbedConfig, StoreConfig
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


def test_heading_chunker_splits_on_headings():
    config = Config()
    chunks = chunk_documents([make_document()], config.chunk)
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc1" for c in chunks)
    assert any("Lubrication" in " > ".join(c.heading_path) for c in chunks)


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = get_embedder(EmbedConfig(backend="hash", dimension=64))
    a = embedder.encode_documents(["phase imbalance fault"])
    b = embedder.encode_documents(["phase imbalance fault"])
    assert a.shape == (1, 64)
    assert (a == b).all()
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5


def test_faiss_roundtrip_and_retrieval(tmp_path):
    faiss_store = pytest.importorskip(
        "faiss", reason="faiss-cpu not installed"
    ) and None
    from industrial_instruction.store.faiss_store import FaissStore

    embed_config = EmbedConfig(backend="hash", dimension=128)
    chunks = chunk_documents([make_document()], Config().chunk)

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

    chunks = chunk_documents([make_document()], Config().chunk)
    FaissStore(
        embedder=get_embedder(EmbedConfig(backend="hash", dimension=128)),
        directory=tmp_path / "index",
    ).add_chunks(chunks, show_progress=False)

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
