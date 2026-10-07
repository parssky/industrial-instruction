"""Retrieval sources: project index, user-built FAISS (both formats), functions."""

import json
import textwrap

import numpy as np
import pytest

from industrial_instruction.config import Config
from industrial_instruction.schemas import Chunk
from industrial_instruction.store import RetrievalError, get_retriever, register_retriever, retriever_backend

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

PASSAGES = [
    "Use ISO VG 46 hydraulic oil and change it every 2000 hours.",
    "Fault E02 means the stator winding is above 140 C.",
    "The drive-end bearing is a 6309-2Z deep groove ball bearing.",
]
EMBED = {"backend": "hash", "dimension": 64,
         "query_prefix": "task: search result | query: ", "document_prefix": "title: none | text: "}


def config(tmp_path, **retrieval):
    return Config.from_dict({"paths": {"root": str(tmp_path)}, "embed": EMBED, "retrieval": retrieval})


def write_paper_format(directory, embed_cfg):
    """What the original vector_store_module.Retriever.save() wrote."""
    from industrial_instruction.embed.registry import get_embedder

    directory.mkdir(parents=True)
    plain = embed_cfg.model_copy(update={"query_prefix": "", "document_prefix": ""})
    vectors = get_embedder(plain).encode_documents(PASSAGES)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype="float32"))
    faiss.write_index(index, str(directory / "index"))
    (directory / "id_map.json").write_text(json.dumps({str(i): t for i, t in enumerate(PASSAGES)}))


# ----------------------------------------------------------- user-built FAISS


def test_paper_format_index_is_queried_without_prefixes(tmp_path):
    cfg = config(tmp_path, source="faiss", path="paper_index")
    write_paper_format(tmp_path / "paper_index", cfg.embed)
    retriever = get_retriever(cfg)
    assert retriever.embedder.config.query_prefix == ""  # matches how it was built
    hits = retriever.search("Use ISO VG 46 hydraulic oil and change it every 2000 hours.", k=2)
    assert hits[0].chunk.text == PASSAGES[0] and hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_index_file_and_mapping_can_be_given_separately(tmp_path):
    cfg = config(tmp_path, source="faiss", path="idx/index", mapping_path="idx/id_map.json")
    write_paper_format(tmp_path / "idx", cfg.embed)
    assert len(get_retriever(cfg).search("bearing", k=3)) == 3


def test_package_format_keeps_its_prefixes(tmp_path):
    from industrial_instruction.embed.registry import get_embedder
    from industrial_instruction.store.faiss_store import FaissStore

    cfg = config(tmp_path, source="faiss", path="mine")
    store = FaissStore(get_embedder(cfg.embed), directory=tmp_path / "mine")
    store.add_chunks([Chunk(id=str(i), doc_id="d", text=t) for i, t in enumerate(PASSAGES)], show_progress=False)
    store.save()
    retriever = get_retriever(cfg)
    assert retriever.embedder.config.query_prefix.startswith("task:")
    assert retriever.search("E02", k=1)[0].chunk.doc_id == "d"


def test_dict_entries_use_text_field(tmp_path):
    cfg = config(tmp_path, source="faiss", path="idx", text_field="content")
    write_paper_format(tmp_path / "idx", cfg.embed)
    (tmp_path / "idx" / "id_map.json").write_text(
        json.dumps({str(i): {"content": t, "page": i} for i, t in enumerate(PASSAGES)})
    )
    hit = get_retriever(cfg).search(PASSAGES[2], k=1)[0]
    assert hit.chunk.text == PASSAGES[2] and hit.chunk.meta["page"] == 2


def test_wrong_embedding_model_is_reported(tmp_path):
    cfg = config(tmp_path, source="faiss", path="idx")
    write_paper_format(tmp_path / "idx", cfg.embed)
    cfg = cfg.with_override("retrieval.embed", {"backend": "hash", "dimension": 32})
    with pytest.raises(RuntimeError, match="retrieval.embed"):
        get_retriever(cfg)


def test_hub_index_is_downloaded(tmp_path, monkeypatch):
    import huggingface_hub

    cfg = config(tmp_path, source="faiss", path="hf:Parssky/industrial-instruction-faiss")
    snapshot = tmp_path / "snapshot"
    write_paper_format(snapshot / "faiss_panasonic", cfg.embed)  # nested, like a repo folder
    calls = {}

    def fake_download(repo_id, repo_type):
        calls.update(repo_id=repo_id, repo_type=repo_type)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert get_retriever(cfg).search("oil", k=1)
    assert calls == {"repo_id": "Parssky/industrial-instruction-faiss", "repo_type": "dataset"}


def test_missing_project_index_explains_both_options(tmp_path):
    with pytest.raises(RetrievalError, match=r"ii index --corpus.*\n.*retrieval.source=faiss"):
        get_retriever(config(tmp_path))


# ---------------------------------------------------------------- functions


def test_registered_function_with_strings_dicts_and_chunks(tmp_path):
    def search(query, k):
        return ["plain text", {"text": "a dict", "score": 0.7, "source": "es"},
                Chunk(id="c", doc_id="manual", text="a chunk")][:k]

    register_retriever("mine", search)
    hits = get_retriever(config(tmp_path, source="function", backend="mine")).search("q", k=3)
    assert [h.chunk.text for h in hits] == ["plain text", "a dict", "a chunk"]
    assert hits[1].score == 0.7 and hits[1].chunk.meta["source"] == "es"
    assert [h.rank for h in hits] == [0, 1, 2]


def test_function_from_a_file_next_to_the_config(tmp_path):
    (tmp_path / "my_search.py").write_text(textwrap.dedent('''
        def run(query, k):
            return [f"{query} #{i}" for i in range(k)]
    '''))
    hits = get_retriever(config(tmp_path, source="function", backend="my_search.py:run")).search("oil", 2)
    assert [h.chunk.text for h in hits] == ["oil #0", "oil #1"]


def test_factory_retriever_gets_options(tmp_path):
    @retriever_backend("es-like", factory=True)
    def build(cfg):
        host = cfg.options["host"]
        return lambda query, k: [f"from {host}"]

    cfg = config(tmp_path, source="function", backend="es-like", options={"host": "es:9200"})
    assert get_retriever(cfg).search("q", 1)[0].chunk.text == "from es:9200"


# ------------------------------------------------------------ index --corpus


def test_index_corpus_from_passages_jsonl(tmp_path):
    from industrial_instruction.store.runner import index_corpus

    (tmp_path / "passages.jsonl").write_text(
        "\n".join(json.dumps({"id": f"p{i}", "body": t, "doc_id": "manual"}) for i, t in enumerate(PASSAGES))
    )
    cfg = config(tmp_path)
    (report,) = index_corpus(cfg, "passages.jsonl", text_field="body")
    assert report.outputs == 3
    hit = get_retriever(cfg).search(PASSAGES[1], k=1)[0]
    assert hit.chunk.id == "p1" and hit.chunk.doc_id == "manual"
    assert cfg.paths.resolve("chunks").exists()  # usable by seeds.source=self too


def test_index_corpus_from_markdown_directory(tmp_path):
    from industrial_instruction.store.runner import index_corpus

    corpus = tmp_path / "manuals"
    corpus.mkdir()
    (corpus / "pump.md").write_text(
        "# Pump manual\n\n## Lubrication\n\n" + PASSAGES[0] + " " * 5 + ("Detail. " * 40)
    )
    cfg = config(tmp_path)
    reports = index_corpus(cfg, corpus)
    assert [r.stage for r in reports] == ["extract", "chunk", "index"]
    assert get_retriever(cfg).search("hydraulic oil", k=1)[0].chunk.text.startswith("## Pump manual")


def test_cli_index_corpus(tmp_path, monkeypatch):
    from industrial_instruction.cli import main

    (tmp_path / "p.jsonl").write_text("\n".join(json.dumps({"text": t}) for t in PASSAGES))
    monkeypatch.chdir(tmp_path)
    assert main(["index", "--corpus", "p.jsonl", "--set", "embed.backend=hash",
                 "--set", "embed.dimension=64"]) == 0
    assert (tmp_path / "artifacts" / "faiss" / "index.faiss").exists()


# ------------------------------------------------------- used by generate/bench


def test_generation_engine_uses_the_configured_retriever(tmp_path):
    from industrial_instruction.generate.engine import GenerationEngine

    register_retriever("fixed", lambda q, k: ["Fault E02 means overtemperature."] * k)
    cfg = config(tmp_path, source="function", backend="fixed")
    engine = GenerationEngine(cfg, client=object())
    assert engine._ensure_store().search("x", 1)[0].chunk.text.startswith("Fault E02")


def test_bench_retrieved_context_with_a_user_retriever(tmp_path):
    from industrial_instruction.benchmark import BenchItem, Endpoint, register_suite, run_benchmarks

    from test_benchmark import FakeOpenAI

    register_retriever("kb", lambda q, k: ["E02 means overtemperature."][:k])
    register_suite("one", lambda c: [BenchItem(id="1", prompt="What is E02?\nA. overtemp\nB. phase", query="E02", gold=["A"])])
    fake = FakeOpenAI(lambda p: '{"answer": ["A"]}' if "E02 means overtemperature" in p else '{"answer": ["B"]}')
    cfg = config(tmp_path, source="function", backend="kb")
    summary = run_benchmarks(cfg, suites=["one"], contexts=["none", "retrieved"],
                             endpoint=Endpoint(cfg.benchmark.endpoint, client=fake))
    assert summary["results"]["one-retrieved"]["set_match"] == 1.0
    assert summary["results"]["one-none"]["set_match"] == 0.0
    assert summary["retrieval"] == {"source": "function", "backend": "kb", "k": 3}


def test_bench_fails_fast_without_an_index(tmp_path):
    from industrial_instruction.benchmark import BenchItem, Endpoint, register_suite, run_benchmarks

    from test_benchmark import FakeOpenAI

    register_suite("one", lambda c: [BenchItem(id="1", prompt="Q?", query="Q", gold=["A"])])
    fake = FakeOpenAI(lambda p: '{"answer": ["A"]}')
    cfg = config(tmp_path)
    with pytest.raises(RetrievalError):
        run_benchmarks(cfg, suites=["one"], contexts=["retrieved"],
                       endpoint=Endpoint(cfg.benchmark.endpoint, client=fake))
    assert fake.sent == []  # no requests spent before failing
