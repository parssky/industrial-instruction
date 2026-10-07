"""Pluggable OCR: the function contract, attaching backends, extractor wiring."""

import textwrap
import threading

import pytest

from industrial_instruction.config import Config, ExtractConfig, OCRConfig
from industrial_instruction.ocr import (
    OCRError,
    OCRPage,
    PageOCR,
    available_ocr_backends,
    clean_markdown,
    get_ocr,
    ocr_backend,
    register_ocr,
)

pymupdf = pytest.importorskip("pymupdf", reason="PyMuPDF not installed")

from industrial_instruction.extract.base import ExtractionError  # noqa: E402
from industrial_instruction.extract.pymupdf_extractor import (  # noqa: E402
    PyMuPDFExtractor,
)

TEXT_PAGE = (
    "Section 1 Lubrication. Use ISO VG 46 hydraulic oil and replace it every "
    "2000 operating hours. Check the level weekly through the sight glass."
)
OCR_MARKDOWN = "## Fault codes\n\n| Code | Meaning |\n| --- | --- |\n| E01 | Phase imbalance |"


def _scanned_page(doc):
    """A page that is only a picture of text - no text layer, like a scan."""
    src = pymupdf.open()
    tmp = src.new_page()
    tmp.insert_text((72, 72), "Fault codes: E01 phase imbalance", fontsize=12)
    pix = tmp.get_pixmap(dpi=72)
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)


def make_pdf(path, text_pages=1, scanned_pages=1):
    doc = pymupdf.open()
    for _ in range(text_pages):
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 300), TEXT_PAGE, fontsize=11)
    for _ in range(scanned_pages):
        _scanned_page(doc)
    doc.save(path)
    return path


class Recorder:
    """A user-style OCR function that remembers what it was asked."""

    def __init__(self, result=OCR_MARKDOWN, fail_on=()):
        self.pages = []
        self.result = result
        self.fail_on = set(fail_on)
        self.lock = threading.Lock()

    def __call__(self, page: OCRPage) -> str:
        with self.lock:
            self.pages.append(page)
        if page.page_number in self.fail_on:
            raise RuntimeError("model server returned 500")
        assert page.image.startswith(b"\x89PNG")
        return self.result


def extractor(tmp_path, fn, **ocr):
    ocr = {"mode": "auto", "backend": "recorder", "cache_dir": str(tmp_path / "cache"), **ocr}
    register_ocr("recorder", fn)
    ex = PyMuPDFExtractor(ExtractConfig(min_chars_per_doc=20, ocr=OCRConfig(**ocr)))
    ex.root = tmp_path
    return ex


# ------------------------------------------------------------- extractor


def test_auto_mode_ocrs_only_pages_without_text(tmp_path):
    fn = Recorder()
    doc = extractor(tmp_path, fn).extract(make_pdf(tmp_path / "m.pdf"))
    assert [p.page_number for p in fn.pages] == [2]
    assert "ISO VG 46" in doc.markdown  # text page came from the PDF itself
    assert "| E01 | Phase imbalance |" in doc.markdown  # scanned page from OCR
    assert doc.markdown.index("ISO VG 46") < doc.markdown.index("E01")
    assert doc.meta["ocr_pages"] == 1 and doc.meta["ocr_backend"] == "recorder"


def test_always_mode_ocrs_every_page(tmp_path):
    fn = Recorder()
    extractor(tmp_path, fn, mode="always").extract(make_pdf(tmp_path / "m.pdf"))
    assert sorted(p.page_number for p in fn.pages) == [1, 2]
    assert "ISO VG 46" in fn.pages[0].text_layer or "ISO VG 46" in fn.pages[1].text_layer


def test_off_mode_never_calls_backend_and_hints_at_ocr(tmp_path):
    fn = Recorder()
    ex = extractor(tmp_path, fn, mode="off")
    with pytest.raises(ExtractionError, match="extract.ocr.mode"):
        ex.extract(make_pdf(tmp_path / "scan.pdf", text_pages=0, scanned_pages=2))
    assert fn.pages == []


def test_failed_page_keeps_text_layer_and_doc_survives(tmp_path):
    fn = Recorder(fail_on={1})
    doc = extractor(tmp_path, fn, mode="always").extract(make_pdf(tmp_path / "m.pdf"))
    assert "ISO VG 46" in doc.markdown  # page 1 fell back to its text layer
    assert "| E01 |" in doc.markdown
    assert doc.meta["ocr_failed"] == 1 and doc.meta["ocr_pages"] == 1


def test_results_are_cached_by_page_image(tmp_path):
    pdf = make_pdf(tmp_path / "m.pdf")
    extractor(tmp_path, Recorder()).extract(pdf)
    second = Recorder()
    doc = extractor(tmp_path, second).extract(pdf)
    assert second.pages == []
    assert doc.meta["ocr_cached"] == 1 and "| E01 |" in doc.markdown


def test_options_reach_the_function(tmp_path):
    fn = Recorder()
    extractor(tmp_path, fn, options={"lang": "deu"}, dpi=100).extract(
        make_pdf(tmp_path / "m.pdf")
    )
    assert fn.pages[0].options == {"lang": "deu"}
    assert fn.pages[0].dpi == 100 and fn.pages[0].width > 0


def test_many_pages_are_batched_and_ordered(tmp_path):
    fn = Recorder()
    ex = extractor(tmp_path, fn, max_workers=2, cache=False)
    doc = ex.extract(make_pdf(tmp_path / "m.pdf", text_pages=1, scanned_pages=11))
    assert sorted(p.page_number for p in fn.pages) == list(range(2, 13))
    assert doc.meta["ocr_pages"] == 11


# ------------------------------------------------------------- attaching


def test_import_path_to_a_file_next_to_the_config(tmp_path):
    (tmp_path / "my_ocr.py").write_text(
        textwrap.dedent(
            """
            def run(page):
                return f"# OCR page {page.page_number}"
            """
        )
    )
    fn = get_ocr(OCRConfig(backend="my_ocr.py:run"), root=tmp_path)
    page = OCRPage(image=b"", page_number=3, source_path="x.pdf")
    assert fn(page) == "# OCR page 3"


def test_import_path_to_a_module():
    fn = get_ocr(OCRConfig(backend="industrial_instruction.ocr.base:clean_markdown"))
    assert fn is clean_markdown


def test_factory_backend_is_built_once_with_config(tmp_path):
    built = []

    @ocr_backend("with-setup", factory=True)
    def make(config):
        built.append(config.options["weights"])
        return lambda page: f"done with {config.options['weights']}"

    fn = get_ocr(OCRConfig(backend="with-setup", options={"weights": "/w"}))
    assert fn(OCRPage(image=b"", page_number=1, source_path="x")) == "done with /w"
    assert built == ["/w"]


def test_unknown_backend_lists_what_is_available():
    with pytest.raises(OCRError, match="openai"):
        get_ocr(OCRConfig(backend="nope"))
    assert {"openai", "tesseract"} <= set(available_ocr_backends())


def test_mode_is_validated_and_legacy_flag_maps_to_auto():
    with pytest.raises(ValueError):
        ExtractConfig(ocr=OCRConfig(mode="sometimes"))
    assert ExtractConfig(ocr_fallback=True).ocr.mode == "auto"
    # YAML 1.1: an unquoted `off` arrives as False
    assert Config.from_dict({"extract": {"ocr": {"mode": False}}}).extract.ocr.mode == "off"
    assert Config().with_override("extract.ocr.mode", "off").extract.ocr.mode == "off"
    config = Config().with_override("extract.ocr.backend", "pkg.mod:fn")
    assert config.extract.ocr.backend == "pkg.mod:fn"


def test_runner_accepts_a_function_directly(tmp_path):
    runner = PageOCR(OCRConfig(mode="always", cache=False), fn=lambda p: "```markdown\n# Hi\n```")
    out = runner.run([OCRPage(image=b"x", page_number=1, source_path="a.pdf")])
    assert out[1].markdown == "# Hi"


def test_clean_markdown_unwraps_fences():
    assert clean_markdown("```markdown\n# Title\n\ntext\n```") == "# Title\n\ntext"
    assert clean_markdown("  plain  ") == "plain"
    assert clean_markdown(None) == ""


# ---------------------------------------------------------- openai backend


def test_openai_backend_sends_image_and_prompt(monkeypatch):
    import openai

    calls = {}

    class FakeClient:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            calls["create"] = kwargs
            message = type("M", (), {"content": "```markdown\n# Page\n```"})
            choice = type("C", (), {"message": message})
            return type("R", (), {"choices": [choice]})

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    fn = get_ocr(
        OCRConfig(backend="openai", base_url="http://localhost:8000/v1", model="Qwen/VL")
    )
    out = fn(OCRPage(image=b"\x89PNG", page_number=1, source_path="a.pdf"))

    assert out == "# Page"
    assert calls["init"]["base_url"] == "http://localhost:8000/v1"
    assert calls["create"]["model"] == "Qwen/VL"
    content = calls["create"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Markdown" in content[1]["text"]


def test_openai_backend_empty_prompt_sends_image_only(monkeypatch):
    from industrial_instruction.ocr.openai_vision import OpenAIVisionOCR

    page = OCRPage(image=b"x", page_number=1, source_path="a.pdf")
    messages = OpenAIVisionOCR(OCRConfig(prompt="")).messages(page)
    assert [c["type"] for c in messages[0]["content"]] == ["image_url"]


# ------------------------------------------------------------- end to end


def test_extract_stage_uses_backend_file_relative_to_root(tmp_path):
    from industrial_instruction.extract.runner import extract_documents

    (tmp_path / "data" / "pdfs").mkdir(parents=True)
    make_pdf(tmp_path / "data" / "pdfs" / "scan.pdf", text_pages=0, scanned_pages=1)
    (tmp_path / "ocr.py").write_text(
        "def run(page):\n    return '# Scanned manual\\n\\n' + 'Replace the filter monthly. ' * 20\n"
    )
    config = Config.from_dict(
        {
            "paths": {"root": str(tmp_path)},
            "extract": {"ocr": {"mode": "auto", "backend": "ocr.py:run"}},
        }
    )
    report = extract_documents(config)
    assert report.outputs == 1 and report.details["ocr_pages"] == 1
    assert (tmp_path / "artifacts" / "ocr_cache").is_dir()


def test_bad_backend_stops_the_extract_stage(tmp_path):
    from industrial_instruction.extract.runner import extract_documents

    (tmp_path / "data" / "pdfs").mkdir(parents=True)
    make_pdf(tmp_path / "data" / "pdfs" / "scan.pdf")
    config = Config.from_dict(
        {
            "paths": {"root": str(tmp_path)},
            "extract": {"ocr": {"mode": "auto", "backend": "missing.py:run"}},
        }
    )
    with pytest.raises(OCRError, match="not found"):
        extract_documents(config)


def test_empty_result_is_not_a_failure(tmp_path):
    doc = extractor(tmp_path, Recorder(result=""), cache=False).extract(
        make_pdf(tmp_path / "m.pdf")
    )
    assert doc.meta["ocr_failed"] == 0 and doc.meta["ocr_empty"] == 1


def test_pdfplumber_backend_uses_the_same_hook(tmp_path):
    pytest.importorskip("pdfplumber", reason="pdfplumber not installed")
    from industrial_instruction.extract.pdfplumber_extractor import PdfplumberExtractor

    fn = Recorder()
    register_ocr("recorder", fn)
    ex = PdfplumberExtractor(
        ExtractConfig(
            min_chars_per_doc=20,
            ocr=OCRConfig(mode="auto", backend="recorder", cache=False),
        )
    )
    doc = ex.extract(make_pdf(tmp_path / "m.pdf"))
    assert [p.page_number for p in fn.pages] == [2]
    assert "| E01 |" in doc.markdown and "ISO VG 46" in doc.markdown
