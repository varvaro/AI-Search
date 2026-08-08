"""Regression tests for the per-page PDF OCR architecture.

Root cause under test (see production audit): the old extract_pdf() rendered
an entire scanned PDF with a single pdftoppm call and OCR'd pages with
per-subprocess timeouts derived from a shared document budget
(min(60,remaining)/min(30,remaining)). Measured on real production files:
  - a 43-page, 17.3 MB scan needs ~77.8s to render ALL pages in one call -
    deterministically above the old 60s cap, even though each page alone
    renders in 0.4-2.1s.
  - a single large-format technical drawing (~A0) renders fine (11-13s) but
    its huge rendered image needs 36-41s to OCR - above the old 30s cap.

Fix: render/OCR one page at a time, each bounded by its own timeout, with a
document-level budget derived from measured per-page cost. These tests use a
mix of:
  - REAL pdftoppm/tesseract subprocesses against small synthetic image-only
    PDFs (native tools must be installed - same requirement the production
    code already has), for genuine end-to-end coverage of the page-range path
  - monkeypatched ai_search._run_ocr_subprocess for deterministic, fast tests
    of timeout/partial-failure/large-document-budget behavior that would
    otherwise need real minutes of wall-clock time
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

import ai_search

REPORTLAB = pytest.importorskip("reportlab.pdfgen.canvas")
from reportlab.pdfgen import canvas as _canvas  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
PIL_IMAGE = pytest.importorskip("PIL.Image")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def _font(size):
    try: return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception: return ImageFont.load_default()


def _make_ocr_page_image(text, path, size=(1700, 2200)):
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((100, 100), text, fill="black", font=_font(60))
    img.save(path)


def _make_image_only_pdf(tmp_path, texts, name="scan.pdf"):
    """Multi-page PDF with zero extractable text layer - each page is a
    rasterized image, forcing extract_pdf() into the OCR branch, same as a
    real scanned document."""
    pdf_path = tmp_path / name
    c = _canvas.Canvas(str(pdf_path), pagesize=A4)
    for index, text in enumerate(texts):
        image_path = tmp_path / f"_page_src_{index}.png"
        _make_ocr_page_image(text, image_path)
        c.drawImage(str(image_path), 0, 0, width=A4[0], height=A4[1])
        c.showPage()
    c.save()
    return pdf_path


def _make_native_text_pdf(tmp_path, lines, name="native.pdf"):
    pdf_path = tmp_path / name
    c = _canvas.Canvas(str(pdf_path), pagesize=A4)
    y = 700
    for line in lines:
        c.drawString(72, y, line); y -= 20
    c.save()
    return pdf_path


def _no_zombie_ocr_processes(deadline_seconds=5):
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        found = subprocess.run(["pgrep", "-f", "pdftoppm|/opt/homebrew/bin/tesseract"], capture_output=True, text=True)
        if not found.stdout.strip(): return True
        time.sleep(0.2)
    return False


def _leftover_ocr_tempdirs():
    return {p for p in Path("/private/tmp").glob("ai-search-ocr-*")}


# ---------------------------------------------------------------------------
# A) native text layer -> no OCR at all
# ---------------------------------------------------------------------------
def test_a_native_text_pdf_skips_ocr_entirely(tmp_path, monkeypatch):
    pdf = _make_native_text_pdf(tmp_path, ["NATIVNI TEXTOVA VRSTVA PRVNI RADEK", "DRUHY RADEK DOPLNUJE DELKU NAD OSMDESAT ZNAKU CELKEM"])
    def fail_if_called(cmd, timeout): raise AssertionError("OCR subprocess must not run for a native-text PDF: "+str(cmd))
    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fail_if_called)
    text = ai_search.extract_pdf(pdf)
    assert "NATIVNI TEXTOVA VRSTVA" in text


# ---------------------------------------------------------------------------
# B) small image-only PDF -> real per-page OCR
# ---------------------------------------------------------------------------
def test_b_small_image_only_pdf_uses_real_per_page_ocr(tmp_path):
    pdf = _make_image_only_pdf(tmp_path, ["JEDNA STRANKA OCR TEST"])
    text = ai_search.extract_pdf(pdf)
    assert "JEDNA" in text.upper() and "STRANKA" in text.upper()
    assert "[OCR SELHALA" not in text


# ---------------------------------------------------------------------------
# C) multi-page OCR PDF -> correct page order, real tesseract/pdftoppm
# ---------------------------------------------------------------------------
def test_c_multi_page_ocr_preserves_page_order_real_subprocesses(tmp_path):
    before = _leftover_ocr_tempdirs()
    pdf = _make_image_only_pdf(tmp_path, ["PRVNI STRANKA ALFA", "DRUHA STRANKA BETA", "TRETI STRANKA GAMA"])
    text = ai_search.extract_pdf(pdf)
    upper = text.upper()
    pos_alfa, pos_beta, pos_gama = upper.find("ALFA"), upper.find("BETA"), upper.find("GAMA")
    assert pos_alfa != -1 and pos_beta != -1 and pos_gama != -1
    assert pos_alfa < pos_beta < pos_gama, f"page order not preserved: {text!r}"
    assert "[OCR SELHALA" not in text
    # Compared against the pre-test snapshot, not an absolute empty set: a
    # production machine's /private/tmp can carry unrelated leftovers from
    # before this fix (see report) that are out of scope to clean up here.
    assert _leftover_ocr_tempdirs() == before


# ---------------------------------------------------------------------------
# D) page render timeout skips only that page, others still succeed
# ---------------------------------------------------------------------------
def test_d_one_page_render_timeout_skips_only_that_page(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"; pdf.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(ai_search, "_pdf_page_count", lambda path, timeout=10: 3)
    monkeypatch.setattr(ai_search.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            page = int(cmd[cmd.index("-f") + 1])
            if page == 2: raise subprocess.TimeoutExpired(cmd, timeout)
            prefix = Path(cmd[-1]); (prefix.parent / (prefix.name + "-01.png")).write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if name == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, "text uspesne strany", "")
        raise AssertionError("unexpected command: "+str(cmd))
    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert text.count("[OCR SELHALA") == 1
    assert "render překročil" in text
    assert text.count("text uspesne strany") == 2


# ---------------------------------------------------------------------------
# E) Tesseract timeout of one page skips only that page
# ---------------------------------------------------------------------------
def test_e_one_page_tesseract_timeout_skips_only_that_page(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"; pdf.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(ai_search, "_pdf_page_count", lambda path, timeout=10: 3)
    monkeypatch.setattr(ai_search.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    # Fail exactly the second tesseract call (i.e. page 2 of 3).
    calls = {"n": 0}
    def fake2(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            prefix = Path(cmd[-1]); (prefix.parent / (prefix.name + "-01.png")).write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if name == "tesseract":
            calls["n"] += 1
            if calls["n"] == 2: raise subprocess.TimeoutExpired(cmd, timeout)
            return subprocess.CompletedProcess(cmd, 0, "ok text", "")
        raise AssertionError("unexpected command: "+str(cmd))
    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake2)
    text = ai_search.extract_pdf(pdf)
    assert text.count("[OCR SELHALA") == 1
    assert "OCR překročil" in text
    assert text.count("ok text") == 2


# ---------------------------------------------------------------------------
# F) temp files cleaned up after a fully successful extraction
# ---------------------------------------------------------------------------
def test_f_temp_files_cleaned_up_after_success(tmp_path):
    before = _leftover_ocr_tempdirs()
    pdf = _make_image_only_pdf(tmp_path, ["USPESNY TEST STRANKY"])
    ai_search.extract_pdf(pdf)
    assert _leftover_ocr_tempdirs() == before


# ---------------------------------------------------------------------------
# G) temp files cleaned up after the document-level budget is exceeded
# ---------------------------------------------------------------------------
def test_g_temp_files_cleaned_up_after_document_budget_exceeded(tmp_path, monkeypatch):
    before = _leftover_ocr_tempdirs()
    pdf = tmp_path / "scan.pdf"; pdf.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(ai_search, "_pdf_page_count", lambda path, timeout=10: 5)
    monkeypatch.setattr(ai_search.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    def fake(cmd, timeout):
        time.sleep(0.15)  # each render/OCR call "costs" real budget
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            prefix = Path(cmd[-1]); (prefix.parent / (prefix.name + "-01.png")).write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if name == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, "text", "")
        raise AssertionError(cmd)
    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    # Enough budget for ~1 full page (render+OCR = 2*0.15s) but not all 5.
    text = ai_search.extract_pdf(pdf, budget_seconds=0.35)
    assert "text" in text  # at least one page completed
    assert "[OCR SELHALA" in text and "časový limit dokumentu" in text  # remaining pages bailed out cleanly
    assert _leftover_ocr_tempdirs() == before


# ---------------------------------------------------------------------------
# H) temp files cleaned up after an unhandled exception (all pages fail)
# ---------------------------------------------------------------------------
def test_h_temp_files_cleaned_up_after_all_pages_fail(tmp_path, monkeypatch):
    before = _leftover_ocr_tempdirs()
    pdf = tmp_path / "scan.pdf"; pdf.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(ai_search, "_pdf_page_count", lambda path, timeout=10: 2)
    monkeypatch.setattr(ai_search.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm": return subprocess.CompletedProcess(cmd, 1, "", "corrupt page")
        raise AssertionError(cmd)
    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    with pytest.raises(RuntimeError, match="OCR selhalo na všech"):
        ai_search.extract_pdf(pdf)
    assert _leftover_ocr_tempdirs() == before


# ---------------------------------------------------------------------------
# I/J/K) STOP mid-OCR: ParseCancelled raised quickly, no orphan
#        pdftoppm/tesseract processes, no hang.
# ---------------------------------------------------------------------------
def test_ijk_stop_during_real_pdf_ocr_kills_child_with_no_orphans(tmp_path):
    pdf = _make_image_only_pdf(tmp_path, [f"STRANKA CISLO {i}" for i in range(8)])
    watchdog = ai_search.ParsingWatchdog()
    stop = threading.Event()
    outcome = {}
    def run():
        try: outcome["result"] = watchdog.parse(pdf, limit=120, stop_event=stop)
        except Exception as exc: outcome["error"] = exc
    thread = threading.Thread(target=run); thread.start()
    time.sleep(2.0)  # let the real spawned worker get into page render/OCR
    stop.set()
    thread.join(timeout=15)
    try:
        assert not thread.is_alive(), "STOP did not terminate the parse call promptly"
        assert isinstance(outcome.get("error"), ai_search.ParseCancelled), outcome
        assert _no_zombie_ocr_processes(), "orphan pdftoppm/tesseract process left running after STOP"
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# Document-level OCR budget policy (unit-level, no subprocesses)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("pages", "expected"), [
    (1, ai_search.PDF_OCR_MIN_DOCUMENT_BUDGET_SECONDS),
    (10, ai_search.PDF_OCR_MIN_DOCUMENT_BUDGET_SECONDS),  # 8*10=80 < floor 150
    (43, 8 * 43),
    (63, 8 * 63),
    (500, ai_search.PDF_OCR_MAX_DOCUMENT_BUDGET_SECONDS),  # 8*500 way above ceiling
    (None, ai_search.PDF_OCR_MIN_DOCUMENT_BUDGET_SECONDS),
])
def test_pdf_ocr_document_budget_seconds_bounds(pages, expected):
    assert ai_search.pdf_ocr_document_budget_seconds(pages) == expected


def test_document_budget_floor_exceeds_worst_case_single_page_allowance():
    # The whole point of the floor: a single page can legitimately need up to
    # PDF_PAGE_RENDER_TIMEOUT_SECONDS + PDF_PAGE_OCR_TIMEOUT_SECONDS; the
    # document budget must never be tighter than that or it could pre-empt a
    # single slow (but legitimate) page before its own timeout even fires.
    worst_case_single_page = ai_search.PDF_PAGE_RENDER_TIMEOUT_SECONDS + ai_search.PDF_PAGE_OCR_TIMEOUT_SECONDS
    assert ai_search.PDF_OCR_MIN_DOCUMENT_BUDGET_SECONDS > worst_case_single_page


def test_pdf_page_count_valid_and_invalid(tmp_path):
    pdf = _make_image_only_pdf(tmp_path, ["A", "B", "C"])
    assert ai_search._pdf_page_count(pdf) == 3
    broken = tmp_path / "broken.pdf"; broken.write_bytes(b"not a real pdf")
    assert ai_search._pdf_page_count(broken) is None


# ---------------------------------------------------------------------------
# Large-PDF integration test (deterministic/mocked): demonstrates that a
# document whose CUMULATIVE per-page time exceeds the OLD 60s single-subprocess
# cap still completes successfully, because per-page timeouts/budget are now
# independent of any single subprocess call. Uses a monkeypatched clock so the
# test itself runs in well under a second while still exercising the real
# extract_pdf()/_extract_pdf_per_page() deadline arithmetic and page loop.
# ---------------------------------------------------------------------------
def test_large_pdf_completes_beyond_old_60s_single_call_cap(tmp_path, monkeypatch):
    pdf = tmp_path / "big_scan.pdf"; pdf.write_bytes(b"%PDF-fake")
    page_count = 15
    per_page_seconds = 5.0  # 15 * 5s = 75s total > old hard-coded 60s pdftoppm cap
    assert page_count * per_page_seconds > 60

    monkeypatch.setattr(ai_search, "_pdf_page_count", lambda path, timeout=10: page_count)
    monkeypatch.setattr(ai_search.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())

    fake_clock = [1_000_000.0]
    monkeypatch.setattr(ai_search.time, "monotonic", lambda: fake_clock[0])

    def fake(cmd, timeout):
        fake_clock[0] += per_page_seconds / 2  # split across the render+OCR call for this page
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            prefix = Path(cmd[-1]); (prefix.parent / (prefix.name + "-01.png")).write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if name == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, "legitimate page text", "")
        raise AssertionError(cmd)
    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)

    text = ai_search.extract_pdf(pdf)
    assert text.count("[OCR SELHALA") == 0
    assert text.count("legitimate page text") == page_count
    total_simulated_elapsed = fake_clock[0] - 1_000_000.0
    assert total_simulated_elapsed > 60, "test did not actually simulate exceeding the old cap"
    assert total_simulated_elapsed <= ai_search.pdf_ocr_document_budget_seconds(page_count)
