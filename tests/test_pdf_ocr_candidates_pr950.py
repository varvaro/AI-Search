"""PR9.5.0 — Multi-PSM OCR candidate selection.

Unit tests cover the pure scorer. Integration tests drive extract_pdf()
through mocked _run_ocr_subprocess so tesseract/pdftoppm are never spawned
except in the STOP/timeout safety cases that reuse the existing worker path.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import ai_search
import ai_search_config
import pdf_ocr_candidates as ocr


# ---------------------------------------------------------------------------
# Pure scoring helpers
# ---------------------------------------------------------------------------

def _cand(mode, raw, indexable=None):
    text = raw if indexable is None else indexable
    return ocr.OCRCandidate(mode=mode, raw_text=raw, indexable_text=text)


NOISE = "FSS bz REN S BEE FPE"
QUALITY = "Rez retenční nádrží a revizní vstup"
ENGLISH = (
    "The contractor shall provide delivery notes for the concrete mix "
    "after each pour together with the fresh concrete test report."
)


def test_flag_default_is_off():
    assert ai_search_config.PDF_MULTI_PSM_OCR_ENABLED is False
    assert ai_search.PDF_MULTI_PSM_OCR_ENABLED is False


def test_a_quality_text_scores_higher_than_noise():
    quality = ocr.score_candidate(_cand("psm12", QUALITY))
    noise = ocr.score_candidate(_cand("psm6", NOISE))
    assert quality.value > noise.value
    assert quality.word_tokens >= noise.word_tokens
    assert quality.diacritic_chars > noise.diacritic_chars


def test_b_more_raw_but_less_indexable_loses():
    padded_raw = ("glyph soup " * 80) + "xxxx"
    long_raw_short_index = _cand("psm3", padded_raw, indexable="short soup xxxx")
    shorter_raw_long_index = _cand(
        "psm12",
        "Rez retenční nádrží\n" * 8,
        indexable=("Rez retenční nádrží a revizní vstup. " * 12).strip(),
    )
    assert len(long_raw_short_index.raw_text) > len(shorter_raw_long_index.raw_text)
    assert len(long_raw_short_index.indexable_text) < len(shorter_raw_long_index.indexable_text)
    assert ocr.choose_best([long_raw_short_index, shorter_raw_long_index]) is shorter_raw_long_index


def test_c_tie_keeps_earlier_candidate():
    a = _cand("psm6", "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu")
    b = _cand("psm12", a.raw_text)
    assert ocr.score_candidate(a).value == ocr.score_candidate(b).value
    assert ocr.choose_best([a, b]) is a


def test_d_scoring_is_deterministic():
    candidate = _cand("psm12", QUALITY)
    first = ocr.score_candidate(candidate)
    second = ocr.score_candidate(candidate)
    assert first == second
    assert ocr.choose_best([candidate, _cand("psm3", NOISE)]).mode == "psm12"
    assert ocr.choose_best([candidate, _cand("psm3", NOISE)]).mode == "psm12"


def test_e_english_without_diacritics_is_not_weak():
    candidate = _cand("psm6", ENGLISH)
    score = ocr.score_candidate(candidate)
    assert score.diacritic_chars == 0
    assert ocr.is_weak(candidate) is False


def test_f_empty_candidate_is_weak_and_lowest():
    empty = _cand("psm6", "")
    score = ocr.score_candidate(empty)
    assert score.word_tokens == 0
    assert score.useful_chars == 0
    assert score.noise_ratio == 1.0
    assert ocr.is_weak(empty) is True
    assert ocr.choose_best([empty, _cand("psm12", QUALITY)]).mode == "psm12"


def test_g_high_noise_unicode_is_weak():
    symbols = "※▲※ ||| ~~~ ░▒▓ ★☆✦ ¤¶§ " * 8
    candidate = _cand("psm6", symbols)
    assert ocr.is_weak(candidate) is True
    assert ocr.score_candidate(candidate).noise_ratio >= ocr.WEAK_MAX_NOISE_RATIO


def test_choose_best_rejects_empty_sequence():
    with pytest.raises(ValueError):
        ocr.choose_best([])


def test_noise_is_weak_quality_short_text_may_be_weak_but_wins_score():
    assert ocr.is_weak(_cand("psm6", NOISE)) is True
    assert ocr.score_candidate(_cand("psm12", QUALITY)).value > ocr.score_candidate(_cand("psm6", NOISE)).value


# ---------------------------------------------------------------------------
# Integration: extract_pdf via mocked OCR subprocess
# ---------------------------------------------------------------------------

@pytest.fixture
def portable_private_tmp(monkeypatch, tmp_path):
    if Path("/private/tmp").is_dir():
        return
    original = ai_search.tempfile.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        if kwargs.get("dir") == "/private/tmp":
            kwargs["dir"] = str(tmp_path)
        return original(*args, **kwargs)

    monkeypatch.setattr(ai_search.tempfile, "TemporaryDirectory", temporary_directory)


@pytest.fixture
def fake_ocr_tools(monkeypatch):
    monkeypatch.setattr(ai_search, "resolve_system_tool", lambda name: f"/tools/{name}")


def _psm(cmd):
    if "--psm" in cmd:
        return cmd[cmd.index("--psm") + 1]
    return None


def _image_only_pdf_setup(tmp_path, monkeypatch, page_count=1):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(ai_search, "_pdf_page_count", lambda path, timeout=10: page_count)
    monkeypatch.setattr(ai_search.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    return pdf


def _render_ok(cmd):
    prefix = Path(cmd[-1])
    (prefix.parent / (prefix.name + "-01.png")).write_bytes(b"x")
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_flag_off_uses_only_psm6(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", False)
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            return subprocess.CompletedProcess(cmd, 0, NOISE, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert text == NOISE
    assert psms == ["6"]


def test_flag_on_strong_psm6_skips_fallbacks(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            return subprocess.CompletedProcess(cmd, 0, ENGLISH, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert text == ENGLISH
    assert psms == ["6"]


def test_flag_on_weak_psm6_runs_psm12(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            if _psm(cmd) == "6":
                return subprocess.CompletedProcess(cmd, 0, NOISE, "")
            return subprocess.CompletedProcess(cmd, 0, QUALITY, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    ai_search.extract_pdf(pdf)
    assert psms[0] == "6"
    assert "12" in psms


def test_psm12_wins_when_indexable_is_best(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    # Consecutive ALL-CAPS headings plus a one-char body: chunks() keeps
    # only the last heading + "x". Raw is long; indexable is tiny.
    psm3_raw = "\n".join(["NADPIS JEDEN DVA"] * 20 + ["x"])
    by_psm = {"6": NOISE, "12": QUALITY + "\nrevizní vstup a maximální hladina vody", "3": psm3_raw}

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, by_psm[_psm(cmd)], "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert text == by_psm["12"]
    assert "retenční" in text


def test_psm3_runs_when_psm12_still_weak_and_budget_allows(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            return subprocess.CompletedProcess(cmd, 0, NOISE if _psm(cmd) != "3" else ENGLISH, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert psms == ["6", "12", "3"]
    assert text == ENGLISH


def test_budget_blocks_psm3_keeps_best_so_far(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    clock = [1_000_000.0]
    monkeypatch.setattr(ai_search.time, "monotonic", lambda: clock[0])
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            clock[0] += 1.0
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            clock[0] += ai_search.PDF_OCR_SECONDS_PER_PAGE_BUDGET
            payload = NOISE if _psm(cmd) == "6" else QUALITY
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf, budget_seconds=ai_search.PDF_OCR_SECONDS_PER_PAGE_BUDGET * 2 + 2)
    assert "3" not in psms
    assert "12" in psms
    assert "retenční" in text


def test_multipage_never_uses_multi_psm(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch, page_count=3)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            return subprocess.CompletedProcess(cmd, 0, NOISE, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert psms == ["6", "6", "6"]
    assert text.count(NOISE) == 3


def test_whole_document_single_image_uses_same_multi_psm(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(ai_search, "_pdf_page_count", lambda path, timeout=10: None)
    monkeypatch.setattr(ai_search.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            return subprocess.CompletedProcess(cmd, 0, NOISE if _psm(cmd) == "6" else QUALITY, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert psms[0] == "6"
    assert "12" in psms
    assert "retenční" in text


def test_extra_psm_timeout_keeps_primary(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    psms = []

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            psms.append(_psm(cmd))
            if _psm(cmd) == "6":
                return subprocess.CompletedProcess(cmd, 0, NOISE, "")
            raise subprocess.TimeoutExpired(cmd, timeout)
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    text = ai_search.extract_pdf(pdf)
    assert text == NOISE
    assert "[OCR SELHALA" not in text
    assert psms == ["6", "12"]


def test_extra_psm_uses_existing_subprocess_wrapper(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    wrappers = []

    def fake(cmd, timeout):
        wrappers.append(Path(cmd[0]).name)
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            return _render_ok(cmd)
        if name == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, NOISE if _psm(cmd) == "6" else QUALITY, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    ai_search.extract_pdf(pdf)
    assert wrappers.count("tesseract") >= 2
    assert wrappers.count("pdftoppm") == 1


def test_render_happens_once_for_multi_psm(tmp_path, monkeypatch, portable_private_tmp, fake_ocr_tools):
    pdf = _image_only_pdf_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(ai_search, "PDF_MULTI_PSM_OCR_ENABLED", True)
    renders = {"n": 0}

    def fake(cmd, timeout):
        name = Path(cmd[0]).name
        if name == "pdftoppm":
            renders["n"] += 1
            return _render_ok(cmd)
        if name == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, NOISE if _psm(cmd) == "6" else QUALITY, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(ai_search, "_run_ocr_subprocess", fake)
    ai_search.extract_pdf(pdf)
    assert renders["n"] == 1


def test_stop_during_multi_psm_kills_active_subprocess(tmp_path, monkeypatch):
    """SIGTERM handler still sees the live Popen published by _run_ocr_subprocess."""
    published = []

    class FakeProc:
        def __init__(self):
            self._alive = True
        def poll(self):
            return None if self._alive else 0
        def kill(self):
            self._alive = False
            published.append("killed")

    proc = FakeProc()
    ai_search._active_ocr_subprocess[0] = proc
    try:
        import parsing_worker
        with pytest.raises(SystemExit) as exc:
            parsing_worker._handle_sigterm(15, None)
        assert exc.value.code == 143
        assert published == ["killed"]
        assert proc.poll() == 0
    finally:
        ai_search._active_ocr_subprocess[0] = None
