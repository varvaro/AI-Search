"""Regression tests for the legacy .xls extraction fix.

Root cause under test (production audit, 2026-08-07): the old .xls path fed
the ENTIRE OLE compound-file container through `textutil -convert txt`, which
has no concept of BIFF record structure - it dumped formatting/embedded
object/revision-remnant bytes as if they were prose. A 3MB OLE file whose
sheets held a few dozen real rows produced ~3MB of "text", which chunked into
dozens of huge nonsense blocks whose cumulative embedding time exceeded
EMBEDDING_TIMEOUT_SECONDS (ERROR_TIMEOUT). ai_search.extract_xls() replaces
that with a real BIFF parse (xlrd): only actual cell values are emitted, so
output size tracks real content, not container size.

Fixtures here are built with xlwt (test-only dependency, see requirements.txt
- it never runs in production code, only to synthesize .xls binaries for
these tests, since there is no dependency-free way to author real BIFF
bytes). Production-file validation (the actual previously-failing documents)
is done separately, read-only, outside the automated suite, since the
production corpus lives outside this repository.
"""
from __future__ import annotations

import datetime

import pytest
import xlwt

import ai_search


def _write_xls(tmp_path, sheets: dict, name="book.xls"):
    """sheets: {sheet_name: [[cell, cell, ...], ...]} using plain Python
    values (str/int/float/bool/datetime); an empty row list means an empty
    sheet (no cells written at all)."""
    wb = xlwt.Workbook(encoding="utf-8")
    date_style = xlwt.XFStyle(); date_style.num_format_str = "YYYY-MM-DD"
    for sheet_name, rows in sheets.items():
        ws = wb.add_sheet(sheet_name)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value is None: continue
                if isinstance(value, datetime.datetime): ws.write(r, c, value, date_style)
                else: ws.write(r, c, value)
    path = tmp_path / name
    wb.save(str(path))
    return path


# ---------------------------------------------------------------------------
# A) simple sheet with text
# ---------------------------------------------------------------------------
def test_a_simple_text_sheet(tmp_path):
    path = _write_xls(tmp_path, {"Přehled": [["Popis", "Stav"], ["Základová deska", "Hotovo"]]})
    text = ai_search.extract_xls(path)
    assert "[Sheet: Přehled]" in text
    assert "- Popis | Stav" in text
    assert "- Základová deska | Hotovo" in text


# ---------------------------------------------------------------------------
# B) numbers - integers stay clean, floats keep precision without FP noise
# ---------------------------------------------------------------------------
def test_b_numbers_are_formatted_cleanly(tmp_path):
    path = _write_xls(tmp_path, {"Čísla": [["Popis", "Množství", "Cena"], ["Beton", 24, 1580.5]]})
    text = ai_search.extract_xls(path)
    assert "- Beton | 24 | 1580.5" in text
    assert "24.0" not in text  # integers must not render with a trailing ".0"


# ---------------------------------------------------------------------------
# C) date
# ---------------------------------------------------------------------------
def test_c_date_is_rendered_as_iso_date(tmp_path):
    path = _write_xls(tmp_path, {"Termíny": [["Akce", "Datum"], ["Předání", datetime.datetime(2026, 3, 15)]]})
    text = ai_search.extract_xls(path)
    assert "- Předání | 2026-03-15" in text


# ---------------------------------------------------------------------------
# D) multiple sheets, each labeled and in order
# ---------------------------------------------------------------------------
def test_d_multiple_sheets_each_labeled(tmp_path):
    path = _write_xls(tmp_path, {
        "Sheet1": [["alpha obsah"]],
        "Sheet2": [["beta obsah"]],
    })
    text = ai_search.extract_xls(path)
    assert text.index("[Sheet: Sheet1]") < text.index("alpha obsah") < text.index("[Sheet: Sheet2]") < text.index("beta obsah")


# ---------------------------------------------------------------------------
# E) empty sheet is skipped entirely (no heading, no stray content)
# ---------------------------------------------------------------------------
def test_e_empty_sheet_is_skipped(tmp_path):
    path = _write_xls(tmp_path, {
        "Data": [["obsah"]],
        "Prazdny": [],
    })
    text = ai_search.extract_xls(path)
    assert "[Sheet: Data]" in text
    assert "Prazdny" not in text


# ---------------------------------------------------------------------------
# F) rows with empty cells: trailing empties trimmed, interior gaps kept
# ---------------------------------------------------------------------------
def test_f_trailing_empty_cells_trimmed_interior_gaps_kept(tmp_path):
    path = _write_xls(tmp_path, {"Sheet": [
        ["A", "", "C", "", "", ""],   # interior gap kept, trailing empties dropped
        ["Jen první buňka", "", "", ""],
    ]})
    text = ai_search.extract_xls(path)
    assert "- A |  | C" in text
    assert "A |  | C | " not in text  # no trailing " | " noise
    assert "Jen první buňka" in text
    lines = [l for l in text.split("\n") if l.strip()]
    assert not any(line.endswith("|") or line.endswith("| ") for line in lines)


# ---------------------------------------------------------------------------
# G) formula results: xlwt cannot write a cached formula result (a writer
# limitation, not a reader/extractor one - verified separately against real
# production files with real formulas), so this exercises _xls_cell_text()
# directly with a Cell shaped exactly like what xlrd hands back for a real
# formula cell (BIFF formula records store a cached result of ordinary
# NUMBER/TEXT/BOOL/ERROR type - xlrd surfaces only that cached value, with no
# separate "this came from a formula" ctype). If our formatting is correct
# for a NUMBER-typed cell, it is correct for a formula-derived one too.
# ---------------------------------------------------------------------------
def test_g_formula_result_cell_is_rendered_like_any_other_value():
    import xlrd
    from xlrd.sheet import Cell
    cell = Cell(xlrd.XL_CELL_NUMBER, 149630.01)
    assert ai_search._xls_cell_text(cell, 0) == "149630.01"


# ---------------------------------------------------------------------------
# H) Unicode / Czech diacritics survive
# ---------------------------------------------------------------------------
def test_h_czech_diacritics_are_preserved(tmp_path):
    path = _write_xls(tmp_path, {"Sheet": [["Základová deska", "Střešní plášť", "žlutá"]]})
    text = ai_search.extract_xls(path)
    assert "- Základová deska | Střešní plášť | žlutá" in text


# ---------------------------------------------------------------------------
# I) no control/binary garbage in output, and output stays proportional to
#    real content (the actual OLE/binary-garbage regression guard)
# ---------------------------------------------------------------------------
def test_i_no_control_characters_in_output(tmp_path):
    dirty = "text s\x00null\x01a\x02control\x1fchars"
    path = _write_xls(tmp_path, {"Sheet": [[dirty, "čistá buňka"]]})
    text = ai_search.extract_xls(path)
    assert all(ord(ch) >= 32 or ch in "\n" for ch in text)
    assert "\x00" not in text and "\x01" not in text and "\x1f" not in text
    assert "čistá buňka" in text


def test_i_many_trailing_empty_rows_do_not_bloat_output(tmp_path):
    rows = [["skutečný obsah řádku"]] + [[] for _ in range(500)]
    path = _write_xls(tmp_path, {"Sheet": rows})
    text = ai_search.extract_xls(path)
    # Real content is one short line; 500 trailing empty rows must not add
    # any output (this is exactly the "megabytes of garbage from a mostly-
    # empty container" failure mode this fix targets).
    assert text == "[Sheet: Sheet]\n- skutečný obsah řádku"


# ---------------------------------------------------------------------------
# Regression: rows whose first cell is a bare number/code ("1", "01", item
# codes) or an ALL-CAPS label must survive ai_search.chunks() intact. Found
# during forensic validation: chunks() treats a line starting with a bare
# number or an ALL-CAPS run as a section heading and folds it into the NEXT
# chunk's `heading` field instead of its body - without the leading "- "
# marker, an item-numbered row's own cell values (description, price, ...)
# would silently vanish from the indexed/searchable text. This exercises the
# REAL chunks() function, not a mock, since the bug is in their interaction.
# ---------------------------------------------------------------------------
def test_numbered_and_allcaps_rows_survive_real_chunking(tmp_path):
    path = _write_xls(tmp_path, {"ZL 001": [
        ["Kód", "Popis", "Cena"],
        [1, "Bourání energokanálu, směsný odpad", 73963.71],
        [2, "Zřízení a údržba záborů", 124565],
        ["HSV", "HSV práce celkem", 73963.71],
    ]})
    text = ai_search.extract_xls(path)
    pieces = ai_search.chunks(text)
    joined = "\n".join(heading + "\n" + body for heading, body in pieces)
    assert "Bourání energokanálu" in joined
    assert "73963.71" in joined
    assert "Zřízení a údržba záborů" in joined
    assert "124565" in joined
    assert "HSV práce celkem" in joined


# ---------------------------------------------------------------------------
# J) invalid .xls -> clear extraction error, never a textutil-style fallback
# ---------------------------------------------------------------------------
def test_j_invalid_xls_raises_clear_error_not_silent_garbage(tmp_path):
    path = tmp_path / "corrupt.xls"
    path.write_bytes(b"toto neni platny OLE/BIFF soubor" * 50)
    with pytest.raises(ValueError, match="OLE/BIFF"):
        ai_search.extract_xls(path)


def test_j_dispatch_never_falls_back_to_textutil_for_xls(tmp_path):
    path = tmp_path / "corrupt.xls"
    path.write_bytes(b"not a real xls")
    with pytest.raises(ValueError):
        ai_search.extract(path)  # must propagate the extraction error, not silently succeed via textutil


# ---------------------------------------------------------------------------
# K) .xlsx behavior is unchanged (different code path entirely, untouched)
# ---------------------------------------------------------------------------
def test_k_xlsx_extraction_unchanged(tmp_path):
    import zipfile
    path = tmp_path / "book.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", '<sst><t>GAMMA obsah tabulky Excel.</t></sst>')
    text, method = ai_search.extract(path)
    assert method == "office_xml"
    assert "GAMMA obsah tabulky Excel." in text


# ---------------------------------------------------------------------------
# End-to-end: real .xls now indexes successfully through sync() with
# meaningful content, and a corrupt one is recorded as an explicit error
# (not silently indexed as garbage) - isolated tmp_path index only.
# ---------------------------------------------------------------------------
class FakeEmbeddings:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_xls_indexes_successfully_through_sync_with_real_content(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    _write_xls(root, {"ZL 001": [["Kód", "Popis", "Cena"], ["01", "Bourání energokanálu", 73963.71]]}, name="zl.xls")
    db, lance = tmp_path / "state.sqlite3", tmp_path / "lance"
    result = ai_search.sync(root, db, lance, FakeEmbeddings())
    assert result["added"] == 1 and result["errors"] == 0
    with ai_search.database(db) as con:
        status = con.execute("SELECT status FROM index_status WHERE path LIKE '%zl.xls'").fetchone()[0]
        assert status == "NOVÝ"
        row = con.execute("SELECT extraction FROM documents WHERE path LIKE '%zl.xls'").fetchone()
        assert row[0] == "xls_biff"
        hit = con.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'energokanálu'").fetchone()[0]
        assert hit == 1


def test_corrupt_xls_is_recorded_as_error_not_indexed_as_garbage(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    (root / "bad.xls").write_bytes(b"not a real ole/biff container" * 20)
    db, lance = tmp_path / "state.sqlite3", tmp_path / "lance"
    result = ai_search.sync(root, db, lance, FakeEmbeddings())
    assert result["errors"] == 1
    with ai_search.database(db) as con:
        status, error = con.execute("SELECT status,error FROM index_status WHERE path LIKE '%bad.xls'").fetchone()
        assert status == "CHYBA"
        assert "OLE/BIFF" in error
        assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
