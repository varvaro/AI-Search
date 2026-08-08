"""Regression tests for the .xlsx extraction fix (Fáze 3.1).

Root cause under test (production audit, 2026-08-07): .xlsx sheets went
through clean_xml_text(), which concatenates every <t>/<v> node with spaces.
A text cell in a sheet part is `<c t="s"><v>4974</v></c>` - the text itself
lives in xl/sharedStrings.xml and the cell only stores an index into it. The
extract of a real 455k-char production workbook therefore began
"4974 4975 52 4976 4977 187 ..." : a flat stream of shared-string indices,
with the actual text dumped separately in string-table order and no row,
column or sheet association at all. It had 48 newlines in 455k characters, so
chunks() saw one enormous block per sheet and the only remaining boundary was
CHUNK_MAX_SIZE, which cuts mid-row.

Fixtures are hand-built OOXML zips rather than a writer library: the bug is in
how the XML is read, so the tests must control the exact XML shape (shared vs
inline strings, sheet part numbering, row-number gaps).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import ai_search
import document_extractors as de

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _write_xlsx(tmp_path: Path, sheets: dict, name: str = "book.xlsx", first_part: int = 4) -> Path:
    """sheets: {sheet_name: [row, ...]} where a row is a list of str/int/float
    cells (None = empty cell) and an empty list means a blank source row.

    Strings go through the shared string table, like real Excel. Sheet parts
    are numbered from `first_part` (default 4, as in the production workbook
    that exposed this) so that resolving them positionally instead of through
    the relationship table would produce the wrong sheet.
    """
    shared: list[str] = []
    index: dict[str, int] = {}

    def share(value: str) -> int:
        if value not in index:
            index[value] = len(shared)
            shared.append(value)
        return index[value]

    def column_ref(number: int) -> str:
        letters = ""
        while number > 0:
            number, remainder = divmod(number - 1, 26)
            letters = chr(65 + remainder) + letters
        return letters

    parts: dict[str, str] = {}
    entries: list[tuple[str, str, str]] = []  # (rId, sheet name, part)
    for offset, (sheet_name, rows) in enumerate(sheets.items()):
        part = f"xl/worksheets/sheet{first_part + offset}.xml"
        xml = [f'<worksheet xmlns="{MAIN}"><sheetData>']
        row_number = 0
        for row in rows:
            row_number += 1
            if not row:
                continue  # Excel omits an empty row entirely, leaving a gap in r=""
            cells = []
            for position, value in enumerate(row, start=1):
                if value is None or value == "":
                    continue
                reference = f"{column_ref(position)}{row_number}"
                if isinstance(value, str):
                    cells.append(f'<c r="{reference}" t="s"><v>{share(value)}</v></c>')
                else:
                    cells.append(f'<c r="{reference}"><v>{value}</v></c>')
            xml.append(f'<row r="{row_number}">' + "".join(cells) + "</row>")
        xml.append("</sheetData></worksheet>")
        parts[part] = "".join(xml)
        entries.append((f"rId{offset + 1}", sheet_name, part))

    workbook = (f'<workbook xmlns="{MAIN}" xmlns:r="{RELS}"><sheets>'
                + "".join(f'<sheet name="{n}" sheetId="{i + 1}" r:id="{rid}"/>'
                          for i, (rid, n, _) in enumerate(entries))
                + "</sheets></workbook>")
    relationships = (f'<Relationships xmlns="{PKG_RELS}">'
                     + "".join(f'<Relationship Id="{rid}" Target="{part[3:]}" '
                               f'Type="{RELS}/worksheet"/>' for rid, _, part in entries)
                     + "</Relationships>")
    shared_xml = (f'<sst xmlns="{MAIN}" count="{len(shared)}">'
                  + "".join(f"<si><t>{value}</t></si>" for value in shared) + "</sst>")

    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", shared_xml)
        for part, xml in parts.items():
            archive.writestr(part, xml)
    return path


def _rows_of(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.startswith("- ")]


# ---------------------------------------------------------------------------
# The actual bug: a text cell must extract as its text, never as its
# shared-string index.
# ---------------------------------------------------------------------------
def test_shared_strings_are_resolved_not_emitted_as_indices(tmp_path):
    path = _write_xlsx(tmp_path, {"Rozpočet": [
        ["Kód", "Popis", "Cena"],
        ["23FBC202", "Bourání energokanálu", 73963.71],
    ]})
    text = de.extract_office(path)
    assert "- Kód | Popis | Cena" in text
    assert "- 23FBC202 | Bourání energokanálu | 73963.71" in text
    # The pre-fix output was a stream of table indices; none of these rows may
    # render as bare "0 1 2 ..." positions.
    assert "- 0 | 1 | 2" not in text


def test_sheet_parts_are_resolved_through_the_relationship_table(tmp_path):
    # Part files are sheet4/sheet5 while tab order is Souhrn, Detail: reading
    # them positionally (or by sorted part name) would mislabel the sheets.
    path = _write_xlsx(tmp_path, {"Souhrn": [["souhrnná data"]], "Detail": [["detailní data"]]})
    text = de.extract_office(path)
    assert text.index("[Sheet: Souhrn]") < text.index("souhrnná data")
    assert text.index("[Sheet: Detail]") < text.index("detailní data")


# ---------------------------------------------------------------------------
# A) more sheets: each separated, rows not glued into one long line
# ---------------------------------------------------------------------------
def test_a_each_sheet_is_separated(tmp_path):
    path = _write_xlsx(tmp_path, {
        "List1": [["alfa", 1], ["beta", 2]],
        "List2": [["gama", 3]],
    })
    text = de.extract_office(path)
    assert "[Sheet: List1]" in text and "[Sheet: List2]" in text
    assert text.index("[Sheet: List1]") < text.index("alfa") < text.index("[Sheet: List2]") < text.index("gama")
    # A blank line between sheets is what makes chunks() start a new chunk.
    assert "\n\n[Sheet: List2]" in text


def test_a_rows_are_not_glued_into_one_long_text(tmp_path):
    path = _write_xlsx(tmp_path, {"List1": [["a", 1], ["b", 2], ["c", 3]]})
    text = de.extract_office(path)
    assert _rows_of(text) == ["- a | 1", "- b | 2", "- c | 3"]


def test_a_every_sheet_reaches_its_own_chunk(tmp_path):
    path = _write_xlsx(tmp_path, {
        "Rekapitulace": [["celkem", 1000]],
        "Položky": [["beton", 500]],
        "Přílohy": [["výkres", 3]],
    })
    pieces = ai_search.chunks(de.extract_office(path))
    bodies = [body for _, body in pieces]
    assert len(bodies) == 3
    assert any("Rekapitulace" in b and "celkem" in b for b in bodies)
    assert any("Položky" in b and "beton" in b for b in bodies)
    assert any("Přílohy" in b and "výkres" in b for b in bodies)


# ---------------------------------------------------------------------------
# B) table: every value survives, in order
# ---------------------------------------------------------------------------
def test_b_all_cell_values_are_preserved(tmp_path):
    table = [
        ["Poř.", "Položka", "MJ", "Množství", "Cena"],
        [1, "Základová deska", "m3", 24, 1580.5],
        [2, "Střešní plášť", "m2", 310, 899],
        [3, "Zábor staveniště", "ks", 1, 124565],
    ]
    path = _write_xlsx(tmp_path, {"Soupis": table})
    text = de.extract_office(path)
    for row in table:
        for value in row:
            assert str(value) in text, f"chybí hodnota {value!r}"
    assert _rows_of(text) == [
        "- Poř. | Položka | MJ | Množství | Cena",
        "- 1 | Základová deska | m3 | 24 | 1580.5",
        "- 2 | Střešní plášť | m2 | 310 | 899",
        "- 3 | Zábor staveniště | ks | 1 | 124565",
    ]


def test_b_row_and_column_order_is_preserved(tmp_path):
    path = _write_xlsx(tmp_path, {"S": [["r1c1", "r1c2"], ["r2c1", "r2c2"]]})
    text = de.extract_office(path)
    assert _rows_of(text) == ["- r1c1 | r1c2", "- r2c1 | r2c2"]


def test_b_interior_column_gaps_are_preserved(tmp_path):
    path = _write_xlsx(tmp_path, {"S": [["A", None, "C", None, None]]})
    text = de.extract_office(path)
    assert "- A |  | C" in text
    assert not any(line.rstrip().endswith("|") for line in _rows_of(text))


def test_b_numbers_keep_xls_formatting_rules(tmp_path):
    # Raw <v> often carries floating-point noise ("5.2784999999999993"); the
    # .xls path already normalises this and .xlsx must match it.
    path = _write_xlsx(tmp_path, {"S": [["Beton", 24, 5.2784999999999993]]})
    text = de.extract_office(path)
    assert "- Beton | 24 | 5.2785" in text
    assert "24.0" not in text


def test_b_a_cell_containing_a_newline_stays_on_one_row(tmp_path):
    path = _write_xlsx(tmp_path, {"S": [["první\nřádek buňky", "vedle"], ["druhý řádek", "x"]]})
    text = de.extract_office(path)
    assert _rows_of(text) == ["- první řádek buňky | vedle", "- druhý řádek | x"]


def test_b_inline_strings_are_read(tmp_path):
    path = tmp_path / "inline.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml",
                         f'<worksheet xmlns="{MAIN}"><sheetData><row r="1">'
                         '<c r="A1" t="inlineStr"><is><t>vložený text</t></is></c>'
                         '<c r="B1"><v>7</v></c></row></sheetData></worksheet>')
    assert "- vložený text | 7" in de.extract_office(path)


def test_b_boolean_cells_are_readable(tmp_path):
    path = tmp_path / "bool.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml",
                         f'<worksheet xmlns="{MAIN}"><sheetData><row r="1">'
                         '<c r="A1" t="b"><v>1</v></c><c r="B1" t="b"><v>0</v></c>'
                         "</row></sheetData></worksheet>")
    assert "- TRUE | FALSE" in de.extract_office(path)


# ---------------------------------------------------------------------------
# C) long sheet: boundaries fall on rows, not on CHUNK_MAX_SIZE
# ---------------------------------------------------------------------------
def _long_sheet(tmp_path, rows: int = 400):
    table = [[f"pol-{i:04d}", f"Popis položky číslo {i} pro dlouhý list rozpočtu", i * 13, i * 1000.5]
             for i in range(rows)]
    return _write_xlsx(tmp_path, {"Dlouhý": table}), table


def test_c_long_sheet_has_natural_row_boundaries(tmp_path):
    path, table = _long_sheet(tmp_path)
    text = de.extract_office(path)
    lines = _rows_of(text)
    assert len(lines) == len(table)
    assert lines[0].startswith("- pol-0000 |") and lines[-1].startswith("- pol-0399 |")


def test_c_long_sheet_splits_into_several_blocks_below_the_chunk_cap(tmp_path):
    path, _ = _long_sheet(tmp_path)
    text = de.extract_office(path)
    blocks = text.split("\n\n")
    assert len(blocks) > 1, "dlouhý list musí mít víc než jeden blok"
    assert all(len(block) <= ai_search.CHUNK_MAX_SIZE for block in blocks)
    # Sheet context must not be lost on the second and later blocks.
    assert all(block.startswith("[Sheet: Dlouhý]") for block in blocks)


def test_c_chunks_never_cut_a_row_in_half(tmp_path):
    path, table = _long_sheet(tmp_path)
    pieces = ai_search.chunks(de.extract_office(path))
    assert len(pieces) > 1
    complete = {"- " + " | ".join(_render(v) for v in row) for row in table}
    for _, body in pieces:
        assert len(body) <= ai_search.CHUNK_MAX_SIZE
        for line in body.split("\n"):
            if line.startswith("[Sheet: "):
                continue
            assert line in complete, f"řádek byl rozřezán uprostřed: {line!r}"


def test_c_no_row_is_lost_across_chunks(tmp_path):
    path, table = _long_sheet(tmp_path)
    pieces = ai_search.chunks(de.extract_office(path))
    joined = "\n".join(body for _, body in pieces)
    for row in table:
        assert "- " + " | ".join(_render(v) for v in row) in joined


def _render(value):
    if isinstance(value, str):
        return value
    return f"{float(value):.10g}"


def _section(tag: str, rows: int = 8):
    """A section big enough to clear SHEET_BLOCK_MIN_CHARS on its own."""
    return [[f"{tag}-{i}", f"popis položky {tag} číslo {i} v sekci rozpočtu", i * 100] for i in range(rows)]


def test_c_blank_source_row_starts_a_new_block(tmp_path):
    path = _write_xlsx(tmp_path, {"S": _section("A") + [[]] + _section("B")})
    text = de.extract_office(path)
    blocks = text.split("\n\n")
    assert len(blocks) == 2
    assert "popis položky A" in blocks[0] and "popis položky A" not in blocks[1]
    assert "popis položky B" in blocks[1] and "popis položky B" not in blocks[0]


def test_c_blank_row_does_not_break_a_block_that_is_still_tiny(tmp_path):
    # Production spreadsheets use a blank row as layout roughly every three
    # rows; honouring each one made a third of all blocks a single short row.
    path = _write_xlsx(tmp_path, {"S": [["a", 1], [], ["b", 2], [], ["c", 3]]})
    text = de.extract_office(path)
    assert text.count("[Sheet: S]") == 1
    assert _rows_of(text) == ["- a | 1", "- b | 2", "- c | 3"]


def test_c_no_block_is_a_single_short_row_in_a_layout_heavy_sheet(tmp_path):
    rows = []
    for group in range(40):
        rows.extend(_section(f"G{group}", rows=3))
        rows.append([])
    path = _write_xlsx(tmp_path, {"S": rows})
    text = de.extract_office(path)
    blocks = [b for b in text.split("\n\n") if b.strip()]
    assert all(len(_rows_of(block)) > 1 for block in blocks[:-1])
    assert all(len(block) <= ai_search.CHUNK_MAX_SIZE for block in blocks)


def test_c_trailing_blank_rows_do_not_produce_empty_blocks(tmp_path):
    path = _write_xlsx(tmp_path, {"S": [["obsah"], [], [], []]})
    text = de.extract_office(path)
    assert text == "[Sheet: S]\n- obsah"


# ---------------------------------------------------------------------------
# D) regression: other formats are untouched
# ---------------------------------------------------------------------------
DOCX_TWO_PARAGRAPHS = (
    '<w:document xmlns:w="w"><w:body>'
    "<w:p><w:r><w:t>První</w:t></w:r><w:r><w:t>odstavec</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Druhý</w:t></w:r><w:r><w:t>odstavec</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def test_d_docx_extraction_is_unchanged(tmp_path):
    path = tmp_path / "smlouva.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", DOCX_TWO_PARAGRAPHS)
    text, method = de.extract_text(path)
    assert method == "office_xml"
    assert [line for line in text.split("\n") if line.strip()] == ["První odstavec", "Druhý odstavec"]
    assert "[Sheet:" not in text and "- " not in text


def test_d_pptx_still_uses_the_flat_join(tmp_path):
    path = tmp_path / "prezentace.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml",
                         '<p:sld xmlns:p="p"><p:t>Nadpis</p:t><p:t>obsah snímku</p:t></p:sld>')
    text = de.extract_office(path)
    assert text == "Nadpis obsah snímku"


def test_d_clean_xml_text_itself_is_unchanged(tmp_path):
    # PPTX and the clean_docx_paragraphs fallback still depend on this exact
    # behaviour, so the .xlsx fix must not have altered it.
    assert de.clean_xml_text(b"<sst><t>alfa</t><t>beta</t><v>12</v></sst>") == "alfa beta 12"


def test_d_eml_extraction_is_unchanged(tmp_path):
    from email.message import EmailMessage
    message = EmailMessage()
    message["Subject"] = "Předání staveniště"
    message["From"] = "a@example.com"
    message["To"] = "b@example.com"
    message.set_content("Tělo zprávy o předání.")
    path = tmp_path / "mail.eml"
    path.write_bytes(message.as_bytes())
    text = ai_search.extract_eml(path)
    assert "Předmět: Předání staveniště" in text
    assert "Tělo zprávy o předání." in text
    assert "[Sheet:" not in text


def test_d_minimal_workbook_without_sheets_still_yields_its_text(tmp_path):
    # Guards the fallback for damaged/minimal workbooks: text that only exists
    # in the shared string table must still be indexed.
    path = tmp_path / "minimal.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", "<sst><t>GAMMA obsah tabulky Excel.</t></sst>")
    text, method = ai_search.extract(path)
    assert method == "office_xml"
    assert "GAMMA obsah tabulky Excel." in text


# ---------------------------------------------------------------------------
# End-to-end through sync(): the values must be findable in FTS, which is what
# the index-level failure was about.
# ---------------------------------------------------------------------------
class FakeEmbeddings:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_xlsx_cell_values_are_searchable_after_indexing(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _write_xlsx(root, {"ZL 001": [
        ["Kód", "Popis", "Cena"],
        ["01", "Bourání energokanálu", 73963.71],
    ]}, name="rozpocet.xlsx")
    db, lance = tmp_path / "state.sqlite3", tmp_path / "lance"
    result = ai_search.sync(root, db, lance, FakeEmbeddings())
    assert result["added"] == 1 and result["errors"] == 0
    with ai_search.database(db) as con:
        assert con.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'energokanálu'").fetchone()[0] == 1
        assert con.execute("""SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH '"73963.71"'""").fetchone()[0] == 1
