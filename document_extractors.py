"""Minimal document text extraction helpers used by ai_search.py.

Adapted from the standalone box_index.py tool (external Codex CORE project) to
remove AI Search's dependency on a path outside this repository. PDF
extraction is intentionally not duplicated here: ai_search.py already
implements its own extract_pdf() with an OCR fallback and always intercepts
".pdf" before consulting INDEXED_EXTS.
"""
from __future__ import annotations

import html
import re
import zipfile
import xml.etree.ElementTree as ET
from email import policy as email_policy
from email.parser import BytesParser
from pathlib import Path

TEXT_EXTS = {".txt", ".md"}
OFFICE_EXTS = {".docx", ".xlsx", ".pptx"}
METADATA_ONLY_EXTS = {".mpp", ".dwg"}
INDEXED_EXTS = TEXT_EXTS | OFFICE_EXTS | METADATA_ONLY_EXTS

# A sheet is emitted as blocks of rows separated by a blank line, because a
# blank line is what ai_search.chunks() splits on. Without it an entire sheet
# is one block and the only remaining boundary is CHUNK_MAX_SIZE, which cuts
# mid-row. Kept below that cap so the cap never has to fire on a table.
SHEET_BLOCK_MAX_CHARS = 3000
# A blank row only ends a block that already stands on its own as a retrieval
# unit. Measured on the production cost estimates (read-only, 2026-08-07):
# spreadsheets there put a blank row roughly every three rows as layout, not as
# a section break, so honouring every one of them made 32% of blocks a single
# ~50-char row. 300 is the smallest value that removes single-row blocks
# entirely (1822 -> 474 blocks, median 174 -> 2756 chars); higher values only
# discard more of the genuine section breaks.
SHEET_BLOCK_MIN_CHARS = 300

_CELL_LINEBREAK = re.compile(r"[\r\n\t]+")
_CELL_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_cell_text(text: str) -> str:
    """One spreadsheet cell -> one line-safe string. A cell may legally hold
    newlines (Alt+Enter) and stray control bytes; both would break the
    one-row-per-line structure the chunker relies on."""
    return _CELL_CONTROL_CHARS.sub("", _CELL_LINEBREAK.sub(" ", text)).strip()


def format_sheet_section(sheet_name: str, rows, max_block_chars: int = SHEET_BLOCK_MAX_CHARS,
                         min_block_chars: int = SHEET_BLOCK_MIN_CHARS) -> str:
    """Render one sheet as `[Sheet: name]` + `- cell | cell` lines, grouped
    into blank-line-separated blocks.

    `rows` is an iterable of cell-string lists; an empty (or all-empty) row is
    the sheet's own section break, honoured once the current block has reached
    `min_block_chars` (see that constant for why it is not honoured
    unconditionally). A block is also closed once it approaches
    `max_block_chars`, so a sheet with no blank rows at all still gets
    boundaries that fall on row edges instead of mid-row.

    The `[Sheet: ...]` label is repeated on every block: chunks() turns each
    block into its own chunk, and without the label only the sheet's first
    chunk would say which sheet it came from. A sheet that fits in one block
    is emitted exactly as before this grouping existed.

    Leading "- " is not cosmetic: chunks() treats a bare-number-first line
    ("1 ...", "01 ...") or an ALL-CAPS-first line as a section heading and
    folds it into the NEXT chunk's heading field instead of its body -
    silently dropping that row's actual cell values from the indexed text.
    Item/code columns ("Kód položky", "Poř. č.") are extremely common in these
    spreadsheets, so this is not a hypothetical edge case. A leading "-" can't
    start any of chunks()' three heading patterns.
    """
    label = f"[Sheet: {sheet_name}]"
    blocks: list[list[str]] = []
    block: list[str] = []
    size = 0
    for cells in rows:
        cells = list(cells)
        while cells and not cells[-1]:
            cells.pop()  # trim only trailing empty cells, not interior gaps
        if not any(cells):
            if block and size >= min_block_chars:
                blocks.append(block); block = []; size = 0
            continue
        line = "- " + " | ".join(cells)
        if block and size + len(line) + 1 > max_block_chars:
            blocks.append(block); block = []; size = 0
        block.append(line); size += len(line) + 1
    if block:
        blocks.append(block)
    return "\n\n".join(label + "\n" + "\n".join(rows) for rows in blocks)


def clean_xml_text(xml_bytes: bytes) -> str:
    """Flat text join for PPTX (slide shapes) and as a last-resort fallback:
    no paragraph concept that chunks() could use.

    Note for XLSX: this is NOT how sheets are read anymore (see
    extract_xlsx()). On a real sheet part it yields shared-string *indices*
    rather than text, because a text cell stores `<c t="s"><v>4974</v></c>`.
    """
    root = ET.fromstring(xml_bytes)
    parts = []
    for node in root.iter():
        if node.text and node.tag.rsplit("}", 1)[-1] in {"t", "v"}:
            parts.append(node.text)
    return " ".join(parts)


def clean_docx_paragraphs(xml_bytes: bytes) -> str:
    """Paragraph-aware text join for Word XML parts (document/header/footer/
    footnotes/endnotes).

    Word splits a single paragraph's text across many <w:t> runs purely for
    formatting reasons (no implied whitespace between them), but a <w:p>
    boundary always means a new line. Joining runs within a paragraph with a
    space and paragraphs with "\n" lets chunks() split on real paragraph
    breaks instead of collapsing an entire document into one giant line, as
    the previous flat " ".join(parts) over the whole XML part used to do.
    """
    root = ET.fromstring(xml_bytes)
    paragraphs = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "p"]
    if not paragraphs:
        # No <w:p> markup found (e.g. malformed/minimal XML) - fall back to
        # the flat join so we still extract whatever text exists.
        return clean_xml_text(xml_bytes)
    lines = []
    for paragraph in paragraphs:
        runs = [node.text for node in paragraph.iter() if node.text and node.tag.rsplit("}", 1)[-1] == "t"]
        lines.append(" ".join(runs))
    return "\n".join(lines)


_XLSX_SHEET_PART = re.compile(r"^xl/worksheets/sheet\d+\.xml$")
_XLSX_CELL_REF = re.compile(r"^([A-Z]+)")


def _local(node) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    """The shared string table: every text cell in the workbook stores an
    index into this list instead of its own text."""
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except (KeyError, ET.ParseError):
        return []
    strings = []
    for item in root:
        # <si> holds one string, possibly split across several <r><t> runs for
        # formatting reasons only - no whitespace is implied between them.
        strings.append("".join(node.text or "" for node in item.iter() if _local(node) == "t"))
    return strings


def _xlsx_sheet_parts(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(sheet name, zip part name) in workbook tab order. The mapping needs
    the relationship table: part names are not positional (a real production
    workbook starts at xl/worksheets/sheet4.xml)."""
    names = set(zf.namelist())
    try:
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        targets = {node.get("Id"): node.get("Target", "") for node in rels.iter() if _local(node) == "Relationship"}
        parts = []
        for node in ET.fromstring(zf.read("xl/workbook.xml")).iter():
            if _local(node) != "sheet":
                continue
            rid = next((v for k, v in node.attrib.items() if k.rsplit("}", 1)[-1] == "id"), None)
            target = targets.get(rid, "").lstrip("/")
            if target and not target.startswith("xl/"):
                target = "xl/" + target
            if target in names:
                parts.append((node.get("name") or "", target))
        if parts:
            return parts
    except (KeyError, ET.ParseError):
        pass
    return [("", name) for name in sorted(name for name in names if _XLSX_SHEET_PART.match(name))]


def _xlsx_number(raw: str) -> str:
    """Match the .xls formatting rule: no trailing ".0" on integers and no
    floating-point noise (a raw <v> is often "5.2784999999999993")."""
    try:
        return f"{float(raw):.10g}"
    except ValueError:
        return raw


def _xlsx_cell_text(cell, shared: list[str]) -> str:
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        return clean_cell_text("".join(node.text or "" for node in cell.iter() if _local(node) == "t"))
    value = next((node.text for node in cell if _local(node) == "v"), None)
    if value is None:
        return ""
    if kind == "s":
        try:
            return clean_cell_text(shared[int(value)])
        except (ValueError, IndexError):
            return ""
    if kind == "b":
        return "TRUE" if value.strip() == "1" else "FALSE"
    if kind in {"str", "e"}:
        return clean_cell_text(value)
    return clean_cell_text(_xlsx_number(value))


def _xlsx_rows(xml_bytes: bytes, shared: list[str]) -> list[list[str]]:
    """Rows of one sheet, in order, with interior column gaps preserved and
    skipped row numbers surfaced as blank rows (Excel omits empty rows from
    the XML entirely, but a human put that gap there as a section break)."""
    root = ET.fromstring(xml_bytes)
    rows: list[list[str]] = []
    previous = 0
    for element in root.iter():
        if _local(element) != "row":
            continue
        try:
            number = int(element.get("r") or previous + 1)
        except ValueError:
            number = previous + 1
        if previous and number > previous + 1:
            rows.append([])
        previous = number
        cells: list[str] = []
        for cell in element:
            if _local(cell) != "c":
                continue
            reference = _XLSX_CELL_REF.match(cell.get("r", "") or "")
            if reference:
                column = 0
                for character in reference.group(1):
                    column = column * 26 + (ord(character) - 64)
                while len(cells) < column - 1:
                    cells.append("")
            cells.append(_xlsx_cell_text(cell, shared))
        rows.append(cells)
    return rows


def extract_xlsx(path: Path) -> str:
    """Read an .xlsx as an actual spreadsheet: resolve the shared string table
    and emit one line per row, per sheet.

    Before this, sheets went through clean_xml_text(), which concatenates
    every <t>/<v> node with spaces. In a sheet part a text cell is
    `<c t="s"><v>4974</v></c>`, so that produced a flat stream of shared-string
    *indices* ("4974 4975 52 4976 ...") with the actual text appearing only
    once, dumped separately from sharedStrings.xml in string-table order with
    no row, column or sheet association at all.

    Dates are left as their raw serial numbers: resolving them needs the
    number-format table in styles.xml, which the previous path did not read
    either, so this is unchanged behaviour rather than a new gap.
    """
    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf)
        sections = []
        for sheet_name, part in _xlsx_sheet_parts(zf):
            try:
                rows = _xlsx_rows(zf.read(part), shared)
            except (ET.ParseError, KeyError):
                continue
            section = format_sheet_section(sheet_name, rows)
            if section:
                sections.append(section)
        if sections:
            return "\n\n".join(sections)
        # No readable sheet rows (minimal or damaged workbook): fall back to
        # the previous flat join over the shared string table, byte for byte,
        # so text that exists is still indexed exactly as it used to be.
        try:
            return clean_xml_text(zf.read("xl/sharedStrings.xml"))
        except (KeyError, ET.ParseError):
            return ""


def extract_office(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return extract_xlsx(path)
    wanted = {
        ".docx": re.compile(r"^word/(document|header\d+|footer\d+|footnotes|endnotes)\.xml$"),
        ".pptx": re.compile(r"^ppt/(slides/slide\d+|notesSlides/notesSlide\d+)\.xml$"),
    }[ext]
    clean = clean_docx_paragraphs if ext == ".docx" else clean_xml_text
    pieces = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if wanted.match(name):
                try:
                    pieces.append(clean(zf.read(name)))
                except (ET.ParseError, KeyError):
                    continue
    return "\n".join(pieces)


_HTML_HIDDEN = re.compile(r"(?is)<(script|style|head)[^>]*>.*?</\1>")
_HTML_BLOCK_BREAK = re.compile(r"(?i)</(p|div|tr|li|h[1-6])>|<br\s*/?>")
_HTML_TAG = re.compile(r"<[^>]+>")


def html_to_text(markup: str) -> str:
    """Minimal, dependency-free HTML-to-text conversion for EML text/html
    bodies: drop script/style, turn block-level closing tags into newlines,
    strip remaining tags, and unescape entities."""
    markup = _HTML_HIDDEN.sub(" ", markup)
    markup = _HTML_BLOCK_BREAK.sub("\n", markup)
    markup = _HTML_TAG.sub(" ", markup)
    markup = html.unescape(markup)
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in markup.splitlines())
    return "\n".join(line for line in lines if line)


def extract_eml(path: Path) -> str:
    """Parse a .eml file as MIME (RFC 822) instead of reading its raw bytes
    as plain text - the previous approach fed MIME headers, boundary
    markers, and base64/quoted-printable encoded bodies straight into the
    chunker. Attachments are intentionally not indexed."""
    with path.open("rb") as handle:
        message = BytesParser(policy=email_policy.default).parse(handle)
    header_lines = [
        f"Předmět: {message.get('subject', '')}",
        f"Od: {message.get('from', '')}",
        f"Komu: {message.get('to', '')}",
        f"Datum: {message.get('date', '')}",
    ]
    body = ""
    plain_part = message.get_body(preferencelist=("plain",))
    if plain_part is not None:
        try:
            body = plain_part.get_content()
        except Exception:
            body = ""
    if not body.strip():
        html_part = message.get_body(preferencelist=("html",))
        if html_part is not None:
            try:
                body = html_to_text(html_part.get_content())
            except Exception:
                body = ""
    return "\n".join(header_lines + ["", body.strip()]).strip()


def extract_text(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in METADATA_ONLY_EXTS:
        return "", "metadata_only"
    if ext in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace"), "text"
    if ext in OFFICE_EXTS:
        return extract_office(path), "office_xml"
    return "", "unsupported"
