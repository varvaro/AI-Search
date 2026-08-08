"""Regresní testy pro FÁZI 1 opravy indexace (extrakce DOCX/RTF/EML).

Rozsah: pouze document_extractors.py + ai_search.extract() dispatch.
Nezasahuje do chunks(), retrievalu, embeddingu, LanceDB, RRF ani scoringu.
"""
import shutil
import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest

import ai_search
import document_extractors as de


def write_office(path: Path, member: str, xml: str):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, xml)


# ---------------------------------------------------------------------------
# DOCX: text run "<w:t>" musí zůstat spojen mezerami v rámci odstavce, ale
# mezi odstavci "<w:p>" musí vzniknout nový řádek.
# ---------------------------------------------------------------------------

DOCX_TWO_PARAGRAPHS = (
    '<w:document xmlns:w="w"><w:body>'
    '<w:p><w:r><w:t>První</w:t></w:r><w:r><w:t>odstavec</w:t></w:r></w:p>'
    '<w:p><w:r><w:t>Druhý</w:t></w:r><w:r><w:t>odstavec</w:t></w:r></w:p>'
    "</w:body></w:document>"
)


def test_docx_paragraphs_are_separated_by_newlines(tmp_path):
    path = tmp_path / "smlouva.docx"
    write_office(path, "word/document.xml", DOCX_TWO_PARAGRAPHS)
    text, method = de.extract_text(path)
    assert method == "office_xml"
    lines = [line for line in text.split("\n") if line.strip()]
    assert lines == ["První odstavec", "Druhý odstavec"]


def test_docx_runs_within_one_paragraph_join_with_space_not_newline(tmp_path):
    path = tmp_path / "runs.docx"
    write_office(
        path,
        "word/document.xml",
        '<w:document xmlns:w="w"><w:p><w:r><w:t>Slovo</w:t></w:r>'
        '<w:r><w:t>jedna</w:t></w:r><w:r><w:t>dva</w:t></w:r></w:p></w:document>',
    )
    text, _ = de.extract_text(path)
    assert text.strip() == "Slovo jedna dva"
    assert "\n" not in text.strip()


def test_docx_without_paragraph_markup_falls_back_to_flat_join(tmp_path):
    """Zpětná kompatibilita: minimální/neúplné XML bez <w:p> se stále
    extrahuje (fallback na starý plochý join), místo aby vrátilo prázdný text."""
    path = tmp_path / "minimal.docx"
    write_office(path, "word/document.xml", '<w:document xmlns:w="w"><w:t>BETA obsah</w:t></w:document>')
    text, _ = de.extract_text(path)
    assert "BETA obsah" in text


def test_docx_chunking_produces_many_chunks_instead_of_one_giant_block(tmp_path):
    """End-to-end důkaz opravy: dokument s N odstavci dřív vznikl jako 1 obří
    chunk (žádný \\n v extrahovaném textu). Word běžně odděluje odstavce
    prázdnými "distančními" odstavci (<w:p></w:p>) - to dnes musí projít až
    do extrahovaného textu jako prázdný řádek, který chunks() (beze změny)
    použije jako existující hranici pro flush()."""
    paragraphs = "".join(
        f'<w:p><w:r><w:t>Odstavec číslo {i} s dostatečně dlouhým obsahem pro chunking testu.</w:t></w:r></w:p><w:p></w:p>'
        for i in range(30)
    )
    path = tmp_path / "dlouha_smlouva.docx"
    write_office(path, "word/document.xml", f'<w:document xmlns:w="w">{paragraphs}</w:document>')
    text, _ = de.extract_text(path)
    pieces = ai_search.chunks(text)
    assert len(pieces) >= 25, f"Očekáváno mnoho chunků místo jednoho obřího bloku, dostal jsem {len(pieces)}"


def test_xlsx_extraction_is_unchanged_by_docx_fix(tmp_path):
    """XLSX (a PPTX) nemá pojem odstavce - musí zůstat na starém plochém joinu."""
    path = tmp_path / "tabulka.xlsx"
    write_office(path, "xl/sharedStrings.xml", '<sst><t>GAMMA</t><t>hodnota</t></sst>')
    text, method = de.extract_text(path)
    assert method == "office_xml"
    assert text.strip() == "GAMMA hodnota"


# ---------------------------------------------------------------------------
# RTF: musí jít přes textutil (jako .doc), ne přes syrové čtení bajtů.
# ---------------------------------------------------------------------------

RTF_WITH_EMBEDDED_BINARY = (
    r"{\rtf1\ansi\deff0"
    r"{\fonttbl{\f0 Arial;}}"
    r"\f0\fs24 Skutecny text technickeho predpisu.\par"
    r"{\pict\pngblip\picw100\pich100 "
    + ("deadbeef" * 5000)  # simuluje desítky KB hex-kódovaného vloženého obrázku
    + r"}"
    r"}"
)

TEXTUTIL_AVAILABLE = shutil.which("/usr/bin/textutil") is not None

requires_textutil = pytest.mark.skipif(
    not TEXTUTIL_AVAILABLE,
    reason="textutil není dostupný (vyžaduje macOS)",
)


@requires_textutil
def test_rtf_extraction_uses_textutil_and_strips_binary_payload(tmp_path):
    path = tmp_path / "predpis.rtf"
    path.write_text(RTF_WITH_EMBEDDED_BINARY, encoding="utf-8")
    text, method = ai_search.extract(path)
    assert method == "textutil"
    assert "Skutecny text technickeho predpisu" in text
    assert "deadbeef" not in text
    assert "pngblip" not in text
    # Reálný text musí být řádově menší než syrový soubor s vloženým obrázkem.
    assert len(text) < path.stat().st_size / 10


@requires_textutil
def test_rtf_no_longer_uses_raw_path_read_text(tmp_path, monkeypatch):
    """Zajišťuje, že se .rtf nedostane do větve syrového path.read_text()."""
    path = tmp_path / "x.rtf"
    path.write_text(r"{\rtf1 obsah}", encoding="utf-8")
    called = {"raw_read": False}
    original_read_text = Path.read_text

    def spying_read_text(self, *args, **kwargs):
        if self == path:
            called["raw_read"] = True
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spying_read_text)
    ai_search.extract(path)
    assert called["raw_read"] is False, ".rtf musí jít přes textutil, ne přes path.read_text()"


# ---------------------------------------------------------------------------
# EML: musí jít přes standardní MIME parser, ne přes syrové čtení bajtů.
# ---------------------------------------------------------------------------

def _write_eml(path: Path, message: EmailMessage) -> None:
    path.write_bytes(bytes(message))


def test_eml_extracts_headers_and_plain_body(tmp_path):
    msg = EmailMessage()
    msg["Subject"] = "Testovací předmět"
    msg["From"] = "odesilatel@example.com"
    msg["To"] = "prijemce@example.com"
    msg["Date"] = "Mon, 1 Jan 2026 10:00:00 +0100"
    msg.set_content("Prostý text těla zprávy s klíčovým slovem ALPHA.")
    path = tmp_path / "mail.eml"
    _write_eml(path, msg)

    text, method = ai_search.extract(path)
    assert method == "email_mime"
    assert "Testovací předmět" in text
    assert "odesilatel@example.com" in text
    assert "prijemce@example.com" in text
    assert "ALPHA" in text


def test_eml_converts_html_only_body_to_text(tmp_path):
    msg = EmailMessage()
    msg["Subject"] = "HTML zpráva"
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg["Date"] = "Mon, 1 Jan 2026 10:00:00 +0100"
    msg.set_content(
        "<html><body><p>První odstavec BETA.</p><p>Druhý odstavec GAMMA.</p></body></html>",
        subtype="html",
    )
    path = tmp_path / "html.eml"
    _write_eml(path, msg)

    text, method = ai_search.extract(path)
    assert method == "email_mime"
    assert "BETA" in text and "GAMMA" in text
    assert "<p>" not in text and "<html>" not in text


def test_eml_does_not_index_attachment_content(tmp_path):
    msg = EmailMessage()
    msg["Subject"] = "S přílohou"
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg["Date"] = "Mon, 1 Jan 2026 10:00:00 +0100"
    msg.set_content("Text zprávy DELTA.")
    msg.add_attachment(
        b"binary-attachment-payload-should-not-be-indexed",
        maintype="application",
        subtype="octet-stream",
        filename="priloha.bin",
    )
    path = tmp_path / "attachment.eml"
    _write_eml(path, msg)

    text, method = ai_search.extract(path)
    assert "DELTA" in text
    assert "priloha.bin" not in text
    assert "binary-attachment-payload" not in text


def test_eml_no_longer_uses_raw_path_read_text(tmp_path, monkeypatch):
    msg = EmailMessage()
    msg["Subject"] = "Test"
    msg["From"] = "a@example.com"
    msg["To"] = "b@example.com"
    msg.set_content("obsah")
    path = tmp_path / "spy.eml"
    _write_eml(path, msg)

    called = {"raw_read": False}
    original_read_text = Path.read_text

    def spying_read_text(self, *args, **kwargs):
        if self == path:
            called["raw_read"] = True
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spying_read_text)
    ai_search.extract(path)
    assert called["raw_read"] is False, ".eml musí jít přes MIME parser, ne přes path.read_text()"


# ---------------------------------------------------------------------------
# End-to-end: sync() nad reálným DOCX souborem s více odstavci musí vytvořit
# více samostatně dohledatelných chunků (ne 1 obří chunk).
# ---------------------------------------------------------------------------

class FakeEmbeddings:
    def encode(self, texts):
        return [[float(len(text) > 0), 0.0, 0.5] for text in texts]


def test_sync_indexes_docx_into_multiple_chunks(tmp_path):
    root = tmp_path / "Projekt"
    root.mkdir()
    paragraphs = "".join(
        f'<w:p><w:r><w:t>Bod {i}: technická specifikace betonáže základové desky.</w:t></w:r></w:p><w:p></w:p>'
        for i in range(20)
    )
    write_office(root / "specifikace.docx", "word/document.xml", f'<w:document xmlns:w="w">{paragraphs}</w:document>')

    state = tmp_path / "state"
    ai_search.sync(root, state / "index.sqlite3", state / "lance", FakeEmbeddings())

    con = ai_search.connect(state / "index.sqlite3")
    count = con.execute(
        "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.name=?",
        ("specifikace.docx",),
    ).fetchone()[0]
    assert count >= 15, f"Očekáváno mnoho chunků z 20 odstavců, dostal jsem {count}"
