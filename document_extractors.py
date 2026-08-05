"""Minimal document text extraction helpers used by ai_search.py.

Adapted from the standalone box_index.py tool (external Codex CORE project) to
remove AI Search's dependency on a path outside this repository. PDF
extraction is intentionally not duplicated here: ai_search.py already
implements its own extract_pdf() with an OCR fallback and always intercepts
".pdf" before consulting INDEXED_EXTS.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

TEXT_EXTS = {".txt", ".md"}
OFFICE_EXTS = {".docx", ".xlsx", ".pptx"}
METADATA_ONLY_EXTS = {".mpp", ".dwg"}
INDEXED_EXTS = TEXT_EXTS | OFFICE_EXTS | METADATA_ONLY_EXTS


def clean_xml_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    parts = []
    for node in root.iter():
        if node.text and node.tag.rsplit("}", 1)[-1] in {"t", "v"}:
            parts.append(node.text)
    return " ".join(parts)


def extract_office(path: Path) -> str:
    wanted = {
        ".docx": re.compile(r"^word/(document|header\d+|footer\d+|footnotes|endnotes)\.xml$"),
        ".pptx": re.compile(r"^ppt/(slides/slide\d+|notesSlides/notesSlide\d+)\.xml$"),
        ".xlsx": re.compile(r"^xl/(sharedStrings|worksheets/sheet\d+)\.xml$"),
    }[path.suffix.lower()]
    pieces = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if wanted.match(name):
                try:
                    pieces.append(clean_xml_text(zf.read(name)))
                except (ET.ParseError, KeyError):
                    continue
    return "\n".join(pieces)


def extract_text(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in METADATA_ONLY_EXTS:
        return "", "metadata_only"
    if ext in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace"), "text"
    if ext in OFFICE_EXTS:
        return extract_office(path), "office_xml"
    return "", "unsupported"
