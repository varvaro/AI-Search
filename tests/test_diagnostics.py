import json
from pathlib import Path

import pytest
import reportlab

import ai_search
import diagnostics

MACOS_REPORT_FONTS = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
)


def configure_portable_report_font(monkeypatch):
    if any(path.is_file() for path in MACOS_REPORT_FONTS):
        return
    fallback = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    if not fallback.is_file():
        pytest.skip("no compatible ReportLab font is available")
    original_path = diagnostics.Path

    def portable_path(value):
        path = original_path(value)
        return fallback if path in MACOS_REPORT_FONTS else path

    monkeypatch.setattr(diagnostics, "Path", portable_path)


class FakeEmbeddings:
    def encode(self,texts,**kwargs): return [[0.1,0.2,0.3] for _ in texts]


def prepared_runtime(tmp_path):
    base=tmp_path/"AI Search"; root=tmp_path/"project"; root.mkdir(); (root/"a.txt").write_text("diagnostický dokument")
    db=base/"database/project.sqlite3"; lance=base/"lance/project"; ai_search.sync(root,db,lance,FakeEmbeddings())
    (base/"state").mkdir(parents=True,exist_ok=True); (base/"state/test-status.json").write_text(json.dumps({"ok":True,"tests":66}))
    return base,root,db,lance


def test_index_verification_and_rc1(tmp_path):
    base,root,db,lance=prepared_runtime(tmp_path)
    check=diagnostics.verify_index(db,lance,base/"database/.index.lock")
    assert check["ok"],check
    assert check["documents"]==1 and check["chunks"]==1 and check["embeddings"]==1
    data=diagnostics.collect_diagnostics(str(root),base)
    assert data["rc1"] and data["counts"]=={"documents":1,"chunks":1,"embeddings":1}


def test_diagnostic_export_creates_html_and_pdf(tmp_path, monkeypatch):
    base,root,_,_=prepared_runtime(tmp_path); data=diagnostics.collect_diagnostics(str(root),base)
    configure_portable_report_font(monkeypatch)
    html,pdf=diagnostics.export_reports(data,base)
    assert html.exists() and "AI SEARCH RC1" in html.read_text(encoding="utf-8")
    assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF") and pdf.stat().st_size>1000


def test_missing_embedding_blocks_rc1(tmp_path):
    base,root,db,lance=prepared_runtime(tmp_path); table=ai_search.lance_table(lance); table.delete('id != "__init__"')
    check=diagnostics.verify_index(db,lance,base/"database/.index.lock")
    assert not check["ok"] and check["missing_embeddings"]==1
