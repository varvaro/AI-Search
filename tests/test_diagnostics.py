import json
import re
from pathlib import Path

import ai_search
import diagnostics


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


def test_app_version_is_a_plain_release_version():
    """APP_VERSION is what the exported report shows as "Verze:", so it must be
    a released X.Y.Z tag - not a pre-release placeholder. It sat at "1.0.0-rc1"
    across v1.1.0-v1.1.2, which made every exported diagnostic misreport the
    version an internal user would quote in a problem report. No assertion
    against the actual Git tag here on purpose: CI checks out without tags."""
    assert re.fullmatch(r"\d+\.\d+\.\d+",diagnostics.APP_VERSION),diagnostics.APP_VERSION


def test_exported_report_states_the_app_version(tmp_path):
    base,root,_,_=prepared_runtime(tmp_path); data=diagnostics.collect_diagnostics(str(root),base)
    assert data["version"]==diagnostics.APP_VERSION
    html,_=diagnostics.export_reports(data,base)
    assert f"Verze: {diagnostics.APP_VERSION}" in html.read_text(encoding="utf-8")


def test_report_font_supports_required_czech_glyphs():
    font_path=diagnostics._report_font_path()
    assert font_path is not None
    assert diagnostics._font_supports_czech(font_path)


def test_diagnostic_export_creates_html_and_pdf(tmp_path):
    base,root,_,_=prepared_runtime(tmp_path); data=diagnostics.collect_diagnostics(str(root),base)
    html,pdf=diagnostics.export_reports(data,base)
    assert html.exists() and "AI SEARCH RC1" in html.read_text(encoding="utf-8")
    assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF") and pdf.stat().st_size>1000


def test_missing_embedding_blocks_rc1(tmp_path):
    base,root,db,lance=prepared_runtime(tmp_path); table=ai_search.lance_table(lance); table.delete('id != "__init__"')
    check=diagnostics.verify_index(db,lance,base/"database/.index.lock")
    assert not check["ok"] and check["missing_embeddings"]==1
