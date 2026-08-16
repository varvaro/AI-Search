"""PR9.6.0 — DRAWING query vs textual/admin documents.

Classifier stays filename/path only. DRAWING queries apply mismatch
penalty to REGULATORY and TECHNICAL_REPORT, and do not boost every
DRAWING filename. Flag default remains OFF.
"""
from __future__ import annotations

import inspect

import pytest

import ai_search
import ai_search_config
import document_class_affinity as dca


DRAWING_Q = "najdi mi výkres retenční nádrže?"


class _FakeEmbeddings:
    name = "fake"

    def encode(self, texts, **kw):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "Projekt"
    root.mkdir()
    (root / "alpha.txt").write_text("ALPHA unikátní fulltextový výraz.")
    (root / "02_B2_podmínky.pdf").write_text("správní podmínky stavby.")
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings


def test_flag_default_is_on():
    assert ai_search_config.DOCUMENT_CLASS_AFFINITY_ENABLED is True
    assert ai_search.DOCUMENT_CLASS_AFFINITY_ENABLED is True


def test_target_query_stays_drawing():
    assert dca.classify_query(DRAWING_Q) is dca.QueryClass.DRAWING
    assert dca.classify_query("výkres retenční nádrže") is dca.QueryClass.DRAWING


@pytest.mark.parametrize("name,path", [
    ("ÚMČ P5 OŽP - rozhodnutí o povolení.pdf", ""),
    ("02_B2_podmínky.pdf", ""),
    ("stanovisko HZS.pdf", ""),
    ("vyjádření správce sítě.pdf", ""),
])
def test_regulatory_from_filename(name, path):
    assert dca.classify_document(name, path) is dca.DocumentClass.REGULATORY


def test_technical_report_unchanged():
    assert dca.classify_document("technická zpráva.pdf", "/reports/technická zpráva.pdf") is (
        dca.DocumentClass.TECHNICAL_REPORT
    )


def test_retence_filename_stays_unknown():
    assert dca.classify_document("32_RETENCE.pdf", "/docs/D1_1_Stavebni/32_RETENCE.pdf") is (
        dca.DocumentClass.UNKNOWN
    )


def test_drawing_query_penalizes_regulatory():
    detail = dca.compute_class_affinity(DRAWING_Q, "02_B2_podmínky.pdf", "")
    assert detail.document_class is dca.DocumentClass.REGULATORY
    assert detail.bonus == pytest.approx(dca.CLASS_MISMATCH_PENALTY)
    assert detail.reason == "class_mismatch"


def test_drawing_query_penalizes_technical_report():
    detail = dca.compute_class_affinity(DRAWING_Q, "technická zpráva.pdf", "")
    assert detail.document_class is dca.DocumentClass.TECHNICAL_REPORT
    assert detail.bonus == pytest.approx(dca.CLASS_MISMATCH_PENALTY)
    assert detail.reason == "class_mismatch"


def test_drawing_query_unknown_is_zero():
    detail = dca.compute_class_affinity(DRAWING_Q, "32_RETENCE.pdf", "")
    assert detail.document_class is dca.DocumentClass.UNKNOWN
    assert detail.bonus == 0.0
    assert detail.reason == "unknown"


def test_drawing_query_does_not_boost_drawing_filenames():
    detail = dca.compute_class_affinity(DRAWING_Q, "výkres_desky.pdf", "")
    assert detail.document_class is dca.DocumentClass.DRAWING
    assert detail.bonus == 0.0
    assert detail.reason == "drawing_match_neutral"


def test_non_drawing_query_does_not_penalize_regulatory():
    detail = dca.compute_class_affinity("jaká je tloušťka desky", "02_B2_podmínky.pdf", "")
    assert detail.query_class is dca.QueryClass.UNKNOWN
    assert detail.bonus == 0.0


def test_contract_query_keeps_match_bonus():
    detail = dca.compute_class_affinity("smlouva na dílo", "SoD_dodavatel.pdf", "")
    assert detail.bonus == pytest.approx(dca.CLASS_MATCH_BONUS)
    assert detail.reason == "class_match"


def test_filename_without_admin_stem_is_not_regulatory():
    assert dca.classify_document("ozp.pdf", "/docs/reports/ozp.pdf") is dca.DocumentClass.UNKNOWN
    assert dca.classify_document("alpha.txt", "/Projekt/alpha.txt") is dca.DocumentClass.UNKNOWN


def test_classifier_has_no_body_parameter():
    sig = inspect.signature(dca.classify_document)
    assert list(sig.parameters) == ["name", "path"]
    # A body mention of "výkres" cannot be passed in and cannot change class.
    assert dca.classify_document("ozp.pdf", "") is dca.DocumentClass.UNKNOWN


def test_module_still_ignores_heading_quote_body():
    src = inspect.getsource(dca.classify_document)
    assert "heading" not in src
    assert "quote" not in src
    assert "body" not in src


def test_flag_off_search_scores_identical_for_drawing_query(backend, monkeypatch):
    root, state, embeddings = backend
    q = DRAWING_Q
    monkeypatch.setattr(ai_search, "DOCUMENT_CLASS_AFFINITY_ENABLED", False)
    off = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings)
    monkeypatch.setattr(ai_search, "DOCUMENT_CLASS_AFFINITY_ENABLED", True)
    on = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings)
    monkeypatch.setattr(ai_search, "DOCUMENT_CLASS_AFFINITY_ENABLED", False)
    assert off
    assert "document_class_affinity" not in off[0]["match"]
    assert "document_class_affinity" in on[0]["match"]
    # Tiny fixture: the drawing query may score the podmínky file; OFF scores
    # must match a second OFF run (identity), and ON may differ only by the
    # documented mismatch on that REGULATORY filename.
    off2 = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings)
    assert [r["score"] for r in off] == [r["score"] for r in off2]
    on_by_doc = {r["document"]: r for r in on}
    off_by_doc = {r["document"]: r for r in off}
    for name, row in off_by_doc.items():
        delta = on_by_doc[name]["score"] - row["score"]
        if name == "02_B2_podmínky.pdf":
            assert delta == pytest.approx(dca.CLASS_MISMATCH_PENALTY)
        else:
            assert delta == pytest.approx(0.0)
