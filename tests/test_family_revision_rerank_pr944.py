"""PR9.4.4 — Intent-gated BM25 admission + family-local revision tournament."""
from __future__ import annotations

from datetime import date

import pytest

import ai_search
import ai_search_config
import family_revision_rerank as frr
import metadata_rerank as mr


class _FakeEmbeddings:
    name = "fake"

    def encode(self, texts, **kw):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "Projekt"
    root.mkdir()
    (root / "alpha.txt").write_text("ALPHA unikátní fulltextový výraz.")
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings


# --- flag / identity ----------------------------------------------------------

def test_flag_default_is_off():
    assert ai_search_config.FAMILY_REVISION_RERANK_ENABLED is False
    assert ai_search.FAMILY_REVISION_RERANK_ENABLED is False


def test_flag_off_search_identical_and_no_new_keys(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", False)
    a = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    b = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    assert [r["score"] for r in a] == [r["score"] for r in b]
    assert "admission_source" not in a[0]["match"]
    assert "family_revision" not in a[0]["match"]
    assert "family_revision_bonus" not in a[0]["match"]


# --- intent -------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("najdi aktuální harmonogram", True),
        ("nejnovější verze", True),
        ("platný dokument", True),
        ("poslední harmonogram", True),
        ("poslední verze výkresu", True),
        ("poslední betonáž", False),
        ("kdy byla poslední betonáž", False),
        ("najdi rozpočet garáží", False),
        ("jaký harmonogram platí pro monolit?", True),
    ],
)
def test_revision_intent(query, expected):
    assert frr.has_revision_intent(query) is expected


# --- safe dates ---------------------------------------------------------------

def test_drawing_codes_are_not_dates():
    assert mr.parse_safe_dates("D.1.2.06 - schéma vyztužení základové desky.pdf") == ()
    assert mr.parse_safe_dates("D.1.2.11 - schéma vyztužení 1.PP.pdf") == ()
    assert frr.revision_date("D.1.2.06 - schéma.pdf") is None
    assert frr.revision_date("D.1.2.11 - schéma.pdf") is None


def test_not_identifier_is_not_a_date():
    assert mr.parse_safe_dates("NOT251110_SoD.pdf") == ()
    assert frr.revision_date("36_monolit_FERI_NOT251110") is None


def test_revision_marker_selects_current_stamp_not_embedded_base_date():
    name = "Project_Schedule_akt.11.12.25_R3_akt_4.08.2026.pdf"
    assert mr.parse_safe_dates(name) == (date(2026, 8, 4),)
    assert frr.revision_date(name) == date(2026, 8, 4)


def test_two_revision_stamps_use_newest_attached():
    name = "Plan_akt_01.04.2026_akt_4.08.2026.pdf"
    assert frr.revision_date(name) == date(2026, 8, 4)


def test_unattached_multiple_dates_are_ambiguous():
    assert frr.revision_date("Report_2025-01-15_and_2026-08-04.pdf") is None


# --- family key ---------------------------------------------------------------

def test_sibling_revisions_share_family_key():
    a = frr.family_key("Project_Plan_akt_01.04.2026.pdf", "/docs/Project_Plan_akt_01.04.2026.pdf")
    b = frr.family_key("Project_Plan_akt_01.06.2026.pdf", "/docs/Project_Plan_akt_01.06.2026.pdf")
    c = frr.family_key("Project_Plan_akt_4.08.2026.pdf", "/docs/Project_Plan_akt_4.08.2026.pdf")
    assert a == b == c
    assert "2026" not in a
    assert "akt" not in a
    assert "plan" in a or "project" in a


def test_foreign_subject_is_different_family():
    a = frr.family_key("Project_Plan_akt_4.08.2026.pdf", "/docs/Project_Plan_akt_4.08.2026.pdf")
    b = frr.family_key("Other_Subject_Schedule_250129.pdf", "/offer/Other_Subject_Schedule_250129.pdf")
    assert a != b


def test_family_key_keeps_subject_tokens():
    key = frr.family_key("Foundation_Slab_Drawing_akt_4.08.2026.pdf", "")
    assert "foundation" in key
    assert "slab" in key
    assert "drawing" in key


# --- tournament ---------------------------------------------------------------

def _row(name, path, score=0.1):
    return {"document": name, "path": path, "score": score}


def test_tournament_singleton_is_zero():
    rows = [_row("Only_akt_4.08.2026.pdf", "/a/Only_akt_4.08.2026.pdf")]
    details = frr.annotate_family_revision(rows, "aktuální harmonogram")
    assert details[0].bonus == 0.0
    assert details[0].is_latest is False


def test_tournament_latest_gets_bonus_older_zero():
    rows = [
        _row("Project_Plan_akt_01.04.2026.pdf", "/a/Project_Plan_akt_01.04.2026.pdf"),
        _row("Project_Plan_akt_4.08.2026.pdf", "/a/Project_Plan_akt_4.08.2026.pdf"),
    ]
    details = frr.annotate_family_revision(rows, "aktuální harmonogram")
    by_name = {r["document"]: d for r, d in zip(rows, details)}
    assert by_name["Project_Plan_akt_4.08.2026.pdf"].bonus == 0.03
    assert by_name["Project_Plan_akt_4.08.2026.pdf"].is_latest is True
    assert by_name["Project_Plan_akt_01.04.2026.pdf"].bonus == 0.0


def test_tournament_no_intent_all_zero():
    rows = [
        _row("Project_Plan_akt_01.04.2026.pdf", "/a/Project_Plan_akt_01.04.2026.pdf"),
        _row("Project_Plan_akt_4.08.2026.pdf", "/a/Project_Plan_akt_4.08.2026.pdf"),
    ]
    details = frr.annotate_family_revision(rows, "najdi harmonogram stavby")
    assert all(d.bonus == 0.0 for d in details)


def test_tournament_ambiguous_dates_zero():
    rows = [
        _row("Report_2025-01-15_and_2026-08-04.pdf", "/a/Report_2025-01-15_and_2026-08-04.pdf"),
        _row("Report_2025-03-01_and_2026-01-01.pdf", "/a/Report_2025-03-01_and_2026-01-01.pdf"),
    ]
    details = frr.annotate_family_revision(rows, "aktuální dokument")
    assert all(d.bonus == 0.0 for d in details)


def test_unrelated_newer_document_does_not_score_against_other_family():
    rows = [
        _row("Project_Plan_akt_01.04.2026.pdf", "/a/Project_Plan_akt_01.04.2026.pdf"),
        _row("Other_Subject_Schedule_4.08.2026.pdf", "/b/Other_Subject_Schedule_4.08.2026.pdf"),
    ]
    details = frr.annotate_family_revision(rows, "aktuální harmonogram")
    assert details[0].family_key != details[1].family_key
    assert all(d.bonus == 0.0 for d in details)


def test_single_old_document_is_not_penalized():
    rows = [_row("Archive_Plan_17.1.2025.pdf", "/old/Archive_Plan_17.1.2025.pdf")]
    details = frr.annotate_family_revision(rows, "aktuální harmonogram")
    assert details[0].bonus == 0.0


# --- admission helper ---------------------------------------------------------

def test_select_bm25_floor_adds_unique_doc_once():
    fts_ids = ["a:0", "b:0", "b:1", "c:0"]
    top_ids = ["a:0"]
    doc_ids = {"a:0": 1, "b:0": 2, "b:1": 2, "c:0": 3}
    extras = frr.select_bm25_floor_chunk_ids(fts_ids, top_ids, doc_ids)
    assert extras == ["b:0", "c:0"]


def test_select_bm25_floor_skips_docs_already_in_window():
    extras = frr.select_bm25_floor_chunk_ids(
        ["a:0", "a:1", "b:0"],
        ["a:0"],
        {"a:0": 1, "a:1": 1, "b:0": 2},
    )
    assert extras == ["b:0"]


# --- search wiring ------------------------------------------------------------

@pytest.fixture
def admission_backend(tmp_path, monkeypatch):
    """Noise docs fill a tiny RRF window; one weaker BM25-only target sits in-pool."""
    monkeypatch.setattr(ai_search, "RETRIEVAL_POOL_SIZE", 20)
    monkeypatch.setattr(ai_search, "RERANK_POOL_SIZE", 3)
    root = tmp_path / "projekt"
    root.mkdir()
    for i in range(8):
        (root / f"noise_{i:02d}.txt").write_text(
            "harmonogram stavby " * 12 + f"noise {i}",
            encoding="utf-8",
        )
    (root / "Project_Plan_akt_4.08.2026.txt").write_text(
        "harmonogram",
        encoding="utf-8",
    )
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    with ai_search.database(db) as con:
        target = con.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id "
            "WHERE d.name='Project_Plan_akt_4.08.2026.txt'"
        ).fetchone()[0]
    return db, lance, embeddings, target


def _fusion_ids(trace):
    return [c["chunk_id"] for c in trace.rrf_candidates]


def _phase3_ids(trace):
    return [c["chunk_id"] for c in trace.candidates_before_precision]


def test_flag_off_top_ids_match_baseline(admission_backend, monkeypatch):
    db, lance, embeddings, target = admission_backend
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", False)
    off = ai_search.SearchTrace()
    ai_search.search(
        "najdi aktuální harmonogram stavby", db, lance, embeddings,
        is_question=False, trace=off, candidate_strategy="legacy",
    )
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", False)
    again = ai_search.SearchTrace()
    ai_search.search(
        "najdi aktuální harmonogram stavby", db, lance, embeddings,
        is_question=False, trace=again, candidate_strategy="legacy",
    )
    assert _phase3_ids(off) == _phase3_ids(again)
    assert target not in _phase3_ids(off) or off.intent["rerank_k"] >= len(off.rrf_candidates)


def test_flag_on_no_intent_same_top_ids(admission_backend, monkeypatch):
    db, lance, embeddings, target = admission_backend
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", False)
    baseline = ai_search.SearchTrace()
    ai_search.search(
        "najdi harmonogram stavby", db, lance, embeddings,
        is_question=False, trace=baseline, candidate_strategy="legacy",
    )
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", True)
    on = ai_search.SearchTrace()
    rows = ai_search.search(
        "najdi harmonogram stavby", db, lance, embeddings,
        is_question=False, trace=on, candidate_strategy="legacy",
    )
    assert _phase3_ids(baseline) == _phase3_ids(on)
    assert _fusion_ids(baseline) == _fusion_ids(on)
    if rows:
        assert "admission_source" not in rows[0]["match"]
        assert "family_revision" not in rows[0]["match"]


def test_flag_on_intent_admits_bm25_only_unique_doc(admission_backend, monkeypatch):
    db, lance, embeddings, target = admission_backend
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", False)
    off = ai_search.SearchTrace()
    ai_search.search(
        "najdi aktuální harmonogram stavby", db, lance, embeddings,
        limit=2, is_question=False, trace=off, candidate_strategy="legacy",
    )
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", True)
    on = ai_search.SearchTrace()
    rows = ai_search.search(
        "najdi aktuální harmonogram stavby", db, lance, embeddings,
        limit=2, is_question=False, trace=on, candidate_strategy="legacy",
    )
    assert _fusion_ids(off) == _fusion_ids(on)
    assert target in {c["chunk_id"] for c in off.bm25_candidates}
    assert target not in _phase3_ids(off)
    assert target in _phase3_ids(on)
    assert target in {c["chunk_id"] for c in on.rerank_candidates}
    assert {r["match"].get("admission_source") for r in rows} <= {"fusion", "bm25_revision_floor"}
    assert "fusion" in {r["match"].get("admission_source") for r in rows}


def test_no_new_retrieval_uses_existing_bm25_list(admission_backend, monkeypatch):
    db, lance, embeddings, _target = admission_backend
    monkeypatch.setattr(ai_search, "FAMILY_REVISION_RERANK_ENABLED", True)
    trace = ai_search.SearchTrace()
    ai_search.search(
        "najdi aktuální harmonogram stavby", db, lance, embeddings,
        is_question=False, trace=trace, candidate_strategy="legacy",
    )
    assert len(trace.bm25_candidates) == trace.intent["retrieval_k"] or len(trace.bm25_candidates) <= 20
    assert trace.metadata["candidate_pool_size_before_truncation"] == len(trace.rrf_candidates)
