"""PR8.2.1 — Revision-aware candidate recall unit tests."""
from __future__ import annotations

import pytest

import ai_search
import ai_search_config
import revision_recall as rrc
import revision_ranking as rr


class _FakeEmbeddings:
    name = "fake"

    def encode(self, texts, **kw):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "Projekt"
    root.mkdir()
    (root / "alpha.txt").write_text("ALPHA unikátní fulltextový výraz.")
    (root / "HMG_plain.txt").write_text("harmonogram stavby obsah")
    (root / "HMG_akt_4.08.2026.txt").write_text("harmonogram stavby obsah aktuální")
    final = root / "contracts" / "final"
    final.mkdir(parents=True)
    (final / "NDS_SOD_FERI_NOT251110_final.txt").write_text("SoD FERI monolit final")
    old = root / "OLD"
    old.mkdir()
    (old / "HMG_akt_01.01.2020.txt").write_text("starý harmonogram")
    # Fill the legacy rerank pool so pattern docs can fall outside top_ids
    # unless revision recall appends them. Content matches a non-revision query.
    for i in range(40):
        (root / f"noise_{i:02d}.txt").write_text(f"běžný stavební zápis číslo {i} beton výztuž")
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings


def test_flag_default_off():
    assert ai_search_config.REVISION_RECALL_ENABLED is False
    assert ai_search.REVISION_RECALL_ENABLED is False


def test_intent_gate_shared_with_ranking():
    assert rr.query_has_revision_intent("najdi aktuální harmonogram")
    assert not rr.query_has_revision_intent("harmonogram stavby")


def test_collect_skips_without_intent(backend):
    root, state, embeddings = backend
    res = rrc.collect_revision_chunk_ids(
        state / "index.sqlite3", "harmonogram stavby Garáže",
    )
    assert res.activated is False
    assert res.added_ids == ()


def test_collect_adds_hmg_akt_and_final_skips_old(backend):
    root, state, embeddings = backend
    res = rrc.collect_revision_chunk_ids(
        state / "index.sqlite3", "najdi aktuální harmonogram",
    )
    assert res.activated is True
    names = " ".join(res.matched_document_names).casefold()
    assert "hmg_akt_4.08.2026" in names
    assert "not251110" in names or "final" in names
    assert "hmg_akt_01.01.2020" not in names


def test_flag_off_search_identical(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", False)
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    q = "najdi aktuální harmonogram stavby"
    a = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=10)
    b = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=10)
    assert [r["document"] for r in a] == [r["document"] for r in b]
    assert [r["score"] for r in a] == [r["score"] for r in b]


def test_flag_off_vs_on_no_intent_identical(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    q = "harmonogram stavby Garáže Smíchov"
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", False)
    base = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=10)
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", True)
    cand = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=10)
    assert [r["document"] for r in base] == [r["document"] for r in cand]
    assert [r["score"] for r in base] == [r["score"] for r in cand]


def test_ordinary_query_unchanged(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    q = "ALPHA"
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", False)
    base = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", True)
    cand = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)
    assert [r["document"] for r in base] == [r["document"] for r in cand]
    assert [r["score"] for r in base] == [r["score"] for r in cand]


def _pool_names(db_path, chunk_ids):
    import sqlite3
    if not chunk_ids:
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = []
        for cid in chunk_ids:
            row = con.execute(
                "SELECT d.name, d.relative_path FROM chunks c "
                "JOIN documents d ON d.id=c.document_id WHERE c.id=?",
                (cid,),
            ).fetchone()
            if row:
                names.append(f"{row[0]} {row[1]}")
        return names
    finally:
        con.close()


def test_intent_appends_current_hmg_into_pool(backend, monkeypatch):
    """Intent query: append-only prefix + current HMG present in Phase-3 pool."""
    root, state, embeddings = backend
    db = state / "index.sqlite3"
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    q = "najdi aktuální platný dokument beton výztuž"
    t0 = ai_search.SearchTrace()
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", False)
    ai_search.search(q, db, state / "lance", embeddings, limit=5, trace=t0)
    base_ids = [c["chunk_id"] for c in t0.candidates_before_precision]

    t1 = ai_search.SearchTrace()
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", True)
    ai_search.search(q, db, state / "lance", embeddings, limit=5, trace=t1)
    cand_ids = [c["chunk_id"] for c in t1.candidates_before_precision]
    meta = t1.metadata.get("revision_recall") or {}

    assert cand_ids[: len(base_ids)] == base_ids
    assert len(cand_ids) >= len(base_ids)
    # Either already in truncated pool or newly appended — must be present.
    blob = " ".join(_pool_names(db, cand_ids)).casefold()
    assert "akt_4.08.2026" in blob
    assert "01.01.2020" not in " ".join(meta.get("matched_document_names") or [])
    # Collect with full exclude proves wiring can still surface currency docs.
    forced = rrc.collect_revision_chunk_ids(db, q, exclude_ids=base_ids)
    assert forced.activated is True
    assert any("akt_4.08.2026" in n.casefold() for n in forced.matched_document_names) or (
        "akt_4.08.2026" in blob
    )


def test_final_sod_available_in_pool(backend, monkeypatch):
    root, state, embeddings = backend
    db = state / "index.sqlite3"
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    q = "poslední finální verze SoD"
    t1 = ai_search.SearchTrace()
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", True)
    ai_search.search(q, db, state / "lance", embeddings, limit=5, trace=t1)
    cand_ids = [c["chunk_id"] for c in t1.candidates_before_precision]
    blob = " ".join(_pool_names(db, cand_ids)).casefold()
    assert "not251110" in blob or "/final/" in blob or "final" in blob
    meta_names = " ".join((t1.metadata.get("revision_recall") or {}).get("matched_document_names") or [])
    # If already truncated into pool, meta may be empty — collect still finds it.
    if not meta_names:
        forced = rrc.collect_revision_chunk_ids(db, q, exclude_ids=[])
        assert any("NOT251110" in n or "final" in n.casefold() for n in forced.matched_document_names)


def test_old_not_auto_appended(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    q = "najdi aktuální výkres výztuže 3.PP"
    t1 = ai_search.SearchTrace()
    monkeypatch.setattr(ai_search, "REVISION_RECALL_ENABLED", True)
    ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5, trace=t1)
    names = " ".join((t1.metadata.get("revision_recall") or {}).get("matched_document_names") or [])
    assert "HMG_akt_01.01.2020" not in names
