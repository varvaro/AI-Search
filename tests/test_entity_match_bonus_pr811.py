"""PR8.1.1 — Entity Match Bonus unit tests.

Covers: flag OFF identity, scoring math/cap, Unicode folding, answer() untouched,
and a synthetic ranking improvement for an explicit entity query.
"""
from __future__ import annotations

import json

import pytest

import ai_search
import ai_search_config
import entity_match_bonus as emb


class _FakeEmbeddings:
    name = "fake"

    def encode(self, texts, **kw):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def backend(tmp_path, monkeypatch):
    root = tmp_path / "Projekt"; root.mkdir()
    (root / "alpha.txt").write_text("ALPHA unikátní fulltextový výraz.")
    (root / "beta.txt").write_text("BETA obecný obsah.")
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings, None


def test_flag_default_is_off():
    assert ai_search_config.ENTITY_MATCH_BONUS_ENABLED is False
    assert ai_search.ENTITY_MATCH_BONUS_ENABLED is False


def test_extract_not_ids_and_tokens():
    signals = emb.extract_entity_signals(
        "zakázky Stafitech zdění NOT250039 NOT250304"
    )
    not_ids = {s.raw.upper() for s in signals if s.kind == "not_id"}
    tokens = {s.folded for s in signals if s.kind == "token"}
    assert "NOT250039" in not_ids
    assert "NOT250304" in not_ids
    assert "stafitech" in tokens
    assert "zakazky" not in tokens  # stopword-ish / filtered if in stop list
    assert all(s.folded not in emb._STOPWORDS for s in signals)


def test_name_hit_beats_path_only_and_cap():
    detail = emb.compute_entity_match_bonus(
        "SafetyPeak NOT250060 Illichman NOT260916 Stafitech",
        document_name="NOT250060_BOZP_SafetyPeak_podepsaná.pdf",
        document_path="/proj/59_Illichman_NOT260916/x.pdf",
    )
    # name: SafetyPeak + NOT250060; path-only: Illichman + NOT260916; Stafitech miss
    # 0.04+0.04+0.02+0.02 = 0.12 → cap 0.06
    assert detail.bonus == pytest.approx(emb.ENTITY_MATCH_BONUS_CAP)
    assert "SafetyPeak" in detail.name_hits or any("safetypeak" in emb.fold(h) for h in detail.name_hits)
    assert detail.path_hits  # Illichman / NOT260916 path-only


def test_path_only_bonus():
    detail = emb.compute_entity_match_bonus(
        "Illichman NOT260916",
        document_name="nabidka.xlsx",
        document_path="/59_tešnění bílé vany_Illichman_NOT260916/NOT260916.pdf",
    )
    assert detail.bonus == pytest.approx(emb.ENTITY_PATH_ONLY_BONUS * 2)
    assert not detail.name_hits
    assert len(detail.path_hits) == 2


def test_unicode_folding_bicik():
    detail = emb.compute_entity_match_bonus(
        "zakázky Bičík NOT260609",
        document_name="NDS_NOT260609_Bicik_zabor.pdf",
        document_path="/51_inzynyring/",
    )
    assert detail.bonus >= emb.ENTITY_NAME_HIT_BONUS
    assert detail.name_hits


def test_no_signals_means_zero_bonus():
    detail = emb.compute_entity_match_bonus(
        "je na boxu podepsaná smlouva?",
        "SOD_HAUS365_podepsané.pdf",
        "/proj/SOD_HAUS365_podepsané.pdf",
    )
    assert detail.signals == ()
    assert detail.bonus == 0.0


def test_answer_module_untouched_by_entity_flag(monkeypatch):
    """answer() must ignore ENTITY_MATCH_BONUS_ENABLED entirely."""
    monkeypatch.setattr(ai_search, "ENTITY_MATCH_BONUS_ENABLED", True)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)

    payload = json.dumps({
        "body": [{"text": "Odpověď.", "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": payload}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    rows = [{
        "document": "SOD_HAUS365.pdf", "path": "/p/SOD_HAUS365.pdf", "project": "P",
        "heading": "", "quote": "text", "score": 1.0,
        "document_id": 1, "chunk_id": "c0",
        "match": {"fts_hit": True, "vector_hit": False, "semantic_similarity": 0.1,
                  "filename_match": False},
    }]
    result = ai_search.answer("technický dotaz bez entity", rows)
    assert "entity_match" not in result
    assert "Odpověď" in result["answer"] or "odpověď" in result["answer"].casefold()


def test_flag_off_search_match_has_no_entity_keys(backend, monkeypatch):
    monkeypatch.setattr(ai_search, "ENTITY_MATCH_BONUS_ENABLED", False)
    _, state, embeddings, _ = backend
    rows = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    assert rows
    assert "entity_match_bonus" not in rows[0]["match"]
    assert "entity_match" not in rows[0]["match"]


def test_flag_on_boosts_entity_filename(backend, monkeypatch):
    root, state, embeddings, _ = backend
    (root / "NOT259999_SafetyPeak_demo.txt").write_text("obecný obsah bez entity ve textu")
    (root / "jiny_dokument.txt").write_text("obecný obsah SafetyPeak zmínka v těle")
    ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)

    monkeypatch.setattr(ai_search, "ENTITY_MATCH_BONUS_ENABLED", False)
    base = ai_search.search(
        "SafetyPeak NOT259999 smlouva",
        state / "index.sqlite3", state / "lance", embeddings, limit=5,
    )
    monkeypatch.setattr(ai_search, "ENTITY_MATCH_BONUS_ENABLED", True)
    cand = ai_search.search(
        "SafetyPeak NOT259999 smlouva",
        state / "index.sqlite3", state / "lance", embeddings, limit=5,
    )
    assert cand
    assert cand[0]["match"].get("entity_match_bonus", 0) > 0
    assert "NOT259999_SafetyPeak_demo.txt" in [r["document"] for r in cand[:3]]


def test_flag_off_byte_identical_scores_on_fixture(backend, monkeypatch):
    """Two OFF runs must be identical; ON may differ only via entity_match keys."""
    _, state, embeddings, _ = backend
    monkeypatch.setattr(ai_search, "ENTITY_MATCH_BONUS_ENABLED", False)
    a = ai_search.search("ALPHA unikátní", state / "index.sqlite3", state / "lance", embeddings)
    b = ai_search.search("ALPHA unikátní", state / "index.sqlite3", state / "lance", embeddings)
    assert [r["document"] for r in a] == [r["document"] for r in b]
    assert [r["score"] for r in a] == [r["score"] for r in b]
