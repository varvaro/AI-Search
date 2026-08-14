"""PR8.1.2 — Subject Entity Alias unit tests."""
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
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings


def test_subject_alias_flag_default_off():
    assert ai_search_config.SUBJECT_ENTITY_ALIAS_ENABLED is False
    assert ai_search.SUBJECT_ENTITY_ALIAS_ENABLED is False


def test_bozp_conjunction_injects_safetypeak():
    signals = emb.extract_subject_alias_signals("je podepsaná smlouva na BOZP?")
    raws = {s.raw for s in signals}
    assert "SafetyPeak" in raws
    assert "NOT250060" in raws
    assert all(s.source == emb.SOURCE_SUBJECT_ALIAS for s in signals)


def test_single_generic_words_do_not_activate():
    assert emb.extract_subject_alias_signals("najdi smlouvu") == ()
    assert emb.extract_subject_alias_signals("jaké těsnění použít?") == ()
    assert emb.extract_subject_alias_signals("monolitická deska") == ()
    assert emb.extract_subject_alias_signals("jen BOZP školení") == ()


def test_white_tank_sealing_and_monolith_supplier():
    ill = emb.extract_subject_alias_signals("kdo dělá těsnění bílé vany?")
    assert {s.raw for s in ill} >= {"Illichman", "NOT260916"}
    feri = emb.extract_subject_alias_signals("kdo je dodavatel monolitu?")
    assert {s.raw for s in feri} >= {"FERI", "NOT251110"}


def test_subject_bonus_cap_and_source_logged():
    detail = emb.compute_entity_match_bonus(
        "je podepsaná smlouva na BOZP?",
        "NOT250060_BOZP_SafetyPeak_podepsaná.pdf",
        "/07_BOZP_ NOT250060_SafetyPeak/podepsaná/",
        include_explicit=False,
        include_subject_aliases=True,
    )
    assert detail.bonus > 0
    assert detail.bonus <= emb.ENTITY_MATCH_BONUS_CAP
    assert emb.SOURCE_SUBJECT_ALIAS in detail.hit_sources
    assert detail.as_trace_dict()["hit_sources"]


def test_tesneni_without_bila_vana_no_bonus():
    detail = emb.compute_entity_match_bonus(
        "jaké těsnění je v pracovní spáře?",
        "NOT260916_Illichman.pdf",
        "/59_Illichman_NOT260916/",
        include_explicit=False,
        include_subject_aliases=True,
    )
    assert detail.signals == ()
    assert detail.bonus == 0.0


def test_both_flags_off_search_unchanged(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "ENTITY_MATCH_BONUS_ENABLED", False)
    monkeypatch.setattr(ai_search, "SUBJECT_ENTITY_ALIAS_ENABLED", False)
    a = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    b = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    assert [r["score"] for r in a] == [r["score"] for r in b]
    assert "entity_match" not in a[0]["match"]


def test_subject_flag_on_changes_only_ranking(backend, monkeypatch):
    root, state, embeddings = backend
    (root / "NOT250060_BOZP_SafetyPeak_demo.txt").write_text("obecný obsah bez BOZP ve textu")
    (root / "jiny.txt").write_text("podepsaná smlouva na BOZP zmínka v těle dokumentu")
    ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)

    monkeypatch.setattr(ai_search, "ENTITY_MATCH_BONUS_ENABLED", False)
    monkeypatch.setattr(ai_search, "SUBJECT_ENTITY_ALIAS_ENABLED", False)
    base = ai_search.search(
        "je podepsaná smlouva na BOZP?",
        state / "index.sqlite3", state / "lance", embeddings, limit=5, is_question=True,
    )
    monkeypatch.setattr(ai_search, "SUBJECT_ENTITY_ALIAS_ENABLED", True)
    cand = ai_search.search(
        "je podepsaná smlouva na BOZP?",
        state / "index.sqlite3", state / "lance", embeddings, limit=5, is_question=True,
    )
    assert cand
    assert cand[0]["match"].get("entity_match_bonus", 0) > 0
    assert emb.SOURCE_SUBJECT_ALIAS in (
        cand[0]["match"].get("entity_match") or {}
    ).get("hit_sources", [])
    assert "SafetyPeak" in cand[0]["document"] or "NOT250060" in cand[0]["document"]
    # Same candidate pool size — only scores/order may change.
    assert len(base) == len(cand)


def test_answer_untouched_by_subject_flag(monkeypatch):
    monkeypatch.setattr(ai_search, "SUBJECT_ENTITY_ALIAS_ENABLED", True)
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
        "document": "x.pdf", "path": "/p/x.pdf", "project": "P", "heading": "",
        "quote": "t", "score": 1.0, "document_id": 1, "chunk_id": "c0",
        "match": {"fts_hit": True, "vector_hit": False, "semantic_similarity": 0.1,
                  "filename_match": False},
    }]
    result = ai_search.answer("je podepsaná smlouva na BOZP?", rows)
    assert "entity_match" not in result
