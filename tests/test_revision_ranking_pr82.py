"""PR8.2 — Revision-aware ranking unit tests."""
from __future__ import annotations

from datetime import date

import pytest

import ai_search
import ai_search_config
import revision_ranking as rr


class _FakeEmbeddings:
    name = "fake"

    def encode(self, texts, **kw):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "Projekt"; root.mkdir()
    (root / "alpha.txt").write_text("ALPHA unikátní fulltextový výraz.")
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings


def test_flag_default_off():
    assert ai_search_config.REVISION_RANKING_ENABLED is False
    assert ai_search.REVISION_RANKING_ENABLED is False


def test_intent_detection():
    assert rr.query_has_revision_intent("najdi aktuální harmonogram")
    assert rr.query_has_revision_intent("jaký harmonogram platí pro monolit?")
    assert rr.query_has_revision_intent("poslední SoD FERI")
    assert rr.query_has_revision_intent("finální verze smlouvy")
    assert not rr.query_has_revision_intent("harmonogram stavby Garáže")
    assert not rr.query_has_revision_intent("porovnání revizí R1 R2")


def test_no_intent_means_zero_bonus():
    detail = rr.compute_revision_score(
        "harmonogram stavby",
        "HMG NDS SIS - Garáže Smíchov_akt.11.12.25_R3_akt_4.08.2026.pdf",
        "/01_HARMONOGRAM/HMG….pdf",
    )
    assert detail.intent is False
    assert detail.bonus == 0.0


def test_current_akt_document_boosted():
    detail = rr.compute_revision_score(
        "najdi aktuální harmonogram stavby",
        "HMG NDS SIS - Garáže Smíchov_akt.11.12.25_R3_akt_4.08.2026.pdf",
        "/02_REALIZACE/01_HARMONOGRAM/HMG….pdf",
        today=date(2026, 8, 12),
    )
    assert detail.intent is True
    assert detail.bonus > 0
    assert any(s.startswith("boost:akt") for s in detail.signals)


def test_old_folder_penalized_not_boosted():
    detail = rr.compute_revision_score(
        "najdi aktuální výkres výztuže 3.PP",
        "D.1.2.07 - schéma vyztužení 3.PP.pdf",
        "/D12_Statika/OLD/D.1.2.07 - schéma vyztužení 3.PP.pdf",
        today=date(2026, 8, 12),
    )
    assert detail.bonus < 0
    assert any("old" in s for s in detail.signals)


def test_draft_penalized_under_intent():
    detail = rr.compute_revision_score(
        "poslední SoD FERI monolit",
        "NDS_SOD_FERI_návrh.docx",
        "/36_monolit_FERI_NOT251110/návrh/NDS_SOD_FERI_návrh.docx",
    )
    assert detail.bonus < 0
    assert any("draft" in s for s in detail.signals)


def test_final_folder_boosted():
    detail = rr.compute_revision_score(
        "poslední SoD FERI monolit final",
        "NDS_SOD_FERI_NOT251110_10022026_SIS.docx",
        "/36_monolit_FERI_NOT251110/final/NDS_SOD_FERI_NOT251110_10022026_SIS.docx",
        today=date(2026, 8, 12),
    )
    assert detail.bonus > 0
    assert any("final" in s for s in detail.signals)


def test_aktualizace_is_not_akt_currency():
    detail = rr.compute_revision_score(
        "poslední SoD FERI monolit final",
        "SP - stanoviska aktualizace stavu k 10_12_2024_SIS_k_SoD__.xlsx",
        "/dotc. org . SP/SP - stanoviska aktualizace stavu.xlsx",
        today=date(2026, 8, 12),
    )
    assert not any(s.startswith("boost:akt") for s in detail.signals)


def test_score_cap():
    detail = rr.compute_revision_score(
        "aktuální finální verze akt_4.08.2026",
        "doc_akt_4.08.2026_final.pdf",
        "/final/doc_akt_4.08.2026_final.pdf",
        today=date(2026, 8, 12),
    )
    assert abs(detail.bonus) <= rr.REVISION_SCORE_CAP


def test_flag_off_search_identical(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    a = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    b = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    assert [r["score"] for r in a] == [r["score"] for r in b]
    assert "revision" not in a[0]["match"]


def test_non_intent_query_unchanged_with_flag_on(backend, monkeypatch):
    root, state, embeddings = backend
    (root / "HMG_akt_4.08.2026.txt").write_text("harmonogram")
    (root / "HMG_old_2024.txt").write_text("harmonogram")
    ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)

    q = "harmonogram stavby Garáže Smíchov"
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    base = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", True)
    cand = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)
    assert [r["document"] for r in base] == [r["document"] for r in cand]
    assert [r["score"] for r in base] == [r["score"] for r in cand]


def test_intent_prefers_akt_over_plain(backend, monkeypatch):
    root, state, embeddings = backend
    (root / "HMG_plain.txt").write_text("harmonogram stavby obsah")
    (root / "HMG_akt_4.08.2026.txt").write_text("harmonogram stavby obsah")
    ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)

    q = "najdi aktuální harmonogram stavby"
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", False)
    base = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)
    monkeypatch.setattr(ai_search, "REVISION_RANKING_ENABLED", True)
    cand = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)
    assert cand[0]["match"].get("revision_score", 0) != 0 or any(
        "akt" in (r.get("document") or "").casefold() for r in cand[:2]
    )
    # Current file should not fall behind after enabling revision ranking.
    def rank_of(rows, name):
        for i, r in enumerate(rows, 1):
            if name in (r.get("document") or ""):
                return i
        return 99

    assert rank_of(cand, "HMG_akt_4.08.2026.txt") <= rank_of(base, "HMG_akt_4.08.2026.txt")
