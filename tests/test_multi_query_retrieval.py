"""PR2 gated multi-query retrieval: flag, gate, planner, merge, QE reuse.

Does not require the production index for unit cases. Integration pieces stub
ai_search.search so flag-OFF call counts stay identical to the single-query path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import ai_search_config
import query_expansion as qe
import query_facets as qf
import ui_services as ui
from query_facets import (
    FacetType,
    PlannedSubquery,
    extract_facets,
    plan_subqueries,
    should_use_multi_query,
)


DESIGN_QUERY = "bude se brokovat základová deska 3PP"


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def test_gate_design_query_is_multi_concept():
    facets = extract_facets(DESIGN_QUERY)
    types = {f.type for f in facets}
    assert FacetType.ACTION in types
    assert FacetType.OBJECT in types
    assert FacetType.LOCATION in types
    assert should_use_multi_query(facets) is True


def test_gate_simple_filename_lookup_is_false():
    assert should_use_multi_query(extract_facets("Pentaflex")) is False
    assert should_use_multi_query(extract_facets("NOT252167")) is False


def test_gate_bare_bp_is_false():
    assert should_use_multi_query(extract_facets("BP")) is False


def test_gate_kzp_feri_does_not_open_without_second_gate_type():
    """DOC_TYPE + OTHER must not open the gate (OTHER does not count)."""
    facets = extract_facets("KZP FERI")
    gate_types = {f.type for f in facets if f.type in qf.MULTI_QUERY_GATE_TYPES}
    assert gate_types == {FacetType.DOC_TYPE} or FacetType.DOC_TYPE in gate_types
    assert should_use_multi_query(facets) is False


def test_gate_other_alone_never_opens():
    assert should_use_multi_query([]) is False
    assert should_use_multi_query(extract_facets("Pentaflex")) is False
    assert should_use_multi_query(extract_facets("půdorys")) is False


def test_gate_ignores_actor_enum_without_detection():
    # Even if a future ACTOR facet appeared, gate types exclude it.
    synthetic = [
        qf.QueryFacet(FacetType.ACTOR, "FERI", ("feri",), "test", 1.0),
        qf.QueryFacet(FacetType.OTHER, "x", ("x",), "test", 0.6),
    ]
    assert should_use_multi_query(synthetic) is False


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def test_planner_design_query_shapes():
    plan = plan_subqueries(DESIGN_QUERY)
    assert plan[0].id == "full"
    assert plan[0].text == DESIGN_QUERY
    assert len(plan) <= ai_search_config.MAX_SUBQUERIES
    assert len(plan) <= 4
    ids = [p.id for p in plan]
    assert "action" in ids
    assert "object_location" in ids
    action = next(p for p in plan if p.id == "action")
    assert "brokov" in qf._fold(action.text)
    ol = next(p for p in plan if p.id == "object_location")
    folded = qf._fold(ol.text)
    assert "zakladova deska" in folded
    assert "3pp" in folded.replace(" ", "").replace(".", "")


def test_planner_dedupes_and_keeps_full_first():
    plan = plan_subqueries(DESIGN_QUERY)
    texts = [qf._fold(p.text) for p in plan]
    assert len(texts) == len(set(texts))
    assert plan[0].id == "full"


def test_planner_has_no_filename_or_document_id_hacks():
    plan = plan_subqueries(DESIGN_QUERY)
    blob = " ".join(p.text for p in plan).casefold()
    assert "techfloor" not in blob
    assert "d11b" not in blob
    assert "1372" not in blob
    assert ".xls" not in blob
    assert ".pdf" not in blob


def test_planner_single_concept_returns_only_full():
    plan = plan_subqueries("Pentaflex")
    assert plan == [PlannedSubquery(id="full", text="Pentaflex", facet_types=())]


def test_planner_respects_max_subqueries_cap():
    plan = plan_subqueries(
        "kontrolní a zkušební plán brokování základová deska 3PP",
        max_subqueries=2,
    )
    assert len(plan) == 2
    assert plan[0].id == "full"


# ---------------------------------------------------------------------------
# QE reuse on planned legs
# ---------------------------------------------------------------------------

def test_action_subquery_uses_existing_qe_surface_prep():
    plan = plan_subqueries(DESIGN_QUERY)
    action = next(p for p in plan if p.id == "action")
    expansion = qe.expand_query(action.text)
    assert "otryskání" in expansion.terms
    assert "broušení" in expansion.terms


def test_object_location_subquery_gets_floor_normalization():
    plan = plan_subqueries(DESIGN_QUERY)
    ol = next(p for p in plan if p.id == "object_location")
    expansion = qe.expand_query(ol.text)
    assert "3.PP" in expansion.terms or "3PP" in ol.text


# ---------------------------------------------------------------------------
# Feature flag OFF — search_all single-leg path
# ---------------------------------------------------------------------------

def test_flag_default_is_off():
    assert ai_search_config.MULTI_QUERY_RETRIEVAL_ENABLED is False


def test_multi_query_plan_off_is_single_full_leg(monkeypatch):
    monkeypatch.setattr(ai_search_config, "MULTI_QUERY_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(ui, "MULTI_QUERY_RETRIEVAL_ENABLED", False)
    legs = ui._multi_query_search_plan(DESIGN_QUERY, fetch_limit=50)
    assert legs == [("full", DESIGN_QUERY, 50)]


def test_multi_query_plan_on_gated_returns_multiple_legs(monkeypatch):
    monkeypatch.setattr(ai_search_config, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(ui, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    legs = ui._multi_query_search_plan(DESIGN_QUERY, fetch_limit=50)
    assert legs[0] == ("full", DESIGN_QUERY, 50)
    assert len(legs) >= 3
    assert all(leg[2] <= 50 for leg in legs)
    # Facet legs use the smaller budget.
    assert all(leg[2] <= ai_search_config.MULTI_QUERY_FACET_FETCH_LIMIT for leg in legs[1:])


def test_search_all_flag_off_calls_search_once_per_source(monkeypatch, tmp_path):
    monkeypatch.setattr(ai_search_config, "MULTI_QUERY_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(ui, "MULTI_QUERY_RETRIEVAL_ENABLED", False)

    calls: list[tuple[str, int]] = []

    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        calls.append((query, limit))
        return [{
            "document": "a.txt",
            "path": str(tmp_path / "a.txt"),
            "project": "p",
            "quote": "brokovat základová deska",
            "heading": "",
            "score": 1.0,
            "match": {},
        }]

    monkeypatch.setattr(ui.ai_search, "search", fake_search)
    monkeypatch.setattr(ui, "metadata_for", lambda path, source: {
        "source": source, "extension": ".txt", "date": "", "author": "", "availability": "local",
    })
    monkeypatch.setattr(ui, "state_paths", lambda state_dir, source: (tmp_path / f"{source}.db", tmp_path / source))
    (tmp_path / "Dokument.db").write_text("", encoding="utf-8")

    settings = ui.Settings(project_root=str(tmp_path), result_count=10)
    rows = ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)
    assert len(calls) == 1
    assert calls[0][0] == DESIGN_QUERY
    assert len(rows) == 1
    assert "_mq_sources" not in rows[0]


def test_search_all_flag_on_runs_multiple_searches_and_merges(monkeypatch, tmp_path):
    monkeypatch.setattr(ai_search_config, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(ui, "MULTI_QUERY_RETRIEVAL_ENABLED", True)

    calls: list[str] = []

    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        calls.append(query)
        # Same document from every leg → must merge to one row.
        return [{
            "document": "techfloor.xls",
            "path": str(tmp_path / "techfloor.xls"),
            "project": "p",
            "quote": f"hit for {query}",
            "heading": "",
            "score": 0.5 + 0.1 * len(calls),
            "match": {},
        }]

    monkeypatch.setattr(ui.ai_search, "search", fake_search)
    monkeypatch.setattr(ui, "metadata_for", lambda path, source: {
        "source": source, "extension": ".xls", "date": "", "author": "", "availability": "local",
    })
    monkeypatch.setattr(ui, "state_paths", lambda state_dir, source: (tmp_path / f"{source}.db", tmp_path / source))
    (tmp_path / "Dokument.db").write_text("", encoding="utf-8")

    settings = ui.Settings(project_root=str(tmp_path), result_count=10)
    rows = ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)
    assert len(calls) >= 3
    assert calls[0] == DESIGN_QUERY
    assert len(rows) == 1
    assert rows[0]["document"] == "techfloor.xls"
    assert "full" in rows[0].get("_mq_sources", ())


def _stub_search_all_env(monkeypatch, tmp_path, fake_search):
    monkeypatch.setattr(ai_search_config, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(ui, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(ui.ai_search, "search", fake_search)
    monkeypatch.setattr(ui, "metadata_for", lambda path, source: {
        "source": source, "extension": ".txt", "date": "", "author": "", "availability": "local",
    })
    monkeypatch.setattr(ui, "state_paths", lambda state_dir, source: (tmp_path / f"{source}.db", tmp_path / source))
    (tmp_path / "Dokument.db").write_text("", encoding="utf-8")
    return ui.Settings(project_root=str(tmp_path), result_count=10)


def test_q_full_relative_order_is_preserved(monkeypatch, tmp_path):
    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        if query == DESIGN_QUERY:
            return [
                {"document": "full-a.txt", "path": str(tmp_path / "full-a.txt"), "project": "p",
                 "quote": "a", "heading": "", "score": 0.9, "match": {}},
                {"document": "full-b.txt", "path": str(tmp_path / "full-b.txt"), "project": "p",
                 "quote": "b", "heading": "", "score": 0.8, "match": {}},
            ]
        # Facet noise with higher numeric scores must not reorder Q_full pair.
        return [
            {"document": "noise.txt", "path": str(tmp_path / "noise.txt"), "project": "p",
             "quote": "noise", "heading": "", "score": 5.0, "match": {}},
            {"document": "full-b.txt", "path": str(tmp_path / "full-b.txt"), "project": "p",
             "quote": "b-facet", "heading": "", "score": 4.0, "match": {}},
        ]

    settings = _stub_search_all_env(monkeypatch, tmp_path, fake_search)
    rows = ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)
    names = [r["document"] for r in rows]
    assert names.index("full-a.txt") < names.index("full-b.txt")
    assert names[0] == "full-a.txt"
    assert names[1] == "full-b.txt"


def test_facet_only_higher_score_cannot_outrank_q_full(monkeypatch, tmp_path):
    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        if query == DESIGN_QUERY:
            return [{
                "document": "from-full.txt", "path": str(tmp_path / "from-full.txt"), "project": "p",
                "quote": "full leg", "heading": "", "score": 0.2, "match": {},
            }]
        return [{
            "document": "from-facet.txt", "path": str(tmp_path / "from-facet.txt"), "project": "p",
            "quote": "facet leg", "heading": "", "score": 9.9, "match": {},
        }]

    settings = _stub_search_all_env(monkeypatch, tmp_path, fake_search)
    rows = ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)
    names = [r["document"] for r in rows]
    assert names[0] == "from-full.txt"
    assert "from-facet.txt" in names
    assert names.index("from-full.txt") < names.index("from-facet.txt")


def test_full_plus_action_same_document_keeps_full_rank_and_provenance(monkeypatch, tmp_path):
    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        if query == DESIGN_QUERY:
            return [
                {"document": "shared.xls", "path": str(tmp_path / "shared.xls"), "project": "p",
                 "quote": "full quote", "heading": "", "score": 0.4, "match": {}},
                {"document": "other-full.txt", "path": str(tmp_path / "other-full.txt"), "project": "p",
                 "quote": "other", "heading": "", "score": 0.3, "match": {}},
            ]
        return [{
            "document": "shared.xls", "path": str(tmp_path / "shared.xls"), "project": "p",
            "quote": "action quote", "heading": "", "score": 8.0, "match": {},
        }]

    settings = _stub_search_all_env(monkeypatch, tmp_path, fake_search)
    rows = ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)
    shared = [r for r in rows if r["document"] == "shared.xls"]
    assert len(shared) == 1
    assert rows[0]["document"] == "shared.xls"
    assert rows[0]["_mq_sources"][0] == "full"
    assert "action" in rows[0]["_mq_sources"] or "object_location" in rows[0]["_mq_sources"]
    # Facet score must not promote shared ahead of its Q_full-relative place:
    # shared stays before other-full (0.4 > 0.3 on full leg).
    assert rows[1]["document"] == "other-full.txt"


def test_facet_only_fills_when_q_full_underfills_result_count(monkeypatch, tmp_path):
    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        if query == DESIGN_QUERY:
            return [{
                "document": "only-full.txt", "path": str(tmp_path / "only-full.txt"), "project": "p",
                "quote": "full", "heading": "", "score": 0.5, "match": {},
            }]
        return [
            {"document": "facet-1.txt", "path": str(tmp_path / "facet-1.txt"), "project": "p",
             "quote": "f1", "heading": "", "score": 0.9, "match": {}},
            {"document": "facet-2.txt", "path": str(tmp_path / "facet-2.txt"), "project": "p",
             "quote": "f2", "heading": "", "score": 0.8, "match": {}},
        ]

    settings = _stub_search_all_env(monkeypatch, tmp_path, fake_search)
    settings.result_count = 3
    rows = ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)
    assert len(rows) == 3
    assert rows[0]["document"] == "only-full.txt"
    assert {r["document"] for r in rows[1:]} == {"facet-1.txt", "facet-2.txt"}


def test_facet_discovery_document_can_appear(monkeypatch, tmp_path):
    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        if query == DESIGN_QUERY:
            return [{
                "document": "full.txt", "path": str(tmp_path / "full.txt"), "project": "p",
                "quote": "full", "heading": "", "score": 1.0, "match": {},
            }]
        return [{
            "document": "new-from-facet.txt", "path": str(tmp_path / "new-from-facet.txt"), "project": "p",
            "quote": "discovered", "heading": "", "score": 0.1, "match": {},
        }]

    settings = _stub_search_all_env(monkeypatch, tmp_path, fake_search)
    rows = ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)
    assert any(r["document"] == "new-from-facet.txt" for r in rows)
    assert rows[0]["document"] == "full.txt"


def test_merge_tie_break_is_deterministic_by_path(monkeypatch, tmp_path):
    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        if query != DESIGN_QUERY:
            return []
        return [
            {"document": "b.txt", "path": str(tmp_path / "b.txt"), "project": "p",
             "quote": "b", "heading": "", "score": 0.5, "match": {}},
            {"document": "a.txt", "path": str(tmp_path / "a.txt"), "project": "p",
             "quote": "a", "heading": "", "score": 0.5, "match": {}},
        ]

    settings = _stub_search_all_env(monkeypatch, tmp_path, fake_search)
    first = [r["document"] for r in ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)]
    second = [r["document"] for r in ui.search_all(DESIGN_QUERY, settings, tmp_path, embeddings=None, expand_query=False)]
    assert first == second == ["a.txt", "b.txt"]
