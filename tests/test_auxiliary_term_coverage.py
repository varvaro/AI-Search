"""PR5 / PR5.1: Auxiliary Term Coverage — provenance + prefix safety.

Unit tests use fake DF. Production checks pin the live index path because
conftest redirects AI_SEARCH_HOME to a temp directory.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import ai_search
import ai_search_config as cfg
import auxiliary_term_coverage as atc

GOLD_PREFIX = "98cca68b6a4762f65f5406281132e8edb785a074bbd0911d87704c57f84e2405"
CRM_QUERY = "jaký svár je požadovaný na CRM destičky"
DESKA_QUERY = "jaká deska"
_REAL_PROD_DB = (
    Path.home() / "Library/Application Support/AI Search/database/project.sqlite3"
)
_REAL_PROD_LANCE = Path.home() / "Library/Application Support/AI Search/lance/project"


def test_aux_flag_default_off():
    assert cfg.AUXILIARY_TERM_COVERAGE_ENABLED is False
    assert ai_search.AUXILIARY_TERM_COVERAGE_ENABLED is False


# ---------------------------------------------------------------------------
# Prefix safety (PR5.1)
# ---------------------------------------------------------------------------


def test_prefix_rent_banned_short_stem():
    """Rent → Ren* must never be emitted (stem length < 4)."""
    parts = atc._constraint_clause("Rent", df_lookup=lambda t: 1, prefix_df_max=10_000)
    assert parts == ['"Rent"']
    assert not any(p.endswith("*") for p in parts)


def test_prefix_sva_banned_short_stem():
    """svár → svá* banned by stem_len < 4 even if DF lookup would allow it."""
    parts = atc._constraint_clause("svár", df_lookup=lambda t: 1, prefix_df_max=10_000)
    assert parts == ['"svár"']
    assert "svá*" not in parts


def test_prefix_deska_banned_high_df():
    def lookup(t: str) -> int:
        if t == "desk*":
            return 5000
        if t == "deska":
            return 1693
        return 0

    parts = atc._constraint_clause("deska", df_lookup=lookup, prefix_df_max=150)
    assert parts == ['"deska"']
    assert "desk*" not in parts


def test_prefix_long_rare_token_allowed():
    def lookup(t: str) -> int:
        if t.endswith("*"):
            return 40  # under AUX_PREFIX_DF_MAX
        return 10

    parts = atc._constraint_clause("podchycení", df_lookup=lookup, prefix_df_max=150)
    assert '"podchycení"' in parts
    assert any(p.endswith("*") for p in parts)
    assert all(not p.startswith("Ren") for p in parts)


def test_plan_hilti_has_no_ren_star():
    dfs = {"Hilti": 68, "Rent": 853, "podchycení": 120, "sloupů": 200}

    def lookup(t: str) -> int:
        if t.endswith("*"):
            # High DF for short dangerous stems; moderate for longer.
            stem = t[:-1]
            if len(stem) < 4:
                return 9999
            if stem.startswith("desk") or stem.startswith("Ren"):
                return 9999
            return 80
        return dfs.get(t, 0)

    plan = atc.plan_auxiliary_query(
        "dodatek č.1 Hilti Rent podchycení sloupů",
        lookup,
        df_rare_max=200,
        prefix_df_max=150,
    )
    assert plan.activated is True
    assert plan.anchor  # rare content token
    assert "Ren*" not in (plan.match or "")
    # Exact Rent may appear; dangerous short prefix must not.
    assert '"Rent"' in (plan.match or "")


def test_plan_crm_no_sva_star():
    dfs = {"svár": 279, "CRM": 36, "destičky": 12}

    def lookup(t: str) -> int:
        if t.endswith("*"):
            return 2000 if t.startswith("svá") or t.startswith("svar") else 20
        return dfs.get(t, 0)

    plan = atc.plan_auxiliary_query(CRM_QUERY, lookup, df_rare_max=200, prefix_df_max=150)
    assert plan.activated is True
    assert plan.anchor == "CRM"
    assert "svá*" not in (plan.match or "")


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def test_candidate_origin_mapping():
    assert atc.candidate_origin(primary=True, aux_hit=False) == atc.ORIGIN_PRIMARY
    assert atc.candidate_origin(primary=False, aux_hit=True) == atc.ORIGIN_AUXILIARY
    assert atc.candidate_origin(primary=True, aux_hit=True) == atc.ORIGIN_BOTH


def test_plan_deska_only_does_not_activate():
    plan = atc.plan_auxiliary_query(DESKA_QUERY, lambda t: 1693 if t == "deska" else 0, df_rare_max=200)
    assert plan.activated is False


def test_collect_returns_matched_and_added(tmp_path, monkeypatch):
    plan = atc.AuxPlan(
        activated=True,
        match='"CRM" AND ("svár")',
        anchor="CRM",
        constraints=("svár",),
        dfs={"CRM": 36, "svár": 10},
        reason="ok",
    )
    monkeypatch.setattr(atc, "plan_auxiliary_query", lambda *a, **k: plan)
    monkeypatch.setattr(
        atc,
        "run_auxiliary_fts",
        lambda *a, **k: ["already", "new-a", "new-b"],
    )
    result = atc.collect_auxiliary_chunk_ids(
        tmp_path / "dummy.sqlite3",
        CRM_QUERY,
        exclude_ids=["already"],
        df_lookup=lambda t: 1,
        max_new_ids=15,
    )
    assert result.matched_ids == ("already", "new-a", "new-b")
    assert result.added_ids == ("new-a", "new-b")


def test_search_does_not_call_aux_when_flag_off(monkeypatch):
    monkeypatch.setattr(ai_search, "AUXILIARY_TERM_COVERAGE_ENABLED", False)
    with mock.patch.object(
        ai_search.auxiliary_term_coverage,
        "collect_auxiliary_chunk_ids",
        side_effect=AssertionError("aux must not run when flag OFF"),
    ):
        assert ai_search.AUXILIARY_TERM_COVERAGE_ENABLED is False
        if ai_search.AUXILIARY_TERM_COVERAGE_ENABLED:
            ai_search.auxiliary_term_coverage.collect_auxiliary_chunk_ids(Path("x"), "q")


# ---------------------------------------------------------------------------
# Production (optional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _REAL_PROD_DB.exists(), reason="production index not available")
def test_prod_prefix_safety_on_live_df():
    lookup = atc.make_fts_df_lookup(_REAL_PROD_DB)
    assert "Ren*" not in atc._constraint_clause("Rent", df_lookup=lookup, prefix_df_max=cfg.AUX_PREFIX_DF_MAX)
    assert "svá*" not in atc._constraint_clause("svár", df_lookup=lookup, prefix_df_max=cfg.AUX_PREFIX_DF_MAX)
    assert "desk*" not in atc._constraint_clause("deska", df_lookup=lookup, prefix_df_max=cfg.AUX_PREFIX_DF_MAX)


@pytest.mark.skipif(not _REAL_PROD_DB.exists(), reason="production index not available")
def test_prod_crm_trace_and_both_primary_origin(monkeypatch):
    """CRM: aux activates with safe MATCH; overlapping hits are BOTH; fts_hit unchanged.

    Prefix safety bans svá*, so CRM gold weld chunks are not aux-recovered here
    (that is out of PR5.1 scope). Overlap with primary FTS still yields BOTH.
    """
    from ai_search_config import QUERY_EXPANSION_MODE

    monkeypatch.setattr(ai_search, "AUXILIARY_TERM_COVERAGE_ENABLED", True)
    emb = ai_search.Embeddings()
    trace = ai_search.SearchTrace()
    rows = ai_search.search(
        CRM_QUERY,
        _REAL_PROD_DB,
        _REAL_PROD_LANCE,
        emb,
        limit=300,
        is_question=True,
        expand_query=QUERY_EXPANSION_MODE,
        trace=trace,
    )
    aux = (trace.metadata or {}).get("auxiliary_term_coverage") or {}
    assert aux.get("activated") is True
    assert aux.get("anchor") == "CRM"
    assert isinstance(aux.get("constraints"), list)
    assert aux.get("match")
    assert "svá*" not in (aux.get("match") or "")
    assert "Ren*" not in (aux.get("match") or "")
    assert aux.get("matched_count", 0) >= 1

    pool = {c["chunk_id"]: c for c in (trace.candidates_before_precision or [])}
    both = [c for c in pool.values() if c.get("candidate_origin") == "BOTH"]
    assert both, "expected BOTH for aux∩primary CRM hits"
    for c in both:
        assert c["aux_hit"] is True

    primary_like = [c for c in pool.values() if c.get("candidate_origin") in {"PRIMARY", "BOTH"}]
    assert primary_like

    cid84 = f"{GOLD_PREFIX}:84"
    if cid84 in pool:
        assert pool[cid84]["candidate_origin"] in {"PRIMARY", "BOTH"}

    for row in rows:
        m = row.get("match") or {}
        assert "fts_hit" in m and "aux_hit" in m and "candidate_origin" in m
        # fts_hit stays primary-channel only (not redefined by aux).
        if m["candidate_origin"] == "BOTH":
            assert m["aux_hit"] is True
            assert m["fts_hit"] or m["vector_hit"]


@pytest.mark.skipif(not _REAL_PROD_DB.exists(), reason="production index not available")
def test_prod_aux_only_origin_on_added_ids(monkeypatch):
    """Aux-appended ids (not in primary FTS/vector) get candidate_origin=AUXILIARY."""
    from ai_search_config import QUERY_EXPANSION_MODE

    monkeypatch.setattr(ai_search, "AUXILIARY_TERM_COVERAGE_ENABLED", True)
    emb = ai_search.Embeddings()
    trace = ai_search.SearchTrace()
    rows = ai_search.search(
        "dodatek č.1 Hilti Rent podchycení sloupů",
        _REAL_PROD_DB,
        _REAL_PROD_LANCE,
        emb,
        limit=300,
        is_question=True,
        expand_query=QUERY_EXPANSION_MODE,
        trace=trace,
    )
    aux = (trace.metadata or {}).get("auxiliary_term_coverage") or {}
    assert aux.get("activated") is True
    assert "Ren*" not in (aux.get("match") or "")
    assert aux.get("added_count", 0) >= 1

    pool = {c["chunk_id"]: c for c in (trace.candidates_before_precision or [])}
    aux_only_ids = [
        cid for cid in (aux.get("added_ids") or [])
        if pool.get(cid, {}).get("candidate_origin") == "AUXILIARY"
    ]
    assert aux_only_ids, f"expected AUXILIARY added ids; aux={aux}"

    by_id = {r["chunk_id"]: r for r in rows}
    for cid in aux_only_ids:
        if cid not in by_id:
            continue
        m = by_id[cid]["match"]
        assert m["aux_hit"] is True
        assert m["fts_hit"] is False
        assert m["vector_hit"] is False
        assert m["candidate_origin"] == "AUXILIARY"


@pytest.mark.skipif(not _REAL_PROD_DB.exists(), reason="production index not available")
def test_prod_off_match_has_no_aux_fields(monkeypatch):
    from ai_search_config import QUERY_EXPANSION_MODE

    monkeypatch.setattr(ai_search, "AUXILIARY_TERM_COVERAGE_ENABLED", False)
    emb = ai_search.Embeddings()
    rows = ai_search.search(
        CRM_QUERY,
        _REAL_PROD_DB,
        _REAL_PROD_LANCE,
        emb,
        limit=50,
        is_question=True,
        expand_query=QUERY_EXPANSION_MODE,
    )
    assert rows
    m = rows[0]["match"]
    assert "fts_hit" in m
    assert "aux_hit" not in m
    assert "candidate_origin" not in m


@pytest.mark.skipif(not _REAL_PROD_DB.exists(), reason="production index not available")
def test_prod_deska_query_aux_not_activated():
    plan = atc.plan_auxiliary_query(
        DESKA_QUERY,
        atc.make_fts_df_lookup(_REAL_PROD_DB),
        df_rare_max=cfg.AUX_DF_RARE_MAX,
        prefix_df_max=cfg.AUX_PREFIX_DF_MAX,
    )
    assert plan.activated is False
