"""Unit tests for deterministic query facet extraction (multi-doc PR1).

These tests do not open SQLite/LanceDB, call search(), or invoke Ollama.
Facet detection must stay decoupled from query expansion emits and from
abbreviation filename bonuses.
"""
from __future__ import annotations

import time

import pytest

import query_expansion as qe
import query_facets as qf
from query_facets import FacetType, MAX_QUERY_FACETS, QueryFacet, extract_facets


def _by_type(facets: list[QueryFacet]) -> dict[FacetType, list[QueryFacet]]:
    out: dict[FacetType, list[QueryFacet]] = {}
    for facet in facets:
        out.setdefault(facet.type, []).append(facet)
    return out


def _surfaces(facets: list[QueryFacet], facet_type: FacetType) -> list[str]:
    return [f.surface for f in facets if f.type == facet_type]


def _all_surfaces(facets: list[QueryFacet]) -> str:
    return " ".join(f.surface for f in facets)


# ---------------------------------------------------------------------------
# A–I acceptance cases
# ---------------------------------------------------------------------------

def test_brokovat_zakladova_deska_3pp_splits_action_object_location():
    facets = extract_facets("bude se brokovat základová deska 3PP")
    by = _by_type(facets)

    assert FacetType.ACTION in by
    assert any(qf._fold(f.surface).startswith("brokov") for f in by[FacetType.ACTION])
    assert any("brokov" in t for f in by[FacetType.ACTION] for t in f.terms)

    assert FacetType.OBJECT in by
    object_surfaces = _surfaces(facets, FacetType.OBJECT)
    assert any(qf._fold(s) == "zakladova deska" for s in object_surfaces), object_surfaces
    assert not any(qf._fold(s) == "zakladova" for s in object_surfaces)
    assert not any(qf._fold(s) == "deska" for s in object_surfaces)

    assert FacetType.LOCATION in by
    assert any(qf._fold(f.surface).replace(" ", "").replace(".", "") == "3pp" for f in by[FacetType.LOCATION])
    assert "3.PP" in by[FacetType.LOCATION][0].terms or "3PP" in by[FacetType.LOCATION][0].terms

    # No single facet swallows the whole query.
    assert all(qf._fold(f.surface) != qf._fold("bude se brokovat základová deska 3PP") for f in facets)


def test_pudorys_3pp_bludne_proudy_location_without_bp_false_positive():
    facets = extract_facets("půdorys 3PP bludné proudy")
    by = _by_type(facets)

    assert FacetType.LOCATION in by
    assert any("3" in f.surface and "PP" in f.surface.upper() for f in by[FacetType.LOCATION])

    blob = _all_surfaces(facets).casefold()
    assert "bludné" in blob or "bludne" in qf._fold(blob)
    assert "proudy" in blob or "proud" in qf._fold(blob)

    # Must not invent a bare BP abbreviation facet (QE also refuses bare BP).
    assert not any(f.surface.strip().upper() == "BP" for f in facets)
    assert not any(t.upper() == "BP" for f in facets for t in f.terms)


def test_kzp_monolit_feri_keeps_doc_type_and_residuals():
    facets = extract_facets("kontrolní a zkušební plán monolit FERI")
    by = _by_type(facets)

    assert FacetType.DOC_TYPE in by
    assert any("kzp" in f.source or "kzp" in " ".join(f.terms) for f in by[FacetType.DOC_TYPE])
    assert any("kontrolní" in f.surface.casefold() for f in by[FacetType.DOC_TYPE])

    blob = qf._fold(_all_surfaces(facets))
    assert "monolit" in blob
    assert "feri" in blob


def test_prehled_zmenovych_listu_gd_keeps_zl_and_gd():
    facets = extract_facets("přehled změnových listů GD")
    blob = qf._fold(_all_surfaces(facets))
    assert "gd" in blob
    assert FacetType.DOC_TYPE in _by_type(facets) or "zmenov" in blob
    # GD must not disappear into a stopword drop.
    assert any("GD" in f.surface or qf._fold(f.surface) == "gd" for f in facets)


def test_predavaci_protokol_keeps_multiword_phrase():
    facets = extract_facets("předávací protokol Zakládání Group")
    surfaces = [qf._fold(f.surface) for f in facets]
    assert any(s == "predavaci protokol" for s in surfaces), surfaces
    # Must not shatter into only unrelated single tokens without the phrase.
    assert not (
        "predavaci" in surfaces
        and "protokol" in surfaces
        and "predavaci protokol" not in surfaces
    )
    blob = qf._fold(_all_surfaces(facets))
    assert "zakladani" in blob
    assert "group" in blob


def test_faktura_nazarenko_keeps_identifier_and_name():
    facets = extract_facets("faktura Nazarenko NOT252167")
    blob = qf._fold(_all_surfaces(facets))
    assert "nazarenko" in blob
    assert "not252167" in blob
    assert any(f.type is FacetType.DOC_TYPE for f in facets) or "faktura" in blob
    # Identifier must stay a separate OTHER span, not glued onto the name.
    assert any(
        f.type is FacetType.OTHER and qf._fold(f.surface) == "not252167"
        for f in facets
    )
    assert not any("nazarenko not252167" in qf._fold(f.surface) for f in facets)


def test_simple_lookup_pentaflex_is_not_forced_into_action_location_doc_type():
    facets = extract_facets("Pentaflex")
    assert facets
    assert all(f.type is FacetType.OTHER for f in facets)
    assert facets[0].surface == "Pentaflex"


def test_empty_query_returns_empty_list():
    assert extract_facets("") == []
    assert extract_facets("   ") == []
    assert extract_facets(None) == []  # type: ignore[arg-type]


def test_case_and_diacritics_variants_same_concepts():
    a = extract_facets("BROKOVÁNÍ podkladu")
    b = extract_facets("brokovani podkladu")
    assert {f.type for f in a} == {f.type for f in b}
    assert any(f.type is FacetType.ACTION for f in a)
    # Surface preserves the original span casing/diacritics where practical.
    assert any("BROKOVÁNÍ" in f.surface for f in a)


def test_floor_np_variants_detected_as_location():
    for query, needle in (
        ("půdorys 2NP", "2NP"),
        ("půdorys 2.NP", "2.NP"),
        ("půdorys 2 NP", "2 NP"),
    ):
        facets = extract_facets(query)
        locs = [f for f in facets if f.type is FacetType.LOCATION]
        assert locs, query
        assert any(needle.replace(" ", "") in f.surface.replace(" ", "") for f in locs), (query, locs)


# ---------------------------------------------------------------------------
# Negative: short abbreviations — no QE / filename side effects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("abbr", ["BP", "TP", "KD", "ZL"])
def test_bare_abbreviation_does_not_touch_query_expansion_or_filename_needles(abbr):
    facets = extract_facets(abbr)
    # Facet extraction itself is allowed to classify DOC_TYPE for real abbr
    # keys (TP/ZL/KD), but must never invoke expansion side effects.
    expansion = qe.expand_query(abbr)
    if abbr == "BP":
        assert expansion.terms == []
        assert not any(f.surface.upper() == "BP" and f.type is FacetType.DOC_TYPE for f in facets)
    # Filename bonus needles stay governed solely by expand_query output.
    import ai_search

    assert ai_search._abbreviation_filename_needles(expansion) == (
        set() if abbr != "KZP" else {"kzp"}
    )


def test_facet_module_does_not_change_expansion_terms_for_design_query():
    query = "bude se brokovat základová deska 3PP"
    before = qe.expand_query(query).terms
    extract_facets(query)  # must be side-effect free w.r.t. QE
    after = qe.expand_query(query).terms
    assert before == after


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "bude se brokovat základová deska 3PP",
        "půdorys 3PP bludné proudy",
        "kontrolní a zkušební plán monolit FERI",
        "přehled změnových listů GD",
        "předávací protokol Zakládání Group",
        "faktura Nazarenko NOT252167",
        "Pentaflex",
        "TP",
    ],
)
def test_extract_facets_is_deterministic(query):
    assert extract_facets(query) == extract_facets(query)


@pytest.mark.parametrize(
    "query",
    [
        "bude se brokovat základová deska 3PP",
        "půdorys 3PP bludné proudy",
        "kontrolní a zkušební plán monolit FERI",
        "přehled změnových listů GD",
        "předávací protokol Zakládání Group",
        "faktura Nazarenko NOT252167",
    ],
)
def test_surfaces_nonempty_bounded_and_nonoverlapping(query):
    facets = extract_facets(query)
    assert len(facets) <= MAX_QUERY_FACETS
    assert all(f.surface.strip() for f in facets)
    assert all(f.source.strip() for f in facets)

    # Non-overlapping char spans for surfaces that occur in the query.
    spans: list[tuple[int, int]] = []
    for facet in facets:
        start = query.find(facet.surface)
        if start < 0:
            continue
        end = start + len(facet.surface)
        for a, b in spans:
            overlap = not (end <= a or start >= b)
            assert not overlap, (facet.surface, spans)
        spans.append((start, end))

    # No duplicate (type, surface, source) triples.
    keys = [(f.type, f.surface, f.source) for f in facets]
    assert len(keys) == len(set(keys))


def test_actor_enum_exists_but_is_not_heuristically_assigned():
    """PR1 reserves ACTOR but must not guess firms from capitalization."""
    facets = extract_facets("FERI monolit")
    assert all(f.type is not FacetType.ACTOR for f in facets)
    assert FacetType.ACTOR in FacetType.__members__


def test_max_facets_bound_on_long_query():
    query = " ".join(
        [
            "kontrolní a zkušební plán",
            "technologický postup",
            "změnový list",
            "smlouva o dílo",
            "brokování",
            "3PP",
            "2NP",
            "bludné proudy",
            "předávací protokol",
            "faktura",
            "extra one two three four",
        ]
    )
    facets = extract_facets(query)
    assert len(facets) <= MAX_QUERY_FACETS
