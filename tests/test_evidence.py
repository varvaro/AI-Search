"""EvidenceSet + conservative facet↔quote matching (multi-doc PR3).

Pure unit tests — no SQLite/LanceDB/Ollama. Does not exercise ranking.
"""
from __future__ import annotations

import time

import pytest

from evidence import (
    EvidenceSet,
    EvidenceSpan,
    JoinStatus,
    build_evidence_set,
    match_facet_in_text,
)
from query_facets import FacetType, QueryFacet, extract_facets

DESIGN_QUERY = "bude se brokovat základová deska 3PP"

TECHFLOOR_QUOTE = (
    "Otryskání podkladu před provedením lité podlahy, brokování, broušení. "
    "podlaha ve 3. pp základová deska"
)

D11B_QUOTE = (
    "Skladba P3/P4 epoxidová litá podlaha CemFlow samonivelační vrstva "
    "ŽB cca 250–300 mm"
)


def _facet(facet_type: FacetType, surface: str, *terms: str) -> QueryFacet:
    return QueryFacet(
        type=facet_type,
        surface=surface,
        terms=tuple(terms) if terms else (_fold_surface(surface),),
        source="test",
        confidence=1.0,
    )


def _fold_surface(surface: str) -> str:
    from evidence import _fold
    return _fold(surface)


def _row(
    *,
    document_id: int,
    chunk_id: str,
    quote: str,
    document: str = "doc.pdf",
    path: str = "/tmp/doc.pdf",
    score: float = 1.0,
    mq_source: str | None = "full",
    mq_sources: tuple[str, ...] | None = None,
) -> dict:
    row = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "document": document,
        "path": path,
        "quote": quote,
        "score": score,
        "project": "p",
        "heading": "",
        "match": {},
    }
    if mq_sources is not None:
        row["_mq_sources"] = mq_sources
    elif mq_source is not None:
        row["_mq_source"] = mq_source
    return row


# ---------------------------------------------------------------------------
# Identity + retrieval provenance
# ---------------------------------------------------------------------------

def test_evidence_span_preserves_identities_and_quote():
    facets = extract_facets(DESIGN_QUERY)
    evidence = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[
            _row(
                document_id=1372,
                chunk_id="abc:0",
                quote=TECHFLOOR_QUOTE,
                document="tech.xls",
                path="/tmp/tech.xls",
            ),
        ],
        facets=facets,
    )
    assert len(evidence.spans) == 1
    span = evidence.spans[0]
    assert span.document_id == 1372
    assert span.chunk_id == "abc:0"
    assert span.document == "tech.xls"
    assert "otrysk" in span.quote.casefold() or "brokov" in span.quote.casefold()


def test_retrieval_provenance_subquery_ids():
    facets = extract_facets(DESIGN_QUERY)
    evidence = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[
            _row(document_id=1, chunk_id="c:0", quote=TECHFLOOR_QUOTE, mq_source="action"),
        ],
        facets=facets,
    )
    assert evidence.spans[0].subquery_ids == ("action",)


def test_duplicate_chunk_unions_subquery_provenance():
    facets = extract_facets(DESIGN_QUERY)
    evidence = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[
            _row(document_id=1, chunk_id="c:0", quote=TECHFLOOR_QUOTE, mq_source="full", score=0.5),
            _row(document_id=1, chunk_id="c:0", quote=TECHFLOOR_QUOTE, mq_source="action", score=0.9),
        ],
        facets=facets,
    )
    assert len(evidence.spans) == 1
    assert set(evidence.spans[0].subquery_ids) == {"full", "action"}
    assert evidence.spans[0].score == 0.9


def test_rows_without_ids_are_skipped_no_fake_ids():
    facets = extract_facets(DESIGN_QUERY)
    evidence = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[{"document": "x", "path": "/x", "quote": TECHFLOOR_QUOTE, "score": 1.0}],
        facets=facets,
    )
    assert evidence.spans == ()


# ---------------------------------------------------------------------------
# Facet matching — positives / negatives
# ---------------------------------------------------------------------------

def test_action_positive_otryskani_brokovani():
    facet = _facet(FacetType.ACTION, "brokovat", "brokovani")
    ok, terms = match_facet_in_text(facet, TECHFLOOR_QUOTE)
    assert ok
    assert any("otryskani" in t or "brokov" in t or "brousen" in t for t in terms)


def test_action_negative_tryskove_injektaze():
    facet = _facet(FacetType.ACTION, "brokovat", "brokovani")
    ok, terms = match_facet_in_text(facet, "Provedení tryskových injektáží podzákladí")
    assert ok is False
    assert terms == ()


def test_object_positive_zakladova_deska():
    facet = _facet(FacetType.OBJECT, "základová deska", "zakladova deska")
    ok, terms = match_facet_in_text(facet, "3. pp základová deska")
    assert ok
    assert any("zakladova deska" in t for t in terms)


def test_object_negative_bare_desky():
    facet = _facet(FacetType.OBJECT, "základová deska", "zakladova deska")
    ok, _ = match_facet_in_text(facet, "Dodávka bednicích desky na stavbu")
    assert ok is False


def test_location_normalizes_3pp_forms():
    facet = _facet(FacetType.LOCATION, "3PP", "3PP", "3.PP", "3 PP")
    for quote in ("podlaha ve 3. pp", "řez 3PP", "úroveň 3 PP"):
        ok, terms = match_facet_in_text(facet, quote)
        assert ok, quote
        assert terms


# ---------------------------------------------------------------------------
# Multi-document separation + design fixture
# ---------------------------------------------------------------------------

def test_multi_document_facets_stay_on_their_spans():
    facets = extract_facets(DESIGN_QUERY)
    evidence = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[
            _row(
                document_id=10,
                chunk_id="a:0",
                quote=TECHFLOOR_QUOTE,
                document="techfloor.xls",
                path="/tmp/techfloor.xls",
                mq_source="action",
            ),
            _row(
                document_id=20,
                chunk_id="b:0",
                quote=D11B_QUOTE,
                document="D11B_skladby.pdf",
                path="/tmp/D11B_skladby.pdf",
                mq_source="object_location",
            ),
        ],
        facets=facets,
    )
    by_doc = {span.document: span for span in evidence.spans}
    tech = by_doc["techfloor.xls"]
    d11b = by_doc["D11B_skladby.pdf"]

    assert FacetType.ACTION in tech.facet_types
    assert FacetType.OBJECT in tech.facet_types
    assert FacetType.LOCATION in tech.facet_types

    assert FacetType.ACTION not in d11b.facet_types
    assert FacetType.LOCATION not in d11b.facet_types
    # D11B fixture has no "základová deska" phrase — do not invent OBJECT.
    assert FacetType.OBJECT not in d11b.facet_types


def test_design_fixture_join_complete_without_cross_doc_action_claim():
    facets = extract_facets(DESIGN_QUERY)
    evidence = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[
            _row(document_id=10, chunk_id="a:0", quote=TECHFLOOR_QUOTE, document="tech.xls"),
            _row(document_id=20, chunk_id="b:0", quote=D11B_QUOTE, document="skladby.pdf"),
        ],
        facets=facets,
    )
    assert evidence.join_status is JoinStatus.COMPLETE
    assert evidence.coverage[FacetType.ACTION] is True
    assert evidence.coverage[FacetType.OBJECT] is True
    assert evidence.coverage[FacetType.LOCATION] is True
    # No span may attribute ACTION to the D11B-like document.
    for span in evidence.spans:
        if span.document == "skladby.pdf":
            assert FacetType.ACTION not in span.facet_types


# ---------------------------------------------------------------------------
# Coverage / join status
# ---------------------------------------------------------------------------

def test_join_status_complete_partial_insufficient():
    facets = [
        _facet(FacetType.ACTION, "brokovat"),
        _facet(FacetType.OBJECT, "základová deska"),
        _facet(FacetType.LOCATION, "3PP", "3PP", "3.PP"),
    ]
    complete = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[_row(document_id=1, chunk_id="c:0", quote=TECHFLOOR_QUOTE)],
        facets=facets,
    )
    assert complete.join_status is JoinStatus.COMPLETE

    partial = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[
            _row(document_id=1, chunk_id="c:0", quote="pouze otryskání a brokování podkladu"),
        ],
        facets=facets,
    )
    assert partial.join_status is JoinStatus.PARTIAL
    assert partial.coverage[FacetType.ACTION] is True
    assert partial.coverage[FacetType.LOCATION] is False

    empty = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[_row(document_id=1, chunk_id="c:0", quote="obecné stavební poznámky")],
        facets=facets,
    )
    assert empty.join_status is JoinStatus.INSUFFICIENT


def test_subquery_alone_does_not_assign_facet_types():
    """Retrieval provenance ≠ text evidence."""
    facets = [_facet(FacetType.ACTION, "brokovat")]
    evidence = build_evidence_set(
        "brokovat",
        retrieval_rows=[
            _row(
                document_id=1,
                chunk_id="c:0",
                quote="epoxidová litá podlaha bez přípravy podkladu",
                mq_source="action",
            ),
        ],
        facets=facets,
    )
    assert evidence.spans[0].subquery_ids == ("action",)
    assert evidence.spans[0].facet_types == ()


# ---------------------------------------------------------------------------
# Performance smoke
# ---------------------------------------------------------------------------

def test_build_evidence_set_is_fast_on_small_pool():
    facets = extract_facets(DESIGN_QUERY)
    rows = [
        _row(document_id=i, chunk_id=f"c:{i}", quote=TECHFLOOR_QUOTE if i % 2 == 0 else D11B_QUOTE)
        for i in range(30)
    ]
    t0 = time.perf_counter()
    for _ in range(50):
        build_evidence_set(DESIGN_QUERY, retrieval_rows=rows, facets=facets)
    elapsed_ms = (time.perf_counter() - t0) / 50 * 1000
    assert elapsed_ms < 10.0, elapsed_ms
