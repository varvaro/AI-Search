"""PR7.1 span capture: additive `_evidence_spans` on search_all() rows.

The capture exists because search_all() returns ONE row per document whose
`quote` is a merge of several chunks' excerpts — the non-best chunks' identities
are dropped by that merge, so no later layer can tell which chunks the answer
could rest on.

What these tests protect, in priority order:
  1. flag OFF → output byte-identical to pre-PR7.1 (no new key at all)
  2. flag ON  → the ONLY difference is the added key (quote/evidence/score/
     ranking/order all unchanged)
  3. captured chunk identity is real and corresponds to the merged quote
  4. evidence.py stays a pure foundation (no runtime import) and EvidenceSpan's
     new `matched_facets` field is backward compatible

No SQLite/LanceDB/Ollama: ai_search.search is faked, so the merge logic under
test is exercised deterministically.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai_search_config
import ui_services as ui
from evidence import EvidenceSpan, build_evidence_set
from query_facets import FacetType, extract_facets

QUERY = "bude se brokovat základová deska 3PP"

# One multi-chunk document (the interesting case: its merged quote spans several
# chunks) plus one single-chunk document (evidence == [] branch).
CHUNK_A0 = "Otryskání podkladu před provedením lité podlahy."
CHUNK_A1 = "Skladba P3 základová deska ŽB 250 mm ve 3. pp."
CHUNK_A2 = "Broušení a penetrace povrchu po otryskání."
CHUNK_A3 = "Otryskání podkladu před provedením lité podlahy."  # exact duplicate of A0
CHUNK_B0 = "Smluvní strany se dohodly na termínu předání."


def _fake_rows(tmp_path: Path) -> list[dict]:
    doc_a = str(tmp_path / "techfloor.xls")
    doc_b = str(tmp_path / "smlouva.pdf")
    common = {"project": "p", "heading": "", "match": {}}
    return [
        {"document": "techfloor.xls", "path": doc_a, "quote": CHUNK_A0, "score": 3.0,
         "document_id": 1, "chunk_id": "a:0", **common},
        {"document": "techfloor.xls", "path": doc_a, "quote": CHUNK_A1, "score": 2.0,
         "document_id": 1, "chunk_id": "a:1", **common},
        {"document": "techfloor.xls", "path": doc_a, "quote": CHUNK_A2, "score": 1.5,
         "document_id": 1, "chunk_id": "a:2", **common},
        {"document": "techfloor.xls", "path": doc_a, "quote": CHUNK_A3, "score": 1.0,
         "document_id": 1, "chunk_id": "a:3", **common},
        {"document": "smlouva.pdf", "path": doc_b, "quote": CHUNK_B0, "score": 0.9,
         "document_id": 2, "chunk_id": "b:0", **common},
    ]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """search_all() with ai_search.search / metadata_for / state_paths faked."""
    monkeypatch.setattr(ai_search_config, "MULTI_QUERY_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(ui, "MULTI_QUERY_RETRIEVAL_ENABLED", False)

    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        return [dict(row) for row in _fake_rows(tmp_path)]

    monkeypatch.setattr(ui.ai_search, "search", fake_search)
    monkeypatch.setattr(ui, "metadata_for", lambda path, source: {
        "source": source, "extension": Path(path).suffix, "date": "",
        "author": "", "availability": "local",
    })
    monkeypatch.setattr(ui, "state_paths",
                        lambda state_dir, source: (tmp_path / f"{source}.db", tmp_path / source))
    (tmp_path / "Dokument.db").write_text("", encoding="utf-8")

    settings = ui.Settings(project_root=str(tmp_path), result_count=10)

    def run(enabled: bool, is_question: bool = False) -> list[dict]:
        monkeypatch.setattr(ui, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", enabled)
        return ui.search_all(QUERY, settings, tmp_path, embeddings=None,
                             is_question=is_question)

    return run


# ---------------------------------------------------------------------------
# Feature flag / OFF identity
# ---------------------------------------------------------------------------

def test_flag_default_is_off():
    assert ai_search_config.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False
    assert ui.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False


def test_off_adds_no_field_at_all(harness):
    rows = harness(False)
    assert rows
    for row in rows:
        assert "_evidence_spans" not in row


def test_on_differs_from_off_only_by_the_new_key(harness):
    """The byte-identity contract: strip `_evidence_spans` from the ON rows and
    they must equal the OFF rows exactly — same order, quotes, evidence,
    scores, metadata."""
    off = copy.deepcopy(harness(False))
    on = copy.deepcopy(harness(True))

    captured = [row.pop("_evidence_spans") for row in on]
    assert captured and all(spans for spans in captured)  # every row got real spans
    assert on == off


def test_on_is_identical_in_question_mode_too(harness):
    off = copy.deepcopy(harness(False, is_question=True))
    on = copy.deepcopy(harness(True, is_question=True))
    for row in on:
        row.pop("_evidence_spans")
    assert on == off


def test_merged_quote_and_score_are_untouched(harness):
    off = {row["path"]: row for row in harness(False)}
    on = {row["path"]: row for row in harness(True)}

    assert list(on) == list(off)  # result order preserved
    for path, row in on.items():
        assert row["quote"] == off[path]["quote"]
        assert row["score"] == off[path]["score"]
        assert row["best_chunk_score"] == off[path]["best_chunk_score"]
        assert row.get("evidence") == off[path].get("evidence")


# ---------------------------------------------------------------------------
# Captured content
# ---------------------------------------------------------------------------

def _row(rows: list[dict], name: str) -> dict:
    return next(row for row in rows if row["document"] == name)


def test_on_captures_spans_for_the_multi_chunk_document(harness):
    row = _row(harness(True), "techfloor.xls")
    spans = row["_evidence_spans"]

    assert [span["chunk_id"] for span in spans] == ["a:0", "a:1", "a:2"]
    assert [span["rank"] for span in spans] == [0, 1, 2]
    assert all(span["document_id"] == 1 for span in spans)


def test_chunk_identity_matches_the_original_retrieval_rows(harness, tmp_path):
    original = {(r["document_id"], r["chunk_id"]): r for r in _fake_rows(tmp_path)}
    for row in harness(True):
        for span in row["_evidence_spans"]:
            key = (span["document_id"], span["chunk_id"])
            assert key in original, key
            source_row = original[key]
            assert span["path"] == source_row["path"]
            assert span["document"] == source_row["document"]
            # The excerpt comes from this chunk's own text, never another's.
            assert source_row["quote"].startswith(span["quote"])


def test_span_quotes_are_chunk_excerpts_not_the_merged_quote(harness):
    """Regression guard for capture ordering: the base row's `quote` is
    overwritten by the merge, so a span captured too late would carry the merged
    string instead of its own chunk's text."""
    row = _row(harness(True), "techfloor.xls")
    quotes = [span["quote"] for span in row["_evidence_spans"]]

    assert quotes == [CHUNK_A0, CHUNK_A1, CHUNK_A2]
    assert row["quote"] not in quotes
    assert len(row["quote"]) > len(quotes[0])


def test_every_captured_excerpt_appears_in_the_merged_quote(harness):
    for row in harness(True):
        for span in row["_evidence_spans"]:
            assert span["quote"] in row["quote"]


def test_spans_align_one_to_one_with_the_evidence_excerpts(harness):
    row = _row(harness(True), "techfloor.xls")
    assert len(row["_evidence_spans"]) == len(row["evidence"])
    for span, item in zip(row["_evidence_spans"], row["evidence"]):
        assert span["rank"] == item["rank"]
        assert span["quote"] == item["text"]


def test_near_duplicate_chunk_is_not_captured(harness):
    """A chunk dropped by the quote merge's near-duplicate filter contributed
    nothing to the answer, so reporting it would overstate the evidence."""
    row = _row(harness(True), "techfloor.xls")
    assert "a:3" not in [span["chunk_id"] for span in row["_evidence_spans"]]


def test_single_chunk_document_captures_exactly_one_span(harness):
    row = _row(harness(True), "smlouva.pdf")
    assert row.get("evidence") is None  # merge reports nothing for one chunk
    assert row["_evidence_spans"] == [{
        "document_id": 2, "chunk_id": "b:0", "path": row["path"],
        "document": "smlouva.pdf", "quote": CHUNK_B0, "score": 0.9,
        "rank": 0, "source": "Dokument",
    }]


def test_span_score_is_the_chunk_score_not_the_document_score(harness):
    """The document row's score may carry the multi-chunk evidence bonus; a span
    must keep the score of the chunk it describes."""
    row = _row(harness(True), "techfloor.xls")
    assert [span["score"] for span in row["_evidence_spans"]] == [3.0, 2.0, 1.5]
    assert row["score"] >= row["best_chunk_score"] == 3.0


def test_spans_carry_source_metadata(harness):
    row = _row(harness(True), "techfloor.xls")
    assert all(span["source"] == "Dokument" for span in row["_evidence_spans"])
    # Single-leg path has no retrieval-leg provenance to report.
    assert all("_mq_source" not in span for span in row["_evidence_spans"])


def test_spans_are_detached_copies_of_the_rows(harness):
    rows = harness(True)
    row = _row(rows, "techfloor.xls")
    span = row["_evidence_spans"][0]
    span["quote"] = "MUTATED"
    span["score"] = -1.0
    assert row["quote"] != "MUTATED"
    assert row["best_chunk_score"] == 3.0


# ---------------------------------------------------------------------------
# Multi-query path (PR2 flag ON) must capture too
# ---------------------------------------------------------------------------

def test_multi_query_path_also_captures_spans(monkeypatch, tmp_path):
    monkeypatch.setattr(ai_search_config, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(ui, "MULTI_QUERY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(ui, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)

    def fake_search(query, db_path, lance_dir, embeddings, limit=8, **kwargs):
        return [dict(row) for row in _fake_rows(tmp_path)]

    monkeypatch.setattr(ui.ai_search, "search", fake_search)
    monkeypatch.setattr(ui, "metadata_for", lambda path, source: {
        "source": source, "extension": "", "date": "", "author": "", "availability": "local",
    })
    monkeypatch.setattr(ui, "state_paths",
                        lambda state_dir, source: (tmp_path / f"{source}.db", tmp_path / source))
    (tmp_path / "Dokument.db").write_text("", encoding="utf-8")

    settings = ui.Settings(project_root=str(tmp_path), result_count=10)
    rows = ui.search_all(QUERY, settings, tmp_path, embeddings=None, is_question=False)

    row = _row(rows, "techfloor.xls")
    assert row["_evidence_spans"]
    for span in row["_evidence_spans"]:
        assert span["chunk_id"] in {"a:0", "a:1", "a:2", "a:3"}
        assert span["quote"] in row["quote"]
        # PR2 retrieval-leg provenance survives into the span.
        assert span["_mq_source"] in {"full", "action", "object_location", "doc_type"}


def test_multi_query_off_by_default_still_captures(harness):
    """The default production path (single leg) is the one that must work."""
    rows = harness(True)
    assert all("_mq_sources" not in row for row in rows)
    assert all(row["_evidence_spans"] for row in rows)


# ---------------------------------------------------------------------------
# Captured spans are consumable by the foundation layer
# ---------------------------------------------------------------------------

def test_captured_spans_feed_build_evidence_set(harness):
    """The point of the capture: the dicts must be a valid `retrieval_rows`
    input, so a future validation layer needs no adapter and no re-retrieval."""
    row = _row(harness(True), "techfloor.xls")
    evidence_set = build_evidence_set(QUERY, retrieval_rows=row["_evidence_spans"],
                                      facets=extract_facets(QUERY))

    assert [span.chunk_id for span in evidence_set.spans] == ["a:0", "a:1", "a:2"]
    assert all(span.document_id == 1 for span in evidence_set.spans)
    # Chunk-level facet evidence is now separable per chunk instead of being
    # collapsed into one merged quote.
    by_chunk = {span.chunk_id: span.facet_types for span in evidence_set.spans}
    assert FacetType.ACTION in by_chunk["a:0"]
    assert FacetType.OBJECT in by_chunk["a:1"]


# ---------------------------------------------------------------------------
# EvidenceSpan.matched_facets — additive, backward compatible
# ---------------------------------------------------------------------------

def test_evidence_span_is_constructible_without_matched_facets():
    span = EvidenceSpan(
        document_id=1, chunk_id="c:0", path="/p/x.pdf", document="x.pdf",
        quote="q", facet_types=(), subquery_ids=("full",), matched_terms=(),
    )
    assert span.matched_facets == ()
    assert span.score == 0.0


def test_matched_facets_preserve_source_and_confidence():
    """`facet_types` keeps only the type, losing exactly what a consumer needs
    to weigh a match: an exact vocabulary hit (1.0) versus a residual span."""
    rows = [{"document_id": 1, "chunk_id": "c:0", "document": "x.xls", "path": "/p/x.xls",
             "quote": CHUNK_A1, "score": 1.0}]
    spans = build_evidence_set(QUERY, retrieval_rows=rows, facets=extract_facets(QUERY)).spans

    assert spans[0].facet_types  # unchanged signal still present
    assert spans[0].matched_facets
    for facet in spans[0].matched_facets:
        assert facet.type in spans[0].facet_types
        assert facet.source
        assert 0.0 < facet.confidence <= 1.0


def test_matched_facets_are_unioned_when_duplicate_spans_merge():
    """Same chunk returned by two legs merges into one span; facet provenance
    must union and dedupe, like facet_types and matched_terms already do."""
    rows = [
        {"document_id": 1, "chunk_id": "c:0", "document": "x.xls", "path": "/p/x.xls",
         "quote": CHUNK_A0, "score": 1.0, "_mq_source": "action"},
        {"document_id": 1, "chunk_id": "c:0", "document": "x.xls", "path": "/p/x.xls",
         "quote": CHUNK_A1, "score": 2.0, "_mq_source": "object_location"},
    ]
    spans = build_evidence_set(QUERY, retrieval_rows=rows, facets=extract_facets(QUERY)).spans

    assert len(spans) == 1
    facets = spans[0].matched_facets
    assert len(facets) >= 2  # guard: both legs' facets really are present
    assert len(facets) == len(set(facets))
    assert {facet.type for facet in facets} == set(spans[0].facet_types)
    # The distinction facet_types cannot express: an exact vocabulary hit next
    # to a 0.6-confidence residual span.
    assert {facet.confidence for facet in facets} == {1.0, 0.6}


# ---------------------------------------------------------------------------
# Layering guard
# ---------------------------------------------------------------------------

def test_foundation_does_not_import_the_runtime():
    """evidence.py must stay a pure foundation: importing ui_services or
    ai_search here would invert the dependency direction PR7.x relies on."""
    source = (Path(__file__).resolve().parents[1] / "evidence.py").read_text(encoding="utf-8")
    for forbidden in ("import ui_services", "import ai_search", "import streamlit",
                      "import sqlite3", "import lancedb"):
        assert forbidden not in source, forbidden


def test_capture_helpers_do_not_touch_retrieval_or_scoring():
    """The capture must be inert: no import of the scoring/ranking helpers into
    it, and no mutation of the row it reads."""
    row = {"document_id": 5, "chunk_id": "z:1", "path": "/p/z.pdf", "document": "z.pdf",
           "quote": "full chunk text", "score": 7.5, "source": "E-mail"}
    frozen = dict(row)
    span = ui._evidence_span_from_row(row, rank=2, excerpt="full chunk")

    assert row == frozen
    assert span["quote"] == "full chunk"
    assert span["score"] == 7.5
    assert span["rank"] == 2
    assert span["source"] == "E-mail"


def test_capture_ignores_out_of_range_ranks():
    """Defensive: a rank that does not address a captured row must be skipped,
    never fabricate identity from a neighbouring chunk."""
    chunk_rows = [{"document_id": 1, "chunk_id": "a:0", "quote": "x", "score": 1.0}]
    evidence = [{"rank": 0, "text": "x"}, {"rank": 9, "text": "y"}]
    spans = ui._capture_evidence_spans(chunk_rows, evidence, "x ... y")
    assert [span["chunk_id"] for span in spans] == ["a:0"]


def test_capture_of_an_empty_document_is_empty():
    assert ui._capture_evidence_spans([], [], "") == []
