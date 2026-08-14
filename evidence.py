"""EvidenceSet: facet-linked provenance over retrieval spans (multi-doc PR3).

Pure, deterministic layer. Does not call search, change ranking, invoke an LLM,
or invent document/chunk identities.

    facets + retrieval rows → build_evidence_set() → EvidenceSet

Two provenance layers (kept separate on purpose):

  * subquery_ids  - retrieval provenance (which search leg returned the row)
  * facet_types   - text evidence (quote actually contains the facet term)

A row found by an ACTION subquery is NOT automatically ACTION evidence.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

import query_expansion as qe
from query_facets import (
    MULTI_QUERY_GATE_TYPES,
    FacetType,
    QueryFacet,
    extract_facets,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Whole-word floor forms for a single level (PP / NP), after folding.
_FLOOR_IN_TEXT = re.compile(r"(?<!\w)(\d+)\s*\.?\s*(pp|np)(?!\w)", re.IGNORECASE)


class JoinStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class EvidenceSpan:
    """One chunk-level evidence item with dual provenance."""

    document_id: int
    chunk_id: str
    path: str
    document: str
    quote: str
    facet_types: tuple[FacetType, ...]
    subquery_ids: tuple[str, ...]
    matched_terms: tuple[str, ...]
    score: float = 0.0
    # PR7.1, additive: the QueryFacet objects whose terms were verified in this
    # span's `quote`. `facet_types` keeps only the type, which discards the two
    # attributes a consumer needs to weigh a match — `source` (exact vocabulary
    # hit vs. residual span) and `confidence` (1.0 / 0.8 / 0.6). Defaults to ()
    # so every existing constructor call and positional usage keeps working;
    # `facet_types` is unchanged and remains the coverage signal.
    matched_facets: tuple[QueryFacet, ...] = ()


@dataclass(frozen=True)
class EvidenceSet:
    """Facet coverage over real retrieval spans — no synthetic join claims."""

    query: str
    facets: tuple[QueryFacet, ...]
    spans: tuple[EvidenceSpan, ...]
    coverage: dict[FacetType, bool]
    join_status: JoinStatus

    @property
    def required_facets(self) -> tuple[FacetType, ...]:
        return tuple(self.coverage.keys())


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_fold(text))


def _phrase_terms(facet: QueryFacet) -> list[str]:
    """Surface + facet.terms + QE emits for the surface (existing lexicon only)."""
    terms: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        folded = _fold(raw).strip()
        if not folded or folded in seen:
            return
        seen.add(folded)
        terms.append(folded)

    _add(facet.surface)
    for term in facet.terms:
        _add(term)
    # Reuse QE for the surface only - never invents synonyms outside the lexicon.
    if facet.surface.strip():
        expansion = qe.expand_query(facet.surface.strip())
        for term in expansion.terms:
            _add(term)
    return terms


def _consecutive_phrase_in_tokens(phrase: str, text_tokens: list[str]) -> bool:
    words = phrase.split()
    if not words:
        return False
    if len(words) == 1:
        return words[0] in text_tokens
    n = len(words)
    for index in range(len(text_tokens) - n + 1):
        if text_tokens[index:index + n] == words:
            return True
    return False


def _location_level(facet: QueryFacet) -> str | None:
    for candidate in (facet.surface, *facet.terms):
        match = _FLOOR_IN_TEXT.search(candidate or "")
        if match:
            return match.group(1)
    return None


def _match_phrases_in_text(
    facet: QueryFacet,
    phrases: list[str],
    text: str,
) -> tuple[bool, tuple[str, ...]]:
    if not (text or "").strip():
        return False, ()

    text_tokens = _tokens(text)
    matched: list[str] = []

    if facet.type is FacetType.LOCATION:
        level = _location_level(facet)
        if level is not None:
            for match in _FLOOR_IN_TEXT.finditer(text):
                if match.group(1) == level:
                    matched.append(_fold(match.group(0)).replace(" ", ""))
            if matched:
                return True, tuple(dict.fromkeys(matched))

    for phrase in phrases:
        if _consecutive_phrase_in_tokens(phrase, text_tokens):
            matched.append(phrase)

    if not matched:
        return False, ()
    return True, tuple(dict.fromkeys(matched))


def match_facet_in_text(facet: QueryFacet, text: str) -> tuple[bool, tuple[str, ...]]:
    """Conservative whole-phrase / whole-token match of a facet against quote text.

    False negatives preferred over false positives:
      * no bare substring stems ("trysk" ⊄ otryskání)
      * multi-word OBJECT requires consecutive tokens ("desky" ≠ "základová deska")
      * LOCATION matches N.PP / N PP / NPP for the facet's level only
    """
    return _match_phrases_in_text(facet, _phrase_terms(facet), text)


def _subquery_ids_from_row(row: dict) -> tuple[str, ...]:
    if row.get("_mq_source"):
        return (str(row["_mq_source"]),)
    sources = row.get("_mq_sources")
    if sources:
        return tuple(str(item) for item in sources)
    return ("full",)


def _required_facet_types(facets: list[QueryFacet] | tuple[QueryFacet, ...]) -> tuple[FacetType, ...]:
    ordered: list[FacetType] = []
    seen: set[FacetType] = set()
    for facet in facets:
        if facet.type in MULTI_QUERY_GATE_TYPES and facet.type not in seen:
            seen.add(facet.type)
            ordered.append(facet.type)
    return tuple(ordered)


def _join_status(coverage: dict[FacetType, bool]) -> JoinStatus:
    if not coverage:
        return JoinStatus.INSUFFICIENT
    values = list(coverage.values())
    if all(values):
        return JoinStatus.COMPLETE
    if any(values):
        return JoinStatus.PARTIAL
    return JoinStatus.INSUFFICIENT


def _merge_duplicate_spans(spans: list[EvidenceSpan]) -> list[EvidenceSpan]:
    """One span per (document_id, chunk_id); union provenance and facet_types."""
    merged: dict[tuple[int, str], EvidenceSpan] = {}
    order: list[tuple[int, str]] = []
    for span in spans:
        key = (span.document_id, span.chunk_id)
        existing = merged.get(key)
        if existing is None:
            merged[key] = span
            order.append(key)
            continue
        facet_types = tuple(dict.fromkeys([*existing.facet_types, *span.facet_types]))
        subquery_ids = tuple(dict.fromkeys([*existing.subquery_ids, *span.subquery_ids]))
        matched_terms = tuple(dict.fromkeys([*existing.matched_terms, *span.matched_terms]))
        # QueryFacet is a frozen dataclass, so identical facets dedupe by value.
        matched_facets = tuple(dict.fromkeys([*existing.matched_facets, *span.matched_facets]))
        quote = existing.quote if len(existing.quote) >= len(span.quote) else span.quote
        score = max(existing.score, span.score)
        merged[key] = EvidenceSpan(
            document_id=existing.document_id,
            chunk_id=existing.chunk_id,
            path=existing.path or span.path,
            document=existing.document or span.document,
            quote=quote,
            facet_types=facet_types,
            subquery_ids=subquery_ids,
            matched_terms=matched_terms,
            score=score,
            matched_facets=matched_facets,
        )
    return [merged[key] for key in order]


def build_evidence_set(
    query: str,
    retrieval_rows: list[dict] | None = None,
    facets: list[QueryFacet] | None = None,
) -> EvidenceSet:
    """Build an EvidenceSet from real retrieval rows. Pure — no I/O, no search.

    Rows without both document_id and chunk_id are skipped (no fake IDs).
    Facet types are assigned only via conservative text match on `quote`.
    """
    query_text = (query or "").strip()
    facet_list = list(facets) if facets is not None else extract_facets(query_text)
    required = _required_facet_types(facet_list)

    # Precompute match phrases once per facet (includes QE) — avoids re-running
    # expand_query for every retrieval row.
    gate_facets = [facet for facet in facet_list if facet.type in MULTI_QUERY_GATE_TYPES]
    facet_phrases = [(facet, _phrase_terms(facet)) for facet in gate_facets]

    spans: list[EvidenceSpan] = []
    for row in retrieval_rows or []:
        document_id = row.get("document_id")
        chunk_id = row.get("chunk_id")
        if document_id is None or not chunk_id:
            continue
        try:
            document_id_int = int(document_id)
        except (TypeError, ValueError):
            continue

        quote = row.get("quote") or ""
        matched_types: list[FacetType] = []
        matched_terms: list[str] = []
        matched_facets: list[QueryFacet] = []
        for facet, phrases in facet_phrases:
            ok, terms = _match_phrases_in_text(facet, phrases, quote)
            if not ok:
                continue
            if facet.type not in matched_types:
                matched_types.append(facet.type)
            if facet not in matched_facets:
                matched_facets.append(facet)
            for term in terms:
                if term not in matched_terms:
                    matched_terms.append(term)

        spans.append(
            EvidenceSpan(
                document_id=document_id_int,
                chunk_id=str(chunk_id),
                path=str(row.get("path") or ""),
                document=str(row.get("document") or ""),
                quote=quote,
                facet_types=tuple(matched_types),
                subquery_ids=_subquery_ids_from_row(row),
                matched_terms=tuple(matched_terms),
                score=float(row.get("score") or 0.0),
                matched_facets=tuple(matched_facets),
            )
        )

    spans = _merge_duplicate_spans(spans)
    coverage = {
        facet_type: any(facet_type in span.facet_types for span in spans)
        for facet_type in required
    }
    return EvidenceSet(
        query=query_text,
        facets=tuple(facet_list),
        spans=tuple(spans),
        coverage=coverage,
        join_status=_join_status(coverage),
    )
