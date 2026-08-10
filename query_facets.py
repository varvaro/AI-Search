"""Deterministic query facet extraction + multi-query planning (PR1/PR2).

Pure decomposition of a user query into typed facets, plus a deterministic
subquery planner used by gated multi-query retrieval:

    query → extract_facets(query) → list[QueryFacet]
         → should_use_multi_query / plan_subqueries

This module does NOT expand queries, call retrieval, touch SQLite/LanceDB,
or invoke an LLM. Expansion still happens inside ai_search.search() via the
existing query_expansion path when ui_services.search_all() runs each planned
subquery.

Matching reuses the same fold/prefix principles as query_expansion.py so
Czech declension behaves consistently, but facet typing is a separate
concern from expansion emits / filename bonuses.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

import query_expansion as qe

# Hard cap so decomposition cannot explode on long / noisy queries.
MAX_QUERY_FACETS = 8

# Vocabulary scopes → facet type. construction_process → ACTION; document-ish
# scopes → DOC_TYPE; a few non-process concepts → OBJECT. Anything else that
# still matches a dictionary surface becomes OTHER (never silently dropped).
_ACTION_SCOPES = frozenset({"construction_process"})
_DOC_TYPE_SCOPES = frozenset({
    "documentation_type",
    "contract",
    "safety",
    "project_administration",
    "finance",
    "handover",
})
_OBJECT_VOCAB_SCOPES = frozenset({"discipline", "survey", "quality"})

# Order / invoice style identifiers (NOT252167) and compact alphanumerics.
_IDENTIFIER_RE = re.compile(r"(?=.*\d)[A-Za-z0-9][A-Za-z0-9._/-]{2,}\Z")

# Floor / storey notation: PP (podzemní podlaží) and NP (nadzemní podlaží).
# QE currently bridges only PP; NP is the same construction convention and is
# listed in the PR1 acceptance examples (2NP / 2.NP / 2 NP). Facet extraction
# detects both; it does not call expand_query or alter QE behaviour.
_FLOOR_PATTERN = re.compile(
    r"(?<!\w)(\d+)\s*\.?\s*(pp|np)(?!\w)",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class FacetType(str, Enum):
    LOCATION = "LOCATION"
    ACTION = "ACTION"
    OBJECT = "OBJECT"
    DOC_TYPE = "DOC_TYPE"
    ACTOR = "ACTOR"  # reserved; no safe firm dictionary in-repo for PR1 detection
    OTHER = "OTHER"


# Facet types that may open the multi-query gate. ACTOR/OTHER never count.
MULTI_QUERY_GATE_TYPES = frozenset({
    FacetType.ACTION,
    FacetType.OBJECT,
    FacetType.LOCATION,
    FacetType.DOC_TYPE,
})


@dataclass(frozen=True)
class QueryFacet:
    """One typed span from the original query.

    `surface` is the original substring (best-effort). `terms` are folded /
    normalized concept labels for later subquery building - unused in PR1.
    `confidence` is a coarse deterministic tier, not a calibrated probability:
      1.0 exact vocabulary / exact floor pattern
      0.8 vocabulary match via prefix / declension
      0.6 residual content span
    `source` explains why the facet exists (required for debugging / traces).
    """

    type: FacetType
    surface: str
    terms: tuple[str, ...]
    source: str
    confidence: float = 1.0


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    folded: str


@dataclass(frozen=True)
class _Candidate:
    words: tuple[str, ...]  # folded phrase words
    facet_type: FacetType
    source: str
    concept: str  # vocabulary key or floor label
    rank: int  # tie-break: lower wins among equal lengths


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


# Query glue / function words. Dropped from residual spans only - never strips
# tokens that are part of a longer vocabulary match ("kontrolní a zkušební plán").
_STOPWORDS = frozenset(
    _fold(word)
    for word in (
        "a", "i", "nebo", "či", "u", "k", "ke", "ku", "s", "se", "si", "z", "ze",
        "do", "na", "nad", "pod", "po", "od", "o", "v", "ve", "při", "pro", "za",
        "je", "jsou", "být", "bude", "budou", "by", "aby", "jak", "jako", "co",
        "kde", "kdy", "proč", "kolik", "kdo", "který", "která", "které", "kterou",
        "najdi", "najděte", "ukaž", "ukažte", "hledej", "hledejte", "dej", "dejte",
        "the", "of", "and", "or", "to", "for", "in", "on",
    )
)


def _tokenize(query: str) -> list[_Token]:
    return [
        _Token(text=m.group(0), start=m.start(), end=m.end(), folded=_fold(m.group(0)))
        for m in _TOKEN_RE.finditer(query or "")
    ]


def _word_matches(dictionary_word: str, token_folded: str) -> bool:
    """Same safety contract as query_expansion._surface_matches per word:
    exact for short forms; query token may extend a longer dictionary stem.
    Never the reverse (no uncontrolled substring / stemmer)."""
    if not dictionary_word or not token_folded:
        return False
    if dictionary_word == token_folded:
        return True
    if (
        len(dictionary_word) >= qe.MIN_PREFIX_MATCH_LENGTH
        and token_folded.startswith(dictionary_word)
    ):
        return True
    return False


def _match_phrase_at(phrase_words: tuple[str, ...], tokens: list[_Token], start: int) -> int | None:
    """Return exclusive end index if `phrase_words` matches consecutive tokens
    starting at `start`, else None."""
    if start + len(phrase_words) > len(tokens):
        return None
    for offset, word in enumerate(phrase_words):
        if not _word_matches(word, tokens[start + offset].folded):
            return None
    return start + len(phrase_words)


def _span_surface(query: str, tokens: list[_Token], start: int, end: int) -> str:
    return query[tokens[start].start:tokens[end - 1].end]


def _is_prefix_match(phrase_words: tuple[str, ...], tokens: list[_Token], start: int, end: int) -> bool:
    for offset, word in enumerate(phrase_words):
        if word != tokens[start + offset].folded:
            return True
    return False


def _floor_forms(level: str, kind: str) -> tuple[str, ...]:
    k = kind.upper()
    return (f"{level}{k}", f"{level}.{k}", f"{level} {k}")


def _build_vocab_candidates() -> list[_Candidate]:
    """Longest-match-first candidates derived from DOMAIN_VOCABULARY only.
    No project-specific hacks and no open-ended verb list."""
    candidates: list[_Candidate] = []
    rank = 0
    for key, rule in qe.DOMAIN_VOCABULARY.items():
        scope = rule.get("scope") or ""
        if scope in _ACTION_SCOPES:
            facet_type = FacetType.ACTION
            # Include documents: for construction_process they name techniques
            # (e.g. otryskání) that are ACTION concepts, not filename emits.
            surfaces = (key, *rule.get("synonyms", ()), *rule.get("documents", ()))
        elif scope in _DOC_TYPE_SCOPES:
            facet_type = FacetType.DOC_TYPE
            surfaces = (
                key,
                *rule.get("abbreviations", ()),
                *rule.get("synonyms", ()),
                *rule.get("documents", ()),
            )
        elif scope in _OBJECT_VOCAB_SCOPES:
            facet_type = FacetType.OBJECT
            surfaces = (key, *rule.get("synonyms", ()))
        elif scope == "project_alias":
            facet_type = FacetType.OTHER
            surfaces = (
                key,
                *rule.get("abbreviations", ()),
                *rule.get("synonyms", ()),
                *rule.get("documents", ()),
            )
        else:
            facet_type = FacetType.OTHER
            surfaces = (key, *rule.get("synonyms", ()), *rule.get("abbreviations", ()))

        source = f"vocabulary:{_fold(key).replace(' ', '_')}"
        seen: set[tuple[str, ...]] = set()
        for surface in surfaces:
            words = tuple(qe._tokens(surface))
            if not words or words in seen:
                continue
            seen.add(words)
            candidates.append(
                _Candidate(
                    words=words,
                    facet_type=facet_type,
                    source=source,
                    concept=key,
                    rank=rank,
                )
            )
            rank += 1
    # Longer phrases first; stable tie-break on rank then words.
    candidates.sort(key=lambda c: (-len(c.words), c.rank, c.words))
    return candidates


# Built once at import - vocabulary is process-static.
_VOCAB_CANDIDATES = _build_vocab_candidates()


def _extract_locations(query: str, tokens: list[_Token], claimed: set[int]) -> list[QueryFacet]:
    facets: list[QueryFacet] = []
    for match in _FLOOR_PATTERN.finditer(query or ""):
        level, kind = match.group(1), match.group(2)
        # Map char span onto tokens that overlap the match.
        idxs = [
            i for i, tok in enumerate(tokens)
            if tok.start < match.end() and tok.end > match.start() and i not in claimed
        ]
        if not idxs:
            continue
        if any(i in claimed for i in idxs):
            continue
        for i in idxs:
            claimed.add(i)
        surface = query[match.start():match.end()]
        forms = _floor_forms(level, kind)
        facets.append(
            QueryFacet(
                type=FacetType.LOCATION,
                surface=surface,
                terms=forms,
                source="floor_pattern",
                confidence=1.0,
            )
        )
    return facets


def _extract_vocabulary_facets(
    query: str,
    tokens: list[_Token],
    claimed: set[int],
) -> list[QueryFacet]:
    facets: list[QueryFacet] = []
    for candidate in _VOCAB_CANDIDATES:
        # Leftmost non-overlapping match for this phrase.
        start = 0
        while start < len(tokens):
            while start < len(tokens) and start in claimed:
                start += 1
            if start >= len(tokens):
                break
            end = _match_phrase_at(candidate.words, tokens, start)
            if end is None or any(i in claimed for i in range(start, end)):
                start += 1
                continue
            for i in range(start, end):
                claimed.add(i)
            surface = _span_surface(query, tokens, start, end)
            prefix = _is_prefix_match(candidate.words, tokens, start, end)
            concept_fold = _fold(candidate.concept)
            terms = (concept_fold, _fold(surface)) if _fold(surface) != concept_fold else (concept_fold,)
            facets.append(
                QueryFacet(
                    type=candidate.facet_type,
                    surface=surface,
                    terms=terms,
                    source=candidate.source,
                    confidence=0.8 if prefix else 1.0,
                )
            )
            break  # one match per candidate phrase is enough for PR1
    return facets


def _is_stopword_token(tok: _Token) -> bool:
    return tok.folded in _STOPWORDS


def _is_identifier(tok: _Token) -> bool:
    return bool(_IDENTIFIER_RE.match(tok.text))


def _emit_residual_run(
    query: str,
    tokens: list[_Token],
    start: int,
    end: int,
) -> QueryFacet:
    surface = _span_surface(query, tokens, start, end)
    if end - start >= 2:
        return QueryFacet(
            type=FacetType.OBJECT,
            surface=surface,
            terms=(_fold(surface),),
            source="residual_span",
            confidence=0.6,
        )
    return QueryFacet(
        type=FacetType.OTHER,
        surface=surface,
        terms=(_fold(surface),),
        source="residual_span",
        confidence=0.6,
    )


def _extract_residuals(
    query: str,
    tokens: list[_Token],
    claimed: set[int],
) -> list[QueryFacet]:
    """Contiguous unclaimed content → OBJECT (2+ tokens) or OTHER (1 token).
    Identifiers (order numbers etc.) are always split out as OTHER so they are
    not glued onto neighbouring names. Stopwords between claimed spans drop."""
    facets: list[QueryFacet] = []
    i = 0
    while i < len(tokens):
        if i in claimed or _is_stopword_token(tokens[i]):
            i += 1
            continue
        # Identifiers stand alone (NOT252167 must not merge into "Nazarenko …").
        if _is_identifier(tokens[i]):
            claimed.add(i)
            surface = tokens[i].text
            facets.append(
                QueryFacet(
                    type=FacetType.OTHER,
                    surface=surface,
                    terms=(_fold(surface),),
                    source="residual_identifier",
                    confidence=0.6,
                )
            )
            i += 1
            continue
        start = i
        while (
            i < len(tokens)
            and i not in claimed
            and not _is_stopword_token(tokens[i])
            and not _is_identifier(tokens[i])
        ):
            i += 1
        end = i
        if end <= start:
            continue
        for j in range(start, end):
            claimed.add(j)
        facets.append(_emit_residual_run(query, tokens, start, end))
    return facets


def _dedupe_facets(facets: list[QueryFacet]) -> list[QueryFacet]:
    """Drop exact duplicates (type, surface, source). Preserve order."""
    seen: set[tuple[str, str, str]] = set()
    out: list[QueryFacet] = []
    for facet in facets:
        key = (facet.type.value, facet.surface, facet.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(facet)
    return out


def _priority(facet: QueryFacet) -> tuple[int, str]:
    order = {
        FacetType.LOCATION: 0,
        FacetType.ACTION: 1,
        FacetType.DOC_TYPE: 2,
        FacetType.OBJECT: 3,
        FacetType.ACTOR: 4,
        FacetType.OTHER: 5,
    }
    return (order.get(facet.type, 9), facet.surface.casefold())


def extract_facets(query: str) -> list[QueryFacet]:
    """Decompose `query` into typed facets. Pure and deterministic.

    Empty / whitespace-only input → []. Never raises on normal string input.
    Does not call expand_query, search, or any I/O.
    """
    if query is None or not str(query).strip():
        return []
    query = str(query)

    tokens = _tokenize(query)
    if not tokens:
        return []

    claimed: set[int] = set()
    facets: list[QueryFacet] = []
    facets.extend(_extract_locations(query, tokens, claimed))
    facets.extend(_extract_vocabulary_facets(query, tokens, claimed))
    facets.extend(_extract_residuals(query, tokens, claimed))
    facets = _dedupe_facets(facets)

    if len(facets) > MAX_QUERY_FACETS:
        facets = sorted(facets, key=_priority)[:MAX_QUERY_FACETS]
        # Keep a stable, readable order: original appearance by surface start.
        def _start(f: QueryFacet) -> int:
            pos = query.find(f.surface)
            return pos if pos >= 0 else 10**9

        facets.sort(key=_start)

    return facets


@dataclass(frozen=True)
class PlannedSubquery:
    """One retrieval leg produced by plan_subqueries().

    `id` is stable (`full`, `action`, `object_location`, `doc_type`).
    `text` is the string passed to ai_search.search(); QE runs there unchanged.
    `facet_types` records which facet types contributed (empty for Q_full).
    """

    id: str
    text: str
    facet_types: tuple[FacetType, ...] = ()


def should_use_multi_query(facets: list[QueryFacet]) -> bool:
    """True when facets justify multi-query retrieval.

    Requires ≥2 distinct types from MULTI_QUERY_GATE_TYPES. OTHER/ACTOR never
    open the gate by themselves and do not count toward the threshold.
    Pure and deterministic - no I/O, no config flag check (callers combine
    this with MULTI_QUERY_RETRIEVAL_ENABLED).
    """
    types = {facet.type for facet in facets if facet.type in MULTI_QUERY_GATE_TYPES}
    return len(types) >= 2


def _surfaces_for(facets: list[QueryFacet], facet_type: FacetType) -> list[str]:
    return [facet.surface for facet in facets if facet.type is facet_type and facet.surface.strip()]


def plan_subqueries(
    query: str,
    facets: list[QueryFacet] | None = None,
    max_subqueries: int = 4,
) -> list[PlannedSubquery]:
    """Build a deterministic subquery plan. Always starts with Q_full.

    Additional legs (action / object_location / doc_type) are appended only
    when should_use_multi_query(facets) is true and the joined surface text
    is non-empty and not a duplicate of an earlier leg (including Q_full).

    Does not emit filenames, document ids, or project-specific hacks.
    Caps at max_subqueries (including Q_full). Pure - no retrieval, no QE.
    """
    original = (query or "").strip()
    if not original:
        return []

    if facets is None:
        facets = extract_facets(original)

    # Clamp defensively; config MAX_SUBQUERIES is the production ceiling.
    limit = max(1, min(int(max_subqueries), 4))
    planned: list[PlannedSubquery] = [
        PlannedSubquery(id="full", text=original, facet_types=()),
    ]
    seen = {_fold(original)}

    if not should_use_multi_query(facets):
        return planned[:limit]

    def _add(subquery_id: str, parts: list[str], facet_types: tuple[FacetType, ...]) -> None:
        if len(planned) >= limit:
            return
        text = " ".join(part.strip() for part in parts if part and part.strip()).strip()
        if not text:
            return
        folded = _fold(text)
        if not folded or folded in seen:
            return
        seen.add(folded)
        planned.append(
            PlannedSubquery(id=subquery_id, text=text, facet_types=facet_types)
        )

    action_surfaces = _surfaces_for(facets, FacetType.ACTION)
    if action_surfaces:
        _add("action", action_surfaces, (FacetType.ACTION,))

    object_surfaces = _surfaces_for(facets, FacetType.OBJECT)
    location_surfaces = _surfaces_for(facets, FacetType.LOCATION)
    object_location = object_surfaces + location_surfaces
    if object_location:
        types: list[FacetType] = []
        if object_surfaces:
            types.append(FacetType.OBJECT)
        if location_surfaces:
            types.append(FacetType.LOCATION)
        _add("object_location", object_location, tuple(types))

    doc_surfaces = _surfaces_for(facets, FacetType.DOC_TYPE)
    if doc_surfaces:
        _add("doc_type", doc_surfaces, (FacetType.DOC_TYPE,))

    return planned[:limit]
