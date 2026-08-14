"""DocumentState foundation (W1).

Pure, deterministic lifecycle-state layer for documents — orthogonal to
QueryFacet coverage (EvidenceSet) and EvidenceNeed (IntentRequirement):

    document/path → classify_document_state → DocumentStateEvidence
    query (+ optional facets) → derive_state_requirement → StateRequirement

This module does NOT:
  * wire into answer() / search() / search_all()
  * change RRF, scoring, bonuses, QE, or Aux
  * parse PDF crypto / metadata / body text
  * call SQLite / LanceDB / Ollama / Streamlit
  * aggregate CONFLICT / StateCoverage across documents
  * invent document_ids

forbidden_states means "must not alone support a positive state claim",
not "remove from the retrieval pool".
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from query_facets import FacetType, QueryFacet, extract_facets

# Bump only when rule semantics change intentionally (tests pin this).
RULES_VERSION = "rules_v1"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Confidence tiers (coarse, not calibrated probabilities).
_CONF_FILENAME = 1.0
_CONF_PATH = 0.5


class DocumentState(str, Enum):
    """Lifecycle state of one document (filename/path heuristics in W1)."""

    SIGNED = "SIGNED"
    FOR_SIGNATURE = "FOR_SIGNATURE"
    DRAFT = "DRAFT"
    TEMPLATE = "TEMPLATE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DocumentStateEvidence:
    """Classification outcome for a single document path/name."""

    document_id: int | None
    path: str
    document: str
    state: DocumentState
    signals: tuple[str, ...]
    confidence: float
    source: str = RULES_VERSION


@dataclass(frozen=True)
class StateRequirement:
    """What document lifecycle state(s) a query intent needs.

    Empty required/preferred/forbidden ⇒ state-insensitive query (no-op).
    """

    query: str
    required_states: tuple[DocumentState, ...]
    preferred_states: tuple[DocumentState, ...]
    forbidden_states: tuple[DocumentState, ...]
    entity_terms: tuple[str, ...]
    doc_type_terms: tuple[str, ...]
    allow_unknown: bool
    source: str = RULES_VERSION


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def _basename(document: str, path: str) -> str:
    name = (document or "").strip()
    if name:
        return name
    raw = (path or "").rstrip("/\\")
    if not raw:
        return ""
    return raw.replace("\\", "/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Filename / path signals (W1 only)
# ---------------------------------------------------------------------------

# Underscore/dot are filename separators — do not use \w lookaround ('_' is \w).
# SIGNED beats FOR_SIGNATURE even when both appear
# (e.g. "...k el. podpisu_podepsané.pdf").
_SIGNED_FILENAME_RE = re.compile(r"(?<![a-z0-9])(podepsan\w*|signed)(?![a-z0-9])", re.UNICODE)
_FOR_SIGNATURE_FILENAME_RE = re.compile(
    r"(k\s*el\.?\s*podpisu|k\s*podpisu|for\s*signature)",
    re.UNICODE,
)
_TEMPLATE_FILENAME_RE = re.compile(r"(?<![a-z0-9])(vzor\w*|template)(?![a-z0-9])", re.UNICODE)
_DRAFT_FILENAME_RE = re.compile(r"(?<![a-z0-9])(navrh\w*|draft)(?![a-z0-9])", re.UNICODE)

# Path is a weak hint only — never overrides a strong filename state.
_TEMPLATE_PATH_RE = re.compile(r"(vzor\w*|template)", re.UNICODE)
_DRAFT_PATH_RE = re.compile(r"(navrh\w*|draft|navrh\s*sod)", re.UNICODE)


def _filename_state(folded_name: str) -> tuple[DocumentState | None, tuple[str, ...]]:
    if not folded_name:
        return None, ()
    if _SIGNED_FILENAME_RE.search(folded_name):
        return DocumentState.SIGNED, ("filename:podepsan*|signed",)
    if _FOR_SIGNATURE_FILENAME_RE.search(folded_name):
        return DocumentState.FOR_SIGNATURE, ("filename:k podpisu|for signature",)
    if _TEMPLATE_FILENAME_RE.search(folded_name):
        return DocumentState.TEMPLATE, ("filename:vzor*|template",)
    if _DRAFT_FILENAME_RE.search(folded_name):
        return DocumentState.DRAFT, ("filename:navrh|draft",)
    return None, ()


def _path_hint_state(folded_path: str) -> tuple[DocumentState | None, tuple[str, ...]]:
    if not folded_path:
        return None, ()
    # TEMPLATE before DRAFT: path ".../vzor.../návrh..." → TEMPLATE.
    if _TEMPLATE_PATH_RE.search(folded_path):
        return DocumentState.TEMPLATE, ("path:vzor*|template",)
    if _DRAFT_PATH_RE.search(folded_path):
        return DocumentState.DRAFT, ("path:navrh|draft",)
    return None, ()


def classify_document_state(
    document: str,
    path: str = "",
    document_id: int | None = None,
) -> DocumentStateEvidence:
    """Classify lifecycle state from filename + weak path hints only.

    Precedence: SIGNED > FOR_SIGNATURE > TEMPLATE > DRAFT > UNKNOWN.
    A strong filename match is never degraded by path.
    """
    name = _basename(document, path)
    folded_name = _fold(name)
    folded_path = _fold(path or "")

    state, signals = _filename_state(folded_name)
    confidence = _CONF_FILENAME
    if state is None:
        state, signals = _path_hint_state(folded_path)
        confidence = _CONF_PATH if state is not None else 0.0
    if state is None:
        state = DocumentState.UNKNOWN
        signals = ()
        confidence = 0.0

    return DocumentStateEvidence(
        document_id=document_id,
        path=path or "",
        document=name,
        state=state,
        signals=signals,
        confidence=confidence,
        source=RULES_VERSION,
    )


# ---------------------------------------------------------------------------
# StateRequirement derivation
# ---------------------------------------------------------------------------

_QUERY_STOPWORDS = frozenset({
    "a", "i", "o", "u", "k", "s", "z", "v", "ve", "ke", "se", "si", "je", "na",
    "do", "od", "za", "po", "pro", "pri", "bez", "jak", "jaka", "jake", "jaky",
    "co", "ci", "nebo", "ale", "tak", "uz", "jen", "the", "of", "and", "or",
    "in", "on", "for", "to", "with", "najdi", "najit", "hledej", "boxu", "box",
    "indexu", "dokumentech", "dokumentu", "soubor", "souboru",
})

_DOC_TYPE_QUERY_TERMS = (
    "smlouva", "smlouvy", "smlouvu", "smlouve", "sod", "bozp", "dodatek",
    "objednavka", "objednávka",
)

# Signed-contract intent markers (folded).
# PR6.2: added general "podpis"/"signatura" variants (audit finding: "Ověř
# podpis smlouvy haus365" was NOOP because only "podepsan*" was recognized).
# Deliberately CZ-only for now - German/English variants are future work.
# Checked AFTER _FOR_SIGNATURE_QUERY_RE below, so "k podpisu" still resolves
# to FOR_SIGNATURE, not SIGNED.
_SIGNED_QUERY_RE = re.compile(
    r"(?<!\w)(podepsan\w*|podpis\w*|signatur\w*|signed)(?!\w)", re.UNICODE,
)
_FOR_SIGNATURE_QUERY_RE = re.compile(
    r"(k\s*el\.?\s*podpisu|k\s*podpisu|for\s*signature)",
    re.UNICODE,
)


def _query_has_doc_type(folded_query: str, facets: Sequence[QueryFacet]) -> bool:
    if any(term in folded_query for term in (_fold(t) for t in _DOC_TYPE_QUERY_TERMS)):
        return True
    return any(f.type == FacetType.DOC_TYPE for f in facets)


def _doc_type_terms(folded_query: str, facets: Sequence[QueryFacet]) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for term in _DOC_TYPE_QUERY_TERMS:
        folded = _fold(term)
        if folded in folded_query and folded not in seen:
            seen.add(folded)
            found.append(folded)
    for facet in facets:
        if facet.type != FacetType.DOC_TYPE:
            continue
        for surface in (facet.surface, *facet.terms):
            folded = _fold(surface).strip()
            if folded and folded not in seen:
                seen.add(folded)
                found.append(folded)
    return tuple(found)


def _entity_terms(folded_query: str) -> tuple[str, ...]:
    """Residual content tokens that look like entity / party identifiers."""
    skip = set(_QUERY_STOPWORDS)
    skip.update(_fold(t) for t in _DOC_TYPE_QUERY_TERMS)
    skip.update({
        "podepsana", "podepsane", "podepsany", "podepsan", "signed",
        "podpisu", "podpis", "signatura", "signaturu", "signatury",
        "navrh", "draft", "vzor", "vzorova", "template",
        "el", "elektronickemu", "elektronicky",
    })
    out: list[str] = []
    seen: set[str] = set()
    for tok in _TOKEN_RE.findall(folded_query):
        if len(tok) < 3 or tok in skip:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return tuple(out)


def _empty_requirement(query: str) -> StateRequirement:
    return StateRequirement(
        query=query,
        required_states=(),
        preferred_states=(),
        forbidden_states=(),
        entity_terms=(),
        doc_type_terms=(),
        allow_unknown=True,
        source=RULES_VERSION,
    )


def derive_state_requirement(
    query: str,
    facets: Sequence[QueryFacet] | None = None,
) -> StateRequirement:
    """Derive lifecycle StateRequirement from the query (optional facets).

    Default: no state intent → empty required/preferred/forbidden.
    """
    q = (query or "").strip()
    if not q:
        return _empty_requirement(q)

    facet_list: tuple[QueryFacet, ...] = (
        tuple(facets) if facets is not None else tuple(extract_facets(q))
    )
    folded = _fold(q)
    has_doc = _query_has_doc_type(folded, facet_list)
    entities = _entity_terms(folded)
    doc_terms = _doc_type_terms(folded, facet_list)

    # FOR_SIGNATURE intent before SIGNED: "k podpisu" is not "podepsaná".
    if has_doc and _FOR_SIGNATURE_QUERY_RE.search(folded):
        return StateRequirement(
            query=q,
            required_states=(DocumentState.FOR_SIGNATURE,),
            preferred_states=(DocumentState.FOR_SIGNATURE,),
            forbidden_states=(DocumentState.DRAFT, DocumentState.TEMPLATE),
            entity_terms=entities,
            doc_type_terms=doc_terms,
            allow_unknown=False,
            source=RULES_VERSION,
        )

    if has_doc and _SIGNED_QUERY_RE.search(folded):
        return StateRequirement(
            query=q,
            required_states=(DocumentState.SIGNED,),
            preferred_states=(DocumentState.SIGNED,),
            forbidden_states=(
                DocumentState.FOR_SIGNATURE,
                DocumentState.DRAFT,
                DocumentState.TEMPLATE,
            ),
            entity_terms=entities,
            doc_type_terms=doc_terms,
            allow_unknown=False,
            source=RULES_VERSION,
        )

    return _empty_requirement(q)
