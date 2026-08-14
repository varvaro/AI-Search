"""IntentRequirement / EvidenceCoverage foundation (PR4).

Pure, deterministic layer orthogonal to QueryFacet coverage (EvidenceSet):

    query → extract_facets → derive_intent_requirement → IntentRequirement
    IntentRequirement + found EvidenceNeed[] → build_evidence_coverage
        → EvidenceCoverage

This module does NOT:
  * call retrieval / search_all
  * touch SQLite / LanceDB / FTS / Ollama
  * classify documents at index time
  * plan second-hop
  * invent evidence from filenames or document_ids

Facet JoinStatus.COMPLETE means "ACTION/OBJECT/LOCATION text spans exist".
EvidenceCoverage.COMPLETE means "all required evidence *needs* are present".
Those are different questions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from query_facets import FacetType, QueryFacet, extract_facets

# Bump only when rule semantics change intentionally (tests pin this).
RULES_VERSION = "rules_v1"


class EvidenceNeed(str, Enum):
    """Coarse evidence kinds required for a safe multi-document answer.

    Keep this set small. Do not grow into a construction ontology.
    """

    TECHNOLOGY = "TECHNOLOGY"
    STRUCTURE = "STRUCTURE"
    CONTRACT = "CONTRACT"
    COST = "COST"
    QUALITY = "QUALITY"


class CoverageStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


# Stable display / tuple order for needs.
_NEED_ORDER = (
    EvidenceNeed.TECHNOLOGY,
    EvidenceNeed.STRUCTURE,
    EvidenceNeed.CONTRACT,
    EvidenceNeed.COST,
    EvidenceNeed.QUALITY,
)


@dataclass(frozen=True)
class IntentRequirement:
    """What evidence kinds the query intent needs — independent of retrieval."""

    query: str
    required_needs: tuple[EvidenceNeed, ...]
    source: str = RULES_VERSION


@dataclass(frozen=True)
class EvidenceCoverage:
    """Satisfaction of IntentRequirement against found evidence need labels.

    `evidence_types` are supplied by the caller (tests / future adapters).
    This PR does not extract needs from quotes or documents.
    """

    required_needs: tuple[EvidenceNeed, ...]
    satisfied_needs: tuple[EvidenceNeed, ...]
    missing_needs: tuple[EvidenceNeed, ...]
    status: CoverageStatus


def _ordered_needs(needs: Iterable[EvidenceNeed]) -> tuple[EvidenceNeed, ...]:
    present = set(needs)
    return tuple(n for n in _NEED_ORDER if n in present)


def _has_vocab_action(facets: Sequence[QueryFacet]) -> bool:
    """True only for lexicon-backed ACTION facets (not residual guesses)."""
    return any(
        f.type == FacetType.ACTION and f.source.startswith("vocabulary:")
        for f in facets
    )


def derive_intent_requirement(
    query: str,
    facets: Sequence[QueryFacet] | None = None,
) -> IntentRequirement:
    """Derive required evidence needs from the query (and optional facets).

    Conservative: if no safe rule matches, return an empty requirement.
    Empty requirement ⇒ coverage COMPLETE (nothing missing by definition).
    """
    q = (query or "").strip()
    facet_list: tuple[QueryFacet, ...] = (
        tuple(facets) if facets is not None else tuple(extract_facets(q))
    )
    types = {f.type for f in facet_list}

    required: list[EvidenceNeed] = []

    # Rule: construction process package.
    # Vocabulary ACTION + OBJECT + LOCATION → technology process evidence and
    # build-up / structure evidence are both needed for a safe join.
    # Drawing / KZP queries typically lack vocabulary ACTION and therefore
    # do not trigger this package (prefer empty over false positives).
    if (
        _has_vocab_action(facet_list)
        and FacetType.OBJECT in types
        and FacetType.LOCATION in types
    ):
        required.extend([EvidenceNeed.TECHNOLOGY, EvidenceNeed.STRUCTURE])

    return IntentRequirement(
        query=q,
        required_needs=_ordered_needs(required),
        source=RULES_VERSION,
    )


def build_evidence_coverage(
    requirement: IntentRequirement,
    evidence_types: Iterable[EvidenceNeed],
) -> EvidenceCoverage:
    """Evaluate which required needs are present in caller-supplied evidence.

    Independent of EvidenceSet, retrieval, and databases. Presence of STRUCTURE
    never invents TECHNOLOGY (or any other need).
    """
    required = _ordered_needs(requirement.required_needs)
    found = set(evidence_types)
    satisfied = _ordered_needs(n for n in required if n in found)
    missing = _ordered_needs(n for n in required if n not in found)

    if not required:
        status = CoverageStatus.COMPLETE
    elif not missing:
        status = CoverageStatus.COMPLETE
    elif not satisfied:
        status = CoverageStatus.INSUFFICIENT
    else:
        status = CoverageStatus.PARTIAL

    return EvidenceCoverage(
        required_needs=required,
        satisfied_needs=satisfied,
        missing_needs=missing,
        status=status,
    )
