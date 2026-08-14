"""IntentRequirement / EvidenceCoverage foundation (PR4).

Pure unit tests — no SQLite/LanceDB/Ollama/retrieval.
"""
from __future__ import annotations

from intent_requirements import (
    RULES_VERSION,
    CoverageStatus,
    EvidenceCoverage,
    EvidenceNeed,
    IntentRequirement,
    build_evidence_coverage,
    derive_intent_requirement,
)
from query_facets import extract_facets

DESIGN_QUERY = "bude se brokovat základová deska 3PP"
BP_QUERY = "půdorys 3PP bludné proudy"
KZP_QUERY = "kontrolní a zkušební plán monolit FERI"


# ---------------------------------------------------------------------------
# Design case: brokování ZD 3PP
# ---------------------------------------------------------------------------


def test_brokovani_requires_technology_and_structure():
    req = derive_intent_requirement(DESIGN_QUERY)
    assert req.query == DESIGN_QUERY
    assert req.source == RULES_VERSION
    assert req.required_needs == (
        EvidenceNeed.TECHNOLOGY,
        EvidenceNeed.STRUCTURE,
    )


def test_brokovani_technology_only_is_partial_missing_structure():
    req = derive_intent_requirement(DESIGN_QUERY)
    cov = build_evidence_coverage(req, [EvidenceNeed.TECHNOLOGY])
    assert cov.status == CoverageStatus.PARTIAL
    assert cov.satisfied_needs == (EvidenceNeed.TECHNOLOGY,)
    assert cov.missing_needs == (EvidenceNeed.STRUCTURE,)
    assert EvidenceNeed.STRUCTURE not in cov.satisfied_needs


def test_brokovani_technology_and_structure_is_complete():
    req = derive_intent_requirement(DESIGN_QUERY)
    cov = build_evidence_coverage(
        req,
        [EvidenceNeed.TECHNOLOGY, EvidenceNeed.STRUCTURE],
    )
    assert cov.status == CoverageStatus.COMPLETE
    assert cov.missing_needs == ()
    assert cov.satisfied_needs == (
        EvidenceNeed.TECHNOLOGY,
        EvidenceNeed.STRUCTURE,
    )


def test_structure_evidence_does_not_invent_technology():
    """D11B-like STRUCTURE must not auto-satisfy TECHNOLOGY."""
    req = derive_intent_requirement(DESIGN_QUERY)
    cov = build_evidence_coverage(req, [EvidenceNeed.STRUCTURE])
    assert EvidenceNeed.TECHNOLOGY not in cov.satisfied_needs
    assert EvidenceNeed.STRUCTURE in cov.satisfied_needs
    assert EvidenceNeed.TECHNOLOGY in cov.missing_needs
    assert cov.status == CoverageStatus.PARTIAL


def test_no_evidence_is_insufficient_when_needs_exist():
    req = derive_intent_requirement(DESIGN_QUERY)
    cov = build_evidence_coverage(req, [])
    assert cov.status == CoverageStatus.INSUFFICIENT
    assert cov.satisfied_needs == ()
    assert cov.missing_needs == (
        EvidenceNeed.TECHNOLOGY,
        EvidenceNeed.STRUCTURE,
    )


# ---------------------------------------------------------------------------
# Negative: drawing / KZP must not over-require
# ---------------------------------------------------------------------------


def test_bp_drawing_query_does_not_require_tech_structure_package():
    req = derive_intent_requirement(BP_QUERY)
    assert EvidenceNeed.TECHNOLOGY not in req.required_needs
    assert EvidenceNeed.STRUCTURE not in req.required_needs
    # Prefer empty over false-positive package.
    assert req.required_needs == ()


def test_kzp_monolit_feri_does_not_require_structure():
    req = derive_intent_requirement(KZP_QUERY)
    assert EvidenceNeed.STRUCTURE not in req.required_needs


def test_empty_requirement_coverage_is_complete():
    req = IntentRequirement(query="x", required_needs=(), source=RULES_VERSION)
    cov = build_evidence_coverage(req, [EvidenceNeed.QUALITY])
    assert cov.status == CoverageStatus.COMPLETE
    assert cov.required_needs == ()
    assert cov.missing_needs == ()
    # Extra found needs do not become required.
    assert cov.satisfied_needs == ()


# ---------------------------------------------------------------------------
# API / purity smoke
# ---------------------------------------------------------------------------


def test_derive_accepts_precomputed_facets():
    facets = extract_facets(DESIGN_QUERY)
    req = derive_intent_requirement(DESIGN_QUERY, facets=facets)
    assert req.required_needs == (
        EvidenceNeed.TECHNOLOGY,
        EvidenceNeed.STRUCTURE,
    )


def test_coverage_ignores_unrequired_found_needs():
    req = IntentRequirement(
        query="q",
        required_needs=(EvidenceNeed.CONTRACT,),
        source=RULES_VERSION,
    )
    cov = build_evidence_coverage(
        req,
        [EvidenceNeed.CONTRACT, EvidenceNeed.COST, EvidenceNeed.QUALITY],
    )
    assert cov.status == CoverageStatus.COMPLETE
    assert cov.satisfied_needs == (EvidenceNeed.CONTRACT,)
    assert cov.missing_needs == ()


def test_evidence_coverage_is_plain_dataclass():
    cov = EvidenceCoverage(
        required_needs=(EvidenceNeed.TECHNOLOGY,),
        satisfied_needs=(),
        missing_needs=(EvidenceNeed.TECHNOLOGY,),
        status=CoverageStatus.INSUFFICIENT,
    )
    assert cov.status is CoverageStatus.INSUFFICIENT
