"""EvidenceSet runtime validation foundation (PR7.0, ADR-007).

Pure unit tests — no SQLite/LanceDB/Ollama/retrieval/answer path. Requirements
and evidences are produced by the REAL foundation functions
(document_state.derive_state_requirement / classify_document_state,
evidence.build_evidence_set) rather than hand-built, so the tests fail if a
foundation rule drifts underneath this layer.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from document_state import (
    DocumentState,
    classify_document_state,
    derive_state_requirement,
)
from evidence import build_evidence_set
from evidence_runtime import (
    RULES_VERSION,
    AnswerValidation,
    GateAction,
    StateCoverage,
    StateVerdict,
    build_state_coverage,
    derive_evidence_needs,
)
from intent_requirements import (
    CoverageStatus,
    EvidenceNeed,
    build_evidence_coverage,
    derive_intent_requirement,
)
from query_facets import FacetType, extract_facets

SIGNED_QUERY = "je na boxu podepsaná smlouva haus365?"
NO_STATE_QUERY = "smlouva haus365"
CRM_QUERY = "jaký svár je požadovaný na CRM destičky"
DESIGN_QUERY = "bude se brokovat základová deska 3PP"

SIGNED_DOC = "SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf"
DRAFT_DOC = "SOD_HAUS365_návrh.docx"
TEMPLATE_DOC = "SOD_HAUS365_vzorová NDS.docx"
FOR_SIGNATURE_DOC = "SOD_HAUS365_NDS_k el. podpisu.pdf"
UNKNOWN_DOC = "SoD_haus365.pdf"
VAHOSTAV_SIGNED_DOC = "SOD_VAHOSTAV_podepsaná.pdf"

# Verbatim filenames of the SoD HAUS365 revision family as indexed in
# production (6342 documents) — the case the CONFLICT rule broke.
HAUS365_FAMILY = (
    "NOT262012_SoD_HAUS365 GmbH & Co. KG_tisk.pdf",
    "Připomínky k SoD_HAUS365.pdf",
    "SOD_HAUS365_NDS_k el. podpisu.pdf",
    "SOD_HAUS365_NDS_k el. podpisu_JS.pdf",
    "SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf",
    "SOD_HAUS365_NDS_návrh_rev2. 260707_VH_rev MH_final.docx",
    "SOD_HAUS365_NDS_návrh_rev2. 260707_VH_rev MH_final_JS el. podpis.pdf",
    "SOD_HAUS365_vzorová NDS_návrh.docx",
    "SOD_HAUS365_vzorová NDS_návrh_rev2. 260707_VH.docx",
    "SOD_HAUS365_vzorová NDS_návrh_rev2. 260707_VH_rev MH 2.docx",
    "SOD_HAUS365_vzorová NDS_návrh_rev2. 260707_VH_rev MH_rev. VH_změna subjektu.docx",
)

# Same fixture text tests/test_evidence.py uses — ACTION+OBJECT+LOCATION all
# verifiably present in the chunk itself.
TECHFLOOR_QUOTE = (
    "Otryskání podkladu před provedením lité podlahy, brokování, broušení. "
    "podlaha ve 3. pp základová deska"
)


def _evidence(document: str, path: str = "", document_id: int | None = None):
    return classify_document_state(document, path or f"/proj/{document}", document_id)


def _spans(quote: str, query: str = DESIGN_QUERY, document: str = "techfloor.xls"):
    facets = extract_facets(query)
    return build_evidence_set(
        query,
        retrieval_rows=[{
            "document_id": 1,
            "chunk_id": "c:0",
            "document": document,
            "path": f"/proj/{document}",
            "quote": quote,
            "score": 1.0,
            "project": "p",
            "heading": "",
            "match": {},
        }],
        facets=facets,
    ).spans


# ---------------------------------------------------------------------------
# TEST 1: signed Haus365 → SIGNED_CONFIRMED
# ---------------------------------------------------------------------------

def test_signed_haus365_is_signed_confirmed():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])

    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.conflict is False
    assert coverage.entity_matched is True
    assert coverage.states_present == frozenset({DocumentState.SIGNED})
    assert [ev.document for ev in coverage.evidences] == [SIGNED_DOC]
    assert coverage.requirement is requirement


def test_signed_confirmed_ignores_unrelated_unknown_documents():
    """An UNKNOWN document in the pool must not weaken a real SIGNED match
    (the HAUS365 audit: a TP dodatek alongside the signed contract)."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [
        _evidence("TP 2.3. - NDS-zajištění stav. jamy - dodatek 2.pdf"),
        _evidence(SIGNED_DOC),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    # The TP document does not match the entity, so it is not citable evidence.
    assert [ev.document for ev in coverage.evidences] == [SIGNED_DOC]


# ---------------------------------------------------------------------------
# TEST 2: draft-only Haus365 → UNSIGNED_CONFIRMED
# ---------------------------------------------------------------------------

def test_draft_only_haus365_is_unsigned_confirmed():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [_evidence(DRAFT_DOC)])

    assert coverage.verdict is StateVerdict.UNSIGNED_CONFIRMED
    assert coverage.conflict is False
    assert coverage.entity_matched is True
    assert coverage.states_present == frozenset({DocumentState.DRAFT})
    assert [ev.document for ev in coverage.evidences] == [DRAFT_DOC]


def test_template_and_draft_only_is_unsigned_confirmed():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [
        _evidence(TEMPLATE_DOC), _evidence(DRAFT_DOC),
    ])
    assert coverage.verdict is StateVerdict.UNSIGNED_CONFIRMED
    assert coverage.states_present == frozenset({DocumentState.TEMPLATE, DocumentState.DRAFT})
    assert len(coverage.evidences) == 2


# ---------------------------------------------------------------------------
# TEST 3: Váhostav signed instead of Haus365 → ENTITY_MISMATCH
# ---------------------------------------------------------------------------

def test_other_party_signed_document_is_entity_mismatch():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [_evidence(VAHOSTAV_SIGNED_DOC)])

    assert coverage.verdict is StateVerdict.ENTITY_MISMATCH
    assert coverage.entity_matched is False
    assert coverage.conflict is False
    # Nothing citable: an unrelated party's contract is not evidence, and it
    # must not leak into a positive verdict (PR6.1 critical finding).
    assert coverage.evidences == ()
    assert coverage.states_present == frozenset()
    assert coverage.verdict is not StateVerdict.SIGNED_CONFIRMED


def test_generic_legal_form_token_cannot_confirm_another_company():
    """PR7.0.1 core regression, verified against the production index: the query
    names "Zakládání Group", the pool holds a SIGNED document of H&B Group, and
    the only shared token is the legal form "group"."""
    requirement = derive_state_requirement("je podepsaný dodatek Zakládání Group?")
    assert requirement.entity_terms == ("zakladani", "group")

    coverage = build_state_coverage(
        requirement, [_evidence("Generální klíč H&B Group_NOT250334-podepsané.pdf")],
    )
    assert coverage.verdict is not StateVerdict.SIGNED_CONFIRMED
    assert coverage.verdict is StateVerdict.ENTITY_MISMATCH
    assert coverage.evidences == ()
    assert coverage.entity_matched is False


def test_correct_company_is_confirmed_and_the_stranger_is_not_cited():
    requirement = derive_state_requirement("je podepsaný dodatek Zakládání Group?")
    coverage = build_state_coverage(requirement, [
        _evidence("Generální klíč H&B Group_NOT250334-podepsané.pdf"),
        _evidence("NDS_DOD1_NOT243136_Zakládání Group, podepsane.pdf"),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert [ev.document for ev in coverage.evidences] == [
        "NDS_DOD1_NOT243136_Zakládání Group, podepsane.pdf"
    ]


def test_partial_multi_token_match_cannot_confirm_a_positive_verdict():
    """Both terms are real names in the pool ('feri' and 'monolit' each occur as
    a whole token) but NO document carries both, so the signed KZP-monolit
    document may not confirm a FERI query — and nothing is citable."""
    requirement = derive_state_requirement("je podepsaná smlouva FERI monolit?")
    assert requirement.entity_terms == ("feri", "monolit")

    coverage = build_state_coverage(requirement, [
        _evidence("KZP monolit Smíchov_podepsané.pdf"),   # SIGNED, only 'monolit'
        _evidence("Zmenovy_list_FERI_001.pdf"),           # UNKNOWN, only 'feri'
    ])
    assert coverage.verdict is not StateVerdict.SIGNED_CONFIRMED
    assert coverage.verdict is StateVerdict.UNVERIFIED
    assert coverage.evidences == ()
    assert coverage.entity_matched is False


def test_stranger_document_is_not_cited_when_a_full_match_exists():
    """"jeřáb JVS": another company's signed crane order must not be cited, but
    the genuine JVS contract must still confirm. Here 'jerab' occurs only inside
    "autojeřábnické" (not a whole token), so it is not a discriminator and the
    verdict rests on 'jvs' alone."""
    requirement = derive_state_requirement("je podepsaná smlouva na jeřáb JVS?")
    coverage = build_state_coverage(requirement, [
        _evidence("NOT260728_autojeřábnické práce_Doškář_OBJ, podepsaná.PDF"),
        _evidence("smlouva JVS-26-8205-01-01-pdf_podepsaná 25.3.2026.pdf"),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert [ev.document for ev in coverage.evidences] == [
        "smlouva JVS-26-8205-01-01-pdf_podepsaná 25.3.2026.pdf"
    ]


def test_noise_token_is_not_a_discriminator_even_as_a_substring():
    """'over' (from "Ověř") occurs as a substring of Czech words like "ověření"
    but never as a whole token — measured across the production index — so the
    whole-token support test keeps it out of the match."""
    requirement = derive_state_requirement("Ověř podpis smlouvy haus365")
    coverage = build_state_coverage(requirement, [
        _evidence("20250306_vyjádření geologa k ověření vrtu P5.pdf"),
        _evidence(SIGNED_DOC),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert [ev.document for ev in coverage.evidences] == [SIGNED_DOC]


def test_declension_and_glued_filenames_still_match():
    """Matching stays substring-based, so 'monolit' matches "monolity" — only
    the discriminativeness test is token-based."""
    requirement = derive_state_requirement("je podepsaná smlouva FERI monolit?")
    coverage = build_state_coverage(requirement, [
        _evidence("SoD FERI monolit_podepsané.pdf"),               # supports both tokens
        _evidence("Připomínky k SoD FERI monolity_podepsané.pdf"),  # declension
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert len(coverage.evidences) == 2


def test_query_naming_only_a_legal_form_confirms_nothing():
    requirement = derive_state_requirement("je podepsaná smlouva Group?")
    assert requirement.entity_terms == ("group",)

    coverage = build_state_coverage(requirement, [
        _evidence("Generální klíč H&B Group_NOT250334-podepsané.pdf"),
    ])
    assert coverage.verdict is StateVerdict.ENTITY_MISMATCH


def test_query_noise_token_does_not_poison_the_entity_match():
    """W1's residual entity extraction leaks non-entity words ("Ověř" → 'over').
    Requiring every raw term would turn a legitimate query into a mismatch;
    terms unsupported by the pool are dropped instead."""
    requirement = derive_state_requirement("Ověř podpis smlouvy haus365")
    assert requirement.entity_terms == ("over", "haus365")

    coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.entity_matched is True


def test_exact_normalized_entity_phrase_matches_across_separators():
    """Filename separators must not defeat the phrase match."""
    requirement = derive_state_requirement("je podepsaná smlouva Zakládání Group?")
    coverage = build_state_coverage(
        requirement, [_evidence("SOD_Zakladani_Group_2026_podepsane.pdf")],
    )
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED


def test_all_entity_terms_present_wins_over_partial_sibling():
    requirement = derive_state_requirement("je podepsaná smlouva FERI monolit?")
    coverage = build_state_coverage(requirement, [
        _evidence("KZP monolit Smíchov_podepsané.pdf"),          # only 'monolit'
        _evidence("NDS_NOT251110_SOD_monolit_Feri_podepsané.pdf"),  # both terms
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert [ev.document for ev in coverage.evidences] == [
        "NDS_NOT251110_SOD_monolit_Feri_podepsané.pdf"
    ]


def test_entity_mismatch_survives_a_pool_full_of_signed_strangers():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [
        _evidence(VAHOSTAV_SIGNED_DOC),
        _evidence("SOD_SafetyPeak_podepsaná.pdf"),
        _evidence("NOT250060_BOZP_podepsané.docx"),
    ])
    assert coverage.verdict is StateVerdict.ENTITY_MISMATCH
    assert coverage.evidences == ()


# ---------------------------------------------------------------------------
# TEST 4: Haus365 signed + draft → CONFLICT
# ---------------------------------------------------------------------------

def test_signed_and_draft_same_entity_is_signed_confirmed():
    """PR7.0.1: lifecycle coexistence is NOT a contradiction. A signed contract
    practically always sits next to the drafts it grew out of."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [
        _evidence(SIGNED_DOC), _evidence(DRAFT_DOC),
    ])

    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.conflict is False
    assert coverage.entity_matched is True
    assert coverage.states_present == frozenset({DocumentState.SIGNED, DocumentState.DRAFT})
    # Only the signed document is citable evidence of signedness...
    assert [ev.document for ev in coverage.evidences] == [SIGNED_DOC]
    # ...but the coexisting draft stays visible to the consumer.
    assert DocumentState.DRAFT in coverage.states_present


def test_signed_and_template_is_signed_confirmed():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [
        _evidence(SIGNED_DOC), _evidence(TEMPLATE_DOC),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.conflict is False
    assert [ev.document for ev in coverage.evidences] == [SIGNED_DOC]


def test_signed_and_for_signature_is_signed_confirmed():
    """The real HAUS365 filename carries BOTH markers ("k el. podpisu_
    podepsané"); a separate for-signature copy must not weaken it either."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [
        _evidence(SIGNED_DOC), _evidence(FOR_SIGNATURE_DOC),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.conflict is False
    assert [ev.document for ev in coverage.evidences] == [SIGNED_DOC]


def test_conflict_is_never_derived_from_lifecycle_states():
    """Matrix guard: no combination of lifecycle states may resurrect the old
    over-broad CONFLICT rule. CONFLICT stays a reserved enum member."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    others = [DRAFT_DOC, TEMPLATE_DOC, FOR_SIGNATURE_DOC, UNKNOWN_DOC]
    for other in others:
        for pool in ([_evidence(SIGNED_DOC), _evidence(other)],
                     [_evidence(other), _evidence(SIGNED_DOC)]):
            coverage = build_state_coverage(requirement, pool)
            assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED, other
            assert coverage.conflict is False, other
    # Whole lifecycle at once.
    coverage = build_state_coverage(requirement, [_evidence(d) for d in [SIGNED_DOC, *others]])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.conflict is False


def test_real_haus365_revision_family_is_signed_confirmed():
    """The production case this layer exists for: the full 11-document SoD
    revision family (real filenames from the index) spans all five states and
    used to return CONFLICT, contradicting the answer PR6 gives today."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [_evidence(d) for d in HAUS365_FAMILY])

    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.conflict is False
    assert coverage.states_present >= frozenset({
        DocumentState.SIGNED, DocumentState.DRAFT,
        DocumentState.TEMPLATE, DocumentState.FOR_SIGNATURE,
    })
    assert [ev.document for ev in coverage.evidences] == [SIGNED_DOC]


def test_two_signed_documents_same_entity_are_both_citable():
    """SoD + LOI for one entity: only the SoD confirms a signed-*smlouva*
    query (PR7.6.1). LOI remains classified SIGNED but is not citable
    contract evidence — the verdict is SIGNED_CONTRACT_CONFIRMED
    (= SIGNED_CONFIRMED wire value) with the SoD alone."""
    requirement = derive_state_requirement("je podepsaná smlouva FERI monolit?")
    coverage = build_state_coverage(requirement, [
        _evidence("NDS_NOT251110_SOD_monolit_Feri_podepsané.pdf"),
        _evidence("NDS_NOT251110_LOI_monolit_Feri_signed.pdf"),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONTRACT_CONFIRMED
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED  # alias
    assert coverage.conflict is False
    assert [ev.document for ev in coverage.evidences] == [
        "NDS_NOT251110_SOD_monolit_Feri_podepsané.pdf",
    ]


def test_competing_signed_revisions_stay_signed_confirmed():
    """Reserved-behaviour pin: telling rev1 from rev2 needs version parsing
    W1 does not have, so competing revisions must NOT fabricate a conflict."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [
        _evidence("SOD_HAUS365_NDS_rev1_podepsané.pdf"),
        _evidence("SOD_HAUS365_NDS_rev2_podepsané.pdf"),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.conflict is False
    assert len(coverage.evidences) == 2


# ---------------------------------------------------------------------------
# TEST 5: no StateRequirement → NOOP
# ---------------------------------------------------------------------------

def test_query_without_state_intent_is_noop():
    requirement = derive_state_requirement(NO_STATE_QUERY)
    assert requirement.required_states == ()

    coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])
    assert coverage.verdict is StateVerdict.NOOP
    assert coverage.evidences == ()
    assert coverage.states_present == frozenset()
    assert coverage.entity_matched is False
    assert coverage.conflict is False


def test_for_signature_intent_is_noop_not_signed_confirmed():
    """SIGNED must never satisfy a "k podpisu" intent (W1 rule). The verdict
    vocabulary is signed-centric, so this layer stays out of it entirely."""
    requirement = derive_state_requirement("najdi smlouvu haus365 k podpisu")
    assert DocumentState.FOR_SIGNATURE in requirement.required_states
    assert DocumentState.SIGNED not in requirement.required_states

    coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])
    assert coverage.verdict is StateVerdict.NOOP


# ---------------------------------------------------------------------------
# TEST 6: derive_evidence_needs is conservative
# ---------------------------------------------------------------------------

def test_unclear_span_yields_no_needs():
    spans = _spans("obecné stavební poznámky bez konkrétního obsahu")
    assert spans and spans[0].facet_types == ()
    assert derive_evidence_needs(spans) == ()


def test_no_spans_yields_no_needs():
    assert derive_evidence_needs([]) == ()
    assert derive_evidence_needs(None) == ()


def test_location_only_span_yields_no_needs():
    """"3.PP" says where, not what — neither TECHNOLOGY nor STRUCTURE."""
    spans = _spans("Poznámka k podlaží 3. pp bez dalšího popisu")
    assert spans[0].facet_types == (FacetType.LOCATION,)  # guard: not vacuous
    assert derive_evidence_needs(spans) == ()


def test_action_and_object_span_yields_technology_and_structure():
    spans = _spans(TECHFLOOR_QUOTE)
    assert derive_evidence_needs(spans) == (
        EvidenceNeed.TECHNOLOGY,
        EvidenceNeed.STRUCTURE,
    )


def test_object_without_action_does_not_invent_technology():
    spans = _spans("skladba P3 základová deska ŽB 250 mm")
    needs = derive_evidence_needs(spans)
    assert EvidenceNeed.STRUCTURE in needs
    assert EvidenceNeed.TECHNOLOGY not in needs


def test_doc_type_span_never_yields_contract_need():
    """A document quoting "smlouva o dílo" is not CONTRACT evidence — CONTRACT/
    COST/QUALITY have no safe deterministic signal in PR7.0."""
    spans = _spans(
        "Tato smlouva o dílo se řídí zákonem č. 89/2012 Sb.",
        query="smlouva o dílo haus365",
        document="SOD_HAUS365.pdf",
    )
    # Guard: the DOC_TYPE facet really did match the text, so the assertions
    # below test the mapping and not an empty span.
    assert spans[0].facet_types == (FacetType.DOC_TYPE,)
    assert derive_evidence_needs(spans) == ()


def test_needs_are_deduplicated_ordered_and_deterministic():
    spans = list(_spans(TECHFLOOR_QUOTE)) * 3
    first = derive_evidence_needs(spans)
    assert first == derive_evidence_needs(spans[::-1])
    assert len(first) == len(set(first))
    assert first.index(EvidenceNeed.TECHNOLOGY) < first.index(EvidenceNeed.STRUCTURE)


def test_provenance_alone_never_creates_a_need():
    """PR3 invariant: a row found BY an ACTION subquery is not ACTION evidence."""
    facets = extract_facets(DESIGN_QUERY)
    spans = build_evidence_set(
        DESIGN_QUERY,
        retrieval_rows=[{
            "document_id": 7, "chunk_id": "c:0", "document": "x.pdf", "path": "/p/x.pdf",
            "quote": "obecný text bez shody", "score": 1.0, "project": "p", "heading": "",
            "match": {}, "_mq_source": "action",
        }],
        facets=facets,
    ).spans
    assert spans[0].subquery_ids == ("action",)
    assert derive_evidence_needs(spans) == ()


def test_needs_adapter_closes_pr4_coverage_gap():
    """The adapter's whole purpose: feed build_evidence_coverage() with labels
    PR4 refused to guess, and reach COMPLETE for a genuinely covered query."""
    requirement = derive_intent_requirement(DESIGN_QUERY)
    assert requirement.required_needs == (EvidenceNeed.TECHNOLOGY, EvidenceNeed.STRUCTURE)

    coverage = build_evidence_coverage(requirement, derive_evidence_needs(_spans(TECHFLOOR_QUOTE)))
    assert coverage.status is CoverageStatus.COMPLETE
    assert coverage.missing_needs == ()


def test_thin_evidence_stays_insufficient():
    requirement = derive_intent_requirement(DESIGN_QUERY)
    coverage = build_evidence_coverage(requirement, derive_evidence_needs(_spans("nic relevantního")))
    assert coverage.status is CoverageStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# TEST 7: CRM svár query must not create a state requirement
# ---------------------------------------------------------------------------

def test_crm_svar_query_creates_no_state_requirement():
    requirement = derive_state_requirement(CRM_QUERY)
    assert requirement.required_states == ()
    assert requirement.preferred_states == ()
    assert requirement.forbidden_states == ()

    # Even with a signed contract in the pool the layer must stay inert.
    coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])
    assert coverage.verdict is StateVerdict.NOOP
    assert coverage.evidences == ()


def test_technical_queries_are_all_noop():
    for query in (CRM_QUERY, "Pentaflex", DESIGN_QUERY, "jaký beton je v základové desce"):
        requirement = derive_state_requirement(query)
        coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])
        assert coverage.verdict is StateVerdict.NOOP, query


# ---------------------------------------------------------------------------
# UNVERIFIED / empty-pool safety
# ---------------------------------------------------------------------------

def test_entity_matched_unknown_state_is_unverified():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [_evidence(UNKNOWN_DOC)])

    assert coverage.verdict is StateVerdict.UNVERIFIED
    assert coverage.entity_matched is True
    assert coverage.states_present == frozenset({DocumentState.UNKNOWN})
    assert [ev.document for ev in coverage.evidences] == [UNKNOWN_DOC]


def test_unknown_alongside_draft_is_unverified_not_unsigned():
    """An UNKNOWN candidate means the signed state cannot be ruled out, so a
    negative claim must not be licensed by the draft sitting next to it."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [_evidence(UNKNOWN_DOC), _evidence(DRAFT_DOC)])
    assert coverage.verdict is StateVerdict.UNVERIFIED


def test_empty_pool_is_unverified_not_unsigned_confirmed():
    """Absence of candidates is not evidence of absence."""
    requirement = derive_state_requirement("je smlouva podepsaná?")
    assert DocumentState.SIGNED in requirement.required_states
    assert requirement.entity_terms == ()

    coverage = build_state_coverage(requirement, [])
    assert coverage.verdict is StateVerdict.UNVERIFIED
    assert coverage.evidences == ()
    assert coverage.entity_matched is False


def test_query_without_entity_uses_general_signed_detection():
    requirement = derive_state_requirement("je smlouva podepsaná?")
    coverage = build_state_coverage(requirement, [_evidence(VAHOSTAV_SIGNED_DOC)])
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    # No entity was named, so no entity match may be claimed.
    assert coverage.entity_matched is False


def test_evidences_default_to_empty_when_omitted():
    requirement = derive_state_requirement(SIGNED_QUERY)
    assert build_state_coverage(requirement).verdict is StateVerdict.ENTITY_MISMATCH


# ---------------------------------------------------------------------------
# AnswerValidation container
# ---------------------------------------------------------------------------

def test_answer_validation_defaults_are_honest_about_what_ran():
    validation = AnswerValidation(query=SIGNED_QUERY)
    assert validation.evidence_set is None
    assert validation.intent_coverage is None
    assert validation.state_coverage is None
    assert validation.gate_action is GateAction.PASSTHROUGH
    assert validation.source == RULES_VERSION


def test_answer_validation_carries_all_layers():
    facets = tuple(extract_facets(SIGNED_QUERY))
    requirement = derive_state_requirement(SIGNED_QUERY)
    state_coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])
    intent_coverage = build_evidence_coverage(
        derive_intent_requirement(DESIGN_QUERY), derive_evidence_needs(_spans(TECHFLOOR_QUOTE))
    )
    validation = AnswerValidation(
        query=SIGNED_QUERY,
        facets=facets,
        evidence_set=None,
        intent_coverage=intent_coverage,
        state_coverage=state_coverage,
        gate_action=GateAction.REWRITTEN_POSITIVE,
    )
    assert validation.state_coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert validation.intent_coverage.status is CoverageStatus.COMPLETE
    assert validation.gate_action is GateAction.REWRITTEN_POSITIVE


def test_dataclasses_are_frozen():
    requirement = derive_state_requirement(SIGNED_QUERY)
    coverage = build_state_coverage(requirement, [_evidence(SIGNED_DOC)])

    with pytest.raises(dataclasses.FrozenInstanceError):
        coverage.verdict = StateVerdict.NOOP
    with pytest.raises(dataclasses.FrozenInstanceError):
        AnswerValidation(query="q").gate_action = GateAction.REWRITTEN_NEGATIVE


# ---------------------------------------------------------------------------
# PR7.0 scope guard: foundation only, no runtime wiring
# ---------------------------------------------------------------------------

def test_module_has_no_runtime_imports():
    """PR7.0 is a pure foundation: importing ai_search/ui_services here would
    invert the dependency direction the next PRs rely on."""
    source = (Path(__file__).resolve().parents[1] / "evidence_runtime.py").read_text(encoding="utf-8")
    for forbidden in ("import ai_search", "import ui_services", "import streamlit",
                      "import sqlite3", "import lancedb", "import urllib"):
        assert forbidden not in source, forbidden


def test_state_coverage_never_reorders_or_grows_the_input():
    """The layer must not behave like a ranking stage: candidate evidences keep
    their input order and no evidence is invented."""
    requirement = derive_state_requirement(SIGNED_QUERY)
    pool = [_evidence(TEMPLATE_DOC), _evidence(DRAFT_DOC)]
    coverage = build_state_coverage(requirement, pool)
    assert [ev.document for ev in coverage.evidences] == [ev.document for ev in pool]
    assert len(coverage.evidences) <= len(pool)
    assert isinstance(coverage, StateCoverage)
