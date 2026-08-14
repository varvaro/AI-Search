"""DocumentState foundation (W1) — pure unit tests, no retrieval/DB/Ollama."""
from __future__ import annotations

from document_state import (
    RULES_VERSION,
    DocumentState,
    classify_document_state,
    derive_state_requirement,
)


def test_signed_haus365_filename():
    ev = classify_document_state("SOD_HAUS365_podepsané.pdf")
    assert ev.state is DocumentState.SIGNED
    assert ev.source == RULES_VERSION
    assert ev.confidence == 1.0
    assert ev.signals


def test_for_signature_haus365_filename():
    ev = classify_document_state("SOD_HAUS365_k el. podpisu.pdf")
    assert ev.state is DocumentState.FOR_SIGNATURE


def test_for_signature_plus_podepsane_is_signed():
    ev = classify_document_state("SOD_HAUS365_k el. podpisu_podepsané.pdf")
    assert ev.state is DocumentState.SIGNED


def test_vzorova_navrh_is_template():
    ev = classify_document_state("SOD_HAUS365_vzorová_návrh.docx")
    assert ev.state is DocumentState.TEMPLATE


def test_navrh_only_is_draft():
    ev = classify_document_state("SOD_HAUS365_návrh.docx")
    assert ev.state is DocumentState.DRAFT


def test_generic_sod_is_unknown():
    ev = classify_document_state("SoD.pdf")
    assert ev.state is DocumentState.UNKNOWN
    assert ev.confidence == 0.0
    assert ev.signals == ()


def test_path_navrh_does_not_degrade_signed_filename():
    ev = classify_document_state(
        "SOD_HAUS365_podepsané.pdf",
        path="/Box/.../01_SoD/02_HAUS365/Návrh SoD/SOD_HAUS365_podepsané.pdf",
    )
    assert ev.state is DocumentState.SIGNED


def test_signed_contract_query_requirement():
    req = derive_state_requirement("je na boxu podepsaná smlouva haus365?")
    assert req.required_states == (DocumentState.SIGNED,)
    assert req.preferred_states == (DocumentState.SIGNED,)
    assert DocumentState.FOR_SIGNATURE in req.forbidden_states
    assert DocumentState.DRAFT in req.forbidden_states
    assert DocumentState.TEMPLATE in req.forbidden_states
    assert req.allow_unknown is False
    assert "haus365" in req.entity_terms
    assert any("smlouv" in t for t in req.doc_type_terms)
    assert req.source == RULES_VERSION


def test_contract_query_without_state_intent_is_empty():
    req = derive_state_requirement("smlouva haus365")
    assert req.required_states == ()
    assert req.preferred_states == ()
    assert req.forbidden_states == ()


def test_safetypeak_podepsana_is_signed_not_haus_specific():
    ev = classify_document_state("SafetyPeak_podepsaná.pdf")
    assert ev.state is DocumentState.SIGNED


def test_for_signature_query_requirement():
    req = derive_state_requirement("najdi smlouvu haus365 k podpisu")
    assert req.required_states == (DocumentState.FOR_SIGNATURE,)
    assert DocumentState.SIGNED not in req.required_states
    assert req.preferred_states == (DocumentState.FOR_SIGNATURE,)
    assert req.allow_unknown is False
    assert "haus365" in req.entity_terms
