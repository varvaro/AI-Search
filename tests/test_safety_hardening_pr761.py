"""PR7.6.1 — Safety Hardening regression tests.

Covers the FAT-derived false-positive classes:
  A) LOI signed must not confirm a SoD query
  B) SafetyPeak signed SoD/BOZP confirms CONTRACT
  C/D) weak lexical near-miss → NO_EVIDENCE abstention
  E) NDS-named file with PALÁC DUNAJ body → DOCUMENT_PROJECT_CONFLICT
  F) Haus365 signed SoD → SIGNED_CONTRACT_CONFIRMED (= SIGNED_CONFIRMED)
"""
from __future__ import annotations

import json

import pytest

import ai_search
from document_classification import DocumentKind, classify_document_kind, detect_project_conflict
from document_state import derive_state_requirement
from evidence_runtime import (
    EvidenceSafetyStatus,
    StateVerdict,
    build_state_coverage,
    evaluate_evidence_safety,
)
from document_state import DocumentState, DocumentStateEvidence


def _ev(document: str, path: str = "") -> DocumentStateEvidence:
    # Mirror evidence_runtime tests: classify via W1, keep path for kind rules.
    from document_state import classify_document_state
    return classify_document_state(document, path)


def _row(document: str, path: str = "", quote: str = "", project: str = "240783160_Garáže_NDS"):
    return {
        "document": document,
        "path": path or f"/fixture/{document}",
        "quote": quote,
        "project": project,
        "heading": "",
        "score": 1.0,
    }


def _mock_ollama(monkeypatch, text: str = "Ano, smlouva je podepsaná."):
    payload = json.dumps({
        "body": [{"text": text, "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": payload}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)


# ---------------------------------------------------------------------------
# A) LOI signed + SoD query → SIGNED_OTHER_DOCUMENT_CONFIRMED
# ---------------------------------------------------------------------------

def test_a_loi_signed_does_not_confirm_sod_query():
    requirement = derive_state_requirement("existuje podepsaná smlouva na monolit s FERI?")
    coverage = build_state_coverage(requirement, [
        _ev("NDS_NOT251110_LOI_monolit_Feri_signed.pdf",
            "…/36_monolit_FERI_NOT251110/podepsané/NDS_NOT251110_LOI_monolit_Feri_signed.pdf"),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_OTHER_DOCUMENT_CONFIRMED
    assert coverage.verdict is not StateVerdict.SIGNED_CONTRACT_CONFIRMED
    assert coverage.evidences and "LOI" in coverage.evidences[0].document


def test_a_loi_gate_never_claims_signed_contract(flags_on, monkeypatch):
    _mock_ollama(monkeypatch, "Ano, podepsaná smlouva s FERI existuje.")
    rows = [_row(
        "NDS_NOT251110_LOI_monolit_Feri_signed.pdf",
        "…/podepsané/NDS_NOT251110_LOI_monolit_Feri_signed.pdf",
        quote="Letter of Intent FERI monolit podepsán.",
    )]
    result = ai_search.answer("existuje podepsaná smlouva na monolit s FERI?", rows)
    text = result["answer"]
    assert "podepsanou smlouvu" in text or "podepsaná smlouva" not in text.lower() or "ne potvrzenou" in text
    assert "ne potvrzenou podepsanou smlouvu" in text
    assert "Ano - na boxu je podepsaná smlouva" not in text
    assert result["validation"]["state_verdict"] == "SIGNED_OTHER_DOCUMENT_CONFIRMED"


# ---------------------------------------------------------------------------
# B) SafetyPeak signed → SIGNED_CONTRACT_CONFIRMED
# ---------------------------------------------------------------------------

def test_b_safetypeak_signed_is_contract_confirmed():
    requirement = derive_state_requirement("je podepsaná smlouva na BOZP?")
    coverage = build_state_coverage(requirement, [
        _ev("NOT250060_BOZP_SafetyPeak_podepsaná.pdf",
            "…/07_BOZP_ NOT250060_SafetyPeak/podepsaná/NOT250060_BOZP_SafetyPeak_podepsaná.pdf"),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONTRACT_CONFIRMED
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED  # wire-compat alias


# ---------------------------------------------------------------------------
# C) kniha betonů → NO_EVIDENCE (not "kniha revizí")
# ---------------------------------------------------------------------------

def test_c_kniha_betonu_near_miss_is_no_evidence():
    safety = evaluate_evidence_safety(
        "najdi knihu betonů",
        [_row(
            "KZP - TEXTOVÁ ČÁST.pdf",
            quote="Zhotovitel je povinen předložit knihu revizí protipožárních klapek. Pevnost betonu C30/37.",
        )],
    )
    assert safety.status is EvidenceSafetyStatus.NO_EVIDENCE
    assert "nenalezeno" in safety.message.casefold()


def test_c_kniha_betonu_not_confirmed_by_beton_filename():
    """FAT doc-10: a TP filename containing 'beton' must not open the gate."""
    safety = evaluate_evidence_safety(
        "najdi knihu betonů",
        [_row(
            "11_06_2026_TP beton monolit konstrukce.pdf",
            quote="Technologický předpis betonáže monolitické desky.",
        )],
    )
    assert safety.status is EvidenceSafetyStatus.NO_EVIDENCE


def test_c_kniha_betonu_answer_abstains(flags_on, monkeypatch):
    calls = {"n": 0}
    real = ai_search._call_ollama

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ai_search, "_call_ollama", counting)
    _mock_ollama(monkeypatch, "Zhotovitel předloží knihu revizí…")
    rows = [_row(
        "KZP - TEXTOVÁ ČÁST.pdf",
        quote="Zhotovitel je povinen předložit knihu revizí protipožárních klapek. Pevnost betonu.",
    )]
    result = ai_search.answer("najdi knihu betonů", rows)
    assert calls["n"] == 0  # LLM must not run
    assert "nenalezeno" in result["answer"].casefold()
    assert "kniha revizí" not in result["answer"]
    assert result["validation"]["evidence_safety"] == "NO_EVIDENCE"


# ---------------------------------------------------------------------------
# D) rovinnost betonu → NO_EVIDENCE
# ---------------------------------------------------------------------------

def test_d_rovinnost_betonu_without_rovinnost_is_no_evidence():
    safety = evaluate_evidence_safety(
        "jaké jsou požadavky na rovinnost betonové desky?",
        [_row(
            "TECHNICKÁ ZPRÁVA.pdf",
            quote="Betonová směs C30/37 dle ČSN EN 206. Betonárna Radlice.",
        )],
    )
    assert safety.status is EvidenceSafetyStatus.NO_EVIDENCE


def test_d_rovinnost_sdk_near_miss_is_no_evidence():
    """FAT qa-10: SDK 'rovinnost Q3' must not answer concrete-slab flatness."""
    safety = evaluate_evidence_safety(
        "jaké jsou požadavky na rovinnost betonové desky?",
        [_row(
            "KZP - TEXTOVÁ ČÁST.pdf",
            quote=(
                "Příplatek k SDK příčce za rovinnost kvality Q3 dle KNAUF. "
                "Samostatně: betonová směs C30/37 pro základovou desku."
            ),
        )],
    )
    assert safety.status is EvidenceSafetyStatus.NO_EVIDENCE


def test_d_rovinnost_betonu_cooccurrence_is_ok():
    safety = evaluate_evidence_safety(
        "jaké jsou požadavky na rovinnost betonové desky?",
        [_row(
            "TP betonová deska.pdf",
            quote="Požadovaná rovinnost betonové desky je max. 5 mm / 2 m.",
        )],
    )
    assert safety.status is EvidenceSafetyStatus.OK


# ---------------------------------------------------------------------------
# E) NDS_seznam TP a KZP + PALÁC DUNAJ → DOCUMENT_PROJECT_CONFLICT
# ---------------------------------------------------------------------------

def test_e_nds_named_file_with_dunaj_content_is_project_conflict():
    assert detect_project_conflict(
        "NDS_seznam TP a KZP.xlsx",
        "02_REALIZACE…/NDS_seznam TP a KZP.xlsx",
        "Projekt: PALÁC DUNAJ - Fit-out pro Evropský parlament",
    )
    safety = evaluate_evidence_safety(
        "jaký je seznam TP a KZP pro ND Smíchov?",
        [_row(
            "NDS_seznam TP a KZP.xlsx",
            "…/NDS_seznam TP a KZP.xlsx",
            quote="Projekt / Project: PALÁC DUNAJ - Fit-out pro Evropský parlament",
        )],
    )
    assert safety.status is EvidenceSafetyStatus.DOCUMENT_PROJECT_CONFLICT
    assert "NDS_seznam TP a KZP.xlsx" in safety.conflicted_documents


def test_e_dunaj_trap_must_not_answer_nds_query(flags_on, monkeypatch):
    _mock_ollama(monkeypatch, "Seznam TP pro Palác Dunaj obsahuje SDK příčky…")
    rows = [_row(
        "NDS_seznam TP a KZP.xlsx",
        "…/NDS_seznam TP a KZP.xlsx",
        quote="Projekt: PALÁC DUNAJ - Fit-out pro Evropský parlament. SDK příčky KNAUF.",
    )]
    result = ai_search.answer("jaký je seznam TP a KZP pro ND Smíchov?", rows)
    text = result["answer"]
    assert "Dunaj" not in text
    assert "nenalezeno" in text.casefold()
    assert result["validation"]["evidence_safety"] == "DOCUMENT_PROJECT_CONFLICT"
    assert result["citations"] == []
    assert not any("NDS_seznam" in str(c.get("document", "")) for c in result["citations"])


def test_bozp_query_does_not_confirm_unrelated_signed_sod():
    """FAT status-01 class: 'smlouva na BOZP' must not be confirmed by a
    random signed SoD that never mentions BOZP."""
    requirement = derive_state_requirement("je podepsaná smlouva na BOZP?")
    coverage = build_state_coverage(requirement, [
        _ev("Podepsaná smlouva - Hejtmanec_Tom.pdf"),
        _ev("SoD objednavatel_podepsaná.pdf"),
    ])
    assert coverage.verdict is StateVerdict.UNVERIFIED
    assert coverage.evidences == ()


def test_bozp_unverified_gate_blocks_invented_technical_answer(flags_on, monkeypatch):
    """Empty-evidence UNVERIFIED/ENTITY_MISMATCH must rewrite even when the LLM
    never says 'podepsaná' — otherwise status-01 invents a BOZP analysis from
    wrong SoDs."""
    _mock_ollama(
        monkeypatch,
        "Podle nalezené smlouvy vyplývají povinnosti BOZP pro zhotovitele monolitů.",
    )
    rows = [_row(
        "Podepsaná smlouva - Hejtmanec_Tom.pdf",
        "…/podepsané/Podepsaná smlouva - Hejtmanec_Tom.pdf",
        quote="Smlouva o dílo – monolitické konstrukce. Podepsáno.",
    )]
    result = ai_search.answer("je podepsaná smlouva na BOZP?", rows)
    text = result["answer"]
    assert "Nelze jednoznačně ověřit" in text
    assert result["validation"]["state_verdict"] in ("UNVERIFIED", "ENTITY_MISMATCH")
    # Must not regurgitate the unrelated SoD's technical content.
    assert "monolit" not in text.casefold()


# ---------------------------------------------------------------------------
# F) Haus365 signed contract → SIGNED_CONTRACT_CONFIRMED
# ---------------------------------------------------------------------------

def test_f_haus365_signed_is_contract_confirmed():
    requirement = derive_state_requirement("je na boxu podepsaná smlouva haus365?")
    coverage = build_state_coverage(requirement, [
        _ev("SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf"),
    ])
    assert coverage.verdict is StateVerdict.SIGNED_CONTRACT_CONFIRMED
    assert coverage.verdict.value == "SIGNED_CONFIRMED"  # benchmark wire compat


# ---------------------------------------------------------------------------
# Kind classifier pins
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kind", [
    ("NDS_NOT251110_LOI_monolit_Feri_signed.pdf", DocumentKind.LOI),
    ("NDS_NOT251110_SOD_monolit_Feri_podepsané.pdf", DocumentKind.CONTRACT),
    ("NOT250060_BOZP_SafetyPeak_podepsaná.pdf", DocumentKind.CONTRACT),
    ("SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf", DocumentKind.CONTRACT),
    ("objednávka_Hilt_Rent_podepsaná.pdf", DocumentKind.ORDER),
])
def test_document_kind_classifier(name, kind):
    assert classify_document_kind(name) is kind


def test_flags_off_keeps_pre_hardening_path(monkeypatch):
    """Flag-OFF must not abstain — byte-compatible with pre-PR7.6.1 for the
    non-state path (LLM still runs)."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    _mock_ollama(monkeypatch, "Nějaká technická odpověď o knize revizí.")
    rows = [_row(
        "KZP - TEXTOVÁ ČÁST.pdf",
        quote="kniha revizí protipožárních klapek",
    )]
    result = ai_search.answer("najdi knihu betonů", rows)
    assert "validation" not in result
    assert "kniha revizí" in result["answer"] or "technická" in result["answer"].lower() or "reviz" in result["answer"]
