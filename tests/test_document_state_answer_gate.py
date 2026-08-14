"""PR6/PR6.2: DocumentState answer safety gate (ai_search._apply_document_state_answer_gate
and the wiring inside ai_search.answer()).

Golden regression tests for the HAUS365 failure class: retrieval correctly
finds a SIGNED contract, but the LLM answer denies it / cites an unrelated
document. Ollama is mocked (see tests/test_answer_quality.py's `_mock_ollama`
pattern) - deterministic, no live server needed.

PR6.2 fixes covered here (see PR6.1 audit):
  * entity fallback removed - an entity-name mismatch (e.g. a different
    party's SIGNED contract) must resolve to NEOVERENO, never POSITIVE_SIGNED,
    with no citation of the unrelated document.
  * negative-claim detection tightened to phrase-adjacent patterns so
    epistemic hedges ("není uvedeno, zda...") are never treated as a claim.
  * signed intent detection widened to "podpis"/"signatura" (CZ only).
  * DOCUMENT_STATE_GATE_ENABLED feature flag - gate is fully inert when OFF
    (the default), byte-identical to pre-PR6 answer() behaviour.

Out of scope here (unchanged, not touched by PR6/PR6.2):
  * retrieval / RRF / scoring / ranking - `results` order is read-only input
  * QE / Aux Term Coverage
  * prompts sent to the LLM (guidance/schema selection)
"""
from __future__ import annotations

import json

import pytest

import ai_search
from document_state import DocumentState
from evidence_runtime import StateVerdict


@pytest.fixture(autouse=True)
def _gate_enabled(monkeypatch):
    """Default flag is OFF (see ai_search_config.DOCUMENT_STATE_GATE_ENABLED) -
    every test in this file exercises the gate, so turn it ON explicitly,
    mirroring the AUXILIARY_TERM_COVERAGE_ENABLED test pattern."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)


def _mock_ollama(monkeypatch, responses):
    calls = []
    remaining = list(responses)

    class FakeResponse:
        def __init__(self, text):
            self._text = text
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": self._text}).encode()

    def fake_urlopen(request, timeout=0):
        calls.append(json.loads(request.data.decode()))
        text = remaining.pop(0) if remaining else calls[-1].get("_last_response", "")
        return FakeResponse(text)

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    return calls


def _row(document, path=None, document_id=None):
    return {
        "document": document,
        "path": path or f"/proj/{document}",
        "project": "Projekt",
        "heading": "",
        "quote": f"Obsah dokumentu {document}.",
        "score": 1.0,
        "document_id": document_id,
        "match": {"fts_hit": True, "vector_hit": True, "semantic_similarity": 0.7, "filename_match": False},
    }


SIGNED_DOC = "SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf"
TP_DOC = "TP 2.3. - NDS-zajištění stav. jamy-TI M1+M2 - dodatek 2.pdf"
TEMPLATE_DOC = "SOD_HAUS365_vzorová NDS_návrh.docx"
UNKNOWN_DOC = "SoD.pdf"
UNKNOWN_MATCHED_DOC = "SoD_haus365.pdf"  # entity-matching but no strong state signal
VAHOSTAV_SIGNED_DOC = "SOD_VAHOSTAV_podepsaná.pdf"  # different party, also SIGNED
QUERY = "je na boxu podepsaná smlouva haus365?"


def _concise_payload(text, zdroj_index=1):
    return {"body": [{"text": text, "zdroj_index": zdroj_index, "typ": "fakt"}], "nenalezeno": False}


# ---------------------------------------------------------------------------
# TEST 1: SIGNED present + unrelated doc - negative claim must be overridden
# ---------------------------------------------------------------------------

def test_signed_present_forbids_negative_absence_claim(monkeypatch):
    rows = [_row(SIGNED_DOC), _row(TP_DOC)]
    payload = _concise_payload(
        "Na boxu není podepsaná smlouva haus365, dostupný je pouze dodatek TP.",
        zdroj_index=2,
    )
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    text = result["answer"]

    # PASS criterion: positive signed classification, citing the SIGNED doc.
    assert "Ano" in text
    assert SIGNED_DOC in text
    # FAIL criterion: the negative-absence claim must not survive the gate.
    assert "není podepsaná" not in text
    assert result["citations"] == rows  # citations/results themselves untouched


def test_signed_present_and_llm_already_correct_is_untouched(monkeypatch):
    """If the model already answers correctly, the gate must not rewrite it -
    only a forbidden claim triggers an override."""
    rows = [_row(SIGNED_DOC), _row(TP_DOC)]
    payload = _concise_payload(f"Ano, na boxu je podepsaná smlouva haus365 ({SIGNED_DOC}).")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    assert "je podepsaná smlouva haus365" in result["answer"]
    assert SIGNED_DOC in result["answer"]


# ---------------------------------------------------------------------------
# TEST 2: only a TEMPLATE/DRAFT document - false positive must be corrected,
# an accurate negative claim must pass through untouched.
# ---------------------------------------------------------------------------

def test_template_only_forbids_false_positive_signed_claim(monkeypatch):
    rows = [_row(TEMPLATE_DOC)]
    payload = _concise_payload("Ano, na boxu je podepsaná smlouva haus365.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    text = result["answer"]

    assert "Ano, na boxu je podepsaná smlouva haus365." not in text
    assert "nebyla nalezena podepsaná" in text
    assert TEMPLATE_DOC in text


def test_template_only_allows_accurate_negative_claim(monkeypatch):
    rows = [_row(TEMPLATE_DOC)]
    payload = _concise_payload(
        "Na boxu není podepsaná smlouva haus365, k dispozici je pouze návrh."
    )
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    # Negative claim is legitimate here (no SIGNED, no UNKNOWN) - must pass through.
    assert "Na boxu není podepsaná smlouva haus365, k dispozici je pouze návrh." in result["answer"]


# ---------------------------------------------------------------------------
# TEST 3: only a generically-named UNKNOWN document that DOES match the
# queried entity - neither a confident positive nor negative claim may stand.
# ---------------------------------------------------------------------------

def test_unknown_state_document_is_classified_unknown():
    from document_state import classify_document_state
    ev = classify_document_state(UNKNOWN_DOC)
    assert ev.state is DocumentState.UNKNOWN


def test_unknown_matched_forbids_negative_absence_claim(monkeypatch):
    rows = [_row(UNKNOWN_MATCHED_DOC)]
    payload = _concise_payload("Na boxu není podepsaná smlouva haus365.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    text = result["answer"]
    assert "není podepsaná smlouva haus365" not in text
    assert "Nelze jednoznačně ověřit" in text
    assert UNKNOWN_MATCHED_DOC in text


def test_unknown_matched_forbids_false_positive_claim_too(monkeypatch):
    rows = [_row(UNKNOWN_MATCHED_DOC)]
    payload = _concise_payload("Ano, na boxu je podepsaná smlouva haus365.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    assert "Nelze jednoznačně ověřit" in result["answer"]


# ---------------------------------------------------------------------------
# PR6.2 TEST ENTITY MISMATCH: a different party's SIGNED contract must never
# confirm the queried entity's signed status - no fallback to all results.
# ---------------------------------------------------------------------------

def test_entity_mismatch_forbids_positive_signed_claim(monkeypatch):
    rows = [_row(VAHOSTAV_SIGNED_DOC)]
    payload = _concise_payload("Ano, je podepsaná smlouva haus365.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    text = result["answer"]

    assert "Ano, je podepsaná smlouva" not in text
    assert "Nelze jednoznačně ověřit" in text
    # Critical: the unrelated party's document must never be cited as proof.
    assert VAHOSTAV_SIGNED_DOC not in text


def test_entity_mismatch_also_forbids_negative_claim(monkeypatch):
    """A denial is equally unsafe here - we simply have no evidence either
    way for the queried entity, so NEOVERENO must override both directions."""
    rows = [_row(VAHOSTAV_SIGNED_DOC)]
    payload = _concise_payload("Ne, na boxu není podepsaná smlouva haus365.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    text = result["answer"]
    assert "není podepsaná smlouva haus365" not in text
    assert "Nelze jednoznačně ověřit" in text
    assert VAHOSTAV_SIGNED_DOC not in text


def test_entity_mismatch_verdict_carries_no_evidence():
    """PR7.3: the verdict comes from evidence_runtime, not from a gate-local
    classification. ENTITY_MISMATCH replaces PR6's NEOVERENO-without-docs."""
    coverage = ai_search._answer_state_coverage(QUERY, [_row(VAHOSTAV_SIGNED_DOC)])
    assert coverage.verdict is StateVerdict.ENTITY_MISMATCH
    assert coverage.requirement.entity_terms  # query does name an entity ("haus365")
    assert coverage.evidences == ()  # no fallback to the unfiltered result set


# ---------------------------------------------------------------------------
# PR6.2 TEST ENTITY MATCH: sanity check that a correctly-named SIGNED
# document for the queried entity still resolves to POSITIVE_SIGNED.
# ---------------------------------------------------------------------------

def test_entity_match_still_resolves_positive_signed(monkeypatch):
    rows = [_row(SIGNED_DOC)]
    payload = _concise_payload("Na boxu není podepsaná smlouva haus365.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    text = result["answer"]
    assert "Ano" in text
    assert SIGNED_DOC in text

    coverage = ai_search._answer_state_coverage(QUERY, rows)
    assert coverage.verdict is StateVerdict.SIGNED_CONFIRMED
    assert coverage.evidences and coverage.evidences[0].document == SIGNED_DOC


# ---------------------------------------------------------------------------
# PR6.2 TEST HEDGE: epistemic hedges must never be read as a negative claim.
# ---------------------------------------------------------------------------

def test_hedge_sentence_is_not_treated_as_negative_claim(monkeypatch):
    rows = [_row(SIGNED_DOC), _row(TP_DOC)]
    hedge = "V nalezených dokumentech není uvedeno, zda byla smlouva podepsána."
    payload = _concise_payload(hedge)
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    assert result["answer"].startswith(hedge)


@pytest.mark.parametrize("hedge", [
    "Není uvedeno, zda byla smlouva podepsána.",
    "Nelze ověřit, zda je smlouva podepsaná.",
    "Není jasné, zda byla smlouva podepsána.",
])
def test_hedge_variants_are_not_matched_by_negative_regex(hedge):
    folded = ai_search._fold_plain(hedge)
    assert not ai_search._STATE_GATE_NEGATIVE_SIGNED_RE.search(folded)


# ---------------------------------------------------------------------------
# PR6.2 TEST NOT FOUND: "nenašel/nenalezl jsem" must be caught as a negative
# claim when the intent requires SIGNED.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Nenašel jsem podepsanou verzi smlouvy.",
    "Nenalezl jsem podepsanou verzi smlouvy.",
    "Nepodařilo se najít podepsanou verzi smlouvy.",
])
def test_not_found_phrasing_is_caught_as_negative_claim(text):
    folded = ai_search._fold_plain(text)
    assert ai_search._STATE_GATE_NEGATIVE_SIGNED_RE.search(folded)


def test_not_found_phrasing_triggers_override_when_signed_present(monkeypatch):
    rows = [_row(SIGNED_DOC)]
    payload = _concise_payload("Nenašel jsem podepsanou verzi smlouvy haus365.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    text = result["answer"]
    assert "Nenašel jsem podepsanou" not in text
    assert "Ano" in text
    assert SIGNED_DOC in text


# ---------------------------------------------------------------------------
# PR6.2 TEST NOOP: unrelated technical queries must never engage the gate.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,answer_text", [
    ("jaký svár je požadovaný na CRM destičky", "Na CRM destičky je požadovaný koutový svár."),
    ("Pentaflex", "Pentaflex je těsnicí pás pro dilatační spáry."),
    ("jaký beton je v základové desce", "V základové desce je beton C25/30."),
])
def test_noop_for_queries_without_signed_intent(monkeypatch, query, answer_text):
    rows = [_row(SIGNED_DOC), _row(TP_DOC)]
    payload = _concise_payload(answer_text)
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(query, rows)
    assert answer_text in result["answer"]

    assert ai_search._answer_state_coverage(query, rows).verdict is StateVerdict.NOOP


# ---------------------------------------------------------------------------
# Regression: queries without a signed-contract state intent are untouched.
# ---------------------------------------------------------------------------

def test_query_without_state_intent_is_a_pure_noop(monkeypatch):
    rows = [_row(SIGNED_DOC), _row(TP_DOC)]
    payload = _concise_payload("Pentaflex je těsnicí pás pro dilatační spáry.")
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer("Pentaflex", rows)
    assert "Pentaflex je těsnicí pás" in result["answer"]

    assert ai_search._answer_state_coverage("Pentaflex", rows).verdict is StateVerdict.NOOP


def test_gate_helper_is_identity_for_noop_outcome():
    rows = [_row(SIGNED_DOC)]
    rendered = "Cokoliv, i něco, co by jinak spustilo bránu."
    coverage = ai_search._answer_state_coverage("Pentaflex", rows)
    assert ai_search._apply_document_state_answer_gate(coverage, rendered) == rendered


# ---------------------------------------------------------------------------
# PR6.2 TEST FLAG: gate is fully inert (byte-identical answer()) when
# DOCUMENT_STATE_GATE_ENABLED is False - including the default config value.
# ---------------------------------------------------------------------------

def test_flag_off_leaves_answer_byte_identical(monkeypatch):
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    rows = [_row(SIGNED_DOC), _row(TP_DOC)]
    payload = _concise_payload(
        "Na boxu není podepsaná smlouva haus365, dostupný je pouze dodatek TP."
    )
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer(QUERY, rows)
    # With the gate OFF, the (otherwise-forbidden) negative claim must survive.
    assert "není podepsaná smlouva haus365" in result["answer"]


def test_default_flag_value_is_off():
    import ai_search_config
    assert ai_search_config.DOCUMENT_STATE_GATE_ENABLED is False
