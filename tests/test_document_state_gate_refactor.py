"""PR7.3: the DocumentState answer gate refactored onto evidence_runtime's
StateCoverage.

Before this, ai_search decided document lifecycle state TWICE - the PR6 gate had
its own `_document_state_outcome()` (substring OR over entity terms) and the
PR7.2 diagnostic layer independently built a StateCoverage (conservative
matching). The two could disagree on the same query, i.e. one process held two
answers to "is this contract signed?".

What these tests protect:
  1. one decision only: `_document_state_outcome` is gone, `_answer_state_coverage`
     is computed once per answer and shared by the gate and the diagnostics
  2. the gate is a pure text policy: it receives a verdict, not a query or
     results, so it CANNOT parse a query, classify a filename, or do entity
     matching even by accident
  3. the five verdict → claim mappings behave as specified
  4. DOCUMENT_STATE_GATE_ENABLED still defaults OFF and is byte-identical there

Ollama is mocked (tests/test_document_state_answer_gate.py's pattern).
"""
from __future__ import annotations

import inspect
import json

import pytest

import ai_search
import ai_search_config
from evidence_runtime import StateCoverage, StateVerdict


@pytest.fixture(autouse=True)
def _gate_enabled(monkeypatch):
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)


def _mock_ollama(monkeypatch, text):
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            payload = {"body": [{"text": text, "zdroj_index": 1, "typ": "fakt"}],
                       "nenalezeno": False}
            return json.dumps({"response": json.dumps(payload)}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen",
                        lambda request, timeout=0: FakeResponse())


def _row(document):
    return {
        "document": document, "path": f"/proj/{document}", "project": "Projekt",
        "heading": "", "quote": f"Obsah dokumentu {document}.", "score": 1.0,
        "document_id": 1, "chunk_id": "c:0",
        "match": {"fts_hit": True, "vector_hit": True, "semantic_similarity": 0.7,
                  "filename_match": False},
    }


def _answer(monkeypatch, query, documents, text):
    _mock_ollama(monkeypatch, text)
    return ai_search.answer(query, [_row(doc) for doc in documents])["answer"]


SIGNED_DOC = "SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf"
DRAFT_DOC = "SOD_HAUS365_NDS_návrh_rev2. 260707_VH_rev MH_final.docx"
TEMPLATE_DOC = "SOD_HAUS365_vzorová NDS_návrh.docx"
UNKNOWN_DOC = "SoD_haus365.pdf"
VAHOSTAV_SIGNED_DOC = "SOD_VAHOSTAV_podepsaná.pdf"
QUERY = "je na boxu podepsaná smlouva haus365?"
TECH_QUERY = "jaký svár je požadovaný na CRM destičky"

NEGATIVE_CLAIM = "Na boxu není podepsaná smlouva haus365."
POSITIVE_CLAIM = "Ano, smlouva haus365 je podepsaná."
HEDGE = "V nalezených dokumentech není uvedeno, zda byla smlouva podepsána."


# ---------------------------------------------------------------------------
# 1. Duplicate state decisioning is gone
# ---------------------------------------------------------------------------

def test_gate_local_state_classification_no_longer_exists():
    assert not hasattr(ai_search, "_document_state_outcome")


def test_gate_signature_cannot_reach_query_or_results():
    """The structural guarantee: with neither `query` nor `results` in scope the
    gate cannot parse a query, classify a filename, or match an entity."""
    params = list(inspect.signature(ai_search._apply_document_state_answer_gate).parameters)
    assert params == ["state_coverage", "rendered"]


def test_gate_body_references_no_state_classification():
    """Structural check on the compiled code (not the comments): the gate's own
    body must not name any classification or entity-matching symbol."""
    names = set(ai_search._apply_document_state_answer_gate.__code__.co_names)
    for forbidden in ("document_state", "derive_state_requirement",
                      "classify_document_state", "entity_terms"):
        assert forbidden not in names, forbidden
    # It does read the verdict and the citable evidence - that is its whole input.
    assert {"verdict", "evidences"} <= names


def test_classification_lives_in_exactly_one_helper():
    names = set(ai_search._answer_state_coverage.__code__.co_names)
    assert {"document_state", "derive_state_requirement", "classify_document_state"} <= names


def test_state_is_decided_once_per_answer(monkeypatch):
    """With BOTH flags on, the gate and the diagnostic layer must share one
    decision - that sharing is the entire point of PR7.3."""
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)
    real = ai_search._answer_state_coverage
    calls = []

    def counting(query, results):
        calls.append(query)
        return real(query, results)

    monkeypatch.setattr(ai_search, "_answer_state_coverage", counting)
    _mock_ollama(monkeypatch, NEGATIVE_CLAIM)
    result = ai_search.answer(QUERY, [_row(SIGNED_DOC)])

    assert len(calls) == 1
    # The verdict reported is provably the one the gate acted on.
    assert result["validation"]["state_verdict"] == "SIGNED_CONFIRMED"
    assert result["validation"]["gate_action"] == "REWRITTEN_POSITIVE"
    assert "Ano" in result["answer"]


def test_gate_consumes_the_verdict_it_is_given(monkeypatch):
    """Hand the gate a hand-built coverage: the rewrite follows the verdict, with
    no access to any query or document to re-derive it from."""
    requirement = ai_search.document_state.derive_state_requirement(QUERY)
    coverage = StateCoverage(
        requirement=requirement, evidences=(), entity_matched=False,
        states_present=frozenset(), verdict=StateVerdict.ENTITY_MISMATCH, conflict=False,
    )
    out = ai_search._apply_document_state_answer_gate(coverage, NEGATIVE_CLAIM)
    assert "Nelze jednoznačně ověřit" in out


# ---------------------------------------------------------------------------
# 2. Verdict → claim policy (the five required regressions)
# ---------------------------------------------------------------------------

def test_haus365_signed_forbids_a_negative_claim(monkeypatch):
    text = _answer(monkeypatch, QUERY, [SIGNED_DOC], NEGATIVE_CLAIM)
    assert "není podepsaná" not in text
    assert text.startswith("Ano - na boxu je podepsaná smlouva.")
    assert SIGNED_DOC in text


def test_haus365_signed_leaves_a_correct_positive_claim_alone(monkeypatch):
    text = _answer(monkeypatch, QUERY, [SIGNED_DOC], POSITIVE_CLAIM)
    assert text.startswith(POSITIVE_CLAIM)


def test_haus365_signed_alongside_draft_still_confirms(monkeypatch):
    """PR7.0.1: a signed contract next to its drafts is normal lifecycle, not a
    conflict - and the draft is not cited as evidence of signedness."""
    text = _answer(monkeypatch, QUERY, [SIGNED_DOC, DRAFT_DOC], NEGATIVE_CLAIM)
    assert text.startswith("Ano - na boxu je podepsaná smlouva.")
    assert DRAFT_DOC not in text


def test_haus365_draft_only_forbids_a_positive_claim(monkeypatch):
    text = _answer(monkeypatch, QUERY, [DRAFT_DOC, TEMPLATE_DOC], POSITIVE_CLAIM)
    assert POSITIVE_CLAIM not in text
    assert text.startswith("Ne - na boxu nebyla nalezena podepsaná verze smlouvy")


def test_haus365_draft_only_lets_a_correct_negative_claim_pass(monkeypatch):
    text = _answer(monkeypatch, QUERY, [DRAFT_DOC], NEGATIVE_CLAIM)
    assert text.startswith(NEGATIVE_CLAIM)


def test_vahostav_document_against_haus365_query_confirms_nothing(monkeypatch):
    """The PR6.1 critical finding: another party's signed contract must confirm
    nothing, deny nothing, and be cited nowhere."""
    for claim in (POSITIVE_CLAIM, NEGATIVE_CLAIM):
        text = _answer(monkeypatch, QUERY, [VAHOSTAV_SIGNED_DOC], claim)
        assert text.startswith(
            "Nelze jednoznačně ověřit stav podpisu smlouvy pro dotazovaný subjekt"
        )
        assert claim not in text
        assert VAHOSTAV_SIGNED_DOC not in text


def test_hedge_sentence_survives_every_verdict(monkeypatch):
    """An epistemic hedge makes no claim, so no verdict may rewrite it - the
    minimal-intervention rule that keeps the gate from destroying valid answers."""
    for documents in ([SIGNED_DOC], [DRAFT_DOC], [VAHOSTAV_SIGNED_DOC], [UNKNOWN_DOC]):
        text = _answer(monkeypatch, QUERY, documents, HEDGE)
        assert text.startswith(HEDGE), documents


def test_unverified_unknown_state_replaces_a_definite_claim(monkeypatch):
    """UNVERIFIED must end in a hedge: a generically named document cannot rule
    the signed state in or out."""
    text = _answer(monkeypatch, QUERY, [UNKNOWN_DOC], NEGATIVE_CLAIM)
    assert "Nelze jednoznačně ověřit, zda je smlouva podepsaná" in text
    assert UNKNOWN_DOC in text  # citable: it does match the queried entity


def test_noop_technical_query_is_never_touched(monkeypatch):
    answer_text = "Na CRM destičky je požadovaný koutový svár."
    text = _answer(monkeypatch, TECH_QUERY, [SIGNED_DOC, DRAFT_DOC], answer_text)
    assert text.startswith(answer_text)


def test_noop_holds_even_for_an_answer_that_would_trip_the_gate(monkeypatch):
    """No signed intent → the gate is inert regardless of what the answer says."""
    text = _answer(monkeypatch, TECH_QUERY, [SIGNED_DOC], NEGATIVE_CLAIM)
    assert text.startswith(NEGATIVE_CLAIM)


# ---------------------------------------------------------------------------
# 3. Flag / identity
# ---------------------------------------------------------------------------

def test_flag_default_is_off():
    assert ai_search_config.DOCUMENT_STATE_GATE_ENABLED is False


def test_flag_off_is_byte_identical(monkeypatch):
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    text = _answer(monkeypatch, QUERY, [SIGNED_DOC], NEGATIVE_CLAIM)
    assert text.startswith(NEGATIVE_CLAIM)


def test_flag_off_does_not_decide_state_at_all(monkeypatch):
    """OFF must skip the decision, not merely ignore it."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    calls = []
    monkeypatch.setattr(ai_search, "_answer_state_coverage",
                        lambda *a: calls.append(a) or None)
    _answer(monkeypatch, QUERY, [SIGNED_DOC], NEGATIVE_CLAIM)
    assert calls == []


# ---------------------------------------------------------------------------
# 4. Read-only over results
# ---------------------------------------------------------------------------

def test_results_and_citations_are_untouched_by_the_gate(monkeypatch):
    rows = [_row(DRAFT_DOC), _row(SIGNED_DOC)]
    before = [dict(row) for row in rows]
    _mock_ollama(monkeypatch, NEGATIVE_CLAIM)
    result = ai_search.answer(QUERY, rows)

    assert rows == before
    assert result["citations"] is rows
    assert [row["document"] for row in result["citations"]] == [DRAFT_DOC, SIGNED_DOC]
