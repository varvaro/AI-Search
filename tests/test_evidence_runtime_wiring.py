"""PR7.2 / PR7.6.1: Evidence Runtime Validation wired into ai_search.answer().

Originally (PR7.2) a DIAGNOSTIC-only layer: flag ON added a `validation` key
and was forbidden from changing answer text. PR7.6.1 adds a safety consumer of
the same layer — weak lexical overlap and project-content conflicts abstain
before the LLM runs. Identity contracts below therefore apply only when the
safety check returns OK (matched evidence); abstention cases are covered in
tests/test_safety_hardening_pr761.py.
"""
from __future__ import annotations

import copy
import json

import pytest

import ai_search
import ai_search_config
from evidence_runtime import RULES_VERSION


@pytest.fixture(autouse=True)
def _gate_off(monkeypatch):
    """PR7.2 must be observable on its own: the PR6 gate stays OFF (its default)
    unless a test turns it on, so a rewritten answer cannot be mistaken for a
    diagnostic-layer side effect."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)


def _mock_ollama(monkeypatch, responses):
    remaining = list(responses)

    class FakeResponse:
        def __init__(self, text):
            self._text = text
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": self._text}).encode()

    def fake_urlopen(request, timeout=0):
        return FakeResponse(remaining.pop(0) if remaining else "")

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)


def _row(document, quote="Obsah dokumentu.", score=1.0, document_id=1, chunk_id="c:0"):
    return {
        "document": document,
        "path": f"/proj/{document}",
        "project": "Projekt",
        "heading": "",
        "quote": quote,
        "score": score,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "match": {"fts_hit": True, "vector_hit": True, "semantic_similarity": 0.7,
                  "filename_match": False},
    }


def _payload(text, zdroj_index=1):
    return json.dumps({"body": [{"text": text, "zdroj_index": zdroj_index, "typ": "fakt"}],
                       "nenalezeno": False})


SIGNED_DOC = "SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf"
DRAFT_DOC = "SOD_HAUS365_vzorová NDS_návrh.docx"
VAHOSTAV_SIGNED_DOC = "SOD_VAHOSTAV_podepsaná.pdf"

SIGNED_QUERY = "je na boxu podepsaná smlouva haus365?"
CRM_QUERY = "jaký svár je požadovaný na CRM destičky"
DESIGN_QUERY = "bude se brokovat základová deska 3PP"

TECH_QUOTE = ("Otryskání podkladu před provedením lité podlahy, brokování. "
              "skladba P3 základová deska ŽB 250 mm ve 3. pp")


def _answer(monkeypatch, query, rows, enabled, text="Odpověď z dokumentu."):
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", enabled)
    _mock_ollama(monkeypatch, [_payload(text)])
    return ai_search.answer(query, rows)


# ---------------------------------------------------------------------------
# Flag OFF: exact identity
# ---------------------------------------------------------------------------

def test_flag_default_is_off():
    assert ai_search_config.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False
    assert ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False


def test_off_adds_no_validation_key(monkeypatch):
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False)
    assert "validation" not in result
    assert set(result) == {"answer", "citations", "model", "confidence"}


def test_on_differs_from_off_only_by_the_validation_key(monkeypatch):
    """The identity contract: pop `validation` from the ON result and the two
    dicts must be equal - same answer text, citations, model, confidence."""
    off = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC), _row(DRAFT_DOC)], enabled=False)
    on = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC), _row(DRAFT_DOC)], enabled=True)

    assert on.pop("validation")
    assert on == off
    assert on["answer"] == off["answer"]


def test_answer_text_is_byte_identical_when_evidence_safety_is_ok(monkeypatch):
    """When retrieved rows actually cover the query tokens, PR7.6.1 safety
    returns OK and the diagnostic layer must not change answer text."""
    cases = [
        (SIGNED_QUERY, [_row(SIGNED_DOC, quote=TECH_QUOTE)]),
        (CRM_QUERY, [_row(
            "TP_CRM_desticky.pdf",
            quote="Požadovaný svár na CRM destičky je koutový.",
        )]),
        (DESIGN_QUERY, [_row(SIGNED_DOC, quote=TECH_QUOTE)]),
        ("Pentaflex", [_row(
            "Pentaflex_KB80_navod.pdf",
            quote="Montážní návod těsnicího pásu Pentaflex KB80.",
        )]),
    ]
    for query, rows in cases:
        off = _answer(monkeypatch, query, copy.deepcopy(rows), enabled=False)
        on = _answer(monkeypatch, query, copy.deepcopy(rows), enabled=True)
        assert on["answer"] == off["answer"], query
        assert on["confidence"] == off["confidence"], query
        assert on["model"] == off["model"], query
        assert on["validation"]["evidence_safety"] == "OK", query


def test_off_does_not_run_the_foundation_layers(monkeypatch):
    """"No new computation when OFF" - the layer's entry point must not be
    reached at all, not merely produce nothing."""
    calls = []
    monkeypatch.setattr(ai_search, "_answer_validation_metadata",
                        lambda *a, **k: calls.append(a) or {})
    _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False)
    assert calls == []

    _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True)
    assert len(calls) == 1


def test_empty_results_path_is_untouched(monkeypatch):
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)
    result = ai_search.answer(SIGNED_QUERY, [])
    assert result == {"answer": "Odpověď nelze vytvořit bez citací.", "citations": [],
                      "confidence": "red"}


def test_ollama_failure_path_carries_no_validation(monkeypatch):
    """No answer was produced, so there is nothing to validate."""
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)

    def boom(request, timeout=0):
        raise TimeoutError("ollama down")

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", boom)
    result = ai_search.answer(SIGNED_QUERY, [_row(SIGNED_DOC)])
    assert "validation" not in result
    assert "Ollama je nedostupná" in result["answer"]


# ---------------------------------------------------------------------------
# Flag ON: shape of the added metadata
# ---------------------------------------------------------------------------

def test_on_adds_validation_metadata(monkeypatch):
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True)
    validation = result["validation"]

    assert validation["rules_version"] == RULES_VERSION
    assert validation["state_verdict"] == "SIGNED_CONFIRMED"
    assert validation["intent_coverage"] in {"COMPLETE", "PARTIAL", "INSUFFICIENT"}
    assert validation["gate_action"] == "PASSTHROUGH"
    assert validation["span_source"] == "merged_rows"
    assert validation["evidence_spans"] == 1


def test_validation_is_json_serializable(monkeypatch):
    """It travels inside answer()'s dict, which app.py and the benchmark
    harness both serialize."""
    result = _answer(monkeypatch, DESIGN_QUERY, [_row("techfloor.xls", quote=TECH_QUOTE)],
                     enabled=True)
    assert json.loads(json.dumps(result["validation"])) == result["validation"]


def test_validation_carries_no_chunk_text(monkeypatch):
    """Diagnostics must not duplicate the quotes `citations` already carries."""
    result = _answer(monkeypatch, DESIGN_QUERY, [_row("techfloor.xls", quote=TECH_QUOTE)],
                     enabled=True)
    assert TECH_QUOTE not in json.dumps(result["validation"], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Real verdicts
# ---------------------------------------------------------------------------

def test_crm_query_is_noop(monkeypatch):
    """A technical query has no signed-contract intent, so the state layer must
    stay inert even with a signed contract sitting in the pool."""
    result = _answer(monkeypatch, CRM_QUERY, [_row(SIGNED_DOC), _row("CRM destička.pdf")],
                     enabled=True)
    validation = result["validation"]

    assert validation["state_verdict"] == "NOOP"
    assert validation["state_documents"] == []
    assert validation["state_entity_matched"] is False


def test_haus365_signed_query_is_signed_confirmed(monkeypatch):
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC), _row(DRAFT_DOC)],
                     enabled=True)
    validation = result["validation"]

    assert validation["state_verdict"] == "SIGNED_CONFIRMED"
    assert validation["state_entity_matched"] is True
    # Only the signed document is citable evidence of signedness; the coexisting
    # draft is normal lifecycle, not a conflict (PR7.0.1).
    assert [doc["document"] for doc in validation["state_documents"]] == [SIGNED_DOC]
    assert validation["state_documents"][0]["state"] == "SIGNED"


def test_entity_mismatch_is_reported(monkeypatch):
    """Another party's signed contract must not be confirmed or cited for a
    HAUS365 query - the PR6.1 critical finding, now observable."""
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(VAHOSTAV_SIGNED_DOC)], enabled=True)
    validation = result["validation"]

    assert validation["state_verdict"] == "ENTITY_MISMATCH"
    assert validation["state_entity_matched"] is False
    assert validation["state_documents"] == []


def test_draft_only_is_unsigned_confirmed(monkeypatch):
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(DRAFT_DOC)], enabled=True)
    assert result["validation"]["state_verdict"] == "UNSIGNED_CONFIRMED"


def test_intent_coverage_reflects_real_evidence(monkeypatch):
    """A design query requires TECHNOLOGY+STRUCTURE; a quote containing both
    covers it, a thin quote does not."""
    covered = _answer(monkeypatch, DESIGN_QUERY,
                      [_row("techfloor.xls", quote=TECH_QUOTE)], enabled=True)["validation"]
    assert covered["required_needs"] == ["TECHNOLOGY", "STRUCTURE"]
    assert covered["intent_coverage"] == "COMPLETE"
    assert covered["missing_needs"] == []

    thin = _answer(monkeypatch, DESIGN_QUERY,
                   [_row("x.pdf", quote="obecné poznámky")], enabled=True)["validation"]
    assert thin["intent_coverage"] == "INSUFFICIENT"
    assert thin["missing_needs"] == ["TECHNOLOGY", "STRUCTURE"]


def test_captured_spans_are_preferred_over_merged_rows(monkeypatch):
    """PR7.1's `_evidence_spans` attributes facet evidence per chunk instead of
    to one concatenated quote."""
    row = _row("techfloor.xls", quote="Otryskání ... skladba")
    row["_evidence_spans"] = [
        {"document_id": 1, "chunk_id": "a:0", "path": row["path"], "document": "techfloor.xls",
         "quote": "Otryskání podkladu před provedením lité podlahy, brokování.", "score": 3.0},
        {"document_id": 1, "chunk_id": "a:1", "path": row["path"], "document": "techfloor.xls",
         "quote": "skladba P3 základová deska ŽB 250 mm ve 3. pp", "score": 2.0},
    ]
    validation = _answer(monkeypatch, DESIGN_QUERY, [row], enabled=True)["validation"]

    assert validation["span_source"] == "captured_spans"
    assert validation["evidence_spans"] == 2
    assert validation["intent_coverage"] == "COMPLETE"


# ---------------------------------------------------------------------------
# Read-only over `results`
# ---------------------------------------------------------------------------

def test_results_are_not_mutated_reordered_or_rescored(monkeypatch):
    rows = [_row(DRAFT_DOC, score=0.4, document_id=2, chunk_id="b:0"),
            _row(SIGNED_DOC, score=9.9, document_id=1, chunk_id="a:0")]
    before = copy.deepcopy(rows)

    result = _answer(monkeypatch, SIGNED_QUERY, rows, enabled=True)

    assert rows == before                      # no mutation, no rescoring
    assert result["citations"] is rows         # same object, same order
    assert [row["document"] for row in result["citations"]] == \
           [row["document"] for row in before]


def test_citations_order_is_identical_with_and_without_the_flag(monkeypatch):
    rows = [_row(DRAFT_DOC, score=0.4), _row(SIGNED_DOC, score=9.9)]
    off = _answer(monkeypatch, SIGNED_QUERY, copy.deepcopy(rows), enabled=False)
    on = _answer(monkeypatch, SIGNED_QUERY, copy.deepcopy(rows), enabled=True)
    assert [r["document"] for r in on["citations"]] == [r["document"] for r in off["citations"]]
    assert [r["score"] for r in on["citations"]] == [r["score"] for r in off["citations"]]


def test_prompt_sent_to_the_llm_is_unchanged(monkeypatch):
    """No prompt/guidance/schema change: the request bodies must match exactly."""
    captured = []

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": _payload("text")}).encode()

    def fake_urlopen(request, timeout=0):
        captured.append(json.loads(request.data.decode()))
        return FakeResponse()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    rows = [_row(SIGNED_DOC)]

    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    ai_search.answer(SIGNED_QUERY, rows)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)
    ai_search.answer(SIGNED_QUERY, rows)

    assert len(captured) == 2
    assert captured[0] == captured[1]


# ---------------------------------------------------------------------------
# Robustness / PR6 coexistence
# ---------------------------------------------------------------------------

def test_diagnostic_failure_never_breaks_the_answer(monkeypatch):
    """PR7.3.1 TEST 1: the validation builder raises → the answer is delivered
    anyway, the failure is recorded, and nothing propagates out of answer()."""
    def boom(*a, **k):
        raise RuntimeError("diagnostic exploded")

    monkeypatch.setattr(ai_search, "_answer_validation_metadata", boom)
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True,
                     text="Smlouva haus365 je podepsaná.")

    assert result["answer"].startswith("Smlouva haus365 je podepsaná.")
    assert result["citations"]
    assert result["validation"] == {
        "error": "RuntimeError: diagnostic exploded",
        "source": "evidence_runtime",
        "status": "FAILED",
    }


def test_failed_validation_leaves_the_answer_byte_identical(monkeypatch):
    """A failed diagnostic must produce the same answer as no diagnostic at all."""
    rows = [_row(SIGNED_DOC), _row(DRAFT_DOC)]
    off = _answer(monkeypatch, SIGNED_QUERY, copy.deepcopy(rows), enabled=False)

    monkeypatch.setattr(ai_search, "_answer_validation_metadata",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("nope")))
    broken = _answer(monkeypatch, SIGNED_QUERY, copy.deepcopy(rows), enabled=True)

    assert broken["answer"] == off["answer"]
    assert broken["confidence"] == off["confidence"]
    assert broken["model"] == off["model"]
    assert [r["document"] for r in broken["citations"]] == \
           [r["document"] for r in off["citations"]]
    assert broken["validation"]["status"] == "FAILED"


def test_successful_validation_reports_status_ok(monkeypatch):
    """PR7.3.1 TEST 2: the normal path is unchanged and carries a real verdict."""
    validation = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)],
                         enabled=True)["validation"]
    assert validation["status"] == "OK"
    assert validation["state_verdict"] == "SIGNED_CONFIRMED"
    assert "error" not in validation


def test_flag_off_never_calls_a_raising_builder(monkeypatch):
    """PR7.3.1 TEST 3: with the flag OFF the builder is not merely ignored - it
    is never called, so even an exploding one cannot affect the answer."""
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise RuntimeError("must never run")

    monkeypatch.setattr(ai_search, "_answer_validation_metadata", boom)
    off = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False,
                  text="Smlouva haus365 je podepsaná.")

    assert calls == []
    assert off["answer"].startswith("Smlouva haus365 je podepsaná.")
    assert "validation" not in off
    assert set(off) == {"answer", "citations", "model", "confidence"}


def test_llm_failure_and_validation_failure_together_do_not_crash(monkeypatch):
    """PR7.3.1 TEST 4: both layers broken at once must preserve the existing
    Ollama error handling, unchanged."""
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)
    monkeypatch.setattr(ai_search, "_answer_validation_metadata",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    def dead(request, timeout=0):
        raise TimeoutError("ollama down")

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", dead)
    result = ai_search.answer(SIGNED_QUERY, [_row(SIGNED_DOC)])

    assert "Ollama je nedostupná: TimeoutError" in result["answer"]
    assert result["error"] == "ollama down"
    assert result["citations"]
    assert "validation" not in result


# ---------------------------------------------------------------------------
# PR7.3.1: the shared state decision must not be able to kill an answer either
# ---------------------------------------------------------------------------

NEGATIVE_CLAIM = "Na boxu není podepsaná smlouva haus365."


def test_gate_rewrites_this_claim_when_the_state_decision_works(monkeypatch):
    """Counterfactual guard: without this the two tests below would pass even if
    the gate never touched NEGATIVE_CLAIM in the first place."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False,
                     text=NEGATIVE_CLAIM)
    assert result["answer"].startswith("Ano - na boxu je podepsaná smlouva.")


def test_state_decision_failure_leaves_the_answer_ungated(monkeypatch):
    """`_answer_state_coverage` feeds the gate. If it fails the answer must pass
    through with its pre-gate text - never raise, never half-rewritten. The gate
    stays silent on purpose: an unknown state cannot justify a rewrite."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    monkeypatch.setattr(ai_search, "_answer_state_coverage",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("state exploded")))

    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False,
                     text=NEGATIVE_CLAIM)
    assert result["answer"].startswith(NEGATIVE_CLAIM)


def test_gate_application_failure_leaves_the_answer_ungated(monkeypatch):
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    monkeypatch.setattr(ai_search, "_apply_document_state_answer_gate",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("gate exploded")))

    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False,
                     text=NEGATIVE_CLAIM)
    assert result["answer"].startswith(NEGATIVE_CLAIM)


def test_state_decision_failure_is_reported_in_validation(monkeypatch):
    """With both flags on, a broken state decision surfaces as FAILED instead of
    silently reporting a verdict the gate never applied."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    monkeypatch.setattr(ai_search, "_answer_state_coverage",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("state exploded")))

    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True)
    assert result["answer"]
    assert result["validation"]["status"] == "FAILED"
    assert result["validation"]["source"] == "evidence_runtime"


def test_gate_flag_off_never_attempts_the_state_decision(monkeypatch):
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    calls = []
    monkeypatch.setattr(ai_search, "_answer_state_coverage",
                        lambda *a: calls.append(a) or None)
    _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False)
    assert calls == []


def test_failures_are_logged_not_swallowed_silently(monkeypatch, caplog):
    """A swallowed exception with no trace anywhere is an operational trap."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    monkeypatch.setattr(ai_search, "_answer_state_coverage",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("state exploded")))
    monkeypatch.setattr(ai_search, "_answer_validation_metadata",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("diag exploded")))

    with caplog.at_level("WARNING", logger="ai_search.search"):
        _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "STATE_GATE_SKIPPED" in messages
    assert "VALIDATION_FAILED" in messages


def test_keyboard_interrupt_is_not_swallowed(monkeypatch):
    """`except Exception` must stay - a cancellation is not a diagnostic error."""
    monkeypatch.setattr(ai_search, "_answer_validation_metadata",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True)


def test_rows_without_chunk_identity_still_produce_a_verdict(monkeypatch):
    """build_evidence_set skips rows lacking document_id/chunk_id; the state
    layer works off filenames and must still report."""
    row = _row(SIGNED_DOC)
    row.pop("document_id"); row.pop("chunk_id")
    validation = _answer(monkeypatch, SIGNED_QUERY, [row], enabled=True)["validation"]

    assert validation["evidence_spans"] == 0
    assert validation["state_verdict"] == "SIGNED_CONFIRMED"


def test_gate_action_records_a_pr6_rewrite(monkeypatch):
    """With both flags on, the diagnostic must not claim PASSTHROUGH after the
    PR6 gate rewrote the answer."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    result = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True,
                     text="Na boxu není podepsaná smlouva haus365.")

    assert result["validation"]["gate_action"] == "REWRITTEN_POSITIVE"
    assert result["validation"]["state_verdict"] == "SIGNED_CONFIRMED"


def test_gate_action_stays_passthrough_when_the_gate_changes_nothing(monkeypatch):
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    result = _answer(monkeypatch, CRM_QUERY, [_row("CRM destička.pdf")], enabled=True)
    assert result["validation"]["gate_action"] == "PASSTHROUGH"


def test_pr6_gate_outcome_is_not_altered_by_the_diagnostic(monkeypatch):
    """PR7.2 does not refactor the gate (that is PR7.3): the gated answer text
    must be identical whether or not the diagnostic layer runs."""
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", True)
    text = "Na boxu není podepsaná smlouva haus365."
    off = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=False, text=text)
    on = _answer(monkeypatch, SIGNED_QUERY, [_row(SIGNED_DOC)], enabled=True, text=text)
    assert on["answer"] == off["answer"]
