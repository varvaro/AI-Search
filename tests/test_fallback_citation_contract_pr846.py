"""PR8.4.6 — free-text fallback citation contract.

When the JSON `format` path fails, `answer()` retries with unconstrained
prose that never goes through `_render_*` / `zdroj_index`. This file tests
`FALLBACK_CITATION_CONTRACT_ENABLED`: ON keeps that prose only when it
mentions a document name from the answer pool; otherwise the canonical
sentinel replaces it. Flag OFF must stay byte-identical to pre-PR8.4.6.

JSON renderers (PR8.4.1–8.4.4) and retrieval are out of scope.
"""
from __future__ import annotations

import json

import ai_search
import ai_search_config


SENTINEL = "Nenalezeno v indexovaných dokumentech."
CONCISE_QUERY = "Pentaflex"
CHECKLIST_QUERY = "Co chybí k předání?"


def _row(document: str):
    return {
        "document": document,
        "path": f"/proj/{document}",
        "quote": f"Obsah {document}",
        "project": "240783160_Garáže_NDS",
        "heading": "",
        "score": 1.0,
        "match": {"fts_hit": True, "vector_hit": True, "semantic_similarity": 0.7},
    }


ROWS = [_row("a.pdf"), _row("b.pdf")]


def _disable_answer_side_flags(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", False)
    monkeypatch.setattr(ai_search, "ABSTENTION_OVERRIDE_ENABLED", False)
    monkeypatch.setattr(ai_search, "STRUCTURED_SUMMARY_CITATION_ENABLED", False)


def _set_fallback_flag(monkeypatch, enabled: bool):
    monkeypatch.setattr(ai_search, "FALLBACK_CITATION_CONTRACT_ENABLED", enabled)
    _disable_answer_side_flags(monkeypatch)


def _mock_sequential_ollama(monkeypatch, responses):
    """Each urlopen returns the next Ollama `response` string."""
    remaining = list(responses)
    calls = []

    class FakeResponse:
        def __init__(self, text):
            self._text = text
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": self._text}).encode()

    def fake_urlopen(request, timeout=0):
        calls.append(json.loads(request.data.decode()))
        return FakeResponse(remaining.pop(0))

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    return calls


def _mock_json_timeout_then(monkeypatch, fallback_text):
    calls = []

    class FakeResponse:
        def __init__(self, text):
            self._text = text
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": self._text}).encode()

    def fake_urlopen(request, timeout=0):
        payload = json.loads(request.data.decode())
        calls.append(payload)
        if "format" in payload:
            raise TimeoutError("timed out")
        return FakeResponse(fallback_text)

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    return calls


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def test_helper_false_for_empty_and_sentinel():
    assert ai_search._fallback_text_has_pool_source("", ROWS) is False
    assert ai_search._fallback_text_has_pool_source(SENTINEL, ROWS) is False


def test_helper_true_for_pool_name_casefold_and_diacritics():
    rows = [_row("Dohoda FERI.pdf")]
    assert ai_search._fallback_text_has_pool_source(
        "Dodavatel je FERI.\n(Zdroj: dohoda feri.pdf)", rows,
    ) is True


def test_helper_false_for_foreign_filename():
    assert ai_search._fallback_text_has_pool_source(
        "Fakt.\n(Zdroj: fake.pdf)", ROWS,
    ) is False


# ---------------------------------------------------------------------------
# 1. Flag default OFF — fallback without a pool source is unchanged
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert ai_search_config.FALLBACK_CITATION_CONTRACT_ENABLED is False
    assert ai_search.FALLBACK_CITATION_CONTRACT_ENABLED is False


def test_flag_off_keeps_fallback_text_without_pool_source(monkeypatch):
    _set_fallback_flag(monkeypatch, False)
    _mock_sequential_ollama(monkeypatch, [
        "Toto neni platny JSON.",
        "Dodavatelem monolitu je FERI s.r.o.",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert "FERI s.r.o." in result["answer"]
    assert not result["answer"].startswith(SENTINEL)
    assert "error" not in result


# ---------------------------------------------------------------------------
# 2. Invalid JSON + no document — ON → sentinel
# ---------------------------------------------------------------------------

def test_invalid_json_no_source_sentinel_when_flag_on(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    _mock_sequential_ollama(monkeypatch, [
        "Toto neni platny JSON.",
        "Dodavatelem monolitu je FERI s.r.o.",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert result["answer"].startswith(SENTINEL)
    assert "FERI s.r.o." not in result["answer"]
    assert "error" not in result


# ---------------------------------------------------------------------------
# 3. Invalid JSON + fake source not in pool — ON → sentinel
# ---------------------------------------------------------------------------

def test_invalid_json_fake_source_sentinel_when_flag_on(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    _mock_sequential_ollama(monkeypatch, [
        "Toto neni platny JSON.",
        "Dodavatelem monolitu je FERI s.r.o.\n(Zdroj: fake.pdf)",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert result["answer"].startswith(SENTINEL)
    assert "fake.pdf" not in result["answer"]


# ---------------------------------------------------------------------------
# 4. Invalid JSON + valid pool source — ON → text stays
# ---------------------------------------------------------------------------

def test_invalid_json_valid_source_kept_when_flag_on(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    _mock_sequential_ollama(monkeypatch, [
        "Toto neni platny JSON.",
        "Pentaflex je těsnicí pás.\n(Zdroj: a.pdf)",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert "Pentaflex je těsnicí pás." in result["answer"]
    assert "(Zdroj: a.pdf)" in result["answer"]
    assert not result["answer"].startswith(SENTINEL)


# ---------------------------------------------------------------------------
# 5 / 6. Timeout on JSON call
# ---------------------------------------------------------------------------

def test_json_timeout_valid_source_kept_when_flag_on(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    calls = _mock_json_timeout_then(monkeypatch, "Odpověď z fallbacku.\n(Zdroj: a.pdf)")
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert len(calls) == 2
    assert "format" in calls[0]
    assert "format" not in calls[1]
    assert "Odpověď z fallbacku." in result["answer"]
    assert "Ollama je nedostupná" not in result["answer"]
    assert "error" not in result


def test_json_timeout_no_source_sentinel_when_flag_on(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    _mock_json_timeout_then(monkeypatch, "Odpověď z fallbacku bez citace.")
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert result["answer"].startswith(SENTINEL)
    assert "Odpověď z fallbacku bez citace." not in result["answer"]
    assert "error" not in result


# ---------------------------------------------------------------------------
# 7. Both Ollama calls fail — ON i OFF
# ---------------------------------------------------------------------------

def test_both_calls_fail_keeps_ollama_unavailable_off(monkeypatch):
    _set_fallback_flag(monkeypatch, False)
    monkeypatch.setattr(
        ai_search.urllib.request, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert "Ollama je nedostupná" in result["answer"]
    assert result["citations"] is ROWS
    assert result["error"]


def test_both_calls_fail_keeps_ollama_unavailable_on(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    monkeypatch.setattr(
        ai_search.urllib.request, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert "Ollama je nedostupná" in result["answer"]
    assert result["citations"] is ROWS
    assert result["error"]


# ---------------------------------------------------------------------------
# 8 / 9. Concise vs checklist query
# ---------------------------------------------------------------------------

def test_concise_query_uses_fallback_contract(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    _mock_sequential_ollama(monkeypatch, [
        "not-json",
        "Fakt bez zdroje.",
    ])
    assert ai_search._is_checklist_query(CONCISE_QUERY) is False
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert result["answer"].startswith(SENTINEL)


def test_checklist_query_uses_fallback_contract(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    _mock_sequential_ollama(monkeypatch, [
        "not-json",
        "Shrnutí:\n- Testovací odpověď bez zdroje.",
    ])
    assert ai_search._is_checklist_query(CHECKLIST_QUERY) is True
    result = ai_search.answer(CHECKLIST_QUERY, ROWS)
    assert result["answer"].startswith(SENTINEL)


def test_checklist_query_keeps_fallback_with_pool_source(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    _mock_sequential_ollama(monkeypatch, [
        "not-json",
        "Shrnutí:\n- Testovací odpověď.\n\nZdroje:\n- a.pdf",
    ])
    result = ai_search.answer(CHECKLIST_QUERY, ROWS)
    assert "Testovací odpověď" in result["answer"]
    assert not result["answer"].startswith(SENTINEL)


# ---------------------------------------------------------------------------
# JSON success is unaffected
# ---------------------------------------------------------------------------

def test_valid_json_path_unaffected_when_flag_on(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    payload = json.dumps({
        "body": [{"text": "FERI je dodavatel.", "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })
    calls = _mock_sequential_ollama(monkeypatch, [payload])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert len(calls) == 1
    assert "FERI je dodavatel." in result["answer"]
    assert "(Zdroj: a.pdf)" in result["answer"]


# ---------------------------------------------------------------------------
# 10. Identity invariant
# ---------------------------------------------------------------------------

def test_citations_identity_when_old_guard_inactive(monkeypatch):
    _set_fallback_flag(monkeypatch, True)
    rows = [_row("a.pdf"), _row("b.pdf")]
    before = [dict(r) for r in rows]
    _mock_sequential_ollama(monkeypatch, [
        "not-json",
        "Fakt bez zdroje.",
    ])
    result = ai_search.answer(CONCISE_QUERY, rows)
    assert result["citations"] is rows
    assert rows == before
    assert result["answer"].startswith(SENTINEL)
