"""PR9.2.1 — free-text fallback after a JSON renderer sentinel.

When the JSON `format` path parses but citation contract drops every
substantive item (typically `zdroj_index=0`), the renderer returns the
canonical sentinel without raising. Pre-PR9.2.1 that meant fallback never
ran. `JSON_SENTINEL_FALLBACK_ENABLED` ON reuses the existing free-text
fallback (still gated by PR8.4.6). Flag OFF is byte-identical to
pre-PR9.2.1. `zdroj_index=0` stays invalid — never remapped to 1.
"""
from __future__ import annotations

import json

import ai_search
import ai_search_config


SENTINEL = "Nenalezeno v indexovaných dokumentech."
CONCISE_QUERY = "kdo je dodavatel monolitu?"
ZERO_INDEX_PAYLOAD = {
    "body": [{
        "text": "Dodavatelem monolitu je společnost FERI s.r.o.",
        "zdroj_index": 0,
        "typ": "fakt",
    }],
    "nenalezeno": False,
}
POOL_FILENAME = "Dohoda o ukončení prací_FERIxSIS.pdf"
FAKE_FILENAME = "neexistujici_zdroj.pdf"


def _row(document: str):
    return {
        "document": document,
        "path": f"/proj/{document}",
        "quote": "FERI je dodavatel monolitu NOT251110.",
        "project": "240783160_Garáže_NDS",
        "heading": "",
        "score": 1.0,
        "match": {"fts_hit": True, "vector_hit": True, "semantic_similarity": 0.7},
    }


ROWS = [_row(POOL_FILENAME), _row("b.pdf")]


def _disable_unrelated_flags(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    monkeypatch.setattr(ai_search, "ABSTENTION_OVERRIDE_ENABLED", False)
    monkeypatch.setattr(ai_search, "STRUCTURED_SUMMARY_CITATION_ENABLED", False)


def _set_pr921_flags(monkeypatch, *, sentinel_fallback: bool, citation: bool = True,
                     fallback_contract: bool = True):
    _disable_unrelated_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", citation)
    monkeypatch.setattr(ai_search, "FALLBACK_CITATION_CONTRACT_ENABLED", fallback_contract)
    monkeypatch.setattr(ai_search, "JSON_SENTINEL_FALLBACK_ENABLED", sentinel_fallback)


def _mock_sequential_ollama(monkeypatch, responses):
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


# ---------------------------------------------------------------------------
# 1. Flag default OFF — JSON sentinel does not start fallback
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert ai_search_config.JSON_SENTINEL_FALLBACK_ENABLED is False
    assert ai_search.JSON_SENTINEL_FALLBACK_ENABLED is False


def test_flag_off_all_zero_json_does_not_call_fallback(monkeypatch):
    _set_pr921_flags(monkeypatch, sentinel_fallback=False, fallback_contract=True)
    calls = _mock_sequential_ollama(monkeypatch, [json.dumps(ZERO_INDEX_PAYLOAD)])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert len(calls) == 1
    assert "format" in calls[0]
    assert result["answer"].startswith(SENTINEL)
    assert "FERI" not in result["answer"]
    assert "error" not in result


# ---------------------------------------------------------------------------
# 2. Concise all-zero JSON, flag ON → free-text fallback runs
# ---------------------------------------------------------------------------

def test_flag_on_all_zero_json_starts_fallback(monkeypatch):
    _set_pr921_flags(monkeypatch, sentinel_fallback=True, fallback_contract=True)
    calls = _mock_sequential_ollama(monkeypatch, [
        json.dumps(ZERO_INDEX_PAYLOAD),
        f"Dodavatelem je FERI.\n(Zdroj: {POOL_FILENAME})",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert len(calls) == 2
    assert "format" in calls[0]
    assert "format" not in calls[1]
    assert "FERI" in result["answer"]
    assert not result["answer"].startswith(SENTINEL)


# ---------------------------------------------------------------------------
# 3. Fallback after all-zero JSON with a real pool filename → keep
# ---------------------------------------------------------------------------

def test_fallback_with_pool_filename_kept_when_contract_on(monkeypatch):
    _set_pr921_flags(monkeypatch, sentinel_fallback=True, fallback_contract=True)
    _mock_sequential_ollama(monkeypatch, [
        json.dumps(ZERO_INDEX_PAYLOAD),
        f"Dodavatelem monolitu je FERI s.r.o. (Zdroj: {POOL_FILENAME})",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert "FERI s.r.o." in result["answer"]
    assert POOL_FILENAME.split(".")[0] in result["answer"] or POOL_FILENAME in result["answer"]
    assert not result["answer"].startswith(SENTINEL)


# ---------------------------------------------------------------------------
# 4. Fallback after all-zero JSON without a filename → sentinel
# ---------------------------------------------------------------------------

def test_fallback_without_filename_is_sentinel_when_contract_on(monkeypatch):
    _set_pr921_flags(monkeypatch, sentinel_fallback=True, fallback_contract=True)
    calls = _mock_sequential_ollama(monkeypatch, [
        json.dumps(ZERO_INDEX_PAYLOAD),
        "Dodavatelem monolitu je FERI s.r.o. [1]",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert len(calls) == 2
    assert result["answer"].startswith(SENTINEL)
    assert "FERI s.r.o." not in result["answer"]


# ---------------------------------------------------------------------------
# 5. Fallback with a fake filename → sentinel
# ---------------------------------------------------------------------------

def test_fallback_with_fake_filename_is_sentinel(monkeypatch):
    _set_pr921_flags(monkeypatch, sentinel_fallback=True, fallback_contract=True)
    _mock_sequential_ollama(monkeypatch, [
        json.dumps(ZERO_INDEX_PAYLOAD),
        f"Dodavatelem monolitu je FERI s.r.o. (Zdroj: {FAKE_FILENAME})",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert result["answer"].startswith(SENTINEL)
    assert FAKE_FILENAME not in result["answer"]
    assert "FERI s.r.o." not in result["answer"]


# ---------------------------------------------------------------------------
# 6. zdroj_index=0 is never remapped to document [1]
# ---------------------------------------------------------------------------

def test_zero_source_index_is_never_remapped_to_first_document(monkeypatch):
    _set_pr921_flags(monkeypatch, sentinel_fallback=True, fallback_contract=True)
    assert ai_search._clamp_source_index(0, len(ROWS)) is None
    calls = _mock_sequential_ollama(monkeypatch, [
        json.dumps(ZERO_INDEX_PAYLOAD),
        f"Dodavatelem je FERI.\n(Zdroj: {POOL_FILENAME})",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    # Remap 0→1 would have resolved the JSON item against ROWS[0] and skipped
    # fallback (one call, JSON renderer emits "(Zdroj: ...)").
    assert len(calls) == 2
    assert ai_search._clamp_source_index(0, len(ROWS)) is None
    json_only = ai_search._render_concise_answer(ZERO_INDEX_PAYLOAD, ROWS)
    assert json_only == SENTINEL
    assert f"(Zdroj: {POOL_FILENAME})" not in json_only
    assert "FERI" in result["answer"]


def test_explicit_abstention_does_not_start_fallback(monkeypatch):
    """nenalezeno=true and no substantive items must stay sentinel."""
    _set_pr921_flags(monkeypatch, sentinel_fallback=True, fallback_contract=True)
    calls = _mock_sequential_ollama(monkeypatch, [
        json.dumps({"body": [], "nenalezeno": True}),
        f"Toto by se nemělo objevit. (Zdroj: {POOL_FILENAME})",
    ])
    result = ai_search.answer(CONCISE_QUERY, ROWS)
    assert len(calls) == 1
    assert result["answer"].startswith(SENTINEL)
    assert "Toto by se nemělo objevit" not in result["answer"]


def test_shrnuti_only_structured_payload_does_not_start_fallback(monkeypatch):
    _set_pr921_flags(monkeypatch, sentinel_fallback=True, fallback_contract=True)
    monkeypatch.setattr(ai_search, "STRUCTURED_SUMMARY_CITATION_ENABLED", True)
    payload = {
        "shrnuti": "Minimální krytí dle TP 124.",
        "oblasti": [{"nazev": "Výztuž", "polozky": []}],
        "nenalezene": [],
    }
    calls = _mock_sequential_ollama(monkeypatch, [
        json.dumps(payload),
        f"Krytí TP 124. (Zdroj: {POOL_FILENAME})",
    ])
    result = ai_search.answer("jaké jsou požadavky na krytí výztuže?", ROWS)
    assert len(calls) == 1
    assert result["answer"].startswith(SENTINEL)
    assert "TP 124" not in result["answer"]
