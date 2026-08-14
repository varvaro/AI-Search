"""PR8.4.1 — citation contract hardening + PR8.3 identity regression.

Root cause (see PR8.4 audit): `_render_answer_item`'s two callers
(`_render_concise_answer`, `_render_structured_answer`) kept a model-produced
claim item even when its `zdroj_index` never resolved to a real result row -
the citation silently disappeared but the claim text survived. This file
tests the CITATION_CONTRACT_ENABLED gate added to close that gap, plus the
`answer_results is results` identity fix (PR8.3 introduced an unconditional
`list(results)` copy that broke object identity even when nothing about
`results` actually changed).

No retrieval/ranking/embeddings/PR8.1/8.2/old_revision_guard behaviour is
touched by these tests - only ai_search.answer()'s rendering step and the
`answer_results` assignment are exercised.
"""
from __future__ import annotations

import json

import ai_search
import ai_search_config


def _row(document: str, path: str, quote: str = "FERI dodavatel monolitu NOT251110"):
    return {
        "document": document,
        "path": path,
        "quote": quote,
        "project": "240783160_Garáže_NDS",
        "heading": "",
        "score": 1.0,
    }


def _mock_structured_ollama(monkeypatch, payload: dict):
    """Mocks the constrained-decoding JSON path (`format=schema`) with an
    arbitrary payload - lets tests hand the model a `zdroj_index` that is
    missing, null, or out of range, exactly like a real LLM sample can."""

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": json.dumps(payload)}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())


CURRENT_FERI = _row(
    "Dohoda o ukončení prací_FERIxSIS.pdf",
    "/36_monolit_FERI_NOT251110/final/Dohoda o ukončení prací_FERIxSIS.pdf",
    quote="FERI je dodavatel monolitu NOT251110.",
)


def _disable_other_flags(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)


# ---------------------------------------------------------------------------
# Flag default / wiring
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert ai_search_config.CITATION_CONTRACT_ENABLED is False
    assert ai_search.CITATION_CONTRACT_ENABLED is False


# ---------------------------------------------------------------------------
# _render_concise_answer (nds-status-04 / nds-qa-09 path)
# ---------------------------------------------------------------------------

def test_flag_off_keeps_pre_pr841_behaviour_uncited_claim_survives(monkeypatch):
    """Reproduces the bug exactly: flag OFF -> claim with an unresolvable
    zdroj_index is still rendered, with no "(Zdroj: ...)" note. This is the
    documented pre-existing behaviour and must not change unless the flag is
    on - regular queries stay byte-identical."""
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", False)
    _disable_other_flags(monkeypatch)
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je společnost FERI s.r.o.",
                   "zdroj_index": 99, "typ": "fakt"}],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", [CURRENT_FERI])
    assert "FERI" in result["answer"]
    assert "(Zdroj:" not in result["answer"]


def test_flag_on_drops_claim_with_unresolvable_zdroj_index(monkeypatch):
    """Flag ON -> the same uncited claim must not survive rendering at all."""
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    _disable_other_flags(monkeypatch)
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je společnost FERI s.r.o.",
                   "zdroj_index": 99, "typ": "fakt"}],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", [CURRENT_FERI])
    assert "FERI" not in result["answer"]
    assert result["answer"].startswith("Nenalezeno v indexovaných dokumentech.")


def test_flag_on_drops_claim_with_missing_zdroj_index(monkeypatch):
    """Same as above, but the model omits zdroj_index entirely rather than
    sending an out-of-range one - both must be treated as unresolvable."""
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    _disable_other_flags(monkeypatch)
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je společnost FERI s.r.o.", "typ": "fakt"}],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", [CURRENT_FERI])
    assert "FERI" not in result["answer"]
    assert "Nenalezeno" in result["answer"]


def test_flag_on_keeps_correctly_cited_claim_unchanged(monkeypatch):
    """UX must not change for a well-formed response: valid zdroj_index ->
    claim AND "(Zdroj: ...)" note both survive, exactly as before."""
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    _disable_other_flags(monkeypatch)
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je společnost FERI s.r.o.",
                   "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", [CURRENT_FERI])
    assert "FERI" in result["answer"]
    assert f"(Zdroj: {CURRENT_FERI['document']})" in result["answer"]


def test_flag_on_keeps_cited_items_and_drops_only_uncited_ones(monkeypatch):
    """Mixed response: one item cites a real row, one does not. Only the
    uncited one must be dropped - the contract is per-item, not all-or-nothing."""
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    _disable_other_flags(monkeypatch)
    other_row = _row("Reakce na dopis SIS.pdf", "/36_monolit_FERI_NOT251110/final/Reakce.pdf",
                      quote="SIS reagoval na dopis.")
    _mock_structured_ollama(monkeypatch, {
        "body": [
            {"text": "Dodavatelem monolitu je společnost FERI s.r.o.", "zdroj_index": 1, "typ": "fakt"},
            {"text": "Nejisté tvrzení bez zdroje.", "zdroj_index": None, "typ": "fakt"},
        ],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", [CURRENT_FERI, other_row])
    assert "FERI" in result["answer"]
    assert "Nejisté tvrzení" not in result["answer"]


# ---------------------------------------------------------------------------
# _render_structured_answer (checklist path, e.g. nds-qa-03)
# ---------------------------------------------------------------------------

CHECKLIST_QUERY = "jaké jsou požadavky na krytí výztuže?"


def test_structured_path_flag_on_drops_uncited_items_and_area(monkeypatch):
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    _disable_other_flags(monkeypatch)
    row = _row("11_06_2026_TP beton monolit konstrukce vc vyztuze.pdf",
               "/TP/11_06_2026_TP beton monolit konstrukce vc vyztuze.pdf",
               quote="TP 124 stanovuje krytí výztuže.")
    _mock_structured_ollama(monkeypatch, {
        "shrnuti": "Požadavky na krytí výztuže.",
        "oblasti": [{
            "nazev": "Krytí výztuže",
            "polozky": [
                {"text": "Minimální krytí je 30 mm dle TP 124.", "zdroj_index": None, "typ": "pozadavek"},
            ],
        }],
        "nenalezene": [],
    })
    result = ai_search.answer(CHECKLIST_QUERY, [row])
    assert "TP 124" not in result["answer"]
    assert "Zdroje:" in result["answer"]
    assert "Nenalezeno v indexovaných dokumentech." in result["answer"]


def test_structured_path_flag_on_keeps_cited_items(monkeypatch):
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    _disable_other_flags(monkeypatch)
    row = _row("11_06_2026_TP beton monolit konstrukce vc vyztuze.pdf",
               "/TP/11_06_2026_TP beton monolit konstrukce vc vyztuze.pdf",
               quote="TP 124 stanovuje krytí výztuže.")
    _mock_structured_ollama(monkeypatch, {
        "shrnuti": "Požadavky na krytí výztuže.",
        "oblasti": [{
            "nazev": "Krytí výztuže",
            "polozky": [
                {"text": "Minimální krytí je 30 mm dle TP 124.", "zdroj_index": 1, "typ": "pozadavek"},
            ],
        }],
        "nenalezene": [],
    })
    result = ai_search.answer(CHECKLIST_QUERY, [row])
    assert "TP 124" in result["answer"]
    assert row["document"] in result["answer"]


# ---------------------------------------------------------------------------
# Regression: nds-status-04
# ---------------------------------------------------------------------------

def test_nds_status_04_regression_unsupported_claim_fixed(monkeypatch):
    """FAT v2 / PR8.3-subset failure: retrieval found the FERI document, the
    model stated the fact, but the rendered answer had zero "(Zdroj: ...)"
    attribution (`unsupported_claim=True`, per benchmark/acceptance_metrics).
    Flag ON must make the case either cite its evidence or abstain - never
    both retrieval-hit AND an uncited claim."""
    from benchmark import acceptance_metrics

    _disable_other_flags(monkeypatch)
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je společnost FERI s.r.o.",
                   "zdroj_index": None, "typ": "fakt"}],
        "nenalezeno": False,
    })

    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", False)
    before = ai_search.answer("kdo je dodavatel monolitu?", [CURRENT_FERI])
    _, _, unsupported_before = acceptance_metrics.evaluate_citations(
        _StubCase(expected_source_contains=[], forbidden_document=None), before,
    )
    assert unsupported_before is True  # bug reproduced pre-fix

    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    after = ai_search.answer("kdo je dodavatel monolitu?", [CURRENT_FERI])
    _, _, unsupported_after = acceptance_metrics.evaluate_citations(
        _StubCase(expected_source_contains=[], forbidden_document=None), after,
    )
    assert unsupported_after is False  # no claim -> nothing left unsupported
    assert "FERI" not in after["answer"]


class _StubCase:
    """Minimal stand-in for BenchmarkCase - evaluate_citations only reads
    .expected_source_contains and .forbidden_document."""
    def __init__(self, expected_source_contains, forbidden_document):
        self.expected_source_contains = expected_source_contains
        self.forbidden_document = forbidden_document


# ---------------------------------------------------------------------------
# PR8.3 identity regression fix: answer_results is results unless transformed
# ---------------------------------------------------------------------------

def test_citations_are_results_object_when_old_guard_disabled(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    rows = [CURRENT_FERI]
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je FERI.", "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", rows)
    assert result["citations"] is rows


def test_citations_are_results_object_when_old_guard_enabled_but_noop(monkeypatch):
    """Guard runs (flag ON) but has nothing to demote - still no transform,
    so identity should still be preserved (an incidental improvement over
    pre-PR8.4.1, not required by PR8.3's own contract, but consistent with
    'copy only on a real transformation')."""
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", True)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    rows = [CURRENT_FERI]  # no OLD/ row present -> guard has nothing to demote
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je FERI.", "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", rows)
    assert result["citations"] is rows


def test_citations_are_a_copy_when_old_guard_actually_demotes(monkeypatch):
    """A real transformation still must produce a new list, not mutate the
    caller's `results` in place."""
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", True)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    old_vop = _row(
        "Příloha č. 1_VOP.pdf",
        "/36_monolit_FERI_NOT251110/návrh/SOD/old/přílohy k SOD/Příloha č. 1_VOP.pdf",
        quote="Dodavatel prohlašuje, že je oprávněn k realizaci Smlouvy.",
    )
    rows = [old_vop, CURRENT_FERI]
    before = list(rows)
    _mock_structured_ollama(monkeypatch, {
        "body": [{"text": "Dodavatelem monolitu je FERI.", "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })
    result = ai_search.answer("kdo je dodavatel monolitu?", rows)
    assert result["citations"] is not rows
    assert rows == before  # caller's list itself is never mutated
