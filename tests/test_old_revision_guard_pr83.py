"""PR8.3 — OLD revision safety guard unit tests."""
from __future__ import annotations

import json

import pytest

import ai_search
import ai_search_config
import old_revision_guard as org
from benchmark.acceptance_run_nds import detect_old_revision_leakage
from benchmark.dataset.schema import load_dataset
from pathlib import Path


def _row(document: str, path: str, quote: str = "FERI dodavatel monolitu NOT251110"):
    return {
        "document": document,
        "path": path,
        "quote": quote,
        "project": "240783160_Garáže_NDS",
        "heading": "",
        "score": 1.0,
    }


def _mock_ollama(monkeypatch, text: str, document_index: int = 1):
    payload = json.dumps({
        "body": [{"text": text, "zdroj_index": document_index, "typ": "fakt"}],
        "nenalezeno": False,
    })

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": payload}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())


OLD_VOP = _row(
    "Příloha č. 1_VOP.pdf",
    "/36_monolit_FERI_NOT251110/návrh/SOD/old/přílohy k SOD/Příloha č. 1_VOP.pdf",
    quote="Dodavatel prohlašuje, že je oprávněn k realizaci Smlouvy.",
)
CURRENT_FERI = _row(
    "Dohoda o ukončení prací_FERIxSIS.pdf",
    "/36_monolit_FERI_NOT251110/final/Dohoda o ukončení prací_FERIxSIS.pdf",
    quote="FERI je dodavatel monolitu NOT251110.",
)
OLD_DRAWING = _row(
    "D.1.2.07 - schéma vyztužení 3.PP.pdf",
    "/D12_Statika/OLD/D.1.2.07 - schéma vyztužení 3.PP.pdf",
    quote="Schéma vyztužení 3.PP.",
)
CURRENT_VOP = _row(
    "Příloha č. 1_VOP.pdf",
    "/36_monolit_FERI_NOT251110/final/přílohy k SOD/Příloha č. 1_VOP.pdf",
    quote="VOP aktuální verze.",
)


def test_flag_default_off():
    assert ai_search_config.OLD_REVISION_GUARD_ENABLED is False
    assert ai_search.OLD_REVISION_GUARD_ENABLED is False


def test_currency_queries_forbid_old():
    assert org.query_forbids_old_authority("kdo je dodavatel monolitu?")
    assert org.query_forbids_old_authority("existuje podepsaná smlouva na monolit?")
    assert org.query_forbids_old_authority("existuje smlouva s FERI?")
    assert org.query_forbids_old_authority("najdi aktuální harmonogram")
    assert org.query_forbids_old_authority("jaký harmonogram platí")


def test_history_queries_allow_old():
    assert org.query_allows_old_history("porovnání revizí harmonogramu")
    assert org.query_allows_old_history("historie smlouvy FERI")
    assert not org.query_forbids_old_authority("porovnání revizí R1 R2")


def test_existuje_vykres_is_not_currency_forbid():
    """nds-status-03: existence of a drawing that only lives in OLD/."""
    assert not org.query_forbids_old_authority("existuje výkres výztuže 3.PP?")


def test_currency_downgrades_old_when_non_old_exists():
    result = org.apply_old_revision_guard(
        "kdo je dodavatel monolitu?", [OLD_VOP, CURRENT_FERI],
    )
    assert result.reason == "currency_forbid_downgrade"
    assert CURRENT_FERI in result.context_results
    assert OLD_VOP in result.historical_results
    paths = [r["path"] for r in result.context_results]
    assert all("/old/" not in p.casefold() for p in paths)


def test_old_only_pool_kept_for_existence():
    result = org.apply_old_revision_guard(
        "existuje výkres výztuže 3.PP?", [OLD_DRAWING],
    )
    assert result.context_results == (OLD_DRAWING,)
    assert result.historical_results == ()


def test_history_keeps_old_in_context():
    result = org.apply_old_revision_guard(
        "porovnání revizí smlouvy FERI", [OLD_VOP, CURRENT_FERI],
    )
    assert result.reason == "history_allowed"
    assert OLD_VOP in result.context_results


def test_soft_same_entity_downgrade():
    result = org.apply_old_revision_guard(
        "jaké jsou VOP monolitu?", [OLD_VOP, CURRENT_VOP],
    )
    assert result.reason == "soft_same_entity_downgrade"
    assert CURRENT_VOP in result.context_results
    assert OLD_VOP in result.historical_results


def test_flag_off_answer_keeps_old_in_citations(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    _mock_ollama(monkeypatch, "Dodavatel je FERI.")
    rows = [OLD_VOP, CURRENT_FERI]
    a = ai_search.answer("kdo je dodavatel monolitu?", rows)
    b = ai_search.answer("kdo je dodavatel monolitu?", rows)
    assert [r["path"] for r in a["citations"]] == [r["path"] for r in b["citations"]]
    assert any("/old/" in (r.get("path") or "").casefold() for r in a["citations"])
    assert "historical_citations" not in a


def test_flag_on_answer_strips_old_from_citations(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", True)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    _mock_ollama(monkeypatch, "Dodavatel monolitu je FERI.")
    result = ai_search.answer("kdo je dodavatel monolitu?", [OLD_VOP, CURRENT_FERI])
    cite_paths = [(r.get("path") or "") for r in result["citations"]]
    assert all("/old/" not in p.casefold() for p in cite_paths)
    hist = result.get("historical_citations") or []
    assert hist and all(h.get("match", {}).get("historical_old") for h in hist)
    assert any("/old/" in (h.get("path") or "").casefold() for h in hist)


def test_nds_status_04_regression_no_old_leak(monkeypatch):
    """FAT v2 blocker: nds-status-04 must not cite /old/ as evidence."""
    from benchmark.dataset.schema import BenchmarkCase

    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", True)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    _mock_ollama(monkeypatch, "Dodavatel monolitu je FERI (NOT251110).")
    pool = [CURRENT_FERI, OLD_VOP, CURRENT_VOP]
    answer = ai_search.answer("kdo je dodavatel monolitu?", pool)
    case = BenchmarkCase(
        id="nds-status-04",
        question="kdo je dodavatel monolitu?",
        category="DOCUMENT_STATUS",
        expected_documents=["36_monolit_FERI_NOT251110"],
        expected_source_contains=["FERI"],
        expected_answer_contains=["feri"],
        expected_outcome="found",
    )
    leaks = detect_old_revision_leakage(case, pool, answer)
    assert leaks == []
    assert all("/old/" not in (r.get("path") or "").casefold() for r in answer["citations"])


def test_pr83_dataset_loads():
    path = Path(__file__).resolve().parents[1] / "benchmark/dataset/pr83_old_revision_guard.jsonl"
    cases = load_dataset(path)
    ids = {c.id for c in cases}
    assert "pr83-status-04-dodavatel" in ids
    assert "pr83-history-allowed" in ids
    assert "pr83-old-only-drawing" in ids
    by_id = {c.id: c for c in cases}
    assert org.query_forbids_old_authority(by_id["pr83-status-04-dodavatel"].question)
    assert org.query_allows_old_history(by_id["pr83-history-allowed"].question)
    assert not org.query_forbids_old_authority(by_id["pr83-old-only-drawing"].question)
