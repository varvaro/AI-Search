"""PR8.4.4 — structured summary citation contract.

Audit finding: `_render_structured_answer` copies `shrnuti` into the answer
as free text with no `zdroj_index`. After PR8.4.1 drops every uncited
`polozky` item, the summary can still stand as an unsupported factual claim
(e.g. "Dodavatelem monolitu je FERI s.r.o." with Zdroje: Nenalezeno).

`STRUCTURED_SUMMARY_CITATION_ENABLED` closes that gap without giving `shrnuti`
its own source field: a factual structured answer is shown iff at least one
`polozky` item survived the existing PR8.4.1/8.4.3 filters; otherwise the
renderer ignores `shrnuti` and returns the canonical sentinel. Flag OFF must
stay byte-identical to pre-PR8.4.4. The concise renderer is out of scope.
"""
from __future__ import annotations

import json

import ai_search
import ai_search_config


def _row(document: str, path: str = "/proj/doc.pdf", quote: str = "quote"):
    return {
        "document": document,
        "path": path,
        "quote": quote,
        "project": "240783160_Garáže_NDS",
        "heading": "",
        "score": 1.0,
    }


ROW_A = _row("D.1.4.j.1_01_TZ.pdf")
ROW_B = _row("11_06_2026_TP beton monolit konstrukce vc vyztuze.pdf")
RESULTS = [ROW_A, ROW_B]
SENTINEL = "Nenalezeno v indexovaných dokumentech."
CHECKLIST_QUERY = "jaké jsou požadavky na krytí výztuže?"


def _set_flags(
    monkeypatch,
    *,
    summary_citation=False,
    citation_contract=False,
    abstention_override=False,
):
    monkeypatch.setattr(ai_search, "STRUCTURED_SUMMARY_CITATION_ENABLED", summary_citation)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", citation_contract)
    monkeypatch.setattr(ai_search, "ABSTENTION_OVERRIDE_ENABLED", abstention_override)


def _structured(*, shrnuti, polozky, nenalezene=None, nenalezeno=None):
    data = {
        "shrnuti": shrnuti,
        "oblasti": [{"nazev": "Oblast 1", "polozky": polozky}],
        "nenalezene": list(nenalezene or []),
    }
    if nenalezeno is not None:
        data["nenalezeno"] = nenalezeno
    return data


# ---------------------------------------------------------------------------
# 1. Flag default OFF
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert ai_search_config.STRUCTURED_SUMMARY_CITATION_ENABLED is False
    assert ai_search.STRUCTURED_SUMMARY_CITATION_ENABLED is False


def test_flag_off_keeps_shrnuti_even_when_all_polozky_are_uncited(monkeypatch):
    """Byte-identical to pre-PR8.4.4: the audit bug still reproduces with
    the flag off, including when PR8.4.1 would drop every item."""
    _set_flags(monkeypatch, summary_citation=False, citation_contract=True)
    data = _structured(
        shrnuti="Dodavatelem monolitu je FERI s.r.o.",
        polozky=[{"text": "FERI je dodavatel.", "zdroj_index": None, "typ": "fakt"}],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert rendered != SENTINEL
    assert "FERI s.r.o." in rendered
    assert "Zdroje:\n- Nenalezeno v indexovaných dokumentech." in rendered


# ---------------------------------------------------------------------------
# 2. shrnuti + žádné validní položky → sentinel při ON
# ---------------------------------------------------------------------------

def test_flag_on_shrnuti_only_returns_sentinel(monkeypatch):
    _set_flags(monkeypatch, summary_citation=True, citation_contract=True)
    data = _structured(
        shrnuti="Dodavatelem monolitu je FERI s.r.o.",
        polozky=[{"text": "FERI je dodavatel.", "zdroj_index": None, "typ": "fakt"}],
    )
    assert ai_search._render_structured_answer(data, RESULTS) == SENTINEL


def test_flag_on_empty_polozky_returns_sentinel(monkeypatch):
    _set_flags(monkeypatch, summary_citation=True)
    data = _structured(shrnuti="Dodavatelem monolitu je FERI s.r.o.", polozky=[])
    assert ai_search._render_structured_answer(data, RESULTS) == SENTINEL


# ---------------------------------------------------------------------------
# 3 / 4. shrnuti + validní citované položky → doprovodný text, ne standalone fakt
# ---------------------------------------------------------------------------

def test_flag_on_cited_items_keep_shrnuti_as_accompanying_text(monkeypatch):
    _set_flags(monkeypatch, summary_citation=True, citation_contract=True)
    data = _structured(
        shrnuti="Krytí výztuže se řídí TP 124.",
        polozky=[{"text": "Minimální krytí dle TP 124.", "zdroj_index": 1, "typ": "pozadavek"}],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert rendered != SENTINEL
    assert "Krytí výztuže se řídí TP 124." in rendered
    assert "Minimální krytí dle TP 124." in rendered
    assert f"(Zdroj: {ROW_A['document']})" in rendered
    assert f"- {ROW_A['document']}" in rendered


def test_flag_on_shrnuti_cannot_stand_as_the_only_claim(monkeypatch):
    """4. A factual `shrnuti` with no surviving evidence item must not appear
    in the rendered answer at all — not even as a Shrnutí section above
    empty areas / Nenalezené informace."""
    _set_flags(monkeypatch, summary_citation=True, citation_contract=True)
    data = _structured(
        shrnuti="Dodavatelem monolitu je FERI s.r.o.",
        polozky=[{"text": "Bez zdroje.", "zdroj_index": 99, "typ": "fakt"}],
        nenalezene=["Nic dalšího."],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert rendered == SENTINEL
    assert "FERI" not in rendered
    assert "Nic dalšího." not in rendered
    assert "Shrnutí:" not in rendered


# ---------------------------------------------------------------------------
# 5. Zachování citation contract PR8.4.1
# ---------------------------------------------------------------------------

def test_flag_on_still_drops_uncited_polozky_when_citation_contract_on(monkeypatch):
    _set_flags(monkeypatch, summary_citation=True, citation_contract=True)
    data = _structured(
        shrnuti="Shrnutí s evidencí.",
        polozky=[
            {"text": "Validní krok.", "zdroj_index": 1, "typ": "pozadavek"},
            {"text": "Nejistý krok.", "zdroj_index": None, "typ": "pozadavek"},
        ],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert "Validní krok." in rendered
    assert "Nejistý krok." not in rendered


def test_flag_on_does_not_implicitly_enable_citation_contract_on_surviving_items(monkeypatch):
    """PR8.4.1 stays independent: with citation contract OFF, an uncited
    polozka still renders. 8.4.4 only refuses a *shrnuti-only* answer."""
    _set_flags(monkeypatch, summary_citation=True, citation_contract=False)
    data = _structured(
        shrnuti="Doprovodné shrnutí.",
        polozky=[{"text": "Položka bez zdroje.", "zdroj_index": None, "typ": "fakt"}],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert rendered != SENTINEL
    assert "Položka bez zdroje." in rendered
    assert "(Zdroj:" not in rendered


# ---------------------------------------------------------------------------
# 6. Zachování abstention override PR8.4.3
# ---------------------------------------------------------------------------

def test_flag_on_does_not_break_abstention_override_with_cited_item(monkeypatch):
    _set_flags(monkeypatch, summary_citation=True, abstention_override=True)
    data = _structured(
        shrnuti="Shrnutí.",
        polozky=[{"text": "Validní krok.", "zdroj_index": 1, "typ": "pozadavek"}],
        nenalezeno=True,
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert "Validní krok." in rendered
    assert rendered != SENTINEL


def test_flag_on_and_override_still_sentinel_when_no_cited_item(monkeypatch):
    _set_flags(monkeypatch, summary_citation=True, abstention_override=True)
    data = _structured(
        shrnuti="FERI je dodavatel.",
        polozky=[{"text": "Bez zdroje.", "zdroj_index": None, "typ": "fakt"}],
        nenalezeno=True,
    )
    assert ai_search._render_structured_answer(data, RESULTS) == SENTINEL


# ---------------------------------------------------------------------------
# 7. Concise renderer beze změny
# ---------------------------------------------------------------------------

def test_concise_renderer_ignores_structured_summary_flag(monkeypatch):
    data = {
        "nenalezeno": False,
        "body": [{"text": "FERI dodavatel.", "zdroj_index": 1, "typ": "fakt"}],
    }
    _set_flags(monkeypatch, summary_citation=False)
    off = ai_search._render_concise_answer(data, RESULTS)
    _set_flags(monkeypatch, summary_citation=True)
    on = ai_search._render_concise_answer(data, RESULTS)
    assert off == on
    assert "FERI dodavatel." in on
    assert f"(Zdroj: {ROW_A['document']})" in on


def test_concise_nenalezeno_path_unchanged_by_structured_summary_flag(monkeypatch):
    data = {
        "nenalezeno": True,
        "body": [{"text": "Validní tvrzení.", "zdroj_index": 1, "typ": "fakt"}],
    }
    _set_flags(monkeypatch, summary_citation=True, abstention_override=False)
    assert ai_search._render_concise_answer(data, RESULTS) == SENTINEL


# ---------------------------------------------------------------------------
# 8. Identity invariant results/citations
# ---------------------------------------------------------------------------

def _mock_ollama(monkeypatch, payload: dict):
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": json.dumps(payload)}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())


def _disable_answer_side_flags(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)


def test_identity_preserved_when_flag_off(monkeypatch):
    _set_flags(monkeypatch, summary_citation=False)
    _disable_answer_side_flags(monkeypatch)
    rows = [ROW_A, ROW_B]
    before = [dict(r) for r in rows]
    _mock_ollama(monkeypatch, _structured(
        shrnuti="Dodavatelem monolitu je FERI s.r.o.",
        polozky=[{"text": "FERI je dodavatel.", "zdroj_index": None, "typ": "fakt"}],
    ))
    result = ai_search.answer(CHECKLIST_QUERY, rows)
    assert result["citations"] is rows
    assert rows == before
    assert "FERI s.r.o." in result["answer"]


def test_identity_preserved_when_flag_on(monkeypatch):
    _set_flags(monkeypatch, summary_citation=True, citation_contract=True)
    _disable_answer_side_flags(monkeypatch)
    rows = [ROW_A, ROW_B]
    before = [dict(r) for r in rows]
    _mock_ollama(monkeypatch, _structured(
        shrnuti="Dodavatelem monolitu je FERI s.r.o.",
        polozky=[{"text": "FERI je dodavatel.", "zdroj_index": None, "typ": "fakt"}],
    ))
    result = ai_search.answer(CHECKLIST_QUERY, rows)
    assert result["citations"] is rows
    assert rows == before
    assert result["answer"].startswith(SENTINEL)
    assert "FERI s.r.o." not in result["answer"]
