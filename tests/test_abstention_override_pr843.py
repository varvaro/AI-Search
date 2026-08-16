"""PR8.4.3 — abstention override hardening.

Audit finding (PR8.4.2, nds-qa-05): the model can return a self-contradictory
JSON payload — a non-empty `body`/`polozky` with valid, resolvable
`zdroj_index` values AND `nenalezeno: true` at the same time.
`_render_concise_answer` trusted `nenalezeno` unconditionally, before ever
looking at `body`, discarding a genuinely cited, evidence-backed answer.

This file tests `ABSTENTION_OVERRIDE_ENABLED`: when ON, `nenalezeno` is only
trusted when NO item in the response survives the PR8.4.1 citation contract
(enforced unconditionally for this decision); when at least one cited item
survives, it is rendered and the conflicting flag is ignored. Flag OFF must
stay byte-identical to pre-PR8.4.3.

No retrieval/ranking/entity/revision/evidence_runtime/prompt code is touched
or exercised here beyond what PR8.4.1's citation contract already covers.
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


ROW_A = _row("PENTAFLEX_KB80_Montážní návod.pdf")
ROW_B = _row("Příloha č. 3 - Technolog. predpis Peri.doc")
RESULTS = [ROW_A, ROW_B]


def _set_flags(monkeypatch, *, abstention_override, citation_contract=False):
    monkeypatch.setattr(ai_search, "ABSTENTION_OVERRIDE_ENABLED", abstention_override)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", citation_contract)


# ---------------------------------------------------------------------------
# 1. Flag default ON (PR9.7.1). OFF remains monkeypatch-testable below.
# ---------------------------------------------------------------------------

def test_flag_default_is_on():
    assert ai_search_config.ABSTENTION_OVERRIDE_ENABLED is True
    assert ai_search.ABSTENTION_OVERRIDE_ENABLED is True


# ---------------------------------------------------------------------------
# Concise renderer path (_render_concise_answer)
# ---------------------------------------------------------------------------

def test_concise_flag_off_nenalezeno_wins_even_with_valid_body(monkeypatch):
    """Byte-identical to pre-PR8.4.3: nenalezeno=true always short-circuits,
    regardless of how good `body` is."""
    _set_flags(monkeypatch, abstention_override=False)
    data = {
        "nenalezeno": True,
        "body": [{"text": "Validní tvrzení.", "zdroj_index": 1, "typ": "fakt"}],
    }
    assert ai_search._render_concise_answer(data, RESULTS) == "Nenalezeno v indexovaných dokumentech."


def test_concise_flag_on_valid_cited_body_overrides_nenalezeno(monkeypatch):
    """2. nenalezeno=true + validní citovaný body -> body se zobrazí při ON."""
    _set_flags(monkeypatch, abstention_override=True)
    data = {
        "nenalezeno": True,
        "body": [{"text": "Validní tvrzení.", "zdroj_index": 1, "typ": "fakt"}],
    }
    rendered = ai_search._render_concise_answer(data, RESULTS)
    assert "Validní tvrzení." in rendered
    assert f"(Zdroj: {ROW_A['document']})" in rendered
    assert rendered != "Nenalezeno v indexovaných dokumentech."


def test_concise_flag_on_no_valid_body_keeps_sentinel(monkeypatch):
    """3. nenalezeno=true + žádné validní body -> sentinel."""
    _set_flags(monkeypatch, abstention_override=True)
    data = {
        "nenalezeno": True,
        "body": [{"text": "Nejisté tvrzení bez zdroje.", "zdroj_index": None, "typ": "fakt"}],
    }
    assert ai_search._render_concise_answer(data, RESULTS) == "Nenalezeno v indexovaných dokumentech."


def test_concise_flag_on_empty_body_keeps_sentinel(monkeypatch):
    _set_flags(monkeypatch, abstention_override=True)
    data = {"nenalezeno": True, "body": []}
    assert ai_search._render_concise_answer(data, RESULTS) == "Nenalezeno v indexovaných dokumentech."


def test_concise_flag_on_mixed_valid_and_invalid_items(monkeypatch):
    """4. Kombinace validní + nevalidní položky: jen validní přežije."""
    _set_flags(monkeypatch, abstention_override=True)
    data = {
        "nenalezeno": True,
        "body": [
            {"text": "Validní tvrzení A.", "zdroj_index": 1, "typ": "fakt"},
            {"text": "Nejisté tvrzení bez zdroje.", "zdroj_index": 99, "typ": "fakt"},
        ],
    }
    rendered = ai_search._render_concise_answer(data, RESULTS)
    assert "Validní tvrzení A." in rendered
    assert "Nejisté tvrzení" not in rendered


def test_concise_override_enforces_citation_even_if_contract_flag_off(monkeypatch):
    """5. Citation contract se dodržuje při override bezpodmínečně - i když
    CITATION_CONTRACT_ENABLED je vypnutý, položka bez zdroje se přes override
    nesmí dostat do vykreslené odpovědi."""
    _set_flags(monkeypatch, abstention_override=True, citation_contract=False)
    data = {
        "nenalezeno": True,
        "body": [{"text": "Nejisté tvrzení bez zdroje.", "zdroj_index": None, "typ": "fakt"}],
    }
    assert ai_search._render_concise_answer(data, RESULTS) == "Nenalezeno v indexovaných dokumentech."


def test_concise_flag_on_nonexistent_source_index_keeps_sentinel(monkeypatch):
    """nenalezeno=true + only an out-of-range zdroj_index → abstention."""
    _set_flags(monkeypatch, abstention_override=True)
    data = {
        "nenalezeno": True,
        "body": [{"text": "Unsupported claim.", "zdroj_index": 99, "typ": "fakt"}],
    }
    assert ai_search._render_concise_answer(data, RESULTS) == "Nenalezeno v indexovaných dokumentech."


def test_concise_nenalezeno_false_valid_body_unchanged(monkeypatch):
    _set_flags(monkeypatch, abstention_override=True)
    data = {
        "nenalezeno": False,
        "body": [{"text": "Validní tvrzení.", "zdroj_index": 1, "typ": "fakt"}],
    }
    rendered = ai_search._render_concise_answer(data, RESULTS)
    assert "Validní tvrzení." in rendered
    assert f"(Zdroj: {ROW_A['document']})" in rendered
    assert rendered != "Nenalezeno v indexovaných dokumentech."


RETENCE_FAIL_ROWS = [
    _row("32_RETENCE.pdf", path="/proj/32_RETENCE.pdf",
         quote="Rez retenční nádrží ... Řez nádrží na dešťovou vodu"),
    _row("03_ÚMČ_společné pravomocné.pdf"),
    _row("PLÁN BOZP NA STAVBU Narodni dum Smichov - garaze A1.doc"),
]
RETENCE_FAIL_PAYLOAD = {
    "body": [
        {
            "text": "Rez retenční nádrží ... Řez nádrží na dešťovou vodu",
            "zdroj_index": 1,
            "typ": "fakt",
        },
        {
            "text": "Retenční železobetonová nádrž o rozměrech 9,9 x 3,8 x 1,7 m a objemu 102,6 m³",
            "zdroj_index": 2,
            "typ": "fakt",
        },
        {
            "text": "Situační výkres stavby",
            "zdroj_index": 3,
            "typ": "fakt",
        },
    ],
    "nenalezeno": True,
}


def test_retence_payload_flag_off_keeps_sentinel(monkeypatch):
    """PR9.7.0 live fail: override OFF discards the cited section."""
    _set_flags(monkeypatch, abstention_override=False)
    rendered = ai_search._render_concise_answer(RETENCE_FAIL_PAYLOAD, RETENCE_FAIL_ROWS)
    assert rendered == "Nenalezeno v indexovaných dokumentech."


def test_retence_payload_flag_on_preserves_cited_section(monkeypatch):
    """PR9.7.1: same payload keeps the cited 32_RETENCE section."""
    _set_flags(monkeypatch, abstention_override=True)
    rendered = ai_search._render_concise_answer(RETENCE_FAIL_PAYLOAD, RETENCE_FAIL_ROWS)
    assert "Rez retenční nádrží" in rendered or "Řez nádrží na dešťovou vodu" in rendered
    assert "32_RETENCE.pdf" in rendered
    assert rendered != "Nenalezeno v indexovaných dokumentech."


def test_concise_flag_on_does_not_affect_non_abstention_path(monkeypatch):
    """ABSTENTION_OVERRIDE_ENABLED must have zero effect when nenalezeno is
    not set - CITATION_CONTRACT_ENABLED alone still governs that path
    (PR8.4.1 behaviour untouched)."""
    _set_flags(monkeypatch, abstention_override=True, citation_contract=False)
    data = {
        "nenalezeno": False,
        "body": [{"text": "Tvrzení bez zdroje, ale nenalezeno=False.", "zdroj_index": None, "typ": "fakt"}],
    }
    rendered = ai_search._render_concise_answer(data, RESULTS)
    assert "Tvrzení bez zdroje" in rendered  # kept - CITATION_CONTRACT_ENABLED is False here
    assert "(Zdroj:" not in rendered


# ---------------------------------------------------------------------------
# Structured renderer path (_render_structured_answer) — checklist queries
# ---------------------------------------------------------------------------

def _structured_payload(*, nenalezeno, polozky):
    return {
        "shrnuti": "Shrnutí odpovědi.",
        "oblasti": [{"nazev": "Oblast 1", "polozky": polozky}],
        "nenalezene": [],
        "nenalezeno": nenalezeno,
    }


def test_structured_flag_off_ignores_nenalezeno_key_entirely(monkeypatch):
    """STRUCTURED_ANSWER_SCHEMA has no official `nenalezeno` field; flag OFF
    must render exactly as pre-PR8.4.3 regardless of any stray key."""
    _set_flags(monkeypatch, abstention_override=False)
    data = _structured_payload(
        nenalezeno=True,
        polozky=[{"text": "Validní krok.", "zdroj_index": 1, "typ": "pozadavek"}],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert "Validní krok." in rendered
    assert rendered != "Nenalezeno v indexovaných dokumentech."


def test_structured_flag_on_valid_item_overrides_nenalezeno(monkeypatch):
    """6/2. structured cesta: nenalezeno=true + validní citovaná položka ->
    položka se zobrazí, konflikt se ignoruje."""
    _set_flags(monkeypatch, abstention_override=True)
    data = _structured_payload(
        nenalezeno=True,
        polozky=[{"text": "Validní krok.", "zdroj_index": 1, "typ": "pozadavek"}],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert "Validní krok." in rendered
    assert ROW_A["document"] in rendered
    assert "Oblast 1" in rendered


def test_structured_flag_on_no_valid_item_keeps_sentinel(monkeypatch):
    """6/3. structured cesta: nenalezeno=true + žádná validní položka ->
    sentinel, ne polovičatý dokument s prázdnými oblastmi."""
    _set_flags(monkeypatch, abstention_override=True)
    data = _structured_payload(
        nenalezeno=True,
        polozky=[{"text": "Nejistý krok.", "zdroj_index": None, "typ": "pozadavek"}],
    )
    assert ai_search._render_structured_answer(data, RESULTS) == "Nenalezeno v indexovaných dokumentech."


def test_structured_flag_on_mixed_items_keeps_only_valid(monkeypatch):
    """4/6. Kombinace validní + nevalidní položky ve structured cestě."""
    _set_flags(monkeypatch, abstention_override=True)
    data = _structured_payload(
        nenalezeno=True,
        polozky=[
            {"text": "Validní krok.", "zdroj_index": 1, "typ": "pozadavek"},
            {"text": "Nejistý krok.", "zdroj_index": None, "typ": "pozadavek"},
        ],
    )
    rendered = ai_search._render_structured_answer(data, RESULTS)
    assert "Validní krok." in rendered
    assert "Nejistý krok." not in rendered


def test_structured_override_enforces_citation_even_if_contract_flag_off(monkeypatch):
    """5/6. Citation contract platí bezpodmínečně i ve structured cestě."""
    _set_flags(monkeypatch, abstention_override=True, citation_contract=False)
    data = _structured_payload(
        nenalezeno=True,
        polozky=[{"text": "Nejistý krok.", "zdroj_index": None, "typ": "pozadavek"}],
    )
    assert ai_search._render_structured_answer(data, RESULTS) == "Nenalezeno v indexovaných dokumentech."


# ---------------------------------------------------------------------------
# 8. Identity invariant: results/citations not mutated or copied, flag OFF
# ---------------------------------------------------------------------------

def _mock_ollama(monkeypatch, payload: dict):
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": json.dumps(payload)}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())


def test_identity_preserved_when_flag_off(monkeypatch):
    """results/citations must remain the SAME object (not a copy, not
    mutated) when ABSTENTION_OVERRIDE_ENABLED is OFF - this PR touches only
    the render decision, never the `answer_results` assignment fixed in
    PR8.4.1."""
    monkeypatch.setattr(ai_search, "ABSTENTION_OVERRIDE_ENABLED", False)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", False)
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    rows = [ROW_A, ROW_B]
    before = [dict(r) for r in rows]
    _mock_ollama(monkeypatch, {
        "body": [{"text": "Validní tvrzení.", "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": True,
    })
    result = ai_search.answer("nejaky dotaz", rows)
    assert result["citations"] is rows
    assert rows == before
    # nenalezeno=True still wins with the flag off, end-to-end through answer()
    assert result["answer"].startswith("Nenalezeno v indexovaných dokumentech.")


def test_identity_preserved_when_flag_on_too(monkeypatch):
    """Turning the override ON must not change the identity contract either
    - only which text gets rendered."""
    monkeypatch.setattr(ai_search, "ABSTENTION_OVERRIDE_ENABLED", True)
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    rows = [ROW_A, ROW_B]
    before = [dict(r) for r in rows]
    _mock_ollama(monkeypatch, {
        "body": [{"text": "Validní tvrzení.", "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": True,
    })
    result = ai_search.answer("nejaky dotaz", rows)
    assert result["citations"] is rows
    assert rows == before
    assert "Validní tvrzení." in result["answer"]
