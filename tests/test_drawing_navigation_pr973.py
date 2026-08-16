"""PR9.7.3 — deterministic drawing-navigation answers."""
from __future__ import annotations

import json
from pathlib import Path

import ai_search
import drawing_navigation as dn
from drawing_navigation import DrawingSubtype


FORBIDDEN_PRODUCTION_VALUES = (
    "RETENCE",
    "32_RETENCE",
    "retenční nádrž",
    "retencni nadrz",
    "FERI",
    "Illichman",
    "NDS",
    "nds-draw",
    "PENTAFLEX",
    "NOT250039",
)

MODULE_PATH = Path(dn.__file__)
SENTINEL = "Nenalezeno v indexovaných dokumentech."


def _row(document, quote="", heading="", path="", score=1.0, project="P"):
    return {
        "document": document,
        "path": path or f"/proj/{document}",
        "relative_path": path or f"proj/{document}",
        "quote": quote,
        "heading": heading,
        "project": project,
        "score": score,
    }


def _disable_other_flags(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", False)
    monkeypatch.setattr(ai_search, "JSON_SENTINEL_FALLBACK_ENABLED", False)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", False)
    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", False)


def _mock_json_ollama(monkeypatch, payload: dict, sink: list | None = None):
    def fake_call(model, prompt, format_schema=None, timeout=240):
        if sink is not None:
            sink.append({"model": model, "prompt": prompt})
        return json.dumps(payload)

    monkeypatch.setattr(ai_search, "_call_ollama", fake_call)


# ---------------------------------------------------------------------------
#  intent
# ---------------------------------------------------------------------------

def test_generic_drawing_query():
    q = "najdi mi výkres retenční nádrže"
    assert dn.is_drawing_navigation_query(q) is True
    assert dn.derive_requested_subtypes(q) == (DrawingSubtype.GENERIC_DRAWING,)


def test_plan_and_generic_query():
    q = "najdi mi výkres půdorys retenční nádrže"
    assert dn.is_drawing_navigation_query(q) is True
    assert dn.derive_requested_subtypes(q) == (
        DrawingSubtype.PLAN, DrawingSubtype.GENERIC_DRAWING,
    )


def test_plan_and_section_query():
    q = "najdi mi půdorys a řez retenční nádrže"
    assert dn.derive_requested_subtypes(q) == (
        DrawingSubtype.PLAN, DrawingSubtype.SECTION,
    )


def test_detail_query():
    q = "kde najdu detail prostupu"
    assert dn.is_drawing_navigation_query(q) is True
    assert dn.derive_requested_subtypes(q) == (DrawingSubtype.DETAIL,)


def test_factual_queries_are_not_navigation():
    for q in (
        "jaký je objem retenční nádrže",
        "kolik stojí retenční nádrž",
        "co říká rozhodnutí o retenční nádrži",
        "je podepsaná smlouva na BOZP?",
        "jaký je stav prací?",
        "jaké je krytí výztuže?",
        "existuje výkres výztuže 3.PP?",
    ):
        assert dn.is_drawing_navigation_query(q) is False, q
        assert dn.try_render(q, [_row("x.pdf", quote="y")]) is None


def test_diacritics_fold():
    assert dn.derive_requested_subtypes("najdi půdorys") == (DrawingSubtype.PLAN,)
    assert dn.derive_requested_subtypes("najdi pudorys") == (DrawingSubtype.PLAN,)
    assert dn.derive_requested_subtypes("najdi řez") == (DrawingSubtype.SECTION,)
    assert dn.derive_requested_subtypes("najdi rez") == (DrawingSubtype.SECTION,)
    assert dn.derive_requested_subtypes("najdi schéma") == (DrawingSubtype.SCHEME,)
    assert dn.derive_requested_subtypes("najdi schema") == (DrawingSubtype.SCHEME,)


# ---------------------------------------------------------------------------
#  source classification
# ---------------------------------------------------------------------------

def test_admin_pdf_with_subject_is_not_drawing():
    row = _row("rozhodnuti.pdf", quote="Povolení retenční nádrže o objemu 102 m3.")
    assert dn.classify_result_subtypes(row) == frozenset()


def test_section_from_quote():
    row = _row("tank_note.pdf", quote="Řez nádrží na dešťovou vodu")
    assert DrawingSubtype.SECTION in dn.classify_result_subtypes(row)


def test_floor_plan_from_filename():
    row = _row("D.1.4-VZT-101-Půdorys 3.PP.pdf", quote="Nádrž dešťové")
    assert DrawingSubtype.PLAN in dn.classify_result_subtypes(row)
    assert dn.is_floor_plan(row) is True


# ---------------------------------------------------------------------------
#  matching / render
# ---------------------------------------------------------------------------

def test_generic_drawing_finds_section_result():
    rows = [
        _row("rozhodnuti.pdf", quote="retenční nádrž na pozemku"),
        _row("tank_note.pdf", heading="Řez retenční nádrží", quote="Řez nádrží na dešťovou vodu"),
    ]
    out = dn.render_drawing_navigation("najdi mi výkres retenční nádrže", rows)
    assert out and not out.abstained
    assert out.matches[0].document == "tank_note.pdf"
    assert out.matches[0].subtype is DrawingSubtype.SECTION
    assert "tank_note.pdf" in out.text
    assert "Řez" in out.text
    assert "rozhodnuti.pdf" not in out.text


def test_plan_and_section_both_found():
    rows = [
        _row("tank_note.pdf", quote="Řez retenční nádrží"),
        _row("Půdorys retenční nádrže.pdf", heading="Půdorys retenční nádrže", quote="Půdorys nádrže"),
    ]
    out = dn.render_drawing_navigation("najdi mi půdorys a řez retenční nádrže", rows)
    assert {m.subtype for m in out.matches} == {DrawingSubtype.PLAN, DrawingSubtype.SECTION}
    assert out.missing == ()
    assert "tank_note.pdf" in out.text
    assert "Půdorys retenční nádrže.pdf" in out.text
    assert SENTINEL not in out.text


def test_partial_section_without_plan():
    rows = [_row("tank_note.pdf", quote="Řez retenční nádrží")]
    out = dn.render_drawing_navigation("najdi mi půdorys a řez retenční nádrže", rows)
    assert not out.abstained
    assert any(m.subtype is DrawingSubtype.SECTION for m in out.matches)
    assert DrawingSubtype.PLAN in out.missing
    assert "tank_note.pdf" in out.text
    assert "nepodařilo doložit" in out.text
    assert SENTINEL not in out.text


def test_unrelated_floor_plan_is_not_a_match():
    rows = [_row("VZT-101-Půdorys 3.PP.pdf", quote="Rozvody vzduchotechniky na 3.PP")]
    out = dn.render_drawing_navigation("najdi mi půdorys retenční nádrže", rows)
    assert out.abstained
    assert out.text == SENTINEL
    assert "VZT-101" not in out.text


def test_floor_plan_with_subject_evidence():
    rows = [_row("VZT-101-Půdorys 3.PP.pdf", quote="Nádrž dešťové vody u rampy")]
    out = dn.render_drawing_navigation("najdi mi půdorys retenční nádrže", rows)
    assert not out.abstained
    match = out.matches[0]
    assert match.floor_plan is True
    assert match.dedicated is False
    assert "VZT-101-Půdorys 3.PP.pdf" in out.text
    assert "podlažní půdorys" in out.text
    assert "samostatný detailní půdorys retenční" not in out.text


def test_dedicated_versus_floor_plan():
    dedicated = _row("Půdorys retenční nádrže.pdf", heading="Půdorys retenční nádrže", quote="Půdorys nádrže")
    floor = _row("VZT-101-Půdorys 3.PP.pdf", quote="Nádrž dešťové")
    assert dn.is_dedicated_plan("najdi půdorys retenční nádrže", dedicated) is True
    assert dn.is_dedicated_plan("najdi půdorys retenční nádrže", floor) is False
    out = dn.render_drawing_navigation("najdi mi půdorys retenční nádrže", [dedicated, floor])
    assert out.matches[0].document == "Půdorys retenční nádrže.pdf"
    assert out.matches[0].dedicated is True
    assert "podlažní půdorys" not in out.text


def test_no_drawing_evidence_abstains():
    rows = [_row("rozhodnuti.pdf", quote="Souřadnice X=1, rozpočet 100, elektro, montáž.")]
    out = dn.render_drawing_navigation("najdi mi výkres retenční nádrže", rows)
    assert out.abstained
    assert out.text == SENTINEL
    assert "rozhodnuti.pdf" not in out.text
    assert "rozpočet" not in out.text


def test_section_is_never_labeled_plan():
    rows = [_row("tank_note.pdf", quote="Řez retenční nádrží")]
    out = dn.render_drawing_navigation("najdi mi výkres půdorys retenční nádrže", rows)
    assert all(m.subtype is not DrawingSubtype.PLAN for m in out.matches)
    assert "Řez:" in out.text
    text_before_missing = out.text.split("Půdorys:")[0]
    assert "tank_note.pdf" in text_before_missing


# ---------------------------------------------------------------------------
#  answer() integration
# ---------------------------------------------------------------------------

def test_drawing_query_skips_ollama(monkeypatch):
    _disable_other_flags(monkeypatch)
    sink = []
    _mock_json_ollama(monkeypatch, {"body": [{"text": "LLM dump", "zdroj_index": 1, "typ": "fakt"}], "nenalezeno": False}, sink)
    rows = [_row("tank_note.pdf", quote="Řez retenční nádrží")]
    result = ai_search.answer("najdi mi výkres retenční nádrže?", rows)
    assert sink == []
    assert result["model"] == "drawing-navigation"
    assert "tank_note.pdf" in result["answer"]
    assert "LLM dump" not in result["answer"]
    assert result["citations"] is rows


def test_non_drawing_query_keeps_ollama_path(monkeypatch):
    _disable_other_flags(monkeypatch)
    sink = []
    _mock_json_ollama(monkeypatch, {"body": [{"text": "Smlouva je podepsaná.", "zdroj_index": 1, "typ": "fakt"}], "nenalezeno": False}, sink)
    rows = [_row("smlouva.pdf", quote="Smlouva je podepsaná.")]
    result = ai_search.answer("je podepsaná smlouva na BOZP?", rows)
    assert len(sink) == 1
    assert result["model"] != "drawing-navigation"
    assert "Smlouva je podepsaná." in result["answer"]
    assert "jednoduchý vyhledávací dotaz" in sink[0]["prompt"]


def test_module_has_no_hardcoded_production_values():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for value in FORBIDDEN_PRODUCTION_VALUES:
        assert value not in source, value
    assert "ui_services" not in source
