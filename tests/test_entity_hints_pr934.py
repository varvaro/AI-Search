"""PR9.3.4 — entity / identifier hint candidates.

The layer offers candidates to verify, never an answer. It must not read a
field the model cannot see, must never emit a WHO candidate from body text,
must never produce source_index 0, and must not hardcode any production
document, firm, or standard value.
"""
from __future__ import annotations

import json
from pathlib import Path

import ai_search
import ai_search_config
import answer_intent
import context_packing
import entity_hints
from entity_hints import HintKind


FORBIDDEN_PRODUCTION_VALUES = (
    "FERI",
    "Illichman",
    "ILLICHMAN",
    "Stafitech",
    "PENTAFLEX",
    "TP 124",
    "TP124",
    "ČBS 02",
    "CBS 02",
    "NOT250039",
    "NOT250304",
    "NOT260916",
    "nds-qa-02",
    "nds-status-05",
    "nds-adv-04",
)

MODULE_PATH = Path(entity_hints.__file__)
INTENT_MODULE_PATH = Path(answer_intent.__file__)


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


def _values(hints, kind=None):
    return [c.value for c in hints.candidates if kind is None or c.kind is kind]


def _disable_other_flags(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", False)
    monkeypatch.setattr(ai_search, "JSON_SENTINEL_FALLBACK_ENABLED", False)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", False)


def _mock_json_ollama(monkeypatch, payload: dict, sink: list | None = None):
    def fake_call(model, prompt, format_schema=None, timeout=240):
        if sink is not None:
            sink.append({"model": model, "prompt": prompt})
        return json.dumps(payload)

    monkeypatch.setattr(ai_search, "_call_ollama", fake_call)


def _payload(text, zdroj_index=1):
    return {
        "body": [{"text": text, "zdroj_index": zdroj_index, "typ": "fakt"}],
        "nenalezeno": False,
    }


# ---------------------------------------------------------------------------
# flag
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert ai_search_config.ENTITY_HINTS_ENABLED is False
    assert ai_search.ENTITY_HINTS_ENABLED is False


def test_flag_off_leaves_prompt_unchanged(monkeypatch):
    _disable_other_flags(monkeypatch)
    rows = [_row("Dodaci_list.pdf", quote="Dodávka betonu.", heading="Dodací list")]

    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", False)
    off_sink = []
    _mock_json_ollama(monkeypatch, _payload("Odpověď."), off_sink)
    off_result = ai_search.answer("kdo dodal beton?", rows)

    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", True)
    on_sink = []
    _mock_json_ollama(monkeypatch, _payload("Odpověď."), on_sink)
    ai_search.answer("kdo dodal beton?", rows)

    assert "KANDIDÁTI K OVĚŘENÍ" not in off_sink[0]["prompt"]
    assert "_entity_hints_debug" not in off_result
    assert off_result["citations"] is rows
    # ON must differ only by the appended block.
    assert on_sink[0]["prompt"].startswith(off_sink[0]["prompt"])


# ---------------------------------------------------------------------------
# status-05 shape — wrong-entity protection
# ---------------------------------------------------------------------------

def test_who_candidate_from_heading_not_from_neighbour_body():
    """A firm named only in another document's body must never be offered."""
    rows = [
        _row(
            "NOT260916.pdf",
            heading="ILLICHMAN Dodací list",
            quote="Dodávka a montáž dle položek rozpočtu.",
        ),
        _row(
            "Smlouva_o_dilo.pdf",
            heading="Smlouva o dílo",
            quote="Zhotovitel FERI a.s. zajistí veškeré práce v rozsahu díla.",
        ),
    ]
    hints = entity_hints.build_entity_hints("kdo dodal a provádí dodávku?", rows)
    who = _values(hints, HintKind.WHO)
    assert "ILLICHMAN" in who
    assert "FERI" not in who
    assert not any("FERI" in value for value in _values(hints))
    assert all(c.field in ("document", "heading") for c in hints.candidates if c.kind is HintKind.WHO)


def test_who_never_reads_quote_even_when_only_source():
    rows = [_row("Zapis.pdf", heading="Zápis", quote="Práce provedla společnost NAZAREN a.s.")]
    hints = entity_hints.build_entity_hints("kdo provádí práce?", rows)
    assert _values(hints, HintKind.WHO) == []


def test_who_rejects_generic_document_words():
    rows = [_row("Technicka_zprava.pdf", heading="TECHNICKÁ ZPRÁVA", quote="Popis konstrukce.")]
    hints = entity_hints.build_entity_hints("kdo je dodavatel?", rows)
    for value in _values(hints, HintKind.WHO):
        assert entity_hints.fold(value) not in {"technicka", "zprava"}


def test_who_rejects_alphanumeric_code_prefix():
    """A code like NOT260916 must not degrade into a bare 'NOT' name."""
    rows = [_row("NOT260916.pdf", heading="NOT260916", quote="Text.")]
    hints = entity_hints.build_entity_hints("kdo dodal?", rows)
    assert "NOT" not in _values(hints, HintKind.WHO)


# ---------------------------------------------------------------------------
# qa-02 shape — standard detection
# ---------------------------------------------------------------------------

def test_standard_detected_with_original_diacritics():
    rows = [
        _row(
            "Technicka_zprava.pdf",
            heading="TECHNICKÁ ZPRÁVA",
            quote="Konstrukce je navržena podle ČBS 02 a doplňujících podkladů.",
        )
    ]
    hints = entity_hints.build_entity_hints("podle jaké normy je konstrukce navržena?", rows)
    standards = _values(hints, HintKind.STANDARD)
    assert any(value.upper().startswith("ČBS") and "02" in value for value in standards)
    assert all(c.source_index == 1 for c in hints.candidates)


def test_standard_families_detected():
    rows = [
        _row("A.pdf", quote="Provedeno dle ČSN EN 1992-1-1 a TP 124."),
        _row("B.pdf", quote="Dále dle ČSN 73 6242."),
    ]
    hints = entity_hints.build_entity_hints("jaké normy a předpisy platí?", rows, max_candidates=6)
    blob = " ".join(_values(hints, HintKind.STANDARD)).upper()
    assert "1992" in blob
    assert "124" in blob or "6242" in blob


def test_standard_prefix_not_reported_as_identifier():
    rows = [_row("A.pdf", quote="Dle TP 124 se postupuje takto.")]
    hints = entity_hints.build_entity_hints("jaké je číslo předpisu a normy?", rows)
    identifiers = [entity_hints.fold(v) for v in _values(hints, HintKind.IDENTIFIER)]
    assert not any(value.startswith("tp") for value in identifiers)


# ---------------------------------------------------------------------------
# adv-04 shape — NOT / numeric identifier detection
# ---------------------------------------------------------------------------

def test_not_identifier_detected_from_filename():
    rows = [_row("NOT250039.pdf", heading="Dodací list", quote="Dodávka materiálu.")]
    hints = entity_hints.build_entity_hints("jaké je číslo zakázky?", rows)
    identifiers = [value.upper() for value in _values(hints, HintKind.IDENTIFIER)]
    assert any("250039" in value for value in identifiers)
    assert all(c.source_index >= 1 for c in hints.candidates)


def test_numeric_document_identifier_detected():
    rows = [_row("Objednavka.pdf", heading="Objednávka 4500123456", quote="Text.")]
    hints = entity_hints.build_entity_hints("jaké je číslo objednávky?", rows)
    assert any("4500123456" in value for value in _values(hints, HintKind.IDENTIFIER))


def test_identifier_extension_is_not_part_of_value():
    rows = [_row("NOT250039.pdf", quote="Text.")]
    hints = entity_hints.build_entity_hints("číslo zakázky?", rows)
    assert all(".pdf" not in value for value in _values(hints))


# ---------------------------------------------------------------------------
# source_index contract
# ---------------------------------------------------------------------------

def test_source_index_is_one_based_and_never_zero():
    rows = [
        _row("A.pdf", quote="Text bez identifikátoru."),
        _row("NOT250039.pdf", quote="Dodávka."),
        _row("C.pdf", quote="Dle ČSN 73 6242."),
    ]
    hints = entity_hints.build_entity_hints("jaké je číslo zakázky a norma?", rows)
    assert hints.candidates
    for candidate in hints.candidates:
        assert candidate.source_index >= 1
        assert candidate.source_index <= len(rows)
    assert 0 not in hints.as_debug_dict()["source_indexes"]


def test_index_matches_row_position_after_packing_subset():
    """Indexes are positions in the sequence handed to the renderer, not ranks."""
    packed = [_row("C.pdf", quote="Dle ČSN 73 6242."), _row("NOT250039.pdf", quote="Dodávka.")]
    hints = entity_hints.build_entity_hints("jaké je číslo zakázky a norma?", packed)
    by_index = {c.source_index: c for c in hints.candidates}
    for index, candidate in by_index.items():
        blob = " ".join(
            str(packed[index - 1].get(field) or "") for field in ("document", "heading", "quote")
        )
        assert entity_hints.fold(candidate.value) in entity_hints.fold(blob)


# ---------------------------------------------------------------------------
# purity / robustness
# ---------------------------------------------------------------------------

def test_rows_are_not_mutated():
    rows = [_row("NOT250039.pdf", heading="ILLICHMAN Dodací list", quote="Dle ČBS 02.")]
    before = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    entity_hints.build_entity_hints("kdo dodal, číslo zakázky a norma?", rows)
    assert json.dumps(rows, sort_keys=True, ensure_ascii=False) == before


def test_deterministic():
    rows = [_row("NOT250039.pdf", heading="ILLICHMAN Dodací list", quote="Dle ČBS 02.")]
    first = entity_hints.build_entity_hints("kdo dodal a jaké je číslo zakázky?", rows)
    second = entity_hints.build_entity_hints("kdo dodal a jaké je číslo zakázky?", rows)
    assert first.candidates == second.candidates


def test_no_intent_produces_no_hints():
    rows = [_row("NOT250039.pdf", heading="ILLICHMAN Dodací list", quote="Dle ČBS 02.")]
    hints = entity_hints.build_entity_hints("jaká je výška konstrukce?", rows)
    assert hints.candidates == ()
    assert hints.as_prompt_block() == ""
    assert not hints


def test_empty_and_malformed_pool():
    for pool in ([], None, [None, "x", 42]):
        hints = entity_hints.build_entity_hints("kdo dodal?", pool)
        assert hints.candidates == ()
        assert hints.as_prompt_block() == ""


def test_candidate_cap_and_prompt_budget():
    rows = [_row(f"NOT25{i:04d}.pdf", quote=f"Dle ČSN 73 62{i:02d}.") for i in range(12)]
    hints = entity_hints.build_entity_hints("jaké je číslo zakázky a jaké normy?", rows)
    assert len(hints.candidates) <= entity_hints.MAX_CANDIDATES
    assert len(hints.as_prompt_block()) <= entity_hints.PROMPT_CHAR_BUDGET


def test_per_row_kind_cap():
    rows = [_row("A.pdf", quote="Dle ČSN 73 6242, ČSN 73 1201, ČSN 73 0540, ČSN 73 0810.")]
    hints = entity_hints.build_entity_hints("jaké normy platí?", rows)
    per_row = [c for c in hints.candidates if c.kind is HintKind.STANDARD and c.source_index == 1]
    assert len(per_row) <= entity_hints.PER_ROW_KIND_CAP


# ---------------------------------------------------------------------------
# candidates only — never an answer
# ---------------------------------------------------------------------------

def test_prompt_block_offers_candidates_not_answers():
    rows = [_row("NOT250039.pdf", heading="ILLICHMAN Dodací list", quote="Dodávka.")]
    block = entity_hints.build_entity_hints("kdo dodal?", rows).as_prompt_block()
    assert "Kandidát ve zdroji [1]:" in block
    lowered = block.casefold()
    for phrase in (
        "správná odpověď",
        "odpověď je",
        "odpověz",
        "použij tohoto kandidáta",
        "je to ",
    ):
        assert phrase not in lowered


def test_prompt_block_only_contains_visible_values():
    rows = [_row("NOT250039.pdf", heading="ILLICHMAN Dodací list", quote="Dodávka.", path="/tajna/cesta/SKRYTA_FIRMA/NOT250039.pdf")]
    hints = entity_hints.build_entity_hints("kdo dodal a číslo zakázky?", rows)
    for candidate in hints.candidates:
        visible = " ".join(
            str(rows[candidate.source_index - 1].get(field) or "")
            for field in ("document", "heading", "quote")
        )
        assert entity_hints.fold(candidate.value) in entity_hints.fold(visible)
    assert "SKRYTA" not in hints.as_prompt_block()


# ---------------------------------------------------------------------------
# no hardcoded production values
# ---------------------------------------------------------------------------

def test_module_has_no_hardcoded_production_values():
    for path in (MODULE_PATH, INTENT_MODULE_PATH):
        source = path.read_text(encoding="utf-8")
        for value in FORBIDDEN_PRODUCTION_VALUES:
            assert value not in source, f"{path.name} hardcodes {value!r}"


def test_static_runtime_prompts_still_free_of_production_values():
    blob = "\n".join(
        (
            ai_search.JSON_ANSWER_GUARD,
            ai_search.CONCISE_JSON_GUIDANCE,
            ai_search.STRUCTURED_JSON_GUIDANCE,
            entity_hints._PROMPT_HEADER,
        )
    )
    for value in FORBIDDEN_PRODUCTION_VALUES:
        assert value not in blob


# ---------------------------------------------------------------------------
# wiring invariants
# ---------------------------------------------------------------------------

def test_flag_on_keeps_citations_and_renderer_contract(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", True)
    rows = [_row("NOT250039.pdf", heading="ILLICHMAN Dodací list", quote="Dodávka betonu.")]
    sink = []
    _mock_json_ollama(monkeypatch, _payload("Dodavatelem je uvedená firma.", 1), sink)
    result = ai_search.answer("kdo dodal beton?", rows)
    assert result["citations"] is rows
    assert "KANDIDÁTI K OVĚŘENÍ" in sink[0]["prompt"]
    assert result["_entity_hints_debug"]["candidate_count"] >= 1
    assert 0 not in result["_entity_hints_debug"]["source_indexes"]


def test_zero_index_payload_still_not_remapped(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", True)
    rows = [_row("NOT250039.pdf", heading="ILLICHMAN Dodací list", quote="Dodávka.")]
    _mock_json_ollama(monkeypatch, _payload("Tvrzení bez zdroje.", 0))
    ai_search.answer("kdo dodal?", rows)
    assert ai_search._clamp_source_index(0, len(rows)) is None


def test_hints_failure_does_not_break_answer(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", True)

    def boom(query, rows, max_candidates=entity_hints.MAX_CANDIDATES):
        raise RuntimeError("hint failure")

    monkeypatch.setattr(ai_search.entity_hints, "build_entity_hints", boom)
    rows = [_row("A.pdf", quote="Dodávka.")]
    sink = []
    _mock_json_ollama(monkeypatch, _payload("Odpověď.", 1), sink)
    result = ai_search.answer("kdo dodal?", rows)
    assert "KANDIDÁTI K OVĚŘENÍ" not in sink[0]["prompt"]
    assert "_entity_hints_debug" not in result
    assert result["citations"] is rows


def test_evidence_gate_runs_before_hints(monkeypatch):
    """An abstaining evidence gate returns before any hint is built."""
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", True)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)
    calls = []

    def tracking(query, rows, max_candidates=entity_hints.MAX_CANDIDATES):
        calls.append(query)
        return entity_hints.EntityHints(query=query, requested_kinds=(), candidates=())

    monkeypatch.setattr(ai_search.entity_hints, "build_entity_hints", tracking)

    class _Status:
        NO_EVIDENCE = "NO_EVIDENCE"

    rows = [_row("Nesouvisejici.pdf", quote="Úplně jiný projekt a jiné téma.")]
    called = []

    def fake_call(model, prompt, format_schema=None, timeout=240):
        called.append(prompt)
        return json.dumps(_payload("Odpověď.", 1))

    monkeypatch.setattr(ai_search, "_call_ollama", fake_call)
    result = ai_search.answer("kdo dodal PENTAFLEX pásku pro jiný projekt?", rows)
    if not called:
        # Gate abstained: hints were never built and no LLM call happened.
        assert calls == []
        assert result["citations"] == []


# ===========================================================================
# PR9.3.4.1
# ===========================================================================

# ---------------------------------------------------------------------------
# 1) shared intent helper
# ---------------------------------------------------------------------------

TECHNICAL_RULES_QUERY = "podle jakých technických pravidel se navrhuje bílá vana?"


def test_standard_intent_active_for_technical_rules_query():
    """The PR9.3.4 defect: this phrasing produced no hint kind at all."""
    kinds = entity_hints._requested_kinds(entity_hints.fold(TECHNICAL_RULES_QUERY))
    assert HintKind.STANDARD in kinds


def test_both_answer_layers_agree_on_technical_rules_query():
    folded = entity_hints.fold(TECHNICAL_RULES_QUERY)
    assert context_packing._intent(folded)["standard"] is True
    assert "STANDARD" in answer_intent.hint_kinds(folded)


def test_context_packing_uses_the_shared_helper():
    """No second copy of the rules: packing delegates, it does not re-implement."""
    source = Path(context_packing.__file__).read_text(encoding="utf-8")
    assert "answer_intent" in source
    for name in ("_WHO_RE", "_ID_INTENT_RE", "_STANDARD_RE", "_TYPE_RE"):
        assert f"{name} = re.compile" not in source
    for query in (TECHNICAL_RULES_QUERY, "kdo je zhotovitel?", "jaký typ konstrukce?"):
        folded = entity_hints.fold(query)
        assert context_packing._intent(folded) == answer_intent.packing_flags(folded)


def test_packing_profile_stays_frozen_at_pr933_behaviour():
    """PR9.3.3 packing selection was A/B validated; widening it needs a new A/B.

    These two phrasings are exactly where the profiles legitimately differ, so a
    future "let's just merge the rule sets" change trips here instead of silently
    re-ranking the packed context.
    """
    inflected_order = entity_hints.fold("jaké zakázky jsou na zdění?")
    plural_norms = entity_hints.fold("jaké normy platí?")
    assert context_packing._intent(inflected_order)["identifier"] is False
    assert context_packing._intent(plural_norms)["standard"] is False
    assert "IDENTIFIER" in answer_intent.hint_kinds(inflected_order)
    assert "STANDARD" in answer_intent.hint_kinds(plural_norms)


def test_hint_kinds_have_stable_order_and_no_type_kind():
    folded = entity_hints.fold("kdo dodal, podle jaké normy a jaké je číslo zakázky?")
    assert answer_intent.hint_kinds(folded) == ("WHO", "STANDARD", "IDENTIFIER")
    assert answer_intent.classify(entity_hints.fold("jaký typ konstrukce?"), answer_intent.HINT_PROFILE).type is True
    assert answer_intent.hint_kinds(entity_hints.fold("jaký typ konstrukce?")) == ()


# ---------------------------------------------------------------------------
# 2) WHO ranking — status-05 analog
# ---------------------------------------------------------------------------

def test_who_ranking_prefers_heading_party_over_repeated_place_names():
    """Synthetic analog of the A/B case: the right name must come first.

    The A/B offered five WHO candidates where place names shared by several
    filenames and a truncated fragment outranked the party named in a heading.
    """
    rows = [
        _row("250114_Mestecko_Ctvrt_zapis.pdf", heading="Mestecko Ctvrt kontrolní den", quote="Projednáno."),
        _row("DOK250114.pdf", heading="KOVOSTAV Dodací list", quote="Zhotovitel VERTEX a.s. zajistí práce."),
        _row("Prehled_Mestecko.pdf", heading="Ctvrt technický list", quote="Popis."),
        _row("Mes_prehled.pdf", heading="Přehled Mestecko", quote="Text."),
    ]
    hints = entity_hints.build_entity_hints("kdo dělá těsnění?", rows)
    who = [c for c in hints.candidates if c.kind is HintKind.WHO]
    assert who, "expected at least one WHO candidate"
    assert who[0].value == "KOVOSTAV"
    assert who[0].field == "heading"
    # A firm named only in body text is never offered, whatever its score.
    assert all("VERTEX" not in c.value for c in hints.candidates)
    values = [c.value for c in who]
    # Truncated filename fragment is gone; repeated place names rank below.
    assert "Mes" not in values
    for place in ("Mestecko", "Ctvrt"):
        if place in values:
            assert values.index(place) > values.index("KOVOSTAV")


def test_repeated_value_scores_lower_than_unique_value():
    repeated = [
        _row("Mestecko_a.pdf", heading="Mestecko zápis", quote="Text."),
        _row("Mestecko_b.pdf", heading="KOVOSTAV Mestecko list", quote="Text."),
    ]
    hints = entity_hints.build_entity_hints("kdo dodal?", repeated)
    by_value = {c.value: c.score for c in hints.candidates}
    assert by_value.get("KOVOSTAV", 0) > by_value.get("Mestecko", -99)
    reasons = [r for c in hints.candidates if c.value == "Mestecko" for r in c.reasons]
    assert any(r.startswith("repeated_in_rows") for r in reasons)


# ---------------------------------------------------------------------------
# 3) IDENTIFIER filters — adv-04 analog
# ---------------------------------------------------------------------------

REGISTRY_QUOTE = (
    "IČO 03747808, DIČ CZ03747808, tel. 736517669, PSČ 15000, "
    "částka 1250000 Kč, pevnost 30 MPa."
)


def test_identifier_returns_document_codes_and_ignores_registry_numbers():
    rows = [
        _row("Faktura_2025.pdf", heading="Faktura", quote=REGISTRY_QUOTE),
        _row("DOK250039.pdf", heading="Zakázka DOK250039", quote="Zdění příček dle DOK250304."),
    ]
    hints = entity_hints.build_entity_hints("jaké zakázky jsou na zdění?", rows)
    identifiers = _values(hints, HintKind.IDENTIFIER)
    assert any("250039" in value for value in identifiers)
    blob = " ".join(identifiers)
    for rejected in ("03747808", "CZ03747808", "736517669", "15000", "1250000"):
        assert rejected not in blob, f"registry/contact number offered: {rejected}"


def test_bare_numbers_in_body_text_are_never_identifiers():
    rows = [_row("A.pdf", heading="Zápis", quote="Hodnoty 123456, 987654321 a 4500123456.")]
    hints = entity_hints.build_entity_hints("jaké je číslo zakázky?", rows)
    assert _values(hints, HintKind.IDENTIFIER) == []


def test_labelled_number_in_heading_is_rejected():
    rows = [_row("A.pdf", heading="IČO 03747808 objednávka", quote="Text.")]
    hints = entity_hints.build_entity_hints("jaké je číslo objednávky?", rows)
    assert all("03747808" not in value for value in _values(hints))


def test_two_letter_prefix_code_is_not_a_document_identifier():
    """A VAT number is a two-letter country prefix plus digits."""
    rows = [_row("A.pdf", heading="Dodavatel", quote="Plátce CZ03747808.")]
    hints = entity_hints.build_entity_hints("jaké je číslo dokumentu?", rows)
    assert all("03747808" not in value for value in _values(hints))


def test_alpha_prefixed_code_is_read_from_body_text():
    """Body text stays readable for codes, only bare digit runs are gated."""
    rows = [_row("A.pdf", heading="Zápis", quote="Vztahuje se k zakázce DOK250304.")]
    hints = entity_hints.build_entity_hints("jaké je číslo zakázky?", rows)
    assert any("250304" in value for value in _values(hints, HintKind.IDENTIFIER))


# ---------------------------------------------------------------------------
# 4) TP rule
# ---------------------------------------------------------------------------

def test_tp_requires_at_least_three_digits():
    rows = [_row("A.pdf", heading="Zápis", quote="Navrženo dle TP 124 a TP 151. Etapa TP 2 a TP 3.")]
    hints = entity_hints.build_entity_hints("podle jakých technických pravidel?", rows)
    standards = [value.upper().replace(" ", "") for value in _values(hints, HintKind.STANDARD)]
    assert any(value.startswith("TP1") for value in standards)
    for rejected in ("TP2", "TP3"):
        assert rejected not in standards, f"{rejected} must not be a standard candidate"


def test_technical_rules_query_yields_standard_candidates_end_to_end():
    rows = [_row("Technicka_zprava.pdf", heading="TECHNICKÁ ZPRÁVA", quote="Navrženo dle TP 151.")]
    hints = entity_hints.build_entity_hints(TECHNICAL_RULES_QUERY, rows)
    assert HintKind.STANDARD in hints.requested_kinds
    assert any("151" in value for value in _values(hints, HintKind.STANDARD))
    assert hints.as_prompt_block() != ""


# ---------------------------------------------------------------------------
# 5) scoring contract
# ---------------------------------------------------------------------------

def test_candidates_are_ordered_by_score_within_a_kind():
    rows = [
        _row("A.pdf", heading="Zápis", quote="Dle ČSN 73 6242."),
        _row("DOK250039.pdf", heading="Zakázka DOK250039", quote="Text."),
    ]
    hints = entity_hints.build_entity_hints("jaké je číslo zakázky a jaká norma?", rows)
    for kind in (HintKind.STANDARD, HintKind.IDENTIFIER):
        scores = [c.score for c in hints.candidates if c.kind is kind]
        assert scores == sorted(scores, reverse=True)


def test_requested_kind_ranks_above_secondary_kind():
    rows = [_row("DOK250039.pdf", heading="KOVOSTAV Dodací list", quote="Text.")]
    hints = entity_hints.build_entity_hints("kdo dodal a jaké je číslo zakázky?", rows)
    kinds = [c.kind for c in hints.candidates]
    assert kinds, "expected candidates"
    assert kinds[0] is HintKind.WHO


def test_debug_dict_exposes_score_and_reasons():
    rows = [_row("DOK250039.pdf", heading="KOVOSTAV Dodací list", quote="Text.")]
    debug = entity_hints.build_entity_hints("kdo dodal?", rows).as_debug_dict()
    assert debug["candidates"]
    first = debug["candidates"][0]
    assert isinstance(first["score"], float)
    assert first["reasons"]
    assert first["source_index"] >= 1


def test_low_score_candidates_are_dropped_instead_of_padding():
    rows = [_row("A.pdf", heading="Zápis", quote="Text.") for _ in range(3)]
    hints = entity_hints.build_entity_hints("kdo dodal?", rows)
    assert all(c.score >= entity_hints.MIN_SCORE for c in hints.candidates)


def test_per_kind_cap():
    names = ("KOVOSTAV", "VERTEXIA", "BETONEX", "IZOLTOP", "STAVIMEX", "PLASTON")
    rows = [_row(f"Zapis_{i}.pdf", heading=f"{name} Dodací list", quote="Text.") for i, name in enumerate(names)]
    hints = entity_hints.build_entity_hints("kdo dodal?", rows)
    who = [c for c in hints.candidates if c.kind is HintKind.WHO]
    assert len(who) == entity_hints.PER_KIND_CAP, [c.value for c in who]
