"""PR9.3.1 — historical NO-GO artifact (prompt-only extraction).

PR9.3.1 added query-focused extraction rules to JSON_ANSWER_GUARD.
Live A/B showed 0/6 extraction; the hunk was removed in PR9.3.3.

This file now asserts the NO-GO prompt is absent and that the PR9.2.2
1-based `zdroj_index` contract (schema without minimum:1) remains.
"""
from __future__ import annotations

import ai_search


FORBIDDEN_PRODUCTION_VALUES = (
    "FERI",
    "Illichman",
    "ILLICHMAN",
    "TP 124",
    "ČBS 02",
    "CBS 02",
    "NOT250039",
    "NOT250304",
    "PENTAFLEX",
    "nds-qa-01",
    "nds-qa-02",
    "nds-qa-03",
    "nds-qa-04",
    "nds-status-04",
    "nds-status-05",
    "nds-adv-04",
)

PR931_NOGO_MARKERS = (
    "Nejdřív urč, jaký konkrétní údaj uživatel požaduje",
    "uveď jej jako první faktickou položku",
    "Nezahlcuj odpověď nesouvisejícími fakty",
    "Norma / předpis",
    "přesný identifikátor předpisu",
    "Nenahrazuj jej pouze obecným popisem",
    "Kdo / dodavatel",
    "konkrétní firmu nebo osobu",
    "Nevracej místo toho jen obecný smluvní text nebo položky rozpočtu",
    "Číslo zakázky / identifikátor",
    "názvu zdroje",
    "ne v těle úryvku",
    "Typ / druh",
    "explicitní označení typu ze zdroje",
    "ne sousední technické informace",
)

RUNTIME_PROMPTS = (
    ai_search.JSON_ANSWER_GUARD,
    ai_search.CONCISE_JSON_GUIDANCE,
    ai_search.STRUCTURED_JSON_GUIDANCE,
)


def _zdroj_index_schema(schema: dict) -> dict:
    return schema["properties"]["zdroj_index"]


def _concise_item_schema() -> dict:
    return ai_search.CONCISE_ANSWER_SCHEMA["properties"]["body"]["items"]


def _structured_item_schema() -> dict:
    oblasti = ai_search.STRUCTURED_ANSWER_SCHEMA["properties"]["oblasti"]
    return oblasti["items"]["properties"]["polozky"]["items"]


def test_pr931_nogo_prompt_hunk_removed():
    blob = "\n".join(RUNTIME_PROMPTS)
    for marker in PR931_NOGO_MARKERS:
        assert marker not in blob, f"PR9.3.1 NO-GO prompt still present: {marker!r}"


def test_pr922_1based_contract_still_present():
    for text in RUNTIME_PROMPTS:
        assert "1-based" in text
        assert "1 až N" in text
        assert "Nikdy nepoužívej 0" in text
        assert "Hodnota 0 je neplatná" in text
        assert "NEVYTVÁŘEJ" in text
        assert "[1] dokument A → zdroj_index: 1" in text
        assert "[2] dokument B → zdroj_index: 2" in text


def test_schema_still_has_no_minimum_or_default():
    for schema in (
        ai_search._ANSWER_ITEM_SCHEMA,
        _concise_item_schema(),
        _structured_item_schema(),
    ):
        field = _zdroj_index_schema(schema)
        assert field["type"] == "integer"
        assert "minimum" not in field
        assert "default" not in field
        assert "1-based" in field["description"]
        assert "Hodnota 0 je neplatná" in field["description"]


def test_zero_is_not_remapped_to_one():
    assert ai_search._clamp_source_index(0, 10) is None
    assert ai_search._clamp_source_index(0, 10) != 1
    assert ai_search._clamp_source_index(1, 10) == 1


def test_runtime_prompt_has_no_hardcoded_production_case_values():
    blob = "\n".join(RUNTIME_PROMPTS)
    for value in FORBIDDEN_PRODUCTION_VALUES:
        assert value not in blob, f"runtime prompt hardcodes {value!r}"
