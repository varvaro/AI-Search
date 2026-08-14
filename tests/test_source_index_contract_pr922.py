"""PR9.2.2 — explicit 1-based `zdroj_index` prompt + schema description.

The runtime contract is already 1-based (`results[zdroj_index - 1]`,
`_clamp_source_index` accepts only `1 <= i <= count`). This file checks
that JSON instructions and the shared item-schema description now state
that contract explicitly, so the model is less likely to emit 0.

Does not add schema `"minimum": 1`, does not remap 0→1, and does not
touch `_clamp_source_index` / renderer behaviour.
"""
from __future__ import annotations

import ai_search


def _zdroj_index_schema(schema: dict) -> dict:
    return schema["properties"]["zdroj_index"]


def _concise_item_schema() -> dict:
    return ai_search.CONCISE_ANSWER_SCHEMA["properties"]["body"]["items"]


def _structured_item_schema() -> dict:
    oblasti = ai_search.STRUCTURED_ANSWER_SCHEMA["properties"]["oblasti"]
    return oblasti["items"]["properties"]["polozky"]["items"]


def test_json_guard_states_1based_rule():
    assert "1-based" in ai_search.JSON_ANSWER_GUARD
    assert "1-based" in ai_search.CONCISE_JSON_GUIDANCE
    assert "1-based" in ai_search.STRUCTURED_JSON_GUIDANCE


def test_json_guard_forbids_zero_index():
    guard = ai_search.JSON_ANSWER_GUARD
    assert "Nikdy nepoužívej 0" in guard
    assert "Hodnota 0 je neplatná" in guard
    assert "Nepoužívej 0 jako" in guard
    assert "Nikdy nepoužívej 0" in ai_search.CONCISE_JSON_GUIDANCE
    assert "Nikdy nepoužívej 0" in ai_search.STRUCTURED_JSON_GUIDANCE


def test_json_guard_maps_bracket_labels_to_source_index():
    guard = ai_search.JSON_ANSWER_GUARD
    assert "[1] dokument A → zdroj_index: 1" in guard
    assert "[2] dokument B → zdroj_index: 2" in guard
    assert "[1] dokument A → zdroj_index: 1" in ai_search.CONCISE_JSON_GUIDANCE
    assert "[2] dokument B → zdroj_index: 2" in ai_search.STRUCTURED_JSON_GUIDANCE


def test_json_guard_valid_range_is_1_to_n():
    guard = ai_search.JSON_ANSWER_GUARD
    assert "1 až N" in guard
    assert "NEVYTVÁŘEJ" in guard


def test_item_schema_description_states_1based_range_and_zero_invalid():
    desc = _zdroj_index_schema(ai_search._ANSWER_ITEM_SCHEMA)["description"]
    assert "1-based" in desc
    assert "[1] až [N]" in desc
    assert "Hodnota 0 je neplatná" in desc


def test_concise_schema_uses_1based_zdroj_index_description():
    item = _concise_item_schema()
    desc = _zdroj_index_schema(item)["description"]
    assert "1-based" in desc
    assert "[1] až [N]" in desc
    assert "Hodnota 0 je neplatná" in desc
    assert "minimum" not in _zdroj_index_schema(item)


def test_structured_schema_uses_1based_zdroj_index_description():
    item = _structured_item_schema()
    desc = _zdroj_index_schema(item)["description"]
    assert "1-based" in desc
    assert "[1] až [N]" in desc
    assert "Hodnota 0 je neplatná" in desc
    assert "minimum" not in _zdroj_index_schema(item)


def test_schema_has_no_minimum_constraint():
    """PR9.2.2 must not add schema minimum:1 — that would coerce zeros to 1."""
    for schema in (
        ai_search._ANSWER_ITEM_SCHEMA,
        _concise_item_schema(),
        _structured_item_schema(),
    ):
        field = _zdroj_index_schema(schema)
        assert field["type"] == "integer"
        assert "minimum" not in field
        assert "default" not in field


def test_clamp_zero_is_none():
    assert ai_search._clamp_source_index(0, 10) is None
    assert ai_search._clamp_source_index(0, 1) is None


def test_clamp_one_stays_one():
    assert ai_search._clamp_source_index(1, 10) == 1
    assert ai_search._clamp_source_index(1, 1) == 1


def test_clamp_out_of_range_is_none():
    assert ai_search._clamp_source_index(11, 10) is None
    assert ai_search._clamp_source_index(-1, 10) is None
    assert ai_search._clamp_source_index(99, 5) is None


def test_zero_is_not_remapped_to_one():
    """0 stays invalid; clamp must never rewrite it to document [1]."""
    assert ai_search._clamp_source_index(0, 10) != 1
    assert ai_search._clamp_source_index(0, 10) is None
    rendered = ai_search._render_answer_item(
        {"text": "Tvrzení.", "zdroj_index": 0, "typ": "fakt"},
        [{"document": "doc-one.pdf", "path": "/proj/doc-one.pdf", "quote": "x"}],
    )
    prefix, text, used_index, document = rendered
    assert used_index is None
    assert document is None
    assert document != "doc-one.pdf"
