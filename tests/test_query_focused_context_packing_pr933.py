"""PR9.3.3 — query-focused pre-LLM context packing.

Packing selects a small ZDROJE subset after OLD guard + evidence gate.
It must not mutate retrieval rows, change citations identity (flag OFF),
or remap zdroj_index 0→1. Production case names must not be hardcoded.
"""
from __future__ import annotations

import json
from pathlib import Path

import ai_search
import ai_search_config
import context_packing
import evidence_runtime
import old_revision_guard


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
    "nds-qa-01",
    "nds-status-04",
    "nds-adv-04",
)


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


def _generic_pool(n=8):
    return [
        _row(
            f"Generic_note_{i}.pdf",
            quote=f"Obecný text o stavebních pracích číslo {i}.",
            heading="Poznámka",
            score=0.5,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1–2 flag / identity
# ---------------------------------------------------------------------------

def test_flag_default_off():
    assert ai_search_config.QUERY_FOCUSED_CONTEXT_PACKING_ENABLED is False
    assert ai_search.QUERY_FOCUSED_CONTEXT_PACKING_ENABLED is False


def test_flag_off_does_not_change_answer_context(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", False)
    rows = _generic_pool(8)
    sink = []
    _mock_json_ollama(monkeypatch, _payload("Obecná odpověď.", 1), sink)
    result = ai_search.answer("jaký je stav prací?", rows)
    prompt = sink[0]["prompt"]
    for row in rows:
        assert f"] {row['document']}" in prompt
    assert prompt.count("\n\n[") + 1 == len(rows) or prompt.count("[") >= len(rows)
    assert result["citations"] is rows
    assert "_packed_context_debug" not in result


# ---------------------------------------------------------------------------
# 3–4 packer purity
# ---------------------------------------------------------------------------

def test_max_four_rows():
    rows = _generic_pool(10)
    packed = context_packing.pack_answer_context("jaký je stav prací?", rows, max_rows=4)
    assert packed.packed_count <= 4
    assert packed.original_count == 10
    assert len(packed.rows) <= 4


def test_original_rows_are_not_mutated():
    rows = [
        _row("Alpha.pdf", quote="alpha token unique", heading="Alpha", score=0.9),
        _row("Beta.pdf", quote="beta token unique", heading="Beta", score=0.8),
        _row("Gamma.pdf", quote="gamma token unique", heading="Gamma", score=0.7),
        _row("Delta.pdf", quote="delta token unique", heading="Delta", score=0.6),
        _row("Epsilon.pdf", quote="epsilon token unique", heading="Epsilon", score=0.5),
    ]
    snapshot = [(id(r), r["quote"], r["score"], r["document"], dict(r)) for r in rows]
    packed = context_packing.pack_answer_context("alpha token unique", rows, max_rows=4)
    assert packed.packed_count <= 4
    for original, (obj_id, quote, score, document, mapping) in zip(rows, snapshot):
        assert id(original) == obj_id
        assert original["quote"] is quote or original["quote"] == quote
        assert original["score"] == score
        assert original["document"] == document
        assert original == mapping
    for packed_row in packed.rows:
        assert packed_row in rows
        assert packed_row is rows[rows.index(packed_row)]


# ---------------------------------------------------------------------------
# 5–9 relevance signals (synthetic names only)
# ---------------------------------------------------------------------------

def test_exact_quote_match_beats_generic_overlap():
    relevant = _row(
        "TZ_objekt.pdf",
        quote=(
            "Pro spodní stavbu je navržen typ konstrukce vodonepropustná "
            "železobetonová vana."
        ),
        heading="Spodní stavba",
        path="/proj/TZ_objekt.pdf",
        score=0.4,
    )
    distractors = [
        _row(
            f"Obecny_beton_{i}.pdf",
            quote=(
                "Beton se používá u pilot, kleneb i stěn. Obecný popis betonových "
                "prací a materiálů bez konkrétní odpovědi na navržený typ. " * 4
            ),
            heading="Beton obecně",
            path=f"/proj/obecne/Obecny_beton_{i}.pdf",
            score=0.95,
        )
        for i in range(6)
    ]
    rows = distractors + [relevant]
    packed = context_packing.pack_answer_context(
        "jaký typ betonové konstrukce je navržen pro spodní stavbu",
        rows,
        max_rows=4,
    )
    docs = [r["document"] for r in packed.rows]
    assert "TZ_objekt.pdf" in docs
    assert docs[0] == "TZ_objekt.pdf"


def test_heading_match():
    relevant = _row(
        "Zapis.pdf",
        quote="Práce probíhají dle smlouvy a Harmonogramu.",
        heading="Hydroizolace - dodavatel",
        path="/proj/Zapis.pdf",
        score=0.3,
    )
    distractors = [
        _row(
            f"VOP_{i}.pdf",
            quote="Provádění prací se řídí obecnými podmínkami objednatele.",
            heading="Obchodní podmínky",
            path=f"/proj/vop/VOP_{i}.pdf",
            score=0.9,
        )
        for i in range(6)
    ]
    packed = context_packing.pack_answer_context(
        "kdo provádí hydroizolace",
        distractors + [relevant],
        max_rows=4,
    )
    assert relevant in packed.rows


def test_document_name_match():
    relevant = _row(
        "36_monolit_ACME_NOT990011_smlouva.pdf",
        quote="Předmětem díla je provádění prací dle této smlouvy.",
        heading="Předmět díla",
        path="/proj/36_monolit_ACME_NOT990011_smlouva.pdf",
        score=0.3,
    )
    distractors = [
        _row(
            f"Rozpocet_obecny_{i}.pdf",
            quote="Dodavatel prací je uveden v rozpočtu a v obchodních podmínkách.",
            heading="Rozpočet",
            path=f"/proj/rozpocet/Rozpocet_obecny_{i}.pdf",
            score=0.9,
        )
        for i in range(6)
    ]
    packed = context_packing.pack_answer_context(
        "kdo je dodavatel monolitu",
        distractors + [relevant],
        max_rows=4,
    )
    assert relevant in packed.rows
    assert packed.rows[0]["document"] == "36_monolit_ACME_NOT990011_smlouva.pdf"


def test_path_entity_match():
    relevant = _row(
        "TZ.pdf",
        quote="Specifikace výrobků je v příloze.",
        heading="Provádění prací",
        path="/zakazky/59_tesneni_bile_vany_NovaFirma/TZ.pdf",
        score=0.3,
    )
    distractors = [
        _row(
            f"Spec_{i}.pdf",
            quote="Těsnění se dodává dle specifikace a výkresů obecně.",
            heading="Specifikace",
            path=f"/generic/Spec_{i}.pdf",
            score=0.9,
        )
        for i in range(6)
    ]
    packed = context_packing.pack_answer_context(
        "kdo dělá těsnění bílé vany",
        distractors + [relevant],
        max_rows=4,
    )
    assert relevant in packed.rows


def test_identifier_match():
    relevant = _row(
        "Objednavka_NOT990011.pdf",
        quote="Předmět díla dle této objednávky.",
        heading="Objednávka",
        path="/zakazky/NOT990011/Objednavka_NOT990011.pdf",
        score=0.2,
    )
    distractors = [
        _row(
            f"Prehled_zakazek_{i}.pdf",
            quote="Přehled zakázek a smluv bez konkrétního čísla této objednávky.",
            heading="Evidence",
            path=f"/admin/Prehled_zakazek_{i}.pdf",
            score=0.9,
        )
        for i in range(6)
    ]
    packed = context_packing.pack_answer_context(
        "jaké je číslo zakázky NOT990011",
        distractors + [relevant],
        max_rows=4,
    )
    assert relevant in packed.rows


# ---------------------------------------------------------------------------
# 10 identifier normalization
# ---------------------------------------------------------------------------

def test_technical_identifiers_survive_normalization():
    assert "cbs02" in context_packing.extract_identifiers("ČBS 02")
    assert "cbs02" in context_packing.extract_identifiers("CBS02")
    assert "tp124" in context_packing.extract_identifiers("TP 124")
    assert "tp124" in context_packing.extract_identifiers("TP124")
    assert "d.1.2.06" in context_packing.extract_identifiers("D.1.2.06")
    assert "not990011" in context_packing.extract_identifiers("NOT990011")
    folded = context_packing.fold("ČBS 02 a TP 124 a D.1.2.06")
    assert "02" in folded
    assert "124" in folded
    assert "1.2.06" in folded or "1" in folded


# ---------------------------------------------------------------------------
# 11–12 multi-doc / no global top-1
# ---------------------------------------------------------------------------

def test_multiple_relevant_documents_can_survive():
    doc_a = _row(
        "Smlouva_NOT880001.pdf",
        quote="Zakázka NOT880001 je uzavřena.",
        heading="Smlouva NOT880001",
        path="/zakazky/NOT880001/Smlouva_NOT880001.pdf",
        score=0.9,
    )
    extra_a = [
        _row(
            "Smlouva_NOT880001.pdf",
            quote=f"Další chunk smlouvy {i}.",
            heading="Příloha",
            path="/zakazky/NOT880001/Smlouva_NOT880001.pdf",
            score=0.85,
        )
        for i in range(4)
    ]
    doc_b = _row(
        "Smlouva_NOT880002.pdf",
        quote="Zakázka NOT880002 je uzavřena.",
        heading="Smlouva NOT880002",
        path="/zakazky/NOT880002/Smlouva_NOT880002.pdf",
        score=0.4,
    )
    packed = context_packing.pack_answer_context(
        "jaká jsou čísla zakázek NOT880001 a NOT880002",
        [doc_a, *extra_a, doc_b],
        max_rows=4,
    )
    docs = {r["document"] for r in packed.rows}
    assert "Smlouva_NOT880001.pdf" in docs
    assert "Smlouva_NOT880002.pdf" in docs
    assert packed.packed_count >= 2


def test_no_global_top1_only_pruning():
    best = _row("Best.pdf", quote="exact unique widget token", heading="Widget", score=0.99)
    others = [
        _row(f"Other_{i}.pdf", quote="other background text", heading="Other", score=0.1)
        for i in range(5)
    ]
    packed = context_packing.pack_answer_context(
        "exact unique widget token",
        [best, *others],
        max_rows=4,
    )
    assert packed.packed_count >= 2
    assert packed.packed_count <= 4


# ---------------------------------------------------------------------------
# 13–15 source index mapping
# ---------------------------------------------------------------------------

def test_packed_source_numbering_is_1_based(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    rows = _generic_pool(8)
    rows[4] = _row(
        "Hit.pdf",
        quote="exact unique widget token in quote",
        heading="Hit",
        path="/proj/Hit.pdf",
        score=0.2,
    )
    sink = []
    _mock_json_ollama(monkeypatch, _payload("Widget nalezen.", 1), sink)
    ai_search.answer("exact unique widget token", rows)
    prompt = sink[0]["prompt"]
    assert "[1] " in prompt
    assert "zdroj_index" in ai_search.JSON_ANSWER_GUARD
    packed_headers = [line for line in prompt.splitlines() if line.startswith("[")]
    assert packed_headers
    assert packed_headers[0].startswith("[1]")
    assert len(packed_headers) <= 4
    assert not any(line.startswith("[0]") for line in packed_headers)


def test_packed_index_maps_to_original_document(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    original = [
        _row("Noise_A.pdf", quote="šum a beton obecně", heading="Šum", score=0.9),
        _row(
            "Target_Alpha.pdf",
            quote="Alpha widget token explicitně uveden.",
            heading="Alpha",
            path="/proj/Target_Alpha.pdf",
            score=0.2,
        ),
        _row("Noise_B.pdf", quote="další šum", heading="Šum", score=0.8),
        _row(
            "Target_Beta.pdf",
            quote="Beta widget token explicitně uveden.",
            heading="Beta",
            path="/proj/Target_Beta.pdf",
            score=0.2,
        ),
        _row("Noise_C.pdf", quote="ještě šum", heading="Šum", score=0.7),
        _row("Noise_D.pdf", quote="šum čtyři", heading="Šum", score=0.6),
        _row("Noise_E.pdf", quote="šum pět", heading="Šum", score=0.5),
        _row("Noise_F.pdf", quote="šum šest", heading="Šum", score=0.4),
    ]
    packed = context_packing.pack_answer_context("Alpha widget token a Beta widget token", original)
    assert packed.packed_count >= 2
    assert packed.rows[0] in original
    second = packed.rows[1]
    sink = []
    _mock_json_ollama(
        monkeypatch,
        _payload(f"Tvrzení ze zdroje {second['document']}.", 2),
        sink,
    )
    result = ai_search.answer("Alpha widget token a Beta widget token", original)
    assert second["document"] in result["answer"]
    assert f"(Zdroj: {second['document']})" in result["answer"]
    debug = result["_packed_context_debug"]
    assert debug["selected_original_ranks"][1] == original.index(second) + 1


def test_index_zero_stays_invalid(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    rows = [
        _row("Keep_A.pdf", quote="alpha widget token", heading="A", score=0.9),
        _row("Keep_B.pdf", quote="beta widget token", heading="B", score=0.8),
        *_generic_pool(6),
    ]
    _mock_json_ollama(monkeypatch, _payload("Neplatný index nula.", 0))
    result = ai_search.answer("alpha widget token", rows)
    assert ai_search._clamp_source_index(0, 4) is None
    assert "Neplatný index nula" not in result["answer"]
    assert result["answer"].startswith("Nenalezeno v indexovaných dokumentech.")


# ---------------------------------------------------------------------------
# 16–19 pipeline identity
# ---------------------------------------------------------------------------

def test_evidence_gate_receives_full_pool_not_packed(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    rows = _generic_pool(8)
    rows[6] = _row("Hit.pdf", quote="exact unique widget token", heading="Hit", score=0.2)
    seen = {}

    def fake_safety(query, results=None):
        pool = list(results or [])
        seen["count"] = len(pool)
        seen["docs"] = [r["document"] for r in pool]
        return evidence_runtime.EvidenceSafety(status=evidence_runtime.EvidenceSafetyStatus.OK)

    monkeypatch.setattr(evidence_runtime, "evaluate_evidence_safety", fake_safety)
    _mock_json_ollama(monkeypatch, _payload("ok", 1))
    result = ai_search.answer("exact unique widget token", rows)
    assert seen["count"] == 8
    assert seen["docs"] == [r["document"] for r in rows]
    assert result["citations"] is rows
    assert result["_packed_context_debug"]["original_count"] == 8
    assert result["_packed_context_debug"]["packed_count"] <= 4


def test_old_guard_runs_before_packing(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", True)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    order = []
    real_guard = old_revision_guard.apply_old_revision_guard
    real_pack = context_packing.pack_answer_context

    def wrap_guard(query, results):
        order.append("old")
        return real_guard(query, results)

    def wrap_pack(query, results, max_rows=4):
        order.append("pack")
        assert "old" in order
        return real_pack(query, results, max_rows=max_rows)

    monkeypatch.setattr(old_revision_guard, "apply_old_revision_guard", wrap_guard)
    monkeypatch.setattr(context_packing, "pack_answer_context", wrap_pack)
    rows = _generic_pool(8)
    _mock_json_ollama(monkeypatch, _payload("ok", 1))
    ai_search.answer("jaký je stav prací?", rows)
    assert order == ["old", "pack"]


def test_citation_mapping_uses_packed_row_not_original_rank(monkeypatch):
    """Packed [1]=orig #2, [2]=orig #5 → zdroj_index=2 cites original #5."""
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", True)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    rows = _generic_pool(8)
    rows[1] = _row(
        "Original_rank2.pdf",
        quote="first widget token here",
        heading="First",
        path="/proj/Original_rank2.pdf",
        score=0.2,
    )
    rows[4] = _row(
        "Original_rank5.pdf",
        quote="second widget token here",
        heading="Second",
        path="/proj/Original_rank5.pdf",
        score=0.2,
    )
    packed = context_packing.pack_answer_context("first widget token second widget token", rows)
    assert 2 in packed.selected_original_ranks
    assert 5 in packed.selected_original_ranks
    packed_pos_of_rank5 = packed.selected_original_ranks.index(5) + 1
    _mock_json_ollama(
        monkeypatch,
        _payload("Citace původního pátého výsledku.", packed_pos_of_rank5),
    )
    result = ai_search.answer("first widget token second widget token", rows)
    assert "Original_rank5.pdf" in result["answer"]
    assert "(Zdroj: Original_rank5.pdf)" in result["answer"]
    assert result["citations"] is rows
    assert len(result["citations"]) == 8


def test_full_citations_invariant_preserved(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    rows = _generic_pool(8)
    _mock_json_ollama(monkeypatch, _payload("ok", 1))
    result = ai_search.answer("jaký je stav prací?", rows)
    assert result["citations"] is rows
    assert len(result["citations"]) == 8
    assert result["_packed_context_debug"]["packed_count"] <= 4


def test_packing_skipped_when_evidence_gate_abstains(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", True)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    pack_calls = []

    def fake_safety(query, results=None):
        return evidence_runtime.EvidenceSafety(
            status=evidence_runtime.EvidenceSafetyStatus.NO_EVIDENCE,
            message="Nenalezeno v indexovaných dokumentech.",
        )

    def fake_pack(*_a, **_k):
        pack_calls.append(1)
        raise AssertionError("packer must not run before safety gate abstention")

    monkeypatch.setattr(evidence_runtime, "evaluate_evidence_safety", fake_safety)
    monkeypatch.setattr(context_packing, "pack_answer_context", fake_pack)
    result = ai_search.answer("jaký je stav prací?", _generic_pool(8))
    assert pack_calls == []
    assert "Nenalezeno" in result["answer"]
    assert "_packed_context_debug" not in result


def test_model_routing_uses_full_pool_not_packed_count(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", True)
    rows = _generic_pool(8)
    sink = []
    _mock_json_ollama(monkeypatch, _payload("ok", 1), sink)
    ai_search.answer("krátký dotaz", rows)
    assert sink[0]["model"] == ai_search.COMPLEX_MODEL
    assert len(rows) > 6


# ---------------------------------------------------------------------------
# 20 no production hardcode
# ---------------------------------------------------------------------------

def test_no_production_specific_hardcode():
    source = Path(context_packing.__file__).read_text(encoding="utf-8")
    blob = source + ai_search.JSON_ANSWER_GUARD
    for value in FORBIDDEN_PRODUCTION_VALUES:
        assert value not in blob, f"hardcoded production value {value!r}"
