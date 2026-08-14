"""PR7.4.1 — benchmark measurement hardening.

Each test pins one audit finding (P1..P7) so a regression re-breaks a named
contract rather than a nameless number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_search  # noqa: E402
from benchmark import acceptance_metrics, answer_evidence, pr74_metrics, pr74_runner  # noqa: E402
from benchmark import report as report_mod  # noqa: E402
from benchmark.acceptance_runner import ACCEPTANCE_DATASET  # noqa: E402
from benchmark.dataset.schema import BenchmarkCase, load_dataset  # noqa: E402

PR74_DATASET = pr74_runner.PR74_DATASET


def case(**overrides) -> BenchmarkCase:
    data = {
        "id": "t", "question": "je podepsaná smlouva haus365?",
        "environment": "fixture", "category": "ENTITY_SAFETY",
    }
    data.update(overrides)
    return BenchmarkCase.from_dict(data)


def answer_dict(text: str, documents: list[str], validation: dict | None = None) -> dict:
    payload = {
        "answer": text,
        "citations": [{"document": d, "path": f"/{d}"} for d in documents],
        "confidence": "yellow",
    }
    if validation is not None:
        payload["validation"] = validation
    return payload


# ---------------------------------------------------------------------------
# P1 — safety judged against used evidence, not the whole pool
# ---------------------------------------------------------------------------

def test_evidence_tiers_separate_pool_from_used_documents():
    answer = answer_dict(
        "Ne - smlouva nebyla nalezena.\n(Zdroj: SOD_HAUS365_podepsané.pdf)",
        ["SOD_HAUS365_podepsané.pdf", "SOD_VAHOSTAV_podepsaná.pdf"],
    )
    tiers = answer_evidence.evidence_tiers(answer)
    assert tiers["retrieved"] == ["SOD_HAUS365_podepsané.pdf", "SOD_VAHOSTAV_podepsaná.pdf"]
    assert tiers["cited"] == ["SOD_HAUS365_podepsané.pdf"]
    assert "SOD_VAHOSTAV_podepsaná.pdf" not in tiers["evidence"]


def test_confidence_block_is_not_part_of_the_answer_body():
    answer = answer_dict(
        "Odpověď.\n\nJistota odpovědi:\nStředn\u00ed\n- 2 zdroje, ale SOD_VAHOSTAV_podepsaná.pdf",
        ["SOD_VAHOSTAV_podepsaná.pdf"],
    )
    assert answer_evidence.cited_documents(answer) == []


def test_perfect_entity_mismatch_answer_is_not_blocking():
    """The audit's headline false positive: a maximally safe answer scored
    BLOCKING only because the distractor was in the retrieval pool."""
    c = case(
        expected_state_verdict="ENTITY_MISMATCH",
        forbidden_sources=["VAHOSTAV"],
        forbidden_answer_contains=["Ano - na boxu je podepsaná"],
    )
    safe = answer_dict(
        "Nelze jednoznačně ověřit stav podpisu smlouvy pro dotazovaný subjekt.",
        ["SOD_VAHOSTAV_podepsaná.pdf"],
        {"state_verdict": "ENTITY_MISMATCH", "intent_coverage": "PARTIAL", "missing_needs": []},
    )
    ev = pr74_metrics.evaluate_case_answers(c, safe, safe)
    assert not ev.has_blocking
    codes = {f.code for f in ev.failures}
    assert "wrong_entity_citation" not in codes
    # The distractor is still reported — as a retrieval observation, severity LOW.
    assert "forbidden_source_retrieved" in codes
    low = next(f for f in ev.failures if f.code == "forbidden_source_retrieved")
    assert low.severity == pr74_metrics.SEVERITY_LOW
    assert low.layer == pr74_metrics.LAYER_RETRIEVAL


def test_answer_that_really_leans_on_forbidden_source_is_blocking():
    c = case(expected_state_verdict="ENTITY_MISMATCH", forbidden_sources=["VAHOSTAV"])
    bad = answer_dict(
        "Ano - na boxu je podepsaná smlouva.\n(Zdroj: SOD_VAHOSTAV_podepsaná.pdf)",
        ["SOD_VAHOSTAV_podepsaná.pdf"],
        {"state_verdict": "ENTITY_MISMATCH"},
    )
    ev = pr74_metrics.evaluate_case_answers(c, bad, bad)
    assert ev.has_blocking
    assert "wrong_entity_citation" in {f.code for f in ev.failures}


def test_state_documents_count_as_used_evidence():
    c = case(expected_state_verdict="UNVERIFIED", forbidden_sources=["VAHOSTAV"])
    answer = answer_dict(
        "Nelze jednoznačně ověřit stav podpisu.",
        ["SOD_VAHOSTAV_podepsaná.pdf"],
        {
            "state_verdict": "UNVERIFIED",
            "state_documents": [{"document": "SOD_VAHOSTAV_podepsaná.pdf", "state": "SIGNED"}],
        },
    )
    ev = pr74_metrics.evaluate_case_answers(c, answer, answer)
    assert "wrong_entity_citation" in {f.code for f in ev.failures}


# ---------------------------------------------------------------------------
# P1/P4 — failure layers
# ---------------------------------------------------------------------------

def test_missing_expected_source_splits_retrieval_from_answer():
    not_retrieved = pr74_metrics.evaluate_case_answers(
        case(expected_source_contains=["HAUS365"]),
        None,
        answer_dict("Odpověď.", ["JINY_DOKUMENT.pdf"]),
    )
    codes = {f.code: f.layer for f in not_retrieved.failures}
    assert codes["expected_source_not_retrieved"] == pr74_metrics.LAYER_RETRIEVAL

    retrieved_unused = pr74_metrics.evaluate_case_answers(
        case(expected_source_contains=["HAUS365"]),
        None,
        answer_dict("Odpověď bez zdroje.", ["SOD_HAUS365.pdf"]),
    )
    codes = {f.code: f.layer for f in retrieved_unused.failures}
    assert codes["expected_source_not_cited"] == pr74_metrics.LAYER_ANSWER


def test_first_failure_layer_is_the_earliest_stage():
    ev = pr74_metrics.evaluate_case_answers(
        case(expected_state_verdict="SIGNED_CONFIRMED", expected_source_contains=["HAUS365"]),
        None,
        answer_dict("Odpověď.", ["JINY.pdf"], {"state_verdict": "UNVERIFIED"}),
    )
    assert ev.first_failure_layer == pr74_metrics.LAYER_RETRIEVAL
    assert pr74_metrics.LAYER_EVIDENCE in ev.failure_layers


def test_aggregate_reports_by_layer():
    evaluations = [
        pr74_metrics.evaluate_case_answers(
            case(id="a", expected_source_contains=["HAUS365"]),
            None, answer_dict("x", ["JINY.pdf"]),
        ),
        pr74_metrics.evaluate_case_answers(
            case(id="b", expected_state_verdict="SIGNED_CONFIRMED"),
            None, answer_dict("x", ["D.pdf"], {"state_verdict": "UNVERIFIED"}),
        ),
    ]
    agg = pr74_metrics.aggregate_evaluations(evaluations)
    assert agg.by_layer["RETRIEVAL"]["cases"] == 1
    assert agg.by_layer["EVIDENCE"]["cases"] == 1
    assert agg.first_failure_layer_counts == {"RETRIEVAL": 1, "EVIDENCE": 1}


# ---------------------------------------------------------------------------
# P6 — improved/degraded symmetry
# ---------------------------------------------------------------------------

def test_identical_violation_in_both_answers_is_not_degraded():
    c = case(expected_state_verdict="UNVERIFIED")
    baseline = answer_dict("Smlouva není podepsaná.", ["D.pdf"], {"state_verdict": "UNVERIFIED"})
    candidate = answer_dict(
        "Smlouva není podepsaná (jinými slovy).", ["D.pdf"], {"state_verdict": "UNVERIFIED"},
    )
    ev = pr74_metrics.evaluate_case_answers(c, baseline, candidate)
    assert ev.answer_delta == "changed_neutral"


def test_baseline_and_candidate_use_the_same_violation_codes():
    """Structural guarantee: one function computes both sides."""
    c = case(expected_state_verdict="UNVERIFIED", forbidden_sources=["VAHOSTAV"])
    both = answer_dict(
        "Ano - je podepsaná.\n(Zdroj: SOD_VAHOSTAV_podepsaná.pdf)",
        ["SOD_VAHOSTAV_podepsaná.pdf"],
        {"state_verdict": "UNVERIFIED"},
    )
    left = pr74_metrics._regression_codes(pr74_metrics._answer_violations(c, both))
    right = pr74_metrics._regression_codes(pr74_metrics._answer_violations(c, both))
    assert left == right and left


def test_trading_one_defect_for_another_is_degraded():
    c = case(expected_state_verdict="UNVERIFIED", forbidden_sources=["VAHOSTAV"])
    baseline = answer_dict("Smlouva není podepsaná.", ["D.pdf"], {"state_verdict": "UNVERIFIED"})
    candidate = answer_dict(
        "Ano - je podepsaná smlouva.", ["D.pdf"], {"state_verdict": "UNVERIFIED"},
    )
    ev = pr74_metrics.evaluate_case_answers(c, baseline, candidate)
    assert ev.answer_delta == "degraded"


# ---------------------------------------------------------------------------
# P3 — synthetic pool is built from document fields only
# ---------------------------------------------------------------------------

def test_synthetic_pool_ignores_answer_assertions():
    c = case(
        expected_documents=["NOT250060_BOZP_SafetyPeak_podepsaná.pdf"],
        expected_source_contains=["podepsaná"],
        forbidden_sources=["Doškář"],
    )
    documents = [row["document"] for row in pr74_runner._build_synthetic_pool(c)]
    assert documents == ["NOT250060_BOZP_SafetyPeak_podepsaná.pdf"]


def test_expected_source_contains_can_no_longer_satisfy_itself():
    c = case(expected_documents=["A.pdf"], expected_source_contains=["podepsaná"])
    pool = pr74_runner._build_synthetic_pool(c)
    assert not any("podepsan" in row["document"].casefold() for row in pool)


# ---------------------------------------------------------------------------
# P2 — warmup excludes latency only
# ---------------------------------------------------------------------------

def _case_dict(case_id: str, *, warmup: bool, blocking: bool) -> dict:
    failures = (
        [{"severity": "BLOCKING", "code": "wrong_entity_citation", "detail": "d", "layer": "SAFETY"}]
        if blocking else []
    )
    return {
        "id": case_id, "warmup": warmup, "error": None, "retrieval_ms": 10.0,
        "retrieval_skipped": False, "live_answer_ms": 100.0, "gate_delta_ms": 1.0,
        "validation_delta_ms": 2.0, "candidate_delta_ms": 3.0,
        "modes": {}, "evaluation": {
            "id": case_id, "category": "SIGNED_DOCUMENT", "answer_delta": "unchanged",
            "failures": failures, "state_verdict_expected": None,
        },
    }


def test_warmup_case_still_counts_for_blocking():
    dicts = [_case_dict("first", warmup=True, blocking=True)]
    evaluation = pr74_metrics.CaseEvaluation(
        id="first", category="SIGNED_DOCUMENT", answer_delta="unchanged",
        failures=[pr74_metrics.Failure("BLOCKING", "wrong_entity_citation", "d", "SAFETY")],
    )
    agg = pr74_metrics.aggregate_evaluations([evaluation])
    go = pr74_runner._go_nogo(agg, dicts)
    assert go["verdict"] == "NO-GO"
    assert "first" in go["blocking_case_ids"]


def test_warmup_case_is_excluded_from_latency_only():
    dicts = [
        _case_dict("first", warmup=True, blocking=False),
        _case_dict("second", warmup=False, blocking=False),
    ]
    latency = pr74_runner._latency_aggregate(dicts)
    assert latency["warmup_excluded_case_ids"] == ["first"]
    assert latency["live_answer_ms"]["n"] == 1


# ---------------------------------------------------------------------------
# P5 — latency deltas
# ---------------------------------------------------------------------------

def test_latency_series_are_deltas_not_relabelled_wall_times():
    latency = pr74_runner._latency_aggregate([_case_dict("a", warmup=False, blocking=False)])
    assert set(latency) >= {
        "retrieval_ms", "live_answer_ms", "end_to_end_ms",
        "state_gate_delta_ms", "validation_delta_ms", "candidate_delta_ms",
    }
    assert "answer_candidate_ms" not in latency
    assert "state_gate_overhead_ms" not in latency
    assert latency["state_gate_delta_ms"]["mean_ms"] == 1.0
    assert latency["end_to_end_ms"]["mean_ms"] == 110.0


def test_deltas_absent_without_replay(monkeypatch):
    c = load_dataset(PR74_DATASET)[0]
    calls = {"n": 0}

    def fake_answer(query, results):
        calls["n"] += 1
        return {"answer": "Odpověď.", "citations": results, "confidence": "green"}

    monkeypatch.setattr(ai_search, "answer", fake_answer)
    monkeypatch.setattr(
        pr74_runner, "run_pipeline_trace",
        lambda *a, **k: type("T", (), {"final_results": [{"document": "A.pdf", "path": "/A.pdf"}]})(),
    )
    result = pr74_runner.evaluate_pr74_case(c, object(), llm_replay=False)
    assert result.error is None
    assert result.gate_delta_ms is None
    assert result.live_answer_ms is None
    assert calls["n"] == 4  # exactly the four modes, no priming call


# ---------------------------------------------------------------------------
# P7 — dataset expectation matches the runtime
# ---------------------------------------------------------------------------

def test_every_synthetic_case_expectation_matches_evidence_runtime():
    """Pin: fixture synthetic-pool expected_state_verdict still matches runtime.

    PR7.6.1 narrows SIGNED_CONFIRMED to contract-kind evidence when the query
    asks for a smlouva. One synthetic case (`pr74-entity-zakladani-hb-with-real`)
    uses a signed DODATEK as the Zakládání hit against a "podepsaná smlouva"
    query — under the new rule that correctly becomes
    SIGNED_OTHER_DOCUMENT_CONFIRMED. Dataset is intentionally not rewritten
    (benchmark freeze); the override below records the known upgrade.
    """
    pr761_upgrades = {
        "pr74-entity-zakladani-hb-with-real": "SIGNED_OTHER_DOCUMENT_CONFIRMED",
    }
    mismatches = []
    for c in load_dataset(PR74_DATASET):
        if c.environment != "fixture" or "synthetic-pool" not in (c.tags or []):
            continue
        pool = pr74_runner._build_synthetic_pool(c)
        coverage = ai_search._answer_state_coverage(c.question, pool)
        actual = coverage.verdict.value
        expected = pr761_upgrades.get(c.id, c.expected_state_verdict)
        if actual != expected:
            mismatches.append((c.id, expected, actual))
    assert mismatches == []


# ---------------------------------------------------------------------------
# Acceptance metrics
# ---------------------------------------------------------------------------

def acceptance_case(**overrides) -> BenchmarkCase:
    data = {
        "id": "acc", "question": "kde najdu kladečské výkresy výztuže?",
        "environment": "fixture", "category": "DOCUMENT_LOOKUP",
        "ground_truth_status": "verified",
    }
    data.update(overrides)
    return BenchmarkCase.from_dict(data)


def test_document_hit_reports_rank():
    rows = [{"document": "a.txt", "path": "/a.txt"}, {"document": "kladecke.txt", "path": "/kladecke.txt"}]
    found, rank = acceptance_metrics.document_hit(["kladecke"], rows)
    assert found and rank == 2


def test_retrieval_failure_is_attributed_to_retrieval():
    c = acceptance_case(expected_documents=["kladecke"])
    rows = [{"document": "jiny.txt", "path": "/jiny.txt"}]
    found, correct, _missing, used = acceptance_metrics.evaluate_acceptance_answer(
        c, rows, answer_dict("Odpověď.", ["jiny.txt"]),
    )
    layer, _detail = acceptance_metrics.classify_failure(c, found, correct, used)
    assert not found and layer == acceptance_metrics.LAYER_RETRIEVAL


def test_answer_failure_when_document_found_but_fact_missing():
    c = acceptance_case(expected_documents=["protokoly"], expected_answer_contains=["28"])
    rows = [{"document": "protokoly.txt", "path": "/protokoly.txt"}]
    found, correct, missing, used = acceptance_metrics.evaluate_acceptance_answer(
        c, rows, answer_dict("Zkoušky se provádějí.", ["protokoly.txt"]),
    )
    layer, _detail = acceptance_metrics.classify_failure(c, found, correct, used)
    assert found and not correct and missing == ["28"]
    assert layer == acceptance_metrics.LAYER_ANSWER


def test_wrong_answer_on_legal_case_is_a_critical_error():
    c = acceptance_case(
        category="CONTRACT_VERIFICATION", criticality="legal",
        expected_documents=["smlouva"], expected_answer_contains=["pokuty"],
    )
    rows = [{"document": "smlouva.txt", "path": "/smlouva.txt"}]
    found, correct, _missing, used = acceptance_metrics.evaluate_acceptance_answer(
        c, rows, answer_dict("Nevím.", ["smlouva.txt"]),
    )
    layer, _detail = acceptance_metrics.classify_failure(c, found, correct, used)
    assert acceptance_metrics.is_critical_error(c, correct, False) is True
    assert layer == acceptance_metrics.LAYER_SAFETY


def test_wrong_answer_on_informational_case_is_not_critical():
    c = acceptance_case(criticality="informational", expected_answer_contains=["28"])
    assert acceptance_metrics.is_critical_error(c, False, False) is False


def test_acceptance_verdict_is_inconclusive_on_fixture():
    agg = acceptance_metrics.aggregate_acceptance([
        acceptance_metrics.AcceptanceCaseResult(
            id="a", question="q", category="DOCUMENT_LOOKUP", environment="fixture",
            criticality="informational", ground_truth_status="verified",
            document_found=True, answer_correct=True, queries_to_answer=1, total_ms=10.0,
        ),
    ])
    verdict = acceptance_metrics.acceptance_verdict(agg, environment="fixture")
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert any("fake embedding" in r for r in verdict["inconclusive_reasons"])


def test_fixture_quality_failure_is_reported_but_not_a_verdict():
    """The mirror of a false GO: a fake embedding model must not be allowed to
    produce a NO-GO about the product either. The failure is still printed."""
    agg = acceptance_metrics.aggregate_acceptance([
        acceptance_metrics.AcceptanceCaseResult(
            id="bad", question="q", category="CONTRACT_VERIFICATION", environment="fixture",
            criticality="legal", ground_truth_status="verified",
            document_found=True, answer_correct=False, critical_error=True, total_ms=10.0,
        ),
    ])
    verdict = acceptance_metrics.acceptance_verdict(agg, environment="fixture")
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert any("critical_errors=1" in f for f in verdict["observed_failures"])
    assert any("NOT attributable" in r for r in verdict["inconclusive_reasons"])
    assert agg.critical_error_case_ids == ["bad"]


def test_critical_error_blocks_on_production():
    results = [
        acceptance_metrics.AcceptanceCaseResult(
            id=f"c{i}", question="q", category="DOCUMENT_LOOKUP", environment="production",
            criticality="informational", ground_truth_status="verified",
            document_found=True, answer_correct=True, queries_to_answer=1, total_ms=10.0,
        )
        for i in range(30)
    ]
    results[0].critical_error = True
    agg = acceptance_metrics.aggregate_acceptance(results)
    verdict = acceptance_metrics.acceptance_verdict(agg, environment="production")
    assert verdict["verdict"] == "NO-GO"
    assert any("critical_errors=1" in b for b in verdict["blockers"])


def test_harness_error_blocks_even_on_fixture():
    agg = acceptance_metrics.aggregate_acceptance([
        acceptance_metrics.AcceptanceCaseResult(
            id="boom", question="q", category="DOCUMENT_LOOKUP", environment="fixture",
            criticality="informational", ground_truth_status="verified",
            error="RuntimeError: boom",
        ),
    ])
    verdict = acceptance_metrics.acceptance_verdict(agg, environment="fixture")
    assert verdict["verdict"] == "NO-GO"
    assert any("errored_cases" in b for b in verdict["blockers"])


def test_acceptance_go_requires_verified_production_dataset():
    results = [
        acceptance_metrics.AcceptanceCaseResult(
            id=f"c{i}", question="q", category="DOCUMENT_LOOKUP", environment="production",
            criticality="informational", ground_truth_status="verified",
            # PR7.5: a found document always carries its rank, and top5_accuracy
            # is now a GO criterion - a result without a rank is a retrieval miss.
            document_found=True, document_rank=1,
            answer_correct=True, queries_to_answer=1, total_ms=10.0,
        )
        for i in range(30)
    ]
    agg = acceptance_metrics.aggregate_acceptance(results)
    assert acceptance_metrics.acceptance_verdict(agg, environment="production")["verdict"] == "GO"
    # One unverified case is enough to withdraw certification.
    results[0].ground_truth_status = "needs_review"
    agg = acceptance_metrics.aggregate_acceptance(results)
    assert acceptance_metrics.acceptance_verdict(agg, environment="production")["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Dataset + report
# ---------------------------------------------------------------------------

def test_acceptance_dataset_loads_and_separates_verified_from_template():
    cases = load_dataset(ACCEPTANCE_DATASET)
    assert len(cases) >= 20
    fixture = [c for c in cases if c.environment == "fixture"]
    production = [c for c in cases if c.environment == "production"]
    assert all(c.ground_truth_status == "verified" for c in fixture)
    assert all(c.ground_truth_status == "needs_review" for c in production)
    covered = {c.category for c in cases}
    assert covered == {
        "DOCUMENT_LOOKUP", "TECHNICAL_INFO", "CONTRACT_VERIFICATION",
        "MEETING_MINUTES", "TECHNICAL_PROCEDURE",
    }


def test_acceptance_categories_are_rejected_outside_the_enum():
    with pytest.raises(ValueError):
        BenchmarkCase.from_dict({"id": "x", "question": "q", "category": "NOT_A_CATEGORY"})
    with pytest.raises(ValueError):
        BenchmarkCase.from_dict({"id": "x", "question": "q", "criticality": "apocalyptic"})
    with pytest.raises(ValueError):
        BenchmarkCase.from_dict({"id": "x", "question": "q", "ground_truth_status": "probably"})


def test_pre_pr741_case_still_loads_unchanged():
    c = BenchmarkCase.from_dict({"id": "legacy", "question": "Pentaflex"})
    assert c.criticality == "informational"
    assert c.ground_truth_status == "unverified"
    assert c.follow_up_questions == []


def test_acceptance_report_states_the_required_numbers():
    artifact = {
        "timestamp": "2026-08-12T00:00:00+00:00",
        "git_sha": "abc",
        "environment": {"name": "fixture", "doc_count": 21, "chunk_count": 21},
        "dataset_file": "acceptance_v1.jsonl",
        "flags": {"DOCUMENT_STATE_GATE_ENABLED": True, "EVIDENCE_RUNTIME_VALIDATION_ENABLED": True, "llm_replay": False},
        "case_count": 1,
        "cases": [{
            "id": "a", "category": "DOCUMENT_LOOKUP", "criticality": "legal",
            "ground_truth_status": "verified", "document_found": True, "document_rank": 1,
            "answer_correct": True, "queries_to_answer": 1, "total_ms": 12.0,
            "failure_layer": "OK", "missing_phrases": [],
        }],
        "aggregate": {
            "case_count": 1, "evaluated_count": 1, "document_found_count": 1,
            "document_found_rate": 1.0, "answer_correct_count": 1, "answer_correct_rate": 1.0,
            "critical_error_count": 0, "critical_error_case_ids": [], "mean_total_ms": 12.0,
            "p95_total_ms": 12.0, "mean_queries_to_answer": 1.0,
            "resolved_within_one_query": 1, "resolved_with_follow_up": 0, "unresolved": 0,
            "verified_case_count": 1, "unverified_case_count": 0, "by_category": {}, "by_layer": {"OK": 1},
        },
        "verdict": {"verdict": "INCONCLUSIVE", "blockers": [], "inconclusive_reasons": ["fixture"], "criteria": {}},
        "warnings": ["Fixture environment uses a FAKE embedding model."],
    }
    markdown = report_mod.render_markdown_acceptance(artifact)
    assert "AI Search Acceptance Report" in markdown
    # PR7.5 renamed the summary rows to the KPI names the acceptance plan uses
    # and split the report into a FAT half and a SAT half.
    for label in (
        "počet testů", "document_found_rate", "answer_correct_rate",
        "počet kritických chyb", "průměrný čas odpovědi", "INCONCLUSIVE",
        "## FAT RESULT", "## SAT STATUS",
    ):
        assert label in markdown
    assert "FAKE embedding model" in markdown
    assert report_mod.render_csv_acceptance(artifact).startswith("id,category")


def test_pr74_report_separates_the_four_views():
    artifact = json.loads(json.dumps({
        "timestamp": "2026-08-12T00:00:00+00:00", "git_sha": "abc",
        "environment": {"name": "fixture", "doc_count": 1, "chunk_count": 1},
        "dataset_file": "pr74_answer_quality.jsonl", "case_count": 0, "llm_replay": True,
        "flags_constant": {"AUXILIARY_TERM_COVERAGE_ENABLED": False, "MULTI_QUERY_RETRIEVAL_ENABLED": False},
        "flag_matrix": {"A": {}, "B": {}, "C": {}, "D": {}},
        "cases": [], "aggregate": {"by_layer": {}, "first_failure_layer_counts": {}},
        "latency": {}, "go_nogo": {"verdict": "GO", "criteria": {}},
    }))
    markdown = report_mod.render_markdown_pr74(artifact)
    for heading in (
        "## Where it broke", "## 1. Retrieval quality", "## 2. Answer quality",
        "## 3. Safety", "## 4. Product usability",
    ):
        assert heading in markdown
    assert "acceptance-run" in markdown
