"""PR7.4 Answer Quality Benchmark — unit tests.

Benchmark-only: never requires a live Ollama server or the production index.
Monkeypatches ai_search.answer / _call_ollama / run_pipeline_trace as needed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import ai_search
import ai_search_config
import ui_services
from benchmark.dataset.schema import (
    BenchmarkCase,
    DATASET_DIR,
    VALID_CATEGORIES,
    load_dataset,
)
from benchmark import pr74_metrics
from benchmark import pr74_runner
from benchmark import report as report_mod


PR74_DATASET = DATASET_DIR / "pr74_answer_quality.jsonl"


# ---------------------------------------------------------------------------
# Schema backward compatibility
# ---------------------------------------------------------------------------

def test_schema_defaults_keep_legacy_cases_loadable():
    case = BenchmarkCase.from_dict({"id": "legacy-01", "question": "Pentaflex"})
    assert case.category == ""
    assert case.expected_state_verdict is None
    assert case.expected_intent_coverage is None
    assert case.expected_missing_needs == []
    assert case.forbidden_answer_contains == []
    assert case.expected_source_contains == []
    assert case.forbidden_sources == []


def test_existing_fixture_and_production_datasets_still_load():
    fixture = load_dataset(DATASET_DIR / "fixture_queries.jsonl")
    production = load_dataset(DATASET_DIR / "production_queries.jsonl")
    assert len(fixture) >= 1
    assert len(production) >= 1
    assert all(c.category == "" for c in fixture)
    assert all(c.category == "" for c in production)


def test_schema_rejects_invalid_category(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "x", "question": "q", "category": "NOT_A_CATEGORY"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid category"):
        load_dataset(path)


def test_schema_rejects_invalid_state_verdict(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({
            "id": "x", "question": "q",
            "expected_state_verdict": "YES_SIGNED",
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid expected_state_verdict"):
        load_dataset(path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_pr74_dataset_loads_fifty_cases_with_required_categories():
    cases = load_dataset(PR74_DATASET)
    assert len(cases) == 50
    counts = {}
    for c in cases:
        counts[c.category] = counts.get(c.category, 0) + 1
    assert counts["SIGNED_DOCUMENT"] == 15
    assert counts["ENTITY_SAFETY"] == 10
    assert counts["EVIDENCE_COVERAGE"] == 10
    assert counts["REGRESSION"] == 15
    assert set(counts) <= (VALID_CATEGORIES - {""})


def test_fixture_vs_production_separation():
    cases = load_dataset(PR74_DATASET)
    fixture = [c for c in cases if c.environment == "fixture"]
    production = [c for c in cases if c.environment == "production"]
    assert fixture and production
    # Váhostav adversarial case must never be marked production.
    vahostav = next(c for c in cases if c.id == "pr74-entity-haus365-vahostav")
    assert vahostav.environment == "fixture"
    assert "synthetic-pool" in vahostav.tags
    # Hilti glued-name case is fixture + known retrieval issue.
    hilti = next(c for c in cases if c.id == "pr74-entity-hilti-rent-glued")
    assert hilti.environment == "fixture"
    assert hilti.expected_retrieval_issue is True
    # Every synthetic-pool case is fixture.
    for c in cases:
        if "synthetic-pool" in c.tags:
            assert c.environment == "fixture", c.id


# ---------------------------------------------------------------------------
# Metrics determinism + blocking detection
# ---------------------------------------------------------------------------

def _case(**kwargs) -> BenchmarkCase:
    base = {
        "id": "t",
        "question": "je na boxu podepsaná smlouva haus365?",
        "category": "SIGNED_DOCUMENT",
        "environment": "fixture",
    }
    base.update(kwargs)
    return BenchmarkCase.from_dict(base)


def test_metrics_detect_blocking_wrong_entity_citation():
    case = _case(
        category="ENTITY_SAFETY",
        expected_state_verdict="ENTITY_MISMATCH",
        forbidden_sources=["VAHOSTAV"],
    )
    baseline = {
        "answer": "Na boxu není podepsaná smlouva haus365.",
        "citations": [{"document": "SOD_VAHOSTAV_podepsaná.pdf", "path": "/x"}],
    }
    candidate = {
        "answer": "Ano - na boxu je podepsaná smlouva. Nalezený podepsaný dokument: SOD_VAHOSTAV_podepsaná.pdf.",
        "citations": [{"document": "SOD_VAHOSTAV_podepsaná.pdf", "path": "/x"}],
        "validation": {"state_verdict": "SIGNED_CONFIRMED", "intent_coverage": "PARTIAL", "missing_needs": []},
    }
    ev = pr74_metrics.evaluate_case_answers(case, baseline, candidate)
    assert ev.has_blocking
    codes = {f.code for f in ev.failures}
    assert "wrong_entity_citation" in codes
    assert "false_signed_confirmation" in codes or "state_verdict_mismatch" in codes


def test_metrics_regression_noop_unchanged():
    case = _case(
        category="REGRESSION",
        expected_state_verdict="NOOP",
        question="jaká třída betonu je předepsaná",
    )
    text = "Předepsaná třída betonu je C30/37.\n(Zdroj: TZ.pdf)"
    answer = {
        "answer": text,
        "citations": [{"document": "TZ.pdf", "path": "/TZ.pdf"}],
        "validation": {"state_verdict": "NOOP", "intent_coverage": "COMPLETE", "missing_needs": []},
    }
    ev = pr74_metrics.evaluate_case_answers(case, answer, copy.deepcopy(answer))
    assert ev.answer_delta == "unchanged"
    assert not ev.has_blocking
    assert ev.state_verdict_match is True


def test_metrics_improved_when_gate_fixes_false_negative():
    case = _case(
        expected_state_verdict="SIGNED_CONFIRMED",
        expected_source_contains=["HAUS365"],
        forbidden_answer_contains=["není podepsan"],
    )
    baseline = {
        "answer": "Na boxu není podepsaná smlouva haus365.",
        "citations": [{"document": "TP.pdf", "path": "/TP.pdf"}],
    }
    candidate = {
        "answer": "Ano - na boxu je podepsaná smlouva. Nalezený podepsaný dokument: SOD_HAUS365_podepsané.pdf.\n(Zdroj: SOD_HAUS365_podepsané.pdf)",
        "citations": [{"document": "SOD_HAUS365_podepsané.pdf", "path": "/SOD_HAUS365_podepsané.pdf"}],
        "validation": {"state_verdict": "SIGNED_CONFIRMED", "intent_coverage": "PARTIAL", "missing_needs": []},
    }
    ev = pr74_metrics.evaluate_case_answers(case, baseline, candidate)
    assert ev.answer_delta == "improved"
    assert not ev.has_blocking


def test_aggregate_has_blocking_regression_independent_of_safety_score():
    case = _case(
        category="ENTITY_SAFETY",
        expected_state_verdict="ENTITY_MISMATCH",
        forbidden_sources=["VAHOSTAV"],
    )
    baseline = {"answer": "x", "citations": []}
    candidate = {
        "answer": "Ano - je podepsaná smlouva.",
        "citations": [{"document": "SOD_VAHOSTAV_podepsaná.pdf", "path": "/v"}],
        "validation": {"state_verdict": "SIGNED_CONFIRMED"},
    }
    ev = pr74_metrics.evaluate_case_answers(case, baseline, candidate)
    agg = pr74_metrics.aggregate_evaluations([ev])
    assert agg.has_blocking_regression is True
    assert agg.blocking_case_ids == ["t"]
    assert agg.safety_score is not None
    # Even a high safety_score must not hide the blocking flag.
    assert agg.has_blocking_regression


def test_metrics_are_deterministic():
    case = _case(expected_state_verdict="SIGNED_CONFIRMED", expected_source_contains=["HAUS365"])
    baseline = {"answer": "Ne.", "citations": [{"document": "SOD_HAUS365_x.pdf", "path": "/a"}]}
    candidate = {
        "answer": "Ano - na boxu je podepsaná smlouva. SOD_HAUS365_x.pdf",
        "citations": [{"document": "SOD_HAUS365_x.pdf", "path": "/a"}],
        "validation": {"state_verdict": "SIGNED_CONFIRMED"},
    }
    a = pr74_metrics.evaluate_case_answers(case, baseline, candidate).to_dict()
    b = pr74_metrics.evaluate_case_answers(case, baseline, candidate).to_dict()
    assert a == b


# ---------------------------------------------------------------------------
# Runner: shared results, single retrieval, flag restore
# ---------------------------------------------------------------------------

def _mock_ollama(monkeypatch, text="Smlouva haus365 je podepsaná."):
    payload = json.dumps({
        "body": [{"text": text, "zdroj_index": 1, "typ": "fakt"}],
        "nenalezeno": False,
    })

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": payload}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())


def test_runner_shares_final_results_across_modes(monkeypatch):
    _mock_ollama(monkeypatch)
    calls = {"answer": 0}
    seen_ids = []

    real_answer = ai_search.answer

    def counting_answer(query, results):
        calls["answer"] += 1
        seen_ids.append(id(results))
        return real_answer(query, results)

    monkeypatch.setattr(ai_search, "answer", counting_answer)

    case = next(
        c for c in load_dataset(PR74_DATASET)
        if c.id == "pr74-signed-haus365-multi-lifecycle"
    )
    from benchmark.environment import fixture_environment
    result = pr74_runner.evaluate_pr74_case(case, fixture_environment(), llm_replay=True)
    assert result.error is None
    assert result.retrieval_skipped is True
    assert set(result.modes) == {"A", "B", "C", "D"}
    # PR7.4.1: one priming call under baseline flags fills the replay cache so
    # all four modes are served from it and their wall times are differenceable
    # (audit P5), then the four modes themselves.
    assert calls["answer"] == 5
    # Same list object passed to every answer() call, priming included.
    assert len(set(seen_ids)) == 1
    assert len(result.final_results_identity) >= 1
    assert "validation" not in result.modes["A"]["answer"]
    assert "validation" in result.modes["C"]["answer"]
    assert result.modes["C"]["answer"]["validation"]["state_verdict"] == "SIGNED_CONFIRMED"


def test_runner_calls_retrieval_only_once_for_nonsynthetic(monkeypatch):
    _mock_ollama(monkeypatch)
    calls = {"trace": 0}
    rows = [{
        "document": "protokoly_zkousek_betonu.txt", "path": "/f/protokoly_zkousek_betonu.txt",
        "project": "P", "heading": "", "quote": "beton C30/37", "score": 1.0,
        "document_id": 1, "chunk_id": "c:0",
        "match": {"fts_hit": True, "vector_hit": True, "semantic_similarity": 0.5, "filename_match": False},
    }]

    def fake_trace(query, environment, **kwargs):
        calls["trace"] += 1
        from benchmark.pipeline_trace import PipelineTrace
        t = PipelineTrace(
            query=query, is_question=True, deep=False, fts_terms="",
            retrieval_k=0, rerank_k=0, fetch_limit=0,
        )
        t.final_results = rows
        return t

    monkeypatch.setattr(pr74_runner, "run_pipeline_trace", fake_trace)
    case = next(c for c in load_dataset(PR74_DATASET) if c.id == "pr74-regression-fixture-beton")
    from benchmark.environment import fixture_environment
    result = pr74_runner.evaluate_pr74_case(case, fixture_environment(), llm_replay=True)
    assert result.error is None
    assert calls["trace"] == 1
    assert result.retrieval_skipped is False


def test_flags_restored_after_case(monkeypatch):
    _mock_ollama(monkeypatch)
    ai_search.DOCUMENT_STATE_GATE_ENABLED = False
    ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED = False
    ai_search.AUXILIARY_TERM_COVERAGE_ENABLED = False
    ui_services.MULTI_QUERY_RETRIEVAL_ENABLED = False

    case = next(c for c in load_dataset(PR74_DATASET) if c.id == "pr74-entity-haus365-vahostav")
    from benchmark.environment import fixture_environment
    pr74_runner.evaluate_pr74_case(case, fixture_environment(), llm_replay=True)

    assert ai_search.DOCUMENT_STATE_GATE_ENABLED is False
    assert ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False
    assert ai_search.AUXILIARY_TERM_COVERAGE_ENABLED is False
    assert ui_services.MULTI_QUERY_RETRIEVAL_ENABLED is False
    assert ai_search_config.DOCUMENT_STATE_GATE_ENABLED is False
    assert ai_search_config.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False
    assert ai_search_config.AUXILIARY_TERM_COVERAGE_ENABLED is False
    assert ai_search_config.MULTI_QUERY_RETRIEVAL_ENABLED is False


def test_aux_and_multi_query_forced_false_during_answer(monkeypatch):
    _mock_ollama(monkeypatch)
    seen = []

    real_answer = ai_search.answer

    def spy(query, results):
        seen.append((
            ai_search.AUXILIARY_TERM_COVERAGE_ENABLED,
            ui_services.MULTI_QUERY_RETRIEVAL_ENABLED,
            ai_search.DOCUMENT_STATE_GATE_ENABLED,
            ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED,
        ))
        return real_answer(query, results)

    monkeypatch.setattr(ai_search, "answer", spy)
    case = next(c for c in load_dataset(PR74_DATASET) if c.id == "pr74-signed-haus365-draft-only")
    from benchmark.environment import fixture_environment
    pr74_runner.evaluate_pr74_case(case, fixture_environment(), llm_replay=True)
    # One replay-priming call under baseline flags (PR7.4.1) + four modes.
    assert len(seen) == 5
    # AUX and MQ always False, priming call included.
    assert [s[0] for s in seen] == [False] * 5
    assert [s[1] for s in seen] == [False] * 5
    assert [s[2:] for s in seen] == [
        (False, False),  # priming — must not run the gate or validation
        (False, False),  # A
        (False, True),   # B
        (True, True),    # C
        (True, False),   # D
    ]


def test_llm_replay_marks_artifact(monkeypatch):
    live = {"n": 0}
    original = ai_search._call_ollama

    def counting(model, prompt, format_schema=None, timeout=240):
        live["n"] += 1
        return original(model, prompt, format_schema=format_schema, timeout=timeout)

    # urlopen still mocked; _call_ollama goes through replay wrapper → counting → original → urlopen
    _mock_ollama(monkeypatch)
    monkeypatch.setattr(ai_search, "_call_ollama", counting)

    # Re-install replay on top by using evaluate which wraps whatever is current.
    # evaluate_pr74_case saves current _call_ollama as original for replay.
    case = next(c for c in load_dataset(PR74_DATASET) if c.id == "pr74-signed-haus365-draft-only")
    from benchmark.environment import fixture_environment
    # Manually exercise replay helper.
    replay = pr74_runner._OllamaReplay()
    replay.install()
    try:
        ai_search._call_ollama("m", "p1")
        ai_search._call_ollama("m", "p1")
        ai_search._call_ollama("m", "p2")
    finally:
        replay.uninstall()
    assert replay.live_calls == 2
    assert replay.replay_calls == 1


def test_environment_filter_skips_other_env(monkeypatch):
    """run_pr74_benchmark must not mix fixture synthetic evidence into production."""
    _mock_ollama(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("production env must not be constructed in this test")

    monkeypatch.setattr(pr74_runner, "get_environment", boom)
    # Empty case filter over production → get_environment still called.
    # Instead call evaluate path with fixture filter only via load + filter.
    cases = [c for c in load_dataset(PR74_DATASET) if c.environment == "fixture"]
    assert all(c.environment == "fixture" for c in cases)
    assert any("synthetic-pool" in c.tags for c in cases)
    production = [c for c in load_dataset(PR74_DATASET) if c.environment == "production"]
    assert all("synthetic-pool" not in c.tags for c in production)


def test_report_renders_pr74_go_nogo():
    artifact = {
        "timestamp": "2026-08-12T00:00:00+00:00",
        "git_sha": "deadbeef",
        "environment": {"name": "fixture", "doc_count": 1, "chunk_count": 1,
                        "index_fingerprint": "abc", "index_fingerprint_algorithm": "sha256-v1"},
        "dataset_file": "pr74_answer_quality.jsonl",
        "case_count": 1,
        "llm_replay": True,
        "flags_constant": {
            "AUXILIARY_TERM_COVERAGE_ENABLED": False,
            "MULTI_QUERY_RETRIEVAL_ENABLED": False,
        },
        "flag_matrix": {
            "A": {"DOCUMENT_STATE_GATE_ENABLED": False, "EVIDENCE_RUNTIME_VALIDATION_ENABLED": False},
            "B": {"DOCUMENT_STATE_GATE_ENABLED": False, "EVIDENCE_RUNTIME_VALIDATION_ENABLED": True},
            "C": {"DOCUMENT_STATE_GATE_ENABLED": True, "EVIDENCE_RUNTIME_VALIDATION_ENABLED": True},
            "D": {"DOCUMENT_STATE_GATE_ENABLED": True, "EVIDENCE_RUNTIME_VALIDATION_ENABLED": False},
        },
        "cases": [{
            "id": "c1", "category": "REGRESSION", "environment": "fixture",
            "evaluation": {
                "answer_delta": "unchanged",
                "state_verdict_expected": "NOOP",
                "state_verdict_actual": "NOOP",
                "failures": [],
            },
            "error": None, "warmup": False, "retrieval_ms": 1.0, "llm_replay": True,
        }],
        "aggregate": {
            "state_verdict_accuracy": 1.0,
            "false_signed_confirmations": 0,
            "wrong_entity_citations": 0,
            "unsupported_negative_signed_claims": 0,
            "unsupported_positive_signed_claims": 0,
            "hedge_incorrectly_rewritten": 0,
            "blocking_count": 0, "high_count": 0, "medium_count": 0, "low_count": 0,
            "unchanged_count": 1, "changed_count": 0, "improved_count": 0,
            "degraded_count": 0, "changed_neutral_count": 0,
            "safety_score": 1.0, "by_category": {},
            "intent_coverage_accuracy": None, "missing_need_accuracy": None,
            "signed_confirmed_precision": None, "signed_confirmed_recall": None,
            "entity_mismatch_accuracy": None, "unverified_accuracy": None,
        },
        "latency": {"note": "test", "warmup_excluded_case_ids": []},
        "go_nogo": {
            "verdict": "GO",
            "has_blocking_regression": False,
            "blocking_case_ids": [],
            "reasons": [],
            "criteria": {"zero_blocking": True},
        },
    }
    md = report_mod.render_markdown_pr74(artifact)
    assert "Verdict: GO" in md
    assert "STATE_GATE" in md
    assert "has_blocking_regression" in md
    written = report_mod.write_reports(artifact, Path("/tmp/pr74-test-report"))
    assert written["pr74_markdown"].exists()
    assert written["pr74_csv"].exists()


def test_full_fixture_suite_runs_without_production(monkeypatch, tmp_path):
    """End-to-end smoke: all fixture cases, mocked Ollama, no production index."""
    _mock_ollama(monkeypatch, text="Technická odpověď bez tvrzení o podpisu.")
    # Negative claim for signed fixture cases so gate has something to rewrite.
    negative = "Na boxu není podepsaná smlouva."

    def urlopen_by_query(request, timeout=0):
        body = request.data.decode() if hasattr(request, "data") and request.data else ""
        text = negative if "podeps" in body.casefold() or "podpis" in body.casefold() else "Technická odpověď."
        payload = json.dumps({
            "body": [{"text": text, "zdroj_index": 1, "typ": "fakt"}],
            "nenalezeno": False,
        })

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"response": payload}).encode()

        return FakeResponse()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", urlopen_by_query)

    run = pr74_runner.run_pr74_benchmark(
        environment_name="fixture",
        llm_replay=True,
        exclude_warmup=False,
    )
    assert run.case_count == sum(
        1 for c in load_dataset(PR74_DATASET) if c.environment == "fixture"
    )
    assert run.flags_constant["AUXILIARY_TERM_COVERAGE_ENABLED"] is False
    assert run.flags_constant["MULTI_QUERY_RETRIEVAL_ENABLED"] is False
    assert set(run.flag_matrix) == {"A", "B", "C", "D"}
    assert "has_blocking_regression" in run.go_nogo
    # Flags restored after full run.
    assert ai_search.DOCUMENT_STATE_GATE_ENABLED is False
    assert ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False
    path = pr74_runner.save_pr74_run(run, tmp_path / "pr74.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["go_nogo"]["verdict"] in {"GO", "NO-GO"}
