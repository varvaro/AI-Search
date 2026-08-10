"""Fast, CI-safe tests for the retrieval regression SUITE (dataset contents,
known-limitation semantics, baseline comparison) - not a measure of retrieval
quality, which needs the real index (`python -m benchmark.run_retrieval_regression`).

Nothing here opens SQLite/LanceDB or loads an embedding model: the one test
that exercises evaluate_case() stubs ui_services.search_all(), because the
point is the bookkeeping around the call, not the call itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import ui_services
from benchmark import run_retrieval_regression as suite
from benchmark.dataset.schema import BenchmarkCase, load_dataset
from benchmark.environment import Environment

DATASET = suite.DATASET_PATH
EXPECTED_CASE_COUNT = 22
# Cases added to the dataset as tripwires before the next baseline refresh
# (user-requested: do not update baseline in the same change).
# test_checked_in_baseline_matches_the_dataset accounts for this gap explicitly.
PENDING_BASELINE_CASE_IDS = frozenset({
    "rr-bp-vyvody-3pp-01",
    "rr-brokovani-zakladova-deska-3pp-01",
})


@pytest.fixture(scope="module")
def cases() -> list[BenchmarkCase]:
    return load_dataset(DATASET)


# --- dataset loading and structure ------------------------------------------

def test_dataset_file_is_checked_in_and_loads(cases):
    """The whole point of the exercise: the suite lives in the repo, not in
    /tmp, so it survives a reboot and can be diffed in review."""
    assert DATASET.exists()
    assert len(cases) == EXPECTED_CASE_COUNT


def test_every_case_has_a_query_and_an_id(cases):
    for case in cases:
        assert case.id.strip(), f"case at line {case.line_number} has no id"
        assert case.question.strip(), f"case {case.id} has no question"


def test_every_case_is_a_production_lookup_with_domain_and_ground_truth(cases):
    """A case with no expected_documents can never fail, so it would be dead
    weight that still costs a real query against the index."""
    for case in cases:
        assert case.environment == "production", case.id
        assert case.domain.strip(), f"case {case.id} has no domain"
        assert case.expected_documents, f"case {case.id} has no expected_documents"
        assert case.notes.strip(), f"case {case.id} has no notes on its ground-truth limitations"


def test_case_ids_are_unique_and_prefixed(cases):
    ids = [case.id for case in cases]
    assert len(set(ids)) == len(ids)
    assert all(case_id.startswith("rr-") for case_id in ids)


def test_corrected_ground_truth_is_the_one_actually_shipped(cases):
    """Guards the two 2026-08-08 ground-truth repairs specifically, because
    both were wrong in a way that still produced plausible-looking numbers.

    'Zmenovy_list' pointed at individual change lists instead of the overview
    the query asks for and manufactured a false regression across the reindex;
    'VT 11_HAUS365' pointed at a quantity spreadsheet instead of the
    technological procedure.
    """
    by_id = {case.id: case for case in cases}
    prehled = by_id["rr-zl-prehled-gd-01"]
    assert prehled.expected_documents == ["přehled ZL GD"]
    assert not any("Zmenovy_list" in needle for needle in prehled.expected_documents)
    assert by_id["rr-haus365-tp-monolit-01"].expected_documents == ["TP Smíchov monolit"]
    assert not any("VT 11" in needle for needle in by_id["rr-haus365-tp-monolit-01"].expected_documents)


def test_the_known_limitations_are_flagged(cases):
    by_id = {case.id: case for case in cases}
    assert by_id["rr-haus365-kladecsky-plan-01"].expected_content_missing is True
    assert by_id["rr-kzp-monolit-feri-01"].expected_retrieval_issue is True
    assert by_id["rr-bp-vyvody-3pp-01"].expected_retrieval_issue is True
    # brokování tripwire is a normal blocking case after the QE surface-prep fix
    assert by_id["rr-brokovani-zakladova-deska-3pp-01"].expected_retrieval_issue is False
    flagged = [c.id for c in cases if c.expected_content_missing or c.expected_retrieval_issue]
    assert sorted(flagged) == [
        "rr-bp-vyvody-3pp-01",
        "rr-haus365-kladecsky-plan-01",
        "rr-kzp-monolit-feri-01",
    ]


# --- schema: new fields are additive ----------------------------------------

def test_new_fields_default_off_so_existing_datasets_are_unchanged():
    case = BenchmarkCase.from_dict({"id": "x", "question": "q"})
    assert case.domain == ""
    assert case.expected_content_missing is False
    assert case.expected_retrieval_issue is False


def test_existing_datasets_still_load_after_the_schema_change():
    for name in ("production_queries.jsonl", "fixture_queries.jsonl"):
        assert load_dataset(DATASET.parent / name)


def test_a_case_cannot_claim_both_missing_content_and_a_retrieval_issue():
    """The two flags are competing diagnoses of the same symptom; allowing both
    would make it impossible to tell which one the case is asserting."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        BenchmarkCase.from_dict({"id": "x", "question": "q",
                                 "expected_content_missing": True, "expected_retrieval_issue": True})


# --- aggregation: known limitations must not poison the headline numbers ----

def _case(case_id, *, hit=True, rank=0, recall=1.0, rr=1.0, ndcg=1.0,
          content_missing=False, retrieval_issue=False, error="", latency=100.0):
    return {"id": case_id, "domain": "d", "question": case_id, "type": "lookup", "difficulty": "medium",
            "expected_documents": ["x"], "expected_content_missing": content_missing,
            "expected_retrieval_issue": retrieval_issue, "notes": "", "is_question": False,
            "error": error, "returned": 10, "best_rank": rank, "hit": hit, "recall_at_k": recall,
            "reciprocal_rank": rr, "ndcg_at_k": ndcg, "latency_ms": latency, "top_paths": []}


def test_expected_content_missing_is_not_counted_as_a_retrieval_fail():
    """A permanently-red case teaches people to ignore the report. Its 0.0 must
    stay out of both the miss count and the means."""
    aggregate = suite._aggregate([_case("ok"), _case("gone", hit=False, rank=None, recall=0.0, rr=0.0,
                                                     ndcg=0.0, content_missing=True)], 10)
    assert aggregate["case_count"] == 2
    assert aggregate["scored_case_count"] == 1
    assert aggregate["content_missing_count"] == 1
    assert aggregate["miss_count"] == 0
    assert aggregate["hit_count"] == 1
    assert aggregate["mean_recall_at_10"] == 1.0
    assert aggregate["mrr"] == 1.0


def test_expected_retrieval_issue_still_counts_toward_the_metrics():
    """Opposite of the case above: the content IS indexed, so the bad score is
    real and must keep dragging the mean until the weakness is fixed -
    otherwise fixing it would show no improvement."""
    aggregate = suite._aggregate([_case("ok"), _case("known", hit=False, rank=None, recall=0.0, rr=0.0,
                                                     ndcg=0.0, retrieval_issue=True)], 10)
    assert aggregate["scored_case_count"] == 2
    assert aggregate["miss_count"] == 1
    assert aggregate["known_retrieval_issue_count"] == 1
    assert aggregate["mean_recall_at_10"] == 0.5


# --- baseline comparison ----------------------------------------------------

def _run(cases_list):
    """A minimal but COMPLETE run artifact - complete enough that
    render_console() can format it, so a test can assert on the report text."""
    return {"timestamp": "t", "k": 10, "dataset": "test.jsonl", "query_expansion_mode": "fts",
            "environment": {"index_fingerprint": "fp", "doc_count": 1, "chunk_count": 1,
                            "embedding_model": "fake"},
            "aggregate": suite._aggregate(cases_list, 10), "cases": cases_list}


def test_hit_to_miss_is_a_blocking_regression():
    comparison = suite.compare_to_baseline(_run([_case("a")]),
                                           _run([_case("a", hit=False, rank=None, recall=0.0, rr=0.0, ndcg=0.0)]))
    assert comparison["has_regression"] is True
    assert comparison["blocking_regression_ids"] == ["a"]
    assert "HIT -> MISS" in comparison["regressed"][0]["reasons"]


def test_a_worse_rank_and_lower_mrr_are_reported_as_regression():
    comparison = suite.compare_to_baseline(_run([_case("a", rank=0, rr=1.0, ndcg=1.0)]),
                                           _run([_case("a", rank=4, rr=0.2, ndcg=0.43)]))
    assert comparison["has_regression"] is True
    reasons = comparison["regressed"][0]["reasons"]
    assert "rank #1 -> #5" in reasons
    assert any(r.startswith("MRR") for r in reasons)
    assert any(r.startswith("nDCG") for r in reasons)


def test_a_better_rank_is_reported_as_improved():
    comparison = suite.compare_to_baseline(_run([_case("a", rank=4, rr=0.2, ndcg=0.43)]),
                                           _run([_case("a", rank=0, rr=1.0, ndcg=1.0)]))
    assert comparison["has_regression"] is False
    assert comparison["improved"][0]["id"] == "a"
    assert "rank #5 -> #1" in comparison["improved"][0]["reasons"]


def test_expected_retrieval_issue_is_reported_but_never_fails_the_run():
    """The KZP vocabulary mismatch is diagnosed and accepted. It must show up
    in the report on every run, and it must not turn the run red."""
    baseline = _run([_case("known", retrieval_issue=True)])
    current = _run([_case("known", hit=False, rank=None, recall=0.0, rr=0.0, ndcg=0.0, retrieval_issue=True)])
    comparison = suite.compare_to_baseline(baseline, current)
    assert comparison["regressed"][0]["id"] == "known"
    assert comparison["has_regression"] is False
    assert comparison["blocking_regression_ids"] == []
    assert comparison["known_limitation_cases"][0]["flag"] == "retrieval_issue"


def test_a_missing_document_appearing_is_surfaced_as_improvement():
    """The tripwire: rr-haus365-kladecsky-plan-01 exists to notice the day the
    file is finally filed."""
    baseline = _run([_case("gone", hit=False, rank=None, recall=0.0, rr=0.0, ndcg=0.0, content_missing=True)])
    current = _run([_case("gone", content_missing=True)])
    comparison = suite.compare_to_baseline(baseline, current)
    assert comparison["improved"][0]["id"] == "gone"
    assert "MISS -> HIT" in comparison["improved"][0]["reasons"]


def test_a_newly_errored_case_is_a_regression():
    comparison = suite.compare_to_baseline(_run([_case("a")]), _run([_case("a", error="RuntimeError: boom")]))
    assert comparison["has_regression"] is True
    assert any("newly errored" in r for r in comparison["regressed"][0]["reasons"])


def test_added_and_removed_cases_do_not_masquerade_as_movement():
    comparison = suite.compare_to_baseline(_run([_case("a"), _case("dropped")]), _run([_case("a"), _case("added")]))
    assert comparison["removed_case_ids"] == ["dropped"]
    assert {"id": "added", "verdict": "NEW", "detail": "not present in baseline"} in comparison["unchanged"]
    assert comparison["has_regression"] is False


def test_comparison_flags_a_changed_query_expansion_mode():
    """A baseline measured under a different expansion branch is not comparable
    - and unlike an index change, nothing in the per-case numbers reveals it."""
    baseline = _run([_case("a")]) | {"query_expansion_mode": None}
    current = _run([_case("a")])
    comparison = suite.compare_to_baseline(baseline, current)
    assert comparison["same_query_expansion_mode"] is False
    assert "query expansion mode changed" in suite.render_console(current, comparison)


def test_comparison_flags_a_different_index():
    baseline = _run([_case("a")])
    current = _run([_case("a")])
    current["environment"] = {"index_fingerprint": "other"}
    assert suite.compare_to_baseline(baseline, current)["same_index"] is False


# --- evaluate_case and the checked-in baseline ------------------------------

def test_evaluate_case_records_rank_and_never_raises_on_a_failing_search(monkeypatch):
    """A broken query must not abort the other 19 cases mid-suite."""
    environment = Environment(name="stub", db_path=Path("/nonexistent"), lance_dir=Path("/nonexistent"),
                              embeddings=object())
    monkeypatch.setattr(ui_services, "search_all",
                        lambda *a, **kw: [{"path": "/x/other.pdf"}, {"path": "/x/TP Smíchov monolit.doc"}])
    result = suite.evaluate_case(BenchmarkCase.from_dict(
        {"id": "t", "question": "technologický postup", "expected_documents": ["TP Smíchov monolit"]}), environment)
    assert result["best_rank"] == 1 and result["hit"] is True and result["error"] == ""

    def boom(*a, **kw):
        raise RuntimeError("index locked")
    monkeypatch.setattr(ui_services, "search_all", boom)
    failed = suite.evaluate_case(BenchmarkCase.from_dict(
        {"id": "t", "question": "q", "expected_documents": ["x"]}), environment)
    assert failed["error"].startswith("RuntimeError") and failed["hit"] is False


def test_the_suite_measures_the_same_expansion_branch_the_ui_runs(monkeypatch):
    """If the suite hardcoded its own expansion setting, it would silently stop
    measuring production the moment the app's setting changed - which is the
    one thing this suite exists to prevent."""
    import ai_search_config
    seen = {}
    environment = Environment(name="stub", db_path=Path("/nonexistent"), lance_dir=Path("/nonexistent"),
                              embeddings=object())
    monkeypatch.setattr(ui_services, "search_all", lambda *a, **k: (seen.update(k), [])[1])
    monkeypatch.setattr(ai_search_config, "QUERY_EXPANSION_MODE", "sentinel")
    suite.evaluate_case(BenchmarkCase.from_dict({"id": "t", "question": "q", "expected_documents": ["x"]}), environment)
    assert seen["expand_query"] == "sentinel"


def test_run_artifact_records_the_expansion_mode_it_was_measured_under():
    import ai_search_config
    assert ai_search_config.QUERY_EXPANSION_MODE == "fts"
    baseline = json.loads(suite.BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["query_expansion_mode"] == ai_search_config.QUERY_EXPANSION_MODE


def test_checked_in_baseline_matches_the_dataset(cases):
    """Catches a dataset edited without refreshing the baseline, which would
    otherwise show up as a pile of NEW/removed entries in the next report.

    PENDING_BASELINE_CASE_IDS are intentionally allowed as the sole gap: the
    case is already in the dataset (so the tripwire is live) while the
    committed baseline has not been refreshed yet."""
    baseline = json.loads(suite.BASELINE_PATH.read_text(encoding="utf-8"))
    dataset_ids = {c.id for c in cases}
    baseline_ids = {c["id"] for c in baseline["cases"]}
    assert baseline_ids == dataset_ids - PENDING_BASELINE_CASE_IDS
    assert dataset_ids - baseline_ids == PENDING_BASELINE_CASE_IDS
    assert baseline["environment"]["index_fingerprint"]
    assert baseline["aggregate"]["case_count"] == EXPECTED_CASE_COUNT - len(PENDING_BASELINE_CASE_IDS)


def test_the_committed_baseline_carries_no_real_document_paths():
    """benchmark/runs/*.json is gitignored because production artifacts embed
    real Box paths. The baseline is the one artifact that must be committed, so
    it has to be stripped - and the comparison never needs the paths anyway."""
    baseline = json.loads(suite.BASELINE_PATH.read_text(encoding="utf-8"))
    assert all("top_paths" not in case for case in baseline["cases"])
    assert "/Users/" not in json.dumps(baseline["cases"], ensure_ascii=False)


def test_baseline_payload_strips_paths_but_keeps_the_comparison_inputs():
    run = {"cases": [_case("a") | {"top_paths": ["/Users/x/secret.pdf"]}]}
    stripped = suite.baseline_payload(run)
    assert "top_paths" not in stripped["cases"][0]
    for key in ("best_rank", "hit", "recall_at_k", "reciprocal_rank", "ndcg_at_k", "error"):
        assert key in stripped["cases"][0]
    assert "top_paths" in run["cases"][0], "must not mutate the run artifact it was given"
