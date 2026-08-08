"""Fast, CI-safe tests for the benchmark FRAMEWORK itself (metrics math,
dataset schema, end-to-end fixture-environment smoke test) - not a measure of
real retrieval quality (see benchmark/dataset/production_queries.jsonl and
`python -m benchmark run --environment production` for that).

These deliberately do not touch ai_search.py's retrieval logic; they only
exercise the benchmark package added alongside it.
"""
from __future__ import annotations

import json

import pytest

import ai_search
from benchmark import metrics, report
from benchmark.compare import LEGACY_FINGERPRINT_ALGORITHM, EnvironmentMismatchError, compare_runs
from benchmark.consistency_check import check_consistency
from benchmark.dataset.chunk_resolution import resolve_expected_chunks
from benchmark.dataset.schema import BenchmarkCase, ExpectedChunk, load_dataset, load_datasets
from benchmark.environment import FINGERPRINT_ALGORITHM, fixture_environment, index_identity
from benchmark.pipeline_trace import run_pipeline_trace
from benchmark.runner import _latency_stats, _percentile, evaluate_case, run_benchmark

# --- metrics.py: pure-function unit tests -----------------------------------

ROWS = [
    {"document": "a.pdf", "path": "/x/a.pdf", "quote": "obsahuje pentaflex a beton"},
    {"document": "b.pdf", "path": "/x/b.pdf", "quote": "smlouva o dilo"},
    {"document": "c.pdf", "path": "/y/FERI/c.pdf", "quote": "geodeticky protokol zakladove desky"},
    {"document": "d.pdf", "path": "/x/d.pdf", "quote": "dalsi neralevantni text"},
]


def test_row_matches_any_is_case_and_diacritics_insensitive():
    assert metrics.row_matches_any(ROWS[2], ["feri"])
    assert metrics.row_matches_any(ROWS[2], ["FERI"])
    assert not metrics.row_matches_any(ROWS[0], ["feri"])


def test_best_rank_returns_first_matching_index_or_none():
    assert metrics.best_rank(ROWS, ["FERI"]) == 2
    assert metrics.best_rank(ROWS, ["nonexistent-xyz"]) is None


def test_recall_at_k_counts_distinct_expected_documents_found():
    assert metrics.recall_at_k(ROWS, ["FERI"], k=10) == 1.0
    assert metrics.recall_at_k(ROWS, ["FERI", "nonexistent-xyz"], k=10) == 0.5
    assert metrics.recall_at_k(ROWS, [], k=10) == 1.0


def test_recall_at_k_respects_k_cutoff():
    assert metrics.recall_at_k(ROWS, ["FERI"], k=2) == 0.0
    assert metrics.recall_at_k(ROWS, ["FERI"], k=3) == 1.0


def test_forbidden_free_rate_flags_forbidden_content():
    # Renamed precision_at_k -> forbidden_rate -> forbidden_free_rate; the
    # formula never changed. 1.0 means "no forbidden content in the top-k",
    # which is why the middle name was inverted. See its docstring.
    assert metrics.forbidden_free_rate(ROWS, ["FERI"], [], k=10) == 1.0
    assert metrics.forbidden_free_rate(ROWS, ["FERI"], ["b.pdf"], k=10) < 1.0


def test_hit_rate_and_reciprocal_rank():
    assert metrics.hit_rate(ROWS, ["FERI"], k=10) == 1.0
    assert metrics.hit_rate(ROWS, ["nonexistent-xyz"], k=10) == 0.0
    assert metrics.reciprocal_rank(ROWS, ["FERI"]) == pytest.approx(1 / 3)
    assert metrics.mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_ndcg_at_k_binary_relevance_is_bounded_and_perfect_when_first():
    assert metrics.ndcg_at_k(ROWS, ["a.pdf"], k=10) == pytest.approx(1.0)
    ndcg_late = metrics.ndcg_at_k(ROWS, ["FERI"], k=10)
    assert 0.0 < ndcg_late < 1.0


def test_channel_agreement_buckets_correctly():
    bm25_rows = [ROWS[0], ROWS[2]]
    vector_rows = [ROWS[2], ROWS[3]]
    agreement = metrics.channel_agreement(bm25_rows, vector_rows, ["a.pdf", "FERI", "nonexistent-xyz"])
    assert agreement == {"both": 1, "bm25_only": 1, "vector_only": 0, "neither": 1}


def test_duplicate_count_and_distinct_ratio():
    rows = [{"document": "a"}, {"document": "a"}, {"document": "b"}]
    assert metrics.duplicate_count(rows) == 1
    assert metrics.distinct_ratio(rows) == pytest.approx(2 / 3)


def test_keyword_coverage_partial_match():
    assert metrics.keyword_coverage("obsahuje pentaflex a beton", ["pentaflex", "beton"]) == 1.0
    assert metrics.keyword_coverage("obsahuje pentaflex", ["pentaflex", "vyztuz"]) == 0.5
    assert metrics.keyword_coverage("x", []) == 1.0


def test_score_histogram_bins_are_monotonic_and_cover_all_values():
    histogram = metrics.score_histogram([0.1, 0.2, 0.9, 0.95], bins=5)
    assert sum(bucket["count"] for bucket in histogram) == 4


# --- metrics.py: chunk-level relevance (2026-08-06 ground-truth repair) ------

CHUNK_TARGET_A = {"chunk_ids": {"chunk-a"}, "document": "a.pdf", "text_anchor": "kontrola dodacich listu", "relevance": 3}
CHUNK_TARGET_B = {"chunk_ids": {"chunk-b"}, "document": "c.pdf", "text_anchor": "geodeticky protokol", "relevance": 2}
CHUNK_ROWS = [
    {"document": "a.pdf", "path": "/x/a.pdf", "chunk_id": "chunk-other", "quote": "nesouvisejici text"},
    {"document": "a.pdf", "path": "/x/a.pdf", "chunk_id": "chunk-a", "quote": "Kontrola dodacich listu betonove smesi"},
    {"document": "c.pdf", "path": "/y/FERI/c.pdf", "chunk_id": "chunk-b", "quote": "Geodeticky protokol zakladove desky"},
]
# A "big folder" row: same document/path substring as a target, but a
# DIFFERENT chunk_id and text that does not contain the target's text_anchor
# - exactly the "wide folder trivially satisfies the case" bug this repair
# must NOT allow for chunk-mode matching.
WIDE_FOLDER_ROW = {"document": "unrelated-other-file.pdf", "path": "/y/FERI/unrelated-other-file.pdf", "chunk_id": "chunk-unrelated", "quote": "smlouva o dilo, zcela jiny obsah"}


def test_row_matches_chunk_target_matches_by_chunk_id_when_present():
    assert metrics.row_matches_chunk_target(CHUNK_ROWS[1], CHUNK_TARGET_A)
    assert not metrics.row_matches_chunk_target(CHUNK_ROWS[0], CHUNK_TARGET_A)  # same document, wrong chunk_id


def test_row_matches_chunk_target_falls_back_to_text_anchor_when_no_chunk_id():
    final_row_hit = {"document": "a.pdf", "path": "/x/a.pdf", "quote": "Kontrola dodacich listu betonove smesi"}
    final_row_miss = {"document": "a.pdf", "path": "/x/a.pdf", "quote": "jiny obsah, jina cast dokumentu"}
    assert metrics.row_matches_chunk_target(final_row_hit, CHUNK_TARGET_A)
    assert not metrics.row_matches_chunk_target(final_row_miss, CHUNK_TARGET_A)


def test_row_matches_chunk_target_wide_folder_substring_alone_is_not_enough():
    # A document/path substring match against an EXPECTED folder is not
    # sufficient on its own for a chunk-mode target - this is the exact
    # benevolence bug (330-document ZMENOVA RIZENI folder, 27-document FERI
    # folder) the ground-truth repair fixes.
    assert not metrics.row_matches_chunk_target(WIDE_FOLDER_ROW, CHUNK_TARGET_B)


def test_chunk_recall_at_k_counts_distinct_targets_not_documents():
    assert metrics.chunk_recall_at_k(CHUNK_ROWS, [CHUNK_TARGET_A, CHUNK_TARGET_B], k=10) == 1.0
    assert metrics.chunk_recall_at_k(CHUNK_ROWS[:1], [CHUNK_TARGET_A, CHUNK_TARGET_B], k=10) == 0.0
    assert metrics.chunk_recall_at_k(CHUNK_ROWS, [], k=10) == 1.0


def test_chunk_best_rank_and_hit_rate_and_reciprocal_rank():
    assert metrics.chunk_best_rank(CHUNK_ROWS, [CHUNK_TARGET_A]) == 1
    assert metrics.chunk_best_rank(CHUNK_ROWS, [{"chunk_ids": {"missing"}, "document": "", "text_anchor": "", "relevance": 1}]) is None
    assert metrics.chunk_hit_rate(CHUNK_ROWS, [CHUNK_TARGET_A], k=2) == 1.0
    assert metrics.chunk_hit_rate(CHUNK_ROWS, [CHUNK_TARGET_A], k=1) == 0.0
    assert metrics.chunk_reciprocal_rank(CHUNK_ROWS, [CHUNK_TARGET_A]) == pytest.approx(1 / 2)


def test_chunk_ndcg_at_k_rewards_higher_relevance_targets_found_first():
    # target A (relevance 3) at rank 0, target B (relevance 2) at rank 1 -
    # the ideal ordering (higher-relevance target first) -> nDCG == 1.0.
    ideal_order_rows = [CHUNK_ROWS[1], CHUNK_ROWS[2], CHUNK_ROWS[0]]
    assert metrics.chunk_ndcg_at_k(ideal_order_rows, [CHUNK_TARGET_A, CHUNK_TARGET_B], k=10) == pytest.approx(1.0)
    # Same two targets found, but in the WORSE order (lower relevance first)
    # must score strictly below the ideal ordering above.
    worse_order_rows = [CHUNK_ROWS[2], CHUNK_ROWS[1], CHUNK_ROWS[0]]
    assert metrics.chunk_ndcg_at_k(worse_order_rows, [CHUNK_TARGET_A, CHUNK_TARGET_B], k=10) < 1.0
    assert metrics.chunk_ndcg_at_k(CHUNK_ROWS, [], k=10) == 1.0


# --- dataset/schema.py --------------------------------------------------------

def test_load_dataset_parses_fixture_queries():
    from benchmark.dataset.schema import DATASET_DIR

    cases = load_dataset(DATASET_DIR / "fixture_queries.jsonl")
    assert len(cases) >= 20
    assert all(isinstance(c, BenchmarkCase) for c in cases)
    assert all(c.environment == "fixture" for c in cases)


def test_load_dataset_parses_production_queries():
    from benchmark.dataset.schema import DATASET_DIR

    cases = load_dataset(DATASET_DIR / "production_queries.jsonl")
    assert len(cases) >= 3
    assert all(c.environment == "production" for c in cases)


def test_load_dataset_rejects_missing_required_field(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps({"id": "no-question"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_dataset(bad_file)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text("\n".join([json.dumps({"id": "same", "question": "a"}), json.dumps({"id": "same", "question": "b"})]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(dup_file)


def test_load_dataset_skips_blank_lines_and_comments(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text("# comment\n\n" + json.dumps({"id": "x", "question": "y"}) + "\n", encoding="utf-8")
    cases = load_dataset(path)
    assert len(cases) == 1


def test_benchmark_case_relevance_mode_defaults_to_document_for_backward_compatibility():
    # Existing/older fixture cases never set relevance_mode - they must keep
    # behaving exactly as before this field was introduced.
    case = BenchmarkCase(id="x", question="y")
    assert case.relevance_mode == "document"
    assert case.expected_chunks == []


def test_load_dataset_rejects_invalid_relevance_mode(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "question": "y", "relevance_mode": "nonsense"}), encoding="utf-8")
    with pytest.raises(ValueError, match="relevance_mode"):
        load_dataset(path)


def test_load_dataset_rejects_chunk_mode_without_expected_chunks(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "question": "y", "relevance_mode": "chunk"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_chunks"):
        load_dataset(path)


def test_load_dataset_parses_structured_expected_chunks(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text(json.dumps({
        "id": "x", "question": "y", "relevance_mode": "chunk",
        "expected_chunks": [{"document": "a.pdf", "ordinal": 3, "text_anchor": "foo", "relevance": 3}],
    }), encoding="utf-8")
    cases = load_dataset(path)
    assert len(cases) == 1
    chunk = cases[0].expected_chunks[0]
    assert isinstance(chunk, ExpectedChunk)
    assert (chunk.document, chunk.ordinal, chunk.text_anchor, chunk.relevance) == ("a.pdf", 3, "foo", 3)


def test_expected_chunk_from_value_accepts_legacy_string_as_chunk_id():
    chunk = ExpectedChunk.from_value("abc123:0")
    assert chunk.chunk_id == "abc123:0"
    assert chunk.document == ""


# --- dataset/chunk_resolution.py ------------------------------------------------

def test_resolve_expected_chunks_resolves_document_ordinal_and_text_anchor(fixture_env):
    spec = ExpectedChunk(document="kniha_betonu.txt", ordinal=0, text_anchor="Kniha betonů obsahuje záznamy")
    targets = resolve_expected_chunks(fixture_env.db_path, [spec])
    assert len(targets) == 1
    assert targets[0]["resolved"] is True
    assert targets[0]["ambiguous"] is False
    assert len(targets[0]["chunk_ids"]) == 1


def test_resolve_expected_chunks_reports_unresolved_when_nothing_matches(fixture_env):
    spec = ExpectedChunk(document="kniha_betonu.txt", text_anchor="text that does not exist in this chunk at all")
    targets = resolve_expected_chunks(fixture_env.db_path, [spec])
    assert targets[0]["resolved"] is False
    assert targets[0]["chunk_ids"] == set()


def test_resolve_expected_chunks_legacy_chunk_id_bypasses_lookup(fixture_env):
    spec = ExpectedChunk(chunk_id="some-explicit-id:0")
    targets = resolve_expected_chunks(fixture_env.db_path, [spec])
    assert targets[0]["chunk_ids"] == {"some-explicit-id:0"}
    assert targets[0]["resolved"] is True


def test_load_datasets_rejects_cross_file_duplicate_ids(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "question": "q1"}), encoding="utf-8")
    b.write_text(json.dumps({"id": "x", "question": "q2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_datasets([a, b])


# --- end-to-end smoke test against the fixture environment -------------------

@pytest.fixture(scope="module")
def fixture_env():
    return fixture_environment()


def test_pipeline_trace_runs_all_stages_for_a_simple_query(fixture_env):
    trace = run_pipeline_trace("kniha betonů", fixture_env, result_count=10)
    assert trace.fts_candidates
    assert trace.fusion_candidates
    assert trace.reranked
    assert trace.final_results
    assert set(trace.stages) >= {"intent_detection", "query_parsing", "fts_retrieval", "vector_retrieval", "fusion_rrf", "candidate_pool", "reranker", "diversification", "prompt_builder"}


def test_pipeline_trace_matches_real_ai_search_search_consistency_check(fixture_env):
    trace = run_pipeline_trace("kniha betonů", fixture_env, result_count=10)
    check_consistency("kniha betonů", trace)  # raises PipelineDriftError on mismatch


def test_evaluate_case_passes_for_an_easy_fixture_case(fixture_env):
    case = BenchmarkCase(id="t1", question="kniha betonů", expected_documents=["kniha_betonu.txt"])
    result = evaluate_case(case, fixture_env, result_count=10)
    assert result.passed, result.failure_reasons
    assert result.metrics["recall_at_k_final"] == 1.0
    assert result.error is None


def test_evaluate_case_fails_for_an_impossible_expectation(fixture_env):
    case = BenchmarkCase(id="t2", question="kniha betonů", expected_documents=["totally-unrelated-document-xyz.pdf"])
    result = evaluate_case(case, fixture_env, result_count=10)
    assert not result.passed
    assert result.failure_reasons


# --- evaluate_case: relevance_mode="chunk" (2026-08-06 ground-truth repair) --

def test_evaluate_case_chunk_mode_passes_when_the_specific_chunk_is_retrieved(fixture_env):
    case = BenchmarkCase(
        id="chunk-ok", question="kniha betonů", relevance_mode="chunk",
        expected_chunks=[ExpectedChunk(document="kniha_betonu.txt", ordinal=0, text_anchor="Kniha betonů obsahuje záznamy", relevance=3)],
    )
    result = evaluate_case(case, fixture_env, result_count=10)
    assert result.relevance_mode == "chunk"
    assert result.passed, result.failure_reasons
    assert result.metrics["recall_at_k_final"] == 1.0
    assert result.metrics["expected_chunks_resolved"][0]["resolved"] is True


def test_evaluate_case_chunk_mode_wide_document_match_alone_does_not_satisfy_the_case(fixture_env):
    # A query that DOES retrieve the right document/folder but whose
    # text_anchor never matches (wrong specific chunk) must fail - this is
    # the exact "big folder trivially satisfies the case" bug the repair
    # targets. Using an existing document but a fabricated impossible anchor
    # to simulate "found the file, not the answer".
    case = BenchmarkCase(
        id="chunk-wide-folder-not-enough", question="kniha betonů", relevance_mode="chunk",
        expected_chunks=[ExpectedChunk(document="kniha_betonu.txt", text_anchor="tento text v dokumentu vubec neexistuje", relevance=3)],
    )
    result = evaluate_case(case, fixture_env, result_count=10)
    assert not result.passed
    assert result.metrics["recall_at_k_final"] == 0.0
    assert result.metrics["expected_chunks_resolved"][0]["resolved"] is False


def test_evaluate_case_chunk_mode_supports_multiple_relevant_chunks(fixture_env):
    case = BenchmarkCase(
        id="chunk-multi", question="beton", relevance_mode="chunk",
        expected_chunks=[
            ExpectedChunk(document="kniha_betonu.txt", ordinal=0, text_anchor="Kniha betonů", relevance=3),
            ExpectedChunk(document="dodaci_listy_betonu.txt", ordinal=0, text_anchor="Dodací listy betonu", relevance=2),
        ],
    )
    result = evaluate_case(case, fixture_env, result_count=10)
    assert len(result.metrics["expected_chunks_resolved"]) == 2
    assert all(t["resolved"] for t in result.metrics["expected_chunks_resolved"])


# --- evaluate_case / _aggregate: relevance_mode="multi_document_reasoning" ---

def test_evaluate_case_multi_document_reasoning_is_never_gated(fixture_env):
    # Even an impossible/unresolvable expectation must not fail the case in
    # this mode - it is informational only (see runner.evaluate_case()).
    case = BenchmarkCase(
        id="multi-doc", question="kniha betonů", relevance_mode="multi_document_reasoning",
        expected_documents=["totally-unrelated-document-xyz.pdf"],
    )
    result = evaluate_case(case, fixture_env, result_count=10)
    assert result.passed is True
    assert result.failure_reasons == []
    assert result.relevance_mode == "multi_document_reasoning"


def test_run_benchmark_excludes_multi_document_reasoning_from_headline_pass_fail(tmp_path, fixture_env):
    dataset = tmp_path / "mixed.jsonl"
    dataset.write_text("\n".join([
        json.dumps({"id": "doc-ok", "question": "kniha betonů", "expected_documents": ["kniha_betonu.txt"]}),
        json.dumps({"id": "multi-doc-impossible", "question": "kniha betonů", "relevance_mode": "multi_document_reasoning", "expected_documents": ["totally-unrelated-xyz.pdf"]}),
    ]), encoding="utf-8")
    run = run_benchmark(environment_name="fixture", dataset_paths=[dataset], result_count=10)
    assert run.aggregate["case_count"] == 2
    assert run.aggregate["gated_case_count"] == 1
    assert run.aggregate["multi_document_reasoning_count"] == 1
    assert run.aggregate["passed"] == 1  # only doc-ok is gated/counted
    assert run.aggregate["failed"] == 0
    by_mode = run.aggregate["by_relevance_mode"]
    assert by_mode["multi_document_reasoning"]["case_count"] == 1
    assert by_mode["document"]["case_count"] == 1


def test_run_benchmark_end_to_end_on_fixture_dataset():
    run = run_benchmark(environment_name="fixture", result_count=10)
    assert run.case_count >= 20
    assert run.aggregate["errored"] == 0
    assert run.aggregate["drift_detected_count"] == 0
    # A regression here (mean recall dropping) is exactly what this benchmark
    # exists to catch - this soft assertion protects the framework's own
    # default dataset from silently degrading unnoticed.
    assert run.aggregate["mean_recall_at_k_final"] >= 0.7


# --- Phase 1 validity fixes -----------------------------------------------

def test_precision_at_k_final_no_longer_exists_as_a_metric_name(fixture_env):
    # The false-precision metric was renamed, not duplicated - a stale
    # `precision_at_k_final` key silently reappearing would mean someone
    # re-added the misleading name.
    case = BenchmarkCase(id="rename-check", question="kniha betonů", expected_documents=["kniha_betonu.txt"])
    result = evaluate_case(case, fixture_env, result_count=10)
    assert "forbidden_free_rate" in result.metrics
    assert "precision_at_k_final" not in result.metrics
    assert "forbidden_rate" not in result.metrics
    assert not hasattr(metrics, "precision_at_k")
    assert not hasattr(metrics, "forbidden_rate")


def test_run_benchmark_aggregate_exposes_index_identity(fixture_env):
    run = run_benchmark(environment_name="fixture", result_count=10)
    env = run.environment
    assert env["db_path"] and env["doc_count"] and env["chunk_count"]
    assert env["index_fingerprint"] is not None


def test_run_benchmark_aggregate_exposes_latency_stats():
    run = run_benchmark(environment_name="fixture", result_count=10)
    latency = run.aggregate["latency"]
    retrieval = latency["retrieval_total_ms"]
    assert retrieval is not None and retrieval["mean_ms"] >= 0 and retrieval["p95_ms"] >= 0
    assert "fts_retrieval" in latency["by_stage_ms"]
    # No live Ollama call was made (include_answer defaults to False) - the
    # final_answer stage must not fabricate a latency number for it.
    assert latency["final_answer_ms"] is None


def test_aggregate_makes_an_errored_case_visible_in_passed_failed_errored_counts():
    # Direct unit test of runner._aggregate() (not compare.py) - an errored
    # case must be counted in case_count and `errored`, and must NOT be
    # silently counted as `passed` (it never produced a pass/fail verdict at
    # all, only an exception) nor vanish from metrics_case_count's context.
    from benchmark.runner import CaseResult, _aggregate

    results = [
        CaseResult(id="ok-1", question="q1", type="retrieval", difficulty="easy", tags=[], passed=True, metrics={"recall_at_k_final": 1.0}),
        CaseResult(id="ok-2", question="q2", type="retrieval", difficulty="easy", tags=[], passed=False, metrics={"recall_at_k_final": 0.0}, failure_reasons=["missed"]),
        CaseResult(id="crash", question="q3", type="retrieval", difficulty="easy", tags=[], passed=False, error="RuntimeError: LanceDB table missing"),
    ]
    agg = _aggregate(results)
    assert agg["case_count"] == 3
    assert agg["passed"] == 1
    assert agg["failed"] == 2  # the errored case IS one of the 2 "not passed" - visible, not hidden
    assert agg["errored"] == 1
    # The errored case must be excluded from the metrics denominator (it has
    # no metrics dict), and that exclusion must be visible, not silent.
    assert agg["metrics_case_count"] == 2
    assert agg["metrics_excluded_errored_count"] == 1


def test_mean_metrics_reports_case_counts_without_mean_prefix():
    # metrics_case_count/metrics_excluded_errored_count must NOT be prefixed
    # "mean_" - see runner._mean_metrics()'s docstring for why (would be
    # misread as a 0-1 ratio metric by compare.py's generic mean_* diffing).
    run = run_benchmark(environment_name="fixture", result_count=10)
    assert run.aggregate["metrics_case_count"] == run.aggregate["gated_case_count"]
    assert run.aggregate["metrics_excluded_errored_count"] == 0
    assert "mean_metrics_case_count" not in run.aggregate


# --- compare.py: environment identity check ---------------------------------

def _run(environment: dict, cases: list[dict], aggregate: dict | None = None) -> dict:
    return {"timestamp": "t", "git_sha": "a", "environment": environment, "aggregate": aggregate or {}, "cases": cases}


def test_compare_runs_raises_on_different_db_path():
    baseline = _run({"db_path": "/a/dev.sqlite3", "doc_count": 5, "chunk_count": 50, "index_fingerprint": "x"}, [])
    current = _run({"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048, "index_fingerprint": "y"}, [])
    with pytest.raises(EnvironmentMismatchError, match="db_path"):
        compare_runs(baseline, current)


def test_compare_runs_allows_forced_comparison_with_visible_mismatch():
    baseline = _run({"db_path": "/a/dev.sqlite3", "doc_count": 5, "chunk_count": 50, "index_fingerprint": "x"}, [])
    current = _run({"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048, "index_fingerprint": "y"}, [])
    comparison = compare_runs(baseline, current, strict_environment=False)
    assert comparison.environment_mismatch
    assert any("db_path" in m for m in comparison.environment_mismatch)


def test_compare_runs_raises_on_different_chunk_count():
    # Same db_path/doc_count, only chunk_count differs (e.g. re-chunking
    # changed the split without adding/removing documents) - must be caught
    # on its own, not only as a side effect of a db_path difference.
    baseline = _run({"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048}, [])
    current = _run({"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 140500}, [])
    with pytest.raises(EnvironmentMismatchError, match="chunk_count"):
        compare_runs(baseline, current)


def test_compare_runs_does_not_raise_when_environment_data_is_missing():
    # Older run artifacts (or a production doc_count query that itself
    # failed) simply omit these fields - absence must not be treated as a
    # mismatch, only an actual disagreement between two present values is.
    baseline = _run({}, [{"id": "c1", "question": "q", "passed": True, "metrics": {}, "failure_reasons": []}])
    current = _run({}, [{"id": "c1", "question": "q", "passed": True, "metrics": {}, "failure_reasons": []}])
    comparison = compare_runs(baseline, current)
    assert comparison.environment_mismatch == []


def test_compare_runs_warns_but_does_not_raise_on_artifact_without_fingerprint():
    # A pre-Phase-1 artifact has no index_fingerprint key at all (not even
    # the legacy timestamp). It must be usable - no raise, no crash - and the
    # report must say out loud that identity could not be fully verified,
    # rather than silently treating "no evidence" as "confirmed match".
    baseline = _run({"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048}, [])
    current = _run({"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048,
                    "index_fingerprint": "f" * 64, "index_fingerprint_algorithm": "sha256-v1"}, [])
    comparison = compare_runs(baseline, current)
    assert comparison.environment_mismatch == []
    assert any("index_fingerprint" in note and "missing" in note for note in comparison.environment_notes)


def test_compare_runs_does_not_raise_when_only_one_side_has_identity_data():
    baseline = _run({}, [])
    current = _run({"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048, "index_fingerprint": "y"}, [])
    comparison = compare_runs(baseline, current)
    assert comparison.environment_mismatch == []


def test_compare_runs_passes_when_identity_matches():
    identity = {"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048, "index_fingerprint": "2026-08-06T10:00:00"}
    baseline = _run(identity, [])
    current = _run(identity, [])
    comparison = compare_runs(baseline, current)
    assert comparison.environment_mismatch == []


# --- compare.py: errored cases must not silently improve the comparison ----

def test_compare_runs_flags_newly_errored_case_as_regression_with_reason():
    baseline = _run({}, [{"id": "c1", "question": "q", "passed": True, "metrics": {"recall_at_k_final": 1.0}, "failure_reasons": []}])
    current = _run({}, [{"id": "c1", "question": "q", "passed": False, "metrics": {}, "failure_reasons": [], "error": "TimeoutError: Ollama timed out"}])
    comparison = compare_runs(baseline, current)
    assert comparison.has_regression is True
    delta = next(d for d in comparison.case_deltas if d.id == "c1")
    assert delta.status == "regression"
    assert any("TimeoutError" in reason for reason in delta.current_failure_reasons)


def test_compare_runs_status_deltas_flag_increased_errored_count_as_regression():
    baseline = _run({}, [], aggregate={"passed": 10, "failed": 0, "errored": 0})
    current = _run({}, [], aggregate={"passed": 5, "failed": 0, "errored": 5})
    comparison = compare_runs(baseline, current)
    assert comparison.has_regression is True
    assert comparison.status_deltas["errored"]["status"] == "regression"
    assert comparison.status_deltas["passed"]["status"] == "regression"


def test_compare_runs_status_deltas_unchanged_when_counts_match():
    baseline = _run({}, [], aggregate={"passed": 10, "failed": 0, "errored": 0})
    current = _run({}, [], aggregate={"passed": 10, "failed": 0, "errored": 0})
    comparison = compare_runs(baseline, current)
    assert all(d["status"] == "unchanged" for d in comparison.status_deltas.values())


# --- compare.py: latency wired to LATENCY_TOLERANCE_MS -----------------------

def test_compare_runs_flags_latency_regression_beyond_tolerance():
    baseline = _run({}, [], aggregate={"latency": {"retrieval_total_ms": {"mean_ms": 500.0, "p95_ms": 600.0, "n": 10}}})
    current = _run({}, [], aggregate={"latency": {"retrieval_total_ms": {"mean_ms": 900.0, "p95_ms": 1000.0, "n": 10}}})
    comparison = compare_runs(baseline, current, latency_tolerance_ms=200.0)
    assert comparison.has_regression is True
    assert comparison.latency_deltas["retrieval_total_ms"]["status"] == "regression"


def test_compare_runs_ignores_latency_noise_within_tolerance():
    baseline = _run({}, [], aggregate={"latency": {"retrieval_total_ms": {"mean_ms": 500.0, "p95_ms": 600.0, "n": 10}}})
    current = _run({}, [], aggregate={"latency": {"retrieval_total_ms": {"mean_ms": 550.0, "p95_ms": 650.0, "n": 10}}})
    comparison = compare_runs(baseline, current, latency_tolerance_ms=200.0)
    assert comparison.latency_deltas["retrieval_total_ms"]["status"] == "unchanged"
    assert comparison.has_regression is False


# --- compare.py ----------------------------------------------------------------

def test_compare_runs_detects_per_case_regression_even_if_aggregate_improves():
    baseline = {
        "timestamp": "t0", "git_sha": "aaa", "aggregate": {"mean_recall_at_k_final": 0.5},
        "cases": [
            {"id": "c1", "question": "q1", "passed": True, "metrics": {"recall_at_k_final": 1.0}, "failure_reasons": []},
            {"id": "c2", "question": "q2", "passed": False, "metrics": {"recall_at_k_final": 0.0}, "failure_reasons": ["x"]},
        ],
    }
    current = {
        "timestamp": "t1", "git_sha": "bbb", "aggregate": {"mean_recall_at_k_final": 0.55},
        "cases": [
            {"id": "c1", "question": "q1", "passed": False, "metrics": {"recall_at_k_final": 0.0}, "failure_reasons": ["regressed"]},
            {"id": "c2", "question": "q2", "passed": True, "metrics": {"recall_at_k_final": 1.0}, "failure_reasons": []},
        ],
    }
    comparison = compare_runs(baseline, current)
    assert comparison.has_regression is True
    statuses = {d.id: d.status for d in comparison.case_deltas}
    assert statuses["c1"] == "regression"
    assert statuses["c2"] == "improvement"


def test_compare_runs_handles_new_and_removed_cases():
    baseline = {"timestamp": "t0", "git_sha": "a", "aggregate": {}, "cases": [{"id": "old", "question": "q", "passed": True, "metrics": {}, "failure_reasons": []}]}
    current = {"timestamp": "t1", "git_sha": "b", "aggregate": {}, "cases": [{"id": "new", "question": "q", "passed": True, "metrics": {}, "failure_reasons": []}]}
    comparison = compare_runs(baseline, current)
    statuses = {d.id: d.status for d in comparison.case_deltas}
    assert statuses == {"old": "removed", "new": "new"}


# --- Phase 2: index fingerprint must survive a rename -----------------------

def _write_index(db_path, documents: list[tuple[int, str, str, str]]) -> None:
    """documents = (id, path, content_hash, indexed_at); one chunk each."""
    with ai_search.database(db_path) as con:
        con.execute("DELETE FROM chunks")
        con.execute("DELETE FROM documents")
        for doc_id, path, content_hash, indexed_at in documents:
            con.execute(
                "INSERT INTO documents(id,path,relative_path,name,project,content_hash,size,mtime_ns,inode,extraction,indexed_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, path, path.lstrip("/"), path.rsplit("/", 1)[-1], "P", content_hash, 1, 1, doc_id, "text", indexed_at),
            )
            con.execute("INSERT INTO chunks(id,document_id,ordinal,heading,text) VALUES(?,?,?,?,?)",
                        (f"c{doc_id}", doc_id, 0, "h", "t"))


def test_index_fingerprint_detects_rename_that_max_indexed_at_could_not(tmp_path):
    # The concrete false-match hole the sha256 fingerprint was built for:
    # ai_search.sync()'s rename branch updates documents.path but neither
    # content_hash nor indexed_at, so doc_count, chunk_count AND
    # MAX(indexed_at) all stay identical while every ground-truth path match
    # (metrics.row_matches_any works on paths) can flip. Verified against the
    # real sync() on 2026-08-07; reproduced here at the DB level so the test
    # stays fast and deterministic.
    db_path = tmp_path / "index.sqlite3"
    _write_index(db_path, [(1, "/corpus/alpha.txt", "hash-a", "2026-08-07 13:23:21"),
                           (2, "/corpus/beta.txt", "hash-b", "2026-08-07 13:23:21")])
    before = index_identity(db_path)

    _write_index(db_path, [(1, "/corpus/alpha-PRESUNUTY.txt", "hash-a", "2026-08-07 13:23:21"),
                           (2, "/corpus/beta.txt", "hash-b", "2026-08-07 13:23:21")])
    after = index_identity(db_path)

    assert (before["doc_count"], before["chunk_count"], before["index_max_indexed_at"]) == \
           (after["doc_count"], after["chunk_count"], after["index_max_indexed_at"])
    assert before["index_fingerprint"] != after["index_fingerprint"]


def test_index_fingerprint_detects_reextraction_with_identical_counts(tmp_path):
    # Same doc/chunk counts, same paths, new content - the other way an index
    # can change without moving any of the old identity fields.
    db_path = tmp_path / "index.sqlite3"
    _write_index(db_path, [(1, "/corpus/alpha.txt", "hash-a", "2026-08-07 13:23:21")])
    before = index_identity(db_path)
    _write_index(db_path, [(1, "/corpus/alpha.txt", "hash-a-v2", "2026-08-07 13:23:21")])
    assert index_identity(db_path)["index_fingerprint"] != before["index_fingerprint"]


def test_index_fingerprint_is_stable_across_reads_of_an_unchanged_index(tmp_path):
    # A fingerprint that drifted between two reads of the same index would
    # make every comparison fail with a bogus mismatch.
    db_path = tmp_path / "index.sqlite3"
    _write_index(db_path, [(1, "/corpus/alpha.txt", "hash-a", "2026-08-07 13:23:21"),
                           (2, "/corpus/beta.txt", "hash-b", "2026-08-07 13:23:22")])
    first, second = index_identity(db_path), index_identity(db_path)
    assert first == second
    assert first["index_fingerprint_algorithm"] == FINGERPRINT_ALGORITHM
    assert len(first["index_fingerprint"]) == 64
    assert first["index_max_indexed_at"] == "2026-08-07 13:23:22"


def test_index_identity_never_raises_on_an_unusable_database(tmp_path):
    # Identity collection runs before every benchmark run; it must degrade to
    # "unknown" (which compare.py treats as not-comparable) rather than take
    # the whole run down.
    broken = tmp_path / "broken.sqlite3"
    broken.write_bytes(b"this is not a sqlite file")
    identity = index_identity(broken)
    assert identity["index_fingerprint"] is None and identity["doc_count"] is None


# --- Phase 2: environment identity in compare_runs --------------------------

def test_compare_runs_flags_fingerprint_mismatch_when_counts_still_match():
    identity = {"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048,
                "index_fingerprint_algorithm": FINGERPRINT_ALGORITHM}
    baseline = _run({**identity, "index_fingerprint": "a" * 64}, [])
    current = _run({**identity, "index_fingerprint": "b" * 64}, [])
    with pytest.raises(EnvironmentMismatchError, match="index_fingerprint"):
        compare_runs(baseline, current)


def test_compare_runs_does_not_diff_fingerprints_from_incompatible_algorithms():
    # Backward compatibility: a pre-sha256 artifact stored a MAX(indexed_at)
    # timestamp under the same key. Diffing it against a sha256 digest would
    # fail 100% of the time, so it is labelled as not-compared instead.
    shared = {"db_path": "/a/prod.sqlite3", "doc_count": 6298, "chunk_count": 139048}
    baseline = _run({**shared, "index_fingerprint": "2026-08-06 10:00:00"}, [])
    current = _run({**shared, "index_fingerprint": "b" * 64, "index_fingerprint_algorithm": FINGERPRINT_ALGORITHM}, [])
    comparison = compare_runs(baseline, current)
    assert comparison.environment_mismatch == []
    assert any(LEGACY_FINGERPRINT_ALGORITHM in note for note in comparison.environment_notes)


def test_compare_runs_records_a_note_for_unverifiable_identity():
    # "No mismatch" must be distinguishable from "no evidence" in the report.
    comparison = compare_runs(_run({}, []), _run({}, []))
    assert comparison.environment_mismatch == []
    assert any("index_fingerprint not compared" in note for note in comparison.environment_notes)
    assert any("doc_count not compared" in note for note in comparison.environment_notes)


# --- Phase 2: a changed case population must not read as progress -----------

_PASS = {"passed": True, "metrics": {"recall_at_k_final": 1.0}, "failure_reasons": []}
_FAIL = {"passed": False, "metrics": {"recall_at_k_final": 0.0}, "failure_reasons": ["missed"]}


def test_compare_runs_marks_means_not_comparable_when_a_failing_case_is_removed():
    # Deleting the one failing case from the dataset raises every mean_*
    # without a single retrieval improvement. That must never be rendered as
    # an "improvement".
    baseline = _run({}, [{"id": "c1", "question": "q", **_PASS}, {"id": "c2", "question": "q", **_FAIL}],
                    aggregate={"mean_recall_at_k_final": 0.5})
    current = _run({}, [{"id": "c1", "question": "q", **_PASS}], aggregate={"mean_recall_at_k_final": 1.0})
    comparison = compare_runs(baseline, current)
    assert comparison.mean_metrics_comparable is False
    assert comparison.aggregate_deltas["mean_recall_at_k_final"]["status"] == "not_comparable"
    assert comparison.dataset_delta["removed_case_ids"] == ["c2"]
    assert comparison.dataset_delta["population_changed"] is True


def test_compare_runs_reports_dataset_delta_when_the_case_count_grows():
    baseline = _run({}, [{"id": "c1", "question": "q", **_PASS}], aggregate={"mean_recall_at_k_final": 1.0})
    current = _run({}, [{"id": "c1", "question": "q", **_PASS}, {"id": "c2", "question": "q", **_FAIL}],
                   aggregate={"mean_recall_at_k_final": 0.5})
    comparison = compare_runs(baseline, current)
    assert comparison.dataset_delta["baseline_case_count"] == 1
    assert comparison.dataset_delta["current_case_count"] == 2
    assert comparison.dataset_delta["new_case_ids"] == ["c2"]
    # A bigger dataset finding more problems is not a code regression, and a
    # mean over a different population is not evidence of one either.
    assert comparison.aggregate_deltas["mean_recall_at_k_final"]["status"] == "not_comparable"
    assert comparison.has_regression is False


def test_compare_runs_marks_means_not_comparable_when_errors_shrink_the_denominator():
    # Same dataset, but errored cases contribute no metrics and drop out of
    # the mean - the surviving cases can push every mean_* upward.
    baseline = _run({}, [{"id": "c1", "question": "q", **_PASS}],
                    aggregate={"mean_recall_at_k_final": 0.5, "metrics_case_count": 10})
    current = _run({}, [{"id": "c1", "question": "q", **_PASS}],
                   aggregate={"mean_recall_at_k_final": 0.9, "metrics_case_count": 5})
    comparison = compare_runs(baseline, current)
    assert comparison.mean_metrics_comparable is False
    assert comparison.aggregate_deltas["mean_recall_at_k_final"]["status"] == "not_comparable"


def test_compare_runs_keeps_normal_mean_verdicts_when_the_population_is_stable():
    # Guard against the not_comparable flag swallowing every real verdict.
    baseline = _run({}, [{"id": "c1", "question": "q", **_PASS}],
                    aggregate={"mean_recall_at_k_final": 0.9, "metrics_case_count": 10})
    current = _run({}, [{"id": "c1", "question": "q", **_PASS}],
                   aggregate={"mean_recall_at_k_final": 0.5, "metrics_case_count": 10})
    comparison = compare_runs(baseline, current)
    assert comparison.mean_metrics_comparable is True
    assert comparison.aggregate_deltas["mean_recall_at_k_final"]["status"] == "regression"
    assert comparison.has_regression is True


def test_compare_runs_flags_a_brand_new_case_that_errors():
    # A case with no baseline is filed as status="new"; without an explicit
    # error check an exception in it would never reach the regression list.
    baseline = _run({}, [{"id": "c1", "question": "q", **_PASS}])
    current = _run({}, [{"id": "c1", "question": "q", **_PASS},
                        {"id": "c2", "question": "q", "passed": False, "metrics": {}, "failure_reasons": [],
                         "error": "RuntimeError: LanceDB table missing"}])
    comparison = compare_runs(baseline, current)
    assert comparison.errored_case_ids == ["c2"]
    assert comparison.newly_errored_case_ids == ["c2"]
    assert comparison.has_regression is True


def test_compare_runs_does_not_re_flag_an_error_present_in_both_runs():
    # A known, pre-existing error stays visible in errored_case_ids but must
    # not permanently mark every future comparison as a regression.
    errored = {"passed": False, "metrics": {}, "failure_reasons": [], "error": "TimeoutError"}
    baseline = _run({}, [{"id": "c1", "question": "q", **errored}])
    current = _run({}, [{"id": "c1", "question": "q", **errored}])
    comparison = compare_runs(baseline, current)
    assert comparison.errored_case_ids == ["c1"]
    assert comparison.newly_errored_case_ids == []
    assert comparison.has_regression is False


# --- Phase 2: latency aggregation -------------------------------------------

def test_latency_stats_expose_min_max_and_sample_count():
    stats = _latency_stats([100.0, 200.0, 300.0, 400.0])
    assert stats["mean_ms"] == 250.0
    assert (stats["min_ms"], stats["max_ms"], stats["n"]) == (100.0, 400.0, 4)


def test_latency_stats_return_none_instead_of_fabricating_zero():
    # "Nothing measured" and "measured 0 ms" must stay distinguishable -
    # a fabricated 0.0 would read as an enormous latency improvement.
    assert _latency_stats([]) is None


def test_latency_stats_make_a_single_outlier_visible():
    # The mean alone hides a 5s outlier among fast queries; max exposes it.
    stats = _latency_stats([100.0] * 19 + [5000.0])
    assert stats["mean_ms"] < 400.0
    assert stats["max_ms"] == 5000.0


def test_percentile_equals_max_on_small_samples():
    # Documents a real property of the nearest-rank p95: below n=20 it IS the
    # maximum, so on the current dataset p95 is the slowest single query
    # rather than a tail estimate. `n` in the report is what makes this
    # readable; the value itself is intentionally unchanged.
    assert _percentile([1.0, 2.0, 3.0, 100.0], 95) == 100.0
    assert _percentile([5.0], 95) == 5.0


# --- Phase 2: non-ratio metrics must not get a ratio verdict / a % sign -----

def test_compare_runs_treats_a_faster_cross_encoder_as_an_improvement():
    # mean_cross_encoder_latency_ms is milliseconds, not a 0-1 ratio: the
    # generic "delta < 0 means regression" rule read a 300ms speed-up as a
    # regression and a slowdown as an improvement.
    baseline = _run({}, [], aggregate={"mean_cross_encoder_latency_ms": 900.0})
    current = _run({}, [], aggregate={"mean_cross_encoder_latency_ms": 400.0})
    comparison = compare_runs(baseline, current)
    assert comparison.aggregate_deltas["mean_cross_encoder_latency_ms"]["status"] == "improvement"
    assert comparison.has_regression is False


def test_compare_runs_flags_a_slower_cross_encoder_as_a_regression():
    baseline = _run({}, [], aggregate={"mean_cross_encoder_latency_ms": 400.0})
    current = _run({}, [], aggregate={"mean_cross_encoder_latency_ms": 900.0})
    comparison = compare_runs(baseline, current)
    assert comparison.aggregate_deltas["mean_cross_encoder_latency_ms"]["status"] == "regression"
    assert comparison.has_regression is True


def test_compare_runs_ignores_cross_encoder_latency_noise():
    # Fallback tolerance for a latency metric used to be 0.0, so a 1ms
    # difference was a verdict.
    baseline = _run({}, [], aggregate={"mean_cross_encoder_latency_ms": 400.0})
    current = _run({}, [], aggregate={"mean_cross_encoder_latency_ms": 401.0})
    comparison = compare_runs(baseline, current)
    assert comparison.aggregate_deltas["mean_cross_encoder_latency_ms"]["status"] == "unchanged"


def test_compare_runs_does_not_score_a_directionless_count():
    # More expansion terms is neither better nor worse on its own; it must
    # never be able to fail a comparison.
    baseline = _run({}, [], aggregate={"mean_expansion_term_count": 0.0})
    current = _run({}, [], aggregate={"mean_expansion_term_count": 3.0})
    comparison = compare_runs(baseline, current)
    assert comparison.aggregate_deltas["mean_expansion_term_count"]["status"] == "informational"
    assert comparison.has_regression is False


def test_report_does_not_render_counts_and_durations_as_percentages():
    assert report._metric_value("mean_cross_encoder_latency_ms", 45.0) == "45 ms"
    assert report._metric_value("mean_expansion_term_count", 3.0) == "3.00"
    assert report._metric_value("mean_recall_at_k_final", 0.5) == "50.0%"
    assert report._metric_value("mean_recall_at_k_final", None) == "-"


def test_markdown_run_report_renders_latency_and_fingerprint_sections():
    # End-to-end guard that the new aggregate/latency/identity fields survive
    # rendering (a KeyError here would break every `report` CLI invocation).
    run = run_benchmark(environment_name="fixture", result_count=10).to_dict()
    markdown = report.render_markdown_run(run)
    assert "| Stage | Mean ms | p95 ms | Min ms | Max ms | n |" in markdown
    assert f"({FINGERPRINT_ALGORITHM}," in markdown
    count_row = next(line for line in markdown.splitlines() if "mean_expansion_term_count" in line)
    assert "%" not in count_row


def test_markdown_comparison_report_surfaces_dataset_and_error_warnings():
    baseline = _run({}, [{"id": "c1", "question": "q", **_PASS}, {"id": "gone", "question": "q", **_FAIL}])
    current = _run({}, [{"id": "c1", "question": "q", **_PASS},
                        {"id": "boom", "question": "q", "passed": False, "metrics": {}, "failure_reasons": [], "error": "RuntimeError"}])
    markdown = report.render_markdown_comparison(compare_runs(baseline, current).to_dict())
    assert "DATASET CHANGED" in markdown
    assert "removed: gone" in markdown
    assert "boom (NEW)" in markdown


def test_run_benchmark_latency_stats_include_min_max_and_count():
    run = run_benchmark(environment_name="fixture", result_count=10)
    retrieval = run.aggregate["latency"]["retrieval_total_ms"]
    assert retrieval["n"] >= 1
    assert retrieval["min_ms"] <= retrieval["mean_ms"] <= retrieval["max_ms"]
    for stats in run.aggregate["latency"]["by_stage_ms"].values():
        assert {"mean_ms", "p95_ms", "min_ms", "max_ms", "n"} <= set(stats)
