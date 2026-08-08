"""Runs a benchmark dataset against an Environment and produces a JSON run
artifact (see `RunArtifact.to_dict()` for the exact shape) that
`compare.py`/`report.py` consume.

Nothing in this module changes retrieval behaviour - it only calls
`pipeline_trace.run_pipeline_trace()` (which itself only calls real
ai_search/ui_services functions, see that module's docstring) and computes
metrics from the result.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ai_search

from . import metrics
from .consistency_check import PipelineDriftError, check_consistency
from .dataset.chunk_resolution import resolve_expected_chunks
from .dataset.schema import BenchmarkCase, load_datasets, default_dataset_paths
from .environment import Environment, get_environment
from .pipeline_trace import run_pipeline_trace

DEFAULT_RECALL_THRESHOLD = 0.5  # a case "passes" if >= half its expected_documents are found in the final top-k, unless overridden per-case
RUNS_DIR = Path(__file__).parent / "runs"


@dataclass
class CaseResult:
    id: str
    question: str
    type: str
    difficulty: str
    tags: list[str]
    passed: bool
    relevance_mode: str = "document"
    failure_reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    drift_detected: bool = False
    error: str | None = None


@dataclass
class RunArtifact:
    timestamp: str
    git_sha: str | None
    environment: dict
    dataset_files: list[str]
    case_count: int
    include_answer: bool
    result_count: int
    candidate_strategy: str
    cases: list[CaseResult]
    aggregate: dict
    expand_query: bool | str = False  # False | "both" | "fts" | "vector", see ai_search.EXPANSION_MODES

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def evaluate_case(
    case: BenchmarkCase, environment: Environment, *, result_count: int = 10, include_answer: bool = False,
    verify_consistency: bool = True, candidate_strategy: str = "legacy", cross_encoder: object = None,
    expand_query: bool | str = False,
) -> CaseResult:
    try:
        trace = run_pipeline_trace(
            case.question, environment, result_count=result_count,
            include_answer=include_answer or bool(case.expected_answer_contains), candidate_strategy=candidate_strategy,
            cross_encoder=cross_encoder, expand_query=expand_query,
        )
    except Exception as exc:  # pragma: no cover - defensive: a crashing case must show up as a failed case, not abort the whole run
        return CaseResult(id=case.id, question=case.question, type=case.type, difficulty=case.difficulty, tags=case.tags, passed=False, relevance_mode=case.relevance_mode, error=f"{type(exc).__name__}: {exc}")

    drift_detected = False
    if verify_consistency:
        try:
            check_consistency(case.question, trace)
        except PipelineDriftError as exc:
            drift_detected = True
            return CaseResult(
                id=case.id, question=case.question, type=case.type, difficulty=case.difficulty, tags=case.tags,
                passed=False, relevance_mode=case.relevance_mode, drift_detected=True, error=str(exc),
            )

    k = result_count
    m: dict = {}
    m["candidate_strategy"] = trace.candidate_strategy
    m["intent_is_question_actual"] = trace.is_question
    # Query Understanding layer (see query_expansion.py). On a baseline run these
    # are False/0/[] - kept present rather than omitted so a baseline-vs-expanded
    # comparison has the same metric keys on both sides.
    m["expand_query"] = trace.expand_query
    m["expansion_term_count"] = len(trace.expansion_terms)
    m["expansion_terms"] = list(trace.expansion_terms)
    m["expansion_matched_rules"] = [rule.get("key") for rule in trace.expansion_matched_rules]
    m["expansion_activated"] = bool(trace.expansion_terms)
    m["stage_latency_ms"] = {name: round(t.latency_ms, 2) for name, t in trace.stages.items()}
    m["stage_pool_size"] = {name: t.pool_size for name, t in trace.stages.items()}

    forbidden = case.forbidden_documents

    # relevance_mode dispatch (2026-08-06 ground-truth repair, see
    # dataset/schema.py's module docstring for the full rationale):
    #   * "document" (default, unchanged): expected_documents, folder/file
    #     substring match - identical behaviour/output to before this field
    #     existed.
    #   * "chunk": expected_chunks, resolved fresh against the CURRENT index,
    #     matched by chunk_id on every pipeline-stage list and by
    #     document+text_anchor on the chunk_id-less terminal result lists - a
    #     wide folder substring can no longer trivially satisfy the case.
    # Every downstream metric KEY name is identical between modes (report.py/
    # compare.py stay untouched); only which rows count as "a hit" changes.
    chunk_mode = case.relevance_mode == "chunk"
    if chunk_mode:
        targets = resolve_expected_chunks(environment.db_path, case.expected_chunks)
        m["expected_chunks_resolved"] = [
            {"document": t["document"], "text_anchor": t["text_anchor"], "relevance": t["relevance"],
             "resolved": t["resolved"], "ambiguous": t["ambiguous"], "chunk_id_count": len(t["chunk_ids"])}
            for t in targets
        ]

        def best_rank(rows):
            return metrics.chunk_best_rank(rows, targets)

        def recall_at(rows, kk):
            return metrics.chunk_recall_at_k(rows, targets, kk)

        def hit_rate(rows, kk):
            return metrics.chunk_hit_rate(rows, targets, kk)

        def reciprocal_rank(rows):
            return metrics.chunk_reciprocal_rank(rows, targets)

        def ndcg_at(rows, kk):
            return metrics.chunk_ndcg_at_k(rows, targets, kk)

        # channel_agreement buckets against string needles - no chunk-target
        # equivalent implemented in this iteration (diagnostic-only, not
        # part of the pass/fail gate); left explicitly None rather than
        # silently reusing the document-substring version against the wrong
        # identity.
        m["channel_agreement"] = None
        has_expectation = bool(case.expected_chunks)
    else:
        expected = case.expected_documents

        def best_rank(rows):
            return metrics.best_rank(rows, expected)

        def recall_at(rows, kk):
            return metrics.recall_at_k(rows, expected, kk)

        def hit_rate(rows, kk):
            return metrics.hit_rate(rows, expected, kk)

        def reciprocal_rank(rows):
            return metrics.reciprocal_rank(rows, expected)

        def ndcg_at(rows, kk):
            return metrics.ndcg_at_k(rows, expected, kk)

        m["channel_agreement"] = metrics.channel_agreement(trace.fts_candidates, trace.vector_candidates, expected)
        has_expectation = bool(expected)

    m["best_rank_fts"] = best_rank(trace.fts_candidates)
    m["best_rank_vector"] = best_rank(trace.vector_candidates)
    m["best_rank_fusion_full"] = best_rank(trace.fusion_candidates)
    m["best_rank_union"] = best_rank(trace.union_candidates)
    pool_after = [row for row in trace.fusion_candidates if row["chunk_id"] in set(trace.pool_after_truncation_ids)]
    m["best_rank_fusion_after_truncation"] = best_rank(pool_after)
    m["recall_fusion_full"] = recall_at(trace.fusion_candidates, len(trace.fusion_candidates) or 1)
    m["recall_pool_after_truncation"] = recall_at(pool_after, len(pool_after) or 1)
    m["pool_survival_rate"] = (m["recall_pool_after_truncation"] / m["recall_fusion_full"]) if m["recall_fusion_full"] > 0 else 1.0

    # candidates_before_precision is what candidate_strategy actually resolved
    # `top_ids` to for THIS run (legacy: fusion_candidates[:rerank_k]; union:
    # union_candidates) - the direct "did the candidate architecture keep this
    # document alive into phase 3 at all" signal, independent of whether the
    # (unchanged in this iteration) cosine reranker then ranks it into the final top-k.
    m["candidates_before_precision_size"] = len(trace.candidates_before_precision)
    m["union_pool_size"] = len(trace.union_candidates)
    m["recall_before_precision"] = recall_at(trace.candidates_before_precision, len(trace.candidates_before_precision) or 1)

    # candidate_strategy="union_ce" only - see ai_search.CrossEncoderReranker.
    # None/0 for legacy/union runs (the cross-encoder never ran), which is the
    # correct/expected value there, not a missing-data gap.
    m["candidates_before_cross_encoder_size"] = len(trace.candidates_before_cross_encoder)
    m["recall_before_cross_encoder"] = recall_at(
        trace.candidates_before_cross_encoder, len(trace.candidates_before_cross_encoder) or 1
    ) if trace.candidates_before_cross_encoder else None
    m["cross_encoder_latency_ms"] = round(trace.cross_encoder_latency_ms, 2) if trace.cross_encoder_latency_ms is not None else None
    m["cross_encoder_model"] = trace.cross_encoder_model
    m["best_rank_cross_encoder"] = best_rank(trace.cross_encoder_candidates) if trace.cross_encoder_candidates else None

    m["recall_at_k_reranked"] = recall_at(trace.reranked, k)
    for depth in (10, 50, 100):
        m[f"recall_at_{depth}_reranked"] = recall_at(trace.reranked, depth)
    m["recall_at_k_final"] = recall_at(trace.final_results, k)
    # Renamed from precision_at_k_final (Phase 1 validity fix) - this is NOT
    # real precision, see metrics.forbidden_free_rate()'s docstring for why.
    m["forbidden_free_rate"] = metrics.forbidden_free_rate(trace.final_results, case.expected_documents, forbidden, k)
    m["hit_rate_final"] = hit_rate(trace.final_results, k)
    m["mrr_final"] = reciprocal_rank(trace.final_results)
    m["ndcg_at_k_final"] = ndcg_at(trace.final_results, k)
    m["duplicate_count_final"] = metrics.duplicate_count(trace.final_results)
    m["distinct_ratio_final"] = metrics.distinct_ratio(trace.final_results)
    m["keyword_coverage_final"] = metrics.keyword_coverage(" ".join(row.get("quote", "") for row in trace.final_results[:k]), case.expected_keywords)
    m["candidates_removed_by_dedup"] = len(trace.search_output) - len(trace.deduplicated)
    m["final_result_count"] = len(trace.final_results)
    m["score_histogram_reranked"] = metrics.score_histogram([row["score"] for row in trace.reranked[: trace.rerank_k]])

    if trace.answer is not None:
        answer_text = trace.answer.get("answer", "")
        m["answer_length"] = len(answer_text)
        m["answer_confidence"] = trace.answer.get("confidence")
        m["expected_answer_contains_coverage"] = metrics.keyword_coverage(answer_text, case.expected_answer_contains)

    # multi_document_reasoning cases (see dataset/schema.py) are never gated
    # on Recall@k: the forensic audit found no single retrievable chunk that
    # answers them, so failing them here would just be re-litigating a known,
    # documented index limitation on every run. Metrics above are still
    # computed (informational/diagnostic - e.g. to see if a future retrieval
    # change ever makes evidence-only documents surface higher), they simply
    # do not produce failure_reasons or flip passed=False.
    multi_document_mode = case.relevance_mode == "multi_document_reasoning"

    reasons: list[str] = []
    if not multi_document_mode:
        threshold = case.min_recall_at_10 if case.min_recall_at_10 is not None else DEFAULT_RECALL_THRESHOLD
        if has_expectation and m["recall_at_k_final"] < threshold:
            reasons.append(f"recall_at_k_final={m['recall_at_k_final']:.2f} below threshold {threshold:.2f}")
        if forbidden and m["forbidden_free_rate"] < 1.0:
            reasons.append(f"forbidden content detected in top-{k} (forbidden_free_rate={m['forbidden_free_rate']:.2f})")
        if case.expected_keywords and m["keyword_coverage_final"] < 0.5:
            reasons.append(f"keyword_coverage_final={m['keyword_coverage_final']:.2f} below 0.5")
        if case.expected_answer_contains and trace.answer is not None and m.get("expected_answer_contains_coverage", 1.0) < 1.0:
            reasons.append(f"expected_answer_contains_coverage={m['expected_answer_contains_coverage']:.2f} < 1.0")

    return CaseResult(
        id=case.id, question=case.question, type=case.type, difficulty=case.difficulty, tags=case.tags,
        passed=not reasons, relevance_mode=case.relevance_mode, failure_reasons=reasons, metrics=m, drift_detected=drift_detected,
    )


_MEAN_METRIC_KEYS = (
    "recall_at_k_final", "forbidden_free_rate", "mrr_final", "ndcg_at_k_final", "hit_rate_final",
    "pool_survival_rate", "distinct_ratio_final", "recall_before_precision",
    "recall_at_10_reranked", "recall_at_50_reranked", "recall_at_100_reranked",
    "recall_before_cross_encoder", "cross_encoder_latency_ms",
    "expansion_term_count",
)


def _mean_metrics(results: list[CaseResult]) -> dict:
    valid = [r for r in results if r.error is None]
    # Explicit so a reviewer never has to guess whether mean_* was computed
    # over the full case set or a smaller subset - an errored case has no
    # metrics dict to average, but it must never be able to silently vanish
    # from the denominator without a visible trace (see compare.py's
    # errored-count regression check, which flags any INCREASE in errored
    # cases on its own, independent of what happens to these means).
    # Deliberately NOT "mean_"-prefixed: compare.py's _aggregate_deltas()
    # treats every "mean_*" key as a 0-1 ratio metric with a small tolerance,
    # which would misreport an unrelated dataset-size change (more/fewer
    # cases between baseline and current) as a regression/improvement.
    out = {"metrics_case_count": len(valid), "metrics_excluded_errored_count": len(results) - len(valid)}
    for key in _MEAN_METRIC_KEYS:
        values = [r.metrics[key] for r in valid if r.metrics.get(key) is not None]
        out[f"mean_{key}"] = sum(values) / len(values) if values else None
    return out


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile without a numpy/scipy dependency (this module
    stays dependency-free, see metrics.py's module docstring)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def _latency_stats(values: list[float]) -> dict | None:
    """mean/p95/min/max/n for one latency series, or None when nothing was
    measured (never a fabricated 0.0 - "no measurement" and "measured 0 ms"
    must stay distinguishable).

    `n` is not decoration: with nearest-rank percentiles, p95 == max for any
    n < 20, so on the current ~33-case dataset p95 IS the slowest single
    query rather than a tail estimate. `min_ms`/`max_ms` make that explicit
    and are also what tells an outlier (mean far below max) apart from a
    uniform slowdown (mean close to max) - the mean alone cannot."""
    if not values:
        return None
    return {
        "mean_ms": round(sum(values) / len(values), 2),
        "p95_ms": round(_percentile(values, 95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
        "n": len(values),
    }


def _latency_aggregate(results: list[CaseResult]) -> dict:
    """Aggregates `stage_latency_ms` (per-case, see evaluate_case()) across
    a run: mean/p95 per pipeline stage, plus two rollups callers actually
    compare release over release - total retrieval latency (every stage
    except final_answer, i.e. what the user waits for before the LLM even
    starts) and final_answer latency (the Ollama call) separately, since the
    two have very different causes and very different acceptable tolerances.
    Only non-errored cases contribute (an errored case has no stage_latency_ms
    at all - it crashed before any timing was recorded for that stage).

    Per-stage `n` is reported precisely because not every stage runs for
    every case: `final_answer` only exists for cases that actually called the
    LLM (include_answer, or a case with expected_answer_contains) and
    `cross_encoder` only under candidate_strategy="union_ce". A stage whose
    `n` is below the run's case count contributed to `retrieval_total_ms`
    for only some of the cases, which is worth knowing before reading a
    total-latency delta as a uniform slowdown."""
    valid = [r for r in results if r.error is None]
    per_stage: dict[str, list[float]] = {}
    retrieval_totals: list[float] = []
    final_answer_totals: list[float] = []
    for r in valid:
        stage_latencies = r.metrics.get("stage_latency_ms") or {}
        if not stage_latencies:
            continue
        retrieval_sum = 0.0
        for stage, value in stage_latencies.items():
            if not isinstance(value, (int, float)):
                continue
            per_stage.setdefault(stage, []).append(value)
            if stage == "final_answer":
                final_answer_totals.append(value)
            else:
                retrieval_sum += value
        retrieval_totals.append(retrieval_sum)
    return {
        "by_stage_ms": {stage: _latency_stats(values) for stage, values in per_stage.items()},
        "retrieval_total_ms": _latency_stats(retrieval_totals),
        "final_answer_ms": _latency_stats(final_answer_totals),
    }


def _aggregate(results: list[CaseResult]) -> dict:
    if not results:
        return {}
    # multi_document_reasoning cases (2026-08-06 ground-truth repair) are
    # excluded from the headline passed/failed/mean_* numbers - they are
    # never gated (see evaluate_case()), so counting them as "passed" would
    # misleadingly inflate the pass rate with cases nothing actually
    # verified. They still run and their diagnostics are kept, just reported
    # separately under "by_relevance_mode".
    gated = [r for r in results if r.relevance_mode != "multi_document_reasoning"]
    agg = {
        "case_count": len(results),
        "gated_case_count": len(gated),
        "passed": sum(1 for r in gated if r.passed),
        "failed": sum(1 for r in gated if not r.passed),
        "errored": sum(1 for r in results if r.error is not None),
        "drift_detected_count": sum(1 for r in results if r.drift_detected),
        "multi_document_reasoning_count": len(results) - len(gated),
    }
    agg.update(_mean_metrics(gated))
    # Latency is computed over ALL results (not just `gated`) - a
    # multi_document_reasoning case still executed the full retrieval
    # pipeline and its timing is just as real, even though its pass/fail
    # gate is intentionally skipped (see the comment on `gated` above).
    agg["latency"] = _latency_aggregate(results)

    by_mode: dict[str, dict] = {}
    for mode in sorted({r.relevance_mode for r in results}):
        mode_results = [r for r in results if r.relevance_mode == mode]
        mode_agg = {
            "case_count": len(mode_results),
            "passed": sum(1 for r in mode_results if r.passed),
            "failed": sum(1 for r in mode_results if not r.passed),
            "errored": sum(1 for r in mode_results if r.error is not None),
        }
        mode_agg.update(_mean_metrics(mode_results))
        by_mode[mode] = mode_agg
    agg["by_relevance_mode"] = by_mode
    return agg


def run_benchmark(
    *,
    environment_name: str = "fixture",
    dataset_paths: list[Path] | None = None,
    result_count: int = 10,
    include_answer: bool = False,
    verify_consistency: bool = True,
    candidate_strategy: str = "legacy",
    cross_encoder: object = None,
    expand_query: bool | str = False,
) -> RunArtifact:
    environment = get_environment(environment_name)
    paths = dataset_paths or default_dataset_paths(environment_name)
    cases = [case for case in load_datasets(paths) if case.environment == environment_name]

    # One shared CrossEncoderReranker instance for the WHOLE run (not one per
    # case) - the model must load exactly once and be reused across every
    # case's query, same requirement as the shared `environment.embeddings`.
    if candidate_strategy == "union_ce" and cross_encoder is None:
        cross_encoder = ai_search.CrossEncoderReranker()

    results = [
        evaluate_case(
            case, environment, result_count=result_count, include_answer=include_answer,
            verify_consistency=verify_consistency, candidate_strategy=candidate_strategy, cross_encoder=cross_encoder,
            expand_query=expand_query,
        )
        for case in cases
    ]

    return RunArtifact(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        environment=environment.describe(),
        dataset_files=[str(p) for p in paths],
        case_count=len(cases),
        include_answer=include_answer,
        result_count=result_count,
        candidate_strategy=candidate_strategy,
        cases=results,
        aggregate=_aggregate(results),
        expand_query=expand_query,
    )


def save_run(run: RunArtifact, path: Path | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sha = run.git_sha or "nogit"
        suffix = f"_expand-{run.expand_query}" if run.expand_query else ""
        path = RUNS_DIR / f"{stamp}_{sha}_{run.environment['name']}_{run.candidate_strategy}{suffix}.json"
    path.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
