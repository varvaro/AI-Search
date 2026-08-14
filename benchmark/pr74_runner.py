"""PR7.4 Answer Quality Benchmark runner.

Compares answer() under flag combinations A/B/C/D over ONE shared retrieval
result list. Never mutates search()/RRF/scoring/QE/prompts. Production modules
are imported only as call targets; this file never edits them.

Flag matrix (AUX and MULTI_QUERY forced False in every mode):
  A  STATE_GATE OFF  VALIDATION OFF   — production default / baseline
  B  STATE_GATE OFF  VALIDATION ON    — diagnostics only
  C  STATE_GATE ON   VALIDATION ON    — full candidate
  D  STATE_GATE ON   VALIDATION OFF   — gate rewrite only

Primary GO/NO-GO compares A (baseline) vs C (candidate). B and D isolate which
layer caused a delta.

LLM handling: optional replay of the first raw `_call_ollama` response so
baseline and candidate see the same rendered text before the gate. Marked
`llm_replay=True` in the artifact — never pretended to be two live samples.

PR7.4.1 measurement fixes (audit P2, P3, P5):

  * warmup excludes a case from LATENCY ONLY. It used to drop the case from the
    aggregate as well, which on a production run silently hid the very first
    case - `pr74-signed-haus365-exists`, the query this whole workstream exists
    for - from the blocking gate and from state accuracy.
  * the synthetic fixture pool is built from document fields only.
    `expected_source_contains` is an assertion ABOUT the answer; feeding it into
    the pool as a document name made the assertion validate itself.
  * latency is measured against a primed LLM cache. The replay cache is filled
    by one live call BEFORE mode A, so all four modes are served from cache and
    differ only by gate/validation work. That makes D−A / B−A / C−A honest
    deltas instead of four whole-answer wall times, one of which (A) used to pay
    the live Ollama round trip while the others did not. The live call is timed
    separately and reported as the real end-user latency.
"""
from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import ai_search
import ai_search_config

from .dataset.schema import BenchmarkCase, DATASET_DIR, load_dataset
from .environment import Environment, get_environment
from .pipeline_trace import run_pipeline_trace
from . import pr74_metrics

PR74_DATASET = DATASET_DIR / "pr74_answer_quality.jsonl"
RUNS_DIR = Path(__file__).parent / "runs"

# Mode → (DOCUMENT_STATE_GATE_ENABLED, EVIDENCE_RUNTIME_VALIDATION_ENABLED)
FLAG_MATRIX = {
    "A": (False, False),
    "B": (False, True),
    "C": (True, True),
    "D": (True, False),
}

BASELINE_MODE = "A"
CANDIDATE_MODE = "C"


@dataclass
class ModeAnswer:
    mode: str
    state_gate: bool
    validation: bool
    answer: dict
    answer_wall_ms: float


@dataclass
class Pr74CaseResult:
    id: str
    question: str
    category: str
    environment: str
    tags: list[str]
    retrieval_ms: float
    retrieval_skipped: bool
    llm_replay: bool
    final_results_count: int
    final_results_identity: list[str]
    modes: dict[str, dict]
    evaluation: dict
    # Honest latency split (see module docstring). `live_answer_ms` is the real
    # end-user cost of one answer under baseline flags; the deltas are the
    # marginal cost of the gate / validation with the LLM held constant.
    live_answer_ms: float | None = None
    gate_delta_ms: float | None = None
    validation_delta_ms: float | None = None
    candidate_delta_ms: float | None = None
    error: str | None = None
    # `warmup=True` means: exclude from latency series only. The case still
    # counts for correctness, safety and GO/NO-GO (audit P2).
    warmup: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Pr74RunArtifact:
    timestamp: str
    git_sha: str | None
    environment: dict
    dataset_file: str
    case_count: int
    llm_replay: bool
    flags_constant: dict
    flag_matrix: dict
    cases: list[dict]
    aggregate: dict
    latency: dict
    go_nogo: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def _synthetic_row(document: str, *, document_id: int, score: float = 1.0) -> dict:
    """Controlled citation row for fixture adversarial pools (no retrieval)."""
    return {
        "document": document,
        "path": f"/fixture/{document}",
        "project": "PR74_FIXTURE",
        "heading": "",
        "quote": f"Obsah dokumentu {document}.",
        "score": score,
        "document_id": document_id,
        "chunk_id": f"synth:{document_id}:0",
        "match": {
            "fts_hit": True, "vector_hit": True,
            "semantic_similarity": 0.7, "filename_match": False,
        },
    }


def _build_synthetic_pool(case: BenchmarkCase) -> list[dict]:
    """Build a controlled results pool from case DOCUMENT fields only.

    Used only for fixture cases tagged `synthetic-pool` — never presented as
    production evidence. Pool = expected_documents ∪ forbidden_documents
    (deduped, stable order).

    `expected_source_contains` and `forbidden_sources` are deliberately NOT
    pool inputs (audit P3): they are assertions about which documents the ANSWER
    must or must not lean on. Injecting them as pool rows made
    `expected_source_contains` satisfy itself and turned a bare fragment like
    "podepsaná" into a document that document_state then classified as SIGNED.
    A case that needs the distractor present must name it in
    `forbidden_documents`.
    """
    names: list[str] = []
    for group in (case.expected_documents, case.forbidden_documents):
        for name in group:
            if name and name not in names:
                names.append(name)
    # Empty names is intentional for ENTITY_MISMATCH empty-pool cases.
    return [
        _synthetic_row(name, document_id=i + 1, score=float(len(names) - i))
        for i, name in enumerate(names)
    ]


def _results_identity(rows: list[dict]) -> list[str]:
    return [
        f"{row.get('document_id')}|{row.get('chunk_id')}|{row.get('document')}|{row.get('score')}"
        for row in rows
    ]


@contextmanager
def _forced_flags(state_gate: bool, validation: bool) -> Iterator[None]:
    """Temporarily set PR7 flags; always restore module + config defaults.

    AUX and MULTI_QUERY are forced False for the whole block so a leaked ON
    from an earlier experiment cannot contaminate the measurement.
    """
    saved = {
        "state": ai_search.DOCUMENT_STATE_GATE_ENABLED,
        "validation": ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED,
        "aux": ai_search.AUXILIARY_TERM_COVERAGE_ENABLED,
        "mq": getattr(ai_search, "MULTI_QUERY_RETRIEVAL_ENABLED", False),
    }
    # ui_services reads MULTI_QUERY from its own import binding.
    import ui_services
    saved_mq_ui = ui_services.MULTI_QUERY_RETRIEVAL_ENABLED
    try:
        ai_search.DOCUMENT_STATE_GATE_ENABLED = state_gate
        ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED = validation
        ai_search.AUXILIARY_TERM_COVERAGE_ENABLED = False
        if hasattr(ai_search, "MULTI_QUERY_RETRIEVAL_ENABLED"):
            ai_search.MULTI_QUERY_RETRIEVAL_ENABLED = False
        ui_services.MULTI_QUERY_RETRIEVAL_ENABLED = False
        yield
    finally:
        ai_search.DOCUMENT_STATE_GATE_ENABLED = saved["state"]
        ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED = saved["validation"]
        ai_search.AUXILIARY_TERM_COVERAGE_ENABLED = saved["aux"]
        if hasattr(ai_search, "MULTI_QUERY_RETRIEVAL_ENABLED"):
            ai_search.MULTI_QUERY_RETRIEVAL_ENABLED = saved["mq"]
        ui_services.MULTI_QUERY_RETRIEVAL_ENABLED = saved_mq_ui


class _OllamaReplay:
    """Benchmark-side cache of the first `_call_ollama` response.

    First call hits the live endpoint (or whatever is already monkeypatched);
    subsequent calls with the same (model, prompt, format) return the cached
    string. Never touches production code paths beyond replacing the function
    attribute on the ai_search module for the duration of one case.
    """

    def __init__(self) -> None:
        self.cache: dict[tuple, str] = {}
        self.live_calls = 0
        self.replay_calls = 0
        self._original = ai_search._call_ollama

    def install(self) -> None:
        replay = self

        def wrapped(model, prompt, format_schema=None, timeout=240):
            key = (model, prompt, json.dumps(format_schema, sort_keys=True, default=str) if format_schema is not None else None)
            if key in replay.cache:
                replay.replay_calls += 1
                return replay.cache[key]
            value = replay._original(model, prompt, format_schema=format_schema, timeout=timeout)
            replay.cache[key] = value
            replay.live_calls += 1
            return value

        ai_search._call_ollama = wrapped  # type: ignore[assignment]

    def uninstall(self) -> None:
        ai_search._call_ollama = self._original  # type: ignore[assignment]


def _timed_answer(query: str, results: list[dict]) -> tuple[dict, float]:
    """One answer() call and its wall time. No attribution guessing here —
    attribution is done by differencing modes that share a primed LLM cache."""
    t0 = time.perf_counter()
    answer = ai_search.answer(query, results)
    return answer, (time.perf_counter() - t0) * 1000


def evaluate_pr74_case(
    case: BenchmarkCase,
    environment: Environment,
    *,
    llm_replay: bool = True,
    result_count: int = 10,
    warmup: bool = False,
) -> Pr74CaseResult:
    """Run one case: one retrieval (or synthetic pool), then modes A–D."""
    use_synthetic = (
        case.environment == "fixture"
        and "synthetic-pool" in (case.tags or [])
    )
    retrieval_ms = 0.0
    retrieval_skipped = False
    try:
        if use_synthetic:
            final_results = _build_synthetic_pool(case)
            retrieval_skipped = True
        else:
            t0 = time.perf_counter()
            # Force AUX/MQ OFF for the retrieval leg too.
            with _forced_flags(False, False):
                trace = run_pipeline_trace(
                    case.question, environment,
                    result_count=result_count, include_answer=False,
                )
            retrieval_ms = (time.perf_counter() - t0) * 1000
            final_results = list(trace.final_results)

        identity_before = _results_identity(final_results)
        replay = _OllamaReplay() if llm_replay else None
        live_answer_ms: float | None = None

        modes: dict[str, ModeAnswer] = {}
        try:
            if replay is not None:
                replay.install()
                # Prime the cache with ONE live generation under baseline flags.
                # After this every mode below is served from cache, so their
                # wall times differ only by gate/validation work and can be
                # differenced. This call is also the honest end-user latency.
                with _forced_flags(False, False):
                    _primed, live_answer_ms = _timed_answer(case.question, final_results)

            for mode, (state_on, val_on) in FLAG_MATRIX.items():
                with _forced_flags(state_on, val_on):
                    # Identity guard: answer() must not mutate results.
                    identity_now = _results_identity(final_results)
                    if identity_now != identity_before:
                        raise RuntimeError(
                            f"final_results mutated before mode {mode}: "
                            f"{identity_before!r} → {identity_now!r}"
                        )
                    answer, wall_ms = _timed_answer(case.question, final_results)
                    if _results_identity(final_results) != identity_before:
                        raise RuntimeError(
                            f"final_results mutated by answer() in mode {mode}"
                        )
                    modes[mode] = ModeAnswer(
                        mode=mode, state_gate=state_on, validation=val_on,
                        answer=answer, answer_wall_ms=wall_ms,
                    )
        finally:
            if replay is not None:
                replay.uninstall()

        # Deltas are only meaningful when every mode saw the same cached LLM
        # response. Without replay the generation noise dwarfs the gate cost, so
        # they stay None rather than reporting a number nobody can trust.
        if replay is not None:
            base_wall = modes[BASELINE_MODE].answer_wall_ms
            gate_delta = modes["D"].answer_wall_ms - base_wall
            validation_delta = modes["B"].answer_wall_ms - base_wall
            candidate_delta = modes[CANDIDATE_MODE].answer_wall_ms - base_wall
        else:
            gate_delta = validation_delta = candidate_delta = None

        baseline = modes[BASELINE_MODE].answer
        candidate = modes[CANDIDATE_MODE].answer
        evaluation = pr74_metrics.evaluate_case_answers(case, baseline, candidate)

        return Pr74CaseResult(
            id=case.id,
            question=case.question,
            category=case.category or "",
            environment=case.environment,
            tags=list(case.tags or []),
            retrieval_ms=retrieval_ms,
            retrieval_skipped=retrieval_skipped,
            llm_replay=bool(llm_replay and replay is not None and replay.replay_calls > 0),
            final_results_count=len(final_results),
            final_results_identity=identity_before,
            modes={
                m: {
                    "state_gate": modes[m].state_gate,
                    "validation": modes[m].validation,
                    "answer": modes[m].answer,
                    "answer_wall_ms": round(modes[m].answer_wall_ms, 3),
                }
                for m in FLAG_MATRIX
            },
            evaluation=evaluation.to_dict(),
            live_answer_ms=None if live_answer_ms is None else round(live_answer_ms, 2),
            gate_delta_ms=None if gate_delta is None else round(gate_delta, 3),
            validation_delta_ms=None if validation_delta is None else round(validation_delta, 3),
            candidate_delta_ms=None if candidate_delta is None else round(candidate_delta, 3),
            warmup=warmup,
        )
    except Exception as exc:
        return Pr74CaseResult(
            id=case.id,
            question=case.question,
            category=case.category or "",
            environment=case.environment,
            tags=list(case.tags or []),
            retrieval_ms=retrieval_ms,
            retrieval_skipped=retrieval_skipped,
            llm_replay=llm_replay,
            final_results_count=0,
            final_results_identity=[],
            modes={},
            evaluation={},
            error=f"{type(exc).__name__}: {exc}",
            warmup=warmup,
        )


def _latency_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    p95_index = min(n - 1, max(0, int((0.95 * n + 0.999999)) - 1))
    # nearest-rank; with small n, p95 ≈ max — report n so readers know.
    return {
        "mean_ms": round(sum(ordered) / n, 2),
        "p95_ms": round(ordered[p95_index], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "n": n,
    }


def _latency_aggregate(case_dicts: list[dict]) -> dict:
    """Latency series. Warmup is excluded HERE and only here (audit P2)."""
    measured = [c for c in case_dicts if not c.get("warmup") and c.get("error") is None]

    def numbers(values) -> list[float]:
        return [v for v in values if isinstance(v, (int, float))]

    retrieval = numbers(c["retrieval_ms"] for c in measured if not c.get("retrieval_skipped"))
    live = numbers(c.get("live_answer_ms") for c in measured)
    gate_delta = numbers(c.get("gate_delta_ms") for c in measured)
    validation_delta = numbers(c.get("validation_delta_ms") for c in measured)
    candidate_delta = numbers(c.get("candidate_delta_ms") for c in measured)
    end_to_end = numbers(
        (c.get("retrieval_ms") or 0.0) + c["live_answer_ms"]
        for c in measured
        if isinstance(c.get("live_answer_ms"), (int, float)) and not c.get("retrieval_skipped")
    )
    return {
        "warmup_excluded_case_ids": [c["id"] for c in case_dicts if c.get("warmup")],
        "retrieval_ms": _latency_stats(retrieval),
        # The only number that represents what a user waits for.
        "live_answer_ms": _latency_stats(live),
        "end_to_end_ms": _latency_stats(end_to_end),
        # Marginal cost of each layer with the LLM response held constant.
        "state_gate_delta_ms": _latency_stats(gate_delta),
        "validation_delta_ms": _latency_stats(validation_delta),
        "candidate_delta_ms": _latency_stats(candidate_delta),
        "note": (
            "live_answer_ms is one real generation under baseline flags and is "
            "the only user-facing latency here. The *_delta_ms series are "
            "differences between modes that all read the SAME cached Ollama "
            "response (cache primed before mode A), so they measure gate and "
            "validation work only - they are not answer latencies and must not "
            "be compared against live_answer_ms as if the candidate were "
            "faster. Deltas are absent when llm_replay=False. Warmup cases are "
            "excluded from latency only; they still count for correctness."
        ),
    }


def _go_nogo(aggregate: pr74_metrics.AggregateMetrics, case_dicts: list[dict]) -> dict:
    regression_unchanged_ok = True
    for c in case_dicts:
        # NOTE: warmup is NOT skipped here — it only excludes a case from the
        # latency series (audit P2).
        if c.get("error"):
            continue
        ev = c.get("evaluation") or {}
        if ev.get("category") == "REGRESSION" and ev.get("answer_delta") not in ("unchanged", None):
            # LOW wording changes are recorded but do not alone flip NO-GO;
            # degraded/blocking do.
            if ev.get("answer_delta") == "degraded" or any(
                f.get("severity") == "BLOCKING" for f in (ev.get("failures") or [])
            ):
                regression_unchanged_ok = False
                break

    reasons = []
    if aggregate.has_blocking_regression:
        reasons.append(f"blocking_case_ids={aggregate.blocking_case_ids}")
    if aggregate.false_signed_confirmations:
        reasons.append(f"false_signed_confirmations={aggregate.false_signed_confirmations}")
    if aggregate.wrong_entity_citations:
        reasons.append(f"wrong_entity_citations={aggregate.wrong_entity_citations}")
    if not regression_unchanged_ok:
        reasons.append("REGRESSION category degraded or blocked under candidate flags")
    errored = [c["id"] for c in case_dicts if c.get("error")]
    if errored:
        reasons.append(f"errored_cases={errored}")

    verdict = "GO" if not reasons else "NO-GO"
    return {
        "verdict": verdict,
        "has_blocking_regression": aggregate.has_blocking_regression,
        "blocking_case_ids": list(aggregate.blocking_case_ids),
        "reasons": reasons,
        "criteria": {
            "zero_blocking": aggregate.blocking_count == 0,
            "zero_wrong_entity_citations": aggregate.wrong_entity_citations == 0,
            "zero_false_signed_confirmations": aggregate.false_signed_confirmations == 0,
            "regression_category_safe": regression_unchanged_ok,
            "no_errored_cases": not errored,
        },
    }


def run_pr74_benchmark(
    *,
    environment_name: str = "production",
    dataset_path: Path | None = None,
    llm_replay: bool = True,
    result_count: int = 10,
    case_filter: set[str] | None = None,
    exclude_warmup: bool = True,
) -> Pr74RunArtifact:
    """Execute the PR7.4 suite.

    Cases whose `environment` field does not match `environment_name` are
    skipped (fixture adversarial cases never run against production and vice
    versa — keeps fixture evidence from being reported as production).
    """
    dataset_path = Path(dataset_path) if dataset_path else PR74_DATASET
    all_cases = load_dataset(dataset_path)
    cases = [c for c in all_cases if c.environment == environment_name]
    if case_filter is not None:
        cases = [c for c in cases if c.id in case_filter]

    environment = get_environment(environment_name)

    # Snapshot defaults so a crash mid-run cannot leave flags ON.
    assert ai_search_config.DOCUMENT_STATE_GATE_ENABLED is False
    assert ai_search_config.EVIDENCE_RUNTIME_VALIDATION_ENABLED is False
    assert ai_search_config.AUXILIARY_TERM_COVERAGE_ENABLED is False
    assert ai_search_config.MULTI_QUERY_RETRIEVAL_ENABLED is False

    results: list[Pr74CaseResult] = []
    for index, case in enumerate(cases):
        warmup = exclude_warmup and index == 0 and not (
            case.environment == "fixture" and "synthetic-pool" in (case.tags or [])
        )
        results.append(
            evaluate_pr74_case(
                case, environment,
                llm_replay=llm_replay,
                result_count=result_count,
                warmup=warmup,
            )
        )

    # Restore config-level defaults explicitly (belt and braces).
    ai_search.DOCUMENT_STATE_GATE_ENABLED = ai_search_config.DOCUMENT_STATE_GATE_ENABLED
    ai_search.EVIDENCE_RUNTIME_VALIDATION_ENABLED = ai_search_config.EVIDENCE_RUNTIME_VALIDATION_ENABLED
    ai_search.AUXILIARY_TERM_COVERAGE_ENABLED = ai_search_config.AUXILIARY_TERM_COVERAGE_ENABLED

    case_dicts = [r.to_dict() for r in results]
    # Rebuild Failure objects (asdict flattened them).
    rebuilt: list[pr74_metrics.CaseEvaluation] = []
    for c in case_dicts:
        # warmup cases ARE included: excluding them from correctness hid the
        # first production case from the blocking gate entirely (audit P2).
        if c.get("error") or not c.get("evaluation"):
            continue
        ev = c["evaluation"]
        rebuilt.append(pr74_metrics.CaseEvaluation(
            id=ev["id"],
            category=ev.get("category") or "",
            answer_delta=ev.get("answer_delta") or "unchanged",
            failures=[pr74_metrics.Failure(**f) for f in (ev.get("failures") or [])],
            state_verdict_actual=ev.get("state_verdict_actual"),
            state_verdict_expected=ev.get("state_verdict_expected"),
            state_verdict_match=ev.get("state_verdict_match"),
            intent_coverage_actual=ev.get("intent_coverage_actual"),
            intent_coverage_expected=ev.get("intent_coverage_expected"),
            intent_coverage_match=ev.get("intent_coverage_match"),
            missing_needs_match=ev.get("missing_needs_match"),
            baseline_answer_len=ev.get("baseline_answer_len") or 0,
            candidate_answer_len=ev.get("candidate_answer_len") or 0,
            evidence_tiers=ev.get("evidence_tiers") or {},
        ))
    aggregate = pr74_metrics.aggregate_evaluations(rebuilt)
    latency = _latency_aggregate(case_dicts)
    go = _go_nogo(aggregate, case_dicts)

    return Pr74RunArtifact(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        environment=environment.describe(),
        dataset_file=str(dataset_path),
        case_count=len(cases),
        llm_replay=llm_replay,
        flags_constant={
            "AUXILIARY_TERM_COVERAGE_ENABLED": False,
            "MULTI_QUERY_RETRIEVAL_ENABLED": False,
        },
        flag_matrix={
            mode: {
                "DOCUMENT_STATE_GATE_ENABLED": pair[0],
                "EVIDENCE_RUNTIME_VALIDATION_ENABLED": pair[1],
            }
            for mode, pair in FLAG_MATRIX.items()
        },
        cases=case_dicts,
        aggregate=aggregate.to_dict(),
        latency=latency,
        go_nogo=go,
    )


def save_pr74_run(run: Pr74RunArtifact, path: Path | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sha = run.git_sha or "nogit"
        path = RUNS_DIR / f"{stamp}_{sha}_pr74_{run.environment['name']}.json"
    path.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
