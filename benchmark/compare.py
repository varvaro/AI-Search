"""Compares two run artifacts (baseline vs. current) and flags regressions.

This is what implements task 6 ("oprava jednoho dotazu nesmí zhoršit dalších
50"): the comparison is done PER CASE, not only on the aggregate mean - a
regression on any single previously-passing case is reported even if the
aggregate mean improves, because an aggregate improvement can otherwise hide
one query getting much worse while nine others improve slightly.

Phase 1 validity fix (2026-08-07): before this, compare_runs() would happily
diff two run artifacts produced against completely different indexes (e.g.
the 492 KB dev/test index vs. the 461 MB production index - the exact mistake
made once already in this project's history, see the module docstring in
environment.py) and print a "comparison" as if the numbers meant anything.
`_environment_mismatches()` below makes that loud instead of silent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Minimum change (in absolute metric units, e.g. 0.02 = 2 percentage points
# for a 0-1 ratio metric) before a delta is reported as a regression/
# improvement rather than noise. Latency uses its own tolerance since it is
# measured in milliseconds, not a 0-1 ratio.
DEFAULT_TOLERANCE = {
    "mean_recall_at_k_final": 0.02,
    "mean_forbidden_free_rate": 0.02,
    "mean_mrr_final": 0.02,
    "mean_ndcg_at_k_final": 0.02,
    "mean_hit_rate_final": 0.02,
    "mean_pool_survival_rate": 0.02,
    "mean_distinct_ratio_final": 0.02,
}
LATENCY_TOLERANCE_MS = 200.0

# Not every mean_* is a 0-1 "higher is better" ratio, and treating them all as
# one was silently inverting verdicts. mean_cross_encoder_latency_ms is
# milliseconds: a FALLING value means the cross encoder got faster, which the
# generic rule reported as a regression (and a slowdown as an "improvement"),
# with a fallback tolerance of 0.0 that made float noise enough to trigger it.
# mean_expansion_term_count is a plain count with no good direction at all -
# more expansion terms is neither better nor worse on its own.
_LOWER_IS_BETTER_METRICS = frozenset({"mean_cross_encoder_latency_ms"})
_INFORMATIONAL_METRICS = frozenset({"mean_expansion_term_count"})

# Environment.describe() keys that identify WHICH index a run's numbers were
# computed against - see _environment_mismatches(). index_fingerprint is
# handled separately because its comparison is gated on both sides having
# been produced by the same fingerprint algorithm.
_IDENTITY_KEYS = ("db_path", "doc_count", "chunk_count")
# Run artifacts written before environment.FINGERPRINT_ALGORITHM existed
# stored a bare MAX(documents.indexed_at) timestamp under `index_fingerprint`
# and no algorithm field. Their digests are not comparable with sha256-v1
# ones, so they are labelled rather than diffed.
LEGACY_FINGERPRINT_ALGORITHM = "legacy-max-indexed-at"


class EnvironmentMismatchError(RuntimeError):
    """Raised by compare_runs() when baseline and current were produced
    against different indexes (different db_path/doc_count/chunk_count/
    index_fingerprint). Comparing retrieval metrics across two different
    indexes is meaningless - pass strict_environment=False to force a
    (clearly-flagged, via ComparisonReport.environment_mismatch) comparison
    anyway, e.g. for manual forensic inspection."""


@dataclass
class CaseDelta:
    id: str
    question: str
    baseline_passed: bool | None
    current_passed: bool | None
    status: str  # "regression" | "improvement" | "unchanged" | "new" | "removed"
    metric_deltas: dict = field(default_factory=dict)
    current_failure_reasons: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    baseline_meta: dict
    current_meta: dict
    aggregate_deltas: dict
    case_deltas: list[CaseDelta]
    has_regression: bool
    environment_mismatch: list[str] = field(default_factory=list)
    status_deltas: dict = field(default_factory=dict)
    latency_deltas: dict = field(default_factory=dict)
    # Identity evidence that could NOT be compared (missing on one side, or
    # two incompatible fingerprint algorithms) - absence of a mismatch is
    # not the same as a proven match, and the report says which it is.
    environment_notes: list[str] = field(default_factory=list)
    dataset_delta: dict = field(default_factory=dict)
    errored_case_ids: list[str] = field(default_factory=list)
    newly_errored_case_ids: list[str] = field(default_factory=list)
    # False when the two runs' mean_* values were averaged over different
    # case populations - see _mean_population_changed().
    mean_metrics_comparable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fingerprint_algorithm(environment: dict) -> str | None:
    """Which algorithm produced this artifact's `index_fingerprint`, or None
    if it has no fingerprint at all. Artifacts predating the sha256
    fingerprint carry no algorithm field - they are mapped to the legacy
    pseudo-algorithm so their timestamp-shaped digest is never diffed
    against a sha256 one (which would be a guaranteed false mismatch)."""
    if environment.get("index_fingerprint") is None:
        return None
    return environment.get("index_fingerprint_algorithm") or LEGACY_FINGERPRINT_ALGORITHM


def _environment_mismatches(baseline: dict, current: dict) -> tuple[list[str], list[str]]:
    """Compares the subset of Environment.describe() that identifies WHICH
    index each run's numbers came from. Returns (mismatches, notes).

    A key is only compared when BOTH sides have a non-None value for it -
    missing data (an older run artifact, or a production run where the
    doc_count query itself failed) is recorded as a NOTE rather than
    silently treated as a match: the comparison is still allowed, but the
    report says out loud which identity evidence was unavailable.

    The fingerprint is additionally gated on both sides declaring the same
    algorithm, which is what keeps pre-sha256 artifacts readable instead of
    failing every comparison against a newer run."""
    before = (baseline.get("environment") or {})
    after = (current.get("environment") or {})
    mismatches: list[str] = []
    notes: list[str] = []
    for key in _IDENTITY_KEYS:
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            notes.append(f"{key} not compared (missing on {'baseline' if b is None else 'current'})")
            continue
        if b != a:
            mismatches.append(f"{key}: baseline={b!r} current={a!r}")

    b_algorithm, a_algorithm = _fingerprint_algorithm(before), _fingerprint_algorithm(after)
    if b_algorithm is None or a_algorithm is None:
        notes.append(f"index_fingerprint not compared (missing on {'baseline' if b_algorithm is None else 'current'})")
    elif b_algorithm != a_algorithm:
        notes.append(f"index_fingerprint not compared (baseline={b_algorithm}, current={a_algorithm} - incompatible algorithms)")
    elif before["index_fingerprint"] != after["index_fingerprint"]:
        mismatches.append(f"index_fingerprint ({b_algorithm}): baseline={before['index_fingerprint']!r} current={after['index_fingerprint']!r}")
    return mismatches, notes


def _dataset_delta(baseline_cases: dict, current_cases: dict) -> dict:
    """How the case population itself changed between the two runs. Any
    change here invalidates a straight mean-vs-mean comparison: dropping a
    failing case from the dataset raises every mean_* without a single
    retrieval improvement, which is exactly the kind of "progress" this
    benchmark must not be able to manufacture."""
    new = sorted(set(current_cases) - set(baseline_cases))
    removed = sorted(set(baseline_cases) - set(current_cases))
    return {
        "baseline_case_count": len(baseline_cases),
        "current_case_count": len(current_cases),
        "new_case_ids": new,
        "removed_case_ids": removed,
        "common_case_count": len(set(baseline_cases) & set(current_cases)),
        "population_changed": bool(new or removed),
    }


def _status_deltas(baseline: dict, current: dict) -> dict:
    """Run-level pass/fail/error counts. An INCREASE in `errored` is always
    reported as its own regression signal, independent of what happens to
    any mean_* metric - see runner._mean_metrics()'s docstring: an errored
    case has no metrics to average, so it is invisible to mean_* deltas by
    construction. Without this explicit check, a code change that makes 5
    previously-slow-but-passing queries start raising exceptions could
    LOOK like an improvement (those 5 low scores simply vanish from the
    average) instead of the severe regression it actually is."""
    b = baseline.get("aggregate") or {}
    c = current.get("aggregate") or {}
    deltas = {}
    for key, worse_is in (("errored", "higher"), ("failed", "higher"), ("passed", "lower")):
        bv, cv = b.get(key), c.get(key)
        if bv is None or cv is None:
            continue
        delta = cv - bv
        if delta == 0:
            status = "unchanged"
        elif (delta > 0) == (worse_is == "higher"):
            status = "regression"
        else:
            status = "improvement"
        deltas[key] = {"before": bv, "after": cv, "delta": delta, "status": status}
    return deltas


def _latency_deltas(baseline: dict, current: dict, tolerance_ms: float) -> dict:
    """Wires runner._latency_aggregate()'s mean_ms rollups into the
    comparison: a mean latency increase beyond `tolerance_ms` is a
    regression, a decrease beyond it an improvement. Only the two headline
    rollups (total retrieval, final LLM answer) are gated here - per-stage
    numbers stay available in each run artifact for manual drill-down but
    are not, on their own, pass/fail signals (a single stage can legitimately
    shift while the end-to-end latency the user feels does not)."""
    b_latency = (baseline.get("aggregate") or {}).get("latency") or {}
    c_latency = (current.get("aggregate") or {}).get("latency") or {}
    deltas = {}
    for key in ("retrieval_total_ms", "final_answer_ms"):
        b = (b_latency.get(key) or {}).get("mean_ms")
        c = (c_latency.get(key) or {}).get("mean_ms")
        if b is None or c is None:
            continue
        delta = c - b
        status = "regression" if delta > tolerance_ms else "improvement" if delta < -tolerance_ms else "unchanged"
        deltas[key] = {"before": b, "after": c, "delta": round(delta, 2), "status": status}
    return deltas


def _failure_reasons(case: dict) -> list[str]:
    """`failure_reasons` plus a visible ERROR entry when the case raised an
    exception (evaluate_case() leaves `failure_reasons` empty for an errored
    case - only `error` is set - so without this an errored regression would
    show up in CaseDelta.status but with an empty, unhelpful reasons list)."""
    reasons = list(case.get("failure_reasons") or [])
    if case.get("error"):
        reasons.append(f"ERROR: {case['error']}")
    return reasons


def _mean_population_changed(baseline: dict, current: dict, dataset_delta: dict) -> bool:
    """True when the two runs' mean_* values were averaged over a different
    set of cases, which makes a mean-vs-mean delta uninterpretable.

    Two independent causes, both real:
      * the dataset itself changed (a case added/removed - dataset_delta), and
      * the same dataset produced a different number of *usable* cases,
        because errored cases contribute no metrics and silently leave the
        denominator (runner._mean_metrics()'s metrics_case_count). Five slow,
        low-recall queries starting to raise exceptions makes every mean_*
        jump upward; without this flag the aggregate table would label that
        an "improvement"."""
    if dataset_delta["population_changed"]:
        return True
    before = (baseline.get("aggregate") or {}).get("metrics_case_count")
    after = (current.get("aggregate") or {}).get("metrics_case_count")
    return before is not None and after is not None and before != after


def _aggregate_deltas(baseline: dict, current: dict, tolerance: dict, population_changed: bool = False) -> dict:
    deltas = {}
    keys = set(baseline.get("aggregate", {})) | set(current.get("aggregate", {}))
    for key in sorted(keys):
        if not key.startswith("mean_"):
            continue
        before = baseline.get("aggregate", {}).get(key)
        after = current.get("aggregate", {}).get(key)
        if before is None or after is None:
            continue
        delta = after - before
        if population_changed:
            # Numbers are still reported (a human may want to eyeball them),
            # but never as "improvement"/"regression" - see
            # _mean_population_changed().
            status = "not_comparable"
        elif key in _INFORMATIONAL_METRICS:
            status = "informational"
        else:
            lower_is_better = key in _LOWER_IS_BETTER_METRICS
            tol = tolerance.get(key, LATENCY_TOLERANCE_MS if lower_is_better else 0.0)
            effective = -delta if lower_is_better else delta
            status = "regression" if effective < -tol else "improvement" if effective > tol else "unchanged"
        deltas[key] = {"before": before, "after": after, "delta": delta, "status": status}
    return deltas


def compare_runs(
    baseline: dict, current: dict, tolerance: dict | None = None, *,
    latency_tolerance_ms: float = LATENCY_TOLERANCE_MS, strict_environment: bool = True,
) -> ComparisonReport:
    """Compares two run artifacts. Raises EnvironmentMismatchError before
    computing anything else if `baseline`/`current` were produced against
    different indexes and `strict_environment` is True (the default) - see
    _environment_mismatches(). Pass strict_environment=False to instead get
    a ComparisonReport with `environment_mismatch` populated as a visible
    warning, e.g. for manual forensic inspection where comparing across an
    index change is intentional."""
    tolerance = {**DEFAULT_TOLERANCE, **(tolerance or {})}
    environment_mismatch, environment_notes = _environment_mismatches(baseline, current)
    if environment_mismatch and strict_environment:
        raise EnvironmentMismatchError(
            "baseline and current runs were produced against different indexes - a metric "
            "comparison between them would be meaningless: " + "; ".join(environment_mismatch)
        )

    baseline_cases = {c["id"]: c for c in baseline.get("cases", [])}
    current_cases = {c["id"]: c for c in current.get("cases", [])}
    all_ids = sorted(set(baseline_cases) | set(current_cases))
    dataset_delta = _dataset_delta(baseline_cases, current_cases)

    case_deltas: list[CaseDelta] = []
    has_regression = False
    for case_id in all_ids:
        before = baseline_cases.get(case_id)
        after = current_cases.get(case_id)
        if before is None:
            case_deltas.append(CaseDelta(id=case_id, question=after["question"], baseline_passed=None, current_passed=after["passed"], status="new", current_failure_reasons=_failure_reasons(after)))
            continue
        if after is None:
            case_deltas.append(CaseDelta(id=case_id, question=before["question"], baseline_passed=before["passed"], current_passed=None, status="removed"))
            continue

        metric_deltas = {}
        for key in ("recall_at_k_final", "forbidden_free_rate", "mrr_final", "ndcg_at_k_final", "hit_rate_final", "pool_survival_rate", "duplicate_count_final"):
            b = before.get("metrics", {}).get(key)
            a = after.get("metrics", {}).get(key)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                metric_deltas[key] = round(a - b, 4)

        # A case that started/stopped erroring is ALWAYS a regression/
        # improvement by itself - it must never fall through to the recall
        # delta below, which would be 0.0/absent for an errored case (no
        # metrics were ever computed) and could misreport it as "unchanged".
        newly_errored = after.get("error") is not None and before.get("error") is None
        no_longer_errored = before.get("error") is not None and after.get("error") is None
        regressed_case = (before["passed"] and not after["passed"]) or newly_errored
        improved_case = (not before["passed"] and after["passed"]) or no_longer_errored
        if regressed_case:
            status = "regression"
        elif improved_case:
            status = "improvement"
        else:
            recall_delta = metric_deltas.get("recall_at_k_final", 0.0)
            status = "regression" if recall_delta < -0.02 else "improvement" if recall_delta > 0.02 else "unchanged"
        if status == "regression":
            has_regression = True
        case_deltas.append(CaseDelta(
            id=case_id, question=after["question"], baseline_passed=before["passed"], current_passed=after["passed"],
            status=status, metric_deltas=metric_deltas, current_failure_reasons=_failure_reasons(after),
        ))

    # Every case erroring in the current run, whether or not it existed in
    # the baseline. A case that is NEW and errors would otherwise be filed
    # as status="new" and disappear from the regression list entirely - an
    # error is never an acceptable outcome, so it counts. A case that
    # already errored in the baseline is listed for visibility but does not
    # re-trigger has_regression on every future comparison.
    errored_case_ids = sorted(cid for cid, case in current_cases.items() if case.get("error"))
    newly_errored_case_ids = sorted(
        cid for cid in errored_case_ids
        if cid not in baseline_cases or not baseline_cases[cid].get("error")
    )

    population_changed = _mean_population_changed(baseline, current, dataset_delta)
    aggregate_deltas = _aggregate_deltas(baseline, current, tolerance, population_changed)
    status_deltas = _status_deltas(baseline, current)
    latency_deltas = _latency_deltas(baseline, current, latency_tolerance_ms)
    if any(d["status"] == "regression" for d in aggregate_deltas.values()):
        has_regression = True
    if any(d["status"] == "regression" for d in status_deltas.values()):
        has_regression = True
    if any(d["status"] == "regression" for d in latency_deltas.values()):
        has_regression = True
    if newly_errored_case_ids:
        has_regression = True

    return ComparisonReport(
        baseline_meta={"timestamp": baseline.get("timestamp"), "git_sha": baseline.get("git_sha")},
        current_meta={"timestamp": current.get("timestamp"), "git_sha": current.get("git_sha")},
        aggregate_deltas=aggregate_deltas,
        case_deltas=case_deltas,
        has_regression=has_regression,
        environment_mismatch=environment_mismatch,
        status_deltas=status_deltas,
        latency_deltas=latency_deltas,
        environment_notes=environment_notes,
        dataset_delta=dataset_delta,
        errored_case_ids=errored_case_ids,
        newly_errored_case_ids=newly_errored_case_ids,
        mean_metrics_comparable=not population_changed,
    )
