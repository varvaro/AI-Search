"""Retrieval regression suite: 20 production lookups, one run, one verdict.

    python -m benchmark.run_retrieval_regression                # run + compare to baseline
    python -m benchmark.run_retrieval_regression --no-baseline  # run only
    python -m benchmark.run_retrieval_regression --update-baseline

Why this lives next to `python -m benchmark run` instead of inside it: that
command answers "how good is the whole pipeline, phase by phase" and drives
`pipeline_trace.py`, which re-runs retrieval with instrumentation and computes
~15 aggregate metrics. This one answers a narrower question - "did any of the
20 known queries move?" - and therefore calls exactly what the user's browser
calls, `ui_services.search_all()`, once per query. Same entry point as the
2026-08-06/07 diagnostics, so the numbers in this suite's baseline are
directly comparable with the measurements those produced. The metric math and
the environment wiring are NOT duplicated here: they come from
benchmark.metrics and benchmark.environment.

Read-only, like the rest of the package: it opens the existing index for
search and never syncs, writes or deletes anything in SQLite/LanceDB. The only
file it writes is its own run artifact (and, with --update-baseline, the
baseline).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_search_config  # noqa: E402
import ui_services  # noqa: E402

from . import metrics  # noqa: E402
from .dataset.schema import BenchmarkCase, load_dataset  # noqa: E402
from .environment import Environment, production_environment  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
DATASET_PATH = PACKAGE_DIR / "dataset" / "retrieval_regression.jsonl"
BASELINE_PATH = PACKAGE_DIR / "baselines" / "retrieval_regression.json"
RUNS_DIR = PACKAGE_DIR / "runs"
DEFAULT_K = 10

# Rank/recall/MRR/nDCG all derive from integer ranks, so any real movement is
# far larger than this - the tolerance only absorbs float round-trips through
# JSON, never a genuine one-position change.
EPSILON = 1e-6


def baseline_payload(run: dict) -> dict:
    """The run artifact minus `top_paths`.

    Run artifacts are gitignored precisely because they embed real Box document
    paths (see benchmark/runs/README.md). The baseline is the one artifact that
    must be committed, so it cannot carry 10 absolute paths per case - and does
    not need to: a comparison is computed from rank/recall/MRR/nDCG/hit/error
    alone. The paths stay in the (local, gitignored) run artifact for debugging
    a case that moved.
    """
    return dict(run, cases=[{k: v for k, v in case.items() if k != "top_paths"} for case in run["cases"]])


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_DIR.parent,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def evaluate_case(case: BenchmarkCase, environment: Environment, k: int = DEFAULT_K) -> dict:
    """One case through the real production search path.

    `is_question` is resolved by production's own classify_query(), and
    `expand_query` comes from the same config constant app.py passes, rather
    than either being hardcoded here: both select a different code path inside
    search_all(), so pinning them would measure a pipeline the UI never runs.
    """
    is_question = ui_services.classify_query(case.question)["mode"] == "otazka"
    started = time.perf_counter()
    error = ""
    results: list[dict] = []
    try:
        results = ui_services.search_all(case.question, environment.settings, environment.state_dir,
                                         environment.embeddings, is_question=is_question,
                                         expand_query=ai_search_config.QUERY_EXPANSION_MODE)
    except Exception as exc:  # noqa: BLE001 - one broken case must not abort the suite
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000

    needles = case.expected_documents
    rank = metrics.best_rank(results, needles)
    hit = rank is not None and rank < k
    return {
        "id": case.id,
        "domain": case.domain,
        "question": case.question,
        "type": case.type,
        "difficulty": case.difficulty,
        "expected_documents": needles,
        "expected_content_missing": case.expected_content_missing,
        "expected_retrieval_issue": case.expected_retrieval_issue,
        "notes": case.notes,
        "is_question": is_question,
        "error": error,
        "returned": len(results),
        "best_rank": rank,
        "hit": hit,
        "recall_at_k": metrics.recall_at_k(results, needles, k),
        "reciprocal_rank": metrics.reciprocal_rank(results, needles),
        "ndcg_at_k": metrics.ndcg_at_k(results, needles, k),
        "latency_ms": round(latency_ms, 1),
        "top_paths": [row.get("path", "") for row in results[:k]],
    }


def _aggregate(cases: list[dict], k: int) -> dict:
    """Headline numbers over the cases that can actually move.

    Cases flagged expected_content_missing are excluded: their ground truth is
    not in the index, so their guaranteed 0.0 would drag every mean down by a
    constant and hide real movement elsewhere. expected_retrieval_issue cases
    ARE included - the content is indexed, so their score is a genuine (bad)
    retrieval result that should improve when the underlying weakness is fixed.
    """
    scored = [c for c in cases if not c["expected_content_missing"]]
    n = len(scored) or 1
    return {
        "case_count": len(cases),
        "scored_case_count": len(scored),
        "content_missing_count": sum(1 for c in cases if c["expected_content_missing"]),
        "known_retrieval_issue_count": sum(1 for c in cases if c["expected_retrieval_issue"]),
        "errored_count": sum(1 for c in cases if c["error"]),
        "hit_count": sum(1 for c in scored if c["hit"]),
        "miss_count": sum(1 for c in scored if not c["hit"]),
        f"mean_recall_at_{k}": sum(c["recall_at_k"] for c in scored) / n,
        "mrr": metrics.mean_reciprocal_rank([c["reciprocal_rank"] for c in scored]),
        f"mean_ndcg_at_{k}": sum(c["ndcg_at_k"] for c in scored) / n,
        "mean_latency_ms": round(sum(c["latency_ms"] for c in cases) / (len(cases) or 1), 1),
        "max_latency_ms": round(max((c["latency_ms"] for c in cases), default=0.0), 1),
    }


def run_regression(environment: Environment | None = None, dataset_path: Path | None = None,
                   k: int = DEFAULT_K) -> dict:
    cases = load_dataset(dataset_path or DATASET_PATH)
    environment = environment or production_environment()
    results = [evaluate_case(case, environment, k) for case in cases]
    return {
        "suite": "retrieval_regression",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "k": k,
        "dataset": str((dataset_path or DATASET_PATH).name),
        # Recorded for the same reason as the index fingerprint: a baseline
        # measured under a different expansion mode is not comparable, and the
        # difference is invisible in the per-case numbers alone.
        "query_expansion_mode": ai_search_config.QUERY_EXPANSION_MODE,
        "environment": environment.describe(),
        "aggregate": _aggregate(results, k),
        "cases": results,
    }


# --------------------------------------------------------------- comparison

def compare_to_baseline(baseline: dict, current: dict, k: int = DEFAULT_K) -> dict:
    """Per-case verdicts plus a single has_regression flag.

    A case counts as REGRESSION on any of: HIT -> MISS, a worse rank, or a drop
    in recall/MRR/nDCG. Cases carrying a known-limitation flag are evaluated
    the same way and reported the same way, but are kept out of has_regression:
    the run must not go red for a limitation already diagnosed and accepted -
    that is what makes people stop reading the report. Their movement in the
    other direction is still surfaced, since "the known issue started passing"
    is exactly the event this suite exists to catch.
    """
    old = {c["id"]: c for c in baseline.get("cases", [])}
    new = {c["id"]: c for c in current.get("cases", [])}

    improved, regressed, unchanged, known_issue = [], [], [], []
    for case_id, now in new.items():
        before = old.get(case_id)
        if before is None:
            unchanged.append({"id": case_id, "verdict": "NEW", "detail": "not present in baseline"})
            continue
        reasons_bad, reasons_good = [], []
        if before["hit"] and not now["hit"]:
            reasons_bad.append("HIT -> MISS")
        if not before["hit"] and now["hit"]:
            reasons_good.append("MISS -> HIT")
        ob, nb = before["best_rank"], now["best_rank"]
        if ob is not None and nb is not None and ob != nb:
            (reasons_good if nb < ob else reasons_bad).append(f"rank #{ob + 1} -> #{nb + 1}")
        for metric, label in (("recall_at_k", "recall"), ("reciprocal_rank", "MRR"), ("ndcg_at_k", "nDCG")):
            delta = now[metric] - before[metric]
            if abs(delta) <= EPSILON:
                continue
            (reasons_good if delta > 0 else reasons_bad).append(f"{label} {before[metric]:.3f} -> {now[metric]:.3f}")
        if now["error"] and not before["error"]:
            reasons_bad.append(f"newly errored ({now['error']})")

        entry = {"id": case_id, "domain": now["domain"], "question": now["question"],
                 "old_rank": ob, "new_rank": nb,
                 "known_limitation": now["expected_content_missing"] or now["expected_retrieval_issue"],
                 "reasons": reasons_bad or reasons_good}
        if reasons_bad:
            regressed.append(entry)
        elif reasons_good:
            improved.append(entry)
        else:
            unchanged.append(entry)
        if now["expected_retrieval_issue"] or now["expected_content_missing"]:
            known_issue.append({"id": case_id, "question": now["question"], "hit": now["hit"],
                                "flag": "content_missing" if now["expected_content_missing"] else "retrieval_issue"})

    removed = [case_id for case_id in old if case_id not in new]
    blocking = [entry for entry in regressed if not entry["known_limitation"]]
    ob_agg, nb_agg = baseline.get("aggregate", {}), current.get("aggregate", {})
    aggregate_deltas = {key: round(nb_agg[key] - ob_agg[key], 4)
                        for key in (f"mean_recall_at_{k}", "mrr", f"mean_ndcg_at_{k}", "hit_count")
                        if key in ob_agg and key in nb_agg}
    return {
        "baseline_timestamp": baseline.get("timestamp"),
        "current_timestamp": current.get("timestamp"),
        "baseline_fingerprint": (baseline.get("environment") or {}).get("index_fingerprint"),
        "current_fingerprint": (current.get("environment") or {}).get("index_fingerprint"),
        "same_index": (baseline.get("environment") or {}).get("index_fingerprint") == (current.get("environment") or {}).get("index_fingerprint"),
        "baseline_query_expansion_mode": baseline.get("query_expansion_mode"),
        "current_query_expansion_mode": current.get("query_expansion_mode"),
        "same_query_expansion_mode": baseline.get("query_expansion_mode") == current.get("query_expansion_mode"),
        "aggregate_deltas": aggregate_deltas,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "removed_case_ids": removed,
        "known_limitation_cases": known_issue,
        "has_regression": bool(blocking),
        "blocking_regression_ids": [entry["id"] for entry in blocking],
    }


# ------------------------------------------------------------------ console

def render_console(run: dict, comparison: dict | None = None) -> str:
    k = run["k"]
    env = run["environment"]
    out = [
        f"Retrieval regression suite - {run['dataset']} ({run['aggregate']['case_count']} cases, k={k})",
        f"index: {env.get('doc_count'):,} docs / {env.get('chunk_count'):,} chunks   model: {env.get('embedding_model')}",
        f"fingerprint: {env.get('index_fingerprint')}",
        "",
        f"{'':<2} {'id':<32} {'domain':<20} {'rank':>6} {'R@'+str(k):>6} {'MRR':>6} {'nDCG':>6} {'ms':>7}",
    ]
    for case in run["cases"]:
        rank = "MISS" if not case["hit"] else f"#{case['best_rank'] + 1}"
        flag = "C" if case["expected_content_missing"] else ("R" if case["expected_retrieval_issue"] else " ")
        out.append(f"{flag:<2} {case['id']:<32} {case['domain']:<20} {rank:>6} "
                   f"{case['recall_at_k']:>6.2f} {case['reciprocal_rank']:>6.2f} {case['ndcg_at_k']:>6.3f} "
                   f"{case['latency_ms']:>7.0f}")
    agg = run["aggregate"]
    out += [
        "",
        "flags: C = expected_content_missing (MISS is correct)   R = expected_retrieval_issue (known, non-blocking)",
        "",
        f"scored cases      {agg['scored_case_count']} of {agg['case_count']} "
        f"({agg['content_missing_count']} excluded: content not in index)",
        f"HIT / MISS        {agg['hit_count']} / {agg['miss_count']}",
        f"mean recall@{k}    {agg[f'mean_recall_at_{k}']:.3f}",
        f"MRR               {agg['mrr']:.3f}",
        f"mean nDCG@{k}      {agg[f'mean_ndcg_at_{k}']:.3f}",
        f"latency           mean {agg['mean_latency_ms']:.0f} ms   max {agg['max_latency_ms']:.0f} ms",
    ]
    if agg["errored_count"]:
        out.append(f"ERRORED           {agg['errored_count']}")

    if comparison is None:
        return "\n".join(out)

    out += ["", "-" * 78, f"vs baseline {comparison['baseline_timestamp']}"]
    if not comparison["same_index"]:
        out.append("WARNING: baseline and current were measured against DIFFERENT indexes "
                   "(fingerprint differs) - rank changes may come from the index, not from code.")
    if not comparison["same_query_expansion_mode"]:
        out.append(f"WARNING: query expansion mode changed "
                   f"({comparison['baseline_query_expansion_mode']!r} -> {comparison['current_query_expansion_mode']!r}) "
                   f"- every rank change below may come from that alone.")
    for key, delta in comparison["aggregate_deltas"].items():
        out.append(f"  {key:<20} {delta:+.3f}")
    for label, entries in (("IMPROVED", comparison["improved"]), ("REGRESSION", comparison["regressed"])):
        if not entries:
            continue
        out.append(f"\n{label}: {len(entries)}")
        for entry in entries:
            note = "  [known limitation, non-blocking]" if entry["known_limitation"] else ""
            out.append(f"  {entry['id']:<32} {', '.join(entry['reasons'])}{note}")
    if comparison["removed_case_ids"]:
        out.append(f"\nremoved from dataset since baseline: {', '.join(comparison['removed_case_ids'])}")
    out.append("")
    out.append("VERDICT: REGRESSION - " + ", ".join(comparison["blocking_regression_ids"])
               if comparison["has_regression"] else "VERDICT: no blocking regression")
    return "\n".join(out)


# --------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.run_retrieval_regression",
                                     description="Run the 20-query retrieval regression suite against the production index.")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help="baseline run artifact to compare against")
    parser.add_argument("--no-baseline", action="store_true", help="run only, skip the baseline comparison")
    parser.add_argument("--update-baseline", action="store_true",
                        help="overwrite the baseline with this run (do this only after verifying the change is intended)")
    parser.add_argument("--output", help="path for the run artifact JSON (default: benchmark/runs/retrieval_regression_<timestamp>.json)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    run = run_regression(dataset_path=Path(args.dataset), k=args.k)

    comparison = None
    baseline_path = Path(args.baseline)
    if not args.no_baseline:
        if baseline_path.exists():
            comparison = compare_to_baseline(json.loads(baseline_path.read_text(encoding="utf-8")), run, args.k)
        else:
            print(f"note: no baseline at {baseline_path} - run with --update-baseline to create one.", file=sys.stderr)

    print(render_console(run, comparison))

    output = Path(args.output) if args.output else RUNS_DIR / f"retrieval_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(run)
    if comparison is not None:
        payload["comparison"] = comparison
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrun artifact: {output}")

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline_payload(run), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"baseline updated: {baseline_path}")

    if run["aggregate"]["errored_count"]:
        return 2
    return 1 if comparison is not None and comparison["has_regression"] else 0


if __name__ == "__main__":
    sys.exit(main())
