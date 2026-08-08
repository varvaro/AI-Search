"""CLI entry point.

    python -m benchmark run [--environment fixture|production] [--candidate-strategy legacy|union] [--expand-query [both|fts|vector]] [--k 10] [--include-answer] [--dataset PATH ...]
    python -m benchmark compare BASELINE.json CURRENT.json [--out-dir DIR]
    python -m benchmark report RUN.json [--out-dir DIR]

See benchmark/README.md for a full walkthrough.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import compare as compare_mod
from . import report as report_mod
from . import runner


def _cmd_run(args: argparse.Namespace) -> int:
    dataset_paths = [Path(p) for p in args.dataset] if args.dataset else None
    run = runner.run_benchmark(
        environment_name=args.environment,
        dataset_paths=dataset_paths,
        result_count=args.k,
        include_answer=args.include_answer,
        verify_consistency=not args.no_consistency_check,
        candidate_strategy=args.candidate_strategy,
        expand_query=args.expand_query,
    )
    path = runner.save_run(run, Path(args.output) if args.output else None)
    print(f"Saved run artifact: {path} (candidate_strategy={run.candidate_strategy} expand_query={run.expand_query})")
    agg = run.aggregate
    print(f"cases: {run.case_count}  passed: {agg.get('passed')}  failed: {agg.get('failed')}  errored: {agg.get('errored')}")
    for key, value in agg.items():
        if key.startswith("mean_") and value is not None:
            print(f"  {key}: {value:.3f}")
    if args.report:
        out_dir = Path(args.report_dir)
        written = report_mod.write_reports(run.to_dict(), out_dir)
        print(f"Reports written to {out_dir}: {', '.join(str(p) for p in written.values())}")
    drift = agg.get("drift_detected_count", 0)
    if drift:
        print(f"WARNING: {drift} case(s) hit PipelineDriftError - see run artifact for details.", file=sys.stderr)
        return 2
    return 0 if agg.get("failed", 0) == 0 and agg.get("errored", 0) == 0 else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = compare_mod.load_run(Path(args.baseline))
    current = compare_mod.load_run(Path(args.current))
    try:
        comparison = compare_mod.compare_runs(baseline, current, strict_environment=not args.allow_environment_mismatch)
    except compare_mod.EnvironmentMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Re-run both baseline and current against the SAME index, or pass --allow-environment-mismatch "
              "to force a (clearly-flagged) comparison anyway.", file=sys.stderr)
        return 3
    print(report_mod.render_markdown_comparison(comparison.to_dict()))
    if args.report_dir:
        current_run = current
        report_mod.write_reports(current_run, Path(args.report_dir), comparison=comparison.to_dict())
    return 1 if comparison.has_regression else 0


def _cmd_report(args: argparse.Namespace) -> int:
    run = compare_mod.load_run(Path(args.run))
    written = report_mod.write_reports(run, Path(args.out_dir))
    print(f"Reports written: {', '.join(str(p) for p in written.values())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run the benchmark dataset against an environment")
    run_parser.add_argument("--environment", choices=["fixture", "production"], default="fixture")
    run_parser.add_argument("--candidate-strategy", choices=["legacy", "union", "union_ce"], default="legacy", help="candidate pool architecture / phase-3 scoring fed into search() (see ai_search.search()'s candidate_strategy)")
    run_parser.add_argument("--expand-query", nargs="?", const="both", default=False, choices=["both", "fts", "vector"], help="enable the Query Understanding layer (query_expansion.py) for this run; the two injection branches can be measured separately. Baseline vs experiment A/B: run once without and once with, then `python -m benchmark compare BASELINE.json EXPANDED.json`")
    run_parser.add_argument("--k", type=int, default=10, help="top-k used for recall/precision/MRR/nDCG (default: 10)")
    run_parser.add_argument("--include-answer", action="store_true", help="also run the final LLM-answer stage (slow, needs a live Ollama server)")
    run_parser.add_argument("--dataset", action="append", help="override dataset file(s); repeatable. Defaults to the environment's dataset file(s)")
    run_parser.add_argument("--output", help="explicit output path for the run artifact JSON (default: benchmark/runs/<timestamp>_<sha>_<env>.json)")
    run_parser.add_argument("--no-consistency-check", action="store_true", help="skip the pipeline-drift consistency check (faster, but phase-level numbers are unverified)")
    run_parser.add_argument("--report", action="store_true", help="also render Markdown/HTML/CSV reports")
    run_parser.add_argument("--report-dir", default=str(Path(__file__).parent / "reports" / "latest"))
    run_parser.set_defaults(func=_cmd_run)

    compare_parser = sub.add_parser("compare", help="compare two run artifacts and flag regressions")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("current")
    compare_parser.add_argument("--report-dir", default=None)
    compare_parser.add_argument("--allow-environment-mismatch", action="store_true", help="compare even if baseline/current were produced against different indexes (db_path/doc_count/chunk_count/index_fingerprint); the mismatch is still reported in the output")
    compare_parser.set_defaults(func=_cmd_compare)

    report_parser = sub.add_parser("report", help="render reports for a single run artifact")
    report_parser.add_argument("run")
    report_parser.add_argument("--out-dir", default=str(Path(__file__).parent / "reports" / "latest"))
    report_parser.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
