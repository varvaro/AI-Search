"""CLI entry point.

    python -m benchmark run [--environment fixture|production] [--candidate-strategy legacy|union] [--expand-query [both|fts|vector]] [--k 10] [--include-answer] [--dataset PATH ...]
    python -m benchmark compare BASELINE.json CURRENT.json [--out-dir DIR]
    python -m benchmark report RUN.json [--out-dir DIR]
    python -m benchmark pr74-run [--environment fixture|production] [--dataset PATH] [--no-llm-replay] [--report]
    python -m benchmark acceptance-run [--environment fixture|production] [--dataset PATH] [--report]

`pr74-run` measures SAFETY (can the gate be made to lie about a contract).
`acceptance-run` measures PRODUCT VALUE (does the tool find the document and
answer the question). They are separate commands with separate verdicts on
purpose - a green safety run is not evidence the tool is usable.

See benchmark/README.md for a full walkthrough.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import compare as compare_mod
from . import report as report_mod
from . import runner
from . import pr74_runner
from . import acceptance_runner


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


def _cmd_pr74_run(args: argparse.Namespace) -> int:
    run = pr74_runner.run_pr74_benchmark(
        environment_name=args.environment,
        dataset_path=Path(args.dataset) if args.dataset else None,
        llm_replay=not args.no_llm_replay,
        result_count=args.k,
        case_filter=set(args.case) if args.case else None,
    )
    path = pr74_runner.save_pr74_run(run, Path(args.output) if args.output else None)
    go = run.go_nogo
    print(f"Saved PR7.4 artifact: {path}")
    print(f"cases: {run.case_count}  verdict: {go.get('verdict')}  "
          f"blocking: {go.get('has_blocking_regression')}  "
          f"llm_replay: {run.llm_replay}")
    agg = run.aggregate
    print(f"  state_verdict_accuracy: {agg.get('state_verdict_accuracy')}")
    print(f"  false_signed_confirmations: {agg.get('false_signed_confirmations')}")
    print(f"  wrong_entity_citations: {agg.get('wrong_entity_citations')}")
    print(f"  unchanged/improved/degraded: "
          f"{agg.get('unchanged_count')}/{agg.get('improved_count')}/{agg.get('degraded_count')}")
    if args.report:
        out_dir = Path(args.report_dir)
        written = report_mod.write_reports(run.to_dict(), out_dir)
        print(f"Reports written to {out_dir}: {', '.join(str(p) for p in written.values())}")
    return 0 if go.get("verdict") == "GO" else 1


def _cmd_acceptance_run(args: argparse.Namespace) -> int:
    run = acceptance_runner.run_acceptance_benchmark(
        environment_name=args.environment,
        dataset_path=Path(args.dataset) if args.dataset else None,
        result_count=args.k,
        case_filter=set(args.case) if args.case else None,
        state_gate=not args.no_state_gate,
        validation=not args.no_validation,
        max_follow_ups=args.max_follow_ups,
    )
    path = acceptance_runner.save_acceptance_run(
        run, Path(args.output) if args.output else None,
    )
    agg = run.aggregate
    verdict = run.verdict
    print(f"Saved acceptance artifact: {path}")
    for warning in run.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print("\nAI Search Acceptance Report")
    print(f"  projekt / dataset:              {run.project_id} / {run.dataset_version or 'bez verze'}")
    print("\nFAT RESULT (automatické měření)")
    print(f"  počet testů:                    {agg.get('case_count')}")
    print(f"  document_found_rate:            {_rate(agg.get('document_found_rate'))}")
    print(f"  top1 / top5 accuracy:           {_rate(agg.get('top1_accuracy'))} / {_rate(agg.get('top5_accuracy'))}")
    print(f"  answer_correct_rate:            {_rate(agg.get('answer_correct_rate'))}")
    print(f"  citation_correct_rate:          {_rate(agg.get('citation_correct_rate'))}")
    print(f"  unsupported_claim_rate:         {_rate(agg.get('unsupported_claim_rate'))}")
    print(f"  forbidden_document_rate:        {_rate(agg.get('forbidden_document_rate'))}")
    print(f"  počet kritických chyb:          {agg.get('critical_error_count')}")
    print(f"  průměrný čas odpovědi:          {_ms(agg.get('mean_total_ms'))}")
    print(f"  průměrný počet dotazů:          {agg.get('mean_queries_to_answer')}")
    print("\nSAT STATUS (lidské ověření)")
    print(f"  ověřeno / čeká:                 {run.verified_cases_count} / {run.pending_cases_count}")
    print(f"  připraveno pro denní použití:   {'ANO' if run.sat_status.get('ready_for_daily_use') else 'NE'}")
    for blocker in run.sat_status.get("blockers") or []:
        print(f"    SAT blocker: {blocker}")
    print(f"\n  GO / NO-GO:                     {verdict.get('verdict')}")
    for blocker in verdict.get("blockers") or []:
        print(f"    blocker: {blocker}")
    for reason in verdict.get("inconclusive_reasons") or []:
        print(f"    inconclusive: {reason}")
    if args.report:
        out_dir = Path(args.report_dir)
        written = report_mod.write_reports(run.to_dict(), out_dir)
        print(f"Reports written to {out_dir}: {', '.join(str(p) for p in written.values())}")
    return 0 if verdict.get("verdict") == "GO" else 1


def _cmd_acceptance_validate(args) -> int:
    from . import dataset_validation
    from .dataset.schema import load_dataset, read_dataset_version

    dataset_path = Path(args.dataset)
    cases = load_dataset(dataset_path)
    db_path = None
    if not args.skip_index:
        from .environment import get_environment

        db_path = get_environment(args.environment).db_path

    report = dataset_validation.validate(cases, db_path)
    print(f"Dataset: {dataset_path.name} (verze {read_dataset_version(dataset_path) or 'neuvedena'})")
    print(f"  case:          {report.checked_case_count}")
    print(f"  index checked: {report.index_checked}")
    print(f"  errors:        {len(report.errors)}")
    print(f"  warnings:      {len(report.warnings)}")
    for error in report.errors:
        print(f"    ERROR   {error}", file=sys.stderr)
    for warning in report.warnings:
        print(f"    WARNING {warning}")
    return 0 if report.ok else 1


def _rate(value) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _ms(value) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


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

    pr74_parser = sub.add_parser(
        "pr74-run",
        help="PR7.4 answer-quality benchmark (baseline A vs candidate C over shared retrieval)",
    )
    pr74_parser.add_argument("--environment", choices=["fixture", "production"], default="fixture")
    pr74_parser.add_argument("--dataset", default=None, help="override pr74_answer_quality.jsonl")
    pr74_parser.add_argument("--k", type=int, default=10)
    pr74_parser.add_argument("--no-llm-replay", action="store_true",
                             help="disable Ollama response replay (two live LLM calls per case)")
    pr74_parser.add_argument("--case", action="append", help="run only these case ids (repeatable)")
    pr74_parser.add_argument("--output", help="explicit output path for the run artifact JSON")
    pr74_parser.add_argument("--report", action="store_true")
    pr74_parser.add_argument("--report-dir", default=str(Path(__file__).parent / "reports" / "pr74-latest"))
    pr74_parser.set_defaults(func=_cmd_pr74_run)

    acceptance_parser = sub.add_parser(
        "acceptance-run",
        help="AI Search Acceptance Test (product value: found document, correct answer, critical errors)",
    )
    acceptance_parser.add_argument("--environment", choices=["fixture", "production"], default="fixture")
    acceptance_parser.add_argument("--dataset", default=None, help="override acceptance_v1.jsonl")
    acceptance_parser.add_argument("--k", type=int, default=10)
    acceptance_parser.add_argument("--case", action="append", help="run only these case ids (repeatable)")
    acceptance_parser.add_argument("--no-state-gate", action="store_true",
                                   help="certify with DOCUMENT_STATE_GATE_ENABLED off")
    acceptance_parser.add_argument("--no-validation", action="store_true",
                                   help="certify with EVIDENCE_RUNTIME_VALIDATION_ENABLED off")
    acceptance_parser.add_argument("--max-follow-ups", type=int, default=None,
                                   help="cap follow-up questions per case (default: all)")
    acceptance_parser.add_argument("--output", help="explicit output path for the run artifact JSON")
    acceptance_parser.add_argument("--report", action="store_true")
    acceptance_parser.add_argument("--report-dir", default=str(Path(__file__).parent / "reports" / "acceptance-latest"))
    acceptance_parser.set_defaults(func=_cmd_acceptance_run)

    validate_parser = sub.add_parser(
        "acceptance-validate",
        help="validate an acceptance dataset (structure + every expected document really in the index)",
    )
    validate_parser.add_argument("--dataset", required=True, help="path to the acceptance .jsonl")
    validate_parser.add_argument(
        "--environment", choices=["fixture", "production"], default="production",
        help="index to resolve expected_document against (default: production)",
    )
    validate_parser.add_argument(
        "--skip-index", action="store_true",
        help="structural checks only - do not open the index",
    )
    validate_parser.set_defaults(func=_cmd_acceptance_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
