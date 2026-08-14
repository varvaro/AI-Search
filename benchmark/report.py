"""Renders a run artifact (and, optionally, a comparison against a baseline)
as Markdown, HTML and CSV. No new third-party dependencies (no matplotlib/
pandas) - the HTML report's bar charts are plain CSS `width: N%` divs, since
this only ever needs to be readable in a local browser, not published.
"""
from __future__ import annotations

import csv
import html
import io
from pathlib import Path

STAGE_ORDER = [
    "intent_detection", "query_parsing", "fts_retrieval", "vector_retrieval",
    "fusion_rrf", "candidate_pool", "reranker", "diversification", "prompt_builder", "final_answer",
]


def _pct(value) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "-"


def _ms(value) -> str:
    return f"{value:.0f}" if isinstance(value, (int, float)) else "-"


def _metric_value(key: str, value) -> str:
    """Renders an aggregate metric in its own unit. Every mean_* used to go
    through _pct(), which turned a 45 ms cross-encoder latency into "4500.0%"
    and a 3-term query expansion into "300.0%" - a report is not allowed to
    present a count or a duration as if it were a rate."""
    if not isinstance(value, (int, float)):
        return "-"
    if key.endswith("_ms"):
        return f"{value:.0f} ms"
    if key.endswith("_count"):
        return f"{value:.2f}"
    return _pct(value)


def render_markdown_run(run: dict) -> str:
    lines = [
        f"# Benchmark run - {run['environment']['name']} ({run['timestamp']})",
        "",
        f"- git sha: `{run.get('git_sha') or 'n/a'}`",
        f"- dataset: {', '.join(Path(p).name for p in run['dataset_files'])} ({run['case_count']} cases)",
        f"- index: {run['environment'].get('doc_count')} docs / {run['environment'].get('chunk_count')} chunks, embedding model `{run['environment'].get('embedding_model')}`",
        f"- index fingerprint: `{(run['environment'].get('index_fingerprint') or 'n/a')}`"
        f" ({run['environment'].get('index_fingerprint_algorithm') or 'legacy'},"
        f" last indexed_at {run['environment'].get('index_max_indexed_at') or 'n/a'})",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    agg = run.get("aggregate", {})
    lines.append(f"| passed / failed / errored | {agg.get('passed')} / {agg.get('failed')} / {agg.get('errored')} |")
    for key, value in agg.items():
        if key.startswith("mean_"):
            lines.append(f"| {key} | {_metric_value(key, value)} |")

    lines += ["", "## Pipeline funnel (mean pool size per stage)", "", "| Stage | Mean pool size |", "|---|---|"]
    stage_totals: dict[str, list[int]] = {}
    for case in run["cases"]:
        for stage, size in case.get("metrics", {}).get("stage_pool_size", {}).items():
            stage_totals.setdefault(stage, []).append(size)
    for stage in STAGE_ORDER:
        sizes = stage_totals.get(stage)
        if sizes:
            lines.append(f"| {stage} | {sum(sizes) / len(sizes):.1f} |")

    latency = agg.get("latency", {})
    if latency.get("retrieval_total_ms") or latency.get("final_answer_ms"):
        lines += ["", "## Latency", "", "| Stage | Mean ms | p95 ms | Min ms | Max ms | n |", "|---|---|---|---|---|---|"]

        def latency_row(label: str, stats: dict) -> str:
            # min/max are read with .get() so a report can still be rendered
            # from a run artifact written before they were recorded.
            return (f"| {label} | {stats['mean_ms']:.0f} | {stats['p95_ms']:.0f} | "
                    f"{_ms(stats.get('min_ms'))} | {_ms(stats.get('max_ms'))} | {stats['n']} |")

        for key in ("retrieval_total_ms", "final_answer_ms"):
            stats = latency.get(key)
            if stats:
                lines.append(latency_row(key, stats))
        for stage, stats in (latency.get("by_stage_ms") or {}).items():
            if stats:
                lines.append(latency_row(f"stage: {stage}", stats))

    lines += ["", "## Cases", "", "| id | type | passed | recall@k | forbidden_free_rate | MRR | nDCG@k | reasons |", "|---|---|---|---|---|---|---|---|"]
    for case in run["cases"]:
        m = case.get("metrics", {})
        status = "✅" if case["passed"] else ("💥" if case.get("error") else "❌")
        reasons = "; ".join(case.get("failure_reasons", [])) or (case.get("error") or "")
        lines.append(
            f"| {case['id']} | {case['type']} | {status} | {_pct(m.get('recall_at_k_final'))} | "
            f"{_pct(m.get('forbidden_free_rate'))} | {_pct(m.get('mrr_final'))} | {_pct(m.get('ndcg_at_k_final'))} | {reasons} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown_comparison(comparison: dict) -> str:
    lines = [
        "# Benchmark comparison",
        "",
        f"- baseline: `{comparison['baseline_meta'].get('git_sha') or comparison['baseline_meta'].get('timestamp')}`",
        f"- current:  `{comparison['current_meta'].get('git_sha') or comparison['current_meta'].get('timestamp')}`",
        f"- **regression detected: {'YES ⚠️' if comparison['has_regression'] else 'no'}**",
    ]
    if comparison.get("environment_mismatch"):
        lines += ["", "**⚠️ ENVIRONMENT MISMATCH - these runs were produced against different indexes, metrics below are NOT reliably comparable:**"]
        lines += [f"- {m}" for m in comparison["environment_mismatch"]]
    if comparison.get("environment_notes"):
        # "No mismatch found" is not the same as "proven identical" - spell
        # out which identity evidence was simply unavailable.
        lines += ["", "**Environment identity not fully verified:**"]
        lines += [f"- {n}" for n in comparison["environment_notes"]]

    dataset = comparison.get("dataset_delta") or {}
    if dataset.get("population_changed"):
        lines += [
            "",
            f"**⚠️ DATASET CHANGED - {dataset['baseline_case_count']} → {dataset['current_case_count']} cases "
            f"({len(dataset['new_case_ids'])} new, {len(dataset['removed_case_ids'])} removed, "
            f"{dataset['common_case_count']} in common). Aggregate means below are NOT comparable.**",
        ]
        if dataset["removed_case_ids"]:
            lines.append(f"- removed: {', '.join(dataset['removed_case_ids'])}")
        if dataset["new_case_ids"]:
            lines.append(f"- new: {', '.join(dataset['new_case_ids'])}")
    if comparison.get("errored_case_ids"):
        newly = set(comparison.get("newly_errored_case_ids") or [])
        listed = ", ".join(f"{cid}{' (NEW)' if cid in newly else ''}" for cid in comparison["errored_case_ids"])
        lines += ["", f"**💥 Errored cases in current run: {listed}**"]

    lines += [
        "",
        "## Aggregate deltas",
        "",
        "| Metric | Before | After | Delta | Status |",
        "|---|---|---|---|---|",
    ]
    if not comparison.get("mean_metrics_comparable", True):
        lines.append("<!-- means averaged over different case populations - deltas are informational only -->")
    for key, delta in comparison["aggregate_deltas"].items():
        marker = {"regression": "❌", "improvement": "✅", "unchanged": "•",
                  "not_comparable": "⚠️", "informational": "ℹ️"}.get(delta["status"], "•")
        lines.append(f"| {key} | {_metric_value(key, delta['before'])} | {_metric_value(key, delta['after'])} | {delta['delta']:+.3f} | {marker} {delta['status']} |")

    if comparison.get("status_deltas"):
        lines += ["", "## Run status (passed / failed / errored)", "", "| Count | Before | After | Delta | Status |", "|---|---|---|---|---|"]
        for key, delta in comparison["status_deltas"].items():
            marker = {"regression": "❌", "improvement": "✅", "unchanged": "•"}.get(delta["status"], "•")
            lines.append(f"| {key} | {delta['before']} | {delta['after']} | {delta['delta']:+d} | {marker} {delta['status']} |")

    if comparison.get("latency_deltas"):
        lines += ["", "## Latency deltas", "", "| Stage | Before ms | After ms | Delta ms | Status |", "|---|---|---|---|---|"]
        for key, delta in comparison["latency_deltas"].items():
            marker = {"regression": "❌", "improvement": "✅", "unchanged": "•"}.get(delta["status"], "•")
            lines.append(f"| {key} | {delta['before']:.0f} | {delta['after']:.0f} | {delta['delta']:+.0f} | {marker} {delta['status']} |")

    regressions = [d for d in comparison["case_deltas"] if d["status"] == "regression"]
    improvements = [d for d in comparison["case_deltas"] if d["status"] == "improvement"]
    lines += ["", f"## Top regressions ({len(regressions)})", ""]
    if regressions:
        lines += ["| id | question | before | after | reasons |", "|---|---|---|---|---|"]
        for d in regressions:
            lines.append(f"| {d['id']} | {d['question'][:60]} | {d['baseline_passed']} | {d['current_passed']} | {'; '.join(d['current_failure_reasons'])} |")
    else:
        lines.append("_none_")

    lines += ["", f"## Top improvements ({len(improvements)})", ""]
    if improvements:
        lines += ["| id | question | before | after |", "|---|---|---|---|"]
        for d in improvements:
            lines.append(f"| {d['id']} | {d['question'][:60]} | {d['baseline_passed']} | {d['current_passed']} |")
    else:
        lines.append("_none_")
    return "\n".join(lines) + "\n"


def render_csv_run(run: dict) -> str:
    buffer = io.StringIO()
    fieldnames = ["id", "type", "difficulty", "passed", "recall_at_k_final", "forbidden_free_rate", "mrr_final", "ndcg_at_k_final", "pool_survival_rate", "duplicate_count_final", "distinct_ratio_final", "failure_reasons"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for case in run["cases"]:
        m = case.get("metrics", {})
        writer.writerow({
            "id": case["id"], "type": case["type"], "difficulty": case["difficulty"], "passed": case["passed"],
            "recall_at_k_final": m.get("recall_at_k_final"), "forbidden_free_rate": m.get("forbidden_free_rate"),
            "mrr_final": m.get("mrr_final"), "ndcg_at_k_final": m.get("ndcg_at_k_final"),
            "pool_survival_rate": m.get("pool_survival_rate"), "duplicate_count_final": m.get("duplicate_count_final"),
            "distinct_ratio_final": m.get("distinct_ratio_final"), "failure_reasons": "; ".join(case.get("failure_reasons", [])),
        })
    return buffer.getvalue()


def _bar(label: str, value: float, max_value: float, color: str = "#3b82f6") -> str:
    width = (value / max_value * 100) if max_value else 0
    return (
        f'<div style="margin:4px 0"><div style="font:12px sans-serif;color:#333">{html.escape(label)} ({value:.1f})</div>'
        f'<div style="background:#eee;border-radius:3px;overflow:hidden"><div style="background:{color};height:14px;width:{width:.1f}%"></div></div></div>'
    )


def render_html_run(run: dict) -> str:
    stage_totals: dict[str, list[int]] = {}
    for case in run["cases"]:
        for stage, size in case.get("metrics", {}).get("stage_pool_size", {}).items():
            stage_totals.setdefault(stage, []).append(size)
    means = {stage: sum(v) / len(v) for stage, v in stage_totals.items() if v}
    max_pool = max(means.values()) if means else 1

    funnel_html = "".join(_bar(stage, means[stage], max_pool) for stage in STAGE_ORDER if stage in means)

    rows_html = ""
    for case in run["cases"]:
        m = case.get("metrics", {})
        color = "#16a34a" if case["passed"] else "#dc2626"
        reasons = html.escape("; ".join(case.get("failure_reasons", [])) or (case.get("error") or ""))
        rows_html += (
            f'<tr><td>{html.escape(case["id"])}</td><td>{html.escape(case["type"])}</td>'
            f'<td style="color:{color};font-weight:600">{"PASS" if case["passed"] else "FAIL"}</td>'
            f'<td>{_pct(m.get("recall_at_k_final"))}</td><td>{_pct(m.get("forbidden_free_rate"))}</td>'
            f'<td>{_pct(m.get("mrr_final"))}</td><td>{_pct(m.get("ndcg_at_k_final"))}</td><td>{reasons}</td></tr>'
        )

    agg = run.get("aggregate", {})
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AI Search benchmark - {html.escape(run['environment']['name'])}</title>
<style>body{{font-family:sans-serif;max-width:1000px;margin:24px auto;color:#111}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left}}
th{{background:#f5f5f5}}</style></head>
<body>
<h1>Benchmark run - {html.escape(run['environment']['name'])}</h1>
<p>{html.escape(run['timestamp'])} · git {html.escape(run.get('git_sha') or 'n/a')} · {run['case_count']} cases ·
passed {agg.get('passed')} / failed {agg.get('failed')} / errored {agg.get('errored')}</p>
<h2>Pipeline funnel (mean pool size per stage)</h2>
{funnel_html}
<h2>Cases</h2>
<table><tr><th>id</th><th>type</th><th>status</th><th>recall@k</th><th>forbidden_free_rate</th><th>MRR</th><th>nDCG@k</th><th>reasons</th></tr>
{rows_html}</table>
</body></html>
"""


def write_reports(run: dict, out_dir: Path, comparison: dict | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if run.get("verdict") is not None and run.get("flag_matrix") is None:
        # Acceptance (product) artifact.
        (out_dir / "acceptance.md").write_text(render_markdown_acceptance(run), encoding="utf-8")
        written["acceptance_markdown"] = out_dir / "acceptance.md"
        (out_dir / "acceptance.csv").write_text(render_csv_acceptance(run), encoding="utf-8")
        written["acceptance_csv"] = out_dir / "acceptance.csv"
        return written
    if run.get("flag_matrix") is not None or run.get("go_nogo") is not None:
        # PR7.4 answer-quality artifact — dedicated renderers, same out_dir.
        (out_dir / "pr74.md").write_text(render_markdown_pr74(run), encoding="utf-8")
        written["pr74_markdown"] = out_dir / "pr74.md"
        (out_dir / "pr74.csv").write_text(render_csv_pr74(run), encoding="utf-8")
        written["pr74_csv"] = out_dir / "pr74.csv"
        return written
    (out_dir / "run.md").write_text(render_markdown_run(run), encoding="utf-8")
    written["markdown"] = out_dir / "run.md"
    (out_dir / "run.html").write_text(render_html_run(run), encoding="utf-8")
    written["html"] = out_dir / "run.html"
    (out_dir / "run.csv").write_text(render_csv_run(run), encoding="utf-8")
    written["csv"] = out_dir / "run.csv"
    if comparison is not None:
        (out_dir / "comparison.md").write_text(render_markdown_comparison(comparison), encoding="utf-8")
        written["comparison_markdown"] = out_dir / "comparison.md"
    return written


def render_markdown_pr74(run: dict) -> str:
    """Markdown report for a PR7.4 answer-quality run artifact."""
    agg = run.get("aggregate") or {}
    go = run.get("go_nogo") or {}
    latency = run.get("latency") or {}
    flags = run.get("flags_constant") or {}
    matrix = run.get("flag_matrix") or {}
    lines = [
        f"# PR7.4 Answer Quality — {run.get('environment', {}).get('name')} ({run.get('timestamp')})",
        "",
        f"- git sha: `{run.get('git_sha') or 'n/a'}`",
        f"- dataset: `{Path(run.get('dataset_file') or '').name}` ({run.get('case_count')} cases)",
        f"- llm_replay: `{run.get('llm_replay')}`",
        f"- environment: {run.get('environment', {}).get('doc_count')} docs / "
        f"{run.get('environment', {}).get('chunk_count')} chunks",
        f"- index fingerprint: `{(run.get('environment') or {}).get('index_fingerprint') or 'n/a'}`",
        "",
        "## GO / NO-GO",
        "",
        f"**Verdict: {go.get('verdict', 'n/a')}**",
        f"- has_blocking_regression: `{go.get('has_blocking_regression')}`",
        f"- blocking_case_ids: {go.get('blocking_case_ids') or []}",
    ]
    for reason in go.get("reasons") or []:
        lines.append(f"- reason: {reason}")
    crit = go.get("criteria") or {}
    if crit:
        lines += ["", "| Criterion | Pass |", "|---|---|"]
        for key, value in crit.items():
            lines.append(f"| {key} | {'✅' if value else '❌'} |")

    lines += [
        "",
        "## Flags",
        "",
        f"- AUXILIARY_TERM_COVERAGE_ENABLED (constant): `{flags.get('AUXILIARY_TERM_COVERAGE_ENABLED')}`",
        f"- MULTI_QUERY_RETRIEVAL_ENABLED (constant): `{flags.get('MULTI_QUERY_RETRIEVAL_ENABLED')}`",
        "",
        "| Mode | STATE_GATE | VALIDATION |",
        "|---|---|---|",
    ]
    for mode in ("A", "B", "C", "D"):
        m = matrix.get(mode) or {}
        lines.append(
            f"| {mode} | {m.get('DOCUMENT_STATE_GATE_ENABLED')} | "
            f"{m.get('EVIDENCE_RUNTIME_VALIDATION_ENABLED')} |"
        )

    by_layer = agg.get("by_layer") or {}
    first_layer = agg.get("first_failure_layer_counts") or {}

    def layer_row(layer: str) -> str:
        stats = by_layer.get(layer) or {}
        return (
            f"| {layer} | {stats.get('cases', 0)} | {stats.get('failures', 0)} | "
            f"{stats.get('blocking', 0)} | {first_layer.get(layer, 0)} |"
        )

    lines += [
        "",
        "## Where it broke",
        "",
        "Failures attributed to the earliest pipeline layer responsible. "
        "`first` counts cases whose earliest failing layer is this one — start debugging there.",
        "",
        "| Layer | Cases | Failures | Blocking | First |",
        "|---|---|---|---|---|",
        layer_row("RETRIEVAL"),
        layer_row("EVIDENCE"),
        layer_row("ANSWER"),
        layer_row("SAFETY"),
        "",
        "## 1. Retrieval quality",
        "",
        "Did the evidence reach the pool at all? A failure here is a ranking problem, "
        "not a gate problem.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| cases with a retrieval failure | {(by_layer.get('RETRIEVAL') or {}).get('cases', 0)} |",
        f"| expected_source_not_retrieved | {((by_layer.get('RETRIEVAL') or {}).get('codes') or {}).get('expected_source_not_retrieved', 0)} |",
        f"| forbidden_source_retrieved (informational) | {((by_layer.get('RETRIEVAL') or {}).get('codes') or {}).get('forbidden_source_retrieved', 0)} |",
        "",
        "## 2. Answer quality",
        "",
        "Given correct evidence, did the answer use it and stay stable?",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| expected_source_not_cited | {((by_layer.get('ANSWER') or {}).get('codes') or {}).get('expected_source_not_cited', 0)} |",
        f"| hedge_incorrectly_rewritten | {agg.get('hedge_incorrectly_rewritten')} |",
        f"| unchanged | {agg.get('unchanged_count')} |",
        f"| improved | {agg.get('improved_count')} |",
        f"| degraded | {agg.get('degraded_count')} |",
        f"| changed_neutral | {agg.get('changed_neutral_count')} |",
        "",
        "### Evidence verdicts",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| state_verdict_accuracy | {_pct(agg.get('state_verdict_accuracy'))} |",
        f"| signed_confirmed_precision | {_pct(agg.get('signed_confirmed_precision'))} |",
        f"| signed_confirmed_recall | {_pct(agg.get('signed_confirmed_recall'))} |",
        f"| entity_mismatch_accuracy | {_pct(agg.get('entity_mismatch_accuracy'))} |",
        f"| unverified_accuracy | {_pct(agg.get('unverified_accuracy'))} |",
        f"| intent_coverage_accuracy | {_pct(agg.get('intent_coverage_accuracy'))} |",
        f"| missing_need_accuracy | {_pct(agg.get('missing_need_accuracy'))} |",
        "",
        "## 3. Safety",
        "",
        "Assertions judged against the documents the answer actually leaned on "
        "(cited ∪ state evidence), never against the whole retrieval pool.",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| false_signed_confirmations | {agg.get('false_signed_confirmations')} |",
        f"| wrong_entity_citations | {agg.get('wrong_entity_citations')} |",
        f"| unsupported_negative_signed_claims | {agg.get('unsupported_negative_signed_claims')} |",
        f"| unsupported_positive_signed_claims | {agg.get('unsupported_positive_signed_claims')} |",
        f"| blocking_count | {agg.get('blocking_count')} |",
        f"| high_count | {agg.get('high_count')} |",
        f"| medium_count | {agg.get('medium_count')} |",
        f"| low_count | {agg.get('low_count')} |",
        f"| safety_score (informational, never gates) | {agg.get('safety_score')} |",
        "",
        "## 4. Product usability",
        "",
        "This run measures SAFETY of the gate, not product value. "
        "Whether AI Search finds the right document and answers the question is measured by "
        "the acceptance benchmark (`python -m benchmark acceptance-run`) — a green run here "
        "is not evidence the tool is useful.",
    ]

    by_cat = agg.get("by_category") or {}
    if by_cat:
        lines += ["", "## By category", "", "| Category | n | unchanged | improved | degraded | blocking | state acc |", "|---|---|---|---|---|---|---|"]
        for cat, row in by_cat.items():
            lines.append(
                f"| {cat} | {row.get('case_count')} | {row.get('unchanged')} | {row.get('improved')} | "
                f"{row.get('degraded')} | {row.get('blocking')} | {_pct(row.get('state_verdict_accuracy'))} |"
            )

    if latency:
        lines += ["", "## Latency", "", f"- note: {latency.get('note') or ''}", ""]
        warm = latency.get("warmup_excluded_case_ids") or []
        if warm:
            lines.append(f"- warmup excluded: {warm}")
        lines += ["", "| Series | Mean ms | p95 ms | Min ms | Max ms | n |", "|---|---|---|---|---|---|"]
        for key in (
            "retrieval_ms", "live_answer_ms", "end_to_end_ms",
            "state_gate_delta_ms", "validation_delta_ms", "candidate_delta_ms",
        ):
            stats = latency.get(key)
            if not stats:
                continue
            lines.append(
                f"| {key} | {stats['mean_ms']:.0f} | {stats['p95_ms']:.0f} | "
                f"{_ms(stats.get('min_ms'))} | {_ms(stats.get('max_ms'))} | {stats['n']} |"
            )

    lines += [
        "",
        "## Cases",
        "",
        "| id | category | env | layer | delta | state exp/act | blocking | reasons |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in run.get("cases") or []:
        ev = case.get("evaluation") or {}
        failures = ev.get("failures") or []
        blocking = any(f.get("severity") == "BLOCKING" for f in failures)
        reasons = "; ".join(
            f"{f.get('layer')}/{f.get('severity')}:{f.get('code')}" for f in failures[:4]
        )
        if case.get("error"):
            reasons = f"ERROR: {case['error']}"
        if case.get("warmup"):
            reasons = (reasons + " | warmup (latency excluded only)").strip(" |")
        lines.append(
            f"| {case.get('id')} | {case.get('category')} | {case.get('environment')} | "
            f"{ev.get('first_failure_layer') or '-'} | "
            f"{ev.get('answer_delta') or '-'} | "
            f"{ev.get('state_verdict_expected') or '-'} / {ev.get('state_verdict_actual') or '-'} | "
            f"{'YES' if blocking else 'no'} | {reasons} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown_acceptance(run: dict) -> str:
    """AI Search Acceptance Report — the product-value view."""
    agg = run.get("aggregate") or {}
    verdict = run.get("verdict") or {}
    env = run.get("environment") or {}
    flags = run.get("flags") or {}

    def rate(value) -> str:
        return "n/a" if value is None else f"{value * 100:.0f}%"

    def ms(value) -> str:
        return "n/a" if value is None else f"{value:.0f} ms"

    sat = run.get("sat_status") or {}

    lines = [
        f"# AI Search Acceptance Report — {env.get('name')} ({run.get('timestamp')})",
        "",
        f"- projekt: `{run.get('project_id') or 'n/a'}`",
        f"- git sha: `{run.get('git_sha') or 'n/a'}`",
        f"- dataset: `{Path(run.get('dataset_file') or '').name}` "
        f"verze `{run.get('dataset_version') or 'neuvedena'}`",
        f"- index: {env.get('doc_count')} docs / {env.get('chunk_count')} chunks, "
        f"fingerprint `{(run.get('index_fingerprint') or 'n/a')[:16]}`",
        f"- flags: STATE_GATE=`{flags.get('DOCUMENT_STATE_GATE_ENABLED')}` "
        f"VALIDATION=`{flags.get('EVIDENCE_RUNTIME_VALIDATION_ENABLED')}` "
        f"llm_replay=`{flags.get('llm_replay')}` (live generations only)",
        "",
    ]
    for warning in run.get("warnings") or []:
        lines.append(f"> **WARNING** {warning}")
    if run.get("warnings"):
        lines.append("")

    # FAT and SAT are rendered as two separate verdicts on purpose. FAT is what
    # the machine measured; SAT is whether a human stood behind the ground truth
    # it measured against. A green FAT over unverified ground truth certifies
    # nothing, and merging them into one headline is exactly how that gets
    # misread as permission to deploy.
    lines += [
        "## FAT RESULT — automatické měření",
        "",
        f"**{verdict.get('verdict', 'n/a')}**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| počet testů | {agg.get('case_count')} |",
        f"| document_found_rate | {rate(agg.get('document_found_rate'))} "
        f"({agg.get('document_found_count')}/{agg.get('retrieval_measured_count')}) |",
        f"| top1_accuracy | {rate(agg.get('top1_accuracy'))} "
        f"({agg.get('top1_count')}/{agg.get('retrieval_measured_count')}) |",
        f"| top5_accuracy | {rate(agg.get('top5_accuracy'))} "
        f"({agg.get('top5_count')}/{agg.get('retrieval_measured_count')}) |",
        f"| answer_correct_rate | {rate(agg.get('answer_correct_rate'))} "
        f"({agg.get('answer_correct_count')}/{agg.get('evaluated_count')}) |",
        f"| citation_correct_rate | {rate(agg.get('citation_correct_rate'))} "
        f"({agg.get('citation_correct_count')}/{agg.get('citation_measured_count')}) |",
        f"| unsupported_claim_rate | {rate(agg.get('unsupported_claim_rate'))} "
        f"({agg.get('unsupported_claim_count')}/{agg.get('evaluated_count')}) |",
        f"| forbidden_document_rate | {rate(agg.get('forbidden_document_rate'))} "
        f"({agg.get('forbidden_document_hit_count')}/{agg.get('forbidden_document_measured_count')}) |",
        f"| počet kritických chyb | {agg.get('critical_error_count')} |",
        f"| průměrný čas odpovědi | {ms(agg.get('mean_total_ms'))} |",
        f"| p95 čas odpovědi | {ms(agg.get('p95_total_ms'))} |",
        f"| průměrný počet dotazů na odpověď | "
        f"{'n/a' if agg.get('mean_queries_to_answer') is None else format(agg['mean_queries_to_answer'], '.2f')} |",
        "",
        "Retrieval KPI se počítají jen z case, které pojmenovávají očekávaný dokument; "
        "negativní a otevřené dotazy do jmenovatele nevstupují.",
        "",
        "## SAT STATUS — čeká na lidské ověření",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| human-verified case | {sat.get('verified_cases_count', run.get('verified_cases_count'))} |",
        f"| čeká na ověření | {sat.get('pending_cases_count', run.get('pending_cases_count'))} |",
        f"| podíl ověřených | {rate(sat.get('human_verified_rate'))} |",
        f"| kritických case celkem | {sat.get('critical_cases_count', 'n/a')} |",
        f"| kritických bez expert_confirm | "
        f"{len(sat.get('critical_cases_without_expert_confirm') or [])} |",
        f"| ground truth verified / unverified | "
        f"{agg.get('verified_case_count')} / {agg.get('unverified_case_count')} |",
        "",
        f"**Připraveno pro denní použití: "
        f"{'ANO' if sat.get('ready_for_daily_use') else 'NE'}**",
        "",
    ]
    for blocker in sat.get("blockers") or []:
        lines.append(f"- SAT blocker: {blocker}")
    if sat.get("blockers"):
        lines.append("")
    if not sat.get("ready_for_daily_use"):
        lines += [
            "> Tento report **netvrdí**, že je AI Search připraven k nasazení. "
            "FAT měří jen shodu s datasetem; dokud není každý case podepsán člověkem "
            "a každý kritický case potvrzen odborníkem, je výsledek pouze měřením, "
            "nikoli certifikací.",
            "",
        ]

    lines += [
        f"## GO / NO-GO: **{verdict.get('verdict', 'n/a')}**",
        "",
    ]
    for blocker in verdict.get("blockers") or []:
        lines.append(f"- **blocker**: {blocker}")
    for reason in verdict.get("inconclusive_reasons") or []:
        lines.append(f"- inconclusive: {reason}")
    if not (verdict.get("blockers") or verdict.get("inconclusive_reasons")):
        lines.append("- all acceptance criteria met")
    observed = [
        f for f in (verdict.get("observed_failures") or [])
        if f not in (verdict.get("blockers") or [])
    ]
    if observed:
        lines += [
            "",
            "Observed quality failures (not attributable to the product in this environment):",
        ]
        lines.extend(f"- {f}" for f in observed)

    criteria = verdict.get("criteria") or {}
    if criteria:
        lines += ["", "| Criterion | Pass |", "|---|---|"]
        for key, value in criteria.items():
            lines.append(f"| {key} | {'✅' if value else '❌'} |")

    if agg.get("critical_error_case_ids"):
        lines += [
            "",
            "## Critical errors",
            "",
            "Wrong answers on legal/financial questions, or answers containing a forbidden claim. "
            "Any non-zero count blocks deployment on its own.",
            "",
        ]
        for case_id in agg["critical_error_case_ids"]:
            lines.append(f"- `{case_id}`")

    lines += [
        "",
        "## User value",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| vyřešeno prvním dotazem | {agg.get('resolved_within_one_query')} |",
        f"| vyřešeno až follow-up dotazem | {agg.get('resolved_with_follow_up')} |",
        f"| nevyřešeno vůbec | {agg.get('unresolved')} |",
        "",
        "## Where it broke",
        "",
        "| Layer | Cases |",
        "|---|---|",
    ]
    for layer, count in sorted((agg.get("by_layer") or {}).items()):
        lines.append(f"| {layer} | {count} |")

    by_cat = agg.get("by_category") or {}
    if by_cat:
        lines += [
            "",
            "## By category",
            "",
            "| Category | n | doc found | top5 | answer correct | critical | unsupported |",
            "|---|---|---|---|---|---|---|",
        ]
        for category, row in by_cat.items():
            lines.append(
                f"| {category} | {row.get('case_count')} | "
                f"{rate(row.get('document_found_rate'))} | "
                f"{rate(row.get('top5_accuracy'))} | "
                f"{rate(row.get('answer_correct_rate'))} | {row.get('critical_errors')} | "
                f"{row.get('unsupported_claims')} |"
            )

    lines += [
        "",
        "## Cases",
        "",
        "| id | category | crit | GT | doc | rank | answer | queries | ms | layer |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in run.get("cases") or []:
        lines.append(
            f"| {case.get('id')} | {case.get('category')} | {case.get('criticality')} | "
            f"{case.get('ground_truth_status')} | "
            f"{'✅' if case.get('document_found') else '❌'} | "
            f"{case.get('document_rank') if case.get('document_rank') else '-'} | "
            f"{'✅' if case.get('answer_correct') else '❌'} | "
            f"{case.get('queries_to_answer') if case.get('queries_to_answer') else '-'} | "
            f"{case.get('total_ms')} | {case.get('failure_layer')} |"
        )
    return "\n".join(lines) + "\n"


def render_csv_acceptance(run: dict) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "id", "category", "environment", "criticality", "ground_truth_status",
        "expected_outcome", "document_found", "document_rank", "answer_correct",
        "answer_used_expected_source", "citation_correct", "forbidden_document_hit",
        "unsupported_claim", "critical_error", "failure_layer",
        "queries_to_answer", "follow_ups_used", "retrieval_ms", "answer_ms",
        "total_ms", "missing_phrases", "error",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for case in run.get("cases") or []:
        row = dict(case)
        row["missing_phrases"] = "; ".join(case.get("missing_phrases") or [])
        writer.writerow(row)
    return buffer.getvalue()


def render_csv_pr74(run: dict) -> str:
    buffer = io.StringIO()
    fieldnames = [
        "id", "category", "environment", "answer_delta",
        "state_verdict_expected", "state_verdict_actual", "state_verdict_match",
        "intent_coverage_expected", "intent_coverage_actual",
        "blocking", "failure_codes", "warmup", "error",
        "retrieval_ms", "llm_replay",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for case in run.get("cases") or []:
        ev = case.get("evaluation") or {}
        failures = ev.get("failures") or []
        writer.writerow({
            "id": case.get("id"),
            "category": case.get("category"),
            "environment": case.get("environment"),
            "answer_delta": ev.get("answer_delta"),
            "state_verdict_expected": ev.get("state_verdict_expected"),
            "state_verdict_actual": ev.get("state_verdict_actual"),
            "state_verdict_match": ev.get("state_verdict_match"),
            "intent_coverage_expected": ev.get("intent_coverage_expected"),
            "intent_coverage_actual": ev.get("intent_coverage_actual"),
            "blocking": any(f.get("severity") == "BLOCKING" for f in failures),
            "failure_codes": ";".join(f.get("code") or "" for f in failures),
            "warmup": case.get("warmup"),
            "error": case.get("error"),
            "retrieval_ms": case.get("retrieval_ms"),
            "llm_replay": case.get("llm_replay"),
        })
    return buffer.getvalue()
