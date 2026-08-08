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
