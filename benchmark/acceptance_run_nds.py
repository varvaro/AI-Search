"""PR7.5.2 — Garáže ND Smíchov FAT Acceptance Runner.

The first automated FAT (Factory Acceptance Test) execution of AI Search
against benchmark/dataset/acceptance_nds_smichov.jsonl. Retrieval and
answer() are called for real, once per case (no LLM replay - acceptance runs
measure the shipping configuration live, the same way pr74_runner's docstring
already explains for the general acceptance runner). Every case's full
evidence trail is stored, not just a pass/fail bit, so a failed row can be
read back to the exact retrieved document, cited source and validation
metadata that produced it.

Deliberately layered ON TOP of the existing acceptance primitives instead of
reimplementing them:
  - benchmark.pipeline_trace.run_pipeline_trace   retrieval, unmodified
  - ai_search.answer                              answer(), unmodified
  - benchmark.answer_evidence                      evidence tiers (cited/state)
  - benchmark.acceptance_metrics                   document_hit / evaluate_citations

This module adds only what none of those already provide: the specific
DOCUMENT_SEARCH / TECHNICAL_QA / DOCUMENT_STATUS / ADVERSARIAL evaluation the
ND Smíchov acceptance plan calls for, an OLD-revision-leakage detector (path-
based, so a folder named OLD/ is caught even for a case that never named it in
forbidden_document), an entity-substitution heuristic for the duplicated-
supplier adversarial cases, the ND-Smíchov GO rule from the PR7.5.2 brief, and
a dedicated Markdown report.

Never touches ai_search.py, retrieval, embeddings, prompts, or the runtime.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ai_search

from . import acceptance_metrics, answer_evidence
from .dataset.schema import (
    DATASET_DIR,
    BenchmarkCase,
    load_dataset,
    read_dataset_version,
)
from .environment import Environment, get_environment
from .pipeline_trace import run_pipeline_trace
from .pr74_runner import _forced_flags, _git_sha

NDS_DATASET = DATASET_DIR / "acceptance_nds_smichov.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"

# The configuration under test. Explicit (not read from ai_search_config) so
# the report states what was actually measured, same convention as
# acceptance_runner.DEFAULT_STATE_GATE / DEFAULT_VALIDATION.
DEFAULT_STATE_GATE = True
DEFAULT_VALIDATION = True

# A path segment exactly "OLD" or "old" (Czech folders use both cases), not a
# bare substring - "Sokolov" or "Podklad" must never match. Matches a leading
# folder ("OLD/D.1.2.07...") or an internal one ("Statika/OLD/D.1.2.07...").
_OLD_SEGMENT_RE = re.compile(r"(^|/)old(/|$)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Step 1-2: load + execute
# ---------------------------------------------------------------------------

def load_nds_dataset(dataset_path: Path | str = NDS_DATASET) -> list[BenchmarkCase]:
    return load_dataset(Path(dataset_path))


def _ask(
    question: str, environment: Environment, *, result_count: int,
) -> tuple[list[dict], dict, float, float]:
    """One retrieval + answer turn, both timed live. Mirrors
    acceptance_runner._ask exactly - kept as a private copy rather than an
    import so this module can evolve its own ND-specific instrumentation
    later without reaching back into the general acceptance runner."""
    t0 = time.perf_counter()
    trace = run_pipeline_trace(
        question, environment, result_count=result_count, include_answer=False,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000
    results = list(trace.final_results)

    t1 = time.perf_counter()
    answer = ai_search.answer(question, results) if results else {
        "answer": "", "citations": [], "confidence": "red",
    }
    answer_ms = (time.perf_counter() - t1) * 1000
    return results, answer, retrieval_ms, answer_ms


# ---------------------------------------------------------------------------
# ND-specific detectors (not covered by acceptance_metrics/answer_evidence)
# ---------------------------------------------------------------------------

def detect_old_revision_leakage(
    case: BenchmarkCase, results: list[dict], answer: dict | None,
) -> list[str]:
    """Evidence paths under an OLD/ folder that represent an ACTUAL wrong
    choice, not merely "the answer used a path that happens to say OLD".

    Two ways a path counts as leakage:

      1. It matches one of the case's own `forbidden_document` fragments.
         Explicit and unambiguous - the dataset author already named this
         exact revision as wrong.
      2. It does not match `forbidden_document`, but a NON-OLD alternative
         for the same `expected_documents` target was present in the
         retrieval pool and the answer leaned on the OLD one anyway. This is
         evidence-based on purpose: filenames repeat across revisions (the
         withdrawn and the current `D.1.2.07 - schéma vyztužení 3.PP.pdf`
         would be spelled identically), so matching `expected_documents`
         against the OLD path's own name proves nothing about which copy is
         correct - only "was a better copy available and ignored" does.

    A case like nds-status-03, where OLD/ is the ONLY place the expected
    drawing still exists, has no non-OLD alternative in the pool and is
    therefore correctly never flagged here.
    """
    rows = answer_evidence.evidence_rows(answer)
    hits: list[str] = []
    for row in rows:
        path = str(row.get("path") or "")
        if not _OLD_SEGMENT_RE.search(path):
            continue
        if case.forbidden_document and answer_evidence.match_any([path], case.forbidden_document):
            hits.append(path)
            continue
        if case.expected_documents and _non_old_alternative_available(case, results):
            hits.append(path)
    return hits


def _non_old_alternative_available(case: BenchmarkCase, results: list[dict]) -> bool:
    for row in results or []:
        path = str(row.get("path") or "")
        if _OLD_SEGMENT_RE.search(path):
            continue
        haystack = [path + " " + str(row.get("document") or "")]
        if answer_evidence.match_any(haystack, case.expected_documents):
            return True
    return False


def detect_entity_substitution(case: BenchmarkCase, answer: dict | None) -> bool:
    """Heuristic proxy for "the answer conflated two similarly-named
    subjects" on the duplicated-supplier ADVERSARIAL cases (Stafitech, Bičík,
    Hilt Rent each have two order numbers in expected_answer_contains).

    Only meaningful when the case names 2+ distinguishing identifiers: if the
    answer contains SOME but not ALL of them, one entity was likely answered
    while the other was dropped or substituted. This is a signal for a human
    reviewer to look at the case, not a certainty - it cannot tell "the model
    picked the wrong one" apart from "the model only found one of two real
    things", which is exactly why SAT sign-off still exists.
    """
    if case.category != "ADVERSARIAL" or len(case.expected_answer_contains) < 2:
        return False
    body = answer_evidence.fold(answer_evidence.answer_body(answer))
    present = [k for k in case.expected_answer_contains if answer_evidence.fold(k) in body]
    return 0 < len(present) < len(case.expected_answer_contains)


# ---------------------------------------------------------------------------
# Per-case result + category evaluation
# ---------------------------------------------------------------------------

@dataclass
class NdsCaseResult:
    id: str
    category: str
    query: str
    criticality: str
    expected_outcome: str
    ground_truth_status: str
    # Raw evidence trail (step 2) - stored so a failed row can be read back
    # without re-running anything.
    retrieved: list[dict] = field(default_factory=list)
    answer_text: str = ""
    cited_sources: list[str] = field(default_factory=list)
    validation: dict | None = None
    # Retrieval outcome (DOCUMENT_SEARCH KPIs)
    document_found: bool | None = None
    document_rank: int | None = None
    top1: bool | None = None
    top5: bool | None = None
    # Answer outcome (TECHNICAL_QA / DOCUMENT_STATUS "statement correctness")
    answer_correct: bool | None = None
    missing_phrases: list[str] = field(default_factory=list)
    # Source outcome (DOCUMENT_STATUS "source correctness")
    citation_correct: bool | None = None
    # ADVERSARIAL / global safety checks
    wrong_document_citation: bool = False
    old_revision_leak_paths: list[str] = field(default_factory=list)
    entity_substitution_suspected: bool = False
    unsupported_claim: bool = False
    # Verdict
    critical_error: bool = False
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    answer_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def old_revision_leak(self) -> bool:
        return bool(self.old_revision_leak_paths)


def _evaluate_result(
    case: BenchmarkCase, results: list[dict], answer: dict,
) -> tuple[dict, list[str]]:
    """Everything derived from ONE (case, results, answer) triple. Returns
    (fields_for_NdsCaseResult, failure_reasons)."""
    found, correct, missing, used_expected = acceptance_metrics.evaluate_acceptance_answer(
        case, results, answer,
    )
    _found_ignored, rank = acceptance_metrics.document_hit(case.expected_documents, results)
    citation_correct, forbidden_document_hit, unsupported = acceptance_metrics.evaluate_citations(
        case, answer,
    )
    old_hits = detect_old_revision_leakage(case, results, answer)
    entity_substitution = detect_entity_substitution(case, answer)
    forbidden_phrase_hit = any(m.startswith("forbidden:") for m in missing)

    critical_error = acceptance_metrics.is_critical_error(
        case, correct, forbidden_phrase_hit, forbidden_document_hit=forbidden_document_hit,
    ) or bool(old_hits)

    reasons: list[str] = []
    if old_hits:
        reasons.append(f"OLD revision leakage: {old_hits}")
    if forbidden_document_hit:
        reasons.append(f"wrong document citation (forbidden_document={case.forbidden_document})")
    if unsupported:
        reasons.append("unsupported claim (no cited/state evidence)")
    if entity_substitution:
        reasons.append(
            f"entity substitution suspected: only some of {case.expected_answer_contains} present"
        )

    if case.category == "DOCUMENT_SEARCH":
        if case.expected_outcome == "not_found":
            if not correct:
                reasons.append(f"expected not_found but answer failed contract: {missing}")
        elif found is False:
            reasons.append(f"expected document never reached the result pool: {case.expected_documents}")
    elif case.category == "TECHNICAL_QA":
        if not correct:
            reasons.append(f"answer missing required fact / contains forbidden claim: {missing}")
    elif case.category == "DOCUMENT_STATUS":
        if not correct:
            reasons.append(f"statement incorrect: {missing}")
        if citation_correct is False:
            reasons.append("source incorrect: answer does not lean on expected_source_contains")
    elif case.category == "CONSTRUCTION_MGMT":
        if not correct:
            reasons.append(f"answer missing required fact: {missing}")
    elif case.category == "ADVERSARIAL":
        if not correct:
            reasons.append(f"answer failed its contract: {missing}")

    passed = not reasons
    fields = {
        "retrieved": [
            {"rank": i, "document": r.get("document"), "path": r.get("path")}
            for i, r in enumerate(results, start=1)
        ],
        "answer_text": answer_evidence.answer_body(answer),
        "cited_sources": answer_evidence.evidence_documents(answer),
        "validation": answer.get("validation") if isinstance(answer, dict) else None,
        "document_found": found,
        "document_rank": rank,
        "top1": (found is True and rank == 1),
        "top5": (found is True and rank is not None and rank <= 5),
        "answer_correct": correct,
        "missing_phrases": missing,
        "citation_correct": citation_correct,
        "wrong_document_citation": forbidden_document_hit,
        "old_revision_leak_paths": old_hits,
        "entity_substitution_suspected": entity_substitution,
        "unsupported_claim": unsupported,
        "critical_error": critical_error,
        "passed": passed,
        "failure_reasons": reasons,
    }
    return fields, reasons


def evaluate_nds_case(
    case: BenchmarkCase,
    environment: Environment,
    *,
    result_count: int = 10,
    state_gate: bool = DEFAULT_STATE_GATE,
    validation: bool = DEFAULT_VALIDATION,
) -> NdsCaseResult:
    result = NdsCaseResult(
        id=case.id, category=case.category, query=case.question,
        criticality=case.criticality, expected_outcome=case.expected_outcome,
        ground_truth_status=case.ground_truth_status,
    )
    try:
        with _forced_flags(state_gate, validation):
            results, answer, retrieval_ms, answer_ms = _ask(
                case.question, environment, result_count=result_count,
            )
        fields, _reasons = _evaluate_result(case, results, answer)
        for name, value in fields.items():
            setattr(result, name, value)
        result.retrieval_ms = round(retrieval_ms, 2)
        result.answer_ms = round(answer_ms, 2)
        result.total_ms = round(retrieval_ms + answer_ms, 2)
    except Exception as exc:  # a harness failure must not look like a case failure
        result.error = f"{type(exc).__name__}: {exc}"
        result.failure_reasons = [result.error]
        result.critical_error = False
        result.passed = False
    return result


# ---------------------------------------------------------------------------
# Aggregate + GO/NO-GO
# ---------------------------------------------------------------------------

@dataclass
class NdsAggregate:
    case_count: int = 0
    evaluated_count: int = 0
    errored_count: int = 0
    passed_count: int = 0
    failed_case_ids: list[str] = field(default_factory=list)
    critical_error_case_ids: list[str] = field(default_factory=list)
    wrong_document_citation_case_ids: list[str] = field(default_factory=list)
    old_revision_leak_case_ids: list[str] = field(default_factory=list)
    unsupported_claim_case_ids: list[str] = field(default_factory=list)
    entity_substitution_case_ids: list[str] = field(default_factory=list)
    by_category: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def aggregate_nds_results(results: list[NdsCaseResult]) -> NdsAggregate:
    agg = NdsAggregate(case_count=len(results))
    evaluated = [r for r in results if r.error is None]
    agg.evaluated_count = len(evaluated)
    agg.errored_count = len(results) - len(evaluated)

    agg.failed_case_ids = [r.id for r in results if not r.passed]
    agg.passed_count = len(results) - len(agg.failed_case_ids)
    agg.critical_error_case_ids = [r.id for r in evaluated if r.critical_error]
    agg.wrong_document_citation_case_ids = [r.id for r in evaluated if r.wrong_document_citation]
    agg.old_revision_leak_case_ids = [r.id for r in evaluated if r.old_revision_leak]
    agg.unsupported_claim_case_ids = [r.id for r in evaluated if r.unsupported_claim]
    agg.entity_substitution_case_ids = [r.id for r in evaluated if r.entity_substitution_suspected]

    buckets: dict[str, list[NdsCaseResult]] = {}
    for r in evaluated:
        buckets.setdefault(r.category or "UNCATEGORIZED", []).append(r)

    for category, items in sorted(buckets.items()):
        row: dict = {"case_count": len(items), "passed": sum(1 for r in items if r.passed)}
        if category == "DOCUMENT_SEARCH":
            measured = [r for r in items if r.document_found is not None]
            row["document_found_rate"] = _ratio(
                sum(1 for r in measured if r.document_found), len(measured),
            )
            row["top1_accuracy"] = _ratio(sum(1 for r in measured if r.top1), len(measured))
            row["top5_accuracy"] = _ratio(sum(1 for r in measured if r.top5), len(measured))
        elif category == "TECHNICAL_QA":
            row["answer_correct_rate"] = _ratio(sum(1 for r in items if r.answer_correct), len(items))
        elif category == "DOCUMENT_STATUS":
            row["statement_correct_rate"] = _ratio(
                sum(1 for r in items if r.answer_correct), len(items),
            )
            source_measured = [r for r in items if r.citation_correct is not None]
            row["source_correct_rate"] = _ratio(
                sum(1 for r in source_measured if r.citation_correct), len(source_measured),
            )
        elif category == "ADVERSARIAL":
            row["old_revision_leaks"] = sum(1 for r in items if r.old_revision_leak)
            row["wrong_document_citations"] = sum(1 for r in items if r.wrong_document_citation)
            row["entity_substitutions_suspected"] = sum(
                1 for r in items if r.entity_substitution_suspected
            )
        agg.by_category[category] = row
    return agg


def nds_go_nogo(agg: NdsAggregate) -> dict:
    """PR7.5.2 GO rule, exactly as specified: four blocking conditions, no
    positive thresholds. This is a SAFETY gate over one FAT run's evidence,
    not a deployment certification - retrieval quality (top1/top5/
    document_found_rate) is reported per-category above but does not gate
    GO here, and ground-truth verification (SAT, see acceptance_runner.
    sat_status) is a separate, still-pending question this function does not
    answer.
    """
    blockers: list[str] = []
    if agg.critical_error_case_ids:
        blockers.append(f"critical_failures={len(agg.critical_error_case_ids)} {agg.critical_error_case_ids}")
    if agg.wrong_document_citation_case_ids:
        blockers.append(
            f"wrong_document_citation={len(agg.wrong_document_citation_case_ids)} "
            f"{agg.wrong_document_citation_case_ids}"
        )
    if agg.old_revision_leak_case_ids:
        blockers.append(
            f"old_revision_leakage={len(agg.old_revision_leak_case_ids)} "
            f"{agg.old_revision_leak_case_ids}"
        )
    if agg.unsupported_claim_case_ids:
        blockers.append(
            f"unsupported_claims={len(agg.unsupported_claim_case_ids)} "
            f"{agg.unsupported_claim_case_ids}"
        )
    if agg.errored_count:
        blockers.append(f"errored_cases={agg.errored_count}")
    return {
        "verdict": "NO-GO" if blockers else "GO",
        "blockers": blockers,
        "criteria": {
            "zero_critical_failures": not agg.critical_error_case_ids,
            "zero_wrong_document_citation": not agg.wrong_document_citation_case_ids,
            "zero_old_revision_leakage": not agg.old_revision_leak_case_ids,
            "zero_unsupported_claims": not agg.unsupported_claim_case_ids,
            "no_errored_cases": agg.errored_count == 0,
        },
    }


# ---------------------------------------------------------------------------
# Run artifact + report
# ---------------------------------------------------------------------------

@dataclass
class NdsRunArtifact:
    timestamp: str
    git_sha: str | None
    environment: dict
    dataset_file: str
    dataset_version: str
    flags: dict
    case_count: int
    cases: list[dict]
    aggregate: dict
    verdict: dict

    def to_dict(self) -> dict:
        return asdict(self)


def run_nds_acceptance(
    *,
    environment_name: str = "production",
    dataset_path: Path | str = NDS_DATASET,
    result_count: int = 10,
    case_filter: set[str] | None = None,
    state_gate: bool = DEFAULT_STATE_GATE,
    validation: bool = DEFAULT_VALIDATION,
) -> NdsRunArtifact:
    dataset_path = Path(dataset_path)
    cases = load_nds_dataset(dataset_path)
    if case_filter is not None:
        cases = [c for c in cases if c.id in case_filter]

    environment = get_environment(environment_name)
    results = [
        evaluate_nds_case(
            case, environment,
            result_count=result_count, state_gate=state_gate, validation=validation,
        )
        for case in cases
    ]
    aggregate = aggregate_nds_results(results)
    verdict = nds_go_nogo(aggregate)

    return NdsRunArtifact(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        environment=environment.describe(),
        dataset_file=str(dataset_path),
        dataset_version=read_dataset_version(dataset_path),
        flags={
            "DOCUMENT_STATE_GATE_ENABLED": state_gate,
            "EVIDENCE_RUNTIME_VALIDATION_ENABLED": validation,
            "AUXILIARY_TERM_COVERAGE_ENABLED": False,
            "MULTI_QUERY_RETRIEVAL_ENABLED": False,
        },
        case_count=len(cases),
        cases=[r.to_dict() for r in results],
        aggregate=aggregate.to_dict(),
        verdict=verdict,
    )


def render_nds_report(run: NdsRunArtifact | dict) -> str:
    data = run.to_dict() if isinstance(run, NdsRunArtifact) else run
    env = data.get("environment") or {}
    agg = data.get("aggregate") or {}
    verdict = data.get("verdict") or {}

    def pct(value) -> str:
        return "n/a" if value is None else f"{value * 100:.0f}%"

    lines = [
        f"# AI Search FAT Acceptance Report — Garáže ND Smíchov ({data.get('timestamp')})",
        "",
        f"- git sha: `{data.get('git_sha') or 'n/a'}`",
        f"- dataset: `{Path(data.get('dataset_file') or '').name}` "
        f"verze `{data.get('dataset_version') or 'neuvedena'}`",
        f"- index fingerprint: `{env.get('index_fingerprint') or 'n/a'}`",
        f"- index: {env.get('doc_count')} docs / {env.get('chunk_count')} chunks "
        f"({env.get('name')})",
        f"- flags: STATE_GATE=`{(data.get('flags') or {}).get('DOCUMENT_STATE_GATE_ENABLED')}` "
        f"VALIDATION=`{(data.get('flags') or {}).get('EVIDENCE_RUNTIME_VALIDATION_ENABLED')}`",
        "",
        f"## Počet testů: {agg.get('case_count')}",
        "",
        f"- vyhodnoceno: {agg.get('evaluated_count')}",
        f"- prošlo: {agg.get('passed_count')}",
        f"- selhalo: {len(agg.get('failed_case_ids') or [])}",
        f"- chyba harness: {agg.get('errored_count')}",
        "",
        "## Výsledky podle kategorií",
        "",
    ]

    by_cat = agg.get("by_category") or {}
    for category, row in by_cat.items():
        lines.append(f"### {category} (n={row.get('case_count')}, prošlo={row.get('passed')})")
        lines.append("")
        if category == "DOCUMENT_SEARCH":
            lines.append(f"- top1_accuracy: {pct(row.get('top1_accuracy'))}")
            lines.append(f"- top5_accuracy: {pct(row.get('top5_accuracy'))}")
            lines.append(f"- document_found_rate: {pct(row.get('document_found_rate'))}")
        elif category == "TECHNICAL_QA":
            lines.append(f"- answer_correct_rate: {pct(row.get('answer_correct_rate'))}")
        elif category == "DOCUMENT_STATUS":
            lines.append(f"- statement_correct_rate: {pct(row.get('statement_correct_rate'))}")
            lines.append(f"- source_correct_rate: {pct(row.get('source_correct_rate'))}")
        elif category == "ADVERSARIAL":
            lines.append(f"- OLD revision leakage: {row.get('old_revision_leaks')}")
            lines.append(f"- wrong document citation: {row.get('wrong_document_citations')}")
            lines.append(f"- entity substitution suspected: {row.get('entity_substitutions_suspected')}")
        else:
            lines.append(f"- passed/total: {row.get('passed')}/{row.get('case_count')}")
        lines.append("")

    lines += ["## Failed cases", ""]
    failed_ids = set(agg.get("failed_case_ids") or [])
    if not failed_ids:
        lines.append("- žádný")
    else:
        by_id = {c["id"]: c for c in data.get("cases") or []}
        for case_id in agg.get("failed_case_ids") or []:
            case = by_id.get(case_id, {})
            reasons = "; ".join(case.get("failure_reasons") or [])
            lines.append(f"- `{case_id}` ({case.get('category')}): {reasons}")
    lines.append("")

    lines += ["## Kritické chyby", ""]
    critical_ids = agg.get("critical_error_case_ids") or []
    if not critical_ids:
        lines.append("- žádné")
    else:
        for case_id in critical_ids:
            lines.append(f"- `{case_id}`")
    lines.append("")

    lines += [
        f"## GO / NO-GO: **{verdict.get('verdict', 'n/a')}**",
        "",
        "GO pouze pokud: critical failures = 0, wrong document citation = 0, "
        "OLD revision leakage = 0, unsupported claims = 0.",
        "",
    ]
    for blocker in verdict.get("blockers") or []:
        lines.append(f"- blocker: {blocker}")
    if not verdict.get("blockers"):
        lines.append("- všechny čtyři podmínky splněny")
    lines += [
        "",
        "> Toto GO/NO-GO je bezpečnostní brána nad JEDNÍM FAT během, nikoli certifikace "
        "nasazení. Ground truth datasetu je needs_review/unverified, dokud neproběhne "
        "lidské ověření (SAT, viz benchmark/sat_protocol.py) - i GO zde neznamená, "
        "že je AI Search připraven k denní práci.",
        "",
    ]

    criteria = verdict.get("criteria") or {}
    if criteria:
        lines += ["| Criterion | Pass |", "|---|---|"]
        for key, value in criteria.items():
            lines.append(f"| {key} | {'✅' if value else '❌'} |")

    return "\n".join(lines) + "\n"


def save_nds_report(run: NdsRunArtifact, out_dir: Path | str = REPORTS_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"acceptance_nds_smichov_{stamp}.md"
    path.write_text(render_nds_report(run), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m benchmark.acceptance_run_nds",
        description="FAT acceptance run of AI Search against Garáže ND Smíchov.",
    )
    parser.add_argument("--environment", choices=["fixture", "production"], default="production")
    parser.add_argument("--dataset", default=str(NDS_DATASET))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--case", action="append", help="run only these case ids (repeatable)")
    parser.add_argument("--no-state-gate", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--out-dir", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)

    run = run_nds_acceptance(
        environment_name=args.environment,
        dataset_path=args.dataset,
        result_count=args.k,
        case_filter=set(args.case) if args.case else None,
        state_gate=not args.no_state_gate,
        validation=not args.no_validation,
    )
    path = save_nds_report(run, args.out_dir)
    print(f"FAT report: {path}")
    print(f"  case count: {run.case_count}")
    print(f"  GO/NO-GO:   {run.verdict.get('verdict')}")
    for blocker in run.verdict.get("blockers") or []:
        print(f"    blocker: {blocker}")
    return 0 if run.verdict.get("verdict") == "GO" else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
