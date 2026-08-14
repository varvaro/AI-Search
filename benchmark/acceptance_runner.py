"""AI Search Acceptance Test runner — the product benchmark.

One configuration, one question at a time, measured the way the user
experiences it: retrieve, answer, check whether the right document was found
and whether the answer says the right thing, and how long it took.

Differences from pr74_runner on purpose:

  * No A/B/C/D matrix. Acceptance asks "is the shipping configuration good
    enough", not "which flag caused a delta". The flags under test default to
    the full candidate (gate ON, validation ON) and are recorded in the
    artifact so a run can never be misread as testing something else.
  * No LLM replay. Replay exists to hold generation constant while comparing
    two configurations; here the live generation IS the product, so every
    answer is a real call and every latency is a real latency.
  * Follow-up questions are executed when the first answer misses, which is
    what a site manager actually does, and `queries_to_answer` records how many
    it took.

Never writes to the index and never touches retrieval, scoring or prompts.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ai_search

from . import acceptance_metrics
from .dataset.schema import (
    CRITICAL_CRITICALITY_NAMES,
    DATASET_DIR,
    BenchmarkCase,
    load_dataset,
    read_dataset_version,
)
from .environment import Environment, get_environment
from .pipeline_trace import run_pipeline_trace
from .pr74_runner import RUNS_DIR, _forced_flags, _git_sha

ACCEPTANCE_DATASET = DATASET_DIR / "acceptance_v1.jsonl"

# The configuration being certified. Kept explicit (not read from
# ai_search_config) so the artifact states what was actually measured.
DEFAULT_STATE_GATE = True
DEFAULT_VALIDATION = True


@dataclass
class AcceptanceRunArtifact:
    timestamp: str
    git_sha: str | None
    environment: dict
    dataset_file: str
    flags: dict
    case_count: int
    cases: list[dict]
    aggregate: dict
    verdict: dict
    warnings: list[str] = field(default_factory=list)
    # PR7.5 run metadata. Identifies WHICH archive, WHICH dataset and HOW MUCH
    # of it a human has stood behind, so a stored artifact can be read months
    # later without guessing what it certified.
    project_id: str = ""
    index_fingerprint: str = ""
    dataset_version: str = ""
    verified_cases_count: int = 0
    pending_cases_count: int = 0
    sat_status: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def sat_status(cases: list[BenchmarkCase], environment_name: str) -> dict:
    """Human-verification state of the dataset — the SAT half of the report.

    FAT is what the machine measured. SAT is whether anyone stood behind the
    ground truth it measured against, and it cannot be computed from a run: it
    is a property of the dataset. Splitting them is the whole point - a green
    FAT over unverified ground truth certifies nothing, and the report must not
    let those two be read as one number.
    """
    verified = [c for c in cases if c.human_verified and c.ground_truth_status == "verified"]
    pending = [c for c in cases if c not in verified]
    critical = [c for c in cases if c.criticality in CRITICAL_CRITICALITY_NAMES]
    missing_expert = [
        c.id for c in critical if c.verification_method != "expert_confirm"
    ]
    stale = [
        c.id for c in cases
        if c.human_verified and not c.index_fingerprint_at_verification
    ]
    blockers: list[str] = []
    if pending:
        blockers.append(
            f"{len(pending)}/{len(cases)} case(s) not human-verified"
        )
    if missing_expert:
        blockers.append(
            f"{len(missing_expert)} critical case(s) lack expert_confirm: {missing_expert}"
        )
    if stale:
        blockers.append(
            f"{len(stale)} case(s) claim human verification without an index fingerprint: {stale}"
        )
    if environment_name != "production":
        blockers.append(
            f"environment={environment_name!r} is not the production index"
        )
    return {
        "verified_cases_count": len(verified),
        "pending_cases_count": len(pending),
        "critical_cases_count": len(critical),
        "critical_cases_without_expert_confirm": missing_expert,
        "human_verified_rate": (len(verified) / len(cases)) if cases else None,
        # Never True until a human has signed off every case AND every critical
        # case carries a professional's confirmation. The report renders this
        # verbatim so no summary can claim readiness on its own.
        "ready_for_daily_use": not blockers,
        "blockers": blockers,
    }


def _ask(
    question: str,
    environment: Environment,
    *,
    result_count: int,
) -> tuple[list[dict], dict, float, float]:
    """One full user-visible turn: retrieval + answer, both timed live."""
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


def evaluate_acceptance_case(
    case: BenchmarkCase,
    environment: Environment,
    *,
    result_count: int = 10,
    state_gate: bool = DEFAULT_STATE_GATE,
    validation: bool = DEFAULT_VALIDATION,
    max_follow_ups: int | None = None,
) -> acceptance_metrics.AcceptanceCaseResult:
    result = acceptance_metrics.AcceptanceCaseResult(
        id=case.id,
        question=case.question,
        category=case.category or "",
        environment=case.environment,
        criticality=case.criticality,
        ground_truth_status=case.ground_truth_status,
        expected_outcome=case.expected_outcome,
    )
    follow_ups = list(case.follow_up_questions or [])
    if max_follow_ups is not None:
        follow_ups = follow_ups[:max_follow_ups]
    questions = [case.question] + follow_ups

    try:
        with _forced_flags(state_gate, validation):
            retrieval_total = answer_total = 0.0
            found = correct = False
            missing: list[str] = []
            used_expected: bool | None = None
            results: list[dict] = []
            answer: dict = {}

            for attempt, question in enumerate(questions, start=1):
                results, answer, retrieval_ms, answer_ms = _ask(
                    question, environment, result_count=result_count,
                )
                retrieval_total += retrieval_ms
                answer_total += answer_ms
                found, correct, missing, used_expected = (
                    acceptance_metrics.evaluate_acceptance_answer(case, results, answer)
                )
                if correct:
                    result.queries_to_answer = attempt
                    result.follow_ups_used = attempt - 1
                    break
            else:
                # Never satisfied, even after every follow-up.
                result.queries_to_answer = None
                result.follow_ups_used = len(questions) - 1

        result.document_found = found
        _hit, rank = acceptance_metrics.document_hit(case.expected_documents, results)
        result.document_rank = rank
        result.answer_correct = correct
        result.answer_used_expected_source = used_expected
        result.missing_phrases = missing
        result.retrieval_ms = round(retrieval_total, 2)
        result.answer_ms = round(answer_total, 2)
        result.total_ms = round(retrieval_total + answer_total, 2)
        result.retrieved_documents = [
            str(r.get("document") or "") for r in results
        ]
        result.evidence_documents = _evidence(answer)
        citation_correct, forbidden_document_hit, unsupported = (
            acceptance_metrics.evaluate_citations(case, answer)
        )
        result.citation_correct = citation_correct
        result.forbidden_document_hit = forbidden_document_hit
        result.forbidden_document_measured = bool(case.forbidden_document)
        result.unsupported_claim = unsupported
        forbidden_hit = any(m.startswith("forbidden:") for m in missing)
        result.critical_error = acceptance_metrics.is_critical_error(
            case, correct, forbidden_hit, forbidden_document_hit=forbidden_document_hit,
        )
        layer, detail = acceptance_metrics.classify_failure(
            case, found, correct, used_expected,
            forbidden_document_hit=forbidden_document_hit,
        )
        result.failure_layer = layer
        result.failure_detail = detail
    except Exception as exc:  # a harness failure must not look like a product failure
        result.error = f"{type(exc).__name__}: {exc}"
        result.failure_layer = "ERROR"
        result.failure_detail = str(exc)
    return result


def _evidence(answer: dict) -> list[str]:
    from . import answer_evidence

    return answer_evidence.evidence_documents(answer)


def run_acceptance_benchmark(
    *,
    environment_name: str = "production",
    dataset_path: Path | None = None,
    result_count: int = 10,
    case_filter: set[str] | None = None,
    state_gate: bool = DEFAULT_STATE_GATE,
    validation: bool = DEFAULT_VALIDATION,
    thresholds: dict | None = None,
    max_follow_ups: int | None = None,
) -> AcceptanceRunArtifact:
    dataset_path = Path(dataset_path) if dataset_path else ACCEPTANCE_DATASET
    cases = [c for c in load_dataset(dataset_path) if c.environment == environment_name]
    if case_filter is not None:
        cases = [c for c in cases if c.id in case_filter]

    environment = get_environment(environment_name)
    results = [
        evaluate_acceptance_case(
            case, environment,
            result_count=result_count,
            state_gate=state_gate,
            validation=validation,
            max_follow_ups=max_follow_ups,
        )
        for case in cases
    ]

    aggregate = acceptance_metrics.aggregate_acceptance(results)
    verdict = acceptance_metrics.acceptance_verdict(
        aggregate, environment=environment_name, thresholds=thresholds,
    )

    warnings: list[str] = []
    if environment_name != "production":
        warnings.append(
            "Fixture environment uses a FAKE embedding model. Results prove "
            "pipeline mechanics only and are not evidence of retrieval quality."
        )
    if aggregate.unverified_case_count:
        warnings.append(
            f"{aggregate.unverified_case_count} case(s) carry unverified ground "
            "truth - their pass/fail cannot certify anything until a human "
            "confirms the expected document and fact against the real index."
        )
    if not cases:
        warnings.append(f"No acceptance cases for environment {environment_name!r}.")

    sat = sat_status(cases, environment_name)
    if sat["critical_cases_without_expert_confirm"]:
        warnings.append(
            f"{len(sat['critical_cases_without_expert_confirm'])} legal/financial/safety "
            "case(s) have no expert_confirm - existence of a file in the index is not "
            "confirmation that its content is current or correct."
        )
    project_ids = sorted({c.project for c in cases if c.project})

    return AcceptanceRunArtifact(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        environment=environment.describe(),
        dataset_file=str(dataset_path),
        flags={
            "DOCUMENT_STATE_GATE_ENABLED": state_gate,
            "EVIDENCE_RUNTIME_VALIDATION_ENABLED": validation,
            "AUXILIARY_TERM_COVERAGE_ENABLED": False,
            "MULTI_QUERY_RETRIEVAL_ENABLED": False,
            "llm_replay": False,
        },
        case_count=len(cases),
        cases=[r.to_dict() for r in results],
        aggregate=aggregate.to_dict(),
        verdict=verdict,
        warnings=warnings,
        project_id=project_ids[0] if len(project_ids) == 1 else ", ".join(project_ids),
        index_fingerprint=str(environment.describe().get("index_fingerprint") or ""),
        dataset_version=read_dataset_version(dataset_path),
        verified_cases_count=sat["verified_cases_count"],
        pending_cases_count=sat["pending_cases_count"],
        sat_status=sat,
    )


def save_acceptance_run(run: AcceptanceRunArtifact, path: Path | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sha = run.git_sha or "nogit"
        path = RUNS_DIR / f"{stamp}_{sha}_acceptance_{run.environment['name']}.json"
    path.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
