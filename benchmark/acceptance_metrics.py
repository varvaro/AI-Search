"""Acceptance (product) metrics — "is AI Search usable for work over Box docs?"

Deliberately separate from pr74_metrics. That module answers a safety question
("can the tool make a false claim about a signed contract"); this one answers a
product question ("does the tool find the right document and tell me the right
thing faster than opening the folder myself"). The audit found those two being
scored together, which lets a green safety run read as a usable product.

Four things are measured per case, in pipeline order, so a failure names the
layer that broke:

  RETRIEVAL  document_found  — an expected document reached the top-k at all
  ANSWER     answer_correct  — the answer states the fact the case asked for,
                               and leans on the expected source
  SAFETY     critical_error  — the answer is wrong (or forbidden) on a case
                               whose criticality is legal/financial
  USER VALUE queries_to_answer, total_ms

`answer_correct` is a substring contract over `expected_answer_contains`, not an
LLM judge: a benchmark that asks a model to grade a model cannot be used as
deployment evidence. The cost is that the dataset must spell out the fact in
terms that will actually appear in the text; `expected_fact` carries the
human-readable version for the reviewer who writes the case.

Pure functions. No I/O, no ai_search import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from . import answer_evidence
from .dataset.schema import CRITICAL_CRITICALITY_NAMES, BenchmarkCase

LAYER_RETRIEVAL = "RETRIEVAL"
LAYER_ANSWER = "ANSWER"
LAYER_SAFETY = "SAFETY"
LAYER_NONE = "OK"

CRITICAL_CRITICALITIES = CRITICAL_CRITICALITY_NAMES

# The literal string HALLUCINATION_GUARD instructs the model to emit when the
# sources do not answer the question, and the one both renderers fall back to.
NOT_FOUND_MARKER = _NF = "nenalezeno v indexovanych dokumentech"
# Fixed scaffolding written by _render_structured_answer regardless of content.
# Removing it is what lets "an answer with no attributed source" be told apart
# from "an answer that correctly said it found nothing".
_ANSWER_SCAFFOLD = (
    "shrnuti:", "pozadovane dokumenty / kroky:", "nenalezene informace:", "zdroje:",
)
_ANSWER_FILLER = ("zadne", _NF, _NF + ".")


def _fold(text: str) -> str:
    return answer_evidence.fold(text)


def document_hit(expected: list[str], rows: list[dict]) -> tuple[bool, int | None]:
    """Did any expected document reach the pool, and at which 1-based rank?

    Substring match against path (falling back to document name) — the same
    rule benchmark/metrics.py already uses for expected_documents, so an
    acceptance case is written exactly like a retrieval case.
    """
    if not expected:
        return True, None
    needles = [_fold(e) for e in expected if e]
    for rank, row in enumerate(rows or [], start=1):
        haystack = _fold(str(row.get("path") or "") + " " + str(row.get("document") or ""))
        if any(needle in haystack for needle in needles):
            return True, rank
    return False, None


def has_substantive_claim(answer: dict | None) -> bool:
    """Does the rendered answer assert anything, once scaffolding is removed?

    Deliberately crude and deliberately not an LLM judge. Everything the
    renderers emit unconditionally (section headings, the "Žádné" filler, the
    not-found sentence) is dropped; whatever survives is content the model chose
    to write. The cost of this simplicity is that it cannot tell a supported
    claim from an unsupported one - only whether a claim was made at all, which
    is exactly what `unsupported_claim` pairs with an empty evidence set.
    """
    for raw in _fold(answer_evidence.answer_body(answer)).splitlines():
        line = raw.strip().lstrip("-").strip()
        # Drop a trailing "(zdroj: x)" note - it is attribution, not a claim.
        head, marker, _tail = line.partition("(zdroj:")
        line = (head if marker else line).strip()
        if not line or line in _ANSWER_SCAFFOLD or line in _ANSWER_FILLER:
            continue
        # Numbered area headings ("1. oblast") are content, so they stay.
        return True
    return False


def evaluate_citations(
    case: BenchmarkCase, answer: dict | None,
) -> tuple[bool | None, bool, bool]:
    """(citation_correct, forbidden_document_hit, unsupported_claim).

    Separate from `evaluate_acceptance_answer` because these three read the
    evidence tiers rather than the answer text, and because keeping the older
    function's signature stable avoids churn in its existing callers.

    citation_correct is None when the case states no citation contract - a
    metric with no ground truth must not be averaged into a rate as if it had
    passed.
    """
    rows = answer_evidence.evidence_rows(answer)
    evidence = answer_evidence.evidence_documents(answer)
    forbidden_hit = bool(answer_evidence.rows_match_any(rows, case.forbidden_document))

    citation_correct: bool | None = None
    if case.expected_source_contains or case.forbidden_document:
        expected_hits = answer_evidence.match_any(evidence, case.expected_source_contains)
        citation_correct = (
            len(expected_hits) == len(case.expected_source_contains) and not forbidden_hit
        )

    unsupported = has_substantive_claim(answer) and not evidence
    return citation_correct, forbidden_hit, unsupported


@dataclass
class AcceptanceCaseResult:
    id: str
    question: str
    category: str
    environment: str
    criticality: str
    ground_truth_status: str
    expected_outcome: str = "found"
    document_found: bool | None = None
    document_rank: int | None = None
    answer_correct: bool | None = None
    answer_used_expected_source: bool | None = None
    citation_correct: bool | None = None
    forbidden_document_hit: bool = False
    # Whether the case declared a forbidden_document contract at all - the
    # denominator for forbidden_document_rate. Without it the rate would be
    # diluted by every case that never had a revision trap to fall into.
    forbidden_document_measured: bool = False
    unsupported_claim: bool = False
    critical_error: bool = False
    failure_layer: str = LAYER_NONE
    failure_detail: str = ""
    queries_to_answer: int | None = None
    follow_ups_used: int = 0
    retrieval_ms: float = 0.0
    answer_ms: float = 0.0
    total_ms: float = 0.0
    retrieved_documents: list[str] = field(default_factory=list)
    evidence_documents: list[str] = field(default_factory=list)
    missing_phrases: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_acceptance_answer(
    case: BenchmarkCase,
    results: list[dict],
    answer: dict | None,
) -> tuple[bool | None, bool, list[str], bool | None]:
    """Evaluate ONE (question, results, answer) triple.

    Returns (document_found, answer_correct, missing_phrases,
    answer_used_expected_source).

    `document_found` is None when the case names no expected document (every
    negative case, plus open-ended management questions). Reporting those as
    found=True - which is what an empty-needle substring match yields - would
    inflate document_found_rate with cases that never tested retrieval.
    """
    found: bool | None
    if case.expected_documents:
        found, _rank = document_hit(case.expected_documents, results)
    else:
        found = None

    body = answer_evidence.answer_body(answer)
    folded = _fold(body)
    missing = [p for p in case.expected_answer_contains if p and _fold(p) not in folded]
    forbidden_present = [p for p in case.forbidden_answer_contains if p and _fold(p) in folded]

    used_expected: bool | None = None
    if case.expected_source_contains:
        evidence = answer_evidence.evidence_documents(answer)
        hits = answer_evidence.match_any(evidence, case.expected_source_contains)
        used_expected = len(hits) == len(case.expected_source_contains)

    correct = (
        not missing
        and not forbidden_present
        and (used_expected is not False)
        and bool(body.strip())
    )
    if forbidden_present:
        missing = missing + [f"forbidden:{p}" for p in forbidden_present]
    return found, correct, missing, used_expected


def classify_failure(
    case: BenchmarkCase,
    document_found: bool | None,
    answer_correct: bool,
    answer_used_expected_source: bool | None,
    *,
    forbidden_document_hit: bool = False,
) -> tuple[str, str]:
    """Which layer is responsible — the whole point of the acceptance report."""
    if document_found is False:
        return LAYER_RETRIEVAL, (
            f"expected document {case.expected_documents} never reached the result pool"
        )
    if forbidden_document_hit:
        return LAYER_SAFETY, (
            f"answer leaned on a forbidden document {case.forbidden_document} - "
            "superseded revision, draft, or another project"
        )
    if answer_correct:
        return LAYER_NONE, ""
    if answer_used_expected_source is False:
        return LAYER_ANSWER, "document was retrieved but the answer does not lean on it"
    if case.criticality in CRITICAL_CRITICALITIES:
        return LAYER_SAFETY, (
            f"wrong answer on a {case.criticality} question - this is a critical error"
        )
    return LAYER_ANSWER, "document was retrieved but the answer misses the required fact"


def is_critical_error(
    case: BenchmarkCase,
    answer_correct: bool,
    forbidden_hit: bool,
    *,
    forbidden_document_hit: bool = False,
) -> bool:
    """A critical error is a WRONG answer where being wrong costs money, creates
    legal exposure or gets built into the structure, plus any forbidden phrase
    or forbidden source regardless of category.

    Sourcing an answer from a superseded revision is critical on its own: the
    text can be entirely accurate about a drawing that was withdrawn.
    """
    if forbidden_hit or forbidden_document_hit:
        return True
    return (not answer_correct) and case.criticality in CRITICAL_CRITICALITIES


@dataclass
class AcceptanceAggregate:
    case_count: int = 0
    evaluated_count: int = 0
    errored_count: int = 0
    document_found_count: int = 0
    document_found_rate: float | None = None
    # Retrieval KPI denominators are the cases that actually name an expected
    # document; negative and open-ended cases are excluded, not counted as hits.
    retrieval_measured_count: int = 0
    top1_count: int = 0
    top1_accuracy: float | None = None
    top5_count: int = 0
    top5_accuracy: float | None = None
    answer_correct_count: int = 0
    answer_correct_rate: float | None = None
    citation_measured_count: int = 0
    citation_correct_count: int = 0
    citation_correct_rate: float | None = None
    forbidden_document_measured_count: int = 0
    forbidden_document_hit_count: int = 0
    forbidden_document_rate: float | None = None
    forbidden_document_case_ids: list[str] = field(default_factory=list)
    unsupported_claim_count: int = 0
    unsupported_claim_rate: float | None = None
    unsupported_claim_case_ids: list[str] = field(default_factory=list)
    critical_error_count: int = 0
    critical_error_case_ids: list[str] = field(default_factory=list)
    mean_total_ms: float | None = None
    p95_total_ms: float | None = None
    mean_queries_to_answer: float | None = None
    resolved_within_one_query: int = 0
    resolved_with_follow_up: int = 0
    unresolved: int = 0
    verified_case_count: int = 0
    unverified_case_count: int = 0
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_layer: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    return ordered[index]


def aggregate_acceptance(results: list[AcceptanceCaseResult]) -> AcceptanceAggregate:
    agg = AcceptanceAggregate(case_count=len(results))
    evaluated = [r for r in results if r.error is None]
    agg.evaluated_count = len(evaluated)
    agg.errored_count = len(results) - len(evaluated)
    if not evaluated:
        return agg

    retrieval_measured = [r for r in evaluated if r.document_found is not None]
    agg.retrieval_measured_count = len(retrieval_measured)
    agg.document_found_count = sum(1 for r in retrieval_measured if r.document_found)
    agg.document_found_rate = _ratio(agg.document_found_count, len(retrieval_measured))
    agg.top1_count = sum(1 for r in retrieval_measured if r.document_rank == 1)
    agg.top1_accuracy = _ratio(agg.top1_count, len(retrieval_measured))
    agg.top5_count = sum(
        1 for r in retrieval_measured if r.document_rank is not None and r.document_rank <= 5
    )
    agg.top5_accuracy = _ratio(agg.top5_count, len(retrieval_measured))

    agg.answer_correct_count = sum(1 for r in evaluated if r.answer_correct)
    agg.answer_correct_rate = _ratio(agg.answer_correct_count, len(evaluated))

    citation_measured = [r for r in evaluated if r.citation_correct is not None]
    agg.citation_measured_count = len(citation_measured)
    agg.citation_correct_count = sum(1 for r in citation_measured if r.citation_correct)
    agg.citation_correct_rate = _ratio(agg.citation_correct_count, len(citation_measured))

    forbidden_measured = [r for r in evaluated if r.forbidden_document_measured]
    agg.forbidden_document_measured_count = len(forbidden_measured)
    agg.forbidden_document_case_ids = [r.id for r in evaluated if r.forbidden_document_hit]
    agg.forbidden_document_hit_count = len(agg.forbidden_document_case_ids)
    agg.forbidden_document_rate = _ratio(
        agg.forbidden_document_hit_count, len(forbidden_measured),
    )

    agg.unsupported_claim_case_ids = [r.id for r in evaluated if r.unsupported_claim]
    agg.unsupported_claim_count = len(agg.unsupported_claim_case_ids)
    agg.unsupported_claim_rate = _ratio(agg.unsupported_claim_count, len(evaluated))

    agg.critical_error_case_ids = [r.id for r in evaluated if r.critical_error]
    agg.critical_error_count = len(agg.critical_error_case_ids)

    totals = [r.total_ms for r in evaluated if r.total_ms]
    if totals:
        agg.mean_total_ms = sum(totals) / len(totals)
        agg.p95_total_ms = _p95(totals)

    resolved = [r.queries_to_answer for r in evaluated if r.queries_to_answer is not None]
    if resolved:
        agg.mean_queries_to_answer = sum(resolved) / len(resolved)
    agg.resolved_within_one_query = sum(1 for r in evaluated if r.queries_to_answer == 1)
    agg.resolved_with_follow_up = sum(
        1 for r in evaluated if r.queries_to_answer is not None and r.queries_to_answer > 1
    )
    agg.unresolved = sum(1 for r in evaluated if r.queries_to_answer is None)

    agg.verified_case_count = sum(1 for r in evaluated if r.ground_truth_status == "verified")
    agg.unverified_case_count = len(evaluated) - agg.verified_case_count

    for r in evaluated:
        agg.by_layer[r.failure_layer] = agg.by_layer.get(r.failure_layer, 0) + 1

    buckets: dict[str, list[AcceptanceCaseResult]] = {}
    for r in evaluated:
        buckets.setdefault(r.category or "UNCATEGORIZED", []).append(r)
    for category, items in sorted(buckets.items()):
        measured = [r for r in items if r.document_found is not None]
        agg.by_category[category] = {
            "case_count": len(items),
            "document_found_rate": _ratio(
                sum(1 for r in measured if r.document_found), len(measured),
            ),
            "top5_accuracy": _ratio(
                sum(1 for r in measured if r.document_rank is not None and r.document_rank <= 5),
                len(measured),
            ),
            "answer_correct_rate": _ratio(sum(1 for r in items if r.answer_correct), len(items)),
            "critical_errors": sum(1 for r in items if r.critical_error),
            "unsupported_claims": sum(1 for r in items if r.unsupported_claim),
        }
    return agg


# --- Acceptance thresholds ---------------------------------------------------
# Defaults follow the PR7.4 audit recommendation. They are values, not code, so
# a stricter contract is a config change rather than an edit here.
DEFAULT_THRESHOLDS = {
    "min_document_found_rate": 0.90,
    "min_answer_correct_rate": 0.80,
    "max_critical_errors": 0,
    "min_verified_cases": 30,
    "max_p95_total_ms": None,  # set once a latency budget is agreed
    # PR7.5 acceptance criteria for one project.
    "min_top5_accuracy": 0.90,
    "min_citation_correct_rate": 0.95,
    "max_unsupported_claims": 0,
    "max_forbidden_document_hits": 0,
}


def acceptance_verdict(
    agg: AcceptanceAggregate,
    *,
    environment: str,
    thresholds: dict | None = None,
) -> dict:
    """GO / NO-GO / INCONCLUSIVE.

    INCONCLUSIVE is a first-class outcome and exists to stop the single most
    likely misuse of this harness: a green fixture run being read as permission
    to deploy. The fixture corpus uses a fake embedding model (benchmark/
    fixtures.py), so it can prove pipeline mechanics and nothing about real
    retrieval quality. Unverified ground truth is the same problem one level up.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    is_production = environment == "production"
    quality_failures: list[str] = []
    harness_failures: list[str] = []
    inconclusive: list[str] = []

    if not is_production:
        inconclusive.append(
            f"environment={environment!r} uses the synthetic corpus and a fake embedding "
            "model - mechanics only, never deployment evidence"
        )
    if agg.unverified_case_count:
        inconclusive.append(
            f"{agg.unverified_case_count} case(s) have unverified ground truth"
        )
    if agg.verified_case_count < limits["min_verified_cases"]:
        inconclusive.append(
            f"only {agg.verified_case_count} verified cases, "
            f"{limits['min_verified_cases']} required"
        )

    if agg.critical_error_count > limits["max_critical_errors"]:
        quality_failures.append(
            f"critical_errors={agg.critical_error_count} "
            f"({agg.critical_error_case_ids})"
        )
    if agg.document_found_rate is not None and agg.document_found_rate < limits["min_document_found_rate"]:
        quality_failures.append(
            f"document_found_rate={agg.document_found_rate:.2f} < {limits['min_document_found_rate']}"
        )
    if agg.answer_correct_rate is not None and agg.answer_correct_rate < limits["min_answer_correct_rate"]:
        quality_failures.append(
            f"answer_correct_rate={agg.answer_correct_rate:.2f} < {limits['min_answer_correct_rate']}"
        )
    if agg.top5_accuracy is not None and agg.top5_accuracy < limits["min_top5_accuracy"]:
        quality_failures.append(
            f"top5_accuracy={agg.top5_accuracy:.2f} < {limits['min_top5_accuracy']}"
        )
    if (
        agg.citation_correct_rate is not None
        and agg.citation_correct_rate < limits["min_citation_correct_rate"]
    ):
        quality_failures.append(
            f"citation_correct_rate={agg.citation_correct_rate:.2f} < "
            f"{limits['min_citation_correct_rate']}"
        )
    if agg.unsupported_claim_count > limits["max_unsupported_claims"]:
        quality_failures.append(
            f"unsupported_claims={agg.unsupported_claim_count} "
            f"({agg.unsupported_claim_case_ids})"
        )
    if agg.forbidden_document_hit_count > limits["max_forbidden_document_hits"]:
        quality_failures.append(
            f"forbidden_document_hits={agg.forbidden_document_hit_count} "
            f"({agg.forbidden_document_case_ids}) - answer used a superseded "
            "revision, a draft, or another project's document"
        )
    budget = limits.get("max_p95_total_ms")
    if budget and agg.p95_total_ms and agg.p95_total_ms > budget:
        quality_failures.append(f"p95_total_ms={agg.p95_total_ms:.0f} > {budget}")

    # A harness failure is attributable everywhere: if cases blew up, the run
    # is broken regardless of which corpus it ran against.
    if agg.errored_count:
        harness_failures.append(f"errored_cases={agg.errored_count}")

    # Quality failures only become a verdict on production. Off production the
    # embedding model is fake, so a low answer_correct_rate says nothing about
    # the product - calling that NO-GO would be the same false-confidence
    # mistake as a green fixture GO, just pointing the other way. The numbers
    # are still reported, labelled as non-attributable.
    if is_production:
        blockers = harness_failures + quality_failures
    else:
        blockers = harness_failures
        inconclusive.extend(
            f"observed on {environment} but NOT attributable to the product: {failure}"
            for failure in quality_failures
        )

    if blockers:
        verdict = "NO-GO"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "GO"

    return {
        "verdict": verdict,
        "blockers": blockers,
        "observed_failures": quality_failures,
        "inconclusive_reasons": inconclusive,
        "thresholds": limits,
        "criteria": {
            "zero_critical_errors": agg.critical_error_count == 0,
            "document_found_rate_ok": (
                agg.document_found_rate is not None
                and agg.document_found_rate >= limits["min_document_found_rate"]
            ),
            "answer_correct_rate_ok": (
                agg.answer_correct_rate is not None
                and agg.answer_correct_rate >= limits["min_answer_correct_rate"]
            ),
            "top5_accuracy_ok": (
                agg.top5_accuracy is not None
                and agg.top5_accuracy >= limits["min_top5_accuracy"]
            ),
            "citation_correct_rate_ok": (
                agg.citation_correct_rate is not None
                and agg.citation_correct_rate >= limits["min_citation_correct_rate"]
            ),
            "zero_unsupported_claims": agg.unsupported_claim_count == 0,
            "zero_forbidden_document_hits": agg.forbidden_document_hit_count == 0,
            "ground_truth_verified": agg.unverified_case_count == 0,
            "production_environment": environment == "production",
            "no_errored_cases": agg.errored_count == 0,
        },
    }
