"""PR7.4 answer-quality metrics — pure, deterministic, no LLM-as-judge.

Operates on baseline/candidate answer dicts (ai_search.answer() output) plus
the BenchmarkCase assertions. No I/O, no ai_search imports.

PR7.4.1 changes (audit P1, P4, P6):

  * Every failure carries a LAYER, so a red run says WHERE it broke instead of
    only that it broke:
      RETRIEVAL  the document never reached the pool (or a distractor did)
      EVIDENCE   the pool was right but the state/intent verdict was wrong
      ANSWER     the evidence was right but the answer did not use it
      SAFETY     the answer asserted something the evidence does not support
  * Safety assertions are judged against the documents the answer actually
    leaned on (answer_evidence.evidence_documents = cited ∪ state), not against
    the whole retrieval pool. Under the old rule `forbidden_sources` measured
    "was a distractor retrieved", which on an adversarial dataset is true by
    construction - a maximally safe "cannot verify" answer scored BLOCKING.
  * Baseline and candidate violations are computed by ONE function
    (_answer_violations), which structurally removes the old asymmetry where a
    code known only to the candidate side turned an unchanged violation into a
    false "degraded".

Severity model (independent of any aggregate score):
  BLOCKING — false signed confirmation, answer leaning on a wrong-entity
             document, unsupported positive signed claim
  HIGH     — unsupported negative/absence claim; missing proof for a verdict
             the case requires
  MEDIUM   — expected evidence retrieved but not used; wrong intent coverage
  LOW      — wording-only change; distractor merely present in the pool

`has_blocking_regression` is a separate bool — never derived from a blended
safety_score. A single blocking failure makes the run NO-GO regardless of any
average.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from . import answer_evidence
from .dataset.schema import BenchmarkCase

# Phrase-adjacent claim detectors — same spirit as ai_search.py's gate
# regexes, duplicated here so the metric layer stays dependency-free of
# ai_search internals (benchmark package must not import production gate
# helpers just to measure them).
_POSITIVE_SIGNED_RE = re.compile(
    r"\b(je|jsou|byla|byl|existuje)\b\s+podepsan\w*",
    re.UNICODE,
)
_NEGATIVE_SIGNED_RE = re.compile(
    r"\b(neni|nebyla|nebyl|nejsou|neexistuje)\b\s+podepsan\w*"
    r"|\bane\b[^\n]{0,40}podepsan\w*",
    re.UNICODE,
)

LAYER_RETRIEVAL = "RETRIEVAL"
LAYER_EVIDENCE = "EVIDENCE"
LAYER_ANSWER = "ANSWER"
LAYER_SAFETY = "SAFETY"
ALL_LAYERS = (LAYER_RETRIEVAL, LAYER_EVIDENCE, LAYER_ANSWER, LAYER_SAFETY)

SEVERITY_BLOCKING = "BLOCKING"
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"
ALL_SEVERITIES = (SEVERITY_BLOCKING, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW)


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def _answer_text(answer: dict | None) -> str:
    if not answer:
        return ""
    return str(answer.get("answer") or "")


def _contains_any(haystack: str, needles: list[str]) -> list[str]:
    folded = _fold(haystack)
    return [n for n in needles if n and _fold(n) in folded]


def _has_positive_signed_claim(text: str) -> bool:
    return bool(_POSITIVE_SIGNED_RE.search(_fold(text)))


def _has_negative_signed_claim(text: str) -> bool:
    return bool(_NEGATIVE_SIGNED_RE.search(_fold(text)))


def _validation(answer: dict | None) -> dict:
    if not answer:
        return {}
    value = answer.get("validation")
    return value if isinstance(value, dict) else {}


@dataclass
class Failure:
    severity: str
    code: str
    detail: str
    layer: str = LAYER_ANSWER


@dataclass
class CaseEvaluation:
    id: str
    category: str
    answer_delta: str  # unchanged | improved | degraded | changed_neutral
    failures: list[Failure] = field(default_factory=list)
    state_verdict_actual: str | None = None
    state_verdict_expected: str | None = None
    state_verdict_match: bool | None = None
    intent_coverage_actual: str | None = None
    intent_coverage_expected: str | None = None
    intent_coverage_match: bool | None = None
    missing_needs_match: bool | None = None
    baseline_answer_len: int = 0
    candidate_answer_len: int = 0
    # PR7.4.1 auditability: what the candidate answer actually leaned on.
    evidence_tiers: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_blocking(self) -> bool:
        return any(f.severity == SEVERITY_BLOCKING for f in self.failures)

    @property
    def failure_layers(self) -> list[str]:
        """Layers that produced at least one failure, in pipeline order."""
        present = {f.layer for f in self.failures}
        return [layer for layer in ALL_LAYERS if layer in present]

    @property
    def first_failure_layer(self) -> str | None:
        """Earliest pipeline layer that broke — where to start debugging."""
        layers = self.failure_layers
        return layers[0] if layers else None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["failure_layers"] = self.failure_layers
        data["first_failure_layer"] = self.first_failure_layer
        data["has_blocking"] = self.has_blocking
        return data


def _answer_violations(case: BenchmarkCase, answer: dict | None) -> list[Failure]:
    """Answer-level and safety-level violations of ONE answer.

    Used for the candidate AND for the baseline, so improved/degraded compares
    like with like. Verdict-level (EVIDENCE) checks live in
    evaluate_case_answers because they only make sense for the candidate - the
    baseline runs with validation OFF and has no verdict to compare.
    """
    failures: list[Failure] = []
    text = _answer_text(answer)
    state_expected = case.expected_state_verdict

    tiers = answer_evidence.evidence_tiers(answer)
    evidence = tiers["evidence"]
    retrieved = tiers["retrieved"]

    forbidden_hits = _contains_any(text, case.forbidden_answer_contains)
    if forbidden_hits:
        failures.append(Failure(
            SEVERITY_BLOCKING, "forbidden_answer_contains",
            f"answer contains forbidden phrases: {forbidden_hits}",
            LAYER_SAFETY,
        ))

    # P1: judged against what the answer leaned on, not against the pool.
    used_forbidden = answer_evidence.match_any(evidence, case.forbidden_sources)
    if used_forbidden:
        failures.append(Failure(
            SEVERITY_BLOCKING, "wrong_entity_citation",
            f"answer leans on forbidden sources: {used_forbidden}",
            LAYER_SAFETY,
        ))
    # A distractor that was merely retrieved is a retrieval observation. It is
    # recorded so the debt stays visible, but it can never fail a run: the gate
    # is not responsible for what the ranker put in the pool.
    only_retrieved_forbidden = [
        n for n in answer_evidence.match_any(retrieved, case.forbidden_sources)
        if n not in used_forbidden
    ]
    if only_retrieved_forbidden:
        failures.append(Failure(
            SEVERITY_LOW, "forbidden_source_retrieved",
            f"forbidden sources present in the pool but unused: {only_retrieved_forbidden}",
            LAYER_RETRIEVAL,
        ))

    for needle in case.expected_source_contains:
        if not needle:
            continue
        if answer_evidence.match_any(evidence, [needle]):
            continue
        if answer_evidence.match_any(retrieved, [needle]):
            # Retrieval did its job; the answer ignored the proof.
            severity = SEVERITY_HIGH if state_expected == "SIGNED_CONFIRMED" else SEVERITY_MEDIUM
            failures.append(Failure(
                severity, "expected_source_not_cited",
                f"expected source {needle!r} was retrieved but the answer does not use it",
                LAYER_ANSWER,
            ))
        else:
            failures.append(Failure(
                SEVERITY_MEDIUM, "expected_source_not_retrieved",
                f"expected source {needle!r} never reached the result pool",
                LAYER_RETRIEVAL,
            ))

    positive = _has_positive_signed_claim(text)
    negative = _has_negative_signed_claim(text)
    if state_expected == "SIGNED_CONFIRMED" and negative and not positive:
        failures.append(Failure(
            SEVERITY_HIGH, "unsupported_negative_signed_claim",
            "expected SIGNED_CONFIRMED but the answer denies signedness",
            LAYER_SAFETY,
        ))
    if state_expected == "UNSIGNED_CONFIRMED" and positive and not negative:
        failures.append(Failure(
            SEVERITY_BLOCKING, "unsupported_positive_signed_claim",
            "expected UNSIGNED_CONFIRMED but the answer affirms signedness",
            LAYER_SAFETY,
        ))
    if state_expected in ("ENTITY_MISMATCH", "UNVERIFIED") and positive:
        failures.append(Failure(
            SEVERITY_BLOCKING if state_expected == "ENTITY_MISMATCH" else SEVERITY_HIGH,
            "false_signed_confirmation",
            f"answer affirms signedness under {state_expected}",
            LAYER_SAFETY,
        ))
    if state_expected in ("ENTITY_MISMATCH", "UNVERIFIED") and negative and not positive:
        failures.append(Failure(
            SEVERITY_HIGH, "unsupported_absence_claim",
            f"answer denies signedness under {state_expected}",
            LAYER_SAFETY,
        ))
    return failures


# Codes that represent a real regression when they appear (or disappear)
# between baseline and candidate. Derived from one place so the baseline and
# candidate sides can never drift apart again (audit P6).
_REGRESSION_CODES = frozenset({
    "forbidden_answer_contains",
    "wrong_entity_citation",
    "expected_source_not_cited",
    "unsupported_positive_signed_claim",
    "unsupported_negative_signed_claim",
    "false_signed_confirmation",
    "unsupported_absence_claim",
})


def _regression_codes(failures: list[Failure]) -> list[str]:
    return sorted(f.code for f in failures if f.code in _REGRESSION_CODES)


def evaluate_case_answers(
    case: BenchmarkCase,
    baseline_answer: dict | None,
    candidate_answer: dict | None,
) -> CaseEvaluation:
    """Deterministic per-case evaluation of baseline vs candidate answers."""
    base_text = _answer_text(baseline_answer)
    cand_text = _answer_text(candidate_answer)
    validation = _validation(candidate_answer)

    failures: list[Failure] = []

    # --- EVIDENCE layer: the verdict the runtime produced -------------------
    state_actual = validation.get("state_verdict")
    state_expected = case.expected_state_verdict
    state_match: bool | None = None
    if state_expected is not None:
        if state_actual is None:
            state_match = False
            failures.append(Failure(
                SEVERITY_HIGH, "missing_state_verdict",
                f"expected {state_expected}, candidate validation has no state_verdict",
                LAYER_EVIDENCE,
            ))
        else:
            state_match = state_actual == state_expected
            if not state_match:
                failures.append(Failure(
                    SEVERITY_HIGH, "state_verdict_mismatch",
                    f"expected {state_expected}, got {state_actual}",
                    LAYER_EVIDENCE,
                ))

    intent_actual = validation.get("intent_coverage")
    intent_expected = case.expected_intent_coverage
    intent_match: bool | None = None
    if intent_expected is not None:
        if intent_actual is None:
            intent_match = False
            failures.append(Failure(
                SEVERITY_MEDIUM, "missing_intent_coverage",
                f"expected {intent_expected}, candidate validation has no intent_coverage",
                LAYER_EVIDENCE,
            ))
        else:
            intent_match = intent_actual == intent_expected
            if not intent_match:
                failures.append(Failure(
                    SEVERITY_MEDIUM, "intent_coverage_mismatch",
                    f"expected {intent_expected}, got {intent_actual}",
                    LAYER_EVIDENCE,
                ))

    missing_match: bool | None = None
    if case.expected_missing_needs:
        actual_missing = set(validation.get("missing_needs") or [])
        expected_missing = set(case.expected_missing_needs)
        missing_match = actual_missing == expected_missing
        if not missing_match:
            failures.append(Failure(
                SEVERITY_MEDIUM, "missing_needs_mismatch",
                f"expected {sorted(expected_missing)}, got {sorted(actual_missing)}",
                LAYER_EVIDENCE,
            ))

    # --- RETRIEVAL / ANSWER / SAFETY layers ---------------------------------
    candidate_violations = _answer_violations(case, candidate_answer)
    failures.extend(candidate_violations)

    # Hedge incorrectly rewritten: baseline made no claim, candidate invented one.
    base_has_claim = _has_positive_signed_claim(base_text) or _has_negative_signed_claim(base_text)
    cand_has_claim = _has_positive_signed_claim(cand_text) or _has_negative_signed_claim(cand_text)
    if not base_has_claim and cand_has_claim and base_text != cand_text:
        if case.category == "REGRESSION" or state_expected == "NOOP":
            failures.append(Failure(
                SEVERITY_MEDIUM, "hedge_incorrectly_rewritten",
                "candidate introduced a signedness claim where the baseline had none",
                LAYER_ANSWER,
            ))

    # --- Answer delta: invariant violations, not text diff -------------------
    if base_text == cand_text:
        answer_delta = "unchanged"
    else:
        base_bad = _regression_codes(_answer_violations(case, baseline_answer))
        cand_bad = _regression_codes(candidate_violations)
        if len(cand_bad) < len(base_bad):
            answer_delta = "improved"
        elif len(cand_bad) > len(base_bad):
            answer_delta = "degraded"
        elif base_bad != cand_bad:
            # Same count, different violations — traded one defect for another.
            answer_delta = "degraded"
        else:
            answer_delta = "changed_neutral"
            if case.category == "REGRESSION" and state_expected in (None, "NOOP"):
                failures.append(Failure(
                    SEVERITY_LOW, "regression_wording_change",
                    "REGRESSION/NOOP case changed answer text under candidate flags",
                    LAYER_ANSWER,
                ))

    return CaseEvaluation(
        id=case.id,
        category=case.category or "",
        answer_delta=answer_delta,
        failures=failures,
        state_verdict_actual=state_actual,
        state_verdict_expected=state_expected,
        state_verdict_match=state_match,
        intent_coverage_actual=intent_actual,
        intent_coverage_expected=intent_expected,
        intent_coverage_match=intent_match,
        missing_needs_match=missing_match,
        baseline_answer_len=len(base_text),
        candidate_answer_len=len(cand_text),
        evidence_tiers=answer_evidence.evidence_tiers(candidate_answer),
    )


@dataclass
class AggregateMetrics:
    case_count: int = 0
    state_verdict_accuracy: float | None = None
    signed_confirmed_precision: float | None = None
    signed_confirmed_recall: float | None = None
    entity_mismatch_accuracy: float | None = None
    unverified_accuracy: float | None = None
    intent_coverage_accuracy: float | None = None
    missing_need_accuracy: float | None = None
    false_signed_confirmations: int = 0
    wrong_entity_citations: int = 0
    unsupported_negative_signed_claims: int = 0
    unsupported_positive_signed_claims: int = 0
    hedge_incorrectly_rewritten: int = 0
    unchanged_count: int = 0
    changed_count: int = 0
    improved_count: int = 0
    degraded_count: int = 0
    changed_neutral_count: int = 0
    blocking_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    has_blocking_regression: bool = False
    blocking_case_ids: list[str] = field(default_factory=list)
    safety_score: float | None = None
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    # PR7.4.1: where failures came from.
    by_layer: dict[str, dict[str, Any]] = field(default_factory=dict)
    first_failure_layer_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def aggregate_evaluations(evaluations: list[CaseEvaluation]) -> AggregateMetrics:
    agg = AggregateMetrics(case_count=len(evaluations))
    if not evaluations:
        return agg

    state_cases = [e for e in evaluations if e.state_verdict_expected is not None]
    state_hits = sum(1 for e in state_cases if e.state_verdict_match)
    agg.state_verdict_accuracy = _ratio(state_hits, len(state_cases))

    pred_signed = [e for e in state_cases if e.state_verdict_actual == "SIGNED_CONFIRMED"]
    exp_signed = [e for e in state_cases if e.state_verdict_expected == "SIGNED_CONFIRMED"]
    tp_signed = sum(1 for e in pred_signed if e.state_verdict_expected == "SIGNED_CONFIRMED")
    agg.signed_confirmed_precision = _ratio(tp_signed, len(pred_signed))
    agg.signed_confirmed_recall = _ratio(
        sum(1 for e in exp_signed if e.state_verdict_actual == "SIGNED_CONFIRMED"),
        len(exp_signed),
    )

    for label, attr in (
        ("ENTITY_MISMATCH", "entity_mismatch_accuracy"),
        ("UNVERIFIED", "unverified_accuracy"),
    ):
        subset = [e for e in state_cases if e.state_verdict_expected == label]
        hits = sum(1 for e in subset if e.state_verdict_match)
        setattr(agg, attr, _ratio(hits, len(subset)))

    intent_cases = [e for e in evaluations if e.intent_coverage_expected is not None]
    agg.intent_coverage_accuracy = _ratio(
        sum(1 for e in intent_cases if e.intent_coverage_match), len(intent_cases),
    )
    missing_cases = [e for e in evaluations if e.missing_needs_match is not None]
    agg.missing_need_accuracy = _ratio(
        sum(1 for e in missing_cases if e.missing_needs_match), len(missing_cases),
    )

    code_counters = {
        "false_signed_confirmation": "false_signed_confirmations",
        "wrong_entity_citation": "wrong_entity_citations",
        "unsupported_negative_signed_claim": "unsupported_negative_signed_claims",
        "unsupported_positive_signed_claim": "unsupported_positive_signed_claims",
        "hedge_incorrectly_rewritten": "hedge_incorrectly_rewritten",
    }
    layer_stats = {
        layer: {"failures": 0, "blocking": 0, "cases": 0, "codes": {}}
        for layer in ALL_LAYERS
    }
    for e in evaluations:
        for f in e.failures:
            attr = code_counters.get(f.code)
            if attr:
                setattr(agg, attr, getattr(agg, attr) + 1)
            if f.severity == SEVERITY_BLOCKING:
                agg.blocking_count += 1
            elif f.severity == SEVERITY_HIGH:
                agg.high_count += 1
            elif f.severity == SEVERITY_MEDIUM:
                agg.medium_count += 1
            elif f.severity == SEVERITY_LOW:
                agg.low_count += 1
            bucket = layer_stats.setdefault(
                f.layer, {"failures": 0, "blocking": 0, "cases": 0, "codes": {}}
            )
            bucket["failures"] += 1
            if f.severity == SEVERITY_BLOCKING:
                bucket["blocking"] += 1
            bucket["codes"][f.code] = bucket["codes"].get(f.code, 0) + 1
        for layer in e.failure_layers:
            layer_stats[layer]["cases"] += 1
        first = e.first_failure_layer
        if first:
            agg.first_failure_layer_counts[first] = agg.first_failure_layer_counts.get(first, 0) + 1
        if e.has_blocking:
            agg.blocking_case_ids.append(e.id)
        if e.answer_delta == "unchanged":
            agg.unchanged_count += 1
        else:
            agg.changed_count += 1
            if e.answer_delta == "improved":
                agg.improved_count += 1
            elif e.answer_delta == "degraded":
                agg.degraded_count += 1
            else:
                agg.changed_neutral_count += 1

    agg.by_layer = {
        layer: dict(stats) for layer, stats in layer_stats.items()
    }
    agg.has_blocking_regression = bool(agg.blocking_case_ids)
    # Informational only — never the GO/NO-GO gate.
    penalty = (
        agg.blocking_count * 1.0
        + agg.high_count * 0.3
        + agg.medium_count * 0.05
    )
    agg.safety_score = max(0.0, 1.0 - penalty / max(1, agg.case_count))

    by_cat: dict[str, list[CaseEvaluation]] = {}
    for e in evaluations:
        by_cat.setdefault(e.category or "UNCATEGORIZED", []).append(e)
    for cat, items in sorted(by_cat.items()):
        agg.by_category[cat] = {
            "case_count": len(items),
            "unchanged": sum(1 for e in items if e.answer_delta == "unchanged"),
            "improved": sum(1 for e in items if e.answer_delta == "improved"),
            "degraded": sum(1 for e in items if e.answer_delta == "degraded"),
            "blocking": sum(1 for e in items if e.has_blocking),
            "state_verdict_accuracy": _ratio(
                sum(1 for e in items if e.state_verdict_match),
                sum(1 for e in items if e.state_verdict_expected is not None),
            ),
        }
    return agg
