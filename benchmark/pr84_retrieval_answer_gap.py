"""PR8.4 — retrieval_hit_answer_miss measurement (read-only audit).

Audit finding
--------------
Retrieval succeeds (`document_found=True`) but the ANSWER layer still fails
its contract. Tracing nds-status-04 / nds-qa-03 / nds-qa-09 back through
`ai_search.py` shows one shared root cause, not three:

  `_render_answer_item()` resolves a model-produced `zdroj_index` via
  `_clamp_source_index(index, len(results))`. When that fails to resolve
  (missing / null / out-of-range index — which an LLM emits often; see
  nds-status-04 flipping between pass and fail across two otherwise-identical
  runs) `document` is `None`, and BOTH renderers keep the claim's text while
  silently dropping the citation:

    `_render_concise_answer`:    `if document: block += f"\\n(Zdroj: {document})"`
    `_render_structured_answer`: `source_note = f" (Zdroj: {document})" if document else ""`

  Neither path drops the *item* itself — only the attribution disappears.
  `benchmark/acceptance_metrics.evaluate_citations` then sees a substantive
  claim with zero parseable evidence documents and marks the case
  `unsupported_claim=True` (nds-status-04 PR8.3-subset run, nds-qa-03,
  nds-qa-09). This is exactly "citovaný dokument musí být zdroj tvrzení" being
  violated at the rendering step — retrieval, ranking, embeddings, PR8.1/8.2
  and the OLD guard are all untouched and irrelevant to it.

  A second, unrelated failure mode is superficially identical in the FAT
  aggregate (`unsupported_claim=True`, retrieval hit) but has nothing to do
  with evidence extraction: `nds-adv-04` failed because `_call_ollama` timed
  out, and `answer()`'s own except-block text ("Ollama je nedostupná: ...")
  is itself a "substantive claim" with no citation. Lumping this in with the
  citation-rendering gap would misattribute an infra flake as an
  evidence-extraction defect, so this module reports it separately
  (`GAP_INFRA_ERROR`) and excludes it from the headline gap rate.

  Running this over the full FAT v2 artifact (see `main()`/CLI below) surfaces
  a THIRD pattern the PR8.4 brief explicitly names ("pokud evidence obsahuje
  fakt, odpověď ho nesmí ignorovat"): several `answer_correct=False` rows
  (e.g. nds-doc-03, nds-adv-01, nds-cm-02) turn out to be plain abstention -
  `answer_text == "Nenalezeno v indexovaných dokumentech."` - even though
  `document_found=True`, i.e. the correct row reached the pool but the
  answer step never engaged with it at all (evidence-safety abstention,
  DocumentState NOOP, or the model itself declining). That is a different
  shape of bug from "wrong/incomplete fact despite a citation" (e.g.
  nds-status-03, where sources ARE cited but the stated fact is wrong), so
  it gets its own label, `GAP_ABSTAINED_DESPITE_HIT`, instead of being
  folded into `GAP_WRONG_FACT`. `has_substantive_claim` (imported from
  `acceptance_metrics`, the existing grading source of truth) is reused
  verbatim to tell the two apart, rather than re-implementing the
  scaffolding/filler detection here.

This module is purely diagnostic: it classifies already-produced FAT/
acceptance case results (dicts shaped like `NdsCaseResult.to_dict()`). It
does not call `ai_search`, does not re-run any case, and does not change
retrieval/ranking/answer generation.

CLI:
  PYTHONPATH=. python -m benchmark.pr84_retrieval_answer_gap <cases.json>
"""
from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import acceptance_metrics

# Substrings (folded, lowercase) that mark an answer() failure as backend/
# infra unavailability rather than an evidence/citation defect. Mirrors the
# literal text ai_search.answer() writes in its own except-block, e.g.
# f"Ollama je nedostupná: {type(exc2).__name__}. ...".
_INFRA_ERROR_MARKERS = (
    "ollama je nedostupna",
    "timeouterror",
    "connectionerror",
    "connection refused",
)

GAP_NO_CITATION = "NO_CITATION"
GAP_ABSTAINED_DESPITE_HIT = "ABSTAINED_DESPITE_HIT"
GAP_WRONG_FACT = "WRONG_FACT"
GAP_INFRA_ERROR = "INFRA_ERROR"
GAP_NONE = "OK"


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").casefold())
    return "".join(c for c in normalized if not unicodedata.combining(c))


def is_infra_error(case: dict) -> bool:
    """True when the case failed because the model backend itself was
    unreachable (harness `error` field, or answer()'s own "Ollama je
    nedostupná" fallback text) - not because a claim lacked a citation or a
    fact was wrong. Must be excluded from the headline gap rate, see module
    docstring."""
    if case.get("error"):
        return True
    body = _fold(str(case.get("answer_text") or ""))
    return any(marker in body for marker in _INFRA_ERROR_MARKERS)


def retrieval_hit(case: dict) -> bool:
    """RETRIEVAL layer succeeded: the expected document reached the pool
    (same signal `document_found_rate`/`top5_accuracy` already use)."""
    return bool(case.get("document_found"))


def has_substantive_claim(case: dict) -> bool:
    """Delegates to `acceptance_metrics.has_substantive_claim` (the existing
    grading source of truth) instead of re-implementing scaffold/filler
    detection here. `answer_text` on a case dict is already the extracted
    body (see `NdsCaseResult.answer_text` / `_evaluate_result`), so wrapping
    it back into `{"answer": ...}` round-trips through `answer_body()`
    as a no-op."""
    return acceptance_metrics.has_substantive_claim({"answer": case.get("answer_text")})


def classify_gap(case: dict) -> str:
    """One case -> one label. Infra errors are checked before citation/fact
    checks so a timeout is never miscounted as an evidence-extraction defect.
    Cases that never had a retrieval hit are out of scope for this metric -
    that failure belongs to the RETRIEVAL layer, not this one.

    Among `answer_correct=False` cases, abstention ("Nenalezeno v
    indexovaných dokumentech.") is split out from a wrong/incomplete stated
    fact: both are "retrieval hit, answer miss", but they point at different
    fixes (an evidence step that gave up vs. one that answered incorrectly)."""
    if not retrieval_hit(case):
        return GAP_NONE
    if is_infra_error(case):
        return GAP_INFRA_ERROR
    if case.get("unsupported_claim"):
        return GAP_NO_CITATION
    if case.get("answer_correct") is False:
        return GAP_WRONG_FACT if has_substantive_claim(case) else GAP_ABSTAINED_DESPITE_HIT
    return GAP_NONE


def is_retrieval_hit_answer_miss(case: dict) -> bool:
    """The PR8.4 headline signal: retrieval succeeded, the backend was
    reachable, and the answer still failed its contract - i.e. the failure
    lives strictly in evidence/citation extraction, not retrieval or infra."""
    return classify_gap(case) in (GAP_NO_CITATION, GAP_ABSTAINED_DESPITE_HIT, GAP_WRONG_FACT)


@dataclass
class RetrievalHitAnswerMissSummary:
    case_count: int = 0
    retrieval_hit_count: int = 0
    gap_count: int = 0
    gap_rate: float | None = None  # gap_count / retrieval_hit_count
    no_citation_case_ids: list[str] = field(default_factory=list)
    abstained_case_ids: list[str] = field(default_factory=list)
    wrong_fact_case_ids: list[str] = field(default_factory=list)
    infra_error_case_ids: list[str] = field(default_factory=list)
    gap_case_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_retrieval_hit_answer_miss(cases: list[dict]) -> RetrievalHitAnswerMissSummary:
    summary = RetrievalHitAnswerMissSummary(case_count=len(cases))
    hits = [c for c in cases if retrieval_hit(c)]
    summary.retrieval_hit_count = len(hits)
    for c in hits:
        label = classify_gap(c)
        if label == GAP_NO_CITATION:
            summary.no_citation_case_ids.append(c.get("id"))
        elif label == GAP_ABSTAINED_DESPITE_HIT:
            summary.abstained_case_ids.append(c.get("id"))
        elif label == GAP_WRONG_FACT:
            summary.wrong_fact_case_ids.append(c.get("id"))
        elif label == GAP_INFRA_ERROR:
            summary.infra_error_case_ids.append(c.get("id"))
    summary.gap_case_ids = (
        summary.no_citation_case_ids + summary.abstained_case_ids + summary.wrong_fact_case_ids
    )
    summary.gap_count = len(summary.gap_case_ids)
    summary.gap_rate = (
        summary.gap_count / summary.retrieval_hit_count if summary.retrieval_hit_count else None
    )
    return summary


def _load_cases_from_artifact(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError(f"{path}: expected a FAT/acceptance artifact with a 'cases' list")
    return cases


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        raise SystemExit(2)
    cases = _load_cases_from_artifact(Path(argv[0]))
    summary = compute_retrieval_hit_answer_miss(cases)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
