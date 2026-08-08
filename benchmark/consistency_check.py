"""Guard against pipeline_trace.py silently drifting out of sync with
`ai_search.search()` in the future.

Before the SearchTrace instrumentation (see ai_search.py), pipeline_trace.py
hand-reimplemented FTS/vector/RRF/rerank to expose per-phase numbers, and
this module's job was to catch that *second implementation* silently
disagreeing with the real one - which required a second, independent call to
`ai_search.search()` and comparing its output against the mirror's.

Now `ai_search.search()` populates the trace itself from inside its one real
code path, so there is no second implementation left to drift against - a
trace's `final_candidates` are written from the exact same `output` list that
produces the function's return value (see SearchTrace's docstring in
ai_search.py). What *can* still drift is search() itself: a future change to
its scoring/truncation logic that isn't matched by an update to the nearby
trace-writing lines. `check_consistency()` now catches exactly that, as a
same-call self-consistency assertion - no second call to `ai_search.search()`
is made, so there is no risk of the check itself flagging harmless
call-to-call nondeterminism as drift.
"""
from __future__ import annotations

from .pipeline_trace import PipelineTrace

SCORE_TOLERANCE = 1e-6


class PipelineDriftError(AssertionError):
    """Raised when a SearchTrace's final_candidates no longer match what the
    very same ai_search.search() call returned. Indicates search()'s
    trace-writing lines (near its `return result[:limit]` statement) were not
    updated to match a change in its scoring/truncation logic - the
    benchmark's phase-level numbers cannot be trusted until they are."""


def check_consistency(query: str, trace: PipelineTrace) -> None:
    search_trace = trace.search_trace
    if search_trace is None:
        raise PipelineDriftError(
            f"trace.search_trace is missing for query {query!r} - run_pipeline_trace() must pass "
            "trace=ai_search.SearchTrace() into ai_search.search() for this check to be meaningful."
        )

    real_output = trace.search_output
    traced_final = search_trace.final_candidates

    if len(traced_final) != len(real_output):
        raise PipelineDriftError(
            f"SearchTrace.final_candidates length ({len(traced_final)}) != ai_search.search()'s "
            f"actual return length ({len(real_output)}) for query {query!r}. search() was likely "
            "changed without updating the matching trace-writing lines near its final return statement."
        )

    for rank, (traced, real) in enumerate(zip(traced_final, real_output)):
        if traced.get("document") != real.get("document") or traced.get("path") != real.get("path"):
            raise PipelineDriftError(
                f"SearchTrace/search() identity mismatch at rank {rank} for query {query!r}: "
                f"traced={traced.get('document')!r} real={real.get('document')!r}"
            )
        if abs((traced.get("score") or 0.0) - (real.get("score") or 0.0)) > SCORE_TOLERANCE:
            raise PipelineDriftError(
                f"SearchTrace/search() score mismatch at rank {rank} for document {real.get('document')!r}: "
                f"traced={traced.get('score')!r} real={real.get('score')!r}"
            )
