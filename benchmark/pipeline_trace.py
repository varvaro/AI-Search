"""Instrumented re-run of the retrieval pipeline that exposes every stage's
intermediate state, for benchmarking - not a reimplementation of retrieval
logic.

Design principle: call the REAL functions wherever they already return
something usable (`ui_services.classify_query`, `ui_services.deduplicate_by_content`,
`ui_services.diversify_results`, `ui_services.search_all`, `ai_search.answer`)
- zero drift risk, because it *is* the production code path.

The BM25-only / vector-only / fusion / candidate-pool / rerank breakdown used
to be the one exception: `ai_search.search()` only returned the *final*
reranked+truncated list, so this module used to hand-reimplement those phases
(FTS query, LanceDB query, RRF math, cosine rerank, chunk-quality/filename
scoring) just to expose the intermediate numbers - a second implementation of
retrieval logic living only to be measured, with its own drift risk against
the real thing.

That mirror is gone. `ai_search.search()` now accepts an optional
`trace=ai_search.SearchTrace()` argument and records those same per-phase
snapshots itself, from inside the one real code path, while computing and
returning the exact same result as `trace=None` would (see `SearchTrace`'s
docstring in ai_search.py). This module now only calls `ai_search.search(...,
trace=search_trace)` once and re-shapes `search_trace`'s chunk_id-keyed
snapshots into `PipelineTrace`'s document/path-keyed ones that
`metrics.py`/`runner.py` already expect (`row_matches_any()` matches against
`document`/`path`, which `SearchTrace` doesn't carry - only `chunks_fts`/
LanceDB's `chunk_id` - so a single batched DB lookup per run still resolves
chunk_id -> document/path/heading/text; that lookup is metadata enrichment,
not retrieval/ranking, and was needed even before this refactor).

`consistency_check.py` now checks something narrower and cheaper than before:
that `SearchTrace.final_candidates` (written inside `ai_search.search()`)
still agrees with what that *same* call returned - a self-consistency
assertion on one call, not a second independent call to `ai_search.search()`
compared against a hand-written mirror. It guards against a future
maintainer changing search()'s core scoring/truncation without updating the
matching trace-writing lines right next to it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import ai_search
import ui_services

from .environment import Environment


@dataclass
class StageTiming:
    name: str
    latency_ms: float
    pool_size: int


@dataclass
class PipelineTrace:
    query: str
    is_question: bool
    deep: bool
    fts_terms: str
    retrieval_k: int
    rerank_k: int
    fetch_limit: int

    candidate_strategy: str = "legacy"

    # Query Understanding layer (ai_search.search(expand_query=...), see
    # query_expansion.py). expand_query=False leaves expansion_terms empty and
    # query_expanded == query, which is what a baseline run records.
    expand_query: bool | str = False
    query_expanded: str = ""
    expansion_terms: list[str] = field(default_factory=list)
    expansion_matched_rules: list[dict] = field(default_factory=list)

    fts_candidates: list[dict] = field(default_factory=list)
    vector_candidates: list[dict] = field(default_factory=list)
    fusion_candidates: list[dict] = field(default_factory=list)
    # ai_search.search()'s _build_candidate_union() output (always populated
    # regardless of candidate_strategy), reshaped with document/path. Every
    # chunk_id either channel found at all, deduplicated, with NO score-based
    # cutoff - the "everything either channel found" reference point.
    union_candidates: list[dict] = field(default_factory=list)
    # The chunk_ids phase 3 (cosine rerank) actually iterated over for THIS
    # call - i.e. what candidate_strategy resolved `top_ids` to. Under
    # "legacy" this equals fusion_candidates truncated to rerank_k (same as
    # pool_after_truncation_ids below); under "union" it equals union_candidates.
    candidates_before_precision: list[dict] = field(default_factory=list)
    pool_after_truncation_ids: list[str] = field(default_factory=list)
    reranked: list[dict] = field(default_factory=list)

    # candidate_strategy="union_ce" only - see ai_search.CrossEncoderReranker.
    candidates_before_cross_encoder: list[dict] = field(default_factory=list)
    cross_encoder_candidates: list[dict] = field(default_factory=list)
    cross_encoder_latency_ms: float | None = None
    cross_encoder_model: str | None = None

    search_output: list[dict] = field(default_factory=list)
    deduplicated: list[dict] = field(default_factory=list)
    diversified: list[dict] = field(default_factory=list)
    final_results: list[dict] = field(default_factory=list)

    context: str | None = None
    answer: dict | None = None

    stages: dict[str, StageTiming] = field(default_factory=dict)

    # Raw ai_search.SearchTrace for this run, kept only so consistency_check.py
    # can compare its final_candidates against search_output without a second
    # call to ai_search.search(). Not consumed by metrics.py/runner.py.
    search_trace: object = None

    def stage_order(self) -> list[str]:
        return [
            "intent_detection",
            "query_parsing",
            "fts_retrieval",
            "vector_retrieval",
            "fusion_rrf",
            "candidate_pool",
            "reranker",
            "cross_encoder",
            "diversification",
            "prompt_builder",
            "final_answer",
        ]


def _fetch_chunk_rows(db_path: Path, chunk_ids: list[str]) -> dict[str, dict]:
    """Batched chunk_id -> {document,path,project,heading,text} lookup, used
    purely to give the phase-level candidate lists (which only carry
    chunk_id/rank/score coming out of ai_search.search()'s trace) the
    document/path fields metrics.py matches on. Metadata enrichment, not
    retrieval - the ranking/order itself already comes from the trace."""
    rows: dict[str, dict] = {}
    if not chunk_ids:
        return rows
    with ai_search.database(db_path) as con:
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            for cid, name, path, project, heading, text in con.execute(
                f"SELECT c.id,d.name,d.path,d.project,c.heading,c.text FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id IN ({placeholders})",
                batch,
            ).fetchall():
                rows[cid] = {"document": name, "path": path, "project": project, "heading": heading, "text": text}
    return rows


def run_pipeline_trace(
    query: str,
    environment: Environment,
    *,
    is_question: bool | None = None,
    result_count: int = 10,
    include_answer: bool = False,
    candidate_strategy: str = "legacy",
    cross_encoder: object = None,
    expand_query: bool | str = False,
) -> PipelineTrace:
    """Runs `query` through every retrieval stage against `environment`,
    recording per-stage candidates/timings. Purely read-only against the
    index (never calls sync())."""
    timings: dict[str, StageTiming] = {}

    # --- intent detection (real ui_services.classify_query) ---
    t0 = time.perf_counter()
    classification = ui_services.classify_query(query)
    resolved_is_question = classification["mode"] == "otazka" if is_question is None else is_question
    timings["intent_detection"] = StageTiming("intent_detection", (time.perf_counter() - t0) * 1000, 1)

    # `fetch_limit` mirrors ui_services.search_all()'s own `fetch_limit` - the
    # *actual* `limit` argument production code passes into ai_search.search(),
    # not `result_count` (the final, post-dedup/diversify UI-facing count).
    candidate_pool = ui_services.QA_CANDIDATE_POOL if resolved_is_question else result_count
    fetch_limit = max(50, candidate_pool * 4, ai_search.QA_RERANK_POOL_SIZE) if resolved_is_question else max(50, candidate_pool * 4)

    # --- the one real call: ai_search.search() does FTS + vector + RRF +
    # rerank itself and records a snapshot of each phase into search_trace,
    # while returning the exact same rows it always would. ---
    search_trace = ai_search.SearchTrace()
    search_output = ai_search.search(
        query, environment.db_path, environment.lance_dir, environment.embeddings,
        limit=fetch_limit, is_question=resolved_is_question, trace=search_trace,
        candidate_strategy=candidate_strategy, cross_encoder=cross_encoder,
        expand_query=expand_query,
    )

    timings["query_parsing"] = StageTiming(
        "query_parsing", search_trace.timings.get("query_parsing", 0.0),
        len(search_trace.query_terms.split(" OR ")) if search_trace.query_terms else 0,
    )
    timings["fts_retrieval"] = StageTiming("fts_retrieval", search_trace.timings.get("fts_retrieval", 0.0), len(search_trace.bm25_candidates))
    timings["vector_retrieval"] = StageTiming("vector_retrieval", search_trace.timings.get("vector_retrieval", 0.0), len(search_trace.vector_candidates))
    timings["fusion_rrf"] = StageTiming("fusion_rrf", search_trace.timings.get("fusion_rrf", 0.0), len(search_trace.rrf_candidates))
    pool_after_size = search_trace.metadata.get("candidate_pool_size_after_truncation", len(search_trace.rrf_candidates))
    timings["candidate_pool"] = StageTiming("candidate_pool", 0.0, pool_after_size)
    timings["reranker"] = StageTiming("reranker", search_trace.timings.get("reranker", 0.0), len(search_trace.rerank_candidates))
    if search_trace.cross_encoder_latency is not None:
        timings["cross_encoder"] = StageTiming("cross_encoder", search_trace.cross_encoder_latency, len(search_trace.cross_encoder_candidates))

    chunk_ids_needing_meta = list(dict.fromkeys(
        [c["chunk_id"] for c in search_trace.bm25_candidates] + [c["chunk_id"] for c in search_trace.vector_candidates]
    ))
    meta = _fetch_chunk_rows(environment.db_path, chunk_ids_needing_meta)

    trace = PipelineTrace(
        query=query,
        is_question=resolved_is_question,
        deep=classification.get("deep", False),
        fts_terms=search_trace.query_terms or "",
        retrieval_k=search_trace.intent.get("retrieval_k", 0),
        rerank_k=search_trace.intent.get("rerank_k", 0),
        fetch_limit=fetch_limit,
        candidate_strategy=search_trace.metadata.get("candidate_strategy", candidate_strategy),
        expand_query=expand_query,
        query_expanded=search_trace.query_expanded or query,
        expansion_terms=list(search_trace.expansion_terms),
        expansion_matched_rules=list(search_trace.expansion_matched_rules),
        stages=timings,
        search_trace=search_trace,
    )

    trace.fts_candidates = [
        {"rank": c["rank"], "chunk_id": c["chunk_id"], "bm25_score": c["score"],
         **{k: meta.get(c["chunk_id"], {}).get(k) for k in ("document", "path")}}
        for c in search_trace.bm25_candidates
    ]
    trace.vector_candidates = [
        {"rank": c["rank"], "chunk_id": c["chunk_id"],
         **{k: meta.get(c["chunk_id"], {}).get(k) for k in ("document", "path")}}
        for c in search_trace.vector_candidates
    ]
    trace.fusion_candidates = [
        {"rank": c["rank"], "chunk_id": c["chunk_id"], "bm25_rank": c["bm25_rank"], "vector_rank": c["vector_rank"], "rrf_score": c["score"],
         **{k: meta.get(c["chunk_id"], {}).get(k) for k in ("document", "path")}}
        for c in search_trace.rrf_candidates
    ]
    trace.pool_after_truncation_ids = [c["chunk_id"] for c in search_trace.rrf_candidates[:trace.rerank_k]]
    trace.union_candidates = [
        {"chunk_id": c["chunk_id"], "fts_rank": c["fts_rank"], "vector_rank": c["vector_rank"], "fts_hit": c["fts_hit"], "vector_hit": c["vector_hit"],
         **{k: meta.get(c["chunk_id"], {}).get(k) for k in ("document", "path")}}
        for c in search_trace.union_candidates
    ]
    trace.candidates_before_precision = [
        {"chunk_id": c["chunk_id"], **{k: meta.get(c["chunk_id"], {}).get(k) for k in ("document", "path")}}
        for c in search_trace.candidates_before_precision
    ]
    trace.reranked = [
        {"chunk_id": c["chunk_id"], "document": c["document"], "path": c["path"], "score": c["score"]}
        for c in search_trace.rerank_candidates
    ]
    trace.candidates_before_cross_encoder = [
        {"chunk_id": c["chunk_id"], **{k: meta.get(c["chunk_id"], {}).get(k) for k in ("document", "path")}}
        for c in search_trace.candidates_before_cross_encoder
    ]
    trace.cross_encoder_candidates = [
        {"chunk_id": c["chunk_id"], "rank": c["rank"], "score": c["score"], "document": c["document"], "path": c["path"]}
        for c in search_trace.cross_encoder_candidates
    ]
    trace.cross_encoder_latency_ms = search_trace.cross_encoder_latency
    trace.cross_encoder_model = search_trace.cross_encoder_model
    trace.search_output = search_output

    return _finish_trace(trace, environment, search_output, result_count, include_answer)


def _finish_trace(trace: PipelineTrace, environment: Environment, search_output: list[dict], result_count: int, include_answer: bool) -> PipelineTrace:
    # --- diversification stage: real ui_services functions, called directly ---
    t0 = time.perf_counter()
    deduplicated = ui_services.deduplicate_by_content(search_output)
    diversified = ui_services.diversify_results(deduplicated)
    trace.deduplicated = deduplicated
    trace.diversified = diversified
    trace.stages["diversification"] = StageTiming("diversification", (time.perf_counter() - t0) * 1000, len(diversified))

    # --- final_results: the real end-to-end ui_services.search_all(), used as
    # the ground truth for "what does the user actually see" - independent of
    # the single-source path above, so any divergence between the two is
    # itself a useful signal. ---
    final_results: list[dict] = []
    if environment.settings is not None and environment.state_dir is not None:
        # expand_query MUST be forwarded here too: recall@k/MRR/nDCG/precision are
        # all computed from final_results, so a baseline search_all() under an
        # expanded run would silently compare expanded phase lists against
        # unexpanded final results and make the whole A/B meaningless.
        final_results = ui_services.search_all(trace.query, environment.settings, environment.state_dir, environment.embeddings, is_question=trace.is_question, expand_query=trace.expand_query)
    else:
        final_results = diversified[: environment.settings.result_count if environment.settings else result_count]
    trace.final_results = final_results

    # --- prompt builder: mirrors the context-assembly one-liner in
    # ai_search.py answer() (as of this writing, the line building `context`
    # right before the Ollama call) - included here only to measure how much
    # context/how many distinct sources reach the LLM, not to change it. ---
    t0 = time.perf_counter()
    if final_results:
        trace.context = "\n\n".join(
            f"[{i}] {r['document']}" + (f" (sekce: {r['heading']})" if r.get("heading") else "") + f" | projekt {r.get('project')}\n{r['quote']}"
            for i, r in enumerate(final_results, 1)
        )
    trace.stages["prompt_builder"] = StageTiming("prompt_builder", (time.perf_counter() - t0) * 1000, len(final_results))

    # --- final answer: real ai_search.answer(), optional because it needs a
    # live Ollama server and is slow/non-deterministic. ---
    if include_answer and final_results:
        t0 = time.perf_counter()
        trace.answer = ai_search.answer(trace.query, final_results)
        trace.stages["final_answer"] = StageTiming("final_answer", (time.perf_counter() - t0) * 1000, 1)

    return trace
