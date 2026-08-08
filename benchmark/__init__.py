"""Retrieval pipeline benchmark framework for AI Search.

This package is intentionally independent of ai_search.py / ui_services.py
internals: it only ever *calls* their existing public functions (search,
classify_query, deduplicate_by_content, diversify_results, search_all,
answer) and never modifies retrieval behaviour, scoring, limits, BM25, RRF,
the reranker, embeddings or chunking.

Phase-by-phase (BM25-only / vector-only / fusion / pool-truncation / rerank)
numbers come from `ai_search.search()`'s own optional `trace=SearchTrace()`
instrumentation, not from a separate reimplementation of retrieval logic -
`pipeline_trace.py` calls `ai_search.search()` once and re-shapes the trace it
records into this package's own dataclasses. `consistency_check.py` still
guards against drift, but now as a same-call self-consistency assertion
(the trace's recorded final candidates vs. what that call actually returned)
rather than a second, independent call to search() compared against a
hand-written mirror.

See benchmark/README.md for usage.
"""
