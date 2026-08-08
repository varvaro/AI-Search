"""Pure, dependency-free metric functions. No I/O, no ai_search/ui_services
imports - these operate on plain lists/dicts so they can be unit-tested with
synthetic data (see tests/test_benchmark_framework.py) independently of any
real or fixture index.

Every function below is documented with WHY it is part of this benchmark,
not just what it computes - see the module-level docstring in each function.
"""
from __future__ import annotations

import math
import unicodedata
from collections import Counter


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", (text or "").casefold()) if not unicodedata.combining(c))


def row_matches_any(row: dict, substrings: list[str]) -> bool:
    """Case- and diacritics-insensitive substring match against a result's
    `path` (falls back to `document`). This is the matching rule behind
    `expected_documents` / `forbidden_documents` in the dataset schema -
    substrings instead of exact chunk ids keep benchmark cases readable and
    resilient to re-chunking."""
    if not substrings:
        return False
    haystack = _fold(row.get("path") or row.get("document") or "")
    return any(_fold(needle) in haystack for needle in substrings)


def keyword_coverage(text: str, keywords: list[str]) -> float:
    """Fraction of `keywords` present (case/diacritics-insensitive substring)
    in `text`. Used to check whether the *content* a correct answer would
    need (not just the right filename) actually reached the top-k quotes -
    a document can be "found" by path yet still miss the specific fact a
    checklist question needs from a different part of the same file."""
    if not keywords:
        return 1.0
    folded = _fold(text)
    hits = sum(1 for kw in keywords if _fold(kw) in folded)
    return hits / len(keywords)


def relevant_ranks(rows: list[dict], expected_substrings: list[str]) -> list[int]:
    """0-indexed ranks (positions) of every row matching `expected_substrings`."""
    return [i for i, row in enumerate(rows) if row_matches_any(row, expected_substrings)]


def best_rank(rows: list[dict], expected_substrings: list[str]) -> int | None:
    """Rank (0-indexed) of the FIRST relevant row, or None if not present at
    all in `rows`. This is the single most diagnostic number when a
    regression happens: "found, but at rank 189" (recoverable by widening a
    pool) is a completely different problem from "never found" (points at
    indexing/chunking/embedding, not pool size) - exactly the distinction
    that mattered in the 2026-08-06 FERI diagnostic (BM25 rank 189, present;
    vector search: absent)."""
    ranks = relevant_ranks(rows, expected_substrings)
    return ranks[0] if ranks else None


def recall_at_k(rows: list[dict], expected_substrings: list[str], k: int) -> float:
    """Fraction of *distinct expected documents* that appear anywhere in the
    top-k. This is the foundational retrieval metric: everything downstream
    (reranking, diversification, the LLM answer) is capped by what recall@k
    lets through - a perfect reranker cannot promote a document recall
    already excluded. Reported per-stage (after FTS, after vector, after
    fusion, after pool truncation, after rerank) so a regression can be
    pinpointed to the exact stage that lost the document, instead of only
    knowing the final list is wrong."""
    if not expected_substrings:
        return 1.0
    top = rows[:k]
    found = {needle for needle in expected_substrings if any(row_matches_any(row, [needle]) for row in top)}
    return len(found) / len(expected_substrings)


def forbidden_free_rate(rows: list[dict], expected_substrings: list[str], forbidden_substrings: list[str], k: int) -> float:
    """1.0 unless the top-k contains an explicitly-forbidden near-duplicate
    spam pattern; degrades toward 0.0 as more of the top-k is forbidden
    content. `expected_substrings` is accepted for signature symmetry with
    the other per-case metrics but is NOT used in the computation below.

    NAMING HISTORY (both renames, 2026-08-07, formula UNCHANGED by either):
    `precision_at_k` -> `forbidden_rate` -> `forbidden_free_rate`. The second
    rename fixed an inverted name: 1.0 means "no forbidden content", so a
    metric called `forbidden_rate` read as the exact opposite of what it
    reports. Run artifacts written before this carry the old key; compare.py
    skips a metric present on only one side, so an old-vs-new comparison
    silently drops it rather than crashing - re-run the baseline to get it
    back.

    Why the first rename: this used to be named `precision_at_k` and its
    result was surfaced as `precision_at_k_final`,
    which is NOT real precision - true precision (relevant / retrieved) needs
    a judgment of relevance for every row, and this only ever checks for a
    fixed set of *forbidden* patterns supplied per-case (usually empty, in
    which case it trivially returns 1.0 - see the "no forbidden_substrings"
    case above). With most cases never setting `forbidden_documents`, the old
    name reported what was effectively a near-constant 1.0 as if it were a
    precision score, which is misleading. The formula/behaviour is UNCHANGED
    by this rename - only the name and the false "this is precision" framing
    were removed. Do not reintroduce a metric named precision_at_k_final
    without it actually being computed from a real relevance judgment over
    every retrieved row.

    Recall alone cannot detect the class of bug fixed by the content-dedup/
    diversify pass (e.g. 8 of 10 slots filled by the same "základovou desku."
    fragment repeated across unrelated "kontrolní den" reports) - a query can
    have perfect recall@10 while still being useless to the user because the
    other 9 slots are redundant. `forbidden_substrings` lets a case
    explicitly flag "this exact duplicate-spam pattern must not dominate",
    complementing the generic relevance/irrelevance check."""
    top = rows[:k]
    if not top:
        return 1.0
    bad = sum(1 for row in top if forbidden_substrings and row_matches_any(row, forbidden_substrings))
    return 1.0 - (bad / len(top))


def hit_rate(rows: list[dict], expected_substrings: list[str], k: int) -> float:
    """Binary "was at least one expected document found in the top-k" - the
    simplest possible top-line KPI (0 or 1 per query, easy to average into a
    single "N% of benchmark queries returned something usable" number for a
    dashboard/PR summary), complementary to the more granular recall@k."""
    if not expected_substrings:
        return 1.0
    return 1.0 if best_rank(rows, expected_substrings) is not None and best_rank(rows, expected_substrings) < k else 0.0


def reciprocal_rank(rows: list[dict], expected_substrings: list[str]) -> float:
    """1/(rank+1) of the first relevant hit, or 0 if absent. Rewards getting
    a relevant result to position 1 far more than position 10 - a good
    single-query proxy for "did the user's first glance already work"."""
    rank = best_rank(rows, expected_substrings)
    return 1.0 / (rank + 1) if rank is not None else 0.0


def mean_reciprocal_rank(values: list[float]) -> float:
    """Aggregate MRR across queries. The standard headline retrieval metric
    in IR literature/enterprise search dashboards precisely because it is a
    single number that already captures "how far down do I usually have to
    look" - good for tracking overall retrieval health release over release."""
    return sum(values) / len(values) if values else 0.0


def dcg_at_k(rows: list[dict], relevance_of: dict, k: int) -> float:
    """Discounted Cumulative Gain: sum of relevance-grade / log2(rank+2). Unlike
    recall/hit-rate (binary), nDCG (below) rewards *graded* relevance and
    heavily discounts hits found deep in the list - the right tool when a
    query has several expected documents of different importance (e.g. the
    primary handover protocol should outrank a merely-related technical
    sheet, and both should outrank finding nothing).

    Each expected item (`relevance_of` key, e.g. a document/folder pattern)
    earns credit at most ONCE, on its first-matching row - `expected_documents`
    is a document-level judgment, and several chunks of the same relevant
    document appearing in the top-k (a *good* outcome, not a bonus) must not
    let the score exceed the ideal-ranking DCG computed below, which only
    ever has one row per expected item."""
    top = rows[:k]
    credited: set[str] = set()
    score = 0.0
    for i, row in enumerate(top):
        grade = 0.0
        for needle, weight in relevance_of.items():
            if needle in credited:
                continue
            if row_matches_any(row, [needle]):
                grade = max(grade, weight)
                credited.add(needle)
        if grade:
            score += grade / math.log2(i + 2)
    return score


def ndcg_at_k(rows: list[dict], expected_substrings: list[str], k: int, relevance_of: dict | None = None) -> float:
    """nDCG@k = DCG@k / ideal-DCG@k, normalized to [0, 1]. Defaults to binary
    relevance (every expected document worth 1.0) when `relevance_of` isn't
    given, which is the common case for this dataset (most cases only assert
    "this folder should be represented somewhere in the top-k", not a strict
    importance ranking between several expected documents)."""
    if not expected_substrings:
        return 1.0
    grades = relevance_of or {needle: 1.0 for needle in expected_substrings}
    dcg = dcg_at_k(rows, grades, k)
    ideal_rows = [{"path": needle} for needle in sorted(grades, key=grades.get, reverse=True)]
    ideal = dcg_at_k(ideal_rows, grades, k)
    return dcg / ideal if ideal else 0.0


def channel_agreement(bm25_rows: list[dict], vector_rows: list[dict], expected_substrings: list[str]) -> dict:
    """For every expected document actually found by either channel, buckets
    it into both / bm25_only / vector_only. Directly diagnoses the exact bug
    class found on 2026-08-06: a document with a BM25-only hit at a
    middling rank gets an artificially low RRF score purely because it has no
    second channel corroborating it, regardless of how relevant it is - this
    metric turns that failure mode into a trackable number instead of
    something only visible via a one-off manual diagnostic."""
    bm25_hit = {needle for needle in expected_substrings if any(row_matches_any(row, [needle]) for row in bm25_rows)}
    vector_hit = {needle for needle in expected_substrings if any(row_matches_any(row, [needle]) for row in vector_rows)}
    both = bm25_hit & vector_hit
    bm25_only = bm25_hit - vector_hit
    vector_only = vector_hit - bm25_hit
    neither = set(expected_substrings) - bm25_hit - vector_hit
    return {"both": len(both), "bm25_only": len(bm25_only), "vector_only": len(vector_only), "neither": len(neither)}


def score_histogram(scores: list[float], bins: int = 10) -> list[dict]:
    """Bucket a list of scores into `bins` equal-width buckets. Not a
    pass/fail metric - a diagnostic aid. A healthy rerank stage usually shows
    a spread-out histogram; a "score collapse" (nearly all candidates
    clustered in one narrow band) means the ranking signal is close to random
    noise even if a headline metric like recall@10 still looks fine, which is
    exactly the kind of latent problem a single scalar metric would hide."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [{"range": (lo, hi), "count": len(scores)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for score in scores:
        index = min(bins - 1, int((score - lo) / width))
        counts[index] += 1
    return [{"range": (lo + i * width, lo + (i + 1) * width), "count": counts[i]} for i in range(bins)]


def row_matches_chunk_target(row: dict, target: dict) -> bool:
    """Chunk-level counterpart to `row_matches_any` (2026-08-06 ground-truth
    repair). `target` is one entry from `chunk_resolution.resolve_expected_chunks()`
    - {"chunk_ids": set[str], "document": str, "text_anchor": str, ...}.

    Every pipeline-stage candidate row (fts/vector/fusion/union/reranked/
    cross_encoder) carries `chunk_id` straight from ai_search.SearchTrace, so
    those are matched by exact id - the same identity the retrieval pipeline
    itself uses, resolved fresh against the current index (see
    chunk_resolution.py's docstring for why that is NOT the same as hardcoding
    a chunk_id string in the dataset file). Terminal rows (final_results/
    search_output) don't carry chunk_id, only `document`/`path`/`quote` - for
    those, a document/path substring match is deliberately NOT sufficient by
    itself (that is exactly the "huge folder trivially satisfies the case"
    bug this repair fixes); a text_anchor substring must ALSO be found in the
    row's own quoted text."""
    chunk_ids = target.get("chunk_ids") or set()
    row_chunk_id = row.get("chunk_id")
    if row_chunk_id:
        return row_chunk_id in chunk_ids
    text_anchor = target.get("text_anchor") or ""
    document = target.get("document") or ""
    if not text_anchor or not document:
        return False
    if not row_matches_any(row, [document]):
        return False
    text = row.get("quote") or row.get("text") or ""
    return _fold(text_anchor) in _fold(text)


def chunk_relevant_ranks(rows: list[dict], targets: list[dict]) -> list[int]:
    """0-indexed ranks of every row matching ANY of `targets`."""
    return [i for i, row in enumerate(rows) if any(row_matches_chunk_target(row, t) for t in targets)]


def chunk_best_rank(rows: list[dict], targets: list[dict]) -> int | None:
    ranks = chunk_relevant_ranks(rows, targets)
    return ranks[0] if ranks else None


def chunk_recall_at_k(rows: list[dict], targets: list[dict], k: int) -> float:
    """Fraction of *distinct expected chunk targets* (not distinct documents)
    found anywhere in the top-k - the chunk-mode counterpart to `recall_at_k`.
    Each target is one physical chunk; a document with 50 unrelated chunks in
    the top-k earns no credit unless one of THOSE 50 is an actual target."""
    if not targets:
        return 1.0
    top = rows[:k]
    found = sum(1 for t in targets if any(row_matches_chunk_target(row, t) for row in top))
    return found / len(targets)


def chunk_hit_rate(rows: list[dict], targets: list[dict], k: int) -> float:
    if not targets:
        return 1.0
    rank = chunk_best_rank(rows, targets)
    return 1.0 if rank is not None and rank < k else 0.0


def chunk_reciprocal_rank(rows: list[dict], targets: list[dict]) -> float:
    rank = chunk_best_rank(rows, targets)
    return 1.0 / (rank + 1) if rank is not None else 0.0


def chunk_dcg_at_k(rows: list[dict], targets: list[dict], k: int) -> float:
    top = rows[:k]
    credited: set[int] = set()
    score = 0.0
    for i, row in enumerate(top):
        grade = 0.0
        for ti, target in enumerate(targets):
            if ti in credited:
                continue
            if row_matches_chunk_target(row, target):
                grade = max(grade, target.get("relevance", 1))
                credited.add(ti)
        if grade:
            score += grade / math.log2(i + 2)
    return score


def chunk_ndcg_at_k(rows: list[dict], targets: list[dict], k: int) -> float:
    """Graded nDCG@k over chunk targets (relevance 0-3 per target, see
    ExpectedChunk.relevance), normalized against the best possible ordering
    of the SAME targets - independent of how many candidates exist overall."""
    if not targets:
        return 1.0
    dcg = chunk_dcg_at_k(rows, targets, k)
    grades = sorted((t.get("relevance", 1) for t in targets), reverse=True)
    ideal = sum(grade / math.log2(i + 2) for i, grade in enumerate(grades[:k]))
    return dcg / ideal if ideal else 0.0


def duplicate_count(rows: list[dict], key=lambda row: row.get("document")) -> int:
    """Number of rows sharing a key (default: document name) with at least
    one earlier row - i.e. how many results in this list are redundant
    repeats rather than distinct evidence. Surfaces regressions in the
    dedup/diversify stage independently of relevance (a dedup bug can look
    "fine" on recall/precision if the duplicated document happens to be
    correct, while still wasting half the LLM's context window)."""
    seen = Counter()
    duplicates = 0
    for row in rows:
        key_value = key(row)
        if seen[key_value] > 0:
            duplicates += 1
        seen[key_value] += 1
    return duplicates


def distinct_ratio(rows: list[dict], key=lambda row: row.get("document")) -> float:
    """distinct(key)/len(rows) - 1.0 means every result is a different
    document, near 0 means the list is dominated by repeats of one source.
    Tracks the "avoid 10 copies of the same fact" goal from the earlier
    dedup/diversify fix over time, independent of any specific query."""
    if not rows:
        return 1.0
    return len({key(row) for row in rows}) / len(rows)
