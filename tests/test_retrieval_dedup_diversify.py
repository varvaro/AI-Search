"""Regression tests for the retrieval-pipeline fix (content deduplication, result
diversification, chunk-quality penalty, wider QA candidate pool, safe FTS prefix
widening). Directly targets the production bug found in the "Co chybí k předání
základové desky investorovi?" audit: ~80 near-identical short "Kontrolní den"
status lines ("základovou desku.") crowded out every genuinely relevant document,
so all 10 documents passed to the LLM were duplicates of the same sentence.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import pytest
import ai_search
import ui_services as ui

DIMENSIONS = 16
STOPWORDS = {"a", "na", "v", "o", "k", "s", "z", "do", "po", "pro", "je", "se", "ze", "za", "ve"}


def _hashed_vector(seed: str) -> list[float]:
    digest = hashlib.sha256(seed.encode()).digest()
    return [(b / 127.5 - 1.0) for b in digest[:DIMENSIONS]]


class BagOfWordsEmbeddings:
    """Minimal bag-of-hashed-stems fake embedding: similarity is driven purely by
    shared vocabulary, with no domain knowledge or special-cased categories - a
    neutral stand-in adequate to prove the retrieval PIPELINE (dedup + diversify)
    behaves correctly. Not a claim about real-world semantic quality - see
    test_search_relevance.py's module docstring for that caveat."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            folded = text.casefold()
            tokens = [t[:6] for t in re.findall(r"\w+", folded) if len(t) > 2 and t not in STOPWORDS]
            vector = [0.0] * DIMENSIONS
            for token in tokens:
                token_vector = _hashed_vector(token)
                vector = [a + b for a, b in zip(vector, token_vector)]
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------

def test_normalize_chunk_text_ignores_case_punctuation_and_spacing():
    assert ui._normalize_chunk_text("Základovou   desku.") == ui._normalize_chunk_text("základovou desku")


def test_deduplicate_by_content_keeps_only_best_scored_near_duplicate():
    rows = [
        {"path": "/a/1.pdf", "document": "1.pdf", "quote": "Základovou desku.", "score": 0.05},
        {"path": "/a/2.pdf", "document": "2.pdf", "quote": "základovou desku", "score": 0.04},
        {"path": "/a/3.pdf", "document": "3.pdf", "quote": "Zcela odlišný obsah o výztuži a kontrole armatury.", "score": 0.03},
    ]
    result = ui.deduplicate_by_content(rows)
    assert [row["path"] for row in result] == ["/a/1.pdf", "/a/3.pdf"]


def test_deduplicate_by_content_keeps_distinct_short_texts():
    rows = [
        {"path": "/a/1.pdf", "document": "1.pdf", "quote": "Pentaflex technický list.", "score": 0.05},
        {"path": "/a/2.pdf", "document": "2.pdf", "quote": "Stavební deník zápis dne 5.", "score": 0.04},
    ]
    assert len(ui.deduplicate_by_content(rows)) == 2


def test_diversify_results_caps_results_per_folder():
    rows = [{"path": f"/kd/{i}.pdf", "document": f"{i}.pdf", "quote": "x", "score": 1 - i / 10} for i in range(5)]
    diversified = ui.diversify_results(rows, max_per_folder=2, max_per_document=3)
    assert len(diversified) == len(rows)  # over-cap rows are deferred, not dropped
    assert sum(1 for row in diversified[:2] if ui._folder_key(row) == "/kd") == 2


def test_diversify_results_prefers_handover_documentation_over_extra_duplicate_folder_slot():
    rows = [
        {"path": f"/kd/{i}.pdf", "document": f"kontrolni_den_{i}.pdf", "quote": "x", "score": 0.9 - i / 100}
        for i in range(3)
    ] + [{"path": "/other/predavaci.pdf", "document": "predavaci_protokol.pdf", "quote": "y", "score": 0.5}]
    diversified = ui.diversify_results(rows, max_per_folder=2, max_per_document=3)
    admitted = {row["document"] for row in diversified[:3]}
    assert "predavaci_protokol.pdf" in admitted


def test_chunk_quality_factor_penalizes_short_chunks_but_exempts_exact_matches():
    assert ai_search._chunk_quality_factor(10, exempt=False) == ai_search.SHORT_CHUNK_PENALTY
    assert ai_search._chunk_quality_factor(100, exempt=False) == ai_search.MEDIUM_CHUNK_PENALTY
    assert ai_search._chunk_quality_factor(500, exempt=False) == 1.0
    assert ai_search._chunk_quality_factor(10, exempt=True) == 1.0


def test_fts_query_terms_adds_safe_prefix_for_long_words():
    terms = ai_search._fts_query_terms("základové desky")
    assert '"základové"' in terms and '"desky"' in terms
    assert "základo*" in terms  # FTS_PREFIX_STRIP=2 chars trimmed from "základové"


def test_fts_query_terms_skips_prefix_for_short_words():
    assert "*" not in ai_search._fts_query_terms("co pes")


# ---------------------------------------------------------------------------
# End-to-end regression: the exact production scenario
# ---------------------------------------------------------------------------

KONTROLNI_DEN_TEXT = "Práce jsou dočasně pozastaveny do nástupu na základovou desku."

RELEVANT_DOCUMENTS = {
    "predavaci_protokol_zakladove_desky.txt": (
        "Předávací protokol základové desky dokumentuje předání investorovi včetně "
        "přílohy knihy betonů, protokolů zkoušek betonu a kontroly výztuže."
    ),
    "kniha_betonu.txt": "Kniha betonů obsahuje záznamy o dodávkách betonu pro základovou desku a datum betonáže.",
    "kontrola_vyztuze.txt": "Kontrola výztuže základové desky před betonáží ověřuje krytí a počet výztužných prutů.",
    "pentaflex_technicky_list.txt": "Technický list těsnicího pásu Pentaflex pro pracovní spáru základové desky.",
    "stavebni_denik_zaklady.txt": "Stavební deník eviduje postup prací na základové desce a předání jednotlivých etap investorovi.",
}


@pytest.fixture()
def duplicate_heavy_backend(tmp_path):
    root = tmp_path / "projekt"
    kd_folder = root / "02_REALIZACE" / "01_KONTROLNI_DNY"
    kd_folder.mkdir(parents=True)
    for i in range(15):
        (kd_folder / f"kontrolni_den_c{i:02d}.txt").write_text(KONTROLNI_DEN_TEXT, encoding="utf-8")
    for name, text in RELEVANT_DOCUMENTS.items():
        (root / name).write_text(text, encoding="utf-8")
    state = tmp_path / "state"
    embeddings = BagOfWordsEmbeddings()
    ai_search.sync(root, state / "database" / "project.sqlite3", state / "lance" / "project", embeddings)
    settings = ui.Settings(project_root=str(root), result_count=10)
    return settings, state, embeddings


def test_production_query_is_not_dominated_by_duplicate_kontrolni_dny(duplicate_heavy_backend):
    """The exact query from the production audit. Before the fix, all 10 results
    were duplicates of the same "základovou desku." sentence from different
    "Kontrolní den" files (e.g. KD č.72, KD č.71, KD č.75, ...)."""
    settings, state, embeddings = duplicate_heavy_backend
    rows = ui.search_all(
        "Co chybí k předání základové desky investorovi?", settings, state, embeddings, is_question=True
    )
    documents = [row["document"] for row in rows]
    duplicate_count = sum(1 for name in documents if name.startswith("kontrolni_den_"))
    assert duplicate_count < len(documents), f"Výsledky jsou stále zaplaveny duplicitami z kontrolních dnů: {documents}"
    assert duplicate_count <= ui.MAX_RESULTS_PER_FOLDER, f"Duplicitní folder cap nefunguje: {documents}"
    relevant_hits = [name for name in documents if name in RELEVANT_DOCUMENTS]
    assert relevant_hits, f"Žádný relevantní dokument (předání/beton/výztuž/Pentaflex/deník) se nedostal do výsledků: {documents}"


def test_document_mode_query_also_avoids_pure_duplicate_flood(duplicate_heavy_backend):
    """Even a plain document-lookup query (is_question=False) must not surface
    only copies of the same sentence - dedup/diversify apply to both modes."""
    settings, state, embeddings = duplicate_heavy_backend
    rows = ui.search_all("základovou desku", settings, state, embeddings, is_question=False)
    documents = [row["document"] for row in rows]
    duplicate_count = sum(1 for name in documents if name.startswith("kontrolni_den_"))
    assert duplicate_count <= ui.MAX_RESULTS_PER_FOLDER, f"Duplicitní folder cap nefunguje v dokumentovém režimu: {documents}"


# ---------------------------------------------------------------------------
# QA retrieval-pool widening (2026-08-06 fix): a natural-language question
# fans a single query out into many FTS5 OR terms, so documents that match
# several common terms with high frequency (e.g. verbose contracts) can push a
# genuinely relevant but lexically-sparse document (a short chunk matching only
# one term once) past a fixed top-100 candidate pool before RRF/rerank ever
# sees it. Production audit: best BM25 rank of the relevant chunk was 138, and
# its LanceDB vector rank was 661 - both past the old RETRIEVAL_POOL_SIZE=100.
# ---------------------------------------------------------------------------

RETRIEVAL_POOL_QUERY = "Co chybí k předání základové desky investorovi?"
RETRIEVAL_POOL_TARGET_TEXT = (
    "Certifikát shody izolačního pásu pro spáru mezi panely. Materiál byl dodán investorovi."
)


@pytest.fixture()
def sparse_relevant_document_backend(tmp_path):
    """~150 verbose 'noise' documents that match every OR term in the production
    query at high frequency, plus a single short, lexically-sparse document that
    matches only one term once - mirrors the production shape (many contracts
    dominating BM25 vs. one relevant scanned certificate). Each noise document
    gets a unique per-file marker so sync()'s content-hash dedup (identical
    files collapse to a single indexed document) doesn't merge them into one."""
    root = tmp_path / "projekt"
    root.mkdir(parents=True)
    for i in range(150):
        text = (
            f"Zápis č.{i}: Co chybí k předání základové desky investorovi? Předání "
            f"investorovi základové desky vyžaduje doplnění chybějících dokladů před "
            f"předáním č.{i}."
        )
        (root / f"noise_{i:03d}.txt").write_text(text, encoding="utf-8")
    (root / "certifikat_izolace.txt").write_text(RETRIEVAL_POOL_TARGET_TEXT, encoding="utf-8")
    state = tmp_path / "state"
    embeddings = BagOfWordsEmbeddings()
    db, lance = state / "database" / "project.sqlite3", state / "lance" / "project"
    ai_search.sync(root, db, lance, embeddings)
    return db, lance, embeddings


def test_document_mode_keeps_original_pool_size_and_excludes_sparse_match(sparse_relevant_document_backend):
    """is_question=False must behave exactly as before the fix: the fixed
    RETRIEVAL_POOL_SIZE=100 pool is dominated by the 150 high-frequency noise
    matches, so the sparse target never reaches the candidate pool at all."""
    db, lance, embeddings = sparse_relevant_document_backend
    results = ai_search.search(RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False)
    documents = [row["document"] for row in results]
    assert "certifikat_izolace.txt" not in documents, (
        "Dokumentový režim by měl zůstat na původní velikosti poolu (beze změny chování)."
    )


def test_question_mode_widened_pool_lets_sparse_relevant_document_through(sparse_relevant_document_backend):
    """is_question=True must widen both the FTS5/vector retrieval pool AND the
    post-RRF rerank pool enough that the sparse-but-relevant document reaches
    the output of search() - i.e. it passes into the next pipeline phase
    (RRF -> rerank -> dedup/diversify) instead of being silently dropped before
    RRF ever runs, which is exactly what happened in production."""
    db, lance, embeddings = sparse_relevant_document_backend
    results = ai_search.search(RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=200, is_question=True)
    documents = [row["document"] for row in results]
    assert "certifikat_izolace.txt" in documents, (
        f"QA režim s rozšířeným poolem nezachytil řídce zastoupený relevantní dokument: {documents[:10]}..."
    )


# ---------------------------------------------------------------------------
# candidate_strategy="union" (2026-08-06 architecture iteration): even with the
# pool-widening fix above, a candidate found by only ONE channel (e.g. a
# BM25-only hit, as in the production FERI case) can still rank at the very
# bottom of the RRF merge - purely for lacking the other channel's
# corroboration - and get discarded by the legacy `rerank_k` truncation before
# phase 3 (cosine rerank) ever sees it. "union" instead lets phase 3 see every
# candidate either channel found at all; RRF is still computed and still
# contributes to score, it just no longer decides what gets discarded before
# reranking. These tests target search()'s new `candidate_strategy` parameter
# and SearchTrace's `union_candidates`/`candidates_before_precision` fields
# directly (not the final result set), matching the case 6 requirement:
# survive-to-precision-stage, not necessarily make the final top-k.
# ---------------------------------------------------------------------------

def test_build_candidate_union_dedupes_by_chunk_id_and_keeps_provenance():
    fts_ids = ["a", "b", "c"]
    vector_ids = ["c", "b", "d"]
    union = ai_search._build_candidate_union(fts_ids, vector_ids)
    assert [c["chunk_id"] for c in union] == ["a", "b", "c", "d"], "deterministic order: first occurrence across fts_ids then vector_ids"
    by_id = {c["chunk_id"]: c for c in union}
    assert by_id["a"] == {"chunk_id": "a", "fts_rank": 0, "vector_rank": None, "fts_hit": True, "vector_hit": False}
    assert by_id["c"] == {"chunk_id": "c", "fts_rank": 2, "vector_rank": 0, "fts_hit": True, "vector_hit": True}
    assert by_id["d"] == {"chunk_id": "d", "fts_rank": None, "vector_rank": 2, "fts_hit": False, "vector_hit": True}


def test_build_candidate_union_handles_empty_inputs():
    assert ai_search._build_candidate_union([], []) == []
    assert [c["chunk_id"] for c in ai_search._build_candidate_union(["x"], [])] == ["x"]


def test_search_rejects_unknown_candidate_strategy(duplicate_heavy_backend):
    settings, state, embeddings = duplicate_heavy_backend
    db, lance = state / "database" / "project.sqlite3", state / "lance" / "project"
    with pytest.raises(ValueError):
        ai_search.search("Pentaflex", db, lance, embeddings, candidate_strategy="cross_encoder")


def test_default_candidate_strategy_is_legacy_and_unchanged(duplicate_heavy_backend):
    """search()'s default behaviour (no candidate_strategy argument at all,
    exactly how every existing call site in ui_services.py/tests/ calls it)
    must stay byte-for-byte identical to explicitly passing "legacy" - the
    production default must not move by adding this parameter."""
    settings, state, embeddings = duplicate_heavy_backend
    db, lance = state / "database" / "project.sqlite3", state / "lance" / "project"
    for query, is_q in (("Pentaflex", False), ("Co chybí k předání základové desky investorovi?", True)):
        default_result = ai_search.search(query, db, lance, embeddings, is_question=is_q)
        explicit_legacy = ai_search.search(query, db, lance, embeddings, is_question=is_q, candidate_strategy="legacy")
        assert default_result == explicit_legacy


@pytest.fixture()
def rrf_tail_backend(tmp_path):
    """60 near-identical noise documents that all out-rank a single sparse
    target on both channels, deterministically pushing the target to the very
    last position (60 of 61) in both the raw BM25 list and the RRF merge -
    below is_question=False's rerank_k=30 floor but still comfortably inside
    its retrieval_k=100 floor. This reproduces the FERI production shape
    (found by a channel, discarded only by the RRF-rank cutoff) with a size
    that is cheap enough to sync() in a unit test."""
    root = tmp_path / "projekt"
    root.mkdir(parents=True)
    for i in range(60):
        text = (
            f"Zápis č.{i}: Co chybí k předání základové desky investorovi? Předání "
            f"investorovi základové desky vyžaduje doplnění chybějících dokladů před "
            f"předáním č.{i}."
        )
        (root / f"noise_{i:03d}.txt").write_text(text, encoding="utf-8")
    (root / "certifikat_izolace.txt").write_text(RETRIEVAL_POOL_TARGET_TEXT, encoding="utf-8")
    state = tmp_path / "state"
    embeddings = BagOfWordsEmbeddings()
    db, lance = state / "database" / "project.sqlite3", state / "lance" / "project"
    ai_search.sync(root, db, lance, embeddings)
    with ai_search.database(db) as con:
        target_chunk_id = con.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.name='certifikat_izolace.txt'"
        ).fetchone()[0]
    return db, lance, embeddings, target_chunk_id


def test_legacy_drops_rrf_tail_candidate_that_union_keeps_before_precision(rrf_tail_backend):
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    legacy_trace = ai_search.SearchTrace()
    ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        trace=legacy_trace, candidate_strategy="legacy",
    )
    union_trace = ai_search.SearchTrace()
    ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        trace=union_trace, candidate_strategy="union",
    )

    # Precondition: the target really was found by BM25 at all (not an
    # indexing/chunking problem) - otherwise this test would prove nothing.
    bm25_hit = any(c["chunk_id"] == target_chunk_id for c in legacy_trace.bm25_candidates)
    assert bm25_hit, "test setup problem: target chunk was not even found by BM25 - fixture no longer reproduces the RRF-tail shape"

    legacy_before = {c["chunk_id"] for c in legacy_trace.candidates_before_precision}
    union_before = {c["chunk_id"] for c in union_trace.candidates_before_precision}

    assert target_chunk_id not in legacy_before, "legacy candidate_strategy should still drop this RRF-tail hit before phase 3 (unchanged production behaviour)"
    assert target_chunk_id in union_before, "union candidate_strategy must let a channel-found candidate reach candidates_before_precision regardless of its RRF rank"

    # Structural invariant, true by construction for ANY query: legacy's pool
    # is always a strict subset of the union pool, because legacy's top_ids
    # come from truncating the exact same fusion_order whose keys equal the
    # union's chunk_id set.
    assert legacy_before.issubset(union_before)
    assert len(union_before) > len(legacy_before)


def test_union_trace_records_provenance_for_the_rrf_tail_candidate(rrf_tail_backend):
    """union_candidates must be populated (independent of candidate_strategy)
    and correctly flag the target as an FTS hit, not a vector hit - matching
    case 6's required A/B/C checks (found by BM25? in the union pool?)."""
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    trace = ai_search.SearchTrace()
    ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        trace=trace, candidate_strategy="legacy",  # union_candidates is populated even under legacy
    )
    entry = next(c for c in trace.union_candidates if c["chunk_id"] == target_chunk_id)
    assert entry["fts_hit"] is True
    assert entry["fts_rank"] is not None


# ---------------------------------------------------------------------------
# candidate_strategy="union_ce" (cross-encoder precision reranker, 2026-08-06):
# builds on "union"'s recall-safe pool, but scores it with an independent
# (query, passage) relevance model instead of cosine similarity. These tests
# use a small deterministic fake in place of the real ~2GB BAAI/bge-reranker-
# v2-m3 model - see the production benchmark for real-model measurements.
# ---------------------------------------------------------------------------

class _ScriptedCrossEncoder:
    """Test double satisfying CrossEncoderReranker's score() contract without
    loading any real model. `scores_by_passage` maps exact passage text to a
    score; `on_score` (if given) runs first and can raise, to exercise the
    fallback path."""
    name = "scripted-test-cross-encoder"
    def __init__(self, scores_by_passage=None, on_score=None):
        self.scores_by_passage = scores_by_passage or {}; self.on_score = on_score; self.calls = 0
    def score(self, query, passages):
        self.calls += 1
        if self.on_score is not None: self.on_score()
        return [self.scores_by_passage.get(p, 0.0) for p in passages]


def test_select_pre_ce_candidates_keeps_each_channels_own_top_budget_independently():
    """A candidate must survive as long as it is within ITS OWN channel's
    top-`budget`, regardless of the other channel or a combined RRF rank -
    the exact property that makes this safe against the FERI-style bug
    (BM25-only hit discarded purely for a poor RRF/vector showing)."""
    fts_ids = [f"fts-noise-{i}" for i in range(5)] + ["bm25_only_target"]
    vector_ids = [f"vector-noise-{i}" for i in range(5)] + ["vector_only_target"]

    wide = ai_search._select_pre_ce_candidates(fts_ids, vector_ids, budget=6)
    wide_ids = {c["chunk_id"] for c in wide}
    assert {"bm25_only_target", "vector_only_target"}.issubset(wide_ids)

    narrow = ai_search._select_pre_ce_candidates(fts_ids, vector_ids, budget=3)
    narrow_ids = {c["chunk_id"] for c in narrow}
    assert "bm25_only_target" not in narrow_ids and "vector_only_target" not in narrow_ids
    assert len(narrow_ids) == 6, "budget=3 per channel, 2 disjoint channels, no overlap here -> exactly 6"


def test_union_ce_bm25_only_rrf_tail_candidate_survives_pre_ce_selection(rrf_tail_backend):
    """The same RRF-tail candidate that "legacy" drops before phase 3 (see
    test_legacy_drops_rrf_tail_candidate_that_union_keeps_before_precision)
    must also survive union_ce's pre-CE budget selection - this is the
    recall-safety property KROK 4/5 require before the cross-encoder ever runs."""
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    ce = _ScriptedCrossEncoder()
    trace = ai_search.SearchTrace()
    ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        trace=trace, candidate_strategy="union_ce", cross_encoder=ce,
    )
    before_ce_ids = {c["chunk_id"] for c in trace.candidates_before_cross_encoder}
    assert target_chunk_id in before_ce_ids


def test_union_ce_final_ranking_is_pure_cross_encoder_score_not_blended_with_rrf(rrf_tail_backend):
    """KROK 7: once the cross-encoder actually ran, ITS score must determine
    the final order/score - not a new blend with RRF/cosine/BM25. Give the
    RRF-tail (BM25-only, worst RRF rank) candidate the single HIGHEST CE
    score of the whole pool and assert it becomes the #1 result."""
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    with ai_search.database(db) as con:
        target_text = con.execute("SELECT text FROM chunks WHERE id=?", (target_chunk_id,)).fetchone()[0]
    ce = _ScriptedCrossEncoder(scores_by_passage={target_text: 999.0})
    results = ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        candidate_strategy="union_ce", cross_encoder=ce,
    )
    assert results, "search() must return results"
    assert results[0]["document"] == "certifikat_izolace.txt"
    assert results[0]["score"] == 999.0
    assert results[0]["match"]["cross_encoder_score"] == 999.0


def test_union_ce_passes_full_indexed_chunk_text_not_ui_truncated_quote(rrf_tail_backend, monkeypatch):
    """KROK 6: the cross-encoder must score the actual indexed chunk text
    (c.text), not the UI's 700-char `quote` preview - assert unmodified by
    making the fake fail the assertion if it ever receives a truncated/short
    stand-in instead of the real (short, in this fixture) chunk body."""
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    with ai_search.database(db) as con:
        target_text = con.execute("SELECT text FROM chunks WHERE id=?", (target_chunk_id,)).fetchone()[0]
    seen_passages = []
    class RecordingCE:
        name = "recording-ce"
        def score(self, query, passages):
            seen_passages.extend(passages)
            return [0.0 for _ in passages]
    ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        candidate_strategy="union_ce", cross_encoder=RecordingCE(),
    )
    assert target_text in seen_passages


def test_union_ce_falls_back_to_union_scoring_on_cross_encoder_exception(rrf_tail_backend):
    """KROK 8: any cross-encoder exception (model error, corrupt state, ...)
    must fall back to identical behaviour as candidate_strategy="union" -
    never raise, never return an emptier/degraded result."""
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    def _boom(): raise RuntimeError("simulated model failure")
    broken_ce = _ScriptedCrossEncoder(on_score=_boom)
    union_ce_results = ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        candidate_strategy="union_ce", cross_encoder=broken_ce,
    )
    union_results = ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        candidate_strategy="union",
    )
    assert union_ce_results == union_results
    assert broken_ce.calls == 1, "the cross-encoder must actually be attempted, not skipped"


def test_union_ce_falls_back_to_union_scoring_on_cross_encoder_timeout(rrf_tail_backend):
    """Same as above, specifically for a timeout (KROK 8 explicitly calls out
    'inference timeoutuje' as a fallback trigger, not just a generic error)."""
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    def _timeout(): raise TimeoutError("cross-encoder predict timed out")
    timing_out_ce = _ScriptedCrossEncoder(on_score=_timeout)
    union_ce_results = ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        candidate_strategy="union_ce", cross_encoder=timing_out_ce,
    )
    union_results = ai_search.search(
        RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
        candidate_strategy="union",
    )
    assert union_ce_results == union_results


def test_union_ce_fallback_logs_error_and_records_it_in_trace(rrf_tail_backend, caplog):
    """KROK 8: 'chybu zaloguj, žádné tiché selhání' - the fallback must be
    observable both via the standard logging module AND via SearchTrace, not
    just swallowed."""
    import logging
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    def _boom(): raise RuntimeError("simulated model failure")
    broken_ce = _ScriptedCrossEncoder(on_score=_boom)
    trace = ai_search.SearchTrace()
    with caplog.at_level(logging.WARNING, logger="ai_search.search"):
        ai_search.search(
            RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False,
            candidate_strategy="union_ce", cross_encoder=broken_ce, trace=trace,
        )
    assert any("CROSS_ENCODER_FALLBACK" in record.message for record in caplog.records)
    assert "simulated model failure" in trace.metadata.get("cross_encoder_error", "")
    assert trace.cross_encoder_model is None, "CE did not successfully run, so this must stay unset (not the fallback strategy's name)"


def test_default_candidate_strategy_still_legacy_after_adding_union_ce(duplicate_heavy_backend):
    """Adding a third strategy must not move the production default - repeats
    test_default_candidate_strategy_is_legacy_and_unchanged's assertion
    explicitly against a codebase that now also has "union_ce" available."""
    settings, state, embeddings = duplicate_heavy_backend
    db, lance = state / "database" / "project.sqlite3", state / "lance" / "project"
    assert ai_search.CANDIDATE_STRATEGY_LEGACY == "legacy"
    assert "union_ce" in ai_search.CANDIDATE_STRATEGIES
    default_result = ai_search.search("Pentaflex", db, lance, embeddings)
    legacy_result = ai_search.search("Pentaflex", db, lance, embeddings, candidate_strategy="legacy")
    assert default_result == legacy_result


# ---------------------------------------------------------------------------
# search_all() post-rerank truncation fix (2026-08-07 Phase 2 diagnostic):
# search_all() used to cut the per-document candidate list down to
# `candidate_pool` (== settings.result_count for a non-question query, =10 in
# production) BEFORE deduplicate_by_content()/diversify_results() ever ran.
# A document correctly found by BM25 at rank 0 and kept through RRF+rerank
# could still be silently dropped here purely because it was the 11th+ best
# individual document by best-chunk score - verified against 3 real
# production queries (ranks 11/16/19 of 27-38 unique documents). These tests
# use a scripted `ai_search.search()` double (same pattern as
# `_ScriptedCrossEncoder` above) for a fully deterministic candidate list,
# instead of relying on real embedding/BM25 ranking to land a document at an
# exact position - which the fix itself has nothing to do with.
# ---------------------------------------------------------------------------

def _fake_row(path: str, score: float, document: str | None = None, quote: str | None = None) -> dict:
    # Default quote is a hash of the document name (verified similarity ratio
    # < 0.4 for any pair among 60 such quotes) so deduplicate_by_content() -
    # which collapses near-identical quote TEXT across different documents -
    # never treats these fixture rows as content duplicates of each other;
    # that would be a false positive unrelated to the truncation-order bug
    # these tests target. A plain "obsah dokumentu {name}" template was tried
    # first and rejected: differing only in a short numeric suffix, its
    # SequenceMatcher ratio across 60 rows peaked at 0.92 - above
    # CONTENT_DUPLICATE_THRESHOLD(0.90) - so it silently collapsed rows.
    document = document or Path(path).name
    quote = quote if quote is not None else hashlib.sha256(document.encode()).hexdigest()
    return {"document": document, "path": path, "project": "P", "heading": "", "quote": quote, "score": score, "match": {}}


_DUPLICATE_QUOTE = "Práce jsou dočasně pozastaveny do nástupu na základovou desku."


def _duplicate_flood_rows(duplicate_count: int, target_score: float, filler_count: int = 0) -> list[dict]:
    """Reproduces the actual production mechanism (not just a plain low
    score): `duplicate_count` documents on distinct paths/names but sharing
    IDENTICAL quote text, ranked above one genuinely-distinct `target.pdf`.
    deduplicate_by_content() collapses the duplicates down to a single
    survivor - which only frees a final-page slot for `target.pdf` (and any
    `filler_*` rows, scored just below target) if all of them were still in
    the list deduplicate_by_content() got to see. Verified against the real
    ui.deduplicate_by_content()/diversify_results() functions on 2026-08-07:
    with the OLD [:candidate_pool] cut applied before this helper's output
    even exists, all `duplicate_count` copies alone already fill the old
    10-wide pool, so target/filler never entered dedup at all."""
    duplicates = [
        _fake_row(f"/kd/{i:03d}/kd.pdf", score=1.0 - i * 0.001, document=f"kd_{i:03d}.pdf", quote=_DUPLICATE_QUOTE)
        for i in range(duplicate_count)
    ]
    target = _fake_row("/other/target.pdf", score=target_score, document="target.pdf",
                        quote="Úplně odlišný text o Pentaflexu a betonáži " + hashlib.sha256(b"target.pdf").hexdigest())
    filler = [
        _fake_row(f"/filler/{i:03d}/f.pdf", score=target_score - 0.001 - i * 0.001, document=f"filler_{i:03d}.pdf")
        for i in range(filler_count)
    ]
    return duplicates + [target] + filler


def _run_scripted_search_all(tmp_path, monkeypatch, scripted_rows, *, is_question=False, result_count=10):
    def fake_search(query, db, lance, embeddings, limit, is_question=False, expand_query=False, trace=None, **kwargs):
        return list(scripted_rows)
    monkeypatch.setattr(ai_search, "search", fake_search)
    state = tmp_path / "state"
    db, _ = ui.state_paths(state, "Dokument")
    db.touch()  # search_all() only calls the (scripted) ai_search.search() if this path exists
    settings = ui.Settings(project_root=str(tmp_path), result_count=result_count)
    return ui.search_all("dotaz", settings, state, embeddings=None, is_question=is_question)


def test_search_all_document_mode_promotes_target_freed_by_widened_dedup_pool(tmp_path, monkeypatch):
    """Core regression test for the task's exact scenario: 12 near-identical
    "Kontrolní den"-style duplicates outrank one genuinely relevant, distinct
    `target.pdf` (rank 13) by best-chunk score, with 20 further distinct
    documents ranked just below it. Before this fix, the OLD
    [:candidate_pool] cut (=settings.result_count=10 for a non-question
    query) ran BEFORE deduplicate_by_content()/diversify_results(), so all 10
    kept slots were duplicate copies of the SAME document - deduplicate_by_
    content() collapsed them to 1 survivor and target/filler never entered
    the pool at all (verified: OLD final = 1 result, no target). After this
    fix, dedup sees all 33 candidates, collapses the 12 duplicates to 1
    survivor, and target/filler fill the remaining page - proving both that
    target now appears AND that it does so via legitimate dedup, not by
    accidentally returning more than settings.result_count rows."""
    scripted_rows = _duplicate_flood_rows(duplicate_count=12, target_score=0.85, filler_count=20)
    results = _run_scripted_search_all(tmp_path, monkeypatch, scripted_rows, is_question=False, result_count=10)
    documents = [row["document"] for row in results]
    assert "target.pdf" in documents, (
        f"Relevantní dokument zmizel jen kvůli předčasnému [:candidate_pool] střihu PŘED "
        f"deduplicate_by_content()/diversify_results(): {documents}"
    )
    assert len(results) == 10, f"Requirement 2: výsledný počet musí zůstat settings.result_count: {documents}"


def test_search_all_document_mode_result_count_unchanged_when_target_absent(tmp_path, monkeypatch):
    """The widened pre-diversify pool must not, by itself, inflate the final
    result count beyond settings.result_count even when there are far more
    than result_count distinct documents available (20 here, none of them
    near-duplicates) - the fix only changes WHICH documents compete for the
    page, not how many are returned."""
    noise = [_fake_row(f"/proj/folder_{i:02d}/noise.pdf", score=1.0 - i * 0.01, document=f"noise_{i:02d}.pdf") for i in range(20)]
    results = _run_scripted_search_all(tmp_path, monkeypatch, noise, is_question=False, result_count=10)
    assert len(results) == 10


def test_search_all_question_mode_dedup_promotion_still_works_unaffected(tmp_path, monkeypatch):
    """Requirement 3, positive half: is_question=True already used
    QA_CANDIDATE_POOL(=50) as `candidate_pool` before this fix, so the exact
    same 12-duplicate/target/20-filler scenario as the document-mode test
    above was ALREADY promoted correctly pre-fix (33 rows < 50) - this proves
    the fix left that pre-existing, already-correct behaviour untouched."""
    scripted_rows = _duplicate_flood_rows(duplicate_count=12, target_score=0.85, filler_count=20)
    results = _run_scripted_search_all(tmp_path, monkeypatch, scripted_rows, is_question=True, result_count=10)
    documents = [row["document"] for row in results]
    assert "target.pdf" in documents, "QA režim už widening měl (QA_CANDIDATE_POOL=50) - toto chování se opravou nesmí změnit"
    assert len(results) == 10


def test_search_all_question_mode_pool_width_still_exactly_qa_candidate_pool(tmp_path, monkeypatch):
    """Requirement 3, negative half: pins diversify_pool for is_question=True
    to EXACTLY QA_CANDIDATE_POOL(=50), not something wider that this fix
    might have accidentally introduced. 52 duplicates (> 50) push target to a
    position deduplicate_by_content() never gets to see even after the fix -
    it must stay dropped, exactly as it was before this change."""
    assert ui.QA_CANDIDATE_POOL == 50, "test assumes today's constant; adjust duplicate_count below if this ever changes"
    scripted_rows = _duplicate_flood_rows(duplicate_count=52, target_score=0.30)
    results = _run_scripted_search_all(tmp_path, monkeypatch, scripted_rows, is_question=True, result_count=10)
    documents = [row["document"] for row in results]
    assert "target.pdf" not in documents, "QA-mode pool šířka se touto opravou nesmí rozšířit nad QA_CANDIDATE_POOL"


def test_search_all_document_mode_pool_width_pinned_to_qa_candidate_pool_constant(tmp_path, monkeypatch):
    """Pins the NEW non-question width to exactly QA_CANDIDATE_POOL(=50), not
    an unbounded or larger widening: 48 duplicates (within 50) still promote
    target, but 52 duplicates (beyond 50) do not - the same boundary
    is_question=True already had."""
    assert ui.QA_CANDIDATE_POOL == 50, "test assumes today's constant; adjust duplicate_count below if this ever changes"
    within_boundary = _duplicate_flood_rows(duplicate_count=48, target_score=0.30)
    results_within = _run_scripted_search_all(tmp_path, monkeypatch, within_boundary, is_question=False, result_count=10)
    assert "target.pdf" in [row["document"] for row in results_within], "48 duplikátů + target (49 celkem) se musí vejít do rozšířeného poolu (50)"

    beyond_boundary = _duplicate_flood_rows(duplicate_count=52, target_score=0.30)
    results_beyond = _run_scripted_search_all(tmp_path, monkeypatch, beyond_boundary, is_question=False, result_count=10)
    assert "target.pdf" not in [row["document"] for row in results_beyond], "52 duplikátů + target (53 celkem) překračuje QA_CANDIDATE_POOL=50 - beze změny musí zůstat zahozen"


# ---------------------------------------------------------------------------
# Document-level evidence aggregation (2026-08-07 ranking audit): search_all()
# used to score a merged document row with a pure MAX over its chunks, so a
# fragmented document with 10 matching chunks ranked identically to one with a
# single matching chunk. Measured on the real index: the correct invoice for
# "faktura Nazarenko stavební práce" had all 10 of its chunks in the 50-slot
# pool and BM25 rank 0, yet placed 15th of 27 documents. The bonus is gated on
# DISTINCT chunks via _select_quote_chunks(), so repeated boilerplate earns
# nothing - the two tests below pin both halves of that contract.
# ---------------------------------------------------------------------------

BOILERPLATE_CHUNK = "Práce jsou dočasně pozastaveny do nástupu na základovou desku."


def _chunk_row(path: str, document: str, score: float, quote: str) -> dict:
    return {"document": document, "path": path, "project": "P", "heading": "", "quote": quote, "score": score, "match": {}}


def _distinct_quote(document: str, index: int) -> str:
    """Chunk text guaranteed distinct from every other chunk of the same
    document (hash-based, verified pairwise similarity < 0.4), so
    _select_quote_chunks() counts it as real supporting evidence."""
    return hashlib.sha256(f"{document}-{index}".encode()).hexdigest()


def test_document_evidence_score_counts_only_distinct_supporting_chunks():
    """Unit-level contract of the bonus itself, independent of search_all():
    3 distinct supporting chunks contribute, a 4th is beyond
    EVIDENCE_BONUS_MAX_CHUNKS, and near-duplicates contribute nothing."""
    quotes = [_distinct_quote("d", i) for i in range(5)]
    scores = [1.0, 0.8, 0.6, 0.4, 0.2]
    expected = 1.0 + ui.EVIDENCE_BONUS_WEIGHT * (0.8 + 0.6 + 0.4)  # 5th chunk excluded by the cap
    assert ui._document_evidence_score(1.0, scores, quotes) == pytest.approx(expected)

    # single chunk -> unchanged, and all-duplicates -> unchanged (pre-fix MAX)
    assert ui._document_evidence_score(1.0, [1.0], [quotes[0]]) == 1.0
    duplicates = [BOILERPLATE_CHUNK] * 50
    assert ui._document_evidence_score(1.0, [1.0] * 50, duplicates) == 1.0


def test_search_all_prefers_document_with_ten_relevant_chunks_over_equally_strong_single_chunk(tmp_path, monkeypatch):
    """Requirement A, literal form: 10 relevant chunks vs ONE equally strong
    chunk (both best-chunk scores exactly 0.50). Under the pre-fix pure MAX
    this was a bare tie decided by insertion order, so ordering alone proves
    nothing here - the meaningful assertion is that the fragmented document now
    carries a strictly higher document score than the single-chunk one."""
    many = [_chunk_row("/proj/many/many.pdf", "many_chunks.pdf", 0.50 - i * 0.01, _distinct_quote("many", i)) for i in range(10)]
    single = [_chunk_row("/proj/single/single.pdf", "single_chunk.pdf", 0.50, _distinct_quote("single", 0))]
    results = _run_scripted_search_all(tmp_path, monkeypatch, many + single, is_question=False, result_count=10)
    by_document = {row["document"]: row for row in results}
    assert by_document["many_chunks.pdf"]["score"] > by_document["single_chunk.pdf"]["score"]
    assert [row["document"] for row in results][0] == "many_chunks.pdf"
    # the bonus is exactly the 3 next distinct chunks (0.49/0.48/0.47), not all 9
    expected = 0.50 + ui.EVIDENCE_BONUS_WEIGHT * (0.49 + 0.48 + 0.47)
    assert by_document["many_chunks.pdf"]["score"] == pytest.approx(expected)
    assert by_document["many_chunks.pdf"]["best_chunk_score"] == 0.50, "původní best-chunk skóre musí zůstat dostupné"
    assert by_document["single_chunk.pdf"]["score"] == 0.50, "jednochunkový dokument nesmí být oprava nijak ovlivněn"


def test_search_all_ten_relevant_chunks_overtake_a_strictly_stronger_single_chunk(tmp_path, monkeypatch):
    """Requirement A, discriminating form - this is the test that actually
    fails without the fix. The single-chunk document has a strictly HIGHER best
    chunk (0.55 vs 0.50), so the pre-fix pure MAX ranked it first; verified
    against the pre-fix behaviour, which returned single_chunk.pdf at position
    0. Accumulated distinct evidence must now flip that order."""
    many = [_chunk_row("/proj/many/many.pdf", "many_chunks.pdf", 0.50 - i * 0.01, _distinct_quote("many", i)) for i in range(10)]
    single = [_chunk_row("/proj/single/single.pdf", "single_chunk.pdf", 0.55, _distinct_quote("single", 0))]
    results = _run_scripted_search_all(tmp_path, monkeypatch, many + single, is_question=False, result_count=10)
    documents = [row["document"] for row in results]
    assert documents[0] == "many_chunks.pdf", f"Fragmentovaný dokument s 10 relevantními chunky musí předběhnout silnější jednochunkový: {documents}"
    assert results[0]["score"] > results[0]["best_chunk_score"], "agregace musí skóre skutečně navýšit"


def test_search_all_does_not_let_fifty_boilerplate_chunks_beat_one_relevant_chunk(tmp_path, monkeypatch):
    """Requirement B: 50 copies of the same boilerplate sentence in one
    document must NOT outrank a single genuinely relevant chunk. This holds
    structurally, not by tuning: _select_quote_chunks() collapses the 50 copies
    to one distinct chunk, so the boilerplate document earns a zero bonus and
    keeps exactly its pre-fix MAX score."""
    boilerplate = [_chunk_row("/proj/spam/spam.pdf", "spam.pdf", 0.40 - i * 0.001, BOILERPLATE_CHUNK) for i in range(50)]
    relevant = [_chunk_row("/proj/good/good.pdf", "good.pdf", 0.45, _distinct_quote("good", 0))]
    results = _run_scripted_search_all(tmp_path, monkeypatch, boilerplate + relevant, is_question=False, result_count=10)
    documents = [row["document"] for row in results]
    assert documents[0] == "good.pdf", f"Boilerplate dokument nesmí předběhnout relevantní: {documents}"
    spam_row = next((r for r in results if r["document"] == "spam.pdf"), None)
    if spam_row is not None:
        assert spam_row["score"] == spam_row["best_chunk_score"], "boilerplate nesmí získat žádný evidence bonus"


def test_search_all_question_mode_scoring_is_byte_identical_to_pre_fix_max(tmp_path, monkeypatch):
    """The aggregation is deliberately gated to is_question=False. In question
    mode the merged row's score must still be exactly its best chunk's score,
    with the same multi-chunk fixture that visibly changes document order in
    document mode."""
    many = [_chunk_row("/proj/many/many.pdf", "many_chunks.pdf", 0.50 - i * 0.01, _distinct_quote("many", i)) for i in range(10)]
    single = [_chunk_row("/proj/single/single.pdf", "single_chunk.pdf", 0.50, _distinct_quote("single", 0))]
    results = _run_scripted_search_all(tmp_path, monkeypatch, many + single, is_question=True, result_count=10)
    for row in results:
        assert row["score"] == row["best_chunk_score"], f"question mode nesmí agregovat: {row['document']}"


def test_union_strategy_unaffected_by_cross_encoder_argument(rrf_tail_backend):
    """`cross_encoder=` must be a no-op for candidate_strategy="union" (and,
    by the same code path, "legacy") - passing one should never change
    behaviour outside candidate_strategy="union_ce"."""
    db, lance, embeddings, target_chunk_id = rrf_tail_backend
    ce = _ScriptedCrossEncoder(scores_by_passage={"anything": 999.0})
    without = ai_search.search(RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False, candidate_strategy="union")
    with_ce_arg = ai_search.search(RETRIEVAL_POOL_QUERY, db, lance, embeddings, limit=8, is_question=False, candidate_strategy="union", cross_encoder=ce)
    assert without == with_ce_arg
    assert ce.calls == 0, "cross_encoder must not even be consulted for a non-union_ce strategy"
