"""Regression tests for quota-based quote aggregation in search_all()'s
path-merge step (see ui_services.py's "Quote aggregation" section docstring).

Production bug this targets: "Jaké doklady musí dodat zhotovitel po betonáži?"
- the document "KZP - TEXTOVÁ ČÁST.pdf" won 21 chunks into the RRF-merged
list and was correctly ranked #1 end-to-end (search -> rerank -> path-merge ->
dedup -> diversify -> final), yet the answer-bearing chunk ("Kontrola
dodacích listů betonové směsi", the document's OWN chunk rank 4) never
appeared in the merged `quote` string handed to ai_search.answer(), because
the OLD merge concatenated chunks strictly in score order until a fixed
1200-char cap was hit - the top 1-2 chunks consumed the whole budget.

Tests A-E below are the exact five cases requested for this fix:
  A) a document with 10+ chunks -> merged quote contains more than one chunk
  B) the relevant chunk is NOT the top-scored one -> it still makes it in
  C) merged quote never exceeds the char budget
  D) duplicate/near-duplicate chunk text is not repeated
  E) a single-chunk document's quote is completely unchanged
"""
from __future__ import annotations

import hashlib
import re

import pytest

import ai_search
import ui_services as ui

DIMENSIONS = 16
STOPWORDS = {"a", "na", "v", "o", "k", "s", "z", "do", "po", "pro", "je", "se", "ze", "za", "ve"}


def _hashed_vector(seed: str) -> list[float]:
    digest = hashlib.sha256(seed.encode()).digest()
    return [(b / 127.5 - 1.0) for b in digest[:DIMENSIONS]]


class BagOfWordsEmbeddings:
    """Same neutral fake embedder used by test_retrieval_dedup_diversify.py -
    similarity driven purely by shared vocabulary, no domain knowledge."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            folded = text.casefold()
            tokens = [t[:6] for t in re.findall(r"\w+", folded) if len(t) > 2 and t not in STOPWORDS]
            vector = [0.0] * DIMENSIONS
            for token in tokens:
                vector = [a + b for a, b in zip(vector, _hashed_vector(token))]
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


# ---------------------------------------------------------------------------
# Unit tests: _select_quote_chunks / _merge_quote_chunks in isolation
# ---------------------------------------------------------------------------

def test_a_document_with_many_chunks_merges_more_than_one():
    topics = [
        "Obecný úvod kontrolního plánu", "Harmonogram kontrolních bodů", "Odpovědné osoby za kvalitu",
        "Přejímka výztuže před betonáží", "Kontrola dodacích listů betonu", "Zkouška krychelné pevnosti",
        "Vizuální kontrola bednění", "Ošetřování betonu po betonáži", "Klimatické podmínky při betonáži",
        "Závěrečné shrnutí plánu", "Příloha - vzor protokolu", "Příloha - vzor předání",
    ]
    chunks = [f"{topic} je popsán v této části dokumentu s doplňujícími technickými detaily." for topic in topics]
    quote, evidence = ui._merge_quote_chunks(chunks)
    assert quote.count(ui.QUOTE_MERGE_SEPARATOR) >= 1, "merged quote must contain more than one chunk"
    assert len(evidence) >= 2


def test_b_relevant_chunk_below_top_score_is_not_dropped():
    """Reproduces the production shape: several of the document's own
    top-scored chunks are near-identical boilerplate (the "~80 near-identical
    Kontrolní den" pattern from the production audit, here within ONE
    document's own chunks) - the OLD algorithm would have appended each of
    them in score order until the 1200-char budget was exhausted, never
    reaching the answer-bearing chunk ranked #4. The NEW quota mechanism
    skips near-duplicates (freeing quota slots) so a genuinely distinct,
    lower-ranked chunk still gets a slot."""
    chunks = [
        "Obecný úvodní odstavec KZP s všeobecnými informacemi o kontrolách kvality na stavbě.",  # rank 0
        "Obecný úvodní odstavec KZP s všeobecnými informacemi o kontrolách kvality na stavbě, verze 2.",  # rank 1 - near-dup of rank 0
        "Harmonogram kontrolních bodů pro zemní práce a základy stavby.",  # rank 2 - distinct
        "Obecný úvodní odstavec KZP s všeobecnými informacemi o kontrolách kvality na stavbě, verze 3.",  # rank 3 - near-dup again
        "Kontrola dodacích listů betonové směsi při každé dodávce na stavbu.",  # rank 4 - the answer
    ]
    quote, evidence = ui._merge_quote_chunks(chunks)
    assert "Kontrola dodacích listů betonové směsi" in quote
    assert any(item["rank"] == 4 for item in evidence)
    assert len(evidence) == ui.QUOTE_MERGE_MAX_CHUNKS


def test_c_merged_quote_never_exceeds_the_budget():
    chunks = [f"Velmi dlouhý textový chunk číslo {i} " * 40 for i in range(6)]
    quote, _ = ui._merge_quote_chunks(chunks)
    assert len(quote) <= ui.QUOTE_MERGE_MAX_CHARS


def test_c_no_single_chunk_can_consume_the_whole_budget():
    """Protection #1 from the task: even the best-scored chunk alone must not
    eat the entire quota when other, distinct chunks exist."""
    huge_first_chunk = "Text " * 1000  # far longer than the whole budget
    chunks = [huge_first_chunk, "Kontrola dodacích listů betonové směsi.", "Zkouška krychelné pevnosti betonu."]
    quote, evidence = ui._merge_quote_chunks(chunks)
    per_chunk_budget = ui.QUOTE_MERGE_MAX_CHARS // ui.QUOTE_MERGE_MAX_CHUNKS
    assert len(evidence[0]["text"]) <= per_chunk_budget
    assert "Kontrola dodacích listů betonové směsi" in quote


def test_d_duplicate_chunk_text_is_not_repeated():
    chunks = [
        "Kontrola dodacích listů betonové směsi při dodávce.",
        "kontrola dodacích listů betonové směsi při dodávce",  # same content, different case/punctuation
        "Zcela odlišný text o zkoušce krychelné pevnosti betonu po 28 dnech.",
    ]
    quote, evidence = ui._merge_quote_chunks(chunks)
    assert quote.count("Kontrola dodacích listů") + quote.count("kontrola dodacích listů") == 1
    assert len(evidence) == 2


def test_two_chunk_document_does_not_waste_the_unused_third_slot():
    """A document with only 2 distinct chunks must not be capped at the same
    per-chunk budget as a 3+-chunk document - that would silently shrink its
    total quote length versus the pre-fix behaviour for no relevance reason
    (confirmed on the production benchmark: a fixed 3-way split reduced
    average context size by 13-18%, outside this fix's own ~5% budget)."""
    chunk_a = "Kontrola dodacích listů betonové směsi při každé dodávce na stavbu. " * 15
    chunk_b = "Zkouška krychelné pevnosti betonu po 28 dnech od data betonáže. " * 15
    quote, evidence = ui._merge_quote_chunks([chunk_a, chunk_b])
    assert len(evidence) == 2
    per_chunk_budget = ui.QUOTE_MERGE_MAX_CHARS // 2
    assert len(evidence[0]["text"]) == per_chunk_budget
    assert len(evidence[1]["text"]) == per_chunk_budget
    assert len(quote) > ui.QUOTE_MERGE_MAX_CHARS // ui.QUOTE_MERGE_MAX_CHUNKS * 2  # strictly more than the old fixed-slot allocation would have given


def test_extra_distinct_chunks_beyond_the_quota_fill_leftover_budget():
    """Phase 2: when the guaranteed max_chunks slots are all short, further
    distinct chunks beyond the quota must still be pulled in - otherwise a
    document with many short chunks (the real "KZP" production shape) ends
    up with an artificially small merged quote versus the pre-fix behaviour,
    just because more than `max_chunks` genuinely distinct pieces exist."""
    topics = [
        "Harmonogram kontrolních bodů", "Odpovědné osoby za kvalitu", "Přejímka výztuže",
        "Kontrola dodacích listů betonu", "Zkouška krychelné pevnosti", "Vizuální kontrola bednění",
        "Ošetřování betonu po betonáži", "Klimatické podmínky při betonáži",
    ]
    short_chunks = [f"{topic}." for topic in topics]
    quote, evidence = ui._merge_quote_chunks(short_chunks)
    assert len(evidence) > ui.QUOTE_MERGE_MAX_CHUNKS, "budget slack must pull in chunks beyond the guaranteed quota"
    assert [item["rank"] for item in evidence] == sorted(item["rank"] for item in evidence)
    assert len(quote) <= ui.QUOTE_MERGE_MAX_CHARS


def test_e_single_chunk_document_quote_is_unchanged():
    text = "Jediný chunk tohoto dokumentu s běžnou délkou textu."
    quote, evidence = ui._merge_quote_chunks([text])
    assert quote == text
    assert evidence == []


def test_empty_chunk_list_returns_empty_quote():
    assert ui._merge_quote_chunks([]) == ("", [])


def test_select_quote_chunks_respects_max_chunks_quota():
    chunks = [
        "Kontrola dodacích listů betonové směsi při dodávce na stavbu.",
        "Zkouška krychelné pevnosti betonu po 28 dnech od betonáže.",
        "Protokol o vizuální kontrole bednění před zahájením betonáže.",
        "Postup ošetřování betonu po betonáži v závislosti na teplotě vzduchu.",
        "Evidence klimatických podmínek naměřených během betonáže základů.",
    ]
    selected = ui._select_quote_chunks(chunks, max_chunks=3)
    assert len(selected) == 3
    assert [rank for rank, _ in selected] == [0, 1, 2]


# ---------------------------------------------------------------------------
# End-to-end: real sync() + search_all(), reproducing the production shape
# ---------------------------------------------------------------------------

@pytest.fixture()
def many_chunk_document_backend(tmp_path):
    """One document built from many short, topically distinct paragraphs so it
    reliably indexes as 10+ chunks, plus a couple of unrelated documents so
    path-merge/dedup/diversify all still have real work to do."""
    root = tmp_path / "projekt"
    root.mkdir()
    paragraphs = [
        "Obecný úvod kontrolního a zkušebního plánu KZP pro stavbu Garáže NDS.",
        "Harmonogram kontrolních bodů pro zemní práce a základové konstrukce.",
        "Odpovědné osoby za kvalitu a jejich role v realizačním týmu.",
        "Postup přejímky výztuže před betonáží základové desky.",
        "Kontrola dodacích listů betonové směsi při každé dodávce na stavbu.",
        "Zkouška krychelné pevnosti betonu po 7 a 28 dnech od betonáže.",
        "Protokol o vizuální kontrole bednění před zahájením betonáže.",
        "Postup ošetřování betonu po betonáži v závislosti na teplotě.",
        "Evidence klimatických podmínek během betonáže základové desky.",
        "Závěrečné shrnutí kontrolního a zkušebního plánu pro investora.",
        "Příloha č. 1 - vzor protokolu o zkoušce betonu.",
        "Příloha č. 2 - vzor předávacího protokolu výztuže.",
    ]
    (root / "KZP_textova_cast.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")
    (root / "smlouva_o_dilo.txt").write_text("Zhotovitel se zavazuje provést dílo řádně a včas dle této smlouvy o dílo.", encoding="utf-8")
    (root / "pentaflex.txt").write_text("Technický list těsnicího pásu Pentaflex pro pracovní spáru základové desky.", encoding="utf-8")
    state = tmp_path / "state"
    embeddings = BagOfWordsEmbeddings()
    ai_search.sync(root, state / "database" / "project.sqlite3", state / "lance" / "project", embeddings)
    settings = ui.Settings(project_root=str(root), result_count=10)
    return settings, state, embeddings


def test_search_all_merges_multiple_chunks_for_a_real_indexed_document_end_to_end(many_chunk_document_backend):
    """Full stack (sync -> ai_search.search -> search_all's path-merge): the
    12-paragraph document must still be found (unchanged retrieval/ranking,
    per requirement #1) AND its merged quote must contain more than one of
    its own chunks (the fix under test), not just the single top-scored one.
    Deliberately does not assert on any ONE specific paragraph's exact
    presence - which chunk scores highest under the fake bag-of-words
    embedder is a retrieval-ranking detail this fix explicitly leaves
    untouched; test_b above already pins down the exact selection mechanism
    with fully controlled input."""
    settings, state, embeddings = many_chunk_document_backend
    rows = ui.search_all("Jaké doklady musí dodat zhotovitel po betonáži?", settings, state, embeddings, is_question=True)
    kzp_rows = [row for row in rows if row["document"] == "KZP_textova_cast.txt"]
    assert kzp_rows, f"the many-chunk document must still be found: {[r['document'] for r in rows]}"
    quote = kzp_rows[0]["quote"]
    assert ui.QUOTE_MERGE_SEPARATOR in quote, f"expected multiple merged chunks, got: {quote!r}"
    assert "evidence" in kzp_rows[0] and len(kzp_rows[0]["evidence"]) >= 2


def test_search_all_merged_quote_still_respects_the_budget_end_to_end(many_chunk_document_backend):
    settings, state, embeddings = many_chunk_document_backend
    rows = ui.search_all("betonáž KZP", settings, state, embeddings, is_question=False)
    for row in rows:
        assert len(row.get("quote") or "") <= ui.QUOTE_MERGE_MAX_CHARS


def test_search_all_evidence_field_is_additive_and_optional(many_chunk_document_backend):
    """Existing consumers only ever read `quote` (a plain string) - `evidence`
    must be a pure addition, never required, never replacing `quote`."""
    settings, state, embeddings = many_chunk_document_backend
    rows = ui.search_all("betonáž", settings, state, embeddings, is_question=True)
    for row in rows:
        assert isinstance(row["quote"], str)
        if "evidence" in row:
            assert isinstance(row["evidence"], list)
            for item in row["evidence"]:
                assert set(item.keys()) == {"rank", "text"}


def test_context_excerpt_does_not_crash_on_multi_chunk_merged_quote(many_chunk_document_backend):
    """context_excerpt() (app.py's UI preview) must keep working unmodified
    against a longer, separator-joined merged quote."""
    settings, state, embeddings = many_chunk_document_backend
    rows = ui.search_all("betonáž KZP", settings, state, embeddings, is_question=True)
    for row in rows:
        excerpt = ui.context_excerpt(row.get("quote", ""), "betonáž")
        assert isinstance(excerpt, str) and excerpt
