"""Regression tests for the Query Understanding layer (query_expansion.py) and
its opt-in wiring into ai_search.search(expand_query=...).

The load-bearing property is the FIRST test group: with the feature flag off,
every observable output - the FTS5 MATCH expression, the text handed to the
embedder, and the returned rows - must be identical to what the pipeline
produced before this layer existed. Everything else here guards the layer's
stated safety rules (conservative, bounded, emit-directional, never rewrites
the query).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_search
import query_expansion as qe
import ui_services as ui

DIMENSIONS = 16


def _hashed_vector(seed: str) -> list[float]:
    digest = hashlib.sha256(seed.encode()).digest()
    return [(b / 127.5 - 1.0) for b in digest[:DIMENSIONS]]


class RecordingEmbeddings:
    """Bag-of-hashed-tokens fake embedder that also records every text it was
    asked to encode - the only way to assert *what* the vector branch was fed
    (the expansion's second injection point) rather than just what came back."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            self.encoded.append(text)
            vector = [0.0] * DIMENSIONS
            for token in re.findall(r"\w+", text.casefold()):
                vector = [a + b for a, b in zip(vector, _hashed_vector(token[:6]))]
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


DOCUMENTS = {
    "KZP - TEXTOVA CAST.pdf": "Kontrolni a zkusebni bod 08. Kontrola dodacich listu betonove smesi pri kazde dodavce.",
    "prehled ZL GD.xlsx": "Poradove cislo ZL, popis zmeny, odsouhlaseno, castka.",
    "pentaflex-ABS.pdf": "PENTAFLEX ABS je bednici a tesnici prvek pro vodonepropustne konstrukce.",
    "smlouva.pdf": "Zhotovitel se zavazuje provest dilo radne a vcas dle teto smlouvy.",
}


@pytest.fixture
def index(tmp_path):
    """Small real SQLite+LanceDB index built through the production sync path,
    so search() runs against genuine FTS5/LanceDB rather than a stub."""
    root = tmp_path / "projekt"
    root.mkdir()
    for name, body in DOCUMENTS.items():
        (root / name.replace(".pdf", ".txt").replace(".xlsx", ".txt")).write_text(f"{name}\n\n{body}", encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    embeddings = RecordingEmbeddings()
    ai_search.sync(root, db, lance, embeddings)
    embeddings.encoded.clear()
    return db, lance, embeddings


# ---------------------------------------------------------------------------
# 1. Flag off == today's behaviour (the rollback guarantee)
# ---------------------------------------------------------------------------

def test_fts_expression_without_extra_terms_is_unchanged():
    assert ai_search._fts_query_terms("základové desky") == ai_search._fts_query_terms("základové desky", extra_terms=())


def test_search_without_expansion_embeds_the_raw_query_only(index):
    db, lance, embeddings = index
    ai_search.search("Jaké doklady musí dodat zhotovitel po betonáži?", db, lance, embeddings, limit=5)
    assert embeddings.encoded == ["Jaké doklady musí dodat zhotovitel po betonáži?"]


def test_search_results_identical_with_flag_off_and_default(index):
    db, lance, embeddings = index
    query = "Jaké doklady musí dodat zhotovitel po betonáži?"
    default_rows = ai_search.search(query, db, lance, embeddings, limit=5)
    explicit_off = ai_search.search(query, db, lance, embeddings, limit=5, expand_query=False)
    assert [(r["document"], r["score"]) for r in default_rows] == [(r["document"], r["score"]) for r in explicit_off]


def test_trace_records_no_expansion_when_flag_off(index):
    db, lance, embeddings = index
    trace = ai_search.SearchTrace()
    ai_search.search("betonáž", db, lance, embeddings, limit=5, trace=trace)
    assert trace.expansion_terms == []
    assert trace.expansion_matched_rules == []
    assert trace.query_expanded == trace.query_original == "betonáž"
    assert trace.metadata["expand_query"] is False


# ---------------------------------------------------------------------------
# 2. Known term expands / unknown term does not
# ---------------------------------------------------------------------------

def test_known_domain_term_expands():
    expansion = qe.expand_query("Jaké doklady musí dodat zhotovitel po betonáži?")
    assert {rule["key"] for rule in expansion.matched_rules} == {"betonáž", "doklady"}
    assert "dodací list" in expansion.terms


def test_declined_form_triggers_rule_via_prefix_match():
    # "betonáži" (locative) must reach the "betonáž" rule without any stemmer.
    assert qe.expand_query("po betonáži").matched_rules


def test_abbreviation_and_project_alias_expand_the_changelist_query():
    expansion = qe.expand_query("změnový list přístavba zázemí Národního domu")
    assert {rule["key"] for rule in expansion.matched_rules} == {"změnový list", "národní dům"}
    assert "ZL" in expansion.terms and "Garáže NDS" in expansion.terms


def test_unknown_term_is_left_untouched():
    expansion = qe.expand_query("Pentaflex")
    assert expansion.terms == [] and expansion.matched_rules == []
    assert expansion.embedding_text == "Pentaflex"
    assert not expansion


def test_everyday_non_domain_query_is_not_enriched():
    for query in ("ahoj jak se máš", "kdy přijde faktura z minulého týdne odpoledne", ""):
        expansion = qe.expand_query(query)
        if query.startswith("kdy"):
            # "faktura" is a legitimate domain trigger - it may expand, but only
            # via that one rule, not by pulling in unrelated construction areas.
            assert {rule["key"] for rule in expansion.matched_rules} <= {"fakturace"}
        else:
            assert expansion.terms == []


def test_terms_already_present_in_the_query_are_not_re_added():
    expansion = qe.expand_query("kontrolní a zkušební plán KZP")
    assert "kontrolní a zkušební plán" not in expansion.terms
    assert "KZP" not in expansion.terms


# ---------------------------------------------------------------------------
# 3. Bounded and directional
# ---------------------------------------------------------------------------

def test_expansion_never_exceeds_the_term_budget():
    # A query deliberately triggering many rules at once.
    expansion = qe.expand_query("doklady betonáž výztuž bednění změnový list BOZP fakturace předání díla")
    assert len(expansion.terms) <= qe.MAX_EXPANSION_TERMS
    assert len(expansion.terms) == len(set(expansion.terms))


def test_custom_max_terms_is_respected():
    assert len(qe.expand_query("Jaké doklady musí dodat zhotovitel po betonáži?", max_terms=3).terms) == 3


def test_budget_is_shared_across_rules_not_consumed_by_the_first():
    expansion = qe.expand_query("Jaké doklady musí dodat zhotovitel po betonáži?")
    keys = {rule["key"] for rule in expansion.matched_rules if rule["terms"]}
    assert keys == {"betonáž", "doklady"}, "round robin must leave both matched rules represented"


def test_document_terms_are_emit_only_and_never_trigger_a_rule():
    # "dodací list" is a `documents` entry of the betonáž rule. Querying it must
    # NOT drag the query back towards betonáž - that inverted, loose relation is
    # exactly what would flood unrelated queries with noise.
    assert "betonáž" not in {rule["key"] for rule in qe.expand_query("dodací list").matched_rules}


def test_original_query_is_never_modified():
    query = "Jaké doklady musí dodat zhotovitel po betonáži?"
    expansion = qe.expand_query(query)
    assert expansion.original == query
    assert expansion.embedding_text.startswith(query)


# ---------------------------------------------------------------------------
# 4. Wiring into search(): both branches actually receive the expansion
# ---------------------------------------------------------------------------

def test_expansion_terms_are_or_ed_into_the_fts_expression():
    terms = ai_search._fts_query_terms("betonáž", extra_terms=["dodací list", "beton"])
    assert '"dodací list"' in terms, "multi-word terms must stay a phrase, not separate OR'd words"
    assert '"beton"' in terms
    assert terms.startswith(ai_search._fts_query_terms("betonáž")), "original query terms must come first, unchanged"


def test_expansion_terms_with_quotes_cannot_break_the_match_expression():
    terms = ai_search._fts_query_terms("beton", extra_terms=['dodací " list', 'AND OR ""'])
    assert terms.count('"') % 2 == 0
    assert '""' not in terms


def test_search_with_expansion_embeds_query_plus_terms(index):
    db, lance, embeddings = index
    query = "Jaké doklady musí dodat zhotovitel po betonáži?"
    ai_search.search(query, db, lance, embeddings, limit=5, expand_query=True)
    assert len(embeddings.encoded) == 1
    encoded = embeddings.encoded[0]
    assert encoded.startswith(query) and len(encoded) > len(query)
    assert "dodací list" in encoded


def test_search_trace_records_expansion_details(index):
    db, lance, embeddings = index
    trace = ai_search.SearchTrace()
    ai_search.search("změnový list přístavba zázemí Národního domu", db, lance, embeddings, limit=5, trace=trace, expand_query=True)
    assert trace.query_original == "změnový list přístavba zázemí Národního domu"
    assert trace.query_expanded != trace.query_original
    assert "ZL" in trace.expansion_terms
    assert {rule["key"] for rule in trace.expansion_matched_rules} == {"změnový list", "národní dům"}
    assert trace.metadata["expand_query"] is True


def test_expansion_recovers_a_document_the_baseline_misses(index):
    """End-to-end proof on the diagnosed failure pattern: the changelist
    register is named "prehled ZL GD" while the query says "změnový list", so
    the baseline finds nothing lexically; expansion supplies "ZL"/"přehled ZL"."""
    db, lance, embeddings = index
    query = "změnový list"
    baseline_terms = ai_search._fts_query_terms(query)
    expanded_terms = ai_search._fts_query_terms(query, extra_terms=qe.expand_query(query).terms)
    with ai_search.database(db) as con:
        baseline = {cid for (cid,) in con.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank", (baseline_terms,))}
        expanded = {cid for (cid,) in con.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank", (expanded_terms,))}
    assert baseline < expanded, "expansion must strictly widen the BM25 candidate set here"


# ---------------------------------------------------------------------------
# 5. Per-branch A/B control (the two injection points are independently gated)
# ---------------------------------------------------------------------------

def test_expansion_branch_modes_resolve_correctly():
    assert ai_search._expansion_branches(False) == (False, False)
    assert ai_search._expansion_branches(True) == (True, True)
    assert ai_search._expansion_branches("both") == (True, True)
    assert ai_search._expansion_branches("fts") == (True, False)
    assert ai_search._expansion_branches("vector") == (False, True)


def test_invalid_expansion_mode_is_rejected_loudly():
    with pytest.raises(ValueError):
        ai_search._expansion_branches("yes")


def test_fts_only_mode_leaves_the_embedded_text_untouched(index):
    db, lance, embeddings = index
    query = "Jaké doklady musí dodat zhotovitel po betonáži?"
    trace = ai_search.SearchTrace()
    ai_search.search(query, db, lance, embeddings, limit=5, trace=trace, expand_query="fts")
    assert embeddings.encoded == [query], "the vector branch must see the raw query in fts-only mode"
    assert trace.expansion_terms, "the FTS branch must still be widened"
    assert '"dodací list"' in trace.query_terms


def test_vector_only_mode_leaves_the_fts_expression_untouched(index):
    db, lance, embeddings = index
    query = "Jaké doklady musí dodat zhotovitel po betonáži?"
    trace = ai_search.SearchTrace()
    ai_search.search(query, db, lance, embeddings, limit=5, trace=trace, expand_query="vector")
    assert trace.query_terms == ai_search._fts_query_terms(query)
    assert embeddings.encoded[0].startswith(query) and len(embeddings.encoded[0]) > len(query)


def test_search_all_defaults_to_no_expansion(monkeypatch, tmp_path):
    captured = {}

    def fake_search(query, db, lance, embeddings, limit, is_question=False, expand_query=False, **kwargs):
        captured["expand_query"] = expand_query
        return []

    monkeypatch.setattr(ai_search, "search", fake_search)
    settings = ui.Settings(project_root=str(tmp_path))
    state_dir = tmp_path / "state"
    db, _ = ui.state_paths(state_dir, "Dokument")
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"")
    ui.search_all("betonáž", settings, state_dir, RecordingEmbeddings())
    assert captured["expand_query"] is False


# ---------------------------------------------------------------------------
# 6. ABBREVIATION_FILENAME_MATCH_BONUS (2026-08-09 rr-kzp-monolit-feri-01 fix)
#
# `ai_search._abbreviation_filename_needles()` is a pure helper (no I/O, no
# search() call needed) - see its docstring in ai_search.py - so the "which
# category can trigger the bonus" contract is tested directly against it
# first, and only the ranking OUTCOME is tested end-to-end via search().
# ---------------------------------------------------------------------------

def test_abbreviation_filename_needles_none_without_expansion():
    """Requirement: 'bez expansion se bonus nikdy nepoužije'. expand_query=False
    makes search() pass expansion=None into this helper - see search()'s own
    `expansion = query_expansion.expand_query(query) if (...) else None`."""
    assert ai_search._abbreviation_filename_needles(None) == set()


def test_abbreviation_filename_needles_extracts_only_the_abbreviations_category():
    """'kontrolní a zkušební plán' matches the 'kzp' rule via its synonym
    trigger; that rule carries an 'abbreviations' term ("KZP") AND
    'documents' terms ("kontrolní bod", "zkušební plán", "plán kontrol").
    Only "kzp" (casefolded) may end up in the needle set - never the looser
    categories, even though they are present in the SAME
    matched_rules[i]["terms"] list expand_query() returns."""
    expansion = qe.expand_query("kontrolní a zkušební plán")
    assert any(rule["key"] == "kzp" for rule in expansion.matched_rules)
    assert ai_search._abbreviation_filename_needles(expansion) == {"kzp"}


def test_abbreviation_filename_needles_drops_abbreviations_shorter_than_the_minimum():
    """2026-08-09 regression fix: "TP" and "ZL" are 2-char abbreviations that
    double as this corpus's own filename numbering convention ("TP 2.4 - ...",
    "ZL č.001 - ..."), so they false-positive-matched dozens of unrelated
    documents - see ABBREVIATION_FILENAME_MIN_LENGTH's comment. Both must be
    dropped, while the 3-char "KZP" (the case this bonus exists for) is kept."""
    assert len("TP") < ai_search.ABBREVIATION_FILENAME_MIN_LENGTH
    assert len("ZL") < ai_search.ABBREVIATION_FILENAME_MIN_LENGTH
    assert len("KZP") >= ai_search.ABBREVIATION_FILENAME_MIN_LENGTH
    tp_expansion = qe.expand_query("technologický postup")
    zl_expansion = qe.expand_query("změnový list")
    assert any(rule["key"] == "tp" for rule in tp_expansion.matched_rules)
    assert any(rule["key"] == "změnový list" for rule in zl_expansion.matched_rules)
    assert ai_search._abbreviation_filename_needles(tp_expansion) == set()
    assert ai_search._abbreviation_filename_needles(zl_expansion) == set()


def test_abbreviation_filename_needles_empty_for_a_rule_without_an_abbreviation():
    """'betonáž' matches its own rule (key trigger), which has NO
    'abbreviations' category at all - only 'synonyms'/'documents'. The needle
    set must stay empty even though the rule DID match and DID contribute
    terms - proves synonym/document terms can never leak into the filename
    bonus, independent of any implementation-detail filtering."""
    expansion = qe.expand_query("betonáž")
    assert expansion.matched_rules, "sanity: the rule must actually have fired"
    assert "abbreviations" not in qe.DOMAIN_VOCABULARY["betonáž"]
    assert ai_search._abbreviation_filename_needles(expansion) == set()


@pytest.fixture
def kzp_index(tmp_path):
    """Three documents isolating ABBREVIATION_FILENAME_MATCH_BONUS from the
    general FILENAME_MATCH_BONUS: none of their names contain the query
    verbatim (so `filename_match` never fires), and none of their bodies
    contain "kontrolní"/"zkušební"/"plán" (so a baseline, non-expanded query
    cannot find them lexically either - mirrors the real rr-kzp-monolit-feri-01
    pattern of a document that ONLY ever spells the concept as "KZP").

    - "KZP silny.txt" and "Bez zkratky.txt" share the SAME body WORDS (only
      trailing punctuation differs, so sync()'s content-hash dedup does not
      collapse them into one document, while RecordingEmbeddings - word-token
      based, see its docstring above - still gives them the exact same
      semantic_similarity) -> tied BM25/vector signal by construction; "KZP"
      in the name is the only difference, isolating exactly what the bonus is
      meant to reward.
    - "KZP slaby.txt" also carries "KZP" in its name (same bonus-eligibility
      as "KZP silny.txt") but unrelated body content, to prove the bonus
      does not paper over a genuine relevance gap between two equally-
      eligible documents (requirement 4)."""
    root = tmp_path / "projekt"
    root.mkdir()
    shared_body = "Technologický postup zajištění stavební jámy mikrozápory"
    docs = {
        "KZP silny.txt": shared_body + " .",
        "Bez zkratky.txt": shared_body + " !",
        "KZP slaby.txt": "Nesouvisející text o fakturaci a platebních podmínkách.",
    }
    for name, body in docs.items():
        (root / name).write_text(body, encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    embeddings = RecordingEmbeddings()
    ai_search.sync(root, db, lance, embeddings)
    return db, lance, embeddings


def test_abbreviation_filename_match_favours_the_document_naming_the_abbreviation(kzp_index):
    """Requirement 1: with expand_query="fts", "KZP silny.txt" must outrank
    its content-identical twin "Bez zkratky.txt" - the only difference
    between them is the "KZP" abbreviation in the filename."""
    db, lance, embeddings = kzp_index
    query = "kontrolní a zkušební plán"
    rows = ai_search.search(query, db, lance, embeddings, limit=10, expand_query="fts")
    documents = [r["document"] for r in rows]
    assert documents.index("KZP silny.txt") < documents.index("Bez zkratky.txt")


def test_abbreviation_filename_match_requires_expansion_to_be_active(kzp_index):
    """Requirement 2: with expand_query=False (default), "KZP silny.txt" gets
    no help from its filename. Two documents with the same semantic_similarity
    still get very slightly different scores from RRF rank-tie-breaking
    (arbitrary iteration order, unrelated to this fix) - so the assertion is a
    delta well below the 0.02 bonus, not exact equality, to avoid asserting
    away that pre-existing tie-breaking noise."""
    db, lance, embeddings = kzp_index
    query = "kontrolní a zkušební plán"
    rows = ai_search.search(query, db, lance, embeddings, limit=10, expand_query=False)
    by_document = {r["document"]: r["score"] for r in rows}
    delta = by_document["KZP silny.txt"] - by_document["Bez zkratky.txt"]
    assert abs(delta) < ai_search.ABBREVIATION_FILENAME_MATCH_BONUS / 2, (
        f"filename must grant no meaningful advantage when expand_query=False (delta={delta})"
    )


def test_two_documents_sharing_the_abbreviation_keep_their_relative_order(kzp_index):
    """Requirement 4: "KZP silny.txt" and "KZP slaby.txt" both carry "KZP" in
    their name and therefore both receive the SAME flat bonus - it must not
    reorder them relative to each other. "KZP silny.txt" is already the more
    relevant of the two by construction (shares no content with the unrelated
    "KZP slaby.txt"), so it must stay ahead with or without expansion."""
    db, lance, embeddings = kzp_index
    query = "kontrolní a zkušební plán"
    for expand_query in (False, "fts"):
        rows = ai_search.search(query, db, lance, embeddings, limit=10, expand_query=expand_query)
        documents = [r["document"] for r in rows]
        assert documents.index("KZP silny.txt") < documents.index("KZP slaby.txt"), f"order changed for expand_query={expand_query!r}"


# ---------------------------------------------------------------------------
# Query Expansion 2.0 MVP - floor notation + bludné proudy → D.1.4.j
# (rr-bp-vyvody-3pp-01). No scoring/RRF changes; dictionary + normalizer only.
# ---------------------------------------------------------------------------

def test_bludne_proudy_emits_project_part_code_not_short_abbreviation():
    """Human phrase must bridge to the project part code used in paths/names.
    Must NOT emit bare "BP" (2-char abbr false-positive class)."""
    expansion = qe.expand_query("bludné proudy")
    assert "D.1.4.j" in expansion.terms
    assert "BP" not in expansion.terms
    assert all(qe._fold(term) != "bp" for term in expansion.terms)
    assert any(rule["key"] == "bludné proudy" for rule in expansion.matched_rules)


def test_floor_pp_normalization_emits_dotted_form_from_compact_query():
    """'3PP' in the query must produce the dotted drawing form '3.PP' so FTS
    can hit body/filename text tokenized as ('3','PP')."""
    expansion = qe.expand_query("půdorys 3PP bludné proudy")
    assert "3.PP" in expansion.terms
    assert "D.1.4.j" in expansion.terms
    assert len(expansion.terms) <= qe.MAX_EXPANSION_TERMS


def test_floor_pp_normalization_covers_spaced_and_dotted_inputs():
    """Dotted and spaced forms already share FTS tokens ('3','PP'), so the
    bridge they still need is the compact single-token '3PP' used in some
    filenames/queries. Compact '3PP' is covered by the previous test."""
    assert "3PP" in qe.expand_query("půdorys 3.PP").terms
    assert "3PP" in qe.expand_query("půdorys 3 PP").terms


def test_bare_bp_does_not_expand():
    """A lone 'BP' query must not fire the bludné proudy rule or emit D.1.4.j -
    short abbreviations are too ambiguous without the full human phrase."""
    expansion = qe.expand_query("BP")
    assert expansion.terms == []
    assert expansion.matched_rules == []
    assert not qe.expand_query("bp")


def test_qe2_mvp_does_not_worsen_short_abbreviation_filename_gating():
    """TP/ZL stay below ABBREVIATION_FILENAME_MIN_LENGTH; KD likewise must not
    become a filename needle. KZP (≥3 chars) remains eligible."""
    tp = qe.expand_query("technologický postup")
    zl = qe.expand_query("změnový list")
    kd = qe.expand_query("kontrolní den")
    kzp = qe.expand_query("kontrolní a zkušební plán")
    assert ai_search._abbreviation_filename_needles(tp) == set()
    assert ai_search._abbreviation_filename_needles(zl) == set()
    assert ai_search._abbreviation_filename_needles(kd) == set()
    assert ai_search._abbreviation_filename_needles(kzp) == {"kzp"}
    # bludné proudy emits a code, not an abbreviations-category term
    bp = qe.expand_query("bludné proudy")
    assert ai_search._abbreviation_filename_needles(bp) == set()


# ---------------------------------------------------------------------------
# Surface preparation: brokování/brokovat → otryskání (+ broušení)
# (rr-brokovani-zakladova-deska-3pp-01). QE-only; no scoring/RRF changes.
# ---------------------------------------------------------------------------

def test_brokovat_zakladova_deska_emits_otryskani_and_keeps_floor_normalization():
    expansion = qe.expand_query("bude se brokovat základová deska 3PP")
    assert "otryskání" in expansion.terms
    assert "broušení" in expansion.terms
    assert "3.PP" in expansion.terms
    assert any(rule["key"] == "brokování" for rule in expansion.matched_rules)
    assert len(expansion.terms) <= qe.MAX_EXPANSION_TERMS


def test_brokovani_podkladu_emits_otryskani():
    expansion = qe.expand_query("brokování podkladu")
    assert "otryskání" in expansion.terms
    assert "broušení" in expansion.terms


def test_brokovat_beton_emits_otryskani():
    expansion = qe.expand_query("brokovat beton")
    assert "otryskání" in expansion.terms
    assert "broušení" in expansion.terms


def test_brokovani_morphology_covers_instrumental_without_stemmer():
    """Prefix match on the rule key covers common declension (brokováním).
    Participles like brokovaný are out of scope (no Czech stemmer)."""
    assert "otryskání" in qe.expand_query("brokováním").terms
    assert "otryskání" not in qe.expand_query("brokovaný").terms


def test_surface_prep_rule_does_not_change_existing_abbreviation_expansions():
    """BP/TP/ZL/KD/KZP expansion contracts stay intact beside the new rule."""
    bp = qe.expand_query("bludné proudy")
    assert "D.1.4.j" in bp.terms
    assert "BP" not in bp.terms

    tp = qe.expand_query("technologický postup")
    assert "TP" in tp.terms

    zl = qe.expand_query("změnový list")
    assert "ZL" in zl.terms

    kd = qe.expand_query("kontrolní den")
    assert "KD" in kd.terms

    kzp = qe.expand_query("kontrolní a zkušební plán")
    assert "KZP" in kzp.terms

    # Bare "BP" stays inert (not an abbreviations trigger). TP/ZL/KD remain
    # legitimate abbreviation keys and may still expand - that is unchanged.
    assert qe.expand_query("BP").terms == []
