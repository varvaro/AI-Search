"""PR9.4.1 — Metadata-aware Phase-3 reranking unit tests.

Covers: flag default/OFF identity, the safe date whitelist parser (accept /
reject shapes), token overlap, discriminator matching (floor / drawing code /
alnum id), the composed score, and a couple of synthetic end-to-end ranking
checks through ai_search.search().
"""
from __future__ import annotations

from datetime import date

import pytest

import ai_search
import ai_search_config
import metadata_rerank as mr


class _FakeEmbeddings:
    name = "fake"

    def encode(self, texts, **kw):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "Projekt"
    root.mkdir()
    (root / "alpha.txt").write_text("ALPHA unikátní fulltextový výraz.")
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings


# --- flag / identity ----------------------------------------------------------

def test_flag_default_is_off():
    assert ai_search_config.METADATA_RERANK_ENABLED is False
    assert ai_search.METADATA_RERANK_ENABLED is False


def test_flag_off_search_identical(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "METADATA_RERANK_ENABLED", False)
    a = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    b = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    assert [r["score"] for r in a] == [r["score"] for r in b]
    assert "metadata_rerank" not in a[0]["match"]
    assert "metadata_rerank_bonus" not in a[0]["match"]


def test_empty_query_or_document_is_a_no_op():
    detail = mr.compute_metadata_score("", "some.pdf", "/a/b/some.pdf")
    assert detail.bonus == 0.0
    assert detail is mr._EMPTY_DETAIL

    detail2 = mr.compute_metadata_score("schéma vyztužení", "", "")
    assert detail2.bonus == 0.0


# --- safe date whitelist -------------------------------------------------------

def test_drawing_code_is_never_read_as_a_date():
    assert mr.parse_safe_dates("D.1.2.06 - schéma vyztužení základové desky.pdf") == ()
    assert mr.parse_safe_dates("D.1.2.103 - HORNÍ VÝZTUŽ ZÁKLADOVÉ DESKY - ao.pdf") == ()
    assert mr.parse_safe_dates("D.1.4.c.4_ND-Smíchov-Garáže-ZTI-103-Půdorys 1.PP.pdf") == ()


def test_alnum_id_digits_are_never_read_as_a_date():
    # An id's digits (e.g. ...251110) must never surface as a parsed date -
    # they are directly preceded by a letter, not isolated by any boundary.
    assert mr.parse_safe_dates("36_monolit_FERI_NOT251110") == ()
    assert mr.parse_safe_dates("NOT251110_LOI_monolit_Feri_signed.pdf") == ()


def test_iso_and_compact_dates_accepted():
    assert mr.parse_safe_dates("Příloha č.2_2025-11-21_Rozpočet Garáže ND.xlsx") == (
        date(2025, 11, 21),
    )
    assert mr.parse_safe_dates("Garaze_NDS_2023_08_17-D_01_TZ.pdf") == (date(2023, 8, 17),)


def test_dmy_dot_date_accepted_when_isolated():
    assert mr.parse_safe_dates("Smlouva o dílo SIS Systémy 24.2.2026.pdf") == (
        date(2026, 2, 24),
    )
    # Short 2-digit year, isolated by an underscore rather than a dot.
    assert mr.parse_safe_dates("HMG_akt_4.08.2026.pdf") == (date(2026, 8, 4),)


def test_dmy_dot_date_rejected_when_glued_to_a_preceding_dot():
    # Documented trade-off: a date immediately preceded by a dot is
    # indistinguishable from a dotted drawing/revision code, so it is not
    # parsed. This module never needs it for recency (only exact match).
    assert mr.parse_safe_dates("akt.11.12.25") == ()


def test_short_year_window_rejects_implausible_years():
    # yy=99 -> year 2099 is outside the plausible 20-35 construction window
    # for the *2-digit* shapes (compact YMD / short DMY); must not silently
    # wrap or produce a nonsensical date.
    assert mr.parse_safe_dates("991231") == ()
    assert mr.parse_safe_dates("31.12.99") == ()


def test_bare_short_numbers_are_never_dates():
    assert mr.parse_safe_dates("D.1.2.06") == ()
    assert mr.parse_safe_dates("1.2.103") == ()
    assert mr.parse_safe_dates("103") == ()


# --- discriminators: floor -----------------------------------------------------

def test_floor_discriminator_hit_and_mismatch():
    q = mr.extract_query_metadata("schéma vyztužení 1.PP")
    floors_q = {d.canonical for d in q.discriminators if d.kind == "floor"}
    assert floors_q == {"1.pp"}

    same = mr.extract_document_metadata("D.1.2.11 - schéma vyztužení 1.PP.pdf", "/x/y.pdf")
    other = mr.extract_document_metadata("D.1.2.09 - schéma vyztužení 2.PP.pdf", "/x/y.pdf")
    floors_same = {d.canonical for d in same.discriminators if d.kind == "floor"}
    floors_other = {d.canonical for d in other.discriminators if d.kind == "floor"}
    assert floors_same == {"1.pp"}
    assert floors_other == {"2.pp"}


def test_floor_does_not_confuse_np_with_pp():
    q = mr.extract_query_metadata("3.NP")
    floors = {d.canonical for d in q.discriminators if d.kind == "floor"}
    assert floors == {"3.np"}
    doc = mr.extract_document_metadata("Půdorys 3.PP.pdf", "")
    doc_floors = {d.canonical for d in doc.discriminators if d.kind == "floor"}
    assert "3.pp" in doc_floors
    assert "3.np" not in doc_floors


# --- discriminators: drawing code ----------------------------------------------

def test_drawing_code_exact_not_prefix():
    d1 = mr.extract_document_metadata("D.1.2.06 - schéma.pdf", "")
    d2 = mr.extract_document_metadata("D.1.2.103 - jiný výkres.pdf", "")
    codes1 = {d.canonical for d in d1.discriminators if d.kind == "drawing_code"}
    codes2 = {d.canonical for d in d2.discriminators if d.kind == "drawing_code"}
    assert codes1 == {"d.1.2.06"}
    assert codes2 == {"d.1.2.103"}
    assert codes1 != codes2


def test_drawing_code_letter_segment():
    d = mr.extract_document_metadata("D.1.4.c.4_ND-Smíchov-Garáže-ZTI-103.pdf", "")
    codes = {c.canonical for c in d.discriminators if c.kind == "drawing_code"}
    assert "d.1.4.c.4" in codes


def test_drawing_code_does_not_absorb_extension():
    d = mr.extract_document_metadata("D.1.2.06.pdf", "")
    codes = [c.canonical for c in d.discriminators if c.kind == "drawing_code"]
    assert codes == ["d.1.2.06"]


# --- discriminators: alnum id ---------------------------------------------------

def test_alnum_id_extraction():
    q = mr.extract_query_metadata("najdi montážní návod Pentaflex KB80")
    ids = {d.canonical for d in q.discriminators if d.kind == "alnum_id"}
    assert "kb80" in ids


def test_alnum_id_is_not_double_counted_as_content_token():
    q = mr.extract_query_metadata("montážní návod KB80")
    assert "kb80" not in q.content_tokens


def test_alnum_id_exact_not_substring():
    q = mr.extract_query_metadata("KB80")
    doc = mr.extract_document_metadata("KB801 varianta.pdf", "")
    ids_q = {d.canonical for d in q.discriminators if d.kind == "alnum_id"}
    ids_doc = {d.canonical for d in doc.discriminators if d.kind == "alnum_id"}
    assert ids_q == {"kb80"}
    assert "kb80" not in ids_doc  # tokenized as one longer token, not a substring hit


# --- token overlap --------------------------------------------------------------

def test_stopword_only_query_yields_no_bonus():
    detail = mr.compute_metadata_score(
        "kdo je dodavatel", "nejaky_soubor.pdf", "/a/b/nejaky_soubor.pdf"
    )
    assert detail.bonus == 0.0


def test_single_shared_token_is_too_weak_alone():
    detail = mr.compute_metadata_score(
        "najdi rozpočet garáží", "Poptávka_Přístavba garáží NDS_monolity.xlsx", "/x/y.xlsx"
    )
    # only "garazi" overlaps (folded) - below MIN_CONTENT_TOKENS_FOR_BONUS
    assert not detail.overlap_name
    assert detail.bonus == 0.0


def test_two_shared_tokens_in_name_score_above_threshold():
    detail = mr.compute_metadata_score(
        "technická zpráva spodní stavba",
        "Technická zpráva - spodní stavba.pdf",
        "/proj/D12_Statika/Technická zpráva - spodní stavba.pdf",
    )
    assert len(detail.overlap_name) >= 2
    assert detail.bonus > 0
    assert detail.bonus <= mr.TOKEN_OVERLAP_CAP


def test_path_only_overlap_scores_less_than_name_overlap():
    name_detail = mr.compute_metadata_score(
        "kontrolní den zápis", "kontrolní den zápis.pdf", "/x/y.pdf"
    )
    path_detail = mr.compute_metadata_score(
        "kontrolní den zápis", "jiny_soubor.pdf", "/kontrolní/den/zápis/y.pdf"
    )
    assert name_detail.bonus > path_detail.bonus > 0


def test_skip_token_overlap_avoids_double_count_with_filename_match():
    normal = mr.compute_metadata_score(
        "technická zpráva spodní stavba",
        "Technická zpráva - spodní stavba.pdf",
        "/x/y.pdf",
    )
    skipped = mr.compute_metadata_score(
        "technická zpráva spodní stavba",
        "Technická zpráva - spodní stavba.pdf",
        "/x/y.pdf",
        skip_token_overlap=True,
    )
    assert normal.bonus > 0
    assert skipped.bonus == 0.0
    assert skipped.overlap_name == ()


def test_unicode_folding_diacritics():
    detail = mr.compute_metadata_score(
        "výztuž základové desky", "vyztuz zakladove desky.pdf", "/x/y.pdf"
    )
    assert len(detail.overlap_name) >= 2


# --- discriminator scoring: cap / mismatch / date is match-only ---------------

def test_discriminator_hit_and_mismatch_scoring():
    hit = mr.compute_metadata_score("1.PP schéma", "schéma 1.PP.pdf", "/x/y.pdf")
    mismatch = mr.compute_metadata_score("1.PP schéma", "schéma 2.PP.pdf", "/x/y.pdf")
    assert hit.discriminator_hits
    assert mismatch.discriminator_mismatches
    assert hit.bonus > mismatch.bonus


def test_date_mismatch_never_penalizes():
    # Document has an unrelated date; query also names a date. No match -> 0
    # contribution from dates specifically (never negative).
    detail = mr.compute_metadata_score(
        "rozpočet 2025-11-21", "Rozpočet 2025-05-14.xlsx", "/x/y.xlsx"
    )
    assert not any(m.startswith("iso_date:") for m in detail.discriminator_mismatches)


def test_date_exact_match_is_a_hit():
    detail = mr.compute_metadata_score(
        "rozpočet 2025-11-21", "Příloha č.2_2025-11-21_Rozpočet.xlsx", "/x/y.xlsx"
    )
    assert any(h.startswith("iso_date:2025-11-21") for h in detail.discriminator_hits)
    assert detail.bonus > 0


def test_discriminator_cap_and_floor():
    # Three independent floor-style mismatches (synthetic) must clamp at the
    # floor, not stack unboundedly negative.
    many_mismatch_query = "1.PP 2.PP 3.PP"
    doc_name = "4.PP 5.PP 6.PP.pdf"
    detail = mr.compute_metadata_score(many_mismatch_query, doc_name, "")
    assert detail.bonus >= mr.DISCRIMINATOR_FLOOR - 1e-9
    assert detail.bonus == pytest.approx(mr.DISCRIMINATOR_FLOOR)


# --- no chunk-body reads / no hardcoded project values -------------------------

def _source_without_comments_and_docstrings(module) -> str:
    """Strip '#' comments and the module's own leading docstring.

    Good enough for these guard tests (not a general Python tokenizer): the
    invariants below care about actual code (dict/attr access, string
    literals used as values), not prose that *describes* the invariant.
    """
    import inspect

    src = inspect.getsource(module)
    doc = module.__doc__ or ""
    if doc:
        src = src.replace(doc, "", 1)
    lines = []
    for line in src.splitlines():
        stripped = line.split("#", 1)[0]
        lines.append(stripped)
    return "\n".join(lines)


def test_module_never_reads_heading_or_quote():
    code = _source_without_comments_and_docstrings(mr)
    assert "heading" not in code
    assert "quote" not in code


def test_module_has_no_hardcoded_project_values():
    code = mr.fold(_source_without_comments_and_docstrings(mr)).replace(" ", "")
    for forbidden in ("feri", "illichman", "stafitech", "safetypeak", "not250039", "not251110", "cbs02"):
        assert forbidden not in code, f"unexpected project-specific literal: {forbidden}"


# --- synthetic end-to-end ranking (through ai_search.search) -------------------

def test_flag_on_no_signal_query_keeps_score_identical(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "METADATA_RERANK_ENABLED", False)
    base = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    monkeypatch.setattr(ai_search, "METADATA_RERANK_ENABLED", True)
    cand = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    # "ALPHA" already triggers the verbatim FILENAME_MATCH_BONUS against
    # alpha.txt (skip_token_overlap=True) and carries no discriminator, so
    # turning the flag on must not change anything for this query.
    assert [r["score"] for r in base] == [r["score"] for r in cand]


def test_floor_discriminator_reorders_synthetic_pair(backend, monkeypatch):
    # The discriminator only ever reads document name/path (never chunk
    # content, see the module's "Does NOT read heading/quote" contract), so
    # the floor marker must live in the filename for this to exercise it.
    root, state, embeddings = backend
    (root / "schema vyztuzeni 2.PP.txt").write_text(
        "schéma vyztužení základové desky obecný technický popis konstrukce"
    )
    (root / "schema vyztuzeni 1.PP.txt").write_text(
        "schéma vyztužení základové desky obecný technický popis konstrukce"
    )
    ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)

    q = "schéma vyztužení 1.PP"
    monkeypatch.setattr(ai_search, "METADATA_RERANK_ENABLED", False)
    base = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)
    monkeypatch.setattr(ai_search, "METADATA_RERANK_ENABLED", True)
    cand = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings, limit=5)

    def rank_of(rows, name):
        for i, r in enumerate(rows, 1):
            if name in (r.get("document") or ""):
                return i
        return 99

    assert rank_of(cand, "1.PP") <= rank_of(base, "1.PP")
    assert cand[0]["match"].get("metadata_rerank_bonus", 0) != 0 or any(
        r["match"].get("metadata_rerank_bonus", 0) for r in cand
    )
