"""PR9.4.3 — Revision-safe content dedup unit tests.

Content-near-duplicates still collapse, except when both filenames carry
distinct safe dates. Exact-normalized duplicates always collapse. No
newer/final/OLD preference; output order stays score order.
"""
from __future__ import annotations

import inspect

import pytest

import metadata_rerank as mr
import ui_services as ui


NEAR_DUP_A = (
    "sheet list1 rozpocet garaze nd cislo n datum zadavatel systemy "
    "nedilna soucast smlouvy polozkovy rozpoctovy vypis konstrukce "
    "monolit zakladova deska steny stropy"
)
NEAR_DUP_B = (
    "sheet list1 rozpocet garaze nd cislo n datum zadavatel systemy "
    "nedilna soucast smlouvy polozkovy rozpoctovy vypis konstrukce "
    "monolit zakladova deska steny preklady"
)


def _sim() -> float:
    return ui._content_similarity(
        ui._normalize_chunk_text(NEAR_DUP_A),
        ui._normalize_chunk_text(NEAR_DUP_B),
    )


def test_near_dup_quotes_are_between_threshold_and_one():
    sim = _sim()
    assert ui.CONTENT_DUPLICATE_THRESHOLD < sim < 1.0


def test_distinct_filename_dates_keep_near_duplicates():
    rows = [
        {"document": "Budget_2025-05-14.xlsx", "path": "/old/Budget_2025-05-14.xlsx",
         "quote": NEAR_DUP_A, "score": 0.09},
        {"document": "Budget_2025-11-21.xlsx", "path": "/cur/Budget_2025-11-21.xlsx",
         "quote": NEAR_DUP_B, "score": 0.08},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == [
        "Budget_2025-05-14.xlsx",
        "Budget_2025-11-21.xlsx",
    ]


def test_same_filename_date_still_dedups():
    rows = [
        {"document": "Budget_2025-11-21_A.xlsx", "path": "/a/Budget_2025-11-21_A.xlsx",
         "quote": NEAR_DUP_A, "score": 0.09},
        {"document": "Budget_2025-11-21_B.xlsx", "path": "/b/Budget_2025-11-21_B.xlsx",
         "quote": NEAR_DUP_B, "score": 0.08},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == ["Budget_2025-11-21_A.xlsx"]


def test_date_on_only_one_side_still_dedups():
    rows = [
        {"document": "Budget_2025-11-21.xlsx", "path": "/a/Budget_2025-11-21.xlsx",
         "quote": NEAR_DUP_A, "score": 0.09},
        {"document": "Budget.xlsx", "path": "/b/Budget.xlsx",
         "quote": NEAR_DUP_B, "score": 0.08},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == ["Budget_2025-11-21.xlsx"]


def test_no_dates_still_dedups():
    rows = [
        {"document": "Budget_A.xlsx", "path": "/a/Budget_A.xlsx",
         "quote": NEAR_DUP_A, "score": 0.09},
        {"document": "Budget_B.xlsx", "path": "/b/Budget_B.xlsx",
         "quote": NEAR_DUP_B, "score": 0.08},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == ["Budget_A.xlsx"]


def test_exact_normalized_content_dedups_even_with_distinct_dates():
    """Exact-normalized quotes carry no extra information; dated filenames
    must not keep a second copy of the same text (KD-style boilerplate)."""
    quote = "Základovou   desku."
    rows = [
        {"document": "KD_2025-05-14.pdf", "path": "/a/KD_2025-05-14.pdf",
         "quote": quote, "score": 0.09},
        {"document": "KD_2025-11-21.pdf", "path": "/b/KD_2025-11-21.pdf",
         "quote": "základovou desku", "score": 0.08},
    ]
    assert ui._normalize_chunk_text(rows[0]["quote"]) == ui._normalize_chunk_text(rows[1]["quote"])
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == ["KD_2025-05-14.pdf"]


def test_drawing_codes_are_not_dates():
    assert mr.parse_safe_dates("D.1.2.06 - schema.pdf") == ()
    assert mr.parse_safe_dates("D.1.2.07 - schema.pdf") == ()
    rows = [
        {"document": "D.1.2.06 - schema.pdf", "path": "/a/D.1.2.06 - schema.pdf",
         "quote": NEAR_DUP_A, "score": 0.09},
        {"document": "D.1.2.07 - schema.pdf", "path": "/b/D.1.2.07 - schema.pdf",
         "quote": NEAR_DUP_B, "score": 0.08},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == ["D.1.2.06 - schema.pdf"]
    assert ui._should_keep_near_duplicate_due_to_distinct_dates(rows[1], rows[0]) is False


def test_not_identifiers_are_not_dates():
    assert mr.parse_safe_dates("NOT251110_draft.pdf") == ()
    assert mr.parse_safe_dates("NOT260101_draft.pdf") == ()
    rows = [
        {"document": "NOT251110_draft.pdf", "path": "/a/NOT251110_draft.pdf",
         "quote": NEAR_DUP_A, "score": 0.09},
        {"document": "NOT260101_draft.pdf", "path": "/b/NOT260101_draft.pdf",
         "quote": NEAR_DUP_B, "score": 0.08},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == ["NOT251110_draft.pdf"]


def test_kd_boilerplate_without_dated_filename_still_dedups():
    rows = [
        {"document": "kontrolni_den_52.pdf", "path": "/kd/52/kontrolni_den_52.pdf",
         "quote": "Základovou desku.", "score": 0.05},
        {"document": "kontrolni_den_53.pdf", "path": "/kd/53/kontrolni_den_53.pdf",
         "quote": "základovou desku", "score": 0.04},
        {"document": "jiny_protokol.pdf", "path": "/other/jiny_protokol.pdf",
         "quote": "Zcela odlišný obsah o výztuži a kontrole armatury.", "score": 0.03},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["document"] for r in kept] == ["kontrolni_den_52.pdf", "jiny_protokol.pdf"]


def test_exception_does_not_reorder_by_date():
    rows = [
        {"document": "Budget_2025-05-14.xlsx", "path": "/old/Budget_2025-05-14.xlsx",
         "quote": NEAR_DUP_A, "score": 0.09},
        {"document": "Budget_2025-11-21.xlsx", "path": "/cur/Budget_2025-11-21.xlsx",
         "quote": NEAR_DUP_B, "score": 0.08},
    ]
    kept = ui.deduplicate_by_content(rows)
    assert [r["score"] for r in kept] == [0.09, 0.08]
    assert kept[0]["document"].endswith("2025-05-14.xlsx")


def test_helper_is_filename_only():
    older = {
        "document": "Budget.xlsx",
        "path": "/archive/2025-05-14/Budget.xlsx",
        "quote": NEAR_DUP_A,
    }
    newer = {
        "document": "Budget.xlsx",
        "path": "/current/2025-11-21/Budget.xlsx",
        "quote": NEAR_DUP_B,
    }
    assert ui._filename_safe_dates(older) == frozenset()
    assert ui._should_keep_near_duplicate_due_to_distinct_dates(newer, older) is False


def test_iso_and_dmy_filename_dates_are_recognized():
    assert mr.parse_safe_dates("Budget_2025-11-21.xlsx")
    assert mr.parse_safe_dates("HMG_akt_4.08.2026.pdf")


def test_new_helpers_have_no_hardcoded_project_values():
    src = (
        inspect.getsource(ui._filename_safe_dates)
        + inspect.getsource(ui._should_keep_near_duplicate_due_to_distinct_dates)
        + inspect.getsource(ui.deduplicate_by_content)
    )
    code = mr.fold(src).replace(" ", "")
    for forbidden in (
        "feri", "illichman", "stafitech", "safetypeak",
        "not250039", "not251110", "cbs02", "smichov",
    ):
        assert forbidden not in code, f"unexpected project-specific literal: {forbidden}"
