"""PR7.5.2 — SAT Human Verification Protocol.

This module only re-projects facts the acceptance dataset already declares
into a sign-off sheet. These tests check the projection is lossless (every
case survives, in order, with the fields it actually has) and that the one
derived column (review_status) follows the documented mapping rather than
inventing a verdict.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import sat_protocol  # noqa: E402
from benchmark.dataset.schema import (  # noqa: E402
    CRITICAL_CRITICALITY_NAMES,
    DATASET_DIR,
    BenchmarkCase,
    load_dataset,
)

NDS_DATASET = DATASET_DIR / "acceptance_nds_smichov.jsonl"


@pytest.fixture(scope="module")
def cases() -> list[BenchmarkCase]:
    return load_dataset(NDS_DATASET)


@pytest.fixture(scope="module")
def rows(cases) -> list[sat_protocol.SatRow]:
    return sat_protocol.build_sat_table(cases)


# ---------------------------------------------------------------------------
# Row count / no case disappears / order preserved
# ---------------------------------------------------------------------------

def test_row_count_matches_case_count(cases, rows):
    assert len(rows) == len(cases)
    assert len(rows) >= 34


def test_no_case_disappears(cases, rows):
    assert {r.case_id for r in rows} == {c.id for c in cases}
    assert len(rows) == len({r.case_id for r in rows}), "duplicate case_id in SAT table"


def test_dataset_order_is_preserved(cases, rows):
    assert [r.case_id for r in rows] == [c.id for c in cases]


def test_load_cases_wrapper_matches_direct_load(cases):
    assert [c.id for c in sat_protocol.load_cases(NDS_DATASET)] == [c.id for c in cases]


# ---------------------------------------------------------------------------
# expert_required taken from the dataset, not invented
# ---------------------------------------------------------------------------

def test_expert_required_is_taken_from_dataset_criticality(cases, rows):
    by_id = {c.id: c for c in cases}
    for row in rows:
        case = by_id[row.case_id]
        expected = case.criticality in CRITICAL_CRITICALITY_NAMES
        assert row.expert_required is expected, row.case_id
        assert row.criticality == case.criticality


def test_at_least_one_row_requires_an_expert(rows):
    assert any(r.expert_required for r in rows)


# ---------------------------------------------------------------------------
# Critical cases must have room for expert sign-off
# ---------------------------------------------------------------------------

def test_critical_unverified_rows_are_open_with_empty_reviewer_slot(cases, rows):
    """A critical case that has not been human-verified must present an empty,
    writable sign-off slot - not a pre-filled or auto-approved row. This is
    what "prostor pro expert potvrzení" means: the columns exist and are
    blank, waiting for a named person."""
    by_id = {c.id: c for c in cases}
    for row in rows:
        case = by_id[row.case_id]
        if row.expert_required and not case.human_verified:
            assert row.review_status == sat_protocol.REVIEW_STATUS_OPEN, row.case_id
            assert row.reviewer_name == "", row.case_id
            assert row.review_date == "", row.case_id
            assert row.review_comment == "", row.case_id


def test_every_expert_required_row_declares_a_verification_method(rows):
    for row in rows:
        if row.expert_required:
            assert row.verification_method, row.case_id


def test_current_dataset_has_no_verified_cases_yet(rows):
    """acceptance_nds_smichov.jsonl is entirely human_verified=False today
    (PR7.5.1) - the protocol must reflect that honestly, not report progress
    that has not happened."""
    assert all(not r.human_verified for r in rows)
    assert all(r.review_status == sat_protocol.REVIEW_STATUS_OPEN for r in rows)


# ---------------------------------------------------------------------------
# review_status derivation
# ---------------------------------------------------------------------------

def _case(**overrides) -> BenchmarkCase:
    data = {"id": "x", "query": "q", "expected_document": ["a"]}
    data.update(overrides)
    return BenchmarkCase.from_dict(data)


def test_review_status_open_when_not_human_verified():
    assert sat_protocol.derive_review_status(_case(human_verified=False)) == "OPEN"


def test_review_status_verified_requires_ground_truth_verified():
    case = _case(human_verified=True, ground_truth_status="verified",
                 verified_by="M. V.", verification_date="2026-08-12")
    assert sat_protocol.derive_review_status(case) == "VERIFIED"


def test_review_status_found_not_verified_when_reviewed_but_not_signed_off():
    case = _case(human_verified=True, ground_truth_status="needs_review",
                 verified_by="M. V.", verification_date="2026-08-12")
    assert sat_protocol.derive_review_status(case) == "FOUND_NOT_VERIFIED"


def test_review_status_failed_when_reviewer_rejects_ground_truth():
    case = _case(human_verified=True, ground_truth_status="unverified",
                 verified_by="M. V.", verification_date="2026-08-12")
    assert sat_protocol.derive_review_status(case) == "FAILED"


def test_all_four_statuses_are_the_documented_enum():
    assert sat_protocol.REVIEW_STATUSES == {"OPEN", "VERIFIED", "FAILED", "FOUND_NOT_VERIFIED"}


# ---------------------------------------------------------------------------
# Field fidelity — case_id -> query -> expected_documents -> verification_method -> criticality
# ---------------------------------------------------------------------------

def test_row_fields_trace_back_to_the_exact_case(cases, rows):
    by_id = {c.id: c for c in cases}
    for row in rows:
        case = by_id[row.case_id]
        assert row.query == case.question
        assert row.category == case.category
        assert row.verification_method == case.verification_method
        assert row.reviewer_name == case.verified_by
        assert row.review_date == case.verification_date
        for fragment in case.expected_documents:
            assert fragment in row.expected_documents


def test_no_new_facts_are_invented_for_blank_fields(rows):
    """A case with no reviewer yet must show an EMPTY reviewer_name/date, not
    a placeholder value that could be mistaken for a real sign-off."""
    for row in rows:
        if not row.human_verified:
            assert row.reviewer_name == ""
            assert row.review_date == ""


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_csv_export_round_trips_every_row(tmp_path, rows):
    path = sat_protocol.export_csv(rows, tmp_path / "sat.csv")
    assert path.exists()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames) == list(sat_protocol.COLUMNS)
        read_rows = list(reader)
    assert len(read_rows) == len(rows)
    assert [r["case_id"] for r in read_rows] == [r.case_id for r in rows]


def test_csv_export_creates_parent_directories(tmp_path, rows):
    path = sat_protocol.export_csv(rows, tmp_path / "nested" / "dir" / "sat.csv")
    assert path.exists()


# ---------------------------------------------------------------------------
# XLSX export
# ---------------------------------------------------------------------------

def test_xlsx_export_round_trips_every_row(tmp_path, rows):
    openpyxl = pytest.importorskip("openpyxl")
    path = sat_protocol.export_xlsx(rows, tmp_path / "sat.xlsx")
    assert path.exists()
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    header = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    assert header == list(sat_protocol.COLUMNS)
    data_rows = list(sheet.iter_rows(min_row=2, values_only=True))
    assert len(data_rows) == len(rows)
    case_id_col = sat_protocol.COLUMNS.index("case_id")
    assert [r[case_id_col] for r in data_rows] == [r.case_id for r in rows]


def test_xlsx_export_marks_expert_required_rows(tmp_path, rows):
    openpyxl = pytest.importorskip("openpyxl")
    path = sat_protocol.export_xlsx(rows, tmp_path / "sat.xlsx")
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    expert_col = sat_protocol.COLUMNS.index("expert_required") + 1
    for row_index, row in enumerate(rows, start=2):
        cell = sheet.cell(row=row_index, column=expert_col)
        if row.expert_required:
            assert cell.fill.start_color.rgb in ("00FFF3CD", "FFFFF3CD", "FFF3CD")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test_generate_sat_protocol_writes_both_exports(tmp_path):
    pytest.importorskip("openpyxl")
    protocol = sat_protocol.generate_sat_protocol(
        NDS_DATASET,
        csv_path=tmp_path / "sat.csv",
        xlsx_path=tmp_path / "sat.xlsx",
    )
    assert (tmp_path / "sat.csv").exists()
    assert (tmp_path / "sat.xlsx").exists()
    assert protocol.case_count == len(protocol.rows)
    assert protocol.open_count == protocol.case_count
    assert protocol.expert_required_count == sum(1 for r in protocol.rows if r.expert_required)


def test_generate_sat_protocol_without_export_paths_only_builds_table():
    protocol = sat_protocol.generate_sat_protocol(NDS_DATASET)
    assert protocol.case_count > 0


def test_cli_main_writes_reports(tmp_path, capsys):
    pytest.importorskip("openpyxl")
    exit_code = sat_protocol.main([
        "--dataset", str(NDS_DATASET),
        "--out-dir", str(tmp_path),
    ])
    assert exit_code == 0
    assert (tmp_path / "sat_protocol.csv").exists()
    assert (tmp_path / "sat_protocol.xlsx").exists()
    captured = capsys.readouterr()
    assert "case count" in captured.out
