"""Acceptance dataset validation — is the measuring stick itself sound?

An acceptance dataset is a claim about reality ("this document exists, this
fact is in it, this person checked"). A claim nobody validates is worse than no
dataset: it produces confident numbers about nothing. Two independent checks:

  validate_schema()          structural, runs anywhere, no index needed
  validate_against_index()   every expected_document really is in the index

They are separate because only the first can run in CI. The production index
lives in Application Support on one machine, so the index check is a tool the
reviewer runs (`python -m benchmark acceptance-validate`), and the CI test skips
when the index is absent rather than passing vacuously.

Read-only throughout: opens SQLite with mode=ro and never touches retrieval.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .answer_evidence import fold
from .dataset.schema import (
    CRITICAL_CRITICALITY_NAMES,
    BenchmarkCase,
)

# A fragment matching more of the archive than this asserts almost nothing —
# "01_KONTROLNÍ DNY" is satisfied by any of 165 documents. Legitimate for a
# folder-level lookup, misleading for a factual case, so it is surfaced as a
# warning and left to the reviewer rather than failed automatically.
WIDE_MATCH_THRESHOLD = 50


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_case_count: int = 0
    index_checked: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_case_count": self.checked_case_count,
            "index_checked": self.index_checked,
        }


def validate_schema(cases: list[BenchmarkCase]) -> ValidationReport:
    """Rules that make a case answerable, independent of any index."""
    report = ValidationReport(checked_case_count=len(cases))
    err = report.errors.append
    warn = report.warnings.append

    for case in cases:
        where = f"{case.id}"

        # A critical case with no declared verification method can never be
        # signed off, because nobody knows what signing it off would mean.
        if case.criticality in CRITICAL_CRITICALITY_NAMES and not case.verification_method:
            err(f"{where}: criticality={case.criticality!r} requires a verification_method")

        # Existence of a file proves nothing about a legal or safety claim.
        if (
            case.criticality in CRITICAL_CRITICALITY_NAMES
            and case.verification_method == "index_query"
        ):
            warn(
                f"{where}: criticality={case.criticality!r} verified only by index_query - "
                "a file being present is not confirmation that its content is current"
            )

        # An anonymous sign-off is not a sign-off.
        if case.human_verified and not case.verified_by.strip():
            err(f"{where}: human_verified=true without verified_by")
        if case.human_verified and not case.verification_date.strip():
            err(f"{where}: human_verified=true without verification_date")

        # Verification is scoped to the index it was performed against; without
        # the fingerprint the claim silently outlives the data it describes.
        if case.human_verified and not case.index_fingerprint_at_verification.strip():
            err(f"{where}: human_verified=true without index_fingerprint_at_verification")

        # A verified status the human never granted is the false-confidence
        # failure this whole layer exists to prevent. Scoped to production:
        # on the fixture corpus the documents and the case are authored
        # together by the test suite, so "verified" is definitional there and
        # there is no archive for a person to check it against.
        if (
            case.environment == "production"
            and case.ground_truth_status == "verified"
            and not case.human_verified
        ):
            err(f"{where}: ground_truth_status='verified' but human_verified=false")

        if case.expected_outcome == "not_found":
            if case.expected_documents:
                err(
                    f"{where}: expected_outcome='not_found' but expected_document is set - "
                    "a negative case must not also assert a positive retrieval target"
                )
            if not (case.expected_answer_contains or case.forbidden_answer_contains
                    or case.forbidden_document):
                err(
                    f"{where}: expected_outcome='not_found' with no answer contract - "
                    "the case would pass on any answer whatsoever"
                )
            if case.verification_method not in ("folder_listing", "document_read", "expert_confirm"):
                warn(
                    f"{where}: negative case verified by {case.verification_method!r} - "
                    "absence from a full-text index is not proof of absence from the archive"
                )
        elif not (
            case.expected_documents
            or case.expected_answer_contains
            or case.expected_source_contains
        ):
            err(f"{where}: positive case asserts nothing - it will pass unconditionally")

    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        err(f"duplicate case ids: {duplicates}")
    return report


def _index_haystack(db_path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT relative_path, name FROM documents").fetchall()
    finally:
        connection.close()
    # Folded once: paths on macOS are stored NFD, dataset fragments are written
    # NFC, and only folding both sides makes "KD č.72" match itself.
    return [fold(f"{path or ''} {name or ''}") for path, name in rows]


def validate_against_index(
    cases: list[BenchmarkCase],
    db_path: Path,
    *,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """Every fragment the dataset names must resolve to a real document."""
    report = report or ValidationReport(checked_case_count=len(cases))
    report.index_checked = True
    haystack = _index_haystack(Path(db_path))

    def hits(fragment: str) -> int:
        needle = fold(fragment)
        return sum(1 for entry in haystack if needle in entry)

    for case in cases:
        for fragment in case.expected_documents:
            count = hits(fragment)
            if count == 0:
                report.errors.append(
                    f"{case.id}: expected_document {fragment!r} matches no document in the index"
                )
            elif count > WIDE_MATCH_THRESHOLD:
                report.warnings.append(
                    f"{case.id}: expected_document {fragment!r} matches {count} documents - "
                    "the assertion is folder-wide, not document-level"
                )
        for fragment in case.expected_source_contains:
            if hits(fragment) == 0:
                report.errors.append(
                    f"{case.id}: expected_source {fragment!r} matches no document in the index"
                )
        for fragment in case.forbidden_document:
            # A forbidden document that does not exist is a trap that can never
            # spring: the case looks protective and tests nothing.
            if hits(fragment) == 0:
                report.errors.append(
                    f"{case.id}: forbidden_document {fragment!r} matches no document - "
                    "the trap cannot fire, so the case gives false assurance"
                )
    return report


def validate(
    cases: list[BenchmarkCase], db_path: Path | None = None,
) -> ValidationReport:
    report = validate_schema(cases)
    if db_path is not None and Path(db_path).exists():
        validate_against_index(cases, Path(db_path), report=report)
    return report
