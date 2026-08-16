"""PR7.5 — Garáže ND Smíchov acceptance dataset and its measurement layer.

The dataset is the measuring stick. These tests check the stick before anyone
uses it to measure the product: that it loads, that its categories and negative
cases are what the acceptance plan says, that no case asserts a document nobody
confirmed exists, and that the new KPIs count what they claim to count.

The one check that needs the real archive (`expected_document` resolves to a
document actually in the production index) skips when that index is absent
instead of passing vacuously — a green CI run must never be evidence that
ground truth was validated.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import acceptance_metrics, acceptance_runner, dataset_validation  # noqa: E402
from benchmark import report as report_mod  # noqa: E402
from benchmark.dataset.schema import (  # noqa: E402
    CRITICAL_CRITICALITY_NAMES,
    DATASET_DIR,
    PROJECT_ACCEPTANCE_CATEGORIES,
    BenchmarkCase,
    load_dataset,
    read_dataset_version,
)

NDS_DATASET = DATASET_DIR / "acceptance_nds_smichov.jsonl"
PROJECT_ID = "240783160_Garáže_NDS"

EXPECTED_DISTRIBUTION = {
    "DOCUMENT_SEARCH": 12,
    "TECHNICAL_QA": 10,
    "DOCUMENT_STATUS": 5,
    "CONSTRUCTION_MGMT": 4,
    "ADVERSARIAL": 6,
}


@pytest.fixture(scope="module")
def cases() -> list[BenchmarkCase]:
    return load_dataset(NDS_DATASET)


def _production_db() -> Path | None:
    """The REAL index, or None when this machine has never run the app.

    Resolved from the default Application Support location rather than from
    ai_search_config.APP_SUPPORT_DIR, because tests/conftest.py redirects
    AI_SEARCH_HOME to a temp dir to keep the suite off the user's state. That
    isolation is right for everything that writes; this check has to read the
    actual archive, since its whole purpose is proving the dataset's ground
    truth resolves against the documents the site manager really has. Opened
    read-only (mode=ro) and never written to.
    """
    db_path = Path.home() / "Library/Application Support/AI Search/database/project.sqlite3"
    return db_path if db_path.exists() else None


# ---------------------------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------------------------

def test_dataset_loads_all_cases(cases):
    assert len(cases) >= 34
    assert len(cases) == sum(EXPECTED_DISTRIBUTION.values())


def test_dataset_declares_a_version():
    assert read_dataset_version(NDS_DATASET)


def test_category_distribution_matches_the_acceptance_plan(cases):
    assert Counter(c.category for c in cases) == Counter(EXPECTED_DISTRIBUTION)
    assert set(EXPECTED_DISTRIBUTION) <= PROJECT_ACCEPTANCE_CATEGORIES


def test_every_case_belongs_to_the_one_project_under_certification(cases):
    assert {c.project for c in cases} == {PROJECT_ID}
    assert {c.environment for c in cases} == {"production"}


def test_negative_cases_exist_and_carry_an_answer_contract(cases):
    negatives = [c for c in cases if c.expected_outcome == "not_found"]
    assert len(negatives) >= 3
    for case in negatives:
        assert not case.expected_documents, f"{case.id} negative case names a positive target"
        assert (
            case.expected_answer_contains
            or case.forbidden_answer_contains
            or case.forbidden_document
        ), f"{case.id} would pass on any answer"


def test_the_three_absent_documents_are_negative_not_positive(cases):
    """kniha betonů / stavební deník / výkres bednění are not in the index.

    Asserting them as positives would create permanently red rows; this pins
    that nobody quietly "fixes" the dataset by inventing a target for them.
    """
    by_id = {c.id: c for c in cases}
    assert by_id["nds-doc-10"].expected_outcome == "not_found"
    assert by_id["nds-doc-10"].expected_content_missing
    assert by_id["nds-qa-10"].expected_outcome == "not_found"
    for case in cases:
        haystack = f"{case.question} {' '.join(case.expected_documents)}".casefold()
        if "kniha beton" in haystack or "stavební deník" in haystack:
            assert case.expected_outcome == "not_found", case.id


def test_required_adversarial_traps_are_present(cases):
    """The four traps PR7.5 names explicitly, each tied to real index data."""
    adversarial = {c.id: c for c in cases if c.category == "ADVERSARIAL"}
    assert len(adversarial) >= 5

    # A) superseded harmonogram revisions must be forbidden as an answer source
    assert any(
        "R1" in f or "R2" in f
        for f in adversarial["nds-adv-01"].forbidden_document
    )
    # B) SoD FERI: the draft must be forbidden, the final expected
    assert adversarial["nds-adv-02"].forbidden_document
    assert any("final" in d for d in adversarial["nds-adv-02"].expected_documents)
    # C) NDS_seznam TP a KZP.xlsx carries PALÁC DUNAJ content
    dunaj = adversarial["nds-adv-03"]
    assert dunaj.expected_outcome == "not_found"
    assert any("dunaj" in k.casefold() for k in dunaj.forbidden_answer_contains)
    assert any("seznam TP a KZP" in d for d in dunaj.forbidden_document)
    # D) all three duplicated suppliers, each requiring BOTH order numbers
    blob = " ".join(
        f"{c.question} {' '.join(c.expected_answer_contains)}" for c in adversarial.values()
    ).casefold()
    for supplier in ("stafitech", "bičík", "hilt rent"):
        assert supplier in blob, supplier
    for case_id in ("nds-adv-04", "nds-adv-05", "nds-adv-06"):
        assert len(adversarial[case_id].expected_answer_contains) == 2, case_id


def test_no_case_claims_verified_ground_truth_without_a_human(cases):
    """The false-confidence guard. Nothing in this dataset is signed off yet,
    and `verified` must never be reachable without a named person."""
    for case in cases:
        if case.ground_truth_status == "verified":
            assert case.human_verified and case.verified_by.strip(), case.id
        if case.human_verified:
            assert case.verified_by.strip(), case.id


def test_critical_cases_declare_a_verification_method(cases):
    for case in cases:
        if case.criticality in CRITICAL_CRITICALITY_NAMES:
            assert case.verification_method, case.id


# ---------------------------------------------------------------------------
# Validation module
# ---------------------------------------------------------------------------

def test_schema_validation_passes_on_the_shipped_dataset(cases):
    report = dataset_validation.validate_schema(cases)
    assert report.ok, report.errors


def test_schema_validation_rejects_an_anonymous_sign_off():
    case = BenchmarkCase.from_dict({
        "id": "x", "query": "q", "expected_document": ["a"],
        "human_verified": True, "ground_truth_status": "verified",
    })
    errors = dataset_validation.validate_schema([case]).errors
    assert any("verified_by" in e for e in errors)
    assert any("index_fingerprint_at_verification" in e for e in errors)


def test_schema_validation_rejects_verified_status_without_human():
    case = BenchmarkCase.from_dict({
        "id": "x", "query": "q", "expected_document": ["a"],
        "environment": "production", "ground_truth_status": "verified",
    })
    assert any(
        "human_verified=false" in e for e in dataset_validation.validate_schema([case]).errors
    )
    # On the fixture corpus the case and the documents are authored together,
    # so there is no archive for a person to verify against.
    fixture = BenchmarkCase.from_dict({
        "id": "x", "query": "q", "expected_document": ["a"],
        "environment": "fixture", "ground_truth_status": "verified",
    })
    assert not dataset_validation.validate_schema([fixture]).errors


def test_the_earlier_acceptance_dataset_still_loads_and_its_gaps_are_pinned():
    """PR7.4.1's acceptance_v1.jsonl predates the PR7.5 fields, so it still
    loads (that is the compatibility contract) but does not satisfy the new
    rules. Left unmodified on purpose - it belongs to PR7.4.1 - and its gaps
    are pinned here so they stay visible instead of being quietly tolerated.
    """
    v1_cases = load_dataset(DATASET_DIR / "acceptance_v1.jsonl")
    assert v1_cases

    errors = dataset_validation.validate_schema(v1_cases).errors
    # Gap 1: verification_method did not exist yet, so no critical case has one.
    assert any("requires a verification_method" in e for e in errors)
    # Gap 2: one production template case asserts nothing and passes on any
    # answer. Worth fixing when acceptance_v1 is next revised.
    assert any("acc-prod-lookup-kzp-monolit" in e for e in errors)
    # And nothing worse than those two classes.
    unexpected = [
        e for e in errors
        if "requires a verification_method" not in e and "asserts nothing" not in e
    ]
    assert not unexpected, unexpected


def test_schema_validation_rejects_a_case_that_asserts_nothing():
    case = BenchmarkCase.from_dict({"id": "x", "query": "q"})
    assert any(
        "asserts nothing" in e for e in dataset_validation.validate_schema([case]).errors
    )


def test_schema_validation_requires_verification_method_on_critical_cases():
    case = BenchmarkCase.from_dict({
        "id": "x", "query": "q", "criticality": "safety", "expected_document": ["a"],
    })
    assert any(
        "verification_method" in e for e in dataset_validation.validate_schema([case]).errors
    )


@pytest.mark.skipif(_production_db() is None, reason="production index not present on this machine")
def test_every_expected_document_exists_in_the_production_index(cases):
    report = dataset_validation.validate_against_index(cases, _production_db())
    assert report.ok, report.errors


@pytest.mark.skipif(_production_db() is None, reason="production index not present on this machine")
def test_every_forbidden_document_is_a_trap_that_can_actually_fire(cases):
    """A forbidden_document naming nothing real looks protective and tests
    nothing — the case would pass no matter how badly the tool behaves."""
    report = dataset_validation.validate_against_index(
        [c for c in cases if c.forbidden_document], _production_db(),
    )
    assert not [e for e in report.errors if "forbidden_document" in e], report.errors


def test_index_validation_flags_a_fragment_that_matches_nothing(tmp_path):
    db_path = tmp_path / "tiny.sqlite3"
    import sqlite3

    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE documents (relative_path TEXT, name TEXT)")
    connection.execute("INSERT INTO documents VALUES ('a/b/Smlouva.pdf', 'Smlouva.pdf')")
    connection.commit()
    connection.close()

    case = BenchmarkCase.from_dict({
        "id": "ghost", "query": "q", "expected_document": ["Neexistuje.pdf"],
    })
    report = dataset_validation.validate_against_index([case], db_path)
    assert any("matches no document" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------

def test_pr75_field_aliases_map_onto_the_canonical_names():
    case = BenchmarkCase.from_dict({
        "id": "x",
        "query": "jaký beton?",
        "expected_document": ["TZ.pdf"],
        "expected_source": ["TZ"],
        "expected_answer_keywords": ["bílá vana"],
        "forbidden_answer_keywords": ["dunaj"],
    })
    assert case.question == "jaký beton?"
    assert case.expected_documents == ["TZ.pdf"]
    assert case.expected_source_contains == ["TZ"]
    assert case.expected_answer_contains == ["bílá vana"]
    assert case.forbidden_answer_contains == ["dunaj"]


def test_setting_both_an_alias_and_its_canonical_name_is_an_error():
    with pytest.raises(ValueError, match="same field"):
        BenchmarkCase.from_dict({"id": "x", "query": "a", "question": "b"})


def test_invalid_expected_outcome_and_verification_method_are_rejected():
    with pytest.raises(ValueError):
        BenchmarkCase.from_dict({"id": "x", "query": "q", "expected_outcome": "maybe"})
    with pytest.raises(ValueError):
        BenchmarkCase.from_dict({"id": "x", "query": "q", "verification_method": "vibes"})


def test_safety_is_a_critical_criticality():
    assert "safety" in CRITICAL_CRITICALITY_NAMES
    case = BenchmarkCase.from_dict({"id": "x", "query": "q", "criticality": "safety"})
    assert acceptance_metrics.is_critical_error(case, answer_correct=False, forbidden_hit=False)


# ---------------------------------------------------------------------------
# New KPIs
# ---------------------------------------------------------------------------

def _answer(text: str, documents: list[str], paths: list[str] | None = None) -> dict:
    paths = paths or [f"/{d}" for d in documents]
    return {
        "answer": text,
        "citations": [{"document": d, "path": p} for d, p in zip(documents, paths)],
    }


def test_document_found_is_none_when_the_case_names_no_document():
    """Otherwise an empty needle list matches everything and a negative case
    silently counts as a retrieval success."""
    case = BenchmarkCase.from_dict({
        "id": "n", "query": "najdi knihu betonů", "expected_outcome": "not_found",
        "expected_answer_keywords": ["nenalezeno"],
    })
    found, _correct, _missing, _used = acceptance_metrics.evaluate_acceptance_answer(
        case, [{"document": "jiny.pdf", "path": "/jiny.pdf"}],
        _answer("Nenalezeno v indexovaných dokumentech.", []),
    )
    assert found is None


def test_top1_and_top5_exclude_cases_with_no_retrieval_contract():
    results = [
        acceptance_metrics.AcceptanceCaseResult(
            id="rank1", question="q", category="DOCUMENT_SEARCH", environment="production",
            criticality="technical", ground_truth_status="needs_review",
            document_found=True, document_rank=1, answer_correct=True,
        ),
        acceptance_metrics.AcceptanceCaseResult(
            id="rank7", question="q", category="DOCUMENT_SEARCH", environment="production",
            criticality="technical", ground_truth_status="needs_review",
            document_found=True, document_rank=7, answer_correct=True,
        ),
        acceptance_metrics.AcceptanceCaseResult(
            id="negative", question="q", category="DOCUMENT_SEARCH", environment="production",
            criticality="technical", ground_truth_status="needs_review",
            document_found=None, document_rank=None, answer_correct=True,
        ),
    ]
    agg = acceptance_metrics.aggregate_acceptance(results)
    assert agg.retrieval_measured_count == 2
    assert agg.top1_accuracy == 0.5
    assert agg.top5_accuracy == 0.5
    assert agg.document_found_rate == 1.0


def test_forbidden_document_is_detected_by_path_not_only_by_name():
    """The revision trap: `D.1.2.07` is spelled identically inside and outside
    `OLD/`, so only the path can tell a withdrawn drawing from a current one."""
    case = BenchmarkCase.from_dict({
        "id": "rev", "query": "výkres 3.PP", "expected_document": ["D.1.2.07"],
        "forbidden_document": ["D12_Statika/OLD"],
    })
    answer = _answer(
        "Výkres je k dispozici.\n(Zdroj: D.1.2.07.pdf)",
        ["D.1.2.07.pdf"], ["Komplet/D12_Statika/OLD/D.1.2.07.pdf"],
    )
    citation_correct, forbidden_hit, unsupported = acceptance_metrics.evaluate_citations(
        case, answer,
    )
    assert forbidden_hit is True
    assert citation_correct is False
    assert unsupported is False
    layer, _detail = acceptance_metrics.classify_failure(
        case, True, True, None, forbidden_document_hit=True,
    )
    assert layer == acceptance_metrics.LAYER_SAFETY
    assert acceptance_metrics.is_critical_error(
        case, answer_correct=True, forbidden_hit=False, forbidden_document_hit=True,
    )


def test_citation_correct_is_none_without_a_citation_contract():
    case = BenchmarkCase.from_dict({"id": "x", "query": "q", "expected_document": ["a"]})
    citation_correct, _hit, _unsupported = acceptance_metrics.evaluate_citations(
        case, _answer("Odpověď.\n(Zdroj: a.pdf)", ["a.pdf"]),
    )
    assert citation_correct is None


def test_unsupported_claim_is_an_assertion_with_no_evidence():
    case = BenchmarkCase.from_dict({"id": "x", "query": "q", "expected_document": ["a"]})
    _cc, _hit, unsupported = acceptance_metrics.evaluate_citations(
        case, _answer("Beton je C30/37.", ["a.pdf"]),
    )
    assert unsupported is True


def test_a_not_found_answer_is_not_an_unsupported_claim():
    case = BenchmarkCase.from_dict({"id": "x", "query": "q", "expected_document": ["a"]})
    _cc, _hit, unsupported = acceptance_metrics.evaluate_citations(
        case, _answer("Nenalezeno v indexovaných dokumentech.", ["a.pdf"]),
    )
    assert unsupported is False


def test_structured_scaffolding_alone_is_not_an_unsupported_claim():
    """The structured renderer always emits its headings and the "Žádné"
    filler. Counting that as a claim would flag every empty answer."""
    body = (
        "Shrnutí:\n- Nenalezeno v indexovaných dokumentech.\n\n"
        "Požadované dokumenty / kroky:\n\n"
        "Nenalezené informace:\n- Žádné\n\n"
        "Zdroje:\n- Nenalezeno v indexovaných dokumentech."
    )
    assert acceptance_metrics.has_substantive_claim(_answer(body, ["a.pdf"])) is False


def test_unsupported_claims_and_forbidden_hits_block_a_production_go():
    results = [
        acceptance_metrics.AcceptanceCaseResult(
            id=f"c{i}", question="q", category="DOCUMENT_SEARCH", environment="production",
            criticality="informational", ground_truth_status="verified",
            document_found=True, document_rank=1, answer_correct=True,
            citation_correct=True, queries_to_answer=1, total_ms=10.0,
        )
        for i in range(30)
    ]
    assert acceptance_metrics.acceptance_verdict(
        acceptance_metrics.aggregate_acceptance(results), environment="production",
    )["verdict"] == "GO"

    results[0].unsupported_claim = True
    verdict = acceptance_metrics.acceptance_verdict(
        acceptance_metrics.aggregate_acceptance(results), environment="production",
    )
    assert verdict["verdict"] == "NO-GO"
    assert any("unsupported_claims=1" in b for b in verdict["blockers"])

    results[0].unsupported_claim = False
    results[1].forbidden_document_measured = True
    results[1].forbidden_document_hit = True
    verdict = acceptance_metrics.acceptance_verdict(
        acceptance_metrics.aggregate_acceptance(results), environment="production",
    )
    assert verdict["verdict"] == "NO-GO"
    assert any("forbidden_document_hits=1" in b for b in verdict["blockers"])


# ---------------------------------------------------------------------------
# SAT status and report
# ---------------------------------------------------------------------------

def test_sat_status_is_not_ready_while_the_dataset_is_unverified(cases):
    status = acceptance_runner.sat_status(cases, "production")
    assert status["ready_for_daily_use"] is False
    assert status["verified_cases_count"] == 0
    assert status["pending_cases_count"] == len(cases)
    assert status["critical_cases_without_expert_confirm"]


def test_sat_status_becomes_ready_only_with_full_human_sign_off():
    signed = [
        BenchmarkCase.from_dict({
            "id": f"c{i}", "query": "q", "expected_document": ["a"],
            "criticality": "legal", "verification_method": "expert_confirm",
            "ground_truth_status": "verified", "human_verified": True,
            "verified_by": "M. Varvarovský", "verification_date": "2026-08-12",
            "index_fingerprint_at_verification": "abc123",
        })
        for i in range(3)
    ]
    assert acceptance_runner.sat_status(signed, "production")["ready_for_daily_use"] is True
    # A single unsigned case withdraws readiness.
    signed[0].human_verified = False
    assert acceptance_runner.sat_status(signed, "production")["ready_for_daily_use"] is False


def test_report_separates_fat_from_sat_and_refuses_to_claim_readiness():
    artifact = {
        "timestamp": "2026-08-12T00:00:00+00:00",
        "git_sha": "abc",
        "project_id": PROJECT_ID,
        "index_fingerprint": "deadbeefdeadbeefdeadbeef",
        "dataset_version": "0.1.0-draft",
        "environment": {"name": "production", "doc_count": 6342, "chunk_count": 157037},
        "dataset_file": "acceptance_nds_smichov.jsonl",
        "flags": {"DOCUMENT_STATE_GATE_ENABLED": True,
                  "EVIDENCE_RUNTIME_VALIDATION_ENABLED": True, "llm_replay": False},
        "case_count": 35,
        "cases": [],
        "aggregate": {
            "case_count": 35, "evaluated_count": 35, "retrieval_measured_count": 31,
            "document_found_count": 28, "document_found_rate": 0.9,
            "top1_count": 12, "top1_accuracy": 0.39, "top5_count": 25, "top5_accuracy": 0.81,
            "answer_correct_count": 20, "answer_correct_rate": 0.57,
            "citation_measured_count": 20, "citation_correct_count": 18,
            "citation_correct_rate": 0.9, "unsupported_claim_count": 2,
            "unsupported_claim_rate": 0.06, "forbidden_document_measured_count": 8,
            "forbidden_document_hit_count": 1, "forbidden_document_rate": 0.125,
            "critical_error_count": 1, "critical_error_case_ids": ["nds-status-02"],
            "mean_total_ms": 4200.0, "p95_total_ms": 9000.0, "mean_queries_to_answer": 1.2,
            "resolved_within_one_query": 18, "resolved_with_follow_up": 2, "unresolved": 15,
            "verified_case_count": 0, "unverified_case_count": 35,
            "by_category": {}, "by_layer": {"OK": 20},
        },
        "verdict": {"verdict": "INCONCLUSIVE", "blockers": [],
                    "inconclusive_reasons": ["35 case(s) have unverified ground truth"],
                    "criteria": {}},
        "warnings": [],
        "verified_cases_count": 0,
        "pending_cases_count": 35,
        "sat_status": {
            "verified_cases_count": 0, "pending_cases_count": 35, "critical_cases_count": 18,
            "critical_cases_without_expert_confirm": ["nds-doc-05"],
            "human_verified_rate": 0.0, "ready_for_daily_use": False,
            "blockers": ["35/35 case(s) not human-verified"],
        },
    }
    markdown = report_mod.render_markdown_acceptance(artifact)
    for heading in ("## FAT RESULT", "## SAT STATUS"):
        assert heading in markdown
    for label in ("top1_accuracy", "top5_accuracy", "citation_correct_rate",
                  "unsupported_claim_rate", "forbidden_document_rate"):
        assert label in markdown
    assert "Připraveno pro denní použití: NE" in markdown
    assert "netvrdí" in markdown
    assert PROJECT_ID in markdown
    assert "0.1.0-draft" in markdown
