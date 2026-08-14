"""PR7.5.2 — Garáže ND Smíchov FAT Acceptance Runner.

Benchmark-only: no live Ollama server or production index is required except
for the one end-to-end smoke test, which uses the synthetic fixture corpus and
a monkeypatched Ollama response - the same pattern tests/test_pr74_answer_quality.py
already established. Category-dispatch logic is tested as pure functions
against synthetic (case, results, answer) triples, with no I/O at all.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_search  # noqa: E402

from benchmark import acceptance_run_nds as runner  # noqa: E402
from benchmark.dataset.schema import (  # noqa: E402
    DATASET_DIR,
    PROJECT_ACCEPTANCE_CATEGORIES,
    BenchmarkCase,
    load_dataset,
)

NDS_DATASET = DATASET_DIR / "acceptance_nds_smichov.jsonl"

# The five runtime files the brief forbids touching. Checked by content hash
# before/after the end-to-end run, not just mtime, so an in-place rewrite that
# preserves the mtime would still be caught.
RUNTIME_FILES = [
    Path(__file__).resolve().parent.parent / name
    for name in ("ai_search.py", "ui_services.py", "evidence_runtime.py",
                 "document_state.py", "evidence.py")
]


def _runtime_snapshot() -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in RUNTIME_FILES if p.exists()}


@pytest.fixture(scope="module")
def cases() -> list[BenchmarkCase]:
    return load_dataset(NDS_DATASET)


def _answer(text: str, documents: list[str], paths: list[str] | None = None,
            validation: dict | None = None) -> dict:
    paths = paths or [f"/{d}" for d in documents]
    out = {"answer": text, "citations": [{"document": d, "path": p} for d, p in zip(documents, paths)]}
    if validation is not None:
        out["validation"] = validation
    return out


def _rows(documents: list[str], paths: list[str] | None = None) -> list[dict]:
    paths = paths or [f"/{d}" for d in documents]
    return [{"document": d, "path": p} for d, p in zip(documents, paths)]


def _case(**overrides) -> BenchmarkCase:
    data = {"id": "x", "query": "q", "category": "TECHNICAL_QA", "project": "240783160_Garáže_NDS"}
    data.update(overrides)
    return BenchmarkCase.from_dict(data)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_dataset_loads_all_cases(cases):
    assert len(cases) >= 34
    assert {c.category for c in cases} <= PROJECT_ACCEPTANCE_CATEGORIES


def test_load_nds_dataset_matches_direct_load(cases):
    assert [c.id for c in runner.load_nds_dataset(NDS_DATASET)] == [c.id for c in cases]


# ---------------------------------------------------------------------------
# OLD-revision-leakage detector
# ---------------------------------------------------------------------------

def test_old_revision_leakage_detected_via_explicit_forbidden_document():
    case = _case(
        category="DOCUMENT_SEARCH", expected_document=["D.1.2.11"],
        forbidden_document=["D12_Statika/OLD"],
    )
    answer = _answer(
        "Výkres je k dispozici.\n(Zdroj: D.1.2.11.pdf)",
        ["D.1.2.11.pdf"], ["Komplet/D12_Statika/OLD/D.1.2.11.pdf"],
    )
    hits = runner.detect_old_revision_leakage(case, [], answer)
    assert hits == ["Komplet/D12_Statika/OLD/D.1.2.11.pdf"]


def test_old_revision_leakage_detected_when_a_non_old_alternative_was_available():
    """No forbidden_document declared, but the retrieval pool DID contain a
    non-OLD copy of the expected target and the answer leaned on the OLD one
    anyway - filenames repeat across revisions, so this can only be told
    apart from the legitimate case below by looking at the pool."""
    case = _case(category="DOCUMENT_SEARCH", expected_document=["D.1.2.11"])
    pool = [
        {"document": "D.1.2.11.pdf", "path": "Komplet/D12_Statika/D.1.2.11.pdf"},
        {"document": "D.1.2.11.pdf", "path": "Komplet/D12_Statika/OLD/D.1.2.11.pdf"},
    ]
    answer = _answer(
        "Výkres je k dispozici.\n(Zdroj: D.1.2.11.pdf)",
        ["D.1.2.11.pdf"], ["Komplet/D12_Statika/OLD/D.1.2.11.pdf"],
    )
    hits = runner.detect_old_revision_leakage(case, pool, answer)
    assert hits == ["Komplet/D12_Statika/OLD/D.1.2.11.pdf"]


def test_old_revision_leakage_excludes_the_deliberately_expected_target():
    """nds-status-03: D.1.2.07 legitimately exists only inside OLD/ - no
    non-OLD alternative was ever in the pool, so this is the correct answer,
    not leakage."""
    case = _case(
        category="DOCUMENT_STATUS", expected_document=["D.1.2.07 - schéma vyztužení 3.PP"],
    )
    pool = [{
        "document": "D.1.2.07 - schéma vyztužení 3.PP.pdf",
        "path": "Komplet/D12_Statika/OLD/D.1.2.07 - schéma vyztužení 3.PP.pdf",
    }]
    answer = _answer(
        "Výkres existuje.\n(Zdroj: D.1.2.07 - schéma vyztužení 3.PP.pdf)",
        ["D.1.2.07 - schéma vyztužení 3.PP.pdf"],
        ["Komplet/D12_Statika/OLD/D.1.2.07 - schéma vyztužení 3.PP.pdf"],
    )
    assert runner.detect_old_revision_leakage(case, pool, answer) == []


def test_old_revision_leakage_ignores_non_old_paths():
    case = _case(expected_document=["a.pdf"])
    answer = _answer("Text.\n(Zdroj: a.pdf)", ["a.pdf"], ["Komplet/D12_Statika/a.pdf"])
    assert runner.detect_old_revision_leakage(case, [], answer) == []


def test_old_revision_leakage_does_not_false_positive_on_substring():
    """A path containing "old" as part of a longer word (not a folder
    segment) must not be flagged."""
    case = _case(expected_document=["a.pdf"])
    answer = _answer("Text.\n(Zdroj: a.pdf)", ["a.pdf"], ["Podklady/a.pdf"])
    assert runner.detect_old_revision_leakage(case, [], answer) == []


# ---------------------------------------------------------------------------
# Entity-substitution heuristic
# ---------------------------------------------------------------------------

def test_entity_substitution_suspected_when_only_one_of_two_present():
    case = _case(category="ADVERSARIAL", expected_answer_keywords=["NOT250039", "NOT250304"])
    answer = _answer("Zakázka NOT250039 na zdění.", ["a.pdf"])
    assert runner.detect_entity_substitution(case, answer) is True


def test_entity_substitution_not_suspected_when_both_present():
    case = _case(category="ADVERSARIAL", expected_answer_keywords=["NOT250039", "NOT250304"])
    answer = _answer("Zakázky NOT250039 a NOT250304 na zdění.", ["a.pdf"])
    assert runner.detect_entity_substitution(case, answer) is False


def test_entity_substitution_not_suspected_when_neither_present():
    case = _case(category="ADVERSARIAL", expected_answer_keywords=["NOT250039", "NOT250304"])
    assert runner.detect_entity_substitution(case, _answer("Nenalezeno.", [])) is False


def test_entity_substitution_only_applies_to_adversarial_category():
    case = _case(category="TECHNICAL_QA", expected_answer_keywords=["NOT250039", "NOT250304"])
    answer = _answer("Zakázka NOT250039.", ["a.pdf"])
    assert runner.detect_entity_substitution(case, answer) is False


# ---------------------------------------------------------------------------
# Category dispatch (pure - no retrieval, no Ollama)
# ---------------------------------------------------------------------------

def test_document_search_case_passes_on_top1_hit():
    case = _case(category="DOCUMENT_SEARCH", expected_document=["kladecke"])
    results = [{"document": "kladecke.txt", "path": "/kladecke.txt"}]
    answer = _answer("Nalezeno.\n(Zdroj: kladecke.txt)", ["kladecke.txt"])
    fields, reasons = runner._evaluate_result(case, results, answer)
    assert fields["document_found"] is True
    assert fields["top1"] is True
    assert fields["top5"] is True
    assert not reasons


def test_document_search_case_fails_when_document_not_retrieved():
    case = _case(category="DOCUMENT_SEARCH", expected_document=["kladecke"])
    results = [{"document": "jiny.txt", "path": "/jiny.txt"}]
    fields, reasons = runner._evaluate_result(case, results, _answer("Nenalezeno.", []))
    assert fields["document_found"] is False
    assert fields["top1"] is False
    assert any("never reached the result pool" in r for r in reasons)


def test_document_search_negative_case_passes_on_correct_not_found():
    case = _case(
        category="DOCUMENT_SEARCH", expected_outcome="not_found",
        expected_answer_keywords=["nenalezeno"],
    )
    fields, reasons = runner._evaluate_result(
        case, [{"document": "jiny.txt", "path": "/jiny.txt"}],
        _answer("Nenalezeno v indexovaných dokumentech.", []),
    )
    assert fields["document_found"] is None
    assert not reasons


def test_document_search_negative_case_fails_when_tool_claims_found():
    case = _case(
        category="DOCUMENT_SEARCH", expected_outcome="not_found",
        forbidden_answer_keywords=["kniha betonů existuje"],
    )
    fields, reasons = runner._evaluate_result(
        case, [], _answer("Kniha betonů existuje ve složce X.", ["x.pdf"]),
    )
    assert any("not_found" in r for r in reasons)


def test_technical_qa_case_fails_when_fact_missing():
    case = _case(category="TECHNICAL_QA", expected_answer_keywords=["vodonepropustn"])
    fields, reasons = runner._evaluate_result(
        case, [{"document": "tz.pdf", "path": "/tz.pdf"}],
        _answer("Beton je kvalitní.", ["tz.pdf"]),
    )
    assert fields["answer_correct"] is False
    assert any("answer missing required fact" in r for r in reasons)


def test_document_status_case_fails_when_source_is_wrong_even_if_statement_reads_right():
    """The statement text ("SafetyPeak") reads right, but the answer leans
    on the wrong citation - evaluate_acceptance_answer requires the expected
    source to actually be used, so BOTH answer_correct and the standalone
    citation_correct check must catch this."""
    case = _case(
        category="DOCUMENT_STATUS", expected_answer_keywords=["safetypeak"],
        expected_source=["SafetyPeak_podepsaná"],
    )
    fields, reasons = runner._evaluate_result(
        case, [{"document": "SafetyPeak_podepsaná.pdf", "path": "/x/SafetyPeak_podepsaná.pdf"}],
        _answer("Ano, SafetyPeak.\n(Zdroj: jiny.pdf)", ["jiny.pdf"]),
    )
    assert fields["answer_correct"] is False
    assert fields["citation_correct"] is False
    assert any("source incorrect" in r for r in reasons)


def test_document_status_case_passes_when_statement_and_source_both_correct():
    case = _case(
        category="DOCUMENT_STATUS", expected_answer_keywords=["safetypeak"],
        expected_source=["SafetyPeak_podepsaná"],
    )
    fields, reasons = runner._evaluate_result(
        case, [{"document": "SafetyPeak_podepsaná.pdf", "path": "/x/SafetyPeak_podepsaná.pdf"}],
        _answer("Ano, SafetyPeak.\n(Zdroj: SafetyPeak_podepsaná.pdf)", ["SafetyPeak_podepsaná.pdf"]),
    )
    assert fields["answer_correct"] is True
    assert fields["citation_correct"] is True
    assert not reasons


def test_adversarial_case_flags_forbidden_document_use():
    case = _case(
        category="ADVERSARIAL", expected_document=["final"], forbidden_document=["30.07.25"],
    )
    answer = _answer(
        "Termín je 21.10.2026.\n(Zdroj: NDS_SOD_FERI_NOT251110_30.07.25.docx)",
        ["NDS_SOD_FERI_NOT251110_30.07.25.docx"],
    )
    fields, reasons = runner._evaluate_result(case, [], answer)
    assert fields["wrong_document_citation"] is True
    assert any("wrong document citation" in r for r in reasons)


def test_a_clean_pass_produces_no_reasons_and_is_not_critical():
    case = _case(category="TECHNICAL_QA", expected_answer_keywords=["ok"])
    fields, reasons = runner._evaluate_result(
        case, [{"document": "a.pdf", "path": "/a.pdf"}],
        _answer("Vše ok.\n(Zdroj: a.pdf)", ["a.pdf"]),
    )
    assert not reasons
    assert fields["critical_error"] is False


def test_unsupported_claim_is_flagged_regardless_of_category():
    case = _case(category="CONSTRUCTION_MGMT", expected_document=["a.pdf"])
    fields, reasons = runner._evaluate_result(
        case, [{"document": "a.pdf", "path": "/a.pdf"}], _answer("Beton je C30/37.", ["a.pdf"]),
    )
    assert fields["unsupported_claim"] is True
    assert any("unsupported claim" in r for r in reasons)


def test_old_revision_leakage_forces_critical_error_regardless_of_criticality():
    case = _case(
        category="ADVERSARIAL", criticality="informational",
        expected_document=["a.pdf"], forbidden_document=["OLD"],
    )
    answer = _answer("Text.\n(Zdroj: a.pdf)", ["a.pdf"], ["X/OLD/a.pdf"])
    fields, _reasons = runner._evaluate_result(case, [], answer)
    assert fields["critical_error"] is True


# ---------------------------------------------------------------------------
# Full dataset — every real case evaluates without raising
# ---------------------------------------------------------------------------

def test_every_real_case_evaluates_without_raising(cases):
    """Pure dispatch over all 35 real cases with a neutral empty answer -
    confirms _evaluate_result never raises regardless of category/outcome
    combination in the actual dataset."""
    for case in cases:
        fields, _reasons = runner._evaluate_result(case, [], _answer("", []))
        assert isinstance(fields, dict)
        assert "passed" in fields


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _result(**overrides) -> runner.NdsCaseResult:
    data = dict(id="r1", category="TECHNICAL_QA", query="q", criticality="technical",
                expected_outcome="found", ground_truth_status="needs_review", passed=True)
    data.update(overrides)
    return runner.NdsCaseResult(**data)


def test_aggregate_counts_failed_and_critical_cases():
    results = [
        _result(id="ok", passed=True),
        _result(id="bad", passed=False, critical_error=True),
        _result(id="leak", passed=False, old_revision_leak_paths=["X/OLD/a.pdf"]),
    ]
    agg = runner.aggregate_nds_results(results)
    assert agg.case_count == 3
    assert agg.passed_count == 1
    assert set(agg.failed_case_ids) == {"bad", "leak"}
    assert agg.critical_error_case_ids == ["bad"]
    assert agg.old_revision_leak_case_ids == ["leak"]


def test_aggregate_document_search_category_kpis():
    results = [
        _result(id="a", category="DOCUMENT_SEARCH", document_found=True, document_rank=1, top1=True, top5=True),
        _result(id="b", category="DOCUMENT_SEARCH", document_found=True, document_rank=7, top1=False, top5=False),
        _result(id="c", category="DOCUMENT_SEARCH", document_found=None, top1=False, top5=False),
    ]
    agg = runner.aggregate_nds_results(results)
    row = agg.by_category["DOCUMENT_SEARCH"]
    assert row["document_found_rate"] == 1.0  # only the 2 measured cases, both found
    assert row["top1_accuracy"] == 0.5
    assert row["top5_accuracy"] == 0.5


def test_aggregate_errored_case_is_excluded_from_by_category_but_counted():
    results = [_result(id="boom", error="RuntimeError: boom", passed=False)]
    agg = runner.aggregate_nds_results(results)
    assert agg.errored_count == 1
    assert agg.by_category == {}
    assert agg.failed_case_ids == ["boom"]


# ---------------------------------------------------------------------------
# GO / NO-GO
# ---------------------------------------------------------------------------

def test_go_when_all_four_conditions_are_clean():
    agg = runner.aggregate_nds_results([_result(id="a", passed=True)])
    verdict = runner.nds_go_nogo(agg)
    assert verdict["verdict"] == "GO"
    assert verdict["blockers"] == []


@pytest.mark.parametrize("overrides", [
    dict(critical_error=True),
    dict(wrong_document_citation=True),
    dict(old_revision_leak_paths=["X/OLD/a.pdf"]),
    dict(unsupported_claim=True),
])
def test_any_single_blocking_condition_forces_no_go(overrides):
    results = [_result(id="a", passed=False, **overrides)]
    verdict = runner.nds_go_nogo(runner.aggregate_nds_results(results))
    assert verdict["verdict"] == "NO-GO"
    assert verdict["blockers"]


def test_harness_error_also_forces_no_go():
    results = [_result(id="boom", error="RuntimeError: boom", passed=False)]
    verdict = runner.nds_go_nogo(runner.aggregate_nds_results(results))
    assert verdict["verdict"] == "NO-GO"
    assert any("errored_cases" in b for b in verdict["blockers"])


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _synthetic_run() -> dict:
    results = [
        _result(id="nds-doc-01", category="DOCUMENT_SEARCH", document_found=True,
                document_rank=1, top1=True, top5=True, passed=True,
                failure_reasons=[]),
        _result(id="nds-adv-01", category="ADVERSARIAL", passed=False,
                old_revision_leak_paths=["X/OLD/a.pdf"], critical_error=True,
                failure_reasons=["OLD revision leakage: ['X/OLD/a.pdf']"]),
    ]
    agg = runner.aggregate_nds_results(results)
    verdict = runner.nds_go_nogo(agg)
    artifact = runner.NdsRunArtifact(
        timestamp="2026-08-12T00:00:00+00:00",
        git_sha="abc123",
        environment={"name": "production", "doc_count": 6342, "chunk_count": 157037,
                     "index_fingerprint": "deadbeefdeadbeef"},
        dataset_file=str(NDS_DATASET),
        dataset_version="0.1.0-draft",
        flags={"DOCUMENT_STATE_GATE_ENABLED": True, "EVIDENCE_RUNTIME_VALIDATION_ENABLED": True},
        case_count=2,
        cases=[r.to_dict() for r in results],
        aggregate=agg.to_dict(),
        verdict=verdict,
    )
    return artifact.to_dict()


def test_report_contains_all_required_sections():
    markdown = runner.render_nds_report(_synthetic_run())
    for needle in (
        "deadbeefdeadbeef",           # index fingerprint
        "## Počet testů: 2",          # case count
        "### DOCUMENT_SEARCH",        # per-category results
        "### ADVERSARIAL",
        "## Failed cases",
        "nds-adv-01",
        "## Kritické chyby",
        "## GO / NO-GO: **NO-GO**",
    ):
        assert needle in markdown, needle


def test_report_accepts_the_dataclass_directly():
    results = [_result(id="a", passed=True)]
    agg = runner.aggregate_nds_results(results)
    artifact = runner.NdsRunArtifact(
        timestamp="t", git_sha=None, environment={}, dataset_file="d.jsonl",
        dataset_version="", flags={}, case_count=1,
        cases=[r.to_dict() for r in results], aggregate=agg.to_dict(),
        verdict=runner.nds_go_nogo(agg),
    )
    assert "GO / NO-GO" in runner.render_nds_report(artifact)


def test_save_nds_report_uses_the_required_filename_pattern(tmp_path):
    artifact = runner.NdsRunArtifact(
        timestamp="t", git_sha=None, environment={}, dataset_file="d.jsonl",
        dataset_version="", flags={}, case_count=0, cases=[],
        aggregate=runner.NdsAggregate().to_dict(), verdict={"verdict": "GO", "blockers": []},
    )
    path = runner.save_nds_report(artifact, tmp_path)
    assert path.exists()
    assert re.match(r"acceptance_nds_smichov_\d{8}T\d{6}Z\.md", path.name)
    assert path.parent == tmp_path


# ---------------------------------------------------------------------------
# End-to-end smoke test (fixture corpus, mocked Ollama, no production access)
# ---------------------------------------------------------------------------

def _mock_ollama(monkeypatch, text: str = "Testovací odpověď.") -> None:
    payload = json.dumps({"body": [{"text": text, "zdroj_index": 0, "typ": "fakt"}], "nenalezeno": False})

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"response": payload}).encode()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: FakeResponse())


def test_end_to_end_run_against_fixture_leaves_runtime_files_untouched(monkeypatch, tmp_path):
    _mock_ollama(monkeypatch)
    before = _runtime_snapshot()

    run = runner.run_nds_acceptance(
        environment_name="fixture",
        dataset_path=NDS_DATASET,
        case_filter={"nds-doc-01"},
    )

    after = _runtime_snapshot()
    assert before == after, "a runtime file changed during the FAT run"

    assert run.case_count == 1
    assert len(run.cases) == 1
    assert run.cases[0]["id"] == "nds-doc-01"
    assert run.cases[0]["error"] is None

    path = runner.save_nds_report(run, tmp_path)
    assert path.exists()
    markdown = path.read_text(encoding="utf-8")
    assert "GO / NO-GO" in markdown
