"""PR8.4 — tests for the retrieval_hit_answer_miss diagnostic.

Pure measurement over recorded case dicts (shaped like `NdsCaseResult.to_dict()`
from `benchmark/acceptance_run_nds.py`). No ai_search import, no live retrieval
or LLM call - the fixtures below are trimmed, faithful copies of real FAT
artifact rows produced during the PR8.3/FAT-v2 runs referenced in the PR8.4
audit (nds-status-04, nds-qa-03, nds-qa-09, nds-adv-04).
"""
from __future__ import annotations

from benchmark.pr84_retrieval_answer_gap import (
    GAP_ABSTAINED_DESPITE_HIT,
    GAP_INFRA_ERROR,
    GAP_NO_CITATION,
    GAP_NONE,
    GAP_WRONG_FACT,
    classify_gap,
    compute_retrieval_hit_answer_miss,
    has_substantive_claim,
    is_infra_error,
    is_retrieval_hit_answer_miss,
    retrieval_hit,
)

# --- Fixtures: recorded rows (trimmed) ---------------------------------------

# PR8.3-subset run: document found (rank 2), model states the fact ("FERI
# s.r.o.") but the rendered answer has zero "(Zdroj: ...)" attribution because
# _render_answer_item's zdroj_index did not resolve -> unsupported_claim=True.
CASE_STATUS_04_NO_CITATION = {
    "id": "nds-status-04",
    "category": "DOCUMENT_STATUS",
    "query": "kdo je dodavatel monolitu?",
    "document_found": True,
    "document_rank": 2,
    "answer_correct": True,
    "citation_correct": None,
    "cited_sources": [],
    "answer_text": "Dodavatelem monolitu je společnost FERI s.r.o.",
    "unsupported_claim": True,
    "error": None,
}

# FAT v2 run: same case, same document, but the model happened to emit a
# resolvable zdroj_index this time -> citation present, no gap. Kept to prove
# the diagnostic (and the underlying defect) is intermittent, not a fixed
# per-case outcome - see module docstring.
CASE_STATUS_04_WITH_CITATION = {
    "id": "nds-status-04",
    "category": "DOCUMENT_STATUS",
    "document_found": True,
    "document_rank": 2,
    "answer_correct": True,
    "cited_sources": ["Dohoda o ukončení prací_FERIxSIS.pdf"],
    "answer_text": "... (Zdroj: Dohoda o ukončení prací_FERIxSIS.pdf)",
    "unsupported_claim": False,
    "error": None,
}

# FAT v2 run: correct document at rank 2 (D.1.4.j.1_01_TZ.pdf), the specific
# fact ("TP 124") never makes it into the answer at all - generic checklist
# summary instead, no "(Zdroj: ...)" anywhere -> both answer_correct=False
# (missing TP 124) and unsupported_claim=True.
CASE_QA_03_NO_CITATION = {
    "id": "nds-qa-03",
    "category": "TECHNICAL_QA",
    "query": "jaké jsou požadavky na krytí výztuže?",
    "document_found": True,
    "document_rank": 2,
    "answer_correct": False,
    "missing_phrases": ["TP 124"],
    "cited_sources": [],
    "answer_text": "Shrnutí:\n- Text obsahuje technické požadavky ...",
    "unsupported_claim": True,
    "error": None,
}

# FAT v2 run: correct document at rank 4, the correct fact IS written in the
# answer body ("14 geotermálních vrtů", "199 m") but every item is missing its
# "(Zdroj: ...)" suffix -> unsupported_claim=True despite the fact being
# right, which is exactly the "citovaný dokument musí být zdroj tvrzení"
# violation from the PR8.4 prompt.
CASE_QA_09_NO_CITATION = {
    "id": "nds-qa-09",
    "category": "TECHNICAL_QA",
    "query": "jak hluboké jsou geotermální vrty?",
    "document_found": True,
    "document_rank": 4,
    "answer_correct": False,
    "missing_phrases": [],
    "cited_sources": [],
    "answer_text": "Požadavek: - **14 geotermálních vrtů** o přibližné hloubce **199 m** ...",
    "unsupported_claim": True,
    "error": None,
}

# FAT v2 run: document found (rank 2), but the model backend itself timed out
# inside answer()'s _call_ollama - the "unsupported_claim=True" here is a
# side-effect of the fallback text ("Ollama je nedostupná: TimeoutError...")
# looking like a substantive claim, NOT an evidence/citation defect. Must be
# classified separately (GAP_INFRA_ERROR) and excluded from the headline gap.
CASE_ADV_04_INFRA_ERROR = {
    "id": "nds-adv-04",
    "category": "ADVERSARIAL",
    "query": "jaké zakázky má Stafitech na zdění?",
    "document_found": True,
    "document_rank": 2,
    "answer_correct": False,
    "missing_phrases": ["NOT250039", "NOT250304"],
    "cited_sources": [],
    "answer_text": "Ollama je nedostupná: TimeoutError. Nalezené citace zůstávají k dispozici.",
    "unsupported_claim": True,
    "error": None,
}

# FAT v2 run: correct document reached the pool (document_found=True), but
# the answer step abstained outright ("Nenalezeno v indexovaných
# dokumentech.") - the model/evidence-safety gate never engaged with the
# found row at all. Distinct from NO_CITATION (a claim was made, just
# uncited) and from WRONG_FACT (a claim was made and it was wrong) - this is
# the "evidence contains a fact but the answer ignores it" pattern the
# PR8.4 prompt names explicitly.
CASE_DOC_03_ABSTAINED_DESPITE_HIT = {
    "id": "nds-doc-03",
    "category": "DOCUMENT_SEARCH",
    "query": "najdi montážní návod Pentaflex KB80",
    "document_found": True,
    "document_rank": 3,
    "answer_correct": False,
    "missing_phrases": ["pentaflex"],
    "cited_sources": [],
    "answer_text": "Nenalezeno v indexovaných dokumentech.",
    "unsupported_claim": False,
    "error": None,
}

# FAT v2 run: sources ARE cited, but the stated fact is wrong/incomplete -
# the genuine "wrong fact despite good citation" pattern, kept separate from
# abstention above.
CASE_STATUS_03_WRONG_FACT = {
    "id": "nds-status-03",
    "category": "DOCUMENT_STATUS",
    "query": "existuje výkres výztuže 3.PP?",
    "document_found": True,
    "document_rank": 1,
    "answer_correct": False,
    "missing_phrases": [],
    "cited_sources": ["260624_NDS_přístavba zázemí KD č.75.pdf"],
    "answer_text": "Dne 21.7.2026 byl výkres výztuže stropu nad 3.PP zaslán GD mailem ...",
    "unsupported_claim": False,
    "error": None,
}

# Control: retrieval itself missed (document_found=False) - out of scope for
# this metric, must not be counted as an answer-layer gap.
CASE_RETRIEVAL_MISS = {
    "id": "control-retrieval-miss",
    "document_found": False,
    "answer_correct": False,
    "unsupported_claim": True,
    "answer_text": "Nenalezeno v indexovaných dokumentech.",
    "error": None,
}

# Control: fully healthy case - retrieval hit, cited, correct.
CASE_HEALTHY = {
    "id": "control-healthy",
    "document_found": True,
    "document_rank": 1,
    "answer_correct": True,
    "unsupported_claim": False,
    "cited_sources": ["Some Document.pdf"],
    "answer_text": "Fact X. (Zdroj: Some Document.pdf)",
    "error": None,
}


# --- retrieval_hit / is_infra_error --------------------------------------------

def test_retrieval_hit_reads_document_found():
    assert retrieval_hit(CASE_STATUS_04_NO_CITATION) is True
    assert retrieval_hit(CASE_RETRIEVAL_MISS) is False


def test_infra_error_detected_from_error_field():
    assert is_infra_error({"error": "TimeoutError: ...", "answer_text": ""}) is True


def test_infra_error_detected_from_ollama_fallback_text():
    assert is_infra_error(CASE_ADV_04_INFRA_ERROR) is True


def test_healthy_case_is_not_infra_error():
    assert is_infra_error(CASE_HEALTHY) is False


# --- classify_gap / is_retrieval_hit_answer_miss -------------------------------

def test_status_04_no_citation_is_no_citation_gap():
    assert classify_gap(CASE_STATUS_04_NO_CITATION) == GAP_NO_CITATION
    assert is_retrieval_hit_answer_miss(CASE_STATUS_04_NO_CITATION) is True


def test_status_04_with_citation_is_ok():
    """Same case/document, different LLM sample -> no gap. Confirms the
    defect is intermittent (depends on whether zdroj_index resolved), not a
    fixed property of the case."""
    assert classify_gap(CASE_STATUS_04_WITH_CITATION) == GAP_NONE
    assert is_retrieval_hit_answer_miss(CASE_STATUS_04_WITH_CITATION) is False


def test_qa_03_missing_fact_and_citation_is_no_citation_gap():
    # unsupported_claim is checked before answer_correct in classify_gap, so
    # a case that is both uncited AND factually wrong is labelled NO_CITATION
    # (the more specific/actionable diagnosis: fix the citation gap first).
    assert classify_gap(CASE_QA_03_NO_CITATION) == GAP_NO_CITATION
    assert is_retrieval_hit_answer_miss(CASE_QA_03_NO_CITATION) is True


def test_qa_09_correct_fact_without_citation_is_still_a_gap():
    """The fact is right, the citation is missing - still a defect per the
    'citovaný dokument musí být zdroj tvrzení' rule, so this must not be
    silently accepted just because the text happens to contain the right
    number."""
    assert classify_gap(CASE_QA_09_NO_CITATION) == GAP_NO_CITATION
    assert is_retrieval_hit_answer_miss(CASE_QA_09_NO_CITATION) is True


def test_adv_04_is_infra_error_not_evidence_gap():
    assert classify_gap(CASE_ADV_04_INFRA_ERROR) == GAP_INFRA_ERROR
    assert is_retrieval_hit_answer_miss(CASE_ADV_04_INFRA_ERROR) is False


def test_doc_03_abstention_despite_hit_is_its_own_category():
    """Retrieval found the document, the model said 'Nenalezeno' anyway -
    must not be lumped in with NO_CITATION (no claim was made, so
    unsupported_claim is False) or with a generic WRONG_FACT."""
    assert has_substantive_claim(CASE_DOC_03_ABSTAINED_DESPITE_HIT) is False
    assert classify_gap(CASE_DOC_03_ABSTAINED_DESPITE_HIT) == GAP_ABSTAINED_DESPITE_HIT
    assert is_retrieval_hit_answer_miss(CASE_DOC_03_ABSTAINED_DESPITE_HIT) is True


def test_status_03_cited_but_wrong_fact_is_wrong_fact_category():
    """Sources are cited (so it is not a citation-rendering gap) but the
    fact is wrong/incomplete - the genuine WRONG_FACT pattern."""
    assert has_substantive_claim(CASE_STATUS_03_WRONG_FACT) is True
    assert classify_gap(CASE_STATUS_03_WRONG_FACT) == GAP_WRONG_FACT
    assert is_retrieval_hit_answer_miss(CASE_STATUS_03_WRONG_FACT) is True


def test_retrieval_miss_is_out_of_scope():
    assert classify_gap(CASE_RETRIEVAL_MISS) == GAP_NONE
    assert is_retrieval_hit_answer_miss(CASE_RETRIEVAL_MISS) is False


def test_healthy_case_has_no_gap():
    assert classify_gap(CASE_HEALTHY) == GAP_NONE
    assert is_retrieval_hit_answer_miss(CASE_HEALTHY) is False


# --- compute_retrieval_hit_answer_miss (aggregate) -----------------------------

def test_aggregate_separates_citation_abstention_fact_and_infra_buckets():
    cases = [
        CASE_STATUS_04_NO_CITATION,
        CASE_QA_03_NO_CITATION,
        CASE_QA_09_NO_CITATION,
        CASE_ADV_04_INFRA_ERROR,
        CASE_DOC_03_ABSTAINED_DESPITE_HIT,
        CASE_STATUS_03_WRONG_FACT,
        CASE_RETRIEVAL_MISS,
        CASE_HEALTHY,
    ]
    summary = compute_retrieval_hit_answer_miss(cases)

    assert summary.case_count == 8
    # retrieval hit: all but CASE_RETRIEVAL_MISS
    assert summary.retrieval_hit_count == 7
    assert set(summary.no_citation_case_ids) == {"nds-status-04", "nds-qa-03", "nds-qa-09"}
    assert summary.abstained_case_ids == ["nds-doc-03"]
    assert summary.wrong_fact_case_ids == ["nds-status-03"]
    assert summary.infra_error_case_ids == ["nds-adv-04"]
    # the infra error must NOT be counted in the headline gap
    assert set(summary.gap_case_ids) == {
        "nds-status-04", "nds-qa-03", "nds-qa-09", "nds-doc-03", "nds-status-03",
    }
    assert summary.gap_count == 5
    assert summary.gap_rate == 5 / 7


def test_aggregate_gap_rate_is_none_when_nothing_reached_retrieval():
    summary = compute_retrieval_hit_answer_miss([CASE_RETRIEVAL_MISS])
    assert summary.retrieval_hit_count == 0
    assert summary.gap_rate is None
