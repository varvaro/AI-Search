"""PR9.4.2 — Document-class affinity + lookup window unit tests.

Covers: flag default/OFF identity, query/document classifiers, scoring
bounds, status-query safety (LOI/signed must not prove a signed SoD),
lookup-only window 30→80, and a no-hardcode scan of the production module.
"""
from __future__ import annotations

import inspect

import pytest

import ai_search
import ai_search_config
import document_class_affinity as dca


class _FakeEmbeddings:
    name = "fake"

    def encode(self, texts, **kw):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def backend(tmp_path):
    root = tmp_path / "Projekt"
    root.mkdir()
    (root / "alpha.txt").write_text("ALPHA unikátní fulltextový výraz.")
    (root / "smlouva_dodavatele.pdf").write_text("smluvní ujednání stran.")
    embeddings = _FakeEmbeddings()
    db = tmp_path / "index.sqlite3"
    lance = tmp_path / "lance"
    ai_search.sync(root, db, lance, embeddings)
    return root, tmp_path, embeddings


# --- flag / identity ----------------------------------------------------------

def test_flag_default_is_on():
    assert ai_search_config.DOCUMENT_CLASS_AFFINITY_ENABLED is True
    assert ai_search.DOCUMENT_CLASS_AFFINITY_ENABLED is True


def test_flag_off_has_no_trace_key(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "DOCUMENT_CLASS_AFFINITY_ENABLED", False)
    rows = ai_search.search(
        "smlouva dodavatele", state / "index.sqlite3", state / "lance", embeddings,
    )
    assert rows
    assert "document_class_affinity" not in rows[0]["match"]


def test_flag_on_is_deterministic(backend, monkeypatch):
    root, state, embeddings = backend
    monkeypatch.setattr(ai_search, "DOCUMENT_CLASS_AFFINITY_ENABLED", True)
    a = ai_search.search(
        "smlouva dodavatele", state / "index.sqlite3", state / "lance", embeddings,
    )
    b = ai_search.search(
        "smlouva dodavatele", state / "index.sqlite3", state / "lance", embeddings,
    )
    assert [r["score"] for r in a] == [r["score"] for r in b]
    assert a[0]["match"]["document_class_affinity"] == b[0]["match"]["document_class_affinity"]
    trace = a[0]["match"]["document_class_affinity"]
    assert {"query_class", "document_class", "bonus", "reason"} <= set(trace)


def test_flag_off_does_not_change_scores_vs_unknown_on(backend, monkeypatch):
    """Class layer itself is a no-op when OFF. An UNKNOWN query with flag ON
    must also keep scores identical (bonus 0)."""
    root, state, embeddings = backend
    q = "ALPHA unikátní"
    monkeypatch.setattr(ai_search, "DOCUMENT_CLASS_AFFINITY_ENABLED", False)
    off = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings)
    monkeypatch.setattr(ai_search, "DOCUMENT_CLASS_AFFINITY_ENABLED", True)
    on = ai_search.search(q, state / "index.sqlite3", state / "lance", embeddings)
    assert [r["score"] for r in off] == [r["score"] for r in on]
    assert "document_class_affinity" not in off[0]["match"]
    assert on[0]["match"]["document_class_affinity"]["bonus"] == 0.0


# --- query class --------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("výkres základové desky", dca.QueryClass.DRAWING),
    ("schéma vyztužení", dca.QueryClass.DRAWING),
    ("úkoly z kontrolních dnů", dca.QueryClass.MINUTES),
    ("zápis z KD", dca.QueryClass.MINUTES),
    ("KD", dca.QueryClass.MINUTES),
    ("harmonogram výstavby", dca.QueryClass.SCHEDULE),
    ("HMG", dca.QueryClass.SCHEDULE),
    ("rozpočet garáží", dca.QueryClass.BUDGET),
    ("smlouva na monolit", dca.QueryClass.CONTRACT),
    ("SoD", dca.QueryClass.CONTRACT),
    ("technická zpráva", dca.QueryClass.TECHNICAL_REPORT),
    ("technické zprávě k desce", dca.QueryClass.TECHNICAL_REPORT),
    ("jaká je tloušťka základové desky", dca.QueryClass.UNKNOWN),
    ("technická specifikace betonu", dca.QueryClass.UNKNOWN),
])
def test_query_class(query, expected):
    assert dca.classify_query(query) is expected


def test_kd_requires_token_boundary():
    assert dca.classify_query("KD") is dca.QueryClass.MINUTES
    assert dca.classify_query("zápis z KD č. 12") is dca.QueryClass.MINUTES
    assert dca.classify_query("skladovací") is dca.QueryClass.UNKNOWN
    assert dca.classify_query("ukolKD") is dca.QueryClass.UNKNOWN


def test_plan_is_drawing_only_with_drawing_context():
    assert dca.classify_query("plán") is dca.QueryClass.UNKNOWN
    assert dca.classify_query("plán výkresové dokumentace") is dca.QueryClass.DRAWING
    assert dca.classify_query("plán kontrolních dnů") is dca.QueryClass.MINUTES


# --- document class -----------------------------------------------------------

def test_budget_from_filename_only():
    assert dca.classify_document("rozpočet_garaze.xlsx", "/docs/finance/rozpočet_garaze.xlsx") is dca.DocumentClass.BUDGET


def test_parent_folder_rozpocet_plus_cn_is_not_budget():
    assert dca.classify_document("CN.xlsx", "/docs/rozpočet/CN.xlsx") is not dca.DocumentClass.BUDGET
    assert dca.classify_document("CN.xlsx", "/docs/nabidky/CN.xlsx") is not dca.DocumentClass.BUDGET


def test_bare_cn_is_not_budget():
    assert dca.classify_document("CN.xlsx", "/docs/CN.xlsx") is not dca.DocumentClass.BUDGET
    assert dca.classify_document("cn.pdf", "") is not dca.DocumentClass.BUDGET


def test_sod_filename_is_contract():
    assert dca.classify_document("SoD_dodavatel.pdf", "/contracts/SoD_dodavatel.pdf") is dca.DocumentClass.CONTRACT


def test_loi_is_not_full_contract():
    assert dca.classify_document("LOI_dodavatel_signed.pdf", "/letters/LOI_dodavatel_signed.pdf") is dca.DocumentClass.LETTER_OF_INTENT
    assert dca.classify_document("LOI_dodavatel_signed.pdf", "") is not dca.DocumentClass.CONTRACT


def test_minutes_from_kd_filename():
    assert dca.classify_document("KD č.72.xlsx", "/zapis/KD č.72.xlsx") is dca.DocumentClass.MINUTES


def test_schedule_from_harmonogram_or_hmg():
    assert dca.classify_document("harmonogram_vystavby.xlsx", "") is dca.DocumentClass.SCHEDULE
    assert dca.classify_document("HMG_2026.xlsx", "/plan/HMG_2026.xlsx") is dca.DocumentClass.SCHEDULE


def test_technical_report_from_filename():
    assert dca.classify_document("technická zpráva.pdf", "/reports/technická zpráva.pdf") is dca.DocumentClass.TECHNICAL_REPORT


def test_drawing_from_vykres_or_schema():
    assert dca.classify_document("výkres_desky.pdf", "") is dca.DocumentClass.DRAWING
    assert dca.classify_document("schéma vyztužení.pdf", "/docs/schéma vyztužení.pdf") is dca.DocumentClass.DRAWING


def test_arbitrary_pdf_is_not_a_drawing():
    assert dca.classify_document("zapis_jednani.pdf", "/docs/zapis_jednani.pdf") is dca.DocumentClass.UNKNOWN


def test_signed_alone_is_not_contract_class():
    assert dca.classify_document("dodatek_signed.pdf", "") is not dca.DocumentClass.CONTRACT
    assert dca.classify_document("podepsané_potvrzení.pdf", "") is not dca.DocumentClass.CONTRACT


# --- scoring ------------------------------------------------------------------

def test_exact_class_match_bonus():
    detail = dca.compute_class_affinity("smlouva na dílo", "SoD_dodavatel.pdf", "/c/SoD_dodavatel.pdf")
    assert detail.bonus == pytest.approx(0.03)
    assert detail.reason == "class_match"


def test_clear_mismatch_is_capped():
    detail = dca.compute_class_affinity("smlouva na dílo", "KD č.12.xlsx", "/zapis/KD č.12.xlsx")
    assert detail.bonus == pytest.approx(-0.015)
    assert detail.bonus >= -0.015
    assert detail.reason == "class_mismatch"


def test_unknown_is_zero():
    assert dca.compute_class_affinity("tloušťka desky", "SoD.pdf", "").bonus == 0.0
    assert dca.compute_class_affinity("smlouva", "poznamky.txt", "").bonus == 0.0


def test_no_final_newer_old_effect():
    q = "smlouva na dílo"
    a = dca.compute_class_affinity(q, "SoD_final.pdf", "")
    b = dca.compute_class_affinity(q, "SoD_návrh.pdf", "")
    c = dca.compute_class_affinity(q, "OLD_SoD.pdf", "")
    assert a.bonus == b.bonus == c.bonus == pytest.approx(0.03)


def test_no_signed_bonus():
    q = "smlouva na dílo"
    unsigned = dca.compute_class_affinity(q, "SoD_dodavatel.pdf", "")
    signed = dca.compute_class_affinity(q, "SoD_dodavatel_signed.pdf", "")
    assert unsigned.bonus == signed.bonus == pytest.approx(0.03)


# --- status safety ------------------------------------------------------------

def test_status_query_plus_loi_signed_is_not_positive_contract_proof():
    q = "existuje podepsaná SoD na monolit?"
    detail = dca.compute_class_affinity(q, "LOI_dodavatel_signed.pdf", "/letters/LOI_dodavatel_signed.pdf")
    assert detail.bonus == 0.0
    assert detail.bonus <= 0.0
    assert detail.status_query is True
    assert detail.document_class is dca.DocumentClass.LETTER_OF_INTENT


@pytest.mark.parametrize("query", [
    "je podepsaná smlouva",
    "máme podepsanou SoD",
    "je smlouva podepsaná",
    "status smlouvy",
    "existuje podepsaná smlouva na monolit?",
])
def test_status_queries_always_zero(query):
    sod = dca.compute_class_affinity(query, "SoD_dodavatel.pdf", "")
    loi = dca.compute_class_affinity(query, "LOI_signed.pdf", "")
    assert sod.bonus == 0.0
    assert loi.bonus == 0.0
    assert sod.status_query is True


def test_signed_without_subject_relevance_is_zero():
    detail = dca.compute_class_affinity(
        "existuje podepsaná smlouva?",
        "potvrzeni_signed.pdf",
        "/misc/potvrzeni_signed.pdf",
    )
    assert detail.bonus == 0.0


def test_loi_never_gets_contract_match_even_on_document_search():
    detail = dca.compute_class_affinity("smlouva na monolit", "LOI_dodavatel.pdf", "")
    assert detail.bonus == 0.0
    assert detail.document_class is dca.DocumentClass.LETTER_OF_INTENT
    assert detail.reason == "loi_excluded"


# --- window (lookup-only) -----------------------------------------------------

def test_lookup_rerank_window_is_80_question_unchanged():
    assert ai_search.RERANK_POOL_SIZE == 80
    assert ai_search.QA_RERANK_POOL_SIZE == 300
    assert ai_search.RETRIEVAL_POOL_SIZE == 100
    assert ai_search.QA_RETRIEVAL_POOL_SIZE == 500


def test_window_change_is_lookup_branch_only(backend):
    root, state, embeddings = backend
    lookup = ai_search.SearchTrace()
    ai_search.search(
        "ALPHA", state / "index.sqlite3", state / "lance", embeddings,
        is_question=False, trace=lookup,
    )
    question = ai_search.SearchTrace()
    ai_search.search(
        "Jaká je tloušťka desky?", state / "index.sqlite3", state / "lance", embeddings,
        is_question=True, trace=question,
    )
    assert lookup.intent["rerank_k"] == 80
    assert lookup.intent["retrieval_k"] == 100
    assert question.intent["rerank_k"] == 300
    assert question.intent["retrieval_k"] == 500


def test_search_branches_use_distinct_pool_constants():
    src = inspect.getsource(ai_search.search)
    assert "QA_RERANK_POOL_SIZE" in src
    assert "RERANK_POOL_SIZE" in src
    assert "if is_question" in src


# --- no hardcode / no body ----------------------------------------------------

def _source_without_comments_and_docstrings(mod) -> str:
    src = inspect.getsource(mod)
    for doc in (mod.__doc__ or "",):
        src = src.replace(doc, "", 1)
    lines = []
    for line in src.splitlines():
        stripped = line.split("#", 1)[0]
        lines.append(stripped)
    return "\n".join(lines)


def test_module_never_reads_heading_or_quote():
    code = _source_without_comments_and_docstrings(dca)
    assert "heading" not in code
    assert "quote" not in code


def test_module_has_no_hardcoded_project_values():
    code = dca.fold(_source_without_comments_and_docstrings(dca)).replace(" ", "")
    for forbidden in (
        "feri", "illichman", "stafitech", "safetypeak",
        "not250039", "not251110", "cbs02", "smichov",
    ):
        assert forbidden not in code, f"unexpected project-specific literal: {forbidden}"
