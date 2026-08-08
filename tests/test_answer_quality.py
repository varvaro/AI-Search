"""Regression tests for RAG answer quality (ai_search.answer()): the structured
JSON output contract (checklist vs. concise schema), the hallucination guard,
per-item source attribution resolved deterministically from `results` (never
generated as free text by the model), the text-prompt fallback used when the
model/server doesn't honor the JSON `format` schema, and the metadata-only
confidence indicator. Retrieval itself (RRF/embedding/scoring/dedup/diversify)
is untouched and out of scope here.
"""
from __future__ import annotations
import json
import pytest
import ai_search


class FakeEmbeddings:
    def encode(self, texts, **kwargs):
        return [[1.0, 0.5, 0.1] for _ in texts]


def _mock_ollama(monkeypatch, responses):
    """Replaces urllib.request.urlopen with a fake that returns `responses[i]`
    (as the Ollama `response` field) for the i-th call, repeating the last item
    if there are more calls than responses. Returns the list of decoded request
    payloads, in call order, so tests can inspect prompt/format/etc."""
    calls = []
    remaining = list(responses)

    class FakeResponse:
        def __init__(self, text):
            self._text = text
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": self._text}).encode()

    def fake_urlopen(request, timeout=0):
        calls.append(json.loads(request.data.decode()))
        text = remaining.pop(0) if remaining else calls[-1].get("_last_response", "")
        return FakeResponse(text)

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    return calls


def _rows(*documents):
    return [
        {"document": name, "path": f"/proj/{name}", "project": "Projekt", "heading": "", "quote": f"Obsah dokumentu {name}.",
         "score": 1.0 - i * 0.05, "match": {"fts_hit": True, "vector_hit": True, "semantic_similarity": 0.7, "filename_match": False}}
        for i, name in enumerate(documents)
    ]


# ---------------------------------------------------------------------------
# Query classification: checklist/completeness vs. simple lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "Co chybí k předání základové desky investorovi?",
    "Jaké doklady musí dodat zhotovitel po betonáži?",
    "Jaká dokumentace je potřeba?",
    "Zkontroluj kompletnost předání.",
])
def test_is_checklist_query_detects_completeness_questions(query):
    assert ai_search._is_checklist_query(query) is True


@pytest.mark.parametrize("query", ["Pentaflex", "kniha betonů", "změnový list", "FERI"])
def test_is_checklist_query_false_for_simple_lookups(query):
    assert ai_search._is_checklist_query(query) is False


# ---------------------------------------------------------------------------
# Confidence indicator (metadata-only, no extra LLM call)
# ---------------------------------------------------------------------------

def test_answer_confidence_green_for_multiple_independent_strong_sources():
    rows = [
        {"document": "a.pdf", "path": "/proj/A/a.pdf", "match": {"fts_hit": True, "semantic_similarity": 0.8}},
        {"document": "b.pdf", "path": "/proj/B/b.pdf", "match": {"fts_hit": True, "semantic_similarity": 0.7}},
        {"document": "c.pdf", "path": "/proj/C/c.pdf", "match": {"fts_hit": False, "semantic_similarity": 0.2}},
    ]
    level, reason = ai_search._answer_confidence("cokoliv", rows)
    assert level == "green" and "zdrojů" in reason


def test_answer_confidence_yellow_for_single_relevant_source():
    rows = [{"document": "a.pdf", "match": {"fts_hit": True, "semantic_similarity": 0.6}}]
    level, reason = ai_search._answer_confidence("cokoliv", rows)
    assert level == "yellow" and "jeden" in reason


def test_answer_confidence_red_for_weak_matches_only():
    rows = [
        {"document": "a.pdf", "match": {"fts_hit": False, "semantic_similarity": 0.1}},
        {"document": "b.pdf", "match": {"fts_hit": False, "semantic_similarity": 0.05}},
    ]
    level, reason = ai_search._answer_confidence("cokoliv", rows)
    assert level == "red" and "málo podkladů" in reason


def test_answer_confidence_downgraded_to_yellow_when_all_sources_share_folder():
    """3 distinct documents with strong hits would normally be 'green', but if
    they all sit in the same folder they aren't independent corroboration - see
    the 'kontrolní den' duplication case from the production audit."""
    rows = [
        {"document": f"kd_{i}.pdf", "path": f"/proj/kontrolni_dny/kd_{i}.pdf", "match": {"fts_hit": True, "semantic_similarity": 0.8}}
        for i in range(3)
    ]
    level, reason = ai_search._answer_confidence("cokoliv", rows)
    assert level == "yellow" and "stejné složky" in reason


def test_answer_confidence_mentions_same_type_when_folder_and_extension_both_shared():
    rows = [
        {"document": f"kd_{i}.pdf", "path": f"/proj/kontrolni_dny/kd_{i}.pdf", "extension": "pdf", "match": {"fts_hit": True, "semantic_similarity": 0.8}}
        for i in range(3)
    ]
    level, reason = ai_search._answer_confidence("cokoliv", rows)
    assert level == "yellow" and "stejného typu dokumentu" in reason


def test_answer_confidence_green_unaffected_when_folders_differ():
    rows = [
        {"document": f"doc_{i}.pdf", "path": f"/proj/folder_{i}/doc_{i}.pdf", "match": {"fts_hit": True, "semantic_similarity": 0.8}}
        for i in range(3)
    ]
    level, _reason = ai_search._answer_confidence("cokoliv", rows)
    assert level == "green"


# ---------------------------------------------------------------------------
# Checklist-query confidence: source *type* must also be relevant, not just
# match strength/diversity - see the "Co chybí k předání základové desky
# investorovi?" production case (10 contract/offer documents, strong lexical
# match, but LLM correctly answered "nenalezeno" while confidence said green).
# ---------------------------------------------------------------------------

def _strong_row(document, folder):
    return {"document": document, "path": f"/proj/{folder}/{document}", "match": {"fts_hit": True, "semantic_similarity": 0.8}}


def test_checklist_confidence_is_not_high_when_sources_are_all_contract_documents():
    rows = [_strong_row(name, f"folder_{i}") for i, name in enumerate([
        "SOD_SUB_vzorová NDS_návrh_rev.BB.docx", "Průvodní dopis_Přístavba garáží.docx",
        "SOD_NDS_Perla x SIS_Návrh RP.docx", "Cenová nabídka_garáže.xlsx",
        "Objednávka_dodatek_1.pdf", "Rozpočet_garáže_v2.xlsx",
        "NOT262012_SoD_HAUS365.pdf", "NOT261709_Martin Bičík_inženýring.pdf",
        "Smlouva o dílo_dodatek.docx", "Nabídka_generální dodavatel.docx",
    ])]
    level, reason = ai_search._answer_confidence("Co chybí k předání základové desky investorovi?", rows)
    assert level != "green"
    assert "smluvní" in reason or "technickou" in reason


def test_checklist_confidence_can_stay_high_with_technical_sources():
    rows = [_strong_row(name, f"folder_{i}") for i, name in enumerate([
        "TP_beton_monoliticke_konstrukce.pdf", "KZP_zakladova_deska.pdf",
        "Protokol_zkousky_betonu_01.pdf",
    ])]
    level, _reason = ai_search._answer_confidence("Jaké doklady musí dodat zhotovitel po betonáži?", rows)
    assert level == "green"


def test_non_checklist_confidence_unaffected_by_document_type_heuristic():
    """Regression guard: a plain product-lookup query ("Pentaflex") must not be
    subject to the checklist-only contract/technical downgrade at all - existing
    behaviour (pure match-strength/diversity signals) stays identical."""
    rows = [_strong_row(name, f"folder_{i}") for i, name in enumerate([
        "SOD_SUB_vzorová NDS_návrh_rev.BB.docx", "Cenová nabídka_garáže.xlsx", "Objednávka_dodatek_1.pdf",
    ])]
    level, reason = ai_search._answer_confidence("Pentaflex", rows)
    assert level == "green"  # same as before: 3 distinct docs, different folders, 2+ strong hits
    assert "smluvní" not in reason and "technickou" not in reason


# ---------------------------------------------------------------------------
# End-to-end answer() behaviour with structured JSON output (Ollama mocked -
# deterministic, no live server needed)
# ---------------------------------------------------------------------------

def test_structured_json_schema_requested_for_checklist_query(monkeypatch):
    """Test 1: "Co chybí k předání základové desky investorovi?" - must request
    the STRUCTURED_ANSWER_SCHEMA, must include the hallucination guard in the
    prompt, and the rendered answer must have the checklist sections with
    sources resolved only from the documents actually retrieved."""
    rows = _rows("TP beton monolitické konstrukce.pdf", "predavaci_protokol.pdf", "kontrola_vyztuze.pdf")
    payload = {
        "shrnuti": "Chybí zejména předávací protokol a doklady o kontrole výztuže.",
        "oblasti": [
            {"nazev": "Dokumentace betonáže", "polozky": [
                {"text": "Protokoly zkoušek betonu", "zdroj_index": 1, "typ": "fakt"},
            ]},
            {"nazev": "Předání investorovi", "polozky": [
                {"text": "Předávací protokol musí být doložen", "zdroj_index": 2, "typ": "pozadavek"},
            ]},
        ],
        "nenalezene": ["Zápis o předání kabeláže"],
    }
    calls = _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer("Co chybí k předání základové desky investorovi?", rows)

    assert calls[0]["format"] == ai_search.STRUCTURED_ANSWER_SCHEMA
    assert "nevymýšlej" in calls[0]["prompt"] and "zdroj_index" in calls[0]["prompt"]
    assert result["citations"] == rows  # only ever the documents actually retrieved

    text = result["answer"]
    assert "Požadované dokumenty / kroky" in text
    assert "Nenalezené informace" in text and "Zápis o předání kabeláže" in text
    assert "Zdroje:" in text
    assert "TP beton monolitické konstrukce.pdf" in text and "predavaci_protokol.pdf" in text
    assert "kontrola_vyztuze.pdf" not in text  # never cited by the model -> not listed
    assert "Jistota odpovědi" in text
    assert result["confidence"] in {"green", "yellow", "red"}


def test_structured_json_schema_for_documents_after_concrete_pour_query(monkeypatch):
    """Test 2: "Jaké doklady musí dodat zhotovitel po betonáži?" - checklist
    schema + citations limited to the actually supplied documents."""
    rows = _rows("dodaci_listy_betonu.pdf", "protokoly_zkousek_betonu.pdf")
    payload = {
        "shrnuti": "Zhotovitel musí doložit dodací listy betonu a protokoly zkoušek.",
        "oblasti": [{"nazev": "Doklady o betonu", "polozky": [
            {"text": "Dodací listy betonu", "zdroj_index": 1, "typ": "fakt"},
            {"text": "Protokoly zkoušek betonu", "zdroj_index": 2, "typ": "fakt"},
        ]}],
        "nenalezene": [],
    }
    calls = _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer("Jaké doklady musí dodat zhotovitel po betonáži?", rows)

    assert calls[0]["format"] == ai_search.STRUCTURED_ANSWER_SCHEMA
    assert {row["document"] for row in result["citations"]} == {row["document"] for row in rows}
    text = result["answer"]
    assert "dodaci_listy_betonu.pdf" in text and "protokoly_zkousek_betonu.pdf" in text
    assert "Nenalezené informace" in text and "Žádné" in text  # empty list renders as "Žádné"


def test_concise_json_schema_used_for_simple_product_lookup(monkeypatch):
    """Test 3: "Pentaflex" - short technical answer, no forced checklist report."""
    rows = _rows("PENTAFLEX_KB80_Montážní návod.pdf")
    payload = {"body": [{"text": "Pentaflex je těsnicí pás pro dilatační spáry.", "zdroj_index": 1, "typ": "fakt"}], "nenalezeno": False}
    calls = _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer("Pentaflex", rows)

    assert calls[0]["format"] == ai_search.CONCISE_ANSWER_SCHEMA
    text = result["answer"]
    assert "Požadované dokumenty / kroky" not in text and "Shrnutí:" not in text  # concise has no checklist headings
    assert "Pentaflex je těsnicí pás" in text
    assert "(Zdroj: PENTAFLEX_KB80_Montážní návod.pdf)" in text  # inline source right under the claim
    assert result["citations"] == rows


def test_concise_nenalezeno_true_renders_explicit_not_found_phrase(monkeypatch):
    rows = _rows("nesouvisejici_dokument.pdf")
    payload = {"body": [], "nenalezeno": True}
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer("Nějaký produkt XYZ", rows)
    assert "Nenalezeno v indexovaných dokumentech" in result["answer"]


def test_out_of_range_source_index_is_dropped_not_fabricated(monkeypatch):
    """A model-supplied zdroj_index outside the retrieved documents must never
    resolve to a document name - the claim is kept, but with no source line,
    rather than risking a wrong/fabricated citation."""
    rows = _rows("a.pdf", "b.pdf")
    payload = {"body": [{"text": "Tvrzení s neplatným zdrojem.", "zdroj_index": 99, "typ": "fakt"}], "nenalezeno": False}
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer("Nějaký jednoduchý dotaz", rows)
    text = result["answer"]
    assert "Tvrzení s neplatným zdrojem." in text
    assert "(Zdroj:" not in text  # no source could be resolved -> none shown, none invented


def test_falls_back_to_text_prompt_when_json_output_invalid(monkeypatch):
    """If the model/server ignores or doesn't support the JSON `format` schema
    (e.g. a different LLM backend), answer() must retry with the old free-text
    prompt instead of failing outright."""
    rows = _rows("a.pdf", "b.pdf")
    calls = _mock_ollama(monkeypatch, ["Toto neni platny JSON vubec.", "Shrnutí:\n- Testovací odpověď.\n\nZdroje:\n- a.pdf"])
    result = ai_search.answer("Co chybí k předání?", rows)

    assert len(calls) == 2
    assert "format" in calls[0]  # first attempt requests structured JSON
    assert "format" not in calls[1]  # fallback call is plain free-text prompting
    assert "Testovací odpověď" in result["answer"]
    assert "Jistota odpovědi" in result["answer"]
    assert result["citations"] == rows


def test_falls_back_to_text_prompt_when_structured_call_times_out(monkeypatch):
    """Root-cause fix: a TimeoutError/ConnectionError on the *first* (structured
    JSON, format=schema) Ollama call must NOT return "Ollama je nedostupná"
    immediately - it must retry with the same free-text prompt/context/results,
    exactly like an invalid-JSON response already does, so a slow structured
    response degrades to the old behaviour instead of producing an apology with
    no real answer."""
    rows = _rows("a.pdf", "b.pdf")
    calls = []

    class FakeResponse:
        def __init__(self, text):
            self._text = text
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": self._text}).encode()

    def fake_urlopen(request, timeout=0):
        payload = json.loads(request.data.decode())
        calls.append(payload)
        if "format" in payload:
            raise TimeoutError("timed out")
        return FakeResponse("Shrnutí:\n- Odpověď z fallbacku po timeoutu.\n\nZdroje:\n- a.pdf")

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    result = ai_search.answer("Co chybí k předání základové desky investorovi?", rows)

    assert len(calls) == 2
    assert "format" in calls[0]  # first attempt: structured JSON, which times out
    assert "format" not in calls[1]  # fallback: plain free-text prompt, same model
    assert calls[1]["prompt"].endswith(calls[0]["prompt"].split("ZDROJE:\n", 1)[1])  # same context/results reused
    assert result["answer"]  # never empty
    assert "Odpověď z fallbacku po timeoutu" in result["answer"]
    assert "error" not in result  # a successful fallback is a normal answer(), not an error response
    assert result["citations"] == rows


def test_falls_back_to_text_prompt_on_connection_error(monkeypatch):
    """Same guarantee for a connection-level failure (not just timeout)."""
    rows = _rows("a.pdf")
    calls = []

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": "Pentaflex je těsnicí pás. Zdroj: a.pdf"}).encode()

    def fake_urlopen(request, timeout=0):
        payload = json.loads(request.data.decode())
        calls.append(payload)
        if "format" in payload:
            raise ConnectionError("connection refused")
        return FakeResponse()

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    result = ai_search.answer("Pentaflex", rows)

    assert len(calls) == 2
    assert result["answer"]
    assert "error" not in result


def test_returns_error_dict_only_when_fallback_also_fails(monkeypatch):
    """If BOTH the structured call and the free-text fallback fail, answer()
    still degrades gracefully (never raises) and keeps the original citations -
    this is the one legitimate case for the "Ollama je nedostupná" message."""
    rows = _rows("a.pdf")

    def fake_urlopen(request, timeout=0):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ai_search.urllib.request, "urlopen", fake_urlopen)
    result = ai_search.answer("Pentaflex", rows)

    assert "Ollama je nedostupná" in result["answer"]
    assert result["citations"] == rows
    assert result["error"]


def test_answer_never_returns_citations_beyond_what_was_retrieved(monkeypatch):
    rows = _rows("a.pdf", "b.pdf")
    payload = {"shrnuti": "ok", "oblasti": [], "nenalezene": []}
    _mock_ollama(monkeypatch, [json.dumps(payload)])
    result = ai_search.answer("Jaké jsou požadavky investora?", rows)
    assert len(result["citations"]) == len(rows)
    assert {c["document"] for c in result["citations"]} <= {"a.pdf", "b.pdf"}


def test_empty_results_still_reports_low_confidence_without_calling_ollama(monkeypatch):
    called = []
    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: called.append(1))
    result = ai_search.answer("cokoliv", [])
    assert result["citations"] == [] and result["confidence"] == "red" and not called
