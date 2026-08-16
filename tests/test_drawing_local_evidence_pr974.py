"""PR9.7.4 — document-local evidence selection for drawing navigation."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import ai_search
import drawing_local_evidence as dle
import drawing_navigation as dn
from drawing_navigation import DrawingSubtype


QUERY_PLAN = "najdi mi výkres půdorys retenční nádrže?"


def _row(
    document,
    quote="",
    heading="",
    path="",
    document_id=1,
    score=1.0,
    project="P",
    source="Dokument",
):
    return {
        "document": document,
        "path": path or f"/proj/{document}",
        "relative_path": path or f"proj/{document}",
        "quote": quote,
        "heading": heading,
        "project": project,
        "score": score,
        "source": source,
        "document_id": document_id,
        "chunk_id": f"current:{document_id}",
    }


def _database(tmp_path: Path, row: dict, chunks: list[tuple[int, str, str]]) -> Path:
    database_dir = tmp_path / "database"
    database_dir.mkdir(exist_ok=True)
    db_path = database_dir / "project.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE documents(
                id INTEGER PRIMARY KEY,
                path TEXT,
                name TEXT,
                content_hash TEXT
            );
            CREATE TABLE chunks(
                id TEXT PRIMARY KEY,
                document_id INTEGER,
                ordinal INTEGER,
                heading TEXT,
                text TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO documents(id,path,name,content_hash) VALUES(?,?,?,?)",
            (row["document_id"], row["path"], row["document"], "digest"),
        )
        connection.executemany(
            "INSERT INTO chunks(id,document_id,ordinal,heading,text) VALUES(?,?,?,?,?)",
            [
                (
                    f"digest:{ordinal}",
                    row["document_id"],
                    ordinal,
                    heading,
                    text,
                )
                for ordinal, heading, text in chunks
            ],
        )
    return database_dir


def _disable_other_flags(monkeypatch):
    monkeypatch.setattr(ai_search, "OLD_REVISION_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_search, "DOCUMENT_STATE_GATE_ENABLED", False)
    monkeypatch.setattr(ai_search, "EVIDENCE_RUNTIME_VALIDATION_ENABLED", False)
    monkeypatch.setattr(ai_search, "CITATION_CONTRACT_ENABLED", False)
    monkeypatch.setattr(ai_search, "JSON_SENTINEL_FALLBACK_ENABLED", False)
    monkeypatch.setattr(ai_search, "QUERY_FOCUSED_CONTEXT_PACKING_ENABLED", False)
    monkeypatch.setattr(ai_search, "ENTITY_HINTS_ENABLED", False)


def _mock_json_ollama(monkeypatch, sink: list):
    def fake_call(model, prompt, format_schema=None, timeout=240):
        sink.append({"model": model, "prompt": prompt})
        return json.dumps(
            {
                "body": [
                    {
                        "text": "Smlouva je podepsaná.",
                        "zdroj_index": 1,
                        "typ": "fakt",
                    }
                ],
                "nenalezeno": False,
            }
        )

    monkeypatch.setattr(ai_search, "_call_ollama", fake_call)


def test_vzt_floor_plan_replaces_airflow_quote(tmp_path):
    row = _row(
        "D.1.4.b.2-VZT-101-Půdorys 3.PP.pdf",
        quote="5400 m³/h",
    )
    database_dir = _database(
        tmp_path,
        row,
        [
            (0, "VZT", "5400 m³/h"),
            (13, "22.4 m²", "Nádrž dešťové\n vody 205 m3\n -10,240"),
            (90, "OBSAH", "VZT – Půdorys 3.PP"),
        ],
    )

    evidence = dle.select_local_evidence(
        QUERY_PLAN,
        row["document_id"],
        row,
        DrawingSubtype.PLAN,
        database_dir=database_dir,
    )

    assert evidence is not None
    assert evidence.chunk_id == "digest:13"
    assert evidence.quote == "Nádrž dešťové vody 205 m3 -10,240"
    enriched = dle.enrich_results(
        QUERY_PLAN,
        [row],
        (DrawingSubtype.PLAN, DrawingSubtype.GENERIC_DRAWING),
        database_dir=database_dir,
    )
    assert enriched[0]["quote"] == "Nádrž dešťové vody 205 m3 -10,240"
    assert enriched[0]["chunk_id"] == "digest:13"
    assert enriched[0]["score"] == row["score"]


def test_admin_pdf_stays_unchanged(tmp_path):
    row = _row(
        "Rozhodnutí o povolení.pdf",
        quote="Souřadnice retenční nádrže",
    )
    database_dir = _database(
        tmp_path,
        row,
        [(1, "Rozhodnutí", "Nádrž dešťové vody 205 m3")],
    )
    rows = [row]

    enriched = dle.enrich_results(
        QUERY_PLAN,
        rows,
        (DrawingSubtype.PLAN,),
        database_dir=database_dir,
    )

    assert enriched is rows
    assert enriched[0]["quote"] == "Souřadnice retenční nádrže"


def test_no_subject_overlap_keeps_current_result(tmp_path):
    row = _row(
        "D.1.4.b.2-VZT-101-Půdorys 3.PP.pdf",
        quote="5400 m³/h",
    )
    database_dir = _database(
        tmp_path,
        row,
        [(1, "VZT", "Rozvody vzduchotechniky a požární klapky")],
    )
    rows = [row]

    enriched = dle.enrich_results(
        QUERY_PLAN,
        rows,
        (DrawingSubtype.PLAN,),
        database_dir=database_dir,
    )

    assert enriched is rows
    assert enriched[0]["quote"] == "5400 m³/h"


def test_floor_plan_remains_non_dedicated_after_override(tmp_path):
    row = _row(
        "D.1.4.b.2-VZT-101-Půdorys 3.PP.pdf",
        quote="5400 m³/h",
    )
    database_dir = _database(
        tmp_path,
        row,
        [(13, "22.4 m²", "Nádrž dešťové vody 205 m3")],
    )
    enriched = dle.enrich_results(
        QUERY_PLAN,
        [row],
        (DrawingSubtype.PLAN,),
        database_dir=database_dir,
    )

    rendered = dn.render_drawing_navigation(QUERY_PLAN, enriched)

    assert rendered is not None
    assert rendered.matches[0].floor_plan is True
    assert rendered.matches[0].dedicated is False
    assert "podlažní půdorys" in rendered.text
    assert "Nejde nutně o samostatný detailní půdorys." in rendered.text


def test_q1_section_has_no_regression_or_database_lookup(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(
        dle,
        "select_local_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic drawing query must not perform local lookup")
        ),
    )
    ollama_calls = []
    monkeypatch.setattr(
        ai_search,
        "_call_ollama",
        lambda *args, **kwargs: ollama_calls.append(1),
    )
    rows = [
        _row(
            "32_RETENCE.pdf",
            heading="Řez retenční nádrží",
            quote="Řez nádrží na dešťovou vodu",
        )
    ]

    result = ai_search.answer("najdi mi výkres retenční nádrže?", rows)

    assert ollama_calls == []
    assert result["citations"] is rows
    assert "Řez:" in result["answer"]
    assert "32_RETENCE.pdf" in result["answer"]
    assert "Řez retenční nádrží" in result["answer"]


def test_non_drawing_query_does_not_call_local_helper(monkeypatch):
    _disable_other_flags(monkeypatch)
    monkeypatch.setattr(
        dle,
        "enrich_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-drawing query entered local evidence helper")
        ),
    )
    ollama_calls = []
    _mock_json_ollama(monkeypatch, ollama_calls)
    rows = [_row("smlouva.pdf", quote="Smlouva je podepsaná.")]

    result = ai_search.answer("je podepsaná smlouva na BOZP?", rows)

    assert len(ollama_calls) == 1
    assert result["model"] != "drawing-navigation"
    assert "Smlouva je podepsaná." in result["answer"]


def test_answer_uses_local_plan_evidence_without_ollama(tmp_path, monkeypatch):
    _disable_other_flags(monkeypatch)
    row = _row(
        "D.1.4.b.2-VZT-101-Půdorys 3.PP.pdf",
        quote="5400 m³/h",
    )
    database_dir = _database(
        tmp_path,
        row,
        [(13, "22.4 m²", "Nádrž dešťové vody 205 m3")],
    )
    monkeypatch.setattr(dle, "DATABASE_DIR", database_dir)
    ollama_calls = []
    monkeypatch.setattr(
        ai_search,
        "_call_ollama",
        lambda *args, **kwargs: ollama_calls.append(1),
    )

    result = ai_search.answer(QUERY_PLAN, [row])

    assert ollama_calls == []
    assert result["model"] == "drawing-navigation"
    assert "Nádrž dešťové vody 205 m3" in result["answer"]
    assert "5400 m³/h" not in result["answer"]
    assert result["citations"][0]["quote"] == "Nádrž dešťové vody 205 m3"
    assert result["citations"][0]["score"] == row["score"]


def test_helper_has_no_project_hardcode():
    source = Path(dle.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "RETENCE",
        "32_RETENCE",
        "retenční nádrž",
        "retencni nadrz",
        "NDS",
        "VZT-101",
        "ZTI-102",
    ):
        assert forbidden not in source
