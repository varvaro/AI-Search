import os
import json
import sqlite3
import threading
import time
from pathlib import Path
import zipfile

import pytest
import ai_search
import ui_services
from ai_search_config import APP_SUPPORT_DIR


class FakeEmbeddings:
    def encode(self, texts):
        return [[float("alpha" in text.lower()), float("beta" in text.lower()), 0.5] for text in texts]


def write_office(path: Path, member: str, xml: str):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, xml)


def test_resolve_system_tool_uses_path_first(monkeypatch):
    monkeypatch.setattr(ai_search.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert ai_search.resolve_system_tool("pdfinfo") == "/usr/bin/pdfinfo"


def test_resolve_system_tool_uses_homebrew_fallback(monkeypatch):
    monkeypatch.setattr(ai_search.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "/usr/local/bin/pdfinfo")
    monkeypatch.setattr(ai_search.os, "access", lambda path, mode: True)
    assert ai_search.resolve_system_tool("pdfinfo") == "/usr/local/bin/pdfinfo"


def test_resolve_system_tool_prefers_apple_silicon_homebrew(monkeypatch):
    monkeypatch.setattr(ai_search.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setattr(ai_search.os, "access", lambda path, mode: True)
    assert ai_search.resolve_system_tool("pdfinfo") == "/opt/homebrew/bin/pdfinfo"


def test_resolve_system_tool_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr(ai_search.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda path: False)
    assert ai_search.resolve_system_tool("pdfinfo") is None


@pytest.fixture
def backend(tmp_path, monkeypatch):
    root = tmp_path / "Projekt Alpha"; root.mkdir()
    files = {
        "alpha.txt": "ALPHA unikátní fulltextový výraz a harmonogram.",
        "beta.docx": "BETA obsah dokumentu Word.",
        "gamma.xlsx": "GAMMA obsah tabulky Excel.",
        "scan.pdf": "DELTA text PDF.",
        "ocr.png": "OCR EPSILON rozpoznaný sken.",
    }
    (root / "alpha.txt").write_text(files["alpha.txt"])
    write_office(root / "beta.docx", "word/document.xml", f'<w:document xmlns:w="w"><w:t>{files["beta.docx"]}</w:t></w:document>')
    write_office(root / "gamma.xlsx", "xl/sharedStrings.xml", f'<sst><t>{files["gamma.xlsx"]}</t></sst>')
    (root / "scan.pdf").write_bytes(b"%PDF-test")
    (root / "ocr.png").write_bytes(b"image-test")
    real_extract = ai_search.extract
    def controlled(path):
        if path.suffix == ".pdf": return files[path.name], "pdftotext"
        if path.suffix == ".png": return files[path.name], "stav_skenu_ocr"
        return real_extract(path)
    monkeypatch.setattr(ai_search, "extract", controlled)
    state = tmp_path / "state"; embeddings = FakeEmbeddings()
    result = ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)
    return root, state, embeddings, result


def test_database_and_indexes_created(backend):
    _, state, _, _ = backend
    assert (state / "index.sqlite3").is_file() and (state / "lance").is_dir()


@pytest.mark.parametrize(("name", "term"), [("alpha.txt","ALPHA"),("beta.docx","BETA"),("gamma.xlsx","GAMMA"),("scan.pdf","DELTA")])
def test_text_formats_are_indexed(backend, name, term):
    _, state, _, _ = backend
    con = ai_search.connect(state / "index.sqlite3")
    assert con.execute("SELECT count(*) FROM chunks_fts WHERE name=? AND chunks_fts MATCH ?", (name, term)).fetchone()[0] == 1


def test_ocr_branch(backend):
    _, state, _, _ = backend
    con = ai_search.connect(state / "index.sqlite3")
    assert con.execute("SELECT extraction FROM documents WHERE name='ocr.png'").fetchone()[0] == "stav_skenu_ocr"


def test_embedding_wrapper_loads_configured_model(monkeypatch):
    loaded = {}
    class Model:
        def __init__(self, name): loaded["name"] = name
        def encode(self, texts, normalize_embeddings): return type("Array", (), {"tolist": lambda self:[[1.0, 0.0]]})()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Model)
    emb = ai_search.Embeddings()
    assert emb.encode(["test"]) == [[1.0, 0.0]]
    assert loaded["name"] == ai_search.EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# CrossEncoderReranker (2026-08-06 cross-encoder precision reranker) - unit
# tests against a fake sentence_transformers.CrossEncoder, so these never
# download/load the real ~2GB BAAI/bge-reranker-v2-m3 model. Real-model
# behaviour is exercised separately by the production benchmark
# (candidate_strategy="union_ce").
# ---------------------------------------------------------------------------

class _FakeSTCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder: records how many times
    it was constructed (to prove lazy-loading/no-reload) and returns a
    deterministic score per (query, passage) pair based on passage length."""
    instances_created = 0
    def __init__(self, name, max_length=None):
        _FakeSTCrossEncoder.instances_created += 1
        self.name = name; self.max_length = max_length
    def predict(self, pairs, batch_size=32):
        import numpy as np
        return np.array([float(len(passage)) for _, passage in pairs])


def test_cross_encoder_reranker_lazy_loads_model_only_on_first_score_call(monkeypatch):
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeSTCrossEncoder)
    _FakeSTCrossEncoder.instances_created = 0
    reranker = ai_search.CrossEncoderReranker()
    assert reranker.model is None, "model must not load at construction time"
    reranker.score("dotaz", ["krátký", "mnohem delší text"])
    assert reranker.model is not None
    assert _FakeSTCrossEncoder.instances_created == 1


def test_cross_encoder_reranker_does_not_reload_model_across_repeated_calls(monkeypatch):
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeSTCrossEncoder)
    _FakeSTCrossEncoder.instances_created = 0
    reranker = ai_search.CrossEncoderReranker()
    reranker.score("dotaz", ["a"])
    reranker.score("jiný dotaz", ["b", "c"])
    reranker.score("třetí dotaz", [])
    assert _FakeSTCrossEncoder.instances_created == 1, "the underlying model must be constructed exactly once, not once per query"


def test_cross_encoder_reranker_score_matches_passage_order_and_count(monkeypatch):
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeSTCrossEncoder)
    reranker = ai_search.CrossEncoderReranker()
    passages = ["x", "xxxxx", "xxx"]
    scores = reranker.score("dotaz", passages)
    assert scores == [1.0, 5.0, 3.0], "scores must be a plain list, same length/order as passages"


def test_cross_encoder_reranker_score_of_empty_passages_is_empty_list_without_loading_model(monkeypatch):
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeSTCrossEncoder)
    _FakeSTCrossEncoder.instances_created = 0
    reranker = ai_search.CrossEncoderReranker()
    assert reranker.score("dotaz", []) == []
    assert _FakeSTCrossEncoder.instances_created == 0, "no passages means no reason to ever load the model"


def test_get_default_cross_encoder_returns_the_same_instance_every_time():
    first = ai_search._get_default_cross_encoder()
    second = ai_search._get_default_cross_encoder()
    assert first is second


def test_fts5_search(backend):
    _, state, _, _ = backend
    con = ai_search.connect(state / "index.sqlite3")
    assert con.execute("SELECT name FROM chunks_fts WHERE chunks_fts MATCH 'unikátní'").fetchone()[0] == "alpha.txt"


def test_vector_search(backend):
    _, state, embeddings, _ = backend
    rows = ai_search.lance_table(state / "lance").search(embeddings.encode(["beta"])[0]).limit(3).to_list()
    assert rows and "id" in rows[0]


def test_hybrid_ranking_prefers_exact_match(backend):
    _, state, embeddings, _ = backend
    rows = ai_search.search("ALPHA unikátní", state / "index.sqlite3", state / "lance", embeddings)
    assert rows[0]["document"] == "alpha.txt"


def test_exact_filename_beats_semantic_similarity(backend):
    root, state, embeddings, _ = backend
    (root / "DL_5690045027.txt").write_text("obecný dodací list")
    ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)
    assert ai_search.search("5690045027", state / "index.sqlite3", state / "lance", embeddings)[0]["document"] == "DL_5690045027.txt"


def test_citations_have_existing_paths(backend):
    _, state, embeddings, _ = backend
    row = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)[0]
    assert {"document", "path", "quote", "project"} <= row.keys()
    assert Path(row["path"]).exists() and row["quote"]


def test_add_file(backend):
    root, state, embeddings, _ = backend
    (root / "new.txt").write_text("nově přidaný soubor")
    assert ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)["added"] == 1


def test_change_file(backend):
    root, state, embeddings, _ = backend
    (root / "alpha.txt").write_text("změněný obsah")
    assert ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)["changed"] == 1


def test_rename_file(backend):
    root, state, embeddings, _ = backend
    (root / "alpha.txt").rename(root / "renamed.txt")
    assert ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)["renamed"] == 1


def test_remove_file(backend):
    root, state, embeddings, _ = backend
    (root / "alpha.txt").unlink()
    assert ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)["removed"] == 1


def test_repeat_does_not_reindex(backend):
    root, state, embeddings, _ = backend
    result = ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)
    assert {key:result[key] for key in ("added","changed","renamed","removed","unchanged","duplicates","skipped","errors","timeouts","stopped")} == {"added":0,"changed":0,"renamed":0,"removed":0,"unchanged":5,"duplicates":0,"skipped":0,"errors":0,"timeouts":0,"stopped":False}


def test_duplicate_hash_is_skipped_without_unique_error(backend):
    root, state, embeddings, _ = backend
    (root / "alpha-copy.txt").write_bytes((root / "alpha.txt").read_bytes())
    result = ai_search.sync(root, state / "index.sqlite3", state / "lance", embeddings)
    assert result["errors"] == 0 and result["unchanged"] == 5 and result["duplicates"] == 1


def test_pdf_repeat_change_rename_and_remove(backend):
    root, state, embeddings, _ = backend
    db, lance = state / "index.sqlite3", state / "lance"
    assert ai_search.sync(root, db, lance, embeddings)["errors"] == 0
    pdf = root / "scan.pdf"; pdf.write_bytes(b"%PDF-zmeneny")
    assert ai_search.sync(root, db, lance, embeddings)["changed"] == 1
    renamed = root / "scan-prejmenovany.pdf"; pdf.rename(renamed)
    assert ai_search.sync(root, db, lance, embeddings)["renamed"] == 1
    renamed.unlink()
    assert ai_search.sync(root, db, lance, embeddings)["removed"] == 1


def test_sqlite_configuration_and_busy_wait(tmp_path):
    db=tmp_path/"state.sqlite3"
    with ai_search.database(db) as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_interruption_rolls_back_and_releases_lock(tmp_path, monkeypatch):
    root=tmp_path/"root"; root.mkdir(); (root/"a.txt").write_text("alpha")
    db,lance=tmp_path/"state.sqlite3",tmp_path/"lance"; embeddings=FakeEmbeddings()
    original=ai_search.extract; monkeypatch.setattr(ai_search,"extract",lambda path: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): ai_search.sync(root,db,lance,embeddings)
    monkeypatch.setattr(ai_search,"extract",original)
    assert ai_search.sync(root,db,lance,embeddings)["added"] == 1


def test_lance_failure_rolls_back_sqlite(tmp_path, monkeypatch):
    root=tmp_path/"root"; root.mkdir(); (root/"a.txt").write_text("alpha")
    db,lance=tmp_path/"state.sqlite3",tmp_path/"lance"; embeddings=FakeEmbeddings()
    table=ai_search.lance_table(lance,3); monkeypatch.setattr(ai_search,"lance_table",lambda *args,**kwargs:table)
    monkeypatch.setattr(table,"add",lambda rows: (_ for _ in ()).throw(RuntimeError("lance failure")))
    result=ai_search.sync(root,db,lance,embeddings)
    assert result["errors"] == 1
    with ai_search.database(db) as con: assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_two_indexations_cannot_run_concurrently(tmp_path):
    root=tmp_path/"root"; root.mkdir(); (root/"a.txt").write_text("alpha")
    db,lance=tmp_path/"state.sqlite3",tmp_path/"lance"; entered=threading.Event(); release=threading.Event()
    class SlowEmbeddings(FakeEmbeddings):
        def encode(self,texts): entered.set(); release.wait(5); return super().encode(texts)
    thread=threading.Thread(target=ai_search.sync,args=(root,db,lance,SlowEmbeddings()))
    thread.start(); assert entered.wait(5)
    with pytest.raises(ai_search.IndexingInProgress): ai_search.sync(root,db,lance,FakeEmbeddings())
    release.set(); thread.join(5); assert not thread.is_alive()


def test_busy_database_waits_instead_of_failing(tmp_path):
    root=tmp_path/"root"; root.mkdir(); file=root/"a.txt"; file.write_text("alpha")
    db,lance=tmp_path/"state.sqlite3",tmp_path/"lance"; embeddings=FakeEmbeddings(); ai_search.sync(root,db,lance,embeddings)
    file.write_text("alpha změna")
    blocker=sqlite3.connect(db,timeout=1,check_same_thread=False); blocker.execute("BEGIN IMMEDIATE")
    releaser=threading.Thread(target=lambda:(time.sleep(.2),blocker.commit(),blocker.close())); releaser.start()
    started=time.perf_counter(); result=ai_search.sync(root,db,lance,embeddings); elapsed=time.perf_counter()-started; releaser.join()
    assert result["changed"] == 1 and elapsed >= .15


def test_application_support_layout_and_safe_migration(tmp_path):
    legacy=tmp_path/"legacy"; legacy.mkdir(); base=tmp_path/"Application Support"/"AI Search"
    (legacy/"settings.json").write_text('{"project_root":"/test"}')
    con=ai_search.connect(legacy/"project.sqlite3"); con.close()
    (legacy/"project-lance").mkdir(); (legacy/"project-lance"/"marker").write_text("ok")
    result=ui_services.ensure_runtime_layout(base,legacy)
    assert {p.name for p in base.iterdir()} == {"database","lance","cache","logs","state"}
    assert (base/"database/project.sqlite3").exists() and (base/"lance/project/marker").exists()
    assert (legacy/"project.sqlite3").exists() and result["migrated"]
    assert APP_SUPPORT_DIR == Path(os.environ["AI_SEARCH_HOME"])


def test_stop_then_resume_uses_completed_documents(tmp_path):
    root=tmp_path/"root"; root.mkdir(); [(root/f"{i}.txt").write_text(f"obsah {i}") for i in range(10)]
    db,lance=tmp_path/"state.sqlite3",tmp_path/"lance"; stop=threading.Event()
    def progress(event):
        if event["phase"]=="LanceDB zápis" and event["current"]==3: stop.set()
    first=ai_search.sync(root,db,lance,FakeEmbeddings(),progress=progress,stop_event=stop)
    assert first["stopped"] and first["added"] == 3
    second=ai_search.sync(root,db,lance,FakeEmbeddings())
    assert second["added"] == 7 and second["unchanged"] == 3 and not second["stopped"]


def test_simulated_crash_preserves_checkpoint_and_resumes(tmp_path):
    root=tmp_path/"root"; root.mkdir(); [(root/f"{i}.txt").write_text(f"obsah {i}") for i in range(6)]
    db,lance=tmp_path/"state.sqlite3",tmp_path/"lance"
    class CrashEmbeddings(FakeEmbeddings):
        calls=0
        def encode(self,texts):
            self.calls+=1
            if self.calls==3: raise KeyboardInterrupt()
            return super().encode(texts)
    with pytest.raises(KeyboardInterrupt): ai_search.sync(root,db,lance,CrashEmbeddings())
    with ai_search.database(db) as con: assert con.execute("SELECT count(*) FROM documents").fetchone()[0] == 2
    result=ai_search.sync(root,db,lance,FakeEmbeddings())
    assert result["added"] == 4 and result["unchanged"] == 2


def test_broken_pdf_is_recorded_and_other_files_continue(tmp_path,monkeypatch):
    root=tmp_path/"root"; root.mkdir(); (root/"bad.pdf").write_bytes(b"broken"); (root/"good.txt").write_text("funguje")
    original=ai_search.extract
    monkeypatch.setattr(ai_search,"extract",lambda path: (_ for _ in ()).throw(ValueError("bad pdf")) if path.suffix==".pdf" else original(path))
    db,lance=tmp_path/"state.sqlite3",tmp_path/"lance"; result=ai_search.sync(root,db,lance,FakeEmbeddings())
    assert result["errors"] == 1 and result["added"] == 1
    with ai_search.database(db) as con: assert con.execute("SELECT status FROM index_status WHERE path LIKE '%bad.pdf'").fetchone()[0] == "CHYBA"


def test_pdf_parser_timeout_is_bounded(tmp_path,monkeypatch):
    pdf=tmp_path/"slow.pdf"; pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(ai_search.subprocess,"run",lambda *args,**kwargs: (_ for _ in ()).throw(ai_search.subprocess.TimeoutExpired(args[0],1)))
    with pytest.raises(TimeoutError): ai_search.extract_pdf(pdf,budget_seconds=1)


@pytest.mark.parametrize("phase",["parsování","chunking","embedding"])
def test_phase_watchdog_records_timeout_and_continues(tmp_path,monkeypatch,phase):
    root=tmp_path/"root"; root.mkdir(); (root/"slow.txt").write_text("obsah")
    class SlowEmbeddings(FakeEmbeddings):
        def encode(self,texts):
            if phase=="embedding": time.sleep(.02)
            return super().encode(texts)
    if phase=="parsování":
        monkeypatch.setattr(ai_search,"PARSE_TIMEOUT_SECONDS",.001); monkeypatch.setattr(ai_search,"extract",lambda path:(time.sleep(.02) or ("text","test")))
    else: monkeypatch.setattr(ai_search,"extract",lambda path:(("řádek\n"*1000) if phase=="chunking" else "text","test"))
    if phase=="chunking": monkeypatch.setattr(ai_search,"CHUNK_TIMEOUT_SECONDS",0)
    if phase=="embedding": monkeypatch.setattr(ai_search,"EMBEDDING_TIMEOUT_SECONDS",.001)
    db,lance=tmp_path/"runtime/database/index.sqlite3",tmp_path/"runtime/lance"; result=ai_search.sync(root,db,lance,SlowEmbeddings())
    assert result["errors"]==1 and result["timeouts"]==1 and Path(result["slow_report"]).exists()
    with ai_search.database(db) as con: status=con.execute("SELECT status,error FROM index_status").fetchone()
    assert status[0]=="ERROR_TIMEOUT" and "fáze=" in status[1]
    log=(tmp_path/"runtime/logs/index.log").read_text()
    assert "ERROR_TIMEOUT" in log and "Traceback" in log


def test_embedding_timeout_config_value():
    """The raise from 60 s to 300 s (Fáze 3.2). 60 s was already being hit in
    production before the reindex: the slow-phase log recorded embedding
    timeouts on .xls documents holding as few as 9 monolithic chunks."""
    import ai_search_config
    assert ai_search_config.EMBEDDING_TIMEOUT_SECONDS == 300
    assert ai_search.EMBEDDING_TIMEOUT_SECONDS == ai_search_config.EMBEDDING_TIMEOUT_SECONDS


def test_embedding_timeout_is_read_from_config_not_hardcoded(tmp_path,monkeypatch):
    """sync() must pass the configured value through to the watchdog. Changing
    the constant has to be enough to change the real limit - the existing
    ERROR_TIMEOUT test above proves a low value still trips, this proves the
    exact value reaches EmbeddingWatchdog.encode()."""
    root=tmp_path/"root"; root.mkdir(); (root/"a.txt").write_text("obsah dokumentu")
    seen=[]
    class RecordingWatchdog(ai_search.EmbeddingWatchdog):
        def encode(self,texts,limit,progress=None):
            seen.append(limit); return super().encode(texts,limit,progress)
    monkeypatch.setattr(ai_search,"EmbeddingWatchdog",RecordingWatchdog)
    monkeypatch.setattr(ai_search,"EMBEDDING_TIMEOUT_SECONDS",1234)
    ai_search.sync(root,tmp_path/"db.sqlite3",tmp_path/"lance",FakeEmbeddings())
    assert seen==[1234]


def test_lancedb_writes_share_the_embedding_timeout(tmp_path,monkeypatch):
    """The same constant also bounds the LanceDB add/delete calls, so raising
    it raises those too - asserted so the coupling is not discovered by
    accident later."""
    root=tmp_path/"root"; root.mkdir(); (root/"a.txt").write_text("obsah dokumentu")
    seen=[]
    real=ai_search.run_with_timeout
    def recording(operation,timeout_seconds,phase_name):
        seen.append((phase_name,timeout_seconds)); return real(operation,timeout_seconds,phase_name)
    monkeypatch.setattr(ai_search,"run_with_timeout",recording)
    monkeypatch.setattr(ai_search,"EMBEDDING_TIMEOUT_SECONDS",1234)
    ai_search.sync(root,tmp_path/"db.sqlite3",tmp_path/"lance",FakeEmbeddings())
    lance_calls=[limit for phase,limit in seen if phase.startswith("lancedb")]
    assert lance_calls and all(limit==1234 for limit in lance_calls)


def test_outlook_msg_parser_extracts_standard_fields(tmp_path,monkeypatch):
    import extract_msg
    class Attachment: longFilename="nabidka.pdf"; shortFilename=None
    class Message:
        subject="Předmět Outlook"; sender="odesilatel@example.com"; to="prijemce@example.com"; cc=""; date="2026-08-04"; body="Text zprávy"; attachments=[Attachment()]
        def __enter__(self): return self
        def __exit__(self,*args): return None
    monkeypatch.setattr(extract_msg,"openMsg",lambda path:Message())
    text,method=ai_search.extract_outlook_msg(tmp_path/"mail.msg")
    assert method=="extract-msg" and "Předmět Outlook" in text and "Text zprávy" in text and "nabidka.pdf" in text


def test_msg_worker_timing_and_extension_report(tmp_path,monkeypatch):
    root=tmp_path/"root"; root.mkdir(); (root/"mail.msg").write_bytes(b"outlook-msg")
    monkeypatch.setattr(ai_search,"extract_outlook_msg",lambda path:(time.sleep(.01) or ("Předmět: Test\n\nTělo Outlook zprávy","extract-msg")))
    db,lance=tmp_path/"runtime/database/index.sqlite3",tmp_path/"runtime/lance"; result=ai_search.sync(root,db,lance,FakeEmbeddings())
    stats=result["by_extension"]["MSG"]; assert stats["successful"]==1 and stats["parse_seconds"]>=.01
    report=json.loads(Path(result["slow_report"]).read_text()); assert report["by_extension"]["MSG"]["successful"]==1


@pytest.mark.parametrize("timeout",[False,True])
def test_msg_error_or_timeout_is_error_msg_parse(tmp_path,monkeypatch,timeout):
    root=tmp_path/"root"; root.mkdir(); (root/"bad.msg").write_bytes(b"bad")
    if timeout:
        monkeypatch.setattr(ai_search,"MSG_PARSE_TIMEOUT_SECONDS",.001); monkeypatch.setattr(ai_search,"extract_outlook_msg",lambda path:(time.sleep(.02) or ("text","extract-msg")))
    else: monkeypatch.setattr(ai_search,"extract_outlook_msg",lambda path:(_ for _ in ()).throw(ValueError("neplatný Outlook MSG")))
    db,lance=tmp_path/"runtime/database/index.sqlite3",tmp_path/"runtime/lance"; result=ai_search.sync(root,db,lance,FakeEmbeddings())
    with ai_search.database(db) as con: status,error=con.execute("SELECT status,error FROM index_status").fetchone()
    assert status=="ERROR_MSG_PARSE" and result["errors"]==1 and (result["timeouts"]==1)==timeout
    assert result["by_extension"]["MSG"]["timeouts" if timeout else "errors"]==1 and error


def test_ollama_offline_preserves_citations(backend, monkeypatch):
    _, state, embeddings, _ = backend
    rows = ai_search.search("ALPHA", state / "index.sqlite3", state / "lance", embeddings)
    monkeypatch.setattr(ai_search.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    result = ai_search.answer("dotaz", rows)
    assert result["citations"] == rows and "nedostup" in result["answer"].lower()
