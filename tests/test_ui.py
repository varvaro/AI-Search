from pathlib import Path
from email.message import EmailMessage
import subprocess

import pytest
from streamlit.testing.v1 import AppTest
import ai_search
import ui_services as ui

class FakeEmbeddings:
    def encode(self,texts): return [[1.0,0.5,0.1] for _ in texts]

def test_ui_starts_in_czech(tmp_path,monkeypatch):
    monkeypatch.setattr(ui,"ollama_status",lambda:False)
    app=AppTest.from_file("app.py",default_timeout=10).run()
    assert not app.exception
    assert any("Najděte informace" in h.value for h in app.header)
    assert any("Aktualizovat index" in b.label for b in app.button)

def test_settings_roundtrip(tmp_path):
    path=tmp_path/"settings.json"; value=ui.Settings(project_root="/projekt",result_count=17)
    ui.save_settings(path,value); loaded=ui.load_settings(path)
    assert loaded.project_root=="/projekt" and loaded.result_count==17

def test_rendered_search_result_shape(tmp_path):
    path=tmp_path/"a.txt"; path.write_text("obsah")
    row={"document":"a.txt","path":str(path),"project":"P","quote":"obsah","score":1.0,"source":"Dokument","date":"2026-08-04","extension":"txt","author":""}
    assert ui.apply_filters([row],source="Dokument",extension="txt")==[row]

def test_search_results_are_unique_by_path(tmp_path,monkeypatch):
    settings=ui.Settings(project_root=str(tmp_path)); db,lance=ui.state_paths(tmp_path,"Dokument"); db.touch()
    duplicate={"document":"a.txt","path":str(tmp_path/"a.txt"),"project":"P","quote":"x","score":1.0}
    monkeypatch.setattr(ai_search,"search",lambda *a,**k:[duplicate.copy(),{**duplicate,"score":0.9}])
    monkeypatch.setattr(ui,"metadata_for",lambda *a,**k:{"source":"Dokument","date":"","author":"","extension":"txt"})
    assert len(ui.search_all("x",settings,tmp_path,FakeEmbeddings()))==1

def test_search_requests_enough_chunks_for_lazy_results(tmp_path,monkeypatch):
    settings=ui.Settings(project_root=str(tmp_path),result_count=20); db,_=ui.state_paths(tmp_path,"Dokument"); db.touch(); captured={}
    def fake_search(*args): captured["limit"]=args[-1]; return []
    monkeypatch.setattr(ai_search,"search",fake_search)
    ui.search_all("x",settings,tmp_path,FakeEmbeddings())
    assert captured["limit"]>=80

def test_filters_remove_nonmatching_rows():
    rows=[{"source":"E-mail","project":"P","path":"/x/a.eml","extension":"eml","author":"Jan","date":"2026-01-01"}]
    assert not ui.apply_filters(rows,source="Poznámka")
    assert ui.apply_filters(rows,author="jan")

def test_source_citation_path_exists(tmp_path):
    path=tmp_path/"zdroj.txt"; path.write_text("citace")
    citation={"document":path.name,"path":str(path),"quote":"citace","project":"P"}
    assert Path(citation["path"]).exists() and citation["quote"]

def test_no_results():
    assert ui.apply_filters([],source="Vše")==[]

def test_ollama_offline_is_false(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**k:(_ for _ in ()).throw(OSError("offline")))
    assert ui.ollama_status() is False

def test_local_eml_import(tmp_path):
    message=EmailMessage(); message["Subject"]="Kontrolní den"; message["From"]="stavbyvedouci@example.cz"; message["To"]="tym@example.cz"; message["Message-ID"]="<vlakno-1>"; message.set_content("Termín betonáže je pátek."); message.add_attachment(b"data",maintype="application",subtype="pdf",filename="priloha.pdf")
    path=tmp_path/"mail.eml"; path.write_bytes(message.as_bytes()); parsed=ui.parse_eml(path)
    assert parsed["subject"]=="Kontrolní den" and "stavbyvedouci" in parsed["sender"] and parsed["attachments"]==["priloha.pdf"] and parsed["thread_id"]=="<vlakno-1>"

@pytest.mark.parametrize("suffix",[".txt",".md",".pdf",".docx"])
def test_local_note_source_type(tmp_path,suffix):
    path=tmp_path/("note"+suffix); path.write_bytes(b"note")
    assert ui.metadata_for(path,"Poznámka")["source"]=="Poznámka"

def test_open_existing_path(tmp_path):
    path=tmp_path/"a.txt"; path.write_text("a")
    class Result: returncode=0
    ok,_=ui.open_path(path,runner=lambda *a,**k:Result())
    assert ok

def test_open_missing_path(tmp_path):
    ok,message=ui.open_path(tmp_path/"missing.txt",runner=lambda *a,**k:None)
    assert not ok and "neexistuje" in message

def test_backend_regression_contract():
    assert callable(ai_search.sync) and callable(ai_search.search) and callable(ai_search.answer)

def test_index_summary_has_real_latest_date(tmp_path):
    db,_=ui.state_paths(tmp_path,"Dokument"); con=ai_search.connect(db); con.close()
    summary=ui.index_summary(tmp_path)
    assert summary["latest"]!="—"

def test_context_is_meaningful_and_bounded():
    text="Úvod dokumentu. "+("Doplňující informace o stavbě. "*8)+"Hydroizolace Pentaflex musí být provedena kolem výztuže. "+("Další technický text. "*8)
    excerpt=ui.context_excerpt(text,"Pentaflex")
    assert 150<=len(excerpt)<=302 and "Pentaflex" in excerpt
    assert len(ui.context_excerpt("A","A"))>20

def test_human_relevance_labels():
    assert "Vysoká" in ui.match_label(1.0,1.0)
    assert "Střední" in ui.match_label(0.5,1.0)
    assert "Nízká" in ui.match_label(0.1,1.0)

def test_display_location_hides_home_path(tmp_path):
    root=tmp_path/"Projekt"; path=root/"Složka"/"dokument.pdf"
    row={"path":str(path),"source":"Dokument","project":"Projekt"}
    project,folder=ui.display_location(row,ui.Settings(project_root=str(root)))
    assert project=="Projekt" and folder=="Složka" and "/Users/" not in folder

def test_native_folder_picker(monkeypatch):
    class Result: returncode=0; stdout="/tmp/Projekt/\n"
    ok,path=ui.choose_folder("Projekt",runner=lambda *a,**k:Result())
    assert ok and path=="/tmp/Projekt"

def test_preview_text_and_missing(tmp_path):
    text=tmp_path/"note.md"; text.write_text("Český náhled")
    assert ui.preview_document(text)["content"]=="Český náhled"
    assert ui.preview_document(tmp_path/"missing.pdf")["kind"]=="error"

def test_system_overview_counts_and_size(tmp_path):
    db,_=ui.state_paths(tmp_path,"Dokument"); con=ai_search.connect(db)
    con.execute("INSERT INTO documents(path,relative_path,name,project,content_hash,size,mtime_ns,inode,extraction) VALUES(?,?,?,?,?,?,?,?,?)",("/x/a.pdf","a.pdf","a.pdf","P","a"*64,1,1,1,"text")); con.commit(); con.close()
    summary=ui.index_summary(tmp_path)
    assert summary["pdf"]==1 and summary["total"]==1 and summary["size_bytes"]>0

@pytest.mark.parametrize("page,heading",[("settings","Nastavení"),("history","Historie a diagnostika"),("diagnostics","Diagnostika")])
def test_secondary_screens_render(page,heading):
    app=AppTest.from_file("app.py",default_timeout=10); app.session_state["page"]=page; app.run()
    assert not app.exception and any(heading in item.value for item in app.header)

def _mock_ready_ui(monkeypatch,tmp_path,rows):
    monkeypatch.setattr(ui,"index_summary",lambda state:{"total":24,"latest":"04. 08. 2026 20:00","ready":True,"pdf":20,"emails":2,"notes":2,"size_bytes":1024})
    monkeypatch.setattr(ui,"indexed_root",lambda state:str(tmp_path))
    monkeypatch.setattr(ui,"ollama_status",lambda:False)
    monkeypatch.setattr(ai_search,"Embeddings",lambda:FakeEmbeddings())
    monkeypatch.setattr(ui,"search_all",lambda *a,**k:rows)

def test_twenty_plus_results_are_lazy_and_compact(tmp_path,monkeypatch):
    rows=[]
    for i in range(24):
        path=tmp_path/(f"Velmi dlouhý název dokumentu číslo {i} s technickým popisem.pdf")
        rows.append({"document":path.name,"title":path.name,"path":str(path),"project":"Projekt","quote":"Hydroizolace Pentaflex musí být provedena kolem výztuže. "+("Technický kontext. "*12),"score":1/(i+1),"source":"Dokument","date":"2026-08-04","extension":"pdf","author":""})
    _mock_ready_ui(monkeypatch,tmp_path,rows)
    app=AppTest.from_file("app.py",default_timeout=10).run(); app.text_input[0].input("Pentaflex").run()
    assert not app.exception
    assert any("Výsledky (24)" in h.value for h in app.subheader)
    assert len([b for b in app.button if b.label=="Náhled"])==8
    assert any(b.label=="Zobrazit další výsledky" for b in app.button)

def test_no_results_screen(tmp_path,monkeypatch):
    _mock_ready_ui(monkeypatch,tmp_path,[])
    app=AppTest.from_file("app.py",default_timeout=10).run(); app.text_input[0].input("nenalezitelný výraz").run()
    assert not app.exception and any("Nebyly nalezeny" in warning.value for warning in app.warning)

def test_ai_citations_render_as_blocks(tmp_path,monkeypatch):
    path=tmp_path/"citovaný dokument.pdf"; path.write_bytes(b"pdf")
    row={"document":path.name,"title":path.name,"path":str(path),"project":"Projekt","quote":"Citovaný kontext dokumentu s dostatečnou délkou. "+("Další věta. "*12),"score":1.0,"source":"Dokument","date":"2026-08-04","extension":"pdf","author":""}
    _mock_ready_ui(monkeypatch,tmp_path,[row]); monkeypatch.setattr(ai_search,"answer",lambda *a,**k:{"answer":"Ověřená odpověď [1].","citations":[row]})
    app=AppTest.from_file("app.py",default_timeout=10).run(); app.radio[0].set_value("Položit otázku").run(); app.text_input[0].input("Jaký je závěr?").run()
    assert not app.exception and any("Odpověď" in h.value for h in app.subheader)
    assert any(b.label=="Přejít na citovaný dokument" for b in app.button)
