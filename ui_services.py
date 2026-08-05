"""Tenká česká aplikační vrstva nad ověřeným backendem AI Search."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from email import policy
from email.parser import BytesParser
import base64, json, os, shutil, sqlite3, subprocess, time
from pathlib import Path
import ai_search
from ai_search_config import APP_SUPPORT_DIR

@dataclass
class Settings:
    project_root: str = ""
    email_root: str = ""
    notes_root: str = ""
    default_llm: str = "qwen3:8b"
    deep_llm: str = "qwen3:14b"
    embedding_model: str = "BAAI/bge-m3"
    ocr: bool = True
    result_count: int = 10
    threads: int = 1

def load_settings(path: Path) -> Settings:
    if not path.exists(): return Settings()
    try: return Settings(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError): return Settings()

def save_settings(path: Path, settings: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(".tmp"); temp.write_text(json.dumps(asdict(settings),ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(temp,path)

def parse_eml(path: Path) -> dict:
    message=BytesParser(policy=policy.default).parse(path.open("rb"))
    body=""
    part=message.get_body(preferencelist=("plain",))
    if part:
        try: body=part.get_content()
        except Exception: body=""
    attachments=[p.get_filename() for p in message.iter_attachments() if p.get_filename()]
    return {"subject":str(message.get("subject",path.name)),"sender":str(message.get("from","")),"recipients":str(message.get("to","")),"date":str(message.get("date","")),"body":body,"path":str(path),"attachments":attachments,"thread_id":str(message.get("thread-index",message.get("references",message.get("message-id",""))))}

def metadata_for(path: Path, source: str) -> dict:
    stat=path.stat(); data={"source":source,"date":datetime.fromtimestamp(stat.st_mtime).date().isoformat(),"author":"","extension":path.suffix.lower().lstrip(".")}
    if source=="E-mail" and path.suffix.lower()==".eml":
        mail=parse_eml(path); data.update({"title":mail["subject"],"author":mail["sender"],"email":mail})
    return data

def state_paths(state_dir: Path, source: str):
    key={"Dokument":"project","E-mail":"emails","Poznámka":"notes"}[source]
    db,lance=state_dir/"database"/f"{key}.sqlite3",state_dir/"lance"/key; db.parent.mkdir(parents=True,exist_ok=True); lance.parent.mkdir(parents=True,exist_ok=True); return db,lance

def state_file(state_dir: Path, name: str) -> Path: return state_dir/"state"/name

def ensure_runtime_layout(base: Path = APP_SUPPORT_DIR, legacy: Path | None = None) -> dict:
    for name in ("database","lance","cache","logs","state"): (base/name).mkdir(parents=True,exist_ok=True)
    migrated=[]
    if not legacy or not legacy.exists(): return {"base":base,"migrated":migrated}
    mapping={"settings.json":base/"state/settings.json","history.jsonl":base/"state/history.jsonl"}
    for source in ("project","emails","notes"):
        mapping[f"{source}.sqlite3"]=base/"database"/f"{source}.sqlite3"
        mapping[f"{source}.sqlite3-wal"]=base/"database"/f"{source}.sqlite3-wal"
        mapping[f"{source}.sqlite3-shm"]=base/"database"/f"{source}.sqlite3-shm"
    for name,target in mapping.items():
        source=legacy/name
        if source.exists() and not target.exists(): target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target); migrated.append(str(target))
    for source in ("project","emails","notes"):
        old=legacy/f"{source}-lance"; target=base/"lance"/source
        if old.exists() and not target.exists(): shutil.copytree(old,target); migrated.append(str(target))
    for db in (base/"database").glob("*.sqlite3"):
        with sqlite3.connect(db) as con:
            if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok": raise RuntimeError(f"Migrace databáze {db.name} neprošla kontrolou integrity")
    return {"base":base,"migrated":migrated}

def index_source(root: Path, source: str, state_dir: Path, embeddings, progress=None, stop_event=None) -> dict:
    db,lance=state_paths(state_dir,source); started=time.perf_counter(); counts=ai_search.sync(root,db,lance,embeddings,progress=progress,stop_event=stop_event); elapsed=time.perf_counter()-started
    record_history(state_dir,{"time":datetime.now().isoformat(timespec="seconds"),"source":source,"root":str(root),"counts":counts,"seconds":elapsed,"embedding_model":ai_search.EMBEDDING_MODEL})
    return {**counts,"seconds":elapsed}

def record_history(state_dir: Path, event: dict) -> None:
    path=state_file(state_dir,"history.jsonl"); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(event,ensure_ascii=False)+"\n")

def read_history(state_dir: Path) -> list[dict]:
    path=state_file(state_dir,"history.jsonl")
    if not path.exists(): return []
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        try: rows.append(json.loads(line))
        except ValueError: pass
    return list(reversed(rows[-50:]))

def search_all(query: str, settings: Settings, state_dir: Path, embeddings) -> list[dict]:
    roots=[("Dokument",settings.project_root),("E-mail",settings.email_root),("Poznámka",settings.notes_root)]; output=[]
    for source,root in roots:
        db,lance=state_paths(state_dir,source)
        if not root or not db.exists(): continue
        # Vyžádej více chunků, aby po sloučení nebyl jeden dokument zobrazen
        # opakovaně a zůstalo dost unikátních výsledků pro postupné načítání.
        for row in ai_search.search(query,db,lance,embeddings,max(50,settings.result_count*4)):
            row.update(metadata_for(Path(row["path"]),source)); row["title"]=row.get("title",row["document"]); output.append(row)
    unique={}
    for row in sorted(output,key=lambda item:item["score"],reverse=True):
        if row["path"] not in unique:
            unique[row["path"]]=row
        elif row.get("quote") and row["quote"] not in unique[row["path"]].get("quote",""):
            unique[row["path"]]["quote"]=(unique[row["path"]].get("quote","")+" "+row["quote"]).strip()[:1200]
    return list(unique.values())[:settings.result_count]

def context_excerpt(text: str, query: str, minimum: int = 150, maximum: int = 300) -> str:
    clean=" ".join((text or "").split())
    if not clean: return "Pro tento výsledek není dostupný textový náhled. Dokument lze otevřít přímo."
    terms=[term.casefold() for term in query.split() if len(term)>2]
    folded=clean.casefold(); positions=[folded.find(term) for term in terms if folded.find(term)>=0]
    center=min(positions) if positions else 0
    start=max(0,center-maximum//3); end=min(len(clean),start+maximum)
    if end-start<minimum and len(clean)>minimum: start=max(0,end-minimum)
    excerpt=clean[start:end].strip(" ,.;:-")
    if len(excerpt)<20: return "Textový úryvek je příliš krátký. Otevřete dokument pro zobrazení úplného kontextu."
    return ("…" if start else "")+excerpt+("…" if end<len(clean) else "")

def match_label(score: float, best_score: float) -> str:
    ratio=score/best_score if best_score>0 else 0
    if ratio>=0.72: return "★★★★★ Vysoká shoda"
    if ratio>=0.38: return "★★★☆☆ Střední shoda"
    return "★☆☆☆☆ Nízká shoda"

def display_location(row: dict, settings: Settings) -> tuple[str,str]:
    path=Path(row["path"]); source=row.get("source","Dokument")
    root_value={"Dokument":settings.project_root,"E-mail":settings.email_root,"Poznámka":settings.notes_root}.get(source,"")
    project=row.get("project") or (Path(root_value).name if root_value else "Neznámý projekt")
    try: relative=path.relative_to(Path(root_value)); folder=str(relative.parent) if str(relative.parent)!="." else "Kořen projektu"
    except (ValueError,TypeError): folder=path.parent.name or "Kořen projektu"
    return project,folder

def choose_folder(prompt: str, runner=subprocess.run) -> tuple[bool,str]:
    script=f'POSIX path of (choose folder with prompt "{prompt.replace(chr(34), chr(39))}")'
    result=runner(["/usr/bin/osascript","-e",script],check=False,capture_output=True,text=True)
    if result.returncode!=0: return False,""
    return True,result.stdout.strip().rstrip("/")

def preview_document(path: Path, fallback: str = "") -> dict:
    if not path.exists(): return {"kind":"error","content":"Soubor už neexistuje. Aktualizujte index."}
    suffix=path.suffix.lower()
    if suffix==".pdf" and path.stat().st_size<=12*1024*1024:
        return {"kind":"pdf","content":base64.b64encode(path.read_bytes()).decode("ascii")}
    if suffix==".eml": return {"kind":"text","content":parse_eml(path)["body"] or fallback}
    if suffix in {".txt",".md",".csv",".rtf"}: return {"kind":"text","content":path.read_text(encoding="utf-8",errors="replace")[:12000]}
    return {"kind":"text","content":fallback or "Pro tento formát není přímý náhled dostupný. Dokument lze otevřít."}

def apply_filters(rows: list[dict], source="Vše", project="Vše", folder="", extension="Vše", author="", date_from=None, date_to=None):
    filtered=[]
    for row in rows:
        date=row.get("date","")
        if source!="Vše" and row.get("source")!=source: continue
        if project!="Vše" and row.get("project")!=project: continue
        if folder and folder.casefold() not in row.get("path","").casefold(): continue
        if extension!="Vše" and row.get("extension")!=extension: continue
        if author and author.casefold() not in row.get("author","").casefold(): continue
        if date_from and date and date < str(date_from): continue
        if date_to and date and date > str(date_to): continue
        filtered.append(row)
    return filtered

def open_path(path: Path, reveal=False, runner=subprocess.run) -> tuple[bool,str]:
    if not path.exists(): return False,"Soubor už na uvedené cestě neexistuje. Aktualizujte index."
    args=["/usr/bin/open","-R",str(path)] if reveal else ["/usr/bin/open",str(path)]
    result=runner(args,check=False,capture_output=True)
    return (result.returncode==0,"Soubor byl otevřen." if result.returncode==0 else "Soubor se nepodařilo otevřít. Zkontrolujte oprávnění macOS.")

def ollama_status(timeout=1.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=timeout): return True
    except Exception: return False

def index_summary(state_dir: Path) -> dict:
    total=0; latest=""; counts={"pdf":0,"emails":0,"notes":0}
    for source in ("Dokument","E-mail","Poznámka"):
        db,_=state_paths(state_dir,source)
        if db.exists():
            with ai_search.database(db) as con:
                source_total=con.execute("SELECT count(*) FROM documents").fetchone()[0]; total+=source_total
                if source=="Dokument": counts["pdf"]+=con.execute("SELECT count(*) FROM documents WHERE lower(name) LIKE '%.pdf'").fetchone()[0]
                elif source=="E-mail": counts["emails"]+=source_total
                else: counts["notes"]+=source_total
            latest=max(latest,datetime.fromtimestamp(db.stat().st_mtime).strftime("%d. %m. %Y %H:%M"))
    size=sum(path.stat().st_size for folder in (state_dir/"database",state_dir/"lance") for path in folder.rglob("*") if path.is_file()) if state_dir.exists() else 0
    return {"total":total,"latest":latest or "—","ready":total>0,**counts,"size_bytes":size}

def indexed_root(state_dir: Path) -> str:
    db,_=state_paths(state_dir,"Dokument")
    if not db.exists(): return ""
    with ai_search.database(db) as con: row=con.execute("SELECT value FROM settings WHERE key='root'").fetchone()
    return row[0] if row else ""
