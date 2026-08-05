"""Read-only production diagnostics and export for AI Search."""
from __future__ import annotations
import fcntl, html, json, os, resource, sqlite3, time
from datetime import datetime
from pathlib import Path

import ai_search
from ai_search_config import APP_SUPPORT_DIR, EMBEDDING_MODEL, DEFAULT_MODEL

APP_VERSION="1.0.0-rc1"
KNOWN_EXTENSIONS=("PDF","DOCX","XLSX","TXT","RTF","EML","MSG","HTML")

def folder_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0

def _latest_json(path: Path, pattern: str) -> dict:
    files=sorted(path.glob(pattern),key=lambda item:item.stat().st_mtime,reverse=True)
    if not files: return {}
    try: return json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError,ValueError): return {}

def _history(base: Path) -> dict:
    path=base/"state/history.jsonl"
    if not path.exists(): return {}
    for line in reversed(path.read_text(encoding="utf-8",errors="replace").splitlines()):
        try: return json.loads(line)
        except ValueError: continue
    return {}

def _lock_active(path: Path) -> bool:
    path.parent.mkdir(parents=True,exist_ok=True); handle=path.open("a+")
    try:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB); return False
        except BlockingIOError: return True
    finally:
        try: fcntl.flock(handle,fcntl.LOCK_UN)
        except OSError: pass
        handle.close()

def _lance_ids(lance_dir: Path) -> tuple[set[str],str]:
    try:
        table=ai_search.lance_table(lance_dir)
        if not table: return set(),"chybí tabulka chunks"
        rows=table.search().select(["id"]).to_arrow().column("id").to_pylist()
        return {value for value in rows if value!="__init__"},"ok"
    except Exception as exc: return set(),f"{type(exc).__name__}: {exc}"

def verify_index(db_path: Path,lance_dir: Path,lock_path: Path) -> dict:
    checks={}; chunk_ids=set(); document_count=chunk_count=0
    try:
        con=sqlite3.connect(f"file:{db_path}?mode=ro",uri=True,timeout=5)
        try:
            integrity=con.execute("PRAGMA integrity_check").fetchone()[0]; checks["SQLite integrita"]=integrity=="ok"
            document_count=con.execute("SELECT count(*) FROM documents").fetchone()[0]; chunk_count=con.execute("SELECT count(*) FROM chunks").fetchone()[0]
            chunk_ids={row[0] for row in con.execute("SELECT id FROM chunks")}
            checks["Duplicitní hashe"]=con.execute("SELECT count(*) FROM (SELECT content_hash FROM documents GROUP BY content_hash HAVING count(*)>1)").fetchone()[0]==0
            checks["Osiřelé chunky"]=con.execute("SELECT count(*) FROM chunks c LEFT JOIN documents d ON d.id=c.document_id WHERE d.id IS NULL").fetchone()[0]==0
        finally: con.close()
    except Exception as exc: checks["SQLite integrita"]=False; checks["SQLite chyba"]=str(exc)
    lance_ids,lance_status=_lance_ids(lance_dir); checks["LanceDB integrita"]=lance_status=="ok"
    missing=chunk_ids-lance_ids; orphan=lance_ids-chunk_ids
    checks["Chybějící embeddingy"]=not missing; checks["Osiřelé embeddingy"]=not orphan; checks["Aktivní lock"]=not _lock_active(lock_path)
    ok=all(value is True for value in checks.values() if isinstance(value,bool))
    return {"ok":ok,"checks":checks,"documents":document_count,"chunks":chunk_count,"embeddings":len(lance_ids),"missing_embeddings":len(missing),"orphan_embeddings":len(orphan)}

def collect_diagnostics(project_path: str,base: Path=APP_SUPPORT_DIR) -> dict:
    db=base/"database/project.sqlite3"; lance=base/"lance/project"; slow=_latest_json(base/"logs","project-slow-phases.json"); history=_history(base)
    extension_stats={name:{"total":0,"successful":0,"timeouts":0,"errors":0,"skipped":0} for name in (*KNOWN_EXTENSIONS,"OSTATNÍ")}; errors=[]; documents=chunks=0
    if db.exists():
        con=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
        try:
            documents=con.execute("SELECT count(*) FROM documents").fetchone()[0]; chunks=con.execute("SELECT count(*) FROM chunks").fetchone()[0]
            rows=con.execute("SELECT d.name,coalesce(s.status,'NEZNÁMÝ'),coalesce(s.updated_at,''),coalesce(s.error,'') FROM documents d LEFT JOIN index_status s ON s.path=d.path").fetchall()
            error_rows=con.execute("SELECT updated_at,path,status,error FROM index_status WHERE status IN ('CHYBA','ERROR_TIMEOUT','ERROR_MSG_PARSE') ORDER BY updated_at DESC LIMIT 50").fetchall()
        finally: con.close()
        for name,status,_,_ in rows:
            ext=Path(name).suffix.upper().lstrip("."); key=ext if ext in KNOWN_EXTENSIONS else "OSTATNÍ"; stat=extension_stats[key]; stat["total"]+=1
            if status in ("NOVÝ","AKTUALIZOVANÝ"): stat["successful"]+=1
            elif status in ("NEZMĚNĚNÝ","DUPLIKÁT","PŘESKOČENÝ"): stat["skipped"]+=1
            elif status in ("ERROR_TIMEOUT",): stat["timeouts"]+=1
            elif status.startswith("ERROR") or status=="CHYBA": stat["errors"]+=1
        errors=[{"čas":row[0],"soubor":row[1],"typ":row[2],"důvod":row[3],"traceback":row[3]} for row in error_rows]
    lance_ids,lance_status=_lance_ids(lance); slowest=slow.get("slowest",[]); by_phase={phase:max((row for row in slowest if row.get("phase")==phase),key=lambda row:row.get("seconds",0),default={}) for phase in ("parsování","msg_parsing","embedding","chunking")}
    timeout_counts={phase:0 for phase in ("parsování","chunking","embedding")}
    for error in errors:
        reason=error["důvod"].lower()
        for phase in timeout_counts:
            if phase in reason: timeout_counts[phase]+=1
    seconds=float(history.get("seconds",0) or 0); counts=history.get("counts",{}); processed=sum(int(counts.get(key,0) or 0) for key in ("added","changed","renamed","unchanged","duplicates","errors")); slow_chunks=sum(int(row.get("chunks",0) or 0) for row in slowest); phase_seconds=sum(float(row.get("seconds",0) or 0) for row in slowest)
    test_status=_latest_json(base/"state","test-status.json")
    verification=verify_index(db,lance,base/"database/.index.lock") if db.exists() else {"ok":False,"checks":{"SQLite integrita":False},"documents":0,"chunks":0,"embeddings":0,"missing_embeddings":0,"orphan_embeddings":0}
    rc1=verification["ok"] and bool(test_status.get("ok"))
    return {"version":APP_VERSION,"created":datetime.now().isoformat(timespec="seconds"),"paths":{"projekt":project_path,"SQLite":str(db),"LanceDB":str(lance),"cache":str(base/"cache"),"logy":str(base/"logs"),"state":str(base/"state")},"models":{"embedding":EMBEDDING_MODEL,"ollama":DEFAULT_MODEL},"sizes":{"database":db.stat().st_size if db.exists() else 0,"lance":folder_size(lance),"cache":folder_size(base/"cache")},"last_index":history.get("time","—"),"counts":{"documents":documents,"chunks":chunks,"embeddings":len(lance_ids)},"extensions":extension_stats,"watchdog":{"timeouts":timeout_counts,"slowest_document":slowest[0] if slowest else {},"slowest_parser":max((by_phase["parsování"],by_phase["msg_parsing"]),key=lambda row:row.get("seconds",0),default={}),"slowest_embedding":by_phase["embedding"],"slowest_chunking":by_phase["chunking"]},"performance":{"seconds":seconds,"documents_per_second":processed/seconds if seconds else 0,"chunks_per_second":slow_chunks/phase_seconds if phase_seconds else 0,"embeddings_per_second":slow_chunks/phase_seconds if phase_seconds else 0,"max_ram_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"max_cpu_seconds":resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime},"errors":errors,"verification":verification,"tests":test_status,"rc1":rc1,"lance_status":lance_status,"slowest":slowest}

def export_reports(data: dict,base: Path=APP_SUPPORT_DIR) -> tuple[Path,Path]:
    reports=base/"reports"; reports.mkdir(parents=True,exist_ok=True); stamp=time.strftime("%Y%m%d-%H%M%S"); html_path=reports/f"AI-Search-diagnostika-{stamp}.html"; pdf_path=reports/f"AI-Search-diagnostika-{stamp}.pdf"
    sections=[]
    for title,value in (("Stav systému",data["paths"]|data["models"]),("Stav indexu",data["verification"]),("Statistiky přípon",data["extensions"]),("Watchdog",data["watchdog"]),("Výkon",data["performance"]),("Nejpomalejší fáze",data["slowest"]),("Chyby",data["errors"])): sections.append(f"<h2>{html.escape(title)}</h2><pre>{html.escape(json.dumps(value,ensure_ascii=False,indent=2,default=str))}</pre>")
    html_path.write_text(f"<!doctype html><html lang='cs'><meta charset='utf-8'><title>AI Search diagnostika</title><style>body{{font:14px -apple-system,BlinkMacSystemFont,sans-serif;max-width:1100px;margin:40px auto;color:#172033}}h1{{color:#175cd3}}pre{{white-space:pre-wrap;background:#f4f7fb;padding:16px;border-radius:8px}}.rc{{padding:18px;background:{'#dcfce7' if data['rc1'] else '#fee2e2'};border-radius:10px}}</style><h1>AI Search - produkční diagnostika</h1><div class='rc'><b>AI SEARCH RC1</b><br>Produkční stav: {'PŘIPRAVENO' if data['rc1'] else 'NEPŘIPRAVENO'}</div><p>Vytvořeno: {html.escape(data['created'])} | Verze: {APP_VERSION}</p>{''.join(sections)}</html>",encoding="utf-8")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    from reportlab.lib import colors
    font_path=Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if not font_path.exists(): font_path=Path("/Library/Fonts/Arial Unicode.ttf")
    pdfmetrics.registerFont(TTFont("AISearchUnicode",str(font_path)))
    styles=getSampleStyleSheet()
    for style_name in ("Title","Heading2","BodyText","Code"): styles[style_name].fontName="AISearchUnicode"
    story=[Paragraph("AI Search - produkční diagnostika",styles["Title"]),Paragraph(f"AI SEARCH RC1 - {'PŘIPRAVENO' if data['rc1'] else 'NEPŘIPRAVENO'}",styles["Heading2"]),Paragraph(f"Vytvořeno: {data['created']} | Verze: {APP_VERSION}",styles["BodyText"]),Spacer(1,6*mm)]
    for title,value in (("Stav systému",data["paths"]|data["models"]),("Stav indexu",data["verification"]),("Statistiky",data["extensions"]),("Watchdog",data["watchdog"]),("Výkon",data["performance"]),("Chyby",data["errors"])):
        story.extend([Paragraph(title,styles["Heading2"]),Paragraph(html.escape(json.dumps(value,ensure_ascii=False,indent=2,default=str)).replace("\n","<br/>"),styles["Code"]),Spacer(1,4*mm)])
    SimpleDocTemplate(str(pdf_path),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=15*mm,bottomMargin=15*mm,title="AI Search diagnostika").build(story)
    return html_path,pdf_path
