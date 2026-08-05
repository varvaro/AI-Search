#!/usr/bin/env python3
"""Hybridní backend AI Search nad lokální složkou Box Drive."""
from __future__ import annotations
import argparse, collections, contextlib, fcntl, gc, hashlib, json, logging, multiprocessing, os, queue, re, resource, sqlite3, subprocess, tempfile, threading, time, traceback, urllib.request, uuid
from pathlib import Path
from ai_search_config import BOX_ROOT, STATE_DIR, EMBEDDING_MODEL, OLLAMA_ENDPOINT, DEFAULT_MODEL, COMPLEX_MODEL, PARSE_TIMEOUT_SECONDS, CHUNK_TIMEOUT_SECONDS, EMBEDDING_TIMEOUT_SECONDS, EMBEDDING_BATCH_SIZE, MSG_PARSE_TIMEOUT_SECONDS
from document_extractors import INDEXED_EXTS, extract_text

SUPPORTED = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".md", ".csv", ".rtf", ".eml", ".msg", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

def sha256_file(path):
    result = [None]; error = [None]
    def _read():
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
            result[0] = digest.hexdigest()
        except BaseException as exc: error[0] = exc
    t = threading.Thread(target=_read, daemon=True); t.start(); t.join(PARSE_TIMEOUT_SECONDS)
    if t.is_alive(): raise PhaseTimeout("hashování", PARSE_TIMEOUT_SECONDS)
    if error[0] is not None: raise error[0]
    return result[0]

def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True); con = sqlite3.connect(path,timeout=60)
    con.execute("PRAGMA busy_timeout=60000"); con.execute("PRAGMA synchronous=NORMAL"); con.execute("PRAGMA foreign_keys=ON")
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'").fetchone():
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript("""
    CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY,path TEXT UNIQUE NOT NULL,relative_path TEXT NOT NULL,name TEXT NOT NULL,project TEXT NOT NULL,content_hash TEXT UNIQUE NOT NULL,size INTEGER NOT NULL,mtime_ns INTEGER NOT NULL,inode INTEGER NOT NULL,extraction TEXT NOT NULL,indexed_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,ordinal INTEGER NOT NULL,heading TEXT NOT NULL,text TEXT NOT NULL);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED,name,relative_path,project,heading,body,tokenize='unicode61 remove_diacritics 2');
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS index_status(path TEXT PRIMARY KEY,content_hash TEXT NOT NULL DEFAULT '',status TEXT NOT NULL,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,error TEXT NOT NULL DEFAULT '');
        """)
    con.execute("CREATE TABLE IF NOT EXISTS index_status(path TEXT PRIMARY KEY,content_hash TEXT NOT NULL DEFAULT '',status TEXT NOT NULL,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,error TEXT NOT NULL DEFAULT '')")
    return con

@contextlib.contextmanager
def database(path):
    con=connect(path)
    try:
        yield con
        con.commit()
    except BaseException: con.rollback(); raise
    finally: con.close()

class IndexingInProgress(RuntimeError): pass
class PhaseTimeout(TimeoutError):
    def __init__(self,phase,limit,chunk_number=0): self.phase=phase; self.limit=limit; self.chunk_number=chunk_number; super().__init__(f"Fáze {phase} překročila limit {limit} s (chunk {chunk_number})")
_INDEX_LOCK=threading.Lock()

@contextlib.contextmanager
def index_guard(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not _INDEX_LOCK.acquire(blocking=False): raise IndexingInProgress("Indexace již probíhá.")
    handle=path.open("a+")
    try:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise IndexingInProgress("Indexace již probíhá.") from exc
        yield
    finally:
        try: fcntl.flock(handle,fcntl.LOCK_UN)
        finally: handle.close(); _INDEX_LOCK.release()

def index_logger(db_path):
    log_dir=db_path.parent.parent/"logs"; log_dir.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger("ai_search.index."+hashlib.sha1(str(db_path).encode()).hexdigest())
    if not logger.handlers:
        logger.setLevel(logging.INFO); handler=logging.FileHandler(log_dir/(db_path.stem+".log"),encoding="utf-8"); handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")); logger.addHandler(handler); logger.propagate=False
    return logger

def report(progress, logger, phase, current=0, total=0, path=None, started=None):
    elapsed=time.perf_counter()-started if started else 0; eta=(elapsed/current*(total-current)) if current and total else None
    event={"phase":phase,"current":current,"total":total,"path":str(path) if path else "","elapsed":elapsed,"eta":eta}; logger.info("%s %s/%s %s",phase,current,total,event["path"])
    if progress: progress(event)

def save_status(db_path,path,digest,status,error=""):
    with database(db_path) as con:
        con.execute("INSERT INTO index_status(path,content_hash,status,error) VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET content_hash=excluded.content_hash,status=excluded.status,updated_at=CURRENT_TIMESTAMP,error=excluded.error",(str(path),digest or "",status,error[:2000]))

def encode_batched(embeddings,texts,batch_size=EMBEDDING_BATCH_SIZE):
    output=[]
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        try: output.extend(embeddings.encode(batch,batch_size=batch_size))
        except TypeError: output.extend(embeddings.encode(batch))
    return output

def memory_bytes(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def _embedding_worker(model_name,requests,responses):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(model_name)
    while True:
        item=requests.get()
        if item is None: return
        request_id,texts,batch_size=item
        try: responses.put((request_id,model.encode(texts,normalize_embeddings=True,batch_size=batch_size).tolist(),None))
        except BaseException: responses.put((request_id,None,traceback.format_exc()))

class EmbeddingWatchdog:
    def __init__(self,embeddings):
        self.embeddings=embeddings; self.process=None; self.context=multiprocessing.get_context("spawn") if isinstance(embeddings,Embeddings) else None
    def start(self):
        self.requests=self.context.Queue(); self.responses=self.context.Queue(); self.process=self.context.Process(target=_embedding_worker,args=(self.embeddings.name,self.requests,self.responses),daemon=True); self.process.start()
    def encode(self,texts,limit,progress=None):
        if not texts: return []
        if not self.context:
            started=time.monotonic(); result=encode_batched(self.embeddings,texts)
            if time.monotonic()-started>limit: raise PhaseTimeout("embedding",limit,len(texts))
            return result
        if not self.process or not self.process.is_alive(): self.start()
        started=time.monotonic(); output=[]
        for offset in range(0,len(texts),EMBEDDING_BATCH_SIZE):
            request_id=uuid.uuid4().hex; batch=texts[offset:offset+EMBEDDING_BATCH_SIZE]; self.requests.put((request_id,batch,EMBEDDING_BATCH_SIZE))
            while True:
                elapsed=time.monotonic()-started; chunk_number=offset+1
                if progress: progress(elapsed,chunk_number,len(output))
                if elapsed>=limit:
                    self.process.terminate(); self.process.join(5)
                    if self.process.is_alive(): self.process=None
                    raise PhaseTimeout("embedding",limit,chunk_number)
                try: response_id,vectors,error=self.responses.get(timeout=min(.25,limit-elapsed))
                except queue.Empty: continue
                if response_id!=request_id: continue
                if error: raise RuntimeError(error)
                output.extend(vectors); break
        return output
    def close(self):
        if self.process and self.process.is_alive(): self.requests.put(None); self.process.join(5)
        if self.process and self.process.is_alive():
            self.process.terminate(); self.process.join(5)
            if self.process.is_alive(): self.process=None

def _parsing_worker(requests,responses):
    while True:
        item=requests.get()
        if item is None: return
        request_id,path=item
        try: responses.put((request_id,extract(Path(path)),None))
        except BaseException: responses.put((request_id,None,traceback.format_exc()))

class ParsingWatchdog:
    def __init__(self): self.context=multiprocessing.get_context("spawn"); self.process=None; self.requests=None; self.responses=None
    def start(self):
        self.requests=self.context.Queue(); self.responses=self.context.Queue(); self.process=self.context.Process(target=_parsing_worker,args=(self.requests,self.responses),daemon=True); self.process.start()
    def parse(self,path,limit,progress=None):
        if extract.__module__!=__name__:
            started=time.monotonic(); result=extract(path)
            if time.monotonic()-started>limit: raise PhaseTimeout("parsování",limit,0)
            return result
        if not self.process or not self.process.is_alive(): self.start()
        request_id=uuid.uuid4().hex; self.requests.put((request_id,str(path))); started=time.monotonic()
        while True:
            elapsed=time.monotonic()-started
            if progress: progress(elapsed)
            if elapsed>=limit:
                self.process.terminate(); self.process.join(5)
                if self.process.is_alive(): self.process=None
                raise PhaseTimeout("parsování",limit,0)
            try: response_id,result,error=self.responses.get(timeout=min(.25,limit-elapsed))
            except queue.Empty: continue
            if response_id!=request_id: continue
            if error: raise RuntimeError(error)
            return result
    def close(self):
        if self.process and self.process.is_alive(): self.requests.put(None); self.process.join(5)
        if self.process and self.process.is_alive():
            self.process.terminate(); self.process.join(5)
            if self.process.is_alive(): self.process=None

def extract_outlook_msg(path):
    import extract_msg
    with extract_msg.openMsg(path) as message:
        attachments=[getattr(item,"longFilename",None) or getattr(item,"shortFilename",None) for item in message.attachments]
        fields=[f"Předmět: {message.subject or ''}",f"Odesílatel: {message.sender or ''}",f"Komu: {message.to or ''}",f"Kopie: {message.cc or ''}",f"Datum: {message.date or ''}","",message.body or ""]
        if attachments: fields.extend(["","Přílohy: "+", ".join(name for name in attachments if name)])
        text="\n".join(fields).strip()
        if not text: raise ValueError("Outlook MSG neobsahuje čitelná pole ani tělo")
        return text,"extract-msg"

def _msg_worker(requests,responses):
    while True:
        item=requests.get()
        if item is None: return
        request_id,path=item
        try: responses.put((request_id,extract_outlook_msg(Path(path)),None))
        except BaseException: responses.put((request_id,None,traceback.format_exc()))

class MsgParsingWatchdog(ParsingWatchdog):
    def start(self):
        self.requests=self.context.Queue(); self.responses=self.context.Queue(); self.process=self.context.Process(target=_msg_worker,args=(self.requests,self.responses),daemon=True); self.process.start()
    def parse(self,path,limit,progress=None):
        if extract_outlook_msg.__module__!=__name__:
            started=time.monotonic(); result=extract_outlook_msg(path)
            if time.monotonic()-started>limit: raise PhaseTimeout("msg_parsing",limit,0)
            return result
        if not self.process or not self.process.is_alive(): self.start()
        request_id=uuid.uuid4().hex; self.requests.put((request_id,str(path))); started=time.monotonic()
        while True:
            elapsed=time.monotonic()-started
            if progress: progress(elapsed)
            if elapsed>=limit:
                self.process.terminate(); self.process.join(5)
                if self.process.is_alive(): self.process=None
                raise PhaseTimeout("msg_parsing",limit,0)
            try: response_id,result,error=self.responses.get(timeout=min(.25,limit-elapsed))
            except queue.Empty: continue
            if response_id!=request_id: continue
            if error: raise RuntimeError(error)
            return result

def iter_documents(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            path = Path(directory) / name
            if not name.startswith(".") and path.suffix.lower() in SUPPORTED: yield path

def extract(path):
    ext = path.suffix.lower()
    if ext == ".pdf": return extract_pdf(path), "stav_skenu_pdf"
    if ext in INDEXED_EXTS: return extract_text(path)
    if ext in {".txt", ".md", ".csv", ".rtf", ".eml"}: return path.read_text(encoding="utf-8", errors="replace"), "text"
    if ext in {".doc", ".xls"}:
        run = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)], capture_output=True, check=False, timeout=PARSE_TIMEOUT_SECONDS)
        return run.stdout.decode("utf-8", errors="replace"), "textutil"
    run = subprocess.run(["/opt/homebrew/bin/tesseract", str(path), "stdout", "-l", "ces+eng"], capture_output=True, check=False, timeout=PARSE_TIMEOUT_SECONDS)
    return run.stdout.decode("utf-8", errors="replace"), "stav_skenu_ocr"

def extract_pdf(path, budget_seconds=180):
    deadline=time.monotonic()+budget_seconds
    try: direct=subprocess.run(["/opt/homebrew/bin/pdftotext","-layout",str(path),"-"],capture_output=True,text=True,check=False,timeout=min(30,budget_seconds)).stdout
    except subprocess.TimeoutExpired as exc: raise TimeoutError("Textová vrstva PDF překročila časový limit") from exc
    if len(direct.strip())>=80: return direct
    with tempfile.TemporaryDirectory(prefix="ai-search-ocr-",dir="/private/tmp") as folder:
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError("PDF překročilo časový limit před OCR")
        try: rendered=subprocess.run(["/opt/homebrew/bin/pdftoppm","-r","300","-png",str(path),str(Path(folder)/"page")],capture_output=True,text=True,check=False,timeout=min(60,remaining))
        except subprocess.TimeoutExpired as exc: raise TimeoutError("Převod PDF pro OCR překročil časový limit") from exc
        if rendered.returncode: raise RuntimeError("PDF nelze převést pro OCR: "+(rendered.stderr.strip() or str(rendered.returncode)))
        parts=[]
        for image in sorted(Path(folder).glob("page-*.png")):
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError("OCR PDF překročilo celkový časový limit")
            try: run=subprocess.run(["/opt/homebrew/bin/tesseract",str(image),"stdout","-l","ces+eng","--psm","6"],capture_output=True,text=True,check=False,timeout=min(30,remaining))
            except subprocess.TimeoutExpired as exc: raise TimeoutError("OCR stránky překročilo časový limit") from exc
            if run.returncode: raise RuntimeError("OCR stránky selhalo: "+(run.stderr.strip() or str(run.returncode)))
            parts.append(run.stdout)
        return "\n".join(parts)

def chunks(text, deadline=None, progress=None):
    result, block, heading = [], [], ""
    def flush():
        value = "\n".join(block).strip(); block.clear()
        if value: result.append((heading, value))
    for line_number,line in enumerate(text.replace("\r\n", "\n").split("\n"),1):
        if deadline and time.monotonic()>=deadline: raise PhaseTimeout("chunking",CHUNK_TIMEOUT_SECONDS,len(result)+1)
        if progress and line_number%250==0: progress(len(result)+1)
        value = line.strip()
        is_heading = bool(re.match(r"^(#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+|[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ\s-]{3,})", value))
        if is_heading and len(value) < 160: flush(); heading = re.sub(r"^#{1,6}\s+", "", value)
        elif not value: flush()
        else: block.append(line)
    flush(); return result or ([("", text.strip())] if text.strip() else [])

class Embeddings:
    def __init__(self, name=EMBEDDING_MODEL):
        self.name=name; self.model=None
    def encode(self, texts, **kwargs):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(self.name)
        return self.model.encode(texts,normalize_embeddings=True,**kwargs).tolist()

def lance_table(directory, dimension=None):
    import lancedb
    db = lancedb.connect(directory)
    if "chunks" in db.table_names(): return db.open_table("chunks")
    return db.create_table("chunks", data=[{"id":"__init__","vector":[0.0]*dimension,"document_id":0}]) if dimension else None

def sync(root, db_path, lance_dir, embeddings, progress=None, stop_event=None):
    root=Path(root); db_path=Path(db_path); lance_dir=Path(lance_dir); logger=index_logger(db_path); started=time.perf_counter()
    counts={k:0 for k in ("added","changed","renamed","removed","unchanged","duplicates","skipped","errors","timeouts")}; counts["stopped"]=False
    raw_progress=progress
    if raw_progress: progress=lambda event: raw_progress({**event,"counts":counts.copy(),"memory_bytes":memory_bytes()})
    with index_guard(db_path.parent/".index.lock"):
        parsing_watchdog=ParsingWatchdog(); msg_watchdog=MsgParsingWatchdog(); embedding_watchdog=EmbeddingWatchdog(embeddings); phase_records=[]; extension_stats=collections.defaultdict(lambda:{"total":0,"successful":0,"skipped":0,"timeouts":0,"errors":0,"parse_seconds":0.0})
        logger.info("START INDEXACE\nkořenová složka: %s",root)
        report(progress,logger,"SQLite otevřena",started=started)
        with database(db_path) as con:
            known={r[0]:r for r in con.execute("SELECT path,id,content_hash,size,mtime_ns,inode FROM documents")}
        report(progress,logger,"Načítám soubory",started=started)
        paths=list(iter_documents(root)); current={str(p):p.stat() for p in paths}; total=len(paths)
        logger.info("celkem nalezených kandidátů: %s",total); by_inode={r[5]:r for r in known.values()}; by_hash={r[2]:r for r in known.values()}
        for number,path in enumerate(paths,1):
            if stop_event and stop_event.is_set(): counts["stopped"]=True; logger.info("STOP vyžádán; bezpečně ukončuji před dokumentem %s",number); break
            absolute=str(path); stat=current[absolute]; old=known.get(absolute); extension=path.suffix.upper().lstrip(".") or "BEZ_PŘÍPONY"; extension_stats[extension]["total"]+=1
            document_started=time.perf_counter(); digest=old[2] if old else ""; current_phase="metadata"
            try:
                if old and old[3:] == (stat.st_size,stat.st_mtime_ns,stat.st_ino):
                    counts["unchanged"]+=1; extension_stats[extension]["skipped"]+=1; save_status(db_path,path,digest,"NEZMĚNĚNÝ"); report(progress,logger,"NEZMĚNĚNÝ",number,total,path,started); continue
                report(progress,logger,"Hashování",number,total,path,started); digest=sha256_file(path); owner=by_hash.get(digest)
                renamed=not old and ((stat.st_ino in by_inode and by_inode[stat.st_ino][2]==digest) or (owner and owner[0] not in current))
                if renamed:
                    old=by_inode.get(stat.st_ino,owner)
                    report(progress,logger,"SQLite BEGIN",number,total,path,started)
                    with database(db_path) as con:
                        con.execute("BEGIN IMMEDIATE")
                        con.execute("UPDATE documents SET path=?,relative_path=?,name=?,size=?,mtime_ns=?,inode=? WHERE id=?",(absolute,str(path.relative_to(root)),path.name,stat.st_size,stat.st_mtime_ns,stat.st_ino,old[1]))
                    logger.info("SQLite COMMIT"); counts["renamed"]+=1; extension_stats[extension]["successful"]+=1; save_status(db_path,path,digest,"AKTUALIZOVANÝ"); known.pop(old[0],None); continue
                if owner and (not old or owner[1]!=old[1]):
                    counts["duplicates"]+=1; counts["skipped"]+=1; extension_stats[extension]["skipped"]+=1; save_status(db_path,path,digest,"DUPLIKÁT"); report(progress,logger,"DUPLIKÁT",number,total,path,started); continue
                def phase_progress(phase,elapsed,chunk_number=0,processed_chunks=0):
                    if progress: progress({"phase":phase,"current":number,"total":total,"path":str(path),"elapsed":time.perf_counter()-started,"eta":None,"phase_elapsed":elapsed,"chunk_number":chunk_number,"chunks_per_second":processed_chunks/elapsed if elapsed else 0})
                current_phase="msg_parsing" if path.suffix.lower()==".msg" else "parsování"; phase_started=time.monotonic(); parser=msg_watchdog if path.suffix.lower()==".msg" else parsing_watchdog; phase_name="MSG parsování" if path.suffix.lower()==".msg" else "Parsování"; limit=MSG_PARSE_TIMEOUT_SECONDS if path.suffix.lower()==".msg" else PARSE_TIMEOUT_SECONDS; report(progress,logger,phase_name,number,total,path,started); text,method=parser.parse(path,limit,lambda elapsed:phase_progress(phase_name,elapsed)); parse_seconds=time.monotonic()-phase_started; extension_stats[extension]["parse_seconds"]+=parse_seconds; phase_records.append({"document":str(path),"extension":extension,"phase":current_phase,"seconds":parse_seconds,"chunks":0})
                current_phase="chunking"; phase_started=time.monotonic(); report(progress,logger,"Chunking",number,total,path,started); pieces=chunks(text,deadline=time.monotonic()+CHUNK_TIMEOUT_SECONDS,progress=lambda chunk:phase_progress("Chunking",time.monotonic()-phase_started,chunk,chunk)); phase_records.append({"document":str(path),"phase":"chunking","seconds":time.monotonic()-phase_started,"chunks":len(pieces)})
                current_phase="embedding"; phase_started=time.monotonic(); report(progress,logger,"Embedding",number,total,path,started); vectors=embedding_watchdog.encode([body for _,body in pieces],EMBEDDING_TIMEOUT_SECONDS,lambda elapsed,chunk,processed:phase_progress("Embedding",elapsed,chunk,processed)); phase_records.append({"document":str(path),"phase":"embedding","seconds":time.monotonic()-phase_started,"chunks":len(vectors)})
                additions=[{"id":f"{digest}:{ordinal}","vector":vector,"document_id":old[1] if old else -1} for ordinal,vector in enumerate(vectors)]
                old_chunk_ids=[]
                if old:
                    with database(db_path) as con: old_chunk_ids=[r[0] for r in con.execute("SELECT id FROM chunks WHERE document_id=?",(old[1],))]
                table=lance_table(lance_dir,len(vectors[0]) if vectors else None); lance_version=table.version if table and (old_chunk_ids or additions) else None
                try:
                    report(progress,logger,"SQLite BEGIN",number,total,path,started)
                    with database(db_path) as con:
                        con.execute("BEGIN IMMEDIATE")
                        if old:
                            doc_id=old[1]; con.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",(doc_id,)); con.execute("DELETE FROM chunks WHERE document_id=?",(doc_id,)); counts["changed"]+=1
                        else: doc_id=None; counts["added"]+=1
                        con.execute("""INSERT INTO documents(id,path,relative_path,name,project,content_hash,size,mtime_ns,inode,extraction)
                            VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO UPDATE SET path=excluded.path,relative_path=excluded.relative_path,name=excluded.name,project=excluded.project,content_hash=excluded.content_hash,size=excluded.size,mtime_ns=excluded.mtime_ns,inode=excluded.inode,extraction=excluded.extraction,indexed_at=CURRENT_TIMESTAMP""",
                            (doc_id,absolute,str(path.relative_to(root)),path.name,root.name,digest,stat.st_size,stat.st_mtime_ns,stat.st_ino,method))
                        doc_id=con.execute("SELECT id FROM documents WHERE path=?",(absolute,)).fetchone()[0]
                        for ordinal,(heading,body) in enumerate(pieces):
                            cid=f"{digest}:{ordinal}"; con.execute("INSERT INTO chunks VALUES(?,?,?,?,?)",(cid,doc_id,ordinal,heading,body)); con.execute("INSERT INTO chunks_fts(chunk_id,name,relative_path,project,heading,body) VALUES(?,?,?,?,?,?)",(cid,path.name,str(path.relative_to(root)),root.name,heading,body))
                        report(progress,logger,"LanceDB zápis",number,total,path,started)
                        if table and old_chunk_ids: table.delete("id IN ("+ ",".join(json.dumps(i) for i in old_chunk_ids)+")")
                        if table and additions:
                            table.delete('id = "__init__"')
                            for row in additions: row["document_id"]=doc_id
                            table.add(additions)
                    logger.info("SQLite COMMIT")
                except Exception:
                    logger.exception("ROLLBACK")
                    if table and lance_version is not None:
                        logger.info("Obnovuji LanceDB verzi %s",lance_version); table.checkout(lance_version); table.restore(); table.checkout_latest()
                    raise
                by_hash[digest]=(absolute,doc_id,digest,stat.st_size,stat.st_mtime_ns,stat.st_ino)
                status="AKTUALIZOVANÝ" if old else "NOVÝ"; save_status(db_path,path,digest,status); logger.info("[%s/%s] %s %s %.2f s velikost=%s",number,total,path,status,time.perf_counter()-document_started,stat.st_size)
                extension_stats[extension]["successful"]+=1
                del text,pieces,vectors,additions; gc.collect()
                if memory_bytes()>12*1024**3: logger.warning("RAM překročila 12 GB; řízená pauza"); time.sleep(.25)
            except PhaseTimeout as exc:
                counts["errors"]+=1; counts["timeouts"]+=1; extension_stats[extension]["timeouts"]+=1; is_msg_parse=path.suffix.lower()==".msg" and exc.phase=="msg_parsing"; error_status="ERROR_MSG_PARSE" if is_msg_parse else "ERROR_TIMEOUT"; phase_records.append({"document":str(path),"extension":extension,"phase":exc.phase,"seconds":exc.limit,"chunks":exc.chunk_number,"timeout":True});
                if is_msg_parse: extension_stats[extension]["parse_seconds"]+=exc.limit
                logger.exception("[%s/%s] %s %s fáze=%s chunk=%s",number,total,path,error_status,exc.phase,exc.chunk_number)
                try: save_status(db_path,path,digest,error_status,f"fáze={exc.phase}; chunk={exc.chunk_number}; {exc}")
                except Exception: logger.exception("Nelze uložit stav timeoutu")
                report(progress,logger,error_status,number,total,path,started); continue
            except Exception as exc:
                counts["errors"]+=1; extension_stats[extension]["errors"]+=1; is_msg_parse=path.suffix.lower()==".msg" and current_phase=="msg_parsing"; error_status="ERROR_MSG_PARSE" if is_msg_parse else "CHYBA"
                if is_msg_parse:
                    failed_seconds=time.monotonic()-phase_started; extension_stats[extension]["parse_seconds"]+=failed_seconds; phase_records.append({"document":str(path),"extension":extension,"phase":"msg_parsing","seconds":failed_seconds,"chunks":0,"error":True})
                logger.exception("[%s/%s] %s %s",number,total,path,error_status)
                try: save_status(db_path,path,digest,error_status,str(exc))
                except Exception: logger.exception("Nelze uložit stav chyby")
                report(progress,logger,error_status,number,total,path,started); continue
            except BaseException:
                parsing_watchdog.close(); msg_watchdog.close(); embedding_watchdog.close(); raise
        for absolute,row in list(known.items()):
            if counts["stopped"]: break
            if absolute in current: continue
            with database(db_path) as con: old_ids=[r[0] for r in con.execute("SELECT id FROM chunks WHERE document_id=?",(row[1],))]
            table=lance_table(lance_dir); lance_version=table.version if table and old_ids else None
            try:
                with database(db_path) as con:
                    con.execute("BEGIN IMMEDIATE"); con.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",(row[1],)); con.execute("DELETE FROM documents WHERE id=?",(row[1],))
                    if table and old_ids: table.delete("id IN ("+ ",".join(json.dumps(i) for i in old_ids)+")")
                counts["removed"]+=1
            except Exception:
                if table and lance_version is not None: table.checkout(lance_version); table.restore(); table.checkout_latest()
                raise
        with database(db_path) as con:
            con.execute("INSERT OR REPLACE INTO settings VALUES('root',?)",(str(root),)); con.execute("INSERT OR REPLACE INTO settings VALUES('embedding_model',?)",(EMBEDDING_MODEL,))
        parsing_watchdog.close(); msg_watchdog.close(); embedding_watchdog.close(); slow_report=sorted(phase_records,key=lambda row:row["seconds"],reverse=True)[:20]; report_path=db_path.parent.parent/"logs"/(db_path.stem+"-slow-phases.json"); report_path.write_text(json.dumps({"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"slowest":slow_report,"by_extension":dict(sorted(extension_stats.items()))},ensure_ascii=False,indent=2),encoding="utf-8"); counts["slow_report"]=str(report_path); counts["by_extension"]=dict(extension_stats)
        elapsed=time.perf_counter()-started; completed=sum(counts[k] for k in ("added","changed","renamed","unchanged","duplicates","errors")); report(progress,logger,"Zastaveno" if counts["stopped"] else "Hotovo",completed,total,started=started)
        logger.info("Celkem=%s Nové=%s Aktualizované=%s Nezměněné=%s Duplicitní=%s Přeskočené=%s Chyby=%s Celkový čas=%.2f Průměrný čas/dokument=%.3f",total,counts["added"],counts["changed"]+counts["renamed"],counts["unchanged"],counts["duplicates"],counts["skipped"],counts["errors"],elapsed,elapsed/max(completed,1)); logger.info("SQLite zavřena; uzavírám LanceDB; konec")
        return counts

def search(query, db_path, lance_dir, embeddings, limit=8):
    terms=" OR ".join('"'+t+'"' for t in re.findall(r"\w+",query) if len(t)>1)
    with database(db_path) as con: lexical=con.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",(terms,limit*4)).fetchall() if terms else []
    scores={cid:1/(60+rank) for rank,(cid,) in enumerate(lexical)}; table=lance_table(lance_dir)
    if table:
        for rank,row in enumerate(table.search(embeddings.encode([query])[0]).limit(limit*4).to_list()): scores[row["id"]]=scores.get(row["id"],0)+1/(60+rank)
    # Přesná shoda dotazu v názvu dokumentu je silnější důkaz než
    # sémantická podobnost obecného textu dodacího listu.
    needle=query.casefold().strip(); output=[]
    with database(db_path) as con:
        for cid,score in sorted(scores.items(),key=lambda x:x[1],reverse=True)[:limit]:
            row=con.execute("SELECT d.name,d.path,d.project,c.heading,c.text FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?",(cid,)).fetchone()
            if row:
                if needle and needle in row[0].casefold(): score += 1.0
                output.append({"document":row[0],"path":row[1],"project":row[2],"quote":row[4][:700],"heading":row[3],"score":score})
    return sorted(output,key=lambda row:row["score"],reverse=True)

def answer(query, results):
    if not results: return {"answer":"Odpověď nelze vytvořit bez citací.","citations":[]}
    context="\n\n".join(f"[{i}] {r['document']} | {r['path']} | projekt {r['project']}\n{r['quote']}" for i,r in enumerate(results,1)); model=COMPLEX_MODEL if len(query)>180 or len(results)>6 else DEFAULT_MODEL
    payload=json.dumps({"model":model,"stream":False,"think":False,"prompt":f"Odpověz česky pouze podle zdrojů a cituj [1], [2].\nDOTAZ: {query}\nZDROJE:\n{context}"}).encode(); req=urllib.request.Request(OLLAMA_ENDPOINT,data=payload,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=240) as response:
            generated=json.loads(response.read())["response"]
    except Exception as exc:
        return {"answer":f"Ollama je nedostupná: {type(exc).__name__}. Nalezené citace zůstávají k dispozici.","citations":results,"model":model,"error":str(exc)}
    return {"answer":generated,"citations":results,"model":model}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=BOX_ROOT); parser.add_argument("--state-dir",type=Path,default=STATE_DIR); sub=parser.add_subparsers(dest="command",required=True); sub.add_parser("index"); sp=sub.add_parser("search"); sp.add_argument("query"); ap=sub.add_parser("answer"); ap.add_argument("query"); args=parser.parse_args(); emb=Embeddings(); db=args.state_dir/"ai_search.sqlite3"; lance=args.state_dir/"lancedb"
    result=sync(args.root,db,lance,emb) if args.command=="index" else search(args.query,db,lance,emb); result=answer(args.query,result) if args.command=="answer" else result; print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
