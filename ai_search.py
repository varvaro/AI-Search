#!/usr/bin/env python3
"""Hybridní backend AI Search nad lokální složkou Box Drive."""
from __future__ import annotations
import argparse, collections, contextlib, fcntl, gc, hashlib, json, logging, multiprocessing, os, queue, re, resource, sqlite3, subprocess, tempfile, threading, time, traceback, urllib.request, uuid
import numpy as np
from pathlib import Path
from ai_search_config import BOX_ROOT, STATE_DIR, EMBEDDING_MODEL, OLLAMA_ENDPOINT, DEFAULT_MODEL, COMPLEX_MODEL, PARSE_TIMEOUT_SECONDS, CHUNK_TIMEOUT_SECONDS, EMBEDDING_TIMEOUT_SECONDS, EMBEDDING_BATCH_SIZE, MSG_PARSE_TIMEOUT_SECONDS
from document_extractors import INDEXED_EXTS, extract_text, extract_eml, clean_cell_text, format_sheet_section
import parsing_worker  # stable multiprocessing.Process targets, see parsing_worker.py docstring
import query_expansion  # Query Understanding layer, opt-in via search(expand_query=True)

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

def run_with_timeout(operation, timeout_seconds, phase_name):
    """Run operation in daemon thread with timeout."""
    result = [None]; error = [None]
    def _run():
        try: result[0] = operation()
        except BaseException as exc: error[0] = exc
    t = threading.Thread(target=_run, daemon=True); t.start(); t.join(timeout_seconds)
    if t.is_alive(): raise PhaseTimeout(phase_name, timeout_seconds)
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
class ParseCancelled(RuntimeError):
    """Raised by ParsingWatchdog.parse() when stop_event fires mid-parse, so sync()
    can tell a user-requested STOP apart from ERROR_TIMEOUT/CHYBA and abort cleanly
    instead of recording a spurious error for the in-flight document."""
    pass
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

class ParsingWatchdog:
    def __init__(self): self.context=multiprocessing.get_context("spawn"); self.process=None; self.requests=None; self.responses=None
    def start(self):
        self.requests=self.context.Queue(); self.responses=self.context.Queue(); self.process=self.context.Process(target=parsing_worker.parsing_worker_main,args=(self.requests,self.responses),daemon=True); self.process.start()
    def parse(self,path,limit,progress=None,stop_event=None):
        if extract.__module__!=__name__:
            started=time.monotonic(); result=extract(path)
            if time.monotonic()-started>limit: raise PhaseTimeout("parsování",limit,0)
            return result
        if not self.process or not self.process.is_alive(): self.start()
        request_id=uuid.uuid4().hex; self.requests.put((request_id,str(path))); started=time.monotonic()
        while True:
            elapsed=time.monotonic()-started
            if progress: progress(elapsed)
            # Checked every poll tick (<=0.25s) so STOP takes effect during a
            # single long-running parse (e.g. per-page PDF OCR) instead of only
            # between documents - see extract_pdf()'s per-page loop, which has
            # no direct access to stop_event since it runs in a separate
            # (spawned) process; killing the whole worker is the only lever we
            # have from here, same as the timeout branch below.
            if stop_event is not None and stop_event.is_set():
                self.process.terminate(); self.process.join(5)
                if self.process.is_alive(): self.process=None
                raise ParseCancelled("STOP vyžádán během parsování")
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

class MsgParsingWatchdog(ParsingWatchdog):
    def start(self):
        self.requests=self.context.Queue(); self.responses=self.context.Queue(); self.process=self.context.Process(target=parsing_worker.msg_parsing_worker_main,args=(self.requests,self.responses),daemon=True); self.process.start()
    def parse(self,path,limit,progress=None,stop_event=None):
        # stop_event accepted only for call-site signature compatibility with
        # ParsingWatchdog.parse() (sync() calls both through one `parser`
        # variable); MSG parsing/STOP behaviour is unchanged and out of scope
        # here - mid-parse STOP still isn't checked below, exactly as before.
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

# Legacy BIFF .xls files are OLE Compound File containers (see extract_xls()
# docstring for the forensic finding that made this necessary): naively
# converting the whole container to text (as the old textutil path did) picks
# up megabytes of OLE/binary structure, producing dozens of huge nonsense
# chunks whose cumulative embedding time blows past EMBEDDING_TIMEOUT_SECONDS.
# Real cell content in the production corpus is tiny (max observed: 505 rows x
# 55 cols) - these ceilings are a defensive backstop against a hypothetical
# pathological DIMENSIONS record, not a tuning knob for real files.
XLS_MAX_ROWS = 50000
XLS_MAX_COLS = 1000

def _xls_cell_text(cell, datemode):
    import xlrd
    if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK): return ""
    if cell.ctype == xlrd.XL_CELL_TEXT: text = str(cell.value)
    elif cell.ctype == xlrd.XL_CELL_NUMBER: text = f"{cell.value:.10g}"
    elif cell.ctype == xlrd.XL_CELL_BOOLEAN: text = "TRUE" if cell.value else "FALSE"
    elif cell.ctype == xlrd.XL_CELL_ERROR:
        text = xlrd.biffh.error_text_from_code.get(cell.value, f"#ERR{cell.value}")
    elif cell.ctype == xlrd.XL_CELL_DATE:
        try:
            moment = xlrd.xldate_as_datetime(cell.value, datemode)
            text = moment.strftime("%Y-%m-%d") if moment.time() == moment.min.time() else moment.strftime("%Y-%m-%d %H:%M")
        except xlrd.xldate.XLDateError: text = f"{cell.value:.10g}"
    else: text = str(cell.value)
    return clean_cell_text(text)

def _xls_sheet_rows(sheet, datemode):
    """Raw cell rows of one sheet, blank rows included - format_sheet_section()
    turns a blank row into the block separator that lets chunks() split the
    sheet on its own structure."""
    return [[_xls_cell_text(sheet.cell(r, c), datemode) for c in range(min(sheet.ncols, XLS_MAX_COLS))]
            for r in range(min(sheet.nrows, XLS_MAX_ROWS))]

def extract_xls(path):
    """Parse legacy BIFF .xls as an actual spreadsheet instead of feeding its
    raw OLE container bytes to textutil.

    Forensic finding behind this function (production audit, 2026-08-07):
    every .xls hitting ERROR_TIMEOUT in the embedding phase turned out to have
    real cell content of a few hundred rows at most (verified with xlrd on
    the actual production files - e.g. "Garáže NDS_ZL Best Truck 7.11.2025.xls"
    is a 3MB OLE file but its 4 sheets contain only 9/62/14/14 rows of real
    change-order data), yet the old `textutil -convert txt` path produced ~3MB
    of "text" per file - textutil has no concept of the BIFF record structure,
    so it dumps the whole OLE compound file (formatting, embedded objects,
    revision remnants, etc.) as if it were prose, and that binary noise
    chunks into dozens of huge nonsense blocks whose cumulative embedding
    time exceeds EMBEDDING_TIMEOUT_SECONDS. xlrd reads only the actual BIFF
    Workbook stream's cell records, so the extracted text is bounded by real
    content, not container size.

    No textutil fallback on failure by design (see project audit item 9):
    silently re-indexing OLE garbage is worse than a clear extraction error.
    A handful of production .xls (~2%, all small files) hit a known xlrd 2.0.x
    limitation unpacking certain shared-string-table records
    (AssertionError deep in xlrd.book.unpack_SST_table) - this is a rare
    pathological xlrd limitation on specific files, not a systemic gap, and is
    intentionally surfaced as an extraction error rather than worked around
    with a second parser library.
    """
    import xlrd
    try:
        book = xlrd.open_workbook(str(path), formatting_info=False, on_demand=True)
    except Exception as exc:
        raise ValueError(f"XLS soubor nelze otevřít jako platný sešit (OLE/BIFF): {type(exc).__name__}: {exc}") from exc
    try:
        sections = []
        for sheet_index in range(book.nsheets):
            try:
                sheet = book.sheet_by_index(sheet_index)
            except Exception as exc:
                raise ValueError(f"List č. {sheet_index+1} v XLS nelze načíst: {type(exc).__name__}: {exc}") from exc
            # Hidden sheets are not skipped: visibility is a presentation
            # choice in Excel, not a signal that the data is irrelevant for
            # retrieval - a hidden calculation/lookup sheet can still hold
            # answer-bearing content (see project audit item 6).
            section = format_sheet_section(sheet.name, _xls_sheet_rows(sheet, book.datemode))
            if section: sections.append(section)
        return "\n\n".join(sections)
    finally:
        book.release_resources()

def extract(path):
    ext = path.suffix.lower()
    if ext == ".pdf": return extract_pdf(path), "stav_skenu_pdf"
    if ext == ".eml": return extract_eml(path), "email_mime"
    if ext == ".xls": return extract_xls(path), "xls_biff"
    if ext in INDEXED_EXTS: return extract_text(path)
    if ext in {".txt", ".md", ".csv"}: return path.read_text(encoding="utf-8", errors="replace"), "text"
    if ext in {".doc", ".rtf"}:
        # .rtf must go through textutil like .doc: reading it as raw text would
        # feed RTF control words and hex-encoded embedded binaries (images)
        # straight into the chunker as if they were prose.
        run = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)], capture_output=True, check=False, timeout=PARSE_TIMEOUT_SECONDS)
        return run.stdout.decode("utf-8", errors="replace"), "textutil"
    run = subprocess.run(["/opt/homebrew/bin/tesseract", str(path), "stdout", "-l", "ces+eng"], capture_output=True, check=False, timeout=PARSE_TIMEOUT_SECONDS)
    return run.stdout.decode("utf-8", errors="replace"), "stav_skenu_ocr"

# --- PDF OCR: per-page architecture -----------------------------------------
# Forensic root cause (production audit): the old extract_pdf() rendered the
# WHOLE document in a single pdftoppm call and OCR'd each page with
# per-subprocess timeouts hard-capped at min(60,remaining)/min(30,remaining).
# Measured on real production files (read-only, see task):
#   - SoD_NOT243136_Zakládání group - podepsaná.pdf: 43 pages, 17.3 MB, no text
#     layer. Rendering ALL 43 pages in one pdftoppm call takes ~77.8s -
#     deterministically above the old 60s cap, regardless of per-page speed
#     (individually, its pages render in 0.4-2.1s each).
#   - R1639-11 / R1639-12: single-page large-format technical drawings
#     (~4930x2830 pt, i.e. roughly A0). Render is fine (11-13s) but OCR-ing the
#     resulting ~7000x4000px image takes 36-41s - above the old 30s tesseract
#     cap.
# Fix: render and OCR one page at a time (pdftoppm -f N -l N), each bounded by
# its own timeout, so one slow/huge page cannot exhaust a shared subprocess
# budget meant for the whole document, and a 40-60 page scan is bounded by
# cumulative (not single-subprocess) time instead of a fixed wall-clock cap.
PDF_NATIVE_TEXT_TIMEOUT_SECONDS = 30
# Per-page render timeout. Measured max 13s (R1639 A0-ish drawing at 300 DPI);
# normal A4 scans render in 0.4-2.1s. ~3.5x margin over the worst observed case.
PDF_PAGE_RENDER_TIMEOUT_SECONDS = 45
# Per-page OCR timeout. Measured max 41s (same R1639 A0-ish drawing); normal
# A4 scans OCR in 0.4-4.5s. ~2.2x margin over the worst observed case.
PDF_PAGE_OCR_TIMEOUT_SECONDS = 90
# Document-level OCR budget, sized from measured per-page cost rather than a
# flat number: average (render+OCR) per page across production A4 scans was
# 2.4-6.4s; rounded up with margin.
PDF_OCR_SECONDS_PER_PAGE_BUDGET = 8
# Floor: must exceed one page's own worst-case allowance
# (PDF_PAGE_RENDER_TIMEOUT_SECONDS + PDF_PAGE_OCR_TIMEOUT_SECONDS = 135s) so the
# document budget can never pre-empt a single legitimately slow page before
# that page's own per-page timeout even has a chance to fire.
PDF_OCR_MIN_DOCUMENT_BUDGET_SECONDS = 150
# Ceiling: hard stop regardless of page count, so a pathological scan cannot
# run for tens of minutes. At 8s/page this covers up to 75 pages before
# capping; documents above that get a bounded partial result (see partial-page
# policy below) instead of an unbounded run.
PDF_OCR_MAX_DOCUMENT_BUDGET_SECONDS = 600
# Safety margin added on top of the internal OCR budget when sizing the
# *outer* ParsingWatchdog `limit` passed from sync() (see call site), so the
# watchdog's independent wall-clock kill can never fire before extract_pdf()'s
# own internal deadline has had a chance to. Covers pdfinfo/pdftotext's own
# (already-budgeted) time plus IPC/process-start overhead.
PDF_PARSE_WATCHDOG_MARGIN_SECONDS = 30

def _pdf_page_count(path, timeout=10):
    """Cheap page-count probe via `pdfinfo` (~20-30ms measured on production
    files). Returns None if it cannot be determined (corrupt/unusual PDF,
    pdfinfo missing, or a non-PDF fed to this path in a test) so callers fall
    back to a conservative whole-document path rather than guessing a range."""
    try:
        run=subprocess.run(["/opt/homebrew/bin/pdfinfo",str(path)],capture_output=True,text=True,check=False,timeout=timeout)
    except (subprocess.TimeoutExpired, OSError): return None
    if run.returncode: return None
    for line in run.stdout.splitlines():
        if line.startswith("Pages:"):
            try: return int(line.split(":",1)[1].strip())
            except ValueError: return None
    return None

def pdf_ocr_document_budget_seconds(page_count):
    """Document-level OCR safety budget derived from measured per-page cost;
    see the module-level constants above for the measurements behind the
    floor/ceiling/per-page rate."""
    pages=page_count if page_count and page_count>0 else 1
    return min(PDF_OCR_MAX_DOCUMENT_BUDGET_SECONDS,max(PDF_OCR_MIN_DOCUMENT_BUDGET_SECONDS,PDF_OCR_SECONDS_PER_PAGE_BUDGET*pages))

# Single-slot registry for the currently in-flight pdftoppm/tesseract Popen, so
# parsing_worker._handle_sigterm can find and kill it if ParsingWatchdog
# terminates this worker process mid-page (timeout or STOP). The worker
# process is single-threaded (one request at a time, see parsing_worker.py),
# so a one-element list is sufficient - no locking needed.
_active_ocr_subprocess=[None]

def _run_ocr_subprocess(cmd,timeout):
    """subprocess.run()-equivalent (same TimeoutExpired/CompletedProcess
    contract) that additionally exposes the live Popen via
    _active_ocr_subprocess while it runs, so an external SIGTERM can kill it
    instead of orphaning it when the whole worker process is terminated."""
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    _active_ocr_subprocess[0]=proc
    try: out,err=proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try: proc.communicate(timeout=5)
        except Exception: pass
        raise
    finally: _active_ocr_subprocess[0]=None
    return subprocess.CompletedProcess(cmd,proc.returncode,out,err)

def _cleanup_page_images(folder_path):
    for image in folder_path.glob("page-*.png"):
        try: image.unlink()
        except OSError: pass

def _pdf_page_marker(page,reason):
    # Explicit, greppable marker so a partial extraction is never silently
    # indistinguishable from a fully successful one (see partial-page failure
    # policy) - logged elsewhere too, but this keeps the signal inside the
    # indexed text itself in case the log is rotated away.
    return f"[OCR SELHALA STRÁNKA {page}: {reason}]"

def _extract_pdf_per_page(path,page_count,deadline):
    with tempfile.TemporaryDirectory(prefix="ai-search-ocr-",dir="/private/tmp") as folder:
        folder_path=Path(folder); parts=[]; failed=0
        for page in range(1,page_count+1):
            remaining=deadline-time.monotonic()
            if remaining<=0: parts.append(_pdf_page_marker(page,"překročen časový limit dokumentu")); failed+=1; continue
            prefix=str(folder_path/"page")
            try: rendered=_run_ocr_subprocess(["/opt/homebrew/bin/pdftoppm","-r","300","-png","-f",str(page),"-l",str(page),str(path),prefix],min(PDF_PAGE_RENDER_TIMEOUT_SECONDS,remaining))
            except subprocess.TimeoutExpired:
                parts.append(_pdf_page_marker(page,"render překročil časový limit")); failed+=1; _cleanup_page_images(folder_path); continue
            if rendered.returncode:
                parts.append(_pdf_page_marker(page,"render selhal")); failed+=1; _cleanup_page_images(folder_path); continue
            images=sorted(folder_path.glob("page-*.png"))
            if not images:
                parts.append(_pdf_page_marker(page,"render nevytvořil obrázek")); failed+=1; continue
            image=images[0]; remaining=deadline-time.monotonic()
            if remaining<=0:
                parts.append(_pdf_page_marker(page,"překročen časový limit dokumentu")); failed+=1; _cleanup_page_images(folder_path); continue
            try: run=_run_ocr_subprocess(["/opt/homebrew/bin/tesseract",str(image),"stdout","-l","ces+eng","--psm","6"],min(PDF_PAGE_OCR_TIMEOUT_SECONDS,remaining))
            except subprocess.TimeoutExpired: parts.append(_pdf_page_marker(page,"OCR překročil časový limit")); failed+=1
            else:
                if run.returncode: parts.append(_pdf_page_marker(page,"OCR selhal")); failed+=1
                else:
                    text=run.stdout.strip()
                    parts.append(text if text else f"[OCR PRÁZDNÁ STRÁNKA {page}]")
            _cleanup_page_images(folder_path)
        if failed==page_count: raise RuntimeError(f"OCR selhalo na všech {page_count} stránkách PDF")
        return "\n".join(parts)

def _extract_pdf_whole_document(path,deadline):
    # Fallback for the rare case pdfinfo cannot report a page count (corrupt or
    # unusual PDF structure - pdftoppm/pdftotext are the same poppler family
    # and are likely to struggle on the same file, so a page range probe would
    # not meaningfully help here). Mirrors the old whole-document render but
    # keeps per-page OCR timeouts/partial-page policy for the OCR half, and is
    # bounded by the same page-count-derived budget (page_count treated as 1).
    with tempfile.TemporaryDirectory(prefix="ai-search-ocr-",dir="/private/tmp") as folder:
        folder_path=Path(folder); remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError("PDF překročilo časový limit před OCR")
        try: rendered=_run_ocr_subprocess(["/opt/homebrew/bin/pdftoppm","-r","300","-png",str(path),str(folder_path/"page")],remaining)
        except subprocess.TimeoutExpired as exc: raise TimeoutError("Převod PDF pro OCR překročil časový limit") from exc
        if rendered.returncode: raise RuntimeError("PDF nelze převést pro OCR: "+(rendered.stderr.strip() or str(rendered.returncode)))
        images=sorted(folder_path.glob("page-*.png")); parts=[]; failed=0
        for index,image in enumerate(images,1):
            remaining=deadline-time.monotonic()
            if remaining<=0: parts.append(_pdf_page_marker(index,"překročen časový limit dokumentu")); failed+=1
            else:
                try: run=_run_ocr_subprocess(["/opt/homebrew/bin/tesseract",str(image),"stdout","-l","ces+eng","--psm","6"],min(PDF_PAGE_OCR_TIMEOUT_SECONDS,remaining))
                except subprocess.TimeoutExpired: parts.append(_pdf_page_marker(index,"OCR překročil časový limit")); failed+=1
                else:
                    if run.returncode: parts.append(_pdf_page_marker(index,"OCR selhal")); failed+=1
                    else:
                        text=run.stdout.strip()
                        parts.append(text if text else f"[OCR PRÁZDNÁ STRÁNKA {index}]")
            try: image.unlink()
            except OSError: pass
        if images and failed==len(images): raise RuntimeError(f"OCR selhalo na všech {len(images)} stránkách PDF")
        return "\n".join(parts)

def extract_pdf(path, budget_seconds=None):
    page_count=_pdf_page_count(path)
    if budget_seconds is None: budget_seconds=pdf_ocr_document_budget_seconds(page_count)
    deadline=time.monotonic()+budget_seconds
    try: direct=subprocess.run(["/opt/homebrew/bin/pdftotext","-layout",str(path),"-"],capture_output=True,text=True,check=False,timeout=min(PDF_NATIVE_TEXT_TIMEOUT_SECONDS,budget_seconds)).stdout
    except subprocess.TimeoutExpired as exc: raise TimeoutError("Textová vrstva PDF překročila časový limit") from exc
    if len(direct.strip())>=80: return direct
    remaining=deadline-time.monotonic()
    if remaining<=0: raise TimeoutError("PDF překročilo časový limit před OCR")
    if page_count is None: return _extract_pdf_whole_document(path,deadline)
    return _extract_pdf_per_page(path,page_count,deadline)

# --- Oversized-chunk protection (chunking audit, 2026-08-07) ---
#
# WHY THIS EXISTS: chunks() below splits ONLY on blank lines and heading-like
# lines, so a document whose extracted text contains neither - a spreadsheet
# sheet flattened into one line, a pre-fix DOCX export, textutil output of a
# table - collapsed into ONE chunk of arbitrary size. The audit measured 186
# such chunks across 125 documents above BGE-M3's 8192-token limit, the worst
# holding 32 488 tokens, and Embeddings.encode() (SentenceTransformer) truncates
# past that limit SILENTLY: ~10M tokens of indexed content were invisible to
# vector search while still looking indexed. A size cap is the only guard that
# does not depend on some extractor emitting the right whitespace, so it stays
# effective even if a future extractor regresses.
CHUNK_MAX_SIZE = 4000   # chars; ~1140 tokens of Czech prose (3.52 chars/token measured), far inside BGE-M3's 8192
CHUNK_OVERLAP = 400     # chars of the previous part repeated in the next one, so a sentence cut by a split stays retrievable from both sides

def _split_oversized(text, max_size=CHUNK_MAX_SIZE, overlap=CHUNK_OVERLAP):
    """Split `text` into pieces of at most `max_size` chars, in original order,
    repeating the last `overlap` chars of each piece at the start of the next.

    Text at or below the limit is returned as a single piece, byte for byte -
    that is what keeps every existing chunk boundary (and every existing test)
    untouched, the audited index having a median chunk of 93 chars against this
    4000-char cap.

    Word boundaries: the cut is the last space in the second half of the window,
    so a piece is always at least `max_size // 2` long and no word is broken. A
    run longer than the window containing NO space (base64 blobs, RTF
    control-word soup - both found in the audited index) offers no boundary to
    respect and is hard-split at `max_size`; leaving it whole would send it back
    into the silent truncation this function exists to prevent."""
    if len(text) <= max_size: return [text]
    overlap = max(0, min(overlap, max_size // 2))
    parts = []; start = 0
    while start < len(text):
        if len(text) - start <= max_size:
            parts.append(text[start:]); break
        window_end = start + max_size
        cut = text.rfind(" ", start + max_size // 2, window_end + 1)
        end = cut if cut > start else window_end
        parts.append(text[start:end])
        # Begin the overlap at a word boundary inside the tail just emitted, and
        # never at or before `start` - `end` always advances, so falling back to
        # it keeps the loop finite even for a degenerate max_size/overlap pair.
        tail_start = end - overlap
        space = text.find(" ", tail_start, end)
        next_start = (space + 1) if space != -1 else tail_start
        start = next_start if next_start > start else end
    return [piece for piece in (part.strip() for part in parts) if piece]

def chunks(text, deadline=None, progress=None):
    result, block, heading = [], [], ""
    def flush():
        value = "\n".join(block).strip(); block.clear()
        # All parts of an oversized block keep that block's own heading - the
        # split changes how much text a chunk holds, never which section it
        # belongs to.
        if value: result.extend((heading, piece) for piece in _split_oversized(value))
    for line_number,line in enumerate(text.replace("\r\n", "\n").split("\n"),1):
        if deadline and time.monotonic()>=deadline: raise PhaseTimeout("chunking",CHUNK_TIMEOUT_SECONDS,len(result)+1)
        if progress and line_number%250==0: progress(len(result)+1)
        value = line.strip()
        is_heading = bool(re.match(r"^(#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+|[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ\s-]{3,})", value))
        if is_heading and len(value) < 160: flush(); heading = re.sub(r"^#{1,6}\s+", "", value)
        elif not value: flush()
        else: block.append(line)
    flush()
    if result: return result
    # Fallback for text that produced no block at all (e.g. every line matched
    # the heading pattern): still capped, since this path used to be the other
    # way a whole document became one chunk.
    remainder = text.strip()
    return [("", piece) for piece in _split_oversized(remainder)] if remainder else []

class Embeddings:
    def __init__(self, name=EMBEDDING_MODEL):
        self.name=name; self.model=None
    def encode(self, texts, **kwargs):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(self.name)
        return self.model.encode(texts,normalize_embeddings=True,**kwargs).tolist()

_search_logger = logging.getLogger("ai_search.search")

# See the 2026-08-06 "cross-encoder precision reranker" analysis (chat transcript)
# for the 3-model comparison behind this choice. Summary: BAAI/bge-reranker-base
# (278M) is officially trained/documented as Chinese+English ONLY - a real risk
# for Czech construction terminology, despite being the smallest/fastest option.
# jinaai/jina-reranker-v2-base-multilingual (278M, genuinely 100+ languages) needs
# trust_remote_code=True (executes remote code from the model repo) and a flash-
# attention dependency that is unreliable/CPU-falls-back on Apple Silicon outside
# CUDA. BAAI/bge-reranker-v2-m3 (568M, well under the "not 1B+" ceiling) is
# explicitly trained multilingual (same lineage as the already-in-use bge-m3
# embedding model), needs ZERO new Python dependencies (plain
# sentence_transformers.CrossEncoder, no trust_remote_code), and measured
# ~10ms/pair batched inference on this Mac's MPS backend (Apple GPU) once warm -
# see CROSS_ENCODER_PRE_CE_BUDGET below for the latency/recall trade-off that
# batching cost drives.
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_MAX_LENGTH = 512   # tokens; deterministic tokenizer truncation, see CrossEncoderReranker docstring
CROSS_ENCODER_BATCH_SIZE = 32
CROSS_ENCODER_TIMEOUT_SECONDS = 60  # whole-batch predict() safety net, not per-pair - see CROSS_ENCODER_MAX_PASSAGE_CHARS for why this can stay this low
# Measured directly on the production index (2026-08-06): passage length is
# heavy-tailed (median 202 chars, p90 2637, max 377151 - the long tail is a
# handful of huge un-split legacy chunks, not the norm). The HF tokenizer has
# to scan/BPE-encode a passage's FULL text before it can truncate to
# CROSS_ENCODER_MAX_LENGTH tokens, so a single 377KB passage in a batch made
# EVERY item in that batch pay for it (measured 51-70ms/pair average across a
# 969-candidate batch containing outliers, vs ~10ms/pair on uniformly-short
# synthetic text - a ~5-7x inflation from a handful of pathological passages,
# not from typical chunk size). Pre-truncating by character count here is a
# cheap, deterministic guard *before* the tokenizer ever sees the outliers -
# not lossy for the vast majority of chunks: 4000 chars is several times
# larger than CROSS_ENCODER_MAX_LENGTH=512 tokens could ever use anyway for
# Czech/multilingual text (roughly 3-6 chars/token for this tokenizer), so it
# only ever discards text the tokenizer would have truncated away regardless -
# except for the pathological long tail, where it turns an unbounded
# tokenization cost into a bounded one.
CROSS_ENCODER_MAX_PASSAGE_CHARS = 4000

class CrossEncoderReranker:
    """Lazy-loaded wrapper around an independent (query, passage) relevance
    model - deliberately has NO knowledge of BM25, LanceDB, RRF or chunk
    metadata; it only ever sees plain (query_text, passage_text) string pairs
    handed to it by the caller, so it cannot be "fixed" into any one retrieval
    channel's internals.

    - lazy loading: the (~2GB) model is only downloaded/loaded into memory on
      the first call to score(), never at import time or app startup.
    - loaded once: `self.model` is cached on the instance after first load, so
      repeated score() calls (repeated queries) reuse the same in-memory model
      - the CALLER is responsible for keeping one instance alive across
      queries (e.g. one instance per benchmark run, or one long-lived instance
      in application state), exactly like the existing `Embeddings` class.
    - batch inference: all pairs for one query go through a single
      model.predict() call (internally chunked into `batch_size`-sized
      batches), not a Python loop calling predict() once per pair.
    - truncation: sentence_transformers.CrossEncoder truncates each side of
      the (query, passage) pair to `max_length` WordPiece/BPE tokens via the
      model's own tokenizer (deterministic, same input -> same truncation
      point every time) - not a naive character slice. 512 tokens covers the
      vast majority of chunks produced by chunks() while keeping batched
      inference latency practical; bge-reranker-v2-m3 supports up to 8192
      tokens if a future audit shows 512 truncates away the relevant part of
      unusually long chunks often enough to matter.
    - errors/timeouts: score() lets timeouts/model errors propagate as
      exceptions (via run_with_timeout) rather than swallowing them - search()
      is the one place that decides what a failure should fall back to (see
      candidate_strategy="union_ce"), so this class never silently returns a
      degraded-but-plausible-looking result.
    """
    def __init__(self, name=CROSS_ENCODER_MODEL, max_length=CROSS_ENCODER_MAX_LENGTH):
        self.name=name; self.max_length=max_length; self.model=None
    def _ensure_loaded(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model=CrossEncoder(self.name,max_length=self.max_length)
    def score(self, query, passages, batch_size=CROSS_ENCODER_BATCH_SIZE, timeout=CROSS_ENCODER_TIMEOUT_SECONDS):
        """Returns a plain list[float] of relevance scores, same length/order
        as `passages` (higher = more relevant; raw model logits, not a
        calibrated probability - only the relative order/magnitude across this
        one call is meaningful, do not compare scores across different
        queries or batch sizes). Raises on failure - callers decide the
        fallback."""
        if not passages: return []
        self._ensure_loaded()
        # See CROSS_ENCODER_MAX_PASSAGE_CHARS: a cheap character-count guard
        # applied BEFORE the tokenizer ever runs, so a handful of pathological
        # (hundreds-of-KB) chunks cannot inflate an entire batch's tokenization
        # cost. Chosen generously larger than max_length could ever consume,
        # so normal-sized chunks are unaffected either way.
        pairs=[(query,(passage or "")[:CROSS_ENCODER_MAX_PASSAGE_CHARS]) for passage in passages]
        return run_with_timeout(lambda: self.model.predict(pairs,batch_size=batch_size).tolist(), timeout, "cross-encoder predict")

# One process-wide default instance, created lazily on first use (not at
# import/app-startup time) so a caller that never asks for candidate_strategy=
# "union_ce" never pays the ~1GB RAM / ~2GB-on-disk cost of this model at all.
# A caller that wants tighter control (e.g. the benchmark, which runs many
# queries back-to-back and wants one guaranteed-shared instance) can instead
# construct and pass its own CrossEncoderReranker() via search(cross_encoder=...).
_default_cross_encoder = None

def _get_default_cross_encoder():
    global _default_cross_encoder
    if _default_cross_encoder is None: _default_cross_encoder=CrossEncoderReranker()
    return _default_cross_encoder

def lance_table(directory, dimension=None):
    import lancedb
    db = lancedb.connect(directory)
    if "chunks" in db.table_names(): return db.open_table("chunks")
    return db.create_table("chunks", data=[{"id":"__init__","vector":[0.0]*dimension,"document_id":0}]) if dimension else None

class SourceUnavailableError(RuntimeError):
    """Sken zdrojové složky vrátil implausibilně málo souborů - sync se ukončí
    dřív, než odstraňovací fáze smaže index dokumentů, které na disku existují."""

# Odstraňovací fáze sync() maže každý indexovaný dokument, který sken neviděl.
# Když je zdrojová složka nedostupná (nepřipojený/nesynchronizovaný Box, špatný
# root v nastavení), sken legitimně vrátí prázdno a jediný běh by smazal celý
# index. Sken počítá KANDIDÁTNÍ SOUBORY, zatímco `known` počítá DEDUPLIKOVANÉ
# indexované dokumenty, takže ve zdravém běhu je sken vyšší číslo (produkce
# 2026-08-06: 8919 skenovaných vs. 6298 indexovaných). Sken pod polovinou
# `known` proto nemůže vzniknout běžným mazáním souborů mezi dvěma syncy.
# Poměrová kontrola se uplatní až od REMOVAL_GUARD_MIN_DOCUMENTS, aby malé
# indexy a testovací fixtures - kde je smazání většiny souborů normální operace
# - fungovaly beze změny; úplně prázdný sken je odmítnut při jakékoli velikosti.
REMOVAL_GUARD_MIN_DOCUMENTS = 25
REMOVAL_GUARD_MIN_SCAN_RATIO = 0.5

def sync(root, db_path, lance_dir, embeddings, progress=None, stop_event=None):
    root=Path(root); db_path=Path(db_path); lance_dir=Path(lance_dir); logger=index_logger(db_path); started=time.perf_counter()
    counts={k:0 for k in ("added","changed","renamed","removed","unchanged","duplicates","skipped","errors","timeouts","removal_skipped")}; counts["stopped"]=False
    raw_progress=progress
    if raw_progress: progress=lambda event: raw_progress({**event,"counts":counts.copy(),"memory_bytes":memory_bytes()})
    with index_guard(db_path.parent/".index.lock"):
        parsing_watchdog=ParsingWatchdog(); msg_watchdog=MsgParsingWatchdog(); embedding_watchdog=EmbeddingWatchdog(embeddings); phase_records=[]; extension_stats=collections.defaultdict(lambda:{"total":0,"successful":0,"skipped":0,"timeouts":0,"errors":0,"parse_seconds":0.0})
        logger.info("START INDEXACE\nkořenová složka: %s",root)
        report(progress,logger,"SQLite otevřena",started=started)
        with database(db_path) as con:
            known={r[0]:r for r in con.execute("SELECT path,id,content_hash,size,mtime_ns,inode FROM documents")}
        report(progress,logger,"Načítám soubory",started=started)
        if stop_event and stop_event.is_set():
            counts["stopped"]=True; logger.info("STOP vyžádán před načtením souborů")
            parsing_watchdog.close(); msg_watchdog.close(); embedding_watchdog.close()
            return counts
        paths=[]
        for path in iter_documents(root):
            if stop_event and stop_event.is_set():
                counts["stopped"]=True; logger.info("STOP vyžádán během načítání souborů")
                parsing_watchdog.close(); msg_watchdog.close(); embedding_watchdog.close()
                return counts
            paths.append(path)
        # Ochrana proti hromadnému mazání - viz REMOVAL_GUARD_* výše.
        if known and (not paths or (len(known)>=REMOVAL_GUARD_MIN_DOCUMENTS and len(paths)<len(known)*REMOVAL_GUARD_MIN_SCAN_RATIO)):
            message=(f"Sken složky {root} našel {len(paths)} souborů, ale index obsahuje {len(known)} dokumentů. "
                     "Zdrojová složka je pravděpodobně nedostupná (nepřipojený Box nebo špatně nastavený kořen). "
                     "Indexace i mazání byly zastaveny, index zůstal beze změny.")
            logger.error("SYNC ZASTAVEN: %s",message)
            parsing_watchdog.close(); msg_watchdog.close(); embedding_watchdog.close()
            raise SourceUnavailableError(message)
        current={}; stat_failed=set()
        for p in paths:
            if stop_event and stop_event.is_set():
                counts["stopped"]=True; logger.info("STOP vyžádán během statování souborů")
                parsing_watchdog.close(); msg_watchdog.close(); embedding_watchdog.close()
                return counts
            try:
                stat_value=run_with_timeout(lambda: p.stat(), PARSE_TIMEOUT_SECONDS, "stat")
                current[str(p)]=stat_value
            except (PhaseTimeout, Exception) as exc:
                # stat_error: soubor sken NAŠEL, jen ho nešlo přečíst. Nesmí se
                # dostat do odstraňovací fáze - drží se stranou v stat_failed.
                stat_failed.add(str(p))
                logger.warning("Přeskakuji soubor %s kvůli chybě stat (dokument NEBUDE odstraněn z indexu): %s", p, exc)
                continue
        paths=[p for p in paths if str(p) in current]; total=len(paths)
        logger.info("celkem nalezených kandidátů: %s",total); by_inode={r[5]:r for r in known.values()}; by_hash={r[2]:r for r in known.values()}
        for number,path in enumerate(paths,1):
            if stop_event and stop_event.is_set(): counts["stopped"]=True; logger.info("STOP vyžádán; bezpečně ukončuji před dokumentem %s",number); break
            absolute=str(path); stat=current[absolute]; old=known.get(absolute); extension=path.suffix.upper().lstrip(".") or "BEZ_PŘÍPONY"; extension_stats[extension]["total"]+=1
            document_started=time.perf_counter(); digest=old[2] if old else ""; current_phase="metadata"
            try:
                if old and old[3:] == (stat.st_size,stat.st_mtime_ns,stat.st_ino):
                    logger.info("NEZMĚNĚNÝ %s",path)
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
                current_phase="msg_parsing" if path.suffix.lower()==".msg" else "parsování"; phase_started=time.monotonic(); parser=msg_watchdog if path.suffix.lower()==".msg" else parsing_watchdog; phase_name="MSG parsování" if path.suffix.lower()==".msg" else "Parsování"
                if path.suffix.lower()==".msg": limit=MSG_PARSE_TIMEOUT_SECONDS
                elif path.suffix.lower()==".pdf":
                    # Dynamic per-file limit so this outer watchdog (wall-clock,
                    # independent of extract_pdf()'s own internal budget) never
                    # kills a large scan before extract_pdf() itself has a
                    # chance to finish or hit its own bounded per-page timeouts.
                    # See PDF_OCR_* constants above extract_pdf() for the
                    # measurements this is derived from.
                    limit=pdf_ocr_document_budget_seconds(_pdf_page_count(path))+PDF_PARSE_WATCHDOG_MARGIN_SECONDS
                else: limit=PARSE_TIMEOUT_SECONDS
                report(progress,logger,phase_name,number,total,path,started)
                text,method=parser.parse(path,limit,lambda elapsed:phase_progress(phase_name,elapsed),stop_event=stop_event); parse_seconds=time.monotonic()-phase_started; extension_stats[extension]["parse_seconds"]+=parse_seconds; phase_records.append({"document":str(path),"extension":extension,"phase":current_phase,"seconds":parse_seconds,"chunks":0})
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
                        if table and old_chunk_ids: run_with_timeout(lambda: table.delete("id IN ("+ ",".join(json.dumps(i) for i in old_chunk_ids)+")"), EMBEDDING_TIMEOUT_SECONDS, "lancedb delete")
                        if table and additions:
                            run_with_timeout(lambda: table.delete('id = "__init__"'), EMBEDDING_TIMEOUT_SECONDS, "lancedb delete")
                            for row in additions: row["document_id"]=doc_id
                            run_with_timeout(lambda: table.add(additions), EMBEDDING_TIMEOUT_SECONDS, "lancedb add")
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
            except ParseCancelled:
                counts["stopped"]=True; logger.info("STOP vyžádán během parsování %s; bezpečně ukončuji",path); break
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
            # Z indexu se smí odstranit pouze POTVRZENĚ chybějící dokument.
            # stat_error: cestu sken viděl, jen selhal stat() (oprávnění, I/O,
            # timeout na cloudovém svazku) - soubor na disku je.
            if absolute in stat_failed:
                counts["removal_skipped"]+=1; logger.warning("PONECHÁN V INDEXU (stat selhal, ne potvrzené chybění): %s",absolute); continue
            # transient_failure: sken cestu nevrátil, ale soubor existuje
            # (nedostupný rodičovský adresář, race s probíhající synchronizací).
            try: still_present=os.path.exists(absolute)
            except OSError as exc: still_present=True; logger.warning("Kontrola existence selhala (%s), dokument ponechán: %s",exc,absolute)
            if still_present:
                counts["removal_skipped"]+=1; logger.warning("PONECHÁN V INDEXU (soubor na disku existuje, ale nebyl ve skenu): %s",absolute); continue
            logger.info("POTVRZENĚ CHYBĚJÍCÍ, odstraňuji z indexu: %s",absolute)
            with database(db_path) as con: old_ids=[r[0] for r in con.execute("SELECT id FROM chunks WHERE document_id=?",(row[1],))]
            table=lance_table(lance_dir); lance_version=table.version if table and old_ids else None
            try:
                with database(db_path) as con:
                    con.execute("BEGIN IMMEDIATE"); con.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",(row[1],)); con.execute("DELETE FROM documents WHERE id=?",(row[1],))
                    if table and old_ids: run_with_timeout(lambda: table.delete("id IN ("+ ",".join(json.dumps(i) for i in old_ids)+")"), EMBEDDING_TIMEOUT_SECONDS, "lancedb delete")
                counts["removed"]+=1; logger.info("ODSTRANĚN %s",absolute)
            except Exception:
                if table and lance_version is not None: table.checkout(lance_version); table.restore(); table.checkout_latest()
                raise
        with database(db_path) as con:
            con.execute("INSERT OR REPLACE INTO settings VALUES('root',?)",(str(root),)); con.execute("INSERT OR REPLACE INTO settings VALUES('embedding_model',?)",(EMBEDDING_MODEL,))
        parsing_watchdog.close(); msg_watchdog.close(); embedding_watchdog.close(); slow_report=sorted(phase_records,key=lambda row:row["seconds"],reverse=True)[:20]; report_path=db_path.parent.parent/"logs"/(db_path.stem+"-slow-phases.json"); report_path.write_text(json.dumps({"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"slowest":slow_report,"by_extension":dict(sorted(extension_stats.items()))},ensure_ascii=False,indent=2),encoding="utf-8"); counts["slow_report"]=str(report_path); counts["by_extension"]=dict(extension_stats)
        elapsed=time.perf_counter()-started; completed=sum(counts[k] for k in ("added","changed","renamed","unchanged","duplicates","errors")); report(progress,logger,"Zastaveno" if counts["stopped"] else "Hotovo",completed,total,started=started)
        logger.info("Celkem=%s Nové=%s Aktualizované=%s Nezměněné=%s Duplicitní=%s Přeskočené=%s Chyby=%s Celkový čas=%.2f Průměrný čas/dokument=%.3f",total,counts["added"],counts["changed"]+counts["renamed"],counts["unchanged"],counts["duplicates"],counts["skipped"],counts["errors"],elapsed,elapsed/max(completed,1)); logger.info("SQLite zavřena; uzavírám LanceDB; konec")
        logger.info("SOUHRN: nezměněné=%s aktualizované=%s nové=%s odstraněné=%s ponechané_v_indexu=%s",counts["unchanged"],counts["changed"],counts["added"],counts["removed"],counts["removal_skipped"])
        return counts

RRF_RANK_CONSTANT = 60          # standard reciprocal-rank-fusion smoothing constant
RETRIEVAL_POOL_SIZE = 100       # Phase 1: candidates pulled from each of FTS5 / vector search
RERANK_POOL_SIZE = 30           # Phase 2: candidates re-scored with exact cosine similarity
# Question-style queries ("Co chybí k předání základové desky investorovi?") spread
# their signal across many common words ORed together in the FTS5 query, so a
# lexically sparse but genuinely relevant chunk (e.g. a short scanned handover
# certificate that only matches one of the OR terms) can rank well outside a
# 100-candidate FTS/vector pool even though it is clearly relevant - production
# audit found the best matching chunk at raw BM25 rank 138 and, after RRF fusion,
# at merged-list position 284. Both pool sizes below must widen together: a
# larger QA_RETRIEVAL_POOL_SIZE alone does not help if QA_RERANK_POOL_SIZE still
# truncates the RRF-merged list before the candidate is ever considered.
QA_RETRIEVAL_POOL_SIZE = 500    # Phase 1 pool for question-style queries
QA_RERANK_POOL_SIZE = 300       # Phase 2 pool for question-style queries
SEMANTIC_RERANK_MULTIPLIER = 2.0  # semantic-rank RRF contribution vs. a single retrieval list
FILENAME_MATCH_BONUS = 0.03     # exact query match in document name; > max single RRF contribution (1/60)
CHUNK_LENGTH_SHORT = 50         # chars; below this a chunk carries almost no standalone context
CHUNK_LENGTH_MEDIUM = 150       # chars; below this the chunk is still fairly thin
SHORT_CHUNK_PENALTY = 0.5
MEDIUM_CHUNK_PENALTY = 0.8
FTS_PREFIX_MIN_LENGTH = 6       # only expand longer tokens - short words carry too little signal for a prefix match to stay precise
FTS_PREFIX_STRIP = 2            # trims common Czech case endings (-é/-á/-ou/-ých/...)

# candidate_strategy for search() (see its docstring):
#   "legacy" (default, unchanged production behaviour) - the RRF-merged list is
#            truncated to rerank_k *before* phase 3 ever sees the rest.
#   "union"  - phase 3 sees the full BM25-top-N ∪ vector-top-N pool; RRF is
#            still computed and still contributes to the final score (so it
#            keeps acting as a *soft* ordering signal), but no longer decides
#            which candidates are discarded before reranking. Added to let a
#            future real (cross-encoder) precision stage see BM25-only hits
#            that a low RRF rank would otherwise have dropped - see the FERI
#            production case in benchmark/dataset/production_queries.jsonl.
CANDIDATE_STRATEGY_LEGACY = "legacy"
CANDIDATE_STRATEGY_UNION = "union"
# "union_ce": same recall-safe union pool as "union", but phase 3 uses a real
# independent cross-encoder (see CrossEncoderReranker below) instead of the
# cosine-similarity-vs-the-same-embeddings pseudo-rerank. Because a cross-
# encoder scores each (query, passage) pair with a full transformer forward
# pass, the pool handed to it must be pre-shrunk from the (1000+ candidate)
# union down to a measured, recall-safe CROSS_ENCODER_PRE_CE_BUDGET - see
# _select_pre_ce_candidates.
CANDIDATE_STRATEGY_UNION_CE = "union_ce"
CANDIDATE_STRATEGIES = (CANDIDATE_STRATEGY_LEGACY, CANDIDATE_STRATEGY_UNION, CANDIDATE_STRATEGY_UNION_CE)

# Measured directly on the production index+dataset (2026-08-06, see chat
# transcript / benchmark reports) before picking this number - NOT guessed:
#
#   case                              budget=100  200   300   500  full-union
#   prod-feri-handover-checklist-01    0.00      1.00  1.00  1.00   1.00
#   prod-feri-doklady-po-betonazi-01   0.00      0.00  0.00  1.00   1.00
#   (the 3 non-checklist/document-mode cases hit 1.00 at every budget - is_question=False
#    caps retrieval_k at 100/channel regardless of this constant, so the budget never binds there)
#
# recall_before_ce@300 still MISSES an expected document entirely (0.00, not a
# marginal drop) on prod-feri-doklady-po-betonazi-01 - only budget=500 matches
# the full unbounded union's recall on every case in this dataset. Per the
# stated decision rule ("candidate budget must follow recall, only prefer
# speed when the recall difference is minimal") that 0.00->1.00 gap at 300 is
# not minimal, so 500 is kept despite its higher latency (~5s/query batched
# cross-encoder inference on this Mac's MPS backend at 500 candidates, vs ~2s
# at 200 - see the final report's latency section). A future iteration could
# revisit this with a cheaper first-pass filter ahead of the cross-encoder
# instead of raising this constant further - not done here (no premature
# optimization). Kept as a named constant so a larger/more adversarial dataset
# can re-run the same sweep and revisit it.
CROSS_ENCODER_PRE_CE_BUDGET = 500

def _select_pre_ce_candidates(fts_ids, vector_ids, budget):
    """Recall-safe reduction of the (potentially 1000+) union pool down to a
    cross-encoder-affordable size - WITHOUT reintroducing the RRF-hard-cutoff
    bug this whole candidate_strategy effort exists to fix. Keeps the top
    `budget` ids from EACH channel INDEPENDENTLY (not a shared RRF-ranked
    cutoff across both), then unions/dedupes them via `_build_candidate_union`.
    A candidate that is strong in only one channel survives exactly as long as
    it is within that channel's own top-`budget` - regardless of what the
    OTHER channel or a combined RRF rank thinks of it. This is the same
    "quota per channel" mechanism `_build_candidate_union` already provides;
    this helper only adds the slicing."""
    return _build_candidate_union(fts_ids[:budget], vector_ids[:budget])

def _build_candidate_union(fts_ids, vector_ids):
    """Pure, dependency-free merge of two independently-ranked candidate id
    lists into one pool, deduplicated strictly by chunk_id, keeping each
    candidate's provenance (which channel(s) found it, and at what rank) -
    no scoring, no DB access, no RRF math. Order is deterministic: first
    occurrence across `fts_ids` then `vector_ids` (stable for the same two
    inputs). Never drops a candidate that appears in either input list -
    that is the whole point relative to an RRF-rank cutoff, which can and
    does drop a BM25-only hit purely for lacking a second channel's
    corroboration, regardless of how relevant it is."""
    fts_rank_of = {cid: rank for rank, cid in enumerate(fts_ids)}
    vector_rank_of = {cid: rank for rank, cid in enumerate(vector_ids)}
    ordered_ids = list(dict.fromkeys(list(fts_ids) + list(vector_ids)))
    return [
        {"chunk_id": cid, "fts_rank": fts_rank_of.get(cid), "vector_rank": vector_rank_of.get(cid),
         "fts_hit": cid in fts_rank_of, "vector_hit": cid in vector_rank_of}
        for cid in ordered_ids
    ]

def _cosine_similarities(query_vector, vectors):
    query=np.asarray(query_vector,dtype=np.float32); matrix=np.asarray(vectors,dtype=np.float32)
    query_norm=np.linalg.norm(query)+1e-8; row_norms=np.linalg.norm(matrix,axis=1)+1e-8
    return (matrix @ query)/(row_norms*query_norm)

def _chunk_quality_factor(text_length, exempt):
    """Softly downweight very short, low-context chunks (e.g. a two-word status
    fragment) that would otherwise compete equally with substantive passages just
    because they happen to contain the query's exact words - root cause found in
    the "Co chybí k předání základové desky" production audit, where a 17-
    character fragment ("základovou desku.") repeated across ~80 site-meeting
    reports out-scored every genuinely relevant document. An exact FTS hit or a
    filename match (contract number, product code, technical designation) is
    exempt: a short-but-precise lexical match must not be punished for being
    short. This is a soft multiplier, not a hard filter."""
    if exempt: return 1.0
    if text_length<CHUNK_LENGTH_SHORT: return SHORT_CHUNK_PENALTY
    if text_length<CHUNK_LENGTH_MEDIUM: return MEDIUM_CHUNK_PENALTY
    return 1.0

# search(expand_query=...) accepts a mode, not just a boolean, because the two
# injection points the Query Understanding layer offers have measurably
# OPPOSITE risk profiles and have to be evaluable independently:
#   "fts"    - OR the expansion terms into the FTS5 MATCH expression. Purely
#              additive to the candidate set; existing lexical hits keep their
#              rank.
#   "vector" - embed "query + terms" instead of "query". This REPLACES the query
#              vector, so it changes every cosine score in the rerank, including
#              those of candidates the expansion had nothing to do with.
# True/"both" enables both (the full layer). See tests/test_query_expansion.py
# and the 2026-08-07 A/B run for what each branch actually did to production
# benchmark cases.
EXPANSION_MODES={False:(False,False),None:(False,False),"off":(False,False),True:(True,True),"both":(True,True),"fts":(True,False),"vector":(False,True)}

def _expansion_branches(expand_query):
    """(use_in_fts, use_in_vector) for an `expand_query` argument."""
    try: return EXPANSION_MODES[expand_query]
    except (KeyError,TypeError): raise ValueError(f"expand_query must be one of {sorted(str(k) for k in EXPANSION_MODES)}, got {expand_query!r}")

def _fts_query_terms(query, extra_terms=()):
    """Build the FTS5 MATCH expression. Each word is matched exactly, and for
    longer words a native FTS5 prefix match ('stem*') is OR'd in as well, so a
    query using one grammatical case (e.g. "základové") still finds a document
    using a different case of the same word (e.g. "základovou") - the unicode61
    tokenizer does no stemming on its own. This is intentionally NOT general
    stemming: only a conservative prefix widening using SQLite's built-in
    prefix-query syntax, applied only to longer words to keep it precise.

    `extra_terms` are query-expansion terms (see query_expansion.py), OR'd in
    ADDITIONALLY - the terms derived from `query` itself are never replaced or
    reordered, so `extra_terms=()` (the default) produces a byte-identical
    expression to the one this function returned before the parameter existed.
    A multi-word term is emitted as an FTS5 phrase ("dodací list") rather than
    as separate OR'd words, which would match every document containing the
    word "list" on its own. Non-word characters are stripped from expansion
    terms because an unescaped quote would break the whole MATCH expression."""
    terms=[]
    for token in re.findall(r"\w+",query):
        if len(token)<=1: continue
        terms.append('"'+token+'"')
        if len(token)>=FTS_PREFIX_MIN_LENGTH: terms.append(token[:-FTS_PREFIX_STRIP]+"*")
    for term in extra_terms:
        words=re.findall(r"\w+",term or "")
        if not words: continue
        if len(words)>1: terms.append('"'+" ".join(words)+'"'); continue
        word=words[0]
        if len(word)<=1: continue
        terms.append('"'+word+'"')
        if len(word)>=FTS_PREFIX_MIN_LENGTH: terms.append(word[:-FTS_PREFIX_STRIP]+"*")
    return " OR ".join(terms)

class SearchTrace:
    """Optional instrumentation sink for search(). Pass an instance as
    search()'s `trace=` argument to additionally record a snapshot of every
    retrieval phase (candidates + latency) - the computation and the
    returned rows are byte-for-byte identical to calling search() without
    `trace` at all; nothing about retrieval/scoring/ranking is affected by
    whether a trace is attached. `trace=None` (the default) is the exact
    code that ran before this parameter existed, plus one `is not None`
    check per phase.

    Fields (populated by search(), read-only from the caller's side):
      query, query_terms      - the raw query and its FTS5 MATCH expression
      query_original          - the query exactly as the caller passed it (same
                                 value as `query`; named separately because it is
                                 the reference point the expansion fields below
                                 are read against)
      query_expanded          - the text actually handed to the embedder: equals
                                 query_original when expand_query is off or no
                                 dictionary rule matched
      expansion_terms         - [str] terms OR'd into the FTS expression and
                                 appended to the embedded text ([] when off)
      expansion_matched_rules - [{"key","scope","trigger","terms"}] which
                                 dictionary rules fired and why (see
                                 query_expansion.expand_query)
      intent                  - {"is_question","retrieval_k","rerank_k","limit"}
      bm25_candidates         - [{"chunk_id","rank","document_id","score"}] straight
                                 from FTS5, ORDER BY rank (score always None - FTS5's
                                 internal bm25() weight isn't exposed by this query,
                                 only relative rank is)
      vector_candidates       - [{"chunk_id","rank","document_id","score"}] straight
                                 from LanceDB (score = raw "_distance", NOT cosine
                                 similarity - see rerank_candidates for that)
      rrf_candidates          - [{"chunk_id","rank","score","bm25_rank","vector_rank",
                                 "document_id"}] the FULL fused+sorted pool, i.e.
                                 *before* the rerank_k truncation - deliberately not
                                 pre-truncated so a benchmark can measure exactly how
                                 much recall that truncation costs (see
                                 candidate_pool_size_before/after_truncation in
                                 `metadata`). Always computed regardless of
                                 candidate_strategy - RRF is kept as a diagnostic /
                                 soft-ordering signal either way, see `union_candidates`.
      union_candidates        - [{"chunk_id","fts_rank","vector_rank","fts_hit",
                                 "vector_hit"}] output of `_build_candidate_union()` -
                                 every chunk_id seen by BM25 and/or vector retrieval,
                                 deduplicated, with NO score-based cutoff of any kind.
                                 Always populated (independent of candidate_strategy)
                                 so a benchmark can compare "what RRF would have kept"
                                 (rrf_candidates[:rerank_k]) against "everything either
                                 channel actually found" for the same query.
      candidates_before_precision - [{"chunk_id"}] the *actual* candidate ids phase 3
                                 (cosine rerank + chunk-quality + filename scoring)
                                 iterated over for this call - i.e. `top_ids`. Under
                                 candidate_strategy="legacy" this is rrf_candidates
                                 truncated to rerank_k (same set as before this field
                                 existed); under "union" it is every id in
                                 union_candidates, unfiltered by RRF rank.
      rerank_candidates       - [{"chunk_id","rank","score","document","path"}] the
                                 candidates_before_precision pool after phase-3
                                 semantic reranking, sorted the same way as the
                                 function's return value
      final_candidates        - same shape as rerank_candidates, truncated to `limit`
                                 - this is what the function actually returns, with
                                 chunk_id/rank added for identification
      timings                 - {"query_parsing","fts_retrieval","vector_retrieval",
                                 "fusion_rrf","reranker"} in milliseconds
      metadata                - free-form extras: candidate_strategy plus the two
                                 pool sizes mentioned above
    """
    def __init__(self):
        self.query=None; self.query_terms=None; self.intent={}
        self.query_original=None; self.query_expanded=None
        self.expansion_terms=[]; self.expansion_matched_rules=[]
        self.bm25_candidates=[]; self.vector_candidates=[]; self.rrf_candidates=[]
        self.union_candidates=[]; self.candidates_before_precision=[]
        self.rerank_candidates=[]; self.final_candidates=[]
        self.timings={}; self.metadata={}
        # candidate_strategy="union_ce" only (see CrossEncoderReranker). Score
        # and rank are kept both inside cross_encoder_candidates[*] (ordered,
        # like every other *_candidates list above) AND as separate
        # chunk_id-keyed dicts, purely for O(1) lookup convenience - both are
        # derived from the exact same single scoring pass, never computed
        # twice, so they cannot drift apart from each other.
        self.candidates_before_cross_encoder=[]   # [{"chunk_id"}] pool actually sent to the cross-encoder
        self.cross_encoder_candidates=[]          # [{"chunk_id","rank","score"}] CE output, sorted by score desc
        self.cross_encoder_score={}                # chunk_id -> raw CE score
        self.cross_encoder_rank={}                 # chunk_id -> rank (0=best) in the CE ordering
        self.cross_encoder_latency=None            # ms spent in CrossEncoderReranker.score(), or None if CE did not run
        self.cross_encoder_model=None              # model name actually used, or None if CE did not run (incl. fallback)

def search(query, db_path, lance_dir, embeddings, limit=8, is_question=False, trace=None, candidate_strategy=CANDIDATE_STRATEGY_LEGACY, cross_encoder=None, expand_query=False):
    """Hybrid retrieval pipeline: (1) broad FTS5 + vector retrieval, (2) reciprocal
    rank fusion merge, (3) a precision reranking pass over the merged pool.
    is_question widens both pool sizes (see QA_RETRIEVAL_POOL_SIZE/QA_RERANK_POOL_SIZE)
    for natural-language questions, which need more recall than a plain document
    lookup; document search (is_question=False) keeps the original pool sizes.

    `trace`: optional SearchTrace instance - see its docstring. Purely an
    observability side-channel (used by benchmark/pipeline_trace.py instead of
    that module's former hand-mirrored reimplementation of this function);
    never changes what is computed or returned.

    `candidate_strategy`: which candidates phase 3 gets to see, and how phase 3
    scores them.
    - "legacy" (default, unchanged production behaviour): truncates the
      RRF-merged list to rerank_k *before* phase 3, then reranks with exact
      cosine similarity against the SAME bge-m3 embeddings retrieval already
      used - a real hit that only one channel found (e.g. BM25-only) can be
      discarded here purely for lacking the other channel's corroboration,
      regardless of relevance (see the FERI production case in
      benchmark/dataset/production_queries.jsonl).
    - "union": phase 3 sees the full BM25-top-N ∪ vector-top-N pool (see
      `_build_candidate_union`) instead of the rerank_k-truncated list, still
      scored with the same cosine-similarity formula as "legacy" - RRF still
      contributes to the score as a *soft* signal, it just no longer decides
      which candidates are discarded before reranking.
    - "union_ce": same recall-safe pool-building philosophy as "union" (see
      `_select_pre_ce_candidates`), but phase 3 scores candidates with an
      independent cross-encoder relevance model (see CrossEncoderReranker)
      instead of cosine similarity - a genuinely new relevance signal, not a
      restatement of the retrieval embeddings. Falls back to "union" behaviour
      (same pool, cosine scoring) if the cross-encoder is unavailable, errors,
      or times out - see the try/except around the cross-encoder call below.
    Only "legacy" vs "union"/"union_ce" changes the *size/membership* of the
    pool phase 3 iterates over and, for "union_ce" only, *how* phase 3 scores
    each candidate; nothing about phase 1/2 (FTS5, vector retrieval, RRF
    fusion) ever changes.

    `cross_encoder`: optional CrossEncoderReranker instance, only consulted
    when candidate_strategy="union_ce". Passing your own instance lets a
    caller that runs many queries (e.g. the benchmark) guarantee the model is
    loaded once and reused; if omitted, a lazily-created process-wide default
    instance is used instead (see `_get_default_cross_encoder`) so search()
    still works correctly, just without that caller-level control. Ignored
    for "legacy"/"union" - passing one then is a no-op, not an error.

    `expand_query`: opt-in Query Understanding layer (see query_expansion.py and
    EXPANSION_MODES above for the accepted values). False (the default) is
    byte-for-byte the behaviour that existed before this parameter, down to the
    FTS expression and the embedded text - the expansion module is not even
    consulted. When enabled, a curated construction-domain dictionary widens the
    selected retrieval channel(s): matched terms are OR'd into the FTS5
    expression and/or appended after the query in the text handed to the
    embedder. The query itself is never rewritten: `query` remains what the
    filename-match bonus and every scoring formula below see, and no scoring
    weight changes. Note the unavoidable consequence of the "vector" branch:
    `query_vector` is then computed from the expanded text, so the cosine rerank
    scores candidates against that vector - the formula is untouched, its input
    is deliberately widened. That is why this is opt-in and A/B-measured
    (benchmark run --expand-query [both|fts|vector]) rather than default-on."""
    if candidate_strategy not in CANDIDATE_STRATEGIES:
        raise ValueError(f"candidate_strategy must be one of {CANDIDATE_STRATEGIES!r}, got {candidate_strategy!r}")
    _tracing=trace is not None
    if is_question: retrieval_k=max(QA_RETRIEVAL_POOL_SIZE,limit*2); rerank_k=max(QA_RERANK_POOL_SIZE,limit)
    else: retrieval_k=max(RETRIEVAL_POOL_SIZE,limit*2); rerank_k=max(RERANK_POOL_SIZE,limit)
    if _tracing:
        trace.query=query; trace.intent={"is_question":is_question,"retrieval_k":retrieval_k,"rerank_k":rerank_k,"limit":limit}
        trace.metadata["candidate_strategy"]=candidate_strategy
    _t=time.perf_counter() if _tracing else None
    _expand_fts,_expand_vector=_expansion_branches(expand_query)
    expansion=query_expansion.expand_query(query) if (_expand_fts or _expand_vector) else None
    _embed_text=expansion.embedding_text if (expansion and _expand_vector) else query
    terms=_fts_query_terms(query,extra_terms=expansion.terms if (expansion and _expand_fts) else ())
    if _tracing:
        trace.query_terms=terms; trace.timings["query_parsing"]=(time.perf_counter()-_t)*1000
        trace.query_original=query; trace.query_expanded=_embed_text
        trace.expansion_terms=list(expansion.terms) if expansion else []
        trace.expansion_matched_rules=[dict(rule) for rule in expansion.matched_rules] if expansion else []
        trace.metadata["expand_query"]=expand_query; trace.metadata["expansion_branches"]={"fts":_expand_fts,"vector":_expand_vector}
    # ORDER BY rank is required: FTS5 otherwise returns matches in rowid order,
    # not by BM25 relevance, which would make the RRF rank-fusion below meaningless.
    _t=time.perf_counter() if _tracing else None
    with database(db_path) as con: lexical=con.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",(terms,retrieval_k)).fetchall() if terms else []
    fts_ids=[cid for (cid,) in lexical]; table=lance_table(lance_dir)
    doc_id_by_chunk={}
    if _tracing and fts_ids:
        # One batched lookup (not one query per candidate) purely to populate
        # SearchTrace.bm25_candidates[*]["document_id"] - chunks_fts itself has
        # no document_id column. Only ever runs when a trace is attached.
        with database(db_path) as con:
            for start in range(0,len(fts_ids),500):
                batch=fts_ids[start:start+500]
                placeholders=",".join("?"*len(batch))
                for cid,doc_id in con.execute(f"SELECT id,document_id FROM chunks WHERE id IN ({placeholders})",batch).fetchall(): doc_id_by_chunk[cid]=doc_id
    if _tracing:
        trace.timings["fts_retrieval"]=(time.perf_counter()-_t)*1000
        trace.bm25_candidates=[{"chunk_id":cid,"rank":i,"document_id":doc_id_by_chunk.get(cid),"score":None} for i,cid in enumerate(fts_ids)]
    query_vector=embeddings.encode([_embed_text])[0] if table else None
    _t=time.perf_counter() if _tracing else None
    vector_rows=run_with_timeout(lambda: table.search(query_vector).limit(retrieval_k).to_list(), 30, "lancedb search") if table and query_vector is not None else []
    vector_ids=[row["id"] for row in vector_rows]
    if _tracing:
        trace.timings["vector_retrieval"]=(time.perf_counter()-_t)*1000
        trace.vector_candidates=[{"chunk_id":row["id"],"rank":i,"document_id":row.get("document_id"),"score":row.get("_distance")} for i,row in enumerate(vector_rows)]

    # Phase 2: Reciprocal Rank Fusion merges the two independently-ranked lists
    # into one candidate pool without needing comparable raw score scales.
    _t=time.perf_counter() if _tracing else None
    rrf_scores={}
    for rank,cid in enumerate(fts_ids): rrf_scores[cid]=rrf_scores.get(cid,0)+1.0/(RRF_RANK_CONSTANT+rank)
    for rank,cid in enumerate(vector_ids): rrf_scores[cid]=rrf_scores.get(cid,0)+1.0/(RRF_RANK_CONSTANT+rank)
    if not rrf_scores:
        if _tracing: trace.timings["fusion_rrf"]=(time.perf_counter()-_t)*1000
        return []
    # sorted() always sorts the full dict regardless of the slice taken below,
    # so keeping the untruncated reference (`fusion_order`) instead of
    # discarding it costs nothing extra - same sort, same result, `top_ids`
    # is unchanged either way.
    fusion_order=sorted(rrf_scores,key=rrf_scores.get,reverse=True)
    # candidate_strategy decides ONLY which ids phase 3 below iterates over -
    # "legacy" keeps the original RRF-rank hard cutoff (top_ids=fusion_order[:rerank_k]);
    # "union" instead uses every id either channel found at all (fts_ids ∪
    # vector_ids - exactly rrf_scores.keys(), just in a fixed, RRF-independent
    # order), so a BM25-only hit can no longer be discarded here purely for
    # ranking low in the RRF merge. rrf_scores[cid] below is unaffected either
    # way and still feeds every candidate's final score (RRF stays a *soft*
    # signal, see search()'s docstring), and is guaranteed defined for every
    # id in either pool since rrf_scores' keys are exactly fts_ids ∪ vector_ids.
    pre_ce_candidates=None
    if candidate_strategy==CANDIDATE_STRATEGY_UNION:
        union=_build_candidate_union(fts_ids,vector_ids); top_ids=[c["chunk_id"] for c in union]
    elif candidate_strategy==CANDIDATE_STRATEGY_UNION_CE:
        # Same recall-safe philosophy as "union", but additionally shrunk to a
        # cross-encoder-affordable budget via independent per-channel quotas
        # (see _select_pre_ce_candidates) - NOT via the RRF-ranked cutoff that
        # "legacy" uses, which is exactly the mechanism this whole
        # candidate_strategy effort exists to avoid.
        union=None; pre_ce_candidates=_select_pre_ce_candidates(fts_ids,vector_ids,CROSS_ENCODER_PRE_CE_BUDGET)
        top_ids=[c["chunk_id"] for c in pre_ce_candidates]
    else:
        union=None; top_ids=fusion_order[:rerank_k]
    if _tracing:
        fts_rank_of={cid:i for i,cid in enumerate(fts_ids)}; vector_rank_of={cid:i for i,cid in enumerate(vector_ids)}
        vector_doc_id_by_chunk={row["id"]:row.get("document_id") for row in vector_rows}
        trace.rrf_candidates=[{"chunk_id":cid,"rank":i,"score":rrf_scores[cid],"bm25_rank":fts_rank_of.get(cid),"vector_rank":vector_rank_of.get(cid),
            "document_id":doc_id_by_chunk.get(cid,vector_doc_id_by_chunk.get(cid))} for i,cid in enumerate(fusion_order)]
        # union_candidates is always populated (independent of candidate_strategy)
        # so a benchmark can compare "what RRF would have kept" against
        # "everything either channel found" even on a legacy-strategy run.
        trace.union_candidates=union if union is not None else _build_candidate_union(fts_ids,vector_ids)
        trace.candidates_before_precision=[{"chunk_id":cid} for cid in top_ids]
        if candidate_strategy==CANDIDATE_STRATEGY_UNION_CE:
            trace.candidates_before_cross_encoder=[{"chunk_id":cid} for cid in top_ids]
        trace.timings["fusion_rrf"]=(time.perf_counter()-_t)*1000
        trace.metadata["candidate_pool_size_before_truncation"]=len(fusion_order)
        trace.metadata["candidate_pool_size_after_truncation"]=len(top_ids)

    # Phase 3: exact cosine similarity reranking. Vectors already fetched during
    # retrieval are reused; only candidates missing a vector (FTS-only hits) need
    # a lookup by id - no re-embedding of text is required.
    _t=time.perf_counter() if _tracing else None
    vector_by_id={row["id"]:row["vector"] for row in vector_rows}
    missing_ids=[cid for cid in top_ids if cid not in vector_by_id]
    if missing_ids and table:
        id_filter=" OR ".join(f'id = {json.dumps(mid)}' for mid in missing_ids)
        try:
            extra=run_with_timeout(lambda: table.search().where(id_filter).limit(len(missing_ids)).to_list(), 15, "lancedb vector lookup")
            for row in extra: vector_by_id[row["id"]]=row["vector"]
        except Exception: pass
    similarity_by_id={}
    if query_vector is not None:
        available=[(cid,vector_by_id[cid]) for cid in top_ids if cid in vector_by_id]
        if available:
            ids_with_vector,vectors=zip(*available)
            for cid,similarity in zip(ids_with_vector,_cosine_similarities(query_vector,list(vectors))): similarity_by_id[cid]=float(similarity)

    # The semantic rerank is fused back in by RANK (not raw cosine value), same
    # as the Phase 1 lists. This keeps the whole pipeline scale-independent: a
    # clearly-winning lexical/retrieval match can't be casually overturned by
    # small numeric noise in the semantic score, while a candidate that is
    # genuinely the best semantic match still gets a strong, consistent boost.
    semantic_order=sorted(similarity_by_id,key=similarity_by_id.get,reverse=True)
    for rank,cid in enumerate(semantic_order): rrf_scores[cid]=rrf_scores.get(cid,0)+SEMANTIC_RERANK_MULTIPLIER/(RRF_RANK_CONSTANT+rank)

    # Přesná shoda dotazu v názvu dokumentu je silnější důkaz než
    # sémantická podobnost obecného textu dodacího listu.
    needle=query.casefold().strip(); fts_id_set=set(fts_ids); vector_id_set=set(vector_ids)
    # Fetched as a separate pass (not interleaved with scoring like before)
    # because candidate_strategy="union_ce" needs every candidate's full chunk
    # text collected up front for ONE batched cross-encoder call - a per-
    # candidate "fetch, then immediately score" loop cannot batch. Output is
    # identical either way; this only changes *when* the DB rows are read, not
    # which rows or what they contain.
    rows_by_id={}
    with database(db_path) as con:
        for cid in top_ids:
            row=con.execute("SELECT d.name,d.path,d.project,c.heading,c.text FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?",(cid,)).fetchone()
            if row: rows_by_id[cid]=row

    # candidate_strategy="union_ce": score each (query, full_chunk_text) pair
    # with an independent cross-encoder instead of the cosine-similarity
    # formula below - see CrossEncoderReranker. Passage text is the actual
    # indexed chunk text (row[4], same source `quote` below truncates from),
    # not the UI's 700-char preview. ANY failure (model load, timeout,
    # inference error) falls back to the exact same cosine-similarity scoring
    # "union" uses for every remaining candidate - search() never raises or
    # returns emptier results just because the cross-encoder had a bad day
    # (KROK 8: no silent failure, always logged).
    ce=None; ce_score_by_id=None; ce_latency_ms=None
    if candidate_strategy==CANDIDATE_STRATEGY_UNION_CE and rows_by_id:
        ce=cross_encoder if cross_encoder is not None else _get_default_cross_encoder()
        ce_ids=list(rows_by_id.keys())
        try:
            _ce_t=time.perf_counter()
            ce_scores=ce.score(query,[rows_by_id[cid][4] or "" for cid in ce_ids])
            ce_latency_ms=(time.perf_counter()-_ce_t)*1000
            ce_score_by_id=dict(zip(ce_ids,ce_scores))
        except Exception as exc:
            _search_logger.warning("CROSS_ENCODER_FALLBACK query=%r model=%s candidates=%d error=%r - falling back to union cosine scoring",query,ce.name,len(ce_ids),exc)
            if _tracing: trace.metadata["cross_encoder_error"]=repr(exc)

    output=[]; trace_ids=[] if _tracing else None
    for cid in top_ids:
        row=rows_by_id.get(cid)
        if not row: continue
        filename_match=bool(needle) and needle in row[0].casefold(); fts_hit=cid in fts_id_set
        quality=_chunk_quality_factor(len(row[4] or ""),exempt=fts_hit or filename_match)
        # KROK 7: when the cross-encoder actually ran, its score IS the final
        # precision score - not blended with RRF/cosine/BM25 via new manual
        # weights. Those signals stay available for diagnostics in `match`/
        # SearchTrace, they just no longer determine the order.
        score=ce_score_by_id[cid] if ce_score_by_id is not None else rrf_scores[cid]*quality+(FILENAME_MATCH_BONUS if filename_match else 0.0)
        match={"fts_hit":fts_hit,"vector_hit":cid in vector_id_set,"semantic_similarity":max(similarity_by_id.get(cid,0.0),0.0),"filename_match":filename_match,"chunk_quality":quality}
        if ce_score_by_id is not None: match["cross_encoder_score"]=ce_score_by_id[cid]
        output.append({"document":row[0],"path":row[1],"project":row[2],"quote":row[4][:700],"heading":row[3],"score":score,"match":match})
        # Cheap `if` on an already-existing per-candidate DB round trip -
        # negligible next to the SQLite query on the line above; only
        # runs at all when a trace is attached.
        if _tracing: trace_ids.append(cid)
    result=sorted(output,key=lambda row:row["score"],reverse=True)
    if _tracing:
        # Sorting the *indices* by the same key/reverse as `result` above is
        # guaranteed (stable sort) to reproduce the identical order, so
        # trace_ids can be reordered in lockstep without re-deriving scores.
        order=sorted(range(len(output)),key=lambda i:output[i]["score"],reverse=True)
        trace.rerank_candidates=[{"chunk_id":trace_ids[i],"rank":rank,"score":output[i]["score"],"document":output[i]["document"],"path":output[i]["path"]} for rank,i in enumerate(order)]
        trace.final_candidates=trace.rerank_candidates[:limit]
        trace.timings["reranker"]=(time.perf_counter()-_t)*1000
        if ce_score_by_id is not None:
            trace.cross_encoder_model=ce.name
            trace.cross_encoder_latency=ce_latency_ms
            trace.cross_encoder_score=dict(ce_score_by_id)
            ce_rank_order=sorted(ce_score_by_id,key=ce_score_by_id.get,reverse=True)
            trace.cross_encoder_rank={cid:rank for rank,cid in enumerate(ce_rank_order)}
            trace.cross_encoder_candidates=[{"chunk_id":cid,"rank":rank,"score":ce_score_by_id[cid],"document":rows_by_id[cid][0],"path":rows_by_id[cid][1]} for rank,cid in enumerate(ce_rank_order)]
    return result[:limit]

# Hallucination guard shared by both prompt variants below: forbids inventing
# facts/norms/obligations, forces an explicit "not found" phrase, and requires a
# per-point source instead of one undifferentiated [1][2][3] list at the end.
HALLUCINATION_GUARD = (
    "Jsi stavební asistent. Odpovídej výhradně česky a výhradně na základě dodaného kontextu ZDROJE - nic si nevymýšlej.\n"
    "Nevymýšlej technické požadavky, normy ani povinnosti, které nejsou doslova uvedené ve ZDROJÍCH.\n"
    "Pokud informace v dodaných ZDROJÍCH není, napiš doslova: \"Nenalezeno v indexovaných dokumentech.\" - nedomýšlej si ji.\n"
    "Rozlišuj typ každého tvrzení tónem věty:\n"
    "- FAKT - doslovná informace ze zdroje, např. \"V dokumentu <název> je uvedeno...\"\n"
    "- POŽADAVEK - povinnost/podmínka uvedená ve zdroji, např. \"Dokumentace obsahuje požadavek...\"\n"
    "- DOPORUČENÍ - tvůj návrh nad rámec zdrojů, jasně tak označený, např. \"Pro kontrolu doporučuji ověřit...\"\n"
    "Za KAŽDÝM důležitým bodem/tvrzením uveď jeho zdroj: buď číselnou citaci [n], nebo - pokud píšeš souvislou větou/odstavcem "
    "a číselná citace by odpověď rozbíjela - ukonči jej řádkem 'Zdroj: <název dokumentu>'. Nikdy neuváděj zdroje jen jako "
    "jeden souhrnný výčet [1][2][3] na konci celé odpovědi bez vazby na to, které tvrzení z kterého zdroje pochází."
)

# Checklist/completeness-style questions ("co chybí", "jaké doklady", ...) get the
# full structured template; see CONCISE_ANSWER_INSTRUCTIONS below for everything else.
# --- Text-prompt fallback path -------------------------------------------------
# Used only if the model/server doesn't honor the JSON `format` schema below (see
# _request_structured_answer / _JSON schemas). Kept verbatim from the previous
# iteration so behaviour degrades gracefully to "best effort formatting" instead
# of breaking outright when swapping to a different LLM backend.
STRUCTURED_ANSWER_INSTRUCTIONS = HALLUCINATION_GUARD+"\n\n"+(
    "Dotaz je kontrola úplnosti/požadavků na dokumentaci. Odpověz PŘESNĚ v této struktuře a POUŽIJ DOSLOVA "
    "tyto čtyři nadpisy, v tomto pořadí, přesně takto napsané - nevymýšlej si vlastní nadpisy, nepoužívej "
    "markdown nadpisy (#, ##), emoji ani tabulky:\n\n"
    "Shrnutí:\n"
    "- krátká přímá odpověď na dotaz (1-2 věty)\n\n"
    "Požadované dokumenty / kroky:\n\n"
    "1. <Oblast, např. dokumentace betonáže>\n"
    "- <konkrétní položka> (Zdroj: <název dokumentu>)\n"
    "- <konkrétní položka> (Zdroj: <název dokumentu>)\n\n"
    "2. <další Oblast>\n"
    "- <konkrétní položka> (Zdroj: <název dokumentu>)\n\n"
    "(pokračuj dalšími číslovanými oblastmi jen podle témat, která skutečně najdeš ve ZDROJÍCH)\n\n"
    "Nenalezené informace:\n"
    "- co nebylo v dodaných ZDROJÍCH nalezeno (nebo napiš 'Žádné', pokud ZDROJE dotaz pokrývají beze zbytku)\n\n"
    "Zdroje:\n"
    "- <název dokumentu>\n"
    "- <název dokumentu>\n"
    "(vyjmenuj všechny použité zdrojové dokumenty, každý jen jednou)\n\n"
    "Nepřidávej žádný text ani nadpis navíc mimo těchto čtyř sekcí."
)

# A one-word product lookup or similarly simple query doesn't need the full
# checklist template - forcing empty "Nenalezené informace" / "Požadované
# dokumenty" sections on it just produces noise.
CONCISE_ANSWER_INSTRUCTIONS = HALLUCINATION_GUARD+"\n\n"+(
    "Toto je jednoduchý vyhledávací dotaz (např. název produktu nebo dokumentu), NE kontrola úplnosti. "
    "Odpověz stručně a věcně v 1-4 krátkých větách nebo odrážkách. NEPOUŽÍVEJ nadpisy 'Shrnutí' / 'Požadované dokumenty' / "
    "'Nenalezené informace' / 'Zdroje' a nevytvářej z odpovědi zbytečně dlouhý report - jen přímou odpověď se zdrojem u každého bodu."
)

# --- Structured JSON output path (primary) -------------------------------------
# Ollama's constrained decoding (`format`: JSON schema) forces the *shape* of the
# response at the token-generation level, so qwen3 physically cannot emit markdown
# headers/emoji/tables here - unlike the free-text prompt above, which it was
# repeatedly observed to ignore. Content quality (facts/požadavek/doporučení,
# hallucination guard) is still driven entirely by the guidance text; only the
# *structure* is delegated to the schema. The final markdown seen by the user is
# then assembled deterministically in Python (_render_structured_answer /
# _render_concise_answer), so a source name can never be garbled by the model -
# it is looked up directly from `results[zdroj_index-1]`, never generated as text.
JSON_ANSWER_GUARD = (
    "Jsi stavební asistent. Odpovídej výhradně česky a výhradně na základě dodaného kontextu ZDROJE - nic si nevymýšlej.\n"
    "Nevymýšlej technické požadavky, normy ani povinnosti, které nejsou doslova uvedené ve ZDROJÍCH.\n"
    "Ke KAŽDÉ položce uveď 'zdroj_index' - celé číslo zdroje [n] z kontextu ZDROJE, ze kterého tvrzení doslova pochází. "
    "Nikdy si zdroj nevymýšlej ani neuváděj číslo mimo rozsah uvedených ZDROJŮ.\n"
    "Pole 'typ' u každé položky vyjadřuje jistotu tvrzení:\n"
    "- \"fakt\" - doslovná informace ze zdroje\n"
    "- \"pozadavek\" - povinnost/podmínka uvedená ve zdroji\n"
    "- \"doporuceni\" - tvůj návrh nad rámec zdrojů, méně jistý, jasně tak označ\n"
)

STRUCTURED_JSON_GUIDANCE = JSON_ANSWER_GUARD+(
    "Dotaz je kontrola úplnosti/požadavků na dokumentaci. Rozděl odpověď do tematických oblastí (např. dokumentace "
    "betonáže, výztuž, geodetické zaměření) jen podle témat, která skutečně najdeš ve ZDROJÍCH. Do 'nenalezene' napiš, "
    "co nebylo v ZDROJÍCH k dotazu nalezeno - prázdné pole, pokud ZDROJE dotaz pokrývají beze zbytku."
)

CONCISE_JSON_GUIDANCE = JSON_ANSWER_GUARD+(
    "Toto je jednoduchý vyhledávací dotaz (např. název produktu nebo dokumentu), NE kontrola úplnosti. Odpověz stručně "
    "- 1 až 4 krátké položky v 'body', bez zbytečného rozšiřování. Nastav 'nenalezeno' na true, pokud ZDROJE dotaz "
    "vůbec nepokrývají."
)

_ANSWER_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Konkrétní tvrzení, dokument nebo krok - jedna myšlenka."},
        "zdroj_index": {"type": "integer", "description": "Číslo zdroje [n] z kontextu ZDROJE, ze kterého tvrzení pochází."},
        "typ": {"type": "string", "enum": ["fakt", "pozadavek", "doporuceni"]},
    },
    "required": ["text", "zdroj_index", "typ"],
}

STRUCTURED_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "shrnuti": {"type": "string", "description": "Krátká přímá odpověď na dotaz, 1-2 věty."},
        "oblasti": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nazev": {"type": "string", "description": "Název tematické oblasti, např. 'Dokumentace betonáže'."},
                    "polozky": {"type": "array", "items": _ANSWER_ITEM_SCHEMA},
                },
                "required": ["nazev", "polozky"],
            },
        },
        "nenalezene": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["shrnuti", "oblasti", "nenalezene"],
}

CONCISE_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "body": {"type": "array", "items": _ANSWER_ITEM_SCHEMA},
        "nenalezeno": {"type": "boolean", "description": "true, pokud dotaz nelze zodpovědět ze ZDROJŮ."},
    },
    "required": ["body", "nenalezeno"],
}

_TYPE_PREFIXES = {"pozadavek": "Požadavek: ", "doporuceni": "Doporučení: "}

def _clamp_source_index(raw_index, count):
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return None
    return index if 1 <= index <= count else None

def _render_answer_item(item, results):
    """Turns one model-produced {text, zdroj_index, typ} item into a rendered
    line plus the (validated) source index actually used, or None if the item
    has no usable text. The document name is looked up from `results` here -
    it never passes through the model as free text, so it cannot be garbled."""
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    prefix = _TYPE_PREFIXES.get(item.get("typ"), "")
    index = _clamp_source_index(item.get("zdroj_index"), len(results))
    document = results[index - 1]["document"] if index else None
    return prefix, text, index, document

def _render_structured_answer(data, results):
    lines = ["Shrnutí:", f"- {str(data.get('shrnuti') or '').strip() or 'Nenalezeno v indexovaných dokumentech.'}", "", "Požadované dokumenty / kroky:", ""]
    used_indexes = set()
    areas = [a for a in (data.get("oblasti") or []) if isinstance(a, dict)]
    for position, area in enumerate(areas, 1):
        name = str(area.get("nazev") or "").strip() or f"Oblast {position}"
        rendered_items = []
        for item in area.get("polozky") or []:
            rendered = _render_answer_item(item, results)
            if not rendered:
                continue
            prefix, text, index, document = rendered
            if index:
                used_indexes.add(index)
            source_note = f" (Zdroj: {document})" if document else ""
            rendered_items.append(f"- {prefix}{text}{source_note}")
        if not rendered_items:
            continue
        lines.append(f"{position}. {name}")
        lines.extend(rendered_items)
        lines.append("")
    missing = [str(m).strip() for m in (data.get("nenalezene") or []) if str(m).strip()]
    lines.append("Nenalezené informace:")
    lines.extend(f"- {m}" for m in missing) if missing else lines.append("- Žádné")
    lines.append("")
    lines.append("Zdroje:")
    if used_indexes:
        lines.extend(f"- {results[i - 1]['document']}" for i in sorted(used_indexes))
    else:
        lines.append("- Nenalezeno v indexovaných dokumentech.")
    return "\n".join(lines).strip()

def _render_concise_answer(data, results):
    if data.get("nenalezeno"):
        return "Nenalezeno v indexovaných dokumentech."
    blocks = []
    for item in data.get("body") or []:
        rendered = _render_answer_item(item, results)
        if not rendered:
            continue
        prefix, text, _index, document = rendered
        block = f"{prefix}{text}"
        if document:
            block += f"\n(Zdroj: {document})"
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else "Nenalezeno v indexovaných dokumentech."

# Mirrors ui_services.DEEP_ANALYSIS_KEYWORDS by design (ai_search.py must stay
# independent of the ui_services application layer, so this is intentionally a
# small, self-contained duplicate rather than a shared import).
CHECKLIST_QUERY_KEYWORDS = ("co chybí","jaké doklady","jaká dokumentace","zkontroluj","kompletnost","jaké jsou požadavky","požadavky investora","musí dodat","co je potřeba","soupis dokladů","co všechno")

def _is_checklist_query(query):
    folded=(query or "").casefold()
    return any(keyword in folded for keyword in CHECKLIST_QUERY_KEYWORDS)

CONFIDENCE_LABELS = {"green":"🟢 Vysoká","yellow":"🟡 Střední","red":"🔴 Nízká"}
CONFIDENCE_STRONG_SIMILARITY = 0.5

# Document-type heuristics for checklist-query confidence only (see
# _answer_confidence below). Independent, self-contained keyword lists - not
# shared with ui_services.PREFERRED_DOCUMENT_KEYWORDS by design, same rationale
# as CHECKLIST_QUERY_KEYWORDS above: this module stays free of the application
# layer. Root cause this addresses: a checklist query like "co chybí k předání
# základové desky investorovi?" can retrieve documents with strong lexical/
# semantic similarity to the query words (contracts/offers mentioning "základová
# deska" as an everyday construction term) without containing any of the
# technical/handover content the question actually needs - the old confidence
# score only measured match strength and source diversity, not topical fit, so
# it reported "🟢 Vysoká" even when the model correctly answered "nenalezeno".
CONTRACT_DOCUMENT_KEYWORDS = ("sod","smlouva","smlouvy","nabídka","cenová","objednávka","dopis","rozpočet")
TECHNICAL_DOCUMENT_KEYWORDS = ("technick","protokol","zápis","kniha betonů","beton","kzp","tp_","předávací","zkoušk","kontrol","výztuž","pentaflex")

def _document_type_haystack(row):
    return f"{row.get('document','')} {row.get('heading','')}".casefold()

def _is_contract_document(row):
    return any(keyword in _document_type_haystack(row) for keyword in CONTRACT_DOCUMENT_KEYWORDS)

def _is_technical_document(row):
    return any(keyword in _document_type_haystack(row) for keyword in TECHNICAL_DOCUMENT_KEYWORDS)

def _answer_confidence(query, results):
    """Deterministic confidence estimate from retrieval metadata only - no extra
    LLM call. Signals used: number of distinct source documents, how many have a
    "strong" match (exact FTS hit or high semantic similarity), and - to reward
    genuinely *independent* corroboration rather than just document count - whether
    all results come from the same folder and/or share the same file type (e.g.
    a folder full of near-identical "kontrolní den" scans isn't independent
    evidence even if it technically spans several documents).

    For checklist-style questions specifically, match strength/diversity alone is
    not enough - the sources also need to be topically relevant (technical/
    handover documentation), not just lexically similar. See CONTRACT_DOCUMENT_
    KEYWORDS/TECHNICAL_DOCUMENT_KEYWORDS above."""
    distinct_documents={row.get("document") for row in results if row.get("document")}
    distinct_folders={str(Path(row["path"]).parent) for row in results if row.get("path")}
    distinct_extensions={row.get("extension") for row in results if row.get("extension")}
    matches=[row.get("match") or {} for row in results]
    strong_hits=sum(1 for m in matches if m.get("fts_hit") or (m.get("semantic_similarity") or 0)>=CONFIDENCE_STRONG_SIMILARITY)
    same_folder=len(results)>=2 and len(distinct_folders)==1
    same_type=len(results)>=2 and len(distinct_extensions)==1
    if strong_hits==0:
        level,reason="red","málo podkladů se silnou shodou - odpověď ověřte v původních dokumentech"
    elif len(distinct_documents)<=1:
        level,reason="yellow","jeden relevantní zdroj s odpovídající shodou"
    elif same_folder:
        detail=" a stejného typu dokumentu" if same_type else ""
        level,reason="yellow",f"{len(distinct_documents)} zdroje, ale všechny ze stejné složky{detail} - nezávislost zdrojů je nižší"
    elif len(distinct_documents)>=3 and strong_hits>=2:
        level,reason="green",f"{len(distinct_documents)} nezávislých zdrojů z různých míst s odpovídající shodou"
    else:
        level,reason="yellow",f"{len(distinct_documents)} zdroje, ale jen omezená shoda mezi nimi"

    if results and _is_checklist_query(query):
        downgrade_notes=[]
        if not any(_is_technical_document(row) for row in results) and level=="green":
            level="yellow"; downgrade_notes.append("Zdroje neobsahují technickou/předávací dokumentaci odpovídající checklist dotazu.")
        contract_only=sum(1 for row in results if _is_contract_document(row) and not _is_technical_document(row))
        if contract_only>len(results)/2:
            level={"green":"yellow","yellow":"red","red":"red"}[level]
            downgrade_notes.append("Zdroje jsou převážně smluvní/obchodní dokumenty.")
        if downgrade_notes:
            reason=" ".join(downgrade_notes)
    return level,reason

def _call_ollama(model, prompt, format_schema=None, timeout=240):
    payload={"model":model,"stream":False,"think":False,"prompt":prompt}
    if format_schema is not None: payload["format"]=format_schema
    req=urllib.request.Request(OLLAMA_ENDPOINT,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.loads(response.read())["response"]

def answer(query, results):
    if not results: return {"answer":"Odpověď nelze vytvořit bez citací.","citations":[],"confidence":"red"}
    confidence_level,confidence_reason=_answer_confidence(query,results)
    confidence_block=f"\n\nJistota odpovědi:\n{CONFIDENCE_LABELS[confidence_level]}\n- {confidence_reason}"
    checklist=_is_checklist_query(query)
    context="\n\n".join(f"[{i}] {r['document']}"+(f" (sekce: {r['heading']})" if r.get("heading") else "")+f" | projekt {r['project']}\n{r['quote']}" for i,r in enumerate(results,1)); model=COMPLEX_MODEL if len(query)>180 or len(results)>6 else DEFAULT_MODEL
    guidance=STRUCTURED_JSON_GUIDANCE if checklist else CONCISE_JSON_GUIDANCE
    schema=STRUCTURED_ANSWER_SCHEMA if checklist else CONCISE_ANSWER_SCHEMA
    # The structured JSON path (format=schema) is preferred, but constrained
    # decoding measurably increases latency/variance under load and was observed
    # to occasionally exceed the timeout (TimeoutError) even when a plain-text
    # generation for the same prompt/context would have completed in time. Any
    # failure of the structured attempt - network/timeout/connection error *or*
    # the model returning something that isn't valid/renderable JSON - falls back
    # to the same free-text prompt/instructions used before this feature existed,
    # with the same `context`/`results`, so a slow or non-conforming response
    # degrades to the old behaviour instead of surfacing as "Ollama je nedostupná".
    try:
        raw=_call_ollama(model,f"{guidance}\n\nDOTAZ: {query}\n\nZDROJE:\n{context}",format_schema=schema)
        data=json.loads(raw)
        rendered=_render_structured_answer(data,results) if checklist else _render_concise_answer(data,results)
    except Exception:
        instructions=STRUCTURED_ANSWER_INSTRUCTIONS if checklist else CONCISE_ANSWER_INSTRUCTIONS
        try:
            rendered=_call_ollama(model,f"{instructions}\n\nDOTAZ: {query}\n\nZDROJE:\n{context}").strip()
        except Exception as exc2:
            return {"answer":f"Ollama je nedostupná: {type(exc2).__name__}. Nalezené citace zůstávají k dispozici.","citations":results,"model":model,"error":str(exc2),"confidence":confidence_level}
    final={"answer":rendered+confidence_block,"citations":results,"model":model,"confidence":confidence_level}
    return final

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=BOX_ROOT); parser.add_argument("--state-dir",type=Path,default=STATE_DIR); sub=parser.add_subparsers(dest="command",required=True); sub.add_parser("index"); sp=sub.add_parser("search"); sp.add_argument("query"); ap=sub.add_parser("answer"); ap.add_argument("query"); args=parser.parse_args(); emb=Embeddings(); db=args.state_dir/"ai_search.sqlite3"; lance=args.state_dir/"lancedb"
    result=sync(args.root,db,lance,emb) if args.command=="index" else search(args.query,db,lance,emb); result=answer(args.query,result) if args.command=="answer" else result; print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
