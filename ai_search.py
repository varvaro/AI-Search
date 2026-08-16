#!/usr/bin/env python3
"""Hybridní backend AI Search nad lokální složkou Box Drive."""
from __future__ import annotations
import argparse, collections, contextlib, fcntl, gc, hashlib, json, logging, multiprocessing, os, queue, re, resource, shutil, sqlite3, subprocess, tempfile, threading, time, traceback, unicodedata, urllib.request, uuid
import numpy as np
from pathlib import Path
from ai_search_config import (
    BOX_ROOT,
    STATE_DIR,
    EMBEDDING_MODEL,
    OLLAMA_ENDPOINT,
    DEFAULT_MODEL,
    COMPLEX_MODEL,
    PARSE_TIMEOUT_SECONDS,
    CHUNK_TIMEOUT_SECONDS,
    EMBEDDING_TIMEOUT_SECONDS,
    EMBEDDING_BATCH_SIZE,
    MSG_PARSE_TIMEOUT_SECONDS,
    AUXILIARY_TERM_COVERAGE_ENABLED,
    AUX_FTS_LIMIT,
    AUX_MAX_NEW_IDS,
    AUX_DF_RARE_MAX,
    AUX_PREFIX_DF_MAX,
    DOCUMENT_STATE_GATE_ENABLED,
    EVIDENCE_RUNTIME_VALIDATION_ENABLED,
    ENTITY_MATCH_BONUS_ENABLED,
    SUBJECT_ENTITY_ALIAS_ENABLED,
    REVISION_RANKING_ENABLED,
    REVISION_RECALL_ENABLED,
    OLD_REVISION_GUARD_ENABLED,
    CITATION_CONTRACT_ENABLED,
    ABSTENTION_OVERRIDE_ENABLED,
    STRUCTURED_SUMMARY_CITATION_ENABLED,
    FALLBACK_CITATION_CONTRACT_ENABLED,
    JSON_SENTINEL_FALLBACK_ENABLED,
    QUERY_FOCUSED_CONTEXT_PACKING_ENABLED,
    ENTITY_HINTS_ENABLED,
    METADATA_RERANK_ENABLED,
    DOCUMENT_CLASS_AFFINITY_ENABLED,
    FAMILY_REVISION_RERANK_ENABLED,
    PDF_MULTI_PSM_OCR_ENABLED,
)
import entity_match_bonus  # PR8.1.1/8.1.2: optional Phase-3 entity name/path bonus
import revision_ranking  # PR8.2: optional Phase-3 revision intent score
import revision_recall  # PR8.2.1: optional append-only revision candidate recall
import old_revision_guard  # PR8.3: optional OLD/ authority demotion in answer()
import context_packing  # PR9.3.3: optional pre-LLM query-focused context packing
import entity_hints  # PR9.3.4: optional entity/identifier candidates in the answer prompt
import metadata_rerank  # PR9.4.1: optional Phase-3 token overlap / date / discriminator bonus
import document_class_affinity  # PR9.4.2: optional Phase-3 query↔document class bonus
import drawing_navigation  # PR9.7.3: deterministic drawing-navigation answers
import drawing_local_evidence  # PR9.7.4: document-local drawing evidence quote
import family_revision_rerank  # PR9.4.4: intent-gated BM25 floor + family latest bonus
import pdf_ocr_candidates  # PR9.5.0: multi-PSM OCR scoring (flag-gated)
from document_extractors import INDEXED_EXTS, extract_text, extract_eml, clean_cell_text, format_sheet_section
import parsing_worker  # stable multiprocessing.Process targets, see parsing_worker.py docstring
import query_expansion  # Query Understanding layer, opt-in via search(expand_query=True)
import auxiliary_term_coverage  # PR5 spike: optional pre-rerank FTS candidate union
import document_state  # PR6: deterministic signed-contract answer safety gate

SUPPORTED = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".md", ".csv", ".rtf", ".eml", ".msg", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

def resolve_system_tool(name):
    """Resolve a CLI tool from PATH, then common Homebrew prefixes."""
    resolved=shutil.which(name)
    if resolved: return resolved
    for prefix in (Path("/opt/homebrew/bin"),Path("/usr/local/bin")):
        candidate=prefix/name
        if candidate.is_file() and os.access(candidate,os.X_OK): return str(candidate)
    return None

def _required_system_tool(name):
    resolved=resolve_system_tool(name)
    if resolved is None: raise RuntimeError(f"Systémový nástroj '{name}' není dostupný.")
    return resolved

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
    tesseract=_required_system_tool("tesseract")
    run = subprocess.run([tesseract, str(path), "stdout", "-l", "ces+eng"], capture_output=True, check=False, timeout=PARSE_TIMEOUT_SECONDS)
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
    pdfinfo=resolve_system_tool("pdfinfo")
    if pdfinfo is None: return None
    try:
        run=subprocess.run([pdfinfo,str(path)],capture_output=True,text=True,check=False,timeout=timeout)
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

def _tesseract_command(tesseract, image, psm):
    return [tesseract, str(image), "stdout", "-l", "ces+eng", "--psm", str(psm)]

def _ocr_indexable_text(raw_text):
    """Join heading+body from chunks() so scoring sees what FTS will see."""
    parts = []
    for heading, body in chunks(raw_text or ""):
        if heading: parts.append(heading)
        if body: parts.append(body)
    return "\n".join(parts)

def _ocr_rendered_image(tesseract, image, deadline, allow_multi_psm):
    """OCR one already-rendered PNG. Extra PSM runs only when the flag is on,
    the page is a single-page OCR fallback, and psm6 indexable text is weak.
    Extra attempts use the same _run_ocr_subprocess slot; their timeout never
    fails the page — the best candidate so far is kept."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(_tesseract_command(tesseract, image, "6"), 0)
    primary_run = _run_ocr_subprocess(
        _tesseract_command(tesseract, image, "6"),
        min(PDF_PAGE_OCR_TIMEOUT_SECONDS, remaining),
    )
    if primary_run.returncode:
        return None
    primary = (primary_run.stdout or "").strip()
    if not (PDF_MULTI_PSM_OCR_ENABLED and allow_multi_psm):
        return primary
    candidates = [pdf_ocr_candidates.OCRCandidate("psm6", primary, _ocr_indexable_text(primary))]
    if not pdf_ocr_candidates.is_weak(candidates[0]):
        return primary
    for psm in ("12", "3"):
        remaining = deadline - time.monotonic()
        if remaining < PDF_OCR_SECONDS_PER_PAGE_BUDGET:
            break
        try:
            extra_run = _run_ocr_subprocess(
                _tesseract_command(tesseract, image, psm),
                min(PDF_PAGE_OCR_TIMEOUT_SECONDS, remaining),
            )
        except subprocess.TimeoutExpired:
            break
        if extra_run.returncode:
            continue
        extra = (extra_run.stdout or "").strip()
        candidates.append(pdf_ocr_candidates.OCRCandidate(f"psm{psm}", extra, _ocr_indexable_text(extra)))
    return pdf_ocr_candidates.choose_best(candidates).raw_text

def _extract_pdf_per_page(path,page_count,deadline):
    pdftoppm=_required_system_tool("pdftoppm"); tesseract=_required_system_tool("tesseract")
    with tempfile.TemporaryDirectory(prefix="ai-search-ocr-") as folder:
        folder_path=Path(folder); parts=[]; failed=0
        for page in range(1,page_count+1):
            remaining=deadline-time.monotonic()
            if remaining<=0: parts.append(_pdf_page_marker(page,"překročen časový limit dokumentu")); failed+=1; continue
            prefix=str(folder_path/"page")
            try: rendered=_run_ocr_subprocess([pdftoppm,"-r","300","-png","-f",str(page),"-l",str(page),str(path),prefix],min(PDF_PAGE_RENDER_TIMEOUT_SECONDS,remaining))
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
            try: text=_ocr_rendered_image(tesseract,image,deadline,allow_multi_psm=(page_count==1))
            except subprocess.TimeoutExpired: parts.append(_pdf_page_marker(page,"OCR překročil časový limit")); failed+=1
            else:
                if text is None: parts.append(_pdf_page_marker(page,"OCR selhal")); failed+=1
                else: parts.append(text if text else f"[OCR PRÁZDNÁ STRÁNKA {page}]")
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
    pdftoppm=_required_system_tool("pdftoppm"); tesseract=_required_system_tool("tesseract")
    with tempfile.TemporaryDirectory(prefix="ai-search-ocr-") as folder:
        folder_path=Path(folder); remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError("PDF překročilo časový limit před OCR")
        try: rendered=_run_ocr_subprocess([pdftoppm,"-r","300","-png",str(path),str(folder_path/"page")],remaining)
        except subprocess.TimeoutExpired as exc: raise TimeoutError("Převod PDF pro OCR překročil časový limit") from exc
        if rendered.returncode: raise RuntimeError("PDF nelze převést pro OCR: "+(rendered.stderr.strip() or str(rendered.returncode)))
        images=sorted(folder_path.glob("page-*.png")); parts=[]; failed=0
        allow_multi_psm=len(images)==1
        for index,image in enumerate(images,1):
            remaining=deadline-time.monotonic()
            if remaining<=0: parts.append(_pdf_page_marker(index,"překročen časový limit dokumentu")); failed+=1
            else:
                try: text=_ocr_rendered_image(tesseract,image,deadline,allow_multi_psm=allow_multi_psm)
                except subprocess.TimeoutExpired: parts.append(_pdf_page_marker(index,"OCR překročil časový limit")); failed+=1
                else:
                    if text is None: parts.append(_pdf_page_marker(index,"OCR selhal")); failed+=1
                    else: parts.append(text if text else f"[OCR PRÁZDNÁ STRÁNKA {index}]")
            try: image.unlink()
            except OSError: pass
        if images and failed==len(images): raise RuntimeError(f"OCR selhalo na všech {len(images)} stránkách PDF")
        return "\n".join(parts)

def extract_pdf(path, budget_seconds=None):
    page_count=_pdf_page_count(path)
    if budget_seconds is None: budget_seconds=pdf_ocr_document_budget_seconds(page_count)
    deadline=time.monotonic()+budget_seconds
    pdftotext=resolve_system_tool("pdftotext")
    try: direct=subprocess.run([pdftotext,"-layout",str(path),"-"],capture_output=True,text=True,check=False,timeout=min(PDF_NATIVE_TEXT_TIMEOUT_SECONDS,budget_seconds)).stdout if pdftotext else ""
    except subprocess.TimeoutExpired as exc: raise TimeoutError("Textová vrstva PDF překročila časový limit") from exc
    except OSError: direct=""
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
RERANK_POOL_SIZE = 80           # Phase 2 lookup window (PR9.4.2); questions use QA_RERANK_POOL_SIZE
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
# Weaker sibling of FILENAME_MATCH_BONUS, scoped ONLY to expand_query's
# abbreviation-category terms (query_expansion.py's "abbreviations" list per
# rule - never "synonyms"/"processes"/"documents"). Fixes the 2026-08-09
# rr-kzp-monolit-feri-01 diagnostic: a spelled-out query ("kontrolní a
# zkušební plán") never satisfies the full-string `needle` filename check
# above even when the file itself is named with the acronym ("KZP monolit
# Smíchov - formulář.xls"), because `needle` is the whole query, not its
# individual tokens/expansion terms. Deliberately smaller than
# FILENAME_MATCH_BONUS (weaker evidence - a 2-4 char acronym in a filename is
# far less discriminative than the full query matching verbatim) but still
# > 1/60 so it can override a pure RRF-rank tie the same way FILENAME_MATCH_BONUS does.
ABBREVIATION_FILENAME_MATCH_BONUS = 0.02
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

# 2026-08-09 retrieval regression benchmark (rr-tp-strikane-betony-01,
# rr-tp-odvodneni-watersystem-01, rr-zl-roznaseci-deska-01): the 2-char
# abbreviations "TP" and "ZL" are also this corpus's own filename NUMBERING
# CONVENTION ("TP 2.4 - ...", "ZL č.001 - ..."), so they match dozens of
# unrelated documents' filenames and, amplified by search_all()'s multi-chunk
# evidence aggregation, pushed the actually-relevant document out of the
# top 10 for other, unrelated queries. Excluding needles shorter than this
# closes that hole while keeping the target "KZP"/3-char+ abbreviations
# (which did not reproduce the false-positive pattern in that same benchmark
# run) eligible.
ABBREVIATION_FILENAME_MIN_LENGTH = 3

def _abbreviation_filename_needles(expansion: "query_expansion.QueryExpansion | None") -> set[str]:
    """Casefolded abbreviation terms (query_expansion.py's "abbreviations"
    category only - never "synonyms"/"processes"/"documents") that survived
    THIS query's expand_query() budget, for ABBREVIATION_FILENAME_MATCH_BONUS.
    Pure/side-effect-free so it can be unit-tested without a full search()
    call - see tests/test_query_expansion.py.

    Always empty when `expansion` is None, i.e. expand_query=False (the
    default) or a query that matched no dictionary rule - the caller's bonus
    is then unconditionally skipped, leaving scoring byte-for-byte identical
    to before this function existed. Terms shorter than
    ABBREVIATION_FILENAME_MIN_LENGTH are dropped - see its comment."""
    if expansion is None:
        return set()
    needles: set[str] = set()
    for rule in expansion.matched_rules:
        vocab_abbreviations = query_expansion.DOMAIN_VOCABULARY.get(rule["key"], {}).get("abbreviations", ())
        needles.update(
            term.casefold() for term in rule["terms"]
            if term in vocab_abbreviations and len(term) >= ABBREVIATION_FILENAME_MIN_LENGTH
        )
    return needles

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
    _family_revision_intent=bool(FAMILY_REVISION_RERANK_ENABLED) and family_revision_rerank.has_revision_intent(query)
    if (_tracing or _family_revision_intent) and fts_ids:
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

    # PR5 / PR5.1: optional auxiliary conjunctive FTS — APPEND-only onto top_ids
    # before Phase 3. Never mutates fts_ids/vector_ids/rrf_scores/fusion_order.
    # Flag OFF (default) skips this block entirely → bit-identical to pre-PR5.
    # PR5.1: aux_matched_ids powers diagnostic candidate_origin / aux_hit only.
    aux_trace=None
    aux_matched_ids=set()
    if AUXILIARY_TERM_COVERAGE_ENABLED:
        try:
            aux_result=auxiliary_term_coverage.collect_auxiliary_chunk_ids(
                db_path,
                query,
                exclude_ids=top_ids,
                df_rare_max=AUX_DF_RARE_MAX,
                fts_limit=AUX_FTS_LIMIT,
                max_new_ids=AUX_MAX_NEW_IDS,
                prefix_df_max=AUX_PREFIX_DF_MAX,
            )
            aux_trace=aux_result.as_trace_dict()
            aux_matched_ids=set(aux_result.matched_ids)
            if aux_result.added_ids:
                top_ids=list(top_ids)+list(aux_result.added_ids)
        except Exception as exc:
            # Spike must never break retrieval — degrade to the pre-aux pool.
            aux_trace={"activated":False,"reason":"error","error":repr(exc),
                       "added_ids":[],"matched_ids":[],"matched_count":0,"added_count":0,
                       "anchor":None,"constraints":[],"match":None}

    # PR8.2.1: revision-intent append-only recall — AFTER aux, BEFORE Phase 3.
    # Flag OFF → skip entirely (byte-identical). Seeds a floor RRF score for
    # newly appended ids so Phase-3 `rrf_scores[cid]` never KeyErrors.
    rev_recall_trace=None
    rev_recall_added_ids=set()
    if REVISION_RECALL_ENABLED:
        try:
            rr_result=revision_recall.collect_revision_chunk_ids(
                db_path, query, exclude_ids=top_ids,
            )
            rev_recall_trace=rr_result.as_trace_dict()
            if rr_result.added_ids:
                floor=1.0/(RRF_RANK_CONSTANT+max(rerank_k,1))
                top_ids=list(top_ids)
                for cid in rr_result.added_ids:
                    if cid not in rrf_scores:
                        rrf_scores[cid]=floor
                    top_ids.append(cid)
                    rev_recall_added_ids.add(cid)
        except Exception as exc:
            rev_recall_trace={"activated":False,"reason":"error","error":repr(exc),
                              "added_ids":[],"added_count":0,"matched_document_names":[]}

    # PR9.4.4: intent-gated BM25-floor admission. Append-only onto top_ids.
    # Uses chunk ids already in fts_ids (no new FTS). fusion_order / rrf_scores
    # stay unchanged. Flag OFF or no revision intent → this block is a no-op.
    family_admission_ids=set()
    if _family_revision_intent:
        vector_doc_id_by_chunk={row["id"]:row.get("document_id") for row in vector_rows}
        for cid,did in vector_doc_id_by_chunk.items():
            if cid not in doc_id_by_chunk and did is not None:
                doc_id_by_chunk[cid]=did
        extras=family_revision_rerank.select_bm25_floor_chunk_ids(fts_ids,top_ids,doc_id_by_chunk)
        if extras:
            top_ids=list(top_ids)+list(extras)
            family_admission_ids=set(extras)

    if _tracing:
        fts_rank_of={cid:i for i,cid in enumerate(fts_ids)}; vector_rank_of={cid:i for i,cid in enumerate(vector_ids)}
        vector_doc_id_by_chunk={row["id"]:row.get("document_id") for row in vector_rows}
        trace.rrf_candidates=[{"chunk_id":cid,"rank":i,"score":rrf_scores[cid],"bm25_rank":fts_rank_of.get(cid),"vector_rank":vector_rank_of.get(cid),
            "document_id":doc_id_by_chunk.get(cid,vector_doc_id_by_chunk.get(cid))} for i,cid in enumerate(fusion_order)]
        # union_candidates is always populated (independent of candidate_strategy)
        # so a benchmark can compare "what RRF would have kept" against
        # "everything either channel found" even on a legacy-strategy run.
        trace.union_candidates=union if union is not None else _build_candidate_union(fts_ids,vector_ids)
        primary_for_trace=set(fts_ids)|set(vector_ids)
        if AUXILIARY_TERM_COVERAGE_ENABLED:
            trace.candidates_before_precision=[{
                "chunk_id":cid,
                "aux_hit":cid in aux_matched_ids,
                "candidate_origin":auxiliary_term_coverage.candidate_origin(
                    primary=cid in primary_for_trace, aux_hit=cid in aux_matched_ids),
            } for cid in top_ids]
        else:
            trace.candidates_before_precision=[{"chunk_id":cid} for cid in top_ids]
        if candidate_strategy==CANDIDATE_STRATEGY_UNION_CE:
            trace.candidates_before_cross_encoder=[{"chunk_id":cid} for cid in top_ids]
        trace.timings["fusion_rrf"]=(time.perf_counter()-_t)*1000
        trace.metadata["candidate_pool_size_before_truncation"]=len(fusion_order)
        trace.metadata["candidate_pool_size_after_truncation"]=len(top_ids)
        if aux_trace is not None:
            trace.metadata["auxiliary_term_coverage"]=aux_trace
        if rev_recall_trace is not None:
            trace.metadata["revision_recall"]=rev_recall_trace
            if REVISION_RECALL_ENABLED:
                for entry in trace.candidates_before_precision:
                    entry["revision_recall_hit"]=entry.get("chunk_id") in rev_recall_added_ids

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
    abbreviation_needles=_abbreviation_filename_needles(expansion)
    # Fetched as a separate pass (not interleaved with scoring like before)
    # because candidate_strategy="union_ce" needs every candidate's full chunk
    # text collected up front for ONE batched cross-encoder call - a per-
    # candidate "fetch, then immediately score" loop cannot batch. Output is
    # identical either way; this only changes *when* the DB rows are read, not
    # which rows or what they contain.
    rows_by_id={}
    with database(db_path) as con:
        for cid in top_ids:
            # d.id is additive identity for EvidenceSet (PR3); chunk id is `cid`.
            # Column order of name/path/project/heading/text stays unchanged so
            # existing row[0]..row[4] consumers (cross-encoder, quote) are untouched.
            row=con.execute("SELECT d.name,d.path,d.project,c.heading,c.text,d.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?",(cid,)).fetchone()
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
        # Internal-only signal (deliberately not added to `match` below): does
        # an expand_query() abbreviation term appear in this candidate's
        # filename. See ABBREVIATION_FILENAME_MATCH_BONUS's docstring - never
        # computed when filename_match already fired (no double bonus).
        abbreviation_filename_match=(not filename_match) and bool(abbreviation_needles) and any(abbr in row[0].casefold() for abbr in abbreviation_needles)
        quality=_chunk_quality_factor(len(row[4] or ""),exempt=fts_hit or filename_match or abbreviation_filename_match)
        # KROK 7: when the cross-encoder actually ran, its score IS the final
        # precision score - not blended with RRF/cosine/BM25 via new manual
        # weights. Those signals stay available for diagnostics in `match`/
        # SearchTrace, they just no longer determine the order.
        score=ce_score_by_id[cid] if ce_score_by_id is not None else rrf_scores[cid]*quality+(FILENAME_MATCH_BONUS if filename_match else ABBREVIATION_FILENAME_MATCH_BONUS if abbreviation_filename_match else 0.0)
        # PR8.1.1 / PR8.1.2: additive entity name/path bonus. Both flags OFF →
        # no-op (historical `match` keys only). Explicit tokens and subject
        # aliases share one capped bonus; sources are logged in entity_match.
        entity_detail=None
        if ENTITY_MATCH_BONUS_ENABLED or SUBJECT_ENTITY_ALIAS_ENABLED:
            entity_detail=entity_match_bonus.compute_entity_match_bonus(
                query,row[0],row[1] or "",
                include_explicit=ENTITY_MATCH_BONUS_ENABLED,
                include_subject_aliases=SUBJECT_ENTITY_ALIAS_ENABLED,
            )
            score=score+entity_detail.bonus
        # fts_hit remains PRIMARY-channel FTS only (unchanged meaning). PR5.1
        # diagnostics (aux_hit / candidate_origin) are additive and only when
        # the aux feature flag is on — OFF keeps the historical match keys.
        match={"fts_hit":fts_hit,"vector_hit":cid in vector_id_set,"semantic_similarity":max(similarity_by_id.get(cid,0.0),0.0),"filename_match":filename_match,"chunk_quality":quality}
        if entity_detail is not None:
            match["entity_match_bonus"]=entity_detail.bonus
            match["entity_match"]=entity_detail.as_trace_dict()
        # PR8.2: intent-gated revision score. Flag OFF → no-op. Non-intent
        # queries also contribute 0 even when the flag is ON.
        revision_detail=None
        if REVISION_RANKING_ENABLED:
            revision_detail=revision_ranking.compute_revision_score(query,row[0],row[1] or "")
            score=score+revision_detail.bonus
            match["revision_score"]=revision_detail.bonus
            match["revision"]=revision_detail.as_trace_dict()
        # PR9.4.1: additive Phase-3 metadata bonus (generic token overlap /
        # date whitelist / discriminator matching against name+path only).
        # Flag OFF → no-op. skip_token_overlap avoids double-counting a query
        # that already earned FILENAME_MATCH_BONUS via a full verbatim match.
        metadata_detail=None
        if METADATA_RERANK_ENABLED:
            metadata_detail=metadata_rerank.compute_metadata_score(
                query,row[0],row[1] or "",
                skip_token_overlap=filename_match,
            )
            score=score+metadata_detail.bonus
            match["metadata_rerank_bonus"]=metadata_detail.bonus
            match["metadata_rerank"]=metadata_detail.as_trace_dict()
        # PR9.4.2: additive Phase-3 query-class ↔ document-class affinity.
        # Flag OFF → no-op. Status/signed-contract queries always contribute 0
        # (see document_class_affinity) so LOI/signed never becomes a SoD proof.
        class_detail=None
        if DOCUMENT_CLASS_AFFINITY_ENABLED:
            class_detail=document_class_affinity.compute_class_affinity(
                query,row[0],row[1] or "",
            )
            score=score+class_detail.bonus
            match["document_class_affinity"]=class_detail.as_trace_dict()
        if _family_revision_intent:
            match["admission_source"]="bm25_revision_floor" if cid in family_admission_ids else "fusion"
        if AUXILIARY_TERM_COVERAGE_ENABLED:
            aux_hit=cid in aux_matched_ids
            match["aux_hit"]=aux_hit
            match["candidate_origin"]=auxiliary_term_coverage.candidate_origin(
                primary=fts_hit or cid in vector_id_set, aux_hit=aux_hit)
        if ce_score_by_id is not None: match["cross_encoder_score"]=ce_score_by_id[cid]
        # document_id/chunk_id are additive (EvidenceSet PR3). Ranking/score/
        # existing public fields are unchanged; callers that only read the
        # historical keys keep identical behaviour.
        output.append({"document":row[0],"path":row[1],"project":row[2],"quote":row[4][:700],"heading":row[3],"score":score,"match":match,"document_id":row[5],"chunk_id":cid})
        # Cheap `if` on an already-existing per-candidate DB round trip -
        # negligible next to the SQLite query on the line above; only
        # runs at all when a trace is attached.
        if _tracing: trace_ids.append(cid)
    if _family_revision_intent and output:
        for row, detail in zip(output, family_revision_rerank.annotate_family_revision(output, query)):
            row["score"]=row["score"]+detail.bonus
            row["match"]["family_revision_bonus"]=detail.bonus
            row["match"]["family_revision"]=detail.as_trace_dict()
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
    "Ke KAŽDÉ položce uveď 'zdroj_index' - 1-based číslo zdroje z kontextu ZDROJE, ze kterého tvrzení doslova pochází.\n"
    "Platné hodnoty jsou pouze 1 až N, kde N je počet zdrojů zobrazených v sekci ZDROJE.\n"
    "Příklad: [1] dokument A → zdroj_index: 1; [2] dokument B → zdroj_index: 2.\n"
    "Nikdy nepoužívej 0, záporné číslo ani číslo větší než N. Hodnota 0 je neplatná.\n"
    "Nepoužívej 0 jako „žádný zdroj“ ani jako odkaz na první dokument.\n"
    "Pokud tvrzení nelze přiřadit ke konkrétnímu zdroji, takovou faktickou položku NEVYTVÁŘEJ.\n"
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
        "zdroj_index": {
            "type": "integer",
            "description": (
                "1-based index zdroje z kontextu [1] až [N]. "
                "Hodnota 0 je neplatná. Použij pouze index konkrétního zdroje, "
                "který podporuje tuto položku."
            ),
        },
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
    # PR8.4.3: same override contract as _render_concise_answer, applied
    # defensively here too even though STRUCTURED_ANSWER_SCHEMA has no
    # official "nenalezeno" field today (bool(None) is False, so this is a
    # no-op unless a future schema/prompt change adds one) — keeps both
    # renderers symmetric under the same contract.
    override_abstention = ABSTENTION_OVERRIDE_ENABLED and bool(data.get("nenalezeno"))
    require_citation = CITATION_CONTRACT_ENABLED or override_abstention
    any_valid_item = False
    for position, area in enumerate(areas, 1):
        name = str(area.get("nazev") or "").strip() or f"Oblast {position}"
        rendered_items = []
        for item in area.get("polozky") or []:
            rendered = _render_answer_item(item, results)
            if not rendered:
                continue
            prefix, text, index, document = rendered
            # PR8.4.1: citation contract — same rule as _render_concise_answer.
            if require_citation and not document:
                continue
            any_valid_item = True
            if index:
                used_indexes.add(index)
            source_note = f" (Zdroj: {document})" if document else ""
            rendered_items.append(f"- {prefix}{text}{source_note}")
        if not rendered_items:
            continue
        lines.append(f"{position}. {name}")
        lines.extend(rendered_items)
        lines.append("")
    # PR8.4.3: conflicting `nenalezeno` is ignored only when a surviving item
    # exists; otherwise the sentinel wins.
    # PR8.4.4: `shrnuti` is not a source of truth — the same "at least one
    # surviving polozky item" rule decides whether a factual structured
    # answer may be shown at all. Flag OFF → this branch is a no-op and
    # `shrnuti` is emitted even when every item was dropped (pre-PR8.4.4).
    if (override_abstention or STRUCTURED_SUMMARY_CITATION_ENABLED) and not any_valid_item:
        return "Nenalezeno v indexovaných dokumentech."
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
    # PR8.4.3: abstention override — a model-declared `nenalezeno` is only
    # trusted when no item in the same response resolves to real evidence.
    # Citation is enforced UNCONDITIONALLY for this decision (independent of
    # CITATION_CONTRACT_ENABLED's own, separate default) — an explicit "not
    # found" signal must not be overridden by anything less than a genuinely
    # cited claim. Flag OFF (or no `nenalezeno`) → identical to pre-PR8.4.3.
    override_abstention = ABSTENTION_OVERRIDE_ENABLED and bool(data.get("nenalezeno"))
    if data.get("nenalezeno") and not override_abstention:
        return "Nenalezeno v indexovaných dokumentech."
    # PR8.4.1: citation contract — a claim item whose zdroj_index never
    # resolved to a real result row has no verifiable source and must not
    # survive rendering. Flag OFF keeps the pre-PR8.4.1 behaviour (item kept,
    # just missing its "(Zdroj: ...)" note) UNLESS the override above already
    # forces citation regardless of the flag.
    require_citation = CITATION_CONTRACT_ENABLED or override_abstention
    blocks = []
    for item in data.get("body") or []:
        rendered = _render_answer_item(item, results)
        if not rendered:
            continue
        prefix, text, _index, document = rendered
        if require_citation and not document:
            continue
        block = f"{prefix}{text}"
        if document:
            block += f"\n(Zdroj: {document})"
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else "Nenalezeno v indexovaných dokumentech."

_FALLBACK_SENTINEL = "Nenalezeno v indexovaných dokumentech."

def _fallback_text_has_pool_source(text, results):
    """PR8.4.6: does unconstrained fallback prose mention a pool document?

    Filename substring match after the same ASCII-fold used by
    `benchmark.answer_evidence` / `_fold_plain`. Deliberately not an NLI
    check — a real name from `results` in the text is enough to keep the
    fallback; anything else (empty, the not-found sentinel, a name that is
    not in the pool) is not a verifiable source.
    """
    body = (text or "").strip()
    if not body:
        return False
    folded = _fold_plain(body)
    if folded == _fold_plain(_FALLBACK_SENTINEL):
        return False
    for row in results or ():
        name = str((row or {}).get("document") or "").strip()
        if name and _fold_plain(name) in folded:
            return True
    return False

def _answer_item_has_text(item):
    """True when a model JSON item attempted a non-empty claim (`text`)."""
    if not isinstance(item, dict):
        return False
    return bool(str(item.get("text") or "").strip())

def _json_payload_has_substantive_answer_item(data):
    """Did the parsed JSON try to answer with at least one claim item?

    Counts concise `body` items and structured `oblasti[].polozky` items
    that have non-empty `text`. Ignores `shrnuti`, `nenalezene`, empty
    items, and other metadata — those must not trigger PR9.2.1 fallback.
    """
    if not isinstance(data, dict):
        return False
    for item in data.get("body") or []:
        if _answer_item_has_text(item):
            return True
    for area in data.get("oblasti") or []:
        if not isinstance(area, dict):
            continue
        for item in area.get("polozky") or []:
            if _answer_item_has_text(item):
                return True
    return False

def _apply_free_text_fallback(query, context, answer_results, checklist, model):
    """One unconstrained Ollama call + the PR8.4.6 filename contract.

    Shared by the JSON-exception branch and PR9.2.1 (JSON rendered to
    sentinel despite substantive items). Raises if the fallback call itself
    fails — callers decide whether that is 'Ollama nedostupná' (exception
    path) or 'keep the JSON sentinel' (PR9.2.1 path).
    """
    instructions=STRUCTURED_ANSWER_INSTRUCTIONS if checklist else CONCISE_ANSWER_INSTRUCTIONS
    rendered=_call_ollama(model,f"{instructions}\n\nDOTAZ: {query}\n\nZDROJE:\n{context}").strip()
    if FALLBACK_CITATION_CONTRACT_ENABLED and not _fallback_text_has_pool_source(rendered, answer_results):
        rendered=_FALLBACK_SENTINEL
    return rendered

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

# --- PR6/PR7.3: DocumentState answer safety gate --------------------------
# Deterministic, filename/path-only guard against the "SIGNED contract exists
# in `results` but the LLM answer denies it" failure class (see the HAUS365
# audit: SOD_HAUS365_NDS_k el. podpisu_podepsané.pdf ranked #1, answer wrongly
# said "není podepsaná" citing an unrelated TP document instead).
#
# Runs AFTER the LLM answer is rendered and BEFORE it is returned to the
# caller - it only rewrites `rendered` text when a forbidden claim pattern is
# detected; an already-compliant answer passes through byte-identical. It
# NEVER touches retrieval, RRF, scoring, ranking, or `results` order - see
# document_state.py's own docstring for the same guarantee at the
# classification layer. State decisions are 100% rule-based - no LLM is
# consulted for the state itself, only for the free-text answer being validated.
#
# PR7.3 split this in two: the state DECISION moved out to
# evidence_runtime.build_state_coverage() (via _answer_state_coverage below),
# leaving the gate with text policy only. The regexes below are therefore the
# gate's whole remaining domain - what the answer CLAIMS - while what is TRUE
# comes from one StateCoverage shared with the diagnostic layer.
def _fold_plain(text):
    """ASCII-folded casefold, same normalization document_state.py uses
    internally (duplicated here - see CHECKLIST_QUERY_KEYWORDS above for the
    established precedent of small, self-contained duplication across this
    module boundary rather than exposing a private helper as public API)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )

# PR6.2 fix: the original {0,80}-char lookahead window was far too wide - it
# flagged epistemic hedges ("není UVEDENO, ZDA byla smlouva podepsána") as a
# hard negative claim, because "neni" and "podeps" both occurred somewhere in
# the same sentence regardless of what stood between them. Replaced with a
# tight, phrase-adjacent pattern list (word(s) immediately before "podepsan"),
# which naturally excludes hedges - "zda byla ... podepsana" never matches
# because "zda"/"uvedeno"/"smlouva" sit directly between the trigger word and
# "podepsan", breaking the required adjacency - with no separate denylist
# needed for "není uvedeno, zda...", "nelze ověřit, zda...", "není jasné, zda...".
_STATE_GATE_NEGATIVE_SIGNED_PATTERNS = (
    r"\bneni\b\s+podepsan\w*",
    r"\bnebyla\b\s+podepsan\w*",
    r"\bnebyl\b\s+podepsan\w*",
    r"\bnejsou\b\s+podepsan\w*",
    r"\bneexistuje\b\s+podepsan\w*",
    r"\bnenas\w*\s+jsem\s+podepsan\w*",  # "nenašel/nenašla jsem podepsanou"
    r"\bnenalez\w*\s+jsem\s+podepsan\w*",  # "nenalezl/nenalezla jsem podepsanou"
    r"\bnepodarilo\s+se\s+najit\s+podepsan\w*",  # "nepodařilo se najít podepsanou"
)
_STATE_GATE_NEGATIVE_SIGNED_RE = re.compile("|".join(_STATE_GATE_NEGATIVE_SIGNED_PATTERNS))
_STATE_GATE_GENERIC_NOT_FOUND_RE = re.compile(r"nenalezeno v indexovanych dokumentech")
_STATE_GATE_POSITIVE_SIGNED_PATTERNS = (
    r"\bje\b\s+podepsan\w*",
    r"\bjsou\b\s+podepsan\w*",
    r"\bbyla\b\s+podepsan\w*",
    r"\bbyl\b\s+podepsan\w*",
    r"\bexistuje\b\s+podepsan\w*",
)
_STATE_GATE_POSITIVE_SIGNED_RE = re.compile("|".join(_STATE_GATE_POSITIVE_SIGNED_PATTERNS))


def _answer_state_coverage(query, results):
    """THE single lifecycle-state decision for one answered query (PR7.3).

    Before this, ai_search made that decision twice: the PR6 gate had its own
    _document_state_outcome() and the PR7.2 diagnostic layer independently built
    a StateCoverage. The two used different entity-matching rules, so they could
    disagree on the same query - two answers to "is this contract signed?" in one
    process. evidence_runtime.build_state_coverage() is now the only one, and
    both consumers read it.

    Pure w.r.t. document_state's pure functions: `results` is read-only input,
    never reordered, filtered, or rescored - it is read here only to classify
    filenames, never to influence ranking.
    """
    from evidence_runtime import build_state_coverage
    return build_state_coverage(
        document_state.derive_state_requirement(query),
        [
            document_state.classify_document_state(
                row.get("document", ""), row.get("path", "") or "", row.get("document_id"),
            )
            for row in results
        ],
    )


def _apply_document_state_answer_gate(state_coverage, rendered):
    """Rewrite `rendered` only when it contradicts `state_coverage`; otherwise
    return it byte-identical.

    PR7.3: the gate is now a pure CONSUMER of one verdict. It receives neither
    `query` nor `results`, which is what structurally guarantees the properties
    PR7.3 required - it cannot parse the query, classify a filename, or do its
    own entity matching, because it has no access to any of those inputs. Its
    only remaining job is text: decide whether the rendered answer makes a claim
    the verdict forbids, and if so replace it.

    Verdict → claim policy:
      SIGNED_CONFIRMED / SIGNED_CONTRACT_CONFIRMED
                         a negative claim is false        → positive rewrite
      SIGNED_OTHER_DOCUMENT_CONFIRMED
                         a positive "podepsaná smlouva" claim is false
                         (only a signed LOI/order/… exists) → clarify rewrite
      UNSIGNED_CONFIRMED a positive claim is false        → negative rewrite
      ENTITY_MISMATCH    neither direction is supported   → "neověřeno"
      UNVERIFIED         the state cannot be ruled out    → "neověřeno" (hedge)
      NOOP               no signed intent                 → untouched

    Minimal intervention throughout: an answer that makes no claim about
    signedness at all is returned unchanged under EVERY verdict. Forcing a
    "neověřeno" sentence onto an answer that never mentioned a signature would
    destroy a useful answer to a question the gate was not asked to police.
    """
    from evidence_runtime import StateVerdict

    verdict = state_coverage.verdict
    if verdict is StateVerdict.NOOP:
        return rendered

    folded = _fold_plain(rendered)
    has_negative = bool(
        _STATE_GATE_NEGATIVE_SIGNED_RE.search(folded)
        or _STATE_GATE_GENERIC_NOT_FOUND_RE.search(folded)
    )
    has_positive = bool(_STATE_GATE_POSITIVE_SIGNED_RE.search(folded))
    docs = ", ".join(sorted({ev.document for ev in state_coverage.evidences}))

    if verdict in (StateVerdict.SIGNED_CONFIRMED, StateVerdict.SIGNED_CONTRACT_CONFIRMED):
        if not has_negative:
            return rendered
        return (
            f"Ano - na boxu je podepsaná smlouva. Nalezený podepsaný dokument: {docs}.\n"
            f"(Zdroj: {docs})"
        )

    if verdict is StateVerdict.SIGNED_OTHER_DOCUMENT_CONFIRMED:
        # PR7.6.1: signed LOI/order/minutes ≠ signed SoD. Always rewrite to the
        # clarifying sentence — a silent pass-through would leave an LLM claim
        # like "podepsaná smlouva existuje" standing on LOI evidence alone.
        kind_hint = docs or "jiný dokument"
        return (
            f"Nalezl jsem podepsaný dokument ({kind_hint}), ale ne potvrzenou "
            f"podepsanou smlouvu.\n(Zdroj: {docs})"
            if docs else
            "Nalezl jsem podepsaný dokument, ale ne potvrzenou podepsanou smlouvu."
        )

    if verdict is StateVerdict.UNSIGNED_CONFIRMED:
        # Every candidate is conclusively FOR_SIGNATURE/DRAFT/TEMPLATE, so a
        # negative claim is accurate and passes through untouched - only a false
        # POSITIVE ("je podepsaná") needs correcting.
        if has_positive and not has_negative:
            return (
                "Ne - na boxu nebyla nalezena podepsaná verze smlouvy; k dispozici je "
                f"pouze nepodepsaná verze.\n(Zdroj: {docs})"
            )
        return rendered

    # ENTITY_MISMATCH / UNVERIFIED: no definite claim in either direction may
    # stand. The two differ only in what may be cited, which `docs` already
    # encodes - ENTITY_MISMATCH and a partial-entity-match UNVERIFIED both carry
    # no citable evidence, because naming a document there would itself be the
    # fabricated confirmation/denial this gate exists to prevent (PR6.1).
    #
    # PR7.6.1: with *no* citable evidence, also block invented technical answers
    # that never mention signedness (FAT status-01: BOZP analysis from an
    # unrelated SoD). Genuine hedges ("není uvedeno" / "nelze ověřit") still
    # pass — minimal intervention for already-safe text.
    if not docs:
        already_hedged = any(
            marker in folded
            for marker in ("nelze", "neni uvedeno", "neover", "nenalezen")
        )
        if already_hedged and not (has_negative or has_positive):
            return rendered
        return (
            "Nelze jednoznačně ověřit stav podpisu smlouvy pro dotazovaný "
            "subjekt - nalezené dokumenty se k němu jednoznačně nevztahují."
        )
    if not (has_negative or has_positive):
        return rendered
    return (
        "Nelze jednoznačně ověřit, zda je smlouva podepsaná - nalezený dokument "
        f"nemá v názvu ani dostupných datech jednoznačný signál o stavu podpisu.\n"
        f"(Zdroj: {docs})"
    )

def _call_ollama(model, prompt, format_schema=None, timeout=240):
    payload={"model":model,"stream":False,"think":False,"prompt":prompt}
    if format_schema is not None: payload["format"]=format_schema
    req=urllib.request.Request(OLLAMA_ENDPOINT,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.loads(response.read())["response"]

# --- PR7.2: Evidence Runtime Validation as a DIAGNOSTIC layer -----------------
#
# WHY THIS EXISTS: the foundation layers (EvidenceSet PR3, IntentRequirement PR4,
# DocumentState W1, StateCoverage/needs PR7.0.1) are fully tested but nothing in
# the runtime ever ran them together over a real answer. This wiring makes their
# verdict OBSERVABLE on production queries before any of it is allowed to change
# an answer.
#
# It is strictly read-only: it renders no text, rewrites nothing, and adds one
# additive key to answer()'s return dict. `results` is read but never reordered,
# filtered, scored, or mutated - retrieval, RRF, rerank, bonuses, QE and the
# prompt are untouched by construction (no call into any of them below).
#
# The PR6 gate is deliberately NOT refactored to use build_state_coverage()
# (that is PR7.3). The two entity-matching rules differ - PR6 accepts any single
# entity token as a substring, PR7.0.1 requires every discriminative term - so
# for some queries `state_verdict` here and the gate's own outcome will disagree.
# Surfacing that disagreement on real queries is the point of this PR; resolving
# it is the next one.

def _validation_retrieval_rows(results):
    """Rows for build_evidence_set(), preferring PR7.1's chunk-level capture.

    `_evidence_spans` (ui_services.search_all, flag-gated) carries one entry per
    chunk that actually contributed to a document's merged quote, with real
    document_id/chunk_id. Facet evidence is then attributed per chunk instead of
    to one concatenated quote whose parts came from different chunks.

    Without the capture the merged rows are the only thing available. That is a
    coarser but still honest input - identity comes from the best chunk of the
    document and the quote is what the LLM was given - so the layer degrades
    instead of going silent. `span_source` records which one was used, because
    the two are not equally precise and a diagnostic must not hide that.
    """
    spans = [span for row in results for span in (row.get("_evidence_spans") or ())]
    if spans:
        return spans, "captured_spans"
    return list(results), "merged_rows"


def _answer_validation_metadata(query, results, gate_rewrote=False, state_coverage=None):
    """Build an AnswerValidation for one answered query and summarize it.

    `state_coverage` is passed in when the gate already computed it, so the same
    verdict the gate acted on is the one reported here (PR7.3 - previously these
    were two independent decisions that could disagree).

    Local imports keep the default-OFF path from paying any import cost - the
    same convention ui_services._multi_query_search_plan() uses for facets, and
    it matters here because ai_search is imported by the parsing worker
    subprocesses too.

    Returns a JSON-serializable dict of scalars/lists only: it travels into
    answer()'s result, which app.py and the benchmark harness both serialize.
    """
    from evidence import build_evidence_set
    from evidence_runtime import (
        AnswerValidation, GateAction, derive_evidence_needs, evaluate_evidence_safety,
    )
    from intent_requirements import build_evidence_coverage, derive_intent_requirement
    from query_facets import extract_facets

    facets = extract_facets(query)
    rows, span_source = _validation_retrieval_rows(results)
    evidence_set = build_evidence_set(query, retrieval_rows=rows, facets=facets)
    intent_coverage = build_evidence_coverage(
        derive_intent_requirement(query, facets),
        derive_evidence_needs(evidence_set.spans),
    )
    if state_coverage is None:
        state_coverage = _answer_state_coverage(query, results)
    evidence_safety = evaluate_evidence_safety(query, results)
    validation = AnswerValidation(
        query=query,
        facets=tuple(facets),
        evidence_set=evidence_set,
        intent_coverage=intent_coverage,
        state_coverage=state_coverage,
        evidence_safety=evidence_safety,
        # This layer never rewrites; a non-PASSTHROUGH value reports what the
        # gate did. Since PR7.3 the gate acts on this very StateCoverage, so the
        # direction named here is the one it actually applied, not a guess.
        gate_action=_validation_gate_action(state_coverage.verdict) if gate_rewrote else GateAction.PASSTHROUGH,
    )
    return _validation_summary(validation, span_source)


def _validation_gate_action(verdict):
    from evidence_runtime import GateAction, StateVerdict
    if verdict in (StateVerdict.SIGNED_CONFIRMED, StateVerdict.SIGNED_CONTRACT_CONFIRMED):
        return GateAction.REWRITTEN_POSITIVE
    if verdict is StateVerdict.UNSIGNED_CONFIRMED:
        return GateAction.REWRITTEN_NEGATIVE
    if verdict is StateVerdict.SIGNED_OTHER_DOCUMENT_CONFIRMED:
        return GateAction.REWRITTEN_UNVERIFIED
    return GateAction.REWRITTEN_UNVERIFIED


def _validation_summary(validation, span_source):
    """Flatten AnswerValidation into diagnostic metadata.

    Only aggregates and document names - never chunk text. This dict is attached
    to every answer when the flag is on, and duplicating quotes into it would
    grow the payload without telling a reader anything `citations` does not.
    """
    state = validation.state_coverage
    intent = validation.intent_coverage
    evidence_set = validation.evidence_set
    safety = validation.evidence_safety
    out = {
        # PR7.3.1: symmetric with the FAILED payload in answer(), so a consumer
        # can branch on one field instead of probing for an "error" key.
        "status": "OK",
        "rules_version": validation.source,
        "span_source": span_source,
        "evidence_spans": len(evidence_set.spans),
        "facet_join_status": evidence_set.join_status.value,
        "state_verdict": state.verdict.value,
        "state_entity_matched": state.entity_matched,
        "state_documents": [
            {"document": ev.document, "state": ev.state.value, "confidence": ev.confidence}
            for ev in state.evidences
        ],
        "intent_coverage": intent.status.value,
        "required_needs": [need.value for need in intent.required_needs],
        "satisfied_needs": [need.value for need in intent.satisfied_needs],
        "missing_needs": [need.value for need in intent.missing_needs],
        "gate_action": validation.gate_action.value,
    }
    if safety is not None:
        out["evidence_safety"] = safety.status.value
        out["conflicted_documents"] = list(safety.conflicted_documents)
    return out


def answer(query, results):
    if not results: return {"answer":"Odpověď nelze vytvořit bez citací.","citations":[],"confidence":"red"}
    # PR9.7.3: drawing-navigation queries are answered from ranked results
    # without Ollama. Non-drawing queries never enter this branch.
    drawing_results=results
    if drawing_navigation.is_drawing_navigation_query(query):
        drawing_results=drawing_local_evidence.enrich_results(
            query,results,drawing_navigation.derive_requested_subtypes(query)
        )
    drawing_text=drawing_navigation.try_render(query,drawing_results)
    if drawing_text is not None:
        confidence_level,confidence_reason=_answer_confidence(query,results)
        confidence_block=f"\n\nJistota odpovědi:\n{CONFIDENCE_LABELS[confidence_level]}\n- {confidence_reason}"
        return {"answer":drawing_text+confidence_block,"citations":drawing_results,"model":"drawing-navigation","confidence":confidence_level}
    # PR8.3: demote OLD/ rows from authoritative context on currency/status
    # queries. Flag OFF → answer_results IS results (same object, not just
    # equal) — byte-identical AND identity-identical to pre-PR8.3. search()
    # is untouched; this only chooses which rows the LLM / citations / state
    # gate may treat as current evidence. Historical OLD copies stay on an
    # additive `historical_citations` key.
    #
    # PR8.4.1: a copy (`list(...)`) is made ONLY where a real transformation
    # happens (guard actually demoted something) or where the transformed
    # result had to be discarded (error / empty). Every other path keeps
    # `results` itself so `final["citations"] is results` still holds for
    # callers that never trigger the guard — this was a latent PR8.3
    # regression (identity, not value, broke) surfaced by PR8.4 test runs.
    answer_results=results
    historical_citations=[]
    old_guard_trace=None
    if OLD_REVISION_GUARD_ENABLED:
        try:
            guard=old_revision_guard.apply_old_revision_guard(query,results)
            old_guard_trace=guard.as_trace_dict()
            if guard.historical_results:
                answer_results=list(guard.context_results)
                historical_citations=old_revision_guard.annotate_historical(guard.historical_results)
        except Exception as exc:
            _search_logger.warning("OLD_REVISION_GUARD_SKIPPED query=%r error=%r - using original results",query,exc)
            answer_results=results; historical_citations=[]; old_guard_trace={"activated":False,"reason":"error","error":repr(exc)}
    if not answer_results:
        answer_results=results; historical_citations=[]
    confidence_level,confidence_reason=_answer_confidence(query,answer_results)
    confidence_block=f"\n\nJistota odpovědi:\n{CONFIDENCE_LABELS[confidence_level]}\n- {confidence_reason}"
    # PR7.6.1: evidence-safety abstention (project conflict / weak lexical
    # overlap). Decision lives in evidence_runtime.evaluate_evidence_safety —
    # answer() only consumes it. Runs only when the validation flag is on so
    # the flag-OFF path stays byte-identical to pre-PR7.6.1. When it abstains
    # we skip the LLM entirely: a factual claim must not be invented from a
    # trap file or a generic-token near-miss.
    evidence_safety=None
    if EVIDENCE_RUNTIME_VALIDATION_ENABLED:
        try:
            from evidence_runtime import EvidenceSafetyStatus, evaluate_evidence_safety
            evidence_safety=evaluate_evidence_safety(query,answer_results)
            if evidence_safety.status in (
                EvidenceSafetyStatus.NO_EVIDENCE,
                EvidenceSafetyStatus.DOCUMENT_PROJECT_CONFLICT,
                EvidenceSafetyStatus.UNVERIFIED,
            ) and evidence_safety.message:
                # Never hand conflicted / irrelevant rows back as citable
                # evidence — the abstention text must stand alone.
                safe_citations=[]
                final={"answer":evidence_safety.message+confidence_block,"citations":safe_citations,"model":DEFAULT_MODEL,"confidence":confidence_level}
                try: final["validation"]=_answer_validation_metadata(query,answer_results,False,None)
                except Exception as exc:
                    _search_logger.warning("VALIDATION_FAILED query=%r error=%r - answer unaffected",query,exc)
                    final["validation"]={"error":f"{type(exc).__name__}: {exc}","source":"evidence_runtime","status":"FAILED"}
                return final
        except Exception as exc:
            _search_logger.warning("EVIDENCE_SAFETY_SKIPPED query=%r error=%r - continuing to LLM",query,exc)
            evidence_safety=None
    # PR9.3.3: pack ZDROJE after OLD guard + evidence gate, before the LLM.
    # Flag OFF → llm_results is answer_results (same object). Retrieval,
    # citations, document-state, and model routing stay on the full pool.
    llm_results=answer_results
    packed_debug=None
    if QUERY_FOCUSED_CONTEXT_PACKING_ENABLED:
        try:
            packed=context_packing.pack_answer_context(query,answer_results)
            if packed.rows:
                llm_results=packed.rows
                packed_debug=packed.as_debug_dict()
        except Exception as exc:
            _search_logger.warning("CONTEXT_PACKING_SKIPPED query=%r error=%r - using full answer_results",query,exc)
            llm_results=answer_results; packed_debug=None
    # PR9.3.4: entity/identifier candidates for the rows the LLM will actually
    # see. Flag OFF → hints_block is "" and the prompt below is byte-identical
    # to pre-PR9.3.4. The block only lists values already present in ZDROJE
    # with their 1-based index; it never states an answer, is not added to
    # `context` (so the PR8.4.6/PR9.2.1 free-text fallback prompt is untouched),
    # and does not reach citations, the renderer, or `answer_results`.
    hints_block=""
    hints_debug=None
    if ENTITY_HINTS_ENABLED:
        try:
            hints=entity_hints.build_entity_hints(query,llm_results)
            hints_block=hints.as_prompt_block()
            hints_debug=hints.as_debug_dict()
        except Exception as exc:
            _search_logger.warning("ENTITY_HINTS_SKIPPED query=%r error=%r - prompt unchanged",query,exc)
            hints_block=""; hints_debug=None
    checklist=_is_checklist_query(query)
    context="\n\n".join(f"[{i}] {r['document']}"+(f" (sekce: {r['heading']})" if r.get("heading") else "")+f" | projekt {r['project']}\n{r['quote']}" for i,r in enumerate(llm_results,1)); model=COMPLEX_MODEL if len(query)>180 or len(answer_results)>6 else DEFAULT_MODEL
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
    json_data=None
    try:
        raw=_call_ollama(model,f"{guidance}\n\nDOTAZ: {query}\n\nZDROJE:\n{context}{hints_block}",format_schema=schema)
        json_data=json.loads(raw)
        rendered=_render_structured_answer(json_data,llm_results) if checklist else _render_concise_answer(json_data,llm_results)
    except Exception:
        try:
            rendered=_apply_free_text_fallback(query,context,llm_results,checklist,model)
        except Exception as exc2:
            return {"answer":f"Ollama je nedostupná: {type(exc2).__name__}. Nalezené citace zůstávají k dispozici.","citations":answer_results,"model":model,"error":str(exc2),"confidence":confidence_level}
    else:
        # PR9.2.1: JSON parsed and rendered, but citation contract dropped
        # every substantive item (typically zdroj_index=0) → sentinel. That
        # is not an exception, so the branch above never ran. Flag OFF →
        # keep the sentinel (byte-identical to pre-PR9.2.1). Flag ON → the
        # same free-text fallback as the exception path, still gated by
        # PR8.4.6. Explicit abstention (no substantive items) stays sentinel.
        # A failed fallback call must NOT replace the JSON sentinel with
        # "Ollama je nedostupná" — JSON already produced a safe answer.
        if (JSON_SENTINEL_FALLBACK_ENABLED
                and rendered == _FALLBACK_SENTINEL
                and _json_payload_has_substantive_answer_item(json_data)):
            try:
                rendered=_apply_free_text_fallback(query,context,llm_results,checklist,model)
            except Exception as exc:
                _search_logger.warning("JSON_SENTINEL_FALLBACK_SKIPPED query=%r error=%r - keeping JSON sentinel",query,exc)
                rendered=_FALLBACK_SENTINEL
    # PR6/PR6.2: deterministic signed-contract safety gate - rewrites `rendered`
    # only if it violates the DocumentState rule; a no-op for every query
    # without a signed-contract intent (see _document_state_outcome). Runs
    # after both the structured and free-text fallback paths so either one is
    # covered. Feature-flagged (default OFF, consistent with the AUX pattern):
    # when disabled, answer() behaviour is byte-identical to pre-PR6.
    # Pre-gate text, kept only when BOTH flags are on, so the diagnostic layer
    # below can report whether the gate rewrote the answer.
    pre_gate=rendered if (DOCUMENT_STATE_GATE_ENABLED and EVIDENCE_RUNTIME_VALIDATION_ENABLED) else None
    # PR7.3: one state decision, computed here only for the gate. When only the
    # diagnostic flag is on it is computed inside _answer_validation_metadata()
    # instead, i.e. under that function's try/except.
    #
    # PR7.3.1: neither the state decision nor the gate may destroy an answer. On
    # failure `rendered` keeps its pre-gate value - exactly what the flag-OFF
    # path returns - so the user still gets their answer, and the gate
    # deliberately does NOT fire: an unknown state cannot justify rewriting a
    # claim. Logged like CROSS_ENCODER_FALLBACK above, so a swallowed failure
    # still leaves a trace when diagnostics are off.
    state_coverage=None
    if DOCUMENT_STATE_GATE_ENABLED:
        try:
            state_coverage=_answer_state_coverage(query,answer_results)
            rendered=_apply_document_state_answer_gate(state_coverage,rendered)
        except Exception as exc:
            _search_logger.warning("STATE_GATE_SKIPPED query=%r error=%r - answer passes through ungated",query,exc)
    final={"answer":rendered+confidence_block,"citations":answer_results,"model":model,"confidence":confidence_level}
    if historical_citations:
        final["historical_citations"]=historical_citations
    # PR7.2 diagnostics: additive `validation` key, nothing else. Never raises -
    # a failure in a read-only diagnostic must not cost the user their answer.
    # Reuses the gate's `state_coverage` when both flags are on, so the verdict
    # reported here is provably the one the gate acted on (PR7.3).
    if EVIDENCE_RUNTIME_VALIDATION_ENABLED:
        try: final["validation"]=_answer_validation_metadata(query,answer_results,pre_gate is not None and pre_gate!=rendered,state_coverage)
        except Exception as exc:
            _search_logger.warning("VALIDATION_FAILED query=%r error=%r - answer unaffected",query,exc)
            final["validation"]={"error":f"{type(exc).__name__}: {exc}","source":"evidence_runtime","status":"FAILED"}
        if old_guard_trace is not None and isinstance(final.get("validation"), dict):
            final["validation"]["old_revision_guard"]=old_guard_trace
        if packed_debug is not None and isinstance(final.get("validation"), dict):
            final["validation"]["context_packing"]=packed_debug
        if hints_debug is not None and isinstance(final.get("validation"), dict):
            final["validation"]["entity_hints"]=hints_debug
    if packed_debug is not None:
        final["_packed_context_debug"]=packed_debug
    if hints_debug is not None:
        final["_entity_hints_debug"]=hints_debug
    return final

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=BOX_ROOT); parser.add_argument("--state-dir",type=Path,default=STATE_DIR); sub=parser.add_subparsers(dest="command",required=True); sub.add_parser("index"); sp=sub.add_parser("search"); sp.add_argument("query"); ap=sub.add_parser("answer"); ap.add_argument("query"); args=parser.parse_args(); emb=Embeddings(); db=args.state_dir/"ai_search.sqlite3"; lance=args.state_dir/"lancedb"
    result=sync(args.root,db,lance,emb) if args.command=="index" else search(args.query,db,lance,emb); result=answer(args.query,result) if args.command=="answer" else result; print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
