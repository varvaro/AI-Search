"""Tenká česká aplikační vrstva nad ověřeným backendem AI Search."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from email import policy
from email.parser import BytesParser
import base64, difflib, json, logging, os, re, shutil, sqlite3, subprocess, tempfile, time
from pathlib import Path
import ai_search
from ai_search_config import APP_SUPPORT_DIR

_logger = logging.getLogger("ai_search.ui_services")

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
    """Metadata jednoho výsledku. Nikdy nevyhazuje výjimku na chybu filesystemu.

    Index a filesystem se mezi dvěma běhy sync() rozcházejí - přejmenovaný nebo
    přesunutý soubor (typicky v Boxu) zmizí z disku, ale jeho chunky v indexu
    zůstanou. Nechráněný path.stat() zde propagoval FileNotFoundError až ven ze
    search_all(), takže jediný zastaralý řádek shodil celé vyhledávání.

    Klíč `availability` odlišuje dva stavy, které se nesmí zaměňovat:
      "available"   - stat() prošel, metadata jsou skutečná
      "missing"     - soubor potvrzeně neexistuje (FileNotFoundError)
      "unavailable" - cestu nelze přečíst z jiného důvodu (oprávnění,
                      odpojený svazek, I/O chyba). To NENÍ důkaz, že dokument
                      má zmizet z indexu - viz ochranu v ai_search.sync().
    """
    data={"source":source,"date":"","author":"","extension":path.suffix.lower().lstrip("."),"availability":"available"}
    try:
        stat=path.stat()
    except FileNotFoundError:
        data["availability"]="missing"; _logger.warning("METADATA: soubor neexistuje, vracím bezpečná metadata: %s",path); return data
    except OSError as exc:
        data["availability"]="unavailable"; _logger.warning("METADATA: soubor dočasně nedostupný (%s), vracím bezpečná metadata: %s",exc,path); return data
    data["date"]=datetime.fromtimestamp(stat.st_mtime).date().isoformat()
    if source=="E-mail" and path.suffix.lower()==".eml":
        try:
            mail=parse_eml(path); data.update({"title":mail["subject"],"author":mail["sender"],"email":mail})
        except (OSError, ValueError) as exc:
            data["availability"]="unavailable"; _logger.warning("METADATA: .eml se nepodařilo načíst (%s): %s",exc,path)
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

QA_CANDIDATE_POOL = 50  # candidate pool size for question-style queries before dedup/diversify narrows it back down

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
CONTENT_DUPLICATE_THRESHOLD = 0.90

def _normalize_chunk_text(text: str) -> str:
    """lowercase + strip punctuation + collapse whitespace, so near-identical
    boilerplate (e.g. the same status line repeated across many reports, only
    differing by trailing period/spacing) compares as equal."""
    folded=(text or "").casefold(); stripped=_PUNCTUATION_PATTERN.sub(" ",folded)
    return " ".join(stripped.split())

def _content_similarity(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None,a,b).ratio()

def deduplicate_by_content(rows: list[dict], threshold: float = CONTENT_DUPLICATE_THRESHOLD) -> list[dict]:
    """Collapse near-duplicate chunk text across DIFFERENT documents (not just
    repeated chunks within the same path, which search_all already merges by
    path). Root cause from the "Co chybí k předání základové desky" production
    audit: the same short status sentence ("základovou desku.") was copy-pasted
    across ~80 dated "Kontrolní den" reports, so each counted as a distinct
    "unique" result and filled the entire top-10 sent to the LLM. Assumes `rows`
    is already sorted by score descending, so the first (best-scored) member of
    each near-duplicate cluster is the one kept."""
    kept: list[dict] = []; kept_normalized: list[str] = []
    for row in rows:
        normalized=_normalize_chunk_text(row.get("quote",""))
        if normalized and any(_content_similarity(normalized,existing)>threshold for existing in kept_normalized):
            continue
        kept.append(row); kept_normalized.append(normalized)
    return kept

# --- Quote aggregation for multi-chunk documents (search_all's path-merge step) ---
#
# WHY THIS EXISTS: the "doklady po betonáži" production audit found that a
# document can be genuinely and correctly retrieved - present, top-ranked,
# surviving reranking/dedup/diversify unchanged - while the single quote
# string handed to ai_search.answer() still doesn't contain the passage that
# actually answers the query. Root cause: a document can win 10-20+ chunks
# into the RRF-merged list (e.g. "KZP - TEXTOVÁ ČÁST.pdf" had 21), and the OLD
# merge below appended chunk quotes strictly in relevance-score order until a
# fixed 1200-char cap was hit - so the two or three highest-scored chunks
# consumed the entire budget before a still-relevant chunk lower in that same
# document's own ranking (rank 4 in the audited case) ever got a chance. The
# document was found; the evidence for the LLM's answer was not.
#
# FIX: allocate the same 1200-char budget as a fixed number of QUOTA SLOTS
# (QUOTE_MERGE_MAX_CHUNKS) instead of first-come-first-served concatenation, so
# no single chunk - however well it scored - can consume the whole budget and
# starve out chunks ranked lower within the SAME document. This changes only
# the TEXT presented for an already-selected, already-ranked result; it does
# not touch scoring, ranking, candidate pool size, or which documents/rows
# make it into search_all()'s output - see search_all()'s call site below,
# which still sorts by the same `score` before AND after this aggregation.
QUOTE_MERGE_MAX_CHARS = 1200    # unchanged total budget from before this fix
QUOTE_MERGE_MAX_CHUNKS = 3      # GUARANTEED protected slots (phase 1); more distinct chunks may still be added in phase 2 if budget remains - see _merge_quote_chunks()
QUOTE_MERGE_SEPARATOR = " ... "  # unambiguous marker that two slots are non-adjacent excerpts, not one continuous sentence

def _select_quote_chunks(chunks: list[str], max_chunks: int = QUOTE_MERGE_MAX_CHUNKS, dedup_threshold: float = CONTENT_DUPLICATE_THRESHOLD) -> list[tuple[int, str]]:
    """Pick up to `max_chunks` distinct chunk texts out of `chunks` (already in
    the caller's relevance order - best chunk for this document first), skipping
    near-duplicates via the same _content_similarity() used by
    deduplicate_by_content() above, so two chunks that only differ by
    whitespace/punctuation don't both consume a quota slot. Returns
    (original_rank, text) pairs - the rank lets a caller show which of the
    document's own chunks (by relevance) contributed each excerpt."""
    selected: list[tuple[int, str]] = []; selected_normalized: list[str] = []
    for rank, text in enumerate(chunks):
        if len(selected) >= max_chunks: break
        if not text: continue
        normalized = _normalize_chunk_text(text)
        if not normalized: continue
        if any(_content_similarity(normalized, existing) > dedup_threshold for existing in selected_normalized): continue
        selected.append((rank, text)); selected_normalized.append(normalized)
    return selected

def _merge_quote_chunks(chunks: list[str], max_chars: int = QUOTE_MERGE_MAX_CHARS, max_chunks: int = QUOTE_MERGE_MAX_CHUNKS, separator: str = QUOTE_MERGE_SEPARATOR) -> tuple[str, list[dict]]:
    """Build the merged `quote` string for one document out of its chunks'
    already-truncated (<=700 char, see ai_search.py's search()) quote texts,
    in the SAME relevance order search_all() already sorts `output` by -
    ranking/scoring is fully decided before this function is ever called, it
    only decides which excerpts are shown and how much of each.

    Single-chunk documents are returned completely unchanged (just the
    original text, capped at `max_chars` as before this fix) - the quota
    logic below only applies once there is more than one chunk to fit,
    which is the only case this fix targets. For 2+ chunks, the first
    `max_chunks` distinct chunks get a protected, water-filled share of the
    budget (protection #1), and any budget they leave unused flows to
    further, lower-ranked distinct chunks beyond that quota (phase 2) - see
    the inline comments below for why both parts are needed.

    Returns (merged_quote, evidence) - `evidence` is `[]` for the single-chunk
    case (nothing new to report) and otherwise `[{"rank","text"}, ...]` for
    every excerpt that made it into the merged quote, in the order they
    appear in it. This is purely additive: `quote` remains the plain string
    every existing consumer (ai_search.answer(), context_excerpt(),
    metrics.row_matches_chunk_target()) already expects."""
    texts = [text for text in chunks if text]
    if not texts: return "", []
    if len(texts) == 1: return texts[0][:max_chars], []
    # Two-phase selection:
    #   Phase 1 ("guaranteed"): the first `max_chunks` distinct chunks (in
    #   relevance order) each get a protected, water-filled share of the
    #   budget - see the loop below. This is protection #1 ("jeden chunk
    #   nesmí vyplnit celý quote budget"): a chunk's cap here is fixed at the
    #   moment it's considered, as `remaining_budget // remaining_slots`
    #   scoped to this guaranteed group ONLY, so no single early chunk -
    #   however long - can claim more than its even share of it.
    #   Phase 2 ("extra"): real indexed chunks are very often shorter than
    #   an even share (e.g. a 350-char answer-bearing sentence against a
    #   400-600 char slot). If the guaranteed group leaves budget unused,
    #   that slack goes to FURTHER distinct, lower-ranked chunks beyond the
    #   `max_chunks` quota rather than being wasted - needed to keep total
    #   context size close to the pre-fix concatenation for documents with
    #   more than `max_chunks` short chunks (confirmed against the
    #   production benchmark: without phase 2, average context size shrank
    #   13-18% versus pre-fix for no relevance reason, outside this fix's
    #   own "context size roughly unchanged" acceptance bar). Phase 2 can
    #   only ever spend budget phase 1 left over - it cannot take budget
    #   away from a guaranteed slot, so protection #1 is unaffected by it.
    all_selected = _select_quote_chunks(texts, max_chunks=len(texts))
    guaranteed, extra = all_selected[:max_chunks], all_selected[max_chunks:]
    remaining_budget = max_chars; remaining_slots = len(guaranteed); evidence = []
    for rank, text in guaranteed:
        cap = max(1, remaining_budget // remaining_slots)
        piece = text[:cap]
        evidence.append({"rank": rank, "text": piece})
        remaining_budget -= len(piece); remaining_slots -= 1
    for rank, text in extra:
        if remaining_budget <= 0: break
        piece = text[:remaining_budget]
        if not piece: break
        evidence.append({"rank": rank, "text": piece}); remaining_budget -= len(piece)
    # Protection #3 (hard cap): per-chunk truncation alone does not bound the
    # JOINED string, since len(separator) * (n-1) is added on top of it -
    # e.g. 3 chunks at the full 400-char quota plus two 5-char " ... "
    # separators is 1210, not 1200. `evidence` above still reports each
    # excerpt's own untruncated-by-this-step text (useful for diagnostics);
    # only the final `quote` string is clipped to the caller-visible budget.
    quote = separator.join(item["text"] for item in evidence)[:max_chars]
    return quote, evidence

# --- Document-level evidence aggregation (search_all's path-merge step) ---
#
# WHY THIS EXISTS: search_all() collapses a document's many retrieved chunks
# into one result row and used to score that row with a pure MAX over its
# chunks - so "10 of my chunks matched this query" counted exactly the same as
# "1 of my chunks matched". The 2026-08-07 ranking audit measured the
# consequence on the real index: for the query "faktura Nazarenko stavební
# práce" the correct invoice had ALL 10 of its chunks in the 50-slot candidate
# pool (20% of the whole pool, the strongest evidence concentration of any
# document there) and was found by BM25 at rank 0, yet its document score was
# just its single best 179-char fragment's score, so it landed 15th of 27
# documents and never reached the top 10. Documents with one longer,
# semantically richer chunk won every slot. Scanned/tabular PDFs (invoices,
# protocols, KZP tables) are exactly the shape that fragments into many small
# chunks, so they were systematically under-scored.
#
# The bonus deliberately uses `_select_quote_chunks()` - the SAME
# within-document near-duplicate filter the quote merge already applies - so
# only DISTINCT supporting chunks can contribute. A document that repeats one
# boilerplate sentence across 50 chunks collapses to a single distinct chunk
# and therefore earns NO bonus at all: its score stays exactly the pre-fix
# MAX. That structural property, not the size of the weight, is what keeps
# this from becoming a "more chunks always wins" spam lever.
EVIDENCE_BONUS_WEIGHT = 0.25      # weight of each supporting chunk relative to the best chunk
EVIDENCE_BONUS_MAX_CHUNKS = 3     # how many supporting chunks may contribute at all (bounded so a 200-chunk document cannot outweigh relevance)

def _document_evidence_score(best_score: float, chunk_scores: list[float], chunk_quotes: list[str],
                            weight: float = EVIDENCE_BONUS_WEIGHT, max_chunks: int = EVIDENCE_BONUS_MAX_CHUNKS) -> float:
    """`best_score` plus a bounded bonus for up to `max_chunks` further
    DISTINCT chunks of the same document, each discounted by `weight`.

    `chunk_scores` and `chunk_quotes` are the document's own chunks in
    relevance order (best first) and must be index-aligned - search_all()
    appends to both under the same condition. Only `chunk_quotes` entries that
    survive `_select_quote_chunks()`'s near-duplicate filter can contribute,
    and the first survivor is skipped because it IS the best chunk already
    counted in `best_score` (never double-counted).

    Single-chunk documents and documents whose chunks are all near-duplicates
    of each other return `best_score` unchanged, i.e. exactly the pre-fix
    behaviour - the bonus is strictly additive evidence, never a penalty."""
    if len(chunk_quotes) < 2:
        return best_score
    distinct = _select_quote_chunks(chunk_quotes, max_chunks=max_chunks + 1)  # +1 = the best chunk itself, skipped below
    supporting = [chunk_scores[rank] for rank, _text in distinct[1:] if rank < len(chunk_scores)]
    return best_score + weight * sum(supporting)

MAX_RESULTS_PER_FOLDER = 2
MAX_RESULTS_PER_DOCUMENT_NAME = 3
PREFERRED_DOCUMENT_KEYWORDS = ("předávací","předání","technick","zápis","protokol","změnov")

def _folder_key(row: dict) -> str: return str(Path(row.get("path","")).parent)

def _is_preferred_document_type(row: dict) -> bool:
    haystack=f"{row.get('document','')} {row.get('heading','')}".casefold()
    return any(keyword in haystack for keyword in PREFERRED_DOCUMENT_KEYWORDS)

def diversify_results(rows: list[dict], max_per_folder: int = MAX_RESULTS_PER_FOLDER, max_per_document: int = MAX_RESULTS_PER_DOCUMENT_NAME) -> list[dict]:
    """Cap how many of the final results may come from the same folder or share
    the same document name, so one burst of similar reports (many dated site-
    meeting minutes in one folder) cannot crowd out other, differently-typed
    relevant documents (handover documentation, technical sheets, protocols).
    Preferred document types are admitted first so they win any contested slot.
    If not enough diverse candidates exist, the caps are relaxed rather than
    returning fewer results than available - matches the explicit requirement
    that the rule "lze překročit" when nothing else relevant remains. The
    caller's original score-sorted order is preserved in the output."""
    if not rows: return rows
    admission_order=sorted(rows,key=lambda row:0 if _is_preferred_document_type(row) else 1)
    folder_counts: dict[str,int] = {}; document_counts: dict[str,int] = {}; admitted_ids=set()
    for row in admission_order:
        folder=_folder_key(row); document=row.get("document","")
        if folder_counts.get(folder,0)<max_per_folder and document_counts.get(document,0)<max_per_document:
            admitted_ids.add(id(row)); folder_counts[folder]=folder_counts.get(folder,0)+1; document_counts[document]=document_counts.get(document,0)+1
    return [row for row in rows if id(row) in admitted_ids]+[row for row in rows if id(row) not in admitted_ids]

def search_all(query: str, settings: Settings, state_dir: Path, embeddings, is_question: bool = False, expand_query: bool | str = False) -> list[dict]:
    # `expand_query` is a pass-through to ai_search.search()'s Query Understanding
    # layer (see query_expansion.py). Default False = today's behaviour exactly;
    # app.py deliberately does not pass it yet, so the UI is unaffected until the
    # A/B benchmark justifies flipping it - the flag exists here so the benchmark
    # and a future settings toggle share one code path instead of two.
    # A question-style query leans much more heavily on the semantic channel and
    # is more exposed to near-duplicate boilerplate (see the "Co chybí k
    # předání základové desky" production audit) - pull a wider candidate pool
    # before dedup/diversification narrow it back down, instead of truncating
    # to result_count immediately like a plain document lookup does.
    candidate_pool=QA_CANDIDATE_POOL if is_question else settings.result_count
    # Vyžádej více chunků, aby po sloučení nebyl jeden dokument zobrazen opakovaně
    # a zůstalo dost unikátních výsledků pro postupné načítání. For QA queries this
    # must be at least ai_search.QA_RERANK_POOL_SIZE - search()'s own widened RRF
    # pool for is_question=True is otherwise truncated again by search()'s final
    # `[:limit]` before it ever reaches the dedup/diversify step below (production
    # audit: a genuinely relevant handover-document chunk ranked #226 of 300 after
    # RRF/rerank, well past a plain 50*4=200 cutoff).
    fetch_limit=max(50,candidate_pool*4,ai_search.QA_RERANK_POOL_SIZE) if is_question else max(50,candidate_pool*4)
    roots=[("Dokument",settings.project_root),("E-mail",settings.email_root),("Poznámka",settings.notes_root)]; output=[]
    for source,root in roots:
        db,lance=state_paths(state_dir,source)
        if not root or not db.exists(): continue
        for row in ai_search.search(query,db,lance,embeddings,fetch_limit,is_question=is_question,expand_query=expand_query):
            row.update(metadata_for(Path(row["path"]),source)); row["title"]=row.get("title",row["document"]); output.append(row)
    # Path-merge: one output row per document, carrying the BEST-scored chunk's
    # row as the base (so score/match/heading/title are unaffected by this
    # step - identical to before this fix) while its `quote` is built from
    # ALL of that document's chunks via quota-based _merge_quote_chunks()
    # instead of first-come-first-served concatenation (see that function's
    # docstring for why - the "doklady po betonáži" production bug this
    # replaces). `chunks_by_path` collects quotes in the same relevance order
    # `unique`'s winning row was picked in, since both loop over the same
    # score-descending `sorted(output, ...)` sequence.
    unique={}; chunks_by_path={}; scores_by_path={}
    for row in sorted(output,key=lambda item:item["score"],reverse=True):
        path=row["path"]
        if path not in unique:
            unique[path]=row; chunks_by_path[path]=[]; scores_by_path[path]=[]
        # Both lists are appended under the SAME condition, so index i of one
        # always describes the same chunk as index i of the other -
        # _document_evidence_score() relies on that alignment.
        if row.get("quote"): chunks_by_path[path].append(row["quote"]); scores_by_path[path].append(row["score"])
    for path,row in unique.items():
        quote,evidence=_merge_quote_chunks(chunks_by_path[path])
        row["quote"]=quote
        if evidence: row["evidence"]=evidence
        # Document-level evidence aggregation (see _document_evidence_score).
        # `best_chunk_score` preserves the pre-aggregation value the row was
        # selected by - kept for diagnostics/benchmark inspection, and it is
        # what `score` still equals whenever no bonus applies.
        row["best_chunk_score"]=row["score"]
        # Question-mode ranking is deliberately left byte-identical: a QA query
        # already pulls a 300-chunk rerank pool, so many documents arrive with
        # a large and very uneven number of chunks, and the audit that motivated
        # this bonus measured only document-lookup queries. Widening it to QA
        # mode is a separate change that needs its own benchmark A/B first.
        if not is_question:
            row["score"]=_document_evidence_score(row["score"],scores_by_path[path],chunks_by_path[path])
    # 2026-08-07 post-rerank truncation fix: `candidate_pool` above doubles as
    # BOTH the fetch_limit multiplier basis AND, for a non-question query,
    # settings.result_count itself (the final UI page size, =10 in
    # production) - so this cut used to shrink the per-document list to the
    # top 10 BEFORE deduplicate_by_content()/diversify_results() ever ran.
    # A document correctly found by BM25 at rank 0 and kept all the way
    # through RRF+rerank could still be discarded HERE, purely because some
    # other document's single best chunk scored marginally higher and pushed
    # the correct one to position 11+ - diversify never got a chance to
    # trade that other document out. Verified against 3 real production
    # queries on 2026-08-07: the correct document ranked 11th/16th/19th of
    # 27-38 unique documents by best-chunk score, discarded here despite
    # surviving retrieval+rerank untouched (see PR description / Phase 2
    # diagnostic report for the exact queries and ranks).
    #
    # Question-mode queries already used QA_CANDIDATE_POOL(=50) as
    # `candidate_pool` and are UNCHANGED by this fix: diversify_pool equals
    # candidate_pool exactly as before. For a non-question query,
    # diversify_pool now widens to that same QA_CANDIDATE_POOL instead of
    # the tight result_count page size - `unique` is already naturally
    # bounded by fetch_limit (<= ~50 chunks' worth of distinct paths for a
    # non-question query), so this is not a new fetch/retrieval, only a
    # later, less aggressive cut of data already in hand. The FINAL page
    # size handed to the caller is still exactly settings.result_count,
    # unchanged, on the last line below.
    diversify_pool=candidate_pool if is_question else max(candidate_pool,QA_CANDIDATE_POOL)
    candidates=sorted(unique.values(),key=lambda item:item["score"],reverse=True)[:diversify_pool]
    diversified=diversify_results(deduplicate_by_content(candidates))
    return diversified[:settings.result_count]

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

QUESTION_STARTERS = {"jaké","jaká","jaký","jací","jak","co","kdy","kde","proč","kolik","kdo","která","který","které","kterou","zda"}
QUESTION_VERBS = {"musí","musím","musíme","potřebuji","potřebujeme","potřebuje","chybí","zkontroluj","zkontrolujte","porovnej","porovnejte","shrň","shrňte","doložit","doložte","vysvětli","popiš"}
DEEP_ANALYSIS_KEYWORDS = ("jaké jsou požadavky","co chybí","zkontroluj","porovnej","shrň","všechny dokumenty","rizika","povinnosti")

def classify_query(query: str) -> dict:
    """Heuristically decide document-lookup vs. question mode and whether the query
    warrants the deeper/complex LLM. Runs on every search, so must stay instant
    (no LLM call) - a fixed keyword/pattern heuristic instead of a model call."""
    text=(query or "").strip(); folded=text.casefold(); words=folded.split()
    is_question=text.endswith("?") or (words and words[0] in QUESTION_STARTERS) or any(verb in words for verb in QUESTION_VERBS)
    deep=any(keyword in folded for keyword in DEEP_ANALYSIS_KEYWORDS)
    return {"mode":"otazka" if is_question else "dokument","deep":deep}

def match_label(score: float, best_score: float) -> str:
    ratio=score/best_score if best_score>0 else 0
    if ratio>=0.72: return "★★★★★ Vysoká shoda"
    if ratio>=0.38: return "★★★☆☆ Střední shoda"
    return "★☆☆☆☆ Nízká shoda"

def match_reason(row: dict) -> str:
    """Human-readable explanation of why a result was ranked here, built from the
    fts_hit/vector_hit/semantic_similarity breakdown produced by ai_search.search()."""
    match=row.get("match")
    if not match: return ""
    fts_hit=match.get("fts_hit"); vector_hit=match.get("vector_hit"); similarity=match.get("semantic_similarity",0.0) or 0.0
    if fts_hit and vector_hit: kind="🔀 kombinovaná shoda (klíčová slova i význam)"
    elif fts_hit: kind="🔤 přesná shoda (klíčová slova)"
    elif vector_hit: kind="🧠 významová shoda"
    else: kind="slabá shoda"
    parts=[kind]
    if vector_hit: parts.append(f"podobnost {similarity*100:.0f} %")
    if match.get("filename_match"): parts.append("shoda v názvu dokumentu")
    return " · ".join(parts)

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

PDFTOPPM_BIN = "/opt/homebrew/bin/pdftoppm"
PDF_PREVIEW_TIMEOUT_SECONDS = 20

def render_pdf_first_page(path: Path, runner=subprocess.run) -> bytes | None:
    """Rasterize the first page of a PDF to PNG bytes using poppler's pdftoppm -
    already a project dependency (used for OCR) - so no new dependency, fully
    local/offline, and renders reliably in Safari unlike the data:-URL iframe."""
    if not Path(PDFTOPPM_BIN).exists(): return None
    with tempfile.TemporaryDirectory(prefix="ai-search-preview-") as folder:
        prefix=Path(folder)/"page"
        try:
            result=runner([PDFTOPPM_BIN,"-png","-r","110","-f","1","-l","1",str(path),str(prefix)],capture_output=True,check=False,timeout=PDF_PREVIEW_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode!=0: return None
        rendered=sorted(Path(folder).glob("page*.png"))
        return rendered[0].read_bytes() if rendered else None

def preview_document(path: Path, fallback: str = "") -> dict:
    if not path.exists(): return {"kind":"error","content":"Soubor už neexistuje. Aktualizujte index."}
    suffix=path.suffix.lower()
    if suffix==".pdf" and path.stat().st_size<=12*1024*1024:
        image=render_pdf_first_page(path)
        if image: return {"kind":"image","content":base64.b64encode(image).decode("ascii")}
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
