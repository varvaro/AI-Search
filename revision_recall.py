"""PR8.2.1 — Revision-aware candidate recall (experimental, flag-gated).

Append-only: when the query has revision intent, pull additional chunk ids
for currency-marked / schedule / final documents into search()'s pre-Phase-3
pool. Does not reorder the existing pool and does not change scoring.

Uses read-only SQLite lookups on documents.name / relative_path (existing
tables). Does NOT alter the FTS index, embeddings, Lance, answer(), or safety.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from revision_ranking import _parse_dates, fold, query_has_revision_intent

# Cap how many *new* chunk ids we may append (one chunk per document).
REVISION_RECALL_MAX_NEW_IDS = 40


@dataclass(frozen=True)
class RevisionRecallResult:
    activated: bool
    reason: str
    added_ids: tuple[str, ...] = ()
    matched_document_names: tuple[str, ...] = ()

    def as_trace_dict(self) -> dict:
        return {
            "activated": self.activated,
            "reason": self.reason,
            "added_ids": list(self.added_ids),
            "added_count": len(self.added_ids),
            "matched_document_names": list(self.matched_document_names),
        }


def _connect_ro(db_path: Path | str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path)}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _is_old_path(name: str, relative_path: str) -> bool:
    rel = (relative_path or "").casefold().replace("\\", "/")
    nm = (name or "").casefold().replace("\\", "/")
    return (
        rel == "old"
        or rel.startswith("old/")
        or "/old/" in f"/{rel}/"
        or nm.startswith("old/")
        or "/old/" in f"/{nm}/"
    )


def _append_priority(query: str, name: str, relative_path: str) -> tuple:
    """Higher tuple sorts first — query-aware, then currency, then fresher date.

    This only decides *which* matching docs fit into max_new_ids. It does not
    reorder the baseline search() pool.
    """
    q = fold(query)
    n = fold(name or "")
    p = fold(relative_path or "").replace("\\", "/")
    blob = f"{n} {p}"
    score = 0
    wants_hmg = "hmg" in q or "harmonogram" in q
    wants_final = "final" in q or "sod" in q
    if wants_hmg and ("hmg" in blob or "harmonogram" in blob):
        score += 100
    if wants_final and ("/final/" in f"/{p}/" or n.endswith("final") or "_final" in n):
        score += 100
    if "akt_" in n or "_akt" in n:
        score += 40
    if "/final/" in f"/{p}/":
        score += 30
    if any(tag in n for tag in ("_r1_", "_r2_", "_r3_")):
        score += 10
    dates = _parse_dates(blob)
    latest = max(dates).toordinal() if dates else 0
    # Negate name for stable tie-break without reversing currency preference.
    return (score, latest, n)


def collect_revision_chunk_ids(
    db_path: Path | str,
    query: str,
    *,
    exclude_ids: Iterable[str] = (),
    max_new_ids: int = REVISION_RECALL_MAX_NEW_IDS,
) -> RevisionRecallResult:
    """Return chunk ids (ordinal 0) for revision-currency documents not yet pooled.

    Match rules (OR), excluding OLD/ folders:
      * name/path contains HMG or harmonogram
      * name contains akt_ / _akt (currency stamp)
      * path contains /final/
      * name contains R\\d_akt / _R\\d_ revision stamp
    """
    if not query_has_revision_intent(query):
        return RevisionRecallResult(activated=False, reason="no_revision_intent")

    exclude = set(exclude_ids or ())
    # Use instr() for literal '_' (SQLite LIKE treats '_' as a single-char wildcard).
    sql = """
        SELECT c.id, d.name, d.relative_path
        FROM documents d
        JOIN chunks c ON c.document_id = d.id AND c.ordinal = 0
        WHERE (
            instr(lower(d.name), 'hmg') > 0
            OR instr(lower(d.relative_path), 'hmg') > 0
            OR instr(lower(d.name), 'harmonogram') > 0
            OR instr(lower(d.relative_path), 'harmonogram') > 0
            OR instr(lower(d.name), 'akt_') > 0
            OR instr(lower(d.name), '_akt') > 0
            OR instr(lower(replace(d.relative_path, '\\', '/')), '/final/') > 0
            OR lower(replace(d.relative_path, '\\', '/')) LIKE '%/final'
            OR instr(lower(d.name), '_r1_') > 0
            OR instr(lower(d.name), '_r2_') > 0
            OR instr(lower(d.name), '_r3_') > 0
        )
        LIMIT ?
    """
    # Over-fetch, filter OLD/exclude, then query-aware priority (not alpha order).
    fetch_limit = max(max_new_ids * 20, 400)
    try:
        con = _connect_ro(db_path)
    except Exception as exc:
        return RevisionRecallResult(activated=False, reason=f"db_error:{type(exc).__name__}")

    try:
        rows = con.execute(sql, (fetch_limit,)).fetchall()
    except Exception as exc:
        return RevisionRecallResult(activated=False, reason=f"query_error:{type(exc).__name__}")
    finally:
        try:
            con.close()
        except Exception:
            pass

    candidates: list[tuple[str, str, str]] = []
    seen_docs: set[str] = set()
    for cid, name, rel in rows:
        if cid in exclude:
            continue
        if _is_old_path(name or "", rel or ""):
            continue
        key = (name or "").casefold()
        if key in seen_docs:
            continue
        seen_docs.add(key)
        candidates.append((cid, name or "", rel or ""))

    candidates.sort(
        key=lambda row: _append_priority(query, row[1], row[2]),
        reverse=True,
    )
    selected = candidates[: max(0, max_new_ids)]
    added = tuple(cid for cid, _, _ in selected)
    names = tuple(name for _, name, _ in selected)

    if not added:
        return RevisionRecallResult(
            activated=True, reason="no_new_matches", added_ids=(), matched_document_names=(),
        )
    return RevisionRecallResult(
        activated=True,
        reason="ok",
        added_ids=added,
        matched_document_names=names,
    )
