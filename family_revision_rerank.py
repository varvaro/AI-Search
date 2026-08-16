"""PR9.4.4 — Intent-gated BM25 admission helpers + family-local revision bonus.

Deterministic, no I/O, no project/vendor/filename hardcode. Dates come only
from metadata_rerank.parse_safe_dates (never revision_ranking._parse_dates).

Does NOT:
  * change FTS / Lance / embeddings / RRF / retrieval pool sizes
  * apply global recency, final/OLD/draft/signed preference
  * activate without FAMILY_REVISION_RERANK_ENABLED and revision intent
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from metadata_rerank import (
    _DATE_DMY_RE,
    _DATE_ISO_RE,
    _DATE_YMD_COMPACT_RE,
    _DATE_YMD_SHORT_RE,
    _SHORT_YEAR_MAX,
    _SHORT_YEAR_MIN,
    parse_safe_dates,
)

FAMILY_LATEST_BONUS = 0.03

_CURRENT_RE = re.compile(
    r"(?<![a-z0-9])(?:aktualni(?:e)?|nejnovejsi|platn[yae]|plati|platnost)(?![a-z0-9])"
)
_POSLEDNI_RE = re.compile(r"(?<![a-z0-9])posledni(?![a-z0-9])")
# Generic revision/document nouns — required next to bare "poslední".
_REVISION_NOUN_RE = re.compile(
    r"(?<![a-z0-9])(?:reviz\w*|verze|version|harmonogram|dokument\w*|vykres\w*)(?![a-z0-9])"
)
_REV_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    r"rev(?:ision)?"
    r"|r\d+"
    r"|akt(?:ual(?:izace|ni|ne|ny)*)?"
    r"|ver(?:sion|ze)?"
    r")(?![a-z0-9])"
)
# Revision marker immediately before a date span (akt_4.08.2026, rev-2026-08-04).
_REV_PREFIX_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:akt|rev(?:ision)?|ver(?:sion|ze)?)[_.\-]*$"
)
_EXT_RE = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|txt|msg)$")
_SEP_RE = re.compile(r"[_\-./,;:()\[\]{}]+")
_WS_RE = re.compile(r"\s+")
_GENERIC_PARENTS = frozenset({"old", "final", "new", "tmp", "temp", "files", "docs"})


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def has_revision_intent(query: str) -> bool:
    """True only for explicit current/revision wording.

    Bare ``poslední`` is not enough — a generic revision/document noun must
    also appear (``poslední harmonogram`` yes, ``poslední betonáž`` no).
    """
    blob = fold(query)
    if not blob:
        return False
    if _CURRENT_RE.search(blob):
        return True
    if _POSLEDNI_RE.search(blob) and _REVISION_NOUN_RE.search(blob):
        return True
    return False


def _strip_date_spans(blob: str) -> str:
    out = blob
    for rx in (_DATE_ISO_RE, _DATE_YMD_COMPACT_RE, _DATE_YMD_SHORT_RE, _DATE_DMY_RE):
        out = rx.sub(" ", out)
    return out


def family_key(filename: str, path: str = "") -> str:
    """Deterministic family identity from filename (parent only if generic)."""
    name = fold(Path(filename or path or "").name)
    name = _EXT_RE.sub("", name)
    name = _REV_TOKEN_RE.sub(" ", name)
    name = _strip_date_spans(name)
    name = _SEP_RE.sub(" ", name)
    name = _WS_RE.sub(" ", name).strip()
    tokens = [t for t in name.split() if t]
    if len(tokens) <= 2:
        parent = fold(Path(path or filename or ".").parent.name)
        parent = _REV_TOKEN_RE.sub(" ", parent)
        parent = _strip_date_spans(parent)
        parent = _SEP_RE.sub(" ", parent)
        parent = _WS_RE.sub(" ", parent).strip()
        if parent and parent not in _GENERIC_PARENTS:
            tokens = tokens + [f"p:{parent}"]
    return " ".join(tokens)


def _valid_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _safe_date_spans(filename: str) -> list[tuple[date, int, int]]:
    """Spans that parse_safe_dates would accept, with offsets in the folded name."""
    blob = fold(filename or "")
    seen: set[date] = set()
    spans: list[tuple[date, int, int]] = []

    def add(parsed: date | None, start: int, end: int) -> None:
        if parsed is None or parsed in seen:
            return
        seen.add(parsed)
        spans.append((parsed, start, end))

    for m in _DATE_ISO_RE.finditer(blob):
        add(_valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.start(), m.end())
    for m in _DATE_YMD_COMPACT_RE.finditer(blob):
        add(_valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.start(), m.end())
    for m in _DATE_YMD_SHORT_RE.finditer(blob):
        yy = int(m.group(1))
        if _SHORT_YEAR_MIN <= yy <= _SHORT_YEAR_MAX:
            add(_valid_date(2000 + yy, int(m.group(2)), int(m.group(3))), m.start(), m.end())
    for m in _DATE_DMY_RE.finditer(blob):
        day, month, year_raw = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(year_raw) == 2:
            year = int(year_raw)
            if not (_SHORT_YEAR_MIN <= year <= _SHORT_YEAR_MAX):
                continue
            year += 2000
        else:
            year = int(year_raw)
        add(_valid_date(year, month, day), m.start(), m.end())
    return spans


def revision_date(filename: str) -> date | None:
    """Single revision date for a filename, or None if missing/ambiguous.

    ``parse_safe_dates`` is the whitelist. When several safe dates exist, a
    date immediately after a generic revision marker (akt_/rev_/ver_) is the
    revision stamp. Several such stamps → newest among *those* only (not a
    global max over unattached dates). Unattached multiples → None.
    """
    accepted = set(parse_safe_dates(filename or ""))
    if not accepted:
        return None
    if len(accepted) == 1:
        return next(iter(accepted))

    blob = fold(filename or "")
    attached: list[date] = []
    for parsed, start, _end in _safe_date_spans(filename or ""):
        if parsed not in accepted:
            continue
        if _REV_PREFIX_RE.search(blob[:start]):
            attached.append(parsed)
    unique_attached = set(attached)
    if len(unique_attached) == 1:
        return next(iter(unique_attached))
    if len(unique_attached) > 1:
        return max(unique_attached)
    return None


def select_bm25_floor_chunk_ids(
    fts_ids: list[str],
    top_ids: list[str],
    doc_id_by_chunk: dict[str, object],
) -> list[str]:
    """Append-only: one BM25 chunk per document already retrieved but not in top_ids.

    Does not change ``fts_ids`` or fusion order. Bounded by the existing BM25 list.
    """
    present = set(top_ids)
    admitted_docs = {
        doc_id_by_chunk[cid]
        for cid in top_ids
        if doc_id_by_chunk.get(cid) is not None
    }
    extras: list[str] = []
    for cid in fts_ids:
        if cid in present:
            continue
        doc_id = doc_id_by_chunk.get(cid)
        if doc_id is None or doc_id in admitted_docs:
            continue
        extras.append(cid)
        admitted_docs.add(doc_id)
        present.add(cid)
    return extras


@dataclass(frozen=True)
class FamilyRevisionDetail:
    family_key: str
    revision_date: date | None
    is_latest: bool
    bonus: float
    reason: str

    def as_trace_dict(self) -> dict:
        return {
            "family_key": self.family_key,
            "revision_date": self.revision_date.isoformat() if self.revision_date else None,
            "is_latest": self.is_latest,
            "bonus": self.bonus,
            "reason": self.reason,
        }


def _doc_key(document: str, path: str) -> str:
    return path or document or ""


def annotate_family_revision(
    rows: list[dict],
    query: str,
) -> list[FamilyRevisionDetail]:
    """Per-row family-local latest bonus. Intent-gated; singleton/ambiguous → 0."""
    empty = FamilyRevisionDetail("", None, False, 0.0, "no_intent")
    if not has_revision_intent(query):
        return [empty for _ in rows]

    per_doc: dict[str, dict] = {}
    keys: list[str] = []
    for row in rows:
        name = row.get("document") or ""
        path = row.get("path") or ""
        dk = _doc_key(name, path)
        keys.append(dk)
        if dk not in per_doc:
            per_doc[dk] = {
                "family": family_key(name, path),
                "date": revision_date(name),
            }

    by_family: dict[str, list[str]] = {}
    for dk, info in per_doc.items():
        by_family.setdefault(info["family"], []).append(dk)

    latest_of: dict[str, str] = {}
    reason_of: dict[str, str] = {}
    for fam, members in by_family.items():
        if len(members) < 2:
            for dk in members:
                reason_of[dk] = "singleton"
            continue
        dated = {dk: per_doc[dk]["date"] for dk in members if per_doc[dk]["date"] is not None}
        if not dated:
            for dk in members:
                reason_of[dk] = "no_date"
            continue
        newest = max(dated.values())
        winners = [dk for dk, d in dated.items() if d == newest]
        if len(winners) != 1:
            for dk in members:
                reason_of[dk] = "ambiguous"
            continue
        latest_of[fam] = winners[0]
        for dk in members:
            reason_of[dk] = "latest" if dk == winners[0] else "older_or_undated"

    details: list[FamilyRevisionDetail] = []
    for row, dk in zip(rows, keys):
        info = per_doc[dk]
        is_latest = latest_of.get(info["family"]) == dk
        bonus = FAMILY_LATEST_BONUS if is_latest else 0.0
        details.append(
            FamilyRevisionDetail(
                family_key=info["family"],
                revision_date=info["date"],
                is_latest=is_latest,
                bonus=bonus,
                reason=reason_of.get(dk, "ok"),
            )
        )
    return details
