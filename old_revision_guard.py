"""PR8.3 — OLD revision safety guard (experimental, flag-gated).

Prevents OLD/ folder documents from being used as *authoritative* answer
evidence on currency / status queries, while keeping them available as
historical context when the user asks for history/revision/comparison, or
when no non-OLD alternative exists in the pool.

Does NOT:
  * change search() / FTS / Lance / embeddings / ranking
  * reorder the retrieval pool for search consumers
  * invent new answers — only chooses which rows answer() may treat as
    authoritative context/citations
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_OLD_SEGMENT_RE = re.compile(r"(?:^|/)(?:old)(?:/|$)", re.IGNORECASE)

# Queries that must NOT treat OLD paths as current/authoritative evidence.
# Bare "aktual"/"final"/"existuje" are omitted — they match aktualizace / final/
# filenames / "existuje výkres" (nds-status-03, OLD-only drawing).
_CURRENCY_AUTHORITY_PATTERNS = (
    "aktualni", "aktualne",
    "platny", "platna", "platne", "plati", "platnost",
    "posledni",
    "finalni verze", "finalni",
    "podepsan",
    "dodavatel",
    "existuje smlouv", "existuje sod",
    "je smlouva", "je sod",
    "stav smlouvy",
)

# Queries that explicitly want historical / revision material — OLD is OK.
_HISTORY_ALLOW_PATTERNS = (
    "historie", "historick",
    "revize", "revizi", "revizni",
    "porovnani", "porovnej", "porovnat",
    "starsi", "stary", "stare",
    "archiv", "archivni",
    "predchozi verze", "puvodni verze",
    "old/", "slozka old",
)


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def path_is_old(path: str | None, document: str | None = None) -> bool:
    blob = f"{path or ''} {document or ''}".replace("\\", "/")
    return bool(_OLD_SEGMENT_RE.search(blob))


def query_allows_old_history(query: str) -> bool:
    q = fold(query)
    return any(p in q for p in _HISTORY_ALLOW_PATTERNS)


def query_forbids_old_authority(query: str) -> bool:
    """True when the query asks for current/status/supplier/signed facts."""
    if query_allows_old_history(query):
        return False
    q = fold(query)
    return any(p in q for p in _CURRENCY_AUTHORITY_PATTERNS)


def _basename(row: dict) -> str:
    name = str(row.get("document") or "")
    # Strip a trailing extension for same-entity matching across .pdf/.docx.
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name)
    return fold(stem)


def has_non_old_same_entity(old_row: dict, pool: list[dict]) -> bool:
    """True if a non-OLD row shares the same document stem (entity copy)."""
    target = _basename(old_row)
    if not target:
        return False
    for row in pool:
        if path_is_old(row.get("path"), row.get("document")):
            continue
        other = _basename(row)
        if not other:
            continue
        if other == target or target in other or other in target:
            return True
    return False


@dataclass(frozen=True)
class OldRevisionGuardResult:
    activated: bool
    reason: str
    context_results: tuple[dict, ...]
    historical_results: tuple[dict, ...]
    downgraded_documents: tuple[str, ...] = ()

    def as_trace_dict(self) -> dict:
        return {
            "activated": self.activated,
            "reason": self.reason,
            "context_count": len(self.context_results),
            "historical_count": len(self.historical_results),
            "downgraded_documents": list(self.downgraded_documents),
        }


def apply_old_revision_guard(
    query: str, results: list[dict] | None,
) -> OldRevisionGuardResult:
    """Split retrieval rows into authoritative context vs historical OLD.

    Policy:
      * history/revize/porovnání → no change (OLD may stay authoritative)
      * currency/status/dodavatel/podepsaná + any non-OLD in pool →
        OLD rows leave context, kept only as historical_results
      * soft (other queries): downgrade an OLD row only when a non-OLD
        same-entity copy exists in the pool
      * if the pool is OLD-only under a forbid query → keep OLD in context
        (existence-only corpora like D.1.2.07 in OLD/)
    """
    rows = list(results or [])
    if not rows:
        return OldRevisionGuardResult(
            activated=False, reason="empty", context_results=(), historical_results=(),
        )

    if query_allows_old_history(query):
        return OldRevisionGuardResult(
            activated=True,
            reason="history_allowed",
            context_results=tuple(rows),
            historical_results=(),
        )

    old_rows = [r for r in rows if path_is_old(r.get("path"), r.get("document"))]
    non_old = [r for r in rows if not path_is_old(r.get("path"), r.get("document"))]

    if query_forbids_old_authority(query):
        if non_old and old_rows:
            return OldRevisionGuardResult(
                activated=True,
                reason="currency_forbid_downgrade",
                context_results=tuple(non_old),
                historical_results=tuple(old_rows),
                downgraded_documents=tuple(
                    str(r.get("document") or "") for r in old_rows if r.get("document")
                ),
            )
        return OldRevisionGuardResult(
            activated=True,
            reason="currency_forbid_keep_only_old" if old_rows and not non_old else "currency_forbid_noop",
            context_results=tuple(rows),
            historical_results=(),
        )

    # Soft path: only demote OLD copies that have a non-OLD twin.
    context: list[dict] = []
    historical: list[dict] = []
    downgraded: list[str] = []
    for row in rows:
        if path_is_old(row.get("path"), row.get("document")) and has_non_old_same_entity(row, rows):
            historical.append(row)
            if row.get("document"):
                downgraded.append(str(row["document"]))
        else:
            context.append(row)
    if not historical:
        return OldRevisionGuardResult(
            activated=True,
            reason="soft_noop",
            context_results=tuple(rows),
            historical_results=(),
        )
    return OldRevisionGuardResult(
        activated=True,
        reason="soft_same_entity_downgrade",
        context_results=tuple(context),
        historical_results=tuple(historical),
        downgraded_documents=tuple(downgraded),
    )


def annotate_historical(rows: list[dict] | tuple[dict, ...]) -> list[dict]:
    """Shallow-copy rows with a historical marker (does not mutate inputs)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        copy = dict(row)
        match = dict(copy.get("match") or {})
        match["historical_old"] = True
        copy["match"] = match
        out.append(copy)
    return out
