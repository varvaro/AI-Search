"""Auxiliary Term Coverage Retrieval (PR5 / PR5.1).

Flag-gated candidate-generation helper. When enabled, builds ONE conjunctive
FTS5 MATCH from a rare query anchor + content constraints and returns chunk
ids to APPEND onto search()'s pre-rerank pool.

PR5.1 adds:
  * prefix safety (stem length + DF gate) for constraint wildcards
  * helpers for candidate_origin / aux_hit diagnostics (scoring unchanged)

Does NOT:
  * alter RRF math or channel lists
  * invent a parallel score / boost / floor
  * call vector search / Ollama / QE dictionaries
  * map queries to document ids
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# Keep in sync with ai_search.FTS_PREFIX_* (duplicated to avoid import cycles).
_FTS_PREFIX_MIN_LENGTH = 6
_FTS_PREFIX_STRIP = 2
# PR5.1 default; overridden from ai_search_config at call sites.
_DEFAULT_PREFIX_DF_MAX = 150

# Czech / query function words — never anchors or constraints.
_STOPWORDS = frozenset({
    "a", "i", "o", "u", "k", "s", "z", "v", "ve", "ke", "ku", "se", "si",
    "na", "do", "od", "za", "po", "pro", "pri", "při", "bez", "nad", "pod",
    "je", "jsou", "byt", "být", "by", "aby", "jak", "jaka", "jaká", "jake",
    "jaké", "jaky", "jaký", "co", "ci", "či", "nebo", "ale", "tak", "uz",
    "už", "jen", "take", "také", "toto", "tato", "tento", "tyto", "to",
    "ten", "ta", "the", "of", "and", "or", "in", "on", "for", "to", "with",
    "pozadovany", "požadovaný", "pozadovane", "požadované", "pozadovana",
    "požadovaná",
})

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

ORIGIN_PRIMARY = "PRIMARY"
ORIGIN_AUXILIARY = "AUXILIARY"
ORIGIN_BOTH = "BOTH"

DfLookup = Callable[[str], int]


@dataclass(frozen=True)
class AuxPlan:
    """Deterministic plan for one auxiliary MATCH (or a no-op)."""

    activated: bool
    match: str | None = None
    anchor: str | None = None
    constraints: tuple[str, ...] = ()
    dfs: dict[str, int] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class AuxResult:
    """Outcome of running the auxiliary FTS leg."""

    plan: AuxPlan
    added_ids: tuple[str, ...] = ()
    matched_ids: tuple[str, ...] = ()
    matched_count: int = 0

    def as_trace_dict(self) -> dict:
        return {
            "activated": self.plan.activated,
            "match": self.plan.match,
            "anchor": self.plan.anchor,
            "constraints": list(self.plan.constraints),
            "dfs": dict(self.plan.dfs),
            "reason": self.plan.reason,
            "added_ids": list(self.added_ids),
            "matched_ids": list(self.matched_ids),
            "matched_count": self.matched_count,
            "added_count": len(self.added_ids),
        }


def candidate_origin(*, primary: bool, aux_hit: bool) -> str:
    """Map primary-channel membership × aux FTS membership → origin label."""
    if primary and aux_hit:
        return ORIGIN_BOTH
    if aux_hit:
        return ORIGIN_AUXILIARY
    return ORIGIN_PRIMARY


def _tokenize(query: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(query or "") if len(t) > 1]


def _is_stopword(token: str) -> bool:
    return token.casefold() in _STOPWORDS


def _content_tokens(query: str) -> list[str]:
    """Ordered unique content tokens (case preserved from first occurrence)."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _tokenize(query):
        if _is_stopword(tok):
            continue
        if len(tok) < 3:
            continue
        key = tok.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return out


def _anchor_score(token: str, df: int) -> float:
    """Higher = better rare anchor. Acronyms beat slightly-rarer wrong surfaces."""
    idf = 1.0 / (1.0 + max(df, 0))
    acronym = 2.0 if token.isupper() and token.isalpha() and 3 <= len(token) <= 8 else 0.0
    length = min(len(token), 12) / 12.0
    return idf * 3.0 + acronym + 0.2 * length


def _escape_fts_token(token: str) -> str:
    return token.replace('"', "")


def _prefix_stem(safe: str) -> str | None:
    """Return a prefix stem of length >= 4, or None if unsafe / unnecessary."""
    if len(safe) < 4:
        return None
    if len(safe) >= _FTS_PREFIX_MIN_LENGTH:
        stem = safe[:-_FTS_PREFIX_STRIP]
    else:
        # len 4–5: drop one case ending only when the remaining stem stays >= 4
        # (blocks Ren* from Rent, svá* from svár).
        stem = safe[:-1]
    if len(stem) < 4 or stem == safe:
        return None
    return stem


def _constraint_clause(
    token: str,
    *,
    df_lookup: DfLookup | None = None,
    prefix_df_max: int = _DEFAULT_PREFIX_DF_MAX,
) -> list[str]:
    """Exact always; prefix* only with stem_len>=4 and DF(prefix*)<=max.

    Prefixes are only ever embedded inside ANCHOR AND (...); this helper never
    emits a standalone MATCH. High-DF stems (desk*, svar*-class) are rejected.
    """
    safe = _escape_fts_token(token)
    if not safe:
        return []
    parts = [f'"{safe}"']
    stem = _prefix_stem(safe)
    if not stem:
        return parts
    if df_lookup is not None:
        try:
            prefix_df = int(df_lookup(f"{stem}*"))
        except Exception:
            prefix_df = prefix_df_max + 1
        if prefix_df <= 0 or prefix_df > prefix_df_max:
            return parts
    parts.append(f"{stem}*")
    return parts


def plan_auxiliary_query(
    query: str,
    df_lookup: DfLookup,
    *,
    df_rare_max: int = 200,
    max_constraints: int = 3,
    prefix_df_max: int = _DEFAULT_PREFIX_DF_MAX,
) -> AuxPlan:
    """Build at most one ANCHOR AND (C OR C* …) plan from DF-gated terms."""
    tokens = _content_tokens(query)
    if len(tokens) < 2:
        return AuxPlan(activated=False, reason="need_two_content_tokens")

    dfs: dict[str, int] = {}
    for tok in tokens:
        try:
            dfs[tok] = int(df_lookup(tok))
        except Exception:
            dfs[tok] = 0

    # Drop tokens absent from the index — they cannot form a useful AND.
    present = [t for t in tokens if dfs.get(t, 0) > 0]
    if len(present) < 2:
        return AuxPlan(activated=False, dfs=dfs, reason="insufficient_indexed_tokens")

    rare = [t for t in present if dfs[t] <= df_rare_max]
    if not rare:
        return AuxPlan(activated=False, dfs=dfs, reason="no_rare_anchor")

    anchor = max(rare, key=lambda t: (_anchor_score(t, dfs[t]), -dfs[t], len(t)))
    constraints = [t for t in present if t.casefold() != anchor.casefold()]
    if not constraints:
        return AuxPlan(activated=False, dfs=dfs, reason="no_constraint")

    # Prefer rarer constraints first; cap width.
    constraints = sorted(constraints, key=lambda t: (dfs[t], len(t)))[:max_constraints]

    clause_parts: list[str] = []
    for c in constraints:
        clause_parts.extend(
            _constraint_clause(c, df_lookup=df_lookup, prefix_df_max=prefix_df_max)
        )
    if not clause_parts:
        return AuxPlan(activated=False, dfs=dfs, reason="empty_constraint_clause")

    safe_anchor = _escape_fts_token(anchor)
    match = f'"{safe_anchor}" AND ({" OR ".join(clause_parts)})'
    return AuxPlan(
        activated=True,
        match=match,
        anchor=anchor,
        constraints=tuple(constraints),
        dfs=dfs,
        reason="ok",
    )


def _connect_ro(db_path: Path | str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path)}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def make_fts_df_lookup(db_path: Path | str) -> DfLookup:
    """DF lookup for exact tokens (`"tok"`) or prefix patterns (`stem*`)."""

    def lookup(token: str) -> int:
        raw = (token or "").strip()
        if not raw:
            return 0
        if raw.endswith("*"):
            stem = _escape_fts_token(raw[:-1])
            if not stem:
                return 0
            expr = f"{stem}*"
        else:
            safe = _escape_fts_token(raw)
            if not safe:
                return 0
            expr = f'"{safe}"'
        try:
            with _connect_ro(db_path) as con:
                return int(
                    con.execute(
                        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                        (expr,),
                    ).fetchone()[0]
                )
        except sqlite3.Error:
            return 0

    return lookup


def run_auxiliary_fts(
    db_path: Path | str,
    match: str,
    *,
    limit: int = 25,
) -> list[str]:
    """Execute one FTS MATCH; return chunk_ids in BM25 rank order."""
    if not match or limit <= 0:
        return []
    try:
        with _connect_ro(db_path) as con:
            rows = con.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        return [cid for (cid,) in rows if cid]
    except sqlite3.Error:
        return []


def collect_auxiliary_chunk_ids(
    db_path: Path | str,
    query: str,
    *,
    exclude_ids: Iterable[str] = (),
    df_lookup: DfLookup | None = None,
    df_rare_max: int = 200,
    fts_limit: int = 25,
    max_new_ids: int = 15,
    prefix_df_max: int = _DEFAULT_PREFIX_DF_MAX,
) -> AuxResult:
    """Plan + run aux FTS; return matched ids and ids not already in exclude_ids."""
    lookup = df_lookup or make_fts_df_lookup(db_path)
    plan = plan_auxiliary_query(
        query,
        lookup,
        df_rare_max=df_rare_max,
        prefix_df_max=prefix_df_max,
    )
    if not plan.activated or not plan.match:
        return AuxResult(plan=plan)

    matched = run_auxiliary_fts(db_path, plan.match, limit=fts_limit)
    excluded = set(exclude_ids)
    added: list[str] = []
    for cid in matched:
        if cid in excluded:
            continue
        added.append(cid)
        if len(added) >= max_new_ids:
            break
    return AuxResult(
        plan=plan,
        added_ids=tuple(added),
        matched_ids=tuple(matched),
        matched_count=len(matched),
    )
