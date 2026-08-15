"""PR9.3.3 — query-focused pre-LLM context packing.

Pure, deterministic selection of a small evidence subset from an already-
retrieved answer pool. Does not call search, Ollama, SQLite, or Lance, and
does not mutate input rows or their scores.

    pack_answer_context(query, results, max_rows=4) -> PackedContext

Flag wiring lives in ai_search.answer(); this module only ranks and picks.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

import answer_intent
from query_facets import FacetType, extract_facets

MAX_ROWS = 4
MIN_ROWS = 2

# Question / function words. Content tokens (typ, konstrukce, dodavatel, …)
# stay; IDF across the candidate pool down-weights those that appear in
# every row.
_STOPWORDS = frozenset({
    "jak", "jaky", "jaka", "jake", "jakou", "jsou", "je", "byt",
    "na", "pro", "se", "si", "do", "od", "za", "po", "pri", "ke", "k",
    "a", "i", "u", "o", "ve", "v", "z", "s",
    "podle", "navrzen", "navrzena", "navrzeno", "pouzit", "pouzita",
    "existuje", "ma", "maji", "muze", "mohou",
    "ktere", "ktery", "ktera", "techto", "teto", "tohoto",
    "the", "of", "and", "or", "to",
})

_IDENTIFIER_RES = (
    re.compile(r"\bnot\d{4,}\b"),
    re.compile(r"\btp\s*\d+\b"),
    re.compile(r"\bcbs\s*\d+\b"),
    re.compile(r"\bcsn(?:\s*en)?\s*[\d.]+"),
    re.compile(r"(?<![a-z])en\s*[\d.]+"),
    re.compile(r"\bd(?:\.\d+){1,}[a-z0-9.]*"),
    re.compile(r"\bdn\s*\d+\b"),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def fold(text: str) -> str:
    """Casefold + strip combining marks. Digits and ASCII identifiers survive."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def extract_identifiers(text: str) -> tuple[str, ...]:
    """Stable identifier keys: spaces removed, digits kept (cbs02, not250039)."""
    folded = fold(text)
    found: list[str] = []
    seen: set[str] = set()
    for rx in _IDENTIFIER_RES:
        for match in rx.finditer(folded):
            key = re.sub(r"\s+", "", match.group(0))
            if key and key not in seen:
                seen.add(key)
                found.append(key)
    return tuple(found)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(fold(text)))


def _content_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(t for t in tokens if t not in _STOPWORDS and len(t) >= 2)


def _bigrams(tokens: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(zip(tokens, tokens[1:]))


def _intent(query_folded: str) -> dict[str, bool]:
    """PR9.3.4.1: the rules moved to answer_intent.PACKING_PROFILE unchanged.

    That profile is frozen precisely because these four flags drive the score
    boosts below; widening them would re-rank the packed context the PR9.3.3
    A/B validated. See answer_intent's module docstring.
    """
    return answer_intent.packing_flags(query_folded)


def _doc_key(row: dict) -> str:
    path = str(row.get("path") or "").strip()
    if path:
        return path
    return str(row.get("document") or "")


def _prefix_hit(needle: str, hay_tokens: set[str]) -> bool:
    if needle in hay_tokens:
        return True
    if len(needle) < 5:
        return False
    stem = needle[:5]
    return any(tok.startswith(stem) or stem.startswith(tok[:5]) for tok in hay_tokens if len(tok) >= 5)


@dataclass(frozen=True)
class PackedContext:
    rows: list
    original_count: int
    packed_count: int
    selected_original_ranks: tuple[int, ...]
    scores: tuple[float, ...]
    reasons: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    def as_debug_dict(self) -> dict:
        return {
            "original_count": self.original_count,
            "packed_count": self.packed_count,
            "selected_original_ranks": list(self.selected_original_ranks),
            "scores": list(self.scores),
            "reasons": [list(r) for r in self.reasons],
            "documents": [str((row or {}).get("document") or "") for row in self.rows],
        }


def pack_answer_context(query: str, results, max_rows: int = MAX_ROWS) -> PackedContext:
    """Return up to `max_rows` original row objects, never mutating them.

    Always keeps at least min(MIN_ROWS, len(results)) rows so multi-document
    queries are not collapsed to a single source.
    """
    rows = list(results or [])
    original_count = len(rows)
    if original_count == 0:
        return PackedContext([], 0, 0, (), ())
    cap = max(1, min(int(max_rows or MAX_ROWS), MAX_ROWS))
    if original_count <= cap and original_count <= MIN_ROWS:
        ranks = tuple(range(1, original_count + 1))
        return PackedContext(rows, original_count, original_count, ranks, (0.0,) * original_count)

    query_folded = fold(query)
    q_tokens = _content_tokens(tokenize(query))
    q_bigrams = _bigrams(q_tokens)
    q_ids = set(extract_identifiers(query))
    intent = _intent(query_folded)
    facets = []
    try:
        facets = extract_facets(query or "")
    except Exception:
        facets = []
    facet_terms = tuple(
        fold(term)
        for facet in facets
        for term in (facet.surface, *facet.terms)
        if fold(term)
    )
    actor_terms = tuple(
        fold(term)
        for facet in facets
        if facet.type == FacetType.ACTOR
        for term in (facet.surface, *facet.terms)
        if fold(term)
    )
    object_terms = tuple(
        fold(term)
        for facet in facets
        if facet.type in (FacetType.OBJECT, FacetType.DOC_TYPE)
        for term in (facet.surface, *facet.terms)
        if fold(term)
    )

    fields = []
    for row in rows:
        quote = fold(str(row.get("quote") or ""))
        heading = fold(str(row.get("heading") or ""))
        name = fold(str(row.get("document") or ""))
        path = fold(str(row.get("path") or "") + " " + str(row.get("relative_path") or ""))
        quote_toks = set(tokenize(quote))
        heading_toks = set(tokenize(heading))
        name_toks = set(tokenize(name))
        path_toks = set(tokenize(path))
        fields.append({
            "quote": quote,
            "heading": heading,
            "name": name,
            "path": path,
            "quote_toks": quote_toks,
            "heading_toks": heading_toks,
            "name_toks": name_toks,
            "path_toks": path_toks,
            "all_toks": quote_toks | heading_toks | name_toks | path_toks,
            "ids_quote": set(extract_identifiers(quote)),
            "ids_heading": set(extract_identifiers(heading)),
            "ids_name": set(extract_identifiers(name)),
            "ids_path": set(extract_identifiers(path)),
        })

    n = max(len(fields), 1)

    def _idf(df_count: int) -> float:
        return math.log((n + 1) / (df_count + 1)) + 1.0

    df_all: dict[str, int] = {}
    df_quote: dict[str, int] = {}
    df_struct: dict[str, int] = {}
    for token in q_tokens:
        df_all[token] = sum(1 for f in fields if _prefix_hit(token, f["all_toks"]))
        df_quote[token] = sum(1 for f in fields if _prefix_hit(token, f["quote_toks"]))
        df_struct[token] = sum(
            1 for f in fields
            if _prefix_hit(token, f["heading_toks"] | f["name_toks"] | f["path_toks"])
        )
    bigram_df: dict[tuple[str, str], int] = {}
    for left, right in q_bigrams:
        needle = f"{left} {right}"
        bigram_df[(left, right)] = sum(
            1 for f in fields
            if needle in f["quote"] or needle in f["heading"] or needle in f["name"]
            or (_prefix_hit(left, f["all_toks"]) and _prefix_hit(right, f["all_toks"]))
        )

    scored: list[tuple[float, int, tuple[str, ...]]] = []
    for index, spec in enumerate(fields):
        score = 0.0
        reasons: list[str] = []
        name_heading_path_boost = 1.6 if intent["who"] else 1.0
        id_boost = 1.5 if (intent["identifier"] or intent["standard"]) else 1.0
        type_boost = 1.4 if intent["type"] else 1.0
        struct_toks = spec["heading_toks"] | spec["name_toks"] | spec["path_toks"]

        for token in q_tokens:
            quote_w = _idf(df_quote.get(token, 0))
            struct_w = _idf(df_struct.get(token, 0))
            all_w = _idf(df_all.get(token, 0))
            if _prefix_hit(token, spec["quote_toks"]):
                score += 0.8 * quote_w
                reasons.append(f"quote:{token}")
            if _prefix_hit(token, spec["heading_toks"]):
                score += 5.0 * struct_w * name_heading_path_boost * type_boost
                reasons.append(f"heading:{token}")
            if _prefix_hit(token, spec["name_toks"]):
                score += 5.0 * struct_w * name_heading_path_boost
                reasons.append(f"name:{token}")
            if _prefix_hit(token, spec["path_toks"]):
                score += 4.0 * struct_w * name_heading_path_boost
                reasons.append(f"path:{token}")
            if _prefix_hit(token, spec["all_toks"]):
                score += 0.6 * all_w

        for left, right in q_bigrams:
            needle = f"{left} {right}"
            bw = _idf(bigram_df.get((left, right), 0))
            quote_hit = needle in spec["quote"] or (
                _prefix_hit(left, spec["quote_toks"]) and _prefix_hit(right, spec["quote_toks"])
            )
            heading_hit = needle in spec["heading"] or (
                _prefix_hit(left, spec["heading_toks"]) and _prefix_hit(right, spec["heading_toks"])
            )
            name_hit = needle in spec["name"] or (
                _prefix_hit(left, spec["name_toks"]) and _prefix_hit(right, spec["name_toks"])
            )
            if quote_hit:
                score += 3.5 * type_boost * bw
                reasons.append(f"quote_bigram:{left}_{right}")
            if heading_hit:
                score += 6.5 * type_boost * bw
                reasons.append(f"heading_bigram:{left}_{right}")
            if name_hit:
                score += 6.0 * bw
                reasons.append(f"name_bigram:{left}_{right}")

        # Distinctive query terms in heading/name/path beat quote-only overlap.
        struct_hits = sum(1 for token in q_tokens if _prefix_hit(token, struct_toks))
        if struct_hits:
            score += 2.0 * struct_hits * name_heading_path_boost
            reasons.append(f"struct_hits:{struct_hits}")

        for ident in q_ids:
            if ident in spec["ids_quote"]:
                score += 6.0 * id_boost
                reasons.append(f"id_quote:{ident}")
            if ident in spec["ids_heading"]:
                score += 7.0 * id_boost
                reasons.append(f"id_heading:{ident}")
            if ident in spec["ids_name"]:
                score += 8.0 * id_boost
                reasons.append(f"id_name:{ident}")
            if ident in spec["ids_path"]:
                score += 7.0 * id_boost
                reasons.append(f"id_path:{ident}")

        # Identifiers present on the row even when the query only *asks* for
        # numbers (zakázka / NOT / smlouva) — filename/heading still count.
        if intent["identifier"] or intent["standard"]:
            row_ids = spec["ids_name"] | spec["ids_heading"] | spec["ids_path"] | spec["ids_quote"]
            if row_ids:
                score += 3.0 * id_boost * min(len(row_ids), 3)
                reasons.append("row_identifiers")

        for term in actor_terms:
            if term and (term in spec["name"] or term in spec["heading"] or term in spec["path"] or term in spec["quote"]):
                score += 3.0 * name_heading_path_boost
                reasons.append(f"actor:{term}")
        for term in object_terms:
            if term and (term in spec["quote"] or term in spec["heading"] or term in spec["name"]):
                score += 1.5
                reasons.append(f"object:{term}")
        for term in facet_terms:
            if len(term) >= 4 and term in spec["heading"]:
                score += 1.0
                reasons.append(f"facet_heading:{term}")

        scored.append((score, index, tuple(reasons[:12])))

    scored.sort(key=lambda item: (-item[0], item[1]))
    target = min(cap, original_count)
    if original_count >= MIN_ROWS:
        target = max(target, min(MIN_ROWS, original_count))

    # Diversity: one best row per document first, then fill leftover
    # slots by score (same-document extra chunks only if they still rank).
    selected_idx: list[int] = []
    selected_set: set[int] = set()
    used_docs: set[str] = set()
    for score, index, _reasons in scored:
        if len(selected_idx) >= target:
            break
        key = _doc_key(rows[index])
        if key in used_docs:
            continue
        selected_idx.append(index)
        selected_set.add(index)
        used_docs.add(key)
    for _score, index, _reasons in scored:
        if len(selected_idx) >= target:
            break
        if index in selected_set:
            continue
        selected_idx.append(index)
        selected_set.add(index)

    score_by_index = {index: score for score, index, _reasons in scored}
    reasons_by_index = {index: reasons for _score, index, reasons in scored}
    packed_rows = [rows[i] for i in selected_idx]
    ranks = tuple(i + 1 for i in selected_idx)
    scores = tuple(round(score_by_index[i], 4) for i in selected_idx)
    reasons = tuple(reasons_by_index[i] for i in selected_idx)
    return PackedContext(
        rows=packed_rows,
        original_count=original_count,
        packed_count=len(packed_rows),
        selected_original_ranks=ranks,
        scores=scores,
        reasons=reasons,
    )
