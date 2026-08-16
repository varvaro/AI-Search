"""PR9.7.3 — deterministic drawing-navigation answers.

Pure helpers. No I/O, no LLM, no retrieval, no project vocabulary.
answer() may short-circuit here and skip Ollama for navigation queries.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


SENTINEL = "Nenalezeno v indexovaných dokumentech."

_HEADING_ORDER = (
    "PLAN",
    "SECTION",
    "SCHEME",
    "DETAIL",
    "SITUATION",
    "GENERIC_DRAWING",
)

_HEADING_LABEL = {
    "PLAN": "Půdorys",
    "SECTION": "Řez",
    "SCHEME": "Schéma",
    "DETAIL": "Detail",
    "SITUATION": "Situace",
    "GENERIC_DRAWING": "Výkres",
}


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


class DrawingSubtype(str, Enum):
    GENERIC_DRAWING = "GENERIC_DRAWING"
    PLAN = "PLAN"
    SECTION = "SECTION"
    SCHEME = "SCHEME"
    DETAIL = "DETAIL"
    SITUATION = "SITUATION"


_CORE = re.compile(r"vykres|pudorys|rez\b|schem")
_SPECIFIC = re.compile(r"pudorys|rez\b|schem|\bdetail\b|situac")
_DETAIL = re.compile(r"\bdetail\b")
_PLAN_WORD = re.compile(r"\bplan(?:u|em|y|e)?\b")
_SITUACE = re.compile(r"situac")
_NAV = re.compile(
    r"najdi|kde\s+(?:je|najdu|lezi)|"
    r"v\s+kter\w*\s+vykres|ve\s+kter\w*\s+plan|"
    r"kter\w*\s+(?:vykres|plan|pudorys|rez)"
)
_NAME_PLAN = re.compile(r"pudorys")
_NAME_SECTION = re.compile(r"\brez\b")
_NAME_SCHEME = re.compile(r"schem")
_NAME_DETAIL = re.compile(r"detail")
_NAME_SITUATION = re.compile(r"situac")
_NAME_DRAWING = re.compile(r"vykres|\.dwg\b")
_TEXT_PLAN = re.compile(r"\bpudorys\b")
_TEXT_SECTION = re.compile(r"\brez\b")
_TEXT_SCHEME = re.compile(r"schem")
_TEXT_DETAIL = re.compile(r"\bdetail\b")
_TEXT_SITUATION = re.compile(r"situac")
_TEXT_DRAWING = re.compile(r"vykres")
_FLOOR = re.compile(r"\d+\s*\.?\s*(?:pp|np)\b|\bpodlaz")
_TOKEN = re.compile(r"[a-z0-9]{3,}")
_WORD = re.compile(r"[^\s?!.,;:()\[\]\"“”]+")

_STOP = frozenset({
    "najdi", "najdu", "kde", "je", "lezi", "v", "ve", "mi", "me", "nam",
    "ktery", "ktera", "ktere", "kterou", "kterem", "kterych",
    "vykres", "vykresu", "vykresy", "pudorys", "pudorysu",
    "rez", "rezu", "schema", "schematu", "schem",
    "detail", "detailu", "plan", "planu", "planem", "plany",
    "situace", "situaci", "situacni", "situac",
    "a", "i", "o", "na", "pro", "z", "do", "u", "se", "si",
    "to", "ten", "ta", "jaky", "jake", "jakou", "jak",
    "dokument", "dokumentu", "soubor", "souboru",
})


@dataclass(frozen=True)
class DrawingMatch:
    subtype: DrawingSubtype
    document: str
    path: str
    quote: str
    source_index: int
    floor_plan: bool
    dedicated: bool


@dataclass(frozen=True)
class DrawingNavigationAnswer:
    text: str
    matches: tuple[DrawingMatch, ...]
    missing: tuple[DrawingSubtype, ...]
    abstained: bool


def is_drawing_navigation_query(query: str) -> bool:
    q = fold(query)
    if not q:
        return False
    has_core = bool(_CORE.search(q))
    has_specific = bool(_SPECIFIC.search(q))
    has_nav = bool(_NAV.search(q))
    has_detail = bool(_DETAIL.search(q))
    has_plan = bool(_PLAN_WORD.search(q))
    has_situace = bool(_SITUACE.search(q))
    has_drawing = has_core or (has_detail and has_nav) or (has_plan and (has_nav or has_core)) or (
        has_situace and (has_nav or has_core)
    )
    if not has_drawing:
        return False
    return has_nav or has_specific


def derive_requested_subtypes(query: str) -> tuple[DrawingSubtype, ...]:
    q = fold(query)
    if not q:
        return ()
    needs: list[DrawingSubtype] = []
    drawing_ctx = bool(_CORE.search(q) or _NAV.search(q))
    if _NAME_PLAN.search(q) or (_PLAN_WORD.search(q) and drawing_ctx):
        needs.append(DrawingSubtype.PLAN)
    if _NAME_SECTION.search(q):
        needs.append(DrawingSubtype.SECTION)
    if _NAME_SCHEME.search(q):
        needs.append(DrawingSubtype.SCHEME)
    if _DETAIL.search(q):
        needs.append(DrawingSubtype.DETAIL)
    if _SITUACE.search(q) and (drawing_ctx or _NAV.search(q)):
        needs.append(DrawingSubtype.SITUATION)
    if re.search(r"vykres", q):
        needs.append(DrawingSubtype.GENERIC_DRAWING)
    return tuple(needs)


def classify_result_subtypes(result: dict | None) -> frozenset[DrawingSubtype]:
    row = result or {}
    name_f = fold(f"{row.get('document') or ''} {row.get('path') or ''}")
    text_f = fold(f"{row.get('heading') or ''} {row.get('quote') or ''}")
    found: set[DrawingSubtype] = set()
    if _NAME_PLAN.search(name_f):
        found.add(DrawingSubtype.PLAN)
    if _NAME_SECTION.search(name_f):
        found.add(DrawingSubtype.SECTION)
    if _NAME_SCHEME.search(name_f):
        found.add(DrawingSubtype.SCHEME)
    if _NAME_DETAIL.search(name_f):
        found.add(DrawingSubtype.DETAIL)
    if _NAME_SITUATION.search(name_f):
        found.add(DrawingSubtype.SITUATION)
    if _NAME_DRAWING.search(name_f):
        found.add(DrawingSubtype.GENERIC_DRAWING)
    if found:
        return frozenset(found)
    if _TEXT_PLAN.search(text_f):
        found.add(DrawingSubtype.PLAN)
    if _TEXT_SECTION.search(text_f):
        found.add(DrawingSubtype.SECTION)
    if _TEXT_SCHEME.search(text_f):
        found.add(DrawingSubtype.SCHEME)
    if _TEXT_DETAIL.search(text_f):
        found.add(DrawingSubtype.DETAIL)
    if _TEXT_SITUATION.search(text_f):
        found.add(DrawingSubtype.SITUATION)
    if _TEXT_DRAWING.search(text_f):
        found.add(DrawingSubtype.GENERIC_DRAWING)
    return frozenset(found)


def subject_tokens(query: str) -> tuple[str, ...]:
    return tuple(
        tok for tok in _TOKEN.findall(fold(query))
        if tok not in _STOP
    )


def subject_phrase(query: str) -> str:
    kept = []
    for raw in _WORD.findall(query or ""):
        folded = fold(raw)
        if not folded or folded in _STOP:
            continue
        if _SPECIFIC.fullmatch(folded) or folded.startswith("vykres"):
            continue
        kept.append(raw)
    return " ".join(kept).strip()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(fold(text)))


def tokens_overlap(query_tokens: Iterable[str], result_tokens: Iterable[str]) -> bool:
    qtoks = [t for t in query_tokens if t]
    rtoks = [t for t in result_tokens if t]
    if not qtoks:
        return True
    for q in qtoks:
        for r in rtoks:
            if q == r:
                return True
            if min(len(q), len(r)) >= 4 and (q.startswith(r) or r.startswith(q)):
                return True
    return False


def result_text(result: dict | None) -> str:
    row = result or {}
    return " ".join(
        str(row.get(key) or "")
        for key in ("document", "path", "heading", "quote")
    )


def has_object_overlap(query: str, result: dict | None) -> bool:
    return tokens_overlap(subject_tokens(query), _tokens(result_text(result)))


def is_floor_plan(result: dict | None) -> bool:
    row = result or {}
    name_f = fold(f"{row.get('document') or ''} {row.get('path') or ''}")
    return bool(_NAME_PLAN.search(name_f) and _FLOOR.search(name_f))


def is_dedicated_plan(query: str, result: dict | None) -> bool:
    if DrawingSubtype.PLAN not in classify_result_subtypes(result):
        return False
    if is_floor_plan(result):
        return False
    row = result or {}
    title = f"{row.get('document') or ''} {row.get('heading') or ''}"
    return tokens_overlap(subject_tokens(query), _tokens(title))


def _subtype_matches_request(requested: DrawingSubtype, found: frozenset[DrawingSubtype]) -> bool:
    if requested is DrawingSubtype.GENERIC_DRAWING:
        return bool(found)
    return requested in found


def _pick_quote(result: dict, query: str) -> str:
    heading = str(result.get("heading") or "").strip()
    quote = re.sub(r"\s+", " ", str(result.get("quote") or "")).strip()
    name_f = fold(f"{result.get('document') or ''} {result.get('path') or ''}")
    subject = subject_tokens(query)
    hay = f"{heading} {quote}"
    hay_f = fold(hay)
    has_signal = bool(
        _TEXT_PLAN.search(hay_f) or _TEXT_SECTION.search(hay_f) or _TEXT_SCHEME.search(hay_f)
        or _TEXT_DETAIL.search(hay_f) or _TEXT_SITUATION.search(hay_f) or _TEXT_DRAWING.search(hay_f)
        or tokens_overlap(subject, _tokens(hay))
    )
    if not has_signal:
        return ""
    if heading and (
        tokens_overlap(subject, _tokens(heading))
        or _TEXT_SECTION.search(fold(heading))
        or _TEXT_PLAN.search(fold(heading))
    ):
        return heading[:160]
    if not quote:
        return heading[:160]
    window = _quote_window(quote, subject)
    if window:
        return window
    if _NAME_PLAN.search(name_f) or _NAME_SECTION.search(name_f) or _NAME_DRAWING.search(name_f):
        return ""
    return quote[:160]


def _quote_window(quote: str, subject: tuple[str, ...], limit: int = 140) -> str:
    folded = fold(quote)
    hits = []
    for pat in (r"\brez\b", r"pudorys", r"schem", r"\bdetail\b", r"situac", r"vykres"):
        m = re.search(pat, folded)
        if m:
            hits.append(m.start())
    for tok in subject:
        idx = folded.find(tok)
        if idx >= 0:
            hits.append(idx)
        else:
            for m in re.finditer(r"[a-z0-9]{4,}", folded):
                if tok.startswith(m.group()) or m.group().startswith(tok):
                    hits.append(m.start())
                    break
    if not hits:
        return ""
    pos = min(hits)
    start = max(0, pos - 20)
    end = min(len(quote), start + limit)
    snippet = quote[start:end].strip(" .,;:-")
    return snippet


def _find_match(
    requested: DrawingSubtype,
    query: str,
    results: list[dict],
    used_indexes: set[int],
) -> DrawingMatch | None:
    for i, row in enumerate(results or (), start=1):
        if i in used_indexes:
            continue
        found = classify_result_subtypes(row)
        if not _subtype_matches_request(requested, found):
            continue
        if not has_object_overlap(query, row):
            continue
        actual = requested
        if requested is DrawingSubtype.GENERIC_DRAWING:
            for pref in (
                DrawingSubtype.SECTION, DrawingSubtype.PLAN, DrawingSubtype.SCHEME,
                DrawingSubtype.DETAIL, DrawingSubtype.SITUATION, DrawingSubtype.GENERIC_DRAWING,
            ):
                if pref in found:
                    actual = pref
                    break
        quote = _pick_quote(row, query)
        return DrawingMatch(
            subtype=actual if requested is DrawingSubtype.GENERIC_DRAWING else requested,
            document=str(row.get("document") or ""),
            path=str(row.get("path") or ""),
            quote=quote,
            source_index=i,
            floor_plan=is_floor_plan(row),
            dedicated=is_dedicated_plan(query, row),
        )
    return None


def _missing_line(subtype: DrawingSubtype, query: str) -> str:
    label = _HEADING_LABEL[subtype.value]
    subject = subject_phrase(query)
    if subtype is DrawingSubtype.PLAN and subject:
        return f"Samostatný půdorys {subject} se mi v nalezených dokumentech nepodařilo doložit."
    if subject:
        return f"{label} {subject} se mi v nalezených dokumentech nepodařilo doložit."
    return f"{label} se mi v nalezených dokumentech nepodařilo doložit."


def _match_block(match: DrawingMatch, query: str) -> str:
    heading = _HEADING_LABEL[match.subtype.value]
    lines = [f"{heading}:", match.document]
    if match.subtype is DrawingSubtype.PLAN and match.floor_plan and not match.dedicated:
        subject = subject_phrase(query) or "objekt"
        if match.quote:
            lines.append(
                f"— „{match.quote}“ — podlažní půdorys, ve kterém je {subject} zakreslený/popsaný. "
                "Nejde nutně o samostatný detailní půdorys."
            )
        else:
            lines.append(
                f"— {subject} je v tomto podlažním půdorysu zakreslený/popsaný. "
                "Nejde nutně o samostatný detailní půdorys."
            )
    elif match.quote:
        lines.append(f"— „{match.quote}“")
    return "\n".join(lines)


def render_drawing_navigation(query: str, results: list[dict] | None) -> DrawingNavigationAnswer | None:
    if not is_drawing_navigation_query(query):
        return None
    requested = derive_requested_subtypes(query)
    if not requested:
        requested = (DrawingSubtype.GENERIC_DRAWING,)
    rows = list(results or ())
    used: set[int] = set()
    matches: list[DrawingMatch] = []
    missing: list[DrawingSubtype] = []
    specific = [s for s in requested if s is not DrawingSubtype.GENERIC_DRAWING]
    generic = DrawingSubtype.GENERIC_DRAWING in requested
    for subtype in specific:
        hit = _find_match(subtype, query, rows, used)
        if hit:
            used.add(hit.source_index)
            matches.append(hit)
        else:
            missing.append(subtype)
    if generic:
        hit = _find_match(DrawingSubtype.GENERIC_DRAWING, query, rows, used)
        if hit:
            used.add(hit.source_index)
            matches.append(hit)
        elif not specific:
            missing.append(DrawingSubtype.GENERIC_DRAWING)
    if not matches:
        return DrawingNavigationAnswer(SENTINEL, (), tuple(missing or requested), True)
    order = {name: i for i, name in enumerate(_HEADING_ORDER)}
    matches.sort(key=lambda m: order.get(m.subtype.value, 99))
    blocks = [_match_block(m, query) for m in matches]
    for subtype in missing:
        blocks.append(f"{_HEADING_LABEL[subtype.value]}:\n{_missing_line(subtype, query)}")
    return DrawingNavigationAnswer("\n\n".join(blocks), tuple(matches), tuple(missing), False)


def try_render(query: str, results: list[dict] | None) -> str | None:
    rendered = render_drawing_navigation(query, results)
    if rendered is None:
        return None
    return rendered.text
