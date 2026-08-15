"""PR9.4.1 — Metadata-aware Phase-3 reranking (experimental, flag-gated).

Additive Phase-3 score term built from three *generic, project-agnostic*
signals computed purely from the query string and one candidate's document
name/path:

  1. token overlap   — shared content words between the query and the
                        document's filename/path (name/path only, never the
                        chunk body).
  2. date whitelist   — a date literally present in both the query and the
                        filename/path. Exact-match only; a newer or older
                        document date is never treated as better or worse.
  3. discriminator    — structural tokens (floor notation like "1.PP",
                        letter-prefixed dotted drawing-style codes like
                        "X.1.2.03", and generic alphanumeric ids mixing
                        letters and digits) that must match *exactly*
                        between query and document. A same-kind mismatch
                        (query says "1.PP", candidate names "2.PP") is a
                        negative signal; dates never carry a mismatch
                        penalty (a document may simply have an unrelated
                        date stamp).

Does NOT:
  * change FTS / Lance / embeddings / candidate_strategy / top_ids
  * read chunk `heading` or `quote` — name/path only
  * express any document-class, revision, "final beats draft", or recency
    ("newer wins") preference
  * call an LLM or touch answer() / context_packing / entity_hints /
    document_state / entity_match_bonus / revision_ranking
  * hardcode any vendor, NOT-id, drawing code, or other project-specific
    value — every pattern below is a generic structural regex

See tests/test_metadata_rerank_pr941.py for the full behavioural contract.

`fold()` is defined locally rather than imported, matching this repo's
per-module normalization convention (query_facets, entity_match_bonus,
revision_ranking, entity_hints each keep their own copy) so a change to one
layer's matching contract can never silently change another's.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

# Tunables — deliberately smaller than FILENAME_MATCH_BONUS (0.03, see
# ai_search.py) so this layer can only break a near-tie inside the already
# rerank_k-truncated pool; it can never override a verbatim full-query
# filename match or a cross-encoder score on its own.
CONTENT_TOKEN_NAME_BONUS = 0.012
CONTENT_TOKEN_PATH_BONUS = 0.006
TOKEN_OVERLAP_CAP = 0.036
# A single generic shared word is too weak a signal on its own (e.g. every
# contract candidate shares "smlouva"-adjacent vocabulary); require at least
# two independent overlaps (or a discriminator hit, scored separately).
MIN_CONTENT_TOKENS_FOR_BONUS = 2

DISCRIMINATOR_HIT_BONUS = 0.030
DISCRIMINATOR_MISMATCH_PENALTY = -0.025
DISCRIMINATOR_CAP = 0.06
DISCRIMINATOR_FLOOR = -0.05

MIN_CONTENT_TOKEN_LEN = 4  # same floor as entity_match_bonus.extract_entity_signals
MIN_ALNUM_ID_LEN = 3
MAX_ALNUM_ID_LEN = 14

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Function words + generic document/legal vocabulary. No firm names, no
# project identifiers, no document-class preference words — this is purely
# noise removal for the token-overlap signal, same spirit as
# entity_match_bonus._STOPWORDS but kept as an independent local copy.
_STOPWORDS = frozenset({
    "a", "i", "o", "u", "k", "s", "z", "v", "ve", "ke", "se", "si", "na", "do",
    "od", "za", "po", "pro", "pri", "bez", "je", "jsou", "jak", "jake", "jaka",
    "jaky", "jakych", "co", "ci", "nebo", "ale",
    "the", "and", "or", "of", "in", "on", "to", "with", "for", "at",
    "najdi", "najit", "hledej", "hledejte", "existuje", "existuji", "ukaz",
    "ukazte", "dej", "dejte",
    "dokument", "dokumentu", "dokumenty", "soubor", "souboru", "slozka",
    "slozky", "ma", "maji", "byl", "byla", "bylo", "byly", "kdo", "kdy",
    "kde", "proc", "kolik", "ktery", "ktera", "ktere", "kterou",
})

# --- discriminator patterns ---------------------------------------------------
# All patterns run against a folded (casefold + diacritics-stripped) blob, so
# no IGNORECASE flag is needed and Czech declension of surrounding words never
# affects ASCII structural matching (digits/dots/letters survive folding).

# Floor / storey notation: "1.PP" (podzemní podlaží) / "2.NP" (nadzemní
# podlaží) — a generic Czech construction-drawing convention, not tied to any
# project. Requires the literal unit suffix so it never fires on a bare
# drawing-code fragment. Boundaries exclude letters/digits but allow "_"/"-"/
# space/dot, since filenames glue tokens together with any of those.
_FLOOR_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})\s*\.?\s*(pp|np)(?![A-Za-z0-9])")

# Letter-prefixed dotted numbering used broadly in technical drawing sets
# (e.g. a leading discipline letter followed by dotted sheet/revision
# numbers, optionally with a single-letter sub-segment): a leading letter,
# then 2-5 dot-separated segments that are each either pure digits or a
# single letter. Restricting each segment this way (rather than any alnum
# run) keeps a trailing file extension like ".pdf" from ever being absorbed
# into the code. Boundaries exclude letters/digits but deliberately allow
# "_"/"-"/space, since filenames commonly glue a separator right after the
# code (e.g. "<code>_ProjectName-Sheet.pdf").
_DRAWING_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])[a-z]\.(?:\d+|[a-z])(?:\.(?:\d+|[a-z])){1,4}(?![A-Za-z0-9])"
)

# Generic alphanumeric identifier: letters+digits mixed in either order
# (e.g. a short product/order/sheet code combining letters and digits). This
# is a structural shape, not a specific prefix — no project vendor code or
# document-id prefix is named anywhere in this module.
_ALNUM_ID_RE = re.compile(r"(?:[a-z]+\d+[a-z0-9]*|\d+[a-z]+[a-z0-9]*)")

# --- safe date whitelist ------------------------------------------------------
# Deliberately conservative. The goal is to accept only shapes that are
# structurally unambiguous so this can never misread a drawing/revision code
# (e.g. a letter-prefixed dotted sheet number like "X.1.2.03") or the digits
# inside an unrelated alphanumeric id (e.g. "AB251110") as a calendar date —
# without denylisting any specific project value. A 2-4 digit number is
# never treated as a year on its own; every branch below requires an
# unambiguous separator/isolation shape. Trade-off: a small number of
# legitimate but ambiguous stamps (e.g. a dotted date immediately glued to a
# preceding word by a dot, such as "rev.03.04.25") are intentionally not
# parsed — recall loss here is acceptable because this module only uses
# dates for exact query<->document matching, never for "which document is
# newest".
_DATE_ISO_RE = re.compile(
    r"(?<![a-z0-9])((?:19|20)\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])(?![a-z0-9])"
)
_DATE_YMD_COMPACT_RE = re.compile(
    r"(?<![a-z0-9])((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?![a-z0-9])"
)
# Six-digit construction-stamp shorthand (YYMMDD), isolated from surrounding
# letters/digits so it can never fire on part of a longer alphanumeric id
# (an id's digits are directly preceded/followed by letters, not a boundary).
_DATE_YMD_SHORT_RE = re.compile(
    r"(?<![a-z0-9])(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?![a-z0-9])"
)
# Dot-separated D.M.Y — excluded whenever immediately preceded by a dot or
# digit, which is exactly what a longer dotted numeric chain (a drawing code)
# looks like; this needs no project-specific denylist to reject a code such
# as "X.1.2.03".
_DATE_DMY_RE = re.compile(
    r"(?<![.\d])([0-9]{1,2})\.([0-9]{1,2})\.((?:19|20)\d{2}|[0-9]{2})(?!\d)(?!\.\d)"
)
# Plausible construction-document year window for 2-digit-year shapes only
# (compact YYMMDD and short DMY years). Four-digit years are never bounded.
_SHORT_YEAR_MIN = 20
_SHORT_YEAR_MAX = 35


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def _valid_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_safe_dates(text: str) -> tuple[date, ...]:
    """Extract dates from `text` using a conservative, generic whitelist.

    Only structurally unambiguous shapes are accepted — see the module
    docstring and the regex comments above for exactly which ones and why.
    Returns () when nothing safely parses; never raises.
    """
    blob = fold(text)
    seen: set[date] = set()
    out: list[date] = []

    def add(parsed: date | None) -> None:
        if parsed is not None and parsed not in seen:
            seen.add(parsed)
            out.append(parsed)

    for m in _DATE_ISO_RE.finditer(blob):
        add(_valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _DATE_YMD_COMPACT_RE.finditer(blob):
        add(_valid_date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _DATE_YMD_SHORT_RE.finditer(blob):
        yy = int(m.group(1))
        if not (_SHORT_YEAR_MIN <= yy <= _SHORT_YEAR_MAX):
            continue
        add(_valid_date(2000 + yy, int(m.group(2)), int(m.group(3))))
    for m in _DATE_DMY_RE.finditer(blob):
        day, month, year_raw = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(year_raw) == 2:
            year = int(year_raw)
            if not (_SHORT_YEAR_MIN <= year <= _SHORT_YEAR_MAX):
                continue
            year += 2000
        else:
            year = int(year_raw)
        add(_valid_date(year, month, day))
    return tuple(sorted(out))


@dataclass(frozen=True)
class Discriminator:
    kind: str        # "floor" | "drawing_code" | "alnum_id" | "iso_date"
    canonical: str    # normalized value used for cross-side equality


def _extract_floor(blob: str) -> list[Discriminator]:
    return [
        Discriminator("floor", f"{int(m.group(1))}.{m.group(2)}")
        for m in _FLOOR_RE.finditer(blob)
    ]


def _extract_drawing_codes(blob: str) -> list[Discriminator]:
    return [Discriminator("drawing_code", m.group(0)) for m in _DRAWING_CODE_RE.finditer(blob)]


def _extract_alnum_ids(tokens: tuple[str, ...]) -> list[Discriminator]:
    out = []
    for tok in tokens:
        if MIN_ALNUM_ID_LEN <= len(tok) <= MAX_ALNUM_ID_LEN and _ALNUM_ID_RE.fullmatch(tok):
            out.append(Discriminator("alnum_id", tok))
    return out


def _extract_date_discriminators(blob: str) -> list[Discriminator]:
    return [Discriminator("iso_date", d.isoformat()) for d in parse_safe_dates(blob)]


def _tokenize_folded(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(fold(text)))


def _content_tokens(tokens: tuple[str, ...], exclude: frozenset[str]) -> tuple[str, ...]:
    out = []
    for tok in tokens:
        if tok in exclude or tok in _STOPWORDS:
            continue
        if len(tok) < MIN_CONTENT_TOKEN_LEN or tok.isdigit():
            continue
        out.append(tok)
    return tuple(out)


@dataclass(frozen=True)
class QueryMetadata:
    content_tokens: tuple[str, ...]
    discriminators: tuple[Discriminator, ...]


@dataclass(frozen=True)
class DocumentMetadata:
    name_tokens: tuple[str, ...]
    path_tokens: tuple[str, ...]
    discriminators: tuple[Discriminator, ...]


def extract_query_metadata(query: str) -> QueryMetadata:
    """Pure decomposition of a query into content tokens + discriminators.

    No I/O, no retrieval, no project-specific vocabulary.
    """
    raw = query or ""
    blob = fold(raw)
    tokens = _tokenize_folded(raw)
    discriminators = (
        tuple(_extract_floor(blob))
        + tuple(_extract_drawing_codes(blob))
        + tuple(_extract_alnum_ids(tokens))
        + tuple(_extract_date_discriminators(blob))
    )
    alnum_values = frozenset(d.canonical for d in discriminators if d.kind == "alnum_id")
    content = _content_tokens(tokens, exclude=alnum_values)
    return QueryMetadata(content_tokens=content, discriminators=discriminators)


def extract_document_metadata(name: str, path: str) -> DocumentMetadata:
    """Pure decomposition of a document's name+path. Never reads chunk text."""
    name_raw, path_raw = name or "", path or ""
    name_tokens = _tokenize_folded(name_raw)
    path_tokens = _tokenize_folded(path_raw)
    blob = fold(f"{name_raw} {path_raw}")
    discriminators = (
        tuple(_extract_floor(blob))
        + tuple(_extract_drawing_codes(blob))
        + tuple(_extract_alnum_ids(name_tokens + path_tokens))
        + tuple(_extract_date_discriminators(blob))
    )
    return DocumentMetadata(name_tokens=name_tokens, path_tokens=path_tokens, discriminators=discriminators)


@dataclass(frozen=True)
class MetadataScoreDetail:
    bonus: float
    overlap_name: tuple[str, ...] = ()
    overlap_path: tuple[str, ...] = ()
    discriminator_hits: tuple[str, ...] = ()
    discriminator_mismatches: tuple[str, ...] = ()

    def as_trace_dict(self) -> dict:
        return {
            "bonus": self.bonus,
            "overlap_name": list(self.overlap_name),
            "overlap_path": list(self.overlap_path),
            "discriminator_hits": list(self.discriminator_hits),
            "discriminator_mismatches": list(self.discriminator_mismatches),
        }


_EMPTY_DETAIL = MetadataScoreDetail(bonus=0.0)


def compute_metadata_score(
    query: str,
    document_name: str,
    document_path: str,
    *,
    skip_token_overlap: bool = False,
) -> MetadataScoreDetail:
    """Additive Phase-3 bonus from token overlap + discriminator matching.

    `skip_token_overlap`: set True when the caller already scored a verbatim
    full-query filename match (ai_search.FILENAME_MATCH_BONUS) for this
    candidate, so the same evidence is never counted twice.

    Pure function: no I/O, no project-specific values, name/path only.
    """
    qmeta = extract_query_metadata(query)
    if not qmeta.content_tokens and not qmeta.discriminators:
        return _EMPTY_DETAIL

    dmeta = extract_document_metadata(document_name, document_path)

    overlap_name: list[str] = []
    overlap_path: list[str] = []
    token_bonus = 0.0
    if not skip_token_overlap and qmeta.content_tokens:
        name_set, path_set = set(dmeta.name_tokens), set(dmeta.path_tokens)
        seen: set[str] = set()
        for tok in qmeta.content_tokens:
            if tok in seen:
                continue
            if tok in name_set:
                seen.add(tok)
                overlap_name.append(tok)
            elif tok in path_set:
                seen.add(tok)
                overlap_path.append(tok)
        if len(overlap_name) + len(overlap_path) >= MIN_CONTENT_TOKENS_FOR_BONUS:
            token_bonus = min(
                len(overlap_name) * CONTENT_TOKEN_NAME_BONUS
                + len(overlap_path) * CONTENT_TOKEN_PATH_BONUS,
                TOKEN_OVERLAP_CAP,
            )
        else:
            overlap_name, overlap_path = [], []

    disc_hits: list[str] = []
    disc_mismatches: list[str] = []
    disc_bonus = 0.0
    if qmeta.discriminators and dmeta.discriminators:
        doc_by_kind: dict[str, set[str]] = {}
        for d in dmeta.discriminators:
            doc_by_kind.setdefault(d.kind, set()).add(d.canonical)
        seen_q: set[tuple[str, str]] = set()
        for qd in qmeta.discriminators:
            key = (qd.kind, qd.canonical)
            if key in seen_q:
                continue
            seen_q.add(key)
            doc_values = doc_by_kind.get(qd.kind)
            if not doc_values:
                continue
            if qd.canonical in doc_values:
                disc_hits.append(f"{qd.kind}:{qd.canonical}")
                disc_bonus += DISCRIMINATOR_HIT_BONUS
            elif qd.kind != "iso_date":
                # Dates are match-only (see module docstring): a document
                # simply having a *different* date is not evidence against
                # it, so only floor/drawing_code/alnum_id ever penalize.
                disc_mismatches.append(f"{qd.kind}:{qd.canonical}")
                disc_bonus += DISCRIMINATOR_MISMATCH_PENALTY
        disc_bonus = max(min(disc_bonus, DISCRIMINATOR_CAP), DISCRIMINATOR_FLOOR)

    bonus = token_bonus + disc_bonus
    if not (overlap_name or overlap_path or disc_hits or disc_mismatches):
        return _EMPTY_DETAIL

    return MetadataScoreDetail(
        bonus=bonus,
        overlap_name=tuple(overlap_name),
        overlap_path=tuple(overlap_path),
        discriminator_hits=tuple(disc_hits),
        discriminator_mismatches=tuple(disc_mismatches),
    )
