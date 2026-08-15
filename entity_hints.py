"""PR9.3.4 — deterministic entity / identifier hint candidates for the LLM prompt.

Pure, deterministic extraction of *candidates to verify* from an already
retrieved and already packed evidence pool:

    build_entity_hints(query, llm_results) -> EntityHints

The layer answers nothing. It only tells the model which concrete strings are
present in the context it was given, and under which 1-based source index:

    Kandidát ve zdroji [1]: ACME

Hard contracts (see tests/test_entity_hints_pr934.py):

* Never calls search, Ollama, SQLite, or Lance; never mutates input rows.
* A candidate value is only ever a substring of a field that answer() actually
  renders into ZDROJE (`document`, `heading`, `quote`). `path`/`relative_path`
  are deliberately never read: a value the model cannot see cannot be cited,
  and offering it would invite an unsupported claim.
* WHO candidates come from `document` and `heading` only, never from `quote`.
  A firm name mentioned inside a neighbouring document's body text is exactly
  the wrong-entity failure this layer exists to prevent, so body text is not a
  source of WHO candidates at all — the restriction is structural, not a score.
* `source_index` is the 1-based position in the `llm_results` sequence handed
  to the renderer, i.e. the same basis as `results[zdroj_index-1]`. 0 is never
  produced and no index is ever remapped.
* No project-specific document, firm, or standard value is hardcoded. Kind
  detection uses generic patterns (ČSN/EN/TP/ČBS prefixes, NOT-style codes)
  and generic Czech document vocabulary only.

Prompt-only *instructions* were already tried and rejected: PR9.3.1 added
query-focused extraction rules to JSON_ANSWER_GUARD and measured 0/6
extraction in live A/B (see tests/test_query_focused_extraction_pr931.py).
This module therefore emits per-query candidate *values with indexes* — data,
not rules — and the static prompt constants stay untouched.

`fold()` is defined locally rather than imported. Every layer in this repo
(query_facets, evidence, evidence_runtime, context_packing) keeps its own
normalization so a downstream flag-gated module cannot change another layer's
matching contract; PR9.3.4 follows that convention and does not import
context_packing.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

import answer_intent
import query_expansion as qe

# Total candidates offered to the model. Small on purpose: the point is to
# narrow the choice, and a long list re-creates the context dilution PR9.3.3
# removed (and its latency cost).
MAX_CANDIDATES = 6
# Per row and per kind, so one verbose heading cannot fill the whole budget.
PER_ROW_KIND_CAP = 2
# Upper bound on the rendered block, enforced by dropping trailing candidates.
PROMPT_CHAR_BUDGET = 400
# PR9.3.4.1: candidates below this score are dropped instead of padding the
# block. The PR9.3.4 A/B offered five candidates for a WHO query of which one
# was right; a short honest list beats a long noisy one.
MIN_SCORE = 2.0
# Per kind and per query, so one over-generating shape rule (WHO especially)
# cannot consume the whole budget and hide a second requested kind.
PER_KIND_CAP = 3

_WHO_MIN_LEN = 4
_WHO_MAX_LEN = 24
# A Capitalized (not ALL-CAPS) unit needs more length to count as a name, which
# is what removes truncated filename fragments.
_WHO_TITLE_MIN_LEN = 5

_PROMPT_HEADER = (
    "KANDIDÁTI K OVĚŘENÍ (nejde o odpověď — každého kandidáta ověř ve ZDROJÍCH "
    "a použij jen toho, který dotaz skutečně zodpovídá):"
)


class HintKind(str, Enum):
    WHO = "WHO"
    STANDARD = "STANDARD"
    IDENTIFIER = "IDENTIFIER"


# Fields answer() renders into ZDROJE. WHO is restricted further (no quote).
_VALUE_FIELDS = ("document", "heading", "quote")
_WHO_FIELDS = ("document", "heading")


def fold(text: str) -> str:
    """Casefold + strip combining marks. Digits and ASCII identifiers survive."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def _fold_with_map(text: str) -> tuple[str, tuple[int, ...]]:
    """Fold `text` and keep, per folded character, its original index.

    Needed because a candidate must be shown to the model in the exact form it
    appears in the context (with its original diacritics and case) while
    matching runs on the folded form. NFKD is not length-preserving, so offsets
    cannot be derived from the folded string alone.
    """
    folded: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text or ""):
        for expanded in unicodedata.normalize("NFKD", char.casefold()):
            if unicodedata.combining(expanded):
                continue
            folded.append(expanded)
            offsets.append(index)
    return "".join(folded), tuple(offsets)


# --- value patterns -----------------------------------------------------------
# Standard / technical-rule identifiers. Most specific first: "ČSN EN 1992-1-1"
# must win over the bare "EN 1992-1-1" and "ČSN" readings of the same span.
_STANDARD_RES = (
    re.compile(r"\bcsn\s*en\s*(?:iso\s*)?\d+(?:[-.\u2013/]\d+)*"),
    re.compile(r"\bcsn\s*(?:iso\s*)?\d+(?:[-.\u2013/]\d+)*"),
    re.compile(r"(?<![a-z])en\s*(?:iso\s*)?\d+(?:[-.\u2013/]\d+)*"),
    re.compile(r"\bcbs\s*\d+(?:[-./]\d+)*"),
    # PR9.3.4.1: technical rules are numbered in the hundreds, so a one- or
    # two-digit "TP n" is a list marker or a fragment, not a rule reference.
    # The A/B produced two such false candidates.
    re.compile(r"\btp\s*\d{3,}[a-z]?\b"),
)
# Alpha-prefixed document codes (e.g. NOT-style order numbers) and bare long
# numeric document ids. The alpha prefix is bounded so a sentence word cannot
# become a code, and standard prefixes are excluded so a rule number stays a
# STANDARD instead of turning into a document identifier.
_IDENTIFIER_RES = (
    re.compile(r"\b([a-z]{3,5})[-_ ]?(\d{4,}(?:[-/]\d+)*)\b"),
    re.compile(r"\b(\d{5,}(?:[-/]\d+)*)\b"),
)
_STANDARD_PREFIXES = frozenset({"csn", "en", "tp", "cbs", "iso", "eta", "etag"})

# PR9.3.4.1 — registry / contact numbers are not document identifiers. The A/B
# produced six candidates for an order-number query, all of them company
# registration or VAT numbers pulled out of body text.
#
# Three generic rules, no dataset-specific value:
#  1. A label immediately before the number names it (IČO / DIČ / tel. / PSČ …).
#  2. A VAT number carries a two-letter country prefix, so the alpha-prefix
#     pattern above requires at least three letters.
#  3. Bare digit runs are only read from `document` / `heading`, never from
#     `quote`: a filename or section title numbers the document itself, while a
#     number inside body text is far more often a registry, contact, or measured
#     value. Alpha-prefixed codes stay readable everywhere.
_NUMBER_LABELS = frozenset({
    "ico", "ic", "dic", "vat", "tel", "telefon", "mobil", "fax", "psc", "zip",
    "iban", "swift", "bic", "ucet", "uctu", "banka", "bankovni", "vs",
    "variabilni", "ks", "rc", "nar", "dat", "datum", "kc", "czk", "eur",
    "mpa", "mm", "cm", "kg", "m2", "m3", "str", "strana", "verze", "rev",
})
_NUMBER_LABEL_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_LABELS)) + r")\b[\s.:_/-]{0,4}$"
)
# Exact widths of the Czech registry / contact formats we must not offer:
# PSČ (5), IČO (8), a local phone number (9).
_REJECTED_BARE_WIDTHS = frozenset({5, 8, 9})
# Fields that number the document itself rather than quoting body text.
_BARE_NUMBER_FIELDS = frozenset({"document", "heading"})

# Word units including digits, so an alphanumeric code stays one unit and is
# rejected as a WHO candidate instead of contributing its bare letter prefix.
_WORD_UNIT_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Generic Czech document / administrative vocabulary. These are the words that
# make a heading look like a name ("TECHNICKÁ ZPRÁVA", "Dodací list") without
# naming any party. Generic terminology only — no project, firm, or document
# value from any concrete dataset.
_GENERIC_DOC_WORDS = frozenset({
    "dokument", "dokumenty", "dokumentace", "priloha", "prilohy", "protokol", "protokoly",
    "zprava", "zpravy", "technicka", "technicky", "technicke", "souhrnna", "souhrnne",
    "celkova", "celkovy", "zaverecna", "zaverecny", "prubezna", "prubezny",
    "list", "listy", "listu", "dodaci", "dodavka", "dodavky", "faktura", "faktury",
    "objednavka", "objednavky", "smlouva", "smlouvy", "dodatek", "dodatky",
    "vykres", "vykresy", "situace", "schema", "detail", "detaily", "rez", "rezy",
    "plan", "plany", "denik", "deniku", "zapis", "zapisy", "kniha", "knihy",
    "revize", "revizni", "kontrola", "kontrolni", "zkouska", "zkousky", "zkusebni",
    "cast", "casti", "strana", "stran", "verze", "kopie", "original", "final", "finalni",
    "stavba", "stavby", "stavebni", "projekt", "projektu", "projektova", "projektove",
    "investor", "investora", "zhotovitel", "zhotovitele", "dodavatel", "dodavatele",
    "subdodavatel", "objednatel", "zadavatel", "ucastnik", "ucastnici",
    "nova", "novy", "nove", "stara", "stary", "stare", "aktualni", "platny", "platna",
    "mesic", "rok", "datum", "cislo", "ks", "kus", "celkem", "poznamka", "poznamky",
    "prehled", "seznam", "soupis", "vzor", "formular", "sablona",
    "zakazka", "zakazky", "zakazek", "zakazce", "etapa", "etapy", "objekt",
    "polozka", "polozky", "rozpocet", "vykaz", "vymer", "termin", "harmonogram",
})


def _vocabulary_words() -> frozenset[str]:
    """Folded single words from the shared domain vocabulary.

    Reused as a WHO rejection filter: a lexicon term is a concept, never a
    party. Read-only use of query_expansion's dictionary — no expansion is
    performed and the dictionary is not modified.
    """
    words: set[str] = set()
    vocabulary = getattr(qe, "DOMAIN_VOCABULARY", {}) or {}
    for key, rule in vocabulary.items():
        surfaces = [key]
        for field in ("synonyms", "documents", "abbreviations"):
            surfaces.extend((rule or {}).get(field, ()) or ())
        for surface in surfaces:
            for word in _WORD_UNIT_RE.findall(fold(surface)):
                if word:
                    words.add(word)
    return frozenset(words)


_VOCABULARY_WORDS = _vocabulary_words()


@dataclass(frozen=True)
class HintCandidate:
    """One concrete string present in one rendered source.

    `source_index` is 1-based into the row sequence passed to
    build_entity_hints(), matching the renderer's `results[zdroj_index-1]`.
    """

    kind: HintKind
    value: str
    source_index: int
    field: str
    score: float = 0.0
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "source_index": self.source_index,
            "field": self.field,
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EntityHints:
    query: str
    requested_kinds: tuple[HintKind, ...]
    candidates: tuple[HintCandidate, ...]

    def __bool__(self) -> bool:
        return bool(self.candidates)

    def as_prompt_block(self) -> str:
        """Additive prompt section, or "" when there is nothing to offer.

        Lists candidates only. It never states or ranks an answer, so a wrong
        candidate costs the model nothing beyond ignoring a line.
        """
        if not self.candidates:
            return ""
        lines = [_PROMPT_HEADER]
        for candidate in self.candidates:
            line = f"Kandidát ve zdroji [{candidate.source_index}]: {candidate.value}"
            block_length = len("\n\n" + "\n".join(lines + [line]))
            if block_length > PROMPT_CHAR_BUDGET and len(lines) > 1:
                break
            lines.append(line)
        if len(lines) == 1:
            return ""
        return "\n\n" + "\n".join(lines)

    def as_debug_dict(self) -> dict:
        return {
            "requested_kinds": [kind.value for kind in self.requested_kinds],
            "candidate_count": len(self.candidates),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "source_indexes": sorted({c.source_index for c in self.candidates}),
        }


def _requested_kinds(query_folded: str) -> tuple[HintKind, ...]:
    """PR9.3.4.1: one shared classifier (answer_intent.HINT_PROFILE).

    The PR9.3.4 A/B found this module's own copy of the rules disagreeing with
    the packing layer's — a standard-intent query produced no hint at all.
    """
    return tuple(HintKind(name) for name in answer_intent.hint_kinds(query_folded))


def _field_text(row: dict, field: str) -> str:
    if field == "document":
        # Extension carries no candidate and would leak into a value.
        raw = str(row.get("document") or "")
        return raw.rsplit(".", 1)[0] if "." in raw[1:] else raw
    return str(row.get(field) or "")


def _accept_span(accepted: list[tuple[int, int]], span: tuple[int, int]) -> bool:
    """Reject a span already covered by a more specific earlier match."""
    start, end = span
    for other_start, other_end in accepted:
        if start >= other_start and end <= other_end:
            return False
    return True


def _extract_standards(text: str) -> tuple[tuple[str, tuple[int, int]], ...]:
    folded, offsets = _fold_with_map(text)
    if not folded:
        return ()
    accepted: list[tuple[int, int]] = []
    found: list[tuple[str, tuple[int, int]]] = []
    for pattern in _STANDARD_RES:
        for match in pattern.finditer(folded):
            span = match.span()
            if not _accept_span(accepted, span):
                continue
            accepted.append(span)
            original = text[offsets[span[0]]:offsets[span[1] - 1] + 1].strip()
            if original:
                found.append((original, span))
    return tuple(found)


def _extract_identifiers(
    text: str,
    standard_spans: tuple[tuple[int, int], ...],
    field: str,
) -> tuple[tuple[str, bool], ...]:
    """Document identifiers as (value, alpha_prefixed) pairs.

    `field` gates the bare-numeric pattern: body text numbers are registry,
    contact, and measured values far more often than document ids (PR9.3.4.1).
    """
    folded, offsets = _fold_with_map(text)
    if not folded:
        return ()
    allow_bare = field in _BARE_NUMBER_FIELDS
    accepted: list[tuple[int, int]] = list(standard_spans)
    found: list[tuple[str, bool]] = []
    for pattern in _IDENTIFIER_RES:
        for match in pattern.finditer(folded):
            span = match.span()
            if not _accept_span(accepted, span):
                continue
            groups = match.groups()
            alpha_prefixed = len(groups) == 2
            if alpha_prefixed and groups[0] in _STANDARD_PREFIXES:
                continue
            # "IČO 03747808" / "tel 736517669": the label itself matched as the
            # alpha prefix, so a leading-context check alone would miss it.
            if alpha_prefixed and groups[0] in _NUMBER_LABELS:
                continue
            if not alpha_prefixed:
                if not allow_bare:
                    continue
                digits = re.sub(r"\D", "", match.group(0))
                if len(digits) in _REJECTED_BARE_WIDTHS:
                    continue
            if _NUMBER_LABEL_RE.search(folded[:span[0]]):
                continue
            accepted.append(span)
            original = text[offsets[span[0]]:offsets[span[1] - 1] + 1].strip()
            if original:
                found.append((original, alpha_prefixed))
    return tuple(found)


def _shares_query_stem(folded: str, query_words: frozenset[str], length: int = 5) -> bool:
    """True when a candidate and a query word share a long prefix.

    Czech inflection means the exact-word block misses "zakázka" for a query
    that said "zakázky", which the PR9.3.4 A/B surfaced as a false WHO candidate.
    """
    if len(folded) < length:
        return False
    stem = folded[:length]
    return any(word.startswith(stem) for word in query_words if len(word) >= length)


def _extract_who(
    text: str,
    blocked: frozenset[str],
    query_words: frozenset[str] = frozenset(),
) -> tuple[tuple[str, bool], ...]:
    """Name-like units from a structural field.

    Deterministic and dictionary-free: a unit qualifies when it is purely
    alphabetic, ALL-CAPS or Capitalized, and is not domain vocabulary, generic
    document terminology, or a word already present in the query. No firm list
    exists in this repo and none is introduced here, so this is a shape rule —
    it can over-generate, which is why the output is labelled a candidate.
    """
    found: list[tuple[str, bool]] = []
    for unit in _WORD_UNIT_RE.findall(text or ""):
        if any(char.isdigit() for char in unit):
            continue
        if not (_WHO_MIN_LEN <= len(unit) <= _WHO_MAX_LEN):
            continue
        all_caps = unit.isupper()
        if not (all_caps or unit.istitle()):
            continue
        # PR9.3.4.1: a Capitalized short unit is a filename fragment, not a name.
        if not all_caps and len(unit) < _WHO_TITLE_MIN_LEN:
            continue
        folded = fold(unit)
        if not folded or folded in blocked:
            continue
        if _shares_query_stem(folded, query_words):
            continue
        found.append((unit, all_caps))
    return tuple(found)


# --- candidate scoring (PR9.3.4.1) -------------------------------------------
# The PR9.3.4 A/B ordered candidates by field and row order alone, so noise
# crowded out the right value even when it was present in the packed context.
# Scoring uses only signals available on the rows the model already sees; it
# never consults retrieval scores, ranks, or the index.
#
# Field weight is per kind on purpose. A heading that names a party ("<NAME>
# delivery note") is stronger evidence of *who* than a filename token, which
# often carries a product or system name; for a document identifier the filename
# is the stronger field.
_FIELD_WEIGHTS = {
    HintKind.WHO: {"heading": 3.0, "document": 2.0, "quote": 0.0},
    HintKind.STANDARD: {"document": 3.0, "heading": 2.5, "quote": 1.5},
    HintKind.IDENTIFIER: {"document": 3.0, "heading": 2.5, "quote": 1.0},
}
# A value repeated across the packed rows is descriptive of the project, not an
# answer to the query — this is what demotes place names shared by filenames.
_REPEAT_PENALTY = 1.5
_PRIMARY_KIND_BONUS = 2.0
_ALL_CAPS_BONUS = 2.0
_LONG_NAME_BONUS = 1.0
_LONG_NAME_LEN = 6
_ALPHA_CODE_BONUS = 3.0
_NUMBERED_STANDARD_BONUS = 1.0


def _score_candidate(
    kind: HintKind,
    value: str,
    field: str,
    *,
    primary_kind: HintKind | None,
    all_caps: bool,
    alpha_prefixed: bool,
    row_hits: int,
) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    score = _FIELD_WEIGHTS.get(kind, {}).get(field, 0.0)
    reasons.append(f"field:{field}")
    if kind is primary_kind:
        score += _PRIMARY_KIND_BONUS
        reasons.append("primary_kind")
    if kind is HintKind.WHO:
        if all_caps:
            score += _ALL_CAPS_BONUS
            reasons.append("all_caps")
        if len(value) >= _LONG_NAME_LEN:
            score += _LONG_NAME_BONUS
            reasons.append("long_name")
    elif kind is HintKind.IDENTIFIER:
        if alpha_prefixed:
            score += _ALPHA_CODE_BONUS
            reasons.append("alpha_code")
    elif kind is HintKind.STANDARD:
        if any(char.isdigit() for char in value):
            score += _NUMBERED_STANDARD_BONUS
            reasons.append("numbered")
    if row_hits > 1:
        score -= _REPEAT_PENALTY * (row_hits - 1)
        reasons.append(f"repeated_in_rows:{row_hits}")
    return score, tuple(reasons)


def build_entity_hints(
    query: str,
    llm_results,
    max_candidates: int = MAX_CANDIDATES,
) -> EntityHints:
    """Collect and rank candidates from `llm_results` for the requested kinds.

    `llm_results` is the row sequence that will be rendered into ZDROJE (after
    PR9.3.3 packing when that flag is on). Rows are read, never mutated. An
    empty pool, an unsupported intent, or a pool with nothing extractable all
    return an EntityHints whose prompt block is "".
    """
    rows = list(llm_results or [])
    query_text = str(query or "")
    requested = _requested_kinds(fold(query_text))
    if not rows or not requested:
        return EntityHints(query=query_text, requested_kinds=requested, candidates=())

    query_words = frozenset(_WORD_UNIT_RE.findall(fold(query_text)))
    cap = max(1, min(int(max_candidates or MAX_CANDIDATES), MAX_CANDIDATES))
    primary_kind = requested[0]

    # Folded visible text per row, for the cross-row repetition signal.
    row_blobs: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            row_blobs.append(fold(" ".join(_field_text(row, f) for f in _VALUE_FIELDS)))
        else:
            row_blobs.append("")

    raw: list[tuple[HintKind, str, int, str, bool, bool]] = []
    for position, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        project_words = frozenset(_WORD_UNIT_RE.findall(fold(str(row.get("project") or ""))))
        blocked = _VOCABULARY_WORDS | _GENERIC_DOC_WORDS | query_words | project_words
        for field in _VALUE_FIELDS:
            text = _field_text(row, field)
            if not text:
                continue
            standards = _extract_standards(text)
            if HintKind.STANDARD in requested:
                for value, _span in standards:
                    raw.append((HintKind.STANDARD, value, position, field, False, False))
            if HintKind.IDENTIFIER in requested:
                spans = tuple(span for _v, span in standards)
                for value, alpha_prefixed in _extract_identifiers(text, spans, field):
                    raw.append((HintKind.IDENTIFIER, value, position, field, False, alpha_prefixed))
            if HintKind.WHO in requested and field in _WHO_FIELDS:
                for value, all_caps in _extract_who(text, blocked, query_words):
                    raw.append((HintKind.WHO, value, position, field, all_caps, False))

    scored: list[HintCandidate] = []
    seen: set[tuple[HintKind, str]] = set()
    for kind, value, position, field, all_caps, alpha_prefixed in raw:
        cleaned = " ".join(str(value or "").split())
        if not cleaned:
            continue
        folded_value = fold(cleaned)
        key = (kind, folded_value)
        if key in seen:
            continue
        row_hits = sum(1 for blob in row_blobs if folded_value and folded_value in blob)
        score, reasons = _score_candidate(
            kind, cleaned, field,
            primary_kind=primary_kind,
            all_caps=all_caps,
            alpha_prefixed=alpha_prefixed,
            row_hits=row_hits,
        )
        if score < MIN_SCORE:
            continue
        seen.add(key)
        scored.append(
            HintCandidate(
                kind=kind,
                value=cleaned,
                source_index=position,
                field=field,
                score=score,
                reasons=reasons,
            )
        )

    # Rank first, then apply the caps, so a per-row cap keeps the best
    # candidates for that row instead of the first ones encountered.
    kind_priority = {kind: index for index, kind in enumerate(requested)}
    scored.sort(
        key=lambda c: (
            kind_priority.get(c.kind, len(requested)),
            -c.score,
            c.source_index,
            fold(c.value),
        )
    )
    selected: list[HintCandidate] = []
    budget: dict[tuple[HintKind, int], int] = {}
    per_kind: dict[HintKind, int] = {}
    for candidate in scored:
        if len(selected) >= cap:
            break
        slot = (candidate.kind, candidate.source_index)
        if budget.get(slot, 0) >= PER_ROW_KIND_CAP:
            continue
        if per_kind.get(candidate.kind, 0) >= PER_KIND_CAP:
            continue
        budget[slot] = budget.get(slot, 0) + 1
        per_kind[candidate.kind] = per_kind.get(candidate.kind, 0) + 1
        selected.append(candidate)
    return EntityHints(
        query=query_text,
        requested_kinds=requested,
        candidates=tuple(selected),
    )
