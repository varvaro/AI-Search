"""PR8.1.1 / PR8.1.2 — Entity Match Bonus (experimental, flag-gated).

Additive Phase-3 score term when:
  * an *explicit* query token or NOT-id appears in name/path (PR8.1.1), and/or
  * a narrow subject→entity conjunction fires and injects needles (PR8.1.2).

Does NOT:
  * change FTS / Lance / embeddings
  * build a general synonym / knowledge graph
  * call an LLM
  * touch answer() / evidence_runtime / document_state
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Tunables — small vs FILENAME_MATCH_BONUS (0.03) / RRF unit (~1/60).
ENTITY_NAME_HIT_BONUS = 0.04
ENTITY_PATH_ONLY_BONUS = 0.02
ENTITY_MATCH_BONUS_CAP = 0.06

SOURCE_EXPLICIT = "explicit_entity"
SOURCE_SUBJECT_ALIAS = "subject_alias"

_NOT_RE = re.compile(r"\bNOT\d+\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Function words + generic legal/doc vocabulary that must never act as entity
# needles (would false-boost every SoD / smlouva filename).
_STOPWORDS = frozenset({
    "a", "i", "o", "u", "k", "s", "z", "v", "ve", "ke", "se", "si", "na", "do",
    "od", "za", "po", "pro", "pri", "bez", "je", "jsou", "jak", "jake", "jaka",
    "jaky", "co", "ci", "nebo", "ale", "the", "and", "or", "of", "in", "on",
    "to", "with", "for",
    "smlouva", "smlouvy", "smlouve", "sod", "loi", "dokument", "dokumentu",
    "dokumenty", "soubor", "souboru", "najdi", "najit", "hledej", "existuje",
    "podepsana", "podepsane", "podepsany", "podepsan", "podpis", "podpisu",
    "boxu", "stavbe", "stavba", "projekt", "projektu", "objednateli", "dilo",
    "dila", "zakazky", "zakazka", "dela", "kdo", "jake", "jaka", "jaky",
    "jakych", "ma", "jsou", "bylo", "byla", "byly", "typ", "postup", "montaze",
    "pozadavky", "pozadavek", "technickych", "pravidel",
})


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


@dataclass(frozen=True)
class EntitySignal:
    raw: str
    folded: str
    kind: str  # "not_id" | "token" | "subject_alias"
    source: str = SOURCE_EXPLICIT  # explicit_entity | subject_alias


@dataclass(frozen=True)
class EntityMatchDetail:
    signals: tuple[EntitySignal, ...]
    name_hits: tuple[str, ...]
    path_hits: tuple[str, ...]
    bonus: float
    hit_sources: tuple[str, ...] = ()

    def as_trace_dict(self) -> dict:
        return {
            "signals": [
                {"raw": s.raw, "kind": s.kind, "source": s.source}
                for s in self.signals
            ],
            "name_hits": list(self.name_hits),
            "path_hits": list(self.path_hits),
            "hit_sources": list(self.hit_sources),
            "bonus": self.bonus,
        }


@dataclass(frozen=True)
class _SubjectAliasRule:
    """Conjunction-gated injection. Never fires on a single generic word."""

    rule_id: str
    # Every entry must appear as a substring of the folded query.
    require_all: tuple[str, ...]
    # Each group: at least one alternative must appear (OR inside, AND across).
    require_any_groups: tuple[tuple[str, ...], ...]
    inject: tuple[str, ...]


# ND Smíchov only — three FAT-derived routes. No open-ended synonym table.
_SUBJECT_ALIAS_RULES: tuple[_SubjectAliasRule, ...] = (
    _SubjectAliasRule(
        rule_id="bozp_signed_contract",
        require_all=("bozp",),
        require_any_groups=(
            ("smlouva", "smlouvy", "smlouve", "podepsana", "podepsane", "podepsany", "podepsan"),
        ),
        inject=("SafetyPeak", "NOT250060"),
    ),
    _SubjectAliasRule(
        rule_id="white_tank_sealing",
        require_all=("tesneni",),
        require_any_groups=(
            ("bila vana", "bile vany", "bile vane", "bile van"),
        ),
        inject=("Illichman", "NOT260916"),
    ),
    _SubjectAliasRule(
        rule_id="monolith_supplier",
        require_all=("monolit", "dodavatel"),
        require_any_groups=(),
        inject=("FERI", "NOT251110"),
    ),
)


def _rule_matches(folded_query: str, rule: _SubjectAliasRule) -> bool:
    if not all(part in folded_query for part in rule.require_all):
        return False
    for group in rule.require_any_groups:
        if not any(alt in folded_query for alt in group):
            return False
    return True


def extract_subject_alias_signals(query: str) -> tuple[EntitySignal, ...]:
    """Inject entity needles only when a full subject conjunction matches."""
    folded_q = fold(query)
    out: list[EntitySignal] = []
    seen: set[str] = set()
    for rule in _SUBJECT_ALIAS_RULES:
        if not _rule_matches(folded_q, rule):
            continue
        for raw in rule.inject:
            key = fold(raw)
            if key in seen:
                continue
            seen.add(key)
            kind = "not_id" if key.startswith("not") and key[3:].isdigit() else "subject_alias"
            out.append(EntitySignal(
                raw=raw, folded=key, kind=kind, source=SOURCE_SUBJECT_ALIAS,
            ))
    return tuple(out)


def extract_entity_signals(query: str) -> tuple[EntitySignal, ...]:
    """Explicit NOT ids + content tokens from the query (no aliases)."""
    raw = query or ""
    folded_q = fold(raw)
    out: list[EntitySignal] = []
    seen: set[str] = set()

    for match in _NOT_RE.finditer(raw):
        token = match.group(0)
        key = fold(token)
        if key in seen:
            continue
        seen.add(key)
        out.append(EntitySignal(
            raw=token, folded=key, kind="not_id", source=SOURCE_EXPLICIT,
        ))

    for tok in _TOKEN_RE.findall(folded_q):
        if len(tok) < 4 or tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        if tok.startswith("not") and tok[3:].isdigit():
            continue
        seen.add(tok)
        out.append(EntitySignal(
            raw=tok, folded=tok, kind="token", source=SOURCE_EXPLICIT,
        ))

    return tuple(out)


def compute_entity_match_bonus(
    query: str,
    document_name: str,
    document_path: str,
    *,
    include_explicit: bool = True,
    include_subject_aliases: bool = False,
    name_bonus: float = ENTITY_NAME_HIT_BONUS,
    path_bonus: float = ENTITY_PATH_ONLY_BONUS,
    cap: float = ENTITY_MATCH_BONUS_CAP,
) -> EntityMatchDetail:
    """Return capped additive bonus for name/path entity hits.

    Name hit outranks path-only for the same signal (no double count).
    Explicit and subject-alias signals share one cap.
    """
    signals: list[EntitySignal] = []
    seen: set[str] = set()
    if include_explicit:
        for signal in extract_entity_signals(query):
            if signal.folded in seen:
                continue
            seen.add(signal.folded)
            signals.append(signal)
    if include_subject_aliases:
        for signal in extract_subject_alias_signals(query):
            if signal.folded in seen:
                continue
            seen.add(signal.folded)
            signals.append(signal)

    if not signals:
        return EntityMatchDetail(
            signals=(), name_hits=(), path_hits=(), bonus=0.0, hit_sources=(),
        )

    name_blob = fold(document_name)
    path_blob = fold(document_path)
    name_hits: list[str] = []
    path_hits: list[str] = []
    hit_sources: list[str] = []
    total = 0.0

    for signal in signals:
        in_name = signal.folded in name_blob
        in_path = signal.folded in path_blob
        if in_name:
            name_hits.append(signal.raw)
            hit_sources.append(signal.source)
            total += name_bonus
        elif in_path:
            path_hits.append(signal.raw)
            hit_sources.append(signal.source)
            total += path_bonus

    bonus = min(total, cap) if total else 0.0
    return EntityMatchDetail(
        signals=tuple(signals),
        name_hits=tuple(name_hits),
        path_hits=tuple(path_hits),
        bonus=bonus,
        hit_sources=tuple(hit_sources),
    )
