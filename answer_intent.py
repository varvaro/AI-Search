"""PR9.3.4.1 — shared query-intent classification for the answer-side layers.

One implementation, used by both `context_packing` (PR9.3.3) and `entity_hints`
(PR9.3.4). Pure: no I/O, no retrieval, no LLM, and nothing here is reachable
from `ai_search.search()`, so retrieval and ranking are untouched by
construction.

    classify(query_folded, profile) -> QueryIntent
    packing_flags(query_folded)     -> dict   # what PR9.3.3 packing consumes
    hint_kinds(query_folded)        -> tuple  # what PR9.3.4 hints consume

WHY TWO PROFILES AND NOT ONE RULE SET
-------------------------------------
The PR9.3.4 live A/B found the two layers disagreeing: for "podle jakých
technických pravidel …" packing detected a standard intent while the hint layer
did not, so no hint was ever produced for that case. The fix is a single shared
implementation — but deliberately not a single shared rule set.

Packing's four flags feed multiplicative score boosts inside
`pack_answer_context`, so widening them changes which rows are selected, and
that selection is exactly what the PR9.3.3 live A/B validated. Two divergences
measured on the real query corpus: an order-number query phrased with the
inflected plural of "zakázka" yields identifier=False under the packing profile
(its `zakazk` alternative is anchored by `\b` and so never matches an inflected
form), and "jaké normy platí?" yields standard=False for the same reason. A
union rule set would flip both to True and silently re-rank the packed context.

So PACKING_PROFILE is frozen at the PR9.3.3 rules — moved here verbatim, with
`packing_flags()` returning the same dict `context_packing._intent()` returned
before — while HINT_PROFILE carries the corrected, inflection-tolerant rules
that only the hint layer reads. Changing PACKING_PROFILE requires re-running the
PR9.3.3 A/B; changing HINT_PROFILE does not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- profile: PR9.3.3 packing (FROZEN — do not widen without a new A/B) -------
_PACKING_WHO = re.compile(r"\b(kdo|dodavatel|zhotovitel|provadi|provadet|dela|delat)\b")
_PACKING_IDENTIFIER = re.compile(r"\b(zakazk|cislo|smluv|objednavk|\bnot\b)\b")
_PACKING_STANDARD = re.compile(r"\b(norma|norem|predpis|pravidel|pravidla|\btp\b|\bcsn\b|\ben\b)\b")
_PACKING_TYPE = re.compile(r"\b(jaky typ|jaka konstrukce|jaky druh|jaky typ|typ konstrukce)\b")

# --- profile: PR9.3.4.1 hints (inflection-tolerant) ---------------------------
# Czech declension is handled by open suffixes on the stem rather than by
# listing forms, which is what the frozen packing profile does and why it misses
# "zakázky" / "normy" / "smlouva".
_HINT_WHO = re.compile(
    r"\b(kdo|dodal\w*|dodala|dodali|dodava\w*|dodavatel\w*|zhotovitel\w*|"
    r"provadi\w*|provadel\w*|provede\w*|provedl\w*|provadet|vyrobil\w*|vyrobce|"
    r"autor\w*|firma|firmy|firmu|firmou|spolecnost\w*|subdodavatel\w*|dela|delat)\b"
)
_HINT_IDENTIFIER = re.compile(
    r"\b(cislo|cisl\w*|zakazk\w*|smlouv\w*|smluv\w*|objednavk\w*|not|"
    r"identifikator\w*|evidencn\w*|oznaceni\w*)\b"
)
_HINT_STANDARD = re.compile(
    r"\b(norm\w*|predpis\w*|pravidel|pravidla|pravidly|standard\w*|"
    r"\btp\b|\bcsn\b|\bcbs\b|\ben\b)\b"
)
_HINT_TYPE = _PACKING_TYPE


@dataclass(frozen=True)
class IntentProfile:
    """A named set of intent patterns. Immutable and shared, never per-query."""

    name: str
    who: re.Pattern
    identifier: re.Pattern
    standard: re.Pattern
    type: re.Pattern


PACKING_PROFILE = IntentProfile(
    name="packing_pr933",
    who=_PACKING_WHO,
    identifier=_PACKING_IDENTIFIER,
    standard=_PACKING_STANDARD,
    type=_PACKING_TYPE,
)

HINT_PROFILE = IntentProfile(
    name="hints_pr9341",
    who=_HINT_WHO,
    identifier=_HINT_IDENTIFIER,
    standard=_HINT_STANDARD,
    type=_HINT_TYPE,
)

# Stable order for hint kinds: the narrower, more answerable intents first.
_HINT_KIND_ORDER = ("WHO", "STANDARD", "IDENTIFIER")


@dataclass(frozen=True)
class QueryIntent:
    profile: str
    who: bool
    identifier: bool
    standard: bool
    type: bool

    def as_flags(self) -> dict[str, bool]:
        """Flag dict in the shape `context_packing.pack_answer_context` reads."""
        return {
            "who": self.who,
            "identifier": self.identifier,
            "standard": self.standard,
            "type": self.type,
        }

    def kinds(self) -> tuple[str, ...]:
        """Requested hint kinds, in stable order. TYPE has no hint extractor."""
        active = {
            "WHO": self.who,
            "STANDARD": self.standard,
            "IDENTIFIER": self.identifier,
        }
        return tuple(name for name in _HINT_KIND_ORDER if active[name])


def classify(query_folded: str, profile: IntentProfile) -> QueryIntent:
    """Classify an ALREADY FOLDED query against `profile`.

    Folding stays with the caller so this module does not introduce a fourth
    normalization contract; every caller already folds with the same
    casefold + strip-combining-marks algorithm.
    """
    text = query_folded or ""
    return QueryIntent(
        profile=profile.name,
        who=bool(profile.who.search(text)),
        identifier=bool(profile.identifier.search(text)),
        standard=bool(profile.standard.search(text)),
        type=bool(profile.type.search(text)),
    )


def packing_flags(query_folded: str) -> dict[str, bool]:
    return classify(query_folded, PACKING_PROFILE).as_flags()


def hint_kinds(query_folded: str) -> tuple[str, ...]:
    return classify(query_folded, HINT_PROFILE).kinds()
