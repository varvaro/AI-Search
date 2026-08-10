"""Query Understanding layer - conservative, dictionary-driven query expansion
that runs BEFORE retrieval (see `expand_query=` on ai_search.search()).

WHY THIS EXISTS
The 2026-08-07 retrieval audit found the dominant production failure mode is
NOT ranking/precision but RECALL through vocabulary mismatch: the answering
document is indexed and extractable, yet never enters any candidate pool
because the user's register and the document's register differ.

    query "Jaké doklady musí dodat zhotovitel po betonáži?"
    document "Kontrola dodacích listů betonové směsi" / "Protokoly o kvalitě betonů"
        -> best_rank_fts = None (absent from BM25 top-600), best_rank_vector = 312

    query "změnový list přístavba zázemí Národního domu"
    document "Garáže NDS_prehled ZL GD akt. <date>.xlsx"
        -> best_rank_fts = None AND best_rank_vector = None (absent from both top-100s)

No reranker can promote a candidate that never entered a pool, which is why the
cross-encoder experiment could not fix either case. The fix has to happen
before retrieval, and for a closed, finite domain vocabulary (construction
terminology, ZL/SoD/KZP abbreviations, project aliases) a curated dictionary is
both the cheapest and the only deterministic option - no LLM call is added to a
pipeline whose LLM stage is already the dominant latency cost (35-190 s/query).

SAFETY PROPERTIES (all deliberate, all tested in tests/test_query_expansion.py)
  * The original query is NEVER modified or replaced - `QueryExpansion.original`
    is preserved verbatim and remains what filename matching and every scoring
    formula in ai_search.search() sees. Expansion only ever ADDS OR-terms to the
    FTS5 MATCH expression and appends terms to the text handed to the embedder.
  * Expansion is emit-directional, never bidirectional. Only a rule's own KEY,
    `abbreviations` and `synonyms` (tight, same-concept surface forms) can
    TRIGGER it; `documents` and `processes` are emit-only. So "betonáž" may
    expand towards "dodací list", but a query about "dodací list" is never
    dragged back to "betonáž" - the loose relation is not invertible and
    inverting it is what would flood unrelated queries with noise.
  * Bounded blast radius: at most MAX_EXPANSION_TERMS terms per query, chosen by
    a two-level round robin (categories within a rule, then rules against each
    other) so neither a single concept nor a single category can consume the
    whole budget - see `_rule_term_stream`.
  * Terms already present in the query are never re-added.

NOT IN SCOPE HERE (deliberately, see the 2026-08-07 architecture review):
LLM query rewrite, HyDE, contextual retrieval, embedding-model changes and
cross-encoder work are all out of scope for this layer.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

_logger = logging.getLogger("ai_search.query_expansion")

# Upper bound on how many terms a single query may gain. The audit's
# recommendation was 5-8; 8 is the ceiling, and the round-robin selection below
# means a query matching several rules spends the budget across all of them
# instead of exhausting it on the first.
MAX_EXPANSION_TERMS = 8

# A dictionary surface form shorter than this must match a query token exactly;
# longer ones may match as a prefix, which is what makes Czech declension work
# ("betonáži" triggers the "betonáž" rule) without a stemmer. Short forms are
# exact-only because a 2-3 character prefix match ("ZL", "TP", "KD") would fire
# on far too many unrelated tokens.
MIN_PREFIX_MATCH_LENGTH = 4

# Only these categories may TRIGGER a rule (together with the rule's key
# itself). See the "emit-directional" safety property above.
TRIGGER_CATEGORIES = ("abbreviations", "synonyms")
# Order in which a matched rule's categories are consumed by the round robin -
# highest-confidence first, so a tight budget spends itself on the safest terms.
EMIT_CATEGORIES = ("abbreviations", "synonyms", "processes", "documents")

# --- General construction-domain vocabulary ---------------------------------
# Curated from general Czech construction/QA terminology and the document
# taxonomy actually present in the corpus - deliberately NOT reverse-engineered
# from the two known failing benchmark queries. Those two cases are covered as a
# side effect of the general "betonáž"/"doklady"/"změnový list" entries, which
# is the point: an entry that only helps one benchmark question is overfitting,
# an entry a site engineer would recognise as ordinary domain vocabulary is not.
#
# Category meanings:
#   abbreviations - unambiguous 1:1 short forms (highest confidence, safe to expand)
#   synonyms      - same-concept surface variants (high confidence)
#   processes     - the construction activity the concept belongs to (medium)
#   documents     - document GENRES that typically carry the answer (loosest,
#                   highest false-positive risk - hence last in EMIT_CATEGORIES
#                   and never a trigger)
#   scope         - audit/curation tag, lets a future regression report break
#                   expansion quality down per domain area
CONSTRUCTION_VOCABULARY: dict[str, dict] = {
    "betonáž": {
        "scope": "construction_process",
        "synonyms": ["betonování", "beton", "betonová směs", "čerstvý beton"],
        "documents": ["dodací list", "zkouška betonu", "protokol o kvalitě betonu", "kniha betonáže", "krychelná pevnost"],
    },
    "výztuž": {
        "scope": "construction_process",
        "synonyms": ["armatura", "armování", "vyztužení", "betonářská ocel"],
        "documents": ["protokol o převzetí výztuže", "zápis o kontrole výztuže", "hutní atest", "výkres výztuže"],
    },
    "bednění": {
        "scope": "construction_process",
        "synonyms": ["bednící systém", "odbednění"],
        "processes": ["betonáž"],
        "documents": ["technologický postup", "protokol o převzetí bednění"],
    },
    "pilota": {
        "scope": "construction_process",
        "synonyms": ["piloty", "mikropilota", "vrtaná pilota"],
        "documents": ["výrobní dokumentace pilot", "protokol o provedení piloty", "technologický postup"],
    },
    "hydroizolace": {
        "scope": "construction_process",
        "synonyms": ["izolace proti vodě", "vodotěsná izolace", "těsnění spár"],
        "documents": ["technický list", "montážní návod", "protokol o zkoušce těsnosti"],
    },
    "kzp": {
        "scope": "documentation_type",
        "abbreviations": ["KZP"],
        "synonyms": ["kontrolní a zkušební plán"],
        "documents": ["kontrolní bod", "zkušební plán", "plán kontrol"],
    },
    "tp": {
        "scope": "documentation_type",
        "abbreviations": ["TP"],
        "synonyms": ["technologický postup", "technický předpis"],
        "documents": ["pracovní postup", "technologický předpis"],
    },
    "dsps": {
        "scope": "documentation_type",
        "abbreviations": ["DSPS"],
        "synonyms": ["dokumentace skutečného provedení stavby"],
        "documents": ["dokumentace skutečného provedení", "výkresová dokumentace"],
    },
    "stavební deník": {
        "scope": "documentation_type",
        "synonyms": ["deník stavby"],
        "documents": ["zápis do stavebního deníku", "denní záznam"],
    },
    "sod": {
        "scope": "contract",
        "abbreviations": ["SoD"],
        "synonyms": ["smlouva o dílo"],
        "documents": ["dodatek smlouvy", "příloha smlouvy", "obchodní podmínky"],
    },
    "bozp": {
        "scope": "safety",
        "abbreviations": ["BOZP"],
        "synonyms": ["bezpečnost práce", "bezpečnost a ochrana zdraví při práci"],
        "documents": ["plán BOZP", "koordinátor BOZP", "školení BOZP", "registr rizik"],
    },
    "změnový list": {
        "scope": "project_administration",
        "abbreviations": ["ZL"],
        "synonyms": ["změnové řízení", "změna díla"],
        "documents": ["přehled změn", "přehled ZL", "soupis změn", "vícepráce", "méněpráce"],
    },
    "kontrolní den": {
        "scope": "project_administration",
        "abbreviations": ["KD"],
        "synonyms": ["kontrolní dny"],
        "documents": ["zápis z kontrolního dne", "zápis z jednání", "protokol z porady"],
    },
    "fakturace": {
        "scope": "finance",
        "synonyms": ["faktura", "úhrada", "platba"],
        "documents": ["zjišťovací protokol", "soupis provedených prací", "daňový doklad", "dílčí faktura"],
    },
    "předání stavby": {
        "scope": "handover",
        "synonyms": ["předání díla", "převzetí stavby", "předání a převzetí"],
        "documents": ["předávací protokol", "zápis o předání", "předávací dokumentace", "dokumentace skutečného provedení"],
    },
    "doklady": {
        "scope": "generic_document_request",
        "synonyms": ["dokumenty", "podklady", "doklad"],
        "documents": ["protokol", "certifikát", "atest", "prohlášení o shodě", "dodací list", "zkušební protokol"],
    },
    "geodetické zaměření": {
        "scope": "survey",
        "synonyms": ["geodet", "geodetické měření", "zaměření"],
        "documents": ["geodetický protokol", "geodetický předávací protokol", "výškopis", "polohopis"],
    },
    "revize": {
        "scope": "quality",
        "synonyms": ["revizní zpráva"],
        "documents": ["protokol o revizi", "výchozí revize", "revizní technik"],
    },
    "vada": {
        "scope": "quality",
        "synonyms": ["závada", "nedodělek", "reklamace"],
        "documents": ["seznam vad a nedodělků", "protokol o odstranění vad", "reklamační protokol"],
    },
    # Query Expansion 2.0 MVP: human phrase "bludné proudy" vs project part code
    # D.1.4.j used in filenames/folders (rr-bp-vyvody-3pp-01). Emit the CODE, not
    # the 2-letter abbreviation "BP" - short abbrs (BP/TP/ZL/KD) historically
    # flooded FTS/filename matching with false positives. "BP" is intentionally
    # absent from abbreviations/synonyms so a bare "BP" query never triggers this.
    "bludné proudy": {
        "scope": "discipline",
        "documents": ["D.1.4.j"],
    },
    # Surface-preparation synonymy (rr-brokovani-zakladova-deska-3pp-01): site
    # language says "brokovat/brokování" while bills of quantities and supplier
    # VV sheets title the same process as "otryskání" and list the techniques
    # side by side (corpus: "Otryskání podkladu … brokování, broušení").
    # Both emits are under `documents` (emit-only) so a query that already says
    # otryskání/broušení is not dragged sideways into brokování expansions.
    # A/B on the production index (2026-08-10): emit "otryskání" alone or with
    # "otryskání podkladu" enters the BM25 pool but still misses search_all
    # top-10; adding the co-listed technique "broušení" reaches top-10 without
    # a construction-context gate and without suite regressions. Only
    # "brokovat" is listed as a synonym trigger - multi-word elaborations
    # ("brokování podkladu/betonu") still match via the key token alone, and
    # listing them as synonyms would re-emit those phrases into FTS and dilute
    # the rare-term signal again. Czech declension of the key ("brokováním")
    # is covered by MIN_PREFIX_MATCH_LENGTH prefix matching; "brokovat" cannot
    # prefix-match "brokování", so it must be an explicit synonym.
    "brokování": {
        "scope": "construction_process",
        "synonyms": ["brokovat"],
        "documents": ["otryskání", "broušení"],
    },
}

# --- Deployment-specific project aliases ------------------------------------
# Kept SEPARATE from the general vocabulary above on purpose: this is instance
# data (one customer's project naming), not construction terminology, and it
# carries a different maintenance and overfitting profile - it must be re-curated
# per deployment, and it is the part a reviewer should scrutinise hardest for
# "was this added just to pass a benchmark case". Split out so that question can
# be answered by looking at one small, clearly-labelled dict.
#
# The entry below encodes a real, documented naming inconsistency in the source
# data (recorded during the 2026-08-06 ground-truth repair): the same project
# appears as the internal short code "Garáže NDS" and as the contractual name
# "Přístavba zázemí Národního domu" across its own documents.
PROJECT_ALIASES: dict[str, dict] = {
    "národní dům": {
        "scope": "project_alias",
        "abbreviations": ["NDS"],
        # "Národního domu" is listed explicitly because the per-word prefix rule
        # in _surface_matches() cannot bridge a Czech vowel alternation: the
        # genitive "domu" does not start with the nominative stem "dum". Where
        # declension changes the stem rather than only appending to it, the
        # declined surface form has to be an explicit synonym.
        "synonyms": ["Národní dům Smíchov", "Národního domu"],
        "documents": ["Garáže NDS", "přístavba zázemí Národního domu"],
    },
}

DOMAIN_VOCABULARY: dict[str, dict] = {**CONSTRUCTION_VOCABULARY, **PROJECT_ALIASES}


def _fold(text: str) -> str:
    """casefold + strip diacritics. Mirrors what the FTS5 index itself does
    (`tokenize='unicode61 remove_diacritics 2'`), so dictionary matching agrees
    with how the index will later tokenize the terms this module emits."""
    return "".join(c for c in unicodedata.normalize("NFKD", (text or "").casefold()) if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", _fold(text))


# Floor / storey notation: users type "3PP", drawings and filenames often use
# "3.PP" or "3 PP". FTS tokenizes on non-word chars, so "3PP" (one token) does
# not match "3.PP" (tokens "3","PP") without an explicit bridge. Match every
# NPP / N.PP / N PP form in the query and emit the missing siblings as
# expansion terms - pure notation aliases, not domain synonyms.
_FLOOR_PP_PATTERN = re.compile(r"(?<!\w)(\d+)\s*\.?\s*pp(?!\w)", re.IGNORECASE)


def _floor_pp_forms(level: str) -> tuple[str, str, str]:
    """Canonical alternate spellings for one floor level number."""
    return (f"{level}PP", f"{level}.PP", f"{level} PP")


def _floor_level_expansion_terms(query: str) -> list[str]:
    """Return floor-notation variants present in spirit but not yet usable as
    FTS bridges for `query`. Empty when the query names no NPP/N.PP/N PP form.
    Deterministic and order-stable: ascending level, then (NPP, N.PP, N PP).
    "3.PP" and "3 PP" share the same FTS tokens, so only the first surviving
    form per token-set is kept (prefers dotted "N.PP" over spaced "N PP")."""
    present = set(_tokens(query))
    selected: list[str] = []
    seen_levels: set[str] = set()
    seen_token_sets: set[frozenset[str]] = set()
    for match in _FLOOR_PP_PATTERN.finditer(query or ""):
        level = match.group(1)
        if level in seen_levels:
            continue
        seen_levels.add(level)
        for form in _floor_pp_forms(level):
            form_tokens = set(_tokens(form))
            if not form_tokens or form_tokens <= present:
                continue
            token_key = frozenset(form_tokens)
            if token_key in seen_token_sets:
                continue
            seen_token_sets.add(token_key)
            selected.append(form)
    return selected


def _surface_matches(surface: str, query_tokens: list[str]) -> bool:
    """True when every word of `surface` is present in `query_tokens`, exactly
    for short words and as a prefix for longer ones. The per-word prefix rule is
    what lets "Národního domu" trigger the "národní dům" rule and "betonáži"
    trigger "betonáž" without any stemming - the query token must EXTEND the
    dictionary word, never the other way round, so a rule can only fire on a
    query that actually contains its (possibly declined) term."""
    words = _tokens(surface)
    if not words:
        return False
    for word in words:
        if word in query_tokens:
            continue
        if len(word) >= MIN_PREFIX_MATCH_LENGTH and any(token.startswith(word) for token in query_tokens):
            continue
        return False
    return True


def _rule_term_stream(rule: dict) -> list[str]:
    """One rule's emittable terms, round-robined ACROSS its categories rather
    than concatenated. Concatenation would let a rule's several synonyms consume
    the entire per-query budget before a single `documents` term is reached -
    and `documents` terms ("dodací list", "zkouška betonu") are precisely the
    ones that bridge the register gap this layer exists to close."""
    buckets = [list(rule.get(category, ())) for category in EMIT_CATEGORIES]
    stream: list[str] = []
    for index in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if index < len(bucket):
                stream.append(bucket[index])
    return stream


@dataclass
class QueryExpansion:
    """Result of one expansion pass. `original` is authoritative: callers must
    keep using it for anything other than the two widened retrieval inputs."""

    original: str
    terms: list[str] = field(default_factory=list)
    matched_rules: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.terms)

    @property
    def embedding_text(self) -> str:
        """Text to embed for the vector branch: the untouched query first (so it
        keeps dominating the resulting vector), expansion terms appended."""
        return f"{self.original} {' '.join(self.terms)}".strip() if self.terms else self.original


def expand_query(query: str, vocabulary: dict[str, dict] | None = None, max_terms: int = MAX_EXPANSION_TERMS) -> QueryExpansion:
    """Detect domain terms in `query` and return the terms retrieval should be
    widened with. Pure and deterministic: same query -> same terms, no I/O, no
    model, no network. Returns an empty QueryExpansion (falsy) when nothing
    matches, which callers treat as "behave exactly as before"."""
    vocab = DOMAIN_VOCABULARY if vocabulary is None else vocabulary
    query_tokens = _tokens(query)
    if not query_tokens:
        return QueryExpansion(original=query)

    present = set(query_tokens)
    matched: list[dict] = []
    streams: list[list[str]] = []

    # Floor notation bridges first: tiny, high-precision, and they must not lose
    # the MAX_EXPANSION_TERMS race to a multi-rule domain match on the same query
    # (e.g. "půdorys 3PP bludné proudy" needs both 3.PP and D.1.4.j).
    floor_terms = _floor_level_expansion_terms(query)
    if floor_terms:
        matched.append({
            "key": "floor_level_pp",
            "scope": "notation",
            "trigger": "floor_pp",
            "terms": list(floor_terms),
        })
        streams.append(list(floor_terms))

    for key, rule in vocab.items():
        triggers = [key] + [surface for category in TRIGGER_CATEGORIES for surface in rule.get(category, ())]
        trigger = next((surface for surface in triggers if _surface_matches(surface, query_tokens)), None)
        if trigger is None:
            continue
        # A term whose every word is already in the query adds nothing to either
        # branch - it would only consume budget and inflate the trace.
        stream = [term for term in _rule_term_stream(rule) if not set(_tokens(term)) <= present]
        if not stream:
            continue
        matched.append({"key": key, "scope": rule.get("scope"), "trigger": trigger, "terms": stream})
        streams.append(stream)

    selected: list[str] = []
    seen: set[str] = set()
    for index in range(max((len(stream) for stream in streams), default=0)):
        if len(selected) >= max_terms:
            break
        for stream in streams:
            if len(selected) >= max_terms:
                break
            if index >= len(stream):
                continue
            term = stream[index]
            folded = _fold(term)
            if folded in seen:
                continue
            seen.add(folded)
            selected.append(term)

    for rule in matched:
        rule["terms"] = [term for term in rule["terms"] if term in selected]

    expansion = QueryExpansion(original=query, terms=selected, matched_rules=[rule for rule in matched if rule["terms"]])
    if expansion.terms:
        _logger.info(
            "QUERY_EXPANSION query=%r rules=%s terms=%s",
            query, [rule["key"] for rule in expansion.matched_rules], expansion.terms,
        )
    return expansion
