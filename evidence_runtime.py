"""EvidenceSet runtime validation foundation (PR7.0, ADR-007).

Pure, deterministic aggregation ON TOP of the existing foundation layers —
this module adds only the two pieces ADR-007 identified as missing, plus the
container that a future runtime consumer will fill in:

    StateRequirement + DocumentStateEvidence[] → build_state_coverage → StateCoverage
    EvidenceSpan[]                             → derive_evidence_needs → EvidenceNeed[]

Everything else is REUSED, not redefined: EvidenceSet/EvidenceSpan (PR3),
IntentRequirement/EvidenceCoverage/EvidenceNeed (PR4) — `EvidenceCoverage`
*is* the intent coverage type, there is deliberately no second one — and
DocumentState/DocumentStateEvidence/StateRequirement (W1).

This module does NOT (ADR-007 "mimo scope"):
  * wire into ai_search.answer() / search() or ui_services.search_all()
  * change retrieval, RRF, fusion, scoring, bonuses, QE, or result order
  * filter, reorder, promote, or otherwise touch retrieval rows
  * call SQLite / LanceDB / Ollama / Streamlit / the network
  * classify documents from filenames (that is document_state.py's job and it
    is intentionally NOT reused for need derivation — see derive_evidence_needs)
  * consult an LLM for any decision
  * store state at index time
  * rewrite answers — GateAction only *records* what a future gate did

Nothing imports this module yet; PR7.0 is foundation only.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from document_classification import (
    detect_project_conflict,
    query_requires_signed_contract,
    signed_evidence_is_contract_for_query,
    subject_doc_type_terms,
)
from document_state import DocumentState, DocumentStateEvidence, StateRequirement
from evidence import EvidenceSet, EvidenceSpan
from intent_requirements import EvidenceCoverage, EvidenceNeed
from query_facets import FacetType, QueryFacet

RULES_VERSION = "rules_v1"


class StateVerdict(str, Enum):
    """Aggregated lifecycle verdict over ALL candidate documents.

    W1 classifies one document at a time; this is the multi-document
    aggregation W1's docstring explicitly left out of scope.

    NOOP              query has no SIGNED state intent — layer is inert.
    SIGNED_CONFIRMED  a SIGNED *contract*/SoD exists among the candidates
                      (PR7.6.1: LOI / order / minutes no longer qualify when
                      the query asks for a smlouva). Alias of
                      SIGNED_CONTRACT_CONFIRMED — same wire value so PR7.4
                      benchmarks that expect SIGNED_CONFIRMED keep matching.
    SIGNED_CONTRACT_CONFIRMED
                      Explicit name for SIGNED_CONFIRMED (Enum alias).
    SIGNED_OTHER_DOCUMENT_CONFIRMED
                      A signed non-contract document (LOI, order, …) exists,
                      but it must NOT be read as a signed smlouva/SoD.
    UNSIGNED_CONFIRMED every candidate is conclusively unsigned
                      (FOR_SIGNATURE / DRAFT / TEMPLATE) — a negative claim is
                      accurate.
    UNVERIFIED        no SIGNED document, but the state cannot be ruled out
                      (an UNKNOWN candidate, an empty candidate pool, or only
                      a partial entity match).
    ENTITY_MISMATCH   the query names an entity and NO candidate document
                      matches it — nothing may be confirmed or denied.
    CONFLICT          RESERVED, never derived (PR7.0.1). See
                      build_state_coverage for why lifecycle coexistence is
                      not a conflict and what signals a real one would need.
    """

    NOOP = "NOOP"
    SIGNED_CONFIRMED = "SIGNED_CONFIRMED"
    SIGNED_CONTRACT_CONFIRMED = "SIGNED_CONFIRMED"  # alias — same value
    SIGNED_OTHER_DOCUMENT_CONFIRMED = "SIGNED_OTHER_DOCUMENT_CONFIRMED"
    UNSIGNED_CONFIRMED = "UNSIGNED_CONFIRMED"
    UNVERIFIED = "UNVERIFIED"
    ENTITY_MISMATCH = "ENTITY_MISMATCH"
    CONFLICT = "CONFLICT"


class EvidenceSafetyStatus(str, Enum):
    """Retrieval-evidence safety for answer abstention (PR7.6.1).

    Orthogonal to StateVerdict: this judges whether the retrieved pool is
    strong enough to support *any* factual claim, not the signedness of a
    contract.
    """

    OK = "OK"
    NO_EVIDENCE = "NO_EVIDENCE"
    UNVERIFIED = "UNVERIFIED"
    DOCUMENT_PROJECT_CONFLICT = "DOCUMENT_PROJECT_CONFLICT"


@dataclass(frozen=True)
class EvidenceSafety:
    """Outcome of evaluate_evidence_safety() — consumed by the answer gate."""

    status: EvidenceSafetyStatus
    message: str = ""
    conflicted_documents: tuple[str, ...] = ()
    source: str = RULES_VERSION


class GateAction(str, Enum):
    """What a validation gate DID to a rendered answer (diagnostics only).

    PR7.0 never rewrites anything — this records the outcome so a future
    consumer's behaviour is observable instead of inferred from the text.
    """

    PASSTHROUGH = "PASSTHROUGH"
    REWRITTEN_POSITIVE = "REWRITTEN_POSITIVE"
    REWRITTEN_UNVERIFIED = "REWRITTEN_UNVERIFIED"
    REWRITTEN_NEGATIVE = "REWRITTEN_NEGATIVE"


@dataclass(frozen=True)
class StateCoverage:
    """Lifecycle-state coverage of the candidate documents for one query.

    `evidences` carries ONLY the documents that actually support `verdict`
    (citable evidence), never the unfiltered pool — an entity mismatch
    therefore carries none at all, because citing an unrelated party's
    contract is exactly the PR6.1 failure this layer exists to prevent.
    """

    requirement: StateRequirement
    evidences: tuple[DocumentStateEvidence, ...]
    entity_matched: bool
    states_present: frozenset[DocumentState]
    verdict: StateVerdict
    conflict: bool


@dataclass(frozen=True)
class AnswerValidation:
    """Container for one answer's validation result (ADR-007 lifecycle).

    Every analytical field is optional: a consumer that only evaluates state
    (today's PR6 gate) must be able to record that honestly instead of
    fabricating an empty EvidenceSet/coverage that would read as "evaluated
    and found nothing".
    """

    query: str
    facets: tuple[QueryFacet, ...] = ()
    evidence_set: EvidenceSet | None = None
    intent_coverage: EvidenceCoverage | None = None
    state_coverage: StateCoverage | None = None
    evidence_safety: EvidenceSafety | None = None
    gate_action: GateAction = GateAction.PASSTHROUGH
    source: str = RULES_VERSION


# ---------------------------------------------------------------------------
# EvidenceSpan → EvidenceNeed adapter
# ---------------------------------------------------------------------------

# Stable output order, mirroring intent_requirements._NEED_ORDER (that helper
# is private; a 5-item tuple is not worth widening its public API for).
_NEED_ORDER = (
    EvidenceNeed.TECHNOLOGY,
    EvidenceNeed.STRUCTURE,
    EvidenceNeed.CONTRACT,
    EvidenceNeed.COST,
    EvidenceNeed.QUALITY,
)

# The ONLY two mappings that survive a false-positive review:
#   ACTION - a process/action term verified in the chunk text is technology
#            (process) evidence.
#   OBJECT - a construction element verified in the chunk text is structure
#            (build-up) evidence.
# Deliberately NOT mapped:
#   LOCATION - "3.PP" is where, not what; it evidences neither need.
#   DOC_TYPE - a technical procedure that merely mentions "smlouva" is not
#              CONTRACT evidence. CONTRACT/COST/QUALITY have no signal here
#              that is safe without content classification, so PR7.0 never
#              derives them (false negative by design).
#   ACTOR / OTHER - never in MULTI_QUERY_GATE_TYPES, never on a span.
_FACET_NEED_MAP = {
    FacetType.ACTION: EvidenceNeed.TECHNOLOGY,
    FacetType.OBJECT: EvidenceNeed.STRUCTURE,
}


def derive_evidence_needs(spans: Iterable[EvidenceSpan] | None) -> tuple[EvidenceNeed, ...]:
    """Derive which evidence needs the given spans actually satisfy.

    This closes the gap PR4 left open on purpose: build_evidence_coverage()
    takes caller-supplied need labels and PR4 refused to guess them. The only
    signal used here is `EvidenceSpan.facet_types`, which evidence.py assigns
    exclusively via conservative whole-phrase matching against the chunk's own
    `quote` (no substring stems, multi-word phrases must be consecutive).

    Filenames, paths, projects, scores, and retrieval provenance
    (`subquery_ids`) are NEVER consulted:
      * a filename classifier here would let "SoD_xy.pdf" fabricate CONTRACT
        evidence that no retrieved text supports
      * a row found BY an ACTION subquery is not ACTION evidence (PR3's core
        invariant) — provenance is not proof

    Conservative by construction: an unrecognized or unmatched span
    contributes nothing, and an empty result is a valid, safe answer. Because
    an empty IntentRequirement yields CoverageStatus.COMPLETE (PR4), under-
    reporting here can never block a query that works today; over-reporting
    could silently legitimize a thin answer. False negatives preferred.
    """
    found: set[EvidenceNeed] = set()
    for span in spans or ():
        # A span with facet_types but no quote/matched_terms cannot be produced
        # by build_evidence_set(); the guard documents the invariant this
        # adapter relies on and keeps hand-built spans from bypassing it.
        if not (span.quote or "").strip() or not span.matched_terms:
            continue
        for facet_type in span.facet_types:
            need = _FACET_NEED_MAP.get(facet_type)
            if need is not None:
                found.add(need)
    return tuple(need for need in _NEED_ORDER if need in found)


# ---------------------------------------------------------------------------
# StateRequirement + DocumentStateEvidence[] → StateCoverage
# ---------------------------------------------------------------------------

# States that conclusively mean "not signed" (W1's non-SIGNED, non-UNKNOWN set).
_UNSIGNED_STATES = frozenset({
    DocumentState.FOR_SIGNATURE,
    DocumentState.DRAFT,
    DocumentState.TEMPLATE,
})


def _fold(text: str) -> str:
    """ASCII-folded casefold — same normalization document_state.py/evidence.py
    use internally (small, self-contained duplication instead of exporting a
    private helper across the module boundary)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


# Legal forms and corporate suffixes carry no identity of their own: "H&B
# Group" and "Zakládání Group" share only "group". Matching on such a token
# previously let one company's signed contract confirm another's (verified
# against the production index: query "je podepsaný dodatek Zakládání Group?"
# accepted "Generální klíč H&B Group_NOT250334-podepsané.pdf" as evidence).
# Tokens shorter than 3 chars ("as", "co") can never reach here — W1's
# _entity_terms already drops them — they are listed for completeness only.
_GENERIC_ENTITY_TOKENS = frozenset({
    "group", "holding", "gmbh", "kg", "sro", "spol", "as", "ag", "co",
    "company", "ltd", "llc", "inc", "plc", "partners", "invest", "trade",
    "servis", "service", "consulting",
})


def _normalize_name(document: str) -> str:
    """Fold and flatten filename separators to single spaces.

    `path` is deliberately NOT part of this: a project/folder name in the path
    ("/Haus365/...") would mark every unrelated document in that folder as the
    queried entity's evidence — the same over-reach class as PR6.1's finding.
    """
    folded = _fold(document)
    return " ".join("".join(c if c.isalnum() else " " for c in folded).split())


def _discriminative_terms(
    entity_terms: tuple[str, ...],
    names: tuple[str, ...],
) -> tuple[str, ...]:
    """Entity terms that can actually discriminate between candidates.

    Two filters, both required:
      * generic legal-form tokens are dropped (see _GENERIC_ENTITY_TOKENS)
      * a term must occur as a WHOLE TOKEN in at least one filename of the pool

    The whole-token support test is what keeps query noise out. W1's
    _entity_terms is a residual token list and does leak non-entity words
    (verified: "Ověř podpis smlouvy haus365" yields ('over', 'haus365')), and
    requiring 'over' to appear in a filename would turn a legitimate query into
    an entity mismatch. Measured over the 6342-document production index,
    'over' occurs as a substring in 12 names (inside "ověření", "b-over"…) but
    as a token in ZERO, while real entity names are unaffected ('haus365',
    'jvs', 'zakladani', 'safetypeak', 'hilti': identical counts either way).

    Matching itself stays substring-based (see _entity_candidates), so Czech
    declension and glued filenames still match — 'monolit' inside "monolity",
    'rent' inside "hiltrent" — without letting a substring coincidence promote
    a noise word into a discriminator.

    Stays pure: "support" is computed from the supplied pool, never from the
    index or a DF lookup.
    """
    tokens = set()
    for name in names:
        tokens.update(name.split())
    return tuple(
        term for term in entity_terms
        if term and term not in _GENERIC_ENTITY_TOKENS and term in tokens
    )


def _entity_candidates(
    pool: tuple[DocumentStateEvidence, ...],
    entity_terms: tuple[str, ...],
) -> tuple[tuple[DocumentStateEvidence, ...], tuple[DocumentStateEvidence, ...]]:
    """Split the pool into (strong, partial) entity matches.

    STRONG — may support a positive signed claim:
      * the whole normalized entity phrase occurs in the filename, or
      * EVERY discriminative term occurs in the filename

    PARTIAL — some but not all discriminative terms occur. Never sufficient for
    a positive verdict: "Zakládání Group" must not be confirmed by "H&B Group",
    and "jeřáb JVS" must not be confirmed by an unrelated crane contract that
    merely shares one token. This trades recall for safety on purpose; ranking
    partial matches by corpus rarity would need a DF lookup, which belongs in
    the runtime consumer (cf. auxiliary_term_coverage's injected df_lookup),
    not in a pure layer.

    Neither list reorders the pool.
    """
    names = tuple(_normalize_name(ev.document) for ev in pool)
    if not any(term and term not in _GENERIC_ENTITY_TOKENS for term in entity_terms):
        # The query named nothing but legal forms ("je podepsaná smlouva
        # Group?") — no document can be attributed, not even by phrase.
        return (), ()

    phrase = " ".join(term for term in entity_terms if term)
    required = _discriminative_terms(entity_terms, names)

    strong: list[DocumentStateEvidence] = []
    partial: list[DocumentStateEvidence] = []
    for evidence, name in zip(pool, names):
        if phrase and phrase in name:
            strong.append(evidence)
            continue
        if not required:
            continue
        matched = sum(1 for term in required if term in name)
        if matched == len(required):
            strong.append(evidence)
        elif matched:
            partial.append(evidence)
    return tuple(strong), tuple(partial)


def _coverage(
    requirement: StateRequirement,
    verdict: StateVerdict,
    evidences: tuple[DocumentStateEvidence, ...] = (),
    entity_matched: bool = False,
    states_present: frozenset[DocumentState] = frozenset(),
    conflict: bool = False,
) -> StateCoverage:
    return StateCoverage(
        requirement=requirement,
        evidences=evidences,
        entity_matched=entity_matched,
        states_present=states_present,
        verdict=verdict,
        conflict=conflict,
    )


def build_state_coverage(
    requirement: StateRequirement,
    evidences: Iterable[DocumentStateEvidence] | None = None,
) -> StateCoverage:
    """Aggregate per-document W1 classifications into one lifecycle verdict.

    Pure: `requirement` comes from document_state.derive_state_requirement()
    and `evidences` from document_state.classify_document_state() — no I/O, no
    retrieval, no LLM, and the input order is never treated as ranking.

    Scope: SIGNED intent only. A state-insensitive query (empty
    required_states) and a FOR_SIGNATURE intent both resolve to NOOP, matching
    the existing runtime contract in ai_search._document_state_outcome. The
    verdict vocabulary is signed-centric, so answering a "k podpisu" intent
    with SIGNED_CONFIRMED would be a misnomer — FOR_SIGNATURE coverage is
    future work, not a silent reinterpretation.

    Decision order (first match wins):
      1. SIGNED not required                        → NOOP
      2. entity named, no candidate matches it      → ENTITY_MISMATCH
      3. entity named, only PARTIAL matches         → UNVERIFIED (not citable)
      4. empty candidate pool                       → UNVERIFIED
      5. SIGNED present                             → SIGNED_CONFIRMED
      6. UNKNOWN present                            → UNVERIFIED
      7. every candidate conclusively unsigned      → UNSIGNED_CONFIRMED
      8. anything else (unmodelled state)           → UNVERIFIED

    Step 2 has no fallback to the unfiltered pool — that fallback was the
    PR6.1 critical finding (a different party's signed contract confirmed the
    queried entity). Step 4 keeps an empty pool from reading as "confirmed not
    signed": absence of candidates is not evidence of absence.

    CONFLICT is deliberately NOT derived (PR7.0.1). The previous rule — SIGNED
    coexisting with a forbidden state (DRAFT/TEMPLATE/FOR_SIGNATURE) — read
    normal document lifecycle as contradiction: a signed contract almost always
    sits next to the drafts it grew out of. Measured against the production
    index, 6 of 7 realistic signed-intent queries returned CONFLICT, including
    the canonical HAUS365 case that PR6 answers correctly today (its 11-document
    revision family spans all five states), and even a minimal two-document pool
    (one signed + one draft) conflicted. `requirement.forbidden_states` means
    "must not ALONE support a positive claim" (W1) — a citation rule, not a
    contradiction rule. A real conflict needs a signal W1 does not have: an
    explicit revocation/termination state, or two competing SIGNED revisions of
    one document identity. Until such a signal exists, SIGNED evidence always
    yields SIGNED_CONFIRMED and consumers can see coexisting drafts in
    `states_present`.
    """
    pool = tuple(evidences or ())

    if DocumentState.SIGNED not in requirement.required_states:
        return _coverage(requirement, StateVerdict.NOOP)

    if requirement.entity_terms:
        candidates, partial = _entity_candidates(pool, requirement.entity_terms)
        if not candidates:
            if partial:
                # Only weakly related documents: a partial (single-token) entity
                # match may never support a positive signed claim, and citing
                # such a document would repeat the PR6.1 failure one level down.
                return _coverage(requirement, StateVerdict.UNVERIFIED)
            return _coverage(requirement, StateVerdict.ENTITY_MISMATCH)
        entity_matched = True
    else:
        # No entity in the query → no entity match can be established. Reported
        # as False rather than vacuously True, so a consumer can never read
        # `entity_matched` as "the queried party was verified".
        candidates = pool
        entity_matched = False

    if not candidates:
        return _coverage(requirement, StateVerdict.UNVERIFIED, entity_matched=entity_matched)

    states_present = frozenset(ev.state for ev in candidates)
    signed = tuple(ev for ev in candidates if ev.state is DocumentState.SIGNED)

    if signed:
        # PR7.6.1: a signed LOI / order / minutes must NEVER confirm a
        # "podepsaná smlouva" query. Split signed evidence into contract vs
        # other; only contract-kind docs produce SIGNED_CONFIRMED
        # (= SIGNED_CONTRACT_CONFIRMED alias). Other signed docs alone yield
        # SIGNED_OTHER_DOCUMENT_CONFIRMED so the gate can say "found signed
        # LOI, not a signed contract" instead of affirming a SoD.
        if query_requires_signed_contract(requirement.doc_type_terms):
            contract_signed = tuple(
                ev for ev in signed
                if signed_evidence_is_contract_for_query(
                    requirement.doc_type_terms, ev.document, ev.path,
                )
            )
            other_signed = tuple(
                ev for ev in signed if ev not in contract_signed
            )
            if contract_signed:
                return _coverage(
                    requirement, StateVerdict.SIGNED_CONTRACT_CONFIRMED,
                    contract_signed, entity_matched, states_present,
                )
            if other_signed:
                # LOI / order / minutes → clarify. Wrong-subject contracts
                # (e.g. any SoD when the query asked for BOZP) must NOT be
                # cited either — that was the status-01 false positive.
                if subject_doc_type_terms(requirement.doc_type_terms):
                    return _coverage(
                        requirement, StateVerdict.UNVERIFIED,
                        (), entity_matched, states_present,
                    )
                return _coverage(
                    requirement, StateVerdict.SIGNED_OTHER_DOCUMENT_CONFIRMED,
                    other_signed, entity_matched, states_present,
                )
        return _coverage(
            requirement, StateVerdict.SIGNED_CONFIRMED, signed,
            entity_matched, states_present,
        )

    unknown = tuple(ev for ev in candidates if ev.state is DocumentState.UNKNOWN)
    if unknown:
        return _coverage(
            requirement, StateVerdict.UNVERIFIED, unknown,
            entity_matched, states_present,
        )

    if states_present <= _UNSIGNED_STATES:
        return _coverage(
            requirement, StateVerdict.UNSIGNED_CONFIRMED, candidates,
            entity_matched, states_present,
        )

    # Unreachable with today's DocumentState members. Kept as the default so a
    # future state (e.g. REVOKED) cannot silently license a negative claim by
    # falling through — it must be classified explicitly first.
    return _coverage(
        requirement, StateVerdict.UNVERIFIED, candidates,
        entity_matched, states_present,
    )


# ---------------------------------------------------------------------------
# PR7.6.1 — retrieval evidence safety (abstention + project conflict)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Tokens too generic to alone justify a document-hit claim. Sharing only one
# of these between the query and a retrieved name/quote is "lexical bait",
# not evidence (FAT: "kniha betonů" ← "kniha revizí …").
_GENERIC_QUERY_TOKENS = frozenset({
    "kniha", "knihu", "knihy", "knih", "seznam", "dokument", "dokumentu", "dokumenty",
    "soubor", "souboru", "najdi", "najit", "hledej", "jake", "jaka", "jaky", "jak",
    "jsou", "pozadavky", "pozadavek", "typ", "podle", "pro", "na", "do",
    "od", "za", "po", "pri", "bez", "the", "and", "or", "of", "in", "on",
    "to", "with", "a", "i", "o", "u", "k", "s", "z", "v", "ve", "ke", "se",
    "je", "co", "ci", "nebo", "ale",
})

# Canonical abstention sentence — must stay byte-compatible with the answer
# renderer's not-found filler so acceptance `unsupported_claim` treats it as
# scaffolding, not a fabricated fact.
_ABSTAIN_MESSAGE = "Nenalezeno v indexovaných dokumentech."


def _token_variants(tok: str) -> tuple[str, ...]:
    """Folded token + a light Czech stem so 'betonove' matches 'betonu'."""
    out = [tok]
    for suf in ("ove", "ova", "ovy", "ych", "ymi", "em", "um", "u", "y", "e", "a", "i"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            out.append(tok[: -len(suf)])
            break
    return tuple(dict.fromkeys(out))


def _query_content_tokens(query: str) -> tuple[str, ...]:
    folded = _fold(query)
    out: list[str] = []
    seen: set[str] = set()
    for tok in _TOKEN_RE.findall(folded):
        if len(tok) < 3 or tok in _GENERIC_QUERY_TOKENS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return tuple(out)


def _row_blob(row: dict) -> str:
    return _fold(
        f"{row.get('document') or ''} {row.get('path') or ''} "
        f"{row.get('heading') or ''} {row.get('quote') or ''}"
    )


def _row_name_blob(row: dict) -> str:
    return _fold(f"{row.get('document') or ''} {row.get('path') or ''}")


def _blob_has_token(blob: str, tok: str) -> bool:
    return any(v in blob for v in _token_variants(tok))


def _row_discriminative_hits(tokens: tuple[str, ...], row: dict, *, name_only: bool = False) -> int:
    blob = _row_name_blob(row) if name_only else _row_blob(row)
    return sum(1 for tok in tokens if _blob_has_token(blob, tok))


def _token_positions(blob: str, tok: str) -> list[int]:
    positions: list[int] = []
    for variant in _token_variants(tok):
        start = 0
        while True:
            idx = blob.find(variant, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + 1
    return positions


# Domain markers that, when nearer to a property token than the queried
# material, mean the property hit is about a different building system
# (FAT nds-qa-10: "rovinnost" of SDK/Q3 ≠ rovinnost of a concrete slab).
_FOREIGN_PROPERTY_DOMAINS = frozenset({
    "sdk", "sadrokarton", "knauf", "pricka", "pricky", "q3", "q2", "q1",
})


def _strict_material_positions(blob: str, tok: str) -> list[int]:
    """Positions of material tokens, excluding the over-broad bare stem 'beton'.

    `_token_variants('betonove')` yields 'beton', which matches every concrete
    passage in a KZP and would defeat the rovinnost grounding check.
    """
    variants = [
        v for v in _token_variants(tok)
        if v != "beton" and (len(v) >= 6 or v in {"betonu", "betony"})
    ]
    if not variants and tok.startswith("beton"):
        variants = ["betonov", tok]
    positions: list[int] = []
    for variant in variants:
        start = 0
        while True:
            idx = blob.find(variant, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + 1
    return positions


def _property_grounded_in_query_material(
    blob: str,
    property_tok: str,
    material_tokens: tuple[str, ...],
    *,
    window: int = 64,
) -> bool:
    """Property token must sit next to the queried material, not a foreign domain."""
    prop_hits = _token_positions(blob, property_tok)
    if not prop_hits:
        return False
    mat_hits: list[int] = []
    for tok in material_tokens:
        mat_hits.extend(_strict_material_positions(blob, tok))
    if not mat_hits:
        return False
    foreign_hits: list[int] = []
    for marker in _FOREIGN_PROPERTY_DOMAINS:
        foreign_hits.extend(_token_positions(blob, marker))

    for p in prop_hits:
        near_mat = [m for m in mat_hits if abs(m - p) <= window]
        if not near_mat:
            continue
        near_foreign = [f for f in foreign_hits if abs(f - p) <= window]
        if near_foreign and min(abs(f - p) for f in near_foreign) < min(abs(m - p) for m in near_mat):
            continue
        return True
    return False


def _row_has_discriminative_overlap(tokens: tuple[str, ...], row: dict, *, name_only: bool = False) -> bool:
    """True when the row covers enough of the query's discriminative tokens.

    One-token queries ("Pentaflex") need that one hit. Multi-token queries
    need at least two hits in the same row. The special rovinnost+beton*
    pair additionally requires local material grounding so an SDK 'rovinnost'
    hit cannot be combined with a distant 'beton' mention (FAT nds-qa-10).
    """
    if not tokens:
        return False
    blob = _row_name_blob(row) if name_only else _row_blob(row)
    if len(tokens) == 1:
        return _blob_has_token(blob, tokens[0])

    material = tuple(t for t in tokens if t.startswith("beton"))
    if "rovinnost" in tokens and material:
        return _property_grounded_in_query_material(blob, "rovinnost", material)

    need = min(2, len(tokens))
    return _row_discriminative_hits(tokens, row, name_only=name_only) >= need


def _is_document_lookup_query(query: str) -> bool:
    folded = _fold(query)
    return any(
        folded.startswith(k + " ") or f" {k} " in f" {folded} "
        for k in ("najdi", "najit", "hledej")
    )


def _lookup_residual_phrase(query: str) -> str:
    """Query minus the leading lookup verb — the document identity being sought."""
    folded = _fold(query)
    for verb in ("najdi", "najit", "hledej"):
        if folded.startswith(verb + " "):
            folded = folded[len(verb):].strip()
            break
        needle = f" {verb} "
        if needle in f" {folded} ":
            folded = folded.replace(verb, " ", 1)
            break
    return " ".join(folded.split())


def _row_covers_lookup_phrase(phrase: str, row: dict) -> bool:
    """Document-lookup hit: the sought phrase must appear as a whole.

    Token-level overlap against a quote that merely shares a domain word
    ("betonu" inside a TP about concrete) is exactly the FAT false-positive
    class this guard exists to stop.
    """
    if not phrase:
        return False
    return phrase in _row_blob(row)


def evaluate_evidence_safety(
    query: str,
    results: Iterable[dict] | None = None,
) -> EvidenceSafety:
    """Decide whether the retrieval pool may support a factual answer.

    Pure: reads `results` rows (document/path/quote) only — never reorders
    them, never calls an LLM.

      1. PROJECT CONFLICT — name claims local project, quote names a foreign
         one. If no non-conflicted overlap remains → DOCUMENT_PROJECT_CONFLICT.
      2. DOCUMENT LOOKUP ("najdi …") — residual phrase after the lookup verb
         must appear as a whole in name/path/quote (FAT: "knihu betonu" must
         not match a TP filename that merely contains "beton", nor a KZP
         quote about "kniha revizí").
      3. WEAK / NO OVERLAP — otherwise, not enough discriminative tokens in
         any row → NO_EVIDENCE.

    Signed-contract intents skip (3) — StateCoverage / the state gate hedges.
    Abstention messages are the canonical not-found sentence only, so the
    acceptance harness does not score them as unsupported claims.
    """
    from document_state import derive_state_requirement

    rows = tuple(results or ())
    conflicted = tuple(
        str(row.get("document") or "")
        for row in rows
        if detect_project_conflict(
            str(row.get("document") or ""),
            str(row.get("path") or ""),
            str(row.get("quote") or ""),
        )
    )
    seen: set[str] = set()
    conflicted_docs: list[str] = []
    for name in conflicted:
        if name and name not in seen:
            seen.add(name)
            conflicted_docs.append(name)

    tokens = _query_content_tokens(query)
    if not rows:
        return EvidenceSafety(
            status=EvidenceSafetyStatus.NO_EVIDENCE,
            message=_ABSTAIN_MESSAGE,
            conflicted_documents=tuple(conflicted_docs),
        )

    lookup = _is_document_lookup_query(query)
    lookup_phrase = _lookup_residual_phrase(query) if lookup else ""
    if lookup:
        overlapping = [
            row for row in rows
            if _row_covers_lookup_phrase(lookup_phrase, row)
        ]
    else:
        overlapping = [
            row for row in rows
            if _row_has_discriminative_overlap(tokens, row)
        ]
    non_conflict_overlap = [
        row for row in overlapping
        if not detect_project_conflict(
            str(row.get("document") or ""),
            str(row.get("path") or ""),
            str(row.get("quote") or ""),
        )
    ]

    if conflicted_docs and overlapping and not non_conflict_overlap:
        return EvidenceSafety(
            status=EvidenceSafetyStatus.DOCUMENT_PROJECT_CONFLICT,
            message=_ABSTAIN_MESSAGE,
            conflicted_documents=tuple(conflicted_docs),
        )

    if conflicted_docs and not non_conflict_overlap and tokens:
        name_hits = [
            row for row in rows
            if detect_project_conflict(
                str(row.get("document") or ""),
                str(row.get("path") or ""),
                str(row.get("quote") or ""),
            )
        ]
        if name_hits:
            return EvidenceSafety(
                status=EvidenceSafetyStatus.DOCUMENT_PROJECT_CONFLICT,
                message=_ABSTAIN_MESSAGE,
                conflicted_documents=tuple(conflicted_docs),
            )

    state_req = derive_state_requirement(query)
    if state_req.required_states:
        return EvidenceSafety(
            status=EvidenceSafetyStatus.OK,
            conflicted_documents=tuple(conflicted_docs),
        )

    if tokens and not overlapping:
        return EvidenceSafety(
            status=EvidenceSafetyStatus.NO_EVIDENCE,
            message=_ABSTAIN_MESSAGE,
            conflicted_documents=tuple(conflicted_docs),
        )

    return EvidenceSafety(
        status=EvidenceSafetyStatus.OK,
        conflicted_documents=tuple(conflicted_docs),
    )
