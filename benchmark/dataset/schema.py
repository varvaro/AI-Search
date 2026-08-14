"""Benchmark dataset schema.

A benchmark case is ONE JSON object on ONE line in a `.jsonl` file under
`benchmark/dataset/`. Adding a new regression case never requires touching
any code - just append a line to an existing file (or add a new file and
point `--dataset` at it).

Only `id` and `question` are required. Everything else has a sensible
default, so a minimal case is just:

    {"id": "pentaflex-01", "question": "Pentaflex"}

A fully-specified case looks like:

    {
      "id": "feri-handover-01",
      "question": "Co chybí k předání základové desky investorovi?",
      "type": "checklist",
      "difficulty": "hard",
      "environment": "production",
      "relevance_mode": "multi_document_reasoning",
      "project": "240783160_Garáže_NDS",
      "expected_documents": ["FERI/Předávací dokumentace"],
      "expected_keywords": ["pentaflex", "beton", "geodetick"],
      "forbidden_documents": ["zákládovou desku."],
      "expected_answer_contains": [],
      "min_recall_at_10": 0.0,
      "tags": ["production-regression"],
      "notes": "2026-08-06 diagnostika: FERI dokumenty ztraceny na rerank_k truncation."
    }

`relevance_mode` (2026-08-06 ground-truth repair) - what KIND of ground truth
this case has, and therefore which metrics gate pass/fail:

  * "document" (default, unchanged behaviour): `expected_documents` is a
    document/folder-level judgment - "this file should be represented
    somewhere in the top-k", nothing asserted about which specific chunk of
    it. This is what every case used before this field existed, so omitting
    `relevance_mode` entirely keeps 100% of the old semantics.
  * "chunk": the case additionally asserts that a SPECIFIC chunk (not just
    "the file, somewhere") must be found - use this for checklist/factual
    questions where a document merely existing in a huge folder is not proof
    the *answer* was retrievable. Requires a non-empty `expected_chunks`.
    Metrics are then computed by matching `expected_chunks`, not by
    substring-matching `expected_documents` against the whole folder (a wide
    folder substring can no longer trivially satisfy the case just because
    *some* unrelated chunk of *some* file in that folder showed up).
  * "multi_document_reasoning": the question requires comparing several
    documents (e.g. "what checklist item is MISSING" requires knowing both
    the full checklist AND everything actually delivered) and the forensic
    audit found no single chunk that answers it. Excluded from the standard
    Recall@k pass/fail gate (kept in the dataset for future RAG/answer-level
    evaluation, but `evaluate_case()` always reports it as passed=True and
    `runner._aggregate()` counts it separately, not inside passed/failed).

expected_chunks (2026-08-06, superseded the old unused "exact chunk-id pins"
idea) - each entry identifies ONE physical chunk in the CURRENT index without
hardcoding its runtime `chunk_id` string (chunk_id = sha256(file bytes):ordinal
in ai_search.py - stable only as long as the file's bytes AND the chunking
algorithm don't change, so pinning it directly in a checked-in dataset file
would silently break on the next re-chunk). Instead:

    {"document": "KZP - TEXTOVÁ ČÁST.pdf", "ordinal": 143,
     "text_anchor": "Kontrola dodacích listů betonové směsi", "relevance": 3}

  * document: substring match against path/name (same rule as
    expected_documents) - which file.
  * ordinal: the chunk's 0-indexed position within that file (optional but
    recommended - narrows "the file" down to "this one chunk of the file").
  * text_anchor: a short, verified-real substring of that chunk's actual
    text (optional but recommended) - the redundancy that makes the pin
    resilient to ordinal drift from a future re-chunk: `resolve_expected_chunks()`
    (benchmark/dataset/chunk_resolution.py) re-resolves document+ordinal+
    text_anchor against whatever index is loaded *at run time*, so the
    dataset file itself never stores an index-version-specific id.
  * relevance: 0-3 graded relevance (0=irrelevant, 1=related, 2=relevant,
    3=direct answer) used by the chunk-mode nDCG calculation. Defaults to 2.
  * chunk_id: legacy escape hatch - an exact chunk_id string, used as-is with
    no DB resolution, for the rare case a test wants to pin a specific
    already-known id (e.g. a synthetic fixture-environment test).

Matching semantics (see benchmark/metrics.py for the actual comparisons):
  * expected_documents / forbidden_documents: case-insensitive SUBSTRING match
    against the result's `path` (falls back to `document`). Using substrings
    instead of exact chunk ids keeps cases easy to write and resilient to
    re-chunking / re-indexing - you name the folder/file you expect, not an
    opaque id. Only used for pass/fail when relevance_mode="document".
  * expected_chunks: see above. Only used for pass/fail when
    relevance_mode="chunk".
  * expected_keywords: case-insensitive substring match against the `quote`
    text of the top-k results (did the *content* the answer would need make
    it into the context, not just the file). Supplementary evidence in every
    mode - never a substitute for expected_documents/expected_chunks.
  * min_recall_at_10: per-case override of the global pass/fail recall
    threshold (see benchmark/runner.py DEFAULT_RECALL_THRESHOLD). Use this to
    intentionally mark a known-hard case as "should still find nothing" (0.0)
    or as a stricter contract (e.g. 1.0) than the project-wide default. Not
    applicable to relevance_mode="multi_document_reasoning" (never gated).

Known-limitation flags (2026-08-08, added with
benchmark/dataset/retrieval_regression.jsonl) - both default to False, so
every pre-existing case keeps its exact previous semantics:

  * expected_content_missing: the ground truth does NOT exist in the index at
    all, and the case is kept only to detect the day it appears (e.g. the file
    is finally filed in the source folder). A MISS here is the CORRECT result,
    so it must never be counted as a retrieval failure - otherwise every run
    carries a permanent red mark that trains people to ignore the report.
  * expected_retrieval_issue: the content IS indexed but a diagnosed, still-open
    retrieval weakness keeps it out of the top-k (e.g. the query spells out
    "kontrolní a zkušební plán" while the documents only ever use the acronym
    "KZP"). Reported on every run so the debt stays visible, but never fails
    the run on its own - the point is to notice when it starts passing, not to
    block on a limitation already known and accepted.

A case cannot be both: "the content is missing" and "retrieval cannot find the
content" are mutually exclusive diagnoses, and marking both would hide which
one is actually being asserted.

  * domain: coarse subject grouping ("SoD", "BOZP", "fakturace", ...) used to
    aggregate results per area of the archive. Free-form on purpose - the set
    of domains follows the customer's folder structure, which changes.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

VALID_TYPES = {"lookup", "checklist", "factual", "comparison", "negative"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_ENVIRONMENTS = {"fixture", "production"}
VALID_RELEVANCE_MODES = {"document", "chunk", "multi_document_reasoning"}
# PR7.4 answer-quality (safety) categories. Empty string keeps every pre-PR7.4
# case byte-compatible (category is optional and defaults to "").
SAFETY_CATEGORIES = {"SIGNED_DOCUMENT", "ENTITY_SAFETY", "EVIDENCE_COVERAGE", "REGRESSION"}
# PR7.4.1 acceptance (product) categories - the work a site manager actually
# does. Kept as a SEPARATE set from the safety categories on purpose: a safety
# benchmark answers "can this tool lie about a contract", an acceptance
# benchmark answers "does this tool save me a trip into the Box folders". Mixing
# them into one score is how a green safety run gets mistaken for a usable
# product (audit section 5).
ACCEPTANCE_CATEGORIES = {
    "DOCUMENT_LOOKUP",       # "kde najdu kladečské výkresy výztuže?"
    "TECHNICAL_INFO",        # "v jakém stáří se zkouší krychle betonu?"
    "CONTRACT_VERIFICATION", # "co upravuje dodatek smlouvy o dílo?"
    "MEETING_MINUTES",       # zápisy z kontrolních dnů
    "TECHNICAL_PROCEDURE",   # montážní/technologické postupy
}
# PR7.5 project acceptance categories - the FAT/SAT agenda of ONE building site
# (240783160_Garáže_NDS), phrased the way its site manager phrases it. Kept
# separate from ACCEPTANCE_CATEGORIES because those describe a generic document
# assistant while these describe the construction agenda under certification.
PROJECT_ACCEPTANCE_CATEGORIES = {
    "DOCUMENT_SEARCH",    # najdi výkres výztuže základové desky
    "TECHNICAL_QA",       # jaký beton je na spodní stavbě?
    "DOCUMENT_STATUS",    # existuje podepsaná smlouva na monolit?
    "CONSTRUCTION_MGMT",  # jaké jsou závazné termíny pro FERI?
    "ADVERSARIAL",        # stará revize vs. nová, draft vs. final, dvojí dodavatel
}
VALID_CATEGORIES = (
    {""} | SAFETY_CATEGORIES | ACCEPTANCE_CATEGORIES | PROJECT_ACCEPTANCE_CATEGORIES
)
VALID_STATE_VERDICTS = {
    "SIGNED_CONFIRMED", "UNSIGNED_CONFIRMED", "ENTITY_MISMATCH", "UNVERIFIED", "NOOP",
}
VALID_INTENT_COVERAGES = {"COMPLETE", "PARTIAL"}
# How bad a wrong answer is. Only legal/financial mistakes count as CRITICAL
# errors in the acceptance report - an imprecise answer about a technical detail
# the user will verify anyway is not the same class of harm as a wrong claim
# about a signed contract or an invoiced amount.
# "safety" (PR7.5) sits next to legal/financial as a CRITICAL class: a wrong
# answer about concrete cover or a superseded reinforcement drawing is a defect
# that gets built into the structure, not an inconvenience the reader corrects.
VALID_CRITICALITIES = {"legal", "financial", "safety", "technical", "informational"}
# Whether a human has confirmed this case's ground truth against the real index.
# Acceptance GO is impossible while unverified cases are present: a dataset
# nobody checked cannot certify a tool for daily use.
VALID_GROUND_TRUTH_STATUSES = {"verified", "needs_review", "unverified"}
# PR7.5. What the case asserts about the WORLD, independent of what retrieval
# does: "found" = the thing exists and must be produced, "not_found" = the thing
# genuinely is not in the index and the only correct answer says so. Negative
# cases are first-class here because "kniha betonů" and "stavební deník" do not
# exist in 240783160_Garáže_NDS at all - writing them as positives would mint a
# permanently red dataset, and omitting them would drop the failure mode most
# likely to mislead a site manager.
VALID_EXPECTED_OUTCOMES = {"found", "not_found"}
# PR7.5. HOW a case's ground truth was established. `index_query` is the only
# machine-checkable one and is deliberately the weakest: a file existing in the
# index proves neither that it is current nor that its content answers the
# question. Critical cases therefore require `expert_confirm`, and negative
# cases require `folder_listing` - absence from a full-text search is not proof
# of absence from the archive.
VALID_VERIFICATION_METHODS = {
    "",                # not yet decided
    "index_query",     # file provably present in the index (read-only SQL)
    "folder_listing",  # a human walked the folder, absence included
    "document_read",   # a human opened the document and read the fact
    "expert_confirm",  # the responsible professional confirmed it
    "cross_document",  # the fact agrees across two independent documents
}
# Criticality classes where a wrong answer is a critical error, and where
# `index_query` alone can never be sufficient verification.
CRITICAL_CRITICALITY_NAMES = frozenset({"legal", "financial", "safety"})


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", (text or "").casefold()) if not unicodedata.combining(c))


# PR7.5 spells its fields in the singular, product-facing form ("query",
# "expected_document"). The canonical names predate it and are load-bearing in
# ~800 tests and every existing .jsonl, so the two are reconciled by aliasing on
# read rather than by renaming. Writing BOTH spellings is an error: silently
# preferring one would make a case mean something other than what it says.
FIELD_ALIASES = {
    "query": "question",
    "expected_document": "expected_documents",
    "expected_source": "expected_source_contains",
    "expected_answer_keywords": "expected_answer_contains",
    "forbidden_answer_keywords": "forbidden_answer_contains",
}


def _resolve_aliases(data: dict, *, source_file: str = "", line_number: int = 0) -> dict:
    if not any(alias in data for alias in FIELD_ALIASES):
        return data
    resolved = dict(data)
    for alias, canonical in FIELD_ALIASES.items():
        if alias not in resolved:
            continue
        if canonical in resolved:
            raise ValueError(
                f"{source_file}:{line_number}: case {data.get('id')!r} sets both {alias!r} and "
                f"{canonical!r} - they are the same field, pick one"
            )
        resolved[canonical] = resolved.pop(alias)
    return resolved


@dataclass
class ExpectedChunk:
    """One physical-chunk identity for a relevance_mode="chunk" case. See the
    module docstring above for the full field-by-field rationale."""
    document: str = ""
    ordinal: int | None = None
    text_anchor: str = ""
    relevance: int = 2
    chunk_id: str = ""

    @staticmethod
    def from_value(value) -> "ExpectedChunk":
        if isinstance(value, str):
            # legacy form: a bare string used to mean "exact chunk_id pin".
            return ExpectedChunk(chunk_id=value)
        if isinstance(value, dict):
            return ExpectedChunk(
                document=str(value.get("document", "")),
                ordinal=value.get("ordinal"),
                text_anchor=str(value.get("text_anchor", "")),
                relevance=int(value.get("relevance", 2)),
                chunk_id=str(value.get("chunk_id", "")),
            )
        raise ValueError(f"expected_chunks entry must be a string or object, got {type(value).__name__}")


@dataclass
class BenchmarkCase:
    id: str
    question: str
    type: str = "lookup"
    difficulty: str = "medium"
    environment: str = "fixture"
    relevance_mode: str = "document"
    project: str | None = None
    domain: str = ""
    expected_content_missing: bool = False
    expected_retrieval_issue: bool = False
    expected_documents: list[str] = field(default_factory=list)
    expected_chunks: list[ExpectedChunk] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_documents: list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    min_recall_at_10: float | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    source_file: str = ""
    line_number: int = 0
    # PR7.4 answer-quality fields (all optional; empty defaults keep every
    # pre-PR7.4 .jsonl line loadable without change).
    category: str = ""
    expected_state_verdict: str | None = None
    expected_intent_coverage: str | None = None
    expected_missing_needs: list[str] = field(default_factory=list)
    forbidden_answer_contains: list[str] = field(default_factory=list)
    expected_source_contains: list[str] = field(default_factory=list)
    forbidden_sources: list[str] = field(default_factory=list)
    # PR7.4.1 acceptance fields (all optional; defaults keep every earlier
    # .jsonl line loadable unchanged).
    criticality: str = "informational"
    ground_truth_status: str = "unverified"
    expected_fact: str = ""
    follow_up_questions: list[str] = field(default_factory=list)
    # PR7.5 project-acceptance fields (all optional; defaults keep every earlier
    # .jsonl line loadable unchanged).
    expected_outcome: str = "found"
    forbidden_document: list[str] = field(default_factory=list)
    verification_method: str = ""
    human_verified: bool = False
    verified_by: str = ""
    verification_date: str = ""
    index_fingerprint_at_verification: str = ""

    @staticmethod
    def from_dict(data: dict, *, source_file: str = "", line_number: int = 0) -> "BenchmarkCase":
        data = _resolve_aliases(data, source_file=source_file, line_number=line_number)
        if "id" not in data or not str(data["id"]).strip():
            raise ValueError(f"{source_file}:{line_number}: case is missing required field 'id'")
        if "question" not in data or not str(data["question"]).strip():
            raise ValueError(f"{source_file}:{line_number}: case {data.get('id')!r} is missing required field 'question'")
        case_type = data.get("type", "lookup")
        if case_type not in VALID_TYPES:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid type {case_type!r} (expected one of {sorted(VALID_TYPES)})")
        difficulty = data.get("difficulty", "medium")
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid difficulty {difficulty!r} (expected one of {sorted(VALID_DIFFICULTIES)})")
        environment = data.get("environment", "fixture")
        if environment not in VALID_ENVIRONMENTS:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid environment {environment!r} (expected one of {sorted(VALID_ENVIRONMENTS)})")
        relevance_mode = data.get("relevance_mode", "document")
        if relevance_mode not in VALID_RELEVANCE_MODES:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid relevance_mode {relevance_mode!r} (expected one of {sorted(VALID_RELEVANCE_MODES)})")
        expected_chunks = [ExpectedChunk.from_value(v) for v in data.get("expected_chunks", [])]
        if relevance_mode == "chunk" and not expected_chunks:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has relevance_mode='chunk' but no expected_chunks - a chunk-level case needs at least one chunk identity to gate on")
        content_missing = bool(data.get("expected_content_missing", False))
        retrieval_issue = bool(data.get("expected_retrieval_issue", False))
        if content_missing and retrieval_issue:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} sets both expected_content_missing and expected_retrieval_issue - these are mutually exclusive diagnoses (content absent vs. content present but unreachable)")
        category = str(data.get("category", "") or "")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid category {category!r} (expected one of {sorted(VALID_CATEGORIES - {''})} or omit)")
        expected_state_verdict = data.get("expected_state_verdict")
        if expected_state_verdict is not None:
            expected_state_verdict = str(expected_state_verdict)
            if expected_state_verdict not in VALID_STATE_VERDICTS:
                raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid expected_state_verdict {expected_state_verdict!r}")
        expected_intent_coverage = data.get("expected_intent_coverage")
        if expected_intent_coverage is not None:
            expected_intent_coverage = str(expected_intent_coverage)
            if expected_intent_coverage not in VALID_INTENT_COVERAGES:
                raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid expected_intent_coverage {expected_intent_coverage!r}")
        criticality = str(data.get("criticality", "informational") or "informational")
        if criticality not in VALID_CRITICALITIES:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid criticality {criticality!r} (expected one of {sorted(VALID_CRITICALITIES)})")
        ground_truth_status = str(data.get("ground_truth_status", "unverified") or "unverified")
        if ground_truth_status not in VALID_GROUND_TRUTH_STATUSES:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid ground_truth_status {ground_truth_status!r} (expected one of {sorted(VALID_GROUND_TRUTH_STATUSES)})")
        expected_outcome = str(data.get("expected_outcome", "found") or "found")
        if expected_outcome not in VALID_EXPECTED_OUTCOMES:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid expected_outcome {expected_outcome!r} (expected one of {sorted(VALID_EXPECTED_OUTCOMES)})")
        verification_method = str(data.get("verification_method", "") or "")
        if verification_method not in VALID_VERIFICATION_METHODS:
            raise ValueError(f"{source_file}:{line_number}: case {data['id']!r} has invalid verification_method {verification_method!r} (expected one of {sorted(VALID_VERIFICATION_METHODS - {''})} or omit)")
        return BenchmarkCase(
            id=str(data["id"]),
            question=str(data["question"]),
            type=case_type,
            difficulty=difficulty,
            environment=environment,
            relevance_mode=relevance_mode,
            project=data.get("project"),
            domain=str(data.get("domain", "")),
            expected_content_missing=content_missing,
            expected_retrieval_issue=retrieval_issue,
            expected_documents=list(data.get("expected_documents", [])),
            expected_chunks=expected_chunks,
            expected_keywords=list(data.get("expected_keywords", [])),
            forbidden_documents=list(data.get("forbidden_documents", [])),
            expected_answer_contains=list(data.get("expected_answer_contains", [])),
            min_recall_at_10=data.get("min_recall_at_10"),
            tags=list(data.get("tags", [])),
            notes=str(data.get("notes", "")),
            source_file=source_file,
            line_number=line_number,
            category=category,
            expected_state_verdict=expected_state_verdict,
            expected_intent_coverage=expected_intent_coverage,
            expected_missing_needs=list(data.get("expected_missing_needs", [])),
            forbidden_answer_contains=list(data.get("forbidden_answer_contains", [])),
            expected_source_contains=list(data.get("expected_source_contains", [])),
            forbidden_sources=list(data.get("forbidden_sources", [])),
            criticality=criticality,
            ground_truth_status=ground_truth_status,
            expected_fact=str(data.get("expected_fact", "")),
            follow_up_questions=list(data.get("follow_up_questions", [])),
            expected_outcome=expected_outcome,
            forbidden_document=list(data.get("forbidden_document", [])),
            verification_method=verification_method,
            human_verified=bool(data.get("human_verified", False)),
            verified_by=str(data.get("verified_by", "")),
            verification_date=str(data.get("verification_date", "")),
            index_fingerprint_at_verification=str(data.get("index_fingerprint_at_verification", "")),
        )


def load_dataset(path: Path) -> list[BenchmarkCase]:
    """Parse one .jsonl file. Blank lines and lines starting with '#' (used
    for human comments/section markers) are skipped. Raises ValueError with a
    file:line pointer on the first malformed entry, rather than silently
    dropping it."""
    path = Path(path)
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON ({exc})") from exc
        case = BenchmarkCase.from_dict(data, source_file=path.name, line_number=line_number)
        if case.id in seen_ids:
            raise ValueError(f"{path.name}:{line_number}: duplicate case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
    return cases


DATASET_VERSION_PREFIX = "# dataset_version:"


def read_dataset_version(path: Path) -> str:
    """Version declared by a `# dataset_version: X` comment in the .jsonl head.

    Kept as a comment rather than a per-case field so the version cannot drift
    between lines of the same file. Returns "" when absent, which the acceptance
    artifact reports verbatim - an unversioned dataset is a fact worth showing,
    not something to paper over with a default.
    """
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(DATASET_VERSION_PREFIX):
            return line[len(DATASET_VERSION_PREFIX):].strip()
        if line and not line.startswith("#"):
            break
    return ""


def load_datasets(paths: list[Path]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for path in paths:
        for case in load_dataset(path):
            if case.id in seen_ids:
                raise ValueError(f"duplicate case id {case.id!r} across dataset files (also defined earlier)")
            seen_ids.add(case.id)
            cases.append(case)
    return cases


DATASET_DIR = Path(__file__).parent
DEFAULT_DATASETS = {
    "fixture": [DATASET_DIR / "fixture_queries.jsonl"],
    "production": [DATASET_DIR / "production_queries.jsonl"],
}


def default_dataset_paths(environment: str) -> list[Path]:
    try:
        return DEFAULT_DATASETS[environment]
    except KeyError as exc:
        raise ValueError(f"unknown environment {environment!r} (expected one of {sorted(DEFAULT_DATASETS)})") from exc
