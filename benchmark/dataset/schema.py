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


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", (text or "").casefold()) if not unicodedata.combining(c))


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

    @staticmethod
    def from_dict(data: dict, *, source_file: str = "", line_number: int = 0) -> "BenchmarkCase":
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
