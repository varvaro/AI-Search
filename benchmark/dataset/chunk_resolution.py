"""Resolves `ExpectedChunk` identities (document substring + ordinal +
text_anchor, see schema.py's module docstring) into concrete `chunk_id`
values by querying whatever index is loaded RIGHT NOW - never by trusting a
chunk_id string baked into the dataset file.

This is read-only metadata lookup, exactly like `pipeline_trace._fetch_chunk_rows`
- it does not touch retrieval/ranking logic, does not call sync(), and does
not write anything. Living here (not in pipeline_trace.py) keeps the
dataset-schema-adjacent concern ("what does this case's ground truth actually
mean against the current index") separate from the pipeline-instrumentation
concern (pipeline_trace.py mirrors ai_search.search()'s phases).
"""
from __future__ import annotations

from pathlib import Path

import ai_search

from .schema import ExpectedChunk, _fold


def resolve_expected_chunks(db_path: Path, expected_chunks: list[ExpectedChunk]) -> list[dict]:
    """For each `ExpectedChunk`, returns a "target" dict:

        {"chunk_ids": set[str], "document": str, "text_anchor": str,
         "relevance": int, "resolved": bool, "ambiguous": bool}

    `chunk_ids` is the set of CURRENT chunk_id values matching that spec
    against `db_path` right now (0, 1, or - if the document/ordinal/anchor
    combination is genuinely ambiguous in the index - more than one).
    `resolved=False` means the spec's document/ordinal/text_anchor no longer
    matches anything in the current index (deleted file, re-chunked away,
    text_anchor typo'd, etc.) - a real signal that the ground truth itself
    needs re-verification, kept distinct from "resolved fine, just not
    retrieved" (which is what recall@k is supposed to measure)."""
    targets: list[dict] = []
    if not expected_chunks:
        return targets

    # One connection, one full documents+chunks scan, filtered in Python with
    # the same diacritics/case-insensitive fold used everywhere else in
    # metrics.py - substring matching against Czech filenames cannot rely on
    # SQLite's ASCII-only LIKE folding.
    with ai_search.database(db_path) as con:
        rows = con.execute(
            "SELECT d.name, d.path, c.id, c.ordinal, c.text FROM chunks c JOIN documents d ON d.id=c.document_id"
        ).fetchall()

    for spec in expected_chunks:
        if spec.chunk_id:
            # Legacy/explicit escape hatch: trust the given id as-is, no lookup.
            targets.append({
                "chunk_ids": {spec.chunk_id}, "document": spec.document, "text_anchor": spec.text_anchor,
                "relevance": spec.relevance, "resolved": True, "ambiguous": False,
            })
            continue

        needle_doc = _fold(spec.document)
        needle_anchor = _fold(spec.text_anchor)
        matches: set[str] = set()
        for name, path, chunk_id, ordinal, text in rows:
            if needle_doc and needle_doc not in _fold(name) and needle_doc not in _fold(path):
                continue
            if spec.ordinal is not None and ordinal != spec.ordinal:
                continue
            if needle_anchor and needle_anchor not in _fold(text):
                continue
            matches.add(chunk_id)

        targets.append({
            "chunk_ids": matches, "document": spec.document, "text_anchor": spec.text_anchor,
            "relevance": spec.relevance, "resolved": bool(matches), "ambiguous": len(matches) > 1,
        })

    return targets
