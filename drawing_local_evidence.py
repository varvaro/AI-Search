"""PR9.7.4 — document-local evidence selection for drawing navigation.

This module may read SQLite, but only chunks belonging to a document that is
already present in ranked search results. It never performs global retrieval,
changes result order, or infers a drawing subtype from chunk text.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ai_search_config import DATABASE_DIR
import document_class_affinity
import drawing_navigation
from drawing_navigation import DrawingSubtype


_SOURCE_DATABASE = {
    "Dokument": "project.sqlite3",
    "E-mail": "emails.sqlite3",
    "Poznámka": "notes.sqlite3",
}
_TOKEN = re.compile(r"[a-z0-9]{3,}")
_SUBTYPE_TOKENS = {
    DrawingSubtype.PLAN: ("pudorys",),
    DrawingSubtype.SECTION: ("rez",),
    DrawingSubtype.SCHEME: ("schema", "schem"),
    DrawingSubtype.DETAIL: ("detail",),
    DrawingSubtype.SITUATION: ("situac",),
}


@dataclass(frozen=True)
class LocalEvidence:
    chunk_id: str
    ordinal: int
    heading: str
    quote: str
    subject_overlap: int
    subtype_overlap: int


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(drawing_navigation.fold(text or "")))


def _token_matches(left: str, right: str) -> bool:
    return left == right or (
        min(len(left), len(right)) >= 4
        and (left.startswith(right) or right.startswith(left))
    )


def _overlap_count(needles: Iterable[str], haystack: Iterable[str]) -> int:
    words = tuple(haystack)
    return sum(any(_token_matches(needle, word) for word in words) for needle in needles)


def _filename_subtypes(result: dict) -> frozenset[DrawingSubtype]:
    return drawing_navigation.classify_result_subtypes(
        {
            "document": result.get("document"),
            "path": result.get("path"),
            "heading": "",
            "quote": "",
        }
    )


def _supports_subtype(result: dict, subtype: DrawingSubtype) -> bool:
    found = _filename_subtypes(result)
    if subtype is DrawingSubtype.GENERIC_DRAWING:
        return bool(found)
    return subtype in found


def _database_path(result: dict, database_dir: Path | None) -> Path | None:
    source = str(result.get("source") or "Dokument")
    filename = _SOURCE_DATABASE.get(source)
    if filename is None:
        return None
    return Path(database_dir or DATABASE_DIR) / filename


def _clean_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def select_local_evidence(
    query: str,
    document_id: int,
    current_result: dict,
    subtype: DrawingSubtype,
    *,
    database_dir: Path | None = None,
) -> LocalEvidence | None:
    """Return the best subject-bearing chunk from one already-selected drawing.

    The filename/path must independently prove both DRAWING class and the
    requested subtype. Chunk text is used only to find subject evidence.
    """
    if not drawing_navigation.is_drawing_navigation_query(query):
        return None
    if document_class_affinity.classify_document(
        str(current_result.get("document") or ""),
        str(current_result.get("path") or ""),
    ) is not document_class_affinity.DocumentClass.DRAWING:
        return None
    if not _supports_subtype(current_result, subtype):
        return None

    subject = drawing_navigation.subject_tokens(query)
    if not subject:
        return None
    try:
        normalized_document_id = int(document_id)
    except (TypeError, ValueError):
        return None

    db_path = _database_path(current_result, database_dir)
    if db_path is None or not db_path.is_file():
        return None

    uri = f"file:{db_path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            document = connection.execute(
                "SELECT name, path, content_hash FROM documents WHERE id=?",
                (normalized_document_id,),
            ).fetchone()
            if document is None:
                return None
            indexed_name, indexed_path, content_hash = document
            if (
                str(indexed_name) != str(current_result.get("document") or "")
                or str(indexed_path) != str(current_result.get("path") or "")
            ):
                return None
            chunk_prefix = f"{content_hash}:"
            chunks = connection.execute(
                """
                SELECT id, ordinal, heading, text
                FROM chunks
                WHERE id>=? AND id<? AND document_id=?
                ORDER BY ordinal
                """,
                (chunk_prefix, f"{content_hash};", normalized_document_id),
            ).fetchall()
    except sqlite3.Error:
        return None

    subtype_tokens = _SUBTYPE_TOKENS.get(subtype, ())
    best: tuple[tuple[int, int, int, int], LocalEvidence] | None = None
    for chunk_id, ordinal, heading, text in chunks:
        heading_tokens = _tokens(heading)
        body_tokens = _tokens(text)
        all_tokens = heading_tokens + body_tokens
        subject_overlap = _overlap_count(subject, all_tokens)
        if subject_overlap == 0:
            continue
        subtype_overlap = _overlap_count(subtype_tokens, all_tokens)
        heading_overlap = _overlap_count(subject + subtype_tokens, heading_tokens)
        body_overlap = _overlap_count(subject + subtype_tokens, body_tokens)
        quote = _clean_quote(text)
        if not quote:
            continue
        evidence = LocalEvidence(
            chunk_id=str(chunk_id),
            ordinal=int(ordinal),
            heading=str(heading or ""),
            quote=quote,
            subject_overlap=subject_overlap,
            subtype_overlap=subtype_overlap,
        )
        score = (
            subject_overlap,
            subtype_overlap,
            heading_overlap,
            body_overlap,
        )
        if best is None or score > best[0]:
            best = (score, evidence)
    return best[1] if best is not None else None


def enrich_results(
    query: str,
    results: list[dict],
    requested_subtypes: Iterable[DrawingSubtype],
    *,
    database_dir: Path | None = None,
) -> list[dict]:
    """Override evidence on eligible rows without changing order or ranking."""
    specific_subtypes = tuple(
        subtype
        for subtype in requested_subtypes
        if subtype is not DrawingSubtype.GENERIC_DRAWING
    )
    if not specific_subtypes:
        return results

    enriched = results
    for index, result in enumerate(results):
        document_id = result.get("document_id")
        if document_id is None:
            continue
        subtype = next(
            (
                candidate
                for candidate in specific_subtypes
                if _supports_subtype(result, candidate)
            ),
            None,
        )
        if subtype is None:
            continue
        evidence = select_local_evidence(
            query,
            document_id,
            result,
            subtype,
            database_dir=database_dir,
        )
        if evidence is None:
            continue
        if enriched is results:
            enriched = list(results)
        updated = dict(result)
        updated["quote"] = evidence.quote
        updated["heading"] = evidence.heading
        updated["chunk_id"] = evidence.chunk_id
        updated["_drawing_local_evidence"] = {
            "ordinal": evidence.ordinal,
            "subject_overlap": evidence.subject_overlap,
            "subtype_overlap": evidence.subtype_overlap,
        }
        enriched[index] = updated
    return enriched
