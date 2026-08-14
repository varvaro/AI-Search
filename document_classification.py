"""Document kind + project-conflict classification (PR7.6.1).

Pure, additive helpers for EvidenceRuntime safety decisions. Classifies *what
kind of document* a filename/path names (LOI vs SoD vs order …) and whether
a retrieved chunk's *content project* contradicts the name/path project.

Does NOT:
  * touch retrieval / ranking / embeddings
  * call SQLite / LanceDB / Ollama
  * rewrite answers (consumers in evidence_runtime / the answer gate do that)
"""
from __future__ import annotations

import re
import unicodedata
from enum import Enum


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


class DocumentKind(str, Enum):
    """Coarse document-type classification from filename/path only."""

    CONTRACT = "CONTRACT"          # SoD / smlouva / signed agreement
    LOI = "LOI"                    # letter of intent
    ORDER = "ORDER"                # objednávka
    AMENDMENT = "AMENDMENT"        # dodatek
    MINUTES = "MINUTES"            # zápis / KD
    TECH_SPEC = "TECH_SPEC"        # TP / KZP / technická specifikace
    EMAIL = "EMAIL"                # .msg / .eml
    OTHER = "OTHER"


# Filename/path markers (folded). First match wins — order matters.
_KIND_RULES: tuple[tuple[DocumentKind, re.Pattern[str]], ...] = (
    (DocumentKind.EMAIL, re.compile(r"\.(msg|eml)\b")),
    (DocumentKind.LOI, re.compile(r"(?<![a-z0-9])(loi|letter\s*of\s*intent)(?![a-z0-9])")),
    (DocumentKind.AMENDMENT, re.compile(r"(?<![a-z0-9])(dodatek|dod\d*|amendment)(?![a-z0-9])")),
    (DocumentKind.ORDER, re.compile(r"(?<![a-z0-9])(objednavk\w*|obj(?:\.|,|_|\b)|order\b)")),
    (DocumentKind.MINUTES, re.compile(r"(?<![a-z0-9])(zapis\w*|kontrolni\s*den|\bkd\b)(?![a-z0-9])")),
    (DocumentKind.TECH_SPEC, re.compile(
        r"(?<![a-z0-9])(technick\w*\s*spec|technick\w*\s*list|\btp\b|\bkzp\b|"
        r"technologick\w*\s*postup)(?![a-z0-9])"
    )),
    # CONTRACT last among positives so SoD/smlouva wins over weaker noise.
    (DocumentKind.CONTRACT, re.compile(
        r"(?<![a-z0-9])(smlouv\w*|\bsod\b|contract|dohoda\s+o\s+d[ií]lo)(?![a-z0-9])"
    )),
)


def classify_document_kind(document: str, path: str = "") -> DocumentKind:
    """Classify document kind from filename + path. Pure; no I/O."""
    name = (document or "").strip()
    if not name:
        raw = (path or "").rstrip("/\\")
        name = raw.replace("\\", "/").rsplit("/", 1)[-1] if raw else ""
    haystack = _fold(f"{name} {path or ''}")
    if not haystack.strip():
        return DocumentKind.OTHER
    for kind, pattern in _KIND_RULES:
        if pattern.search(haystack):
            return kind
    # A signed BOZP / SoD-style agreement often lacks the literal "smlouva"
    # token (e.g. NOT250060_BOZP_SafetyPeak_podepsaná.pdf). Treat signed
    # agreement-looking names as CONTRACT when they are not an excluded kind.
    if re.search(r"(?<![a-z0-9])(podepsan\w*|signed)(?![a-z0-9])", haystack):
        if re.search(r"(?<![a-z0-9])(bozp|sod|smlouv)", haystack) or "/podepsan" in haystack:
            return DocumentKind.CONTRACT
    return DocumentKind.OTHER


# Doc-type terms (folded) that mean the query asks for a signed *contract*/SoD.
_CONTRACT_QUERY_TERMS = frozenset({
    "smlouva", "smlouvy", "smlouvu", "smlouve", "sod", "contract", "bozp",
})

# Doc-type tokens that ALSO identify the *subject* of the agreement
# ("smlouva na BOZP" must not be confirmed by an unrelated signed SoD).
_SUBJECT_DOC_TYPE_TERMS = frozenset({"bozp"})


def query_requires_signed_contract(doc_type_terms: tuple[str, ...] | list[str]) -> bool:
    """True when the signed-intent query is specifically about a smlouva/SoD."""
    return any(_fold(t) in _CONTRACT_QUERY_TERMS for t in (doc_type_terms or ()))


def subject_doc_type_terms(doc_type_terms: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Doc-type tokens that also identify the agreement subject (e.g. BOZP)."""
    return tuple(_fold(t) for t in (doc_type_terms or ()) if _fold(t) in _SUBJECT_DOC_TYPE_TERMS)


def _subject_terms(doc_type_terms: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return subject_doc_type_terms(doc_type_terms)


def signed_evidence_is_contract_for_query(
    doc_type_terms: tuple[str, ...] | list[str],
    document: str,
    path: str = "",
) -> bool:
    """Whether a SIGNED document may confirm a signed-*contract* query.

    LOI / order / amendment / minutes / email / tech-spec never confirm
    "podepsaná smlouva", even when their filename contains 'signed'/'podepsan*'.
    When the query names a subject doc-type (BOZP), the signed document must
    also carry that subject — otherwise any signed SoD would falsely confirm
    "smlouva na BOZP".
    """
    if not query_requires_signed_contract(doc_type_terms):
        return True
    kind = classify_document_kind(document, path)
    if kind is not DocumentKind.CONTRACT:
        return False
    subjects = _subject_terms(doc_type_terms)
    if not subjects:
        return True
    haystack = _fold(f"{document} {path}")
    return any(subject in haystack for subject in subjects)


# ---------------------------------------------------------------------------
# Project content conflict (name says project A, body says project B)
# ---------------------------------------------------------------------------

# Local project name fragments that appear in NDS filenames / paths.
_LOCAL_PROJECT_NAME_RE = re.compile(
    r"(?<![a-z0-9])(nds|nd\s*smichov|garaze?\s*nd|240783160)(?![a-z0-9])",
)

# Foreign projects observed as content traps in the NDS index.
_FOREIGN_PROJECT_CONTENT_RE = re.compile(
    r"(?<![a-z0-9])("
    r"palac\s*dunaj|palace?\s*dunaj|"
    r"evropsk\w*\s*parlament|european\s*parliament"
    r")(?![a-z0-9])",
)


def document_name_implies_local_project(document: str, path: str = "") -> bool:
    return bool(_LOCAL_PROJECT_NAME_RE.search(_fold(f"{document} {path}")))


def content_implies_foreign_project(quote: str) -> bool:
    return bool(_FOREIGN_PROJECT_CONTENT_RE.search(_fold(quote or "")))


def detect_project_conflict(document: str, path: str = "", quote: str = "") -> bool:
    """True when the filename/path claims the local project but the chunk body
    names a different project (e.g. NDS_seznam…xlsx containing PALÁC DUNAJ)."""
    if not document_name_implies_local_project(document, path):
        return False
    return content_implies_foreign_project(quote)
