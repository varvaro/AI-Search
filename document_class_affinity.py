"""PR9.4.2 / PR9.6.0 — Query-class ↔ document-class affinity (flag-gated).

Additive Phase-3 score from a *generic* compatibility between the query's
document-type intent and one candidate's filename/path/extension.

Does NOT:
  * change FTS / Lance / embeddings / retrieval pool / QA rerank
  * read chunk heading, quote, or body
  * apply recency, revision, OLD/final/návrh, or signed-state preference
  * hardcode vendors, NOT-ids, drawing numbers, or project folders
  * replace document_state (signed-contract answer safety stays there)
  * boost every DRAWING filename on a DRAWING query (PR9.6.0: match bonus
    stays 0 for that pair; only textual/admin mismatch is applied)

Status / signed-contract questions always return bonus=0. A letter of intent
(LOI) is never treated as a full CONTRACT match. See
tests/test_document_class_affinity_pr942.py and
tests/test_document_class_affinity_pr960.py.

`fold()` is local, matching this repo's per-module normalization convention.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

CLASS_MATCH_BONUS = 0.03
CLASS_MISMATCH_PENALTY = -0.015


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


class QueryClass(str, Enum):
    DRAWING = "DRAWING"
    MINUTES = "MINUTES"
    SCHEDULE = "SCHEDULE"
    CONTRACT = "CONTRACT"
    BUDGET = "BUDGET"
    TECHNICAL_REPORT = "TECHNICAL_REPORT"
    UNKNOWN = "UNKNOWN"


class DocumentClass(str, Enum):
    DRAWING = "DRAWING"
    MINUTES = "MINUTES"
    SCHEDULE = "SCHEDULE"
    CONTRACT = "CONTRACT"
    LETTER_OF_INTENT = "LETTER_OF_INTENT"
    BUDGET = "BUDGET"
    OFFER = "OFFER"
    TECHNICAL_REPORT = "TECHNICAL_REPORT"
    REGULATORY = "REGULATORY"
    UNKNOWN = "UNKNOWN"


# Query intent. Declension-tolerant stems; no project vocabulary.
_Q_DRAWING = re.compile(r"vykres|schem")
_Q_DRAWING_PLAN = re.compile(r"\bplan")  # only with a drawing-doc companion
_Q_DRAWING_CTX = re.compile(r"vykres|schem|pudorys|rez\b|dokumentac")
_Q_MINUTES = re.compile(r"kontroln\w*\s+dn|zapis\w*\s+(?:z\s+)?(?:kontroln|kd)|ukol\w*.{0,24}kontroln")
_Q_KD = re.compile(r"(?<![a-z0-9])kd(?![a-z0-9])")
_Q_SCHEDULE = re.compile(r"harmonogram|(?<![a-z0-9])hmg(?![a-z0-9])")
_Q_CONTRACT = re.compile(
    r"smlouv|(?<![a-z0-9])sod(?![a-z0-9])|predan\w*\s+dil|zavazn\w*\s+smluv"
)
_Q_BUDGET = re.compile(r"rozpocet")
_Q_TZ = re.compile(r"technick\w+\s+zprav")
_Q_TZ_TOKEN = re.compile(r"(?<![a-z0-9])tz(?![a-z0-9])")

# Contract *status* questions — affinity is forced to 0 so a generic
# CONTRACT/LOI/signed filename can never become evidence that a SoD exists.
_Q_SIGNED = re.compile(r"podepsan")
_Q_STATUS_CONTRACT = re.compile(
    r"smlouv|(?<![a-z0-9])sod(?![a-z0-9])"
)
_Q_STATUS_WORD = re.compile(r"status")

# Document filename/path. BUDGET is filename-only (see classify_document).
_D_LOI = re.compile(r"(?<![a-z0-9])loi(?![a-z0-9])|letter\s+of\s+intent")
_D_MINUTES = re.compile(
    r"kontroln\w*\s+dn|(?<![a-z0-9])kd(?![a-z0-9])|(?<![a-z0-9])\d{1,3}\.kd(?![a-z0-9])"
)
_D_SCHEDULE = re.compile(r"harmonogram|(?<![a-z0-9])hmg(?![a-z0-9])")
_D_BUDGET = re.compile(r"rozpocet")
_D_OFFER = re.compile(r"nabidk|cenova\s+nabidka|poptavk")
_D_CONTRACT = re.compile(r"smlouv|(?<![a-z0-9])sod(?![a-z0-9])")
_D_TZ = re.compile(r"technick\w*\s*zprav|(?<![a-z0-9])tz(?![a-z0-9])")
# Administrative / regulatory filings. Filename+path only; stems are
# declension-tolerant and must not fire on ordinary construction nouns.
_D_REGULATORY = re.compile(r"rozhodnut|povolen|podmink|stanovisk|vyjadren")
_D_DRAWING_WORD = re.compile(r"vykres|schem")
_D_DRAWING_CODE = re.compile(r"(?<![a-z0-9])[a-z]\.\d+(?:\.\d+){1,4}(?![a-z0-9])")


def is_signed_contract_status_query(query: str) -> bool:
    """True when the user asks about contract/SoD existence or signed state.

    Smallest safety gate from the PR9.4.2 preflight: status questions must
    not receive any class-affinity bonus (positive or negative).
    """
    q = fold(query)
    if not _Q_STATUS_CONTRACT.search(q):
        return False
    return bool(_Q_SIGNED.search(q) or _Q_STATUS_WORD.search(q))


def classify_query(query: str) -> QueryClass:
    q = fold(query or "")
    if not q:
        return QueryClass.UNKNOWN
    if _Q_DRAWING.search(q) or (_Q_DRAWING_PLAN.search(q) and _Q_DRAWING_CTX.search(q)):
        return QueryClass.DRAWING
    if _Q_MINUTES.search(q) or _Q_KD.search(q):
        return QueryClass.MINUTES
    if _Q_SCHEDULE.search(q):
        return QueryClass.SCHEDULE
    if _Q_BUDGET.search(q):
        return QueryClass.BUDGET
    if _Q_TZ.search(q) or _Q_TZ_TOKEN.search(q):
        return QueryClass.TECHNICAL_REPORT
    if _Q_CONTRACT.search(q):
        return QueryClass.CONTRACT
    return QueryClass.UNKNOWN


def classify_document(name: str, path: str = "") -> DocumentClass:
    """Classify from filename + extension + path. Budget is filename-only."""
    name = name or ""
    path = path or ""
    ext = Path(name).suffix.lower()
    name_f = fold(name)
    path_f = fold(path)
    blob = f"{name_f} {path_f}"

    if _D_LOI.search(blob):
        return DocumentClass.LETTER_OF_INTENT
    if _D_MINUTES.search(blob):
        return DocumentClass.MINUTES
    if _D_SCHEDULE.search(blob):
        return DocumentClass.SCHEDULE
    # Budget: filename only. A parent folder named "rozpočet" plus an offer
    # filename must not become BUDGET (PR9.4.2 preflight).
    if _D_BUDGET.search(name_f):
        return DocumentClass.BUDGET
    if _D_OFFER.search(name_f):
        return DocumentClass.OFFER
    if _D_CONTRACT.search(blob):
        return DocumentClass.CONTRACT
    if _D_TZ.search(name_f) or _D_TZ.search(path_f):
        return DocumentClass.TECHNICAL_REPORT
    if _D_REGULATORY.search(blob):
        return DocumentClass.REGULATORY
    if _D_DRAWING_WORD.search(blob):
        return DocumentClass.DRAWING
    if ext == ".pdf" and _D_DRAWING_CODE.search(name_f):
        return DocumentClass.DRAWING
    return DocumentClass.UNKNOWN


# query class → (match document classes, mismatch document classes)
_AFFINITY: dict[QueryClass, tuple[frozenset[DocumentClass], frozenset[DocumentClass]]] = {
    QueryClass.DRAWING: (
        frozenset({DocumentClass.DRAWING}),
        frozenset({
            DocumentClass.CONTRACT, DocumentClass.MINUTES,
            DocumentClass.BUDGET, DocumentClass.OFFER, DocumentClass.LETTER_OF_INTENT,
            DocumentClass.TECHNICAL_REPORT, DocumentClass.REGULATORY,
        }),
    ),
    QueryClass.MINUTES: (
        frozenset({DocumentClass.MINUTES}),
        frozenset({DocumentClass.CONTRACT, DocumentClass.LETTER_OF_INTENT}),
    ),
    QueryClass.SCHEDULE: (
        frozenset({DocumentClass.SCHEDULE}),
        frozenset({DocumentClass.CONTRACT, DocumentClass.MINUTES, DocumentClass.LETTER_OF_INTENT}),
    ),
    QueryClass.CONTRACT: (
        frozenset({DocumentClass.CONTRACT}),
        frozenset({DocumentClass.MINUTES, DocumentClass.BUDGET, DocumentClass.OFFER}),
    ),
    QueryClass.BUDGET: (
        frozenset({DocumentClass.BUDGET}),
        frozenset({DocumentClass.OFFER, DocumentClass.CONTRACT, DocumentClass.LETTER_OF_INTENT}),
    ),
    QueryClass.TECHNICAL_REPORT: (
        frozenset({DocumentClass.TECHNICAL_REPORT}),
        frozenset({DocumentClass.OFFER, DocumentClass.MINUTES, DocumentClass.CONTRACT}),
    ),
}


@dataclass(frozen=True)
class ClassAffinityDetail:
    query_class: QueryClass
    document_class: DocumentClass
    bonus: float
    reason: str
    status_query: bool = False

    def as_trace_dict(self) -> dict:
        return {
            "query_class": self.query_class.value,
            "document_class": self.document_class.value,
            "bonus": self.bonus,
            "reason": self.reason,
            "status_query": self.status_query,
        }


def compute_class_affinity(query: str, document_name: str, document_path: str = "") -> ClassAffinityDetail:
    """Additive Phase-3 bonus. Pure; name/path only."""
    qclass = classify_query(query)
    dclass = classify_document(document_name, document_path)
    if is_signed_contract_status_query(query):
        return ClassAffinityDetail(
            query_class=qclass, document_class=dclass, bonus=0.0,
            reason="status_query_zero", status_query=True,
        )
    if qclass is QueryClass.UNKNOWN or dclass is DocumentClass.UNKNOWN:
        return ClassAffinityDetail(
            query_class=qclass, document_class=dclass, bonus=0.0, reason="unknown",
        )
    if dclass is DocumentClass.LETTER_OF_INTENT:
        # LOI is never a full CONTRACT match (status-02 safety).
        return ClassAffinityDetail(
            query_class=qclass, document_class=dclass, bonus=0.0, reason="loi_excluded",
        )
    match, mismatch = _AFFINITY.get(qclass, (frozenset(), frozenset()))
    if dclass in match:
        # PR9.6.0: a DRAWING query must not boost every DRAWING filename
        # (+0.03 promoted off-topic sheets over the topic hit). Other
        # query classes keep the historical match bonus.
        if qclass is QueryClass.DRAWING:
            return ClassAffinityDetail(
                query_class=qclass, document_class=dclass,
                bonus=0.0, reason="drawing_match_neutral",
            )
        return ClassAffinityDetail(
            query_class=qclass, document_class=dclass,
            bonus=CLASS_MATCH_BONUS, reason="class_match",
        )
    if dclass in mismatch:
        return ClassAffinityDetail(
            query_class=qclass, document_class=dclass,
            bonus=CLASS_MISMATCH_PENALTY, reason="class_mismatch",
        )
    return ClassAffinityDetail(
        query_class=qclass, document_class=dclass, bonus=0.0, reason="neutral",
    )
