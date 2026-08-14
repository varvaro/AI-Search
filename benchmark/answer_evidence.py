"""Evidence tiers of one ai_search.answer() result — benchmark-side, read-only.

PR7.4 conflated three very different things under the single name "citations".
`answer()` returns `"citations": results` verbatim (ai_search.py), i.e. the whole
retrieval pool, including every distractor that happened to rank in the top-k.
Measuring "did the answer cite the wrong entity" against that blob answers a
different question — "was a distractor retrieved at all" — and on an adversarial
dataset that is true by construction.

Three tiers are therefore separated here:

  retrieved   every document in the pool. A distractor here is a RETRIEVAL
              observation, never an answer defect on its own.
  cited       documents the rendered answer actually leans on. The renderers
              (_render_answer_item / _render_structured_answer /
              _render_concise_answer) look the document name up from `results`
              and write it into the text verbatim - as "(Zdroj: <name>)" or
              under the trailing "Zdroje:" list - so a pool document whose exact
              name occurs in the answer body is one the model attributed a claim
              to. Nothing is parsed positionally; only exact name containment is
              used, which survives any future change of the render layout.
  state       documents the DocumentState verdict rested on, read from the
              additive `validation.state_documents` diagnostic key (PR7.2).

`evidence` = cited ∪ state is what a safety assertion must be judged against:
the union of "the answer said this document supports the claim" and "the gate
decided the lifecycle state from this document".

Pure functions over the answer dict. No ai_search import, no I/O.
"""
from __future__ import annotations

import unicodedata

# answer() appends this block to every rendered answer. It carries no document
# names (only counts and fixed wording, see _answer_confidence), but stripping
# it keeps name containment exact even for a document literally called e.g.
# "jistota.pdf".
_CONFIDENCE_MARKER = "\n\nJistota odpovědi:\n"


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def answer_body(answer: dict | None) -> str:
    """Rendered answer without the appended confidence block."""
    if not answer:
        return ""
    text = str(answer.get("answer") or "")
    head, marker, _tail = text.partition(_CONFIDENCE_MARKER)
    return head if marker else text


def retrieved_documents(answer: dict | None) -> list[str]:
    """Every document in the pool handed to answer(), in pool order."""
    if not answer:
        return []
    seen: list[str] = []
    for row in answer.get("citations") or []:
        name = str((row or {}).get("document") or "")
        if name and name not in seen:
            seen.append(name)
    return seen


def cited_documents(answer: dict | None) -> list[str]:
    """Pool documents whose exact name occurs in the rendered answer body.

    A document is "cited" when the renderer wrote its name into the text, which
    only happens for an item whose `zdroj_index` the model actually returned
    (_render_answer_item resolves the name from `results`, so the model can
    never invent or garble it).
    """
    body = fold(answer_body(answer))
    if not body:
        return []
    return [name for name in retrieved_documents(answer) if fold(name) in body]


def state_documents(answer: dict | None) -> list[str]:
    """Documents the DocumentState verdict rested on (validation diagnostics)."""
    if not answer:
        return []
    validation = answer.get("validation")
    if not isinstance(validation, dict):
        return []
    names: list[str] = []
    for entry in validation.get("state_documents") or []:
        name = str((entry or {}).get("document") or "") if isinstance(entry, dict) else str(entry or "")
        if name and name not in names:
            names.append(name)
    return names


def evidence_documents(answer: dict | None) -> list[str]:
    """cited ∪ state — what a safety assertion is judged against."""
    names = list(cited_documents(answer))
    for name in state_documents(answer):
        if name not in names:
            names.append(name)
    return names


def evidence_rows(answer: dict | None) -> list[dict]:
    """The pool rows behind `evidence_documents`, so a check can see the PATH.

    `cited`/`state` are document NAMES because that is all the rendered answer
    exposes. A revision trap ("do not answer from OLD/") lives in the path, not
    the name - `D.1.2.07 - schéma vyztužení 3.PP.pdf` is spelled identically
    inside and outside `OLD/`. Mapping the names back onto their pool rows is
    the only way to assert on the folder an answer leaned on.
    """
    if not answer:
        return []
    wanted = {fold(name) for name in evidence_documents(answer)}
    rows: list[dict] = []
    for row in answer.get("citations") or []:
        if not isinstance(row, dict):
            continue
        if fold(str(row.get("document") or "")) in wanted:
            rows.append(row)
    return rows


def rows_match_any(rows: list[dict], needles: list[str]) -> list[str]:
    """Needles matching any row's path (falling back to its document name)."""
    if not rows or not needles:
        return []
    blob = fold(" ".join(
        str(row.get("path") or "") + " " + str(row.get("document") or "") for row in rows
    ))
    return [n for n in needles if n and fold(n) in blob]


def _blob(names: list[str]) -> str:
    return fold(" ".join(names))


def match_any(names: list[str], needles: list[str]) -> list[str]:
    """Needles (substring, diacritics-insensitive) present in `names`."""
    blob = _blob(names)
    return [n for n in needles if n and fold(n) in blob]


def evidence_tiers(answer: dict | None) -> dict[str, list[str]]:
    """All tiers at once — what the runner stores per case for auditability."""
    cited = cited_documents(answer)
    state = state_documents(answer)
    evidence = list(cited)
    for name in state:
        if name not in evidence:
            evidence.append(name)
    return {
        "retrieved": retrieved_documents(answer),
        "cited": cited,
        "state": state,
        "evidence": evidence,
    }
