"""PR8.2 — Revision-aware ranking (experimental, flag-gated).

When the query expresses a *current/final revision* intent, apply a small
additive Phase-3 score from filename/path revision markers.

Does NOT:
  * change FTS / Lance / embeddings
  * apply "newer is always better" without revision intent
  * touch answer() / evidence_runtime / document_state
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

# Tunables — same order of magnitude as entity / filename bonuses.
REVISION_CURRENCY_BONUS = 0.03   # akt_ / final marker
REVISION_DATE_BONUS_MAX = 0.03   # fresher parsed date (intent-gated only)
REVISION_TOPIC_BONUS = 0.03      # query topic aligns with HMG vs contract/final
REVISION_OLD_PENALTY = -0.05     # OLD/ or /old/ path segment
REVISION_DRAFT_PENALTY = -0.03   # draft / návrh / vzor
REVISION_SCORE_CAP = 0.08        # |net| clamp (allows topic + currency + date)

# Intent: only these activate any revision adjustment.
_INTENT_PATTERNS = (
    "aktualni", "aktualne", "aktual",
    "platny", "platna", "platne", "plati", "platnost",
    "posledni",
    "finalni verze", "finalni",
)

_OLD_SEGMENT_RE = re.compile(r"(?:^|/)(?:old)(?:/|$)", re.IGNORECASE)
# Currency stamp only — must NOT match Czech "aktualizace" / "aktualni".
_AKT_RE = re.compile(
    r"(?:^|[_\s.])akt_\d|(?:^|[_\s.])akt\.\d|_akt(?:[_.]|$)|akt_",
    re.IGNORECASE,
)
_FINAL_RE = re.compile(r"(?:^|/)final(?:/|$)|(?:^|[_\s.])final(?:[_\s.]|$)", re.IGNORECASE)
_DRAFT_RE = re.compile(
    r"(?:^|[_\s./])(?:draft|navrh|návrh|vzor|vzorova|vzorová)(?:[_\s./]|$)",
    re.IGNORECASE,
)
# Dates common in NDS filenames: 4.08.2026, 11.12.25, 250129, 14.8.24
_DATE_DMY_RE = re.compile(
    r"(?<!\d)(\d{1,2})[.](\d{1,2})[.](\d{2,4})(?!\d)"
)
_DATE_YMD_COMPACT_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")  # YYMMDD heuristic

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def query_has_revision_intent(query: str) -> bool:
    folded = fold(query)
    return any(p in folded for p in _INTENT_PATTERNS)


def _parse_dates(blob: str) -> list[date]:
    """Extract plausible document dates from a folded name/path blob."""
    out: list[date] = []
    for m in _DATE_DMY_RE.finditer(blob):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        if y < 1990 or y > 2100 or mo < 1 or mo > 12 or d < 1 or d > 31:
            continue
        try:
            out.append(date(y, mo, d))
        except ValueError:
            continue
    # Compact YYMMDD only when it looks like a construction stamp (20–29 year).
    for m in _DATE_YMD_COMPACT_RE.finditer(blob):
        yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yy < 20 or yy > 35 or mm < 1 or mm > 12 or dd < 1 or dd > 31:
            continue
        try:
            out.append(date(2000 + yy, mm, dd))
        except ValueError:
            continue
    return out


def _newest_date(blob: str) -> date | None:
    dates = _parse_dates(blob)
    return max(dates) if dates else None


def _date_freshness_bonus(newest: date | None, *, today: date | None = None) -> float:
    """Mild boost for newer dates — only meaningful under revision intent.

    Maps document date to [0, REVISION_DATE_BONUS_MAX] relative to a 1-year
    window ending at `today` (short enough that months separate akt_ stamps).
    Undated → 0 (no global preference).
    """
    if newest is None:
        return 0.0
    ref = today or date.today()
    age_days = (ref - newest).days
    if age_days < 0:
        return REVISION_DATE_BONUS_MAX
    window = 365
    if age_days >= window:
        return 0.0
    return REVISION_DATE_BONUS_MAX * (1.0 - age_days / window)


@dataclass(frozen=True)
class RevisionScoreDetail:
    intent: bool
    bonus: float
    signals: tuple[str, ...]

    def as_trace_dict(self) -> dict:
        return {
            "intent": self.intent,
            "bonus": self.bonus,
            "signals": list(self.signals),
        }


def compute_revision_score(
    query: str,
    document_name: str,
    document_path: str,
    *,
    today: date | None = None,
) -> RevisionScoreDetail:
    """Intent-gated revision adjustment for one candidate.

    Returns bonus=0 when the query has no revision intent (byte-identical
    ranking contribution for non-revision queries).
    """
    if not query_has_revision_intent(query):
        return RevisionScoreDetail(intent=False, bonus=0.0, signals=())

    name = document_name or ""
    path = document_path or ""
    blob = f"{name} {path}"
    folded_blob = fold(blob)
    signals: list[str] = []
    total = 0.0

    # Penalties first — stale/draft must not be rescued by a coincidental date.
    if _OLD_SEGMENT_RE.search(path) or _OLD_SEGMENT_RE.search(name):
        total += REVISION_OLD_PENALTY
        signals.append("penalty:old_segment")
    elif "/old/" in folded_blob or folded_blob.startswith("old/") or "/old " in folded_blob:
        total += REVISION_OLD_PENALTY
        signals.append("penalty:old_segment")

    if _DRAFT_RE.search(blob) or any(
        tok in folded_blob for tok in ("draft", "navrh", "vzorova", "vzorove", "vzor")
    ):
        # Avoid penalizing every path containing unrelated 'vzor' substring in
        # the middle of Czech words by requiring draft/navrh or vzor as token.
        draft_hit = bool(_DRAFT_RE.search(blob))
        if not draft_hit:
            tokens = set(_TOKEN_RE.findall(folded_blob))
            draft_hit = bool(tokens & {"draft", "navrh", "vzor", "vzorova", "vzorove"})
        if draft_hit:
            total += REVISION_DRAFT_PENALTY
            signals.append("penalty:draft")

    # Currency markers.
    qf = fold(query)
    wants_schedule = "harmonogram" in qf or "hmg" in qf
    wants_contract = any(t in qf for t in ("sod", "smlouva", "final", "finalni"))
    has_hmg = "hmg" in folded_blob or "harmonogram" in folded_blob
    has_final_path = bool(
        _FINAL_RE.search(path) or _FINAL_RE.search(name) or "/final/" in folded_blob
    )

    has_currency = False
    if _AKT_RE.search(name) or _AKT_RE.search(path) or "akt_" in folded_blob or re.search(r"akt\.\d", folded_blob):
        total += REVISION_CURRENCY_BONUS
        signals.append("boost:akt")
        has_currency = True
    # Final-folder currency: skip on pure schedule queries so SoD/final does not
    # outrank HMG when the user asked for harmonogram/HMG.
    if has_final_path and not (wants_schedule and not wants_contract):
        total += REVISION_CURRENCY_BONUS
        signals.append("boost:final")
        has_currency = True

    if wants_schedule and has_hmg:
        total += REVISION_TOPIC_BONUS
        signals.append("boost:topic_hmg")
    elif wants_contract and has_final_path:
        total += REVISION_TOPIC_BONUS
        signals.append("boost:topic_final")

    # Fresher date only stacks onto currency markers — never a standalone
    # "newer filename date wins" rule (KD_* dates would otherwise outrank HMG).
    newest = _newest_date(folded_blob)
    date_bonus = _date_freshness_bonus(newest, today=today) if has_currency else 0.0
    if date_bonus > 0:
        total += date_bonus
        signals.append(f"boost:date:{newest.isoformat()}")

    # Clamp net adjustment.
    if total > REVISION_SCORE_CAP:
        total = REVISION_SCORE_CAP
    elif total < -REVISION_SCORE_CAP:
        total = -REVISION_SCORE_CAP

    return RevisionScoreDetail(intent=True, bonus=total, signals=tuple(signals))
