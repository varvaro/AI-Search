"""PR9.5.0 — Deterministic OCR candidate scoring.

Pure functions over already-extracted text. No filesystem, no subprocess,
no PDF libraries, no I/O, no project/filename hardcode.

The caller must pass ``indexable_text`` built from ``chunks()`` output
(heading + body), not the raw Tesseract string. Scoring therefore prefers
text that will actually land in FTS/Lance over a longer raw dump that the
chunker would drop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# Word tokens: Unicode letters, length >= 3. Digits/underscores excluded so
# isolated CAD measures ("196", "R1639") do not inflate the count.
_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_DIACRITICS = frozenset("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")

# Weak-OCR gates. Tuned to fire on short glyph soup ("FSS bz REN S") while
# leaving a normal English or Czech paragraph alone. Diacritics are NOT a
# gate — a valid English scan must not be treated as weak.
WEAK_MIN_WORD_TOKENS = 12
WEAK_MIN_USEFUL_CHARS = 80
WEAK_MAX_NOISE_RATIO = 0.50


@dataclass(frozen=True)
class OCRCandidate:
    mode: str
    raw_text: str
    indexable_text: str


@dataclass(frozen=True)
class OCRQualityScore:
    word_tokens: int
    useful_chars: int
    diacritic_chars: int
    noise_ratio: float
    mean_token_length: float
    value: float


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def score_candidate(candidate: OCRCandidate) -> OCRQualityScore:
    """Score ``indexable_text`` only. ``raw_text`` is ignored on purpose."""
    text = candidate.indexable_text or ""
    tokens = _tokens(text)
    word_tokens = len(tokens)
    useful_chars = sum(1 for ch in text if ch.isalnum() or ch.isspace())
    diacritic_chars = sum(1 for ch in text if ch in _DIACRITICS)
    length = len(text)
    if length == 0:
        noise_ratio = 1.0
    else:
        noise_ratio = 1.0 - (useful_chars / length)
    mean_token_length = (sum(len(tok) for tok in tokens) / word_tokens) if word_tokens else 0.0
    value = (
        word_tokens * 10.0
        + useful_chars * 1.0
        + diacritic_chars * 0.25
        + min(mean_token_length, 12.0) * 0.5
        - noise_ratio * 40.0
    )
    return OCRQualityScore(
        word_tokens=word_tokens,
        useful_chars=useful_chars,
        diacritic_chars=diacritic_chars,
        noise_ratio=noise_ratio,
        mean_token_length=mean_token_length,
        value=value,
    )


def is_weak(candidate: OCRCandidate) -> bool:
    """True when the indexable text is too thin or too noisy to trust.

    Diacritics are not consulted. English prose with enough word tokens
    and useful characters is not weak.
    """
    score = score_candidate(candidate)
    if score.word_tokens < WEAK_MIN_WORD_TOKENS:
        return True
    if score.useful_chars < WEAK_MIN_USEFUL_CHARS:
        return True
    if score.noise_ratio >= WEAK_MAX_NOISE_RATIO:
        return True
    return False


def choose_best(candidates: Sequence[OCRCandidate]) -> OCRCandidate:
    """Highest ``value`` wins; a tie keeps the earlier candidate."""
    if not candidates:
        raise ValueError("choose_best requires at least one OCR candidate")
    winner = candidates[0]
    winner_value = score_candidate(winner).value
    for candidate in candidates[1:]:
        value = score_candidate(candidate).value
        if value > winner_value:
            winner = candidate
            winner_value = value
    return winner
