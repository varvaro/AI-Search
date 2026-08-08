"""Tests for ai_search.chunks()' oversized-chunk protection (chunking audit,
2026-08-07).

Background the assertions are built on: the audit measured that chunks() splits
only on blank lines and heading-like lines, so extracted text with neither
collapsed into a single chunk - 186 chunks in 125 documents exceeded BGE-M3's
8192-token limit and were silently truncated at embedding time, the worst
holding 32 488 tokens. CHUNK_MAX_SIZE/CHUNK_OVERLAP bound that; these tests pin
both the new guarantee and the fact that documents under the cap are split
exactly as before.

No database, no index, no model - chunks() is a pure function.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ai_search


def _old_chunks(text):
    """The pre-cap chunking algorithm, verbatim except for the deadline/progress
    plumbing. Test B compares against this rather than against a hand-written
    expectation, so it fails if the cap changes ANY boundary for a document that
    stays under the limit - not just the boundaries someone thought to list."""
    result, block, heading = [], [], ""

    def flush():
        value = "\n".join(block).strip()
        block.clear()
        if value:
            result.append((heading, value))

    for line in text.replace("\r\n", "\n").split("\n"):
        value = line.strip()
        is_heading = bool(re.match(
            r"^(#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+|[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ\s-]{3,})", value))
        if is_heading and len(value) < 160:
            flush()
            heading = re.sub(r"^#{1,6}\s+", "", value)
        elif not value:
            flush()
        else:
            block.append(line)
    flush()
    return result or ([("", text.strip())] if text.strip() else [])


# ---------------------------------------------------------------------------
# A) long text with no blank line - the exact shape that used to become one
#    monolithic chunk (measured in production: a 114 221-char contract in a
#    single line, of which BGE-M3 encoded only 25.2%)
# ---------------------------------------------------------------------------
TOKEN_COUNT = 3000  # ~27 000 chars of unique, individually identifiable words


def _token_text():
    """One single line, no blank lines, no heading-like line. Every word is
    unique, which lets the assertions below verify completeness, ordering and
    overlap without reconstructing the overlap arithmetic."""
    return " ".join(f"slovo{i:05d}" for i in range(TOKEN_COUNT))


def _tokens(text):
    return re.findall(r"slovo\d+", text)


def test_a_long_text_without_blank_lines_is_capped():
    text = _token_text()
    assert len(text) > 20_000, "vstupní podmínka testu: >20 000 znaků"
    assert "\n" not in text and "\n\n" not in text

    pieces = ai_search.chunks(text)

    assert len(pieces) > 1, "monolit musí být rozdělen"
    oversized = [len(body) for _, body in pieces if len(body) > ai_search.CHUNK_MAX_SIZE]
    assert not oversized, f"žádný chunk nesmí přesáhnout {ai_search.CHUNK_MAX_SIZE} znaků, nalezeno: {oversized}"


def test_a_long_text_content_is_fully_preserved_and_ordered():
    """Completeness AND no broken words in one pass: the set of words in the
    output must equal the input's (nothing lost), and every word in the output
    must be a whole input word (nothing cut in half, which would produce a
    token like 'slovo012' out of 'slovo01234')."""
    text = _token_text()
    pieces = ai_search.chunks(text)
    expected = [f"slovo{i:05d}" for i in range(TOKEN_COUNT)]

    seen = []
    for _, body in pieces:
        seen.extend(_tokens(body))

    assert set(seen) == set(expected), "žádné slovo se nesmí ztratit ani vzniknout"
    assert all(len(token) == len("slovo00000") for token in seen), "žádné slovo nesmí být rozříznuto"

    # order preserved: each chunk's first word comes no earlier than the previous chunk's first word
    first_indices = [int(_tokens(body)[0][5:]) for _, body in pieces if _tokens(body)]
    assert first_indices == sorted(first_indices), f"pořadí textu musí být zachováno: {first_indices[:10]}"


def test_a_adjacent_chunks_overlap():
    text = _token_text()
    pieces = ai_search.chunks(text)
    bodies = [body for _, body in pieces]

    for index in range(len(bodies) - 1):
        shared = set(_tokens(bodies[index])) & set(_tokens(bodies[index + 1]))
        assert shared, f"chunky {index} a {index + 1} musí mít překryv"

    # the overlap must be bounded by CHUNK_OVERLAP, not an arbitrary duplication
    for index in range(len(bodies) - 1):
        shared_chars = sum(len(token) + 1 for token in
                           set(_tokens(bodies[index])) & set(_tokens(bodies[index + 1])))
        assert shared_chars <= ai_search.CHUNK_OVERLAP + len("slovo00000") + 1, \
            f"překryv chunků {index}/{index + 1} je {shared_chars} znaků, limit je CHUNK_OVERLAP={ai_search.CHUNK_OVERLAP}"


# ---------------------------------------------------------------------------
# B) ordinary document with headings - splitting must be byte-identical to the
#    pre-cap algorithm
# ---------------------------------------------------------------------------
DOCUMENT_WITH_HEADINGS = """SMLOUVA O DÍLO

1. Předmět smlouvy
Zhotovitel se zavazuje provést dílo v rozsahu dle přílohy č. 1.
Objednatel se zavazuje dílo převzít a zaplatit cenu.

2. Cena díla
Cena je stanovena jako pevná a nejvýše přípustná.

ZÁVĚREČNÁ USTANOVENÍ
Tato smlouva nabývá účinnosti dnem podpisu obou stran.
"""


def test_b_document_with_headings_splits_exactly_as_before():
    pieces = ai_search.chunks(DOCUMENT_WITH_HEADINGS)
    assert pieces == _old_chunks(DOCUMENT_WITH_HEADINGS), "dělení dokumentu pod limitem se nesmí změnit"


def test_b_headings_and_blank_lines_remain_the_split_points():
    pieces = ai_search.chunks(DOCUMENT_WITH_HEADINGS)
    headings = [heading for heading, _ in pieces]
    assert "1. Předmět smlouvy" in headings
    assert "2. Cena díla" in headings
    assert "ZÁVĚREČNÁ USTANOVENÍ" in headings
    assert all(len(body) <= ai_search.CHUNK_MAX_SIZE for _, body in pieces)


def test_b_oversized_block_keeps_its_heading_on_every_part():
    """A split must not orphan the section a chunk belongs to."""
    body = " ".join(f"veta{i:04d}" for i in range(1200))  # ~10 800 chars in one paragraph
    pieces = ai_search.chunks(f"3. Technologický postup\n{body}\n")
    assert len(pieces) > 1
    assert {heading for heading, _ in pieces} == {"3. Technologický postup"}


# ---------------------------------------------------------------------------
# C) short document - still exactly one chunk
# ---------------------------------------------------------------------------
def test_c_short_document_stays_one_chunk():
    text = "Předávací protokol byl podepsán dne 3. 2. 2026 na stavbě."
    pieces = ai_search.chunks(text)
    assert len(pieces) == 1
    assert pieces[0][1] == text
    assert pieces == _old_chunks(text)


def test_c_text_exactly_at_the_cap_is_not_split():
    """Boundary: <= CHUNK_MAX_SIZE must be returned untouched, so the cap can
    never alter a document that was already fine."""
    text = "a" * ai_search.CHUNK_MAX_SIZE
    pieces = ai_search.chunks(text)
    assert len(pieces) == 1
    assert pieces[0][1] == text

    over = "a" * (ai_search.CHUNK_MAX_SIZE + 1)
    assert len(ai_search.chunks(over)) == 2, "o jeden znak nad limitem se už dělit musí"


# ---------------------------------------------------------------------------
# D) empty input - no chunks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\r\n\r\n", "\t \n \t"])
def test_d_empty_input_produces_no_chunks(text):
    assert ai_search.chunks(text) == []


# ---------------------------------------------------------------------------
# Degenerate input found in the audited index: a 22.8M-char RTF blob and base64
# attachment payloads contain runs with no space at all. There is no word
# boundary to respect, but leaving such a run whole would put it straight back
# into silent embedding truncation - so it must still be capped.
# ---------------------------------------------------------------------------
def test_run_without_any_space_is_still_capped():
    text = "x" * (ai_search.CHUNK_MAX_SIZE * 3 + 137)
    pieces = ai_search.chunks(text)
    assert all(len(body) <= ai_search.CHUNK_MAX_SIZE for _, body in pieces)
    assert "".join(body for _, body in pieces).count("x") >= len(text), "obsah se nesmí ztratit"


def test_split_oversized_terminates_and_covers_text_for_degenerate_settings():
    """Guards the loop's forward-progress fallback: an overlap as large as the
    window must not livelock or drop content."""
    text = " ".join(f"w{i:04d}" for i in range(400))
    parts = ai_search._split_oversized(text, max_size=100, overlap=100)
    assert parts, "musí vrátit alespoň jednu část"
    assert all(len(part) <= 100 for part in parts)
    joined = " ".join(parts)
    assert all(f"w{i:04d}" in joined for i in range(400)), "žádné slovo se nesmí ztratit"
