#!/usr/bin/env python3
"""Experimental PR5 report — does NOT update the retrieval regression baseline.

Usage (from repo root, with production index available):

    PYTHONPATH=. .venv/bin/python benchmark/report_auxiliary_term_coverage.py

Prints per-query: aux activated, added ids, latency delta OFF→ON, HIT/MISS
for a small diagnostic set. Default config flag stays OFF after the run.
"""
from __future__ import annotations

import time
from pathlib import Path

import ai_search
import ai_search_config as cfg
import auxiliary_term_coverage as atc
from ai_search_config import APP_SUPPORT_DIR, QUERY_EXPANSION_MODE

CASES = [
    {
        "id": "atc-crm-weld-01",
        "query": "jaký svár je požadovaný na CRM destičky",
        "gold_doc": "D.1.4.j.1_01_TZ",
        "is_question": True,
    },
    {
        "id": "atc-crm-weld-02",
        "query": "CRM destičky svár",
        "gold_doc": "D.1.4.j.1_01_TZ",
        "is_question": False,
    },
    {
        "id": "atc-neg-deska",
        "query": "jaká deska",
        "gold_doc": None,
        "is_question": True,
    },
    {
        "id": "atc-neg-common",
        "query": "co je ve smlouvě",
        "gold_doc": None,
        "is_question": True,
    },
]


def _doc_hit(rows: list[dict], needle: str | None) -> bool:
    if not needle:
        return False
    n = needle.casefold()
    for r in rows:
        blob = f"{r.get('document', '')} {r.get('path', '')}".casefold()
        if n in blob:
            return True
    return False


def _run(query: str, *, is_question: bool, aux_on: bool, emb, db: Path, lance: Path):
    ai_search.AUXILIARY_TERM_COVERAGE_ENABLED = aux_on
    trace = ai_search.SearchTrace()
    t0 = time.perf_counter()
    rows = ai_search.search(
        query,
        db,
        lance,
        emb,
        limit=300 if is_question else 80,
        is_question=is_question,
        expand_query=QUERY_EXPANSION_MODE,
        trace=trace,
    )
    ms = (time.perf_counter() - t0) * 1000
    aux = (trace.metadata or {}).get("auxiliary_term_coverage")
    pool = [c["chunk_id"] for c in (trace.candidates_before_precision or [])]
    return rows, ms, aux, pool


def main() -> int:
    db = APP_SUPPORT_DIR / "database" / "project.sqlite3"
    lance = APP_SUPPORT_DIR / "lance" / "project"
    if not db.exists():
        print(f"SKIP: production DB missing at {db}")
        return 0

    print("PR5 Auxiliary Term Coverage — experimental report")
    print(f"config default AUXILIARY_TERM_COVERAGE_ENABLED={cfg.AUXILIARY_TERM_COVERAGE_ENABLED}")
    print(f"db={db}")
    emb = ai_search.Embeddings()

    activated = 0
    added_total = 0
    print()
    for case in CASES:
        q = case["query"]
        rows_off, ms_off, aux_off, pool_off = _run(
            q, is_question=case["is_question"], aux_on=False, emb=emb, db=db, lance=lance
        )
        rows_on, ms_on, aux_on, pool_on = _run(
            q, is_question=case["is_question"], aux_on=True, emb=emb, db=db, lance=lance
        )
        # Restore default OFF immediately after each case.
        ai_search.AUXILIARY_TERM_COVERAGE_ENABLED = False

        gold = case["gold_doc"]
        hit_off = _doc_hit(rows_off, gold) if gold else None
        hit_on = _doc_hit(rows_on, gold) if gold else None
        aux_meta = aux_on or {}
        if aux_meta.get("activated"):
            activated += 1
        added_total += int(aux_meta.get("added_count") or 0)

        print(f"=== {case['id']}")
        print(f"  query: {q}")
        print(f"  aux_activated: {aux_meta.get('activated')} reason={aux_meta.get('reason')}")
        print(f"  match: {aux_meta.get('match')}")
        print(f"  added_count: {aux_meta.get('added_count')} matched={aux_meta.get('matched_count')}")
        print(f"  latency_ms OFF={ms_off:.1f} ON={ms_on:.1f} delta={ms_on - ms_off:.1f}")
        if gold:
            print(f"  gold={gold} search_HIT OFF={hit_off} ON={hit_on}")
            gold_pool_off = sum(1 for c in pool_off if "98cca68b" in c)
            gold_pool_on = sum(1 for c in pool_on if "98cca68b" in c)
            print(f"  gold_chunks_in_pre_rerank_pool OFF={gold_pool_off} ON={gold_pool_on}")
        print(f"  aux_off_trace_present: {aux_off is not None}")  # should be False
        print()

    print("SUMMARY")
    print(f"  aux_activated_count: {activated}/{len(CASES)}")
    print(f"  aux_added_ids_total: {added_total}")
    print(f"  flag restored OFF: {ai_search.AUXILIARY_TERM_COVERAGE_ENABLED is False}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
