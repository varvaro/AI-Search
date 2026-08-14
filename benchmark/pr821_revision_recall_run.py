"""PR8.2.1 retrieval-only runner — baseline vs revision-recall candidate pool.

Baseline: REVISION_RECALL_ENABLED OFF (ranking + entity flags OFF).
Candidate: REVISION_RECALL_ENABLED ON (REVISION_RANKING_ENABLED still OFF).

Measures pool membership via SearchTrace.candidates_before_precision, not top-k rank.

Usage:
  PYTHONPATH=. python -m benchmark.pr821_revision_recall_run --environment production
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dataset.schema import DATASET_DIR
from . import environment as env_mod

DATASET_PATH = DATASET_DIR / "pr821_revision_recall.jsonl"


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(c)
    )


def _load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def _pool_docs(db_path: Path, chunk_ids: list[str]) -> list[tuple[str, str]]:
    if not chunk_ids:
        return []
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        out = []
        for cid in chunk_ids:
            row = con.execute(
                "SELECT d.name, d.relative_path FROM chunks c "
                "JOIN documents d ON d.id=c.document_id WHERE c.id=?",
                (cid,),
            ).fetchone()
            if row:
                out.append((row[0] or "", row[1] or ""))
        return out
    finally:
        con.close()


def _pool_blob(docs: list[tuple[str, str]]) -> str:
    return _fold(" ".join(f"{n} {p}" for n, p in docs))


def _expected_in_pool(docs: list[tuple[str, str]], needles: list[str]) -> bool:
    if not needles:
        return True
    blob = _pool_blob(docs)
    # All needles must match (AND) — allows requiring HMG + akt date together.
    return all(_fold(n) in blob for n in needles if n)


def _forbidden_hits(docs: list[tuple[str, str]], forbidden: list[str]) -> list[str]:
    blob = _pool_blob(docs)
    return [n for n in forbidden if n and _fold(n) in blob]


@dataclass
class CaseResult:
    id: str
    type: str
    query: str
    baseline_pool_size: int
    candidate_pool_size: int
    expected_hit: bool
    pool_identical: bool
    old_auto_appended: bool
    baseline_added_count: int
    candidate_added_count: int
    matched_names_sample: list[str] = field(default_factory=list)
    forbidden_in_candidate_pool: list[str] = field(default_factory=list)


def run_case(case: dict, db, lance, emb, *, limit: int = 10) -> CaseResult:
    import ai_search

    query = case["query"]
    expected = list(case.get("expected_in_pool") or [])
    forbidden = list(case.get("forbidden_in_pool_auto") or [])
    ctype = case.get("type", "positive_pool")

    ai_search.ENTITY_MATCH_BONUS_ENABLED = False
    ai_search.SUBJECT_ENTITY_ALIAS_ENABLED = False
    ai_search.REVISION_RANKING_ENABLED = False
    ai_search.REVISION_RECALL_ENABLED = False
    t0 = ai_search.SearchTrace()
    ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query, trace=t0)
    base_ids = [c["chunk_id"] for c in (t0.candidates_before_precision or [])]
    base_docs = _pool_docs(db, base_ids)
    base_meta = (t0.metadata or {}).get("revision_recall") or {}

    ai_search.REVISION_RECALL_ENABLED = True
    t1 = ai_search.SearchTrace()
    ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query, trace=t1)
    ai_search.REVISION_RECALL_ENABLED = False
    cand_ids = [c["chunk_id"] for c in (t1.candidates_before_precision or [])]
    cand_docs = _pool_docs(db, cand_ids)
    cand_meta = (t1.metadata or {}).get("revision_recall") or {}

    # Append-only: baseline ids must be a prefix of candidate ids.
    pool_identical = base_ids == cand_ids
    prefix_ok = cand_ids[: len(base_ids)] == base_ids
    expected_hit = _expected_in_pool(cand_docs, expected) if expected else True
    forb = _forbidden_hits(cand_docs, forbidden)
    # Only count OLD that were newly appended (not already in baseline pool).
    base_blob = _pool_blob(base_docs)
    newly = [
        (n, p)
        for n, p in cand_docs
        if _fold(f"{n} {p}") not in base_blob
        and ("/old/" in _fold(f"{n} {p}") or _fold(p).startswith("old/"))
    ]
    old_auto = bool(newly) if ctype == "negative_old" else False
    if ctype == "negative_old" and forb:
        # Prefer precise: forbidden substrings only among *appended* docs.
        appended = cand_docs[len(base_docs) :] if prefix_ok else cand_docs
        forb = _forbidden_hits(appended, forbidden)
        old_auto = bool(forb)

    names = list(cand_meta.get("matched_document_names") or [])[:8]
    return CaseResult(
        id=case["id"],
        type=ctype,
        query=query,
        baseline_pool_size=len(base_ids),
        candidate_pool_size=len(cand_ids),
        expected_hit=expected_hit if ctype == "positive_pool" else True,
        pool_identical=pool_identical if ctype.startswith("control") else prefix_ok,
        old_auto_appended=old_auto,
        baseline_added_count=int(base_meta.get("added_count") or 0),
        candidate_added_count=int(cand_meta.get("added_count") or 0),
        matched_names_sample=names,
        forbidden_in_candidate_pool=forb,
    )


def run(environment: str = "production", out: Path | None = None) -> dict:
    import ai_search
    import ai_search_config

    assert ai_search_config.REVISION_RECALL_ENABLED is False
    ai_search.REVISION_RECALL_ENABLED = False
    ai_search.REVISION_RANKING_ENABLED = False

    if environment != "production":
        raise SystemExit("pr821_revision_recall_run supports --environment production only")

    env = env_mod.production_environment()
    cases = _load_cases(DATASET_PATH)
    results = [run_case(c, env.db_path, env.lance_dir, env.embeddings) for c in cases]

    positives = [r for r in results if r.type == "positive_pool"]
    controls = [r for r in results if r.type.startswith("control")]
    neg_old = [r for r in results if r.type == "negative_old"]

    pool_hit_rate = (
        sum(1 for r in positives if r.expected_hit) / len(positives) if positives else 0.0
    )
    controls_identical = all(r.pool_identical for r in controls)
    # Flag OFF never writes revision_recall metadata → baseline_added_count always 0.
    flag_off_clean = all(r.baseline_added_count == 0 for r in results)
    old_auto_rate = (
        sum(1 for r in neg_old if r.old_auto_appended) / max(1, len(neg_old))
    )
    append_only_ok = all(
        r.candidate_pool_size >= r.baseline_pool_size for r in results
    ) and all(
        r.pool_identical or r.type == "positive_pool" or r.type == "negative_old"
        for r in results
    )
    # For positives/negatives, require prefix preservation (stored in pool_identical for those).
    prefix_ok = all(r.pool_identical for r in results if r.type in ("positive_pool", "negative_old"))

    go = (
        pool_hit_rate >= 0.66
        and controls_identical
        and flag_off_clean
        and old_auto_rate == 0.0
        and append_only_ok
        and prefix_ok
    )

    report = {
        "dataset": str(DATASET_PATH),
        "environment": environment,
        "n_cases": len(results),
        "pool_hit_rate": pool_hit_rate,
        "controls_pool_identical": controls_identical,
        "flag_off_clean": flag_off_clean,
        "old_auto_append_rate": old_auto_rate,
        "append_only_prefix_ok": prefix_ok,
        "go_nogo": "GO" if go else "NO-GO",
        "cases": [asdict(r) for r in results],
    }
    out_path = out or (
        Path(__file__).parent / "reports" / f"pr821_revision_recall_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR8.2.1 revision recall A/B")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(environment=args.environment, out=args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))
    print("go_nogo=", report["go_nogo"])
    return 0 if report["go_nogo"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
