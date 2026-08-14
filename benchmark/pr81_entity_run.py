"""PR8.1.1 retrieval-only runner — baseline (flag OFF) vs entity boost (flag ON).

Usage:
  PYTHONPATH=. python -m benchmark.pr81_entity_run --environment production

Writes a JSON report under benchmark/reports/ (or --out). Does not modify
indexes, embeddings, or answer().
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dataset.schema import DATASET_DIR
from . import environment as env_mod

DATASET_PATH = DATASET_DIR / "pr81_entity_retrieval.jsonl"


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


def _blob(row: dict) -> str:
    return _fold(f"{row.get('document') or ''} {row.get('path') or ''}")


def _first_rank(rows: list[dict], needles: list[str]) -> int | None:
    if not needles:
        return None
    folded_needles = [_fold(n) for n in needles if n]
    for i, row in enumerate(rows, 1):
        hay = _blob(row)
        if any(n in hay for n in folded_needles):
            return i
    return None


def _coverage(rows: list[dict], needles: list[str], *, k: int) -> float:
    if not needles:
        return 1.0
    blob = " ".join(_blob(r) for r in rows[:k])
    hits = sum(1 for n in needles if n and _fold(n) in blob)
    return hits / len(needles)


@dataclass
class CaseResult:
    id: str
    type: str
    query: str
    baseline_rank: int | None
    candidate_rank: int | None
    entity_hit_at_5: bool
    multi_entity_coverage_at_10_baseline: float
    multi_entity_coverage_at_10_candidate: float
    false_boost: bool
    baseline_top5: list[str] = field(default_factory=list)
    candidate_top5: list[str] = field(default_factory=list)
    baseline_ms: float = 0.0
    candidate_ms: float = 0.0


def run_case(case: dict, db, lance, emb, *, limit: int = 10) -> CaseResult:
    import ai_search

    query = case["query"]
    expected = list(case.get("expected_name_contains") or [])
    all_needles = list(case.get("expected_all_contains") or expected)
    forbidden = list(case.get("forbidden_name_contains") or [])
    ctype = case.get("type", "positive")

    ai_search.ENTITY_MATCH_BONUS_ENABLED = False
    t0 = time.perf_counter()
    base = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)
    base_ms = (time.perf_counter() - t0) * 1000

    ai_search.ENTITY_MATCH_BONUS_ENABLED = True
    t1 = time.perf_counter()
    cand = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)
    cand_ms = (time.perf_counter() - t1) * 1000

    ai_search.ENTITY_MATCH_BONUS_ENABLED = False

    rank_needles = expected or all_needles
    base_rank = _first_rank(base, rank_needles)
    cand_rank = _first_rank(cand, rank_needles)
    hit5 = cand_rank is not None and cand_rank <= 5
    cov_b = _coverage(base, all_needles, k=10)
    cov_c = _coverage(cand, all_needles, k=10)

    false_boost = False
    if ctype == "negative" and forbidden:
        top_blob = " ".join(_blob(r) for r in cand[:5])
        # False boost: a forbidden entity entered top-5 while baseline top-5
        # did not already contain it (boost introduced the contamination).
        base_blob = " ".join(_blob(r) for r in base[:5])
        for n in forbidden:
            fn = _fold(n)
            if fn in top_blob and fn not in base_blob:
                false_boost = True
                break

    return CaseResult(
        id=case["id"],
        type=ctype,
        query=query,
        baseline_rank=base_rank,
        candidate_rank=cand_rank,
        entity_hit_at_5=hit5 if ctype == "positive" else (cand_rank is not None and cand_rank <= 5),
        multi_entity_coverage_at_10_baseline=cov_b,
        multi_entity_coverage_at_10_candidate=cov_c,
        false_boost=false_boost,
        baseline_top5=[r.get("document") or "" for r in base[:5]],
        candidate_top5=[r.get("document") or "" for r in cand[:5]],
        baseline_ms=base_ms,
        candidate_ms=cand_ms,
    )


def run(environment: str = "production", out: Path | None = None) -> dict:
    import ai_search
    import ai_search_config

    assert ai_search_config.ENTITY_MATCH_BONUS_ENABLED is False
    ai_search.ENTITY_MATCH_BONUS_ENABLED = False

    if environment == "production":
        env = env_mod.production_environment()
    else:
        raise SystemExit("pr81_entity_run currently supports --environment production only")

    cases = _load_cases(DATASET_PATH)
    results = [
        run_case(c, env.db_path, env.lance_dir, env.embeddings) for c in cases
    ]

    positives = [r for r in results if r.type == "positive"]
    negatives = [r for r in results if r.type == "negative"]
    improved = [
        r for r in positives
        if r.candidate_rank is not None and (
            r.baseline_rank is None or r.candidate_rank < r.baseline_rank
            or r.multi_entity_coverage_at_10_candidate > r.multi_entity_coverage_at_10_baseline
        )
    ]
    false_boost_rate = (
        sum(1 for r in negatives if r.false_boost) / len(negatives) if negatives else 0.0
    )
    entity_hit_at_5_rate = (
        sum(1 for r in positives if r.entity_hit_at_5) / len(positives) if positives else 0.0
    )

    report = {
        "dataset": str(DATASET_PATH),
        "environment": environment,
        "n_cases": len(results),
        "entity_hit_at_5_rate": entity_hit_at_5_rate,
        "false_boost_rate": false_boost_rate,
        "improved_positive_ids": [r.id for r in improved],
        "continue_pr812_subject_aliases": (
            false_boost_rate == 0.0 and len(improved) >= 1
        ),
        "cases": [asdict(r) for r in results],
    }

    out_path = out or (Path(__file__).parent / "reports" / f"pr81_entity_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR8.1.1 entity match bonus A/B")
    parser.add_argument("--environment", default="production", choices=["production"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(environment=args.environment, out=args.out)
    print(json.dumps({
        "report_path": report["report_path"],
        "entity_hit_at_5_rate": report["entity_hit_at_5_rate"],
        "false_boost_rate": report["false_boost_rate"],
        "improved_positive_ids": report["improved_positive_ids"],
        "continue_pr812_subject_aliases": report["continue_pr812_subject_aliases"],
    }, ensure_ascii=False, indent=2))
    for case in report["cases"]:
        print(
            f"  {case['id']}: base_rank={case['baseline_rank']} "
            f"cand_rank={case['candidate_rank']} "
            f"cov {case['multi_entity_coverage_at_10_baseline']:.2f}→"
            f"{case['multi_entity_coverage_at_10_candidate']:.2f} "
            f"false_boost={case['false_boost']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
