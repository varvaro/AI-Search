"""PR8.2.2 end-to-end A/B — revision recall + ranking together.

Baseline: REVISION_RECALL_ENABLED OFF, REVISION_RANKING_ENABLED OFF
          (entity flags OFF).
Candidate: both revision flags ON.

Usage:
  PYTHONPATH=. python -m benchmark.pr822_revision_e2e_run --environment production
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

DATASET_PATH = DATASET_DIR / "pr822_revision_ranking.jsonl"


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
    """Return 1-based rank of first row matching ALL needles (AND)."""
    if not needles:
        return None
    folded = [_fold(n) for n in needles if n]
    if not folded:
        return None
    for i, row in enumerate(rows, 1):
        hay = _blob(row)
        if all(n in hay for n in folded):
            return i
    return None


def _hits_forbidden(rows: list[dict], forbidden: list[str], *, k: int = 5) -> list[str]:
    blob = " ".join(_blob(r) for r in rows[:k])
    return [n for n in forbidden if n and _fold(n) in blob]


@dataclass
class CaseResult:
    id: str
    type: str
    query: str
    baseline_rank: int | None
    candidate_rank: int | None
    revision_hit_at_5: bool
    old_leak: bool
    order_identical: bool
    rank_improved: bool
    baseline_top5: list[str] = field(default_factory=list)
    candidate_top5: list[str] = field(default_factory=list)
    forbidden_in_candidate_top5: list[str] = field(default_factory=list)
    candidate_revision_scores_top5: list[float] = field(default_factory=list)
    evidence: str = ""


def run_case(case: dict, db, lance, emb, *, limit: int = 10) -> CaseResult:
    import ai_search

    query = case["query"]
    expected = list(case.get("expected_contains") or [])
    forbidden = list(case.get("forbidden_top5") or [])
    ctype = case.get("type", "positive")

    ai_search.ENTITY_MATCH_BONUS_ENABLED = False
    ai_search.SUBJECT_ENTITY_ALIAS_ENABLED = False
    ai_search.REVISION_RECALL_ENABLED = False
    ai_search.REVISION_RANKING_ENABLED = False
    base = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)

    ai_search.REVISION_RECALL_ENABLED = True
    ai_search.REVISION_RANKING_ENABLED = True
    cand = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)
    ai_search.REVISION_RECALL_ENABLED = False
    ai_search.REVISION_RANKING_ENABLED = False

    base_rank = _first_rank(base, expected) if expected else None
    cand_rank = _first_rank(cand, expected) if expected else None
    hit5 = bool(expected) and cand_rank is not None and cand_rank <= 5
    forbidden_hits = _hits_forbidden(cand, forbidden, k=5)
    top_blob = " ".join(_blob(r) for r in cand[:5])
    old_leak = (
        ctype == "positive"
        and ("/old/" in top_blob or top_blob.startswith("old/"))
    )
    order_identical = [r.get("document") for r in base] == [r.get("document") for r in cand]
    rank_improved = (
        ctype == "positive"
        and cand_rank is not None
        and (base_rank is None or cand_rank < base_rank)
    )

    rev_scores = [
        float((r.get("match") or {}).get("revision_score") or 0.0) for r in cand[:5]
    ]

    if ctype == "positive":
        evidence = (
            f"expected={expected!r} base_rank={base_rank} cand_rank={cand_rank} "
            f"hit@5={hit5} old_leak={old_leak} "
            f"cand_top1={cand[0].get('document') if cand else None!r}"
        )
    elif ctype.startswith("control"):
        evidence = (
            f"order_identical={order_identical} "
            f"base_top1={base[0].get('document') if base else None!r} "
            f"cand_top1={cand[0].get('document') if cand else None!r}"
        )
    else:
        evidence = ""

    return CaseResult(
        id=case["id"],
        type=ctype,
        query=query,
        baseline_rank=base_rank,
        candidate_rank=cand_rank,
        revision_hit_at_5=hit5 if ctype == "positive" else False,
        old_leak=old_leak,
        order_identical=order_identical,
        rank_improved=rank_improved,
        baseline_top5=[r.get("document") or "" for r in base[:5]],
        candidate_top5=[r.get("document") or "" for r in cand[:5]],
        forbidden_in_candidate_top5=forbidden_hits,
        candidate_revision_scores_top5=rev_scores,
        evidence=evidence,
    )


def run(environment: str = "production", out: Path | None = None) -> dict:
    import ai_search
    import ai_search_config

    assert ai_search_config.REVISION_RECALL_ENABLED is False
    assert ai_search_config.REVISION_RANKING_ENABLED is False
    ai_search.REVISION_RECALL_ENABLED = False
    ai_search.REVISION_RANKING_ENABLED = False

    if environment != "production":
        raise SystemExit("pr822_revision_e2e_run supports --environment production only")

    env = env_mod.production_environment()
    cases = _load_cases(DATASET_PATH)
    results = [run_case(c, env.db_path, env.lance_dir, env.embeddings) for c in cases]

    positives = [r for r in results if r.type == "positive"]
    controls = [r for r in results if r.type.startswith("control")]

    revision_hit_at_5_rate = (
        sum(1 for r in positives if r.revision_hit_at_5) / len(positives) if positives else 0.0
    )
    old_leak_rate = (
        sum(1 for r in positives if r.old_leak) / len(positives) if positives else 0.0
    )
    ordinary_regression = not all(r.order_identical for r in controls)
    controls_identical = all(r.order_identical for r in controls)

    go = (
        revision_hit_at_5_rate >= 0.66
        and old_leak_rate == 0.0
        and controls_identical
        and not ordinary_regression
    )

    report = {
        "dataset": str(DATASET_PATH),
        "environment": environment,
        "flags_candidate": {
            "REVISION_RECALL_ENABLED": True,
            "REVISION_RANKING_ENABLED": True,
        },
        "n_cases": len(results),
        "revision_hit_at_5_rate": revision_hit_at_5_rate,
        "old_leak_rate": old_leak_rate,
        "ordinary_regression": ordinary_regression,
        "controls_order_identical": controls_identical,
        "go_nogo": "GO" if go else "NO-GO",
        "cases": [asdict(r) for r in results],
    }
    out_path = out or (
        Path(__file__).parent / "reports" / f"pr822_revision_e2e_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR8.2.2 revision ranking E2E A/B")
    parser.add_argument("--environment", default="production", choices=["production"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(environment=args.environment, out=args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))
    for case in report["cases"]:
        print(
            f"  {case['id']}: base={case['baseline_rank']} cand={case['candidate_rank']} "
            f"hit5={case['revision_hit_at_5']} old_leak={case['old_leak']} "
            f"identical={case['order_identical']}"
        )
        print(f"    evidence: {case['evidence']}")
    return 0 if report["go_nogo"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
