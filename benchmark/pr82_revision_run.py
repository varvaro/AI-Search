"""PR8.2 retrieval-only runner — baseline vs revision-ranking candidate.

Baseline: REVISION_RANKING_ENABLED OFF (and entity flags OFF).
Candidate: REVISION_RANKING_ENABLED ON.

Usage:
  PYTHONPATH=. python -m benchmark.pr82_revision_run --environment production
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

DATASET_PATH = DATASET_DIR / "pr82_revision_ranking.jsonl"


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
    folded = [_fold(n) for n in needles if n]
    for i, row in enumerate(rows, 1):
        hay = _blob(row)
        if any(n in hay for n in folded):
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
    wrong_current_revision: bool
    order_identical: bool
    baseline_top5: list[str] = field(default_factory=list)
    candidate_top5: list[str] = field(default_factory=list)
    forbidden_in_candidate_top5: list[str] = field(default_factory=list)


def run_case(case: dict, db, lance, emb, *, limit: int = 10) -> CaseResult:
    import ai_search

    query = case["query"]
    expected = list(case.get("expected_contains") or [])
    forbidden = list(case.get("forbidden_top5") or [])
    ctype = case.get("type", "positive")

    ai_search.ENTITY_MATCH_BONUS_ENABLED = False
    ai_search.SUBJECT_ENTITY_ALIAS_ENABLED = False
    ai_search.REVISION_RANKING_ENABLED = False
    base = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)

    ai_search.REVISION_RANKING_ENABLED = True
    cand = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)
    ai_search.REVISION_RANKING_ENABLED = False

    base_rank = _first_rank(base, expected) if expected else None
    cand_rank = _first_rank(cand, expected) if expected else None
    hit5 = bool(expected) and cand_rank is not None and cand_rank <= 5
    forbidden_hits = _hits_forbidden(cand, forbidden, k=5)
    # old_leak: OLD/old path segment in top-5 under revision-intent cases.
    top_blob = " ".join(_blob(r) for r in cand[:5])
    old_leak = (
        ctype in ("positive", "negative_old")
        and ("/old/" in top_blob or top_blob.startswith("old/"))
    )
    wrong_current = bool(expected) and (
        cand_rank is None or (base_rank is not None and cand_rank > base_rank and cand_rank > 5)
    )
    order_identical = [r.get("document") for r in base] == [r.get("document") for r in cand]

    return CaseResult(
        id=case["id"],
        type=ctype,
        query=query,
        baseline_rank=base_rank,
        candidate_rank=cand_rank,
        revision_hit_at_5=hit5 if ctype == "positive" else False,
        old_leak=old_leak if ctype in ("positive", "negative_old") else False,
        wrong_current_revision=wrong_current if ctype == "positive" else False,
        order_identical=order_identical,
        baseline_top5=[r.get("document") or "" for r in base[:5]],
        candidate_top5=[r.get("document") or "" for r in cand[:5]],
        forbidden_in_candidate_top5=forbidden_hits,
    )


def run(environment: str = "production", out: Path | None = None) -> dict:
    import ai_search
    import ai_search_config

    assert ai_search_config.REVISION_RANKING_ENABLED is False
    ai_search.REVISION_RANKING_ENABLED = False

    if environment != "production":
        raise SystemExit("pr82_revision_run supports --environment production only")

    env = env_mod.production_environment()
    cases = _load_cases(DATASET_PATH)
    results = [run_case(c, env.db_path, env.lance_dir, env.embeddings) for c in cases]

    positives = [r for r in results if r.type == "positive"]
    controls = [r for r in results if r.type == "control_no_intent"]
    neg_old = [r for r in results if r.type == "negative_old"]

    revision_hit_at_5_rate = (
        sum(1 for r in positives if r.revision_hit_at_5) / len(positives) if positives else 0.0
    )
    old_leak_rate = (
        sum(1 for r in positives + neg_old if r.old_leak) / max(1, len(positives) + len(neg_old))
    )
    wrong_current_revision_rate = (
        sum(1 for r in positives if r.wrong_current_revision) / len(positives) if positives else 0.0
    )
    controls_identical = all(r.order_identical for r in controls)

    go = (
        revision_hit_at_5_rate >= 0.66
        and old_leak_rate == 0.0
        and wrong_current_revision_rate == 0.0
        and controls_identical
    )

    report = {
        "dataset": str(DATASET_PATH),
        "environment": environment,
        "n_cases": len(results),
        "revision_hit_at_5_rate": revision_hit_at_5_rate,
        "old_leak_rate": old_leak_rate,
        "wrong_current_revision_rate": wrong_current_revision_rate,
        "controls_order_identical": controls_identical,
        "go_nogo": "GO" if go else "NO-GO",
        "cases": [asdict(r) for r in results],
    }
    out_path = out or (Path(__file__).parent / "reports" / f"pr82_revision_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR8.2 revision ranking A/B")
    parser.add_argument("--environment", default="production", choices=["production"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(environment=args.environment, out=args.out)
    print(json.dumps({
        "report_path": report["report_path"],
        "revision_hit_at_5_rate": report["revision_hit_at_5_rate"],
        "old_leak_rate": report["old_leak_rate"],
        "wrong_current_revision_rate": report["wrong_current_revision_rate"],
        "controls_order_identical": report["controls_order_identical"],
        "go_nogo": report["go_nogo"],
    }, ensure_ascii=False, indent=2))
    for case in report["cases"]:
        print(
            f"  {case['id']}: base={case['baseline_rank']} cand={case['candidate_rank']} "
            f"hit5={case['revision_hit_at_5']} old_leak={case['old_leak']} "
            f"identical={case['order_identical']}"
        )
    return 0 if report["go_nogo"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
