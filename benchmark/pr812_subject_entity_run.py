"""PR8.1.2 retrieval-only runner — baseline vs subject-alias candidate.

Baseline: both ENTITY_MATCH_BONUS and SUBJECT_ENTITY_ALIAS OFF.
Candidate: SUBJECT_ENTITY_ALIAS ON (explicit entity bonus stays OFF so the
delta isolates subject routing).

Usage:
  PYTHONPATH=. python -m benchmark.pr812_subject_entity_run --environment production
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

DATASET_PATH = DATASET_DIR / "pr812_subject_entity.jsonl"


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


def _forbidden_in_top(rows: list[dict], forbidden: list[str], *, k: int = 5) -> list[str]:
    blob = " ".join(_blob(r) for r in rows[:k])
    return [n for n in forbidden if n and _fold(n) in blob]


@dataclass
class CaseResult:
    id: str
    type: str
    query: str
    baseline_rank: int | None
    candidate_rank: int | None
    entity_rank_delta: int | None
    subject_hit_at_5: bool
    false_subject_boost: bool
    baseline_top5: list[str] = field(default_factory=list)
    candidate_top5: list[str] = field(default_factory=list)
    forbidden_in_candidate_top5: list[str] = field(default_factory=list)


def run_case(case: dict, db, lance, emb, *, limit: int = 10) -> CaseResult:
    import ai_search

    query = case["query"]
    expected = list(case.get("expected_contains") or [])
    forbidden = list(case.get("forbidden_contains") or [])
    ctype = case.get("type", "positive")

    ai_search.ENTITY_MATCH_BONUS_ENABLED = False
    ai_search.SUBJECT_ENTITY_ALIAS_ENABLED = False
    base = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)

    ai_search.ENTITY_MATCH_BONUS_ENABLED = False
    ai_search.SUBJECT_ENTITY_ALIAS_ENABLED = True
    cand = ai_search.search(query, db, lance, emb, limit=limit, is_question="?" in query)

    ai_search.SUBJECT_ENTITY_ALIAS_ENABLED = False

    base_rank = _first_rank(base, expected) if expected else None
    cand_rank = _first_rank(cand, expected) if expected else None
    if base_rank is None and cand_rank is None:
        delta = None
    elif base_rank is None and cand_rank is not None:
        delta = 99  # appeared from outside top-k
    elif cand_rank is None and base_rank is not None:
        delta = -(base_rank)  # lost
    else:
        delta = base_rank - cand_rank  # positive = improved

    hit5 = bool(expected) and cand_rank is not None and cand_rank <= 5
    forbidden_hits = _forbidden_in_top(cand, forbidden, k=5)
    base_forbidden = _forbidden_in_top(base, forbidden, k=5)
    # False subject boost: candidate top-5 newly contains a forbidden supplier.
    false_boost = bool(set(_fold(x) for x in forbidden_hits) - set(_fold(x) for x in base_forbidden))

    return CaseResult(
        id=case["id"],
        type=ctype,
        query=query,
        baseline_rank=base_rank,
        candidate_rank=cand_rank,
        entity_rank_delta=delta,
        subject_hit_at_5=hit5 if ctype == "positive" else False,
        false_subject_boost=false_boost if ctype == "negative" else False,
        baseline_top5=[r.get("document") or "" for r in base[:5]],
        candidate_top5=[r.get("document") or "" for r in cand[:5]],
        forbidden_in_candidate_top5=forbidden_hits,
    )


def run(environment: str = "production", out: Path | None = None) -> dict:
    import ai_search
    import ai_search_config

    assert ai_search_config.SUBJECT_ENTITY_ALIAS_ENABLED is False
    assert ai_search_config.ENTITY_MATCH_BONUS_ENABLED is False
    ai_search.SUBJECT_ENTITY_ALIAS_ENABLED = False
    ai_search.ENTITY_MATCH_BONUS_ENABLED = False

    if environment != "production":
        raise SystemExit("pr812_subject_entity_run supports --environment production only")

    env = env_mod.production_environment()
    cases = _load_cases(DATASET_PATH)
    results = [run_case(c, env.db_path, env.lance_dir, env.embeddings) for c in cases]

    positives = [r for r in results if r.type == "positive"]
    negatives = [r for r in results if r.type == "negative"]
    subject_hit_at_5_rate = (
        sum(1 for r in positives if r.subject_hit_at_5) / len(positives) if positives else 0.0
    )
    false_subject_boost_rate = (
        sum(1 for r in negatives if r.false_subject_boost) / len(negatives) if negatives else 0.0
    )
    go = subject_hit_at_5_rate >= 0.9 and false_subject_boost_rate == 0.0

    report = {
        "dataset": str(DATASET_PATH),
        "environment": environment,
        "n_cases": len(results),
        "subject_hit_at_5_rate": subject_hit_at_5_rate,
        "false_subject_boost_rate": false_subject_boost_rate,
        "go_nogo": "GO" if go else "NO-GO",
        "continue_pr82_revision_ranking": go,
        "cases": [asdict(r) for r in results],
    }
    out_path = out or (Path(__file__).parent / "reports" / f"pr812_subject_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR8.1.2 subject entity routing A/B")
    parser.add_argument("--environment", default="production", choices=["production"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(environment=args.environment, out=args.out)
    print(json.dumps({
        "report_path": report["report_path"],
        "subject_hit_at_5_rate": report["subject_hit_at_5_rate"],
        "false_subject_boost_rate": report["false_subject_boost_rate"],
        "go_nogo": report["go_nogo"],
        "continue_pr82_revision_ranking": report["continue_pr82_revision_ranking"],
    }, ensure_ascii=False, indent=2))
    for case in report["cases"]:
        print(
            f"  {case['id']}: base={case['baseline_rank']} cand={case['candidate_rank']} "
            f"delta={case['entity_rank_delta']} hit5={case['subject_hit_at_5']} "
            f"false_boost={case['false_subject_boost']}"
        )
    return 0 if report["go_nogo"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
