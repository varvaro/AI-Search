# Retrieval pipeline benchmark

Measures the AI Search retrieval pipeline's quality (recall, MRR, nDCG,
duplication, latency) end-to-end and per pipeline stage, so any future
change to retrieval can be checked objectively against a "before" run instead
of relying on manually re-running one or two example queries.

This package never modifies retrieval behaviour. It only *calls* the existing
public functions in `ai_search.py` / `ui_services.py` (`search`,
`classify_query`, `deduplicate_by_content`, `diversify_results`,
`search_all`, `answer`) and reads their output, including phase-level
BM25/vector/RRF/rerank candidates - `ai_search.search()` accepts an optional
`trace=ai_search.SearchTrace()` argument that records a snapshot of every
phase from inside its one real code path, with the return value staying
byte-for-byte identical either way. See `benchmark/__init__.py`/
`pipeline_trace.py` for details, and `consistency_check.py` for how a future
change to `search()` that forgets to update its own trace-writing lines is
still caught.

## Quick start

```bash
# Fast, offline, deterministic - runs against a small synthetic corpus
# (benchmark/fixtures.py), not the real Box index. This is what CI should run.
.venv/bin/python -m benchmark run

# Same, but against the REAL production index (needs the app to have been
# indexed at least once; loads the real BAAI/bge-m3 model - slower).
.venv/bin/python -m benchmark run --environment production

# Also render Markdown/HTML/CSV reports next to the run artifact.
.venv/bin/python -m benchmark run --environment production --report

# Compare two runs (exit code 1 if any case regressed).
.venv/bin/python -m benchmark compare benchmark/runs/OLD.json benchmark/runs/NEW.json

# The 20-query retrieval regression suite (see "Retrieval regression suite"
# below). Runs against the production index and compares to a checked-in
# baseline in one step; exit code 1 on a blocking regression.
.venv/bin/python -m benchmark.run_retrieval_regression

# A/B the Query Understanding layer (query_expansion.py). `--expand-query`
# alone means both injection branches; the two can also be measured separately
# because they behave very differently (see "Query expansion A/B" below).
.venv/bin/python -m benchmark run --environment production --output benchmark/runs/ab_baseline.json
.venv/bin/python -m benchmark run --environment production --expand-query --output benchmark/runs/ab_expand.json
.venv/bin/python -m benchmark run --environment production --expand-query fts     --output benchmark/runs/ab_fts.json
.venv/bin/python -m benchmark run --environment production --expand-query vector  --output benchmark/runs/ab_vector.json
.venv/bin/python -m benchmark compare benchmark/runs/ab_baseline.json benchmark/runs/ab_expand.json
```

## Query expansion A/B

`--expand-query [both|fts|vector]` toggles `ai_search.search(expand_query=...)`.
Every run artifact records `expand_query` at the top level and, per case,
`expansion_activated` / `expansion_terms` / `expansion_matched_rules` /
`expansion_term_count`, so a comparison can always tell whether a case's change
came from the expansion firing at all. Baseline runs carry the same keys with
empty/False values, so both sides of an A/B have an identical metric schema.

The branch split matters: the `fts` branch only ADDS OR-terms to the candidate
set, while the `vector` branch REPLACES the embedded query text and therefore
changes every cosine score in the rerank - including for candidates the
expansion had nothing to do with. Measure them apart before drawing conclusions.

Every run writes a JSON artifact to `benchmark/runs/<timestamp>_<gitsha>_<environment>.json`
(gitignored - see `benchmark/runs/README.md`). Keep one aside before a risky
change (`cp benchmark/runs/latest.json benchmark/runs/baseline.json`) and diff
against it afterwards.

## Architecture

```
Question
  -> intent detection      (ui_services.classify_query - real call)
  -> query parsing         }
  -> FTS retrieval (BM25)  }
  -> vector retrieval      }  ai_search.search(..., trace=SearchTrace()) - one real call;
  -> fusion (RRF)          }  SearchTrace records each phase's candidates/timings as
  -> candidate pool        }  search() computes them, without changing what it returns
  -> reranker               }
  -> diversification        (ui_services.deduplicate_by_content/diversify_results - real calls)
  -> prompt builder          (mirrors the context one-liner in ai_search.answer())
  -> final answer (optional) (ai_search.answer() - real call, needs live Ollama)
```

`benchmark/pipeline_trace.py` produces one `PipelineTrace` per query with the
candidate list *at every stage*, not just the final output, by calling
`ai_search.search()` once with a `SearchTrace` attached and re-shaping its
chunk_id-keyed snapshots into `PipelineTrace`'s document/path-keyed ones
(one small batched DB lookup resolves chunk_id -> document/path for that
reshaping - metadata enrichment, not retrieval). See `ai_search.SearchTrace`'s
docstring for the instrumentation contract, and `consistency_check.py` for
how a future change to `search()`'s scoring/truncation that isn't matched by
an update to its own nearby trace-writing lines is still caught.

### Why the environments (fixture vs. production)

* **fixture**: `benchmark/fixtures.py`'s 21-document synthetic corpus +
  `FakeCategoryEmbeddings` (the exact same trade-off already used in
  `tests/test_search_relevance.py`, duplicated here so `benchmark/` has no
  dependency on `tests/`). Fast, deterministic, no model download - the
  default, and what CI should run on every PR.
* **production**: the real `~/Library/Application Support/AI Search` index +
  the real `BAAI/bge-m3` embeddings. Read-only - never calls `sync()`. This is
  what actually answers "did retrieval get better or worse on the real
  index", and is the only environment that can validate the real embedding
  model's semantic quality (the fixture's fake embeddings only validate
  pipeline *mechanics* - see `fixtures.py`'s docstring).

## Metrics (why each one is here)

See `benchmark/metrics.py` - every function's docstring explains what failure
mode it exists to catch. Summary:

| Metric | Stage | Why |
|---|---|---|
| `best_rank` / `recall_at_k` per stage | BM25, vector, fusion, pool | Pinpoints exactly *which* stage drops a relevant document - the 2026-08-06 FERI diagnostic only found the root cause (rerank_k truncation) by comparing recall stage-by-stage, not from the final result alone. |
| `pool_survival_rate` | candidate pool | `recall(after truncation) / recall(before truncation)`. This single number would have flagged the FERI bug directly. |
| `channel_agreement` | fusion | both/bm25-only/vector-only/neither - detects RRF's "single-channel penalty" bias. |
| `recall_at_k`, `hit_rate`, `MRR`, `nDCG@k` | final results | Standard IR metrics; nDCG additionally rewards *graded*, *ranked* relevance, MRR rewards a good #1 result, recall/hit-rate are the simplest top-line numbers. There is deliberately **no precision metric**: real precision needs a relevance judgment for every retrieved row, which this dataset does not have. |
| `forbidden_free_rate` | final results | 1.0 = no explicitly-forbidden near-duplicate pattern in the top-k, degrading toward 0.0 as more of the top-k is forbidden content. Renamed twice on 2026-08-07 with no change to the formula: `precision_at_k` (never measured precision) -> `forbidden_rate` (inverted name) -> `forbidden_free_rate`. |
| `duplicate_count`, `distinct_ratio` | final results | Tracks the "no 10 copies of the same fact" dedup/diversify goal over time, independent of relevance. |
| `keyword_coverage` | final results / answer | Did the *content* a correct answer needs make it through, independent of whether the LLM used it well. |
| `stage_latency_ms`, `stage_pool_size` | every stage | Regression guard for speed (explicit project priority: "rychlost > přesnost") and sanity check (an empty pool at any stage is an infra bug, not a relevance bug). |
| `score_histogram` | reranker | Diagnostic aid: a collapsed histogram (all scores clustered) means the ranking signal is close to noise even if `recall@k` still looks fine. |

## Dataset

One JSON object per line in `benchmark/dataset/*.jsonl` - see
`benchmark/dataset/schema.py`'s docstring for the full field reference.
Minimal case:

```json
{"id": "pentaflex-01", "question": "Pentaflex"}
```

Add a case by appending a line - no code changes needed. `fixture_queries.jsonl`
targets the synthetic corpus; `production_queries.jsonl` targets the real
index and must only reference paths/keywords you've actually verified exist
in it; `retrieval_regression.jsonl` is the 20-query regression suite described
below.

## Retrieval regression suite

```bash
.venv/bin/python -m benchmark.run_retrieval_regression                # run + compare to baseline
.venv/bin/python -m benchmark.run_retrieval_regression --no-baseline  # run only
.venv/bin/python -m benchmark.run_retrieval_regression --update-baseline
```

20 production lookups (`benchmark/dataset/retrieval_regression.jsonl`), one per
real failure mode found during the 2026-08-06/07 diagnostics, measured against
a checked-in baseline (`benchmark/baselines/retrieval_regression.json`). The
report names every case that moved, in either direction, and the command exits
non-zero on a blocking regression.

**Why it is separate from `python -m benchmark run`.** That command answers
"how good is the pipeline, phase by phase" and drives `pipeline_trace.py`
(instrumented retrieval, ~15 aggregate metrics, 5 answer-oriented cases). This
one answers "did any of the 20 known queries move", and calls exactly what the
user's browser calls - `ui_services.search_all()`, once per query. Keeping the
entry point identical to the diagnostics that produced the ground truth is what
makes the baseline numbers comparable with them. Metric math and environment
wiring are not duplicated: both come from `metrics.py` / `environment.py`.

The split earns its keep empirically: across the 2026-08-08 reindex the 5-case
production dataset did not move at all (2 passed / 2 failed, every mean
identical), while this suite recorded four rank changes including two
MISS -> HIT.

### Known-limitation flags

Two dataset fields (see `schema.py`) keep permanent, diagnosed problems visible
without making the run permanently red - a report that is always failing is a
report nobody reads:

* `expected_content_missing` - the ground truth is not in the index at all
  (`rr-haus365-kladecsky-plan-01`: no file or chunk matches "kladeč"). MISS is
  the *correct* result, so the case is excluded from the headline means and
  from the miss count. It stays in the suite as a tripwire: the day the file is
  filed, the case flips to HIT and the report says so.
* `expected_retrieval_issue` - the content is indexed but a diagnosed, still
  -open weakness keeps it out of the top-k (`rr-kzp-monolit-feri-01`: the query
  spells out "kontrolní a zkušební plán" while the documents only use the
  acronym "KZP"). Its bad score *does* count toward the means - otherwise
  fixing the weakness would show no improvement - but it never fails the run
  on its own.

A case cannot set both: "content absent" and "content present but unreachable"
are competing diagnoses, and the schema rejects a case asserting both.

### What the suite measures

`evaluate_case()` resolves both `is_question` (via production's `classify_query`)
and `expand_query` (via `ai_search_config.QUERY_EXPANSION_MODE`, the same
constant `app.py` passes) at run time rather than hardcoding them, so the suite
cannot drift into measuring a pipeline the UI never runs. The mode is recorded
in every artifact and the runner warns when it differs from the baseline's -
a mode change moves ranks on its own and is invisible in the per-case numbers.

That constant is itself a result of this suite. Measured 2026-08-08:

| expansion | recall@10 | MRR | nDCG@10 | improved | regressed |
|---|---|---|---|---|---|
| off | 0.947 | 0.772 | 0.817 | - | - |
| `"fts"` | 0.974 | 0.778 | 0.827 | 1 | 0 |
| `True` (both branches) | 0.921 | 0.697 | 0.751 | 2 | **4** |

Enabling both branches looks obviously right - it is the flag the query
expansion layer was built for - and is a net loss, because the vector branch
REPLACES the embedded query text rather than adding to it, so every cosine
score in the rerank moves. It took `rr-kzp-seznam-tp-01` from rank 1 to a miss.
The 5-case `production_queries.jsonl` dataset registered none of this.

### Ground truth

Every needle was resolved by SQL against the real index before being written,
and the resulting document count is recorded in the case's `notes`. Prefer a
needle matching one document, or one tight family of revisions - a
folder-wide needle makes the case pass trivially and blind. Two cases were
repaired on 2026-08-08 after the post-reindex audit and are pinned by
`tests/test_retrieval_regression_suite.py`:

* `rr-zl-prehled-gd-01` used `Zmenovy_list` (individual change lists) instead
  of `přehled ZL GD` (the overview the query asks for) and reported a false
  regression (#7 -> #9) at the exact moment the real overview spreadsheets
  entered the top 10 for the first time. With the correct needle the same two
  runs read MISS -> #3.
* `rr-haus365-tp-monolit-01` used `VT 11_HAUS365`, a quantity spreadsheet, not
  the technological procedure `TP Smíchov monolit.doc`.

## Automatic comparison & regression policy

`benchmark/compare.py` diffs two run artifacts **per case**, not only on the
aggregate mean - a case that flips from passing to failing is always reported
as a regression even if the aggregate mean improves (this is what prevents
"fixing query A silently breaks 50 others" from hiding inside a rosy average).
`python -m benchmark compare` exits non-zero if any regression is detected,
so it can gate a CI step or a pre-push hook.

## Reports

`python -m benchmark run --report` or `python -m benchmark report RUN.json`
writes `run.md` (PR-description-friendly, git-diffable), `run.html`
(browsable, includes a pipeline-funnel bar chart of mean pool size per stage)
and `run.csv` (per-case metrics, for pivoting elsewhere) to
`benchmark/reports/latest/` (gitignored, see `.gitignore`).

## CI (proposal - not implemented, this is a local Mac app with no hosted CI)

* On every PR touching `ai_search.py` / `ui_services.py`: run
  `python -m benchmark run --environment fixture` (fast, no external
  dependencies) and fail the check if `aggregate.failed > 0` or
  `aggregate.errored > 0`. This is a pure regression net (mechanics only,
  fake embeddings) so it's safe to run in a hosted/ephemeral runner.
* On demand (or nightly, on the machine that has the real index + Ollama):
  `python -m benchmark run --environment production --report`, then
  `python -m benchmark compare <last-known-good>.json <new>.json` and post the
  Markdown comparison as a PR/commit comment. Requires a self-hosted runner
  with access to the real Box index/BAAI-m3/Ollama - not something a hosted
  GitHub Actions runner can do out of the box, which is why this stays a
  manual/self-hosted step rather than a hard PR gate.
* `--include-answer` (real Ollama call per case) should stay opt-in/manual
  only - it's slow (~5-30s/query) and non-deterministic, unsuitable for a
  required check.

## Regression testing vs. tests/

`tests/test_search_relevance.py` and friends pin an *exact* expected rank for
a small, hand-picked set of queries and fail hard on any deviation - tight,
fast, deterministic, meant to run on every commit. This benchmark instead
tracks *aggregate* quality metrics (recall/MRR/nDCG, plus
regression-per-case detection) over time and across a larger, more loosely
specified dataset (substring/keyword matching instead of exact ranks) -
complementary, not a replacement: `tests/` says "did we break this specific
known-good case", `benchmark/` says "is retrieval quality, in aggregate,
trending up or down".

## Interpreting a run

* `aggregate.failed`/`errored` == 0 and no `drift_detected` warnings: healthy.
* A case in `benchmark/dataset/production_queries.jsonl` tagged
  `production-regression` that's failing on purpose (see its `notes` field,
  e.g. `prod-feri-handover-checklist-01`) is a **known, tracked, open issue**,
  not a bug in the benchmark - it exists so the fix for that issue has an
  objective, automatic "did it actually work" signal instead of one more
  manual query test.
* `pool_survival_rate` well below 1.0 on a case that otherwise has decent
  `recall_fusion_full` means: the document *is* being retrieved by BM25/vector,
  it's just getting cut by the `rerank_k` pool truncation - a real, actionable,
  distinct signal from "not indexed" or "not semantically matched".

## Risk of this framework: 3/10

Read-only against the real index (never calls `sync()`). `ai_search.py` gained
one addition for this framework's benefit: `search()`'s optional `trace=`
parameter (default `None`, unchanged behaviour/return value either way - see
`ai_search.SearchTrace`'s docstring). The main risks are (a) a future change
to `search()`'s scoring/truncation logic not being matched by an update to
its own nearby trace-writing lines - mitigated by `consistency_check.py`'s
`PipelineDriftError`, which every `python -m benchmark run` performs by
default; and (b) dataset cases making false claims about what exists in the
production index - mitigated by requiring every `production_queries.jsonl`
case to reference something actually verified in the index (documented in
that file's header comment).
