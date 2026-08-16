import os
from pathlib import Path

BOX_ROOT = Path.home() / "Library/CloudStorage/Box-Box/160_Construction/02_Realizace/240783160_Garáže_NDS"
APP_SUPPORT_DIR = Path(os.environ.get("AI_SEARCH_HOME",Path.home() / "Library/Application Support/AI Search"))
DATABASE_DIR = APP_SUPPORT_DIR / "database"
LANCE_DIR = APP_SUPPORT_DIR / "lance"
CACHE_DIR = APP_SUPPORT_DIR / "cache"
LOGS_DIR = APP_SUPPORT_DIR / "logs"
STATE_DIR = APP_SUPPORT_DIR / "state"
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"
COMPLEX_MODEL = "qwen3:14b"
VISION_MODEL = "gemma4"
PARSE_TIMEOUT_SECONDS = 120
CHUNK_TIMEOUT_SECONDS = 60
# Applies to a whole document's embedding phase (all its chunks together), not
# to a batch or a single call - see EmbeddingWatchdog.encode(), where the clock
# starts once before the batch loop. Raised from 60 s after the pre-reindex
# audit (2026-08-07) measured 60 s being hit already: the production log
# recorded embedding timeouts on .xls documents holding as few as 9 chunks,
# because a monolithic token-dense chunk costs ~4.5 s on its own. The row-aware
# XLS/XLSX extractor makes each chunk far cheaper but produces many more of
# them per document, and after a timeout the watchdog kills its subprocess, so
# the next document pays ~13 s to reload the model - a tight limit makes
# failures cascade. 300 s covers the slowest measured document with ~4x margin
# while still bounding a hung embedding.
EMBEDDING_TIMEOUT_SECONDS = 300
EMBEDDING_BATCH_SIZE = 8
# Which branch of the Query Understanding layer (query_expansion.py) the app
# runs with. "fts" widens only the FTS5 MATCH expression with extra OR-terms;
# False disables expansion entirely; True/"vector"/"both" additionally REPLACE
# the text handed to the embedder.
#
# "fts" and not True, measured 2026-08-08 on the 20-query retrieval regression
# suite against the post-reindex index:
#
#   off            recall@10 0.947  MRR 0.772  nDCG 0.817
#   "fts"          recall@10 0.974  MRR 0.778  nDCG 0.827   1 improved, 0 regressed
#   True (both)    recall@10 0.921  MRR 0.697  nDCG 0.751   2 improved, 4 REGRESSED
#
# The vector branch loses because it does not add to the query, it replaces the
# embedded text - so every cosine score in the rerank moves, including for
# candidates the expansion had nothing to do with. Enabling both branches took
# "seznam TP a KZP dodavatelů" from rank 1 to a miss. The FTS branch only ever
# ADDS candidates, which is why it carries no such risk.
#
# Read by app.py and by benchmark/run_retrieval_regression.py, so the suite
# always measures the same path the UI takes - a constant here rather than a
# literal at each call site is what keeps those two from drifting apart.
QUERY_EXPANSION_MODE = "fts"
# Multi-Document Retrieval PR2: gated multi-query orchestration inside
# ui_services.search_all(). Default OFF keeps the single-query path byte-identical
# to pre-PR2 behaviour (verified against the retrieval regression suite).
# When True, search_all may run up to MAX_SUBQUERIES facet subqueries for
# multi-concept queries only (see query_facets.should_use_multi_query).
MULTI_QUERY_RETRIEVAL_ENABLED = False
MAX_SUBQUERIES = 4
# Per-facet subquery fetch budget (chunk rows from ai_search.search). Q_full
# still uses search_all's normal fetch_limit; facet legs stay smaller so
# MAX_SUBQUERIES cannot multiply the QA 500-pool.
MULTI_QUERY_FACET_FETCH_LIMIT = 30
# PR5 spike: auxiliary conjunctive FTS leg that ADDS candidates into search()
# before Phase 3 rerank. Default OFF → search() path is bit-identical to pre-PR5
# (no DF lookups, no extra MATCH). Does not change RRF, bonuses, or QE.
AUXILIARY_TERM_COVERAGE_ENABLED = False
# Max chunk rows returned by the single auxiliary FTS MATCH.
AUX_FTS_LIMIT = 25
# Max new chunk_ids appended onto top_ids (after dedupe against the RRF pool).
AUX_MAX_NEW_IDS = 15
# Absolute DF ceiling for an "anchor" (rare) term. Calibrated on the NDS index
# where CRM≈36 and deska≈1693 — high-DF construction nouns must not be anchors.
AUX_DF_RARE_MAX = 200
# PR5.1: max chunk hits for an aux constraint prefix* (COUNT MATCH stem*).
# Blocks Ren*/svá*/desk*-class expansions while allowing longer rare stems.
AUX_PREFIX_DF_MAX = 150
# PR6/PR6.2: deterministic signed-contract answer safety gate in ai_search.answer().
# Default OFF → answer() is bit-identical to pre-PR6 (gate is never invoked).
DOCUMENT_STATE_GATE_ENABLED = False
# PR7.1: additive chunk-span capture in ui_services.search_all(). Default OFF →
# search_all() returns byte-identical rows to pre-PR7.1 (the `_evidence_spans`
# key is never created). ON only ADDS that key: quote, evidence, score, ranking
# and result order stay untouched, and nothing consumes the key yet.
EVIDENCE_RUNTIME_VALIDATION_ENABLED = False
# PR8.1.1: additive Phase-3 entity match bonus (explicit query token / NOT-id
# ⊆ document name or path). Default OFF → search() scoring is bit-identical to
# pre-PR8.1.1. Does not change FTS, Lance, embeddings, answer(), or safety.
ENTITY_MATCH_BONUS_ENABLED = False
# PR8.1.2: subject→entity conjunction aliases (BOZP+smlouva→SafetyPeak, …).
# Default OFF → bit-identical to PR8.1.1-only / pre-alias behaviour. Requires
# the Phase-3 entity bonus hook; when ON alone it injects alias needles only.
SUBJECT_ENTITY_ALIAS_ENABLED = False
# PR8.2: intent-gated revision ranking (aktuální/platný/poslední/finální).
# Default OFF → search() scoring bit-identical to pre-PR8.2. Applies only when
# the query expresses revision intent; never a global "newer is better" rule.
REVISION_RANKING_ENABLED = False
# PR8.2.1: append-only revision-intent candidate recall (HMG/akt_/final/…).
# Default OFF → search() bit-identical to pre-PR8.2.1. Does not reorder the
# baseline pool; Phase-3 revision ranking stays a separate flag.
REVISION_RECALL_ENABLED = False
# PR8.3: OLD/ path safety guard in answer() — demotes OLD rows from
# authoritative context/citations on currency/status queries. Default OFF →
# answer() byte-identical to pre-PR8.3. Does not change search()/FTS/Lance.
OLD_REVISION_GUARD_ENABLED = False
# PR8.4.1: citation contract in answer() rendering (_render_answer_item and its
# two callers). Default OFF → _render_concise_answer/_render_structured_answer
# byte-identical to pre-PR8.4.1 (a claim item with no resolvable zdroj_index is
# still rendered, just without its "(Zdroj: ...)" note). ON → that same item is
# dropped instead of kept unattributed, so a factual claim can never survive
# rendering without a verifiable source. Does not touch retrieval, ranking,
# embeddings, PR8.1/8.2, or old_revision_guard — purely a rendering-layer gate
# on model output that already passed through those stages.
CITATION_CONTRACT_ENABLED = False
# PR8.4.3: abstention override in answer() rendering. Audit (PR8.4.2) found the
# model can return a self-contradictory JSON payload — a non-empty `body`/
# `polozky` with valid, resolvable `zdroj_index` values AND `nenalezeno: true`
# at the same time (nds-qa-05). `_render_concise_answer` trusted `nenalezeno`
# unconditionally and before ever looking at `body`, discarding a genuinely
# cited, evidence-backed answer. Default OFF → both renderers byte-identical
# to pre-PR8.4.3 (an explicit `nenalezeno` always wins). ON → `nenalezeno` is
# only trusted when NO item in the same response survives the PR8.4.1 citation
# contract (enforced unconditionally for this decision, regardless of
# CITATION_CONTRACT_ENABLED's own separate default); if at least one cited
# item survives, it is rendered and the conflicting flag is ignored. Does not
# touch retrieval, ranking, entity/revision (PR8.x), evidence_runtime, or the
# Ollama prompts — a rendering-layer decision over model output only.
ABSTENTION_OVERRIDE_ENABLED = False
# PR8.4.4: structured-summary citation contract in `_render_structured_answer`.
# Audit found `shrnuti` is copied into the answer as free text with no
# `zdroj_index`, so a factual claim can survive after every `polozky` item has
# been dropped by PR8.4.1. Default OFF → structured renderer byte-identical to
# pre-PR8.4.4 (`shrnuti` is always emitted). ON → `shrnuti` is accompanying
# text only: a factual structured answer is shown iff at least one `polozky`
# item survived the existing PR8.4.1/8.4.3 filters; otherwise the renderer
# ignores `shrnuti`/`nenalezene` and returns the canonical sentinel. Does not
# add a source field for `shrnuti`, does not touch the concise renderer,
# retrieval, ranking, evidence_runtime, or Ollama prompts/schema.
STRUCTURED_SUMMARY_CITATION_ENABLED = False
# PR8.4.6: citation contract on the free-text fallback in answer(). When the
# JSON `format` path fails (invalid JSON / timeout / connection error) the
# second Ollama call returns unconstrained prose and never goes through
# `_render_*` / zdroj_index. Default OFF → that fallback text is used as-is
# (byte-identical to pre-PR8.4.6). ON → keep the fallback only when the text
# contains at least one document name from the answer() pool; otherwise
# replace it with the canonical sentinel. Does not apply to a successful JSON
# render, nor to the double-failure "Ollama je nedostupná" path. Separate from
# CITATION_CONTRACT_ENABLED (JSON renderer only).
FALLBACK_CITATION_CONTRACT_ENABLED = False
# PR9.2.1: if the JSON path parses and renders, but citation contract drops
# every substantive item (typically zdroj_index=0) leaving the canonical
# sentinel, run the existing free-text fallback. Default OFF → answer() is
# byte-identical to pre-PR9.2.1 (JSON sentinel is final; fallback stays
# exception-only). Does not remap zdroj_index, does not weaken PR8.4.1–8.4.6,
# and does not fire on explicit abstention (nenalezeno / empty items).
JSON_SENTINEL_FALLBACK_ENABLED = False
# PR9.3.3: pre-LLM query-focused context packing (max 4 evidence rows).
# Default OFF → answer() sends the same answer_results to the LLM as
# pre-PR9.3.3. ON packs ZDROJE only; retrieval, OLD guard, evidence gate,
# document state, and citations stay on the full pool. Does not remap
# zdroj_index 0→1 and does not change ranking.
QUERY_FOCUSED_CONTEXT_PACKING_ENABLED = False
# PR9.3.4: deterministic entity / identifier hint candidates appended to the
# answer prompt after PR9.3.3 packing. Default OFF → the prompt string is
# byte-identical to pre-PR9.3.4. ON adds one "KANDIDÁTI K OVĚŘENÍ" section
# listing values already visible in ZDROJE with their 1-based source index;
# it never formulates an answer. Retrieval, ranking, OLD guard, evidence gate,
# renderer, and citations are untouched, and no zdroj_index is remapped.
ENTITY_HINTS_ENABLED = False
# PR9.4.1: additive Phase-3 metadata bonus in search() — generic token
# overlap (query <-> document name/path), a safe date whitelist, and
# structural discriminator matching (floor notation, drawing-style codes,
# alphanumeric ids). Default OFF → search() scoring is bit-identical to
# pre-PR9.4.1. Does not change FTS, Lance, embeddings, candidate_strategy,
# top_ids, or any document-class/revision/recency preference; reads only
# document name/path, never chunk heading/quote. See metadata_rerank.py.
METADATA_RERANK_ENABLED = False
# PR9.4.2 / PR9.6.1: additive Phase-3 query-class ↔ document-class affinity.
# Default ON after PR9.6.0 validation (drawing queries demote REGULATORY /
# TECHNICAL_REPORT; DRAWING filenames are not boosted). Flag OFF remains a
# no-op. Lookup rerank window (RERANK_POOL_SIZE=80) is a separate, always-on
# pool-size change. Does not change FTS, Lance, embeddings, retrieval pool,
# QA rerank, or document_state. See document_class_affinity.py.
DOCUMENT_CLASS_AFFINITY_ENABLED = True
# PR9.4.4: intent-gated BM25-floor Phase-3 admission + family-local latest
# revision bonus (+0.03). Default OFF → search() bit-identical to pre-PR9.4.4
# (no extras, no new match keys). Does not change retrieval/RRF/embeddings
# and does not enable REVISION_RANKING_ENABLED. See family_revision_rerank.py.
FAMILY_REVISION_RERANK_ENABLED = False
# PR9.5.0: multi-PSM OCR candidate selection for weak single-page scans.
# Default OFF → extract_pdf() stays on --psm 6 only, no extra tesseract
# calls. Does not change chunks(), search(), ranking, or embeddings.
# See pdf_ocr_candidates.py.
PDF_MULTI_PSM_OCR_ENABLED = False
MSG_PARSE_TIMEOUT_SECONDS = 120
