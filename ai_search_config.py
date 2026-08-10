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
MSG_PARSE_TIMEOUT_SECONDS = 120
