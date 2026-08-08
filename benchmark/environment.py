"""Wires the benchmark up to a concrete (db, lance, embeddings) target.

Two environments:

  fixture     Small synthetic in-memory corpus + a deterministic fake
              embedding model (see fixtures.py). Builds in a few seconds,
              needs no network/model download/production data. This is the
              default and is what CI / `python -m benchmark run` uses.

  production  The real Application Support paths (same ones ui_services.py
              uses) + the real BAAI/bge-m3 model via ai_search.Embeddings().
              Slow (model load + real index size), requires the app to have
              been run at least once so the index exists. Use
              `--environment production` explicitly.

Neither environment path writes to ai_search.py / ui_services.py, and the
production environment only ever *reads* the existing index - it never syncs
or mutates it.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_search  # noqa: E402
import ai_search_config  # noqa: E402
import ui_services  # noqa: E402

from . import fixtures  # noqa: E402

# Recorded next to the digest in every run artifact so compare.py can refuse
# to compare two digests produced by different algorithms. Run artifacts
# written before 2026-08-07 stored a bare MAX(indexed_at) timestamp under the
# same `index_fingerprint` key - compare.py maps those to the pseudo-algorithm
# "legacy-max-indexed-at" and skips the digest comparison rather than
# reporting a bogus mismatch (see _fingerprint_algorithm() there).
FINGERPRINT_ALGORITHM = "sha256-v1"
_FIELD_SEP = b"\x1f"
_RECORD_SEP = b"\x1e"


def index_identity(db_path: Path) -> dict:
    """Deterministic identity of the index at `db_path`, used by compare.py to
    refuse comparing two runs measured against different indexes.

    `index_fingerprint` is a SHA256 over (doc_count, chunk_count) plus every
    document's (id, path, content_hash, indexed_at) in `ORDER BY id` order.
    It replaces the previous fingerprint, a bare MAX(documents.indexed_at),
    which had two demonstrated false-match holes:

      1. A RENAMED document. ai_search.sync()'s rename branch updates
         documents.path/name/size/mtime_ns/inode but deliberately not
         content_hash (same bytes) and not indexed_at (not a re-index) - so
         doc_count, chunk_count AND MAX(indexed_at) all stay identical while
         every path changes. Benchmark ground truth matches on path
         substrings (metrics.row_matches_any), so every metric could move
         while the identity check reported a perfect match. Verified against
         the real sync() on 2026-08-07: renamed=1, all three old identity
         fields unchanged.
      2. SQLite's CURRENT_TIMESTAMP has one-second resolution
         ('2026-08-07 13:23:21'), so two different indexes built within the
         same second collide.

    Both are covered by hashing per-document identity instead. `path` catches
    renames/moves, `content_hash` catches content changes that keep the chunk
    count stable, `indexed_at` catches a re-index of unchanged content, and
    `id` ordering makes the digest independent of row/scan order.

    Chunk-level rows are covered by `chunk_count` plus the per-document
    `content_hash` rather than by hashing all ~139k chunk ids - chunks are
    derived from the document content that content_hash already pins, so the
    extra scan would cost time without adding discriminating power.

    Never raises: on any DB problem every field comes back None, which
    compare.py treats as "not comparable", never as a silent match."""
    identity = {
        "doc_count": None, "chunk_count": None, "index_fingerprint": None,
        "index_fingerprint_algorithm": None, "index_max_indexed_at": None,
    }
    try:
        digest = hashlib.sha256()
        with ai_search.database(db_path) as con:
            doc_count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            max_indexed_at = con.execute("SELECT MAX(indexed_at) FROM documents").fetchone()[0]
            digest.update(f"{FINGERPRINT_ALGORITHM}{_FIELD_SEP!r}{doc_count}{_FIELD_SEP!r}{chunk_count}".encode())
            digest.update(_RECORD_SEP)
            for row in con.execute("SELECT id,path,content_hash,indexed_at FROM documents ORDER BY id"):
                digest.update(_FIELD_SEP.join(str(value).encode("utf-8", "surrogatepass") for value in row))
                digest.update(_RECORD_SEP)
    except Exception:
        return identity
    identity.update(
        doc_count=doc_count, chunk_count=chunk_count, index_fingerprint=digest.hexdigest(),
        index_fingerprint_algorithm=FINGERPRINT_ALGORITHM, index_max_indexed_at=max_indexed_at,
    )
    return identity


@dataclass
class Environment:
    name: str
    db_path: Path
    lance_dir: Path
    embeddings: object
    state_dir: Path | None = None
    root: Path | None = None
    settings: "ui_services.Settings | None" = None
    doc_count: int | None = None
    chunk_count: int | None = None
    # See index_identity() above for what the fingerprint covers and why the
    # algorithm id travels with it. None when unavailable (empty index, or
    # the query itself failed) - compare.py treats None as "not comparable",
    # never as a silent match.
    index_fingerprint: str | None = None
    index_fingerprint_algorithm: str | None = None
    # The old (pre-sha256) fingerprint value, kept as a human-readable
    # diagnostic in every report - "when was this index last written to".
    index_max_indexed_at: str | None = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "db_path": str(self.db_path),
            "lance_dir": str(self.lance_dir),
            "embedding_model": getattr(self.embeddings, "name", type(self.embeddings).__name__),
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "index_fingerprint": self.index_fingerprint,
            "index_fingerprint_algorithm": self.index_fingerprint_algorithm,
            "index_max_indexed_at": self.index_max_indexed_at,
        }


_fixture_cache: dict[str, "Environment"] = {}


def fixture_environment(tmp_root: Path | None = None) -> Environment:
    """Builds (or reuses, if already built once in this process) the tiny
    synthetic corpus and indexes it with the real `ai_search.sync()` - so the
    indexing/chunking/FTS/LanceDB code under test is 100% real, only the
    corpus and the embedding model are fake."""
    if tmp_root is None and "default" in _fixture_cache:
        return _fixture_cache["default"]
    import tempfile

    base = Path(tmp_root) if tmp_root else Path(tempfile.mkdtemp(prefix="ai-search-benchmark-fixture-"))
    root = fixtures.write_fixture_corpus(base / "corpus")
    embeddings = fixtures.FakeCategoryEmbeddings()
    db_path = base / "state" / "database" / "project.sqlite3"
    lance_dir = base / "state" / "lance" / "project"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lance_dir.parent.mkdir(parents=True, exist_ok=True)
    ai_search.sync(root, db_path, lance_dir, embeddings)
    identity = index_identity(db_path)
    env = Environment(
        name="fixture",
        db_path=db_path,
        lance_dir=lance_dir,
        embeddings=embeddings,
        state_dir=base / "state",
        root=root,
        settings=ui_services.Settings(project_root=str(root), result_count=10),
        **identity,
    )
    if tmp_root is None:
        _fixture_cache["default"] = env
    return env


def production_environment(source: str = "Dokument") -> Environment:
    """Points at the real Application Support state (same as the running
    app). Read-only: does not call sync()/index, only opens the existing
    database/LanceDB table for search()."""
    state_dir = ai_search_config.APP_SUPPORT_DIR
    db_path, lance_dir = ui_services.state_paths(state_dir, source)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Production index not found at {db_path}. Run the app at least once (Aktualizovat "
            f"index) before using --environment production, or pass --state-dir explicitly."
        )
    settings = ui_services.load_settings(ui_services.state_file(state_dir, "settings.json"))
    embeddings = ai_search.Embeddings(name=settings.embedding_model)
    return Environment(
        name="production",
        db_path=db_path,
        lance_dir=lance_dir,
        embeddings=embeddings,
        state_dir=state_dir,
        root=Path(settings.project_root) if settings.project_root else None,
        settings=settings,
        **index_identity(db_path),
    )


def get_environment(name: str, **kwargs) -> Environment:
    if name == "fixture":
        return fixture_environment(**kwargs)
    if name == "production":
        return production_environment(**kwargs)
    raise ValueError(f"unknown environment {name!r} (expected 'fixture' or 'production')")
