"""Fáze 0 - konzistence filesystem vs. index.

Pokrývá dvě třídy produkčních selhání:
1. Dokument je v indexu, ale na disku chybí -> vyhledávání nesmí spadnout.
2. Sken zdrojové složky selže nebo je neúplný -> sync nesmí smazat index.
"""
from pathlib import Path

import pytest

import ai_search
import ui_services as ui


class FakeEmbeddings:
    def encode(self, texts):
        return [[1.0, 0.5, 0.1] for _ in texts]


@pytest.fixture
def indexed(tmp_path):
    root = tmp_path / "Projekt"
    root.mkdir()
    for name in ("alpha.txt", "beta.txt", "gama.txt"):
        (root / name).write_text(f"Dokument {name} popisuje betonáž a harmonogram.", encoding="utf-8")
    db, lance = tmp_path / "state" / "index.sqlite3", tmp_path / "state" / "lance"
    embeddings = FakeEmbeddings()
    ai_search.sync(root, db, lance, embeddings)
    return root, db, lance, embeddings


def document_count(db: Path) -> int:
    with ai_search.database(db) as con:
        return con.execute("SELECT count(*) FROM documents").fetchone()[0]


# --- 1. Dokument v indexu, ale fyzicky chybí --------------------------------

def test_metadata_for_missing_file_returns_safe_metadata(tmp_path):
    data = ui.metadata_for(tmp_path / "neexistuje.pdf", "Dokument")
    assert data["availability"] == "missing"
    assert data == {"source": "Dokument", "date": "", "author": "", "extension": "pdf", "availability": "missing"}


def test_metadata_for_transient_error_is_not_reported_as_missing(tmp_path, monkeypatch):
    path = tmp_path / "box.pdf"
    path.write_bytes(b"%PDF-x")
    real_stat = Path.stat

    def failing_stat(self, *args, **kwargs):
        if self.name == "box.pdf":
            raise PermissionError("svazek dočasně nedostupný")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)
    data = ui.metadata_for(path, "Dokument")
    assert data["availability"] == "unavailable"
    assert data["extension"] == "pdf" and data["date"] == ""


def test_search_all_survives_document_missing_from_disk(tmp_path, monkeypatch):
    settings = ui.Settings(project_root=str(tmp_path))
    db, _ = ui.state_paths(tmp_path, "Dokument")
    db.touch()
    missing = tmp_path / "Projekt" / "presunuty.pdf"
    row = {"document": "presunuty.pdf", "path": str(missing), "project": "P", "quote": "citace", "score": 1.0}
    monkeypatch.setattr(ai_search, "search", lambda *a, **k: [row.copy()])
    results = ui.search_all("haus365 kladečský plán", settings, tmp_path, FakeEmbeddings())
    assert len(results) == 1
    assert results[0]["availability"] == "missing" and results[0]["quote"] == "citace"


# --- 2. Chyba stat() nesmí odstranit dokument -------------------------------

def test_stat_error_does_not_remove_document(indexed, monkeypatch):
    root, db, lance, embeddings = indexed
    real_stat = Path.stat

    def failing_stat(self, *args, **kwargs):
        if self.name == "beta.txt":
            raise PermissionError("dočasně nedostupné")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)
    counts = ai_search.sync(root, db, lance, embeddings)
    assert counts["removed"] == 0 and counts["removal_skipped"] == 1
    assert document_count(db) == 3


def test_file_present_but_missed_by_scan_is_kept(indexed, monkeypatch):
    root, db, lance, embeddings = indexed
    real_iter = ai_search.iter_documents
    monkeypatch.setattr(ai_search, "iter_documents", lambda base: [p for p in real_iter(base) if p.name != "gama.txt"])
    counts = ai_search.sync(root, db, lance, embeddings)
    assert counts["removed"] == 0 and counts["removal_skipped"] == 1
    assert document_count(db) == 3


# --- 3. Prázdný root nesmí smazat index -------------------------------------

def test_empty_root_aborts_sync_without_deleting(indexed):
    root, db, lance, embeddings = indexed
    for path in root.iterdir():
        path.unlink()
    with pytest.raises(ai_search.SourceUnavailableError):
        ai_search.sync(root, db, lance, embeddings)
    assert document_count(db) == 3


def test_empty_root_with_empty_index_is_allowed(tmp_path):
    root = tmp_path / "Prazdny"
    root.mkdir()
    counts = ai_search.sync(root, tmp_path / "state" / "index.sqlite3", tmp_path / "state" / "lance", FakeEmbeddings())
    assert counts["removed"] == 0 and counts["added"] == 0


def test_scan_far_below_index_size_aborts_sync(tmp_path):
    """Poměrová ochrana se uplatní až nad REMOVAL_GUARD_MIN_DOCUMENTS."""
    root = tmp_path / "Velky"
    root.mkdir()
    names = [f"dokument-{i:02d}.txt" for i in range(30)]
    for name in names:
        (root / name).write_text(f"Obsah {name}", encoding="utf-8")
    db, lance = tmp_path / "state" / "index.sqlite3", tmp_path / "state" / "lance"
    embeddings = FakeEmbeddings()
    ai_search.sync(root, db, lance, embeddings)
    assert document_count(db) == 30
    for name in names[5:]:
        (root / name).unlink()
    with pytest.raises(ai_search.SourceUnavailableError):
        ai_search.sync(root, db, lance, embeddings)
    assert document_count(db) == 30


# --- 4. Potvrzeně chybějící dokument se odstraní ----------------------------

def test_confirmed_missing_document_is_removed(indexed):
    root, db, lance, embeddings = indexed
    (root / "alpha.txt").unlink()
    counts = ai_search.sync(root, db, lance, embeddings)
    assert counts["removed"] == 1 and counts["removal_skipped"] == 0
    assert document_count(db) == 2


# --- 5. Sken vynechává ~BROMIUM (HP Sure Click) -----------------------------
# Produkce 2026-08-09: 10 z 17 chyb indexace byly několikasetbajtové stuby,
# které Sure Click zrcadlí do "~BROMIUM"; originály byly zaindexované ze svých
# skutečných umístění. Viz BROMIUM_DIRECTORY v ai_search.py.

def bromium_tree(tmp_path):
    root = tmp_path / "Projekt"
    (root / "~BROMIUM").mkdir(parents=True)
    (root / "podslozka" / "~BROMIUM").mkdir(parents=True)
    (root / "~bromium_zaloha").mkdir()
    files = {
        root / "dokument.pdf": True,                          # běžný dokument mimo izolaci
        root / "podslozka" / "vnoreny.pdf": True,
        root / "~bromium_zaloha" / "zaloha.pdf": True,        # jen podobný název, ne izolace
        root / "~BROMIUM" / "stub.pdf": False,
        root / "podslozka" / "~BROMIUM" / "vnoreny-stub.pdf": False,
    }
    for path in files:
        path.write_bytes(b"%PDF-1.4 obsah")
    return root, {path.name for path, indexed in files.items() if indexed}


def test_bromium_files_are_skipped_and_normal_files_are_kept(tmp_path):
    root, expected = bromium_tree(tmp_path)
    assert {path.name for path in ai_search.iter_documents(root)} == expected


def test_bromium_match_is_case_insensitive(tmp_path):
    """APFS/HFS+ i Box jsou case-insensitive, takže složka může být uložená
    jako "~Bromium" i "~BROMIUM" - obojí je stejná Sure Click izolace."""
    root = tmp_path / "Projekt"
    (root / "~Bromium").mkdir(parents=True)
    (root / "~Bromium" / "stub.pdf").write_bytes(b"%PDF-1.4 obsah")
    (root / "dokument.pdf").write_bytes(b"%PDF-1.4 obsah")
    assert [path.name for path in ai_search.iter_documents(root)] == ["dokument.pdf"]


def test_sync_does_not_report_bromium_stubs_as_errors(tmp_path):
    root = tmp_path / "Projekt"
    (root / "~BROMIUM").mkdir(parents=True)
    (root / "~BROMIUM" / "stub.pdf").write_bytes(b"tohle neni PDF")   # stub, který by parser neotevřel
    (root / "dokument.txt").write_text("Betonáž základové desky.", encoding="utf-8")
    counts = ai_search.sync(root, tmp_path / "state" / "index.sqlite3", tmp_path / "state" / "lance", FakeEmbeddings())
    assert counts["errors"] == 0
    assert counts["added"] == 1
