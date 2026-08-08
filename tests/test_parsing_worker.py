"""Regression tests for the ParsingWatchdog / parsing_worker multiprocessing
lifecycle fix (PicklingError under Streamlit hot-reload during production
indexing).

Root cause under test: multiprocessing.Process(target=<fn>) pickles <fn> "by
reference" (module name + qualname) and pickle re-validates identity
(``getattr(sys.modules[mod], name) is fn``) at ``Process.start()`` time. When
the worker function lived inside ai_search.py, an ``importlib.reload(ai_search)``
(what Streamlit's LocalSourcesWatcher does whenever ai_search.py is edited on
disk while the background ``sync()`` indexing thread keeps running) replaces
that name with a brand new function object, so a subsequent ``.start()``
using a now-stale reference fails with "Can't pickle ... not the same
object". The fix moves the Process targets into parsing_worker.py, a module
ai_search.py imports but never itself edits/reloads.

These tests exercise the REAL multiprocessing "spawn" boundary end to end -
no monkeypatching of ``ai_search.extract``/``ai_search.extract_outlook_msg``,
since that would make ``ParsingWatchdog.parse()`` take its in-process
shortcut and never touch pickle/spawn at all (that shortcut is exactly why
the existing test suite never caught this bug) - and a REAL
``importlib.reload(ai_search)`` to reproduce the original failure condition.

Per the project safety rule, these tests talk to ParsingWatchdog /
MsgParsingWatchdog directly (never ai_search.sync() against iter_documents on
a monkeypatched/targeted path) and only ever touch throwaway tmp_path
fixtures - never a production index.
"""
from __future__ import annotations

import importlib
import multiprocessing
import multiprocessing.reduction
import sys
import threading
import time

import pytest

import ai_search
import parsing_worker


class FakeEmbeddings:
    name = "fake"

    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _assert_no_zombies():
    """No leftover multiprocessing children of the current test process."""
    assert not multiprocessing.active_children()


# ---------------------------------------------------------------------------
# A) worker targets are standard-pickle friendly under the real spawn context
# ---------------------------------------------------------------------------
def test_a_worker_targets_are_picklable_under_spawn_context():
    # multiprocessing.reduction.ForkingPickler is exactly what Process.start()
    # uses internally - the same identity-checking path that raised
    # PicklingError (spawn always uses ForkingPickler regardless of platform).
    multiprocessing.reduction.ForkingPickler.dumps(parsing_worker.parsing_worker_main)
    multiprocessing.reduction.ForkingPickler.dumps(parsing_worker.msg_parsing_worker_main)


# ---------------------------------------------------------------------------
# Mechanism proof: reproduces the ORIGINAL bug shape on a disposable module
# (never touches ai_search.py), so we know test B below is a meaningful guard
# and not a tautology.
# ---------------------------------------------------------------------------
def test_mechanism_stale_reference_reproduces_original_picklingerror(tmp_path):
    module_path = tmp_path / "old_style_worker_fixture.py"
    module_path.write_text("def _parsing_worker(requests, responses):\n    pass\n")
    sys.path.insert(0, str(tmp_path))
    try:
        import old_style_worker_fixture as old_module
        stale_target = old_module._parsing_worker
        # Simulate Streamlit's LocalSourcesWatcher reloading the module while
        # a background thread already holds a reference captured pre-reload.
        module_path.write_text("def _parsing_worker(requests, responses):\n    pass  # recompiled\n")
        importlib.reload(old_module)
        with pytest.raises(Exception) as exc_info:
            multiprocessing.reduction.ForkingPickler.dumps(stale_target)
        assert "not the same object" in str(exc_info.value)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("old_style_worker_fixture", None)


def test_parsing_worker_module_identity_stable_across_ai_search_reload():
    """Why the fix works: parsing_worker.py is never re-executed by an
    ai_search.py reload, so its function objects' identity never changes."""
    before = ai_search.parsing_worker.parsing_worker_main
    importlib.reload(ai_search)
    after = ai_search.parsing_worker.parsing_worker_main
    assert before is after


# ---------------------------------------------------------------------------
# B) real reload + real spawn: the actual reproduction the bug report asked for
# ---------------------------------------------------------------------------
def test_b_worker_survives_importlib_reload_of_ai_search(tmp_path):
    importlib.reload(ai_search)
    target = tmp_path / "a.txt"; target.write_text("alpha text")
    watchdog = ai_search.ParsingWatchdog()
    try:
        text, method = watchdog.parse(target, limit=10)
        assert "alpha text" in text
    finally:
        watchdog.close()
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# C) multiple sequential worker starts, before AND after reload
# ---------------------------------------------------------------------------
def test_c_multiple_sequential_starts_before_and_after_reload(tmp_path):
    files = [tmp_path / f"{i}.txt" for i in range(3)]
    for i, f in enumerate(files): f.write_text(f"obsah {i}")
    watchdog = ai_search.ParsingWatchdog()
    try:
        text, _ = watchdog.parse(files[0], limit=10); assert "obsah 0" in text
        importlib.reload(ai_search)
        text, _ = watchdog.parse(files[1], limit=10); assert "obsah 1" in text
        watchdog.close()  # force a fresh process start, simulating a restart
        text, _ = watchdog.parse(files[2], limit=10); assert "obsah 2" in text
    finally:
        watchdog.close()
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# D) parser success via real spawn
# ---------------------------------------------------------------------------
def test_d_parser_success_via_real_subprocess(tmp_path):
    target = tmp_path / "a.txt"; target.write_text("obsah pro parsing")
    watchdog = ai_search.ParsingWatchdog()
    try:
        text, method = watchdog.parse(target, limit=10)
    finally:
        watchdog.close()
    assert text == "obsah pro parsing" and method == "text"
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# E / J) parser exception propagation, and cleanup after an exception response
# ---------------------------------------------------------------------------
def test_e_parser_exception_propagates_and_cleans_up(tmp_path):
    missing = tmp_path / "missing.txt"  # never created -> extract() raises
    watchdog = ai_search.ParsingWatchdog()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            watchdog.parse(missing, limit=10)
        assert "FileNotFoundError" in str(exc_info.value)
        # Worker loop catches the exception per-request and stays alive to
        # serve the next document - confirm it is still usable.
        assert watchdog.process is not None and watchdog.process.is_alive()
    finally:
        watchdog.close()
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# F / I) parsing timeout kills the child and cleans it up
# ---------------------------------------------------------------------------
def test_f_parsing_timeout_kills_child_and_raises(tmp_path):
    target = tmp_path / "a.txt"; target.write_text("alpha")
    watchdog = ai_search.ParsingWatchdog()
    try:
        # Real subprocess spawn + import + extract() always takes far longer
        # than 1ms, so this is a genuine timeout, not a contrived sleep.
        with pytest.raises(ai_search.PhaseTimeout):
            watchdog.parse(target, limit=0.001)
        process = watchdog.process
        assert process is not None
        deadline = time.monotonic() + 5
        while process.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process.is_alive()
    finally:
        watchdog.close()
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# G) STOP/cancel through the real sync() path (isolated tmp_path index only)
# ---------------------------------------------------------------------------
def test_g_stop_during_real_parsing_terminates_quickly_without_zombies(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    for i in range(5): (root / f"{i}.txt").write_text(f"obsah {i}")
    db, lance = tmp_path / "state.sqlite3", tmp_path / "lance"
    stop = threading.Event()
    def progress(event):
        if event.get("phase") == "Parsování" and event.get("current") == 2: stop.set()
    started = time.perf_counter()
    result = ai_search.sync(root, db, lance, FakeEmbeddings(), progress=progress, stop_event=stop)
    elapsed = time.perf_counter() - started
    assert result["stopped"] is True
    assert elapsed < 15  # bounded - no hang
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# H) child cleanup - no live process after a clean success + close()
# ---------------------------------------------------------------------------
def test_h_no_live_process_after_success_and_close(tmp_path):
    target = tmp_path / "a.txt"; target.write_text("alpha")
    watchdog = ai_search.ParsingWatchdog()
    watchdog.parse(target, limit=10)
    assert watchdog.process.is_alive()
    watchdog.close()
    assert not watchdog.process.is_alive()
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# K) output parity: spawn-based worker returns the same result as the direct,
#    in-process extract() call for the same file (nothing about parsing
#    behavior/content changed, only where the Process target function lives)
# ---------------------------------------------------------------------------
def test_k_spawn_worker_output_matches_direct_extract(tmp_path):
    target = tmp_path / "a.txt"; target.write_text("stejný obsah pro oba")
    direct = ai_search.extract(target)
    watchdog = ai_search.ParsingWatchdog()
    try:
        via_worker = watchdog.parse(target, limit=10)
    finally:
        watchdog.close()
    assert via_worker == direct
    _assert_no_zombies()


# ---------------------------------------------------------------------------
# Same architecture applies to MsgParsingWatchdog (1 of the 113 original
# errors was a .msg file) - real spawn, real reload, real error propagation.
# ---------------------------------------------------------------------------
def test_msg_watchdog_survives_reload_and_propagates_errors(tmp_path):
    importlib.reload(ai_search)
    bad_msg = tmp_path / "bad.msg"; bad_msg.write_bytes(b"not a real msg/ole file")
    watchdog = ai_search.MsgParsingWatchdog()
    try:
        with pytest.raises(RuntimeError):
            watchdog.parse(bad_msg, limit=15)
    finally:
        watchdog.close()
    _assert_no_zombies()
