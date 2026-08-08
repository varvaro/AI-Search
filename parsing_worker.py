"""Stable multiprocessing.Process entrypoints for document/message parsing.

Root cause this module fixes (see project audit, PicklingError cluster of 113
events in one production indexing run): `ParsingWatchdog`/`MsgParsingWatchdog`
in ai_search.py spawn (multiprocessing context "spawn") a child process whose
`target=` is a plain top-level function that lives in ai_search.py itself.
`pickle` serializes such a function "by reference" (module name + qualified
name) and, at the exact moment `Process.start()` pickles it, re-resolves
`getattr(sys.modules[module], qualname)` and requires it `is` the object being
pickled (see cpython `pickle.py:save_global`/`Pickler.save_global`). Streamlit
runs the whole app inside one process and its `LocalSourcesWatcher` calls
`importlib.reload()` on locally-edited modules such as ai_search.py - reload
re-executes the module's top-level code *in place* (same module object, same
`__dict__`), which creates a *new* function object for every `def` at module
scope, including the old worker target. `sync()` normally runs in a long-lived
background daemon thread (see app.py's indexing job thread) while Streamlit's
main thread keeps handling reruns/reloads triggered by ordinary edits to
ai_search.py during development - so a reload can land in the (small but
nonzero) window between the watchdog grabbing its target reference and
`Process.start()` pickling it, or simply mean any later `.start()` call
(process restart after a timeout-kill, or watchdog recreated for the next
sync() run) captures a target object whose module-level name has since been
rebound to a different function object. Either way, pickle's identity check
fails with "Can't pickle <function ...>: it's not the same object as
ai_search.<name>".

Fix: keep these Process targets as plain, top-level, closure-free functions
(so they stay standard-pickle friendly) but move them OUT of ai_search.py into
this dedicated module. ai_search.py is the file that actually changes during
active development (retrieval, scoring, answer(), ...) and therefore the file
Streamlit's watcher actually reloads; this module intentionally contains
*nothing* except the two worker loops below, so it has no reason to be edited
- and therefore no reason to be reloaded - while a sync() run is in flight.
`import parsing_worker` from ai_search.py just rebinds the (already cached,
identical) module object on every reload of ai_search.py; it does not
re-execute parsing_worker.py, so `parsing_worker.parsing_worker_main` and
`parsing_worker.msg_parsing_worker_main` keep the same object identity across
any number of ai_search.py reloads, which is exactly what pickle's identity
check needs.

Each worker imports ai_search lazily, inside the function body, once it is
actually running (as the target of a spawned child process, or - for tests
that monkeypatch ai_search.extract/extract_outlook_msg with a plain callable -
never at all, since ParsingWatchdog.parse() then takes the in-process shortcut
and skips these functions entirely). This keeps importing parsing_worker.py
itself free of side effects (no ai_search import, no process/model/DB/queue
creation at import time) and avoids any import-order coupling back to
ai_search.py, which imports this module at its own top level.
"""
from __future__ import annotations

import signal
import traceback
from pathlib import Path


def _handle_sigterm(signum, frame):
    """Installed only in parsing_worker_main (see below).

    ParsingWatchdog.parse() kills this process with SIGTERM both on timeout and
    now on STOP-during-parse (see ai_search.ParseCancelled). Without a handler,
    the default SIGTERM disposition kills the interpreter immediately: any
    pdftoppm/tesseract child that ai_search.extract_pdf() currently has running
    (tracked in ai_search._active_ocr_subprocess while its Popen.communicate()
    is blocking) is orphaned and keeps running un-timed, and the
    `with tempfile.TemporaryDirectory(...)` guarding the page PNGs never gets to
    run its cleanup. Converting SIGTERM into a raised SystemExit instead lets it
    propagate as a normal Python exception through extract_pdf()'s call stack,
    so the temp directory's __exit__ still runs, while we explicitly kill the
    tracked subprocess first since normal exception unwinding does not know
    about a raw Popen handle sitting outside any try/finally in scope yet.
    """
    import ai_search
    proc = ai_search._active_ocr_subprocess[0]
    if proc is not None and proc.poll() is None:
        try: proc.kill()
        except Exception: pass
    raise SystemExit(143)


def parsing_worker_main(requests, responses):
    """Process target used by ParsingWatchdog. Runs until it receives None."""
    signal.signal(signal.SIGTERM, _handle_sigterm)
    import ai_search
    while True:
        item = requests.get()
        if item is None:
            return
        request_id, path = item
        try:
            responses.put((request_id, ai_search.extract(Path(path)), None))
        except Exception:
            responses.put((request_id, None, traceback.format_exc()))
        # SystemExit/KeyboardInterrupt (notably the SystemExit raised by
        # _handle_sigterm above) intentionally propagate out of this loop
        # instead of being caught: they must actually terminate the process so
        # ParsingWatchdog's terminate()+join() completes promptly instead of
        # racing a worker that swallowed its own kill signal.


def msg_parsing_worker_main(requests, responses):
    """Process target used by MsgParsingWatchdog. Runs until it receives None."""
    import ai_search
    while True:
        item = requests.get()
        if item is None:
            return
        request_id, path = item
        try:
            responses.put((request_id, ai_search.extract_outlook_msg(Path(path)), None))
        except BaseException:
            responses.put((request_id, None, traceback.format_exc()))
