"""Drive reads its runs' states and evidence several at a time, each result kept with its run."""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from research import drive as D


def test_run_states_reads_runs_in_parallel_and_keeps_each_state_with_its_run(tmp_path, monkeypatch):
    """Reading about 40 runs one `research status` (an ssh round trip) at a time took a tick 33 min. Several are read at once, and each answer stays with the run it is about, whatever order they finish in."""
    seen: set[int] = set()
    lock = threading.Lock()

    def run(argv, **kw):
        rid = argv[argv.index("status") + 1]
        with lock:
            seen.add(threading.get_ident())
        state = {"r1": "running", "r2": "done", "r3": "failed"}.get(rid)
        if state is None:
            return subprocess.CompletedProcess(argv, 1, "", "no such run")
        d = tmp_path / rid
        d.mkdir(exist_ok=True)
        rc = 0 if state == "done" else (1 if state == "failed" else None)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"runs": [{"state": state, "rc": rc, "dir": str(d)}]}), "")

    monkeypatch.setattr(D.subprocess, "run", run)
    ids = ["r3", "r1", "nope", "r2"]
    out = D.run_states(tmp_path, ids)
    assert list(out) == ids
    assert out["r1"][0] == "running" and out["r2"][:2] == ("done", 0) and out["r3"][:2] == ("failed", 1)
    assert out["nope"] == ("unknown", None, None, None)
    assert seen and threading.get_ident() not in seen, "read on worker threads, not one after another on the caller's"


def test_each_keeps_order_and_runs_a_single_item_inline():
    assert D._each(lambda x: x * 2, [3, 1, 2]) == [6, 2, 4]
    me = threading.get_ident()
    assert D._each(lambda _: threading.get_ident(), ["only"]) == [me]
    assert D._each(lambda x: x, []) == []


def test_evidence_drops_runs_it_cannot_fetch_and_keeps_the_rest_in_order(tmp_path, monkeypatch):
    got = {"r1": tmp_path / "r1", "r3": tmp_path / "r3"}
    for d in got.values():
        d.mkdir()
    monkeypatch.setattr(D, "_fetch", lambda repo, rid, names: got.get(rid))
    assert D.evidence(Path("."), ["r3", "r2", "r1"]) == [got["r3"], got["r1"]]
