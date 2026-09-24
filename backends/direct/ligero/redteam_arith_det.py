"""Red-team (lane red-team-arith): run ``backends.direct.ligero.run`` with every ``os.urandom`` draw made deterministic,
so two trees can be compared byte for byte on the same coins and mask keys.

    python -m backends.direct.ligero.redteam_arith_det TRACE.json -- --relation R bench-vu --zk ... --dump-dir D

The draws of a ``pipeline.prove_many`` pass are keyed by (sub-batch index, draw number within that sub-batch, length);
the per-sub-batch counters restart at every pass, so the warm-up passes (whose count differs between trees) do not
shift the recorded rep's randomness.  Draws outside a pass are keyed by their order among such draws since the last
pass.  TRACE.json records every draw of the last pass (key, length, calling function) for diagnosis."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import threading
import traceback

_STATE = {"passes": 0, "job": None, "ctr": {}, "trace": [], "last_trace": []}
_LOCK = threading.Lock()
SEED = os.environ.get("RTA_SEED", "red-team-arith/v1").encode()


def _det_urandom(n: int) -> bytes:
    with _LOCK:
        job = _STATE["job"] if threading.current_thread() is threading.main_thread() else f"thread:{threading.current_thread().name}"
        key = "none" if job is None else str(job)
        c = _STATE["ctr"].get(key, 0)
        _STATE["ctr"][key] = c + 1
        fr = traceback.extract_stack(limit=2)[0]
        _STATE["trace"].append({"job": key, "i": c, "n": int(n), "at": f"{os.path.basename(fr.filename)}:{fr.name}"})
    return hashlib.shake_256(SEED + f"|{key}|{c}|{n}".encode()).digest(int(n))


def _install() -> None:
    os.urandom = _det_urandom
    from . import pipeline

    orig_pm = pipeline.prove_many
    orig_step = pipeline._Job.step

    def step(self):
        prev = _STATE["job"]
        _STATE["job"] = self.index
        try:
            return orig_step(self)
        finally:
            _STATE["job"] = prev

    def call(i, starter, *a, **k):
        prev = _STATE["job"]
        _STATE["job"] = i
        try:
            return starter(*a, **k)
        finally:
            _STATE["job"] = prev

    def prove_many(starters, *a, **k):
        with _LOCK:
            _STATE["passes"] += 1
            _STATE["ctr"] = {}
            _STATE["trace"] = []
        try:
            return orig_pm([functools.partial(call, i, s) for i, s in enumerate(starters)], *a, **k)
        finally:
            with _LOCK:
                _STATE["last_trace"] = list(_STATE["trace"])
                _STATE["trace"] = []
                _STATE["ctr"] = {}

    pipeline._Job.step = step
    pipeline.prove_many = prove_many
    for m in list(sys.modules.values()):
        if getattr(m, "prove_many", None) is orig_pm:
            m.prove_many = prove_many


def main() -> int:
    trace_path, sep, *rest = sys.argv[1:]
    assert sep == "--", "usage: redteam_arith_det TRACE.json -- RUN_ARGS..."
    _install()
    from . import run
    rc = 1
    try:
        rc = run.main(rest)
    finally:
        with open(trace_path, "w") as f:
            json.dump({"passes": _STATE["passes"], "last_pass": _STATE["last_trace"], "outside": _STATE["trace"]}, f, indent=0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
