"""Process-level cyclic-GC thresholds for the offline analysis processes (the Build's derives (`build`) and workload Program compose
(`global-program`), fold worker, Match checkers, `manifest build` / `build-global`).

Content-neutral by construction: CPython's cyclic collector only frees objects that are already unreachable; no
artifact byte, digest, verdict or population depends on WHEN it runs. What depends on it is wall time. With the
interpreter defaults (700, 10, 10) a full (gen-2) collection is triggered every time the long-lived heap has grown
25 %, and each one walks the whole resident graph -- the fold at llama B8 256/32 ran 24 gen-2 + 3,073 gen-1
collections (mfast M-1147: fold 35.5 -> 23.5 s wall with (100000, 50, 1000), peak RSS +3.5 %); at B8 1024/128 the
resident fold graph is ~27 GB and each full collection is tens of seconds; the B8 1024/128 envelope's derive takes 203 s
at the defaults and 173 s tuned, to the same Program. gc.disable() is NOT the choice: the replay's cyclic garbage then
stays resident (+38 % RSS at 256/32).

At exit the heap is frozen (`gc.freeze`, an atexit hook) so the interpreter's shutdown collections skip it: they walk the whole
resident graph to free memory the process is about to return anyway. The B8 1024/128 envelope's derive ended 18.7 s after its
`main` returned, 1.3 s frozen (the same Program); the workload compose's process ended 30.7 s after its last write. Frozen
objects are still freed by reference count as the modules are torn down; a cycle the shutdown collection would have freed is
left to the OS instead, so every output of these processes is closed before `main` returns (each is written in a `with` or a
one-shot `json.dump(..., open(...))`).

VERITY_GC_THRESHOLDS
  unset / ""         -> TUNED (100000, 50, 1000), heap frozen at exit
  "default" / "off"  -> leave the interpreter alone (the A/B control)
  "a,b,c"            -> gc.set_threshold(a, b, c), heap frozen at exit
"""
from __future__ import annotations

import atexit
import gc
import os

ENV = "VERITY_GC_THRESHOLDS"
TUNED = (100_000, 50, 1000)
LEAVE = ("default", "off", "none", "0")
_freeze_registered = False


def tune(env: str = ENV) -> tuple[int, int, int] | None:
    """Apply the configured thresholds to THIS process and freeze its heap at exit; return the thresholds, or None when the
    interpreter is left alone."""
    global _freeze_registered
    raw = os.environ.get(env, "").strip().lower()
    if raw in LEAVE:
        return None
    if not raw:
        th = TUNED
    else:
        parts = [int(x) for x in raw.split(",")]
        if len(parts) != 3 or min(parts) < 0:
            raise ValueError(f"{env}={raw!r}: want 'default' or three non-negative ints 'a,b,c'")
        th = (parts[0], parts[1], parts[2])
    gc.set_threshold(*th)
    if not _freeze_registered:
        atexit.register(gc.freeze)
        _freeze_registered = True
    return th
