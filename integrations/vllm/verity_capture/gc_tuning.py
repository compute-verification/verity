"""Process-level cyclic-GC thresholds for the offline analysis processes (fold worker, Match checkers).

Content-neutral by construction: CPython's cyclic collector only frees objects that are already unreachable; no
artifact byte, digest, verdict or population depends on WHEN it runs. What depends on it is wall time. With the
interpreter defaults (700, 10, 10) a full (gen-2) collection is triggered every time the long-lived heap has grown
25 %, and each one walks the whole resident graph -- the fold at llama B8 256/32 ran 24 gen-2 + 3,073 gen-1
collections (mfast M-1147: fold 35.5 -> 23.5 s wall with (100000, 50, 1000), peak RSS +3.5 %); at B8 1024/128 the
resident fold graph is ~27 GB and each full collection is tens of seconds. gc.disable() is NOT the choice: the
replay's cyclic garbage then stays resident (+38 % RSS at 256/32).

VERITY_GC_THRESHOLDS
  unset / ""         -> TUNED (100000, 50, 1000)
  "default" / "off"  -> leave the interpreter alone (the A/B control)
  "a,b,c"            -> gc.set_threshold(a, b, c)
"""
from __future__ import annotations

import gc
import os

ENV = "VERITY_GC_THRESHOLDS"
TUNED = (100_000, 50, 1000)
LEAVE = ("default", "off", "none", "0")


def tune(env: str = ENV) -> tuple[int, int, int] | None:
    """Apply the configured thresholds to THIS process; return them, or None when left at the interpreter's."""
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
    return th
