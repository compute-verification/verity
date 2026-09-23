"""One pure module per CPU-only oracle.  Each exposes:

    NAME      the check id (the key in expected/<row>.json)
    TIER      "T0" (inputs <= ~100 MB: laptop-safe in principle), "T1" (multi-GB fold records: pod), "T2" (Program rebuild: big-RAM pod)
    STRENGTH  "strong" (bit-for-bit / counted facts) or "weak" (PASS-side only, or a rule no reference row exercises)
    INPUTS    the `artifact://r19/<row>/<rel>` files it reads (documentation + skip reason)
    applies(row) -> str | None      None when the check applies to the row, else the reason it does not (skip by name)
    expected(row) -> dict           the contract, lifted from the row's OWN documents (what `lift_expected.py` freezes into expected/)
    actual(row) -> dict             the same fields recomputed with the CURRENT code (or read from the v2 candidate's output)

`actual` may raise `NotImplementedError` (a stubbed oracle: the test skips and says so) or `resolver.NotResolvable` (an input is not
reachable from this machine: skip by name).  A check never writes into a row dir.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType

ORDER = ("manifest_digest", "program_digest", "global_match_checks", "executed_prefix", "coverage", "commit_summary", "verdict",
         "decomp_hashes", "replay_partition")


def load(name: str) -> ModuleType:
    return import_module(f"{__name__}.{name}")


def all_checks() -> list[ModuleType]:
    return [load(n) for n in ORDER]
