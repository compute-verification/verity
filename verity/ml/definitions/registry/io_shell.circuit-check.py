"""circuit-check's bindings of `verity.ml.definitions.registry.io_shell`'s `TokenFeed_v1`, `SlotRow_v1`, `SlotMask_v1`,
`ScanSlots_v1`, `OutCollect_v1` and `SlotEntry_v1`, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _scan_slots(ns: int, firsts: tuple[int, int, int, int], stride: int, lanes: int, np_: int) -> Any:
    """`ScanSlots_v1` over the network certifier's scan state row at w63's cells (`verity_vllm.program.predict.scan_state.layout`,
    PR2's form with ``occ``): its ``ns`` words, lane 0's occupancy, prompt length, max_tokens and ids bit offsets
    (``firsts``), the lane ``stride``, at ``lanes`` lanes of ``np_`` prompt ids."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import io_shell as X

    occ, lp, mx, ids = (tuple(f + stride * j for j in range(lanes)) for f in firsts)
    return bind(X.ScanSlots, NS=ns, NP=np_, IDB=17, K0=12, K=45, OCC=occ, LP=lp, MAX=mx, IDS=ids)


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`TokenFeed_v1` at a small class (4 prompt ids) and at the class of record's (1,024), each the first position's and
    every other's; `SlotRow_v1` at both; `SlotMask_v1`; `ScanSlots_v1` at both classes' scan state rows (2 lanes, KO 2, 4
    prompt ids; 4 lanes, KO 64, 1,024); `OutCollect_v1` at KO 2 and 64, the first position's and every other's; and
    `SlotEntry_v1` at the small class (KO 2, 4 prompt ids) with no stop id (TINY's) and at the class of record's (KO 64,
    1,024) with Llama-3.2-1B's (128001)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import io_shell as X

    return [("TokenFeed_v1:4:1", lambda: bind(X.TokenFeed, NP=4, FIRST=True)),
            ("TokenFeed_v1:4:0", lambda: bind(X.TokenFeed, NP=4, FIRST=False)),
            ("TokenFeed_v1:1024:1", lambda: bind(X.TokenFeed, NP=1024, FIRST=True)),
            ("TokenFeed_v1:1024:0", lambda: bind(X.TokenFeed, NP=1024, FIRST=False)),
            ("SlotRow_v1:4", lambda: bind(X.SlotRow, NP=4)),
            ("SlotRow_v1:1024", lambda: bind(X.SlotRow, NP=1024)),
            ("SlotMask_v1", lambda: bind(X.SlotMask)),
            ("ScanSlots_v1:2:2:4", lambda: _scan_slots(4548, (376, 168, 200, 474), 36333, 2, 4)),
            ("ScanSlots_v1:4:64:1024", lambda: _scan_slots(9353, (376, 168, 200, 1528), 37387, 4, 1024)),
            ("OutCollect_v1:2:1", lambda: bind(X.OutCollect, KO=2, FIRST=True)),
            ("OutCollect_v1:2:0", lambda: bind(X.OutCollect, KO=2, FIRST=False)),
            ("OutCollect_v1:64:1", lambda: bind(X.OutCollect, KO=64, FIRST=True)),
            ("OutCollect_v1:64:0", lambda: bind(X.OutCollect, KO=64, FIRST=False)),
            ("SlotEntry_v1:2", lambda: bind(X.SlotEntry, KO=2, IDB=17, NP=4, STOP=())),
            ("SlotEntry_v1:64", lambda: bind(X.SlotEntry, KO=64, IDB=17, NP=1024, STOP=(128001,)))]
