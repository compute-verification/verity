"""circuit-check's bindings of `verity.ml.definitions.registry.io_shell`'s `TokenFeed_v1`, `SlotRow_v1`, `SlotMask_v1`,
`ScanSlots_v1`, `OutCollect_v1` and `SlotEntry_v1`, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`TokenFeed_v1` at a small class (4 prompt ids) and at the class of record's (1,024), each the first position's and
    every other's; `SlotRow_v1` at both; `SlotMask_v1`; `ScanSlots_v1` at both classes' scan state rows; `OutCollect_v1` at
    KO 2 and 64, the first position's and every other's; and `SlotEntry_v1` at the small class (KO 2, 4 prompt ids) with no
    stop id (TINY's) and at the class of record's (KO 64, 1,024) with Llama-3.2-1B's (128001)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import io_shell as X

    # the network certifier's scan state row at w63's cells (`verity_vllm.program.predict.scan_state.layout`, PR2's form with
    # ``occ``): (lanes, KO, NP) -> its words, lane 0's occupancy, prompt length, max_tokens and ids bit offsets, and the lane
    # stride; at a small class (2 lanes, KO 2, 4 prompt ids) and at the class of record's (4 lanes, KO 64, 1,024)
    scan = {(2, 2, 4): (4548, (376, 168, 200, 474), 36333), (4, 64, 1024): (9353, (376, 168, 200, 1528), 37387)}

    def slots(lanes: int, ko: int, np_: int) -> Any:
        ns, firsts, stride = scan[(lanes, ko, np_)]
        occ, lp, mx, ids = (tuple(f + stride * j for j in range(lanes)) for f in firsts)
        return bind(X.ScanSlots, NS=ns, NP=np_, IDB=17, K0=12, K=45, OCC=occ, LP=lp, MAX=mx, IDS=ids)

    return [*[(f"TokenFeed_v1:{NP}:{int(first)}", lambda NP=NP, first=first: bind(X.TokenFeed, NP=NP, FIRST=first))
              for NP in (4, 1024) for first in (True, False)],
            *[(f"SlotRow_v1:{NP}", lambda NP=NP: bind(X.SlotRow, NP=NP)) for NP in (4, 1024)],
            ("SlotMask_v1", lambda: bind(X.SlotMask)),
            *[(f"ScanSlots_v1:{k[0]}:{k[1]}:{k[2]}", lambda k=k: slots(*k)) for k in scan],
            *[(f"OutCollect_v1:{ko}:{int(first)}", lambda ko=ko, first=first: bind(X.OutCollect, KO=ko, FIRST=first))
              for ko in (2, 64) for first in (True, False)],
            ("SlotEntry_v1:2", lambda: bind(X.SlotEntry, KO=2, IDB=17, NP=4, STOP=())),
            ("SlotEntry_v1:64", lambda: bind(X.SlotEntry, KO=64, IDB=17, NP=1024, STOP=(128001,)))]
