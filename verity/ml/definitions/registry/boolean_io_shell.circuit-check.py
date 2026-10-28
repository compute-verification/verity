"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_io_shell`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _scan_slots(ns: int, firsts: tuple[int, int, int, int], stride: int, lanes: int, np_: int) -> Any:
    """`ScanSlots_v2` over the network certifier's scan state row at the scan's cells of record
    (`verity.ml.protocols.certifier.ingress_scan.layout`): its ``ns`` words, lane 0's occupancy, prompt length, max_tokens
    and ids bit offsets (``firsts``), the lane ``stride``, at ``lanes`` lanes of ``np_`` prompt ids."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_io_shell as BX

    occ, lp, mx, ids = (tuple(f + stride * j for j in range(lanes)) for f in firsts)
    return bind(BX.ScanSlotsV2, NS=ns, NP=np_, IDB=17, K0=12, K=45, OCC=occ, LP=lp, MAX=mx, IDS=ids)


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`IngressPrompt_v2` at a small layout (3 ids under a 1,000-entry vocabulary) and at a served one (16 ids under a
    128,256-entry vocabulary, 6-character fields); `ScanSlots_v2` at TINY's class's scan state row (2 lanes, KO 2, 4 prompt
    ids) and the class of record's (4 lanes, KO 64, 1,024)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_io_shell as BX

    return [
        ("IngressPrompt_v2:1", lambda: bind(BX.IngressPromptV2, MODEL="m", LP=3, V=1000)),
        ("IngressPrompt_v2:2", lambda: bind(BX.IngressPromptV2, MODEL="served-model", LP=16, V=128256)),
        ("ScanSlots_v2:2:2:4", lambda: _scan_slots(4548, (376, 168, 200, 474), 36333, 2, 4)),
        ("ScanSlots_v2:4:64:1024", lambda: _scan_slots(9353, (376, 168, 200, 1528), 37387, 4, 1024)),
    ]
