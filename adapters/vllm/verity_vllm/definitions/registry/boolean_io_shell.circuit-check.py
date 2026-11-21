"""circuit-check's bindings of `verity_vllm.definitions.registry.boolean_io_shell`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _scan_slots(ns: int, firsts: tuple[int, int, int, int], stride: int, lane: int, np_: int, fn: str = "ScanSlotsV5") -> Any:
    """`ScanSlots_v5` (or ``fn``, its `ScanSlotsBits_v1`) over the network attester's scan state row at the scan's cells of record
    (`verity.protocols.accounting.network.prover.ingress_scan.layout`): its ``ns`` words, lane 0's occupancy, prompt length, max_tokens
    and ids bit offsets (``firsts``), the lane ``stride``, at lane ``lane`` of ``np_`` prompt ids."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import boolean_io_shell as BX

    occ, lp, mx, ids = (f + stride * lane for f in firsts)
    return bind(getattr(BX, fn), NS=ns, NP=np_, IDB=17, K0=12, K=45, OCC=occ, LP=lp, MAX=mx, IDS=ids)


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`IngressPrompt_v2` at a small layout (3 ids under a 1,000-entry vocabulary) and at a served one (16 ids under a
    128,256-entry vocabulary, 6-character fields); `IngressSampled_v1` at the same two with a 6-byte and a 16-byte id;
    `ScanSlots_v5` at TINY's class's scan state row (lane 1 of 2, KO 2, 4 prompt ids), the class of record's (lane 3 of 4,
    KO 64, 1,024) and the remote application's (lane 0 of 4, KO 2, 27, ``ScanSlots_v5:4:2:27``), and its `ScanSlotsBits_v1`
    at each."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import boolean_io_shell as BX

    return [
        ("IngressPrompt_v2:1", lambda: bind(BX.IngressPromptV2, MODEL="m", LP=3, V=1000)),
        ("IngressPrompt_v2:2", lambda: bind(BX.IngressPromptV2, MODEL="served-model", LP=16, V=128256)),
        ("IngressSampled_v1:1", lambda: bind(BX.IngressSampledV1, MODEL="m", LP=3, V=1000, RID=6)),
        ("IngressSampled_v1:2", lambda: bind(BX.IngressSampledV1, MODEL="served-model", LP=16, V=128256, RID=16)),
        ("ScanSlots_v5:2:2:4", lambda: _scan_slots(4548, (376, 168, 200, 474), 36333, 1, 4)),
        ("ScanSlots_v5:4:64:1024", lambda: _scan_slots(9353, (376, 168, 200, 1528), 37387, 3, 1024)),
        ("ScanSlots_v5:4:2:27", lambda: _scan_slots(9090, (376, 168, 200, 474), 36333, 0, 27)),
        ("ScanSlotsBits_v1:2:2:4", lambda: _scan_slots(4548, (376, 168, 200, 474), 36333, 1, 4, "ScanSlotsBitsV1")),
        ("ScanSlotsBits_v1:4:64:1024", lambda: _scan_slots(9353, (376, 168, 200, 1528), 37387, 3, 1024, "ScanSlotsBitsV1")),
        ("ScanSlotsBits_v1:4:2:27", lambda: _scan_slots(9090, (376, 168, 200, 474), 36333, 0, 27, "ScanSlotsBitsV1")),
    ]
