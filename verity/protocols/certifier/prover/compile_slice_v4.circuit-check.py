"""circuit-check's bindings of `verity.protocols.certifier.prover.compile_slice_v4`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_v4_roots() -> list[Any]:
    """The compile slice over the memory-checked pool at a small class: three steps at (R 2, W 1, A 1) of 4-token blocks
    on three blocks (one step from the empty engine would fold to constants), one arrival of 4-bit tokens with one block
    key, one decision of advice, a stream of two slots and a touched list of six dense entries and one M entry. Each
    part once (CompileWindow, CompileBus, BlockKey_v1, PoolSlot_v1, PoolEntry_v1, PoolEnd_v1, CompileVerdict), and the
    wiring their own checks can't reach: the records' scan, the keys' prints into RK's initial values, the stream's two
    entry runs and the verdict. At 260,781 ANDs the whole is past the default whole-lowering budget; its parts are each
    lowered."""
    from verity.protocols.certifier.prover import compile_slice_v2 as V2
    from verity.protocols.certifier.prover.compile_slice_v4 import statics, unit

    engine = {**V2.DEMO_ENGINE, "max_model_len": 8, "num_gpu_blocks": 3}
    return [unit(**statics(2, 1, 1, engine, 1, Q=1, D=2, P=3, K=1, CTW=4, CU=1, DA=1, NS=2, NM=1))]
