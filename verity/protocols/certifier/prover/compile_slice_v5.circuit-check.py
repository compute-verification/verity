"""circuit-check's bindings of `verity.protocols.certifier.prover.compile_slice_v5`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_v5_roots() -> list[Any]:
    """The compile slice at its window's activity at a small class: three steps at (R 2, W 1, A 1) of 4-token blocks on
    three blocks, one arrival of 4-bit tokens, one decision of advice, a stream of two slots and a touched list of four
    dense entries, one M and one RB entry, and a ragged row of one block. Each part once (CompileLoad, CompileStep,
    CompileBus, CompileArrivals, RaggedKey, PoolSlot_v1, PoolEntry_v1, PoolEnd_v1, CompileVerdict), and the wiring their
    own checks can't reach: the steps' scan and its records' columns, the arrivals' and the ragged row's scans into the
    keys bit, the stream's three entry runs and the verdict. At 269,528 ANDs the whole is past the default
    whole-lowering budget; its parts are each lowered."""
    from verity.protocols.certifier.prover import compile_slice_v2 as V2
    from verity.protocols.certifier.prover.compile_slice_v5 import statics, unit

    engine = {**V2.DEMO_ENGINE, "max_model_len": 8, "num_gpu_blocks": 3}
    return [unit(**statics(2, 1, 1, engine, 1, Q=1, D=2, P=3, K=1, CTW=4, CU=1, DA=1, NS=2, NM=1, NRB=1, NRK=1))]
