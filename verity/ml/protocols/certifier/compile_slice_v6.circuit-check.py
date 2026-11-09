"""circuit-check's bindings of `verity.ml.protocols.certifier.compile_slice_v6`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_v6_roots() -> list[Any]:
    """Both units at CompileSlice_v5's small class (three steps at (R 2, W 1, A 1) on three blocks, one arrival of 4-bit
    tokens, one decision of advice, two stream slots, one M and one RB entry, a ragged row of one block). The compile
    slice is CompileWindow (CompileSlice_v5's load and steps, CompileIds), CallsEq over its records and CallsVerdict, the
    records held to the calls row; the pool window is CompileSlice_v5's stream, bus and ragged keys with PoolCounts, the two pool rows' digests
    (one segment each), their keys' prints, PoolOpen, PoolClose and PoolCopies over the rows and the M run, and
    PoolVerdict: the wiring their own checks can't reach is the opened row into the stream's initial values and the
    finals into the close. The pool window is past the default whole-lowering budget; its parts are each lowered."""
    from verity.ml.protocols.certifier import compile_slice_v2 as V2
    from verity.ml.protocols.certifier.compile_slice_v6 import compile_unit, pool_unit, statics

    engine = {**V2.DEMO_ENGINE, "max_model_len": 8, "num_gpu_blocks": 3}
    st = statics(2, 1, 1, engine, 1, Q=1, D=2, P=3, K=1, CTW=4, CU=1, DA=1, NS=2, NM=1, NRB=1, NRK=1)
    return [compile_unit(**st), pool_unit(**st)]


def _network_pool_chain_v1_roots() -> list[Any]:
    """Two consecutive windows at the demonstration class (`compile_slice_v6.demo`): window 1's compile slice and pool
    window, then window 2's, the output root window 1's pool window commits the input root window 2's opens, one input
    both read. Its checks reach the wiring no one window's can: the shared root."""
    from verity.ml.protocols.certifier.compile_slice_v6 import chain_unit, demo

    return [chain_unit(**demo()[0])]
