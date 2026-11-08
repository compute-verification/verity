"""circuit-check's bindings of `verity.ml.protocols.certifier.compile_slice_v5`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_compile_slice_v5_roots() -> list[Any]:
    """Both units at CompileSlice_v4's small class (three steps at (R 2, W 1, A 1) on three blocks, one arrival with one
    block key, one decision of advice, two stream slots, one M entry). The compile slice is CompileWindow and
    CallsVerdict, the records held to the calls row; the pool window adds to CompileSlice_v4's stream the two pool rows'
    digests (one segment each), their keys' prints, PoolOpen, PoolClose and PoolCopies over the rows and the M run, and
    PoolVerdict: the wiring their own checks can't reach is the opened row into the stream's initial values and the
    finals into the close."""
    from verity.ml.protocols.certifier import compile_slice_v2 as V2
    from verity.ml.protocols.certifier.compile_slice_v5 import compile_unit, pool_unit, statics

    engine = {**V2.DEMO_ENGINE, "max_model_len": 8, "num_gpu_blocks": 3}
    st = statics(2, 1, 1, engine, 1, Q=1, D=2, P=3, K=1, CTW=4, CU=1, DA=1, NS=2, NM=1)
    return [compile_unit(**st), pool_unit(**st)]
