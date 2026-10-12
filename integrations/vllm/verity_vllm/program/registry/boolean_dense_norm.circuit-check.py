"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dense_norm`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The rest of Gemma-2's norm chain on bits (over `boolean_dense`'s rows): the mean at N = 16 and at N = 1030 (two
    chunks, one element per thread), the narrowing at N = 16, and the rsqrt with its MUFU at the word id and at mufu's
    `RsqrtApprox_v2`."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import mufu as BM
    from verity_vllm.program.registry import boolean_dense_norm as DN
    from verity_vllm.program.registry import prims as P

    return [
        bind(DN.MeanTriton, N=16),
        bind(DN.MeanTriton, N=1030),
        bind(DN.NarrowF32ToBf16, N=16),
        bind(DN.RsqrtF32, N=1, RSQRT=P.RsqrtApprox),
        bind(DN.RsqrtF32, N=1, RSQRT=bind(BM.RsqrtApprox)),
    ]
