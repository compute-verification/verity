"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dense`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """`registry.dense`'s rows on bits at 8 elements with Gemma-2's constants: `gelu_tanh_and_mul` at I = 8, the RMSNorm
    chain's and the logit softcap's element-wise rows at N = 8."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dense as BDN

    return [
        bind(BDN.GeluTanhMul, I=8),
        *[bind(d, N=8) for d in (BDN.SquareBf16, BDN.SquareF32, BDN.AddWidenedBf16, BDN.ScaleRowBf16, BDN.ScaleRowF32,
                                 BDN.MulVecF32, BDN.Bf16MulScalarTensor)],
        bind(BDN.AddScalarF32, N=8, C=1e-6),
        bind(BDN.AddScalarBf16, N=8, C=1.0),
        bind(BDN.Bf16DivScalar, N=8, C=30.0),
        bind(BDN.Bf16MulScalar, N=8, C=30.0),
    ]
