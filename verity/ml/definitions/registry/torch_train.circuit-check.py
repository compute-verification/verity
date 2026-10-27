"""circuit-check's bindings of `verity.ml.definitions.registry.torch_train`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The 128 x 128 transpose, RMSNorm's forward (RSQRT `rsqrt.approx.f32`'s) and backward at H = 128 (the smallest vectorized
    row), its weight's gradient at 64 tokens (the fewest it builds), SiLU forward at N = 8 and backward at N = 2 under both
    contractions (at N = 8 the two are past `targets.MAX_GATES`, so `select` would keep one), each with sm_120's MUFU.EX2 (the
    composites reach their In and Out parts)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import mufu as BMU
    from verity.ml.definitions.registry import torch_train as TT

    ex2 = bind(BMU.MufuEx2Ftz)
    eps = 0x3727C5AC
    return [
        bind(TT.Transpose, R=128, C=128),
        bind(TT.RmsNormTorch, H=128, LANES=32, EPS=eps, RSQRT=bind(BMU.RsqrtApprox)),
        bind(TT.RmsNormTorchBackward, H=128, LANES=32),
        bind(TT.RmsNormTorchWeightGrad, M=64, H=576),
        bind(TT.SiluTorch, N=8, EX2=ex2),
        bind(TT.SiluTorchBackward, N=2, EX2=ex2, CONTRACT=1),
        bind(TT.SiluTorchBackward, N=2, EX2=ex2, CONTRACT=0),
    ]
