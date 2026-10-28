"""The training-step circuit the upstream certificate is measured on: `TrainStep_v1`'s own builder
(`verity.ml.definitions.registry.torch_step`) and Definitions at a scaled shape.

The measured step, `TrainStep_v1{L=30,B=8,S=512}` at smollm2-135m's shape (benchmarks/rho/data/circuits.toml), is 3.1e15
gates, and the smallest step the registry builds (L = 1, B·S = 256) is about 1e13: neither is traversed gate by gate. The
scaled step is the same body at `SCALED`'s model shape, with B·S = 16 tokens (the bmm's k16 step makes 16 the fewest a
sequence holds). The registry refuses that shape in five places, which `scaled` lifts for as long as it is entered:

* `TrainStep_v1`'s captured M (`cublas_bf16.TRAIN_M`) and attention shapes (`cublas_bf16.TRAIN_BMM_SHAPES`);
* the cuBLAS route table (`cublas_bf16.train_route`): every GEMM of the scaled step takes the unsplit route, the one the
  measured step takes for every forward and input-gradient GEMM but the LM head's (its weight gradients are split-K);
* `RmsNormTorch`'s H ≥ 128 (`torch_train._lanes`, `reduce_lanes`), torch's vectorized reduce, at `setReduceConfig`'s
  lane count for the scaled row;
* its weight gradient's column CTAs (`torch_train.column_ctas`, H ≥ 128 and M ≥ 64): the scaled columns are one CTA;
* the optimizer's and SiLU's run lengths (`ADAM_N`, `ACT_N`), to divide the scaled tensors.

Every Definition below the root is bound as `TrainStep_v1`'s body binds it; none is changed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

#: the scaled model shape: hidden 16, four query heads on two KV heads of 16 (two groups, as smollm2-135m's 9 on 3 has
#: three), MLP 48 (three SiLU calls a token row, as 1536 on 512), and the smallest vocabulary `LogSoftmaxRow_v1` takes
SCALED = {"H": 16, "NH": 4, "KVH": 2, "D": 16, "I": 48, "V": 12288}
#: the scaled step: two layers (so that a layer's input comes from a layer, not only the embedding), one sequence of 16
STEP = {"L": 2, "B": 1, "S": 16}
#: the scaled run lengths: SiLU's words a call and AdamW's elements a block
ACT_N, ADAM_N = 16, 16


@contextmanager
def scaled(config: dict[str, int] | None = None, act_n: int = ACT_N, adam_n: int = ADAM_N) -> Iterator[Any]:
    """`torch_step` with the scaled shape's refusals lifted (the module docstring), restored on exit."""
    from verity.ml.definitions.registry import cublas_bf16 as CB
    from verity.ml.definitions.registry import torch_step as TS
    from verity.ml.definitions.registry import torch_train as TT

    config = dict(SCALED if config is None else config)

    def lanes(S: Any) -> int:
        H, L = TT._count("H", S.H), TT._count("LANES", S.LANES)
        if H % 4:
            raise ValueError(f"H = {H} is not whole float4s")
        if L > TT.WARP or L & (L - 1):
            raise ValueError(f"LANES is a power of two up to {TT.WARP}, not {L}")
        return L

    def reduce_lanes(H: int, rows: int) -> int:
        d0, d1 = TT._last_pow2(H // 4), min(TT._last_pow2(rows), TT.MAX_THREADS)
        width = min(d0, TT.WARP)
        height = min(d1, TT.MAX_THREADS // width)
        return min(d0, TT.MAX_THREADS // height)

    def train_route(role: str, M: int, K: int, N: int) -> dict[str, Any]:
        if role not in CB.TRAIN_LINEAR_ROLES:
            raise ValueError(f"train_route: role {role!r}")
        return {"kernel": "nosplit"}

    def column_ctas(M: int, H: int) -> int:
        M, H = TT._count("M", M), TT._count("H", H)
        if H % 4:
            raise ValueError(f"H = {H} is not whole float4s")
        if -(-M // 4) >= 256:
            raise ValueError(f"M = {M}: the scaled step's columns are one CTA")
        return 1

    saved = [(TS, "CONFIG", TS.CONFIG), (TS, "ACT_N", TS.ACT_N), (TS, "ADAM_N", TS.ADAM_N), (CB, "TRAIN_M", CB.TRAIN_M),
             (CB, "TRAIN_BMM_SHAPES", CB.TRAIN_BMM_SHAPES), (CB, "train_route", CB.train_route), (TT, "_lanes", TT._lanes),
             (TT, "reduce_lanes", TT.reduce_lanes), (TT, "column_ctas", TT.column_ctas)]
    try:
        TS.CONFIG, TS.ACT_N, TS.ADAM_N = config, act_n, adam_n
        CB.TRAIN_M = (*CB.TRAIN_M, *range(16, 257, 16))
        CB.TRAIN_BMM_SHAPES = (*CB.TRAIN_BMM_SHAPES, *((s, config["D"]) for s in range(16, 257, 16)))
        CB.train_route, TT._lanes, TT.reduce_lanes, TT.column_ctas = train_route, lanes, reduce_lanes, column_ctas
        yield TS
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)


def step(L: int = STEP["L"], B: int = STEP["B"], S: int = STEP["S"]):
    """`TrainStep_v1{L, B, S}` at the shape in force (call inside `scaled`) as a Program: the weights and their optimizer
    state are θ (the step's registered parameters, as `references.layer` registers its layer's weights), every other
    parameter a run input."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.program import Program
    from verity.ml.definitions.registry import torch_step as TS

    names = [n for n, _ in TS.params(L, B, S)]
    theta = tuple(n for n in names if n.endswith((".w", ".master", ".exp_avg", ".exp_avg_sq")))
    return Program(bind(TS.TrainStep, L=L, B=B, S=S), registered=theta + ("cs", "n"), theta=theta)
