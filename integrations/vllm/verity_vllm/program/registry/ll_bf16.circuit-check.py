"""circuit-check's bindings of `verity_vllm.program.registry.ll_bf16`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The word Definitions at GLM-5.3's gate (K 6144): the dotprod at BS 128, VEC 8 with and without contraction, the split-K at
    SPLIT_K 6 and 8 (TILE_K 256, WARPS 4); the rows at small statics; the dotprod's two steps."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import ll_bf16 as LL

    HOP = word_of(BG.HopperBF16WgmmaDot16)
    return [
        ("LlBf16DotprodDot_v1", lambda: bind(LL.LlBf16DotprodDotWord, K=6144, BS=128, VEC=8, FMA=1)),
        ("LlBf16DotprodDot_v1:FMA=0", lambda: bind(LL.LlBf16DotprodDotWord, K=6144, BS=128, VEC=8, FMA=0)),
        ("LlBf16SplitKDot_v1", lambda: bind(LL.LlBf16SplitKDotWord, K=6144, SPLIT_K=6, TILE_K=256, WARPS=4, DOT=HOP)),
        ("LlBf16SplitKDot_v1:SPLIT_K=8", lambda: bind(LL.LlBf16SplitKDotWord, K=6144, SPLIT_K=8, TILE_K=256, WARPS=4, DOT=HOP)),
        ("LlBf16DotprodGemm_v1", lambda: bind(LL.LlBf16DotprodGemmWord, N=2, K=256, BS=64, VEC=2, FMA=1)),
        ("LlBf16SplitKGemm_v1", lambda: bind(LL.LlBf16SplitKGemmWord, N=2, K=128, SPLIT_K=2, TILE_K=32, WARPS=2, DOT=HOP)),
        ("LlBf16FmaStep_v1", lambda: bind(LL.LlBf16FmaStepWord)),
        ("LlBf16MulAddStep_v1", lambda: bind(LL.LlBf16MulAddStepWord)),
    ]


def _boolean_roots() -> list[Any]:
    """On bits: the dotprod's two steps; the dotprod over two tiles of two warps' lanes, two elements per vector load, with and
    without contraction; the split-K over two ranks of two warps, two tiles each; and their rows at two W rows."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import ll_bf16 as LL

    HOP = BG.HopperBF16WgmmaDot16
    return [
        bind(LL.LlBf16FmaStep),
        bind(LL.LlBf16MulAddStep),
        bind(LL.LlBf16DotprodDot, K=256, BS=64, VEC=2, FMA=1),
        bind(LL.LlBf16DotprodDot, K=256, BS=64, VEC=2, FMA=0),
        bind(LL.LlBf16SplitKDot, K=128, SPLIT_K=2, TILE_K=32, WARPS=2, DOT=HOP),
        bind(LL.LlBf16DotprodGemm, N=2, K=128, BS=64, VEC=2, FMA=1),
        bind(LL.LlBf16SplitKGemm, N=2, K=128, SPLIT_K=2, TILE_K=32, WARPS=2, DOT=HOP),
    ]
