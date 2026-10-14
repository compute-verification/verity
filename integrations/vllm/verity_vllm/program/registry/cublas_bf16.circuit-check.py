"""circuit-check's bindings of `verity_vllm.program.registry.cublas_bf16`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The word Definitions: the gate's f32-out GEMM at GLM-5.3's K (6144) over two W rows, the batched GEMM at two members,
    K = 32, N = 3, and both default split-K output types at K = 64, N = 3, four splits."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import cublas_bf16 as CB

    HOP = word_of(BG.HopperBF16WgmmaDot16)
    return [
        ("GemmF32Out_v1", lambda: bind(CB.GemmF32OutWord, K=6144, N=2, DOT=HOP)),
        ("BatchedGemm_v1", lambda: bind(CB.BatchedGemmWord, B=2, K=32, N=3, DOT=HOP)),
        ("CublasSplitKGemmF32Out_v1", lambda: bind(CB.CublasSplitKGemmF32OutWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSplitKGemm_v1", lambda: bind(CB.CublasSplitKGemmWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
    ]


def _boolean_roots() -> list[Any]:
    """On bits: the f32-out GEMM at K = 32, N = 3 and the batched GEMM at two members, K = 32, N = 3 (`mla.MlaBmm`'s check
    statics)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import cublas_bf16 as CB

    HOP = BG.HopperBF16WgmmaDot16
    return [
        bind(CB.GemmF32Out, K=32, N=3, DOT=HOP),
        bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP),
        bind(CB.CublasSplitKGemmF32Out, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
    ]
