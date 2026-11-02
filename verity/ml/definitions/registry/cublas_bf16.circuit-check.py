"""circuit-check's bindings of `verity.ml.definitions.registry.cublas_bf16`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The word Definitions: the gate's f32-out GEMM at GLM-5.3's K (6144) over two W rows, the batched GEMM at two members,
    K = 32, N = 3, and both default split-K output types, the bf16-workspace and the serial split-K at K = 64, N = 3, four
    splits."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import cublas_bf16 as CB

    HOP = word_of(BG.HopperBF16WgmmaDot16)
    return [
        ("GemmF32Out_v1", lambda: bind(CB.GemmF32OutWord, K=6144, N=2, DOT=HOP)),
        ("BatchedGemm_v1", lambda: bind(CB.BatchedGemmWord, B=2, K=32, N=3, DOT=HOP)),
        ("CublasSplitKGemmF32Out_v1", lambda: bind(CB.CublasSplitKGemmF32OutWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSplitKGemm_v1", lambda: bind(CB.CublasSplitKGemmWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSplitKGemmBf16Partials_v1", lambda: bind(CB.CublasSplitKGemmBf16PartialsWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSerialSplitKGemm_v1", lambda: bind(CB.CublasSerialSplitKGemmWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
    ]


def _boolean_roots() -> list[Any]:
    """On bits: the f32-out GEMM at K = 32, N = 3 and the batched GEMM at two members, K = 32, N = 3 (`mla.MlaBmm`'s check
    statics); both default split-K output types, the bf16-workspace and the serial split-K at K = 64, N = 3, four splits; and
    the 8-aligned grid.z split-K at K = 64, N = 3, three slices of 24, 24 and 16 words at k-tile 16 (each 24-word slice's
    residue tile of 8 words and 8 zeros before its whole tile).  The f32-out, batched, bf16-workspace, serial and aligned ones
    again on the training routes' step (`CB.TRAIN_DOT`, `HopperBF16WgmmaDot16_v3`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import cublas_bf16 as CB

    HOP, TRAIN = BG.HopperBF16WgmmaDot16, CB.TRAIN_DOT
    return [
        bind(CB.GemmF32Out, K=32, N=3, DOT=HOP),
        bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP),
        bind(CB.CublasSplitKGemmF32Out, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSplitKGemmBf16Partials, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSerialSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSplitKGemmBf16PartialsAligned, K=64, N=3, SPLIT_K=3, ALIGN=8, TILE_K=16, DOT=HOP),
        bind(CB.GemmF32Out, K=32, N=3, DOT=TRAIN),
        bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=TRAIN),
        bind(CB.CublasSplitKGemmBf16Partials, K=64, N=3, SPLIT_K=4, DOT=TRAIN),
        bind(CB.CublasSerialSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=TRAIN),
        bind(CB.CublasSplitKGemmBf16PartialsAligned, K=64, N=3, SPLIT_K=3, ALIGN=8, TILE_K=16, DOT=TRAIN),
    ]
