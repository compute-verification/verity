"""circuit-check's bindings of `verity.ml.definitions.registry.cublas_bf16`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The word Definitions: the padded GEMM at T = 40 of K = 48, N = 3, the three grid.z split-Ks (a bf16 workspace, an
    f32 one and none) at K = 64, N = 3, three 8-aligned slices at k-tile 16, the gate's f32-out GEMM at GLM-5.3's K (6144)
    over two W rows, the batched GEMM at two members,
    K = 32, N = 3, and both default split-K output types, the bf16-workspace and the serial split-K at K = 64, N = 3, four
    splits.  Then every route `cublas_routes.json` records for GLM-5.3's two split-K weights on one TP8 rank (``served-``), up to
    its m_max: the MoE router's gate (K = 6144, N = 256, f32 out) and the indexer's ``wk`` (K = 6144, N = 160), each at two,
    three and four splits and non-split (the indexer's is core's `Gemm_v2`); `test_cublas_served` holds them to the table.
    Last, the TF32 GEMM at the rotation codecs' (K, N) = (64, 64) (`TRAIN_TF32`); at (512, 512) one proof block is past Flock's
    proved block range, and `test_cublas_train` holds that shape to the capture's words."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import cublas_bf16 as CB

    HOP = word_of(BG.HopperBF16WgmmaDot16)
    GRID_Z = {"K": 64, "N": 3, "SPLIT_K": 3, "ALIGN": 8, "TILE_K": 16}
    return [
        ("GemmPad_v1", lambda: bind(CB.GemmPadWord, T=40, K=48, N=3, DOT=HOP)),
        ("CublasSplitKGemmBf16PartialsAligned_v1", lambda: bind(CB.CublasSplitKGemmBf16PartialsAlignedWord, **GRID_Z, DOT=HOP)),
        ("CublasSplitKGemmAligned_v1", lambda: bind(CB.CublasSplitKGemmAlignedWord, **GRID_Z, DOT=HOP)),
        ("CublasSerialSplitKGemmAligned_v1", lambda: bind(CB.CublasSerialSplitKGemmAlignedWord, **GRID_Z, DOT=HOP)),
        ("GemmF32Out_v1", lambda: bind(CB.GemmF32OutWord, K=6144, N=2, DOT=HOP)),
        ("BatchedGemm_v1", lambda: bind(CB.BatchedGemmWord, B=2, K=32, N=3, DOT=HOP)),
        ("CublasSplitKGemmF32Out_v1", lambda: bind(CB.CublasSplitKGemmF32OutWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSplitKGemm_v1", lambda: bind(CB.CublasSplitKGemmWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSplitKGemmBf16Partials_v1", lambda: bind(CB.CublasSplitKGemmBf16PartialsWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSerialSplitKGemm_v1", lambda: bind(CB.CublasSerialSplitKGemmWord, K=64, N=3, SPLIT_K=4, DOT=HOP)),
        ("CublasSplitKGemmF32Out_v1/served-router-2", lambda: bind(CB.CublasSplitKGemmF32OutWord, K=6144, N=256, SPLIT_K=2, DOT=HOP)),
        ("CublasSplitKGemmF32Out_v1/served-router-3", lambda: bind(CB.CublasSplitKGemmF32OutWord, K=6144, N=256, SPLIT_K=3, DOT=HOP)),
        ("CublasSplitKGemmF32Out_v1/served-router-4", lambda: bind(CB.CublasSplitKGemmF32OutWord, K=6144, N=256, SPLIT_K=4, DOT=HOP)),
        ("GemmF32Out_v1/served-router", lambda: bind(CB.GemmF32OutWord, K=6144, N=256, DOT=HOP)),
        ("CublasSplitKGemm_v1/served-indexer-2", lambda: bind(CB.CublasSplitKGemmWord, K=6144, N=160, SPLIT_K=2, DOT=HOP)),
        ("CublasSplitKGemm_v1/served-indexer-3", lambda: bind(CB.CublasSplitKGemmWord, K=6144, N=160, SPLIT_K=3, DOT=HOP)),
        ("CublasSplitKGemm_v1/served-indexer-4", lambda: bind(CB.CublasSplitKGemmWord, K=6144, N=160, SPLIT_K=4, DOT=HOP)),
        ("Gemm_v2/served-indexer", lambda: bind(G.Gemm, K=6144, N=160, DOT=HOP)),
        ("GemmTf32_v1/rotation-64", lambda: bind(CB.GemmTf32, K=64, N=64)),
    ]


def _boolean_roots() -> list[Any]:
    """On bits: the f32-out GEMM at K = 32, N = 3 and the batched GEMM at two members, K = 32, N = 3 (`mla.MlaBmm`'s check
    statics); both default split-K output types, the bf16-workspace and the serial split-K at K = 64, N = 3, four splits;
    the three 8-aligned grid.z split-Ks (a bf16 workspace, an f32 one and none) at K = 64, N = 3, three slices of 24, 24 and
    16 words at k-tile 16 (each 24-word slice's residue tile of 8 words and 8 zeros before its whole tile); and the padded
    GEMM at T = 40 of K = 48, N = 3.  The f32-out, batched, bf16-workspace, serial, grid.z and padded ones again on the
    training routes' step (`CB.TRAIN_DOT`, `HopperBF16WgmmaDot16_v3`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import cublas_bf16 as CB

    HOP, TRAIN = BG.HopperBF16WgmmaDot16, CB.TRAIN_DOT
    GRID_Z = {"K": 64, "N": 3, "SPLIT_K": 3, "ALIGN": 8, "TILE_K": 16}
    return [
        bind(CB.GemmF32Out, K=32, N=3, DOT=HOP),
        bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=HOP),
        bind(CB.CublasSplitKGemmF32Out, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSplitKGemmBf16Partials, K=64, N=3, SPLIT_K=4, DOT=HOP),
        bind(CB.CublasSerialSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=HOP),
        *(bind(d, **GRID_Z, DOT=HOP) for d in (CB.CublasSplitKGemmBf16PartialsAligned, CB.CublasSplitKGemmAligned,
                                                CB.CublasSerialSplitKGemmAligned)),
        bind(CB.GemmPad, T=40, K=48, N=3, DOT=HOP),
        bind(CB.GemmF32Out, K=32, N=3, DOT=TRAIN),
        bind(CB.BatchedGemm, B=2, K=32, N=3, DOT=TRAIN),
        bind(CB.CublasSplitKGemmBf16Partials, K=64, N=3, SPLIT_K=4, DOT=TRAIN),
        bind(CB.CublasSerialSplitKGemm, K=64, N=3, SPLIT_K=4, DOT=TRAIN),
        *(bind(d, **GRID_Z, DOT=TRAIN) for d in (CB.CublasSplitKGemmBf16PartialsAligned, CB.CublasSplitKGemmAligned,
                                                  CB.CublasSerialSplitKGemmAligned)),
        bind(CB.GemmPad, T=40, K=48, N=3, DOT=TRAIN),
    ]
