"""circuit-check's bindings of `verity.definitions.registry.mtp_draft`'s Definitions, and of the draft step's other kernels at the
statics GLM-5.3's MTP draft runs them at, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`MtpEhNorm_v1` at GLM-5.3's draft layer (hidden 6144, its 8192 block on 4 warps, rms_norm_eps 1e-5), held to the block by
    its cut (`mtp_draft.eh_cut`), and the cut's own units at those statics: the position select and each half's scale
    (`dsa_norm.DsaRMSNormApply_v1{G=64}`, the apply, is `dsa_norm`'s binding). Then `eh_proj`, the replicated bf16 linear from
    12288 to 6144 that the H200 runs on cuBLAS: core's `Gemm_v2` where cuBLAS does not split K, and `cublas_bf16`'s
    `CublasSplitKGemm_v1` at the one split count its route table takes up to 8192 tokens, 2
    (`glm53_mtp_eh_proj_capture_gpu`). A whole row of either is past Flock's proved block range, and a served row is cut along
    N into slices, each the same Definition at N = 1 on x and one row of W; so each is bound at its slice too."""
    from verity.circuits.boolean.trace import word_of
    from verity.circuits.defs import bind
    from verity.circuits.ops import gemm as G
    from verity.circuits.ops.boolean import gemm as BG
    from verity.definitions.registry import cublas_bf16 as CB
    from verity.definitions.registry import mtp_draft as MD

    eh = {"BLOCK": 8192, "WARPS": 4, "EPS": 1e-5}
    proj = {"K": 12288, "N": 6144, "DOT": word_of(BG.HopperBF16WgmmaDot16)}
    return [
        ("MtpEhNorm_v1/glm53", lambda: bind(MD.MtpEhNorm, H=6144, **eh)),
        ("MtpEhZero_v1/glm53", lambda: bind(MD.MtpEhZero, H=6144)),
        ("TritonBlockRMSNormSumSq_v1/glm53-mtp-eh", lambda: bind(MD.TritonBlockRMSNormSumSq, N=6144, **eh)),
        ("Gemm_v2/glm53-mtp-eh-proj", lambda: bind(G.Gemm, **proj)),
        ("CublasSplitKGemm_v1/glm53-mtp-eh-proj-2", lambda: bind(CB.CublasSplitKGemmWord, SPLIT_K=2, **proj)),
        ("Gemm_v2/glm53-mtp-eh-proj-slice", lambda: bind(G.Gemm, **{**proj, "N": 1})),
        ("CublasSplitKGemm_v1/glm53-mtp-eh-proj-2-slice", lambda: bind(CB.CublasSplitKGemmWord, SPLIT_K=2, **{**proj, "N": 1})),
    ]
