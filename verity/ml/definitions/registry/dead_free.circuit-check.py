"""circuit-check's bindings of `verity.ml.definitions.registry.dead_free`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The RMSNorm at one warp's worth of columns and at Llama-3.2-1B's hidden size (8 warps' chunks), the selects at V = 16,
    TokenSelect_v4 at four chunks, the last one short, with its chunk and its combine there, a unit of groups of 16 with its last
    group short, and the top-p pieces at the registered split count S = 32, the sampler at V = 64 (where
    `GumbelTopPTokenSelect_v2` has 6,034 dead gates), and FA2's attention with a first block of 16 keys below kBlockN 32 (where
    `Attention_v5` has 2 dead gates a head), v19's head also at a first block of 8 keys (whose V rows `AttnBlockFirst_v1` pads to
    a k-step), and `AttnBlockFirst_v2` at those two first blocks with a key's value in 4 slices of 4 columns, also at Ampere's
    k16 step (FA2's on sm_80 and sm_89)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dead_free as DF, hopper, prims as P

    fa2 = {"D": 16, "BN": 32, "MASKED_FROM": 1, "DOT": hopper.HopperBF16WgmmaDot16, "INV": P.Fa2InvSum}
    return [
        ("AttentionHead_v18", lambda: bind(DF.AttentionHeadV18, T=48, **fa2)),
        ("Attention_v18", lambda: bind(DF.AttentionV18, T=48, NH=2, KVH=1, **fa2)),
        ("AttentionHead_v19", lambda: bind(DF.AttentionHeadV19, T=48, **fa2)),
        ("AttentionHead_v19-padded", lambda: bind(DF.AttentionHeadV19, T=40, **fa2)),
        ("Attention_v19", lambda: bind(DF.AttentionV19, T=48, NH=2, KVH=1, **fa2)),
        ("AttnBlockFirst_v2", lambda: bind(DF.AttnBlockFirstV2, D=16, NVIS=16, BN=32, DOT=fa2["DOT"], VS=4)),
        ("AttnBlockFirst_v2-padded", lambda: bind(DF.AttnBlockFirstV2, D=16, NVIS=8, BN=32, DOT=fa2["DOT"], VS=4)),
        ("AttnBlockFirst_v2-ampere", lambda: bind(DF.AttnBlockFirstV2, D=16, NVIS=8, BN=32, DOT=P.AmpereBF16TcDot16, VS=4)),
        ("RMSNormTriton_v5", lambda: bind(DF.RMSNormTritonV5, N=64, EPS=1e-5)),
        ("RMSNormTriton_v5-wide", lambda: bind(DF.RMSNormTritonV5, N=2048, EPS=1e-5)),
        ("TokenSelect_v3", lambda: bind(DF.TokenSelectV3, V=16)),
        ("TokenSelect_v4", lambda: bind(DF.TokenSelectV4, V=200)),
        ("TokenSelectChunk_v2", lambda: bind(DF.TokenSelectChunkV2, C=DF.CHUNK)),
        ("TokenSelectCombine_v2", lambda: bind(DF.TokenSelectCombineV2, V=200, C=DF.CHUNK)),
        ("TokenSelectGroups_v1", lambda: bind(DF.TokenSelectGroupsV1, C=40, G=16)),
        ("TopPMask_v3", lambda: bind(DF.TopPMaskV3, V=16)),
        ("GumbelSelectF32GivenGreedy_v2", lambda: bind(DF.GumbelSelectF32GivenGreedyV2, V=16)),
        ("GumbelTopPTokenSelectSharedGreedy_v2", lambda: bind(DF.GumbelTopPTokenSelectSharedGreedyV2, V=16)),
        ("TopPKeepWord_v2", lambda: bind(DF.TopPKeepWordV2, V=16, S=32)),
        ("TopPMask_v4", lambda: bind(DF.TopPMaskV4, V=16, S=32)),
        ("GumbelSelectF32_v3", lambda: bind(DF.GumbelSelectF32V3, V=16)),
        ("GumbelTopPTokenSelect_v3", lambda: bind(DF.GumbelTopPTokenSelectV3, V=64, S=32)),
    ]
