"""circuit-check's bindings of `verity.ml.definitions.registry.inductor`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics in Llama-3.2-1B's launch shapes: the norms at two warps a row over one chunk
    and over a partial second chunk, its head dim and decode tile for RoPE and the split head (two splits, the second partial);
    the q/k norm-RoPE at Qwen3's head dim in its combo kernel's two autotune candidates (one chunk of four columns a thread, two
    chunks of two) and in cc 12.0's winner (two chunks of eight columns over eight lanes a row)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import hopper, inductor

    one = {"N": 512, "EPS": 1e-5, "XBLOCK": 2, "R0_BLOCK": 512, "NUM_WARPS": 4}
    two = {**one, "N": 768}
    fa3 = {"D": 64, "BN": 192, "DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
    qk = {"D": 128, "EPS": 1e-6, "XBLOCK": 2, "NUM_WARPS": 2}
    return [
        ("InductorRMSNorm_v1", lambda: bind(inductor.InductorRMSNorm, **one)),
        ("InductorAddRMSNorm_v1", lambda: bind(inductor.InductorAddRMSNorm, **two)),
        ("InductorAdd2RMSNorm_v1", lambda: bind(inductor.InductorAdd2RMSNorm, **one)),
        ("InductorAdd2RMSNormFinal_v1", lambda: bind(inductor.InductorAdd2RMSNormFinal, **two)),
        ("InductorSiluMulElem_v1", lambda: inductor.InductorSiluMulElem),
        ("InductorSiluMul_v1", lambda: bind(inductor.InductorSiluMul, I=8)),
        ("InductorRoPEHead_v1", lambda: bind(inductor.InductorRoPEHead, D=64)),
        ("InductorRoPE_v1", lambda: bind(inductor.InductorRoPE, NHEADS=2, D=64)),
        ("InductorQKNormRoPEHead_v1", lambda: bind(inductor.InductorQKNormRoPEHead, R0_BLOCK=128, **qk)),
        ("InductorQKNormRoPE_v1", lambda: bind(inductor.InductorQKNormRoPE, NHEADS=2, R0_BLOCK=64, **qk)),
        ("InductorQKNormRoPEHead_v1/cc120", lambda: bind(inductor.InductorQKNormRoPEHead, **{**qk, "XBLOCK": 64, "R0_BLOCK": 64,
                                                                                              "NUM_WARPS": 16})),
        ("Fa3AttentionHeadSplit_v1", lambda: bind(inductor.Fa3AttentionHeadSplit, T=300, NS=2, MASKED_FROM=1, **fa3)),
        ("Fa3AttentionSplit_v1", lambda: bind(inductor.Fa3AttentionSplit, T=300, NS=2, NH=4, KVH=2, MASKED_FROM=1, **fa3)),
    ]
