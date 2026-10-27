"""Differential-test adapter (`verity_vllm.properties.admission`) for the fused-MoE block's registered Definitions against vLLM
d9105ea80's `fused_experts` (TritonExperts: `fused_moe_kernel` w13 -> `_C.silu_and_mul` -> `fused_moe_kernel` w2 with the routed
weight -> `_moe_C.moe_sum`), bf16, unquantized, routing given:

    # pod, torch venv:  verity-vllm properties-admission produce --adapter verity_vllm.program.moe_difftest --n 20 --seed 17 --out DIR
    # verity venv:      verity-vllm properties-admission check   --adapter verity_vllm.program.moe_difftest --dir DIR --out RUN/moe.json

`production` runs vLLM's own entry and records what each `fused_moe_kernel` launch wrote, by wrapping the module's
`dispatch_fused_moe_kernel`: the w13 launch's output (`MoeExpertGemm{E,K=H,N=2I}` per (token, slot) pair), the w2 launch's input
(`SiluMul_v1{I}`) and output (`MoeExpertGemmW{E,K=I,N=H}`), then the block's output (`MoeSum_v1{TOPK,H}` per token).  The expert
GEMMs are the ones a config run binds on the capture's device (`frontend.targets.moe_expert_gemm_spec`: `_v1`, or `_v2{DOT}` with the
device's k16 step).  The launch configuration vLLM chose (tile sizes, sorted or naive schedule) and the device's compute capability are
recorded with the case.  H spans several BLOCK_SIZE_K tiles, so the k-chain crosses tile boundaries; the token counts take both the
naive and the sorted block schedule."""
from __future__ import annotations

import struct

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry.moe import MoeExpertGemm, MoeExpertGemmV2, MoeExpertGemmW, MoeExpertGemmWV2, MoeSum
from verity.ml.definitions.registry.prims import BF16, F32, I32

SEMANTIC_ID = "moe/MoeExpertGemm@v1"

SHAPES = ((8, 2, 256, 128, 4), (8, 2, 256, 128, 1), (16, 4, 256, 64, 6), (8, 2, 128, 256, 8), (4, 2, 512, 128, 3))   # (E, TOPK, H, I, M)
KINDS = ("random", "zeros", "max", "subnormal", "alternating")


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def cases(rng, i: int) -> dict:
    E, TOPK, H, I, M = SHAPES[i % len(SHAPES)]
    ids = [e for _ in range(M) for e in rng.sample(range(E), TOPK)]
    g = []
    for _ in range(M):
        w = [rng.random() + 1e-3 for _ in range(TOPK)]
        g += [_f32_bits(v / sum(w)) for v in w] if i % 7 else [_f32_bits(1.0)] + [_f32_bits(0.0)] * (TOPK - 1)
    kx = KINDS[i % len(KINDS)]
    kw = "random" if i % 4 else KINDS[(i // 4) % len(KINDS)]
    return {"statics": {"E": E, "TOPK": TOPK, "H": H, "I": I, "M": M},
            "inputs": [{"shape": [M * H], "dtype": "bfloat16", "kind": kx},
                       {"shape": [E * 2 * I * H], "dtype": "bfloat16", "kind": kw},
                       {"shape": [E * H * I], "dtype": "bfloat16", "kind": kw},
                       {"shape": [M * TOPK], "dtype": "int32", "kind": "topk_ids", "words": ids},
                       {"shape": [M * TOPK], "dtype": "float32", "kind": "topk_weights", "words": g}],
            "edge": kx != "random"}


def production(statics: dict, inputs: list) -> list:
    import inspect

    import torch
    import vllm.model_executor.layers.fused_moe.fused_moe as FM

    from verity_vllm.engine import hooks

    E, TOPK, H, I, M = (int(statics[k]) for k in ("E", "TOPK", "H", "I", "M"))
    x, w13, w2, ids, g = inputs
    x, w13, w2 = x.reshape(M, H), w13.reshape(E, 2 * I, H), w2.reshape(E, H, I)
    ids, g = ids.reshape(M, TOPK), g.reshape(M, TOPK)
    launches: list[dict] = []
    dispatch = FM.dispatch_fused_moe_kernel
    sig = inspect.signature(dispatch)

    def recorded(*args, **kw):
        dispatch(*args, **kw)
        a = sig.bind(*args, **kw).arguments
        launches.append({"A": a["A"].clone(), "C": a["C"].clone(), "config": dict(a["config"]), "top_k": int(a["top_k"]),
                         "mul_routed_weight": bool(a["mul_routed_weight"]), "naive": a["sorted_token_ids"] is None})

    with hooks.patched(FM, "dispatch_fused_moe_kernel", recorded):
        out = FM.fused_experts(x, w13, w2, g, ids)
    torch.cuda.synchronize()
    if len(launches) != 2 or launches[0]["mul_routed_weight"] or not launches[1]["mul_routed_weight"]:
        raise RuntimeError(f"fused_experts ran {len(launches)} fused_moe_kernel launches, not the w13 / w2 pair")
    statics["launches"] = [{k: v for k, v in L.items() if k not in ("A", "C")} for L in launches]
    statics["cc"] = list(torch.cuda.get_device_capability(x.device))
    up, down = launches
    return [up["C"].reshape(-1), down["A"].reshape(-1), down["C"].reshape(-1), out.reshape(-1)]


def _block(B, S, x, w13, w2, ids, g, up_def, down_def):
    up, act, down, out = [], [], [], []
    for t in range(S.M):
        dt = []
        for k in range(S.TOPK):
            u = B.call(up_def, x[t], ids[t][k], w13)
            a = B.call(bind(b1.SiluMul, I=S.I), u)
            d = B.call(down_def, a, ids[t][k], g[t][k], w2)
            up.append(u); act.append(a); dt.append(d)
        down += dt
        out.append(B.call(bind(MoeSum, TOPK=S.TOPK, H=S.H), array_of(dt)))
    return tuple_of(array_of(up), array_of(act), array_of(down), array_of(out))


def _difftest_body(B, S, x, w13, w2, ids, g):
    """Per (token t, slot k): the w13 launch's output (`MoeExpertGemm_v1`), `SiluMul_v1` of it (the w2 launch's input) and the w2
    launch's output (`MoeExpertGemmW_v1`); per token, `MoeSum_v1` over its slots."""
    return _block(B, S, x, w13, w2, ids, g, bind(MoeExpertGemm, E=S.E, K=S.H, N=2 * S.I), bind(MoeExpertGemmW, E=S.E, K=S.I, N=S.H))


def _difftest_body_dot(B, S, x, w13, w2, ids, g):
    """`FusedMoEDifftest_v1` with the expert GEMMs `MoeExpertGemm_v2` / `MoeExpertGemmW_v2` on the k16 step `DOT`."""
    return _block(B, S, x, w13, w2, ids, g, bind(MoeExpertGemmV2, E=S.E, K=S.H, N=2 * S.I, DOT=S.DOT),
                  bind(MoeExpertGemmWV2, E=S.E, K=S.I, N=S.H, DOT=S.DOT))

def _signature(S):
    P = S.M * S.TOPK
    return ((("x", Array(S.M, Array(S.H, BF16))), ("w13", Array(S.E, Array(2 * S.I, Array(S.H, BF16)))),
             ("w2", Array(S.E, Array(S.H, Array(S.I, BF16)))), ("ids", Array(S.M, Array(S.TOPK, I32))), ("g", Array(S.M, Array(S.TOPK, F32)))),
            Tuple(Array(P, Array(2 * S.I, BF16)), Array(P, Array(S.I, BF16)), Array(P, Array(S.H, BF16)), Array(S.M, Array(S.H, BF16))))


FusedMoEDifftest = CompositeDefinition("FusedMoEDifftest", 1, ("E", "TOPK", "H", "I", "M"), _signature, _difftest_body,
                                       doc=_difftest_body.__doc__ or "", conformance="derived: difftest", register=False)
FusedMoEDifftestDot = CompositeDefinition("FusedMoEDifftestDot", 1, ("E", "TOPK", "H", "I", "M", "DOT"), _signature, _difftest_body_dot,
                                          doc=_difftest_body_dot.__doc__ or "", conformance="derived: difftest", register=False)


def spec(statics: dict):
    """The block as a config run on the capture's device binds it (`frontend.targets.moe_expert_dot` of its compute capability,
    recorded by `production`; none recorded = the default target's `_v1` Definitions)."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.frontend import targets as T
    st = {k: int(statics[k]) for k in ("E", "TOPK", "H", "I", "M")}
    cc = statics.get("cc")
    dot = T.moe_expert_dot(TargetProfile(compute_capability=tuple(cc))) if cc else None
    return bind(FusedMoEDifftest, **st) if dot is None else bind(FusedMoEDifftestDot, **st, DOT=dot)
