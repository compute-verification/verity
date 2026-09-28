"""[R15 stoch] Differential-test adapter (gate G4, `verity_vllm.properties.admission`) for `GumbelTopPTokenSelect_v1{V,S}(logits, top_p, seed,
pos, temp)` — temperature + top-p (split pipeline) + seeded Gumbel select — against the pinned production chain of vLLM d9105ea80's
`Sampler.sample` for a request with top_p < 1 and no top-k / penalties / logit bias / min_p:

    logits32 = logits_bf16.to(float32); apply_temperature(logits32, idx, temperature)      # `_temperature_kernel` (skipped at T in {0, 1})
    apply_top_k_top_p(logits32, k=None, p=top_p)                                            # `_apply_topp_split` (batch 1 -> S = splits_for(1, num_sm))
    gumbel_sample(logits32, idx, temperature, seeds, pos, apply_temperature=False, ...)      # `_gumbel_sample_kernel` + argmax

    # pod, torch venv:  VERITY_REAL_LOGITS=<rows.npy> verity-vllm properties-admission produce --adapter verity_vllm.program.registry.sampling_topp_difftest --n 40 --seed 13 --out DIR
    # verity venv:      verity-vllm properties-admission check   --adapter verity_vllm.program.registry.sampling_topp_difftest --dir DIR --out RUN/sampling_topp.json

The production side ALSO emits the masked fp32 row (output 1) so the intermediate `TopPMask_v1{V,S}` value is checked lane by lane,
not only the sampled token.  S is a static of the specialization = the split count the production launch used (recorded per case).
Cases: real rows when `VERITY_REAL_LOGITS` names a `[N, V]` bf16-word .npy (Llama-3.2-1B lm_head outputs, the B1 stoch manifest
prompt), interleaved with edge rows (zeros / max / subnormal / alternating / few-finite / many-duplicates); temperatures = the panel's
0.8 and a spread (never 0: top-p at temperature 0 is greedy — a different family); top_p = the panel's 0.95, 1.0 (the p >= 1 early
return), 0.5, 0.99 and a spread; seeds over the int64 range; positions over the prompt / decode range."""
from __future__ import annotations

import os
import struct

import numpy as np

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity_vllm.program.registry.prims import BF16, F32, I32
from verity_vllm.program.registry.b1 import const
from verity_vllm.program.registry.sampling import I64, GumbelSelectF32, GumbelTopPTokenSelect, TemperatureScale, TopPMask
from verity_vllm.program.registry.topp_geometry import splits_for
from verity.ml.circuits.types import Array, Tuple

SEMANTIC_ID = "sampling/GumbelTopPTokenSelect@v1"

VOCABS = (128256, 1024, 1025, 3000)
KINDS = ("random", "zeros", "max", "subnormal", "alternating")
TEMPS = (0.8, 0.7, 1.0, 0.8, 1.3, 0.3, 2.0)
TOPPS = (0.95, 1.0, 0.5, 0.99, 0.95, 0.8, 0.9999)
SEEDS = (1077402498, 0, 1, 12345, 2**31 - 1, 2**32, 2**62 - 1, 42, 7)
POSS = (255, 0, 31, 256, 287, 1023, 1151, 2**31)


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _real_rows() -> np.ndarray | None:
    p = os.environ.get("VERITY_REAL_LOGITS")
    if not p or not os.path.exists(p):
        return None
    rows = np.load(p)
    return rows.view(np.uint16) if rows.dtype != np.uint16 else rows


def cases(rng, i: int) -> dict:
    real = _real_rows()
    T = TEMPS[i % len(TEMPS)] if i % 2 == 0 else rng.uniform(0.2, 2.0)
    top_p = TOPPS[i % len(TOPPS)] if i % 3 != 1 else rng.uniform(0.3, 0.999)
    seed = SEEDS[i % len(SEEDS)] if i % 3 == 0 else rng.getrandbits(62)
    pos = POSS[i % len(POSS)] if i % 4 == 0 else rng.randrange(0, 1 << 20)
    if real is not None and i % 4 != 3:                      # three of four cases on real rows when available
        r = i % real.shape[0]
        V = int(real.shape[1])
        logits = {"shape": [V], "dtype": "bfloat16", "kind": "real", "npy": os.environ["VERITY_REAL_LOGITS"], "row": r}
        edge = False
    else:
        V = VOCABS[0] if i % 8 in (3, 7) else VOCABS[1 + (i % (len(VOCABS) - 1))]
        kind = KINDS[(i // 4) % len(KINDS)]
        logits = {"shape": [V], "dtype": "bfloat16", "kind": kind}
        edge = kind != "random"
    tb, pb = _f32_bits(T), _f32_bits(top_p)
    return {"statics": {"V": V, "S": None, "seed": int(seed), "pos": int(pos), "temp_bits": tb, "top_p_bits": pb,
                        "temp": struct.unpack("<f", struct.pack("<I", tb))[0], "top_p": struct.unpack("<f", struct.pack("<I", pb))[0]},
            "inputs": [logits,
                       {"shape": [1], "dtype": "float32", "kind": "top_p", "words": [pb]},
                       {"shape": [1], "dtype": "int64", "kind": "seed", "words": [int(seed)]},
                       {"shape": [1], "dtype": "int64", "kind": "pos", "words": [int(pos)]},
                       {"shape": [1], "dtype": "float32", "kind": "temp", "words": [tb]}],
            "edge": edge}


def production(statics: dict, inputs: list) -> list:
    import torch
    from vllm.utils.platform_utils import num_compute_units
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample

    x, top_p, seed, pos, temp = inputs
    V = int(statics["V"])
    statics["S"] = splits_for(1, num_compute_units(x.device.index))     # the launch geometry the spec must be bound with
    logits = torch.empty((1, V), dtype=torch.float32, device=x.device).copy_(x.reshape(1, V))   # the sampler's fp32 copy
    idx = torch.zeros(1, dtype=torch.int32, device=x.device)
    temperature = temp.reshape(1).to(torch.float32)
    apply_temperature(logits, idx, temperature)
    logits = apply_top_k_top_p(logits, None, top_p.reshape(1).to(torch.float32))               # in place (split pipeline at batch 1)
    masked = logits.clone()
    sampled = gumbel_sample(logits, idx, temperature, seed.reshape(1).to(torch.int64), pos.reshape(1).to(torch.int64),
                            apply_temperature=False, is_drafting=False, use_fp64=False)
    return [sampled.to(torch.int32), masked.reshape(V)]


def _difftest_body(B, S, logits, top_p, seed, pos, temp):
    """`GumbelTopPTokenSelect_v1{V}` returning ALSO its intermediate masked fp32 row (production output 1), so the difftest checks
    `TopPMask_v1{V}` lane by lane and not only the sampled token.  Same body as the registered composite; [R16 stoch] S is an OPERAND
    of the registered forms (`splits`, i32) -- here the launch geometry `production` recorded (`splits_for(1, num_SMs)` of the device)
    enters as a `Const32[S]`, the same way the fold carries the split kernels' constexpr."""
    scaled = B.call(bind(TemperatureScale, V=S.V), logits, temp)
    masked = B.call(bind(TopPMask, V=S.V), scaled, top_p, B.call(const(32, int(S.S))))
    tok = B.call(bind(GumbelSelectF32, V=S.V), masked, seed, pos, temp)
    return tuple_of(tok, masked)


GumbelTopPTokenSelectDifftest = CompositeDefinition(
    "GumbelTopPTokenSelectDifftest", 1, ("V", "S"),
    lambda S: ((("logits", Array(S.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32)), Tuple(I32, Array(S.V, F32))),
    _difftest_body, doc=_difftest_body.__doc__ or "", conformance="derived: difftest", register=False)


def spec(statics: dict):
    return bind(GumbelTopPTokenSelectDifftest, V=int(statics["V"]), S=int(statics["S"]))


def spec_registered(statics: dict):
    """The registered family (token only) — what the Program uses; S is its 6th operand (`splits`, i32), not a static."""
    return bind(GumbelTopPTokenSelect, V=int(statics["V"]))
