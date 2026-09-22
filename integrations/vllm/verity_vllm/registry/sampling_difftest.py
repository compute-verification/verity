"""[X-06] Differential-test adapter (gate G4, `verity_capture.difftest`) for `GumbelTokenSelect_v1{V}(logits, seed, pos, temp)` — the
stochastic token select with seed / pos / temperature as OPERANDS — against the pinned production chain of vLLM d9105ea80's
`Sampler.sample` for a request without top-k / top-p:

    logits32 = logits_bf16.to(float32); apply_temperature(logits32, idx, temperature)      # `_temperature_kernel`, skipped at T in {0, 1}
    gumbel_sample(logits32, idx, temperature, seeds, pos, apply_temperature=False, is_drafting=False, use_fp64=False)

    # pod, torch venv:  VERITY_REAL_LOGITS=<rows.npy> python -m verity_capture.difftest produce --adapter verity_vllm.registry.sampling_difftest --n 48 --seed 13 --out DIR
    # verity venv:      python -m verity_capture.difftest check   --adapter verity_vllm.registry.sampling_difftest --dir DIR --out RUN/sampling_x06.json

Cases: REAL logits rows when `VERITY_REAL_LOGITS` names a `[N, V]` bf16-word (uint16) .npy (e.g. Llama-3.2-1B lm_head outputs at
several positions of a real prompt, kind "real"), interleaved with the required edge kinds (zeros / max / subnormal / alternating)
and random rows at V = 128256 (Llama-3.2) and small vocabularies; temperatures = the pinned row's 0.8, 0.7, 1.0 and 0.0 (the kernel's
early returns), and a spread; seeds over the int64 request range; positions over the prompt / decode range.  Every operand is a
`words` input (bound), nothing about the sampler is a static but V."""
from __future__ import annotations

import os
import struct

import numpy as np

from verity.ir.defs import bind
from verity_vllm.registry.sampling import GumbelTokenSelect

SEMANTIC_ID = "sampling/GumbelTokenSelect@v1"

VOCABS = (128256, 1024, 1000, 1025, 3000, 151936)
KINDS = ("random", "zeros", "max", "subnormal", "alternating")
TEMPS = (0.8, 0.7, 1.0, 0.0, 0.8, 1.3, 0.05, 2.0)
SEEDS = (0, 1, 12345, 2**31 - 1, 2**32, 2**62 - 1, 42, 7)
POSS = (0, 31, 255, 256, 287, 1023, 1151, 2**31)


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
    T = TEMPS[(i // 2) % len(TEMPS)] if i % 2 == 0 else rng.uniform(0.05, 2.5)
    seed = SEEDS[i % len(SEEDS)] if i % 3 == 0 else rng.getrandbits(62)
    pos = POSS[i % len(POSS)] if i % 4 == 0 else rng.randrange(0, 1 << 20)
    if real is not None and i % 3 != 2:                      # two of three cases on real rows when available
        r = i % real.shape[0]
        V = int(real.shape[1])
        logits = {"shape": [V], "dtype": "bfloat16", "kind": "real", "npy": os.environ["VERITY_REAL_LOGITS"], "row": r}
        edge = False
    else:
        V = VOCABS[0] if i % 6 in (0, 3) else VOCABS[1 + (i % (len(VOCABS) - 1))]
        kind = KINDS[i % len(KINDS)]
        logits = {"shape": [V], "dtype": "bfloat16", "kind": kind}
        edge = kind != "random"
    tb = _f32_bits(T)
    return {"statics": {"V": V, "seed": int(seed), "pos": int(pos), "temp_bits": tb, "temp": struct.unpack("<f", struct.pack("<I", tb))[0]},
            "inputs": [logits,
                       {"shape": [1], "dtype": "int64", "kind": "seed", "words": [int(seed)]},
                       {"shape": [1], "dtype": "int64", "kind": "pos", "words": [int(pos)]},
                       {"shape": [1], "dtype": "float32", "kind": "temp", "words": [tb]}],
            "edge": edge}


def production(statics: dict, inputs: list) -> list:
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample

    x, seed, pos, temp = inputs
    V = int(statics["V"])
    logits = torch.empty((1, V), dtype=torch.float32, device=x.device).copy_(x.reshape(1, V))   # the sampler's fp32 copy
    idx = torch.zeros(1, dtype=torch.int32, device=x.device)                                    # expanded_idx_mapping: row 0 -> slot 0
    temperature = temp.reshape(1).to(torch.float32)
    apply_temperature(logits, idx, temperature)                                                  # in place; early return at T in {0, 1}
    sampled = gumbel_sample(logits, idx, temperature, seed.reshape(1).to(torch.int64), pos.reshape(1).to(torch.int64),
                            apply_temperature=False, is_drafting=False, use_fp64=False)
    return [sampled.to(torch.int32)]


def spec(statics: dict):
    return bind(GumbelTokenSelect, V=int(statics["V"]))
