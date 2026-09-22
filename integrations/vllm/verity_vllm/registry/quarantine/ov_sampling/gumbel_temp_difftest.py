"""Differential-test adapter (gate G4) for `GumbelArgmaxTempBf16_v1{V, SEED, POS, T}`: vLLM d9105ea80 `apply_temperature`
(`_temperature_kernel`, in place `logits[r] /= T` with Triton's fp32 `/` = `div.full.f32`) followed by `gumbel_sample`
(APPLY_TEMPERATURE=False) — the exact kernel pair `Sampler.sample` launches for a row with T not in {0, 1}.

    # pod, torch venv:   python -m verity_capture.difftest produce --adapter verity_vllm.registry.quarantine.ov_sampling.gumbel_temp_difftest --n 32 --seed 11 --out DIR
    # verity venv:       python -m verity_capture.difftest check   --adapter verity_vllm.registry.quarantine.ov_sampling.gumbel_temp_difftest --dir DIR --out RUN/quarantine.json --append

Temperatures: the served values (0.7) plus a spread over (0.05, 2.5) excluding 1.0; `T` is the fp32 bit pattern (the static of
the Definition).  Edge kinds as in `gumbel_difftest` — the `max` kind at T < 1 overflows to +inf in both the kernel and the
`div.full` model (the tie among +inf lanes is then the lowest index).
"""
from __future__ import annotations


import struct

from verity.ir.defs import bind

from verity_vllm.registry.quarantine.ov_sampling.gumbel import GumbelArgmaxTempBf16
from verity_vllm.registry.quarantine.ov_sampling.gumbel_difftest import KINDS, VOCABS, _seed_pos

SEMANTIC_ID = "ov-sampling/GumbelArgmaxTempBf16@v1"

TEMPS = (0.7, 0.5, 1.3, 0.05, 2.0, 0.9, 0.15, 1.7)


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def cases(rng, i: int) -> dict:
    V = VOCABS[-1] if i in (3, 19) else VOCABS[i % (len(VOCABS) - 1)]
    kind = KINDS[i % len(KINDS)]
    seed, pos = _seed_pos(rng, i)
    T = TEMPS[i % len(TEMPS)] if i % 2 == 0 else rng.uniform(0.05, 2.5)
    if abs(T - 1.0) < 1e-3:
        T = 1.25
    return {"statics": {"V": V, "SEED": seed, "POS": pos, "T": _f32_bits(T), "T_value": struct.unpack("<f", struct.pack("<I", _f32_bits(T)))[0]},
            "inputs": [{"shape": [V], "dtype": "bfloat16", "kind": kind}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample

    (x,) = inputs
    logits = x.reshape(1, int(statics["V"])).to(torch.float32)
    dev = logits.device
    idx = torch.zeros(1, dtype=torch.int32, device=dev)
    t_value = struct.unpack("<f", struct.pack("<I", int(statics["T"])))[0]
    temperature = torch.tensor([t_value], dtype=torch.float32, device=dev)
    apply_temperature(logits, idx, temperature)                           # in place: logits[0] = div.full(logits[0], T)
    seeds = torch.tensor([int(statics["SEED"])], dtype=torch.int64, device=dev)
    pos = torch.tensor([int(statics["POS"])], dtype=torch.int64, device=dev)
    sampled = gumbel_sample(logits, idx, temperature, seeds, pos, apply_temperature=False, is_drafting=False, use_fp64=False)
    return [sampled.to(torch.int32)]


def spec(statics: dict):
    return bind(GumbelArgmaxTempBf16, V=int(statics["V"]), SEED=int(statics["SEED"]), POS=int(statics["POS"]), T=int(statics["T"]))
