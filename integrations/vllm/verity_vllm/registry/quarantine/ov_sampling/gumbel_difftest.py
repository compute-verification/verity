"""Differential-test adapter (gate G4, `verity_capture.difftest`) for `GumbelArgmaxBf16_v1{V, SEED, POS}` against the pinned
production operation: vLLM d9105ea80 `vllm.v1.worker.gpu.sample.gumbel.gumbel_sample` on the fp32 widening of a bf16 logits
row (APPLY_TEMPERATURE=False, IS_DRAFTING=False, USE_FP64=False) — the exact call of `Sampler.sample` for a T=1 row.

    # pod, torch venv:   python -m verity_capture.difftest produce --adapter verity_vllm.registry.quarantine.ov_sampling.gumbel_difftest --n 32 --seed 7 --out DIR
    # verity venv:       python -m verity_capture.difftest check   --adapter verity_vllm.registry.quarantine.ov_sampling.gumbel_difftest --dir DIR --out RUN/quarantine.json [--append]

Torch is imported inside `production` only.  Cases: vocabularies at, below and above one 1024-lane block and the B1 vocabulary
(151936); seeds and positions over the full int64 request range; the required edge kinds (zeros = all-equal logits, so the
served token is the argmax of the noise alone and the tie rule is load-bearing; max = ±max-finite / ±128; subnormal (FTZ);
alternating).
"""
from __future__ import annotations



from verity.ir.defs import bind

from verity_vllm.registry.quarantine.ov_sampling.gumbel import GumbelArgmaxBf16

SEMANTIC_ID = "ov-sampling/GumbelArgmaxBf16@v1"

VOCABS = (1024, 1000, 1025, 2048, 3000, 4096, 151936)
KINDS = ("random", "zeros", "max", "subnormal", "alternating")


def _seed_pos(rng, i: int) -> tuple[int, int]:
    seed = [0, 1, 12345, 2**31 - 1, 2**32, 2**62 - 1][i % 6] if i % 3 == 0 else rng.getrandbits(62)
    pos = [0, 31, 46, 131071, 2**31, 2**40][i % 6] if i % 4 == 0 else rng.randrange(0, 1 << 20)
    return seed, pos


def cases(rng, i: int) -> dict:
    V = VOCABS[-1] if i in (5, 17) else VOCABS[i % (len(VOCABS) - 1)]
    kind = KINDS[i % len(KINDS)]
    seed, pos = _seed_pos(rng, i)
    return {"statics": {"V": V, "SEED": seed, "POS": pos},
            "inputs": [{"shape": [V], "dtype": "bfloat16", "kind": kind}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    (x,) = inputs
    logits = x.reshape(1, int(statics["V"])).to(torch.float32)          # the sampler's `logits.to(float32)` of the bf16 row
    dev = logits.device
    idx = torch.zeros(1, dtype=torch.int32, device=dev)                  # expanded_idx_mapping: row 0 -> request slot 0
    temperature = torch.ones(1, dtype=torch.float32, device=dev)         # unused (APPLY_TEMPERATURE=False) but loaded by the kernel
    seeds = torch.tensor([int(statics["SEED"])], dtype=torch.int64, device=dev)
    pos = torch.tensor([int(statics["POS"])], dtype=torch.int64, device=dev)
    sampled = gumbel_sample(logits, idx, temperature, seeds, pos, apply_temperature=False, is_drafting=False, use_fp64=False)
    return [sampled.to(torch.int32)]


def spec(statics: dict):
    return bind(GumbelArgmaxBf16, V=int(statics["V"]), SEED=int(statics["SEED"]), POS=int(statics["POS"]))
