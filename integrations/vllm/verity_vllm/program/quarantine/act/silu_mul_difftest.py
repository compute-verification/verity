"""difftest adapter for `act/SiluMul@v2`: production = vLLM `torch.ops._C.silu_and_mul(out, x)` on one bf16 row [gate | up].

    verity-vllm properties-admission produce --adapter verity_vllm.program.quarantine.act.silu_mul_difftest --n 60 --seed 7 --out DIR
    verity-vllm properties-admission check   --adapter verity_vllm.program.quarantine.act.silu_mul_difftest --dir DIR --out quarantine.json

The row cycles through the generic edge kinds and, every sixth case, explicit words: gates over the adversarial edge classes
(+-0, +-inf, NaN, -88.5 / -89 / -100 around expf's overflow, subnormals, ordinary) against ups of +-0, +-1, +-inf and NaN.
I = 8 is the vec path's unit of 8 bf16 lanes; 11 leaves a scalar tail; 8192 is Llama-3.2-1B's intermediate size.
"""
from __future__ import annotations

SEMANTIC_ID = "act/SiluMul@v2"
KINDS = ("random", "zeros", "max", "subnormal", "alternating")
SHAPES = (8, 11, 8192)
GATES = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xC2B1, 0xC2B2, 0xC2C8, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x4120, 0xC120)
UPS = (0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0)


def _words(I: int, i: int) -> list[int]:
    n = len(GATES) * len(UPS)
    pairs = [divmod((i + k) % n, len(UPS)) for k in range(I)]
    pairs = [(GATES[a], UPS[b]) for a, b in pairs]
    return [g for g, _ in pairs] + [u for _, u in pairs]


def cases(rng, i: int) -> dict:
    I = SHAPES[(i // len(KINDS)) % len(SHAPES)]
    if i % 6 == 5:
        row = {"shape": [2 * I], "dtype": "bfloat16", "words": _words(I, i * I)}
        return {"statics": {"I": I}, "inputs": [row], "edge": True}
    kind = KINDS[i % len(KINDS)]
    return {"statics": {"I": I}, "inputs": [{"shape": [2 * I], "dtype": "bfloat16", "kind": kind}], "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    (x,) = inputs
    I = int(statics["I"])
    out = torch.empty((1, I), dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul(out, x.view(1, 2 * I))
    return [out.view(I)]


def spec(statics: dict):
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.quarantine.act.silu_mul import SiluMul

    return bind(SiluMul, I=int(statics["I"]))
