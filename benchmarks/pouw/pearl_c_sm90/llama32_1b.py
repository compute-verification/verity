"""Llama-3.2-1B (unsloth/Llama-3.2-1B, served as verity-fp8/Llama-3.2-1B-FP8) under Pearl-C on one H200
(``pearl-c-h100-v1-h3``): its linears as one card serves them (TP 1), each with the sm_90a kernels' domain verdict, and γ at
the H100 record at both FADD readings (`glm53.gamma`, `glm53.FADD_READINGS`).

The FP8 checkpoint is `verity-vllm fp8-checkpoint`'s verity-fp8-block128/v1 (adapters/vllm's
``manifests/checkpoints.json``): every decoder linear a 128 x 128 block-scaled weight, the format GLM-5.3's are, so it takes
the pinned vLLM's FP8 block linear (`glm53.DEEPGEMM`).  ``lm_head`` (tied to the embedding, BF16, not a LinearBase) is not a
linear Pearl-C computes.  Every layer has the same four.

Decode's M is the batch and prefill's the step's tokens (``max_num_batched_tokens``): demos/vllm serves this model at
a batch of 8 sequences and 2,048 batched tokens by default, beside GLM-5.3's M.

  python benchmarks/pouw/pearl_c_sm90/llama32_1b.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from glm53 import DEEPGEMM, FADD_READINGS, SCHEME, Linear, gamma, scheme

#: the checkpoint's config.json, the fields the linears read
CONFIG = {"H": 2048, "NH": 32, "NKV": 8, "HD": 64, "I": 8192, "L": 16}
TP = 1
DECODE_M = (1, 8, 16, 64, 256)
PREFILL_M = (2048, 8192)


def linears(c: dict = CONFIG) -> list[Linear]:
    """A layer's linears, in its order."""
    return [
        Linear("self_attn.qkv_proj", (c["NH"] + 2 * c["NKV"]) * c["HD"], c["H"], "all", DEEPGEMM, "q, k and v merged"),
        Linear("self_attn.o_proj", c["H"], c["NH"] * c["HD"], "all", DEEPGEMM),
        Linear("mlp.gate_up_proj", 2 * c["I"], c["H"], "all", DEEPGEMM, "gate and up merged"),
        Linear("mlp.down_proj", c["H"], c["I"], "all", DEEPGEMM),
    ]


def table(name: str = SCHEME) -> list[dict]:
    """Every linear (`linears`) with its domain verdict (`pouw_pearl_c_device.refusals`, the kernels' grain with the
    record's) and, in the domain, γ at each FADD reading."""
    from verity_vllm.protocols.pouw import pearl_c as PPD

    rows = []
    for i, lin in enumerate(linears()):
        bad = PPD.refusals(name, {i: (lin.n, lin.k)})
        row = {**asdict(lin), "refused": [b.split(": ", 1)[-1] for b in bad]}
        if not bad:
            row["gamma"] = {r: float(gamma(scheme(f, name), lin.n, lin.k)) for r, f in FADD_READINGS.items()}
        rows.append(row)
    return rows


def in_domain(name: str = SCHEME) -> dict[tuple[int, int], dict]:
    """Every in-domain (n, k) of `table`, with its linear's row."""
    return {(r["n"], r["k"]): r for r in table(name) if not r["refused"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    rows = table()
    if a.json:
        print(json.dumps({"scheme": SCHEME, "tp": TP, "decode_m": DECODE_M, "prefill_m": PREFILL_M, "linears": rows}, indent=1))
        return 0
    for r in rows:
        verdict = "refused: " + "; ".join(r["refused"]) if r["refused"] else \
            "γ " + ", ".join(f"{k} {100 * v:.4f}%" for k, v in r["gamma"].items())
        print(f"{r['name']:20s} n {r['n']:5d} k {r['k']:5d} {r['stock']:26s} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
