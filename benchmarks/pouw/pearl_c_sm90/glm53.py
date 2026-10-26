"""GLM-5.3 (zai-org/GLM-5.3 at `glm53_scales.REVISION`) under Pearl-C on the H200 (``pearl-c-h100-v1-h3``): its linears as one
TP8 rank serves them, each with the sm_90a kernels' domain verdict, and γ at the H100 record.

The linears are the pinned vLLM's (integrations/vllm's ``manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json``: each tap's
output shape and program), from the checkpoint's config (`CONFIG`).  Every FP8 linear is a 128 x 128 block-scaled weight, served
by FlashInfer below 32 tokens and DeepGEMM ``fp8_gemm_nt`` from 32 (ue8m0 activation scales, weight scales requantized to ue8m0 at
load); the routed experts are vLLM's fused-MoE Triton kernel over (expert, row block); the indexer's ``wk_weights_proj`` is a
BF16 cuBLAS GEMM.  The router (``mlp.gate``, an FP32 GEMM over E outputs) and ``lm_head`` (BF16, not a LinearBase) are not
linears Pearl-C computes.

At decode a rank sees the whole batch (TP shards weights, not rows): M is the batch, and an expert's M about M·K/E (K of E
experts a row).  At prefill M is the step's tokens (``max_num_batched_tokens``).

γ (`gamma`) is one unit's: 1 − (1 − ρ)²·credit/W_ref, rev1's (`PearlC.credit_of`, `PearlC.wref`; Lean's `creditDevRev1` and
`wrefDevRev1K` with the cast offset c per activation element), which every Lean price twin's γ is
(`Proofs/Pouw/PearlC/DeviceSm120KernelGamma.lean`: v1 under rev1, 8192³, FADD 8, the honest cast at 32, 173561/22302500).
Every term is linear in m, so γ is a function of (n, k).  No Lean twin states γ at the H100 record on GLM-5.3's shapes;
these are its values, not pins.  The FP32 add's price is read two ways (`FADD_READINGS`): the record's 32 (H100_PRICES:
4,096 dense FP8 MACs to 128 FP32 lanes a SM a clock), and the data sheet's rates, 1,979 dense FP8 TFLOPS to 67 FP32
TFLOPS (an FMA two flops), 1979/67, the two clocked differently.

  python benchmarks/pouw/pearl_c_sm90/glm53.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from fractions import Fraction

SCHEME = "pearl-c-h100-v1-h3"
#: the checkpoint's config.json, the fields the linears read
CONFIG = {"H": 6144, "NH": 64, "QL": 2048, "KVL": 512, "DN": 192, "DR": 64, "DV": 256, "IH": 32, "ID": 128, "E": 256, "K": 8,
          "MI": 2048, "NS": 1, "I": 12288, "L": 78, "FD": 3}
TP = 8
#: the FP32 add's W1 price, in dense FP8 MACs: the H100 record's, and the data sheet's rates
FADD_READINGS = {"record": Fraction(32), "sheet": Fraction(1979, 67)}
#: decode's batch and prefill's step tokens, the M a rank's dense linears see
DECODE_M = (1, 16, 64, 256)
PREFILL_M = (2048, 8192)
DEEPGEMM = "fp8-block-linear-deepgemm"


@dataclass(frozen=True)
class Linear:
    name: str
    n: int
    k: int
    layers: str      # which layers have it: all, dense (the first FD), moe, indexer
    stock: str       # the pinned vLLM's program for it
    note: str = ""


def linears(c: dict = CONFIG, tp: int = TP, ep: bool = False) -> list[Linear]:
    """One rank's linears, in the layer's order.  With `ep` (vLLM's ``--enable-expert-parallel`` at DP 1, so EP = TP), a rank
    holds E / tp whole routed experts rather than a 1 / tp slice of each, and the shared expert stays TP-sharded (its MLP is
    sequence-parallel only at DP > 1).  The capture map's MoE backend (Triton) is the one without expert parallelism; no
    capture of record says which one vLLM picks under it."""
    nhl = c["NH"] // tp
    mi, moe = (c["MI"], "fp8-moe-ep") if ep else (c["MI"] // tp, "fp8-moe-triton")
    per = f"per expert, {c['E'] // tp} of them on the rank" if ep else "per expert, E of them"
    return [
        Linear("self_attn.fused_qkv_a_proj", c["QL"] + c["KVL"] + c["DR"], c["H"], "all", DEEPGEMM, "replicated (q_a, kv_a)"),
        Linear("self_attn.q_b_proj", nhl * (c["DN"] + c["DR"]), c["QL"], "all", DEEPGEMM),
        Linear("self_attn.kv_b_proj", nhl * (c["DN"] + c["DV"]), c["KVL"], "all", DEEPGEMM,
               "prefill's dense-MHA route only; decode runs it as W_UK_T and W_UV, BF16 bmms"),
        Linear("self_attn.o_proj", c["H"], nhl * c["DV"], "all", DEEPGEMM),
        Linear("self_attn.indexer.wq_b", c["IH"] * c["ID"], c["QL"], "indexer", DEEPGEMM, "replicated"),
        Linear("self_attn.indexer.wk_weights_proj", c["ID"] + c["IH"], c["H"], "indexer", "cublas-bf16-gemm",
               "replicated; BF16 (weights_proj is unquantized)"),
        Linear("mlp.gate_up_proj", 2 * c["I"] // tp, c["H"], "dense", DEEPGEMM),
        Linear("mlp.down_proj", c["H"], c["I"] // tp, "dense", DEEPGEMM),
        Linear("mlp.shared_experts.gate_up_proj", 2 * c["NS"] * c["MI"] // tp, c["H"], "moe", DEEPGEMM),
        Linear("mlp.shared_experts.down_proj", c["H"], c["NS"] * c["MI"] // tp, "moe", DEEPGEMM),
        Linear("mlp.experts.w13", 2 * mi, c["H"], "moe", moe, per),
        Linear("mlp.experts.w2", c["H"], mi, "moe", moe, per),
    ]


def expert_rows(m: int, c: dict = CONFIG) -> Fraction:
    """An expert's rows on average when M rows each go to K of E experts."""
    return Fraction(m * c["K"], c["E"])


def scheme(fadd: Fraction | None = None, name: str = SCHEME):
    """The registered record's scheme, with the FP32 add's price at `fadd` when given."""
    from verity.ml.protocols.pouw.schemes import pearl_c as C
    from verity.ml.protocols.pouw.schemes.pearl_c_device import DEVICES
    from verity.core.protocols.pouw.serving import SchemeId

    sid = SchemeId.parse(name)
    dev = DEVICES[sid.device]
    prices = dev.prices if fadd is None else replace(dev.prices, fadd=fadd)
    return C.PearlC(forming=sid.forming, device=dev, hashing=sid.hashing, prices=prices)


def gamma(s, n: int, k: int, cast: Fraction = Fraction(0)) -> Fraction:
    """One unit's γ at scheme `s` (module docstring), W_ref with `cast` more per activation element (the kernel's cast over
    the credited one)."""
    from verity.core.protocols.pouw.protocol import Shape

    sh = Shape(16, k, n)
    rho = s.device.cap
    return 1 - (1 - rho) ** 2 * Fraction(s.credit_of(sh)) / (s.wref(sh) + cast * sh.m * sh.k)


def table(name: str = SCHEME, ep: bool = False) -> list[dict]:
    """Every linear (`linears`) with its domain verdict (`pouw_pearl_c_device.refusals`, the kernels' grain with the
    record's) and, in the domain, γ at each FADD reading."""
    from verity_vllm.protocols.pouw import pearl_c as PPD

    rows = []
    for i, lin in enumerate(linears(ep=ep)):
        bad = PPD.refusals(name, {i: (lin.n, lin.k)})
        row = {**asdict(lin), "refused": [b.split(": ", 1)[-1] for b in bad]}
        if not bad:
            row["gamma"] = {r: float(gamma(scheme(f, name), lin.n, lin.k)) for r, f in FADD_READINGS.items()}
        rows.append(row)
    return rows


def in_domain(name: str = SCHEME) -> dict[tuple[int, int], dict]:
    """Every in-domain (n, k) of `table` with the routed experts sharded and expert-parallel: the row of its first linear,
    the sharded table's before the expert-parallel one's."""
    out: dict[tuple[int, int], dict] = {}
    for ep in (False, True):
        for r in table(name, ep):
            if not r["refused"]:
                out.setdefault((r["n"], r["k"]), {**r, "ep": ep})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json", action="store_true")
    p.add_argument("--ep", action="store_true", help="the routed experts expert-parallel (`linears`)")
    a = p.parse_args(argv)
    rows = table(ep=a.ep)
    if a.json:
        print(json.dumps({"scheme": SCHEME, "tp": TP, "ep": a.ep, "decode_m": DECODE_M, "prefill_m": PREFILL_M, "linears": rows},
                         indent=1))
        return 0
    for r in rows:
        verdict = "refused: " + "; ".join(r["refused"]) if r["refused"] else \
            "γ " + ", ".join(f"{k} {100 * v:.4f}%" for k, v in r["gamma"].items())
        print(f"{r['name']:38s} n {r['n']:5d} k {r['k']:5d} {r['layers']:8s} {r['stock']:26s} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
