"""The arithmetic behind the fast-decode routes' ceilings and their effect on P2's timed audit, with the measured rates as
parameters (the defaults are those measured on one RTX PRO 6000 Blackwell Server Edition).

- `tc_ceiling_gbps`: the decode rate if the tensor cores did nothing but the two squarings' MMAs at a given int8 rate.
- `full_pass_gbps`: a kernel and the key expansion run one after the other.
- `seqroot`: P2's SeqRoot margin at width w, derived as `schemes.p2.ROOT_CALL_NS` derives it at 16,448 bits (the Zen 5 floor
  τ_F, (w − 2) + 40 squaring-equivalents a root, the measured 8-core cooperation 3.490×, half the cooperating root as
  `ROOT_CALL_NS`, the RTT cap the largest whole µs with Δ + cap below it), with τ_F scaled to w as w^alpha: log2(3) when
  narrowing drops a Karatsuba level of the floor's plan, log4(7) when it drops a Toom-4 level (the narrower width's best
  case).
- `partial_fraction`: the share of the weights that can be stored only as E when decoding them on every read may take
  `gpu_share` of the GPU.
"""
from __future__ import annotations

import math

from verity.core.protocols.pous.schemes.p2 import DELTA_NS

PAYLOAD = 2054
MAC_PER_MMA = 16 * 8 * 32
TAU_F_NS = 2060.845 / 5.7
COOP = 2716.7 / 778.5
W0 = 16_448

MEASURED = {
    "read_GBps": 1525.0,
    "int8_gemm_tops": 716.5,
    "keys_GBps": 232.8,
    "cuda_core_kernel_GBps": 71.6,
    "cuda_core_full_GBps": 54.75,
}


def tc_ceiling_gbps(mmas_per_square: int, int8_tops: float = MEASURED["int8_gemm_tops"], payload: int = PAYLOAD) -> float:
    macs_per_block = 2 * mmas_per_square * MAC_PER_MMA
    return int8_tops * 1e12 / 2 / macs_per_block * payload / 1e9


def full_pass_gbps(kernel_gbps: float, keys_gbps: float = MEASURED["keys_GBps"]) -> float:
    return 1 / (1 / kernel_gbps + 1 / keys_gbps)


def seqroot(w: int, alpha: float = math.log2(3), delta_ns: int = DELTA_NS) -> dict[str, float]:
    tau = TAU_F_NS * (w / W0) ** alpha
    root_ns = (w - 2 + 40) * tau
    coop_ns = root_ns / COOP
    root_call_ns = math.floor(coop_ns / 2)
    cap_us = math.ceil((root_call_ns - delta_ns) / 1000) - 1
    return {"w": w, "tau_ns": tau, "root_us": root_ns / 1e3, "coop_root_us": coop_ns / 1e3, "root_call_ns": root_call_ns,
            "rtt_cap_us": cap_us if cap_us >= 0 else None}


def partial_fraction(decode_gbps: float = MEASURED["cuda_core_full_GBps"], gpu_share: float = 1.0,
                     read_gbps: float = MEASURED["read_GBps"]) -> float:
    return gpu_share * decode_gbps / read_gbps


if __name__ == "__main__":
    import json

    out = {"tc_ceiling_dense_GBps": tc_ceiling_gbps(1188), "tc_ceiling_kara2_GBps": tc_ceiling_gbps(864),
           "tc_ceiling_ideal_GBps": tc_ceiling_gbps(486),
           "full_pass_ideal_GBps": full_pass_gbps(tc_ceiling_gbps(486)),
           "seqroot": [seqroot(w, a) for w in (16_448, 8_256, 4_160, 2_112) for a in (math.log2(3), math.log(7, 4))],
           "partial_fraction_whole_gpu": partial_fraction(), "partial_fraction_tenth": partial_fraction(gpu_share=0.1)}
    print(json.dumps(out, indent=1))
