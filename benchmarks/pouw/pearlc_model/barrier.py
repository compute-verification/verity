"""Pearl-C under the data-flow model: the circuit barrier on TT_OUT and the output-volume floors (arithmetic only).

1. The barrier.  A Boolean circuit of s gates (fan-in 2) for F = (C̃, U) of one unit runs as a W1 program of at most
   s + N_in + N_out INT32 lane instructions (one `lop3` a gate, one bit extraction an input bit, one packing step an
   output bit) plus what every program pays and nothing credits: the weight side (`job_cost`), the hashing and the
   fetches.  So TT_OUT at W1 implies every such circuit has
   s >= ((1 - rho)·credit - job - fetch·(m + n)·k) / p_int32 - N_in - N_out (hashing left out, which only lowers it).
   The script prints that bound over N = max(N_in, N_out) per shape and device, at fetch 0 and at the DRAM price, beside
   the best known explicit bound for an N-to-N function (about 4N) and the open 5N.  It also prints the concurrent (G2)
   form, where gates run at 96 per SM clock (64 LOP3 lanes and 64 FFMA lanes at two a gate), and the P2 forms, where
   each gate also moves 3 words through shared memory or HBM at their data-sheet rates.
2. The output-volume floor (4g).  Every fresh transcript word is written by an arithmetic instruction (a move copies an
   existing word).  Additively (W1's sum of prices) a word costs at least the cheapest price per output word; under G2
   the device writes at most W_max fresh words per SM clock over all pipes at once.  c0 is the floor's share of the
   credited work, for Pearl-C as designed (C̃ and U: 2 words an output), Dither (one word per 32 MACs), and Pearl-C with
   its atom partials (one word per k32 atom) or window partials (one per 128) committed.
3. Rule L: the pooled datacenter figure of what the floor proves, a week, a month and a year, 1,000 machines.

Usage: python barrier.py [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction

from verity.core.protocols.pouw.protocol import Shape
from verity.core.protocols.pouw.schemes.pearl_c import R, PearlC
from verity.core.protocols.pouw.schemes.pearl_c_device import H100, SM120

# Per SM clock, data-sheet rows (benchmarks/pouw/machine_model DS rows, benchmarks/census/hardware.json).
DEVICES = {
    "h100": dict(device=H100, tc=4096, gate_rate=96, smem_words=32, mac_per_s=1978.9e12 / 2, mem_bytes_per_s=3.35e12,
                 p_int32=64, p_fp32=32, w1_word=32, tc_words=4096 // 16),
    "sm120": dict(device=SM120, tc=1024, gate_rate=96, smem_words=32, mac_per_s=1.0e15 / 2, mem_bytes_per_s=1.597e12,
                  p_int32=16, p_fp32=8, w1_word=8, tc_words=1024 // 16),
}
# Non-TC fresh words per SM clock at most: two IMAD.WIDE issues (64 lanes x 2 words) and two 32-bit issues (64 lanes).
NON_TC_WORDS = 192

SHAPES = [("decode 32x8192x8192", 32, 8192, 8192), ("prefill 4096x8192x8192", 4096, 8192, 8192),
          ("square 8192^3", 8192, 8192, 8192), ("square 16384^3", 16384, 16384, 16384),
          ("deep 16384x65536x16384", 16384, 65536, 16384)]


def io_bits(m: int, k: int, n: int) -> tuple[int, int]:
    """F's input bits (A's FP32 rows, B̃'s codes, P_B's 64 BF16 a column, E_A's lines, F_A and F_B, the seed) and outputs."""
    n_in = 32 * m * k + 8 * n * k + 16 * 2 * R * n + 8 * R * m + 2 * 8 * R * k + 512
    return n_in, 64 * m * n


def barrier(out: dict) -> None:
    rows = []
    for dname, d in DEVICES.items():
        scheme = PearlC(forming="v1", device=d["device"], hashing="h3")
        p_smem = d["tc"] / d["smem_words"]
        p_mem = d["mac_per_s"] / (d["mem_bytes_per_s"] / 4)
        for label, m, k, n in SHAPES:
            credit = float((1 - scheme.cap) * scheme.credit_of(Shape(m, k, n)))
            job = float(scheme.job_cost(n, k))
            fetch = p_mem * (m + n) * k
            n_in, n_out = io_bits(m, k, n)
            big_n = max(n_in, n_out)
            w1 = (credit - job) / d["p_int32"] - n_in - n_out
            w1_fetch = (credit - job - fetch) / d["p_int32"] - n_in - n_out
            g2 = (credit - job) * d["gate_rate"] / d["tc"] - n_in - n_out
            p2_smem = (credit - job) / (d["p_int32"] + 3 * p_smem) - n_in - n_out
            p2_mem = (credit - job) / (d["p_int32"] + 3 * p_mem) - n_in - n_out
            rows.append(dict(device=dname, shape=label, credit=credit, job=job, fetch_dram=fetch, n_in=n_in, n_out=n_out,
                             s_over_N_w1=w1 / big_n, s_over_N_w1_fetch=w1_fetch / big_n, s_over_N_g2=g2 / big_n,
                             s_over_N_p2_smem=p2_smem / big_n, s_over_N_p2_hbm=p2_mem / big_n, p_smem=p_smem, p_hbm=p_mem))
    out["barrier"] = rows
    print(f"{'device':6s} {'shape':24s} {'credit':>10s} {'job':>10s} {'N_in':>10s} {'N_out':>10s}  "
          f"s/N: W1  W1+fetch  G2  P2-smem  P2-HBM")
    for r in rows:
        print(f"{r['device']:6s} {r['shape']:24s} {r['credit']:10.3e} {r['job']:10.3e} {r['n_in']:10.3e} {r['n_out']:10.3e}  "
              f"{r['s_over_N_w1']:7.2f} {r['s_over_N_w1_fetch']:7.2f} {r['s_over_N_g2']:7.2f} {r['s_over_N_p2_smem']:7.3f} "
              f"{r['s_over_N_p2_hbm']:8.4f}")


def floors(out: dict) -> None:
    rows = []
    for dname, d in DEVICES.items():
        scheme = PearlC(forming="v1", device=d["device"], hashing="h3")
        macs_per_word_g2 = d["tc"] / (d["tc_words"] + NON_TC_WORDS)
        for k in (1024, 4096, 16384, 65536):
            m = n = 8192
            per_out = float(scheme.credit_of(Shape(m, k, n))) / (m * n)
            for name, words in (("pearl-c", 2), ("pearl-c-T atoms", k // 32 + 1), ("pearl-c-T windows", k // 128 + 1)):
                rows.append(dict(device=dname, k=k, scheme=name, words_per_output=words, credit_per_output=per_out,
                                 c0_w1=min(1.0, words * d["w1_word"] / per_out),
                                 c0_g2=min(1.0, words * macs_per_word_g2 / per_out)))
        rows.append(dict(device=dname, k=None, scheme="dither (L=32)", words_per_output=None, credit_per_output=32,
                         c0_w1=min(1.0, d["w1_word"] / 32), c0_g2=min(1.0, macs_per_word_g2 / 32)))
    out["floors"] = rows
    print(f"\n{'device':6s} {'k':>6s} {'scheme':18s} {'words/out':>9s} {'credit/out':>10s} {'c0 W1':>8s} {'c0 G2':>8s}")
    for r in rows:
        print(f"{r['device']:6s} {str(r['k']):>6s} {r['scheme']:18s} {str(r['words_per_output']):>9s} "
              f"{r['credit_per_output']:10.1f} {r['c0_w1']:8.4f} {r['c0_g2']:8.4f}")


def rule_l(out: dict) -> None:
    """Pooled over a datacenter of 1,000 machines at the census's dense E4M3 rate (a rate, not a run): the credited MACs,
    and the part the G2 output-volume floor proves of each Pearl-C transcript at k = 1024 and 16384.  The floor is
    deterministic (it holds for every run), so it reads the same at δ = 1% and 1e-6."""
    periods = {"week": 7 * 86400, "month (30 d)": 30 * 86400, "year (365 d)": 365 * 86400}
    rows = []
    for dname, d in DEVICES.items():
        for f in out["floors"]:
            if f["device"] != dname or f["k"] not in (1024, 16384) or f["scheme"] == "pearl-c-T windows":
                continue
            for pname, secs in periods.items():
                total = 1000 * d["mac_per_s"] * secs
                rows.append(dict(device=dname, k=f["k"], scheme=f["scheme"], period=pname, pooled_macs=total,
                                 proved_macs=total * f["c0_g2"], share=f["c0_g2"],
                                 per_window_macs=d["mac_per_s"] * 60, per_window_proved=d["mac_per_s"] * 60 * f["c0_g2"]))
    out["rule_l"] = rows
    print(f"\n{'device':6s} {'k':>6s} {'scheme':16s} {'period':13s} {'pooled MACs':>12s} {'proved (both δ)':>16s} {'share':>8s}"
          f" {'one 60 s window':>16s}")
    for r in rows:
        print(f"{r['device']:6s} {r['k']:6d} {r['scheme']:16s} {r['period']:13s} {r['pooled_macs']:12.3e} "
              f"{r['proved_macs']:16.3e} {r['share']:8.4f} {r['per_window_proved']:16.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    out: dict = {"note": __doc__.splitlines()[0], "devices": {k: {kk: (str(v) if kk == "device" else v) for kk, v in d.items()}
                                                             for k, d in DEVICES.items()}, "non_tc_words": NON_TC_WORDS}
    barrier(out)
    floors(out)
    rule_l(out)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
