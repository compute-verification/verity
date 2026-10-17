"""The TTP-encoding note's derived numbers, from one run's records and named constants (stdlib only).

    python3 derive.py RUN_OUT_DIR [SLOTH_OUT_DIR]   # e.g. ~/.research/runs/r20261017-083410-d961/out
                                                    #      ~/.research/runs/r20261017-090405-2c12/out

Measured inputs are the run's `decode.jsonl`, `chain.jsonl` and `bw.json`, and `sloth_fft.sh`'s `sloth.jsonl` (the FFT
part of one squaring mod 2^1257787 - 1, without carries, so an upper bound on that decode); every other constant names
its source.
"""

from __future__ import annotations

import json
import math
import os
import sys

TAU = 840e-6                 # space audit deadline (definitions page, operating point)
THETA = 0.0169               # space tolerance at p 0.3%, q = eps (workstream A, B1)
G = 8                        # GPUs a node
W_BYTES = 70.6e9             # Llama-3.1-70B FP8 weights (model-v0 section 1)
NODE_READ = 12.31e12         # B of the node's GPUs, measured (workstream A, A6)
CHALLENGE_BYTES = 4096       # b (definitions page)
ATTACK_SPEEDUP = 4           # brute force over the honest decode: incremental first squaring x2, kernel/early abort x2
HOST_SHARE = 0.13            # host CPUs beside the GPUs (squaring-ceiling report: h <= 13%)
DELTA_BITS = 64              # bandwidth audit: delta = 2^-64
SPOT = 100_000               # bandwidth audit spot checks s
X_MAX = 0.01                 # stacked share of the weights we allow
OPENSSL_SIGN = {1024: 13493.3, 2048: 4091.0, 3072: 1520.0}   # this VM, one core, `openssl speed -seconds 3`
OPENSSL_SIGN[1536] = OPENSSL_SIGN[1024] * (OPENSSL_SIGN[2048] / OPENSSL_SIGN[1024]) ** math.log2(1.5)  # interpolated
PRIVATE_OPS_PER_RECORD = 1 / (1 - 0.75 ** 4) + 1            # pad search: y's root when some v is a square, then v's
IMAD_WIDE_PER_SM_CLK = 32 * 0.994                            # squaring-ceiling probe: 0.994 warp IMAD.WIDE a SM-clock
NTT_GBPS_PER_ROUND = (30.0, 50.0)                            # 1e6 public Sloth, modeled from ra-fast-decode's NTT
SLOTH_A = 1_257_787 - 2                                      # squarings a root mod 2^1257787 - 1


def load(out):
    rows = [json.loads(line) for line in open(os.path.join(out, "decode.jsonl"))]
    chain = [json.loads(line) for line in open(os.path.join(out, "chain.jsonl"))]
    bw = json.load(open(os.path.join(out, "bw.json")))
    clock = json.load(open(os.path.join(out, "clock-before.json")))
    return rows, chain, bw, clock


def products_per_record(bits, form):
    nl = bits // 32
    square = nl * (nl + 1) // 2
    reduce = 2 * (nl // 2) ** 2 if form >= 4 else nl * nl
    return 2 * (square + reduce)


def measured_sloth(sloth_out):
    if not sloth_out:
        return []
    rows = [json.loads(line) for line in open(os.path.join(sloth_out, "sloth.jsonl"))]
    best = max(rows, key=lambda r: r["word_gb_per_s_median"])
    print(f"\nFP64 cuFFT squaring mod 2^1257787 - 1, FFT part only: best {best['word_gb_per_s_median']:.2f} GB/s a round "
          f"({best['points']} points, {best['words']} words, max round-off {best['max_roundoff']})")
    return [("FP64 cuFFT, measured", best["word_gb_per_s_median"])]


def main(out, sloth_out=None):
    rows, chain, bw, clock = load(out)
    read = bw["read_gb_per_s_median"]
    print(f"read {read} GB/s (median), clock {clock['clock_mhz']} MHz, {clock['sms']} SMs")
    ceiling = IMAD_WIDE_PER_SM_CLK * clock["sms"] * clock["clock_mhz"] * 1e6
    best = {}
    print("\n| bits | form | records/s | record GB/s | w_d | products/s | of IMAD.WIDE ceiling |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        rate = r["records_per_s_median"]
        prod = rate * products_per_record(r["bits"], r["form"])
        print(f"| {r['bits']} | {r['form']} | {rate:.3e} | {r['record_gb_per_s_median']:.1f} | "
              f"{read / r['record_gb_per_s_median']:.2f} | {prod:.3e} | {prod / ceiling:.0%} |")
        if r["bits"] not in best or rate > best[r["bits"]]["records_per_s_median"]:
            best[r["bits"]] = r
    print("\nper width, best form:")
    print("| bits | form | record GB/s | w_d | payload B | record B | |C|/|W| (tag byte) | |C|/|W| (tag bit) | "
          "records for 70.6 GB | s (4 KiB challenge) | s/n | s (one-record challenge) | s/n | TTP core-h | "
          "TTP min on 128 cores |")
    print("|" + "---|" * 15)
    derived = {}
    for bits, r in sorted(best.items()):
        rec_b = bits // 8 + 1
        pay_b = bits // 8 - 2 if r["form"] >= 4 else bits // 8 - 1
        records = W_BYTES / pay_b
        rate = r["records_per_s_median"]
        budget = G * rate * TAU * ATTACK_SPEEDUP * (1 + HOST_SHARE)
        per_blk = CHALLENGE_BYTES * 8 / bits
        s_blk, s_one = math.log2(budget / per_blk), math.log2(budget)
        sign = OPENSSL_SIGN.get(bits)
        core_h = records * PRIVATE_OPS_PER_RECORD / sign / 3600 if sign else float("nan")
        derived[bits] = dict(w_d=read / r["record_gb_per_s_median"], s_blk=s_blk, s_one=s_one)
        print(f"| {bits} | {r['form']} | {r['record_gb_per_s_median']:.1f} | {read / r['record_gb_per_s_median']:.2f} | "
              f"{pay_b} | {rec_b} | {rec_b / pay_b:.4f} | {(bits / 8 + 1 / 8) / pay_b:.4f} | {records:.3e} | "
              f"{s_blk:.1f} | {s_blk / bits:.2%} | {s_one:.1f} | {s_one / bits:.2%} | {core_h:.1f} | "
              f"{core_h * 60 / 128:.0f} |")

    print(f"\nTTP private ops a record (pad search): {PRIVATE_OPS_PER_RECORD:.2f}")
    print(f"forge ln(1/delta)/s = {DELTA_BITS * math.log(2) / SPOT:.3%}")
    cap = X_MAX * W_BYTES
    print(f"input cap for x <= {X_MAX:.0%}: I_live <= {cap / 1e6:.0f} MB a node; "
          + ", ".join(f"R_in <= {cap / t / 1e3:.0f} kB/s at T_res {name}"
                      for name, t in (("10 min", 600), ("1 h", 3600), ("6 h", 21600), ("1 day", 86400))))
    for name, r_in in (("token ids at prefill (A, R3)", 58e3), ("honest serving, high (F)", 110e3)):
        print(f"  epoch at {name} {r_in / 1e3:.0f} kB/s: E <= {cap / r_in / 3600:.1f} h")

    rates = [("NTT, modeled", g) for g in NTT_GBPS_PER_ROUND] + measured_sloth(sloth_out)
    print("\n1e6 public Sloth:")
    for k_rounds in (1, 2):
        for name, gbps in rates:
            dec = gbps / k_rounds
            w_d = read / dec
            setup_s = W_BYTES / (G * dec * 1e9 / SLOTH_A)
            print(f"  {k_rounds} round(s), {name} {gbps:.1f} GB/s a round: w_d {w_d:.0f}, honest setup of 70.6 GB on "
                  f"the node {setup_s / 3600:.0f} h, x at L = 10 min {600 / setup_s:.3%}")
    for bits in (1024, 2048):
        w = derived[bits]["w_d"]
        for k_rounds in (1, 2):
            print(f"  RW-SMS {bits} vs {k_rounds}-round Sloth: decode "
                  + ", ".join(f"{read / g * k_rounds / w:.0f}x cheaper ({name} {g:.1f})" for name, g in rates))
    print(f"\nnode: weights read once a pass {W_BYTES / NODE_READ * 1e3:.2f} ms; "
          + ", ".join(f"{b}-bit decode adds {W_BYTES / (G * best[b]['record_gb_per_s_median'] * 1e9) * 1e3:.1f} ms"
                      for b in sorted(best)))


if __name__ == "__main__":
    main(*(os.path.expanduser(a) for a in sys.argv[1:3]))
