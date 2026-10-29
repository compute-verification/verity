"""What committing NCP's checkpoint words with SHA-512 costs in Llama-3.2-1B's served step on the H200: stdlib only, no GPU.

    python3 benchmarks/served_commit/ncp_k32_price.py OUT_JSON

NCP computes each linear's dot product as one 3K-deep stacked product and the prover commits its accumulator word at every
step of depth d: 3K/d words of 4 bytes a dot product, so 12/d bytes a useful MAC whatever K is (d = 32, k32: 0.375; d = 16,
ncp-v1 and ncp-v2 today: 0.75). The words go into hm96 SHA-512 rows of 16 KiB. A forward's hashing takes
(B h_x − h_leaf min(B, G)) / (540,672 f) seconds for B bytes (GD-HASH's model, its inputs measured on vy-dc-h200-5), and
NCP's GEMMs cost what GD-DENOISE measured for a 3K-wide cuBLASLt call and fresh noise on both operands. t_H / t_S is the
stock step plus both, over the stock step; "matmuls alone" reads t_S as the plain GEMMs only, without the rest of the
step. Every input is a measurement (its art and run) or an estimate (its source),
and the output says which. Coverage: each matmul's final-word gap d/(3K), against 1%, and its share of the MACs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RULER = 4096 * 132                                  # dense int8 MAC-times a cycle on the H200: 4,096 an SM, 132 SMs
F = 1.965e9                                         # the step loops' SM clock (NVML, r20261028-202719-cc9b)
GAP_MEASURED = 16.8e6 / F                           # the loop's host gap that G was measured on, a 16.8 M-cycle sleep
HBM = 4.8e12                                        # H200 HBM3e bytes a second (data sheet), for words written out unfused

# measured: art:5cf3621a988a946e98dcb8d36eafbecc3496b898df7ae7f481f757f5308ff991 (r20261028-185933-4435, r20261028-202719-cc9b)
ROWS = {4096: {"h_x": 2703.3, "h_leaf": 2569.9, "G": 3.4545e9},
        16384: {"h_x": 2123.4, "h_leaf": 2077.6, "G": 4.4836e9}}
U_RULE = 0.258                                      # GD-HASH's stock utilization on GPU-busy time at T 8,192

LAYERS = 16
LINEARS = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192)}
HEAD = (128256, 2048)                               # tied, bf16 under verity-int8-channel127/v1, so outside NCP
HEADS, HEAD_DIM = 32, 64

# measured: art:43138e61baa8fa11988e5ff9873a51133bde30c26784f90a321c67899822be5e (r20261028-194021-247f, vy-dc-h200-5),
# table.md's model rows: the 16 layers' plain GEMMs (µs a forward) and, over them, fresh A-side noise, the noised GEMM,
# one 3K-wide cuBLASLt call less the noised one, and the weight side formed a call (W′ a call)
GEMM = {64: {"plain_us": 570.8, "a_side": 1.893, "noised": -0.001, "wide": 0.996, "w_side": 5.683},
        512: {"plain_us": 1194.3, "a_side": 2.059, "noised": 0.012, "wide": 1.169, "w_side": 2.739},
        8192: {"plain_us": 12494.2, "a_side": 2.424, "noised": 0.043, "wide": 1.536, "w_side": 0.265}}

# the stock steps (GD-HASH, note:compute-research/20261028T2230Z-report-gd-hash-reframe)
ENGINES = {
    8192: {"sync": (37.77, 8.73, "measured: HASH-56's step, r20261028-023029-1f82; its served gap 8.73 ms"),
           "async": (37.77 - 8.73 + 2.29, 2.29, "estimate: HASH-56's step with p1a-levers' async gap, 2.29 ms")},
    512: {"sync": (0.96 + 2.46, 2.46, "estimate: p1a-levers' T 8,192 parts scaled (kernels 0.96 ms, gap 2.46 ms)"),
          "async": (0.96 + 0.71, 0.71, "estimate: the same with the async gap, 0.71 ms")},
}
POINTS = {"T 8,192": (8192, 8192, 8192), "T 512, 64 real rows": (512, 64, 64), "T 512, 512 rows": (512, 512, 512)}


def words_bytes(n: int, k: int, d: int) -> int:
    return n * 4 * -(-3 * k // d)


def per_token(d: int, head: bool = False) -> dict:
    lin = {name: words_bytes(n, k, d) * LAYERS for name, (n, k) in LINEARS.items()}
    macs = LAYERS * sum(n * k for n, k in LINEARS.values())
    if head:
        lin["head"], macs = words_bytes(*HEAD, d), macs + HEAD[0] * HEAD[1]
    return {"bytes": sum(lin.values()), "by_linear": lin, "macs": macs}


def hash_ms(B: float, rb: int, gap_ms: float) -> tuple[float, float]:
    r = ROWS[rb]
    eta = r["h_leaf"] * r["G"] / (RULER * F) / GAP_MEASURED
    G = eta * gap_ms * 1e-3 * RULER * F / r["h_leaf"]
    excl = B * r["h_x"] / (RULER * F) * 1e3
    return excl, excl - r["h_leaf"] * min(B, G) / (RULER * F) * 1e3


def coverage(ctx: int, decode: bool) -> dict:
    """MAC shares a served token at context ctx (decode: every key, and the head; a prefill of ctx tokens: row j has j
    keys, and the head runs on its last token only), and what NCP covers: the linears; with the head on int7 too; and
    with P·V's dot products of at least 1,067 keys."""
    keys = ctx if decode else (ctx + 1) / 2
    lin, head = LAYERS * sum(n * k for n, k in LINEARS.values()), HEAD[0] * HEAD[1] / (1 if decode else ctx)
    qk = pv = LAYERS * HEADS * HEAD_DIM * keys
    total = lin + head + qk + pv
    short = float(ctx < 1067) if decode else min(1.0, 1066 * 1067 / (ctx * (ctx + 1)))   # P·V MACs on rows < 1,067 keys
    return {"linears": lin / total, "head": head / total, "qk": qk / total, "pv": pv / total, "pv_macs_short": short,
            "gap_pv_full": 32 / (3 * ctx), "covered_linears": lin / total, "covered_with_head": (lin + head) / total,
            "covered_with_head_long_pv": (lin + head + pv * (1 - short)) / total}


def main() -> None:
    out: dict = {"inputs": {"rows": ROWS, "gemm": GEMM, "engines": ENGINES}, "gaps": {}, "points": {}, "coverage": {}}
    for name, (n, k) in {**LINEARS, "head": HEAD, "attention qk": (0, HEAD_DIM)}.items():
        out["gaps"][name] = {"K": k, "k32": 32 / (3 * k), "d16": 16 / (3 * k), "over_1pct_k32": 32 / (3 * k) > 0.01}
    for d, head in ((32, False), (32, True), (16, False)):
        t = per_token(d, head)
        rule = t["bytes"] / t["macs"] * ROWS[16384]["h_x"] * U_RULE
        for pname, (T, real, m) in POINTS.items():
            B, g = t["bytes"] * real, GEMM[min(GEMM, key=lambda x: abs(x - m))]
            gemm_ms = g["plain_us"] * (g["a_side"] + g["noised"] + g["wide"] + g["w_side"]) * 1e-3
            excl, _ = hash_ms(B, 16384, 0.0)
            plain_ms = g["plain_us"] * 1e-3
            row = {"bytes_token": t["bytes"], "GB_forward": B / 1e9, "rows_16k": B / 16384, "exclusive_ms": excl,
                   "hbm_write_ms_unfused": B / HBM * 1e3, "gemm_extra_ms": gemm_ms, "plain_gemm_ms": plain_ms,
                   "tH_tS_matmuls": None if head else (plain_ms + gemm_ms + excl) / plain_ms, "engines": {}}
            for e, (stock, gap, src) in ENGINES[T].items():
                _, gapfirst = hash_ms(B, 16384, gap)
                row["engines"][e] = {"stock_ms": stock, "hash_ms_gap_first": gapfirst, "src": src,
                                     "tH_tS": (stock + gapfirst + gemm_ms) / stock,
                                     "tH_tS_no_gap": (stock + excl + gemm_ms) / stock}
            out["points"].setdefault(f"d {d}{', head too' if head else ''}",
                                     {"bytes_mac": t["bytes"] / t["macs"], "J_rule_T8192": rule})[pname] = row
    for ctx, decode in ((144, True), (288, True), (2048, False), (8192, False)):
        out["coverage"][f"{'decode' if decode else 'prefill'} {ctx}"] = coverage(ctx, decode)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    for d, p in out["points"].items():
        print(f"{d}: {p['bytes_mac']:.3f} B a useful MAC, J by GD-HASH's rule at T 8,192 {p['J_rule_T8192']:.0f}")
        for pname, r in p.items():
            if not isinstance(r, dict):
                continue
            eng = "  ".join(f"{e} {x['tH_tS']:.1f}x (no gap {x['tH_tS_no_gap']:.1f}x)" for e, x in r["engines"].items())
            if r["tH_tS_matmuls"] is not None:
                eng += f"  matmuls alone {r['tH_tS_matmuls']:.0f}x"
            print(f"  {pname:<20} {r['bytes_token'] / 1e6:6.1f} MB/token {r['GB_forward']:8.1f} GB/fwd "
                  f"{r['rows_16k'] / 1e6:7.2f} M rows  hash {r['exclusive_ms']:8.1f} ms  HBM write {r['hbm_write_ms_unfused']:6.1f} ms  "
                  f"GEMM +{r['gemm_extra_ms']:5.1f} ms  t_H/t_S {eng}")
    for name, g in out["gaps"].items():
        print(f"gap {name:<13} K {g['K']:>5}: k32 {g['k32']:.2%}, d16 {g['d16']:.2%}")
    for c, v in out["coverage"].items():
        print(f"coverage {c:<20} linears {v['linears']:.1%}  head {v['head']:.1%}  QK {v['qk']:.1%}  PV {v['pv']:.1%} "
              f"(on rows < 1,067 keys {v['pv_macs_short']:.0%})  covered: linears {v['covered_linears']:.1%}, "
              f"+ head {v['covered_with_head']:.1%}, + long P·V {v['covered_with_head_long_pv']:.1%}")


if __name__ == "__main__":
    main()
