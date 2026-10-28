"""Does the Module-SIS commitment arm lower J?  A model over cited inputs, stdlib only:

    python3 j_effect.py ANDS.json OUT.json

J = (t_remote / t_stock - 1) + proving GPU-s a window / stock GPU-s a window. Both terms divide by the stock GPU-s of a
window (W steps of t_stock on one GPU), so swapping SHA-512 for SIS changes J by

    dJ = (W * d_serve + d_prove) / (W * t_stock),

d_serve the change in a step's commitment time (GPU-s, negative when SIS is faster) and d_prove the change in a window's
proving GPU-s from opening rows under SIS. Its sign does not depend on t_stock. SIS lowers J only when
W > W* = d_prove / (-d_serve).

Serve time is in SM-cycles a committed byte, so it holds at any clock:
- SHA-512 (hm96-sha512 rows in a frame-v3-sha512 tree): COMMIT-BYTES' compressions a step at HASH-38's bar of 0.01803
  compressions an SM-clock (note:compute-research/20261028T1600Z-finding-commit-bytes): 149.0 M a step at 4 KiB rows,
  117.1 M at 16 KiB, over 13,628,428,480 B (b921).
- SIS (q = 2^32, int8 x, A in four int8 limbs, so 4n MACs a byte), at two rates: `datasheet`, the H200's dense int8
  peak, 989 T MAC/s at 1,830 MHz, 4,094 MACs an SM-clock (the coordinator's figure); `measured`, the hash lane's H200
  cuBLASLt rate at n = 256, 737 GB/s at 1,972 MHz including the limb passes (r20261018-065316-6afc,
  note:hash/20261018T0627Z-draft-lattice), 71% of that peak, scaled by n. Then the row's digest (4n B) through SHA-512
  and hm96's and the tree's compressions, as the SHA-512 path pays them. This is the whole GPU on the commitment alone, the most favourable case for SIS: SHA-512 also co-runs beside
  the served GEMM at 0.167 of its bar for free (r20261028-012220-309e), and SIS, a tensor-core GEMM itself, cannot.

Proving: D draws a window open r_d rows each, every row costing `row_ands` ANDs (SHA-512: optimize.row_ands; SIS: the
recompute, SHA-512 over the digest, hm96) at kappa GPU-s an AND. D = ln(1/delta) / eps, Achievability's draws for
uniform weights at delta = 2^-40, eps = 1%.
"""
from __future__ import annotations

import json
import sys
from math import ceil, log

SHA512_ANDS, HM96_ANDS = 58_120, 116_240
BAR = 0.01803                     # SHA-512 compressions an SM-clock (HASH-38)
SMS = 132
CLOCK = 1.769e9                   # b57f's measured clock
STEP_BYTES = 13_628_428_480       # b921, T 8,192
T_STOCK = 26.14e-3                # b57f, T 8,192
TOKEN_BYTES = 1_663_627
T512_REAL = 64                    # V3: T 512 with 64 real tokens; the step commits their rows
MODEL_MACS_TOKEN = 1.03e9         # COMMIT-BYTES, from b57f's tensor work
SIS_CYC_N256 = {"datasheet": 4 * 256 / (989e12 / (SMS * 1.830e9)),   # SM-cycles a byte at n = 256
                "measured": SMS * 1.972e9 / 737e9}
SIS_N = (192, 208, 224, 256)
PER_ROW_FIXED = 9 + 2.8           # hm96's salt, c and h, the frame leaf (9), and the tree's nodes and pads a row
DELTA_LOG = 40 * log(2)
EPS = 0.01
KAPPAS = {"1e-10 (favourable floor)": 1e-10, "1.96e-10 (proofs, zk proving slope)": 1.96e-10,
          "3.57e-10 (P3, 2.8e9 ANDs a GPU-s)": 3.57e-10}


def sha_rows(nbytes: int) -> float:
    return ceil((nbytes + 17) / 128)


def serve_cycles(scheme: str, row: int, n: int = 0, digest_bytes: int = 0, rate: str = "measured") -> float:
    """SM-cycles a committed byte."""
    if scheme == "sha512":
        return (sha_rows(row) + PER_ROW_FIXED) / row / BAR
    return SIS_CYC_N256[rate] * n / 256 + (sha_rows(digest_bytes or 4 * n) + PER_ROW_FIXED) / row / BAR


def serve_cycles_lhl(row: int, n: int, rate: str = "measured") -> float:
    """Ajtai-hiding: d = A x + B r mod 2^32 with r 4n + 33 uniform int8 words (the leftover hash lemma), r expanded by
    SHA-512 as a PRF (64 B a compression), d itself the tree's leaf content (no hm96: d hides), the frame leaf, the
    tree."""
    r = 4 * n + 33
    return (SIS_CYC_N256[rate] * n / 256) * (row + r) / row + (sha_rows(4 * n) + 2 + 2.8 + r / 64) / row / BAR


def break_even(d_serve: float, d_prove: float) -> str:
    """The windows (steps) at which SIS lowers J."""
    if d_serve < 0 < d_prove:
        return f"W > {d_prove / -d_serve:.0f}"
    if d_serve <= 0 and d_prove <= 0:
        return "every W"
    if d_prove < 0 < d_serve:
        return f"W < {-d_prove / d_serve:.0f}"
    return "no W"


def row_ands_sha(row: int) -> float:
    return HM96_ANDS + SHA512_ANDS * sha_rows(row)


def row_ands_sis(row: int, n: int, h: float, digest_bytes: int | None = None, extra: float = 0) -> float:
    return h * row + SHA512_ANDS * sha_rows(digest_bytes or 4 * n) + HM96_ANDS + extra


def main() -> None:
    ands = json.load(open(sys.argv[1]))
    h = {(r["encoding"], r["construction"], r["n"]): r["ands_per_byte"] for r in ands["rows"]}
    proj = {p["m"]: p for p in ands["projection"]}
    draws = DELTA_LOG / EPS
    out = {"inputs": {"step_bytes": STEP_BYTES, "t_stock": T_STOCK, "clock": CLOCK, "bar": BAR,
                      "sis_cycles_per_byte_n256": SIS_CYC_N256, "draws": draws, "kappas": KAPPAS}, "rows": []}
    for row, rate, n in ((row, rate, n) for row in (4096, 16384) for rate in SIS_CYC_N256 for n in SIS_N):
        sha_cyc = serve_cycles("sha512", row)
        p = proj[row]
        variants = [(c, 4 * n, row_ands_sis(row, n, h[("int8", c, n)])) for c in ("pp-naf", "anf-8")]
        for v in ("const", "var"):
            variants.append((f"proj-{v} (2-phase C-Flock)", 7 * n,
                             row_ands_sis(row, n, p[f"{v}_ands_per_byte"], 7 * n, n * 54 * 61 / 3 + 244)))
        variants.append(("proj-lhl (2-phase, Ajtai-hiding)", 0,
                         p["const_ands_per_byte"] * (row + 4 * n + 33) + n * 22 * 61 / 3 + 244))
        for name, digest, sis_row in variants:
            sis_cyc = (serve_cycles_lhl(row, n, rate) if name.startswith("proj-lhl")
                       else serve_cycles("sis", row, n, digest, rate))
            d_serve = STEP_BYTES * (sis_cyc - sha_cyc) / (SMS * CLOCK)
            d_row = sis_row - row_ands_sha(row)
            for kname, kappa in KAPPAS.items():
                for r_d in (1, 2):
                    d_prove = draws * r_d * d_row * kappa
                    out["rows"].append({
                        "row": row, "tensor_rate": rate, "n": n, "opening": name, "kappa": kname,
                        "rows_per_draw": r_d,
                        "serve_cycles_per_byte": {"sha512": round(sha_cyc, 4), "sis": round(sis_cyc, 4)},
                        "d_serve_ms_step": round(d_serve * 1e3, 2), "d_serve_J": round(d_serve / T_STOCK, 3),
                        "row_ands": {"sha512": row_ands_sha(row), "sis": round(sis_row)},
                        "d_prove_gpu_s_window": round(d_prove, 2),
                        "dJ_window_1_step": round((d_serve + d_prove) / T_STOCK, 1),
                        "helps_t8192": break_even(d_serve, d_prove),
                        "helps_t512_64_real": break_even(d_serve * T512_REAL / 8192, d_prove),
                        "helps_t8192_if_sis_were_free": break_even(-STEP_BYTES * sha_cyc / (SMS * CLOCK), d_prove),
                        "sis_macs_over_model_macs": round(4 * n * TOKEN_BYTES / MODEL_MACS_TOKEN, 2)})
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    for r in out["rows"]:
        if r["kappa"].startswith("1.96") and r["rows_per_draw"] == 1:
            print(f"row {r['row']:5} {r['tensor_rate']:9} n={r['n']} {r['opening']:28} serve {r['d_serve_ms_step']:+7.2f} ms/step "
                  f"({r['d_serve_J']:+.3f} J)  row ANDs {r['row_ands']['sis']:>11,} vs {r['row_ands']['sha512']:>10,}"
                  f"  prove {r['d_prove_gpu_s_window']:+10.2f} GPU-s/window  dJ(W=1) {r['dJ_window_1_step']:+9.1f}"
                  f"  T8192 {r['helps_t8192']}, T512/64 {r['helps_t512_64_real']}")


if __name__ == "__main__":
    main()
