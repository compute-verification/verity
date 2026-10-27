"""FP8-g1's and W4A8-g1's committed values through H1's own transcript construction (co32c-u16, the RTX PRO 6000's
record).

g1 serves each operand as its nearest level R plus a fair coin, centred: (R_x + b - 1/2)(R_w + c - 1/2), b and c H1's
own coins (its activation dither and weight coin). It is credited at H1's ε when the one int8 GEMM of R_x + b and
R_w + c is H1's transcript: R_w in H1's committed weights' place (|W| <= 110), R_x in its activations' (|V| <= 126),
and every operand H1 feeds the MMA where `cgemm_ref.py`'s h1 branch checks it: X = V + b and W' = W + c + U in
[-126, 127], U on +-16 (`u_rows16`). For FP8-g1 (activations E4M3's integers to 120, weights to 104) and W4A8-g1
(weights m q, q twice an E2M1 value, m an integer UE4M3 block scale to 9 a block of 16; activations H1's own int8)
this checks:

- the ranges: the largest E4M3 integer weight top, activation top and block scale whose operands fit H1, and the
  next one up with the operand that leaves;
- the transcript: tiles of 128 x 128 at k = 4096 and 14336, the operands at the grids' extremes (every level at the
  top, at the bottom, of opposite signs) and at random levels, keyed as cgemm.cu keys a call (b, c, U and z from one
  key, `cgemm_ref.py`'s draws), through checked_strip's `co32c_words`: X, W' and P_A in H1's ranges, the served word
  O = O' - sum_d 128^d D_d equal to (R_w + c) X (`O_is_served`), and g1's served output from O and its rank-one
  terms equal to the exact centred product;
- the controls: E4M3 weights to 112 and block scales to 10 at the top, refused by the same range checks (W' to 129;
  levels to 120, past H1's 110), while the identities still hold (the arithmetic is no range check).

Exit 1 on any failure, or on a control that isn't refused. Run from the repository root:
    uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit \
      python3 benchmarks/pouw/fpx_design/g1h_check.py [--json OUT]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cgemm_ref", HERE.parent / "edited_gemm" / "cgemm_ref.py")
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)
cs = cg.cs

#: H1's checks (`cgemm_ref.check`'s h1 branch): committed weights to 110, activations V to 126, the MMA's operands
H1_W, H1_V, H1_OP = 110, 126, (-126, 127)
#: every U `u_rows16` draws: -16 + beta . (1, 2, 4, 8, 17)
U16 = sorted({-16 + int(np.dot([(b >> p) & 1 for p in range(5)], cg.W_RE16)) for b in range(32)})
E2M1X2 = (0, 1, 2, 3, 4, 6, 8, 12)
T = cg.T


def e4m3_integers() -> list[int]:
    out = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            v = Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)
            if v.denominator == 1:
                out.add(int(v))
    return sorted(out)


E4M3I = e4m3_integers()


def weight_range(top: int) -> tuple[int, int]:
    """W' = R + c + U over R in [-top, top], c in {0, 1} and U16."""
    return -top + U16[0], top + 1 + U16[-1]


def weight_fits(top: int) -> bool:
    lo, hi = weight_range(top)
    return top <= H1_W and H1_OP[0] <= lo and hi <= H1_OP[1]


def act_fits(top: int) -> bool:
    return top <= H1_V and H1_OP[0] <= -top and top + 1 <= H1_OP[1]


def ranges() -> dict:
    w8 = max(t for t in E4M3I if weight_fits(t))
    a8 = max(t for t in E4M3I if act_fits(t))
    w4 = max(m for m in range(1, 16) if weight_fits(12 * m))
    nw8, na8 = E4M3I[E4M3I.index(w8) + 1], E4M3I[E4M3I.index(a8) + 1]
    return {"U": [U16[0], U16[-1]], "fp8_weight_top": w8, "fp8_weight_operands": weight_range(w8),
            "fp8_weight_next": {"top": nw8, "operands": weight_range(nw8), "fits": weight_fits(nw8)},
            "fp8_act_top": a8, "fp8_act_operands": [-a8, a8 + 1],
            "fp8_act_next": {"top": na8, "over_V": na8 > H1_V},
            "w4_scale_top": w4, "w4_weight_operands": weight_range(12 * w4),
            "w4_scale_next": {"scale": w4 + 1, "level": 12 * (w4 + 1), "over_W": 12 * (w4 + 1) > H1_W,
                              "operands": weight_range(12 * (w4 + 1))}}


def grid(levels) -> np.ndarray:
    return np.array(sorted({s * v for v in levels for s in (1, -1)}), np.int64)


def weights(arm: str, kind: str, rng, k: int, wtop: int) -> np.ndarray:
    """R, rows x k: FP8-g1's E4M3 integers to `wtop`; W4A8-g1's m q with m to `wtop` a block of 16."""
    if arm == "fp8":
        g, top = grid(v for v in E4M3I if v <= wtop), wtop
        rand = g[rng.integers(0, len(g), (T, k))]
    else:
        m = np.repeat(rng.integers(1, wtop + 1, (T, k // 16)), 16, axis=1)
        q = grid(E2M1X2)[rng.integers(0, 2 * len(E2M1X2) - 1, (T, k))]
        rand, top = m * q, wtop * 12
    return {"top": np.full((T, k), top), "bottom": np.full((T, k), -top), "opposite": np.full((T, k), top),
            "random": rand}[kind].astype(np.int64)


def acts(arm: str, kind: str, rng, k: int) -> np.ndarray:
    """V, tokens x k: FP8-g1's E4M3 integers to 120; W4A8-g1's H1's own int8 (floor, to 126)."""
    if arm == "fp8":
        g, top = grid(v for v in E4M3I if v <= 120), 120
        rand = g[rng.integers(0, len(g), (T, k))]
    else:
        rand, top = rng.integers(-H1_V, H1_V + 1, (T, k)), H1_V
    return {"top": np.full((T, k), top), "bottom": np.full((T, k), -top), "opposite": np.full((T, k), -top),
            "random": rand}[kind].astype(np.int64)


#: the weight tops checked (E4M3 integer, block scale) and the next ones up, which must be refused
TOPS = {"fp8": 104, "w4a8": 9}
CONTROLS = {"fp8": 112, "w4a8": 10}


def tile(arm: str, kind: str, k: int, seed: int, wtop: int) -> dict:
    rng = np.random.default_rng(seed)
    key = hashlib.sha512(f"g1h {arm} {kind} {k} {seed}".encode()).digest()[:32]
    nd = cs.n_digits((k // 32) * -128, (k // 32) * 127)
    R, V = weights(arm, kind, rng, k, wtop), acts(arm, kind, rng, k)
    b = cg.patch_bits(key, cg.DITHER_TAG, 0, T, k)
    c = cg.patch_bits(key, cg.C_TAG, 0, T, k)
    U = cg.u_rows16(key, 0, T)
    z = cg.z_rows(key, 0, T, nd)
    X, Ph = V + b, R + c
    Wp = Ph + U[:, np.arange(k) % 32]
    Y = X.reshape(T, k // 32, 32).sum(1).T
    A = cs.digits(Y, z, nd)
    _, D, Op = cs.co32c_words(R, V.T, U, c, b.T, z, 32, 32, nd)
    O = ((Op.astype(np.int64) - sum(dd.astype(np.int64) << (7 * (j + 1)) for j, dd in enumerate(D))) & cs.MASK)
    O = (O ^ 0x80000000) - 0x80000000
    served = Ph @ X.T
    sx, sw = X.sum(1)[None, :], Ph.sum(1)[:, None]
    if arm == "fp8":   # both centred: 4 sum (X - 1/2)(P^ - 1/2) = sum (2X - 1)(2P^ - 1)
        g1, exact = 4 * O - 2 * sx - 2 * sw + k, (2 * Ph - 1) @ (2 * X - 1).T
    else:              # dithered activations, centred weights: 2 sum X (P^ - 1/2) = sum X (2P^ - 1)
        g1, exact = 2 * O - sx, (2 * Ph - 1) @ X.T
    res = {"arm": arm, "kind": kind, "k": k, "weight_top": wtop, "digits": nd,
           "R_range": [int(R.min()), int(R.max())], "V_range": [int(V.min()), int(V.max())],
           "X_range": [int(X.min()), int(X.max())], "Wp_range": [int(Wp.min()), int(Wp.max())],
           "PA_range": [int(min(a.min() for a in A)), int(max(a.max() for a in A))],
           "served_abs_max": int(np.abs(served).max()),
           "O_is_served": bool((O == served).all()), "g1_served_exact": bool((g1 == exact).all())}
    res["ok"] = bool(res["O_is_served"] and res["g1_served_exact"]
                     and abs(res["R_range"][0]) <= H1_W and res["R_range"][1] <= H1_W
                     and -H1_V <= res["V_range"][0] and res["V_range"][1] <= H1_V
                     and H1_OP[0] <= res["X_range"][0] and res["X_range"][1] <= H1_OP[1]
                     and H1_OP[0] <= res["Wp_range"][0] and res["Wp_range"][1] <= H1_OP[1]
                     and -128 <= res["PA_range"][0] and res["PA_range"][1] <= 127
                     and res["served_abs_max"] < 2 ** 31)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    r = ranges()
    print(json.dumps(r))
    ok = (r["fp8_weight_top"] == 104 and not r["fp8_weight_next"]["fits"] and r["fp8_act_top"] == 120
          and r["w4_scale_top"] == 9 and r["w4_scale_next"]["over_W"])
    tiles, controls = [], []
    for arm in ("fp8", "w4a8"):
        for kind in ("top", "bottom", "opposite", "random"):
            for k in (4096, 14336):
                t = tile(arm, kind, k, len(tiles), TOPS[arm])
                tiles.append(t)
                print(json.dumps(t), flush=True)
                ok &= t["ok"]
        c = tile(arm, "top", 4096, 100 + len(controls), CONTROLS[arm])
        c["rejected"] = not c["ok"]
        controls.append(c)
        print("control", json.dumps(c), flush=True)
        ok &= c["rejected"] and c["O_is_served"] and c["g1_served_exact"]
    if a.json:
        Path(a.json).write_text(json.dumps({"ranges": r, "tiles": tiles, "controls": controls, "ok": bool(ok)},
                                           indent=1) + "\n")
    print("ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
