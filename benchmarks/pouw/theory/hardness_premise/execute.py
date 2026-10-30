"""Run a `witness.py` schedule exactly on integers, and count its W1 work at any shape (HT-PREMISE, P1 and P2).

The schedule computes e·Rᵀ with e over `a` consecutive values (the noise after its affine shift; one output write maps
it back to E·Rᵀ) and R uniform int8. Operand value intervals are tracked symbolically through every Strassen or
Winograd form and digit split, so the shifts an atom needs, and their corrections, are the ones the worst-case
operand needs; the exact run asserts the data stays inside them and that every leaf operand fits its format.

Counted work, priced in W1 (CORE.md §2a; H200 measured table, per register written):
- MMA MACs at the format's price (s8 1, b1 0.125, f16 2, nvf4 0.5 from the sm_120 sheet, e4m3 1);
- output-side additions: per scheme node Σ_t uses_t − 4 − #single-use products (each output starts from one
  product's accumulator, single-use products chain onto theirs), per digit split one IMAD, per shifted atom operand
  one correction; reported at one write per addition (IADD) and at one write per two additions (IADD3);
- a float leaf drains its fp32 accumulator to int32 before 2^24 (and an e4m3 leaf on the H200 every k32): two writes
  per drain an output;
- noise-side additions (forming combinations, digits, row sums), four int8 lanes a register.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

S_U = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
S_V = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
S_W = [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]]
W_U = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, -1, -1], [0, 0, 0, 1], [0, 0, 1, 1], [-1, 0, 1, 1], [1, 0, -1, 0]]
W_V = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, -1, 1], [-1, 1, 0, 0], [1, -1, 0, 1], [0, -1, 0, 1]]
W_W = [[1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0], [1, 0, 0, -1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1]]
SCHEMES = {"strassen": (S_U, S_V, S_W), "winograd": (W_U, W_V, W_W)}

# integer window each format holds exactly, its W1 price a MAC, and its accumulator: (k, bits) for float, None for int
FORMATS = {
    "s8": ((-128, 127), 1.0, None),
    "b1": ((0, 1), 0.125, None),
    "e4m3": ((-16, 16), 1.0, (32, 14)),
    "e5m2": ((-8, 8), 1.0, (32, 14)),
    "e2m1": ((-4, 4), 1.0, (32, 24)),
    "nvf4": ((-4, 4), 0.5, (64, 24)),
    "f16": ((-2048, 2048), 2.0, (16, 24)),
    "bf16": ((-256, 256), 2.0, (16, 24)),
    "tf32": ((-2048, 2048), 4.0, (8, 24)),
}


class Ops:
    def __init__(self):
        self.mac = {}
        self.out_adds = 0
        self.e_adds = 0
        self.drain_writes = 0

    def w1(self, iadd3: bool) -> float:
        mma = sum(FORMATS[f][1] * c for f, c in self.mac.items())
        out = self.out_adds / 2 if iadd3 else self.out_adds
        return mma + 32 * (out + self.e_adds / 4 + self.drain_writes)


def _form(iv, coeffs):
    lo = sum(min(c * iv[p][0], c * iv[p][1]) for p, c in enumerate(coeffs) if c)
    hi = sum(max(c * iv[p][0], c * iv[p][1]) for p, c in enumerate(coeffs) if c)
    return (lo, hi)


def _blocks(M):
    h, w = M.shape[0] // 2, M.shape[1] // 2
    return [M[:h, :w], M[:h, w:], M[h:, :w], M[h:, w:]]


def _shift(iv, win):
    """The shift s with iv − s inside win (0 if it already is), or None if iv is wider than win."""
    if iv[1] - iv[0] > win[1] - win[0]:
        return None
    if win[0] <= iv[0] and iv[1] <= win[1]:
        return 0
    return iv[0] - win[0]


def run(node, A, B, ia, ib, shape, ops):
    """A·B for E-side A (m×K) and R-side B (K×n) with value intervals ia, ib; A, B None counts only."""
    m, K, n = shape
    if "atom" in node:
        win, _price, acc = FORMATS[node["atom"]]
        sa, sb = _shift(ia, win), _shift(ib, win)
        assert sa is not None and sb is not None, (node, ia, ib)
        ops.mac[node["atom"]] = ops.mac.get(node["atom"], 0) + m * K * n
        ops.out_adds += ((sa != 0) + (sb != 0)) * m * n
        if sb:
            ops.e_adds += m * K  # row sums of the noise operand
        if acc is not None:
            k, bits = acc
            amax = max(abs(ia[0] - sa), abs(ia[1] - sa))
            bmax = max(abs(ib[0] - sb), abs(ib[1] - sb))
            assert k * amax * bmax <= 2**bits, (node, ia, ib)  # one instruction's k products sum exactly
            chunk = max(k, (2**bits // max(1, amax * bmax)) // k * k)
            ops.drain_writes += 2 * math.ceil(K / chunk) * m * n
        if A is None:
            return None
        A2, B2 = A - sa, B - sb
        assert A2.min() >= win[0] and A2.max() <= win[1] and B2.min() >= win[0] and B2.max() <= win[1]
        if acc is not None:
            assert K * amax * bmax < 2**53
        Y = A2 @ B2
        if sa:
            Y = Y + sa * B2.sum(axis=0)[None, :]
        if sb:
            Y = Y + sb * A2.sum(axis=1)[:, None]
        if sa and sb:
            Y = Y + sa * sb * K
        return Y
    if "scheme" in node:
        assert node["roles"] == "AB", node["roles"]
        U, V, W = SCHEMES[node["scheme"]]
        assert m % 2 == 0 and K % 2 == 0 and n % 2 == 0, shape
        sub = (m // 2, K // 2, n // 2)
        iA = [ia] * 4
        iB = [ib] * 4
        Ab = _blocks(A) if A is not None else None
        Bb = _blocks(B) if B is not None else None
        M = []
        for t, (u, v) in enumerate(zip(U, V)):
            ops.e_adds += (sum(map(abs, u)) - 1) * sub[0] * sub[1]
            a = sum(c * Ab[p] for p, c in enumerate(u) if c) if Ab is not None else None
            b = sum(c * Bb[q] for q, c in enumerate(v) if c) if Bb is not None else None
            M.append(run(node["kids"][t], a, b, _form(iA, u), _form(iB, v), sub, ops))
        uses = [sum(1 for w in W if w[t]) for t in range(7)]
        ops.out_adds += (sum(uses) - 4 - sum(1 for u in uses if u == 1)) * sub[0] * sub[2]
        if A is None:
            return None
        C = [sum(c * M[t] for t, c in enumerate(w) if c) for w in W]
        return np.block([[C[0], C[1]], [C[2], C[3]]])
    d = node["d"]
    lo_kid, hi_kid = node["kids"]
    if node["split"] == "b":
        c = ib[0]
        i0, i1 = (0, d - 1), (0, (ib[1] - c) // d)
        ops.out_adds += (1 + (c != 0)) * m * n
        if c:
            ops.e_adds += m * K
        if B is None:
            run(lo_kid, None, None, ia, i0, shape, ops)
            run(hi_kid, None, None, ia, i1, shape, ops)
            return None
        x0, x1 = (B - c) % d, (B - c) // d
        Y = run(lo_kid, A, x0, ia, i0, shape, ops) + d * run(hi_kid, A, x1, ia, i1, shape, ops)
        return Y + c * A.sum(axis=1)[:, None] if c else Y
    c = ia[0]
    i0, i1 = (0, d - 1), (0, (ia[1] - c) // d)
    ops.out_adds += (1 + (c != 0)) * m * n
    ops.e_adds += 2 * m * K  # the two digits of the noise operand
    if A is None:
        run(lo_kid, None, None, i0, ib, shape, ops)
        run(hi_kid, None, None, i1, ib, shape, ops)
        return None
    x0, x1 = (A - c) % d, (A - c) // d
    Y = run(lo_kid, x0, B, i0, ib, shape, ops) + d * run(hi_kid, x1, B, i1, ib, shape, ops)
    return Y + c * B.sum(axis=0)[None, :] if c else Y


def depth(node) -> int:
    return max([depth(k) + ("scheme" in node) for k in node.get("kids", [])], default=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("witness", type=Path, nargs="+")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    out, ok = {}, True
    for path in args.witness:
        w = json.loads(path.read_text())
        a, b = w["a"], w["b"]
        assert b == 256
        L = depth(w)
        side = 2 ** L * 2
        rec = {"model_cost_eighths": w["cost"], "scheme_depth": L, "exact": []}
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            e = rng.integers(0, a, (side, side), dtype=np.int64)
            R = rng.integers(-128, 128, (side, side), dtype=np.int64)
            R[0, 0], R[-1, -1] = -128, 127
            ops = Ops()
            Y = run(w, e, R.T, (0, a - 1), (-128, 127), (side, side, side), ops)
            rec["exact"].append(bool(np.array_equal(Y, e @ R.T)))
        ok &= all(rec["exact"])
        tensor = sum(FORMATS[f][1] * c for f, c in ops.mac.items()) / side**3
        rec["tensor_w1_per_mac"] = tensor
        rec["mac_mix"] = {f: c / side**3 for f, c in ops.mac.items()}
        for (m, K, n) in ((4096, 2048, 4096), (8192, 8192, 8192), (8192, 28672, 8192), (16384, 53248, 16384),
                          (65536, 53248, 16384)):
            o = Ops()
            run(w, None, None, (0, a - 1), (-128, 127), (m, K, n), o)
            o.out_adds += m * n  # e·Rᵀ back to E·Rᵀ
            macs = m * K * n
            rec[f"m{m}_K{K}_n{n}"] = {
                "w1_iadd": o.w1(False) / macs,
                "w1_iadd3": o.w1(True) / macs,
                "out_adds_per_output": o.out_adds / (m * n),
                "drain_writes_per_output": o.drain_writes / (m * n),
            }
        out[path.stem] = rec
        print(path.stem, json.dumps(rec), file=sys.stderr, flush=True)
    out["ok"] = bool(ok)
    text = json.dumps(out, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
