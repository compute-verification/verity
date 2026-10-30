"""Run a `witness.py` schedule exactly on integers, and count its W1 work at any shape (HT-PREMISE, P1 and P2).

The schedule computes e·Rᵀ with e over `a` consecutive values (the noise after its affine shift; one output write maps
it back to E·Rᵀ) and R uniform int8. Operand value intervals are tracked symbolically through every Strassen or
Winograd form and digit split, so the shifts an atom needs, and their corrections, are the ones the worst-case
operand needs; the exact run asserts the data stays inside them and that every leaf operand fits its format.

Counted work, priced in W1 (CORE.md §2a; H200 measured table, per register written):
- MMA MACs at the format's price (s8 1, b1 0.125, f16 2, nvf4 0.5 from the sm_120 sheet, e4m3 1);
- output-side writes: per scheme node its output program (`PROGRAMS`, executed here), per digit split one IMAD, per
  shifted atom operand one correction. A program's chain step starts a single-use product's accumulator from an
  earlier sum, which is free only when that product's schedule ends in an MMA accumulator (`chainable`), else one
  write;
- a float leaf drains its fp32 accumulator to int32 before 2^24 (and an e4m3 leaf on the H200 every k32): two writes
  per drain an output;
- noise-side additions (forming combinations, digits, row sums).

Three readings: `fast` takes the IADD3 programs and four int8 lanes a register on the noise side (the attacker's best
case), `plain` the IADD programs and two int16 lanes (the noise combinations outgrow int8 below the top levels), and
`spilled` is `plain` plus one L1-priced reload (128) for each output write of a node above `FUSED_LEVELS`. Operands
reach the MMA through shared memory (TMA), which writes no register, as for the honest kernel.
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
# Winograd with product 3's R form negated (same interval, [−510, 510]), so every single-use product enters with +1
W_U = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, -1, -1], [0, 0, 0, 1], [0, 0, 1, 1], [-1, 0, 1, 1], [1, 0, -1, 0]]
W_V = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 1, -1], [-1, 1, 0, 0], [1, -1, 0, 1], [0, -1, 0, 1]]
W_W = [[1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 0, 1, 1, 1]]
SCHEMES = {"strassen": (S_U, S_V, S_W), "winograd": (W_U, W_V, W_W)}

# Output programs: ("sum", dest, ((sign, src), ...)) writes dest; ("chain", dest, t, base) runs product t's MMA from
# base's accumulator (D ≠ C, so base survives and product t alone is gone). A node takes, per add instruction, the
# cheapest of its scheme's programs given which of its products are `chainable`. Sums are written for IADD (2
# operands) and cost ⌈(terms − 1)/2⌉ writes under IADD3.
PROGRAMS = {
    "strassen": (
        (("sum", "C1", ((1, "M2"), (1, "M4"))), ("sum", "C2", ((1, "M1"), (1, "M3"))),
         ("sum", "Z", ((1, "M0"), (-1, "M1"), (-1, "M4"))), ("sum", "P0", ((1, "Z"), (1, "C2"))),
         ("chain", "C0", 6, "P0"), ("sum", "P3", ((1, "Z"), (1, "C1"))), ("chain", "C3", 5, "P3")),
        (("sum", "C1", ((1, "M2"), (1, "M4"))), ("sum", "C2", ((1, "M1"), (1, "M3"))),
         ("sum", "P0", ((1, "M0"), (1, "M3"), (-1, "M4"))), ("chain", "C0", 6, "P0"),
         ("sum", "P3", ((1, "M0"), (-1, "M1"), (1, "M2"))), ("chain", "C3", 5, "P3")),
        (("chain", "C1", 2, "M4"), ("chain", "C2", 1, "M3"), ("sum", "Q", ((1, "M0"), (1, "M3"), (-1, "M4"))),
         ("chain", "C0", 6, "Q"), ("sum", "P3", ((1, "Q"), (-1, "C2"), (1, "C1"))), ("chain", "C3", 5, "P3")),
    ),
    "winograd": (
        (("sum", "S", ((1, "M0"), (1, "M5"))), ("sum", "U", ((1, "S"), (1, "M4"))), ("chain", "C1", 2, "U"),
         ("sum", "T", ((1, "S"), (1, "M6"))), ("chain", "C2", 3, "T"), ("sum", "C3", ((1, "T"), (1, "M4"))),
         ("chain", "C0", 1, "M0")),
        (("sum", "U", ((1, "M0"), (1, "M5"), (1, "M4"))), ("chain", "C1", 2, "U"),
         ("sum", "T", ((1, "M0"), (1, "M5"), (1, "M6"))), ("chain", "C2", 3, "T"),
         ("sum", "C3", ((1, "T"), (1, "M4"))), ("chain", "C0", 1, "M0")),
        (("chain", "S", 5, "M0"), ("chain", "T", 6, "S"), ("chain", "U", 4, "S"), ("chain", "C1", 2, "U"),
         ("chain", "C2", 3, "T"), ("sum", "C3", ((1, "T"), (1, "U"), (-1, "S"))), ("chain", "C0", 1, "M0")),
    ),
}
# Above this many scheme levels the products' accumulators can't all stay in registers (4^L + 1 tiles), so the
# `spilled` reading charges each output write of such a node one reload at the L1 price.
FUSED_LEVELS = 2


def _eval(prog, M):
    env = {f"M{t}": m for t, m in enumerate(M)}
    chained, standalone = set(), set()
    for step in prog:
        if step[0] == "sum":
            srcs = [src for _, src in step[2]]
            env[step[1]] = sum(s * env[src] for s, src in step[2])
        else:
            assert f"M{step[2]}" not in standalone | chained, step
            chained.add(f"M{step[2]}")
            srcs = [step[3]]
            env[step[1]] = env[f"M{step[2]}"] + env[step[3]]
        assert not chained & set(srcs), step  # a chained product exists only inside its chain's result
        standalone |= {s for s in srcs if s.startswith("M")}
    return [env[f"C{i}"] for i in range(4)]


for _name, (_u, _v, _w) in SCHEMES.items():
    for _prog in PROGRAMS[_name]:
        assert [list(c) for c in _eval(_prog, list(np.eye(7, dtype=int)))] == _w, _name


def chainable(node) -> bool:
    """Whether a product computed by `node` can start its accumulator from any value at no write."""
    if "atom" in node:
        return True
    return "split" in node and chainable(node["kids"][0])


def program_writes(prog, kids, iadd3: bool) -> int:
    n = 0
    for step in prog:
        if step[0] == "sum":
            n += -(-(len(step[2]) - 1) // 2) if iadd3 else len(step[2]) - 1
        else:
            n += not chainable(kids[step[2]])
    return n


def heights(node) -> int:
    """Annotate every node with its scheme levels at and below it (`_h`)."""
    node["_h"] = max([heights(k) for k in node.get("kids", [])], default=0) + ("scheme" in node)
    return node["_h"]

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
        self.out = {"iadd": 0, "iadd3": 0, "spilled": 0}
        self.e_adds = 0
        self.drain_writes = 0

    def add_out(self, n: int, node=None, iadd3_n=None):
        self.out["iadd"] += n
        self.out["iadd3"] += n if iadd3_n is None else iadd3_n
        if node is not None and node["_h"] > FUSED_LEVELS:
            self.out["spilled"] += n

    def w1(self, reading: str) -> float:
        mma = sum(FORMATS[f][1] * c for f, c in self.mac.items())
        fast = reading == "fast"
        out = self.out["iadd3" if fast else "iadd"]
        reload = 128 * self.out["spilled"] if reading == "spilled" else 0
        return mma + 32 * (out + self.e_adds / (4 if fast else 2) + self.drain_writes) + reload


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
        ops.add_out(((sa != 0) + (sb != 0)) * m * n)
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
        U, V, _ = SCHEMES[node["scheme"]]
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
        progs = PROGRAMS[node["scheme"]]
        writes = min(program_writes(p, node["kids"], False) for p in progs)
        writes3 = min(program_writes(p, node["kids"], True) for p in progs)
        ops.add_out(writes * sub[0] * sub[2], node, writes3 * sub[0] * sub[2])
        if A is None:
            return None
        C, *rest = (_eval(p, M) for p in progs)
        assert all(np.array_equal(x, y) for other in rest for x, y in zip(C, other))
        return np.block([[C[0], C[1]], [C[2], C[3]]])
    d = node["d"]
    lo_kid, hi_kid = node["kids"]
    if node["split"] == "b":
        c = ib[0]
        i0, i1 = (0, d - 1), (0, (ib[1] - c) // d)
        ops.add_out((1 + (c != 0)) * m * n, node)
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
    ops.add_out((1 + (c != 0)) * m * n, node)
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
    ap.add_argument("--shapes", default="4096,2048,4096 8192,8192,8192 8192,28672,8192 16384,53248,16384",
                    help="space-separated m,K,n to count W1 at (each divisible by 2^depth)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes.split()]
    out, ok = {}, True
    for path in args.witness:
        w = json.loads(path.read_text())
        a, b = w["a"], w["b"]
        assert b == 256
        L = depth(w)
        heights(w)
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
        for (m, K, n) in shapes:
            o = Ops()
            run(w, None, None, (0, a - 1), (-128, 127), (m, K, n), o)
            o.add_out(m * n)  # e·Rᵀ back to E·Rᵀ
            macs = m * K * n
            rec[f"m{m}_K{K}_n{n}"] = {
                "w1_fast": o.w1("fast") / macs,
                "w1_plain": o.w1("plain") / macs,
                "w1_spilled": o.w1("spilled") / macs,
                "out_writes_per_output": {k: v / (m * n) for k, v in o.out.items()},
                "drain_writes_per_output": o.drain_writes / (m * n),
                "noise_adds_per_noise_entry": o.e_adds / (m * K),
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
