"""Conformance vectors for `Definitions.Pouw.Fp8Atom.H1T`: H-1T's forming map, salt fields and checked chain, as Lean data.

    PYTHONPATH=<verity> python scripts/h1t_vectors.py <hardness vectors directory> [--out DIR]

The expected values come from an exact transcription of `docs/pouw/hardness-shaped-matrices.md` §3.1 over fractions,
with the atom from `verity.ml.circuits.silicon` (`tc_dot_total_e4m3` on `HOPPER_E4M3_K32`) and the chain in numpy float32. Before
anything is written, three other computations must agree with it:
* Track H's `h1_tagged.h1t_ref`, driven by a stub generator that returns the vectors' salt fields, and Track H's
  `h1_tagged.chain` on the operands it forms (the pinned map, bit for bit);
* a float16 simulation of the kernel `h1t_form_xq` (`h1_kernel.py`): the salt fields from the raw XOF words through its
  masks, and the real lanes through its f16 `max`, `mul` and `sub`, then `cvt.rn.satfinite.e4m3`. Where the f16
  intermediate is inexact, the simulation's codes are compared and the cases counted;
* `HOPPER_E4M3_K32.step` on every atom.

Kernel-pinned, as Lean data (`Definitions/Pouw/Fp8Atom/H1TVectors.lean`, checks in
`Proofs/Pouw/Fp8Atom/H1TChecks.lean`):
* `formTable`: `formReal` on every finite code, both signs and every `F` in {1/4, 1/2, 1, 2, 4} (2,540 lanes);
* `tagTable`: every tag value, its code and its negation's (80);
* `saltVectors`: raw XOF words and the fields `extract` reads;
* `rowVectors`: full rows (T <= 3) through the forming, the atoms and the chain, every I and R word of one column.
`Definitions/Pouw/Fp8Atom/H1TCoverage.json` records the families and the cross-checks.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from verity.ml.circuits.silicon import models
from verity.ml.circuits.silicon.total_fp8 import tc_dot_total_e4m3

P = models.HOPPER_E4M3_K32
SEED = 2026092901
PART = 100
ROW_PART = 4
FINITE = [c for c in range(256) if c & 0x7F != 0x7F]
F_VALUES = [Q(1, 4), Q(1, 2), Q(1), Q(2), Q(4)]
A0 = list(range(0x68, 0x7F))
A1 = [c | s for s in (0, 0x80) for c in range(0x60, 0x7F)]


# ---------------------------------------------------------------------------------------------------------------
# the exact transcription (§3.1)
# ---------------------------------------------------------------------------------------------------------------

def code_val(c: int) -> Q:
    e, m = (c >> 3) & 0xF, c & 7
    sig = m if e == 0 else 8 + m
    v = Q(sig) * Q(2) ** (max(e, 1) - 10)
    return -v if c & 0x80 else v


MAG = [code_val(c) for c in range(127)]


def enc(y: Q) -> int:
    """RNE on the E4M3 grid, ties to the even code, clamped to +-448."""
    if y == 0:
        return 0
    a = abs(y)
    if a >= 448:
        c = 126
    else:
        lo = max(c for c in range(127) if MAG[c] <= a)
        if MAG[lo] == a:
            c = lo
        else:
            d1, d2 = a - MAG[lo], MAG[lo + 1] - a
            c = lo if d1 < d2 else lo + 1 if d2 < d1 else (lo if lo % 2 == 0 else lo + 1)
    return c | (0x80 if y < 0 else 0)


def binade(x: Q) -> int:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Q(2) ** e > x:
        e -= 1
    while Q(2) ** (e + 1) <= x:
        e += 1
    return e


def floor_f(xs) -> Q:
    M = max(abs(code_val(c)) for c in xs)
    return Q(1, 4) if M == 0 else max(Q(2) ** (binade(M) - 6), Q(1, 4))


def mu(F: Q, x: Q) -> Q:
    p = Q(0) if x == 0 else Q(2) ** binade(abs(x))
    return max(p / 8, F)


def form_real(F: Q, c: int, pos: bool) -> tuple[int, int]:
    s = 1 if pos else -1
    x = code_val(c)
    return enc(x + s * mu(F, x)), enc(-(s * mu(F, x)))


def tag_val(F: Q, neg: bool, m: int) -> Q:
    return (-1 if neg else 1) * 32 * F * (1 + Q(m, 8))


def fields(r) -> tuple[list[bool], list[bool], list[int]]:
    pos = [bool((r[l // 2] >> (15 if l % 2 == 0 else 31)) & 1) for l in range(29)]
    neg = [bool((r[17] >> 31) & 1), bool((r[16] >> 15) & 1), bool((r[16] >> 31) & 1)]
    man = [(r[17] >> 23) & 7, (r[16] >> 7) & 7, (r[16] >> 23) & 7]
    return pos, neg, man


def form_slice(xs, r) -> tuple[list[int], list[int]]:
    pos, neg, man = fields(r)
    F = floor_f(xs)
    x1, x2 = zip(*(form_real(F, xs[l], pos[l]) for l in range(29)))
    t = [tag_val(F, neg[i], man[i]) for i in range(3)]
    return list(x1) + [enc(v) for v in t], list(x2) + [enc(-v) for v in t]


def tuple_codes(j: int) -> list[int]:
    return [A0[j % 23], A1[j // 23 % 62], A1[j // 1426 % 62]]


def f32_bits(v: np.float32) -> int:
    return int(np.array([v], np.float32).view(np.uint32)[0])


def atom(a, b) -> int:
    d = tc_dot_total_e4m3(P, 0, a, b)
    assert d == P.step(0, a, b)
    return d


def row_words(T: int, x, b, j: int, raws):
    """-> (I words, R words R_0..R_(2T-1)) as FP32 bit patterns, and the operands (x1, x2, lanes of the column)."""
    ops, cols = [], []
    for tau in range(T):
        ops.append(form_slice(x[29 * tau:29 * tau + 29], raws[tau]))
        cols.append(list(b[29 * tau:29 * tau + 29]) + tuple_codes(j))
    I = [atom(ops[p][0], cols[p]) for p in range(T)] + [atom(ops[p][1], cols[p]) for p in range(T)]
    vals = np.array(I, np.uint32).view(np.float32)
    R = [I[0]]
    acc = vals[0]
    for p in range(1, 2 * T):
        acc = np.float32(acc + vals[p])
        R.append(f32_bits(acc))
    return I, R, ops, cols


# ---------------------------------------------------------------------------------------------------------------
# the cross-checks: Track H's h1t_ref and chain, and the kernel's f16 path
# ---------------------------------------------------------------------------------------------------------------

class StubRng:
    """Returns the given arrays to `h1t_ref`'s draws, in its order: real signs, tag signs, tag mantissas."""

    def __init__(self, draws):
        self.draws = list(draws)

    def _next(self, size):
        a = self.draws.pop(0)
        assert a.shape == tuple(size), (a.shape, size)
        return a

    def choice(self, _vals, size):
        return self._next(size)

    def integers(self, _lo, _hi, size):
        return self._next(size)


def trackh_row(h1t, T: int, x, b, j: int, raws, I, R):
    """Track H's `h1t_ref` with the row's salt fields and its `chain` on one column: its words must be ours."""
    fs = [fields(r) for r in raws]
    s = np.array([[[1.0 if f[0][l] else -1.0 for l in range(29)] for f in fs]])
    ts = np.array([[[-1.0 if f[1][i] else 1.0 for i in range(3)] for f in fs]])
    tm = np.array([[[f[2][i] for i in range(3)] for f in fs]])
    x1, x2 = h1t.h1t_ref(np.array([x], np.uint8), StubRng([s, ts, tm]))
    Wr = np.array(b, np.uint8).reshape(29 * T, 1)
    t = np.array(tuple_codes(j), np.uint8).reshape(3, 1)
    W = np.concatenate([Wr.reshape(T, 29, 1), np.broadcast_to(t[None], (T, 3, 1))], 1).reshape(32 * T, 1)
    w = h1t.chain(x1, x2, W)[0, 0]
    ours = np.concatenate([np.array(I, np.uint32).view(np.float32), np.array(R[1:], np.uint32).view(np.float32)])
    assert np.array_equal(w.view(np.uint32), ours.view(np.uint32)) or np.array_equal(w, ours), (w, ours)
    return x1[0], x2[0]


def h16(v) -> np.float16:
    return np.float16(v)


def bits16(v: np.float16) -> int:
    return int(np.array([v], np.float16).view(np.uint16)[0])


def of16(u: int) -> np.float16:
    return np.array([u], np.uint16).view(np.float16)[0]


def kernel_slice(xs, r) -> tuple[list[int], list[int]]:
    """`h1t_form_xq`'s f16 path for one slice: masks on the raw words, f16 max/mul/sub, then cvt.rn.satfinite.e4m3."""
    h = [h16(float(code_val(c))) for c in xs] + [h16(0.0)] * 3
    M = max(abs(float(v)) for v in h)
    KE, E8 = 0x7C00, None
    Mb = bits16(h16(M)) & KE
    E8 = max(float(np.float16(of16(Mb) * np.float16(0.125))), 2.0)
    V = np.float16(np.float16(E8) * np.float16(4.0))
    Vb = bits16(V)
    r16, r17 = r[16], r[17]
    v30 = of16(Vb | (r16 & 0x8380))
    v31 = of16(Vb | ((r16 >> 16) & 0x8380))
    v29 = of16(Vb | ((r17 >> 16) & 0x8380))
    x1, x2 = [], []
    for l in range(29):
        half = (r[l // 2] >> (0 if l % 2 == 0 else 16)) & 0xFFFF
        p = of16(bits16(h[l]) & KE)
        u = np.float16(max(float(p), E8))
        t = of16(bits16(u) | (half & 0x8000))
        y = np.float16(t * np.float16(0.125))
        z = np.float16(h[l] - y)
        x1.append(enc(Q(float(z))))
        x2.append(enc(Q(float(y))))
    for v in (v29, v30, v31):
        x1.append(enc(Q(float(v))))
        x2.append(enc(-Q(float(v))))
    return x1, x2


# ---------------------------------------------------------------------------------------------------------------
# the families
# ---------------------------------------------------------------------------------------------------------------

def rand_raw(rng: random.Random):
    return [rng.getrandbits(32) for _ in range(18)]


def salt_family(rng: random.Random):
    out = [("zero", [0] * 18), ("ones", [0xFFFFFFFF] * 18)]
    for b in (0, 7, 8, 9, 14, 15, 16, 22, 23, 24, 25, 26, 30, 31):
        out.append((f"bit{b}", [1 << b] * 18))
    for w in range(18):
        out.append((f"word{w}", [0xFFFFFFFF if i == w else 0 for i in range(18)]))
    out += [("random", rand_raw(rng)) for _ in range(40)]
    return out


def row_family(rng: random.Random):
    fin = FINITE
    sub = [c for c in fin if (c >> 3) & 0xF == 0 and c & 7]
    out = []

    def row(fam, T, x, b, j, raws=None):
        out.append((fam, T, x, b, j, raws or [rand_raw(rng) for _ in range(T)]))

    for T in (1, 2, 3):
        for _ in range(3):
            row("random", T, [rng.choice(fin) for _ in range(29 * T)], [rng.choice(fin) for _ in range(29 * T)],
                rng.randrange(88412))
    for T, k in ((2, 40), (3, 60), (1, 1)):
        x = [rng.choice(fin) for _ in range(k)] + [0] * (29 * T - k)
        b = [rng.choice(fin) for _ in range(k)] + [0] * (29 * T - k)
        row("padded", T, x, b, rng.randrange(88412))
    for T in (2, 3):
        row("loud448", T, [rng.choice((0x7E, 0xFE)) for _ in range(29 * T)], [0x7E] * (29 * T), rng.randrange(88412))
        row("loud416", T, [0x7D] * (29 * T), [rng.choice((0x7E, 0xFE)) for _ in range(29 * T)], rng.randrange(88412))
    for T in (2, 3):
        row("zero-row", T, [0] * (29 * T), [rng.choice(fin) for _ in range(29 * T)], rng.randrange(88412))
    row("zero-both", 2, [0] * 58, [0] * 58, 0)
    for T in (1, 2):
        row("subnormal", T, [rng.choice(sub) for _ in range(29 * T)], [rng.choice(fin) for _ in range(29 * T)],
            rng.randrange(88412))
    for j in (0, 22, 23, 1425, 1426, 88411):
        row("tuple-edge", 2, [rng.choice(fin) for _ in range(58)], [rng.choice(fin) for _ in range(58)], j)
    x = [rng.choice(fin) for _ in range(58)]
    b = [rng.choice(fin) for _ in range(58)]
    raws = [rand_raw(rng) for _ in range(2)]
    for j in (5, 6, 5 + 23, 5 + 1426):
        row("same-row", 2, x, b, j, raws)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Lean
# ---------------------------------------------------------------------------------------------------------------

HEADER = "/-! Generated by `scripts/h1t_vectors.py`; do not edit. -/\n"


def lean_bool(b: bool) -> str:
    return "true" if b else "false"


def lean_list(xs, fmt=str) -> str:
    return "[" + ", ".join(fmt(x) for x in xs) + "]"


def hexw(w: int) -> str:
    return f"0x{w:08X}"


def pack(xs, bits: int) -> str:
    """Little-endian: element i is bits (bits * i)..; as `codes` in `Pouw.Fp8Atom.Atom`."""
    n = sum(v << (bits * i) for i, v in enumerate(xs))
    return f"0x{n:0{max(1, (bits * len(xs) + 3) // 4)}X}"


def parts(name: str, typ: str, items: list[str], doc: str, size: int = PART):
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    defs = []
    for n, c in enumerate(chunks):
        defs.append(f"/-- {doc}, part {n}. -/\ndef {name}{n} : List {typ} := [\n  " + ",\n  ".join(c) + "]\n")
    return defs, [f"{name}{n}" for n in range(len(chunks))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h1_dir")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[6] / "Security"))
    args = ap.parse_args()
    sys.path.insert(0, args.h1_dir)
    import h1_tagged as h1t  # noqa: E402

    rng = random.Random(SEED)
    out = Path(args.out)
    cov: dict = {"seed": SEED}

    # the per-lane table, against Track H's enc and the kernel's f16 path
    form = []
    f16_inexact, f16_differs, reachable = 0, 0, 0
    for x in FINITE:
        for pos in (False, True):
            for fi, F in enumerate(F_VALUES):
                x1, x2 = form_real(F, x, pos)
                s = 1.0 if pos else -1.0
                xv = float(code_val(x))
                m = max((2.0 ** np.floor(np.log2(abs(xv))) if xv else 0.0) / 8, float(F))
                assert (int(h1t.enc(np.array([xv + s * m]))[0]), int(h1t.enc(np.array([-s * m]))[0])) == \
                    (x1 if x1 != 0x80 else 0, x2), (x, pos, F)
                if F >= max(Q(1, 4), Q(2) ** (binade(abs(code_val(x))) - 6) if x & 0x7F else Q(0)):
                    # a slice whose floor is F: x, and a lane at 2^(log2 F + 6) when F > 1/4
                    xs = [x] + ([enc(64 * F)] if F > Q(1, 4) else []) + [0] * 28
                    xs = xs[:29]
                    assert floor_f(xs) == F, (x, F)
                    k1, k2 = kernel_slice(xs, [0xFFFFFFFF if pos else 0] + [0] * 17)
                    exact = Q(float(np.float16(xv) - np.float16(-s * m))) == code_val(x) + s * Q(m)
                    f16_inexact += not exact
                    f16_differs += (k1[0], k2[0]) != (x1, x2)
                    reachable += 1
                form.append((x, pos, fi, x1, x2))
    cov["formTable"] = {"lanes": len(form), "reachable_lanes_kernel_compared": reachable,
                        "f16_intermediate_inexact_reachable": f16_inexact, "kernel_f16_codes_differ": f16_differs}

    tags = []
    for fi, F in enumerate(F_VALUES):
        for neg in (False, True):
            for m in range(8):
                v = tag_val(F, neg, m)
                c1, c2 = enc(v), enc(-v)
                assert code_val(c1) == v and code_val(c2) == -v
                tags.append((fi, neg, m, c1, c2))
    cov["tagTable"] = len(tags)

    salts = []
    for fam, r in salt_family(rng):
        pos, neg, man = fields(r)
        k1, k2 = kernel_slice([0x38] * 29, r)
        o1, o2 = form_slice([0x38] * 29, r)
        assert (k1, k2) == (o1, o2), fam
        salts.append((fam, r, sum(1 << l for l in range(29) if pos[l]), sum(1 << i for i in range(3) if neg[i]),
                      sum(man[i] << (3 * i) for i in range(3))))
    cov["saltVectors"] = len(salts)

    rows = []
    kernel_rows = 0
    for fam, T, x, b, j, raws in row_family(rng):
        I, R, ops, cols = row_words(T, x, b, j, raws)
        trackh_row(h1t, T, x, b, j, raws, I, R)
        for tau in range(T):
            ks = kernel_slice(x[29 * tau:29 * tau + 29], raws[tau])
            kernel_rows += ks == (ops[tau][0], ops[tau][1])
        rows.append((fam, T, x, b, j, raws, I, R))
    cov["rowVectors"] = {"rows": len(rows), "families": sorted({r[0] for r in rows}),
                         "atoms": sum(2 * r[1] for r in rows), "kernel_f16_slices_agree": kernel_rows,
                         "slices": sum(r[1] for r in rows)}

    body = ["import Definitions.Pouw.Fp8Atom.H1TVec", "", HEADER, "namespace Pouw.Fp8Atom.H1T.Vectors", ""]
    names = {}
    for name, typ, items, doc in (
            ("form", "FormVec", [f"⟨{x}, {lean_bool(p)}, {fi}, {a}, {c}⟩" for x, p, fi, a, c in form],
             "`formReal` on every finite code, both signs and every `F`"),
            ("tag", "TagVec", [f"⟨{fi}, {lean_bool(n)}, {m}, {a}, {c}⟩" for fi, n, m, a, c in tags],
             "Every tag value's code and its negation's"),
            ("salt", "SaltVec", [f"⟨{pack(r, 32)}, {hexw(p)}, {n}, {m}⟩" for _, r, p, n, m in salts],
             "Raw XOF words and the fields `extract` reads"),
            ("row", "RowVec", [f"⟨{T}, {j}, {pack(x, 8)}, {pack(b, 8)}, "
                               f"{lean_list(raws, lambda r: pack(r, 32))}, {lean_list(I, hexw)}, "
                               f"{lean_list(R, hexw)}⟩" for _, T, x, b, j, raws, I, R in rows],
             "Full rows through the forming, the atoms and the chain")):
        defs, ns = parts(name, typ, items, doc, ROW_PART if name == "row" else PART)
        body += defs
        names[name] = ns
    body.append("end Pouw.Fp8Atom.H1T.Vectors\n")
    body.append("namespace Pouw.Fp8Atom.H1T\n")
    for name, whole, doc in (("form", "formTable", f"The per-lane table ({len(form)} lanes)"),
                             ("tag", "tagTable", f"The tag table ({len(tags)} values)"),
                             ("salt", "saltVectors", f"The salt vectors ({len(salts)})"),
                             ("row", "rowVectors", f"The full rows ({len(rows)})")):
        typ = {"form": "FormVec", "tag": "TagVec", "salt": "SaltVec", "row": "RowVec"}[name]
        body.append(f"/-- {doc}, from `scripts/h1t_vectors.py` (seed {SEED}). -/\ndef {whole} : List {typ} :=\n  " +
                    " ++\n  ".join(f"Vectors.{n}" for n in names[name]) + "\n")
    body.append("end Pouw.Fp8Atom.H1T\n")
    (out / "Definitions/Pouw/Fp8Atom/H1TVectors.lean").write_text("\n".join(body))

    checks = ["import Definitions.Pouw.Fp8Atom.H1TVectors", "", HEADER, "namespace Pouw.Fp8Atom.H1T.Checks", "",
              "open Pouw.Fp8Atom.H1T", ""]
    for name, holds in (("form", "FormVec.holds"), ("tag", "TagVec.holds"), ("salt", "SaltVec.holds"),
                        ("row", "RowVec.holds")):
        for n in names[name]:
            checks.append(f"/-- `{n}` holds, evaluated by the kernel. -/\n"
                          f"theorem {n} : Vectors.{n}.all {holds} = true := by decide +kernel\n")
    for name, whole, holds in (("form", "formTable", "FormVec.holds"), ("tag", "tagTable", "TagVec.holds"),
                               ("salt", "saltVectors", "SaltVec.holds"), ("row", "rowVectors", "RowVec.holds")):
        parts_ = names[name]
        lemmas = [whole, "List.all_append", *parts_, "Bool.and_self"] if len(parts_) > 1 else [whole, *parts_]
        checks.append(f"/-- Every vector of `{whole}` holds. -/\n"
                      f"theorem {whole}_all : {whole}.all {holds} = true := by\n  simp only [{', '.join(lemmas)}]\n")
    checks.append("end Pouw.Fp8Atom.H1T.Checks\n")
    (out / "Proofs/Pouw/Fp8Atom/H1TChecks.lean").write_text("\n".join(checks))
    (out / "Definitions/Pouw/Fp8Atom/H1TCoverage.json").write_text(json.dumps(cov, indent=1) + "\n")
    print(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
