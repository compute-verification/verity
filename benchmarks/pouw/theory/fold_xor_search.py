"""FOLD-XOR's search: no program of k instructions drawn from DS-100's IMAD- and FP-pipe forms on sm_100a computes
z = x ⊕ y ⊕ K on full 32-bit words, for any constant K (note:20261019T0740Z-draft-rt-fold, FOLD-XOR).

The premise it checks. Priced at k + 1 IMAD-pipe instructions a LOP3, the B200 sheet's headline is under 1% for every
k ≥ 3 (`fold_xor.py`'s curve: 0.9425% at 4). So "ε < 1%" rests on "li's XOR has no IMAD- or FP-pipe program of 3 or
fewer instructions", and this search turns that into a checked fact.

The menu is `fold_xor.py`'s: the opcodes `facet_break.loads.class_of` routes to class IMAD or FFMA, the two classes
DS-100 prices on the IMAD and FP pipes. Each is covered by a modelled form whose behaviour contains the hardware's
(`COVER`, checked against `class_of`), so a program the search rules out is ruled out on the device. Every modifier
below is a free choice of the search, per instruction:
- IMAD: d = ±(a·b) + c' mod 2^32, where c' is c, −c or ~c, plus 0 or 1. It covers IMAD.MOV, IMAD.IADD, IMAD.SHL and
  IMUL, which are IMAD with constant operands, SASS's −b (−(a·b)), −c and ~c, and IMAD.X: its carry-in is a
  predicate, and no instruction on the menu writes one (ISETP, FSETP and IADD3 are class ALU), so on the menu it is
  PT or !PT, the 0 or 1.
- IMAD.HI: both halves of the 64-bit s = ±(a'·b') + C' mod 2^64, where a' is a or −a and b' is b or −b (32-bit), each
  sign- or zero-extended on its own, and C' is the 64-bit register pair {c_hi:c_lo} (each half a register or a
  constant, the convention `fold_xor.py`'s 78 uses: {0, c} and {0, B'}), its negation or complement, plus 0 or 1.
  hi32(s) is IMAD.HI's result, and the pair is IMAD.WIDE's and IMUL.WIDE's, so those are covered too.
- IDP.4A: d = c + Σ_i a.b_i·b.b_i over the 4 bytes, and IDP.2A: d = c + a.h0·b.b_{2s} + a.h1·b.b_{2s+1} (.LO s = 0,
  .HI s = 1), signed or unsigned independently in a and b.
- FFMA2 (FADD2, FMUL2): two FFMA lanes in one instruction, each with its own operands. A lane is IEEE binary32
  fma(±|a|, ±|b|, ±|c|) under any rounding mode (RN, RZ, RM, RP and RNA), with .FTZ (subnormal operands and result to
  sign-preserving zero), .SAT (to [0, 1], NaN to +0, and −0 to either zero) and .FMZ (0 times anything is 0, its sign
  +0 or the operands' xor), and a NaN result is the canonical 0x7FFFFFFF. It covers FFMA, and FADD and FMUL, which
  are FFMA with b = 1.0 and c = −0.0.
- Off the menu, as `fold_xor.py` checks: HFMA2 (class HFMA2) has no DS-100 row, nor IMAD.WIDE (IMADW), which the
  IMAD.HI form covers anyway.

An operand is any earlier value (x, y or a result) or a free 32-bit constant; K is free too. The output is the last
instruction's result: a program whose output is earlier is a shorter program, and k = 0 is x or y, so "no program of
1, ..., k instructions" is "none of k or fewer". The last instruction is never FP: x ⊕ y ⊕ K takes every 32-bit value
(fix y, and x ranges over all), the non-canonical NaN patterns among them, and an FP lane returns only 0x7FFFFFFF.

The search. For each sequence of opcodes, CEGIS over z3's bit-vectors and floats: find wiring, modifiers, constants and
K that compute x ⊕ y ⊕ K on a set of inputs, then look for an input where that program is wrong and add it. When no
program fits the inputs found (unsat), none of that sequence computes it on all inputs: that is the proof, with no
sampled constant anywhere. A program that is right on every input would be reported as FOUND, refuting the premise.
`--target add` and `--target mulhi` are the search finding programs that exist (x + y + K, hi32(x·y) + K), so it
isn't vacuous.

Run: `python fold_xor_search.py --k 3 [--jobs N] [--no-fp] [--timeout-ms T] [--json PATH] [--target xor|add|mulhi]`.
Exit 0 when every sequence is unsat (the bound holds), 1 when one is FOUND, 2 when one is unknown (a timeout).
Stdlib and z3.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import z3

M32 = 0xFFFFFFFF
SEED = 0xF0D5
INT_OPS = ("IMAD", "IMAD.HI", "IDP.4A", "IDP.2A")
FP_OPS = ("FFMA2",)
#: operand slots and results of each modelled form
SLOTS = {"IMAD": 3, "IMAD.HI": 4, "IDP.4A": 3, "IDP.2A": 3, "FFMA2": 6}
OUTS = {"IMAD": 1, "IMAD.HI": 2, "IDP.4A": 1, "IDP.2A": 1, "FFMA2": 2}
#: every opcode DS-100 prices on the IMAD and FP pipes, and the modelled form that covers it (see the docstring)
COVER = {
    "IMAD": "IMAD", "IMAD.MOV.U32": "IMAD", "IMAD.IADD": "IMAD", "IMAD.SHL.U32": "IMAD", "IMAD.X": "IMAD",
    "IMUL": "IMAD", "IMAD.HI": "IMAD.HI", "IMAD.HI.U32": "IMAD.HI", "IMAD.HI.X": "IMAD.HI", "IMUL.HI": "IMAD.HI",
    "IMUL.WIDE": "IMAD.HI", "IMUL.WIDE.U32": "IMAD.HI",
    "IDP.4A.S8.S8": "IDP.4A", "IDP.4A.U8.U8": "IDP.4A", "IDP.4A.S8.U8": "IDP.4A", "IDP.4A.U8.S8": "IDP.4A",
    "IDP.2A.LO.S16.S8": "IDP.2A", "IDP.2A.LO.U16.U8": "IDP.2A", "IDP.2A.HI.S16.S8": "IDP.2A",
    "IDP.2A.HI.U16.U8": "IDP.2A",
    "FFMA": "FFMA2", "FADD": "FFMA2", "FMUL": "FFMA2", "FFMA2": "FFMA2", "FADD2": "FFMA2", "FMUL2": "FFMA2",
    "FFMA.FTZ": "FFMA2", "FADD.FTZ.SAT": "FFMA2", "FMUL.FMZ": "FFMA2", "FFMA.RZ": "FFMA2",
}
OFF_MENU = ("IMAD.WIDE.U32", "IMAD.WIDE", "HFMA2", "HADD2", "HMUL2")
F32 = z3.Float32()
CANON_NAN = 0x7FFFFFFF


def bv(v: int):
    return z3.BitVecVal(v & M32, 32)


def mux(sel, opts):
    if isinstance(sel, int):
        return opts[sel]
    out = opts[-1]
    for i in range(len(opts) - 2, -1, -1):
        out = z3.If(sel == i, opts[i], out)
    return out


# ---- the modelled forms ----------------------------------------------------------------------------------------------

def ext(v, hi: int, lo: int, signed):
    part = z3.Extract(hi, lo, v)
    pad = 32 - (hi - lo + 1)
    return z3.If(signed, z3.SignExt(pad, part), z3.ZeroExt(pad, part))


def _addend(c, m, width: int):
    """SASS's c operand of an IMAD form: c, −c or ~c, plus .X's carry-in (a constant 0 or 1 on the menu)."""
    one = z3.BitVecVal(1, width)
    c = z3.If(m["nc"], -c, c)
    return c - z3.If(m["inv"], one, 0) + z3.If(m["cx"], one, 0)


def imad(o, m, p=None):
    """`p`, when given, stands for the product a·b (`anchored_cegis`)."""
    p = o[0] * o[1] if p is None else p
    return [z3.If(m["np"], -p, p) + _addend(o[2], m, 32)]


def hi_factors(o, m):
    """IMAD.HI's 64-bit multiplicands: a or −a and b or −b, each sign- or zero-extended."""
    a = z3.If(m["na"], -o[0], o[0])
    b = z3.If(m["nb"], -o[1], o[1])
    return (z3.If(m["sa"], z3.SignExt(32, a), z3.ZeroExt(32, a)),
            z3.If(m["sb"], z3.SignExt(32, b), z3.ZeroExt(32, b)))


def imad_hi(o, m, p=None):
    """Both halves of the 64-bit a·b + {c_hi:c_lo}: the high one is IMAD.HI's, and the pair is IMAD.WIDE's and
    IMUL.WIDE's (off the menu, covered anyway). `p`, when given, stands for the product (`anchored_cegis`)."""
    if p is None:
        a, b = hi_factors(o, m)
        p = a * b
    s = z3.If(m["np"], -p, p) + _addend(z3.Concat(o[3], o[2]), m, 64)
    return [z3.Extract(31, 0, s), z3.Extract(63, 32, s)]


def idp4a(o, m):
    acc = o[2]
    for i in range(4):
        acc = acc + ext(o[0], 8 * i + 7, 8 * i, m["sa"]) * ext(o[1], 8 * i + 7, 8 * i, m["sb"])
    return [acc]


def idp2a(o, m):
    lo = [o[2]]
    for half in range(2):
        b0 = z3.If(m["hi"], ext(o[1], 8 * (half + 2) + 7, 8 * (half + 2), m["sb"]),
                   ext(o[1], 8 * half + 7, 8 * half, m["sb"]))
        lo.append(ext(o[0], 16 * half + 15, 16 * half, m["sa"]) * b0)
    return [lo[0] + lo[1] + lo[2]]


def _flush(f, on):
    zero = z3.If(z3.fpIsNegative(f), z3.fpMinusZero(F32), z3.fpPlusZero(F32))
    return z3.If(z3.And(on, z3.fpIsSubnormal(f)), zero, f)


def lane(a, b, c, m, p: str):
    """One FFMA lane: binary32 fma of the operands as floats, under the lane's modifiers; the result's bits."""
    args = []
    for v, n in ((a, "a"), (b, "b"), (c, "c")):
        f = _flush(z3.fpBVToFP(v, F32), m["ftz" + p])
        f = z3.If(m[f"abs{n}{p}"], z3.fpAbs(f), f)
        args.append(z3.If(m[f"neg{n}{p}"], z3.fpNeg(f), f))
    one = z3.FPVal(1.0, F32)
    fa, fb, fc = args
    fmz = z3.And(m["fmz" + p], z3.Or(z3.fpIsZero(fa), z3.fpIsZero(fb)))
    zs = z3.And(m["fmzx" + p], z3.Xor(z3.fpIsNegative(fa), z3.fpIsNegative(fb)))
    fa = z3.If(fmz, z3.If(zs, z3.fpMinusZero(F32), z3.fpPlusZero(F32)), fa)
    fb = z3.If(fmz, one, fb)
    r = _flush(z3.fpFMA(m["rm" + p], fa, fb, fc), m["ftz" + p])
    low = z3.If(z3.And(m["satz" + p], z3.fpIsZero(r)), r, z3.fpPlusZero(F32))
    sat = z3.If(z3.Or(z3.fpIsNaN(r), z3.fpIsZero(r), z3.fpLT(r, z3.fpPlusZero(F32))), low,
                z3.If(z3.fpGT(r, one), one, r))
    r = z3.If(m["sat" + p], sat, r)
    return z3.If(z3.fpIsNaN(r), bv(CANON_NAN), z3.fpToIEEEBV(r))


def ffma2(o, m):
    return [lane(o[0], o[1], o[2], m, "0"), lane(o[3], o[4], o[5], m, "1")]


FORMS = {"IMAD": imad, "IMAD.HI": imad_hi, "IDP.4A": idp4a, "IDP.2A": idp2a, "FFMA2": ffma2}


def modifiers(op: str, tag: str) -> dict:
    B = lambda n: z3.Bool(f"{tag}_{n}")
    signs = {n: B(n) for n in ("np", "nc", "inv", "cx")}
    if op == "IMAD":
        return signs
    if op == "IMAD.HI":
        return {"sa": B("sa"), "sb": B("sb"), "na": B("na"), "nb": B("nb"), **signs}
    if op == "IDP.4A":
        return {"sa": B("sa"), "sb": B("sb")}
    if op == "IDP.2A":
        return {"sa": B("sa"), "sb": B("sb"), "hi": B("hi")}
    if op == "FFMA2":
        m = {}
        for p in "01":
            m["rm" + p] = z3.Const(f"{tag}_rm{p}", z3.RNE().sort())
            for n in ("ftz", "sat", "satz", "fmz", "fmzx", "absa", "absb", "absc", "nega", "negb", "negc"):
                m[n + p] = B(n + p)
        return m
    return {}


# ---- the program skeleton and its evaluation -------------------------------------------------------------------------

def label(ops: tuple) -> str:
    return " ".join(op if isinstance(op, str) else f"{op[0]}{list(op[1])}" for op in ops)


def op_name(op) -> str:
    return op if isinstance(op, str) else op[0]


def skeleton(ops: tuple, fix: tuple = ()):
    """The symbolic program for a sequence of opcodes: per instruction its operand selectors (an index into the values
    so far, or the last index for its own constant), constants and modifiers; and the constraints on the selectors.
    An integer form is (op, (a, b)), its multiplicands' sources fixed (the search enumerates them): a symbolic 32 × 32
    product under a selector is what z3 can't refute, and with x and y concrete at each input a fixed source makes the
    product concrete, or a constant times a concrete word. `fix[i]` maps more of instruction i's slots to a source
    (`splits`)."""
    prog, cons, pool = [], [], 2
    for i, op in enumerate(ops):
        tag, name = f"i{i}", op_name(op)
        slots = [(z3.BitVec(f"{tag}_s{j}", 4), z3.BitVec(f"{tag}_c{j}", 32)) for j in range(SLOTS[name])]
        if not isinstance(op, str):
            slots[0], slots[1] = (op[1][0], slots[0][1]), (op[1][1], slots[1][1])
        m = modifiers(name, tag)
        for j, s in (fix[i] if i < len(fix) else {}).items():
            if isinstance(j, str):
                m[j] = z3.BoolVal(s)
            else:
                slots[j] = (s, slots[j][1])
        cons += [z3.ULE(s, pool) for s, _ in slots if not isinstance(s, int)]
        prog.append((name, slots, m))
        pool += OUTS[name]
    return prog, cons


def evaluate(prog, x, y):
    pool = [x, y]
    for op, slots, m in prog:
        vals = [mux(s, pool + [c]) for s, c in slots]
        pool += FORMS[op](vals, m)
    return pool[-1]


def target(name: str, x, y, K):
    if name == "xor":
        return x ^ y ^ K
    if name == "add":
        return x + y + K
    if name == "mulhi":
        return z3.Extract(63, 32, z3.ZeroExt(32, x) * z3.ZeroExt(32, y)) + K
    raise SystemExit(f"unknown target {name}")


def nan_bits(v):
    """v is a binary32 NaN pattern: exponent all ones, mantissa nonzero, either sign."""
    return z3.And(z3.Extract(30, 23, v) == 0xFF, z3.Extract(22, 0, v) != 0)


def as_nan(v: int) -> int:
    return (v | 0x7F800000) if v & 0x7FFFFF else (v | 0x7F800001)


def initial_inputs(rng: random.Random, edge_only: bool = False, domain: str = "all") -> list[tuple[int, int]]:
    """The edge pairs, whose products with any value are shifts, negations or alternating masks, then 4 random; on the
    nan domain each word is made a NaN pattern (exponent bits set, a zero mantissa made 1)."""
    edge = [(0, 0), (M32, M32), (0, M32), (M32, 0), (1, 0), (0, 1), (0x55555555, 0xAAAAAAAA), (1 << 31, 0)]
    out = edge + ([] if edge_only else [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(4)])
    return [(as_nan(x), as_nan(y)) for x, y in out] if domain == "nan" else out


def splits(ops: tuple) -> list[tuple]:
    """The first instruction's addend slots set to each source in turn (x, y or its constant): the cases of `ops`,
    which together are every program of it. An FP first instruction isn't split."""
    op = ops[0]
    if isinstance(op, str):
        return [()]
    free = range(2, SLOTS[op[0]])
    return [(dict(zip(free, srcs)),) for srcs in itertools.product(range(3), repeat=len(free))]


def mod_splits(ops: tuple, fix: tuple) -> list[tuple]:
    """`fix` (one of `splits`) with the first instruction's modifiers set to each assignment in turn, inv and cx both
    set left out (c − 1 + 1 is c, as with neither): with x, y and its addends' sources fixed, its results are concrete
    at each input, so the next instruction's products are by concrete words."""
    names = sorted(modifiers(ops[0][0], "m"))
    out = []
    for vals in itertools.product((False, True), repeat=len(names)):
        m = dict(zip(names, vals))
        if not (m.get("inv") and m.get("cx")):
            out.append(({**fix[0], **m},))
    return out


def _sw(i: int) -> int:
    return 1 - i if i < 2 else i


def swap(ops: tuple) -> tuple:
    """`ops` with x and y exchanged, in `sequences`' canonical form. x ⊕ y ⊕ K is symmetric, and a program P of `ops`
    gives the program of `swap(ops)` that computes P(y, x), so one computes the target if and only if the other does."""
    return tuple(op if isinstance(op, str) else
                 (op[0], (_sw(op[1][0]), _sw(op[1][1])) if op[0] == "IDP.2A" else tuple(sorted(map(_sw, op[1]))))
                 for op in ops)


def swap_fix(ops: tuple, fix: tuple) -> tuple:
    """A case's `fix` under `swap`: x and y exchanged in its sources, and when the swap reorders the first
    instruction's multiplicands, its per-operand modifiers exchanged with them."""
    if not fix:
        return fix
    a, b = ops[0][1]
    flip = ops[0][0] != "IDP.2A" and _sw(a) > _sw(b)
    other = {"sa": "sb", "sb": "sa", "na": "nb", "nb": "na"} if flip else {}
    return ({other.get(j, j): (_sw(s) if isinstance(j, int) else s) for j, s in fix[0].items()},)


def merge(k: int, records: list[dict]) -> dict:
    """k's verdict from search records (whole sequences, `splits` cases and `mod_splits` sub-cases, from any runs): a
    sequence is unsat when it or its `swap` image is, or when each of its cases is, a case when it or its image is or
    when each of its sub-cases is. -> the tally, the cases still open, and of those searched as sub-cases, the sub-cases
    still open (`subs`). A split run's per-sequence summary (it has "cases") is skipped: under `--cases` it is over
    some cases only, and its cases' own records carry the verdict."""
    res: dict[str, set] = {}
    for r in records:
        if "cases" not in r:
            res.setdefault(r["ops"], set()).add(r["result"])
    ok = lambda *names: any("unsat" in res.get(n, ()) for n in names)
    seqs, unsat, still, subs = sequences(k, True), 0, [], []
    for s in seqs:
        t = swap(s)
        if ok(label(s), label(t)):
            unsat += 1
            continue
        if isinstance(s[0], str):
            still.append(label(s))
            continue
        case = lambda g: (f"{label(s)} /{fix_label(g)}", f"{label(t)} /{fix_label(swap_fix(s, g))}")
        left = [f for f in splits(s) if not ok(*case(f)) and not all(ok(*case(g)) for g in mod_splits(s, f))]
        unsat += not left
        still += [case(f)[0] for f in left]
        for f in left:
            gs = [case(g) for g in mod_splits(s, f)]
            if any(n in res for g in gs for n in g):
                subs += [g[0] for g in gs if not ok(*g)]
    return {"k": k, "sequences": len(seqs), "unsat": unsat, "open": still, "subs": subs,
            "found": sorted(n for n, v in res.items() if "FOUND" in v)}


def fix_label(fix: tuple, mods: bool = True) -> str:
    """Slots first (s2=x), then modifiers (np=1); `mods` False leaves the modifiers out (the `splits` case's label)."""
    def part(i, j, s):
        return f"{j}={int(s)}" if isinstance(j, str) else f"s{j}={'xyC'[s] if i == 0 else s}"
    order = lambda t: (isinstance(t[0], str), str(t[0]))
    return ";".join(",".join(part(i, j, s) for j, s in sorted(f.items(), key=order) if mods or not isinstance(j, str))
                    for i, f in enumerate(fix))


def first_reads(ops: tuple, fix: tuple) -> set:
    """Which of x (0) and y (1) the first instruction reads, once `fix` sets its addends; None if one is still free."""
    op = ops[0]
    if isinstance(op, str):
        return None
    srcs = list(op[1]) + [fix[0].get(j) if fix else None for j in range(2, SLOTS[op[0]])]
    return None if None in srcs else {s for s in srcs if s < 2}


def shared_axis(ops: tuple, fix: tuple, by_mult: bool = False, pow2: bool = False):
    """The word whose values CEGIS shares across its inputs (`shared_inputs`; 0: x, 1: y), or None: the one the first
    instruction reads alone once `fix` sets its addends, and with `by_mult` else the one both its multiplicands read.
    The other word then enters the first results only through the addends, so at one value of the shared word a
    result whose addend is a constant is one term up to a carry, and the next instruction's products by it are few.
    That settles IMAD.HI[0, 0] IMAD.HI[3, 3] /s2=y,s3=C cases unknown at 60 s in 3 s, but it slows
    IMAD[0, 0] IMAD.HI[2, 3] /s2=y cases from 4 s to 26 s, so it is for the cases the default leaves unknown. With
    `pow2`, also the word a multiplicand reads beside the first instruction's constant."""
    reads = first_reads(ops, fix)
    if reads in ({0}, {1}):
        return min(reads)
    if not (by_mult or pow2) or reads is None or isinstance(ops[0], str):
        return None
    mult = set(ops[0][1]) - ({2} if pow2 else set())
    return min(mult) if mult in ({0}, {1}) else None


def collision_inputs(ops: tuple, fix: tuple, rng: random.Random, n: int = 12) -> list[tuple[int, int]]:
    """Pairs of inputs whose x ⊕ y differ while the first instruction's results agree at two random values of its
    constants: when those results don't depend on the constants (x·x + {y:C} at x and −x, or at x with its top bit
    flipped and y shifted to make up the difference), a program's
    outputs at the pair differ only through the second instruction's reads of x and y, and the products by those
    results are equal. Each of n random inputs comes with its partner when z3 finds one (3 s each)."""
    prog, _ = skeleton(ops[:1], fix)
    name, slots, m = prog[0]
    X, Y = z3.BitVec("X2", 32), z3.BitVec("Y2", 32)
    consts = [c for _, c in slots]

    def results(x, y, cv):
        vals = [z3.substitute(mux(s, [x, y, c]), *zip(consts, map(bv, cv))) for s, c in slots]
        return FORMS[name](vals, m)

    out = []
    for _ in range(n):
        x, y = rng.getrandbits(32), rng.getrandbits(32)
        sol = z3.Solver()
        sol.set("timeout", 3000)
        for _ in range(2):
            cv = [rng.getrandbits(32) for _ in consts]
            sol.add(*[a == b for a, b in zip(results(bv(x), bv(y), cv), results(X, Y, cv))])
        sol.add(bv(x ^ y) != X ^ Y)
        out.append((x, y))
        if sol.check() == z3.sat:
            md = sol.model()
            out.append((md.eval(X, model_completion=True).as_long(), md.eval(Y, model_completion=True).as_long()))
    return out


def shared_inputs(rng: random.Random, axis: int, pow2: bool = False) -> list[tuple[int, int]]:
    """4 values of the shared word (`shared_axis`) against 6 of the other: the first instruction's results are then
    few terms, each shared by several inputs, so z3 bit-blasts few products by them rather than one per input. With
    `pow2` the shared word's values are 0 and powers of two, at which every product by it is a shift: that settles
    IMAD.HI[0, 2] IMAD.HI[0, 2] /s2=y,s3=y cases unknown at 120 s in 2 s."""
    few = [0, 1, 1 << 31, 1 << rng.randrange(1, 31)] if pow2 else [0, 1] + [rng.getrandbits(32) for _ in range(2)]
    many = [0, M32, 1 << 31] + [rng.getrandbits(32) for _ in range(3)]
    return [(a, b) if axis == 0 else (b, a) for a in few for b in many]


def axis_inputs() -> list[tuple[int, int]]:
    """Pairs on the axes x = 0 and y = 0, and powers of two: a first product x·y is 0 on the axes, so there the first
    results are their addends, nearly constant along an axis, and the next product is nearly one term; off them it is
    a power of two, so every product by it is a shift."""
    pts = [(0, 0)] + [p for j in (0, 1, 7, 16, 31) for p in ((0, 1 << j), (1 << j, 0))]
    return pts + [(1 << a, 1 << b) for a, b in ((0, 0), (3, 29), (16, 16), (31, 1), (20, 5))]


def cegis(ops: tuple[str, ...], tname: str = "xor", timeout_ms: int = 0, seed: int = SEED,
          edge_only: bool = False, domain: str = "all", fix: tuple = (), by_mult: bool = False,
          collide: bool = False, pow2: bool = False, axes: bool = False) -> dict:
    """One sequence of opcodes: unsat (no program of it computes the target), FOUND (one does, with its wiring), or
    unknown (a timeout), with the seconds it took. On the nan domain both inputs range over the NaN patterns only
    (`reduce_nan`). With `fix` (one of `splits`), only the programs whose addend sources it names; when that gives a
    `shared_axis` (under `by_mult` or `pow2`), CEGIS starts from `shared_inputs` and looks for counterexamples at those
    words first; with `collide` it starts from `collision_inputs` instead, and with `axes` from `axis_inputs`."""
    name = label(ops) + (f" /{fix_label(fix)}" if fix else "")
    t0 = time.monotonic()
    rng = random.Random(f"{seed}:{name}:{tname}")
    prog, cons = skeleton(ops, fix)
    K = z3.BitVec("K", 32)
    syn = z3.Solver()
    if timeout_ms:
        syn.set("timeout", timeout_ms)
    syn.add(cons)
    inputs = []

    def add(x: int, y: int):
        inputs.append((x, y))
        syn.add(evaluate(prog, bv(x), bv(y)) == target(tname, bv(x), bv(y), K))

    axis = shared_axis(ops, fix, by_mult, pow2) if fix and domain == "all" and not (collide or axes) else None
    start = (axis_inputs() if axes and domain == "all" else
             collision_inputs(ops, fix, rng) if collide and fix and domain == "all" else
             shared_inputs(rng, axis, pow2) if axis is not None else initial_inputs(rng, edge_only, domain))
    for x, y in start:
        add(x, y)

    def end(result: str, **more) -> dict:
        return {"ops": name, "result": result, "inputs": len(inputs), **more,
                "seconds": round(time.monotonic() - t0, 1)}

    X, Y = z3.BitVec("X", 32), z3.BitVec("Y", 32)
    general = evaluate(prog, X, Y) != target(tname, X, Y, K)
    if domain == "nan":
        general = z3.And(general, nan_bits(X), nan_bits(Y))
    free = [v for _, slots, m in prog for s, c in slots for v in (s, c) if not isinstance(v, int)]
    free += [v for _, _, m in prog for v in m.values() if not (z3.is_true(v) or z3.is_false(v))]
    while True:
        r = syn.check()
        if r == z3.unsat:
            return end("unsat")
        if r != z3.sat:
            return end("unknown", stage="synthesis")
        mdl = syn.model()
        sol = [(v, mdl.eval(v, model_completion=True)) for v in free + [K]]
        cx = None
        shared = [z3.Or([(X, Y)[axis] == bv(p[axis]) for p in inputs])] if axis is not None else []
        for near in shared + [None]:
            ver = z3.Solver()
            if timeout_ms:
                ver.set("timeout", timeout_ms)
            ver.add(z3.substitute(general, *sol))
            if near is not None:
                ver.add(near)
            rv = ver.check()
            if rv == z3.sat:
                cx = ver.model()
                break
            if rv != z3.unsat:
                return end("unknown", stage="verification")
        if cx is None:
            return end("FOUND", program={str(v): str(val) for v, val in sol})
        add(cx.eval(X, model_completion=True).as_long(), cx.eval(Y, model_completion=True).as_long())


# ---- anchored products and x-slices ----------------------------------------------------------------------------------

def _small(k: int, X):
    """k·X for a small integer k, as additions: a negative constant factor would be bit-blasted as a dense one."""
    out = z3.BitVecVal(0, X.size())
    for _ in range(abs(k)):
        out = out + X
    return -out if k < 0 else out


class ByWord:
    """Products v·X of one symbolic word X by concrete words v. Each v is s·c + k·2^32 + t (mod 2^width) with s = ±1,
    c a 32-bit core of fewest set bits and |k|, |t| ≤ 3, so only each distinct core is multiplied by X, and a core
    that is a power of two is a shift."""

    def __init__(self, X):
        self.X, self.width, self.cores = X, X.size(), {}

    def __call__(self, v: int):
        w, mask = self.width, (1 << self.width) - 1
        best = None
        for t in range(-3, 4):
            for s in (1, -1):
                c = (s * (v - t)) & M32
                k = ((v - t - s * c) & mask) >> 32 if w > 32 else 0
                k = k - (1 << (w - 32)) if w > 32 and k >> (w - 33) else k
                if abs(k) <= 3 and (best is None or c.bit_count() < best[1].bit_count()):
                    best = (s, c, k, t)
        if best is None:
            return z3.BitVecVal(v, w) * self.X
        s, c, k, t = best
        if c not in self.cores:
            self.cores[c] = z3.BitVecVal(c, w) * self.X
        out = self.cores[c] if s == 1 else -self.cores[c]
        if k:
            out = out + (_small(k, self.X) << 32)
        return out + _small(t, self.X) if t else out


def values(cons, term, limit: int, timeout_ms: int = 0):
    """Every value `term` takes under `cons`, by AllSAT: the list ends on an unsat check, so it is complete. None when
    there are more than `limit` (or a check is unknown)."""
    sol = z3.Solver()
    if timeout_ms:
        sol.set("timeout", timeout_ms)
    sol.add(cons)
    out = []
    while len(out) <= limit:
        r = sol.check()
        if r == z3.unsat:
            return out
        if r != z3.sat:
            return None
        out.append(sol.model().eval(term, model_completion=True).as_long())
        sol.add(term != out[-1])
    return None


def _case_of(term, table: list):
    """table's entry for term's value: [(value, expr)], term known to take one of the values."""
    out = table[-1][1]
    for v, e in reversed(table[:-1]):
        out = z3.If(term == v, e, out)
    return out


def anchored(A: list, B: list, cons, limit: int, by: tuple, timeout_ms: int = 0, p0=None):
    """A[-1]·B[-1] as A[0]·B[0] + D·B[0] + A[0]·E + D·E, with D = A[-1] − A[0] and E = B[-1] − B[0]: when, under `cons`,
    D and E each take at most `limit` values (`values`, complete), each product by D or E is a case over concrete
    words (`ByWord`, `by` its pair for A[0] and B[0]), so z3 bit-blasts one symbolic product, A[0]·B[0], for all
    inputs, or none when `p0` stands for it. -> the product, or None past `limit`."""
    D, E = A[-1] - A[0], B[-1] - B[0]
    vd, ve = values(cons, D, limit, timeout_ms), values(cons, E, limit, timeout_ms)
    if vd is None or ve is None:
        return None
    w, mask = A[0].size(), (1 << A[0].size()) - 1
    byA, byB = by
    out = A[0] * B[0] if p0 is None else p0
    if vd != [0]:
        out = out + _case_of(D, [(v, byB(v)) for v in vd])
    if ve != [0]:
        out = out + _case_of(E, [(u, byA(u)) for u in ve])
    if vd != [0] and ve != [0]:
        out = out + _case_of(D, [(v, _case_of(E, [(u, z3.BitVecVal(v * u & mask, w)) for u in ve])) for v in vd])
    return out


#: the x-slices' constants, tried in turn: alternating bits, so that no window of carries makes y ⊕ c from y
SLICES = (0x55555555, 0xA5A55A5A, 0x12345678)


def anchored_cegis(ops: tuple, fix: tuple = (), c_star: int = SLICES[0], timeout_ms: int = 0, limit: int = 64,
                   tname: str = "xor", free_anchor: bool = False, start: list | None = None) -> dict:
    """One two-instruction sequence (or a `splits` case) on its x-slice at c*, CEGIS with the last instruction's
    product anchored (`anchored`): unsat, slice-sat or unknown.

    A program computing x ⊕ y ⊕ K computes y ⊕ c* at x0 = c* ⊕ K, for every y, so no x0 (one symbolic word), wiring,
    modifiers and constants with that slice is no program, and unsat is the proof. In the slice the first
    instruction's product, when its multiplicands read x and constants only, is one free word
    (each program's product is one of its values, so unsat stays the proof), and its results then differ across the
    inputs y only through the addends; CEGIS starts from y = 0, 1, 2, 4 and 8, at which the anchored products' cores
    are shifts (more inputs to start from made the IMAD.HI[0, 2] IMAD.HI[2, 4] cases slower: 11 s at 5, over 60 s at
    33). A slice that some program computes is slice-sat, which settles nothing (try another c*). The modifiers stay
    symbolic, so a `splits` case is searched whole. With `tname` add, the slice is y + c* (x0 = c* − K).

    With `free_anchor`, the anchor product A[0]·B[0] is a free word P0 too, so z3 bit-blasts no symbolic product.
    Verification stays exact; a program whose P0 isn't A[0]·B[0] at its A[0] and B[0] (a, b) gets the lemma
    A[0] = a ∧ B[0] = b → P0 = a·b instead, true of every program, and after `limit` lemmas P0 is tied to the product."""
    name = label(ops) + (f" /{fix_label(fix)}" if fix else "")
    t0 = time.monotonic()
    prog, cons = skeleton(ops, fix)
    (n1, s1, m1), (n2, s2, m2) = prog
    if len(ops) != 2 or n1 not in INT_OPS:
        raise ValueError(f"{name}: anchored CEGIS takes two instructions, the first an integer form")
    if 1 in ops[0][1]:
        raise ValueError(f"{name}: an x-slice needs a first product that doesn't read y (its swap image reads x)")
    if tname not in ("xor", "add"):
        raise ValueError(f"{name}: an x-slice of {tname} isn't y combined with a constant")
    X0 = z3.BitVec("X0", 32)
    slice_target = lambda y: y ^ bv(c_star) if tname == "xor" else y + bv(c_star)
    P1 = z3.BitVec("P1", 64 if n1 == "IMAD.HI" else 32) if n1 in ("IMAD", "IMAD.HI") else None

    def last_operands(x, y):
        pool = [x, y] + FORMS[n1]([mux(s, [x, y, c]) for s, c in s1], m1, *([P1] if P1 is not None else []))
        return [mux(s, pool + [c]) for s, c in s2]

    def factors(vals):
        return hi_factors(vals, m2) if n2 == "IMAD.HI" else (vals[0], vals[1])

    def output(vals, p=None):
        return FORMS[n2](vals, m2, p)[-1] if n2 in ("IMAD", "IMAD.HI") else FORMS[n2](vals, m2)[-1]

    syn = z3.Solver()
    if timeout_ms:
        syn.set("timeout", timeout_ms)
    syn.add(cons)
    inputs, A, B, by, P0 = [], [], [], None, None

    def end(result: str, **more) -> dict:
        return {"ops": name, "result": result, "inputs": len(inputs), "method": "slice", "c_star": f"{c_star:#010x}",
                **more, "seconds": round(time.monotonic() - t0, 1)}

    def add(inp: int) -> bool:
        y = bv(inp)
        want = slice_target(y)
        vals = last_operands(X0, y)
        inputs.append(inp)
        if n2 not in ("IMAD", "IMAD.HI"):
            syn.add(output(vals) == want)
            return True
        nonlocal by, P0
        a, b = factors(vals)
        A.append(a)
        B.append(b)
        if by is None:
            by = (ByWord(a), ByWord(b))
            P0 = z3.BitVec("P0", a.size()) if free_anchor else a * b
        p = P0 if len(A) == 1 else anchored([A[0], a], [B[0], b], cons, limit, by, timeout_ms, P0)
        if p is None:
            return False
        syn.add(output(vals, p) == want)
        return True

    for inp in [0, 1, 2, 4, 8] if start is None else start:
        if not add(inp):
            return end("unknown", stage="anchor", at=inp)
    Y = z3.BitVec("Y", 32)
    general = output(last_operands(X0, Y)) != slice_target(Y)
    free = [v for _, slots, m in prog for s, c in slots for v in (s, c) if not isinstance(v, int)]
    free += [v for _, _, m in prog for v in m.values() if not (z3.is_true(v) or z3.is_false(v))]
    free += [X0] + ([P1] if P1 is not None else [])
    lemmas = 0
    while True:
        r = syn.check()
        if r == z3.unsat:
            return end("unsat", lemmas=lemmas)
        if r != z3.sat:
            return end("unknown", stage="synthesis", lemmas=lemmas)
        mdl = syn.model()
        if free_anchor and P0 is not None:
            a, b = (mdl.eval(v, model_completion=True) for v in (A[0], B[0]))
            if mdl.eval(P0, model_completion=True).as_long() != a.as_long() * b.as_long() % (1 << a.size()):
                lemmas += 1
                syn.add(P0 == A[0] * B[0] if lemmas > limit else
                        z3.Implies(z3.And(A[0] == a, B[0] == b), P0 == z3.BitVecVal(a.as_long() * b.as_long(), a.size())))
                continue
        sol = [(v, mdl.eval(v, model_completion=True)) for v in free]
        ver = z3.Solver()
        if timeout_ms:
            ver.set("timeout", timeout_ms)
        ver.add(z3.substitute(general, *sol))
        rv = ver.check()
        if rv == z3.unsat:
            return end("slice-sat", program={str(v): str(val) for v, val in sol})
        if rv != z3.sat:
            return end("unknown", stage="verification")
        nxt = ver.model().eval(Y, model_completion=True).as_long()
        if nxt in inputs:
            raise AssertionError(f"{name}: verification refuted a program at an input it was synthesized for")
        if not add(nxt):
            return end("unknown", stage="anchor", at=nxt)


def window_check(ops: tuple, fix: tuple, c_star: int = SLICES[0], tname: str = "xor", timeout_ms: int = 0,
                 count: tuple | None = (3, 2)) -> dict:
    """One two-instruction `splits` case on its x-slice at c*, on a window y = 4m, 4m + 1, 4m + 2 where the last
    product is affine: unsat, window-sat (settles nothing) or unknown.

    The family it takes (else ValueError): both instructions IMAD or IMAD.HI, the first's product not reading y nor its
    high addend word, and at most one of the last's multiplicands varying with y (y, or the first's low result when its
    low addend word is y). At x0 = c* ⊕ K the first's sum is A ± y with A fixed, so its low result is (A_lo ± y) mod
    2^32 and its high result changes at one step of y; each of the last's multiplicands, negated or not and extended, is
    then constant or steps by ±1, except at one step of y each (the varying one's wrap, zero-extended, or its sign
    boundary, sign-extended; the high result's change). The 2^30 windows have disjoint steps, so every program has a
    window where they step by τ_a and τ_b, one of them 0, and there its product is a_0·b_0 + i·(τ_a·b_0 + τ_b·a_0) (a
    ring identity mod 2^w): the last sum is affine on the window, and its result steps by q or q + 1 (hi32 of an affine
    map) or by a constant (IMAD). y ⊕ c* steps by 1 − 2c_0, then by 2c_0 − 1 + 2(1 − 2c_1), 4 apart when c_0 ≠ c_1
    (0x55555555: −1, then 3), so no program has the slice.

    z3 checks both halves over every source, modifier and constant, with the first product one free word. The count,
    looser than the one step each, as any finite count does: no program has 2 steps where a constant multiplicand
    changes, nor 3 where the varying one doesn't step by +1 and 3 where it doesn't step by −1 (else the result is
    unknown). The window, exactly over every m, with a_0·b_0 a free word too, the premises (each multiplicand's two
    steps equal, in {0, ±1}, one of them 0) asserted on the forms' own multiplicands: each program at its window is a
    model, so unsat is the proof. With `tname` add, the slice is y + c*, which programs have (the check isn't vacuous). `count` is the numbers of off steps refuted for the varying
    and the constant multiplicands; (1, ·) and (·, 1) are sat on cases with a varying multiplicand and a high result
    that changes, and None checks the window alone, a probe that settles nothing."""
    name = label(ops) + (f" /{fix_label(fix)}" if fix else "")
    t0 = time.monotonic()
    names = [op_name(op) for op in ops]
    first = fix[0] if fix else {}
    if len(ops) != 2 or set(names) - {"IMAD", "IMAD.HI"}:
        raise ValueError(f"{name}: a window check takes two instructions, each IMAD or IMAD.HI")
    if 1 in ops[0][1] or not all(isinstance(first.get(j), int) for j in range(2, SLOTS[names[0]])):
        raise ValueError(f"{name}: a window check needs a first product not reading y and its addends' sources fixed")
    if names[0] == "IMAD.HI" and first[3] == 1:
        raise ValueError(f"{name}: the first's high addend word is y, so its high result varies with y")
    varying = {1} | ({2} if first[2] == 1 else set())
    if sum(s in varying for s in ops[1][1]) > 1:
        raise ValueError(f"{name}: both of the last's multiplicands vary with y")
    if tname not in ("xor", "add"):
        raise ValueError(f"{name}: an x-slice of {tname} isn't y combined with a constant")
    prog, cons = skeleton(ops, fix)
    (n1, s1, m1), (n2, s2, m2) = prog
    X0, M = z3.BitVec("X0", 32), z3.BitVec("M", 32)
    P1 = z3.BitVec("P1", 64 if n1 == "IMAD.HI" else 32)

    def operands(y):
        pool = [X0, y] + FORMS[n1]([mux(s, [X0, y, c]) for s, c in s1], m1, P1)
        return [mux(s, pool + [c]) for s, c in s2]

    def factors(v):
        return hi_factors(v, m2) if n2 == "IMAD.HI" else (v[0], v[1])

    def solver(*fs):
        s = z3.Solver()
        if timeout_ms:
            s.set("timeout", timeout_ms)
        s.add(*cons, *fs)
        return s

    def off_steps(j: int, n: int, tau: int) -> list:
        """n distinct steps y → y + 1 where multiplicand j doesn't step by tau."""
        ys = [z3.BitVec(f"Y{j}_{tau}_{i}", 32) for i in range(n)]
        out = [z3.ULT(a, b) for a, b in itertools.pairwise(ys)] + [z3.ULE(ys[-1], M32 - 1)]
        for y in ys:
            f0, f1 = factors(operands(y))[j], factors(operands(y + 1))[j]
            out.append(f1 - f0 != z3.BitVecVal(tau, f0.size()))
        return out

    for j, src in enumerate(ops[1][1] if count else ()):
        off = off_steps(j, count[0], 1) + off_steps(j, count[0], -1) if src in varying else off_steps(j, count[1], 0)
        r = solver(*off).check()
        if r != z3.unsat:
            return {"ops": name, "result": "unknown", "method": "window", "stage": f"count of multiplicand {j}: {r}",
                    "seconds": round(time.monotonic() - t0, 1)}
    sol = solver(z3.ULE(M, (1 << 30) - 1))
    ys = [M * 4 + i for i in range(3)]
    vals = [operands(y) for y in ys]
    A, B = zip(*(factors(v) for v in vals))
    w = A[0].size()
    one, zero = z3.BitVecVal(1, w), z3.BitVecVal(0, w)
    ta, tb = A[1] - A[0], B[1] - B[0]
    sol.add(A[2] - A[1] == ta, B[2] - B[1] == tb, z3.Or(ta == 0, tb == 0),
            *(z3.Or(t == zero, t == one, t == -one) for t in (ta, tb)))
    step = (z3.If(ta == one, B[0], z3.If(ta == -one, -B[0], zero)) +
            z3.If(tb == one, A[0], z3.If(tb == -one, -A[0], zero)))
    P2 = z3.BitVec("P2", w)
    for i, (v, y) in enumerate(zip(vals, ys)):
        sol.add(FORMS[n2](v, m2, P2 + _small(i, step))[-1] == (y ^ bv(c_star) if tname == "xor" else y + bv(c_star)))
    r = sol.check()
    result = ("window-sat" if r == z3.sat else "unknown" if r != z3.unsat else
              "unsat" if count == (3, 2) else "window-unsat")
    return {"ops": name, "result": result, "method": "window", "c_star": f"{c_star:#010x}",
            "seconds": round(time.monotonic() - t0, 1)}


def settle(ops: tuple, tname: str, timeout_ms: int, seed: int, edge_only: bool, domain: str, fix: tuple,
           by_mult: bool, collide: bool, pow2: bool, slices: tuple = (), free_anchor: bool = False,
           axes: bool = False, window: bool = False) -> dict:
    """One case by `cegis`, or with `slices` by `anchored_cegis` at each x-slice in turn until one is unsat (its
    record names it), else the last one's record, or with `window` by `window_check` at the first x-slice. A case
    `anchored_cegis` or `window_check` doesn't take is unknown."""
    name = label(ops) + (f" /{fix_label(fix)}" if fix else "")
    if window:
        try:
            return window_check(ops, fix, SLICES[0], tname, timeout_ms)
        except ValueError as e:
            return {"ops": name, "result": "unknown", "stage": str(e), "seconds": 0.0}
    if not slices:
        return cegis(ops, tname, timeout_ms, seed, edge_only, domain, fix, by_mult, collide, pow2, axes)
    tried = []
    for c in slices:
        try:
            r = anchored_cegis(ops, fix, c, timeout_ms, tname=tname, free_anchor=free_anchor)
        except ValueError as e:
            return {"ops": name, "result": "unknown", "stage": str(e), "seconds": 0.0}
        tried.append(f"{c:#010x}:{r['result']}")
        if r["result"] == "unsat":
            break
    return {**r, "tried": tried}


def sequences(k: int, fp: bool) -> list[tuple]:
    """Every sequence of k opcodes, the last an integer form (an FP lane's result never is x ⊕ y ⊕ K), each integer
    form with each pair of multiplicand sources: unordered for IMAD, IMAD.HI and IDP.4A (a·b = b·a, their sign and
    negation modifiers swapping with them), ordered for IDP.2A (halves times bytes)."""
    inner = INT_OPS + (FP_OPS if fp else ())
    out = []
    for names in itertools.product(*([inner] * (k - 1) + [INT_OPS])):
        pools, pool = [], 2
        for n in names:
            pools.append(pool)
            pool += OUTS[n]
        choices = [[(n, ab) for ab in itertools.product(range(p + 1), repeat=2)] if n == "IDP.2A"
                   else [(n, ab) for ab in itertools.combinations_with_replacement(range(p + 1), 2)] if n in INT_OPS
                   else [n] for n, p in zip(names, pools)]
        out += list(itertools.product(*choices))
    return out


def reduce_nan(ops: tuple):
    """The integer sequence that covers `ops` on the nan domain, or None when an FP form follows an integer one.

    When x and y are both NaN patterns, an FP lane whose operands are x, y, constants or FP results is a function of
    its constants and modifiers alone: a NaN operand makes the lane's result the canonical NaN (or +0 under .SAT),
    except that .FMZ with a zero multiplicand makes it c ± 0, and c is then a NaN or such a function itself. So a
    sequence whose FP forms all precede its first integer form computes there what its integer suffix computes with
    every FP result replaced by a constant, and the constant slot (free) covers it. The suffix is re-indexed: x and y
    stay, an FP result becomes the instruction's own constant, and an integer result moves down past the FP results.
    No program of the suffix on the nan domain means no program of `ops` on all inputs."""
    names = [op_name(op) for op in ops]
    nfp = next((i for i, n in enumerate(names) if n not in FP_OPS), len(ops))
    if any(n in FP_OPS for n in names[nfp:]):
        return None
    drop = sum(OUTS[n] for n in names[:nfp])
    out, pool = [], 2 + drop
    for op in ops[nfp:]:
        n, ab = op

        def move(i: int) -> int:
            if i < 2:
                return i
            if i < 2 + drop or i == pool:
                return pool - drop
            return i - drop

        a, b = (move(i) for i in ab)
        out.append((n, (a, b) if n == "IDP.2A" else tuple(sorted((a, b)))))
        pool += OUTS[n]
    return tuple(out)


def search(k: int, fp: bool = True, tname: str = "xor", jobs: int = 1, timeout_ms: int = 0, stream=None,
           part: str = "all", shard: tuple[int, int] = (0, 1), labels: set | None = None,
           edge_only: bool = False, via_nan: bool = False, split: bool = False,
           cases: set | None = None, split_mods: bool = False, by_mult: bool = False,
           collide: bool = False, pow2: bool = False, slices: tuple = (), free_anchor: bool = False,
           axes: bool = False, window: bool = False) -> dict:
    """Every sequence of k opcodes (`part` int: those without an FP form, fp: those with one); each result is also
    written to `stream` (JSON lines) as it finishes. With `via_nan`, each sequence is settled by its `reduce_nan`
    suffix on the nan domain (each distinct suffix searched once); one that doesn't reduce is unknown. With `split`,
    each sequence is searched as its `splits`, each case streamed, and it is unsat when every case is, FOUND when
    one is; `cases` (labels "SEQUENCE /CASE") keeps only those, and a sequence's verdict is then over them alone.
    `split_mods` searches each case as its `mod_splits`, and `cases` may then name single sub-cases too. With
    `slices`, each case is searched by `anchored_cegis` instead, and with `window` by `window_check` (`settle`)."""
    seqs = [s for s in sequences(k, fp)
            if part == "all" or (part == "fp") == any(op_name(op) in FP_OPS for op in s)][shard[0]::shard[1]]
    if cases is not None:
        labels = {c.split(" /")[0] for c in cases} & (labels if labels is not None else {label(s) for s in seqs})
    if labels is not None:
        seqs = [s for s in seqs if label(s) in labels]
    res = []

    def parts(c):
        if not split:
            return [()]
        out = []
        for f in splits(c):
            subs = mod_splits(c, f) if split_mods and f else [f]
            whole = cases is None or f"{label(c)} /{fix_label(f)}" in cases
            out += [g for g in subs if whole or f"{label(c)} /{fix_label(g)}" in cases]
        return out

    def done(r):
        res.append(r)
        if stream:
            stream.write(json.dumps({"k": k, "fp": fp, **r}) + "\n")
            stream.flush()

    domain = "nan" if via_nan else "all"
    cover = {label(s): reduce_nan(s) if via_nan else s for s in seqs}
    todo = [(c, f) for c in dict.fromkeys(c for c in cover.values() if c is not None) for f in parts(c)]
    found = {}

    def got(c, f, r):
        found[c, fix_label(f)] = r
        if split:
            stream and stream.write(json.dumps({"k": k, "fp": fp, "case": True, **r}) + "\n")
            stream and stream.flush()
        elif not via_nan:
            done(r)

    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            futs = {ex.submit(settle, c, tname, timeout_ms, SEED, edge_only, domain, f, by_mult, collide, pow2, slices,
                              free_anchor, axes, window): (c, f) for c, f in todo}
            for fut in as_completed(futs):
                got(*futs[fut], fut.result())
    else:
        for c, f in todo:
            got(c, f, settle(c, tname, timeout_ms, SEED, edge_only, domain, f, by_mult, collide, pow2, slices,
                             free_anchor, axes, window))
    if split:
        for name, c in cover.items():
            rs = [found[c, fix_label(f)] for f in parts(c)]
            worst = next((r for want in ("FOUND", "unknown", "slice-sat", "window-sat") for r in rs
                          if r["result"] == want), rs[0])
            done({"ops": name, "result": worst["result"], "cases": len(rs),
                  "open": [r["ops"] for r in rs if r["result"] != "unsat"]})
    elif via_nan:
        for name, c in cover.items():
            done({"ops": name, "result": "unknown", "stage": "no nan reduction"} if c is None
                 else {**found[c, ""], "ops": name, "via": label(c), "domain": "nan"})
    tally = {r: sum(1 for x in res if x["result"] == r) for r in ("unsat", "FOUND", "unknown")}
    return {"k": k, "fp": fp, "part": part, "shard": list(shard), "target": tname, "sequences": len(seqs),
            "tally": tally, "proved": tally["unsat"] == len(seqs), "results": res}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--from-k", type=int, default=1, help="search every k from this one up to --k")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-fp", action="store_true")
    ap.add_argument("--timeout-ms", type=int, default=0)
    ap.add_argument("--target", default="xor", choices=("xor", "add", "mulhi"))
    ap.add_argument("--json")
    ap.add_argument("--stream", help="append each sequence's result to this file as a JSON line")
    ap.add_argument("--part", default="all", choices=("all", "int", "fp"),
                    help="only the sequences without (int) or with (fp) an FP form, to split a run across machines")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th sequence from the i-th")
    ap.add_argument("--labels", help="only the sequences whose labels are this file's lines")
    ap.add_argument("--edge-only", action="store_true", help="start CEGIS from the edge pairs alone")
    ap.add_argument("--via-nan", action="store_true",
                    help="settle each sequence by its integer suffix on NaN-pattern inputs (reduce_nan)")
    ap.add_argument("--split", action="store_true", help="search each sequence as its first instruction's addend cases")
    ap.add_argument("--cases", help="with --split, only the cases whose labels ('SEQUENCE /CASE') are this file's lines")
    ap.add_argument("--split-mods", action="store_true",
                    help="with --split, search each case as its first instruction's modifier assignments")
    ap.add_argument("--by-mult", action="store_true",
                    help="share the word both of the first instruction's multiplicands read (`shared_axis`)")
    ap.add_argument("--collide", action="store_true",
                    help="start CEGIS from input pairs the first instruction can't tell apart (`collision_inputs`)")
    ap.add_argument("--pow2", action="store_true",
                    help="share the word a first multiplicand reads, at 0 and powers of two (`shared_inputs`)")
    ap.add_argument("--axes", action="store_true",
                    help="start CEGIS from pairs on the axes x = 0 and y = 0 and powers of two (`axis_inputs`)")
    ap.add_argument("--slice", action="store_true",
                    help="settle each case by its x-slices (`anchored_cegis`, `SLICES`): two instructions, the first "
                         "product not reading y")
    ap.add_argument("--free-anchor", action="store_true", help="with --slice, the anchor product a free word")
    ap.add_argument("--window", action="store_true",
                    help="settle each case on a window of its x-slice where the last product is affine "
                         "(`window_check`): two IMAD or IMAD.HI, y in at most one last multiplicand")
    ap.add_argument("--merge", nargs="+", metavar="JSONL",
                    help="search nothing: k's verdict from these runs' records, through x <-> y (`merge`)")
    a = ap.parse_args()
    if a.merge:
        recs = [json.loads(line) for p in a.merge for line in open(p, encoding="utf-8") if line.strip()]
        m = merge(a.k, recs)
        print(f"k = {a.k}: {m['sequences']} sequences, {m['unsat']} unsat, {len(m['open'])} cases open "
              f"({len(m['subs'])} sub-cases open), FOUND {m['found']}"
              + ("  -- proved" if m["unsat"] == m["sequences"] else ""))
        for c in m["open"]:
            print(f"  open: {c}")
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=1)
        return 1 if m["found"] else 2 if m["open"] else 0
    out, rc = [], 0
    stream = open(a.stream, "a", encoding="utf-8") if a.stream else None
    for k in range(a.from_k, a.k + 1):
        labels = set(open(a.labels, encoding="utf-8").read().splitlines()) if a.labels else None
        cases = set(open(a.cases, encoding="utf-8").read().splitlines()) - {""} if a.cases else None
        r = search(k, not a.no_fp, a.target, a.jobs, a.timeout_ms, stream, a.part,
                   tuple(int(v) for v in a.shard.split("/")), labels, a.edge_only, a.via_nan, a.split, cases,
                   a.split_mods, a.by_mult, a.collide, a.pow2, SLICES if a.slice else (), a.free_anchor, a.axes,
                   a.window)
        out.append(r)
        print(f"k = {k}: {r['sequences']} sequences, {r['tally']}" + ("  -- proved" if r["proved"] else ""),
              flush=True)
        for x in r["results"]:
            if x["result"] != "unsat":
                print(f"  {x['result']}: {x['ops']} {x.get('program', '')}", flush=True)
        if r["tally"]["FOUND"]:
            rc = 1
        elif r["tally"]["unknown"] and rc == 0:
            rc = 2
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
