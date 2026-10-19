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


def imad(o, m):
    p = o[0] * o[1]
    return [z3.If(m["np"], -p, p) + _addend(o[2], m, 32)]


def imad_hi(o, m):
    """Both halves of the 64-bit a·b + {c_hi:c_lo}: the high one is IMAD.HI's, and the pair is IMAD.WIDE's and
    IMUL.WIDE's (off the menu, covered anyway)."""
    a = z3.If(m["na"], -o[0], o[0])
    b = z3.If(m["nb"], -o[1], o[1])
    a = z3.If(m["sa"], z3.SignExt(32, a), z3.ZeroExt(32, a))
    b = z3.If(m["sb"], z3.SignExt(32, b), z3.ZeroExt(32, b))
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


def shared_inputs(rng: random.Random, axis: int) -> list[tuple[int, int]]:
    """4 values of the word the first instruction reads (axis 0: x, 1: y) against 6 of the other: the first
    instruction's results are then 4 terms, each shared by 6 inputs, so z3 bit-blasts 4 products by its constants
    rather than one per input."""
    few = [0, 1] + [rng.getrandbits(32) for _ in range(2)]
    many = [0, M32, 1 << 31] + [rng.getrandbits(32) for _ in range(3)]
    return [(a, b) if axis == 0 else (b, a) for a in few for b in many]


CHEAP = (0, M32, *(1 << i for i in range(32)), *(M32 ^ (1 << i) for i in range(32)))


def cegis(ops: tuple[str, ...], tname: str = "xor", timeout_ms: int = 0, seed: int = SEED,
          edge_only: bool = False, domain: str = "all", fix: tuple = (), cheap: bool = False) -> dict:
    """One sequence of opcodes: unsat (no program of it computes the target), FOUND (one does, with its wiring), or
    unknown (a timeout). On the nan domain both inputs range over the NaN patterns only (`reduce_nan`). With `fix`
    (one of `splits`), only the programs whose addend sources it names; when the first instruction then reads only x
    (or only y), CEGIS starts from `shared_inputs` and looks for counterexamples at those words first. With `cheap`,
    it starts from the edge pairs and looks for counterexamples among the words with at most one bit set or clear
    (`CHEAP`) before any other: their products by an earlier result are few distinct terms."""
    name = label(ops) + (f" /{fix_label(fix)}" if fix else "")
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

    reads = first_reads(ops, fix) if fix and domain == "all" else None
    axis = 0 if reads == {0} else 1 if reads == {1} else None
    for x, y in (shared_inputs(rng, axis) if axis is not None else initial_inputs(rng, edge_only or cheap, domain)):
        add(x, y)
    X, Y = z3.BitVec("X", 32), z3.BitVec("Y", 32)
    general = evaluate(prog, X, Y) != target(tname, X, Y, K)
    if domain == "nan":
        general = z3.And(general, nan_bits(X), nan_bits(Y))
    free = [v for _, slots, m in prog for s, c in slots for v in (s, c) if not isinstance(v, int)]
    free += [v for _, _, m in prog for v in m.values() if not (z3.is_true(v) or z3.is_false(v))]
    small = [z3.And(*(z3.Or([V == bv(w) for w in CHEAP]) for V in (X, Y)))] if cheap and domain == "all" else []
    while True:
        r = syn.check()
        if r == z3.unsat:
            return {"ops": name, "result": "unsat", "inputs": len(inputs)}
        if r != z3.sat:
            return {"ops": name, "result": "unknown", "inputs": len(inputs), "stage": "synthesis"}
        mdl = syn.model()
        sol = [(v, mdl.eval(v, model_completion=True)) for v in free + [K]]
        cx = None
        shared = [z3.Or([(X, Y)[axis] == bv(p[axis]) for p in inputs])] if axis is not None else []
        for near in small + shared + [None]:
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
                return {"ops": name, "result": "unknown", "inputs": len(inputs), "stage": "verification"}
        if cx is None:
            return {"ops": name, "result": "FOUND", "inputs": len(inputs),
                    "program": {str(v): str(val) for v, val in sol}}
        add(cx.eval(X, model_completion=True).as_long(), cx.eval(Y, model_completion=True).as_long())


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
           edge_only: bool = False, via_nan: bool = False, split: bool = False, cheap: bool = False,
           cases: set | None = None, split_mods: bool = False) -> dict:
    """Every sequence of k opcodes (`part` int: those without an FP form, fp: those with one); each result is also
    written to `stream` (JSON lines) as it finishes. With `via_nan`, each sequence is settled by its `reduce_nan`
    suffix on the nan domain (each distinct suffix searched once); one that doesn't reduce is unknown. With `split`,
    each sequence is searched as its `splits`, each case streamed, and it is unsat when every case is, FOUND when
    one is; `cases` (labels "SEQUENCE /CASE") keeps only those, and a sequence's verdict is then over them alone.
    `split_mods` searches each case as its `mod_splits`."""
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
        fs = [f for f in splits(c) if cases is None or f"{label(c)} /{fix_label(f)}" in cases]
        return [g for f in fs for g in (mod_splits(c, f) if split_mods and f else [f])]

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
            futs = {ex.submit(cegis, c, tname, timeout_ms, SEED, edge_only, domain, f, cheap): (c, f)
                    for c, f in todo}
            for fut in as_completed(futs):
                got(*futs[fut], fut.result())
    else:
        for c, f in todo:
            got(c, f, cegis(c, tname, timeout_ms, SEED, edge_only, domain, f, cheap))
    if split:
        for name, c in cover.items():
            rs = [found[c, fix_label(f)] for f in parts(c)]
            worst = next((r for want in ("FOUND", "unknown") for r in rs if r["result"] == want), rs[0])
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
    ap.add_argument("--cheap", action="store_true",
                    help="start from the edge pairs and look for counterexamples among CHEAP's words first")
    ap.add_argument("--split-mods", action="store_true",
                    help="with --split, search each case as its first instruction's modifier assignments")
    a = ap.parse_args()
    out, rc = [], 0
    stream = open(a.stream, "a", encoding="utf-8") if a.stream else None
    for k in range(a.from_k, a.k + 1):
        labels = set(open(a.labels, encoding="utf-8").read().splitlines()) if a.labels else None
        cases = set(open(a.cases, encoding="utf-8").read().splitlines()) - {""} if a.cases else None
        r = search(k, not a.no_fp, a.target, a.jobs, a.timeout_ms, stream, a.part,
                   tuple(int(v) for v in a.shard.split("/")), labels, a.edge_only, a.via_nan, a.split, a.cheap, cases,
                   a.split_mods)
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
