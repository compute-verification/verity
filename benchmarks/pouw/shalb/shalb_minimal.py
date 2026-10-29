"""Per-function minimality of SHA-512's Σ0, Σ1, σ0 and σ1 halves over sm_90 integer-lane instructions, by SMT.

    uv run --no-project --with z3-solver python benchmarks/pouw/shalb/shalb_minimal.py [--max-k K] [--timeout S]
        [--jobs N] [--only NAMES] [--json OUT]

A target is one 32-bit half of a 64-bit function of x = hi:lo, given the registers lo and hi. A program is k
straight-line instructions, each reading up to three sources (lo, hi, an earlier result, or an immediate of its own);
its last result must equal the target at every x. For each k and each tuple of k opcodes, the search is
counterexample-guided: a finite set of x, a solver asked for the opcodes' modifiers, sources and immediates agreeing
there, and each candidate it returns checked at every x by a second solver, its counterexample added. UNSAT at a finite
set of x proves that no program with those opcodes exists, since a program that agrees everywhere agrees there; UNSAT
at every tuple proves that no k-instruction program does. As a control, the search is asked for the honest
4-instruction program's opcodes (three funnel or plain shifts and a LOP3) for every half, beside the sweep: a control
not "found" says the solver's budget, not the target, decided it.

The instructions, with every modifier as a free choice:
- LOP3.LUT at every 8-bit LUT (bitwise: and, or, xor, not, select, three-way xor, Ch, Maj);
- SHF.R.W / SHF.L.W (funnel shifts of two sources at a constant amount), SHF.R.U32 / SHF.R.S32 / SHF.L.U32 (plain);
- PRMT at every selector (each result byte any byte of the two sources, or its sign replicated);
- IADD3 with each source negated or not; LEA and LEA.HI at every shift; IMAD's low word (a·m + c) at every immediate
  multiplier m; BREV;
- IMNMX (signed and unsigned min and max) and SEL on a source's sign, for the select family.

Outside the search: floating-point lane instructions (FFMA, HFMA2, DFMA, MUFU, conversions), IDP4A/IDP2A, VABSDIFF4,
POPC/FLO/BMSK, IMAD of two registers, IMAD.HI (its 64-bit products time the solver out), shifts by a register amount, and carries or predicates read from another instruction (each instruction
here reads only registers and immediates). Each half is searched alone: a program computing both halves together, sharing
work between them, is outside it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time

import z3

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1


def rotr(x: int, n: int) -> int:
    return (x >> n | x << (64 - n)) & M64


FUNCS = {
    "Σ0": lambda x: rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39),
    "Σ1": lambda x: rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41),
    "σ0": lambda x: rotr(x, 1) ^ rotr(x, 8) ^ x >> 7,
    "σ1": lambda x: rotr(x, 19) ^ rotr(x, 61) ^ x >> 6,
}
OPS = ("LOP3", "FSHR", "FSHL", "SHR", "SAR", "SHL", "PRMT", "IADD3", "LEA", "LEAHI", "IMAD", "BREV", "IMNMX", "SEL")


def zfunc(name, x):
    """The 64-bit target as a z3 term."""
    r = z3.RotateRight
    return {"Σ0": lambda: r(x, 28) ^ r(x, 34) ^ r(x, 39), "Σ1": lambda: r(x, 14) ^ r(x, 18) ^ r(x, 41),
            "σ0": lambda: r(x, 1) ^ r(x, 8) ^ z3.LShR(x, 7), "σ1": lambda: r(x, 19) ^ r(x, 61) ^ z3.LShR(x, 6)}[name]()


class Program:
    """A tuple of opcodes and each instruction's unknowns: three source selectors and immediates, LUT, shift amount,
    negations, PRMT selector, min/max mode."""

    def __init__(self, ops: tuple[str, ...]):
        self.ops, k = ops, len(ops)
        self.src = [[z3.BitVec(f"src{i}_{j}", 4) for j in range(3)] for i in range(k)]
        self.imm = [[z3.BitVec(f"imm{i}_{j}", 32) for j in range(3)] for i in range(k)]
        self.lut = [z3.BitVec(f"lut{i}", 8) for i in range(k)]
        self.sh = [z3.BitVec(f"sh{i}", 32) for i in range(k)]
        self.neg = [[z3.Bool(f"neg{i}_{j}") for j in range(3)] for i in range(k)]
        self.pidx = [[z3.BitVec(f"pidx{i}_{b}", 3) for b in range(4)] for i in range(k)]
        self.psgn = [[z3.Bool(f"psgn{i}_{b}") for b in range(4)] for i in range(k)]
        self.mode = [z3.BitVec(f"mode{i}", 2) for i in range(k)]

    def wellformed(self):
        """Sources in range, multipliers immediate, and every result read by a later instruction: a program with an
        unread result computes the target with fewer instructions, which a smaller k rules out."""
        k = len(self.ops)
        out = []
        for i in range(k):
            out.append(z3.ULT(self.sh[i], 32))
            out += [z3.ULE(self.src[i][j], 2 + i) for j in range(3)]
            if self.ops[i] == "IMAD":
                out.append(self.src[i][1] == 2 + i)
            if i < k - 1:
                out.append(z3.Or([self.src[j][s] == 2 + i for j in range(i + 1, k) for s in range(3)]))
        return out

    def step(self, i, a, b, c):
        op, sh = self.ops[i], self.sh[i]
        if op == "LOP3":
            v = z3.BitVecVal(0, 32)
            for m in range(8):
                term = (a if m & 4 else ~a) & (b if m & 2 else ~b) & (c if m & 1 else ~c)
                v = v | z3.If(z3.Extract(m, m, self.lut[i]) == 1, term, z3.BitVecVal(0, 32))
            return v
        if op == "FSHR":
            return z3.Extract(31, 0, z3.LShR(z3.Concat(b, a), z3.ZeroExt(32, sh)))
        if op == "FSHL":
            return z3.Extract(63, 32, z3.Concat(b, a) << z3.ZeroExt(32, sh))
        if op == "SHR":
            return z3.LShR(a, sh)
        if op == "SAR":
            return a >> sh
        if op == "SHL":
            return a << sh
        if op == "PRMT":
            ab = z3.Concat(b, a)
            pbytes = [z3.Extract(8 * q + 7, 8 * q, ab) for q in range(8)]
            out = []
            for q in range(4):
                sel = pbytes[7]
                for n in range(6, -1, -1):
                    sel = z3.If(self.pidx[i][q] == n, pbytes[n], sel)
                sgn = z3.If(z3.Extract(7, 7, sel) == 1, z3.BitVecVal(0xFF, 8), z3.BitVecVal(0, 8))
                out.append(z3.If(self.psgn[i][q], sgn, sel))
            return z3.Concat(out[3], out[2], out[1], out[0])
        if op == "IADD3":
            return sum((z3.If(self.neg[i][j], -s, s) for j, s in enumerate((a, b, c))), z3.BitVecVal(0, 32))
        if op == "LEA":
            return (a << sh) + b
        if op == "LEAHI":
            return z3.Extract(31, 0, z3.LShR(z3.Concat(c, a), z3.ZeroExt(32, 32 - sh))) + b
        if op == "IMAD":
            return a * b + c
        if op == "IMADHI":
            return z3.Extract(63, 32, z3.ZeroExt(32, a) * z3.ZeroExt(32, b)) + c
        if op == "BREV":
            return z3.Concat(*[z3.Extract(q, q, a) for q in range(32)])
        if op == "IMNMX":
            m = self.mode[i]
            return z3.If(m == 0, z3.If(a < b, a, b), z3.If(m == 1, z3.If(a > b, a, b),
                                                         z3.If(m == 2, z3.If(z3.ULT(a, b), a, b),
                                                               z3.If(z3.UGT(a, b), a, b))))
        if op == "SEL":
            return z3.If(c < 0, a, b)
        raise ValueError(op)

    def run(self, lo, hi):
        vals = [lo, hi]
        for i in range(len(self.ops)):
            srcs = []
            for j in range(3):
                v = self.imm[i][j]
                for n in range(len(vals) - 1, -1, -1):
                    v = z3.If(self.src[i][j] == n, vals[n], v)
                srcs.append(v)
            vals.append(self.step(i, *srcs))
        return vals

    def describe(self, model) -> list[str]:
        out = []
        for i, op in enumerate(self.ops):
            names = ["lo", "hi"] + [f"r{n}" for n in range(i)]
            srcs = []
            for j in range(3):
                s = model.eval(self.src[i][j], model_completion=True).as_long()
                srcs.append(names[s] if s < len(names) else
                            hex(model.eval(self.imm[i][j], model_completion=True).as_long()))
            out.append(f"r{i} = {op}({', '.join(srcs)}; lut {model.eval(self.lut[i], model_completion=True)}, "
                       f"sh {model.eval(self.sh[i], model_completion=True)})")
        return out


def search(task) -> dict:
    """Whether a program with opcodes `ops` computes half `half` (0 lo, 1 hi) of FUNCS[name]."""
    name, half, ops, timeout_s = task
    rng = random.Random(hash((name, half, ops)) & M32)
    prog = Program(ops)
    syn = z3.Tactic("qfbv").solver()
    syn.set(timeout=timeout_s * 1000)
    syn.add(prog.wellformed())
    xs = [1 << b for b in (0, 7, 31, 32, 45, 63)] + [rng.getrandbits(64) for _ in range(2)]
    t0, rounds = time.monotonic(), 0

    def constrain(x):
        out = prog.run(z3.BitVecVal(x & M32, 32), z3.BitVecVal(x >> 32, 32))[-1]
        syn.add(out == z3.BitVecVal((FUNCS[name](x) >> 32 * half) & M32, 32))

    for x in xs:
        constrain(x)
    base = {"function": name, "half": "lo" if half == 0 else "hi", "ops": list(ops)}
    while True:
        rounds += 1
        left = timeout_s - (time.monotonic() - t0)
        if left <= 0:
            return base | {"verdict": "unknown", "samples": len(xs), "rounds": rounds,
                           "seconds": round(time.monotonic() - t0, 2)}
        syn.set(timeout=int(left * 1000))
        r = syn.check()
        if r != z3.sat:
            verdict = "none" if r == z3.unsat else "unknown"
            return base | {"verdict": verdict, "samples": len(xs), "rounds": rounds,
                           "seconds": round(time.monotonic() - t0, 2)}
        model = syn.model()
        lo, hi = z3.BitVec("lo", 32), z3.BitVec("hi", 32)
        out = prog.run(lo, hi)[-1]
        ver = z3.Tactic("qfbv").solver()
        ver.set(timeout=timeout_s * 1000)
        ver.add(model.eval(out, model_completion=True) != z3.Extract(32 * half + 31, 32 * half,
                                                                      zfunc(name, z3.Concat(hi, lo))))
        if ver.check() == z3.unsat:
            return base | {"verdict": "found", "program": prog.describe(model), "samples": len(xs), "rounds": rounds,
                           "seconds": round(time.monotonic() - t0, 2)}
        cm = ver.model()
        xs.append(cm.eval(hi, model_completion=True).as_long() << 32 | cm.eval(lo, model_completion=True).as_long())
        constrain(xs[-1])


def summarize(names, k, rs) -> list[dict]:
    out = []
    for n in names:
        for h in ("lo", "hi"):
            mine = [r for r in rs if r["function"] == n and r["half"] == h]
            v = {f"n_{x}": sum(r["verdict"] == x for r in mine) for x in ("none", "unknown", "found")}
            out.append({"function": n, "half": h, "k": k, "opcode_tuples": len(mine), **v,
                        "max_seconds": max(r["seconds"] for r in mine),
                        "found": [r["program"] for r in mine if r["verdict"] == "found"][:3],
                        "unknown": [r["ops"] for r in mine if r["verdict"] == "unknown"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds a query, all its solver calls together")
    ap.add_argument("--control-timeout", type=int, default=7200, help="seconds a query, for the controls")
    ap.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--only", default="", help="comma-separated function names")
    ap.add_argument("--json", default="", help="the summary, rewritten after each k")
    ap.add_argument("--jsonl", default="", help="every query's result as it finishes")
    args = ap.parse_args()
    names = [n for n in FUNCS if not args.only or n in args.only.split(",")]
    summary: list[dict] = []
    stream = open(args.jsonl, "a") if args.jsonl else None

    def dump():
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"summary": summary}, f, indent=1, ensure_ascii=False)

    with mp.Pool(args.jobs) as pool:
        controls = pool.map_async(search, [(n, h, ("FSHR", "FSHR", "SHR" if n.startswith("σ") and h else "FSHR",
                                                   "LOP3"), args.control_timeout) for n in names for h in (0, 1)])
        for k in range(args.min_k, args.max_k + 1):
            tasks = [(n, h, ops, args.timeout) for n in names for h in (0, 1)
                     for ops in itertools.product(OPS, repeat=k)]
            t0, rs = time.monotonic(), []
            for r in pool.imap_unordered(search, tasks, chunksize=1):
                rs.append(r)
                if stream:
                    stream.write(json.dumps(r, ensure_ascii=False) + "\n")
                    stream.flush()
            for line in summarize(names, k, rs):
                summary.append(line)
                print(json.dumps(line, ensure_ascii=False), flush=True)
            print(f"k {k}: {len(tasks)} queries in {time.monotonic() - t0:.0f} s", flush=True)
            dump()
        for r in controls.get():
            summary.append({"control": r})
            print(json.dumps({"control": r}, ensure_ascii=False), flush=True)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
