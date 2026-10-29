"""Per-function minimality of SHA-512's Σ0, Σ1, σ0 and σ1 halves over sm_90 integer-lane instructions, by SMT.

    uv run --no-project --with z3-solver --with bitwuzla python benchmarks/pouw/shalb/shalb_minimal.py [--max-k K]
        [--timeout S] [--solver bitwuzla] [--extra OPS] [--redo QUERIES_JSONL] [--done QUERIES_JSONL] [--jobs N]
        [--only NAMES] [--json OUT]

A target is one 32-bit half of a 64-bit function of x = hi:lo, given the registers lo and hi. A program is k
straight-line instructions, each reading up to three sources (lo, hi, an earlier result, or an immediate of its own);
its last result must equal the target at every x. For each k and each tuple of k opcodes, the search is
counterexample-guided: a finite set of x, a solver asked for the opcodes' modifiers, sources and immediates agreeing
there, and each candidate it returns checked at every x by a second solver, its counterexample added. UNSAT at a finite
set of x proves that no program with those opcodes exists, since a program that agrees everywhere agrees there; UNSAT
at every tuple proves that no k-instruction program does. A program whose result goes unread computes the target in
fewer instructions, so each k reads "none" at every smaller k over the same opcodes. z3 builds every query and checks
every candidate; the synthesis query goes to z3 or to Bitwuzla (z3's assertions as SMT-LIB2). As a control, the search
is asked for the honest 4-instruction program's opcodes (three funnel or plain shifts and a LOP3) for every half: a
control not "found" says the solver's budget, not the target, decided it.

The instructions (OPS), with every modifier as a free choice:
- LOP3.LUT at every 8-bit LUT (bitwise: and, or, xor, not, select, three-way xor, Ch, Maj);
- SHF.R.W / SHF.L.W (funnel shifts of two sources at a constant amount), SHF.R.U32 / SHF.R.S32 / SHF.L.U32 (plain);
- PRMT at every selector (each result byte any byte of the two sources, or its sign replicated);
- IADD3 with each source negated or not; LEA and LEA.HI at every shift; IMAD's low word (a·m + c) at every immediate
  multiplier m; BREV;
- IMNMX (signed and unsigned min and max) and SEL on a source's sign, for the select family.

EXTRA, searched by `--extra` over the tuples holding at least one of them: funnel shifts by a register amount (wrap or
clamp), IMAD of two registers, POPC, FLO (U32 or S32, plain or .SH), BMSK (wrap or clamp); and IMAD.HI and IDP4A,
encoded but outside the sweeps, since their products leave both solvers undecided at every budget tried.

Outside the search: floating-point lane instructions (FFMA, HFMA2, DFMA, MUFU, conversions), IDP2A, VABSDIFF4, warp-wide
instructions (SHFL, REDUX, VOTE, MATCH), and carries or predicates read from another instruction (each instruction here
reads only registers and immediates). Each half is searched alone: a program computing both halves together, sharing
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
import zlib

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
EXTRA = ("FSHRV", "FSHLV", "IMADR", "IMADHI", "POPC", "FLO", "BMSK", "IDP4A")


def zfunc(name, x):
    """The 64-bit target as a z3 term."""
    r = z3.RotateRight
    return {"Σ0": lambda: r(x, 28) ^ r(x, 34) ^ r(x, 39), "Σ1": lambda: r(x, 14) ^ r(x, 18) ^ r(x, 41),
            "σ0": lambda: r(x, 1) ^ r(x, 8) ^ z3.LShR(x, 7),
            "σ1": lambda: r(x, 19) ^ r(x, 61) ^ z3.LShR(x, 6)}[name]()


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
        if op in ("IMAD", "IMADR"):
            return a * b + c
        if op == "IMADHI":
            signed = z3.Extract(0, 0, self.mode[i]) == 1
            return z3.Extract(63, 32, z3.If(signed, z3.SignExt(32, a) * z3.SignExt(32, b),
                                            z3.ZeroExt(32, a) * z3.ZeroExt(32, b))) + c
        if op in ("FSHRV", "FSHLV"):
            amt = z3.If(z3.Extract(0, 0, self.mode[i]) == 1, z3.If(z3.ULT(c, 32), c, z3.BitVecVal(32, 32)),
                        c & 31)
            if op == "FSHRV":
                return z3.Extract(31, 0, z3.LShR(z3.Concat(b, a), z3.ZeroExt(32, amt)))
            return z3.Extract(63, 32, z3.Concat(b, a) << z3.ZeroExt(32, amt))
        if op == "POPC":
            return sum((z3.ZeroExt(31, z3.Extract(q, q, a)) for q in range(32)), z3.BitVecVal(0, 32))
        if op == "FLO":
            x = z3.If(z3.And(z3.Extract(0, 0, self.mode[i]) == 1, a < 0), ~a, a)
            pos = z3.BitVecVal(M32, 32)
            for q in range(32):
                pos = z3.If(z3.Extract(q, q, x) == 1, z3.BitVecVal(q, 32), pos)
            return z3.If(z3.And(z3.Extract(1, 1, self.mode[i]) == 1, x != 0), 31 - pos, pos)
        if op == "BMSK":
            clamp = z3.Extract(0, 0, self.mode[i]) == 1
            pos, width = (z3.If(clamp, z3.If(z3.ULT(v, 32), v, z3.BitVecVal(32, 32)), v & 31) for v in (a, b))
            ones = (z3.BitVecVal(1, 64) << z3.ZeroExt(32, width)) - 1
            return z3.Extract(31, 0, ones << z3.ZeroExt(32, pos))
        if op == "IDP4A":
            ext = lambda s, v: z3.If(s, z3.SignExt(24, v), z3.ZeroExt(24, v))
            sa, sb = z3.Extract(0, 0, self.mode[i]) == 1, z3.Extract(1, 1, self.mode[i]) == 1
            return sum((ext(sa, z3.Extract(8 * q + 7, 8 * q, a)) * ext(sb, z3.Extract(8 * q + 7, 8 * q, b))
                        for q in range(4)), c)
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

    def unknowns(self) -> list:
        return [v for rows in (self.src, self.imm, self.neg, self.pidx, self.psgn) for row in rows for v in row] + \
            self.lut + self.sh + self.mode

    def describe(self, cand: dict) -> list[str]:
        out = []
        for i, op in enumerate(self.ops):
            names = ["lo", "hi"] + [f"r{n}" for n in range(i)]
            srcs = []
            for j in range(3):
                s = cand[str(self.src[i][j])].as_long()
                srcs.append(names[s] if s < len(names) else hex(cand[str(self.imm[i][j])].as_long()))
            out.append(f"r{i} = {op}({', '.join(srcs)}; lut {cand[str(self.lut[i])]}, sh {cand[str(self.sh[i])]})")
        return out


def z3_synth(syn, prog: Program, left_s: float):
    """The synthesis query by z3: (z3 result, the unknowns' values when sat)."""
    syn.set(timeout=int(left_s * 1000))
    r = syn.check()
    if r != z3.sat:
        return r, None
    m = syn.model()
    return r, {str(v): m.eval(v, model_completion=True) for v in prog.unknowns()}


def bitwuzla_synth(syn, prog: Program, left_s: float):
    """The same query, z3's assertions handed to Bitwuzla as SMT-LIB2."""
    import bitwuzla as bz
    tm, opts = bz.TermManager(), bz.Options()
    opts.set(bz.Option.PRODUCE_MODELS, True)
    opts.set(bz.Option.TIME_LIMIT_PER, max(1, int(left_s * 1000)))
    if os.environ.get("SHALB_ABSTRACTION"):
        opts.set(bz.Option.ABSTRACTION, True)
    parser = bz.Parser(tm, opts)
    parser.parse("(set-logic QF_BV)\n" + syn.sexpr(), True, False)
    solver = parser.bitwuzla()
    r = solver.check_sat()
    if r == bz.Result.UNSAT:
        return z3.unsat, None
    if r != bz.Result.SAT:
        return z3.unknown, None
    got = {t.symbol(): solver.get_value(t).value(10) for t in parser.get_declared_funs()}
    cand = {}
    for v in prog.unknowns():
        g = got.get(str(v))
        if z3.is_bool(v):
            cand[str(v)] = z3.BoolVal(g is True or g == "true")
        else:
            cand[str(v)] = z3.BitVecVal(int(g) if g is not None else 0, v.size())
    return z3.sat, cand


SOLVERS = {"z3": z3_synth, "bitwuzla": bitwuzla_synth}


def search(task) -> dict:
    """Whether a program with opcodes `ops` computes half `half` (0 lo, 1 hi) of FUNCS[name]."""
    name, half, ops, timeout_s, solver, init = task
    rng = random.Random(zlib.crc32(repr((name, half, ops)).encode()))
    prog = Program(ops)
    syn = z3.Tactic("qfbv").solver()
    syn.add(prog.wellformed())
    xs = [rng.getrandbits(64) if s == "r" else 1 << int(s) for s in init.split(",")]
    t0, rounds = time.monotonic(), 0

    def constrain(x):
        out = prog.run(z3.BitVecVal(x & M32, 32), z3.BitVecVal(x >> 32, 32))[-1]
        syn.add(out == z3.BitVecVal((FUNCS[name](x) >> 32 * half) & M32, 32))

    for x in xs:
        constrain(x)
    base = {"function": name, "half": "lo" if half == 0 else "hi", "ops": list(ops), "solver": solver}
    while True:
        rounds += 1
        left = timeout_s - (time.monotonic() - t0)
        if left <= 0:
            return base | {"verdict": "unknown", "samples": len(xs), "rounds": rounds,
                           "seconds": round(time.monotonic() - t0, 2)}
        r, cand = SOLVERS[solver](syn, prog, left)
        if r != z3.sat:
            verdict = "none" if r == z3.unsat else "unknown"
            return base | {"verdict": verdict, "samples": len(xs), "rounds": rounds,
                           "seconds": round(time.monotonic() - t0, 2)}
        lo, hi = z3.BitVec("lo", 32), z3.BitVec("hi", 32)
        out = z3.substitute(prog.run(lo, hi)[-1], *[(v, cand[str(v)]) for v in prog.unknowns()])
        ver = z3.Tactic("qfbv").solver()
        ver.set(timeout=timeout_s * 1000)
        ver.add(out != z3.Extract(32 * half + 31, 32 * half, zfunc(name, z3.Concat(hi, lo))))
        v = ver.check()
        if v != z3.sat:
            return base | {"verdict": "found" if v == z3.unsat else "unknown", "samples": len(xs), "rounds": rounds,
                           "seconds": round(time.monotonic() - t0, 2)} | \
                ({"program": prog.describe(cand)} if v == z3.unsat else {})
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
                        "max_seconds": max((r["seconds"] for r in mine), default=0),
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
    ap.add_argument("--solver", choices=sorted(SOLVERS), default="z3", help="the synthesis queries' solver")
    ap.add_argument("--redo", default="", help="a --jsonl stream: rerun only its unknown queries, by k")
    ap.add_argument("--done", default="", help="a --jsonl stream: skip the queries it decided (none or found), as "
                    "when resuming a sweep cut short")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--init", default="0,7,31,32,45,63,r,r",
                    help="the first samples of x: bit positions set alone, or r for a random x")
    ap.add_argument("--extra", default="",
                    help="comma-separated EXTRA opcodes: search the tuples over OPS and them that hold at least one")
    args = ap.parse_args()
    names = [n for n in FUNCS if not args.only or n in args.only.split(",")]
    extra = tuple(o for o in args.extra.split(",") if o)
    assert set(extra) <= set(EXTRA), extra
    summary: list[dict] = []
    stream = open(args.jsonl, "a") if args.jsonl else None
    redo: dict[int, list] = {}
    if args.redo:
        for line in open(args.redo):
            r = json.loads(line)
            if r["verdict"] == "unknown" and r["function"] in names:
                redo.setdefault(len(r["ops"]), []).append((r["function"], 0 if r["half"] == "lo" else 1,
                                                           tuple(r["ops"]), args.timeout, args.solver, args.init))

    decided: set = set()
    if args.done:
        for line in open(args.done):
            r = json.loads(line)
            if r["verdict"] != "unknown":
                decided.add((r["function"], 0 if r["half"] == "lo" else 1, tuple(r["ops"])))

    def dump():
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"summary": summary}, f, indent=1, ensure_ascii=False)

    with mp.Pool(args.jobs) as pool:
        controls = pool.map_async(search, [] if args.no_controls else [
            (n, h, ("FSHR", "FSHR", "SHR" if n.startswith("σ") and h else "FSHR", "LOP3"), args.control_timeout,
             args.solver, args.init) for n in names for h in (0, 1)])
        for k in sorted(redo) if args.redo else range(args.min_k, args.max_k + 1):
            tuples = [ops for ops in itertools.product(OPS + extra, repeat=k) if set(ops) & set(extra)] \
                if extra else list(itertools.product(OPS, repeat=k))
            tasks = redo[k] if args.redo else [(n, h, ops, args.timeout, args.solver, args.init) for n in names
                                               for h in (0, 1) for ops in tuples if (n, h, ops) not in decided]
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
