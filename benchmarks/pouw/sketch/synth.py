"""The fewest instructions a loop-free program needs to compute a sketch of a dot product's running words, on small
instances, by counterexample-guided synthesis (z3): the falsification test for a sketch's cost premise.

    python3 benchmarks/pouw/sketch/synth.py [--width 8] [--steps 2,3,4] [--sketch all|NAME,...] [--max-len L]
        [--timeout S] [--json OUT]

- **The instance.** The free inputs are a dot product's step increments x_1..x_T (b-bit words, arithmetic mod 2^b); the
  running words are w_t = x_1 + ... + x_t. A sketch is a function of the words. Honest computes the words (w_1 = x_1 is
  free, each later word one write) and then the sketch with its own instructions.
- **The components**, each one instruction writing one register (W1 prices every one at 32 or more, `H32`):
  `lin` (any linear form c_0 + sum c_t x_t of the free inputs with constants chosen by the search; it stands in for an MMA
  output, and is generous to the adversary, since a real one costs a register per 32 MACs and cannot add a constant),
  `add3` (a +- b +- c), `imad` (a * b + c), `lop3` (any bitwise function of three registers, its truth table chosen by the
  search) and `shf` (shift left, logical shift right or rotate by a chosen amount). An operand is an input, an earlier
  result or an immediate.
- **The question.** For each length L below honest's, is there a program of L instructions whose last result equals the
  sketch on every input? z3 synthesizes on a growing set of examples and verifies over all inputs (CEGIS); SAT is a fold
  (a cheaper program, printed), UNSAT at every L below honest's means none exists in this component set at this size.
  The linear sketches are positive controls: their fold must be found.
- **Widths.** With `--ring`, `shf` shifts left only, and every component (and every sketch here) commutes with reduction
  mod 2^b: a 32-bit program correct on every input, reduced mod 2^b, is a b-bit program correct on every input. UNSAT at
  a small width then rules out the same length at 32 bits. SAT at a small width may be a coincidence of the width.
  Without `--ring`, right shifts and rotates don't commute, and a width's UNSAT holds only at that width.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import z3

OPS = ("lin", "add3", "imad", "lop3", "shf")


def words(xs, mask):
    out, acc = [], 0
    for x in xs:
        acc = (acc + x) & mask
        out.append(acc)
    return out


def _odd(i):
    return 2 * i + 3


SKETCHES = {
    # name: (spec on the words, honest length as a function of T, the inputs' arity multiplier, valid T predicate)
    "linear": (lambda w, m: sum(_odd(t) * x for t, x in enumerate(w)) & m, lambda T: (T - 1) + T, 1, lambda T: True),
    "sum": (lambda w, m: sum(w) & m, lambda T: (T - 1) + (T - 1 + 1) // 2, 1, lambda T: True),
    "sumsq": (lambda w, m: sum(x * x for x in w) & m, lambda T: (T - 1) + T, 1, lambda T: True),
    "nh": (lambda w, m: sum(w[2 * k] * w[2 * k + 1] for k in range(len(w) // 2)) & m, lambda T: (T - 1) + T // 2, 1,
           lambda T: T % 2 == 0),
    "xor": (None, lambda T: (T - 1) + (T - 1 + 1) // 2, 1, lambda T: True),
    "chain": (None, lambda T: (T - 1) + (T - 1 + 1) // 2, 1, lambda T: T % 2 == 0),
    "nh2": (None, lambda T: 2 * (T - 1) + T, 2, lambda T: True),
}


def spec(name, xs, T, b):
    m = (1 << b) - 1
    if name == "nh2":
        u, v = words(xs[:T], m), words(xs[T:], m)
        return sum(a * c for a, c in zip(u, v)) & m
    w = words(xs, m)
    if name == "xor":
        s = 0
        for x in w:
            s ^= x
        return s
    if name == "chain":
        s = 0
        for k in range(T // 2):
            a, c = w[2 * k], w[2 * k + 1]
            s = (s ^ a ^ c) if k % 2 == 0 else (s + a + c) & m
        return s
    return SKETCHES[name][0](w, m)


def spec_z3(name, xs, T, b):
    """The same spec over z3 bitvectors (for the verification query)."""
    def ws(v):
        out, acc = [], z3.BitVecVal(0, b)
        for x in v:
            acc = acc + x
            out.append(acc)
        return out
    if name == "nh2":
        u, v = ws(xs[:T]), ws(xs[T:])
        return sum((a * c for a, c in zip(u[1:], v[1:])), u[0] * v[0])
    w = ws(xs)
    if name == "linear":
        return sum((z3.BitVecVal(_odd(t), b) * x for t, x in enumerate(w)), z3.BitVecVal(0, b))
    if name == "sum":
        return sum(w[1:], w[0])
    if name == "sumsq":
        return sum((x * x for x in w), z3.BitVecVal(0, b))
    if name == "nh":
        return sum((w[2 * k] * w[2 * k + 1] for k in range(T // 2)), z3.BitVecVal(0, b))
    if name == "xor":
        s = w[0]
        for x in w[1:]:
            s = s ^ x
        return s
    if name == "chain":
        s = z3.BitVecVal(0, b)
        for k in range(T // 2):
            a, c = w[2 * k], w[2 * k + 1]
            s = (s ^ a ^ c) if k % 2 == 0 else (s + a + c)
        return s
    raise ValueError(name)


class Candidate:
    """L symbolic instructions over n free inputs: opcode, three operand selectors (an input, an earlier result or the
    immediate), immediates, the lop3 truth table, the shift kind and amount, the lin coefficients and the add3 signs."""

    def __init__(self, n, L, b, ring=False):
        self.n, self.L, self.b, self.ring = n, L, b, ring
        self.op = [z3.BitVec(f"op{i}", 3) for i in range(L)]
        self.sel = [[z3.BitVec(f"s{i}_{j}", 8) for j in range(3)] for i in range(L)]
        self.imm = [[z3.BitVec(f"k{i}_{j}", b) for j in range(3)] for i in range(L)]
        self.lut = [z3.BitVec(f"lut{i}", 8) for i in range(L)]
        self.sh = [z3.BitVec(f"sh{i}", b) for i in range(L)]
        self.shk = [z3.BitVec(f"shk{i}", 2) for i in range(L)]
        self.lc = [[z3.BitVec(f"c{i}_{t}", b) for t in range(n + 1)] for i in range(L)]
        self.neg = [[z3.Bool(f"n{i}_{j}") for j in range(2)] for i in range(L)]

    def wf(self):
        c = []
        zero = z3.BitVecVal(0, self.b)
        for i in range(self.L):
            op, sel = self.op[i], self.sel[i]
            c += [z3.ULT(op, len(OPS)), z3.ULT(self.shk[i], 1 if self.ring else 3), z3.ULT(self.sh[i], self.b)]
            for j in range(3):
                # selector n + i means the immediate; below it an input or an earlier result
                c.append(z3.ULE(sel[j], self.n + i))
            # fields an opcode doesn't read are zero, and operand orders it can't tell apart are sorted (lop3 absorbs a
            # permutation into its table, imad's product commutes): neither removes a distinct program
            c += [z3.Implies(op == 0, z3.And(sel[0] == 0, sel[1] == 0, sel[2] == 0)),
                  z3.Implies(op != 0, z3.And([lc == zero for lc in self.lc[i]])),
                  z3.Implies(op != 1, z3.And(z3.Not(self.neg[i][0]), z3.Not(self.neg[i][1]))),
                  z3.Implies(op == 2, z3.ULE(sel[0], sel[1])),
                  z3.Implies(op != 3, self.lut[i] == 0),
                  z3.Implies(op == 3, z3.And(z3.ULE(sel[0], sel[1]), z3.ULE(sel[1], sel[2]))),
                  z3.Implies(op != 4, z3.And(self.sh[i] == 0, self.shk[i] == 0)),
                  z3.Implies(op == 4, z3.And(sel[1] == 0, sel[2] == 0))]
            for j in range(3):
                c.append(z3.Implies(sel[j] != self.n + i, self.imm[i][j] == zero))
        # the last instruction is the sketch, and every earlier one is read by a later non-lin one (a minimal program
        # has no dead code, so this prunes nothing it needs)
        for i in range(self.L - 1):
            c.append(z3.Or([z3.And(self.op[k] != 0, z3.Or([self.sel[k][j] == self.n + i for j in range(3)]))
                            for k in range(i + 1, self.L)]))
        return c

    def run(self, xs):
        """Each instruction's value on the inputs xs (z3 terms), the last one being the program's output."""
        b = self.b
        vals = list(xs)
        for i in range(self.L):
            def operand(j):
                s = self.sel[i][j]
                v = self.imm[i][j]
                for idx in range(len(vals) - 1, -1, -1):
                    v = z3.If(s == idx, vals[idx], v)
                return v
            a, c2, c3 = operand(0), operand(1), operand(2)
            lin = self.lc[i][0]
            for t in range(self.n):
                lin = lin + self.lc[i][t + 1] * xs[t]
            add3 = a + z3.If(self.neg[i][0], -c2, c2) + z3.If(self.neg[i][1], -c3, c3)
            imad = a * c2 + c3
            lop = z3.BitVecVal(0, b)
            for mterm in range(8):
                bit = z3.Extract(mterm, mterm, self.lut[i])
                term = (a if mterm & 4 else ~a) & (c2 if mterm & 2 else ~c2) & (c3 if mterm & 1 else ~c3)
                lop = lop | z3.If(bit == 1, term, z3.BitVecVal(0, b))
            k = self.sh[i]
            rot = z3.RotateLeft(a, k)
            shf = z3.If(self.shk[i] == 0, a << k, z3.If(self.shk[i] == 1, z3.LShR(a, k), rot))
            v = z3.If(self.op[i] == 0, lin, z3.If(self.op[i] == 1, add3, z3.If(self.op[i] == 2, imad,
                                                                                    z3.If(self.op[i] == 3, lop, shf))))
            vals.append(v)
        return vals[-1]

    def describe(self, m):
        lines = []
        names = [f"x{t + 1}" for t in range(self.n)]
        for i in range(self.L):
            op = OPS[m.eval(self.op[i]).as_long()]

            def nm(j):
                s = m.eval(self.sel[i][j]).as_long()
                if s >= self.n + i:
                    return str(m.eval(self.imm[i][j], model_completion=True).as_long())
                return names[s]
            if op == "lin":
                cs = [m.eval(c, model_completion=True).as_long() for c in self.lc[i]]
                rhs = " + ".join([str(cs[0])] + [f"{c}*x{t + 1}" for t, c in enumerate(cs[1:]) if c])
            elif op == "add3":
                s1 = "-" if z3.is_true(m.eval(self.neg[i][0], model_completion=True)) else "+"
                s2 = "-" if z3.is_true(m.eval(self.neg[i][1], model_completion=True)) else "+"
                rhs = f"{nm(0)} {s1} {nm(1)} {s2} {nm(2)}"
            elif op == "imad":
                rhs = f"{nm(0)} * {nm(1)} + {nm(2)}"
            elif op == "lop3":
                rhs = f"lop3[{m.eval(self.lut[i], model_completion=True).as_long():#04x}]({nm(0)}, {nm(1)}, {nm(2)})"
            else:
                kind = ("<<", ">>", "rotl")[m.eval(self.shk[i], model_completion=True).as_long()]
                rhs = f"{nm(0)} {kind} {m.eval(self.sh[i], model_completion=True).as_long()}"
            names.append(f"r{i}")
            lines.append(f"r{i} = {rhs}")
        return lines


def synth(name, T, b, L, timeout_s, seed=1, ring=False):
    """CEGIS for a program of exactly L instructions; returns ('sat', lines) | ('unsat', None) | ('unknown', None)."""
    import random
    rng = random.Random(seed)
    n = T * SKETCHES[name][2]
    prog = Candidate(n, L, b, ring)
    syn = z3.SolverFor("QF_BV")
    syn.set("timeout", int(timeout_s * 1000))
    syn.add(prog.wf())
    mask = (1 << b) - 1
    examples = [[rng.randrange(1 << b) for _ in range(n)] for _ in range(4)]
    for e in examples:
        syn.add(prog.run([z3.BitVecVal(x, b) for x in e]) == spec(name, e, T, b))
    xs = [z3.BitVec(f"x{t}", b) for t in range(n)]
    target = spec_z3(name, xs, T, b)
    t0 = time.time()
    while True:
        left = timeout_s - (time.time() - t0)
        if left <= 0:
            return "unknown", None, len(examples)
        syn.set("timeout", int(left * 1000))
        r = syn.check()
        if r == z3.unsat:
            return "unsat", None, len(examples)
        if r != z3.sat:
            return "unknown", None, len(examples)
        m = syn.model()
        ver = z3.SolverFor("QF_BV")
        ver.set("timeout", int(max(left, 1) * 1000))
        out = z3.substitute(prog.run(xs), *[(v, m.eval(v, model_completion=True)) for v in _vars(prog)])
        ver.add(out != target)
        rv = ver.check()
        if rv == z3.unsat:
            return "sat", prog.describe(m), len(examples)
        if rv != z3.sat:
            return "unknown", None, len(examples)
        cm = ver.model()
        e = [cm.eval(x, model_completion=True).as_long() & mask for x in xs]
        examples.append(e)
        syn.add(prog.run([z3.BitVecVal(x, b) for x in e]) == spec(name, e, T, b))


def _vars(p):
    out = list(p.op) + list(p.lut) + list(p.sh) + list(p.shk)
    for row in p.sel + p.imm + p.lc + p.neg:
        out += list(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--steps", default="2,3,4")
    ap.add_argument("--sketch", default="all")
    ap.add_argument("--max-len", type=int, default=0, help="search lengths up to this (default: honest's - 1)")
    ap.add_argument("--honest", action="store_true", help="also synthesize at honest's length (a completeness check)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per (sketch, T, L) query")
    ap.add_argument("--ring", action="store_true",
                    help="shifts left only, so every component commutes with reduction mod 2^width: UNSAT at this width "
                         "then holds at 32 bits")
    ap.add_argument("--json")
    a = ap.parse_args()
    names = list(SKETCHES) if a.sketch == "all" else a.sketch.split(",")
    rows = []
    for name in names:
        for T in [int(s) for s in a.steps.split(",")]:
            if not SKETCHES[name][3](T):
                continue
            honest = SKETCHES[name][1](T)
            top = a.max_len or honest - 1
            if a.honest:
                top = max(top, honest)
            for L in range(1, top + 1):
                t0 = time.time()
                r, lines, nex = synth(name, T, a.width, L, a.timeout, ring=a.ring)
                row = {"sketch": name, "T": T, "width": a.width, "ring": a.ring, "len": L, "honest_len": honest, "result": r,
                       "examples": nex, "seconds": round(time.time() - t0, 2), "program": lines}
                rows.append(row)
                print(json.dumps(row), flush=True)
                if r == "sat" and L < honest:
                    break
    verdict = {}
    for row in rows:
        key = f"{row['sketch']}/T{row['T']}"
        v = verdict.setdefault(key, {"honest_len": row["honest_len"], "cheaper": None, "unsat_below": [], "unknown": []})
        if row["result"] == "sat" and row["len"] < row["honest_len"]:
            v["cheaper"] = row["len"]
        elif row["result"] == "unsat":
            v["unsat_below"].append(row["len"])
        elif row["result"] == "unknown":
            v["unknown"].append(row["len"])
    print("## verdicts", file=sys.stderr)
    for k, v in verdict.items():
        print(f"- {k}: honest {v['honest_len']}; cheaper program at length {v['cheaper']}; UNSAT at {v['unsat_below']};"
              f" unknown at {v['unknown']}", file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"rows": rows, "verdicts": verdict, "width": a.width, "ring": a.ring}, f, indent=1)


if __name__ == "__main__":
    main()
