"""RT2-HOPPER task 1: is there a program for li's fold on sm_90a under one issue slot a word? (note:20261019T1140Z-draft-rt2-hopper)

li's half step is `y <- (y ^ d0 ^ k) * a + d1` over 32-bit words (`segment._li_absorb`, k = LK[c][h], a = LA[c][h]), one
LOP3 and one IMAD: 1 instruction a word. DS-90 prices every instruction one issue slot, so a fold under 1 a word needs
fewer instructions. FOLD-MIN (`theory/fold_routes.py`, #2709) showed that a half step is no single instruction at 32 bits,
over a pool of test vectors. That is a bound on one half step alone. This script asks the questions a cheaper whole fold
would need a yes to, each as an exact synthesis query (CEGIS over z3 bit-vectors):

- `half2`: two consecutive half steps of one chain in 3 instructions (2 is honest's per half step, so 4 for two).
- `half3`: three in 5.
- `pair`: one half step of each of two chains (different keys) in 3: work shared across chains.
- `pair_xor`: two chains' half steps and the XOR of their results (the checkpoint's per-lane XOR) in 4.
- `*_enc`: the same with the chain state carried in any encoding `E(y) = ((y ^ K) * U + C)` (U odd), chosen freely at the
  window's input and its output, so a cheaper representation of the state between half steps is searched too.

The ISA (`OPS`): LOP3 with any of the 256 LUTs, IADD3 with any signs, IMAD `a * b ± c` (IMAD.WIDE's low word is the same
function), LEA `(a << s) + b`, every operand any input, any earlier result or a free constant (more freedom than SASS's one
immediate an instruction, so an UNSAT is stronger than needed). Each of these commutes with reduction mod 2^w: a 32-bit
program of them, reduced mod 2^w with its constants reduced, computes the reduced target. So `unsat` at width w proves
that no 32-bit program of these forms exists. `--high` adds the forms that move high bits down (IMAD.HI, IMAD.WIDE's
high word, the funnel shifts SHF.L/SHF.R, LEA.HI, and PRMT's byte selection). These don't commute with reduction, so with
`--high` the query runs at w = 32 itself.

Each query has a control: the same target at the honest count (2 an absorbed half step, plus 1 for each XOR) must be
`sat`, with the program z3 returns verified on every input. A control that fails means the encoding can't find
programs, and no `unsat` beside it counts.

Run: `python li_window_sat.py [--width W ...] [--high] [--only NAME ...] [--json OUT]`. It prints one line a query, the
verdict, the time, the CEGIS rounds and the program when sat.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verity.ml.protocols.pouw.circuit import segment as SG  # noqa: E402

LOW = ("LOP3", "IADD3", "IMAD", "LEA")
HIGH = ("IMAD.HI", "IMAD.WIDE.HI", "SHF.L", "SHF.R", "LEA.HI", "PRMT")


class Program:
    """n instructions over `nin` inputs at width w, each with an opcode, three operand selectors (an input, an earlier
    result or the slot's own constant), a LUT, three signs and a shift; every field a z3 variable."""

    def __init__(self, n: int, nin: int, w: int, ops: tuple[str, ...], outs: int):
        self.n, self.nin, self.w, self.ops, self.outs = n, nin, w, ops, outs
        self.vars: list[z3.ExprRef] = []
        self.cons: list[z3.BoolRef] = []
        bits = max(1, (nin + n).bit_length())
        self.ins = []
        for j in range(n):
            op = self._var(z3.BitVec(f"op{j}", 4))
            self.cons.append(z3.ULT(op, len(ops)))
            sel = [self._var(z3.BitVec(f"sel{j}_{t}", bits + 1)) for t in range(3)]
            for s in sel:
                self.cons.append(z3.ULE(s, nin + j))
            k = [self._var(z3.BitVec(f"k{j}_{t}", w)) for t in range(3)]
            lut = self._var(z3.BitVec(f"lut{j}", 8))
            sg = self._var(z3.BitVec(f"sg{j}", 3))
            sh = self._var(z3.BitVec(f"sh{j}", 6))
            self.cons.append(z3.ULT(sh, w))
            prm = [self._var(z3.BitVec(f"prm{j}_{t}", 3)) for t in range(4)]
            self.ins.append(dict(op=op, sel=sel, k=k, lut=lut, sg=sg, sh=sh, prm=prm))
        # which results are the outputs: the last `outs` instructions' results by default, any results when outs > 1
        self.osel = [self._var(z3.BitVec(f"out{i}", bits + 1)) for i in range(outs)]
        for i, o in enumerate(self.osel):
            self.cons.append(z3.UGE(o, nin) if n else o == 0)
            self.cons.append(z3.ULT(o, nin + n))
        if outs == 1 and n:
            self.cons.append(self.osel[0] == nin + n - 1)

    def _var(self, v):
        self.vars.append(v)
        return v

    def run(self, inputs: list, m: z3.ModelRef | None = None) -> list:
        """The outputs on `inputs` (z3 bit-vectors, symbolic or concrete); with `m`, the program the model fixes."""
        w = self.w
        ev = (lambda e: m.eval(e, model_completion=True)) if m is not None else (lambda e: e)
        vals = list(inputs)
        for j, ins in enumerate(self.ins):
            sel = [ev(s) for s in ins["sel"]]
            k = [ev(x) for x in ins["k"]]
            opd = []
            for t in range(3):
                e = k[t]
                for i in range(len(vals) - 1, -1, -1):
                    e = z3.If(sel[t] == i, vals[i], e)
                opd.append(e)
            a, b, c = opd
            res = self._ops(ev(ins["op"]), a, b, c, ev(ins["lut"]), ev(ins["sg"]), ev(ins["sh"]),
                            [ev(p) for p in ins["prm"]])
            vals.append(z3.simplify(res) if m is not None else res)
        out = []
        for o in self.osel:
            o = ev(o)
            e = vals[-1]
            for i in range(len(vals) - 1, -1, -1):
                e = z3.If(o == i, vals[i], e)
            out.append(e)
        return out

    def _ops(self, op, a, b, c, lut, sg, sh, prm):
        w = self.w
        ones = z3.BitVecVal((1 << w) - 1, w)
        forms = []
        for name in self.ops:
            if name == "LOP3":
                r = z3.BitVecVal(0, w)
                for mt in range(8):
                    bit = z3.SignExt(w - 1, z3.Extract(mt, mt, lut))
                    ta = a if mt & 1 else a ^ ones
                    tb = b if mt & 2 else b ^ ones
                    tc = c if mt & 4 else c ^ ones
                    r = r | (bit & ta & tb & tc)
                forms.append(r)
            elif name == "IADD3":
                def s(x, i):
                    return z3.If(z3.Extract(i, i, sg) == 1, -x, x)
                forms.append(s(a, 0) + s(b, 1) + s(c, 2))
            elif name == "IMAD":
                forms.append(a * b + z3.If(z3.Extract(2, 2, sg) == 1, -c, c))
            elif name == "LEA":
                forms.append((a << z3.ZeroExt(w - 6, sh) if w > 6 else a << z3.Extract(w - 1, 0, sh)) + b)
            elif name in ("IMAD.HI", "IMAD.WIDE.HI"):
                p = z3.ZeroExt(w, a) * z3.ZeroExt(w, b)
                if name == "IMAD.HI":
                    forms.append(z3.Extract(2 * w - 1, w, p) + c)
                else:
                    forms.append(z3.Extract(2 * w - 1, w, p + z3.ZeroExt(w, c)))
            elif name in ("SHF.L", "SHF.R", "LEA.HI"):
                pair = z3.Concat(b, a)   # hi:lo
                s6 = z3.ZeroExt(2 * w - 6, sh)
                if name == "SHF.R":
                    forms.append(z3.Extract(w - 1, 0, z3.LShR(pair, s6)))
                else:
                    hi = z3.Extract(2 * w - 1, w, pair << s6)
                    forms.append(hi + c if name == "LEA.HI" else hi)
            elif name == "PRMT":
                assert w % 8 == 0
                src = [z3.Extract(8 * i + 7, 8 * i, a) for i in range(w // 8)] + \
                      [z3.Extract(8 * i + 7, 8 * i, b) for i in range(w // 8)]
                bytes_ = []
                for i in range(w // 8):
                    e = src[-1]
                    for q in range(len(src) - 1, -1, -1):
                        e = z3.If(prm[i] == q, src[q], e)
                    bytes_.append(e)
                forms.append(z3.Concat(*reversed(bytes_)))
            else:
                raise ValueError(name)
        r = forms[-1]
        for i in range(len(forms) - 2, -1, -1):
            r = z3.If(op == i, forms[i], r)
        return r

    def describe(self, m: z3.ModelRef) -> list[str]:
        names = [f"in{i}" for i in range(self.nin)]
        lines = []
        for j, ins in enumerate(self.ins):
            op = self.ops[m.eval(ins["op"], model_completion=True).as_long()]
            ops = []
            for t in range(3):
                s = m.eval(ins["sel"][t], model_completion=True).as_long()
                ops.append(names[s] if s < len(names) else hex(m.eval(ins["k"][t], model_completion=True).as_long()))
            extra = {"LOP3": f"lut={m.eval(ins['lut'], model_completion=True).as_long():#04x}",
                     "IADD3": f"signs={m.eval(ins['sg'], model_completion=True).as_long():03b}",
                     "IMAD": f"negc={m.eval(ins['sg'], model_completion=True).as_long() >> 2 & 1}"}.get(op, "")
            if op in ("LEA", "SHF.L", "SHF.R", "LEA.HI"):
                extra = f"sh={m.eval(ins['sh'], model_completion=True).as_long()}"
            lines.append(f"v{j} = {op}({', '.join(ops)}) {extra}".rstrip())
            names.append(f"v{j}")
        outs = [names[m.eval(o, model_completion=True).as_long()] for o in self.osel]
        lines.append(f"out = {', '.join(outs)}")
        return lines


# ---- the targets ----------------------------------------------------------------------------------------------------

def half(y, d0, d1, k, a):
    return (y ^ d0 ^ k) * a + d1


def key(w: int, c: int, h: int):
    m = (1 << w) - 1
    return z3.BitVecVal(SG.LK[c][h] & m, w), z3.BitVecVal(SG.LA[c][h] & m, w)


class Target:
    """A query's target: `nin` inputs, its outputs as a function of them and of the encoding variables (`enc`), and the
    honest count (`honest`)."""

    def __init__(self, name: str, w: int, encoded: bool = False):
        self.name, self.w, self.encoded = name, w, encoded
        self.enc_vars: list[z3.ExprRef] = []
        self.enc_cons: list[z3.BoolRef] = []
        base = name.removesuffix("_enc")
        self.base = base
        if base in ("half1", "half2", "half3"):
            self.k = int(base[-1])
            self.nin = 1 + 2 * self.k
            self.honest = 2 * self.k
            self.nout = 1
        elif base == "pair":
            self.nin, self.honest, self.nout = 6, 4, 2
        elif base == "pair_xor":
            self.nin, self.honest, self.nout = 6, 5, 3
        else:
            raise ValueError(name)
        if encoded:
            self.din = self._enc("in")
            self.eout = self._enc("out")

    def _enc(self, tag):
        K, U, C = (z3.BitVec(f"enc_{tag}_{x}", self.w) for x in "KUC")
        self.enc_vars += [K, U, C]
        self.enc_cons.append(z3.Extract(0, 0, U) == 1)
        return K, U, C

    @staticmethod
    def _apply(e, x):
        K, U, C = e
        return ((x ^ K) * U) + C

    def outputs(self, xs: list) -> list:
        w = self.w
        if self.base.startswith("half"):
            y = xs[0]
            if self.encoded:
                y = self._apply(self.din, y)
            for h in range(self.k):
                k, a = key(w, 0, h % 2)
                y = half(y, xs[1 + 2 * h], xs[2 + 2 * h], k, a)
            return [self._apply(self.eout, y) if self.encoded else y]
        kA, aA = key(w, 0, 0)
        kB, aB = key(w, 1, 0)
        yA = half(xs[0], xs[1], xs[2], kA, aA)
        yB = half(xs[3], xs[4], xs[5], kB, aB)
        return [yA, yB] if self.base == "pair" else [yA, yB, yA ^ yB]


def cegis(t: Target, n: int, ops: tuple[str, ...], seed: int, max_rounds: int = 400, limit_s: int = 0) -> dict:
    w = t.w
    rng = random.Random(seed)
    prog = Program(n, t.nin, w, ops, t.nout)
    synth = z3.Solver()
    if limit_s:
        synth.set("timeout", 1000 * limit_s)
    synth.add(prog.cons + t.enc_cons)
    t0 = time.time()
    rounds = 0

    def add_point(pt):
        xs = [z3.BitVecVal(v, w) for v in pt]
        got = prog.run(xs)
        want = t.outputs(xs)
        for g, h in zip(got, want):
            synth.add(g == h)

    for _ in range(4):
        add_point([rng.getrandbits(w) for _ in range(t.nin)])
    while rounds < max_rounds:
        rounds += 1
        r = synth.check()
        if r == z3.unsat:
            return {"verdict": "unsat", "rounds": rounds, "sec": round(time.time() - t0, 2)}
        if r != z3.sat:
            return {"verdict": str(r), "rounds": rounds, "sec": round(time.time() - t0, 2)}
        m = synth.model()
        xs = [z3.BitVec(f"x{i}", w) for i in range(t.nin)]
        got = prog.run(xs, m)
        want = [z3.substitute(o, *[(v, m.eval(v, model_completion=True)) for v in t.enc_vars])
                for o in t.outputs(xs)]
        ver = z3.Solver()
        ver.add(z3.Or([g != h for g, h in zip(got, want)]))
        vr = ver.check()
        if vr == z3.unsat:
            enc = {str(v): m.eval(v, model_completion=True).as_long() for v in t.enc_vars}
            return {"verdict": "sat", "rounds": rounds, "sec": round(time.time() - t0, 2),
                    "program": prog.describe(m), "encoding": enc}
        cm = ver.model()
        add_point([cm.eval(x, model_completion=True).as_long() for x in xs])
    return {"verdict": "rounds-exhausted", "rounds": rounds, "sec": round(time.time() - t0, 2)}


QUERIES = ("half1", "half2", "half3", "pair", "pair_xor", "half1_enc", "half2_enc")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--width", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--high", action="store_true", help="add the high-to-low forms; runs at w = 32")
    ap.add_argument("--only", nargs="+", choices=QUERIES)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--seed", type=int, default=0x71)
    ap.add_argument("--limit-s", type=int, default=0, help="z3's limit on one synthesis check (0: none); 'unknown' past it")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    ops = LOW + HIGH if args.high else LOW
    widths = [32] if args.high else args.width
    rows, bad = [], 0
    for w in widths:
        for q in args.only or QUERIES:
            t = Target(q, w, encoded=q.endswith("_enc"))
            plan = [(t.honest - 1, "bound")]
            if not args.no_controls:
                plan.append((t.honest, "control"))
            for n, role in plan:
                r = cegis(t, n, ops, args.seed, limit_s=args.limit_s)
                ok = r["verdict"] == ("unsat" if role == "bound" else "sat")
                bad += not ok
                row = {"query": q, "width": w, "n": n, "role": role, "ops": list(ops), "ok": ok, **r}
                rows.append(row)
                print(f"{'PASS' if ok else 'FAIL'} w={w} {q} n={n} ({role}): {r['verdict']} in {r['sec']} s, "
                      f"{r['rounds']} rounds", flush=True)
                for line in r.get("program", []):
                    print(f"    {line}", flush=True)
                if args.json:
                    args.json.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(rows) - bad} of {len(rows)} as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
