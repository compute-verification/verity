"""Search for programs cheaper than honest over windows of s steps of the multiply-xor fold N <- (N ^ D_t) k_t, by SAT.

Each instance asks: is there a straight-line program of L instructions that maps a window's inputs to its output on
every example? `unsat` at L = honest - 1 means no such program exists over the instruction set, with every free term
below granted; `sat` at L = honest is the sanity check that the encoding finds the honest program.

Targets (k_t a live odd word a step and call, all of a call's keys known at its start):

  zero     D_t = R_t + C_t: R_t the row's coin prefix, C_t the tile column's (zero data); honest 3 a step
  offset   D_t = d + R_t + C_t with d a per-cell offset (chosen data: the data's word fixed after its first step);
           honest 3 a step
  words    D_t given per cell (data the prover can't choose: the MMA writes each word); honest 2 a step beyond it

Instruction set: LOP3 (any of the 256 functions), IADD3 (any operand negated), IMAD (a b +- c, the product negatable;
covers IMUL, IMAD.SHL, IMAD.MOV and IMAD.WIDE's low word) and LEA ((a << s) +- b, any s; covers SHL). Each is a
T-function (output bit i reads input bits <= i), so a 32-bit program's low W bits are a W-bit program of the same
length with its free terms and constants truncated: `unsat` at width W proves there is no 32-bit program of that
length over this set. Instructions that move high bits down (PRMT, IDP.4A, IMAD.WIDE's high word, IMUL.HI, shifts
right, IMNMX) are outside it; `fold_search.py`'s 32-bit search covers them for one step.

Free terms, each an arbitrary function given as one symbolic value per tuple it may read: of the call's keys (F of
them), of a row's coins and the keys (F), and of a column's coins and the keys (F). The inputs R_t, C_t and k_t are
given besides.

Modes (transformed-state loops):

  plain    the state is N on entry and N_s on exit
  basis    the state at each window boundary is phi_b(N) = ((N ^ a) b ^ c) d + e, a..e free per boundary with b, d
           odd (a bijection: a factored key product such as N P^-1, offsets, both), not counted. Consecutive
           windows share their boundary's basis (three chained key groups), as a loop must: with independent bases
           at a window's two ends, the exit basis absorbs the window's last multiply, an edge effect a loop pays back
           at the next window
  basisrc  basis with a row's and a column's own xor and add offsets at each boundary:
           ((N ^ a ^ x_i ^ y_j) b ^ c) d + e + u_i + v_j
  carry    the state is two words (u, v) with N = u + v (or u ^ v, the solver's choice, the same at both ends), u
           arbitrary (examples draw it), on entry and on exit: carry-save forms that accept any split

  full     (32 bits only, no reduction) the set above plus IMAD.WIDE's high word, SHF.R, PRMT with a register
           selector, IDP.4A and IMNMX

An `unsat` proves something only if every example is an input a call can reach. The `window` set draws R_t and C_t
freely at each step: it covers coin layouts whose prefixes take every residue. The `layout` set draws them as the
lead's walks truncated to W bits: a row increment is an 8-bit coin b in [0, 254] (a u8 lane 1 + b beside a -1 lane), a
column increment is odd (three odd coin lanes at odd weights, at most 16,383), so C_t's parity is t's. A window there
starts at a step t0 of parity `phase` (each instance runs at both, as a program may depend on t), or at the call's
start, where N_0 is

  zero     0
  key      kappa, a live word of the call (a source, as the keys are)
  rowcol   a_i ^ b_j, live words of the row and of the column (sources; the free terms read them)

Every window of s ≥ 2 runs at width 8 and again at width 6 (a smaller formula; `unsat` there is the same proof).

`--targets` writes the targets instead: each one's word form, honest cost a step and 32-bit test vectors.

usage: fold_window.py OUT.json [--set=window|layout] [--width=W] [--timeout=S] [--jobs=J] [--only=NAME,...] [--quick]
       fold_window.py OUT.json --targets
"""
import json
import multiprocessing as mp
import random
import sys
import time

import z3

OPS = ("LOP3", "IADD3", "IMAD", "LEA")


def honest(target, s):
    return (2 if target == "words" else 3) * s


def fold_ref(n, ds, ks, mask):
    for d, k in zip(ds, ks):
        n = ((n ^ d) * k) & mask
    return n


class Program:
    """L slots; slot j reads three of the sources and slots before it. `full` adds, at 32 bits, the instructions
    that move high bits down: IMAD.WIDE's high word (a b + c, signed or unsigned; covers IMUL.HI), SHF.R (the funnel
    (b:a) >> s; covers SHR and rotates), PRMT (each byte of (b:a) chosen by c's nibbles, a register selector covering
    every immediate one), IDP.4A (four byte products, either signedness each side, + c) and IMNMX."""

    def __init__(self, tag, L, nsrc, w, full=False):
        self.L, self.nsrc, self.w, self.full = L, nsrc, w, full
        sb = max(1, (nsrc + L).bit_length())
        self.op = [z3.BitVec(f"{tag}op{j}", 3 if full else 2) for j in range(L)]
        self.sub = [z3.BitVec(f"{tag}sub{j}", 2) for j in range(L)]
        self.sel = [[z3.BitVec(f"{tag}sel{j}_{o}", sb) for o in range(3)] for j in range(L)]
        self.lut = [z3.BitVec(f"{tag}lut{j}", 8) for j in range(L)]
        self.neg = [[z3.Bool(f"{tag}neg{j}_{o}") for o in range(3)] for j in range(L)]
        self.sh = [z3.BitVec(f"{tag}sh{j}", w) for j in range(L)]

    def constraints(self):
        cs = []
        for j in range(self.L):
            for o in range(3):
                cs.append(z3.ULT(self.sel[j][o], self.nsrc + j))
            cs.append(z3.ULT(self.sh[j], self.w))
            if self.full:
                cs.append(z3.ULE(self.op[j], 7))
            s0, s1, s2 = self.sel[j]
            # LOP3 and IADD3 are symmetric in their operands up to the lut and the signs; IMAD in its factors
            cs.append(z3.Implies(z3.Or(self.op[j] == 0, self.op[j] == 1), z3.And(z3.ULE(s0, s1), z3.ULE(s1, s2))))
            cs.append(z3.Implies(self.op[j] == 2, z3.ULE(s0, s1)))
        return cs

    def run(self, srcs):
        vals = list(srcs)
        ones = z3.BitVecVal((1 << self.w) - 1, self.w)
        zero = z3.BitVecVal(0, self.w)
        for j in range(self.L):
            ops = []
            for o in range(3):
                v = vals[-1]
                for i in range(len(vals) - 2, -1, -1):
                    v = z3.If(self.sel[j][o] == i, vals[i], v)
                ops.append(v)
            a, b, c = ops
            na, nb, nc = self.neg[j]
            lop = zero
            for m in range(8):
                t = z3.If(z3.Extract(m, m, self.lut[j]) == 1, ones, zero)
                t = t & (a if m & 4 else ~a) & (b if m & 2 else ~b) & (c if m & 1 else ~c)
                lop = lop | t
            add = z3.If(na, -a, a) + z3.If(nb, -b, b) + z3.If(nc, -c, c)
            mad = z3.If(na, -(a * b), a * b) + z3.If(nc, -c, c)
            lea = (a << self.sh[j]) + z3.If(nb, -b, b)
            v = z3.If(self.op[j] == 0, lop, z3.If(self.op[j] == 1, add, z3.If(self.op[j] == 2, mad, lea)))
            if self.full:
                v = z3.If(z3.ULE(self.op[j], 3), v, self.wide_ops(j, a, b, c))
            vals.append(v)
        return vals

    def wide_ops(self, j, a, b, c):
        w, sub = self.w, self.sub[j]
        sx = z3.Extract(0, 0, sub) == 1
        sy = z3.Extract(1, 1, sub) == 1
        ext = lambda x, sgn: z3.If(sgn, z3.SignExt(w, x), z3.ZeroExt(w, x))
        hi = z3.Extract(2 * w - 1, w, ext(a, sx) * ext(b, sy) + z3.ZeroExt(w, c))
        shf = z3.Extract(w - 1, 0, z3.LShR(z3.Concat(b, a), z3.ZeroExt(w, self.sh[j])))
        pair = z3.Concat(b, a)
        byts = [z3.Extract(8 * i + 7, 8 * i, pair) for i in range(8)]
        outb = []
        for i in range(w // 8):
            f = z3.Extract(4 * i + 2, 4 * i, c)
            x = byts[7]
            for q in range(6, -1, -1):
                x = z3.If(f == q, byts[q], x)
            outb.append(x)
        prmt = z3.Concat(*reversed(outb))
        idp = c
        for i in range(w // 8):
            ai, bi = z3.Extract(8 * i + 7, 8 * i, a), z3.Extract(8 * i + 7, 8 * i, b)
            ea = z3.If(sx, z3.SignExt(w - 8, ai), z3.ZeroExt(w - 8, ai))
            eb = z3.If(sy, z3.SignExt(w - 8, bi), z3.ZeroExt(w - 8, bi))
            idp = idp + ea * eb
        lt = z3.If(sub == 2, a < b, z3.ULT(a, b))
        mnmx = z3.If(self.neg[j][0], z3.If(lt, a, b), z3.If(lt, b, a))
        op = self.op[j]
        return z3.If(op == 4, hi, z3.If(op == 5, z3.If(z3.Or(sub == 1, sub == 2), mnmx, shf),
                                        z3.If(op == 6, prmt, idp)))

    def decode(self, m):
        out = []
        for j in range(self.L):
            op = m.eval(self.op[j], model_completion=True).as_long()
            sel = [m.eval(x, model_completion=True).as_long() for x in self.sel[j]]
            neg = [z3.is_true(m.eval(x, model_completion=True)) for x in self.neg[j]]
            sub = m.eval(self.sub[j], model_completion=True).as_long()
            if op >= 4:
                name = {4: "IMAD.WIDE.hi", 5: "IMNMX" if sub in (1, 2) else "SHF.R", 6: "PRMT", 7: "IDP.4A"}[op]
                out.append(f"v{self.nsrc + j} = {name}[sub {sub}, neg {neg}, sh "
                           f"{m.eval(self.sh[j], model_completion=True).as_long()}](s{sel[0]}, s{sel[1]}, s{sel[2]})")
            elif op == 0:
                out.append(f"v{self.nsrc + j} = LOP3.{m.eval(self.lut[j], model_completion=True).as_long():#04x}"
                           f"(s{sel[0]}, s{sel[1]}, s{sel[2]})")
            elif op == 1:
                out.append(f"v{self.nsrc + j} = IADD3(" + ", ".join(("-" if n else "+") + f"s{x}"
                                                                   for n, x in zip(neg, sel)) + ")")
            elif op == 2:
                out.append(f"v{self.nsrc + j} = IMAD({'-' if neg[0] else ''}s{sel[0]} * s{sel[1]}, "
                           f"{'-' if neg[2] else '+'}s{sel[2]})")
            else:
                out.append(f"v{self.nsrc + j} = LEA(s{sel[0]} << {m.eval(self.sh[j], model_completion=True).as_long()}"
                           f", {'-' if neg[1] else '+'}s{sel[1]})")
        return out


def layout_walks(rng, s, t0):
    """Draws of the lead's zero-data row and column prefixes at steps t0 .. t0 + s - 1."""
    def walk(inc):
        x, out = 0, []
        for t in range(1, t0 + s):
            x += inc()
            if t >= t0:
                out.append(x)
        return out
    return (lambda: walk(lambda: rng.randrange(0, 255)),
            lambda: walk(lambda: 2 * rng.randrange(-8192, 8192) + 1))


def instance(spec, w, seed):
    """Build the solver for one (target, s, mode, L, F); return (solver, program, source names)."""
    target, s, mode, L, F = spec["target"], spec["s"], spec["mode"], spec["L"], spec["F"]
    mask = (1 << w) - 1
    rng = random.Random(seed)
    odd = lambda: rng.randrange(0, 1 << w) | 1
    if w == 32:
        # the served domains: row prefixes of an 8-bit coin over at most 293 steps, column prefixes, and words and
        # offsets within the s32 gate's data bound
        row_v = lambda: rng.randrange(0, 255 * 293 + 1)
        col_v = lambda: rng.randrange(-500_000, 500_000) & mask
        word_v = lambda: rng.randrange(-(1 << 27), 1 << 27) & mask
    else:
        row_v = col_v = word_v = lambda: rng.randrange(0, 1 << w)
    chained = mode in ("basis", "basisrc")
    nk, nr, nc, nn = spec.get("grid", (3, 2, 3, 4) if chained else (2, 3, 3, 4))
    # basis modes chain their key groups: group a's window reads keys[a + 1] and maps basis a to basis a + 1, so a
    # boundary's basis is one function for the window that ends there and the one that starts there; on the lead's
    # walks group a reads the walk's a-th segment, as consecutive windows do
    chain = [[odd() for _ in range(s)] for _ in range(nk + 1)]
    keys = chain[1:] if chained else chain[:nk]
    start = spec.get("start")
    if spec.get("layout"):
        row_walk, col_walk = layout_walks(rng, nk * s if chained else s, 1 if start else 40 + spec["phase"])
        rows = [[x & mask for x in row_walk()] for _ in range(nr)]
        cols = [[x & mask for x in col_walk()] for _ in range(nc)]
    else:
        rows = [[row_v() for _ in range(s)] for _ in range(nr)]
        cols = [[col_v() for _ in range(s)] for _ in range(nc)]
    if start:
        kappa = [rng.randrange(0, 1 << w) for _ in range(nk)]
        arow = [rng.randrange(0, 1 << w) for _ in range(nr)]
        bcol = [rng.randrange(0, 1 << w) for _ in range(nc)]
    kf = [[z3.BitVec(f"kf{f}_{a}", w) for a in range(nk)] for f in range(F)]
    rf = [[[z3.BitVec(f"rf{f}_{a}_{r}", w) for r in range(nr)] for a in range(nk)] for f in range(F)]
    cf = [[[z3.BitVec(f"cf{f}_{a}_{c}", w) for c in range(nc)] for a in range(nk)] for f in range(F)]
    names, examples = [], []
    for a, ks in enumerate(keys):
        if target == "words":
            for _ in range(nr * nc * nn):
                n = rng.randrange(0, 1 << w)
                ds = [word_v() for _ in range(s)]
                elem = [n] + ds
                examples.append((a, elem, [], [], fold_ref(n, ds, ks, mask), ks, n))
            names = ["N"] + [f"D{t + 1}" for t in range(s)]
            continue
        seg = slice(a * s, (a + 1) * s) if len(rows[0]) > s else slice(0, s)
        for r, rr in enumerate(row[seg] for row in rows):
            for c, cc in enumerate(col[seg] for col in cols):
                for _ in range(nn):
                    n = rng.randrange(0, 1 << w)
                    lead = [n]
                    if start:
                        n = {"zero": 0, "key": kappa[a], "rowcol": arow[r] ^ bcol[c]}[start]
                        lead = [arow[r], bcol[c]] if start == "rowcol" else [n]
                    d0 = word_v() if target == "offset" else 0
                    ds = [(d0 + rr[t] + cc[t]) & mask for t in range(s)]
                    elem = lead + ([d0] if target == "offset" else [])
                    examples.append((a, elem, [(r, x) for x in rr], [(c, x) for x in cc],
                                     fold_ref(n, ds, ks, mask), ks, n))
        names = {"rowcol": ["a", "b"], "key": ["kappa"], "zero": ["N0"]}.get(start, ["N"]) + \
                (["d"] if target == "offset" else []) + [f"R{t + 1}" for t in range(s)] + \
                [f"C{t + 1}" for t in range(s)]
    names = names + [f"k{t + 1}" for t in range(s)] + [f"keyfun{f}" for f in range(F)]
    if target != "words":
        names += [f"rowfun{f}" for f in range(F)] + [f"colfun{f}" for f in range(F)]
    if mode == "carry":
        names = ["v"] + names
    sol = z3.SolverFor("QF_BV")
    prog = Program("p", L, len(names), w, full=spec.get("full", False))
    sol.add(prog.constraints())
    basis = {}
    if chained:
        basis = [[z3.BitVec(f"basis{x}_{b}", w) for x in "abcde"] for b in range(nk + 1)]
        for b in range(nk + 1):
            sol.add(z3.Extract(0, 0, basis[b][1]) == 1, z3.Extract(0, 0, basis[b][3]) == 1)
        zero = z3.BitVecVal(0, w)
        # basisrc: a row's and a column's own xor and add offsets at each boundary, as precomputation may give them
        rb = [[[z3.BitVec(f"rbasis{x}_{b}_{r}", w) if mode == "basisrc" else zero for x in "xa"] for r in range(nr)]
              for b in range(nk + 1)]
        cb = [[[z3.BitVec(f"cbasis{x}_{b}_{c}", w) if mode == "basisrc" else zero for x in "xa"] for c in range(nc)]
              for b in range(nk + 1)]

        def phi(x, b, r, c):
            p = basis[b]
            return ((((x ^ p[0] ^ rb[b][r][0] ^ cb[b][c][0]) * p[1]) ^ p[2]) * p[3]) + p[4] + rb[b][r][1] + cb[b][c][1]
    if mode == "carry":
        xor_form = z3.Bool("carry_xor")
    W = lambda x: z3.BitVecVal(x & mask, w)
    for (a, elem, rr, cc, want, ks, n) in examples:
        srcs = [W(x) for x in elem]
        if target != "words":
            srcs += [W(x) for _, x in rr] + [W(x) for _, x in cc]
        srcs += [W(k) for k in ks] + [kf[f][a] for f in range(F)]
        if target != "words":
            r_idx, c_idx = rr[0][0], cc[0][0]
            srcs += [rf[f][a][r_idx] for f in range(F)] + [cf[f][a][c_idx] for f in range(F)]
        if chained:
            r_idx, c_idx = (rr[0][0], cc[0][0]) if target != "words" else (0, 0)
            srcs[0] = phi(srcs[0], a, r_idx, c_idx)
            sol.add(prog.run(srcs)[-1] == phi(W(want), a + 1, r_idx, c_idx))
        elif mode == "carry":
            u = W(rng.randrange(0, 1 << w))
            v = z3.If(xor_form, W(n) ^ u, W(n) - u)
            vals = prog.run([u, v] + srcs[1:])
            o1, o2 = vals[-2], vals[-1]
            sol.add(z3.If(xor_form, o1 ^ o2, o1 + o2) == W(want))
        else:
            sol.add(prog.run(srcs)[-1] == W(want))
    if mode == "carry":
        names[0], names[1] = "u", "v"
    return sol, prog, names, len(examples)


def solve(spec):
    w, seed, timeout = spec["width"], spec["seed"], spec["timeout"]
    t0 = time.time()
    sol, prog, names, nex = instance(spec, w, seed)
    sol.set("timeout", int(timeout * 1000))
    r = sol.check()
    out = dict(spec, result=str(r), seconds=round(time.time() - t0, 2), examples=nex, sources=names)
    if r == z3.sat:
        out["program"] = prog.decode(sol.model())
    return out


def specs(width, timeout, quick, which="window"):
    out = []

    def add(target, s, mode, L, w, full=False, grid=None, phase=None, start=None):
        h = honest(target, s)
        name = f"{target}-s{s}-{'start' + start if start else mode}-L{L}-w{w}" + ("-full" if full else "")
        x = {"name": name + (f"-p{phase}" if phase is not None else ""), "target": target, "s": s, "mode": mode,
             "L": L, "F": s, "honest": h, "width": w, "seed": 1000 * s + L, "timeout": timeout, "full": full}
        if grid:
            x["grid"] = grid
        if phase is not None or start:
            x.update(layout=True, phase=phase, start=start)
        out.append(x)

    if which == "layout":
        for target in ("zero", "offset"):
            for p in (0, 1):
                for mode in ("plain", "basis", "basisrc", "carry"):
                    add(target, 1, mode, 2, width, phase=p)
                    add(target, 2, mode, 5, 6, phase=p)
                add(target, 2, "plain", 5, width, phase=p)
                add(target, 3, "plain", 8, 6, phase=p)
            add(target, 1, "plain", 3, width, phase=0)
        add("zero", 1, "basisrc", 3, width, phase=0)
        for start, Ls in (("zero", {1: (1,), 2: (3, 4)}), ("key", {1: (2, 3), 2: (5,)}), ("rowcol", {1: (2, 3), 2: (5,)})):
            for s, ls in Ls.items():
                for L in ls:
                    add("zero", s, "plain", L, width, grid=(3, 4, 4, 1), start=start)
                    if s == 2:
                        add("zero", s, "plain", L, 6, grid=(3, 4, 4, 1), start=start)
        return out
    for target in ("zero", "offset", "words"):
        for s in (1, 2, 3):
            h = honest(target, s)
            for mode in ("plain", "basis", "carry"):
                if mode != "plain" and s == 3:
                    continue
                for L in (h - 1, h):
                    if L == h and mode != "plain" and not (mode == "basis" and s == 1):
                        continue
                    add(target, s, mode, L, width)
            if s >= 2:
                add(target, s, "plain", h - 1, 6)
    # 32 bits with the instructions that move high bits down (no width reduction there)
    add("words", 1, "plain", 1, 32, full=True, grid=(2, 1, 1, 4))
    add("words", 1, "plain", 2, 32, full=True, grid=(2, 1, 1, 4))
    add("zero", 1, "plain", 2, 32, full=True, grid=(2, 2, 2, 1))
    add("words", 2, "plain", 3, 32, full=True, grid=(2, 1, 1, 4))
    if quick:
        out = [x for x in out if x["s"] == 1 and x["mode"] == "plain" and not x["full"]]
    return out


def targets(cases=4, steps=3, seed=0):
    """Each target's word form, honest cost a step and 32-bit test vectors on the lead's walks (inputs, keys, and N_t
    after each step), for another harness to check a strategy against bit for bit."""
    rng = random.Random(seed)
    mask = (1 << 32) - 1
    out = {"fold": "N_t = (N_{t-1} ^ D_t) k_t mod 2^32, k_t a live odd u32 a step and call (all of a call's keys known "
                   "at its start); N_T and D_T committed",
           "row_increment": "b in [0, 254]", "column_increment": "odd, |x| <= 16383"}
    for target, form in (("zero", "R_t + C_t"), ("offset", "d + R_t + C_t"), ("words", "given")):
        for start in (None, "zero", "key", "rowcol") if target == "zero" else (None,):
            row_walk, col_walk = layout_walks(rng, steps, 1 if start else 40)
            vecs = []
            for _ in range(cases):
                v = {"k": [rng.randrange(0, 1 << 32) | 1 for _ in range(steps)]}
                if target == "words":
                    v["D"] = ds = [rng.randrange(-(1 << 27), 1 << 27) & mask for _ in range(steps)]
                else:
                    v["R"], v["C"] = row_walk(), [x & mask for x in col_walk()]
                    v["d"] = rng.randrange(-(1 << 27), 1 << 27) & mask if target == "offset" else 0
                    ds = [(v["d"] + r + c) & mask for r, c in zip(v["R"], v["C"])]
                v["N0"] = 0 if start == "zero" else rng.randrange(0, 1 << 32)
                if start == "key":
                    v["kappa"] = v["N0"]
                if start == "rowcol":
                    v["a"], v["b"] = v["N0"], rng.randrange(0, 1 << 32)
                    v["N0"] = v["a"] ^ v["b"]
                v["N"] = [fold_ref(v["N0"], ds[:t + 1], v["k"][:t + 1], mask) for t in range(steps)]
                vecs.append(v)
            out[target + (f"-start{start}" if start else "")] = {"D_t": form, "honest_a_step": honest(target, 1),
                                                                 "vectors": vecs}
    return out


def main():
    args = sys.argv[1:]
    out_path = args[0]
    opt = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in args[1:]}
    if "--targets" in opt:
        with open(out_path, "w") as f:
            json.dump(targets(), f, indent=1)
        return 0
    width = int(opt.get("--width", 8))
    timeout = float(opt.get("--timeout", 3600))
    jobs = int(opt.get("--jobs", mp.cpu_count()))
    todo = specs(width, timeout, "--quick" in opt, opt.get("--set", "window"))
    if "--only" in opt:
        keep = set(opt["--only"].split(","))
        todo = [x for x in todo if x["name"] in keep]
    todo.sort(key=lambda x: (x["L"], x["s"]))
    results = []
    with mp.Pool(jobs) as pool:
        for res in pool.imap_unordered(solve, todo):
            results.append(res)
            print(json.dumps({k: res[k] for k in ("name", "result", "seconds")}), flush=True)
            with open(out_path, "w") as f:
                json.dump(sorted(results, key=lambda x: x["name"]), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
