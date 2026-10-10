"""Sizing a switching-network universal unit against UniversalUnit_v1's multiplexer trees.

The construction (`Unit`): poles 0..k-1 in gate order (N_IN inputs, G gate poles, N_OUT output poles). Each gate pole is a
universal gate out = a·(ea ⊕ t·b) ⊕ b·eb ⊕ n (3 ANDs, 4 bits: XOR, AND, XNOR, NAND, OR, copy/NOT of either input, constants,
each unary one reading a single input). Its input a comes from edge-universal graph A, b from B, each a Γ1 EUG (in, out <= 1
per pole) built by pair recursion: poles paired (2p, 2p+1), each pair's two outputs swapped into the two half-size sub-EUGs'
pole p (out-X), the sub-EUGs' pole-p inputs swapped back to the pair (in-X), and the odd pole's input a Y between that and the
even pole's output (the intra-pair edge). A switch is 1 AND. An output pole reads Y(A, B).

Every switch setting gives an SSA netlist (each pole input is an earlier pole's output or 0): the network is a DAG whose poles
are in order, which `evaluate`'s topological sort checks, and `decode` reads the netlist off any setting.

    python switching_sizing.py F32MulFtz_v3.gates.json   # from art:a34dcfde092edf623b23d78c103853547c5587fff1b413e7e44d1b929483e84b
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict

ZERO = 0


class Net:
    """Nodes: ('in', i) | ('X0'|'X1', bit, a, b) | ('Y', bit, a, b) | ('UG', gate, a, b). Ref 0 is the constant 0."""

    def __init__(self):
        self.nodes = [("zero",)]
        self.nbits = 0

    def add(self, *n):
        self.nodes.append(n)
        return len(self.nodes) - 1

    def bit(self):
        self.nbits += 1
        return self.nbits - 1

    def X(self, a, b):
        if a == ZERO and b == ZERO:
            return ZERO, ZERO, None
        s = self.bit()
        return self.add("X0", s, a, b), self.add("X1", s, a, b), s

    def Y(self, a, b):
        if a == b:
            return a, None
        s = self.bit()
        return self.add("Y", s, a, b), s


def eug(net: Net, outs: list[int]):
    """A Γ1 EUG over len(outs) poles whose outputs are `outs`: (each pole's input ref, the routing tree)."""
    k = len(outs)
    if k == 1:
        return [ZERO], ("leaf",)
    if k == 2:
        return [ZERO, outs[0]], ("leaf",)
    m = (k + 1) // 2
    ao, bo, oX = [], [], []
    for p in range(m):
        if 2 * p + 1 < k:
            x0, x1, s = net.X(outs[2 * p], outs[2 * p + 1])
            ao.append(x0), bo.append(x1), oX.append(s)
        else:
            ao.append(outs[2 * p]), bo.append(outs[2 * p]), oX.append(None)
    ai, at = eug(net, ao)
    bi, bt = eug(net, bo)
    ins, iX, Ys = [], [], []
    for p in range(m):
        if 2 * p + 1 < k:
            x0, x1, s = net.X(ai[p], bi[p])
            y, sy = net.Y(x1, outs[2 * p])
            ins += [x0, y]
            iX.append(s), Ys.append(sy)
        else:
            y, sy = net.Y(ai[p], bi[p])
            ins.append(y)
            iX.append(None), Ys.append(sy)
    return ins, ("node", k, oX, iX, Ys, at, bt)


def two_colour(edges, deg=2):
    """Edge 2-colouring of a bipartite multigraph of max degree 2: edges (left, right) -> colour 0/1."""
    L, R = defaultdict(list), defaultdict(list)
    for i, (l, r) in enumerate(edges):
        L[l].append(i), R[r].append(i)
    assert all(len(v) <= deg for v in L.values()) and all(len(v) <= deg for v in R.values())
    col = [None] * len(edges)

    def other(adj, key, e):
        return next((f for f in adj[key] if f != e), None)

    for e0 in range(len(edges)):
        if col[e0] is not None:
            continue
        col[e0] = 0
        for side0 in (0, 1):  # walk both directions from e0
            e, side = e0, side0
            while True:
                f = other(R, edges[e][1], e) if side == 0 else other(L, edges[e][0], e)
                if f is None or col[f] is not None:
                    if f is not None:
                        assert col[f] != col[e], "odd cycle"
                    break
                col[f] = 1 - col[e]
                e, side = f, 1 - side
    return col


def route(tree, edges: dict[int, int], bits: dict[int, int]):
    """Set switch bits so pole edges[v] -> v (dst -> src, src < v) are carried: each pole in <= 1, out <= 1."""
    if tree[0] == "leaf":
        assert all(v == 1 and u == 0 for v, u in edges.items()), edges
        return
    _, k, oX, iX, Ys, at, bt = tree

    def setb(s, v):
        if s is None:
            return
        assert bits.setdefault(s, v) == v, "switch conflict"

    cross = []
    for v, u in edges.items():
        assert u < v
        if u // 2 == v // 2:
            setb(Ys[v // 2], 1)
        else:
            cross.append((u, v))
    col = two_colour([(u // 2, v // 2) for u, v in cross])
    sub = ({}, {})
    for (u, v), c in zip(cross, col):
        pu, pv = u // 2, v // 2
        if oX[pu] is not None:
            setb(oX[pu], int(c != u % 2))
        if iX[pv] is not None:
            setb(iX[pv], int(c != v % 2))
            if v % 2:
                setb(Ys[pv], 0)
        else:
            setb(Ys[pv], c)
        assert pv not in sub[c]
        sub[c][pv] = pu
    route(at, sub[0], bits)
    route(bt, sub[1], bits)


class Unit:
    """The switching unit for (G gate poles, N_IN, N_OUT)."""

    def __init__(self, G, N_IN, N_OUT):
        self.G, self.N_IN, self.N_OUT = G, N_IN, N_OUT
        net = self.net = Net()
        xs = [net.add("in", i) for i in range(N_IN)]
        ug = [net.add("UG", g, None, None) for g in range(G)]  # args filled below
        self.ug0 = ug[0]
        outs = xs + ug + [ZERO] * N_OUT
        ia, self.ta = eug(net, outs)
        ib, self.tb = eug(net, outs[:N_IN + G] + [ZERO] * N_OUT)
        for g, r in enumerate(ug):
            net.nodes[r] = ("UG", g, ia[N_IN + g], ib[N_IN + g])
        self.ybits, self.outs = [], []
        for i in range(N_OUT):
            y, s = net.Y(ia[N_IN + G + i], ib[N_IN + G + i])
            self.outs.append(y), self.ybits.append(s)
        self.switch_bits = net.nbits

    def program_bits(self):
        return self.switch_bits + 4 * self.G

    def live(self):
        seen, st = set(), list(self.outs)
        while st:
            r = st.pop()
            if r in seen or r == ZERO:
                continue
            seen.add(r)
            n = self.net.nodes[r]
            if n[0] in ("X0", "X1", "Y", "UG"):
                st += [n[2], n[3]]
        return seen

    def and_count(self):
        seen = self.live()
        sw, ugs = set(), 0
        for r in seen:
            n = self.net.nodes[r]
            if n[0] in ("X0", "X1", "Y"):
                sw.add(n[1])
            elif n[0] == "UG":
                a, b = n[2] != ZERO, n[3] != ZERO
                ugs += 3 if a and b else (1 if a or b else 0)
        return len(sw), ugs, len(sw) + ugs

    def evaluate(self, prog, xbits, lanes):
        """prog: (switch bits dict, gate coeffs [(ea, eb, t, n)], ybits), xbits: per-input lane masks. Topological."""
        bits, coeff = prog
        nodes, M = self.net.nodes, (1 << lanes) - 1
        indeg, users = [0] * len(nodes), defaultdict(list)
        for r, n in enumerate(nodes):
            if n[0] in ("X0", "X1", "Y", "UG"):
                for a in set((n[2], n[3])):
                    indeg[r] += 1
                    users[a].append(r)
        val = [0] * len(nodes)
        order = [r for r in range(len(nodes)) if indeg[r] == 0]
        done = 0
        while order:
            r = order.pop()
            done += 1
            n = nodes[r]
            if n[0] == "in":
                val[r] = xbits[n[1]]
            elif n[0] in ("X0", "X1", "Y"):
                s = bits.get(n[1], 0)
                pick = (n[2], n[3]) if (s == 0) == (n[0] != "X1") else (n[3], n[2])
                val[r] = val[pick[0]]
            elif n[0] == "UG":
                ea, eb, t, ng = coeff[n[1]]
                a, b = val[n[2]], val[n[3]]
                val[r] = ((a & ((M if ea else 0) ^ (b if t else 0))) ^ (b if eb else 0) ^ (M if ng else 0)) & M
            for u in users[r]:
                indeg[u] -= 1
                if indeg[u] == 0:
                    order.append(u)
        assert done == len(nodes), "cycle"
        return [val[o] for o in self.outs]


# ---- netlists (UniversalUnit_v1's format) -------------------------------------------------------------------------------

def interpret(n_in, gates, outputs, x):
    w = list(x)
    for op, a, b in gates:
        va = w[a] if a < len(w) else 0
        vb = w[b] if b < len(w) else 0
        w.append(((va & vb) if op & 1 else (va ^ vb)) ^ (-(op >> 1)))
    return [w[o] if o < len(w) else 0 for o in outputs]


def to_reads(n_in, gates):
    """Each gate as (coefficient kind, reads): unary forms (a == b) read once."""
    out = []
    for op, a, b in gates:
        if a == b:
            out.append({0: ("c0", ()), 2: ("c1", ()), 1: ("copy", (a,)), 3: ("not", (a,))}[op])
        else:
            out.append(({0: "xor", 1: "and", 2: "xnor", 3: "nand"}[op], (a, b)))
    return out


def copies_needed(n_in, gates, outputs):
    fo = Counter()
    for _, rd in to_reads(n_in, gates):
        fo.update(rd)
    fo.update(outputs)
    return sum(max(0, f - 2) for f in fo.values()), fo


def expand(n_in, gates, outputs):
    """Copy-insert to fan-out <= 2: (gate list of (kind, reads) in pole numbering, outputs)."""
    rd = to_reads(n_in, gates)
    readers = defaultdict(list)  # wire -> list of (reader id) in order; reader = ('g', g, slot) | ('o', i)
    for g, (_, r) in enumerate(rd):
        for j, w in enumerate(r):
            readers[w].append(("g", g, j))
    for i, w in enumerate(outputs):
        readers[w].append(("o", i))
    # new order: input copies first, then each gate followed by its copies
    new, newpos = [], {}
    src_of = {}  # reader -> new source pole

    def place_copies(w, wpole):
        rs = readers[w]
        c = max(0, len(rs) - 2)
        cur = wpole
        for i in range(c):
            new.append(("copy", (cur,)))
            nxt = n_in + len(new) - 1
            src_of[rs[i]] = cur
            cur = nxt
        for r in rs[c:]:
            src_of[r] = cur

    for i in range(n_in):
        place_copies(i, i)
    for g, (kind, r) in enumerate(rd):
        new.append((kind, None))
        p = n_in + len(new) - 1
        newpos[g] = len(new) - 1
        place_copies(n_in + g, p)
    for g, (kind, r) in enumerate(rd):
        new[newpos[g]] = (kind, tuple(src_of[("g", g, j)] for j in range(len(r))))
    outs = [src_of[("o", i)] for i in range(len(outputs))]
    return new, outs


COEFF = {"xor": (1, 1, 0, 0), "and": (0, 0, 1, 0), "xnor": (1, 1, 0, 1), "nand": (0, 0, 1, 1), "c0": (0, 0, 0, 0),
         "c1": (0, 0, 0, 1)}


def program(unit: Unit, n_in, gates, outputs):
    """Encode a netlist at the unit's class: switch bits and gate coefficients. Pads with constant-0 gate poles."""
    eg, eo = expand(n_in, gates, outputs)
    G, N_IN = unit.G, unit.N_IN
    assert len(eg) <= G and n_in <= N_IN and len(eo) == unit.N_OUT, (len(eg), G)
    sh = N_IN - n_in

    def at(p):
        return p if p < n_in else p + sh

    eg = eg + [("c0", ())] * (G - len(eg))
    edges, owner = [], []
    for g, (kind, r) in enumerate(eg):
        for j, w in enumerate(r):
            edges.append((at(w), N_IN + g)), owner.append(("g", g))
    for i, w in enumerate(eo):
        edges.append((at(w), N_IN + G + i)), owner.append(("o", i))
    col = two_colour(edges)
    ea_, eb_ = {0: {}, 1: {}}, {}
    side = {0: {}, 1: {}}
    for (u, v), c, ow in zip(edges, col, owner):
        assert v not in side[c]
        side[c][v] = u
    bits, coeff = {}, []
    route(unit.ta, side[0], bits)
    route(unit.tb, side[1], bits)
    for g, (kind, r) in enumerate(eg):
        p = N_IN + g
        if kind in ("copy", "not"):
            onA = p in side[0]
            coeff.append((int(onA), int(not onA), 0, int(kind == "not")))
        else:
            coeff.append(COEFF[kind])
    for i, s in enumerate(unit.ybits):
        v = N_IN + G + i
        if s is not None:
            bits[s] = 0 if v in side[0] else 1
    return (bits, coeff), len(eg) - (G - len(eg) if False else 0)


# ---- today's unit and the comparison constructions ------------------------------------------------------------------------

def _tree(m, k):
    n = 0
    for _ in range(k):
        m = (m + 1) // 2
        n += m
    return n


def today_and_count(G, N_IN, N_OUT):
    k = max(1, (N_IN + G - 1).bit_length())
    return sum(2 * _tree(N_IN + g, k) + 2 for g in range(G)) + N_OUT * _tree(N_IN + G, k)


def today_program_bits(G, N_IN, N_OUT):
    k = max(1, (N_IN + G - 1).bit_length())
    return 2 * G + 2 * G * k + N_OUT * k


def waksman(n):
    """Switches of an n-input Waksman network: sum_{i<=n} ceil(log2 i) (Beauquier-Darrot)."""
    return sum((i - 1).bit_length() for i in range(1, n + 1))


def halves_waksman(k):
    """Left/right recursion, a Waksman network for the cross edges and a Y per right pole, per Γ1."""
    if k <= 1:
        return 0
    a, b = (k + 1) // 2, k // 2
    return halves_waksman(a) + halves_waksman(b) + waksman(max(a, b)) + b


def main():
    sys.setrecursionlimit(10000)
    rng = random.Random(1)
    # 1. correctness on random netlists with heavy fan-out and on F32MulFtz_v3
    for trial in range(30):
        n_in, g0, n_out = rng.randrange(1, 12), rng.randrange(1, 60), rng.randrange(1, 8)
        gates = []
        for g in range(g0):
            hot = rng.random() < 0.5
            a = rng.randrange(min(3, n_in + g)) if hot else rng.randrange(n_in + g)
            b = a if rng.random() < 0.1 else rng.randrange(n_in + g)
            gates.append((rng.randrange(4), a, b))
        outs = [rng.randrange(n_in + g0) for _ in range(n_out)]
        eg, _ = expand(n_in, gates, outs)
        u = Unit(len(eg) + rng.randrange(3), n_in + rng.randrange(3), n_out)
        prog, _ = program(u, n_in, gates, outs)
        L = 64
        xs = [[rng.getrandbits(1) for _ in range(n_in)] for _ in range(L)]
        lanes = [sum(xs[l][i] << l for l in range(L)) for i in range(n_in)] + [0] * (u.N_IN - n_in)
        got = u.evaluate(prog, lanes, L)
        for l in range(L):
            want = interpret(n_in, gates, outs, xs[l])
            assert [(o >> l) & 1 for o in got] == [w & 1 for w in want], trial
    print("random netlists: 30 x 64 inputs agree")

    d = json.load(open(sys.argv[1]))
    n_in, gates, outs = d["n_in"], [tuple(g) for g in d["gates"]], d["outputs"]
    c, fo = copies_needed(n_in, gates, outs)
    eg, _ = expand(n_in, gates, outs)
    print(f"F32MulFtz_v3: G={len(gates)} copies={c} G'={len(eg)} max fan-out={max(fo.values())}")
    res = {}
    for G in (len(eg), 8090 + c):
        u = Unit(G, 64, 32)
        if G == len(eg):
            prog, _ = program(u, n_in, gates, outs)
            L = 64
            xs = [[rng.getrandbits(1) for _ in range(n_in)] for _ in range(L)]
            lanes = [sum(xs[l][i] << l for l in range(L)) for i in range(n_in)]
            got = u.evaluate(prog, lanes, L)
            for l in range(L):
                assert [(o >> l) & 1 for o in got] == [w & 1 for w in interpret(n_in, gates, outs, xs[l])]
            print("  F32MulFtz_v3 through the switching unit: 64 random inputs agree with interpret")
        res[G] = u.and_count(), u.program_bits()
        print(f"  G'={G}: (switch ANDs, gate ANDs, total)={res[G][0]}, program bits={res[G][1]}")


def decode(unit: Unit, bits):
    """The netlist any switch setting describes: for each gate pole and output, the pole (or 0) each input reads."""
    nodes = unit.net.nodes
    pole = {}
    for r, n in enumerate(nodes):
        if n[0] == "in":
            pole[r] = n[1]
        elif n[0] == "UG":
            pole[r] = unit.N_IN + n[1]

    def src(r):
        while True:
            n = nodes[r]
            if n[0] == "zero":
                return None
            if r in pole:
                return pole[r]
            s = bits.get(n[1], 0)
            r = (n[2], n[3])[0 if (s == 0) == (n[0] != "X1") else 1]

    gates = []
    for r, n in enumerate(nodes):
        if n[0] == "UG":
            a, b = src(n[2]), src(n[3])
            p = unit.N_IN + n[1]
            assert (a is None or a < p) and (b is None or b < p), "forward read"
            gates.append((n[1], a, b))
    gates.sort()
    return [(a, b) for _, a, b in gates], [src(o) for o in unit.outs]


def ug_interpret(n_in, srcs, coeff, outs, x):
    w = list(x)
    for (a, b), (ea, eb, t, ng) in zip(srcs, coeff):
        va = 0 if a is None else w[a]
        vb = 0 if b is None else w[b]
        w.append((va & (ea ^ (t & vb))) ^ (vb & eb) ^ ng)
    return [0 if o is None else w[o] for o in outs]


def decision3(rng):
    for trial in range(40):
        u = Unit(rng.randrange(1, 70), rng.randrange(1, 10), rng.randrange(1, 6))
        bits = {s: rng.getrandbits(1) for s in range(u.net.nbits)}
        coeff = [tuple(rng.getrandbits(1) for _ in range(4)) for _ in range(u.G)]
        srcs, outs = decode(u, bits)
        L = 32
        xs = [[rng.getrandbits(1) for _ in range(u.N_IN)] for _ in range(L)]
        got = u.evaluate((bits, coeff), [sum(xs[l][i] << l for l in range(L)) for i in range(u.N_IN)], L)
        for l in range(L):
            assert [(o >> l) & 1 for o in got] == ug_interpret(u.N_IN, srcs, coeff, outs, xs[l])
    print("decision 3: 40 random switch settings each decode to an in-order netlist the unit agrees with")


def table():
    N_IN, N_OUT = 64, 32
    rows = []
    for G in (1024, 4096, 8090, 32768):
        k_of = lambda g: N_IN + g + N_OUT
        t = today_and_count(G, N_IN, N_OUT)
        u = Unit(G, N_IN, N_OUT)
        sw, ug, tot = u.and_count()
        Gw = 3 * G + N_OUT - 2
        uw = Unit(Gw, N_IN, N_OUT)
        _, _, totw = uw.and_count()
        k, kw = k_of(G), k_of(Gw)
        hw = 2 * halves_waksman(k) + 3 * G + N_OUT
        rows.append(dict(G=G, today=t, today_bits=today_program_bits(G, N_IN, N_OUT), sw=sw, ug=ug, total=tot,
                         bits=u.program_bits(), per_pole=round(tot / G, 1), ratio=round(t / tot, 1), Gw=Gw, worst=totw,
                         worst_ratio=round(t / totw, 1), lit45=round(4.5 * k * math.log2(k)),
                         lit45w=round(4.5 * kw * math.log2(kw)), halves=hw,
                         c_klogk=round(sw / (k * math.log2(k)), 3)))
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
    decision3(random.Random(2))
    table()
