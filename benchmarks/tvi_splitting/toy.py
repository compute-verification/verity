"""Toy training graphs, word level, and the split rewrites the covering experiments compare.

A graph is a list of `Node`s in topological order. A node is a registered source (`reg`, in theta, free to read), an
unregistered source (`src`: a token or a target, read at its width) or a computed word (`op`) with its predecessors, a
gate count and an evaluator. Words are integers mod 2**b (ring semantics, the attacker's most splittable case: the low
plane of a ring computation depends on low planes only).

`training_graph` builds T SGD steps of a one-layer model with an RMSNorm-like coupling:

    h_j = W_t x_j                    (m dot products of length n)
    s_j = sum_r h_jr^2               (the norm's inner product)
    a_jr = sigma(h_jr, s_j)          (a nonlinearity of the value and the norm)
    d_jr = (a_jr - y_jr) * tau(h_jr, s_j)
    G_rc = sum_j d_jr x_jc
    W_{t+1} = W_t - G

Variants, each computing the same outputs (`check_equivalent`):

- `mono`: every dot product and gradient entry is one node (a value is computed whole, products whole).
- `inner`: dot products and gradient sums are explicit product and adder chains, so partial sums can cross blocks.
- `shares`: data-parallel additive shares. The batch is split in two halves, each half's gradient updates its own share
  of the weights, W_t = W^A_t + W^B_t is never reconstructed until the final output, and each forward value is h^A + h^B,
  added only where the nonlinearity consumes it.
- `bits` (on top of `inner` or `shares`): every word is two planes (low h bits, high b-h bits); adds pass a carry bit,
  products pass a carry word and two cross terms, and nonlinear ops read both planes.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Node:
    name: str
    kind: str  # "reg", "src", "op"
    bits: int
    preds: list[int] = field(default_factory=list)
    gates: int = 0
    fn: Callable[[list[int]], int] | None = None
    tag: str = ""


class Graph:
    def __init__(self, b: int):
        self.b = b
        self.mask = (1 << b) - 1
        self.nodes: list[Node] = []
        self.index: dict[str, int] = {}
        self.outputs: list[str] = []

    def add(self, node: Node) -> int:
        assert node.name not in self.index, node.name
        for p in node.preds:
            assert p < len(self.nodes)
        self.index[node.name] = len(self.nodes)
        self.nodes.append(node)
        return self.index[node.name]

    def reg(self, name: str) -> int:
        return self.add(Node(name, "reg", self.b))

    def src(self, name: str, bits: int | None = None) -> int:
        return self.add(Node(name, "src", self.b if bits is None else bits))

    def op(self, name, preds, fn, gates, bits=None, tag="") -> int:
        return self.add(Node(name, "op", self.b if bits is None else bits, list(preds), gates, fn, tag))

    def evaluate(self, inputs: dict[str, int]) -> dict[str, int]:
        vals: list[int] = []
        for nd in self.nodes:
            if nd.kind == "op":
                vals.append(nd.fn([vals[p] for p in nd.preds]))
            else:
                vals.append(inputs[nd.name])
        return {nd.name: v for nd, v in zip(self.nodes, vals)}

    def stats(self) -> dict:
        ops = [nd for nd in self.nodes if nd.kind == "op"]
        return {
            "nodes": len(self.nodes),
            "ops": len(ops),
            "gates": sum(nd.gates for nd in ops),
            "src_bits": sum(nd.bits for nd in self.nodes if nd.kind == "src"),
        }


def _tables(b: int, seed: int = 7):
    """Two fixed nonlinear maps Z_{2^b} x Z_{2^b} -> Z_{2^b}, the toy's sigma and tau."""
    rng = random.Random(seed)
    size = 1 << b
    sig = [[rng.randrange(size) for _ in range(size)] for _ in range(size)]
    tau = [[rng.randrange(size) for _ in range(size)] for _ in range(size)]
    return sig, tau


class Words:
    """Word-level arithmetic on a graph: each value is one node id."""

    def __init__(self, g: Graph):
        self.g = g
        b, mk = g.b, g.mask
        self.cost = {"add": b, "mul": b * b, "nl": 4 * b * b}
        self._add = lambda v: (v[0] + v[1]) & mk
        self._sub = lambda v: (v[0] - v[1]) & mk
        self._mul = lambda v: (v[0] * v[1]) & mk

    def reg(self, name):
        return self.g.reg(name)

    def src(self, name):
        return self.g.src(name)

    def add(self, name, x, y, tag=""):
        return self.g.op(name, [x, y], self._add, self.cost["add"], tag=tag)

    def sub(self, name, x, y, tag=""):
        return self.g.op(name, [x, y], self._sub, self.cost["add"], tag=tag)

    def mul(self, name, x, y, tag=""):
        return self.g.op(name, [x, y], self._mul, self.cost["mul"], tag=tag)

    def nl(self, name, table, x, s, tag=""):
        return self.g.op(name, [x, s], lambda v: table[v[0]][v[1]], self.cost["nl"], tag=tag)

    def sum(self, name, terms, tag=""):
        acc = terms[0]
        for i, t in enumerate(terms[1:], 1):
            acc = self.add(f"{name}+{i}", acc, t, tag=tag if i == len(terms) - 1 else "")
        return acc

    def dot(self, name, xs, ys, tag=""):
        """A whole dot product as one node."""
        mk = self.g.mask
        n = len(xs)

        def fn(v):
            return sum(v[i] * v[n + i] for i in range(n)) & mk

        return self.g.op(name, list(xs) + list(ys), fn, n * (self.cost["mul"] + self.cost["add"]), tag=tag)

    def output(self, name, x):
        self.g.outputs.append(self.g.nodes[x].name)
        return x


class Planes:
    """Bit-sliced arithmetic: each value is a pair (lo, hi) of node ids, lo = v mod 2^h, hi = v >> h, b = 2h.

    Ring semantics mod 2^b: (x + y)_lo needs lo planes only, (x + y)_hi needs the hi planes and one carry bit;
    (x y)_lo needs lo planes, (x y)_hi needs a carry word floor(x_lo y_lo / 2^h) and the cross terms x_lo y_hi,
    x_hi y_lo (the hi*hi term vanishes mod 2^b). A nonlinear op reads both planes of its arguments.
    """

    def __init__(self, g: Graph):
        assert g.b % 2 == 0
        self.g = g
        self.h = g.b // 2
        h, hm = self.h, (1 << self.h) - 1
        self.hm = hm
        self.cost_add = h  # per plane
        self.cost_mul = h * h

    def lift(self, name, node_id):
        """Split a whole word into planes (free wires within the producing block; modelled as two cheap ops)."""
        h, hm = self.h, self.hm
        lo = self.g.op(f"{name}.lo", [node_id], lambda v: v[0] & hm, 0, bits=h)
        hi = self.g.op(f"{name}.hi", [node_id], lambda v: (v[0] >> h) & hm, 0, bits=h)
        return lo, hi

    def src(self, name):
        lo = self.g.src(f"{name}.lo", self.h)
        hi = self.g.src(f"{name}.hi", self.h)
        return lo, hi

    def reg(self, name):
        return self.g.reg(f"{name}.lo"), self.g.reg(f"{name}.hi")

    def add(self, name, x, y, sign=1, tag=""):
        h, hm = self.h, self.hm
        g = self.g
        if sign == 1:
            lo = g.op(f"{name}.lo", [x[0], y[0]], lambda v: (v[0] + v[1]) & hm, self.cost_add, bits=h, tag=tag)
            c = g.op(f"{name}.c", [x[0], y[0]], lambda v: (v[0] + v[1]) >> h, 1, bits=1)
            hi = g.op(f"{name}.hi", [x[1], y[1], c], lambda v: (v[0] + v[1] + v[2]) & hm, self.cost_add, bits=h, tag=tag)
        else:
            lo = g.op(f"{name}.lo", [x[0], y[0]], lambda v: (v[0] - v[1]) & hm, self.cost_add, bits=h, tag=tag)
            c = g.op(f"{name}.c", [x[0], y[0]], lambda v: 1 if v[0] < v[1] else 0, 1, bits=1)
            hi = g.op(f"{name}.hi", [x[1], y[1], c], lambda v: (v[0] - v[1] - v[2]) & hm, self.cost_add, bits=h, tag=tag)
        return lo, hi

    def sub(self, name, x, y, tag=""):
        return self.add(name, x, y, sign=-1, tag=tag)

    def mul(self, name, x, y, tag=""):
        h, hm = self.h, self.hm
        g = self.g
        lo = g.op(f"{name}.lo", [x[0], y[0]], lambda v: (v[0] * v[1]) & hm, self.cost_mul, bits=h, tag=tag)
        cw = g.op(f"{name}.cw", [x[0], y[0]], lambda v: ((v[0] * v[1]) >> h) & hm, self.cost_mul, bits=h)
        x1 = g.op(f"{name}.x1", [x[0], y[1]], lambda v: (v[0] * v[1]) & hm, self.cost_mul, bits=h)
        x2 = g.op(f"{name}.x2", [x[1], y[0]], lambda v: (v[0] * v[1]) & hm, self.cost_mul, bits=h)
        hi = g.op(f"{name}.hi", [cw, x1, x2], lambda v: (v[0] + v[1] + v[2]) & hm, 2 * self.cost_add, bits=h, tag=tag)
        return lo, hi

    def nl(self, name, table, x, s, tag=""):
        h, hm = self.h, self.hm
        g = self.g
        preds = [x[0], x[1], s[0], s[1]]

        def whole(v):
            return table[v[0] | (v[1] << h)][v[2] | (v[3] << h)]

        cost = 2 * g.b * g.b
        lo = g.op(f"{name}.lo", preds, lambda v: whole(v) & hm, cost, bits=h, tag=tag)
        hi = g.op(f"{name}.hi", preds, lambda v: (whole(v) >> h) & hm, cost, bits=h, tag=tag)
        return lo, hi

    def sum(self, name, terms, tag=""):
        acc = terms[0]
        for i, t in enumerate(terms[1:], 1):
            acc = self.add(f"{name}+{i}", acc, t, tag=tag if i == len(terms) - 1 else "")
        return acc

    def dot(self, name, xs, ys, tag=""):
        prods = [self.mul(f"{name}*{i}", x, y) for i, (x, y) in enumerate(zip(xs, ys))]
        return self.sum(name, prods, tag=tag)

    def output(self, name, x):
        self.g.outputs.extend([self.g.nodes[x[0]].name, self.g.nodes[x[1]].name])
        return x


def training_graph(n=2, m=2, B=2, T=2, b=4, variant="mono", model="norm") -> Graph:
    """T SGD steps of the toy model, as `variant` computes them (module docstring).

    `model="linear"` drops the norm and the nonlinearities (e = W x - y, G = sum_j e_j x_j^T): training is linear in
    W, so `shares` there are additive shares of W that are never reconstructed, each evolving on every token.
    """
    base, _, bits = variant.partition("+")
    assert base in ("mono", "inner", "shares", "shares_inner"), variant
    assert bits in ("", "bits"), variant
    assert model in ("norm", "linear"), model
    linear = model == "linear"
    g = Graph(b)
    sig, tau = _tables(b)
    A = Planes(g) if bits else Words(g)
    explicit = bits or base in ("inner", "shares_inner")
    shared = base.startswith("shares")

    def dot(name, xs, ys, tag=""):
        if explicit:
            if bits:
                return A.dot(name, xs, ys, tag=tag)
            return A.sum(name, [A.mul(f"{name}*{i}", x, y) for i, (x, y) in enumerate(zip(xs, ys))], tag=tag)
        return A.dot(name, xs, ys, tag=tag)

    theta = [[A.reg(f"theta[{r},{c}]") for c in range(n)] for r in range(m)]
    if shared and linear:
        halves = [list(range(B)), list(range(B))]
    elif shared:
        halves = [list(range(B // 2)), list(range(B // 2, B))]
    else:
        halves = [list(range(B))]
    # W[q] is weight share q; share 1 (only when shared) is zero before the first step, so its forward terms are absent.
    W = [theta] + ([None] if shared else [])
    for t in range(T):
        X = [[A.src(f"x{t}[{j},{c}]") for c in range(n)] for j in range(B)]
        Y = [[A.src(f"y{t}[{j},{r}]") for r in range(m)] for j in range(B)]
        D = {}
        for j in range(B):
            if linear:
                # e = W x - y is linear in W: share q's residual is W^q x, the last share's minus y, never added up.
                # Shared, share 0 is theta's part (it never reads a target) and share 1 the targets' part.
                for q in range(len(W)):
                    for r in range(m):
                        hq = None
                        if W[q] is not None:
                            hq = dot(f"h{t}[{j},{r}]/{q}", W[q][r], X[j], tag="value" if t > 0 else "")
                        if q == len(W) - 1:
                            base_ = hq if hq is not None else _zero(A, g, bits, f"z{t}[{j},{r}]")
                            D[j, r, q] = A.sub(f"e{t}[{j},{r}]/{q}", base_, Y[j][r])
                        else:
                            D[j, r, q] = hq
                continue
            H = []
            for r in range(m):
                parts = [dot(f"h{t}[{j},{r}]/{q}", W[q][r], X[j], tag="value" if t > 0 else "")
                         for q in range(len(W)) if W[q] is not None]
                H.append(parts[0] if len(parts) == 1 else A.add(f"h{t}[{j},{r}]", parts[0], parts[1]))
            sq = [A.mul(f"sq{t}[{j},{r}]", H[r], H[r]) for r in range(m)]
            S = A.sum(f"s{t}[{j}]", sq)
            for r in range(m):
                a = A.nl(f"a{t}[{j},{r}]", sig, H[r], S)
                e = A.sub(f"e{t}[{j},{r}]", a, Y[j][r])
                tt = A.nl(f"tau{t}[{j},{r}]", tau, H[r], S)
                D[j, r] = A.mul(f"d{t}[{j},{r}]", e, tt)
        newW = []
        for q, js in enumerate(halves):
            Wq = []
            for r in range(m):
                row = []
                for c in range(n):
                    if linear:
                        G = dot(f"G{t}/{q}[{r},{c}]", [D[j, r, q] for j in js], [X[j][c] for j in js])
                    else:
                        G = dot(f"G{t}/{q}[{r},{c}]", [D[j, r] for j in js], [X[j][c] for j in js])
                    prev = W[q][r][c] if W[q] is not None else _zero(A, g, bits, f"zero{t}/{q}[{r},{c}]")
                    row.append(A.sub(f"W{t + 1}/{q}[{r},{c}]", prev, G))
                Wq.append(row)
            newW.append(Wq)
        W = newW
    for r in range(m):
        for c in range(n):
            if shared:
                A.output(f"W{T}[{r},{c}]", A.add(f"W{T}[{r},{c}]", W[0][r][c], W[1][r][c]))
            else:
                A.output(f"W{T}[{r},{c}]", W[0][r][c])
    return g


def _zero(A, g, bits, name):
    if bits:
        return (g.op(f"{name}.lo", [], lambda v: 0, 0, bits=A.h), g.op(f"{name}.hi", [], lambda v: 0, 0, bits=A.h))
    return g.op(name, [], lambda v: 0, 0)


def word_inputs(n, m, B, T, b, rng) -> dict[str, int]:
    vals = {}
    for r, c in itertools.product(range(m), range(n)):
        vals[f"theta[{r},{c}]"] = rng.randrange(1 << b)
    for t in range(T):
        for j in range(B):
            for c in range(n):
                vals[f"x{t}[{j},{c}]"] = rng.randrange(1 << b)
            for r in range(m):
                vals[f"y{t}[{j},{r}]"] = rng.randrange(1 << b)
    return vals


def plane_inputs(words: dict[str, int], b: int) -> dict[str, int]:
    h = b // 2
    out = {}
    for k, v in words.items():
        out[f"{k}.lo"] = v & ((1 << h) - 1)
        out[f"{k}.hi"] = v >> h
    return out


def final_weights(g: Graph, vals: dict[str, int]) -> list[int]:
    """The output words, reassembled from planes where the graph is bit-sliced."""
    names = g.outputs
    if names and names[0].endswith(".lo"):
        h = g.b // 2
        return [vals[names[i]] | (vals[names[i + 1]] << h) for i in range(0, len(names), 2)]
    return [vals[x] for x in names]


def check_equivalent(n=2, m=2, B=2, T=2, b=4, variants=None, trials=200, seed=1, model="norm") -> dict[str, bool]:
    """Every variant computes `mono`'s final weights on `trials` random inputs."""
    variants = variants or VARIANTS
    rng = random.Random(seed)
    graphs = {v: training_graph(n, m, B, T, b, v, model) for v in variants}
    ok = {v: True for v in variants}
    for _ in range(trials):
        w = word_inputs(n, m, B, T, b, rng)
        p = plane_inputs(w, b)
        ref = final_weights(graphs["mono"], graphs["mono"].evaluate(w))
        for v, g in graphs.items():
            got = final_weights(g, g.evaluate(p if "bits" in v else w))
            ok[v] &= got == ref
    return ok


VARIANTS = ["mono", "inner", "inner+bits", "shares", "shares_inner", "shares_inner+bits"]
