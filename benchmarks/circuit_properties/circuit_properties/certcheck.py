"""The upstream certificate (`Nci.Capability.Consumers.Upstream`) on a training step, `TrainStep_v1`: its consumers, their
regions and required gates, and what that gives the dependency cap h_w·(k + c/g).

**Uses** (`Consumers.ofUses`). A counted product is one lane of a tensor-core dot (`HopperBF16WgmmaDot16_v2`) in a
forward or input-gradient GEMM: x is the GEMM row's activation, w its weight. Use (kind, layer, token) is one token's row
of a GEMM's x, and its finishing sets are the lanes of that row's dots. `into F` of a lane is the 16 bits of x and the 16
of w it reads (each lane reads exactly those: `tests/test_certcheck.py`), and a row's coordinates share x (the row GEMM's
batch axis None), so a use's distinct `into F ∩ A` are its row's K activation values, each read by N coordinates.
`KINDS` lists the kinds, each with its region: the token's own computation since the previous use's products, as whole
activations of the step's root (a GEMM row, a norm's token, a token's words of an add or a multiply, its SiLU calls), so
that regions of different uses are disjoint. Attention mixes tokens: a use read from attention (o's input, q's, k's and v's
input gradients) takes the whole attention call as its region, the largest one inside attention (a larger region can only
require more gates), shared by the sequence's tokens.

**Outputs** (`Consumers.ofOutputs`): an output bit of a weight-gradient GEMM coordinate, its region the coordinate.

For each consumer the measurement reports |A|, |G| for the largest certified G (`regions.required_size`), and, when G is
empty, why: a class with no gate in A (F reads nothing of the region), or no gate reaching every class.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from verity.ml.circuits.layout import Circuit, Scope

from . import regions as RG
from .ops import Site, flatten, op_shape, param_bases

#: the counted operation: one lane of a tensor-core dot, x its activation a and w its weight b
DOT = re.compile(r"HopperBF16WgmmaDot16_v\d+")
SITE = Site(op=DOT.pattern, x="a", w="b")


@dataclass(frozen=True)
class Kind:
    """A kind of use: its region's activations and its consuming GEMMs, for layer i and token t."""

    name: str
    what: str
    region: Callable[["Step", int, int], list[Scope]]
    consumers: Callable[["Step", int], list[str]]
    per_layer: bool = True
    attention: bool = False


@dataclass
class Verdict:
    kind: str
    layer: int | None
    token: int
    region_gates: int
    required: int
    classes: int
    finishing_sets: int
    absent: int
    why: str


class Step:
    """A built `TrainStep_v1` Program: its root activations by node name, and the consumers' regions and classes."""

    def __init__(self, program, L: int, M: int) -> None:
        self.p, self.L, self.M = program, L, M
        self.C = Circuit(program.root, explicit_limit=10 ** 7)
        self.root = self.C.root_scope
        self.index = {node.name: k for k, node in enumerate(program.root.body.nodes) if node.name}
        self.pull, self.res = RG.Pull(), RG.Resolver()

    def node(self, name: str):
        k = self.index[name]
        return k, self.p.root.body.nodes[k]

    def tok(self, name: str, t: int) -> list[Scope]:
        """Token t's members of the root batch ``name`` (a norm's one, an add's H words, SiLU's I / ACT_N calls)."""
        k, node = self.node(name)
        if node.form != "batch" or node.n % self.M:
            raise ValueError(f"{name} is not a batch over the step's {self.M} tokens")
        per = node.n // self.M
        return [self.root.child(k, m) for m in range(t * per, (t + 1) * per)]

    def row(self, name: str, t: int) -> Scope:
        """Row t of the GEMM ``name`` (`LinearTorch`, `LinearTorchDx`): a row GEMM, batched over the call's rows."""
        k, node = self.node(name)
        call = self.root.child(k, 0)
        inner = call.spec.body.nodes[0]
        if node.form != "call" or inner.form != "batch" or inner.n != self.M:
            raise ValueError(f"{name} is not a call batched over the step's {self.M} rows")
        return call.child(0, t)

    def whole(self, name: str) -> Scope:
        k, node = self.node(name)
        if node.form == "batch" and node.n != 1:
            raise ValueError(f"{name} is a batch of {node.n}: one sequence is one activation")
        return self.root.child(k, 0)

    def embed(self, t: int) -> Scope:
        return self.root.child(self.index["embed"], 0).child(0, t)

    def classes(self, row: Scope) -> tuple[list[frozenset[int]], list[frozenset[int]], int]:
        """`into F` of each lane of a GEMM row's first and last coordinates, and its number of coordinates N."""
        gemm = row.spec.body.nodes[0]
        if gemm.form != "batch" or gemm.axes[0] is not None:
            raise ValueError(f"{row.spec.id}: not a batch of coordinates sharing x")
        ends = []
        for n in (0, gemm.n - 1):
            coord = row.child(0, n)
            out: list[frozenset[int]] = []
            for k, cn in enumerate(coord.spec.body.nodes):
                if cn.form == "call" and DOT.fullmatch(cn.fn.id):
                    out += self.lane_into(coord.child(k, 0))
            ends.append(out)
        return ends[0], ends[1], gemm.n

    def lane_into(self, dot: Scope) -> list[frozenset[int]]:
        """`into m(p)` for each lane p of the dot activation ``dot``: the root gates its lane gates read outside it."""
        fn = dot.spec
        shape, flat, base = op_shape(fn, SITE), flatten(fn), param_bases(fn)
        out = []
        for lane in range(shape.lanes):
            mine = set(shape.members[lane])
            local = {o for g in mine for o in flat.operands[g] if o >= 0 and o not in mine}
            if local:
                raise ValueError(f"{fn.id}: lane {lane} reads gates of the dot outside its own")
            gates = set()
            for g in mine:
                for o in flat.operands[g]:
                    if o < 0:
                        j = -1 - o
                        pi = next(i for i in range(len(base) - 1) if base[i] <= j < base[i + 1])
                        gates.add(self.res.param(dot, pi, j - base[pi]))
            out.append(frozenset(gates))
        return out

    def use(self, kind: Kind, i: int | None, t: int) -> Verdict:
        acts = kind.region(self, i, t)
        inside = _inside(acts)
        classes: Counter = Counter()
        for name in kind.consumers(self, i):
            row = self.row(name, t)
            first, last, N = self.classes(row)
            a = [frozenset(g for g in c if inside(g)) for c in first]
            b = [frozenset(g for g in c if inside(g)) for c in last]
            if a != b:
                raise ValueError(f"{kind.name}: {name}'s first and last coordinates read different gates of the region")
            for c in a:
                classes[c] += N
        distinct = list(classes)
        r = RG.required_size(acts, distinct, self.pull, self.res)
        why = ""
        if r.absent:
            why = f"{len(r.absent)} of {len(distinct)} classes read no gate of the region"
        elif not r.required:
            why = "no gate of the region reaches every class"
        return Verdict(kind.name, i, t, r.gates, r.required, len(distinct), sum(classes.values()), len(r.absent), why)


def _inside(acts: Iterable[Scope]):
    spans = sorted((s.offset, s.offset + s.spec.gates) for s in acts)

    def inside(g: int) -> bool:
        return any(lo <= g < hi for lo, hi in spans)
    return inside


def _lin(*names: str):
    return lambda S, i: [f"l{i}.{n}" for n in names]


def _y1(S: Step, i: int, t: int) -> list[Scope]:
    up = [S.embed(t)] if i == 0 else [S.row(f"l{i - 1}.down", t), *S.tok(f"l{i - 1}.h3", t)]
    return up + S.tok(f"l{i}.ln1", t)


def _dh(S: Step, i: int, t: int) -> list[Scope]:
    if i == S.L - 1:
        return [S.row("lm_head.dx", t), *S.tok("norm.dx", t)]
    j = f"l{i + 1}."
    return [*(S.row(j + f"{w}.dx", t) for w in ("q", "k", "v")),
            *(a for n in ("kv.dx", "qkv.dx", "ln1.dx", "x.grad") for a in S.tok(j + n, t))]


KINDS = (
    Kind("y1", "ln1's output: the layer's input (the embedding, or the previous layer's down row and residual add) and ln1",
         _y1, _lin("q", "k", "v")),
    Kind("o_in", "attention's output row (o's input): the whole attention call", lambda S, i, t: [S.whole(f"l{i}.attn")],
         _lin("o"), attention=True),
    Kind("y2", "ln2's output: o's row, the residual add h2 and ln2",
         lambda S, i, t: [S.row(f"l{i}.o", t), *S.tok(f"l{i}.h2", t), *S.tok(f"l{i}.ln2", t)], _lin("gate", "up")),
    Kind("mul", "down's input: gate's and up's rows, SiLU and the product",
         lambda S, i, t: [S.row(f"l{i}.gate", t), S.row(f"l{i}.up", t), *S.tok(f"l{i}.act", t), *S.tok(f"l{i}.mul", t)],
         _lin("down")),
    Kind("yn", "the final norm's output: the last layer's down row and residual add, and the norm",
         lambda S, i, t: [S.row(f"l{S.L - 1}.down", t), *S.tok(f"l{S.L - 1}.h3", t), *S.tok("norm", t)],
         lambda S, i: ["logits"], per_layer=False),
    Kind("dlogits", "the logits' gradient: the logits row, the log-softmax row and its backward",
         lambda S, i, t: [S.row("logits", t), *S.tok("loss.rows", t), *S.tok("logits.dx", t)],
         lambda S, i: ["lm_head.dx"], per_layer=False),
    Kind("dh", "the layer output's gradient (down's dX input): the next layer's q, k, v dX rows, their sums, ln1's backward "
         "and the residual sum, or for the last layer the LM head's dX row and the final norm's backward", _dh,
         _lin("down.dx")),
    Kind("dup", "up's dX input: down's dX row and the product's gradient into up",
         lambda S, i, t: [S.row(f"l{i}.down.dx", t), *S.tok(f"l{i}.mul.db", t)], _lin("up.dx")),
    Kind("dgate", "gate's dX input: the product's gradient into SiLU and SiLU's backward",
         lambda S, i, t: [*S.tok(f"l{i}.mul.da", t), *S.tok(f"l{i}.act.dx", t)], _lin("gate.dx")),
    Kind("dh2", "o's dX input: up's and gate's dX rows, their sum, ln2's backward and the residual sum",
         lambda S, i, t: [S.row(f"l{i}.up.dx", t), S.row(f"l{i}.gate.dx", t),
                          *(a for n in ("mlp.dx", "ln2.dx", "h2.grad") for a in S.tok(f"l{i}.{n}", t))],
         _lin("o.dx")),
    *(Kind(f"d{w}", f"{w}'s dX input: the whole attention backward call", lambda S, i, t: [S.whole(f"l{i}.attn.dx")],
           _lin(f"{w}.dx"), attention=True) for w in ("v", "k", "q")),
)


def uses(S: Step, tokens: Iterable[int], layers: Iterable[int] | None = None, kinds=KINDS) -> list[Verdict]:
    layers = list(range(S.L) if layers is None else layers)
    out = []
    for kind in kinds:
        for i in (layers if kind.per_layer else [None]):
            for t in tokens:
                out.append(S.use(kind, i, t))
    return out


def disjoint(S: Step, tokens: Iterable[int], kinds=KINDS) -> tuple[bool, int]:
    """Whether the regions of the non-attention uses are pairwise disjoint, and how many activations they hold."""
    spans = []
    for kind in kinds:
        if kind.attention:
            continue
        for i in (range(S.L) if kind.per_layer else [None]):
            for t in tokens:
                spans += [(s.offset, s.offset + s.spec.gates) for s in kind.region(S, i, t)]
    spans.sort()
    return all(spans[a][1] <= spans[a + 1][0] for a in range(len(spans) - 1)), len(spans)


def as_dict(v: Verdict) -> dict:
    return asdict(v)
