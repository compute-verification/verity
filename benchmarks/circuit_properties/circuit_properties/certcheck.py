"""The upstream certificate (`Nci.Capability.Consumers.Upstream`) on a training step, `TrainStep_v1`: its consumers, their
regions and required gates, and what that gives the dependency cap h_w·(k + c/g).

**Counted products.** A product is one lane of a tensor-core dot in a linear layer's forward or input-gradient GEMM: a
whole dot's (`HopperBF16WgmmaDot16_v2`, the unsplit routes) or a dot step's (`HopperBf16Dot16Step_v1`, `StepFirst_v1`,
the split-K routes: the measured step's LM-head input gradient). Its x is the GEMM row's activation and its w the layer's
weight, so product (i, j) multiplies weight slot i by an operand of use j, as `Nci.Capability` reads them. `into F` of a
lane is what its gates read outside it: x's 16 bits (a dot step's lane, its sign bit) and w's 16 (each lane reads exactly
those: `tests/test_certcheck.py`). A GEMM row's coordinates share x, so a use's distinct `into F ∩ A` are its rows' x
values, each read by N coordinates.

**Uses** (`Consumers.ofUses`), in two designs:

* `LAYER` (the one `Nci.Capability.Products` names: one token's input to one layer): use (fwd, i, t) holds the products
  of layer i's seven GEMM rows of token t (q, k, v, o, gate, up, down) and (bwd, i, t) those of its seven input-gradient
  rows; the LM head's forward and input gradient are a use each per token. A use holds one product per weight slot, so a
  forward and a backward use are two. Its region is token t's computation from the previous use's last GEMM to its own
  last GEMM's input: its rows of every GEMM and its members of every per-token batch (a norm's token, an add's words,
  SiLU's calls, attention's per-token rotary, score, softmax and output rows, forward and backward). Regions of different
  uses are disjoint, so ω = 1.
* `GEMM` (finer): one use per GEMM input and token, its region the token's computation since the previous GEMM's
  products. A use read from attention (o's input, q's, k's and v's input gradients) takes the whole attention call, the
  largest region inside attention, shared by the sequence's tokens. It shows which segments hold a gate reaching every
  value of the next GEMM's input.

**Outputs** (`Consumers.ofOutputs`): a weight gradient's coordinate, one output bit each, its region the coordinate's own
activations (its dots or dot steps, partial sums and roundings; the split-K routes' x-side preparation, shared by a row's
coordinates, is outside it). The 16 bits of one coordinate share their region (ω = 1/16).

For each consumer the measurement reports |A|, |G| for the largest certified G (`regions.required_size`), and, when G is
empty, why: a class with no gate in A (F reads nothing of the region), or no gate reaching every class.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Callable, Iterable

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import Circuit, Scope

from . import regions as RG
from .ops import Site, flatten, op_shape, param_bases

#: the counted operations: a whole dot's lanes and a dot step's, x their activation a and w their weight b
OPS = re.compile(r"HopperBF16WgmmaDot16_v\d+|HopperBf16Dot16Step(?:First)?_v\d+")
SITE = Site(op=OPS.pattern, x="a", w="b")
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")


@dataclass(frozen=True)
class Kind:
    """A kind of consumer: its region's activations and its consuming GEMMs, for layer i and token t."""

    name: str
    what: str
    region: Callable[["Step", int | None, int], list[Scope]]
    consumers: Callable[["Step", int | None], list[str]]
    per_layer: bool = True
    attention: bool = False


@dataclass
class Verdict:
    design: str
    kind: str
    layer: int | None
    token: int
    region_gates: int
    required: int
    classes: int
    finishing_sets: int
    absent: int
    why: str


@dataclass
class Output:
    weight: str
    row: int
    coordinate: int
    bit: int
    region_gates: int
    required: int
    addends: int
    why: str


class Step:
    """A built `TrainStep_v1` Program: its root activations by node name, and the consumers' regions and classes."""

    def __init__(self, program, L: int, B: int, S: int) -> None:
        self.p, self.L, self.B, self.S, self.M = program, L, B, S, B * S
        self.C = Circuit(program.root, explicit_limit=10 ** 7)
        self.root = self.C.root_scope
        self.index = {node.name: k for k, node in enumerate(program.root.body.nodes) if node.name}
        self.pull, self.res = RG.Pull(), RG.Resolver()
        theta = set(program.theta)
        self.theta = [(self.root.child(k, 0).offset, self.root.child(k, 0).offset + n.gates)
                      for k, n in enumerate(program.root.body.nodes) if n.name in theta]

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
        """Row t of the GEMM ``name`` (`LinearTorch`, `LinearTorchDx`, `LinearTorchDw`): a row GEMM batched over rows."""
        k, node = self.node(name)
        call = self.root.child(k, 0)
        inner = call.spec.body.nodes[0]
        if node.form != "call" or inner.form != "batch" or not 0 <= t < inner.n:
            raise ValueError(f"{name} is not a call batched over rows, or has no row {t}")
        return call.child(0, t)

    def rows(self, name: str) -> int:
        return self.p.root.body.nodes[self.index[name]].fn.body.nodes[0].n

    def seq(self, name: str, t: int) -> tuple[Scope, int]:
        """Token t's sequence's member of the per-sequence batch ``name`` (attention), and t's place in it."""
        k, node = self.node(name)
        if node.form != "batch" or node.n != self.B:
            raise ValueError(f"{name} is not a batch over the step's {self.B} sequences")
        return self.root.child(k, t // self.S), t % self.S

    def whole(self, name: str, t: int) -> Scope:
        return self.seq(name, t)[0]

    def attn(self, name: str, t: int) -> list[Scope]:
        """Token t's members of every per-token batch inside its sequence's attention call ``name``: a batch of S members
        is one a token, of S·h members h a token, token-major (`tests/test_certcheck.py`)."""
        call, u = self.seq(name, t)
        out = []
        for k, node in enumerate(call.spec.body.nodes):
            if node.form != "batch" or not node.fn.gates:
                continue
            if node.n % self.S:
                raise ValueError(f"{name}: node {k} ({node.fn.id}) is a batch of {node.n}, not per token")
            per = node.n // self.S
            out += [call.child(k, m) for m in range(u * per, (u + 1) * per)]
        return out

    def embed(self, t: int) -> Scope:
        return self.row("embed", t)

    def coordinate(self, row: Scope, k: int) -> list[Scope]:
        return coordinate(row, k)

    def ops(self, scope: Scope) -> Iterable[Scope]:
        return ops(scope)

    def classes(self, row: Scope) -> tuple[list[frozenset[int]], list[frozenset[int]], int]:
        return classes(row, self.res)

    def check_theta(self, acts: Iterable[Scope]) -> None:
        for s in acts:
            lo, hi = s.offset, s.offset + s.spec.gates
            if any(lo < b and a < hi for a, b in self.theta):
                raise ValueError(f"{s.path()} holds a gate of θ")

    def use(self, design: str, kind: Kind, i: int | None, t: int) -> Verdict:
        acts = kind.region(self, i, t)
        self.check_theta(acts)
        inside = _inside(acts)
        classes: Counter = Counter()
        for name in kind.consumers(self, i):
            first, last, N = self.classes(self.row(name, t))
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
        return Verdict(design, kind.name, i, t, r.gates, r.required, len(distinct), sum(classes.values()), len(r.absent),
                       why)

    def output(self, weight: str, r: int, k: int, bit: int) -> Output:
        """Output bit ``bit`` of coordinate k of row r of the weight gradient ``weight`` (`Consumers.ofOutputs`)."""
        row = self.row(weight, r)
        acts = self.coordinate(row, k)
        self.check_theta(acts)
        out = RG.returns(row)[16 * k + bit]
        res = RG.required_size(acts, [frozenset({out})], self.pull, self.res)
        addends = sum(op_shape(op.spec, SITE).lanes for s in acts for op in self.ops(s))
        if res.absent:
            return Output(weight, r, k, bit, res.gates, 0, addends, "the output gate is outside the coordinate")
        # G is what reaches into {out} = out's operands, so out itself, which the class {out} counts, is not in it
        return Output(weight, r, k, bit, res.gates, res.required - 1, addends, "")


def _coords(fn) -> int:
    return fn.signature()[1].n


def coordinate(row: Scope, k: int) -> list[Scope]:
    """Coordinate k's own activations in a row GEMM: member k of every batch over the row's N coordinates, through calls
    (a split's GEMM); a call shared by the coordinates (the x-side preparation) is not among them."""
    N = _coords(row.spec)
    out: list[Scope] = []

    def walk(scope: Scope) -> None:
        for j, node in enumerate(scope.spec.body.nodes):
            if isinstance(node.fn, PrimitiveDefinition):
                continue
            if node.form == "batch" and node.n == N:
                out.append(scope.child(j, k))
            elif node.form == "call":
                walk(scope.child(j, 0))
    walk(row)
    return out


def ops(scope: Scope) -> Iterable[Scope]:
    """The counted-operation activations at or below ``scope``."""
    if OPS.fullmatch(scope.spec.id):
        yield scope
        return
    if isinstance(scope.spec, PrimitiveDefinition):
        return
    for j, node in enumerate(scope.spec.body.nodes):
        if isinstance(node.fn, PrimitiveDefinition):
            continue
        for m in range(node.n if node.form in ("batch", "scan") else 1):
            yield from ops(scope.child(j, m))


def lane_into(op: Scope, res: RG.Resolver) -> list[frozenset[int]]:
    """`into m(p)` for each lane p of the counted-operation activation ``op``: the root gates its lane gates read outside
    it."""
    fn = op.spec
    shape, flat, base = op_shape(fn, SITE), flatten(fn), param_bases(fn)
    out = []
    for lane in range(shape.lanes):
        mine = set(shape.members[lane])
        local = {o for g in mine for o in flat.operands[g] if o >= 0 and o not in mine}
        if local:
            raise ValueError(f"{fn.id}: lane {lane} reads gates of the op outside its own")
        gates = set()
        for g in mine:
            for o in flat.operands[g]:
                if o < 0:
                    j = -1 - o
                    pi = next(i for i in range(len(base) - 1) if base[i] <= j < base[i + 1])
                    gates.add(res.param(op, pi, j - base[pi]))
        out.append(frozenset(gates))
    return out


def classes(row: Scope, res: RG.Resolver) -> tuple[list[frozenset[int]], list[frozenset[int]], int]:
    """`into F` of each product of a GEMM row's first and last coordinates, and its number of coordinates N."""
    N = _coords(row.spec)
    ends = []
    for n in (0, N - 1):
        ends.append([c for s in coordinate(row, n) for op in ops(s) for c in lane_into(op, res)])
    return ends[0], ends[1], N


def _inside(acts: Iterable[Scope]):
    spans = sorted((s.offset, s.offset + s.spec.gates) for s in acts)

    def inside(g: int) -> bool:
        return any(lo <= g < hi for lo, hi in spans)
    return inside


def _lin(*names: str):
    return lambda S, i: [f"l{i}.{n}" for n in names]


def _toks(S: Step, t: int, *names: str) -> list[Scope]:
    return [a for n in names for a in S.tok(n, t)]


def _y1(S: Step, i: int, t: int) -> list[Scope]:
    up = [S.embed(t)] if i == 0 else [S.row(f"l{i - 1}.down", t), *S.tok(f"l{i - 1}.h3", t)]
    return up + S.tok(f"l{i}.ln1", t)


def _dh(S: Step, i: int, t: int) -> list[Scope]:
    if i == S.L - 1:
        return [S.row("lm_head.dx", t), *S.tok("norm.dx", t)]
    j = f"l{i + 1}."
    return [*(S.row(j + f"{w}.dx", t) for w in ("q", "k", "v")), *_toks(S, t, *(j + n for n in ("kv.dx", "qkv.dx",
                                                                                                  "ln1.dx", "x.grad")))]


def _yn(S: Step, i, t: int) -> list[Scope]:
    return [S.row(f"l{S.L - 1}.down", t), *S.tok(f"l{S.L - 1}.h3", t), *S.tok("norm", t)]


def _dlogits(S: Step, i, t: int) -> list[Scope]:
    return [S.row("logits", t), *_toks(S, t, "loss.rows", "logits.dx")]


def _fwd(S: Step, i: int, t: int) -> list[Scope]:
    p = f"l{i}."
    return [*_y1(S, i, t), *(S.row(p + w, t) for w in ("q", "k", "v")), *S.attn(p + "attn", t), S.row(p + "o", t),
            *_toks(S, t, p + "h2", p + "ln2"), S.row(p + "gate", t), S.row(p + "up", t), *_toks(S, t, p + "act", p + "mul")]


def _bwd(S: Step, i: int, t: int) -> list[Scope]:
    p = f"l{i}."
    return [*_dh(S, i, t), S.row(p + "down.dx", t), *_toks(S, t, p + "mul.da", p + "mul.db", p + "act.dx"),
            S.row(p + "up.dx", t), S.row(p + "gate.dx", t), *_toks(S, t, p + "mlp.dx", p + "ln2.dx", p + "h2.grad"),
            S.row(p + "o.dx", t), *S.attn(p + "attn.dx", t)]


LM = (
    Kind("yn", "the LM head's input: the last layer's down row and residual add, and the final norm", _yn,
         lambda S, i: ["logits"], per_layer=False),
    Kind("dlogits", "the LM head's input gradient: the logits row, the log-softmax row and its backward", _dlogits,
         lambda S, i: ["lm_head.dx"], per_layer=False),
)

LAYER = (
    Kind("fwd", "the layer's forward: its input (the embedding, or the previous layer's down row and residual add), ln1, "
         "the q, k, v rows, attention's rows, o's row, the residual add, ln2, the gate and up rows, SiLU and the product",
         _fwd, _lin(*LINEARS)),
    Kind("bwd", "the layer's backward: its output gradient (the next layer's q, k, v input-gradient rows, their sums, ln1's "
         "backward and the residual sum, or the LM head's input-gradient row and the final norm's backward), the down, "
         "up and gate input-gradient rows, the product's and SiLU's backward, ln2's backward, the residual sum, o's "
         "input-gradient row and attention's backward rows", _bwd, _lin(*(f"{w}.dx" for w in LINEARS))),
    *LM,
)

GEMM = (
    Kind("y1", "ln1's output: the layer's input (the embedding, or the previous layer's down row and residual add) and ln1",
         _y1, _lin("q", "k", "v")),
    Kind("o_in", "attention's output row (o's input): the whole attention call", lambda S, i, t: [S.whole(f"l{i}.attn", t)],
         _lin("o"), attention=True),
    Kind("y2", "ln2's output: o's row, the residual add h2 and ln2",
         lambda S, i, t: [S.row(f"l{i}.o", t), *_toks(S, t, f"l{i}.h2", f"l{i}.ln2")], _lin("gate", "up")),
    Kind("mul", "down's input: gate's and up's rows, SiLU and the product",
         lambda S, i, t: [S.row(f"l{i}.gate", t), S.row(f"l{i}.up", t), *_toks(S, t, f"l{i}.act", f"l{i}.mul")],
         _lin("down")),
    *LM,
    Kind("dh", "the layer output's gradient (down's dX input): `bwd`'s first segment", _dh, _lin("down.dx")),
    Kind("dup", "up's dX input: down's dX row and the product's gradient into up",
         lambda S, i, t: [S.row(f"l{i}.down.dx", t), *S.tok(f"l{i}.mul.db", t)], _lin("up.dx")),
    Kind("dgate", "gate's dX input: the product's gradient into SiLU and SiLU's backward",
         lambda S, i, t: _toks(S, t, f"l{i}.mul.da", f"l{i}.act.dx"), _lin("gate.dx")),
    Kind("dh2", "o's dX input: up's and gate's dX rows, their sum, ln2's backward and the residual sum",
         lambda S, i, t: [S.row(f"l{i}.up.dx", t), S.row(f"l{i}.gate.dx", t),
                          *_toks(S, t, *(f"l{i}.{n}" for n in ("mlp.dx", "ln2.dx", "h2.grad")))],
         _lin("o.dx")),
    *(Kind(f"d{w}", f"{w}'s dX input: the whole attention backward call",
           lambda S, i, t: [S.whole(f"l{i}.attn.dx", t)], _lin(f"{w}.dx"), attention=True) for w in ("v", "k", "q")),
)

DESIGNS = {"layer": LAYER, "gemm": GEMM}


def uses(S: Step, tokens: Iterable[int], layers: Iterable[int] | None = None, design: str = "layer",
         kinds: Iterable[str] | None = None) -> list[Verdict]:
    layers = list(range(S.L) if layers is None else layers)
    tokens = list(tokens)
    out = []
    for kind in DESIGNS[design]:
        if kinds is not None and kind.name not in kinds:
            continue
        for i in (layers if kind.per_layer else [None]):
            for t in tokens:
                out.append(S.use(design, kind, i, t))
    return out


def spans(S: Step, design: str, tokens: Iterable[int], layers: Iterable[int] | None = None) -> dict:
    """Each use's region as gate intervals, by (kind, layer, token)."""
    layers = list(range(S.L) if layers is None else layers)
    out = {}
    for kind in DESIGNS[design]:
        for i in (layers if kind.per_layer else [None]):
            for t in tokens:
                out[(kind.name, i, t)] = sorted((s.offset, s.offset + s.spec.gates) for s in kind.region(S, i, t))
    return out


def overlap_weights(regions: dict) -> dict:
    """ω_c = 1 / (the most regions holding any one gate of c's), which `Consumers.OverlapWeights` accepts: at a gate held by
    m regions each has ω ≤ 1/m. Regions are lists of disjoint gate intervals."""
    events = []
    for c, ivs in regions.items():
        for lo, hi in ivs:
            if hi > lo:
                events += [(lo, 1, c), (hi, -1, c)]
    events.sort(key=lambda e: (e[0], e[1]))
    active: Counter = Counter()
    most = {c: 1 for c in regions}
    for _pos, d, c in events:
        active[c] += d
        if active[c] > 1:
            raise ValueError(f"region {c}'s intervals overlap")
        if d > 0:
            m = sum(1 for v in active.values() if v)
            for a, v in active.items():
                if v and m > most[a]:
                    most[a] = m
    return {c: Fraction(1, m) for c, m in most.items()}


def products(fn, memo: dict | None = None) -> int:
    """The counted products in the Definition ``fn``: lanes of its counted-operation activations."""
    memo = {} if memo is None else memo
    hit = memo.get(id(fn))
    if hit is not None:
        return hit[1]
    if OPS.fullmatch(fn.id):
        n = op_shape(fn, SITE).lanes
    elif isinstance(fn, PrimitiveDefinition):
        n = 0
    else:
        n = sum((node.n if node.form in ("batch", "scan") else 1) * products(node.fn, memo) for node in fn.body.nodes)
    memo[id(fn)] = (fn, n)
    return n


def coverage(S: Step) -> dict:
    """The step's counted products by where they are: the uses' GEMMs (forward and input gradient, the LM head's
    included), the weight gradients, attention, and anything else."""
    memo: dict = {}
    fwd = {f"l{i}.{w}" for i in range(S.L) for w in LINEARS} | {"logits"}
    dx = {f"l{i}.{w}.dx" for i in range(S.L) for w in LINEARS} | {"lm_head.dx"}
    dw = {f"l{i}.{w}.dw" for i in range(S.L) for w in LINEARS} | {"lm_head.dw"}
    out: Counter = Counter()
    for node in S.p.root.body.nodes:
        n = (node.n if node.form in ("batch", "scan") else 1) * products(node.fn, memo)
        if not n:
            continue
        where = ("forward" if node.name in fwd else "input_gradient" if node.name in dx else "weight_gradient"
                 if node.name in dw else "attention" if node.name and node.name.split(".", 1)[-1] in ("attn", "attn.dx")
                 else "other")
        out[where] += n
        if where == "other":
            out[f"other:{node.name}"] += n
    out["total"] = sum(v for k, v in out.items() if ":" not in k)
    return dict(out)


def as_dict(v) -> dict:
    return asdict(v)
