"""The work the min-cut bound charges, read off a Program, and the bound at a policy.

The circuit is a Program in the `verity-ir/descriptor/v1` format, and nothing beside it: the reading takes no source
language, trace, model or optimizer. It is computed once per specialized Definition and context and multiplied through
batches and scans, so it never walks the unrolled circuit. It reads, for `min_cut.bound`'s inputs:

* `|C|` (`Work.gates`, `boolean_gates`): Boolean gates, each primitive at its Boolean version's AND, XOR and NOT
  gates and an input node at none; a primitive with no Boolean version is refused by name.
* The products: each activation of a Definition in `PRODUCTS` (outermost first, so a `LinearRef`'s own dots are its
  MACs) multiplies its parameter `x` by its parameter `w`, one MAC a word of the larger of the two, and gives its
  returned leaves. A product's operand reads the arrays its argument's references name.
* Which are charged: a product both of whose operands are wholly reached from a fresh input (a run input outside
  `θ`; a registered input, a `θ` input and a constant are not), neither of which reads a lookup's words, and whose
  two operands are no one array. A word is a lookup's (`LOOKUPS`) where the table it is read from is not reached
  from a fresh input; a value computed from such words and words no fresh input reaches is one too, and one computed
  from any value all of whose words reach a fresh input some other way is not. These are the words `min_cut` names
  "a function of θ and a few fresh gates", whose matmuls a caller leaves out of `N`.
* `N` and `O`: the charged products' MACs, and their returned leaves times their width.
* The kinds: an operand is a weight where it reads nothing of the current iteration's inputs: in a scan, its
  sequence arguments' (`xs`), so a carried value is a weight from the iteration that first finds it reached from a
  fresh input; outside every scan, nothing is a weight.
* The `β` and `b`: a weight's `β` at most `min_cut`'s default (bf16's 16 bits), an activation's at most `β(k)` and
  `b` at most bf16's 16, each at most its width and a `β` at most the source bits its operand's values hold. A fresh
  input's word and, in a scan, a carried word reached from one hold their width; a computed array's words hold on
  average the bits of the source words its node reads, each once, over its words, at most their width. So a weight a
  body decodes from a narrower carry (a ternary code's two bit planes and a scale) holds its code's bits.
* `F`: `share(ω)`, ω the largest set of arrays the work multiplies pairwise (at least 2).

Each status is per array, never per word: a node's output has one status unless it is a call's or a scan's, whose
status is its callee's returned runs. A value is fresh only where every word read is, a lookup's where any may be, and
two arrays are one where any references overlap. Against the statements, the reading's errors go both ways:

* Less work than they count, so a bound true but weak: a product the tables don't name is not counted, a value with
  any word no fresh input reaches is θ's, and overlapping references are one array (F may be higher).
* More than they may support, so a bound the theorem may not give: (H) fails at a `β` above the bits a value holds,
  and the source bits are an average from data flow, which a value holds no more of where words repeat, as a gather's
  rows do over the tokens that share an id; and a value carried through a scan's state is a weight, at bf16's 16
  bits where an activation is at `β(k)`, which a submitter can choose.

`bound` takes the most blocks over the sub-works whose products are at least a threshold β_w·β_x, at least a
contraction length and of one kind or both, each at the least β_w·β_x, the least `b` and the ω of its own products,
through the calculator's `min_cut`.
"""

from __future__ import annotations

import bisect
import math
import re
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.program import is_input, is_registered
from verity.circuits.refs import Affine, Concat, Strided

from .ops import ids


@dataclass(frozen=True)
class Product:
    """Activations of a Definition matching ``op`` are matrix products of parameter ``x`` by parameter ``w``: one MAC a
    word of the larger, each output a returned leaf."""

    op: str
    x: str
    w: str

    @cached_property
    def rx(self) -> re.Pattern:
        return re.compile(self.op)


@dataclass(frozen=True)
class Lookup:
    """Activations of a primitive matching ``op`` read a word of parameter ``table`` at parameter ``index``."""

    op: str
    index: str
    table: str

    @cached_property
    def rx(self) -> re.Pattern:
        return re.compile(self.op)


#: the reference registry's matrix products (`verity.definitions.registry.ref_prims`): a row times a matrix, and a dot
PRODUCTS = (Product(r"LinearRef_v\d+", "x", "w"), Product(r"DotRef_v\d+", "x", "w"))
#: the registered gathers: a column of a table at a token
LOOKUPS = (Lookup(r"Gather\w+x\d+_v\d+", "token", "column"),)


class Status(NamedTuple):
    """What an array's words depend on."""

    #: every word is reached from a fresh input
    fresh: bool
    #: some word may be
    reach: bool
    #: some word may be a function of lookups' words and words no fresh input reaches
    looked: bool
    #: the scan levels (0: outside every scan) whose current inputs some word may read
    data: frozenset
    #: the source bits a word holds on average: a fresh input's and a carried word's width, θ's 0, and a computed
    #: array's the bits of the sources it reads over its words, at most its width
    bits: float = 0.0


THETA = Status(False, False, False, frozenset())


def join(sts, weights=None) -> Status:
    """The status of an array made of arrays of statuses ``sts``, of ``weights`` leaves each (None: its bits the
    most of theirs)."""
    sts = list(sts)
    if not sts:
        return THETA
    if weights is None:
        bits = max(s.bits for s in sts)
    else:
        weights = list(weights)
        bits = sum(w * s.bits for w, s in zip(weights, sts)) / max(1, sum(weights))
    return Status(all(s.fresh for s in sts), any(s.reach for s in sts), any(s.looked for s in sts),
                  frozenset().union(*(s.data for s in sts)), bits)


def computed(sts, bits: float) -> Status:
    """The status of an array each of whose words is computed from words of each of arrays ``sts``, ``bits`` a word."""
    sts = list(sts)
    looked = any(s.looked for s in sts) and not any(s.fresh and not s.looked for s in sts)
    return Status(any(s.fresh for s in sts), any(s.reach for s in sts), looked,
                  frozenset().union(*(s.data for s in sts)), bits)


#: an array's status by leaf: runs (end, status), ends rising, the last the array's leaves
Profile = tuple


def runs_of(parts: list[tuple[int, Status]]) -> Profile:
    out: list[tuple[int, Status]] = []
    for end, s in parts:
        if out and out[-1][1] == s:
            out[-1] = (end, s)
        elif not out or end > out[-1][0]:
            out.append((end, s))
    return tuple(out)


def span(p: Profile, lo: int, hi: int) -> Status:
    """``p``'s status over leaves ``lo`` to ``hi``."""
    out, weights = _over(p, lo, hi)
    return join(out, weights)


def _over(p: Profile, lo: int, hi: int) -> tuple[list[Status], list[int]]:
    """The runs of ``p`` over leaves ``lo`` to ``hi``, and how many of those leaves each holds."""
    ends = [e for e, _ in p]
    i = bisect.bisect_right(ends, lo)
    out, weights = [], []
    while i < len(p):
        start = p[i - 1][0] if i else 0
        out.append(p[i][1])
        weights.append(min(p[i][0], hi) - max(start, lo))
        if p[i][0] >= hi:
            break
        i += 1
    return out, weights


def bits_in(p: Profile, lo: int, hi: int) -> float:
    """The source bits ``p``'s leaves ``lo`` to ``hi`` hold."""
    out, weights = _over(p, lo, hi)
    return sum(w * s.bits for w, s in zip(weights, out))


def whole(p: Profile) -> Status:
    return join((s for _, s in p), (e - b for (e, _), b in zip(p, (0, *(e for e, _ in p)))))


def drop(p: Profile, levels: frozenset) -> Profile:
    return runs_of([(e, s._replace(data=s.data - levels)) for e, s in p])


def capped(p: Profile, bits: float) -> Profile:
    return runs_of([(e, s._replace(bits=min(s.bits, bits))) for e, s in p])


def carried(p: Profile, t) -> Profile:
    """A carry ``p`` of type ``t`` as an iteration's source: a word reached from a fresh input holds its width, any
    other none."""
    cuts = sorted({e for e, _ in p} | {e for e, _ in _width_runs(t)})
    widths = _width_runs(t)
    out, i, j = [], 0, 0
    for e in cuts:
        while p[i][0] < e:
            i += 1
        while widths[j][0] < e:
            j += 1
        s = p[i][1]
        out.append((e, s._replace(bits=float(widths[j][1]) if s.reach else 0.0)))
    return runs_of(out)


def _width_runs(t) -> list[tuple[int, int]]:
    """A type's leaf widths as runs (end, width)."""
    w = t.homogeneous_width()
    if w is not None or not t.leaves:
        return [(t.leaves, w or 0)]
    out: list[tuple[int, int]] = []
    for w in t.leaf_widths():
        if out and out[-1][1] == w:
            out[-1] = (out[-1][0] + 1, w)
        else:
            out.append(((out[-1][0] if out else 0) + 1, w))
    return out


def _parts(refs, memo: dict) -> list[tuple[int, int, str, int, int, int]]:
    """The references ``refs`` as runs (a, b, space, idx, lo, hi): positions a to b of the sequence read leaves inside
    lo to hi of (space, idx). A strided run is its hull, and a prefix shared through a `PartLog` one run a source over
    the whole sequence, computed once a log."""
    out: list = []

    def walk(r, at: int) -> None:
        if isinstance(r, Affine):
            if r.count:
                out.append((at, at + r.count, r.space, r.idx, r.base, r.base + r.count))
        elif isinstance(r, Strided):
            if not r.count:
                return
            rows = -(-r.count // r.inner)
            corners = [r.base + q * r.outer_stride + j * r.inner_stride for q in (0, rows - 1)
                       for j in (0, min(r.inner, r.count) - 1)]
            out.append((at, at + r.count, r.space, r.idx, min(corners), max(corners) + 1))
        elif isinstance(r, Concat) and r.prefix is not None:
            log, j = r.prefix
            for (space, idx), (lo, hi) in _hulls(log, j, memo).items():
                out.append((at, at + r.count, space, idx, lo, hi))
            pos = at + log.starts[j]
            for p in r.parts[j:]:
                walk(p, pos)
                pos += p.count
        elif isinstance(r, Concat):
            pos = at
            for p in r.parts:
                walk(p, pos)
                pos += p.count
        else:
            for j, (space, idx, leaf) in enumerate(r):
                if out and out[-1][1] == at + j and out[-1][2:4] == (space, idx) and out[-1][5] == leaf:
                    out[-1] = (out[-1][0], at + j + 1, space, idx, out[-1][4], leaf + 1)
                else:
                    out.append((at + j, at + j + 1, space, idx, leaf, leaf + 1))

    walk(refs, 0)
    return out


def _hulls(log, j: int, memo: dict) -> dict:
    """The hull each source's leaves span in a `PartLog`'s first ``j`` parts, extended part by part."""
    hs = memo.setdefault(id(log), (log, [{}]))[1]
    while len(hs) <= j:
        h = dict(hs[-1])
        for _a, _b, space, idx, lo, hi in _parts(log.parts[len(hs) - 1], memo):
            old = h.get((space, idx))
            h[(space, idx)] = (lo, hi) if old is None else (min(old[0], lo), max(old[1], hi))
        hs.append(h)
    return hs[j]


@dataclass
class Site:
    """One product node of one Definition's body in one context."""

    product: Product
    node: int
    #: activations a node: a batch's members
    n: int
    macs: int
    outputs: int
    out_width: int
    k_dim: int
    widths: tuple[int, int]
    x: Status
    w: Status
    #: each operand's source runs (space, idx, lo, hi) in the body
    sources: tuple[tuple, tuple]
    #: the source bits each operand's words hold on average
    bits: tuple[float, float] = (0.0, 0.0)


@dataclass
class Summary:
    sid: int
    fn_id: str
    ret: Profile
    sites: list[Site] = field(default_factory=list)
    #: (callee summary, activations a parent activation, bindings (callee param, a, b, space, idx, lo, hi))
    children: list[tuple[Summary, int, list]] = field(default_factory=list)


def _width(t) -> int:
    w = t.homogeneous_width()
    if w is None:
        raise ValueError(f"a product's operand of mixed widths ({sorted(t.width_set())})")
    return w


def _unit(t) -> int:
    """A type's words: its leaves (a word program's leaf is one word)."""
    return t.leaves


class Reader:
    def __init__(self, program, products=PRODUCTS, lookups=LOOKUPS) -> None:
        self.program, self.products, self.lookups = program, products, lookups
        self.memo: dict = {}
        self.logs: dict = {}
        self._match: dict = {}

    def product(self, fn) -> Product | None:
        return self._of(fn, self.products)

    def lookup(self, fn) -> Lookup | None:
        return self._of(fn, self.lookups)

    def _of(self, fn, table):
        key = (id(table), fn.id)
        if key not in self._match:
            self._match[key] = next((p for p in table if any(p.rx.fullmatch(i) for i in ids(fn))), None)
        return self._match[key]

    def parts(self, refs):
        return _parts(refs, self.logs)

    def root(self) -> Summary:
        root = self.program.root
        return self.summary(root, (), 0, theta=frozenset(root.theta))

    def summary(self, fn, ctx: tuple, depth: int, theta: frozenset = frozenset()) -> Summary:
        key = (fn.id, ctx, depth)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        body = fn.body
        S = Summary(-1, fn.id, ())
        prof: list[Profile] = []

        def source(space: str, idx: int) -> Profile:
            return ctx[idx] if space == "p" else prof[idx]

        def arg_profile(a) -> Profile:
            return runs_of([(b_, span(source(s, i), lo, hi)) for _a, b_, s, i, lo, hi in self.parts(a.refs)]) or \
                ((a.refs.count, THETA),)

        def arg_status(a) -> Status:
            return join(span(source(s, i), lo, hi) for _a, _b, s, i, lo, hi in self.parts(a.refs))

        def bindings(i: int, a, mapped: bool) -> list:
            ps = self.parts(a.refs)
            if mapped:
                return [(i, 0, -1, s, j, lo, hi) for _a, _b, s, j, lo, hi in ps]
            return [(i, a_, b_, s, j, lo, hi) for a_, b_, s, j, lo, hi in ps]

        def read_bits(args) -> tuple[float, int]:
            """The source bits ``args`` read between them, each source leaf once, and how many leaves those are."""
            spans: dict = {}
            for a in args:
                for _a, _b, s, i, lo, hi in self.parts(a.refs):
                    spans.setdefault((s, i), []).append((lo, hi))
            total, n = 0.0, 0
            for (s, i), rs in spans.items():
                rs.sort()
                lo, hi = rs[0]
                for a, b in [*rs[1:], (math.inf, math.inf)]:
                    if a > hi:
                        total += bits_in(source(s, i), lo, hi)
                        n += hi - lo
                        lo, hi = a, b
                    else:
                        hi = max(hi, b)
            return total, n

        def made_bits(args, out) -> float:
            """The source bits a word of ``out`` holds on average, computed from ``args``: at most its width."""
            return min(max(out.width_set(), default=0), read_bits(args)[0] / out.leaves) if out.leaves else 0.0

        def operand_bits(a) -> float:
            total, n = read_bits([a])
            return total / n if n else 0.0

        for k, nd in enumerate(body.nodes):
            f, leaves = nd.fn, nd.out.leaves
            if is_input(f):
                st = THETA if is_registered(f) or k in theta else \
                    Status(True, True, False, frozenset({0}), float(max(nd.out.width_set(), default=0)))
                prof.append(((leaves, st),))
                continue
            if isinstance(f, PrimitiveDefinition):
                sts = [arg_status(a) for a in nd.args]
                bits = made_bits(nd.args, nd.out)
                lk = self.lookup(f)
                if lk is not None:
                    names = [n for n, _t in f.params]
                    t, x = sts[names.index(lk.table)], sts[names.index(lk.index)]
                    st = Status(x.fresh, True, True, t.data | x.data, bits) if x.reach and not t.reach else \
                        computed(sts, bits)
                else:
                    st = computed(sts, bits)
                prof.append(((leaves, st),))
                continue
            p = self.product(f)
            if p is not None and nd.form in ("call", "batch"):
                params, ret = f.signature()
                names = [n for n, _t in params]
                ix, iw = names.index(p.x), names.index(p.w)
                tx, tw = params[ix][1], params[iw][1]
                S.sites.append(Site(p, k, nd.n if nd.form == "batch" else 1, max(_unit(tx), _unit(tw)), ret.leaves,
                                    _width(ret), min(_unit(tx), _unit(tw)), (_width(tx), _width(tw)),
                                    arg_status(nd.args[ix]), arg_status(nd.args[iw]),
                                    (tuple(x[2:] for x in self.parts(nd.args[ix].refs)),
                                     tuple(x[2:] for x in self.parts(nd.args[iw].refs))),
                                    (operand_bits(nd.args[ix]), operand_bits(nd.args[iw]))))
                prof.append(((leaves, computed((arg_status(a) for a in nd.args), made_bits(nd.args, nd.out))),))
                continue
            bits = made_bits(nd.args, nd.out)
            if nd.form == "call":
                C = self.summary(f, tuple(arg_profile(a) for a in nd.args), depth)
                S.children.append((C, 1, [b for i, a in enumerate(nd.args) for b in bindings(i, a, False)]))
                prof.append(capped(C.ret, bits))
            elif nd.form == "batch":
                params, _ = f.signature()
                cctx = tuple(arg_profile(a) if ax is None else ((t.leaves, arg_status(a)),)
                             for a, ax, (_n, t) in zip(nd.args, nd.axes, params))
                C = self.summary(f, cctx, depth)
                S.children.append((C, nd.n, [b for i, (a, ax) in enumerate(zip(nd.args, nd.axes))
                                             for b in bindings(i, a, ax is not None)]))
                per = C.ret[-1][0]
                prof.append(capped(runs_of([(m * per + e, s) for m in range(nd.n) for e, s in C.ret])
                                   if len(C.ret) == 1 or nd.n * len(C.ret) <= 4096 else ((leaves, whole(C.ret)),), bits))
            elif nd.form == "scan":
                prof.append(capped(self.scan(S, nd, depth, arg_profile, arg_status, bindings), bits))
            else:
                raise ValueError(f"{fn.id} node {k}: form {nd.form!r}")
        S.ret = runs_of([(b_, span(source(s, i), lo, hi)) for _a, b_, s, i, lo, hi in self.parts(body.ret)]) or \
            ((body.ret.count, THETA),)
        S.sid = len(self.memo)
        self.memo[key] = S
        return S

    def scan(self, S: Summary, nd, depth: int, arg_profile, arg_status, bindings) -> Profile:
        """A scan's iterations by context: the first's carry its init's, each next one's the carry the one before
        returned, until a context repeats. A carried value reads none of the current iteration's inputs, and outside
        every scan, the init and the shared arguments are the loop's, not an iteration's."""
        step, n, nx = nd.fn, nd.n, nd.n_xs
        params, ret = step.signature()
        inner = depth + 1
        gone = frozenset({inner} | ({0} if depth == 0 else set()))
        outer = frozenset({0}) if depth == 0 else frozenset()
        xs = tuple(((params[1 + j][1].leaves, (s := arg_status(nd.args[1 + j]))._replace(data=s.data | {inner})),)
                   for j in range(nx))
        shared = tuple(drop(arg_profile(a), outer) for a in nd.args[1 + nx:])
        carry_leaves = params[0][1].leaves
        carry = carried(drop(arg_profile(nd.args[0]), gone), params[0][1])
        seq: list[tuple[tuple, Summary]] = []
        seen: dict = {}
        while len(seq) < n:
            ctx = (carry, *xs, *shared)
            if ctx in seen:
                break
            seen[ctx] = len(seq)
            C = self.summary(step, ctx, inner)
            seq.append((ctx, C))
            carry = carried(drop(runs_of(_cut(C.ret, 0, carry_leaves)), gone), params[0][1])
        counts = [1] * len(seq)
        last = len(seq) - 1
        if len(seq) < n:
            s0 = seen[(carry, *xs, *shared)]
            cycle = len(seq) - s0
            full, rem = divmod(n - len(seq), cycle)
            for i in range(cycle):
                counts[s0 + i] += full + (i < rem)
            last = s0 + (n - len(seq) - 1) % cycle
        for i, (_ctx, C) in enumerate(seq):
            bs = [b for j, a in enumerate(nd.args) if j or i == 0
                  for b in bindings(j, a, 1 <= j <= nx)]
            S.children.append((C, counts[i], bs))
        ys = join(whole(_cut(C.ret, carry_leaves, C.ret[-1][0])) for _ctx, C in seq)
        y_leaves = ret.leaves - carry_leaves
        return runs_of([*_cut(seq[last][1].ret, 0, carry_leaves), (carry_leaves + n * y_leaves, ys)])


def _cut(p: Profile, lo: int, hi: int) -> list[tuple[int, Status]]:
    """Runs of ``p`` over leaves ``lo`` to ``hi``, ends relative to ``lo``."""
    out, start = [], 0
    for e, s in p:
        a, b = max(start, lo), min(e, hi)
        if b > a:
            out.append((b - lo, s))
        start = e
    return out


class _Classes:
    """Arrays as classes of runs (scope, space, idx, lo, hi): runs of one source that overlap are one array, and so are
    a callee's parameter and the arrays its caller binds to it."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, v):
        p = self.parent.setdefault(v, v)
        while p != v:
            self.parent[v] = self.parent.setdefault(p, p)
            v, p = p, self.parent[p]
        return p

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def overlaps(self) -> None:
        by: dict = {}
        for v in list(self.parent):
            by.setdefault(v[:3], []).append(v)
        for vs in by.values():
            vs.sort(key=lambda v: (v[3], v[4]))
            hi, prev = None, None
            for v in vs:
                if hi is not None and v[3] < hi:
                    self.union(prev, v)
                    hi = max(hi, v[4])
                else:
                    hi = v[4]
                prev = v


@dataclass(frozen=True)
class Group:
    """Charged products of one shape: an operand pair's widths and kinds, an output width and a contraction length."""

    product: str
    #: (x, w) widths in bits, and whether each is a weight
    widths: tuple[int, int]
    weight: tuple[bool, bool]
    out_width: int
    k_dim: int
    macs: int
    wires: int
    #: the pairs of arrays its products multiply
    edges: frozenset
    #: (x, w) source bits a value holds, which cap its β
    bits: tuple[float, float] = (math.inf, math.inf)


@dataclass
class Work:
    """The products of a Program and the ones the bound charges."""

    #: |C| in Boolean gates (`boolean_gates`)
    gates: int
    groups: list[Group]
    #: MACs a product left out by why: "theta" (an operand some word of which no fresh input reaches), "lookup" (an
    #: operand that reads a lookup's words) and "self" (both operands of one array)
    left_out: dict[str, int]

    @property
    def N(self) -> int:
        return sum(g.macs for g in self.groups)

    @property
    def O(self) -> int:
        return sum(g.wires for g in self.groups)


def boolean_gates(program) -> int:
    """|C| in Boolean gates: each primitive at its Boolean version's AND, XOR and NOT gates (`boolean_version`, counted
    by `verity.circuits.evaluation.bits.compile`, as `training_slowdown.gates` prices a traced op), an input node at
    none, summed once per Definition and multiplied through batches and scans. `NoBooleanVersion` names every
    primitive that has none."""
    from verity.circuits.evaluation.bits import compile as compile_bits
    from verity.definitions.boolean import NoBooleanVersion, boolean_version

    memo: dict = {}
    missing: dict = {}

    def primitive(f) -> int:
        if is_input(f):
            return 0
        try:
            return sum(compile_bits(boolean_version(f)).counts.values())
        except NoBooleanVersion as e:
            missing.update(e.missing)
            return 0

    def of(fn) -> int:
        if fn.id not in memo:
            memo[fn.id] = primitive(fn) if isinstance(fn, PrimitiveDefinition) else \
                sum(of(nd.fn) * (nd.n if nd.form in ("batch", "scan") else 1) for nd in fn.body.nodes)
        return memo[fn.id]

    total = of(program.root)
    if missing:
        raise NoBooleanVersion(missing)
    return total


def read(program, products=PRODUCTS, lookups=LOOKUPS) -> Work:
    """The work ``program`` holds (module docstring)."""
    gates = boolean_gates(program)
    R = Reader(program, products, lookups)
    root = R.root()
    order = sorted(R.memo.values(), key=lambda s: -s.sid)
    count = {root.sid: 1}
    for s in order:
        for c, n, _b in s.children:
            count[c.sid] = count.get(c.sid, 0) + count.get(s.sid, 0) * n
    cls = _Classes()
    for s in order:
        for c, _n, bs in s.children:
            for i, a, b, space, idx, lo, hi in bs:
                cls.union((c.sid, "p", i, a if b >= 0 else 0, b if b >= 0 else math.inf), (s.sid, space, idx, lo, hi))
        for site in s.sites:
            for src in site.sources:
                for v in src:
                    cls.find((s.sid, *v))
    cls.overlaps()
    ops = []
    for s in order:
        for site in s.sites:
            xs, ws = ([(s.sid, *v) for v in src] for src in site.sources)
            for vs in (xs, ws):
                for v in vs[1:]:
                    cls.union(vs[0], v)
            ops.append((s, site, xs[0] if xs else None, ws[0] if ws else None))
    left = {"theta": 0, "lookup": 0, "self": 0}
    groups: dict = {}
    for s, site, x, w in ops:
        n = count.get(s.sid, 0) * site.n
        macs = n * site.macs
        if not macs:
            continue
        if not (site.x.fresh and site.w.fresh):
            left["theta"] += macs
        elif site.x.looked or site.w.looked:
            left["lookup"] += macs
        elif x is None or w is None or cls.find(x) == cls.find(w):
            left["self"] += macs
        else:
            key = (site.product.op, site.widths, (not site.x.data, not site.w.data), site.out_width, site.k_dim,
                   tuple(round(b, 6) for b in site.bits))
            g = groups.setdefault(key, [0, 0, set()])
            g[0] += macs
            g[1] += n * site.outputs * site.out_width
            g[2].add(frozenset((cls.find(x), cls.find(w))))
    return Work(gates, [Group(k[0], k[1], k[2], k[3], k[4], m, o, frozenset(e), k[5])
                        for k, (m, o, e) in sorted(groups.items(), key=lambda kv: repr(kv[0]))], left)


def clique(edges) -> int:
    """The most vertices pairwise joined by ``edges`` (pairs), at least 2 where there is an edge."""
    adj: dict = {}
    for e in edges:
        a, b = tuple(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best = 0

    def grow(r: int, p: set, x: set) -> None:
        nonlocal best
        if not p and not x:
            best = max(best, r)
            return
        if r + len(p) <= best:
            return
        u = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[u]):
            grow(r + 1, p & adj[v], x & adj[v])
            p.discard(v)
            x.add(v)

    grow(0, set(adj), set())
    return best


def min_cut():
    """The calculator's `min_cut`, imported as the training benchmark's calculator imports it."""
    rho = str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "inference_only" / "rho")
    if rho not in sys.path:
        sys.path.append(rho)
    from rho import min_cut as mc
    return mc


@dataclass(frozen=True)
class Priced:
    """The bound at a policy on a sub-work: which groups, their ω, and `min_cut.bound`'s result."""

    bound: object
    groups: tuple[Group, ...]
    omega: int
    N: int
    O: int


def betas(g: Group, bk: float, bw: float) -> tuple[float, float]:
    """A group's (β_w, β_x): a weight's at most ``bw``, an activation's at most ``bk``, each at most its width and the
    source bits a value holds; where both are of one kind, the larger first."""
    b = [min(w, bits, bw if weight else bk) for w, bits, weight in zip(g.widths, g.bits, g.weight)]
    if g.weight[0] != g.weight[1]:
        return (b[1], b[0]) if g.weight[1] else (b[0], b[1])
    return tuple(sorted(b, reverse=True))


def bound(work: Work, c: float, k: float, n: float | None = None) -> Priced | None:
    """The best bound at the policy (c, k, n) over ``work``'s sub-works (module docstring); None where it has no
    charged products or (H) states no activation's β at k."""
    mc = min_cut()
    bk = mc.beta(k)
    if bk is None or not work.groups:
        return None
    priced = [(math.prod(betas(g, bk, mc.BF16)), g) for g in work.groups]
    best = None
    for P in sorted({q for q, _ in priced if q > 0}):
        for K0 in sorted({g.k_dim for g in work.groups}):
            for kinds in ("both", "weights", "activations"):
                M = [(q, g) for q, g in priced if q >= P and g.k_dim >= K0 and
                     (kinds == "both" or any(g.weight) == (kinds == "weights"))]
                if not M:
                    continue
                bw, bx = betas(min(M, key=lambda x: x[0])[1], bk, mc.BF16)
                omega = max(2, clique(e for _, g in M for e in g.edges))
                N, O = sum(g.macs for _, g in M), sum(g.wires for _, g in M)
                b = mc.bound(c, k, n, N=N, O=O, size=work.gates, beta_w=bw, beta_x=bx,
                             b=min(mc.BF16, *(g.out_width for _, g in M)), F=mc.share(omega))
                if best is None or b.blocks > best.bound.blocks:
                    best = Priced(b, tuple(g for _, g in M), omega, N, O)
    return best
