"""PoUW as one pass over a served Program (decision 102, `PROTOCOL.md` §`compile`): `compile(program, scheme)`.

* **Find.** Every Call of a registered matmul family (`MATMULS`: each family's A and B ports, its (m, k, n) and the arm it is
  registered at) is one dot product, or N of them (`S8Matmul_v1`'s rows). A Call nested in a Definition, at any depth, is
  found through it when its operands are that Definition's parameters (`ScaledMmS8Coordinate_v1`'s `GemmAccumulatorS8_v1`).
  A Program whose root is the matmul itself is served as one Call of it (`PouwCall_v1{FN}`). A served Program with an edited
  arm, a scan over a matmul or a matmul step outside every registered family is refused.
* **Replace.** Each dot becomes one unit of the scheme (`Scheme.unit`): ``(a, b, noise) -> {z, receipt}``. A's row and B's row
  are its committed inputs, the noise is the Program's coin input `pouw.noise` (`In(kind="coin")` where ports.py is on the
  tree), the receipt is a sink (`Out(sink=True)`) that only the Program's outputs read, and z is the exact product, which feeds
  the Call's consumers in its place, so served values don't change. The units sit at the root, one batch per site (a root
  node's dots of one kind, `Site`), ahead of the node that read the Call's output; that node's Definition F becomes
  `PouwRest_v1{FN=F}`, F without its dots, reading their z (`pouw.z`). A rest of no gate is inlined.
* **Check** (`CompileError` with the rule's code): no registered Call or matmul step is left outside a unit (``uncovered``);
  each noise leaf reaches one unit's noise port once and nothing else (``noise-flow``, ``noise-reuse``; inside a unit,
  COIN-RULES' `E_COIN_FLOW` and `E_COIN_REUSE` where ports.py is on the tree); served values reach a unit only through its z
  (``separator``); every unit outside the units meets `cut.WIDTH_RULE` (``width``). `Compiled.check_gates` adds the gate-level
  cut (no value computed in two units), the separator and the width rule by reachability, for a Program small enough to
  flatten.

The Program is the verifier's, rebuilt from (served Program, scheme): `rebuild`, `Compiled.check_descriptor` (`traced_as` at
Program level). Seeds are the verifier's too: seed_j = SHA-512(`SEED_TAG` ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))
(`seed`), position the site's index; a unit's noise is `Scheme.noise(seed_j, dot, i, j)` for its A row i and B row j.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Callable, Iterator, Mapping, Sequence

from verity.ml.circuits.codec import descriptor_id, program_digest
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.program import REGISTRATION, Program, is_input
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Refs, Strided, concat, runs
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Type

from ..protocol import DomainError, tagged

try:
    from verity.ml.circuits import ports as _ports
except ImportError:  # port types (circuits PROTOCOL.md §12) are not on this tree
    _ports = None

NOISE = "pouw.noise"
Z = "pouw.z"
SEED_TAG = "verity/pouw/seed/v1"
#: a tensor-core k step (`ml.circuits.ops.prims`' ``…Dot<k>`` and their Boolean twins): a matmul wherever it is called
STEP = re.compile(r".*(Imma|Qmma|Omma|Qgmma|Wgmma|Tc|MmaV2)Dot[0-9]+_v[0-9]+")


class CompileError(ValueError):
    """A served Program, or a scheme, the pass refuses. `code` names the rule."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


# -- the registry ------------------------------------------------------------------------------------------------------------

def _any_arm(fn: Any) -> tuple[str, str] | None:
    return None


def _plain_arm(fn: Any) -> tuple[str, str] | None:
    s = fn.bindings
    if s["NOISE"] != "none":
        return "edited-arm", (f"{fn.id} is an edited arm (NOISE {s['NOISE']!r}): a served Program computes the plain arm, and "
                              f"only the scheme's units add noise")
    if s["T0"] != "zero" or s["COMMIT"] != "final" or s["TAIL"] or s["LIMBS"] != "none":
        return "arm", (f"{fn.id} is not the plain arm (T0 'zero', COMMIT 'final', no tail, LIMBS 'none'), so its Call is not "
                       f"a whole dot product")
    return None


@dataclass(frozen=True)
class Matmul:
    """A registered matmul family: its A and B ports, whether B is N rows that A meets each of (`rows`), and the arm it is
    registered at (`arm(fn)`: None, or the refusal's (code, message))."""

    family: str
    a: str
    b: str
    rows: bool = False
    arm: Callable[[Any], tuple[str, str] | None] = _any_arm

    def mkn(self, fn: Any) -> tuple[int, int, int]:
        """One Call's (m, k, n)."""
        return 1, fn.bindings["K"], fn.bindings["N"] if self.rows else 1


MATMULS: dict[str, Matmul] = {m.family: m for m in (
    Matmul("S8Dot_v1", "a", "b", arm=_plain_arm),
    Matmul("S8Dot_v2", "a", "b", arm=_plain_arm),
    Matmul("S8Matmul_v1", "a", "b", rows=True, arm=_plain_arm),
    Matmul("GemmAccumulatorS8_v1", "x", "wrow"),
    Matmul("GemmAccumulatorS8_v2", "x", "wrow"),
)}


@dataclass(frozen=True)
class Dot:
    """One dot product of a registered family: its k, one A row's and one B row's types, the exact product's, the Definition
    computing it (`exact`: the Call's own, or `S8Matmul_v1`'s coordinate), the family's k step and, for the s8 family, its
    operand intervals."""

    family: str
    k: int
    a: Type
    b: Type
    z: Type
    exact: Any
    step: Any
    a_bound: tuple[int, int] | None = None
    b_bound: tuple[int, int] | None = None


def _entry(fn: Any) -> Matmul | None:
    return MATMULS.get(fn.base_id) if isinstance(fn, SpecializedDefinition) else None


def _registered(fn: Any) -> Matmul | None:
    e = _entry(fn)
    if e is not None and (bad := e.arm(fn)) is not None:
        raise CompileError(*bad)
    return e


@lru_cache(maxsize=None)
def _dot(fn: Any) -> Dot:
    e = MATMULS[fn.base_id]
    params, ret = fn.signature()
    types = dict(params)
    b, z, exact = types[e.b], ret, fn
    if e.rows:
        b, z, exact = b.elem, ret.elem, fn.body.nodes[0].fn
    s = fn.bindings
    bounds = (tuple(s["A"]), tuple(s["B"])) if "A" in s else (None, None)
    return Dot(fn.base_id, s["K"], types[e.a], b, z, exact, s["DOT"], *bounds)


def _per_call(fn: Any) -> int:
    return fn.bindings["N"] if MATMULS[fn.base_id].rows else 1


# -- schemes -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scheme:
    """A PoUW scheme as the pass takes it. `unit(dot)`: the Definition of one dot's unit, ``(a: dot.a, b: dot.b, noise: N) ->
    Record(z: dot.z, receipt: R)``, holding the noised arm, the receipt fold and the spot check, its z the exact product.
    `noise(seed, dot, i, j)`: the unit's N leaves for A row i and B row j of a site, from the site's seed."""

    name: str
    unit: Callable[[Dot], Any]
    noise: Callable[[bytes, Dot, int, int], Sequence[int]]


#: the schemes the pass compiles to, by name (protocol structure, keyed by silicon target in the registration). Separate from
#: `circuit.SCHEMES`, whose names the vLLM integration serves as ncp-v2-style traced schemes.
SCHEMES: dict[str, Scheme] = {}


def register(scheme: Scheme) -> Scheme:
    prev = SCHEMES.get(scheme.name)
    if prev is not None and prev is not scheme:
        raise ValueError(f"a different scheme is registered as {scheme.name!r}")
    SCHEMES[scheme.name] = scheme
    return scheme


def seed(salt: bytes, forward: int, position: int, root_a: bytes, root_b: bytes) -> bytes:
    """seed_j for the call at (forward, position) whose operands commit to root_a and root_b."""
    return tagged(SEED_TAG, salt, forward, position, root_a, root_b)


# -- find --------------------------------------------------------------------------------------------------------------------

def _keys(r: Refs) -> Iterator[tuple[str, int]]:
    if isinstance(r, Concat):
        for p in r.parts:
            yield from _keys(p)
    elif isinstance(r, Explicit):
        for s, i, _leaf in r.items:
            yield s, i
    elif r.count:
        yield r.space, r.idx


def _members(nd: Any) -> int:
    return nd.n if nd.form == "batch" else 1


def _merge(out: list, rs: Sequence[tuple[Dot, int]]) -> list:
    for d, c in rs:
        if out and out[-1][0] == d:
            out[-1] = (d, out[-1][1] + c)
        else:
            out.append((d, c))
    return out


def _node_runs(owner: Any, k: int, nd: Any, nested: bool) -> tuple[tuple[Dot, int], ...]:
    """Node k of `owner`'s dots in z order, as runs (dot, count). In a nested Definition, every leaf a dot's operand reads is a
    parameter of it."""
    if isinstance(nd.fn, PrimitiveDefinition):
        return ()
    if _registered(nd.fn) is not None:
        if nested and any(s != "p" for a in nd.args for s, _i in _keys(a.refs)):
            raise CompileError("operand-inside", f"{owner.id} node {k}: {nd.fn.id}'s operands are computed inside {owner.id}, "
                                                 f"so its dots can't be units at the root")
        return ((_dot(nd.fn), _per_call(nd.fn) * _members(nd)),)
    inner = _runs(nd.fn)
    if not inner:
        return ()
    if nd.form == "scan":
        raise CompileError("scan", f"{owner.id} node {k}: a scan of {nd.fn.id} holds a matmul; its steps' dots aren't units yet")
    if nested and any(s != "p" for j in _dot_params(nd.fn) for s, _i in _keys(nd.args[j].refs)):
        raise CompileError("operand-inside", f"{owner.id} node {k}: a matmul in {nd.fn.id} reads a value computed inside "
                                             f"{owner.id}, so its dots can't be units at the root")
    return tuple(_merge([], list(inner) * _members(nd)))


@lru_cache(maxsize=None)
def _runs(G: Any) -> tuple[tuple[Dot, int], ...]:
    out: list = []
    for k, nd in enumerate(G.body.nodes):
        _merge(out, _node_runs(G, k, nd, True))
    return tuple(out)


@lru_cache(maxsize=None)
def _dot_params(G: Any) -> frozenset[int]:
    """G's parameters a dot inside G reads as an operand."""
    out: set[int] = set()
    for nd in G.body.nodes:
        if _entry(nd.fn) is not None:
            out.update(i for a in nd.args for s, i in _keys(a.refs) if s == "p")
        elif not isinstance(nd.fn, PrimitiveDefinition) and _runs(nd.fn):
            out.update(i for j in _dot_params(nd.fn) for s, i in _keys(nd.args[j].refs) if s == "p")
    return frozenset(out)


def _holds(nd: Any) -> bool:
    return not isinstance(nd.fn, PrimitiveDefinition) and _entry(nd.fn) is None and bool(_runs(nd.fn))


@lru_cache(maxsize=None)
def _zt(G: Any) -> TupleT:
    """`pouw.z`'s type in `PouwRest_v1{FN=G}`: per node of G holding dots, its z leaves (an Array over its dots, or its members)."""
    fields = []
    for nd in G.body.nodes:
        if _entry(nd.fn) is not None:
            fields.append(Array(_per_call(nd.fn) * _members(nd), _dot(nd.fn).z))
        elif _holds(nd):
            fields.append(Array(_members(nd), _zt(nd.fn)))
    return TupleT(*fields)


def _member(r: Refs, t: Type, ax: int | None, m: int) -> Refs:
    return r if ax is None else Coll(t, r).take(ax, m).refs


def _call_dots(owner: Any, k: int, fn: Any, args: Sequence[Refs | None]) -> Iterator[tuple[Dot, Refs, Refs]]:
    e, dot = MATMULS[fn.base_id], _dot(fn)
    names = [n for n, _t in fn.signature()[0]]
    a, b = args[names.index(e.a)], args[names.index(e.b)]
    if a is None or b is None:
        raise CompileError("operand-inside", f"{owner.id} node {k}: {fn.id}'s operands are computed inside a Definition")
    if not e.rows:
        yield dot, a, b
        return
    L = dot.b.leaves
    for j in range(_per_call(fn)):
        yield dot, a, b.slice(j * L, (j + 1) * L)


def _node_dots(owner: Any, k: int, nd: Any, args: Sequence[Refs | None]) -> Iterator[tuple[Dot, Refs, Refs]]:
    """Node k's dots in z order with their operands, `args` its arguments' refs in the caller's scope (None: not resolvable)."""
    if nd.form == "call":
        members = [list(args)]
    else:
        members = [[None if r is None else _member(r, a.type, ax, m) for r, a, ax in zip(args, nd.args, nd.axes)]
                   for m in range(nd.n)]
    for margs in members:
        if _entry(nd.fn) is not None:
            yield from _call_dots(owner, k, nd.fn, margs)
        else:
            yield from _dots_in(nd.fn, margs)


def _dots_in(G: Any, params: Sequence[Refs | None]) -> Iterator[tuple[Dot, Refs, Refs]]:
    env = {("p", i): r for i, r in enumerate(params) if r is not None}
    for k, nd in enumerate(G.body.nodes):
        if _entry(nd.fn) is None and not _holds(nd):
            continue
        args = [_compose(a.refs, env) if all(key in env for key in _keys(a.refs)) else None for a in nd.args]
        yield from _node_dots(G, k, nd, args)


# -- replace -----------------------------------------------------------------------------------------------------------------

def _compose(r: Refs, env: Mapping[tuple[str, int], Refs]) -> Refs:
    """`r`, whose references are into the collections `env` names, as references into what they map to."""
    if isinstance(r, Concat):
        return concat([_compose(p, env) for p in r.parts])
    if isinstance(r, Explicit):
        return concat([env[(s, i)].slice(j, j + 1) for s, i, j in r.items])
    if not r.count:
        return Explicit(())
    t = env[(r.space, r.idx)]
    if isinstance(r, Affine):
        return t.slice(r.base, r.base + r.count)
    if isinstance(t, Affine):
        return Strided(t.space, t.idx, t.base + r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
    parts = []
    for q in range(-(-r.count // r.inner)):
        lo, w = r.base + q * r.outer_stride, min(r.inner, r.count - q * r.inner)
        if r.inner_stride == 1:
            parts.append(t.slice(lo, lo + w))
        else:
            parts.extend(t.slice(lo + j * r.inner_stride, lo + j * r.inner_stride + 1) for j in range(w))
    return concat(parts)


def _pack(parts: Sequence[Refs]) -> Refs:
    """Equal-length parts (one a unit) as one sequence: a `Strided` when they are runs of one collection in progression."""
    p0 = parts[0]
    if isinstance(p0, Affine) and all(isinstance(p, Affine) and (p.space, p.idx, p.count) == (p0.space, p0.idx, p0.count)
                                      for p in parts):
        step = parts[1].base - p0.base if len(parts) > 1 else p0.count
        if all(p.base == p0.base + u * step for u, p in enumerate(parts)):
            if step == p0.count:
                return Affine(p0.space, p0.idx, p0.base, p0.count * len(parts))
            return Strided(p0.space, p0.idx, p0.base, p0.count, step, 1, p0.count * len(parts))
    return concat(list(parts))


@dataclass(frozen=True)
class _Rest:
    defn: Any            # `PouwRest_v1{FN}`, None when it has no gate (inlined)
    kept: tuple[int, ...]


@lru_cache(maxsize=None)
def _rest(G: Any) -> _Rest:
    used: set[int] = set()

    def note(r: Refs) -> None:
        used.update(i for s, i in _keys(r) if s == "p")

    gates = False
    for nd in G.body.nodes:
        if _entry(nd.fn) is not None:
            continue
        if _holds(nd):
            inner = _rest(nd.fn)
            for i in inner.kept:
                note(nd.args[i].refs)
            gates = gates or inner.defn is not None
        else:
            for a in nd.args:
                note(a.refs)
            gates = True
    note(G.body.ret)
    kept = tuple(sorted(used))
    if gates and Z in [n for n, _t in G.signature()[0]]:
        raise CompileError("name", f"{G.id} has a parameter named {Z!r}, the rest's z")
    return _Rest(bind(PouwRest, FN=G) if gates else None, kept)


class _ZParam:
    """The z of a rest's dots, taken in order from its `pouw.z`."""

    def __init__(self, z: Refs) -> None:
        self.z, self.at = z, 0

    def __call__(self, nd: Any, args: Sequence[Coll | None]) -> Refs:
        n = nd.out.leaves if _entry(nd.fn) is not None else _members(nd) * _zt(nd.fn).leaves
        r = self.z.slice(self.at, self.at + n)
        self.at += n
        return r


def _emit(B: Any, nd: Any, args: Sequence[Coll]) -> Coll:
    if nd.form in ("primitive", "call"):
        return B.call(nd.fn, *args, name=nd.name)
    if nd.form == "batch":
        return B.batch(nd.fn, *args, axes=nd.axes, n=nd.n, name=nd.name)
    return B.scan(nd.fn, args[0], xs=args[1:1 + nd.n_xs], shared=args[1 + nd.n_xs:], n=nd.n, name=nd.name)


def _emit_rest(B: Any, nd: Any, args: Sequence[Coll | None], z: Refs) -> Refs:
    G, R, zt = nd.fn, _rest(nd.fn), _zt(nd.fn)
    if nd.form == "call":
        if R.defn is None:
            return _replay(B, G, {("p", i): a.refs for i, a in enumerate(args) if a is not None}, _ZParam(z))
        return B.call(R.defn, *[args[i] for i in R.kept], Coll(zt, z), name=nd.name).refs
    if R.defn is None:
        L, parts = zt.leaves, []
        for m in range(nd.n):
            env = {("p", i): _member(a.refs, a.type, ax, m) for i, (a, ax) in enumerate(zip(args, nd.axes)) if a is not None}
            parts.append(_replay(B, G, env, _ZParam(z.slice(m * L, (m + 1) * L))))
        return concat(parts)
    return B.batch(R.defn, *[args[i] for i in R.kept], Coll(Array(nd.n, zt), z),
                   axes=tuple(nd.axes[i] for i in R.kept) + (0,), n=nd.n, name=nd.name).refs


def _replay(B: Any, F: Any, env: dict, zsrc: Callable[[Any, Sequence[Coll]], Refs]) -> Refs:
    """F's body in `B` with its parameters at `env`, every dot's output the z `zsrc` gives it; F's return's refs. A dot's or a
    holder's argument that reads a parameter `env` leaves out (one only the lifted dots read) is None."""
    for k, nd in enumerate(F.body.nodes):
        args = [Coll(a.type, _compose(a.refs, env)) if all(key in env for key in _keys(a.refs)) else None for a in nd.args]
        if _entry(nd.fn) is not None:
            env[("n", k)] = zsrc(nd, args)
        elif _holds(nd):
            env[("n", k)] = _emit_rest(B, nd, args, zsrc(nd, args))
        else:
            env[("n", k)] = _emit(B, nd, args).refs
    return _compose(F.body.ret, env)


def _rest_ports(S: Any) -> Any:
    """FN's ports on what the rest keeps (its kept parameters, its return)."""
    pt = _ports.port_types(S.FN)
    if pt is None:
        return None
    params = S.FN.signature()[0]
    ps = {params[i][0]: _ports.Leaves(tuple((lo, hi, p) for lo, hi, p, _w in pt.params[i]))
          for i in _rest(S.FN).kept if pt.params[i]}
    ret = _ports.Leaves(tuple((lo, hi, o) for lo, hi, o, _w in pt.ret)) if pt.ret else None
    return _ports.Ports(params=ps, ret=ret) if ps or ret is not None else None


def _rest_signature(S: Any) -> tuple:
    params, ret = S.FN.signature()
    return tuple(params[i] for i in _rest(S.FN).kept) + ((Z, _zt(S.FN)),), ret


def _rest_body(B: Any, S: Any, *args: Coll) -> Coll:
    F, kept = S.FN, _rest(S.FN).kept
    zsrc = _ZParam(args[-1].refs)
    out = _replay(B, F, {("p", i): a.refs for i, a in zip(kept, args)}, zsrc)
    assert zsrc.at == args[-1].refs.count
    return Coll(F.signature()[1], out)


def _composite(name: str, statics: tuple[str, ...], signature: Callable, body: Callable, ports: Callable | None,
               doc: str) -> CompositeDefinition:
    kw = {"ports": ports} if _ports is not None else {}
    return CompositeDefinition(name, 1, statics, signature, body, doc=doc, conformance="protocol-defined", register=False, **kw)


PouwRest = _composite(
    "PouwRest", ("FN",), _rest_signature, _rest_body, _rest_ports,
    "FN with its dots lifted out: FN's parameters that the rest reads, then `pouw.z`, the z of FN's dots in order (per "
    "node holding dots, an Array over its dots or members), to FN's return.")

PouwCall = _composite(
    "PouwCall", ("FN",), lambda S: S.FN.signature(), lambda B, S, *args: B.call(S.FN, *args), None,
    "One Call of FN, a registered matmul: the served function of a Program whose root is the matmul itself, so that its dots "
    "are a Call's.")


def _served_fn(fn: Any) -> Any:
    return bind(PouwCall, FN=fn) if _entry(fn) is not None else fn


@dataclass(frozen=True)
class _SitePlan:
    node: int
    dot: Dot
    count: int
    unit: Any
    noise: Type
    receipt: Type


def _unit_types(scheme: Scheme, dot: Dot, unit: Any) -> tuple[Type, Type]:
    params, ret = unit.signature()
    if [n for n, _t in params] != ["a", "b", "noise"] or params[0][1] != dot.a or params[1][1] != dot.b:
        raise CompileError("scheme", f"{scheme.name}'s unit {unit.id} takes {params}, not (a: {dot.a!r}, b: {dot.b!r}, noise)")
    if not (isinstance(ret, Record) and [n for n, _t in ret.fields] == ["z", "receipt"] and ret.fields[0][1] == dot.z
            and ret.fields[1][1].leaves):
        raise CompileError("scheme", f"{scheme.name}'s unit {unit.id} returns {ret!r}, not {{z: {dot.z!r}, receipt}}")
    return params[2][1], ret.fields[1][1]


@lru_cache(maxsize=16)
def _plan(served: Any, scheme_name: str) -> tuple[_SitePlan, ...]:
    """The sites, in order: per node of the served function holding dots, one per kind of dot, in the order first met."""
    scheme = SCHEMES[scheme_name]
    out = []
    for k, nd in enumerate(served.body.nodes):
        counts: dict[Dot, int] = {}
        for d, c in _node_runs(served, k, nd, False):
            counts[d] = counts.get(d, 0) + c
        for d, c in counts.items():
            unit = scheme.unit(d)
            out.append(_SitePlan(k, d, c, unit, *_unit_types(scheme, d, unit)))
    if not out:
        raise CompileError("no-matmul", f"{served.id} has no Call of a registered matmul family ({', '.join(MATMULS)})")
    return tuple(out)


class _Units:
    """The root's z source: each node's dots as unit batches, one per site, ahead of the node."""

    def __init__(self, B: Any, served: Any, sites: Sequence[_SitePlan], noise: Coll) -> None:
        self.B, self.served, self.sites, self.noise, self.at = B, served, sites, noise, 0
        self.receipts: list[Refs] = []

    def __call__(self, nd: Any, args: Sequence[Coll]) -> Refs:
        k = self.sites[self.at].node
        dots = list(_node_dots(self.served, k, nd, [a.refs for a in args]))
        by: dict[Dot, list] = {}
        for d, a, b in dots:
            by.setdefault(d, []).append((a, b))
        at: dict[Dot, tuple[int, int, int]] = {}
        for d, ab in by.items():
            sp = self.sites[self.at]
            assert sp.dot == d and sp.count == len(ab)
            a = Coll(Array(sp.count, d.a), _pack([x for x, _ in ab]))
            b = Coll(Array(sp.count, d.b), _pack([y for _, y in ab]))
            out = self.B.batch(sp.unit, a, b, self.noise[self.at], axes=(0, 0, 0), name=f"pouw[{self.at}]")
            k, lz, lr = out.refs.idx, d.z.leaves, sp.receipt.leaves
            self.receipts.append(Strided("n", k, lz, lr, lz + lr, 1, sp.count * lr))
            at[d] = (k, lz, lz + lr)
            self.at += 1
        parts, used = [], dict.fromkeys(by, 0)
        u = 0
        while u < len(dots):
            d, v = dots[u][0], u
            while v < len(dots) and dots[v][0] == d:
                v += 1
            k, lz, P = at[d]
            parts.append(Strided("n", k, used[d] * P, lz, P, 1, (v - u) * lz))
            used[d] += v - u
            u = v
        return concat(parts)


def _compiled_signature(S: Any) -> tuple:
    sites = _plan(S.SERVED, S.SCHEME)
    params, ret = S.SERVED.signature()
    return (tuple(params) + ((NOISE, TupleT(*[Array(s.count, s.noise) for s in sites])),),
            Record(("served", ret), ("receipts", TupleT(*[Array(s.count, s.receipt) for s in sites]))))


def _compiled_body(B: Any, S: Any, *args: Coll) -> Coll:
    sites = _plan(S.SERVED, S.SCHEME)
    units = _Units(B, S.SERVED, sites, args[-1])
    out = _replay(B, S.SERVED, {("p", i): a.refs for i, a in enumerate(args[:-1])}, units)
    assert units.at == len(sites)
    return Coll(_compiled_signature(S)[1], concat([out] + units.receipts))


def _compiled_ports(S: Any) -> Any:
    return _ports.Ports(params={NOISE: _ports.In(kind="coin")}, ret={"receipts": _ports.Out(sink=True)})


PouwCompiled = _composite(
    "PouwCompiled", ("SERVED", "SCHEME"), _compiled_signature, _compiled_body, _compiled_ports,
    "SERVED with every registered matmul's dots as SCHEME's units: SERVED's parameters and `pouw.noise` (a coin, per site an "
    "Array over its units) to {served: SERVED's return, receipts: per site an Array over its units' receipts (sinks)}.")


# -- the compiled Program ----------------------------------------------------------------------------------------------------

def _row_key(r: Refs) -> tuple:
    return tuple(runs(r))


@dataclass(frozen=True)
class Site:
    """One batch of units: a node of the served function and a kind of dot. `a_rows`, `b_rows`: A's and B's distinct rows, as
    references into the served function's body (parameters ``p``, nodes ``n``), first use first; unit u multiplies A row
    ``pairs[u][0]`` by B row ``pairs[u][1]``. Its position (the seed's) is its index in `Compiled.sites`."""

    position: int
    node: int
    dot: Dot
    unit: Any
    noise: Type
    receipt: Type
    a_rows: tuple[Refs, ...]
    b_rows: tuple[Refs, ...]
    pairs: tuple[tuple[int, int], ...]

    @property
    def count(self) -> int:
        return len(self.pairs)

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.a_rows), self.dot.k, len(self.b_rows)


def _sites(served: Any, plan: Sequence[_SitePlan]) -> tuple[Site, ...]:
    out: list[Site] = []
    for k, nd in enumerate(served.body.nodes):
        here = [sp for sp in plan if sp.node == k]
        if not here:
            continue
        dots = list(_node_dots(served, k, nd, [a.refs for a in nd.args]))
        for sp in here:
            rows: tuple[dict, dict] = ({}, {})
            pairs = []
            for d, a, b in dots:
                if d == sp.dot:
                    pairs.append(tuple(rows[s].setdefault(_row_key(r), (len(rows[s]), r))[0] for s, r in ((0, a), (1, b))))
            out.append(Site(len(out), k, sp.dot, sp.unit, sp.noise, sp.receipt,
                            tuple(r for _i, r in rows[0].values()), tuple(r for _i, r in rows[1].values()), tuple(pairs)))
    return tuple(out)


@dataclass(frozen=True)
class GateCheck:
    """`Compiled.check_gates`: the gate-level cut (`cut.check_cut`, no width rule: every gate certified once, reads across units
    only of committed values, no value computed in two units), the separator (no unit's gate reached backward from a served
    output but through its z) and the width rule on every unit a served output is reached from without crossing a z."""

    cut: Any
    reached: int                              # unit gates reached from the served outputs but through a z
    z_gates: int                              # the z gates that trace stops at
    z_bits: int
    receipts_owned: bool                      # every receipt output is a unit's
    too_wide: tuple[tuple[int, int, int], ...]

    @property
    def ok(self) -> bool:
        return self.cut.ok and not self.reached and self.receipts_owned and not self.too_wide


def _ret_gate(fn: Any, j: int) -> int | None:
    """The gate of `fn` (fn-relative) that computes return leaf j, None when it is a parameter's leaf."""
    space, k, leaf = fn.body.ret[j]
    while space == "n":
        nd, base = fn.body.nodes[k], fn.body.offsets[k]
        if nd.form == "primitive":
            return base
        if nd.form == "scan":
            raise CompileError("scan", f"{fn.id}: a unit's z comes out of a scan")
        m, r = divmod(leaf, nd.fn.out_leaves) if nd.form == "batch" else (0, leaf)
        g = _ret_gate(nd.fn, r)
        return None if g is None else base + m * nd.fn.gates + g
    return None


@dataclass(frozen=True, eq=False)
class Compiled:
    program: Program
    served: Program
    scheme: Scheme
    sites: tuple[Site, ...]

    @cached_property
    def digest(self) -> str:
        return program_digest(self.program)

    @property
    def served_leaves(self) -> int:
        """The compiled Program's first outputs, the served ones."""
        return self.served.fn.signature()[1].leaves

    def query(self) -> dict:
        """`Q_template_instances` over the root's templates, first use first: the units', the rests' and the served ones'."""
        from verity.ml.circuits.partition_object import template_instances_query

        seen: dict[str, Any] = {}
        for nd in self.program.root.body.nodes:
            if not is_input(nd.fn):
                seen.setdefault(descriptor_id(nd.fn), nd.fn)
        return template_instances_query(list(seen.values()))

    def seeds(self, salt: bytes, forward: int, roots: Sequence[tuple[bytes, bytes]]) -> list[bytes]:
        """Each site's seed from its operand roots (root(A_j), root(B_j)), `roots[position]`."""
        if len(roots) != len(self.sites):
            raise ValueError(f"one (root_a, root_b) per site: {len(self.sites)}, not {len(roots)}")
        return [seed(salt, forward, s.position, *roots[s.position]) for s in self.sites]

    def unit_noise(self, seed_j: bytes, site: Site, u: int) -> list[int]:
        """Unit u of `site`'s noise leaves."""
        i, j = site.pairs[u]
        vals = list(self.scheme.noise(seed_j, site.dot, i, j))
        if len(vals) != site.noise.leaves:
            raise ValueError(f"{self.scheme.name}'s noise gives {len(vals)} leaves, its unit takes {site.noise.leaves}")
        return vals

    def noise_input(self, salt: bytes, forward: int, roots: Sequence[tuple[bytes, bytes]]) -> list[int]:
        """`pouw.noise`'s leaves, in order: every unit's, site by site."""
        out: list[int] = []
        for s, sd in zip(self.sites, self.seeds(salt, forward, roots)):
            for u in range(s.count):
                out += self.unit_noise(sd, s, u)
        return out

    def check_descriptor(self, descriptor: dict) -> None:
        """Refuse a descriptor that is not the verifier's own compiled Program."""
        try:
            ok = program_digest(descriptor) == self.digest
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise DomainError(f"{self.scheme.name}: the descriptor is not the verifier's compiled Program of "
                              f"{self.served.fn.id}")

    def check_gates(self) -> GateCheck:
        """The gate-level checks, on the flattened Program: for one small enough to flatten (a test's, a reduced layer)."""
        from verity.ml.circuits.cut import CallGraph, boundary_widths, check_cut, fits

        fn = self.program.fn
        body = fn.body
        g = CallGraph(fn)
        first, at = [], 0
        for nd in body.nodes:
            first.append(at)
            at += _members(nd)
        owner = []
        for c in range(g.n):
            j = g.keep[c]
            k = bisect.bisect_right(body.offsets, j) - 1
            nd = body.nodes[k]
            owner.append(first[k] + ((j - body.offsets[k]) // nd.fn.gates if nd.form == "batch" else 0))
        cut = check_cut(g, owner, limits=None)
        index = {j: c for c, j in enumerate(g.keep)}
        units = {id(s.unit) for s in self.sites}
        pouw: set[int] = set()
        ports: set[int] = set()
        for k, nd in enumerate(body.nodes):
            if id(nd.fn) not in units:
                continue
            lz = nd.fn.signature()[1].fields[0][1].leaves
            local = [_ret_gate(nd.fn, j) for j in range(lz)]
            if None in local:
                raise CompileError("separator", f"{nd.fn.id}: its z returns a parameter's leaf")
            for m in range(nd.n):
                pouw.add(first[k] + m)
                base = body.offsets[k] + m * nd.fn.gates
                ports.update(index[base + j] for j in local if base + j in index)
        receipts = sum(s.count * s.receipt.leaves for s in self.sites)
        outs = list(g.outputs)
        served = outs[:len(outs) - receipts]
        preds: list[list[int]] = [[] for _ in range(g.n)]
        for p, q in zip(g.src, g.dst):
            preds[q].append(p)
        seen, stopped, reached = set(served), {q for q in served if q in ports}, 0
        stack = [q for q in served if q not in ports]
        while stack:
            q = stack.pop()
            if owner[q] in pouw:
                reached += 1
                continue
            for p in preds[q]:
                if p in seen:
                    continue
                seen.add(p)
                if p in ports:
                    stopped.add(p)
                else:
                    stack.append(p)
        ob, og = boundary_widths(g, owner)
        hit = sorted({owner[q] for q in seen if q not in stopped})
        wide = tuple((u, int(ob[u]), int(og[u])) for u in hit if u not in pouw and not fits(ob[u], og[u]))
        return GateCheck(cut, reached, len(stopped), sum(int(g.width[p]) for p in stopped),
                         all(owner[q] in pouw for q in outs[len(outs) - receipts:]), wide)


# -- checks ------------------------------------------------------------------------------------------------------------------

def _reads_coin(served: Program) -> str | None:
    """Why the served Program reads a coin, or None."""
    if _ports is not None:
        pt = _ports.port_types(served.fn)
        if pt is not None and any(p.kind == "coin" for rs in pt.params for _lo, _hi, p, _w in rs):
            return f"{served.fn.id} takes a coin parameter"
    seen, todo = set(), [served.fn]
    while todo:
        fn = todo.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        if isinstance(fn, SpecializedDefinition):
            if fn.defn is PouwCompiled:
                return f"{fn.id} is already compiled: it reads {NOISE}"
            todo.extend(nd.fn for nd in fn.body.nodes)
        elif _ports is not None and (pt := _ports.port_types(fn)) is not None and any(
                p.kind == "coin" for rs in pt.params for _lo, _hi, p, _w in rs):
            return f"it calls {fn.id}, which reads a coin"
    return None


def _uncovered(fn: Any, units: set[int]) -> list[str]:
    bad, seen, todo = [], set(), [fn]
    while todo:
        f = todo.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        for k, nd in enumerate(f.body.nodes):
            if id(nd.fn) in units:
                continue
            if _entry(nd.fn) is not None or STEP.fullmatch(nd.fn.id):
                bad.append(f"{f.id} node {k}: {nd.fn.id}")
            elif isinstance(nd.fn, SpecializedDefinition):
                todo.append(nd.fn)
    return bad


def _check(c: Compiled) -> None:
    from verity.ml.circuits.cut import fits

    fn = c.program.fn
    body = fn.body
    units = {id(s.unit) for s in c.sites}
    bad = _uncovered(fn, units)
    if bad:
        raise CompileError("uncovered", f"a matmul is left outside every unit: {'; '.join(bad[:4])}")
    p_noise = len(body.params) - 1
    period: dict[int, tuple[int, int]] = {}
    for k, nd in enumerate(body.nodes):
        if id(nd.fn) in units:
            ret = nd.fn.signature()[1]
            period[k] = (ret.fields[0][1].leaves, ret.leaves)
    noise: list[tuple[int, int]] = []
    readers: list[tuple[str, Refs, bool]] = []
    for k, nd in enumerate(body.nodes):
        for j, a in enumerate(nd.args):
            readers.append((f"node {k} ({nd.fn.id}) argument {j}", a.refs, k in period and j == 2))
    S = c.served_leaves
    readers.append(("the served outputs", body.ret.slice(0, S), False))
    for what, r, port in readers:
        for s, i, lo, hi in runs(r):
            if s == "p" and i == p_noise:
                if not port:
                    raise CompileError("noise-flow", f"{what} reads {NOISE}: only a unit's noise port does")
                noise.append((lo, hi))
            elif s == "n" and i in period:
                lz, P = period[i]
                if lo % P + (hi - lo) > lz:
                    raise CompileError("separator", f"{what} reads node {i}'s leaves [{lo}, {hi}), past a unit's z")
    for s, i, lo, hi in runs(body.ret.slice(S, body.ret.count)):
        if s != "n" or i not in period or lo % period[i][1] < period[i][0]:
            raise CompileError("separator", f"the receipts read {s}{i} [{lo}, {hi}), not a unit's receipt")
    noise.sort()
    total = body.params[p_noise][1].leaves
    if any(b[0] < a[1] for a, b in zip(noise, noise[1:])):
        raise CompileError("noise-reuse", f"a leaf of {NOISE} is read by two noise ports")
    if sum(hi - lo for lo, hi in noise) != total:
        raise CompileError("noise-flow", f"{NOISE} has {total} leaves; the units read {sum(hi - lo for lo, hi in noise)}")
    read = {k: bytearray(nd.out.leaves) for k, nd in enumerate(body.nodes) if k not in period}
    for r in [a.refs for nd in body.nodes for a in nd.args] + [body.ret]:
        for s, i, lo, hi in runs(r):
            if s == "n" and i in read:
                read[i][lo:hi] = b"\x01" * (hi - lo)
    for k, flags in read.items():
        nd = body.nodes[k]
        n = _members(nd)
        widths = nd.out.leaf_widths() if nd.out.homogeneous_width() is None else None
        w = nd.out.homogeneous_width()
        L = nd.out.leaves // n
        for m in range(n):
            lo = m * L
            og = sum(flags[lo:lo + L])
            ob = og * w if widths is None else sum(widths[j] for j in range(lo, lo + L) if flags[j])
            if not fits(ob, og):
                raise CompileError("width", f"node {k} ({nd.fn.id}) member {m} commits {ob} bits in {og} values, past "
                                            f"the width rule")


# -- the pass ----------------------------------------------------------------------------------------------------------------

def compile(program: Program, scheme: str) -> Compiled:
    """`program` with every registered matmul's dots as `scheme`'s units (the module docstring), checked."""
    if scheme not in SCHEMES:
        raise CompileError("scheme", f"no scheme {scheme!r} is registered ({', '.join(SCHEMES) or 'none'})")
    why = _reads_coin(program)
    if why is not None:
        raise CompileError("coin-read", f"the served Program reads a coin ({why}); noise enters only through compile")
    names = [n for n, _t in program.fn.signature()[0]]
    if NOISE in names:
        raise CompileError("name", f"{program.fn.id} has a parameter named {NOISE!r}")
    registered = tuple(n for n, b in zip(names, program.bindings) if b == REGISTRATION)
    served = _served_fn(program.fn)
    plan = _plan(served, scheme)
    fn = bind(PouwCompiled, SERVED=served, SCHEME=scheme)
    try:
        compiled = Program(fn, registered=registered)
    except ValueError as e:
        if _ports is not None and isinstance(e, _ports.PortError):
            raise CompileError(e.code, str(e).removeprefix(f"{e.code}: ")) from e
        raise
    c = Compiled(compiled, program, SCHEMES[scheme], _sites(served, plan))
    _check(c)
    return c


def rebuild(served: Program, scheme: str, descriptor: dict) -> Compiled:
    """The verifier's compiled Program, from (served, scheme), held to an offered descriptor."""
    c = compile(served, scheme)
    c.check_descriptor(descriptor)
    return c


__all__ = ["CompileError", "Compiled", "Dot", "GateCheck", "MATMULS", "Matmul", "NOISE", "PouwCall", "PouwCompiled", "PouwRest", "SCHEMES",
           "SEED_TAG", "STEP", "Scheme", "Site", "Z", "compile", "rebuild", "register", "seed"]
