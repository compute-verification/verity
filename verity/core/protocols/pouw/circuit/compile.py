"""PoUW as one pass over a served Program (`PROTOCOL.md` §`compile`):
`compile(program, scheme, groups)`.

* **Find.** Every Call of a registered matmul family (`MATMULS`: each family's A and B ports, its (m, k, n) and the arm it is
  registered at) is one dot product, or N of them (`S8Matmul_v1`'s rows). Attention's QKᵀ is `DotBf16`'s, a kind like any
  other. A Call nested in a Definition, at any depth, is found through it when its operands are that
  Definition's parameters (`ScaledMmS8Coordinate_v1`'s `GemmAccumulatorS8_v1`). A Program whose root is the matmul itself is
  served as one Call of it (`PouwCall_v1{FN}`). A served Program with an edited arm, a scan over a matmul or a matmul step
  outside every registered family (P·V as `AttnBlock_v3` serves it, tensor-core steps on a P computed in the block) is
  refused.
* **Tile.** A site is a root node's dots of one kind; its distinct A rows and B rows, first use first, are its operands. Its
  A rows go in blocks of the scheme's r, each block's B rows (first use within the block) in chunks of c, and a tile is the
  site's dots inside one block and chunk (`Tile`: its rows, columns and cells, in z order). A grid site's tiles are full r × c
  grids; a site whose dots pair rows one to one (a population of coordinates) has diagonal ones.
* **Replace.** Each tile is one unit of the scheme, its dot kind's (`Scheme.unit(dot, tile)`: per kind a factory, which
  carries that kind's noise form; `Scheme.plug` sets one), self-contained: ``(a: its A rows, b: its B rows,
  salt, position, roots) -> {z: its cells' exact products, receipt}``. The unit derives its seed in its own proof from the
  salt, its position and its seed group's roots (`seeds.PouwSeed`, or `seeds.PouwUnitSeeds` for a unit whose
  receipt folds take keys: seed_j and σ_R), and its noise from the seed; nothing else reads the salt. The
  salt `pouw.salt` is the Program's one coin (`In(kind="coin")` where ports.py is on the tree); unit u reads position slot
  u (`Compiled.positions`) and its seed group's (root(A), root(B)). The three seed inputs are registered values the
  one-stage verifier fills itself, so the prover chooses none of them (`sampled_proofs.one_stage.registered`'s
  `supplied_rows` and `supplied_root`): `pouw.roots` from the roots the record names for each group's operands (a served
  port's, an AROW row table's included, and the registered weights', which the verifier holds to the root it pinned
  before the salt, `one_stage.registration.check`'s `pinned`), `pouw.positions` after every unit of the log's earlier records
  (`registered.log_base`, so no two records of a log share a position), and `pouw.salt` per window, filled by the
  verifier. R7 refuses a record that registers any other commitment of them (`R7-supplied`); `Compiled.inputs` gives the
  leaves the verifier fills. A seed group is the calls committed together (`groups`; by default a site, one
  call: a batched call over a layer's heads and the batch is one site): one commitment fixes the noise of
  every unit it covers, so the units a group covers are the grinding premium's divisor, at least the
  scheme's `cover`, and its dots of each attention kind carry a grind term of at most 0.1% (`grind`'s premium
  over the group's rows, each row's noise bits from the kind's `RowNoise` and its readers among the group's dots). A group
  never spans calls run one after another. The receipt is a sink (`Out(sink=True)`) only the Program's
  outputs read, and z feeds the Call's consumers in place of the Call's output, so served values don't change. The units sit
  at the root, one batch per site and tile shape, ahead of the node that read the Call's output; that node's Definition F
  becomes `PouwRest_v1{FN=F}`, F without its dots, reading their z (`pouw.z`). A rest of no gate is inlined.
* **Check** (`CompileError` with the rule's code): every seed group covers at least the scheme's `cover` units, and its
  dots of each kind with a `RowNoise` a grind term of at most `grind.FLAG` at `grind.T_BITS` (``cover``: past it, decode at
  a small batch or a short prompt, the group's noise is a live window's per-round coin, not a seed's); no seed group
  holds two calls one of which reads the other's output (``span``);
  no registered Call or matmul step is left outside a unit (``uncovered``);
  the salt reaches only the seed, through the units' salt port (``salt-flow``); each unit's seed reads the unit's own position
  and roots parameters whole, and at the root each unit batch reads its own slots of `pouw.positions` and its site's of
  `pouw.roots` (its group's), so no unit chooses its position and no two units share one (``position``); a unit derives one seed
  (``seed``) and its z reads none of it (``noise-in-z``), or under a one-stream scheme (`Scheme.stream`) each z leaf is
  recombined from words its receipt folds (``stream``: z reads the seed through the recombination, the wrap correction and
  the dither peel); served values reach a unit only through its z (``separator``); every node outside the units meets
  `cut.WIDTH_RULE` (``width``). `Compiled.check_gates` adds the gate-level cut (no value
  computed in two units), the separator and the width rule by reachability, for a Program small enough to flatten.

The Program is the verifier's, rebuilt from (served Program, scheme, groups): `rebuild`, `Compiled.check_descriptor`
(`traced_as` at Program level). Each unit's noise-forced work, the floor compute attribution credits to the block it is
placed at, is `Compiled.floor`.
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Any, Callable, Iterator, Mapping, Sequence

from verity.ml.circuits.codec import descriptor_id, program_digest
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.program import REGISTRATION, Program, is_input
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Refs, Strided, concat, runs
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Type

from .. import grind as GR
from ..protocol import DomainError
from . import seeds as SD

try:
    from verity.ml.circuits import ports as _ports
except ImportError:  # port types (circuits PROTOCOL.md §12) are not on this tree
    _ports = None

SALT = "pouw.salt"
POSITIONS = "pouw.positions"
ROOTS = "pouw.roots"
Z = "pouw.z"
#: a tensor-core k step (`ml.circuits.ops.prims`' ``…Dot<k>`` and their Boolean twins): a matmul wherever it is called
STEP = re.compile(r".*(Imma|Qmma|Omma|Qgmma|Wgmma|Tc|MmaV2)Dot[0-9]+_v[0-9]+")
seed = SD.seed


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
    Matmul("DotBf16_v1", "x", "w"),
    Matmul("DotBf16_v2", "x", "w"),
    Matmul("DotBf16_v3", "x", "w"),
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

    @property
    def boolean(self) -> bool:
        """Whether its operands are bits (a Boolean family's), so its unit's seed and noise are too."""
        return self.a.homogeneous_width() == 1


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
    return Dot(fn.base_id, s["K"], types[e.a], b, z, exact, s.get("DOT"), *bounds)


def _per_call(fn: Any) -> int:
    return fn.bindings["N"] if MATMULS[fn.base_id].rows else 1


# -- schemes -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    """One unit's share of a site: `r` A rows and `c` B rows, and its cells, the site's dots among them in z order, each
    (row, column) into the tile's rows and columns."""

    r: int
    c: int
    pairs: tuple[tuple[int, int], ...]

    @property
    def grid(self) -> bool:
        """Whether its cells are the whole r × c grid, once each."""
        return len(self.pairs) == self.r * self.c == len(set(self.pairs))


ANY = "*"


@dataclass(frozen=True)
class RowNoise:
    """A kind's noise form as its seed group's grind term reads it: the noise bits one A row and one B row
    carry, each ``(fixed, per entry)``, so ``fixed + per_entry·k`` at contraction k."""

    a: tuple[int, int]
    b: tuple[int, int]

    def bits(self, k: int) -> tuple[int, int]:
        return self.a[0] + self.a[1] * k, self.b[0] + self.b[1] * k


#: The attention forms H1 is to carry (`benchmarks/pouw/theory/attn_noise.py`'s `h_table`): QKᵀ's Q row carries
#: its c (a bit an entry) and its U (320 bits), its K row its b (a bit an entry); P·V's P row its U (224 bits), its V column
#: its b. QKᵀ is `DotBf16`'s; P·V's kind is the one a plugged unit takes (`Scheme.plug(..., rows=PV_ROWS)`). H1's scheme has no
#: unit for either yet, so its `rows` are read only once one is plugged: a `DotBf16` dot is refused (`scheme`) first.
QK_ROWS = RowNoise(a=(320, 1), b=(0, 1))
PV_ROWS = RowNoise(a=(224, 0), b=(0, 1))
ROWS = {f"DotBf16_v{v}": QK_ROWS for v in (1, 2, 3)}


@dataclass(frozen=True)
class Scheme:
    """A PoUW scheme as the pass takes it: its tile size (r, c), a scheme parameter; per kind of dot (a family's id, or `ANY`
    for every kind without its own), the factory of one tile's unit, which carries that kind's noise form (attention's
    QKᵀ and P·V are kinds like any other, and `plug` sets a kind's form); `cover`, the fewest units one seed
    group may cover (the grinding premium is a constant number of units per seed group, so GRIND/BUDGET sets
    the least coverage); and `rows`, per kind whose noise a matched row makes visible (attention's, `ROWS`), its
    `RowNoise`. A seed group's dots of such a kind carry GRIND's grind term: `grind.visible_premium` over the group's
    distinct A rows and B rows, each row's noise bits from the form and its readers the group's dots that read it, over
    those dots, at `grind.T_BITS`; over `grind.FLAG` (0.1%) the group is refused. At decode that is every
    (query head, request) pair a call holds, whatever KV heads they share (8B: batch ≥ 13 for QKᵀ); at prefill, the
    (KV head, request) components, fewer the longer the prompt. `noise`, per kind outside `rows`, its `RowNoise` too, which
    the grind term doesn't read and only the work floor does (`Compiled.floor`; H1's s8 kinds, `h1tile.H1_ROWS`): a kind's
    dots are in the floor where its form (`form`) puts noise on both its rows at its k.

    A unit, ``units[kind](dot, tile)``, is ``(a: Array(tile.r, dot.a), b: Array(tile.c, dot.b), salt, position, roots:
    Array(2, digest)) -> Record(z: Array(len(tile.pairs), dot.z), receipt: R)``, the digest and position types
    `seeds.types(dot.boolean)`'s. It calls one seed once on its salt, position and roots: `seeds.PouwSeed`
    (`seeds.seed_call`), or `seeds.PouwUnitSeeds` (`seeds.unit_seeds_call`) where its receipt folds take keys, which come
    from σ_R, never from seed_j. It derives its noise from seed_j (`seeds.noise_blocks`), holds the noised
    arm, the receipt fold and the spot check, and returns its cells' exact products as z. A factory may raise
    `CompileError` ("scheme") for a dot or tile it doesn't take.

    `stream`, for a one-stream scheme (`h1tile.SCHEME_V3`): the Definitions that make a cell's z from the words its receipt
    folds (`h1tile.STREAM`: the recombinations, the wrap correction and the dither peel). Such a unit's z is made from the
    noised words, so it reads the seed by design, and the unit check's ``noise-in-z`` rule gives way to ``stream``: every
    z leaf is the output of a call of one of them, and every leaf such a call reads is either another such call's output or
    read by the receipt's own cone. That z is the Call's value is the stream Definitions' arithmetic, not this check's."""

    name: str
    tile: tuple[int, int]
    units: Mapping[str, Callable[[Dot, Tile], Any]]
    stream: tuple[Any, ...] = field(default=(), kw_only=True)
    cover: int = 1
    rows: Mapping[str, RowNoise] = field(default_factory=lambda: dict(ROWS))
    noise: Mapping[str, RowNoise] = field(default_factory=dict)

    def form(self, kind: str) -> RowNoise | None:
        """kind's noise form, `rows`' or `noise`'s; None where the scheme declares none."""
        return self.rows.get(kind, self.noise.get(kind))

    def unit(self, dot: Dot, tile: Tile) -> Any:
        f = self.units.get(dot.family, self.units.get(ANY))
        if f is None:
            raise CompileError("scheme", f"{self.name} has no unit (noise form) for kind {dot.family}: "
                                         f"it covers {', '.join(sorted(self.units))}")
        return f(dot, tile)

    def plug(self, kind: str, unit: Callable[[Dot, Tile], Any], name: str, rows: RowNoise | None = None) -> "Scheme":
        """This scheme with `unit` as kind's factory (its noise form), under its own name (register it); `rows`, the form's
        `RowNoise` (`PV_ROWS` for P·V), where a matched row is visible."""
        return Scheme(name, self.tile, {**self.units, kind: unit}, self.cover,
                      {**self.rows, **({kind: rows} if rows is not None else {})}, self.noise)


#: the schemes the pass compiles to, by name (protocol structure, keyed by silicon target in the registration). Separate from
#: `circuit.SCHEMES`, whose names the vLLM integration serves as ncp-v2-style traced schemes.
SCHEMES: dict[str, Scheme] = {}
#: this package's modules that register its own schemes (`h1tile`: eps1-sm120-h1fx1), imported at the first name not yet in
#: `SCHEMES`
BUILTIN = ("h1tile",)


def _scheme(name: str) -> Scheme:
    if name not in SCHEMES:
        import importlib

        for m in BUILTIN:
            importlib.import_module(f"{__package__}.{m}")
    if name not in SCHEMES:
        raise CompileError("scheme", f"no scheme {name!r} is registered ({', '.join(SCHEMES) or 'none'})")
    return SCHEMES[name]


def register(scheme: Scheme) -> Scheme:
    prev = SCHEMES.get(scheme.name)
    if prev is not None and prev is not scheme:
        raise ValueError(f"a different scheme is registered as {scheme.name!r}")
    r, c = scheme.tile
    if min(r, c) < 1:
        raise ValueError(f"{scheme.name}: a tile is at least 1 × 1, not {r} × {c}")
    if scheme.cover < 1 or not scheme.units:
        raise ValueError(f"{scheme.name}: a seed group covers at least one unit, and a scheme has a unit for some kind")
    for kind, form in scheme.rows.items():
        if min(*form.a, *form.b) < 0 or min(form.bits(1)) < 1:
            raise ValueError(f"{scheme.name}: {kind}'s rows carry noise on both sides (at least a bit each, {form}): a form "
                             f"with one side clean has no grind bound")
    for kind, form in scheme.noise.items():
        if kind in scheme.rows or min(*form.a, *form.b) < 0:
            raise ValueError(f"{scheme.name}: {kind}'s noise form is in `rows` or in `noise`, not both, and no side carries "
                             f"negative bits ({form})")
    SCHEMES[scheme.name] = scheme
    return scheme


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


# -- the tiling --------------------------------------------------------------------------------------------------------------

def _row_key(r: Refs) -> tuple:
    return tuple(runs(r))


def _rows(dots: Sequence[tuple[Refs, Refs]]) -> tuple[tuple[Refs, ...], tuple[Refs, ...], tuple[tuple[int, int], ...]]:
    """A site's distinct A rows and B rows, first use first, and each dot's (A row, B row)."""
    rows: tuple[dict, dict] = ({}, {})
    pairs = []
    for a, b in dots:
        pairs.append(tuple(rows[s].setdefault(_row_key(r), (len(rows[s]), r))[0] for s, r in ((0, a), (1, b))))
    return tuple(r for _i, r in rows[0].values()), tuple(r for _i, r in rows[1].values()), tuple(pairs)


@dataclass(frozen=True)
class _Tiled:
    rows: tuple[int, ...]       # the site's A rows the tile multiplies
    cols: tuple[int, ...]       # its B rows
    tile: Tile


def _tiles(pairs: Sequence[tuple[int, int]], r: int, c: int) -> tuple[tuple[_Tiled, ...], tuple[tuple[int, int], ...]]:
    """The site's tiles in order (A row block, then B row chunk), and each dot's (tile, cell)."""
    blocks: dict[int, list[int]] = {}
    for d, (i, _j) in enumerate(pairs):
        blocks.setdefault(i // r, []).append(d)
    units: list[_Tiled] = []
    cells: list[tuple[int, int]] = [(0, 0)] * len(pairs)
    for blk in sorted(blocks):
        mine = blocks[blk]
        order = list(dict.fromkeys(pairs[d][1] for d in mine))
        chunk = {j: q // c for q, j in enumerate(order)}
        by: dict[int, list[int]] = {}
        for d in mine:
            by.setdefault(chunk[pairs[d][1]], []).append(d)
        for q in sorted(by):
            ds = by[q]
            rows = tuple(sorted({pairs[d][0] for d in ds}))
            cols = tuple(order[q * c:(q + 1) * c])
            ri, ci = {v: n for n, v in enumerate(rows)}, {v: n for n, v in enumerate(cols)}
            for cell, d in enumerate(ds):
                cells[d] = (len(units), cell)
            units.append(_Tiled(rows, cols, Tile(len(rows), len(cols), tuple((ri[pairs[d][0]], ci[pairs[d][1]]) for d in ds))))
    return tuple(units), tuple(cells)


@dataclass(frozen=True)
class _SitePlan:
    node: int
    dot: Dot
    a_rows: tuple[Refs, ...]
    b_rows: tuple[Refs, ...]
    pairs: tuple[tuple[int, int], ...]
    units: tuple[_Tiled, ...]
    cells: tuple[tuple[int, int], ...]      # each dot's (unit of the site, cell)


@dataclass(frozen=True)
class _Group:
    """One unit batch at the root: a site's units of one tile shape."""

    site: int
    tile: Tile
    unit: Any
    receipt: Type
    members: tuple[int, ...]                # the site's units, in order
    first: int                              # the first's position slot


@dataclass(frozen=True)
class _Plan:
    boolean: bool
    sites: tuple[_SitePlan, ...]
    groups: tuple[_Group, ...]
    nodes: tuple[int, ...]                  # the served function's nodes holding dots

    @property
    def units(self) -> int:
        return sum(len(g.members) for g in self.groups)


def _unit_receipt(scheme: Scheme, dot: Dot, tile: Tile, unit: Any) -> Type:
    D, P = SD.types(dot.boolean)
    want = (("a", Array(tile.r, dot.a)), ("b", Array(tile.c, dot.b)), ("salt", D), ("position", P), ("roots", Array(2, D)))
    params, ret = unit.signature()
    if tuple(params) != want:
        raise CompileError("scheme", f"{scheme.name}'s unit {unit.id} takes {params}, not {want}")
    z = Array(len(tile.pairs), dot.z)
    if not (isinstance(ret, Record) and [n for n, _t in ret.fields] == ["z", "receipt"] and ret.fields[0][1] == z
            and ret.fields[1][1].leaves):
        raise CompileError("scheme", f"{scheme.name}'s unit {unit.id} returns {ret!r}, not {{z: {z!r}, receipt}}")
    return ret.fields[1][1]


@lru_cache(maxsize=16)
def _plan(served: Any, scheme_name: str) -> _Plan:
    """The sites, in order (per node of the served function holding dots, one per kind of dot, in the order first met), their
    tiles, and the unit batches."""
    scheme = SCHEMES[scheme_name]
    r, c = scheme.tile
    sites: list[_SitePlan] = []
    nodes = []
    for k, nd in enumerate(served.body.nodes):
        if not _node_runs(served, k, nd, False):
            continue
        nodes.append(k)
        dots = list(_node_dots(served, k, nd, [a.refs for a in nd.args]))
        for d in dict.fromkeys(d for d, _a, _b in dots):
            a_rows, b_rows, pairs = _rows([(a, b) for e, a, b in dots if e == d])
            sites.append(_SitePlan(k, d, a_rows, b_rows, pairs, *_tiles(pairs, r, c)))
    if not sites:
        raise CompileError("no-matmul", f"{served.id} has no Call of a registered matmul family ({', '.join(MATMULS)})")
    levels = {sp.dot.boolean for sp in sites}
    if len(levels) > 1:
        raise CompileError("level", f"{served.id} holds dots on words and on bits: one compiled Program derives its seeds at one "
                                    f"level")
    groups, slot = [], 0
    for s, sp in enumerate(sites):
        by: dict[Tile, list[int]] = {}
        for u, t in enumerate(sp.units):
            by.setdefault(t.tile, []).append(u)
        for tile, members in by.items():
            unit = scheme.unit(sp.dot, tile)
            groups.append(_Group(s, tile, unit, _unit_receipt(scheme, sp.dot, tile, unit), tuple(members), slot))
            slot += len(members)
    return _Plan(levels.pop(), tuple(sites), tuple(groups), tuple(nodes))


def _gather(locs: Sequence[tuple[int, int, int]]) -> Refs:
    """Runs (node, first leaf, count), in order, as few `Affine` and `Strided` parts as their progression allows."""
    merged: list[list[int]] = []
    for k, b, n in locs:
        if merged and merged[-1][0] == k and merged[-1][1] + merged[-1][2] == b:
            merged[-1][2] += n
        else:
            merged.append([k, b, n])
    parts: list[Refs] = []
    i = 0
    while i < len(merged):
        k, b, n = merged[i]
        j = i + 1
        if j < len(merged) and merged[j][0] == k and merged[j][2] == n and merged[j][1] > b + n:
            step = merged[j][1] - b
            while j < len(merged) and merged[j][0] == k and merged[j][2] == n and merged[j][1] == b + (j - i) * step:
                j += 1
            parts.append(Strided("n", k, b, n, step, 1, (j - i) * n))
        else:
            parts.append(Affine("n", k, b, n))
        i = j
    return concat(parts)


class _Units:
    """The root's z source: each node's tiles as unit batches, one per site and tile shape, ahead of the node."""

    def __init__(self, B: Any, served: Any, plan: _Plan, group_of: tuple[int, ...], salt: Coll, positions: Coll,
                 roots: Coll) -> None:
        self.B, self.served, self.plan, self.group_of = B, served, plan, group_of
        self.salt, self.positions, self.roots = salt, positions, roots
        self.at = 0
        self.receipts: list[Refs] = []

    def __call__(self, nd: Any, args: Sequence[Coll]) -> Refs:
        k = self.plan.nodes[self.at]
        self.at += 1
        dots = list(_node_dots(self.served, k, nd, [a.refs for a in args]))
        PL = SD.types(self.plan.boolean)[1].leaves
        here = {sp.dot: s for s, sp in enumerate(self.plan.sites) if sp.node == k}
        at: dict[tuple[int, int], tuple[int, int]] = {}
        for gi, g in enumerate(self.plan.groups):
            if g.site not in here.values():
                continue
            sp = self.plan.sites[g.site]
            a_rows, b_rows, pairs = _rows([(a, b) for d, a, b in dots if d == sp.dot])
            assert pairs == sp.pairs, f"{self.served.id} node {k}: its dots' rows differ from the plan's"
            us, n = [sp.units[u] for u in g.members], len(g.members)
            a = Coll(Array(n, Array(g.tile.r, sp.dot.a)), _pack([a_rows[i] for t in us for i in t.rows]))
            b = Coll(Array(n, Array(g.tile.c, sp.dot.b)), _pack([b_rows[j] for t in us for j in t.cols]))
            pos = Coll(Array(n, self.positions.type.elem), self.positions.refs.slice(g.first * PL, (g.first + n) * PL))
            out = self.B.batch(g.unit, a, b, self.salt, pos, self.roots[self.group_of[g.site]], axes=(0, 0, None, 0, None),
                               name=f"pouw[{gi}]")
            kk, per = out.refs.idx, out.type.elem.leaves
            lz = per - g.receipt.leaves
            self.receipts.append(_gather([(kk, m * per + lz, g.receipt.leaves) for m in range(n)]))
            for m, u in enumerate(g.members):
                at[(g.site, u)] = (kk, m * per)
        seen = dict.fromkeys(here.values(), 0)
        locs = []
        for d, _a, _b in dots:
            s = here[d]
            u, cell = self.plan.sites[s].cells[seen[s]]
            seen[s] += 1
            kk, base = at[(s, u)]
            locs.append((kk, base + cell * d.z.leaves, d.z.leaves))
        return _gather(locs)


def _compiled_signature(S: Any) -> tuple:
    plan = _plan(S.SERVED, S.SCHEME)
    params, ret = S.SERVED.signature()
    D, P = SD.types(plan.boolean)
    return (tuple(params) + ((SALT, D), (POSITIONS, Array(plan.units, P)), (ROOTS, Array(max(S.GROUPS) + 1, Array(2, D)))),
            Record(("served", ret), ("receipts", TupleT(*[Array(len(g.members), g.receipt) for g in plan.groups]))))


def _compiled_body(B: Any, S: Any, *args: Coll) -> Coll:
    plan = _plan(S.SERVED, S.SCHEME)
    n = len(args) - 3
    units = _Units(B, S.SERVED, plan, S.GROUPS, *args[n:])
    out = _replay(B, S.SERVED, {("p", i): a.refs for i, a in enumerate(args[:n])}, units)
    assert units.at == len(plan.nodes)
    return Coll(_compiled_signature(S)[1], concat([out] + units.receipts))


def _compiled_ports(S: Any) -> Any:
    positions = {POSITIONS: _ports.In(kind="position")} if "position" in _ports.KINDS else {}
    return _ports.Ports(params={SALT: _ports.In(kind="coin"), **positions}, ret={"receipts": _ports.Out(sink=True)})


PouwCompiled = _composite(
    "PouwCompiled", ("SERVED", "SCHEME", "GROUPS"), _compiled_signature, _compiled_body, _compiled_ports,
    "SERVED with every registered matmul's dots in SCHEME's tile units: SERVED's parameters, `pouw.salt` (the coin), "
    "`pouw.positions` (a position per unit) and `pouw.roots` (per seed group, GROUPS[site] the site's, root(A) and root(B)) "
    "to {served: SERVED's return, receipts: per unit batch an Array over its units' receipts (sinks)}.")


# -- the compiled Program ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    """A node of the served function and a kind of dot. `a_rows`, `b_rows`: A's and B's distinct rows, as references into the
    served function's body (parameters ``p``, nodes ``n``), first use first; dot d multiplies A row ``pairs[d][0]`` by B row
    ``pairs[d][1]``. Its seed group (`Compiled.group_of`) is its slot of `pouw.roots`."""

    index: int
    node: int
    dot: Dot
    a_rows: tuple[Refs, ...]
    b_rows: tuple[Refs, ...]
    pairs: tuple[tuple[int, int], ...]

    @property
    def count(self) -> int:
        return len(self.pairs)

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.a_rows), self.dot.k, len(self.b_rows)


@dataclass(frozen=True)
class Unit:
    """One tile's unit: its site, its slot of `pouw.positions` (its position at base 0), the site's A and B rows it
    multiplies, its tile, its Definition and its receipt's type."""

    site: int
    slot: int
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    tile: Tile
    defn: Any
    receipt: Type


@dataclass(frozen=True)
class Floor:
    """One unit's noise-forced work (`Compiled.floor`): its batch's node in the compiled Program's root (`program.root`,
    where a placement puts it), its dots' kind, the dots its noise forces (all its cells where the kind's form,
    `Scheme.form`, puts noise on both rows at its k; none where it has a clean side or no form) and their multiply-adds,
    k each."""

    node: int
    kind: str
    dots: int
    macs: int


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


def _check_roots(roots: Sequence[tuple[bytes, bytes]], n: int) -> None:
    if len(roots) != n:
        raise ValueError(f"one (root_a, root_b) per seed group: {n}, not {len(roots)}")


def _check_salt(salt: bytes) -> None:
    if not isinstance(salt, (bytes, bytearray)) or len(salt) != 64:
        raise ValueError(f"`{SALT}` is the epoch's 64-byte salt")


def _sites(plan: _Plan) -> tuple[Site, ...]:
    return tuple(Site(s, sp.node, sp.dot, sp.a_rows, sp.b_rows, sp.pairs) for s, sp in enumerate(plan.sites))


def _group_of(plan: _Plan, groups: Callable[[Site], Any] | Sequence[Any] | None) -> tuple[int, ...]:
    """Each site's seed group, numbered in first appearance: `groups` is a key per site (sites of one key are committed
    together and share `pouw.roots`' slot), a function of the site to its key, or None for a group per site."""
    sites = _sites(plan)
    if groups is None:
        keys: Sequence[Any] = range(len(sites))
    elif callable(groups):
        keys = [groups(s) for s in sites]
    else:
        keys = list(groups)
        if len(keys) != len(sites):
            raise ValueError(f"a seed group key per site: {len(sites)} sites, not {len(keys)} keys")
    index: dict[Any, int] = {}
    return tuple(index.setdefault(k, len(index)) for k in keys)


def _grind(plan: _Plan, group_of: Sequence[int], scheme: Scheme) -> tuple[dict[str, float], ...]:
    """Per seed group, the grind term of its dots of each kind with a `RowNoise`: `grind.visible_premium`
    over the group's distinct A rows and B rows, each (its noise bits at its k, its readers among the group's dots of the
    kind), over those dots."""
    acc: dict[tuple[int, str], tuple[dict[tuple[int, tuple], tuple[int, int]], list[int]]] = {}
    for sp, g in zip(plan.sites, group_of):
        form = scheme.rows.get(sp.dot.family)
        if form is None:
            continue
        rows, dots = acc.setdefault((g, sp.dot.family), ({}, [0]))
        bits = form.bits(sp.dot.k)
        for side, rs in ((0, sp.a_rows), (1, sp.b_rows)):
            for x, n in Counter(p[side] for p in sp.pairs).items():
                key = (side, _row_key(rs[x]))
                rows[key] = (bits[side], rows.get(key, (0, 0))[1] + n)
        dots[0] += len(sp.pairs)
    out: list[dict[str, float]] = [{} for _ in range(max(group_of) + 1)]
    for (g, kind), (rows, dots) in acc.items():
        out[g][kind] = GR.visible_premium([(n, b, u) for (b, u), n in sorted(Counter(rows.values()).items())],
                                          GR.T_BITS) / dots[0]
    return tuple(out)


def _span(served: Any, plan: _Plan, group_of: Sequence[int]) -> tuple[int, int, int] | None:
    """A seed group holding the sites of two nodes one of which reads the other's output, through any nodes between (layer
    ℓ + 1's operands are layer ℓ's): (group, the earlier node, the later), or None."""
    nodes: dict[int, set[int]] = {}
    for sp, g in zip(plan.sites, group_of):
        nodes.setdefault(g, set()).add(sp.node)
    if all(len(ns) < 2 for ns in nodes.values()):
        return None
    above: list[int] = []
    for nd in served.body.nodes:
        a = 0
        for x in nd.args:
            for s, i, _lo, _hi in runs(x.refs):
                if s == "n":
                    a |= above[i] | 1 << i
        above.append(a)
    for g, ns in nodes.items():
        for n in sorted(ns):
            for m in sorted(ns):
                if above[n] >> m & 1:
                    return g, m, n
    return None


@dataclass(frozen=True, eq=False)
class Compiled:
    program: Program
    served: Program
    scheme: Scheme
    plan: _Plan
    group_of: tuple[int, ...]                 # each site's seed group (its slot of `pouw.roots`)

    @cached_property
    def digest(self) -> str:
        return program_digest(self.program)

    @property
    def boolean(self) -> bool:
        return self.plan.boolean

    @cached_property
    def sites(self) -> tuple[Site, ...]:
        return _sites(self.plan)

    @cached_property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Each seed group's sites: the calls committed together under one (root(A), root(B))."""
        return tuple(tuple(s for s, g in enumerate(self.group_of) if g == k) for k in range(max(self.group_of) + 1))

    @cached_property
    def cover(self) -> tuple[int, ...]:
        """The units each seed group covers (its coverage, which `Scheme.cover` bounds below)."""
        n = [0] * len(self.groups)
        for u in self.units:
            n[self.group_of[u.site]] += 1
        return tuple(n)

    @cached_property
    def grind(self) -> tuple[dict[str, float], ...]:
        """Per seed group, the grind term of its dots of each kind with a `RowNoise` (`Scheme.rows`): π* over
        its dots, each at most `grind.FLAG`."""
        return _grind(self.plan, self.group_of, self.scheme)

    @cached_property
    def units(self) -> tuple[Unit, ...]:
        """Every unit, by slot."""
        out = []
        for g in self.plan.groups:
            sp = self.plan.sites[g.site]
            out += [Unit(g.site, g.first + m, sp.units[u].rows, sp.units[u].cols, g.tile, g.unit, g.receipt)
                    for m, u in enumerate(g.members)]
        return tuple(out)

    @cached_property
    def floor(self) -> tuple[Floor, ...]:
        """Every unit's noise-forced work, by slot: W_b, the floor `Accounting.Guarantees.PouwBlockCredit`
        (cursor/attrib-ea24) attributes to the block the unit is placed at, as `Wb · Nmin` with `Wb` the least of the
        block's units' `macs` and `Nmin` the right units PoUW's draw established. Read off the plan and the scheme's forms,
        as the cover and grind checks are, with no gate walked. It is the attributed floor, not every gate the salt
        reaches: the seed's and the noise's SHA-512 and the receipt fold read the salt too, and the floor counts none of
        them (`Accounting.Assumptions.PouwFloorAdvice`'s scope, the committed words' coin × coin products)."""
        units = {id(g.unit) for g in self.plan.groups}
        nodes = [k for k, nd in enumerate(self.program.root.body.nodes) if id(nd.fn) in units]
        assert len(nodes) == len(self.plan.groups)
        out = []
        for k, g in zip(nodes, self.plan.groups):
            dot = self.plan.sites[g.site].dot
            form = self.scheme.form(dot.family)
            n = len(g.tile.pairs) if form is not None and min(form.bits(dot.k)) > 0 else 0
            out += [Floor(k, dot.family, n, n * dot.k)] * len(g.members)
        return tuple(out)

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

    def positions(self, base: int = 0) -> list[int]:
        """`pouw.positions`' values: unit u's is ``base + u``. A salt seeds each (salt, position) once, so a verifier that runs
        the Program more than once under one salt gives each run its own range (``base``); the one-stage verifier's is
        every unit of the log's earlier records (`sampled_proofs.one_stage.registered.log_base`)."""
        if base < 0 or base + len(self.units) > 1 << 64:
            raise ValueError(f"positions {base} .. {base + len(self.units) - 1} are no u64s")
        return [base + u.slot for u in self.units]

    def seeds(self, salt: bytes, roots: Sequence[tuple[bytes, bytes]], base: int = 0) -> list[bytes]:
        """Each unit's seed_j, as its proof derives it (`seeds.seed`), from its seed group's operand roots ``roots[group]``,
        under the epoch's 64-byte `salt` (a certified epoch's, `audit.Epoch.certified`)."""
        _check_salt(salt)
        _check_roots(roots, len(self.groups))
        return [SD.seed(salt, p, *roots[self.group_of[u.site]]) for u, p in zip(self.units, self.positions(base))]

    def inputs(self, salt: bytes, roots: Sequence[tuple[bytes, bytes]], base: int = 0) -> list[list[int]]:
        """The leaves of `pouw.salt` (the epoch's 64-byte salt, a certified epoch's), `pouw.positions` and `pouw.roots`
        (a pair per seed group), the arguments after the served ones: the rows the one-stage verifier fills
        (`sampled_proofs.one_stage.registered.supplied_rows`) from the window's salt, the log's ``base`` and the roots
        the record names."""
        _check_salt(salt)
        _check_roots(roots, len(self.groups))
        b = self.boolean
        return [SD.leaves(salt, b), [v for p in self.positions(base) for v in SD.position_leaves(p, b)],
                [v for ra, rb in roots for v in SD.leaves(bytes(ra) + bytes(rb), b)]]

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
        units = {id(gr.unit) for gr in self.plan.groups}
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
        receipts = fn.signature()[1].leaves - self.served_leaves
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

def _coin_ports(fn: Any) -> bool:
    pt = _ports.port_types(fn) if _ports is not None else None
    return pt is not None and any(p.kind == "coin" for rs in pt.params for _lo, _hi, p, _w in rs)


def _reads_coin(served: Program) -> str | None:
    """Why the served Program reads a coin, or None."""
    if _coin_ports(served.fn):
        return f"{served.fn.id} takes a coin parameter"
    seen, todo = set(), [served.fn]
    while todo:
        fn = todo.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        if isinstance(fn, SpecializedDefinition):
            if fn.defn is PouwCompiled:
                return f"{fn.id} is already compiled: it reads {SALT}"
            if SD.is_seed(fn):
                return f"it derives a seed ({fn.id}), which reads the salt"
            todo.extend(nd.fn for nd in fn.body.nodes)
        elif _coin_ports(fn):
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


def _reads(r: Refs, space: str, idx: int) -> bool:
    return any(s == space and i == idx for s, i, _lo, _hi in runs(r))


@lru_cache(maxsize=None)
def _check_unit(U: Any, boolean: bool, stream: tuple[Any, ...] = ()) -> None:
    """One unit Definition: one seed, on its salt, position and roots parameters whole; the salt read by nothing else; and its
    z reached from no node the seed reaches, or for a one-stream scheme (`stream`) recombined from words its receipt folds
    (`_check_stream`)."""
    body = U.body
    at = [k for k, nd in enumerate(body.nodes) if SD.is_seed(nd.fn)]
    if len(at) != 1 or body.nodes[at[0]].form != "call":
        raise CompileError("seed", f"{U.id} calls the seed {len(at)} times at its top level: a unit derives its noise from one "
                                   f"call of `seeds.PouwSeed`, or of `seeds.PouwUnitSeeds` for its noise and its fold keys")
    ks = at[0]
    D, P = SD.types(boolean)
    args = body.nodes[ks].args
    if runs(args[0].refs) != [("p", 2, 0, D.leaves)]:
        raise CompileError("salt-flow", f"{U.id}'s seed reads {args[0].refs!r} as its salt, not the unit's salt parameter")
    for j, n, what in ((1, P.leaves, "position"), (2, 2 * D.leaves, "roots")):
        if runs(args[j].refs) != [("p", j + 2, 0, n)]:
            raise CompileError("position", f"{U.id}'s seed reads {args[j].refs!r} as its {what}, not the unit's {what} "
                                           f"parameter: a unit can't choose its position")
    for k, nd in enumerate(body.nodes):
        for j, a in enumerate(nd.args):
            if (k, j) != (ks, 0) and _reads(a.refs, "p", 2):
                raise CompileError("salt-flow", f"{U.id} node {k} ({nd.fn.id}) argument {j} reads the salt: it flows only into "
                                                f"the seed")
    if _reads(body.ret, "p", 2):
        raise CompileError("salt-flow", f"{U.id} returns a leaf of the salt")
    if stream:
        _check_stream(U, stream)
        return
    tainted: set[int] = set()
    for k, nd in enumerate(body.nodes):
        if k == ks or any(s == "n" and i in tainted for a in nd.args for s, i, _lo, _hi in runs(a.refs)):
            tainted.add(k)
    lz = U.signature()[1].fields[0][1].leaves
    hit = sorted({i for s, i, _lo, _hi in runs(body.ret.slice(0, lz)) if s == "n" and i in tainted})
    if hit:
        raise CompileError("noise-in-z", f"{U.id}'s z reads node {hit[0]} ({body.nodes[hit[0]].fn.id}), which the seed reaches: "
                                         f"z is the exact product")


def _check_stream(U: Any, stream: tuple[Any, ...]) -> None:
    """A one-stream unit's z: each leaf an output of a call of a `stream` Definition, and every leaf such a call reads either
    another such call's output or a leaf a node in the receipt's cone reads (a parameter's included), so z is a function of
    the words the receipt folds and of what they are made from, and of nothing else."""
    body = U.body
    lz = U.signature()[1].fields[0][1].leaves
    cone, todo = set(), [i for s, i, _lo, _hi in runs(body.ret.slice(lz, body.ret.count)) if s == "n"]
    while todo:
        k = todo.pop()
        if k not in cone:
            cone.add(k)
            todo += [i for a in body.nodes[k].args for s, i, _lo, _hi in runs(a.refs) if s == "n"]
    folded: dict[tuple[str, int], bytearray] = {}
    for k in cone:
        for a in body.nodes[k].args:
            for s, i, lo, hi in runs(a.refs):
                f = folded.setdefault((s, i), bytearray())
                if len(f) < hi:
                    f.extend(bytes(hi - len(f)))
                f[lo:hi] = b"\x01" * (hi - lo)
    ids = {id(f) for f in stream}
    names = ", ".join(f.id for f in stream)

    def made(nd: Any) -> bool:
        return id(getattr(nd.fn, "defn", nd.fn)) in ids

    for s, i, _lo, _hi in runs(body.ret.slice(0, lz)):
        nd = body.nodes[i] if s == "n" else None
        if nd is None or not made(nd):
            raise CompileError("stream", f"{U.id}'s z reads {s}{i} ({nd.fn.id if nd else 'a parameter'}), not a recombination "
                                         f"of the words its receipt folds ({names})")
    seen: set[int] = set()
    todo = [i for s, i, _lo, _hi in runs(body.ret.slice(0, lz)) if s == "n"]
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        nd = body.nodes[i]
        for a in nd.args:
            for s2, j, lo, hi in runs(a.refs):
                if s2 == "n" and made(body.nodes[j]):
                    todo.append(j)
                elif len(f := folded.get((s2, j), b"")) < hi or not all(f[lo:hi]):
                    raise CompileError("stream", f"{U.id}'s z at node {i} ({nd.fn.id}) reads {s2}{j} [{lo}, {hi}), which the "
                                                 f"receipt doesn't fold: z is the one stream's served output")


def _check(c: Compiled) -> None:
    from verity.ml.circuits.cut import fits

    fn, plan = c.program.fn, c.plan
    body = fn.body
    units = {id(g.unit) for g in plan.groups}
    bad = _uncovered(fn, units)
    if bad:
        raise CompileError("uncovered", f"a matmul is left outside every unit: {'; '.join(bad[:4])}")
    D, P = SD.types(plan.boolean)
    n = len(body.params) - 3
    names = {n: SALT, n + 1: POSITIONS, n + 2: ROOTS}
    code = {n: "salt-flow", n + 1: "position", n + 2: "position"}
    nodes = [k for k, nd in enumerate(body.nodes) if id(nd.fn) in units]
    if len(nodes) != len(plan.groups):
        raise CompileError("position", f"{len(nodes)} unit batches at the root, not the plan's {len(plan.groups)}")
    want: dict[tuple[int, int], tuple[Refs, int | None]] = {}
    period: dict[int, tuple[int, int]] = {}
    for k, g in zip(nodes, plan.groups):
        if body.nodes[k].fn is not g.unit:
            raise CompileError("position", f"node {k} is {body.nodes[k].fn.id}, not unit batch {g.unit.id}")
        m = len(g.members)
        want[(k, 2)] = (Affine("p", n, 0, D.leaves), None)
        want[(k, 3)] = (Affine("p", n + 1, g.first * P.leaves, m * P.leaves), 0)
        want[(k, 4)] = (Affine("p", n + 2, c.group_of[g.site] * 2 * D.leaves, 2 * D.leaves), None)
        ret = g.unit.signature()[1]
        period[k] = (ret.fields[0][1].leaves, ret.leaves)
    slots = sorted((g.first, len(g.members)) for g in plan.groups)
    if [f for f, _m in slots] != [sum(m for _f, m in slots[:i]) for i in range(len(slots))] or \
            sum(m for _f, m in slots) != body.params[n + 1][1].n:
        raise CompileError("position", f"the unit batches' slots {slots} don't cover {POSITIONS} once each")
    S = c.served_leaves
    readers = [(f"node {k} ({nd.fn.id}) argument {j}", k, j, a.refs, nd.axes[j] if nd.form == "batch" else None)
               for k, nd in enumerate(body.nodes) for j, a in enumerate(nd.args)]
    readers.append(("the served outputs", None, None, body.ret.slice(0, S), None))
    for what, k, j, r, ax in readers:
        rs = runs(r)
        if (k, j) in want:
            exp, exp_ax = want[(k, j)]
            if rs != runs(exp) or ax != exp_ax:
                i = exp.idx
                raise CompileError(code[i], f"{what} reads {r!r} (axis {ax}), not its slot of {names[i]} {exp!r} (axis {exp_ax})")
            continue
        for s, i, lo, hi in rs:
            if s == "p" and i in names:
                raise CompileError(code[i], f"{what} reads {names[i]}: only a unit batch's seed inputs do")
            if s == "n" and i in period:
                lz, per = period[i]
                if lo % per + (hi - lo) > lz:
                    raise CompileError("separator", f"{what} reads node {i}'s leaves [{lo}, {hi}), past a unit's z")
    for s, i, lo, hi in runs(body.ret.slice(S, body.ret.count)):
        if s != "n" or i not in period or lo % period[i][1] < period[i][0]:
            raise CompileError("separator", f"the receipts read {s}{i} [{lo}, {hi}), not a unit's receipt")
    for g in plan.groups:
        _check_unit(g.unit, plan.boolean, c.scheme.stream)
    read = {k: bytearray(nd.out.leaves) for k, nd in enumerate(body.nodes) if k not in period}
    for r in [a.refs for nd in body.nodes for a in nd.args] + [body.ret]:
        for s, i, lo, hi in runs(r):
            if s == "n" and i in read:
                read[i][lo:hi] = b"\x01" * (hi - lo)
    for k, flags in read.items():
        nd = body.nodes[k]
        m = _members(nd)
        widths = nd.out.leaf_widths() if nd.out.homogeneous_width() is None else None
        w = nd.out.homogeneous_width()
        L = nd.out.leaves // m
        for u in range(m):
            lo = u * L
            og = sum(flags[lo:lo + L])
            ob = og * w if widths is None else sum(widths[j] for j in range(lo, lo + L) if flags[j])
            if not fits(ob, og):
                raise CompileError("width", f"node {k} ({nd.fn.id}) member {u} commits {ob} bits in {og} values, past "
                                            f"the width rule")


# -- the pass ----------------------------------------------------------------------------------------------------------------

def compile(program: Program, scheme: str, groups: Callable[[Site], Any] | Sequence[Any] | None = None) -> Compiled:
    """`program` with every registered matmul's dots in `scheme`'s tile units (the module docstring), checked. `groups` names
    the calls committed together, each group one seed group (one slot of `pouw.roots`): a key per site, or a function of the
    `Site` to its key; None is a group per site, one call."""
    sch = _scheme(scheme)
    why = _reads_coin(program)
    if why is not None:
        raise CompileError("coin-read", f"the served Program reads a coin ({why}); the salt enters only through compile")
    names = [n for n, _t in program.fn.signature()[0]]
    for n in (SALT, POSITIONS, ROOTS):
        if n in names:
            raise CompileError("name", f"{program.fn.id} has a parameter named {n!r}")
    registered = tuple(n for n, b in zip(names, program.bindings) if b == REGISTRATION)
    served = _served_fn(program.fn)
    plan = _plan(served, scheme)
    group_of = _group_of(plan, groups)
    span = _span(served, plan, group_of)
    if span is not None:
        g, m, n = span
        raise CompileError("span", f"seed group {g} holds the calls of node {m} and of node {n}, which reads node {m}'s output: "
                                   f"a seed binds its calls' operand roots, so one seed can't span calls run one after "
                                   f"another, such as a step's layers")
    cover = [0] * (max(group_of) + 1)
    for sp, g in zip(plan.sites, group_of):
        cover[g] += len(sp.units)
    if min(cover) < sch.cover:
        g = cover.index(min(cover))
        raise CompileError("cover", f"seed group {g} (sites {[s for s, k in enumerate(group_of) if k == g]}) covers "
                                    f"{cover[g]} units, fewer than {scheme}'s {sch.cover}: commit more calls together "
                                    f"(`groups`)")
    for g, by in enumerate(_grind(plan, group_of, sch)):
        for kind, eps in by.items():
            if eps > GR.FLAG:
                raise CompileError("cover", f"seed group {g} (sites {[s for s, k in enumerate(group_of) if k == g]}): the "
                                            f"grind term of its {kind} dots is {100 * eps:.4g}%, over {100 * GR.FLAG:g}% at "
                                            f"T = 2^{GR.T_BITS} (decision 122: GRIND's premium over its rows' noise bits and "
                                            f"readers): commit more of the call's heads and requests together, and below "
                                            f"that (decode at a small batch, a short prompt) the group's noise is a live "
                                            f"per-round coin's (decision 36), not a seed's")
    fn = bind(PouwCompiled, SERVED=served, SCHEME=scheme, GROUPS=group_of)
    try:
        compiled = Program(fn, registered=registered)
    except ValueError as e:
        if _ports is not None and isinstance(e, _ports.PortError):
            raise CompileError(e.code, str(e).removeprefix(f"{e.code}: ")) from e
        raise
    c = Compiled(compiled, program, SCHEMES[scheme], plan, group_of)
    _check(c)
    return c


def rebuild(served: Program, scheme: str, descriptor: dict,
            groups: Callable[[Site], Any] | Sequence[Any] | None = None) -> Compiled:
    """The verifier's compiled Program, from (served, scheme, groups), held to an offered descriptor."""
    c = compile(served, scheme, groups)
    c.check_descriptor(descriptor)
    return c


__all__ = ["ANY", "BUILTIN", "CompileError", "Compiled", "Dot", "Floor", "GateCheck", "MATMULS", "Matmul", "POSITIONS",
           "PV_ROWS", "PouwCall", "PouwCompiled", "PouwRest", "QK_ROWS", "ROOTS", "ROWS", "RowNoise", "SALT", "SCHEMES", "STEP",
           "Scheme", "Site", "Tile", "Unit", "Z", "compile", "rebuild", "register", "seed"]
