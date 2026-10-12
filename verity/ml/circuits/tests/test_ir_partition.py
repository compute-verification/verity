"""`verity.ml.circuits.partition`: the fast path (per-body induction over `Family.tiling()`) against the interval sweep on the
same materialised sets; the sweep on synthetic interval families and on Parts of real Programs; `validate_width`
(bound-only certification vs exact, budget, Unsupported) against an oracle boundary and against `verity.ml.circuits.boundary`;
readiness on Programs with 64-bit gates; the `VuVerdict` composition; `query_ast` families end to end.

Stand-ins, built from explicit `Part`s so the validator is checked independently of `query_ast`:

* `StubGateSet` / `StubFamily`: the `GateSet` / `Family` shape;
* `NodeTiling` / `BodyTiling` / `Tiling`: the field names `validate_partition`'s fast path reads; `materialise` turns a
  tiling into the family it induces, so the fast path can be checked against the sweep on the same sets;
* `OracleBoundary`: `w_out` from the explicit oracle `GateSelection.out` (gate widths from the producing primitive), with
  an adjustable slack on the bound and an optional "exact is Unsupported" mode;
* small Programs from test-local primitives: `gemm_program()` (`TGemm{K=32,N=6}`: Input batches, a composite batch,
  nested calls), `synth_program(N)` (a pair of zip-able batches, a constant, a scan), `twice_program()` and
  `wide_program()`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import pytest

from verity.ml.circuits import boundary, partition, query_ast as G
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.layout import Scope, GateSelection
from verity.ml.circuits.parts import Limits, Part, Unsupported, n_members
from verity.ml.circuits.partition import (PartitionVerdict, ReadinessReport, VuVerdict, WidthVerdict, gate_width_readiness, readiness_32bit,
                                 validate_partition, validate_vu_family, validate_vu_partition, validate_width, vu_status)
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V16, V32 = Value(16), Value(32)
A16 = Array(16, V16)


def prim(name: str, params, ret, fn) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, tuple(params), ret, fn, register=False)


Add32 = prim("TAdd32", [("a", V32), ("b", V32)], V32, lambda a, b: (a + b) & 0xFFFFFFFF)
Mul32 = prim("TMul32", [("a", V32), ("b", V32)], V32, lambda a, b: (a * b) & 0xFFFFFFFF)
Zero32 = prim("TZero32", [], V32, lambda: 0)
Dot16 = prim("TDot16", [("acc", V32), ("a", A16), ("b", A16)], V32, lambda acc, *ab: acc)
Cvt16 = prim("TCvt16", [("a", V32)], V16, lambda a: a >> 16)


def _dot(Bd, S, x, w):
    acc = Bd.call(Zero32)
    for s in range(S.K // 16):
        acc = Bd.call(Dot16, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
    return acc


Dot = CompositeDefinition("TDot", 1, ("K",), lambda S: ((("x", Array(S.K, V16)), ("w", Array(S.K, V16))), V32), _dot,
                          register=False)
Coord = CompositeDefinition("TGemmCoordinate", 1, ("K",), lambda S: ((("x", Array(S.K, V16)), ("wrow", Array(S.K, V16))), V16),
                            lambda Bd, S, x, wrow: Bd.call(Cvt16, Bd.call(bind(Dot, K=S.K), x, wrow)), register=False)
Gemm = CompositeDefinition("TGemm", 1, ("K", "N"),
                           lambda S: ((("x", Array(S.K, V16)), ("w", Array(S.N, Array(S.K, V16)))), Array(S.N, V16)),
                           lambda Bd, S, x, w: Bd.batch(bind(Coord, K=S.K), x, w, axes=(None, 0)), register=False)


# ---------------------------------------------------------------------------------------------------------
# GateSet / Family stubs
# ---------------------------------------------------------------------------------------------------------


def _merge(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ivs = sorted(ivs)
    out: list[tuple[int, int]] = []
    for lo, hi in ivs:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


@dataclass(frozen=True)
class StubGateSet:
    parts: tuple[Part, ...]

    def intervals(self) -> list[tuple[int, int]]:
        return _merge([iv for p in self.parts for iv in p.intervals()])

    def size(self) -> int:
        return sum(hi - lo for lo, hi in self.intervals())

    def contains(self, g: int) -> bool:
        return any(lo <= g < hi for lo, hi in self.intervals())

    def gates(self) -> list[int]:
        return [g for lo, hi in self.intervals() for g in range(lo, hi)]


@dataclass(frozen=True)
class IntervalSet:
    """A synthetic set given directly by intervals (no Parts): for sweep tests on fake programs."""
    ivs: tuple[tuple[int, int], ...]

    def intervals(self) -> list[tuple[int, int]]:
        return list(self.ivs)


class StubFamily:
    def __init__(self, sets: list, tiling: Any = None, count: int | None = None) -> None:
        self._sets = list(sets)
        self._tiling = tiling
        self._count = len(self._sets) if count is None else count

    def count(self) -> int:
        return self._count

    def by_index(self, i: int):
        return self._sets[i]

    def index_of(self, s) -> int | None:
        target = s.intervals()
        for i, x in enumerate(self._sets):
            if x.intervals() == target:
                return i
        return None

    def __iter__(self) -> Iterator:
        return iter(self._sets)

    def tiling(self):
        return self._tiling


class FakeProgram:
    """Only `.gates`: enough for the sweep on synthetic interval families."""

    def __init__(self, gates: int) -> None:
        self.gates = gates


# ---------------------------------------------------------------------------------------------------------
# Tiling stubs and their materialisation
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeTiling:
    node: int
    kind: str                                          # "atomic" | "whole" | "zip" | "descend"
    members: tuple[tuple[int, int], ...]               # member ranges covered by this entry
    k: int = 1                                         # atomic: k adjacent members per set
    group: int | None = None                           # zip: group id within the body
    child: str | None = None                           # descend: plan key of the callee body's tiling


@dataclass(frozen=True)
class BodyTiling:
    key: str
    spec_id: str
    spec: Any
    n_nodes: int
    entries: tuple[NodeTiling, ...]
    count: int


@dataclass(frozen=True)
class Tiling:
    root: str
    bodies: dict[str, BodyTiling]


# -- builders: a tiling is assembled from per-body rule dicts {node: rule}; rules are the tuples
#    ("whole",) | ("atomic", k) | ("ranges", ((a, b), ...)) [= whole per range] | ("zip", group) | ("descend", BodyTiling)
#    | ("descend_partial", BodyTiling, ((a, b), ...)) | ("atomic_ranges", k, ((a, b), ...)) | ("multi", rule, rule, ...)


def _entries_of(spec, rules: Any) -> tuple[NodeTiling, ...]:
    pairs = sorted(rules.items()) if isinstance(rules, dict) else sorted(rules, key=lambda p: p[0])
    out: list[NodeTiling] = []
    for j, rule in pairs:
        n = n_members(spec.body.nodes[j])
        full = ((0, n),) if n else ()
        for r in (rule[1:] if rule[0] == "multi" else (rule,)):
            kind = r[0]
            if kind == "whole":
                out.append(NodeTiling(j, "whole", full))
            elif kind == "atomic":
                out.append(NodeTiling(j, "atomic", full, k=r[1]))
            elif kind == "ranges":
                out.append(NodeTiling(j, "whole", tuple(r[1])))
            elif kind == "atomic_ranges":
                out.append(NodeTiling(j, "atomic", tuple(r[2]), k=r[1]))
            elif kind == "zip":
                out.append(NodeTiling(j, "zip", full, group=r[1]))
            elif kind == "descend":
                out.append(NodeTiling(j, "descend", full, child=r[1].key))
            elif kind == "descend_partial":
                out.append(NodeTiling(j, "descend", tuple(r[2]), child=r[1].key))
            else:
                out.append(NodeTiling(j, kind, full))  # unknown kinds pass through (malformed on purpose)
    return tuple(out)


def _count_of(spec, entries: tuple[NodeTiling, ...], bodies: dict[str, BodyTiling]) -> int:
    total = 0
    zips: dict[Any, int] = {}
    for e in entries:
        n = n_members(spec.body.nodes[e.node])
        if e.kind == "whole":
            total += len(e.members)
        elif e.kind == "atomic":
            total += sum(-(-(b - a) // e.k) for a, b in e.members)
        elif e.kind == "zip":
            zips[e.group] = max(zips.get(e.group, 0), n)
        elif e.kind == "descend":
            total += sum(b - a for a, b in e.members) * bodies[e.child].count
    return total + sum(zips.values())


_KEYS = {"n": 0}


def body_tiling(spec, rules: Any, key: str | None = None, bodies: dict[str, BodyTiling] | None = None, count: int | None = None) -> BodyTiling:
    """A BodyTiling for `spec` from `{node: rule}`; descended children must already be in `bodies` (they are added to
    the dict the caller threads through `tiling()`)."""
    entries = _entries_of(spec, rules)
    _KEYS["n"] += 1
    key = key or f"plan{_KEYS['n']}"
    kids = {}
    for r in (rules.values() if isinstance(rules, dict) else (r for _, r in rules)):
        for rr in (r[1:] if r[0] == "multi" else (r,)):
            if rr[0] in ("descend", "descend_partial"):
                kids[rr[1].key] = rr[1]
                kids.update(_all_bodies(rr[1]))
    bt = BodyTiling(key, spec.id, spec, len(spec.body.nodes), entries,
                    _count_of(spec, entries, kids) if count is None else count)
    bt.__dict__["_kids"] = kids  # carried so `tiling()` can collect every body without a registry
    return bt


def _all_bodies(bt: BodyTiling) -> dict[str, BodyTiling]:
    out = {bt.key: bt}
    for k, v in bt.__dict__.get("_kids", {}).items():
        out[k] = v
        out.update(_all_bodies(v))
    return out


def tiling(root: BodyTiling) -> Tiling:
    return Tiling(root.key, _all_bodies(root))


def whole_body(spec, except_nodes: dict[int, Any] | None = None, **kw) -> BodyTiling:
    """Every node of `spec.body` as one whole set, with `except_nodes` overriding selected node rules."""
    rules = {j: ("whole",) for j in range(len(spec.body.nodes))}
    rules.update(except_nodes or {})
    return body_tiling(spec, rules, **kw)


def members_k(k: int = 1) -> tuple:
    return ("atomic", k)


def ranges_of(*ranges: tuple[int, int]) -> tuple:
    return ("ranges", tuple(ranges))


def materialise(program: Program, t: Tiling) -> list[StubGateSet]:
    """The sets a tiling induces, in entry order (zip groups at the position of their first entry)."""
    out: list[StubGateSet] = []

    def body(sc: Scope, bt: BodyTiling) -> None:
        nodes = bt.spec.body.nodes
        groups: dict[Any, list[int]] = {}
        for e in bt.entries:
            j = e.node
            if e.kind == "whole":
                for a, b in e.members:
                    out.append(StubGateSet((Part(sc, frozenset((j, m) for m in range(a, b))),)))
            elif e.kind == "atomic":
                for a, b in e.members:
                    for c in range(a, b, e.k):
                        out.append(StubGateSet((Part(sc, frozenset((j, m) for m in range(c, min(c + e.k, b)))),)))
            elif e.kind == "zip":
                groups.setdefault(e.group, []).append(j)
            elif e.kind == "descend":
                for a, b in e.members:
                    for m in range(a, b):
                        body(sc.child(j, m), t.bodies[e.child])
            else:
                raise ValueError(e)
        for key, js in groups.items():
            n = max(n_members(nodes[j]) for j in js)
            for m in range(n):
                items = frozenset((j, m) for j in js if m < n_members(nodes[j]))
                out.append(StubGateSet((Part(sc, items),)))

    body(program.circuit.root_scope, t.bodies[t.root])
    return out


def family_of(program: Program, tiling: Tiling, with_tiling: bool = True) -> StubFamily:
    return StubFamily(materialise(program, tiling), tiling if with_tiling else None)


# ---------------------------------------------------------------------------------------------------------
# boundary stub: the explicit oracle
# ---------------------------------------------------------------------------------------------------------


class OracleBoundary:
    """`w_out(part)` = sum of `ret.w` over `GateSelection(circuit, gates).out` (the oracle); `w_out_bound` = exact +
    `slack` (a deliberately loose sound bound).  `exact_unsupported=True` makes `w_out` return Unsupported."""

    def __init__(self, program: Program, slack: int = 0, exact_unsupported: bool = False) -> None:
        self.program, self.slack, self.exact_unsupported = program, slack, exact_unsupported
        self.calls = {"bound": 0, "exact": 0}

    def _exact(self, part: Part) -> int:
        C = self.program.circuit
        gates = [g for lo, hi in part.intervals() for g in range(lo, hi)]
        return sum(C.gate(g).prim.ret.w for g in GateSelection(C, gates).out)

    def w_out_bound(self, part: Part) -> int:
        self.calls["bound"] += 1
        return self._exact(part) + self.slack

    def w_out(self, part: Part):
        self.calls["exact"] += 1
        if self.exact_unsupported:
            return Unsupported("stub", "exact disabled")
        return self._exact(part)


# ---------------------------------------------------------------------------------------------------------
# fixture programs
# ---------------------------------------------------------------------------------------------------------


def gemm_program() -> Program:
    """Root: Input16 x (32), Input16 w (192), batch TGemmCoordinate_v1{K=32} x 6 -> 248 gates.
    TGemmCoordinate body: call TDot (3 gates: TZero32 + 2 TDot16 steps), TCvt16."""
    return Program(bind(Gemm, K=32, N=6))


_ScanStep = CompositeDefinition(
    "TScanStep", 1, (), lambda S: ((("carry", V32), ("x", V32)), Tuple(V32, V32)),
    lambda B, S, carry, x: (lambda c: tuple_of(c, B.call(Mul32, c, x)))(B.call(Add32, carry, x)),
    register=False)


def _synth_body(B, S, x):
    sq = B.batch(Mul32, x, x, axes=(0, 0))          # node 1: n=N
    s = B.batch(Add32, sq, x, axes=(0, 0))          # node 2: n=N, member m reads sq[m]
    z = B.call(Zero32)                              # node 3: constant
    return B.scan(_ScanStep, z, xs=[s])                # node 4: N iterations of 2 gates


_Synth = CompositeDefinition("TSynth", 1, ("N",),
                             lambda S: ((("x", Array(S.N, V32)),), Tuple(V32, Array(S.N, V32))),
                             _synth_body, register=False)


def synth_program(N: int = 4) -> Program:
    """Root: Input32 x (N) | batch TMul32 (N) | batch TAdd32 (N) | TZero32 | scan TScanStep (N x 2) ->
    3N + 1 + 2N gates.  Nodes 1 and 2 are zip-able (equal n, member m of 2 reads member m of 1)."""
    return Program(bind(_Synth, N=N))


_Twice = CompositeDefinition(
    "TTwice", 1, (), lambda S: ((("x", Array(32, V16)), ("w", Array(32, Array(32, V16)))), Array(32, V16)),
    lambda B, S, x, w: B.call(bind(Gemm, K=32, N=32), B.call(bind(Gemm, K=32, N=32), x, w), w),
    register=False)


def twice_program() -> Program:
    """Root: Input16 x (32), Input16 w (1024), call TGemm{K=32,N=32} (node 2), call TGemm{K=32,N=32} (node 3, fed by node 2).
    The same specialization twice from one body: a shared callee for the multiplicity / plan-key tests."""
    return Program(bind(_Twice))


_Wide64 = PrimitiveDefinition("TWide64", 1, (), Value(64), lambda: 0, register=False)


def wide_program() -> Program:
    """A Program with a 64-bit Input and a 64-bit primitive gate (readiness violators of both kinds)."""
    W = CompositeDefinition("TWide", 1, (), lambda S: ((("seed", Value(64)),), Tuple(Value(64), V32)),
                            lambda B, S, seed: tuple_of(B.call(_Wide64), B.call(Add32, B.call(Zero32), B.call(Zero32))),
                            register=False)
    return Program(bind(W))


def test_stub_gemm_layout():
    P_ = gemm_program()
    assert P_.gates == 248
    fam = family_of(P_, tiling(whole_body(P_.root)))
    assert fam.count() == 3
    assert sum(s.size() for s in fam) == 248


def test_stub_synth_layout():
    P_ = synth_program(4)
    assert [n.form for n in P_.root.body.nodes] == ["batch", "batch", "batch", "primitive", "scan"]
    assert P_.gates == 3 * 4 + 1 + 2 * 4


def test_oracle_boundary_on_gemm_member():
    P_ = gemm_program()
    sc = P_.circuit.root_scope
    B = OracleBoundary(P_)
    assert B.w_out(Part(sc, frozenset({(2, 0)}))) == 16        # one GemmCoordinate: its bf16 output
    assert B.w_out(Part(sc, frozenset({(0, 0)}))) == 16        # one Input16 gate read by every coordinate
    assert B.w_out_bound(Part(sc, frozenset({(2, 0)}))) == 16
    assert OracleBoundary(P_, slack=7).w_out_bound(Part(sc, frozenset({(2, 0)}))) == 23


@pytest.fixture
def gemm():
    return gemm_program()


@pytest.fixture
def synth():
    return synth_program(4)


def gemm_vu_tiling(P_):
    """The proof-unit tiling of the Gemm program: every Input gate its own set, every GemmCoordinate one set."""
    return tiling(whole_body(P_.root, {0: members_k(), 1: members_k(), 2: members_k()}))


def gemm_deep_tiling(P_):
    """Descend into every coordinate: TDot's three gates and the TCvt16 each their own set."""
    coord = P_.root.body.nodes[2].fn
    dot = coord.body.nodes[0].fn
    dot_t = whole_body(dot)
    coord_t = body_tiling(coord, {0: ("descend", dot_t), 1: ("whole",)})
    return tiling(whole_body(P_.root, {2: ("descend", coord_t)}))


def both_paths(P_, t):
    """(fast-path verdict, sweep verdict) on the family the tiling induces."""
    fast = validate_partition(P_, family_of(P_, t, with_tiling=True))
    sweep = validate_partition(P_, family_of(P_, t, with_tiling=False))
    assert fast.detail["path"] == "tiling" and sweep.detail["path"] == "sweep"
    return fast, sweep


# -- fast path vs sweep: agreement ---------------------------------------------------------------------------


def test_whole_root_nodes_exact(gemm):
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root)))
    assert fast.status == sweep.status == "exact"
    assert fast.codes == () and fast.detail["sets"] == 3 == sweep.detail["sets"]
    assert fast.detail["bodies"] == 1 and fast.detail["gates"] == 248


def test_vu_tiling_exact(gemm):
    fast, sweep = both_paths(gemm, gemm_vu_tiling(gemm))
    assert fast.status == sweep.status == "exact"
    assert fast.detail["sets"] == 32 + 192 + 6 == sweep.detail["sets"]


def test_deep_tiling_exact_count_by_induction(gemm):
    fast, sweep = both_paths(gemm, gemm_deep_tiling(gemm))
    assert fast.status == sweep.status == "exact"
    assert fast.detail["sets"] == 2 + 6 * (3 + 1) == sweep.detail["sets"]
    assert fast.detail["bodies"] == 3


def test_members_k_ranges_with_partial_last_block(gemm):
    t = tiling(whole_body(gemm.root, {1: members_k(50), 2: members_k(4)}))   # 192 = 3*50 + 42; 6 = 4 + 2
    fast, sweep = both_paths(gemm, t)
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 1 + 4 + 2


def test_unassigned_node_is_gap(gemm):
    rules = {0: ("whole",), 2: members_k()}  # root node 1 (the weights) missing
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "gap"
    assert fast.codes == ("tiling-unassigned",)
    [p] = fast.detail["problems"]
    assert p["code"] == "tiling-unassigned" and p["spec"] == gemm.root.id and p["node"] == 1 and p["fn"] == "Input16_v1"
    assert sweep.detail["gaps"] == [{"gates": [32, 224]}]


def test_unassigned_nested_node_is_gap(gemm):
    coord = gemm.root.body.nodes[2].fn
    coord_t = body_tiling(coord, {0: ("whole",)})  # the TCvt16 of every coordinate uncovered
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root, {2: ("descend", coord_t)})))
    assert fast.status == sweep.status == "gap"
    assert fast.detail["problems"][0]["spec"] == coord.id and fast.detail["problems"][0]["node"] == 1
    assert sweep.detail["gap_count"] == 6


def test_double_assignment_is_overlap(gemm):
    rules = [(0, ("whole",)), (1, ("whole",)), (2, members_k()), (2, ("whole",))]  # node 2 in two entries, full ranges both
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "overlap"
    assert fast.codes == ("tiling-members-overlap",)
    assert fast.detail["problems"][0]["node"] == 2 and fast.detail["sets"] == 2 + 6 + 1 == sweep.detail["sets"]
    # two entries whose ranges complement each other are fine: atomic over [0, 3), whole over [3, 6)
    rules = {0: ("whole",), 1: ("whole",), 2: ("multi", ("atomic_ranges", 1, ((0, 3),)), ("ranges", ((3, 6),)))}
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 2 + 3 + 1
    # a partial descent: members [0, 4) of the coordinates descended, [4, 6) atomic
    coord = gemm.root.body.nodes[2].fn
    rules = {0: ("whole",), 1: ("whole",), 2: ("multi", ("descend_partial", whole_body(coord), ((0, 4),)), ("atomic_ranges", 1, ((4, 6),)))}
    fast, sweep = both_paths(gemm, tiling(body_tiling(gemm.root, rules)))
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 2 + 4 * 2 + 2


def test_member_ranges_gap_and_overlap(gemm):
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root, {2: ranges_of((0, 3))})))
    assert fast.status == sweep.status == "gap" and fast.codes == ("tiling-members-gap",)
    assert fast.detail["problems"][0]["members"] == [3, 6]
    fast, sweep = both_paths(gemm, tiling(whole_body(gemm.root, {2: ranges_of((0, 4), (3, 6))})))
    assert fast.status == sweep.status == "overlap" and fast.codes == ("tiling-members-overlap",)
    # gap and overlap together: overlap wins, both codes listed
    fast, _ = both_paths(gemm, tiling(whole_body(gemm.root, {2: ranges_of((0, 4), (3, 5))})))
    assert fast.status == "overlap" and fast.codes == ("tiling-members-gap", "tiling-members-overlap")


def test_zip_group_exact_and_unequal_n(synth):
    t = tiling(whole_body(synth.root, {0: members_k(), 1: ("zip", "g"), 2: ("zip", "g"), 4: members_k()}))
    fast, sweep = both_paths(synth, t)
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 4 + 4 + 1 + 4
    # zip of a batch (n=4) with the constant (n=1): the sets such a group induces are not defined -> the description
    # is malformed and the sweep decides on whatever the family materialised (here: the covering reading)
    t = tiling(whole_body(synth.root, {2: ("zip", "g"), 3: ("zip", "g")}))
    v = validate_partition(synth, family_of(synth, t, with_tiling=True))
    assert v.detail["path"] == "sweep" and v.codes[0] == "tiling-malformed"
    assert "tiling-zip-unequal-n" in v.detail["tiling_malformed"]


def test_descend_into_scan_iterations(synth):
    step = synth.root.body.nodes[4].fn
    t = tiling(whole_body(synth.root, {4: ("descend", whole_body(step))}))
    fast, sweep = both_paths(synth, t)
    assert fast.status == sweep.status == "exact" and fast.detail["sets"] == 4 + 4 * 2


def test_shared_callee_from_two_parents_two_plans():
    """Gemm called twice from the root; each call descended through its own plan into ONE shared coordinate plan,
    which descends into the dot plan: multiplicities 2 x 32 coordinates, 64 dots.  Discovery order is not
    topological here (the coordinate plan is found under the first Gemm plan before the second Gemm plan exists)."""
    P_ = twice_program()
    gemm_spec = P_.root.body.nodes[2].fn
    assert P_.root.body.nodes[3].fn is gemm_spec
    coord = gemm_spec.body.nodes[0].fn
    dot = coord.body.nodes[0].fn
    dot_t = whole_body(dot, key="dot")
    coord_t = body_tiling(coord, {0: ("descend", dot_t), 1: ("whole",)}, key="coord")
    g1 = whole_body(gemm_spec, {0: ("descend", coord_t)}, key="gemm-a")
    g2 = whole_body(gemm_spec, {0: ("descend", coord_t)}, key="gemm-b")
    t = tiling(whole_body(P_.root, {2: ("descend", g1), 3: ("descend", g2)}))
    assert set(t.bodies) == {t.root, "gemm-a", "gemm-b", "coord", "dot"}
    fast, sweep = both_paths(P_, t)
    assert fast.status == sweep.status == "exact"
    assert fast.detail["sets"] == 2 + 2 * 32 * (3 + 1) == sweep.detail["sets"] and fast.detail["bodies"] == 5
    # width: the tiling path must count 64 coordinate activations (2 parents x 32 members) and 64 dot activations
    B = OracleBoundary(P_)
    saved = partition._BOUNDARY
    partition._BOUNDARY = B
    try:
        v = validate_width(P_, family_of(P_, t), 32)
    finally:
        partition._BOUNDARY = saved
    assert v.detail["sets"] == fast.detail["sets"] and v.detail["path"] == "tiling"
    # the two Input batches as whole sets fail; the rest (32-bit accumulator chain gates, 16-bit cvt) pass
    assert v.status == "violation" and v.detail["violation_count"] == 2 and v.max_seen == 1024 * 16
    assert v.detail["exact_evaluated"] == 2                       # only the two Input sets needed the oracle
    # with the Inputs per member the whole family certifies, and the 64 activations of every nested class are counted
    t = tiling(whole_body(P_.root, {0: members_k(), 1: members_k(), 2: ("descend", g1), 3: ("descend", g2)}))
    partition._BOUNDARY = B
    try:
        v = validate_width(P_, family_of(P_, t), 32)
    finally:
        partition._BOUNDARY = saved
    assert v.status == "ok" and v.detail["sets"] == 32 + 1024 + 2 * 32 * 4


def test_scopes_of_enumerates_every_activation():
    P_ = twice_program()
    gemm_spec = P_.root.body.nodes[2].fn
    coord = gemm_spec.body.nodes[0].fn
    coord_t = whole_body(coord, key="coord")
    g = whole_body(gemm_spec, {0: ("descend", coord_t)}, key="gemm")
    t = tiling(whole_body(P_.root, {2: ("descend", g), 3: ("descend", g)}))
    from verity.ml.circuits.partition import CheckedTiling
    T = CheckedTiling(P_, t)
    assert T.instances == {(t.root,): 1, ("gemm",): 2, ("coord",): 64}
    scopes = list(T.scopes_of(("coord",)))
    assert len(scopes) == 64 and len({sc.offset for sc in scopes}) == 64
    assert all(sc.spec is coord for sc in scopes)
    assert [sc.path()[-1] for sc in scopes[:2]] == [f"{coord.id}#0", f"{coord.id}#1"]


# -- malformed descriptions fall back to the sweep --------------------------------------------------------


def test_descend_on_primitive_node_is_malformed_then_swept(gemm):
    t = tiling(whole_body(gemm.root, {0: ("descend", whole_body(gemm.root))}))
    fam = StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=t)
    v = validate_partition(gemm, fam)
    assert v.status == "exact" and v.detail["path"] == "sweep"
    assert v.codes == ("tiling-malformed",) and "primitive" in v.detail["tiling_malformed"]


def test_descend_into_wrong_spec_is_malformed(gemm, synth):
    t = tiling(whole_body(gemm.root, {2: ("descend", whole_body(synth.root.body.nodes[4].fn))}))
    v = validate_partition(gemm, StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=t))
    assert v.codes[0] == "tiling-malformed" and v.detail["path"] == "sweep"


def test_wrong_root_is_malformed(gemm, synth):
    v = validate_partition(gemm, StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=tiling(whole_body(synth.root))))
    assert v.codes[0] == "tiling-malformed" and v.status == "exact"


def test_unknown_rule_kind_is_malformed(gemm):
    t = tiling(whole_body(gemm.root, {2: ("halve",)}))
    v = validate_partition(gemm, StubFamily(materialise(gemm, tiling(whole_body(gemm.root))), tiling=t))
    assert v.codes[0] == "tiling-malformed"


def test_count_mismatch_falls_back_to_sweep(gemm):
    t = gemm_vu_tiling(gemm)
    sets = materialise(gemm, t)
    v = validate_partition(gemm, StubFamily(sets, tiling=t, count=len(sets) + 1))
    assert v.codes[0] == "tiling-count-mismatch" and v.detail["path"] == "sweep" and v.status == "exact"
    assert v.detail["tiling_count"] == {"structural": 230, "family": 231}


def test_tiling_that_lies_about_its_sets_is_caught_by_the_count(gemm):
    """A family whose sets are NOT what its tiling says (one coordinate dropped): the structural count disagrees
    and the sweep, not the tiling, decides."""
    t = gemm_vu_tiling(gemm)
    sets = materialise(gemm, t)[:-1]
    v = validate_partition(gemm, StubFamily(sets, tiling=t))
    assert v.status == "gap" and v.detail["path"] == "sweep"


def test_fast_path_visits_no_gates(gemm, monkeypatch):
    calls = {"n": 0}
    orig = type(gemm.circuit).gate

    def counting(self, i):
        calls["n"] += 1
        return orig(self, i)

    monkeypatch.setattr(type(gemm.circuit), "gate", counting)
    v = validate_partition(gemm, family_of(gemm, gemm_deep_tiling(gemm)))
    assert v.status == "exact" and calls["n"] == 0


def test_unsupported_limit_applies_to_fallback_only(gemm):
    class L(Limits):
        max_intervals = 4

    t = gemm_vu_tiling(gemm)
    assert validate_partition(gemm, family_of(gemm, t, with_tiling=True), L()).status == "exact"
    assert validate_partition(gemm, family_of(gemm, t, with_tiling=False), L()).status == "unsupported"


# -- width -----------------------------------------------------------------------------------------------------


@pytest.fixture
def oracle(gemm, monkeypatch):
    def install(slack: int = 0, exact_unsupported: bool = False, program=None) -> OracleBoundary:
        B = OracleBoundary(program or gemm, slack=slack, exact_unsupported=exact_unsupported)
        monkeypatch.setattr(partition, "_BOUNDARY", B)
        return B
    return install


def test_width_bound_certifies_every_class(gemm, oracle):
    B = oracle()
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert isinstance(v, WidthVerdict)
    assert v.status == "ok" and v.violators == () and v.unsupported == ()
    assert v.max_seen == 16 and v.w_max == 32
    d = v.detail
    assert d["path"] == "tiling" and d["sets"] == 230 and d["classes"] == 3
    assert d["bound_certified"] == 230 and d["exact_evaluated"] == 0
    # Input16 members: certified structurally (sum w(g) = 16), no boundary call; the 6 coordinates (sum w(g) = 112)
    # need the boundary bound, one call per member of the one root instance
    assert d["struct_certified"] == 224 and d["classes_struct_certified"] == 2
    assert B.calls["exact"] == 0 and B.calls["bound"] == 6


def test_width_bound_fails_exact_certifies(gemm, oracle):
    B = oracle(slack=20)                                             # boundary bound = 36 > 32 for every coordinate
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert v.status == "ok" and v.max_seen == 16
    assert v.detail["bound_certified"] == 224 and v.detail["exact_evaluated"] == 6 == v.detail["exact_certified"]
    assert B.calls["exact"] == 6 and v.detail["classes_exact"] == 1


def test_width_bound_above_w_max_proves_nothing(gemm, oracle):
    oracle(slack=100, exact_unsupported=True)
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert v.status == "unsupported" and v.violators == ()
    assert v.detail["unsupported_count"] == 6 and v.detail["violation_count"] == 0     # the 6 coordinates
    assert "boundary-unsupported" in v.detail["codes"]
    assert v.max_seen == 16                                                             # the Input members


def test_width_violation_exact(gemm, oracle):
    oracle()
    t = tiling(whole_body(gemm.root, {0: members_k(), 1: members_k()}))   # all six coordinates as one set: 96 bits
    v = validate_width(gemm, family_of(gemm, t), 32)
    assert v.status == "violation" and v.max_seen == 96
    assert v.detail["violation_count"] == 1 and v.detail["examples"][0]["value"] == {"w_out": 96}
    assert v.detail["examples"][0]["spec"] == gemm.root.id and v.detail["examples"][0]["node"] == 2
    # the same family without a tiling hint: per-set path, the violator is a family index
    v = validate_width(gemm, family_of(gemm, t, with_tiling=False), 32)
    assert v.status == "violation" and v.violators == (224,) and v.detail["path"] == "sets"
    # a wide enough w_max certifies it
    assert validate_width(gemm, family_of(gemm, t), 96).status == "ok"


def test_width_input_batch_as_one_set_violates(gemm, oracle):
    oracle()
    v = validate_width(gemm, family_of(gemm, tiling(whole_body(gemm.root))), 32)
    assert v.status == "violation" and v.detail["violation_count"] == 3     # x (512 bits), w (3072), coordinates (96)
    assert v.max_seen == 192 * 16
    # the cheap structural bound never certifies these (sum w(g) >= w_out), so the oracle was consulted exactly
    assert v.detail["exact_evaluated"] == 3


def test_width_exact_budget_exhausted_is_unsupported(gemm, oracle):
    class L(Limits):
        max_members_exact = 5

    oracle(slack=20)
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32, L())
    assert v.status == "unsupported" and "limit-members-exact" in v.detail["codes"]
    assert v.detail["exact_evaluated"] == 5 and v.detail["unsupported_count"] == 1


def test_width_multi_part_set_certifies_by_sum_never_refutes(gemm, oracle):
    oracle()
    sc = gemm.circuit.root_scope
    two = StubGateSet((Part(sc, frozenset({(0, 0)})), Part(sc.child(2, 0), frozenset({(1, 0)}))))   # x[0] + one cvt gate
    rest = [StubGateSet((Part(sc, frozenset({(0, m)})),)) for m in range(1, 32)]
    fam = StubFamily([two] + rest)
    assert validate_width(gemm, fam, 32).status == "ok"                  # 16 + 16 <= 32 certifies
    v = validate_width(gemm, fam, 16)
    assert v.status == "unsupported" and v.unsupported == (0,) and "multi-part-exact" in v.detail["codes"]
    assert v.violators == ()


def test_width_sets_path_limit(gemm, oracle):
    class L(Limits):
        max_intervals = 10

    oracle()
    v = validate_width(gemm, family_of(gemm, gemm_vu_tiling(gemm), with_tiling=False), 32, L())
    assert v.status == "unsupported" and "limit-sets" in v.detail["codes"] and v.detail["sets"] == 10
    assert v.detail["unsupported_count"] == 220


def test_width_zip_and_scan_classes(synth, oracle):
    oracle(program=synth)
    step = synth.root.body.nodes[4].fn
    t = tiling(whole_body(synth.root, {0: members_k(), 1: ("zip", "g"), 2: ("zip", "g"), 4: ("descend", whole_body(step))}))
    v = validate_width(synth, family_of(synth, t), 32)
    # zip member m: V32Mul[m] feeds only V32Add[m] (inside) -> out = {V32Add[m]} = 32; scan iteration gates: 32 each
    assert v.status == "ok" and v.max_seen == 32 and v.detail["classes"] == 5
    v = validate_width(synth, family_of(synth, t), 31)
    assert v.status == "violation"
    # whole nodes: the scan as one set returns carry + 4 ys = 160 bits; the Input and the two batches escape 128 each
    v = validate_width(synth, family_of(synth, tiling(whole_body(synth.root))), 64)
    assert v.status == "violation" and v.max_seen == 160 and v.detail["violation_count"] == 4


def test_width_tiling_with_problems_uses_per_set_path(gemm, oracle):
    oracle()
    t = tiling(body_tiling(gemm.root, {0: members_k(), 2: members_k()}))   # node 1 unassigned
    v = validate_width(gemm, family_of(gemm, t), 32)
    assert v.detail["path"] == "sets" and v.status == "ok"


# -- readiness ----------------------------------------------------------------------------------------------


def test_readiness_ok_on_gemm(gemm):
    r = readiness_32bit(gemm)
    assert isinstance(r, ReadinessReport) and r.ok and r.violators == () and r.max_w == 32 and r.specs == 3


def test_readiness_lists_64bit_input_and_primitive():
    P_ = wide_program()
    r = readiness_32bit(P_)
    assert not r.ok and r.max_w == 64 and r.gates_over == 2
    prims = sorted((v.prim, v.form, v.members, v.instances) for v in r.violators)
    assert prims == [("Input64_v1", "batch", 1, 1), ("TWide64_v1", "primitive", 1, 1)]
    assert all(v.spec == P_.root.id for v in r.violators)
    # 64 is a parameter: at w_max = 64 the same Program is ready
    assert gate_width_readiness(P_, 64).ok


def test_readiness_counts_instances_by_multiplicity(gemm):
    # the 32-bit gates sit in TDot, instantiated 6 times: at w_max = 16 each is counted with multiplicity 6
    r = gate_width_readiness(gemm, 16)
    over = {(v.prim.split("_")[0], v.instances) for v in r.violators}
    assert ("TZero32", 6) in over and ("TDot16", 6) in over
    assert r.gates_over == 6 * 3   # TZero32 + 2 TDot16 steps per coordinate; TCvt16 (16) and Input16 are fine


# -- composition -------------------------------------------------------------------------------------------


def test_vu_verdict_ok(gemm, oracle):
    oracle()
    v = validate_vu_family(gemm, family_of(gemm, gemm_vu_tiling(gemm)), 32)
    assert isinstance(v, VuVerdict) and v.status == "ok" and v.count == 230
    assert v.partition.status == "exact" and v.width.status == "ok" and v.readiness_32.ok
    j = v.to_json()
    assert j["status"] == "ok" and j["count"] == 230 and j["partition"]["status"] == "exact"
    import json
    json.dumps(j)  # JSON-serialisable


def test_vu_verdict_fail_and_not_ready_and_unsupported(gemm, oracle):
    oracle()
    t = tiling(whole_body(gemm.root, {0: members_k(), 1: members_k()}))
    assert validate_vu_family(gemm, family_of(gemm, t), 32).status == "fail"           # width violation
    t = tiling(body_tiling(gemm.root, {0: members_k(), 2: members_k()}))
    assert validate_vu_family(gemm, family_of(gemm, t), 32).status == "fail"           # gap
    W = wide_program()
    oracle(program=W)
    v = validate_vu_family(W, family_of(W, tiling(whole_body(W.root))), 64)
    assert v.status == "ok" and v.count == 5                                            # a 64-bit protocol
    v = validate_vu_family(W, family_of(W, tiling(whole_body(W.root))), 32)
    assert v.status == "fail" and not v.readiness_32.ok                                 # the returned 64-bit gate escapes
    # the same Program where the 64-bit gates do not escape any set: readiness alone objects
    fam = StubFamily([StubGateSet((Part(W.circuit.root_scope, frozenset({(0, 0), (1, 0)})),)),
                      StubGateSet((Part(W.circuit.root_scope, frozenset({(2, 0), (3, 0), (4, 0)})),))])
    v = validate_vu_family(W, fam, 64)
    assert v.status == "ok"
    v = validate_vu_family(W, fam, 96)
    assert v.status == "ok"
    assert not gate_width_readiness(W, 32).ok

    class L(Limits):
        max_intervals = 2

    oracle()
    v = validate_vu_family(gemm, family_of(gemm, gemm_vu_tiling(gemm), with_tiling=False), 32, L())
    assert v.status == "unsupported" and v.partition.status == "unsupported"


def test_vu_status_precedence():
    ok_p = PartitionVerdict("exact", (), {})
    gap_p = PartitionVerdict("gap", (), {})
    un_p = PartitionVerdict("unsupported", (), {})
    ok_w = WidthVerdict(32, 16, (), (), "ok")
    bad_w = WidthVerdict(32, 96, (0,), (), "violation")
    un_w = WidthVerdict(32, None, (), (0,), "unsupported")
    ready = ReadinessReport(32, True, (), 32, 1, 0)
    unready = ReadinessReport(32, False, (), 64, 1, 1)
    assert vu_status(ok_p, ok_w, ready) == "ok"
    assert vu_status(gap_p, ok_w, ready) == "fail"
    assert vu_status(ok_p, bad_w, unready) == "fail"
    assert vu_status(ok_p, ok_w, unready) == "not-ready"
    assert vu_status(un_p, ok_w, unready) == "not-ready"
    assert vu_status(un_p, ok_w, ready) == "unsupported"
    assert vu_status(ok_p, un_w, ready) == "unsupported"


# -- the interval sweep (tier B) ---------------------------------------------------------------------------


def fam(*sets: tuple[tuple[int, int], ...]) -> StubFamily:
    return StubFamily([IntervalSet(tuple(s)) for s in sets])


def test_exact_partition():
    v = validate_partition(FakeProgram(10), fam(((0, 3),), ((3, 10),)))
    assert isinstance(v, PartitionVerdict)
    assert v.status == "exact" and v.codes == ()
    assert v.detail["path"] == "sweep" and v.detail["sets"] == 2 and v.detail["intervals"] == 2 and v.detail["gates"] == 10


def test_exact_partition_out_of_order_and_multi_interval_sets():
    # zip-like sets: {[0,2),[4,6)} and {[2,4),[6,8)}, given in reverse order
    v = validate_partition(FakeProgram(8), fam(((6, 8), (2, 4)), ((4, 6), (0, 2))))
    assert v.status == "exact"


def test_overlap():
    v = validate_partition(FakeProgram(10), fam(((0, 5),), ((3, 10),)))
    assert v.status == "overlap" and v.codes == ("sweep-overlap",)
    assert v.detail["overlaps"] == [{"gates": [3, 5], "sets": [0, 1]}]
    assert v.detail["overlap_count"] == 1


def test_gap_interior_and_trailing():
    v = validate_partition(FakeProgram(10), fam(((0, 3),), ((5, 8),)))
    assert v.status == "gap" and v.codes == ("sweep-gap",)
    assert v.detail["gaps"] == [{"gates": [3, 5]}, {"gates": [8, 10]}]
    assert v.detail["gap_count"] == 2


def test_leading_gap():
    v = validate_partition(FakeProgram(4), fam(((1, 4),)))
    assert v.status == "gap" and v.detail["gaps"] == [{"gates": [0, 1]}]


def test_overlap_wins_over_gap_both_codes_listed():
    v = validate_partition(FakeProgram(10), fam(((0, 4),), ((2, 6),), ((8, 10),)))
    assert v.status == "overlap"
    assert v.codes == ("sweep-overlap", "sweep-gap")
    assert v.detail["gaps"] == [{"gates": [6, 8]}]


def test_duplicate_set_is_overlap():
    v = validate_partition(FakeProgram(4), fam(((0, 4),), ((0, 4),)))
    assert v.status == "overlap" and v.detail["overlaps"][0]["gates"] == [0, 4]


def test_nested_interval_overlap():
    v = validate_partition(FakeProgram(10), fam(((0, 10),), ((3, 5),)))
    assert v.status == "overlap" and v.detail["overlaps"] == [{"gates": [3, 5], "sets": [0, 1]}]


def test_empty_family_is_a_gap():
    v = validate_partition(FakeProgram(3), fam())
    assert v.status == "gap" and v.detail["gaps"] == [{"gates": [0, 3]}]


def test_zero_gate_program_with_empty_family_is_exact():
    assert validate_partition(FakeProgram(0), fam()).status == "exact"


def test_empty_set_is_reported_but_does_not_break_partitionhood():
    v = validate_partition(FakeProgram(4), fam(((0, 4),), ()))
    assert v.status == "exact" and v.codes == ("empty-set",)
    assert v.detail["empty_sets"] == [1] and v.detail["empty_count"] == 1


def test_unsupported_by_limit_is_never_a_verdict():
    class L(Limits):
        max_intervals = 3

    sets = [((i, i + 1),) for i in range(5)]
    v = validate_partition(FakeProgram(5), fam(*sets), L())
    assert v.status == "unsupported" and v.codes == ("limit-intervals",)
    assert v.detail["max_intervals"] == 3 and v.detail["intervals_seen"] == 4
    # the same family, under the default limits, is exact
    assert validate_partition(FakeProgram(5), fam(*sets)).status == "exact"
    # and an overlapping family beyond the limit is also unsupported, not "overlap"
    bad = [((0, 5),)] + sets
    assert validate_partition(FakeProgram(5), fam(*bad), L()).status == "unsupported"


def test_interval_outside_program_is_an_error_not_a_verdict():
    with pytest.raises(ValueError):
        validate_partition(FakeProgram(4), fam(((0, 5),)))
    with pytest.raises(ValueError):
        validate_partition(FakeProgram(4), fam(((-1, 2),)))


def test_many_intervals_sweep_is_fast_enough():
    n = 20_000
    sets = [((2 * i, 2 * i + 1), (2 * i + 1, 2 * i + 2)) for i in range(n)]
    v = validate_partition(FakeProgram(2 * n), fam(*sets))
    assert v.status == "exact" and v.detail["intervals"] == 2 * n


# -- the sweep over real Parts -------------------------------------------------------------------------


def test_sweep_over_gemm_parts_exact_and_gap():
    P_ = gemm_program()
    sc = P_.circuit.root_scope
    sets = [StubGateSet((Part(sc, frozenset({(0, m)})),)) for m in range(32)]          # every Input16 x gate
    sets += [StubGateSet((Part(sc, frozenset({(1, m) for m in range(192)})),))]        # w as one set
    sets += [StubGateSet((Part(sc, frozenset({(2, m)})),)) for m in range(6)]          # each GemmCoordinate
    v = validate_partition(P_, StubFamily(sets))
    assert v.status == "exact" and v.detail["sets"] == 39 and v.detail["intervals"] == 39
    # drop one coordinate -> a 4-gate gap at that member's interval
    v = validate_partition(P_, StubFamily(sets[:-1]))
    assert v.status == "gap" and v.detail["gaps"] == [{"gates": [244, 248]}]


def test_sweep_over_nested_parts_and_inner_overlap():
    P_ = gemm_program()
    sc = P_.circuit.root_scope
    inner = sc.child(2, 3)                                        # GemmCoordinate#3
    dot = inner.child(0, 0)                                       # its TDot call
    sets = [StubGateSet((Part(sc, frozenset({(0, m) for m in range(32)})),)),
            StubGateSet((Part(sc, frozenset({(1, m) for m in range(192)})),))]
    sets += [StubGateSet((Part(sc, frozenset({(2, m)})),)) for m in range(6) if m != 3]
    sets += [StubGateSet((Part(dot, frozenset({(j, 0)})),)) for j in range(3)]         # TZero32, 2 TDot16 steps
    sets += [StubGateSet((Part(inner, frozenset({(1, 0)})),))]                        # TCvt16
    assert validate_partition(P_, StubFamily(sets)).status == "exact"
    # the whole coordinate again on top of its pieces -> overlap inside the nested body
    v = validate_partition(P_, StubFamily(sets + [StubGateSet((Part(sc, frozenset({(2, 3)})),))]))
    assert v.status == "overlap" and v.detail["overlap_count"] >= 1


def test_sweep_over_synth_zip_and_scan_iterations():
    P_ = synth_program(4)
    sc = P_.circuit.root_scope
    sets = [StubGateSet((Part(sc, frozenset({(0, m)})),)) for m in range(4)]
    sets += [StubGateSet((Part(sc, frozenset({(1, m), (2, m)})),)) for m in range(4)]  # zip of the two batches
    sets += [StubGateSet((Part(sc, frozenset({(3, 0)})),))]
    sets += [StubGateSet((Part(sc, frozenset({(4, m)})),)) for m in range(4)]          # scan iterations
    v = validate_partition(P_, StubFamily(sets))
    assert v.status == "exact" and v.detail["intervals"] == 4 + 8 + 1 + 4


def test_sweep_matches_materialised_tiling_without_tiling_hint():
    P_ = gemm_program()
    t = tiling(whole_body(P_.root, {0: members_k(), 2: members_k()}))
    v = validate_partition(P_, family_of(P_, t, with_tiling=False))
    assert v.status == "exact" and v.detail["path"] == "sweep" and v.detail["sets"] == 32 + 1 + 6


# -- the real boundary and query_ast families ------------------------------------------------------------------


def test_width_with_the_real_boundary_agrees_with_the_oracle(gemm, synth, monkeypatch):
    """`validate_width` through `verity.ml.circuits.boundary` returns the same verdict as through the explicit oracle (whose
    bound is exact, so its `max_seen` is never above the real one's)."""
    cases = [(gemm, gemm_vu_tiling(gemm)), (gemm, gemm_deep_tiling(gemm)), (gemm, tiling(whole_body(gemm.root))),
             (gemm, tiling(whole_body(gemm.root, {0: members_k(), 1: members_k()}))),
             (synth, tiling(whole_body(synth.root, {0: members_k(), 1: ("zip", "g"), 2: ("zip", "g"),
                                                    4: ("descend", whole_body(synth.root.body.nodes[4].fn))}))),
             (synth, tiling(whole_body(synth.root)))]
    for P_, t in cases:
        for w_max in (16, 32, 96):
            for with_tiling in (True, False):
                monkeypatch.setattr(partition, "_BOUNDARY", None)
                real = validate_width(P_, family_of(P_, t, with_tiling=with_tiling), w_max)
                monkeypatch.setattr(partition, "_BOUNDARY", OracleBoundary(P_))
                oracle = validate_width(P_, family_of(P_, t, with_tiling=with_tiling), w_max)
                key = (P_.root.id, w_max, with_tiling)
                assert (real.status, real.violators, real.unsupported) == (oracle.status, oracle.violators, oracle.unsupported), key
                assert real.max_seen >= oracle.max_seen, key
    monkeypatch.setattr(partition, "_BOUNDARY", None)
    assert partition._boundary() is boundary


def test_query_ast_default_family_is_a_valid_vu_partition(gemm):
    fam = G.evaluate(gemm, G.default(32))
    assert fam.count() == 32 + 192 + 6
    v = validate_vu_partition(gemm, G.default(32))
    assert v.status == "ok" and v.count == 230
    assert v.partition.status == "exact" and v.partition.detail["path"] == "tiling"
    assert v.width.status == "ok" and v.width.max_seen == 16
    swept = validate_partition(gemm, StubFamily(list(fam)))
    assert swept.status == "exact" and swept.detail["path"] == "sweep" and swept.detail["sets"] == 230


def test_query_ast_family_with_a_gap_and_an_overlap(synth):
    rest = G.evaluate(synth, G.default(32))
    assert validate_partition(synth, rest).status == "exact"
    sets = list(rest)
    assert validate_partition(synth, StubFamily(sets[1:])).status == "gap"
    assert validate_partition(synth, StubFamily(sets + sets[:1])).status == "overlap"


# -- hierarchies: subset cover, refinement, members per parent ----------------------------------------------


def test_within_partitions_a_subset_of_gates(gemm):
    """The coordinates alone partition the computed gates [224, 248), leaving the Inputs out."""
    vu = materialise(gemm, gemm_vu_tiling(gemm))
    coords = [s for s in vu if s.intervals()[0][0] >= 224]
    assert len(coords) == 6
    v = validate_partition(gemm, StubFamily(coords), within=[(224, 248)])
    assert v.status == "exact" and v.codes == () and v.detail["within"] == 24 and v.detail["path"] == "sweep"
    as_gateset = validate_partition(gemm, StubFamily(coords), within=IntervalSet(((224, 236), (236, 248))))
    assert as_gateset.status == "exact"
    gap = validate_partition(gemm, StubFamily(coords[1:]), within=[(224, 248)])
    assert gap.status == "gap" and gap.detail["gaps"] == [{"gates": [224, 228]}]
    out = validate_partition(gemm, StubFamily(vu), within=[(224, 248)])
    assert out.status == "outside" and out.codes == ("sweep-outside",)
    assert out.detail["outside"] == [{"gates": [0, 224]}] and out.detail["outside_count"] == 1
    both = validate_partition(gemm, StubFamily(coords + coords[:1]), within=[(224, 248)])
    assert both.status == "overlap"
    assert vu_status(out, WidthVerdict(32, 16, (), (), "ok"), gate_width_readiness(gemm, 32)) == "fail"
    with pytest.raises(ValueError):
        validate_partition(gemm, StubFamily(coords), within=[(224, 249)])


def test_within_the_whole_program_agrees_with_the_full_check(gemm):
    for t in (gemm_vu_tiling(gemm), gemm_deep_tiling(gemm)):
        sets = materialise(gemm, t)
        for fam in (sets, sets[1:], sets + sets[:1]):
            full = validate_partition(gemm, StubFamily(fam))
            within = validate_partition(gemm, StubFamily(fam), within=[(0, gemm.gates)])
            assert (within.status, within.codes) == (full.status, full.codes)
            assert {k: v for k, v in within.detail.items() if k != "within"} == full.detail


def test_refinement_counts_members_per_parent(gemm):
    whole = family_of(gemm, tiling(whole_body(gemm.root)))
    vu = family_of(gemm, gemm_vu_tiling(gemm))
    deep = family_of(gemm, gemm_deep_tiling(gemm))
    assert partition.members_per_parent(gemm, whole, vu) == (32, 192, 6)
    assert partition.members_per_parent(gemm, whole, deep) == (1, 1, 24)
    coords = StubFamily([s for s in vu if s.intervals()[0][0] >= 224])
    inside = StubFamily([s for s in deep if s.intervals()[0][0] >= 224])
    v = partition.validate_refinement(gemm, coords, inside)
    assert v.status == "ok" and v.counts == (4,) * 6 and v.detail == {"coarse_sets": 6, "fine_sets": 24}
    assert partition.validate_refinement(gemm, vu, deep).codes == ("refine-crosses",)   # deep keeps Inputs whole


def test_refinement_on_query_families_and_unequal_sizes(gemm):
    """`each(F, G)` nests by construction; per-parent counts may differ (3 TDot gates, then 1 TCvt16)."""
    coords = G.evaluate(gemm, G.instances("//TGemmCoordinate"))
    subs = G.evaluate(gemm, G.each(G.instances("//TGemmCoordinate"), G.default(32)))
    gates = G.evaluate(gemm, G.each(G.instances("//TGemmCoordinate"), G.default(0)))
    assert partition.members_per_parent(gemm, coords, subs) == (2,) * 6
    assert partition.members_per_parent(gemm, subs, gates) == (3, 1) * 6
    assert partition.validate_refinement(gemm, gates, subs).codes == ("refine-crosses",)


def test_crossing_outside_and_empty_sets_are_not_refinements(gemm):
    whole = family_of(gemm, tiling(whole_body(gemm.root)))
    vu = family_of(gemm, gemm_vu_tiling(gemm))
    crosses = partition.validate_refinement(gemm, vu, whole)
    assert crosses.status == "violation" and crosses.codes == ("refine-crosses",) and crosses.counts == ()
    assert crosses.detail["refine-crosses"] == {"count": 3, "sets": [0, 1, 2]}
    coords_only = StubFamily([s for s in vu if s.intervals()[0][0] >= 224])
    outside = partition.validate_refinement(gemm, coords_only, vu)
    assert outside.codes == ("refine-outside",) and outside.detail["refine-outside"]["count"] == 224
    straddle = StubFamily([IntervalSet(((0, 20),)), IntervalSet(((20, 248),))])
    halves = StubFamily([IntervalSet(((0, 10), (30, 40))), IntervalSet(((10, 30),)), IntervalSet(((40, 248),))])
    assert partition.validate_refinement(gemm, straddle, halves).codes == ("refine-crosses",)
    assert partition.validate_refinement(gemm, halves, StubFamily([IntervalSet(((5, 8), (32, 35)))])).counts == (1, 0, 0)
    empty = partition.validate_refinement(gemm, whole, StubFamily([IntervalSet(())]))
    assert empty.codes == ("refine-empty",)
    overlap = partition.validate_refinement(gemm, StubFamily(list(whole) + list(whole)[:1]), vu)
    assert overlap.codes == ("refine-coarse-overlap",)
    with pytest.raises(ValueError, match="not a refinement"):
        partition.members_per_parent(gemm, vu, whole)


def test_refinement_is_unsupported_beyond_the_interval_limit(gemm):
    def cap(n):
        class L(Limits):
            max_intervals = n
        return L()

    whole = family_of(gemm, tiling(whole_body(gemm.root)))
    vu = family_of(gemm, gemm_vu_tiling(gemm))
    assert partition.validate_refinement(gemm, whole, vu, cap(100)).status == "unsupported"
    assert partition.validate_refinement(gemm, vu, whole, cap(100)).codes == ("limit-intervals",)
    assert partition.validate_refinement(gemm, whole, vu, cap(233)).status == "ok"
