"""In-test stand-ins for the partition tests (qpart-owned).  Until `gateset.py` / `boundary.py` are merged the
partition validator is exercised against:

* `StubGateSet` / `StubFamily` -- the `GateSet` / `Family` shape of the freeze doc, built from explicit `Part`s;
* `BodyTiling` / `Tiling` -- the structural description `validate_partition`'s fast path reads (INTERFACE NOTE in
  `out/gen/window/qpart.md`); `materialise` turns a tiling into the family it induces, so the fast path can be checked
  against the interval sweep on the same sets;
* `OracleBoundary` -- `w_out` from the explicit oracle `GateSelection.out` (gate widths from the producing primitive),
  with an adjustable slack on the bound and an optional "exact is Unsupported" mode;
* two small Programs: `gemm_program()` (`Gemm_v1{K=32,N=6}`: Input batches, a composite batch, nested calls) and
  `synth_program(N)` (a pair of zip-able batches, a constant, a scan).

Nothing here is copied from the other owners' modules; the real modules replace these stubs at merge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.layout import Scope, GateSelection
from verity.ml.circuits.parts import Part, Unsupported, n_members
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16, F32
from verity.ml.circuits.types import Array, Tuple, Value


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


# The Query API owner's structures (qapi INTERFACE NOTE, 21:20Z), same field names, so the real classes are drop-in.
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
    """Root: Input16 x (32), Input16 w (192), batch GemmCoordinate_v1{K=32} x 6 -> 248 gates.
    GemmCoordinate body: call DotBf16 (3 gates: Const32 + 2 Ampere steps), F2fpBf16."""
    return Program(bind(b1.Gemm, K=32, N=6))


_ScanStep = CompositeDefinition(
    "QpartScanStep", 1, (), lambda S: ((("carry", F32), ("x", F32)), Tuple(F32, F32)),
    lambda B, S, carry, x: (lambda c: tuple_of(c, B.call(P.F32Mul, c, x)))(B.call(P.F32Add, carry, x)),
    register=False)


def _synth_body(B, S, x):
    sq = B.batch(P.F32Mul, x, x, axes=(0, 0))          # node 1: n=N
    s = B.batch(P.F32Add, sq, x, axes=(0, 0))          # node 2: n=N, member m reads sq[m]
    z = B.call(b1.ZERO32)                              # node 3: constant
    return B.scan(_ScanStep, z, xs=[s])                # node 4: N iterations of 2 gates


_Synth = CompositeDefinition("QpartSynth", 1, ("N",),
                             lambda S: ((("x", Array(S.N, F32)),), Tuple(F32, Array(S.N, F32))),
                             _synth_body, register=False)


def synth_program(N: int = 4) -> Program:
    """Root: Input32 x (N) | batch F32Mul (N) | batch F32Add (N) | Const32 | scan QpartScanStep (N x 2) ->
    3N + 1 + 2N gates.  Nodes 1 and 2 are zip-able (equal n, member m of 2 reads member m of 1)."""
    return Program(bind(_Synth, N=N))


_Twice = CompositeDefinition(
    "QpartTwice", 1, (), lambda S: ((("x", Array(32, BF16)), ("w", Array(32, Array(32, BF16)))), Array(32, BF16)),
    lambda B, S, x, w: B.call(bind(b1.Gemm, K=32, N=32), B.call(bind(b1.Gemm, K=32, N=32), x, w), w),
    register=False)


def twice_program() -> Program:
    """Root: Input16 x (32), Input16 w (1024), call Gemm{K=32,N=32} (node 2), call Gemm{K=32,N=32} (node 3, fed by node 2).
    The same specialization twice from one body: a shared callee for the multiplicity / plan-key tests."""
    return Program(bind(_Twice))


_Wide64 = PrimitiveDefinition("QpartWide64", 1, (), Value(64), lambda: 0, register=False)


def wide_program() -> Program:
    """A Program with a 64-bit Input and a 64-bit primitive gate (readiness violators of both kinds)."""
    W = CompositeDefinition("QpartWide", 1, (), lambda S: ((("seed", Value(64)),), Tuple(Value(64), F32)),
                            lambda B, S, seed: tuple_of(B.call(_Wide64), B.call(P.F32Add, B.call(b1.ZERO32), B.call(b1.ZERO32))),
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
