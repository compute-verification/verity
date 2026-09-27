"""Per-unit circuits over any partition core evaluates, for C-Flock's circuit prover (prototype; see the Project doc
``docs/workstreams.md``, "The backend over an arbitrary partition").

    uv run python -m verity_flock.partition_units --model tiny [--query '{"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}}']
    uv run python -m verity_flock.partition_units --definition 'RoPE_v1{NHEADS=32,D=64}' --vectors 32

Given a Program and a query that ``verity.ir.partition_object.evaluate`` applies (``Q_word`` v1 or a template query), every unit
becomes a standalone word circuit (:class:`UnitCircuit`):

* its gates are the computing gates the partition gives it, plus every structure gate (constant or ``cut.WIRING``) they read
  through, which the unit's statement hardwires;
* its inputs are the values it reads from outside itself, in first-read order: its Call's parameter leaves, or computing gates of
  another unit of the same Call (both committed under the partition invariant);
* its outputs are its committed gates (``cut.derived_committed``), ascending.

Units whose relative form (primitives, operand wiring, input widths, outputs) is equal share a *shape*, the SHA-256 of that form,
so one lowering serves the class. :func:`lower_class` lowers a class with C-Flock's pieces (``ir_lower.lower_gates``);
:func:`check` evaluates each Call of the Program on synthetic words (every word is a valid input of a total circuit), slices each
unit's boundary values out of the Call's transcript, and runs every unit of every lowered class through its class circuit.

A Program's Calls are the units' outer limit: ``Q_word`` v1 cuts each Call as its Definition's activation, and the template queries
make whole instances units, so no unit here spans two Calls. The extractor is per Definition, like ``Q_word``: activations of one
Definition share their units' circuits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from verity.ir import partition_object as PO
from verity.ir.codec import canonical_json
from verity.ir.cut import CallGraph, call_scope, derived_committed
from verity.randomness import derive

from . import gf2
from . import ir_lower as IL

SOURCE = b"verity partition-units synthetic"
DOMAIN = "verity/flock/partition-units/v0"


@dataclass(frozen=True)
class UnitCircuit:
    """One unit's circuit inside its Call's Definition: Call-local gate indices (the Definition's own gates are ``0 ..
    fn.gates - 1``; an input ``-1 - p`` is the Call's parameter leaf ``p``)."""

    definition: str
    unit: int                      # the unit within its Definition's cut
    gates: tuple[int, ...]         # computing and structure gates, canonical order
    computing: int                 # how many of them the partition assigns to the unit
    inputs: tuple[int, ...]        # first-read order
    outputs: tuple[int, ...]       # committed, ascending
    in_widths: tuple[int, ...]
    out_widths: tuple[int, ...]
    shape: str


class _Activation:
    """One Definition's one-Call Program (``cut.call_scope``): its gates, parameter leaves and ``CallGraph``."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.circuit, call = call_scope(fn)
        self.lo = call.offset
        if self.lo != sum(t.leaves for _, t in fn.signature()[0]):
            raise AssertionError(f"{fn.id}: the one-Call Program does not start with exactly its parameter leaves")
        self.graph = CallGraph(fn)
        self._gates: dict[int, Any] = {}

    def gate(self, j: int) -> Any:
        g = self._gates.get(j)
        if g is None:
            g = self._gates[j] = self.circuit.gate(self.lo + j)
        return g

    def local(self, o: int) -> int:
        """An operand (absolute gate index) as a Call-local index, a parameter leaf as ``-1 - p``."""
        return o - self.lo if o >= self.lo else -1 - o

    def width(self, j: int) -> int:
        return (self.gate(j) if j >= 0 else self.circuit.gate(-1 - j)).prim.ret.w

    def evaluate(self, words: Sequence[int]) -> dict[int, int]:
        """Every gate's value on the parameter leaves ``words`` (Call-local keys; parameters as ``-1 - p``)."""
        t: dict[int, int] = {-1 - p: int(w) for p, w in enumerate(words)}
        for j in range(self.fn.gates):
            g = self.gate(j)
            ops = [self.local(o) for o in g.operands]
            lazy = getattr(g.prim, "evaluate_lazy", None)
            t[j] = lazy(lambda q, _o=ops: t[_o[q]]) if lazy is not None else g.prim.evaluate(*(t[o] for o in ops))
        return t


def _extract(act: _Activation, owner: Sequence[int], committed: set[int], u: int) -> UnitCircuit:
    g = act.graph
    keep = g.keep
    structure = set(range(act.fn.gates)) - set(keep)
    owned = {keep[c] for c in range(g.n) if owner[c] == u}
    members: set[int] = set()
    stack = sorted(owned)
    while stack:
        j = stack.pop()
        if j in members:
            continue
        members.add(j)
        for o in act.gate(j).operands:
            k = act.local(o)
            if k >= 0 and k in structure and k not in members:
                stack.append(k)
    gates = tuple(sorted(members))
    local = {j: i for i, j in enumerate(gates)}
    port: dict[int, int] = {}
    form = []
    for j in gates:
        refs = []
        for o in act.gate(j).operands:
            k = act.local(o)
            refs.append(("g", local[k]) if k in local else ("i", port.setdefault(k, len(port))))
        form.append([act.gate(j).prim.id, refs])
    inputs = sorted(port, key=port.__getitem__)
    outputs = tuple(sorted(keep[c] for c in committed if owner[c] == u))
    in_w = tuple(act.width(k) for k in inputs)
    out_w = tuple(act.width(j) for j in outputs)
    shape = hashlib.sha256(canonical_json({"in": list(in_w), "gates": form, "out": [local[j] for j in outputs]})).hexdigest()
    return UnitCircuit(act.fn.id, u, gates, len(owned), tuple(inputs), outputs, in_w, out_w, shape)


@dataclass(frozen=True)
class Placement:
    """Unit ``index`` of the partition: root-body node ``node``, batch member ``member``, its Definition's unit ``unit``."""

    index: int
    node: int
    member: int
    circuit: UnitCircuit


class PartitionUnits:
    """Every unit of ``query`` applied to ``program`` as a :class:`UnitCircuit`, per Definition (activations share them)."""

    def __init__(self, program: Any, query: Mapping[str, Any]) -> None:
        self.program = program
        self.query = dict(query)
        self.partition = PO.evaluate(program, query)
        self._acts: dict[str, _Activation] = {}
        self._circuits: dict[str, list[UnitCircuit]] = {}

    def activation(self, fn: Any) -> _Activation:
        a = self._acts.get(fn.id)
        if a is None:
            a = self._acts[fn.id] = _Activation(fn)
        return a

    def circuits(self, fn: Any, owner: Sequence[int] | None = None) -> list[UnitCircuit]:
        """The Definition's units in cut order (``owner``: a unit per computing gate; None: the whole activation is one unit)."""
        got = self._circuits.get(fn.id)
        if got is None:
            act = self.activation(fn)
            owner = [0] * act.graph.n if owner is None else list(owner)
            committed = set(derived_committed(act.graph, owner))
            units = (max(owner) + 1) if owner else 0
            got = self._circuits[fn.id] = [_extract(act, owner, committed, u) for u in range(units)]
        return got

    def __iter__(self) -> Iterator[Placement]:
        part, nodes = self.partition, self.program.root.body.nodes
        i = 0
        if isinstance(part, PO.TemplatePartition):
            for k, _tid, _first, n in part.ranges:
                (uc,) = self.circuits(nodes[k].fn)
                for m in range(n):
                    yield Placement(i, k, m, uc)
                    i += 1
            return
        for p, (k, cut) in enumerate(part.calls):
            for uc in self.circuits(cut.fn, part.owner(p)):
                yield Placement(i, k, 0, uc)
                i += 1


@dataclass
class ClassCircuit:
    shape: str
    circuit: gf2.Circuit | None
    counts: dict | None
    missing: tuple[str, ...] = ()
    reason: str = ""


def lower_class(units: PartitionUnits, uc: UnitCircuit, max_gates: int = 4096) -> ClassCircuit:
    """C-Flock's Boolean lowering of one unit (inputs ``in0 ..``, outputs ``out0 ..``, one word per port), or why not."""
    act = units._acts[uc.definition]
    prims = {act.gate(j).prim.id for j in uc.gates}
    missing = tuple(sorted(p for p in prims if p not in IL.PIECES and not p.startswith(("Const", "Input"))))
    if missing:
        return ClassCircuit(uc.shape, None, None, missing, "no piece")
    if len(uc.gates) > max_gates:
        return ClassCircuit(uc.shape, None, None, (), f"more than {max_gates} word gates")
    C = gf2.Circuit()
    wires: dict[int, list] = {}
    for t, (k, w) in enumerate(zip(uc.inputs, uc.in_widths)):
        wires[k + act.lo if k >= 0 else -1 - k] = C.input(f"in{t}", w)
    try:
        IL.lower_gates(C, [act.gate(j) for j in uc.gates], wires)
    except IL.NoPiece as e:
        return ClassCircuit(uc.shape, None, None, (), str(e))
    for t, j in enumerate(uc.outputs):
        C.output(f"out{t}", wires[j + act.lo])
    return ClassCircuit(uc.shape, C, C.counts())


def synthetic_words(key: Any, node: int, member: int, widths: Sequence[int], T: int) -> list[list[int]]:
    """T draws of a Call's parameter leaves, uniform words of each leaf's width, from the counter stream at (node, member, t)."""
    out = []
    nbytes = [(w + 7) // 8 for w in widths]
    for t in range(T):
        raw = key.stream(("call", node, member, t), sum(nbytes))
        words, at = [], 0
        for w, nb in zip(widths, nbytes):
            words.append(int.from_bytes(raw[at:at + nb], "little") & ((1 << w) - 1))
            at += nb
        out.append(words)
    return out


def class_mismatches(C: gf2.Circuit, uc: UnitCircuit, ins: Sequence[Sequence[int]], want: Sequence[Sequence[int]]) -> int:
    """Output words of the class circuit C that differ from ``want`` on the lanes ``ins`` (per port, one word per lane)."""
    n = len(want[0]) if want else 0
    got, _fail = C.evaluate({f"in{i}": IL.slice_words(v, w) for i, (v, w) in enumerate(zip(ins, uc.in_widths))}, n)
    return sum(int((IL.unslice_words(got[f"out{i}"], n) != np.asarray(v, dtype=np.uint64)).sum()) for i, v in enumerate(want))


def check(units: PartitionUnits, T: int = 16, seed: int = 0, max_gates: int = 4096) -> dict[str, Any]:
    """Lower one unit per shape class, then run every unit of every lowered class, in every Call, on its boundary values sliced
    from the Call's reference transcript on T synthetic draws; a summary with the mismatches (expected 0)."""
    t0 = time.time()
    placements = list(units)
    extract_s = time.time() - t0
    by_shape: dict[str, UnitCircuit] = {}
    n_by_shape: Counter = Counter()
    for pl in placements:
        by_shape.setdefault(pl.circuit.shape, pl.circuit)
        n_by_shape[pl.circuit.shape] += 1
    t1 = time.time()
    classes = {s: lower_class(units, uc, max_gates) for s, uc in by_shape.items()}
    lower_s = time.time() - t1
    key = derive(SOURCE, DOMAIN, {"program": PO.program_sha512(units.program)[:32], "seed": int(seed)})
    nodes = units.program.root.body.nodes
    t2 = time.time()
    by_call: dict[tuple[int, int], list[Placement]] = {}
    for pl in placements:
        if classes[pl.circuit.shape].circuit is not None:
            by_call.setdefault((pl.node, pl.member), []).append(pl)
    lanes: dict[str, tuple[list[list[int]], list[list[int]]]] = {}    # shape -> (per input port, per output port) words, one per lane
    for (k, m), pls in sorted(by_call.items()):
        act = units.activation(nodes[k].fn)
        widths = [act.circuit.gate(p).prim.ret.w for p in range(act.lo)]
        runs = [act.evaluate(w) for w in synthetic_words(key, k, m, widths, T)]
        for pl in pls:
            uc = pl.circuit
            ins, outs = lanes.setdefault(uc.shape, ([[] for _ in uc.inputs], [[] for _ in uc.outputs]))
            for port, x in zip(ins, uc.inputs):
                port.extend(r[x] for r in runs)
            for port, j in zip(outs, uc.outputs):
                port.extend(r[j] for r in runs)
    mismatches, lanes_checked = 0, 0
    for s, (ins, want) in lanes.items():
        mismatches += class_mismatches(classes[s].circuit, by_shape[s], ins, want)
        lanes_checked += len(want[0]) if want else 0
    checked = sum(len(p) for p in by_call.values())
    check_s = time.time() - t2
    lowered = [s for s, c in classes.items() if c.circuit is not None]
    missing = Counter(p for s, c in classes.items() for p in c.missing for _ in range(n_by_shape[s]))
    ands = {s: classes[s].counts["ands"] for s in lowered}
    return {
        "query": units.query,
        "partition": PO.digest(PO.build(units.program, units.query)),
        "units": len(placements),
        "computing_gates": sum(pl.circuit.computing for pl in placements),
        "shapes": len(by_shape),
        "shapes_lowered": len(lowered),
        "units_lowered": sum(n_by_shape[s] for s in lowered),
        "units_checked": checked,
        "lanes_checked": lanes_checked,
        "vectors": T,
        "mismatches": mismatches,
        "ands_lowered_units": sum(ands[s] * n_by_shape[s] for s in lowered),
        "largest_class_ands": max(ands.values(), default=0),
        "not_lowered": {"no piece (units per primitive)": dict(missing.most_common()),
                        "other": dict(Counter(c.reason for s, c in classes.items() if c.circuit is None and not c.missing))},
        "seconds": {"extract": round(extract_s, 3), "lower": round(lower_s, 3), "check": round(check_s, 3)},
    }


def _program(a: argparse.Namespace) -> Any:
    from verity.ir.defs import bind
    from verity.ir.program import Program

    from circuit_check import targets as TG

    TG.load_registries()
    if a.model == "tiny":
        from verity_vllm.program.registry import b1
        return Program(bind(b1.Serve, C=TG.TINY, LP=3, STEPS=2))
    return _one_call(TG.definition(a.definition))


def _one_call(fn: Any) -> Any:
    from verity.ir.defs import CompositeDefinition
    from verity.ir.program import Program

    params, ret = fn.signature()
    shell = CompositeDefinition(f"OneCall[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *x: B.call(fn, *x),
                                register=False)
    return Program(shell)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", choices=["tiny"], help="circuit-check's tiny Serve_v1 Program (38 Calls)")
    src.add_argument("--definition", help="a registered Definition id, as a one-Call Program (circuit-check's parser)")
    ap.add_argument("--query", default=json.dumps(PO.query(16, 32)), help="a partition query (JSON)")
    ap.add_argument("--vectors", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gates", type=int, default=4096, help="lower a class only up to this many word gates")
    a = ap.parse_args(argv)
    program = _program(a)
    units = PartitionUnits(program, json.loads(a.query))
    json.dump(check(units, a.vectors, a.seed, a.max_gates), sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
