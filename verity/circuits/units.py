"""Each unit of a partition as a standalone word circuit, and the shape that units with the same circuit share.

A unit's circuit (`ProofBlockCircuit`) lives inside one activation of a *flat* Definition: a Call's Definition, or, for a Definition
`Q_word` cuts member by member (`cut.separable`), the member's callee.  Gates are numbered as `cut.call_scope` numbers them: the
Definition's own gates ``0 .. gates - 1`` in canonical order, its parameter leaf ``p`` as ``-1 - p``.

* **gates**: the computing gates the partition assigns the unit (`cut.CallGraph` order), plus every structure gate (a constant,
  a gate computed from constants alone, or a `cut.WIRING` primitive) they read, directly or through other structure gates: the
  statement hardwires structure, so each unit reading it carries its own copy;
* **inputs**: the values the unit reads from outside itself, in first-read order (gates in canonical order, operands in order):
  a parameter leaf of the activation, or a computing gate of another unit (committed under the partition invariant);
* **outputs**: the unit's committed gates, ascending (`cut.derived_committed`: another unit reads it, or the activation returns
  it); a unit with none (a dead unit) has an empty circuit;
* **shape**: `shape_digest` of ``canon({"in": [input widths], "gates": [[primitive id, [["g", local] | ["i", input]], ...], ...],
  "out": [local, ...]})``, ``local`` a gate's position in ``gates``: ``SHA-512("verity/partition/v2/shape\0" || form)``.  Equal shapes are equal circuits of equal interface, so one
  lowering and one statement serve a class.  A constant's primitive id carries its value (``Const32[0x3f800000]_v1``).

`PartitionUnits` gives every unit of a program's partition in the partition's own numbering (by Call, then within the Call's
cut), either for a query `partition_object.evaluate` applies (``Q_word`` v1, the template queries) or for an owner function
(``owners(fn, graph) -> [unit per computing gate]``, one flat cut per Call's Definition): the extractor reads only owners, so
it serves any partition whose units lie inside Calls.  Circuits are built once per flat Definition and unit, and `classes`
counts shapes through the cut tree without expanding it.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import partition_object as PO
from .codec import canonical_json
from .cut import WIRING, CallGraph, DefinitionCut, call_scope, derived_committed


def shape_digest(form: str) -> str:
    """The SHA-512 of a unit's circuit: of its canonical form (`ProofBlockCircuit.form`)."""
    return hashlib.sha512(b"verity/partition/v2/shape\0" + form.encode()).hexdigest()


@dataclass(frozen=True)
class ProofBlockCircuit:
    definition: str                # the flat Definition's id
    unit: int                      # the unit within that Definition's cut
    gates: tuple[int, ...]
    computing: int                 # how many of `gates` the partition assigns the unit (the rest are structure)
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    in_widths: tuple[int, ...]
    out_widths: tuple[int, ...]
    form: str                      # the canonical JSON the shape hashes
    shape: str


class Activation:
    """One flat Definition in its one-Call Program (`cut.call_scope`): its gates by local index, and its `CallGraph`."""

    def __init__(self, fn: Any, wiring: frozenset[str] = WIRING) -> None:
        self.fn = fn
        self.circuit, call = call_scope(fn)
        self.lo = call.offset
        if self.lo != sum(t.leaves for _, t in fn.signature()[0]):
            raise AssertionError(f"{fn.id}: the one-Call Program does not start with exactly its parameter leaves")
        self.graph = CallGraph(fn, wiring)
        self._gates: dict[int, Any] = {}

    @property
    def params(self) -> int:
        return self.lo

    def gate(self, j: int) -> Any:
        """Local gate j (j < 0: the parameter leaf ``-1 - j``'s Input gate)."""
        if j < 0:
            return self.circuit.gate(-1 - j)
        g = self._gates.get(j)
        if g is None:
            g = self._gates[j] = self.circuit.gate(self.lo + j)
        return g

    def local(self, o: int) -> int:
        """An absolute operand index as a local one."""
        return o - self.lo if o >= self.lo else -1 - o

    def absolute(self, j: int) -> int:
        return j + self.lo if j >= 0 else -1 - j

    def width(self, j: int) -> int:
        return self.gate(j).prim.ret.w

    def circuits(self, owner: Sequence[int]) -> list[ProofBlockCircuit]:
        """The circuit of every unit of the cut ``owner`` (a unit per computing gate, in `CallGraph` order), in unit order."""
        g = self.graph
        if len(owner) != g.n:
            raise ValueError(f"{self.fn.id}: {len(owner)} owners for {g.n} computing gates")
        keep = g.keep
        computing = set(keep)
        committed = set(derived_committed(g, owner))
        n_units = max(owner) + 1 if len(owner) else 0
        owned: list[list[int]] = [[] for _ in range(n_units)]
        outputs: list[list[int]] = [[] for _ in range(n_units)]
        for c in range(g.n):
            owned[owner[c]].append(keep[c])
            if c in committed:
                outputs[owner[c]].append(keep[c])
        return [self._circuit(u, owned[u], outputs[u], computing) for u in range(n_units)]

    def _circuit(self, u: int, owned: list[int], outputs: list[int], computing: set[int]) -> ProofBlockCircuit:
        members: set[int] = set()
        stack = list(owned)
        while stack:
            j = stack.pop()
            if j in members:
                continue
            members.add(j)
            for o in self.gate(j).operands:
                k = self.local(o)
                if k >= 0 and k not in computing and k not in members:
                    stack.append(k)
        gates = tuple(sorted(members))
        local = {j: i for i, j in enumerate(gates)}
        port: dict[int, int] = {}
        form = []
        for j in gates:
            refs = []
            for o in self.gate(j).operands:
                k = self.local(o)
                refs.append(["g", local[k]] if k in local else ["i", port.setdefault(k, len(port))])
            form.append([self.gate(j).prim.id, refs])
        inputs = tuple(sorted(port, key=port.__getitem__))
        in_w = tuple(self.width(k) for k in inputs)
        out_w = tuple(self.width(j) for j in outputs)
        text = canonical_json({"in": list(in_w), "gates": form, "out": [local[j] for j in outputs]}).decode()
        return ProofBlockCircuit(self.fn.id, u, gates, len(owned), inputs, tuple(outputs), in_w, out_w, text,
                           shape_digest(text))


@dataclass(frozen=True)
class Placement:
    """Unit ``index`` of the partition: root-body node ``node`` (a Call), batch member ``member`` (template queries), and
    ``path``, the (body node, member) steps from the Call's Definition down to the flat Definition holding the unit."""

    index: int
    node: int
    member: int
    path: tuple[tuple[int, int], ...]
    circuit: ProofBlockCircuit


@dataclass
class UnitClass:
    shape: str
    representative: ProofBlockCircuit
    definition: Any                # the flat Definition the representative lives in (to lift inputs through it)
    units: int                     # how many units of the partition have this shape


class PartitionUnits:
    """Every unit of a partition of ``program`` as a `ProofBlockCircuit`: of ``query`` (as `partition_object.evaluate` applies it), or of
    ``owners`` (``owners(fn, graph)``: a unit per computing gate of each Call's Definition, cut flat)."""

    def __init__(self, program: Any, query: Mapping[str, Any] | None = None,
                 owners: Callable[[Any, CallGraph], Sequence[int]] | None = None, wiring: frozenset[str] = WIRING) -> None:
        if (query is None) == (owners is None):
            raise ValueError("a query or an owner function, not both")
        self.program = program
        self.query = dict(query) if query is not None else None
        self._owners = owners
        self._wiring = wiring
        self.partition = PO.evaluate(program, query) if query is not None else None
        self._acts: dict[str, Activation] = {}
        self._flat: dict[str, tuple[Any, list[ProofBlockCircuit]]] = {}
        self._counts: dict[str, Counter] = {}
        self._reps: dict[str, tuple[ProofBlockCircuit, Any]] = {}

    def activation(self, fn: Any) -> Activation:
        a = self._acts.get(fn.id)
        if a is None:
            a = self._acts[fn.id] = Activation(fn, self._wiring)
        return a

    def activation_of(self, definition: str) -> Activation:
        """The activation of a flat Definition this partition's units live in, by id."""
        return self._acts[definition]

    def flat(self, fn: Any, owner: Sequence[int] | None) -> list[ProofBlockCircuit]:
        """The circuits of a flat Definition's units (``owner`` None: the whole activation is one unit), memoised by id."""
        got = self._flat.get(fn.id)
        if got is None:
            act = self.activation(fn)
            owner = [0] * act.graph.n if owner is None else list(owner)
            got = self._flat[fn.id] = (fn, act.circuits(owner))
            for uc in got[1]:
                self._reps.setdefault(uc.shape, (uc, fn))
        return got[1]

    # -- the cut tree of each Call ------------------------------------------------------------------------------------------
    def _calls(self) -> Iterator[tuple[int, int, Any, Any]]:
        """(root node, member, the Call's Definition, its cut: a `DefinitionCut`, or None for one unit, or an owner list)."""
        nodes = self.program.root.body.nodes
        if self._owners is not None:
            for k, fn in PO.calls(self.program):
                yield k, 0, fn, list(self._owners(fn, self.activation(fn).graph))
            return
        part = self.partition
        if isinstance(part, PO.TemplatePartition):
            for k, _tid, _first, n in part.ranges:
                for m in range(n):
                    yield k, m, nodes[k].fn, None
            return
        for k, cut in part.calls:
            yield k, 0, cut.fn, cut

    def _expand(self, cut: Any, fn: Any, path: tuple = ()) -> Iterator[tuple[tuple, ProofBlockCircuit]]:
        if isinstance(cut, DefinitionCut) and cut.parts is not None:
            for node, (child, m) in enumerate(cut.parts):
                if child is None:
                    continue
                for member in range(m):
                    yield from self._expand(child, child.fn, path + ((node, member),))
            return
        owner = cut.flat if isinstance(cut, DefinitionCut) else cut
        for uc in self.flat(fn, owner):
            yield path, uc

    def _count(self, cut: Any, fn: Any) -> Counter:
        if not (isinstance(cut, DefinitionCut) and cut.parts is not None):
            owner = cut.flat if isinstance(cut, DefinitionCut) else cut
            return Counter(uc.shape for uc in self.flat(fn, owner))
        got = self._counts.get(fn.id)
        if got is None:
            got = Counter()
            for child, m in cut.parts:
                if child is not None:
                    for s, c in self._count(child, child.fn).items():
                        got[s] += c * m
            self._counts[fn.id] = got
        return got

    def __iter__(self) -> Iterator[Placement]:
        i = 0
        for k, m, fn, cut in self._calls():
            for path, uc in self._expand(cut, fn):
                yield Placement(i, k, m, path, uc)
                i += 1

    def classes(self) -> dict[str, UnitClass]:
        """Every shape with a representative, the flat Definition it lives in, and how many units have it, without expanding
        separable cuts (a Definition's members share their callee's circuits)."""
        total: Counter = Counter()
        by_def: dict[str, Counter] = {}
        for _k, _m, fn, cut in self._calls():
            key = fn.id if cut is None or isinstance(cut, DefinitionCut) else None
            if key is not None and key in by_def:
                total.update(by_def[key])
                continue
            c = self._count(cut, fn)
            if key is not None:
                by_def[key] = c
            total.update(c)
        return {s: UnitClass(s, *self._reps[s], n) for s, n in total.items()}


__all__ = ["shape_digest", "ProofBlockCircuit", "Activation", "Placement", "UnitClass", "PartitionUnits"]
