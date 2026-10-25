"""The stack-neutral serving interface (Daniel, 24 Oct): what a serving stack calls to have its served units proved.

1. **Its workload, from Verity's Definitions** (:class:`Workload`): each proof-unit template a binding of a Definition registered
   in `verity.ml.circuits.defs.REGISTRY` from a ``verity`` module (`catalog`), with its unit count. Its population program
   and template query are the verifier's (`rows.population_program`).
2. **A window** (:class:`ServingWindow`) over the run root of record and the proof service's law: the members the
   population is cut into, ``cap`` units a statement (`members`).
3. **Each unit's values, as served** (`ServingWindow.record`): one array per parameter and one per return field, a leading
   unit axis, each element its bits as an unsigned integer (`evaluation.evaluate_batch`'s convention), with the served
   identity of those units (:class:`Served`).
4. **The commitment** (`ServingWindow.close`): every unit's rows as ``hm96-sha512/row/v2`` rows in ``frame-v3-sha512``
   trees, one tree per port under the served-domain rule (`rows.commit`).
5. **The window it registers with the proof service** (`ServingWindow.write`): `rows.write`'s files, whose registration is
   core's record made by the ``registrar`` the stack passes (core's ``one_stage.registration``). The service takes it from
   there: live coins, the drawn units' C-Flock proofs and the Lean verifier of record.

A unit's rows are its Definition's bit rows (`layout`): one row per parameter, the little-endian bit concatenation of its
leaves, then one output row ``o<j>`` per return field, zero-padded to whole u16 words: C-Flock's `UnitStager` rows
(`unit_rows`), so the verifier composes the unit's circuit from the Definition alone. :class:`Stack` is what a stack's
adapter provides. vLLM's (`verity_vllm.pipeline.serving_rows`) lays out its own members on the same `rows` window.

That a stack's units compute what its model computed is completeness, never proved: the verifier accepts a window exactly
when its drawn units' committed outputs are their Definitions' outputs on their committed inputs (`ServingWindow.mismatches`
is the stack's own check before it commits).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, SpecializedDefinition
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

from . import rows as CS

#: the element widths a parameter's or return field's leaves may have: a row is their little-endian bytes
PACKED = (8, 16, 32, 64)


def catalog(fn: Any) -> None:
    """Refuse (ValueError) a template that is not a complete binding of a Definition Verity registers from a ``verity`` module:
    a stack's own Definition never defines what a unit computes."""
    base = fn.defn if isinstance(fn, SpecializedDefinition) else fn
    if not isinstance(base, CompositeDefinition) or (isinstance(fn, SpecializedDefinition) and not fn.complete):
        raise ValueError(f"{getattr(fn, 'id', fn)!r}: a template is a complete binding of a composite Definition")
    if base.id not in REGISTRY or REGISTRY[base.id] is not base or not base.body.__module__.startswith("verity."):
        raise ValueError(f"{base.id}: not a Definition Verity registers (defined in {base.body.__module__}); a stack builds its "
                         "units from Verity's Definitions and silicon semantics, never its own")


def _fields(ret: Any) -> tuple[Any, ...]:
    return tuple(f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,)))


def _width(name: str, ty: Any) -> int:
    widths = ty.width_set()
    if len(widths) != 1 or not widths <= set(PACKED) or sum(ty.leaf_widths()) % 16:
        raise ValueError(f"{name}: its leaves are {sorted(widths)} bits, {sum(ty.leaf_widths())} in all; a row is whole u16 "
                         f"words of leaves of one of {PACKED} bits")
    return next(iter(widths))


@dataclass(frozen=True)
class Layout:
    """A template's bit rows: ``rows`` each parameter's ``(name, words)``, ``outs`` each return field's ``(o<j>, words)``, and
    each one's element width."""
    rows: tuple[tuple[str, int], ...]
    outs: tuple[tuple[str, int], ...]
    widths: tuple[int, ...]

    @property
    def schemas(self) -> tuple[tuple[str, str], ...]:
        return tuple((p, V2) for p, _ in self.rows + self.outs)


def layout(fn: Any) -> Layout:
    """``fn``'s bit rows (module docstring); refused when a port is not whole u16 words of one ``PACKED`` width."""
    params, ret = fn.signature()
    rows = tuple((name, sum(ty.leaf_widths()) // 16) for name, ty in params)
    outs = tuple((f"o{j}", sum(ty.leaf_widths()) // 16) for j, ty in enumerate(_fields(ret)))
    if len({p for p, _ in rows + outs}) != len(rows) + len(outs):
        raise ValueError(f"{fn.id}: a parameter is named like an output row {[p for p, _ in outs]}")
    widths = tuple(_width(f"{fn.id} {name}", ty) for name, ty in [*params, *zip((p for p, _ in outs), _fields(ret))])
    return Layout(rows, outs, widths)


def unit_rows(fn: Any) -> tuple[list[tuple[str, str, int, list[str]]], list[tuple[str, tuple[int, ...]]]]:
    """``fn``'s rows as C-Flock's `UnitStager` takes them, ``(in_rows, out_rows)``: one ``hm96-sha512/row/v2`` row per parameter,
    then one per return field."""
    lay = layout(fn)
    return [(p, V2, 16 * w, [p]) for p, w in lay.rows], [(V2, (j,)) for j in range(len(lay.outs))]


def words(a: Any, width: int, n: int, count: int) -> np.ndarray:
    """``n`` units' values of one port (``a``: a leading unit axis, each element ``width`` bits, any dtype of that size) as
    their ``(n, count)`` u16 words: the elements' little-endian bytes."""
    a = np.ascontiguousarray(a)
    if a.dtype.itemsize * 8 != width or a.shape[:1] != (n,):
        raise ValueError(f"{a.shape} {a.dtype}: not {n} units of {width}-bit elements")
    w = a.view(f"u{width // 8}").astype(f"<u{width // 8}").reshape(n, -1).view("<u2")
    if w.shape != (n, count):
        raise ValueError(f"{a.shape}: {w.shape[1]} words a unit, not the port's {count}")
    return w.astype(np.uint16)


@dataclass(frozen=True)
class Workload:
    """What a stack serves: ``templates``, each ``(template, units)`` in order, every template `catalog`'s and of its own
    descriptor id."""
    name: str
    templates: tuple[tuple[Any, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "templates", tuple((fn, int(n)) for fn, n in self.templates))
        for fn, n in self.templates:
            catalog(fn)
            layout(fn)
            if n < 1:
                raise ValueError(f"{fn.id}: {n} units; a template of the workload has at least one")
        if len(set(self.ids)) != len(self.ids) or not self.templates:
            raise ValueError(f"{self.name}: templates {self.ids}, each once, at least one")

    @property
    def ids(self) -> list[str]:
        return [descriptor_id(fn) for fn, _ in self.templates]

    def program(self) -> tuple[Any, dict[str, Any]]:
        """The verifier's population program and template query over the templates (`rows.population_program`)."""
        return CS.population_program(list(self.templates))

    def template(self, tid: str) -> Any:
        """The template of descriptor id ``tid``."""
        for fn, _ in self.templates:
            if descriptor_id(fn) == tid:
                return fn
        raise KeyError(f"{tid}: no template of {self.name}")


def members(workload: Workload, cap: int) -> tuple[CS.Member, ...]:
    """The window's members: each template's units in the partition's order, in chunks ``<descriptor id>/c<j>`` of at most
    ``cap`` (one statement each), on the template's bit rows."""
    if cap < 1:
        raise ValueError(f"a statement holds at least one unit, not {cap}")
    program, query = workload.program()
    out = []
    for _k, tid, lo, n in CS.template_partition(program, query).ranges:
        lay = layout(workload.template(tid))
        out += [CS.Member(tid, lay.rows, lay.outs, min(cap, n - at), lo + at, name=f"{tid}/c{j}", schemas=lay.schemas)
                for j, at in enumerate(range(0, n, cap))]
    return tuple(out)


@dataclass(frozen=True)
class Served:
    """Which served computation units are (`rows.index_doc`'s columns): the request, its engine step, layer and op path, the
    output member, the token row and the program row."""
    request: str
    engine_step: int = 0
    layer: int = 0
    op_path: str = ""
    member: str = ""
    token_row: int = 0
    program_row: int = 0


@dataclass(frozen=True)
class _Batch:
    inputs: list[np.ndarray]
    outputs: list[np.ndarray]
    words: list[np.ndarray]      # every port's (n, words) u16, parameters then return fields
    served: Served
    n: int


@dataclass
class ServingWindow:
    """One window of a stack's serving: ``workload``'s units under ``run`` (the record's run root, hex) and ``law`` (the proof
    service's law of the window, as its draw parses it), in members of at most ``cap`` units. Each template's units are
    `record`ed in order, then `close` commits them and `write` writes the window."""
    workload: Workload
    run: str
    law: Mapping[str, Any]
    cap: int
    window: CS.Window = field(init=False)
    _batches: dict[str, list[_Batch]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        program, query = self.workload.program()
        self.window = CS.Window(program, query, dict(self.law), self.run, members(self.workload, self.cap))

    def count(self, tid: str) -> int:
        """Template ``tid``'s units recorded so far."""
        return sum(b.n for b in self._batches.get(tid, ()))

    def record(self, template: Any, inputs: Sequence[Any], outputs: Sequence[Any], served: Served) -> range:
        """``template``'s next units: ``inputs`` one array per parameter, ``outputs`` one per return field (module docstring),
        all of one unit count, the units of one served computation ``served``. -> their indices among the template's units."""
        tid = descriptor_id(template)
        fn, total = self.workload.template(tid), dict(zip(self.workload.ids, (n for _, n in self.workload.templates)))[tid]
        lay = layout(fn)
        if len(inputs) != len(lay.rows) or len(outputs) != len(lay.outs):
            raise ValueError(f"{tid}: {len(inputs)} inputs and {len(outputs)} outputs, not {len(lay.rows)} and {len(lay.outs)}")
        n, at = len(inputs[0]), self.count(tid)
        if n < 1 or at + n > total:
            raise ValueError(f"{tid}: {n} units at {at}, past its {total}")
        values = [np.ascontiguousarray(a) for a in [*inputs, *outputs]]
        ws = [words(a, w, n, c) for a, w, (_, c) in zip(values, lay.widths, lay.rows + lay.outs)]
        self._batches.setdefault(tid, []).append(_Batch(values[:len(inputs)], values[len(inputs):], ws, served, n))
        return range(at, at + n)

    def mismatches(self, kernel: str = "reference") -> dict[str, list[int]]:
        """Per template, the units whose recorded outputs are not its Definition's outputs on their recorded inputs
        (`evaluate_batch` under ``kernel``): what the proof of a drawn one would refuse."""
        out = {}
        for fn, _n in self.workload.templates:
            tid, bad, at = descriptor_id(fn), [], 0
            leaves = [len(f.leaf_widths()) for f in _fields(fn.signature()[1])]
            for b in self._batches.get(tid, ()):
                want = np.asarray(evaluate_batch(fn, *b.inputs, kernel=kernel)).reshape(b.n, -1).astype(np.uint64)
                got = np.concatenate([o.view(f"u{o.dtype.itemsize}").reshape(b.n, -1).astype(np.uint64) for o in b.outputs], axis=1)
                if got.shape != want.shape or want.shape[1] != sum(leaves):
                    raise ValueError(f"{tid}: outputs of {got.shape[1]} leaves a unit, its Definition's {want.shape[1]}")
                bad += [at + int(i) for i in np.flatnonzero((got != want).any(axis=1))]
                at += b.n
            if bad:
                out[tid] = bad
        return out

    def close(self, *, workers: int = 1, salt_key: bytes | None = None) -> CS.Committed:
        """Commit every unit's rows (`rows.commit`), refused unless every template's units are all recorded. ``salt_key``:
        tests only (`rows.commit`)."""
        missing = {tid: (self.count(tid), n) for tid, (_, n) in zip(self.workload.ids, self.workload.templates)
                   if self.count(tid) != n}
        if missing:
            raise ValueError(f"units recorded / served by template: {missing}; a window closes over all of them")
        base = self._bases()
        data = []
        for m in self.window.members:
            lay, got = layout(self.workload.template(m.template)), self._batches[m.template]
            ports = [np.concatenate([b.words[g] for b in got]) for g in range(len(lay.rows) + len(lay.outs))]
            lo = m.base - base[m.template]
            data.append({"rows": {p: ports[g][lo:lo + m.n] for g, (p, _) in enumerate(lay.rows)},
                         "outs": {p: ports[len(lay.rows) + j][lo:lo + m.n] for j, (p, _) in enumerate(lay.outs)}})
        return CS.commit(self.window, data, workers=workers, salt_key=salt_key)

    def _bases(self) -> dict[str, int]:
        return {t: b for _k, t, b, _n in CS.template_partition(self.window.program, self.window.query).ranges}

    def index(self) -> dict[str, Any]:
        """`rows.index_doc` of the window: per member, each served computation its units are of, with its whole range."""
        spans: dict[str, list[tuple[int, int, Served]]] = {}
        for tid, lo in self._bases().items():
            for b in self._batches.get(tid, ()):
                spans.setdefault(tid, []).append((lo, lo + b.n, b.served))
                lo += b.n
        entries = [[[a, b, s.request, s.engine_step, s.layer, s.op_path, s.member, s.token_row, s.program_row]
                    for a, b, s in spans.get(m.template, ()) if a < m.base + m.n and b > m.base] for m in self.window.members]
        return CS.index_doc(self.window.members, entries, self.run)

    def write(self, c: CS.Committed, out_dir: str | Path, provenance: Mapping[str, Any], *, registrar: Any,
              kind: str = "served") -> dict[str, Any]:
        """The window's files (`rows.write`), its registration core's record made by ``registrar`` (core's
        ``one_stage.registration``), its window naming ``provenance`` and the workload."""
        return CS.write(c, out_dir, {**provenance, "workload": self.workload.name}, self.index(), kind=kind, registrar=registrar)


class Stack(Protocol):
    """What a serving stack's adapter provides: the workload it serves, and serving that records every unit of it into a
    window (`ServingWindow.record`) before the window closes."""

    def workload(self) -> Workload: ...

    def serve(self, window: ServingWindow) -> None: ...
