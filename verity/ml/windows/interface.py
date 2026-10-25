"""What an integration hands Verity (Daniel, 24 Oct: integrations are frontends, and Verity is agnostic to them), one window
at a time, in four records:

1. **A circuit in proof units** (:class:`Circuit`): templates, each a complete binding of a Definition in Verity's catalog
   (`catalog`), with its unit count, in order. Their population program and template query (`rows.population_program`)
   partition it into proof units, and each unit's public layout is its Definition's bit rows (`layout`); ``external``
   names the ports that are the system's external I/O. The window's ``circuit.json`` (`circuit_record`) holds the whole
   layout the verifier reads, and the registration binds its SHA-512.
2. **A compiler and its advice** (optional, `advice_binding`): the registered compiler's name and digest (which carries the
   public cap on the advice's length) and the advice's hiding commitment (`verity.ml.commitments.advice`): its root and its
   length in bits, which is charged. Verity never reads the advice.
3. **The values** (:class:`Window`): each unit's values as computed (`Window.record`: one array per parameter and one per
   return field, a leading unit axis, each element its bits as an unsigned integer, `evaluation.evaluate_batch`'s
   convention), committed as ``hm96-sha512/row/v2`` rows in ``frame-v3-sha512`` trees, one tree per port under the
   served-domain rule (`Window.close`, `rows.commit`), and registered before any coin (`Window.write`: `rows.write`'s files,
   the registration core's record made by the ``registrar`` the caller passes, core's ``one_stage.registration``).
4. **On a draw, the openings**: a drawn member's private rows and salts as C-Flock's prover reads them
   (`flock_inputs.flock_input_files`), for the prover behind the firewall.

An integration may label each run of units it records with one opaque JSON value (``label``): the window's
``index.json``, whose SHA-512 the registration binds. Verity never reads a label.

The verifier's window comes from the records alone (`of_records`: ``circuit.json`` held to the registration's digest),
never from the integration's code. A unit's rows are its Definition's bit rows: one row per parameter, the little-endian
bit concatenation of its leaves, then one output row ``o<j>`` per return field, zero-padded to whole u16 words: C-Flock's
`UnitStager` rows (`unit_rows`), so the verifier composes the unit's circuit from the Definition alone.

That the circuit computes what the integration served is completeness, never proved: the verifier accepts a window exactly
when its drawn units' committed outputs are their Definitions' outputs on their committed inputs (`Window.mismatches` is
the integration's own check before it commits).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, SpecializedDefinition
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

from . import rows as CS

#: the element widths a parameter's or return field's leaves may have: a row is their little-endian bytes
PACKED = (8, 16, 32, 64)
CIRCUIT = "verity/window-circuit/v1"
LABELS = "verity/window-labels/v1"
_DIGEST = re.compile("[0-9a-f]{128}")


def catalog(fn: Any) -> None:
    """Refuse (ValueError) a template that is not a complete binding of a Definition Verity registers from a ``verity`` module:
    an integration's own Definition never defines what a unit computes."""
    base = fn.defn if isinstance(fn, SpecializedDefinition) else fn
    if not isinstance(base, CompositeDefinition) or (isinstance(fn, SpecializedDefinition) and not fn.complete):
        raise ValueError(f"{getattr(fn, 'id', fn)!r}: a template is a complete binding of a composite Definition")
    if base.id not in REGISTRY or REGISTRY[base.id] is not base or not base.body.__module__.startswith("verity."):
        raise ValueError(f"{base.id}: not a Definition Verity registers (defined in {base.body.__module__}); an integration "
                         "builds its units from Verity's Definitions and silicon semantics, never its own")


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
class Circuit:
    """An integration's circuit in proof units: ``templates``, each ``(template, units)`` in order, every template `catalog`'s
    and of its own descriptor id; ``external``, the ports (``<descriptor id>/<port>``) that are the system's external I/O."""
    name: str
    templates: tuple[tuple[Any, int], ...]
    external: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "templates", tuple((fn, int(n)) for fn, n in self.templates))
        object.__setattr__(self, "external", tuple(self.external))
        for fn, n in self.templates:
            catalog(fn)
            layout(fn)
            if n < 1:
                raise ValueError(f"{fn.id}: {n} units; a template of the circuit has at least one")
        if len(set(self.ids)) != len(self.ids) or not self.templates:
            raise ValueError(f"{self.name}: templates {self.ids}, each once, at least one")
        ports = {f"{tid}/{p}" for fn, tid in zip((f for f, _ in self.templates), self.ids)
                 for p, _ in layout(fn).rows + layout(fn).outs}
        if bad := sorted(set(self.external) - ports):
            raise ValueError(f"{self.name}: external ports {bad} are no template's port")

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


def members(circuit: Circuit, cap: int) -> tuple[CS.Member, ...]:
    """The window's members: each template's units in the partition's order, in chunks ``<descriptor id>/c<j>`` of at most
    ``cap`` (one statement each), on the template's bit rows."""
    if cap < 1:
        raise ValueError(f"a statement holds at least one unit, not {cap}")
    program, query = circuit.program()
    out = []
    for _k, tid, lo, n in CS.template_partition(program, query).ranges:
        lay = layout(circuit.template(tid))
        out += [CS.Member(tid, lay.rows, lay.outs, min(cap, n - at), lo + at, name=f"{tid}/c{j}", schemas=lay.schemas)
                for j, at in enumerate(range(0, n, cap))]
    return tuple(out)


def advice_binding(compiler: Mapping[str, str], root: str, bits: int) -> dict[str, Any]:
    """The registration's binding of a circuit's compiler and advice: the compiler's ``name`` and ``digest`` and the root and
    length in bits of the advice's hiding commitment (`verity.ml.commitments.advice`), each refused unless well-formed."""
    if set(compiler) != {"name", "digest"} or not isinstance(compiler["name"], str) or not _DIGEST.fullmatch(str(compiler["digest"])):
        raise ValueError(f"a compiler is its name and its 64-byte digest in hex, not {dict(compiler)!r}")
    if not _DIGEST.fullmatch(str(root)):
        raise ValueError(f"the advice's root is 64 bytes, 128 hex digits, not {root!r}")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 0:
        raise ValueError(f"the advice's length is a whole number of bits, not {bits!r}")
    return {"compiler": {"name": compiler["name"], "digest": compiler["digest"]}, "advice": {"root": root, "bits": bits}}


def circuit_record(w: CS.Window, name: str, templates: Sequence[tuple[str, int]],
                   statements: Mapping[str, Mapping[str, Any]] | None = None, external: Sequence[str] = ()) -> dict[str, Any]:
    """``circuit.json``: everything the verifier's window is (`of_records`), none of its values. ``templates``: each
    ``(descriptor id, units)`` in the program's order; ``statements``: by template, how its unit's statement reads its rows
    (``in_rows``, ``out_rows``, ``reads``, as C-Flock's `UnitStager` takes them), for a template whose members are bit rows."""
    return {"format": CIRCUIT, "name": name, "templates": [[t, int(n)] for t, n in templates], "query": dict(w.query),
            "program": w.partition["program"], "partition": w.partition_digest, "run": w.run, "context": w.context,
            "work_table": w.work_table, "external": list(external),
            "members": [{"template": m.template, "name": m.name, "n": m.n, "base": m.base, "rows": [list(r) for r in m.rows],
                         "outs": [list(o) for o in m.outs], "schemas": [list(s) for s in m.schemas],
                         "tables": [list(t) for t in m.tables], "link_units": list(m.link_units),
                         "links": [list(x) for x in m.links], "wired": [[p, o, list(ws)] for p, o, ws in m.wired]}
                        for m in w.members],
            "row_tables": [{"engine_step": t.engine_step, "op_path": t.op_path, "template": t.template, "words": t.words,
                            "rows": [list(r) for r in t.rows]} for t in w.row_tables],
            "statements": {t: {"in_rows": [[p, s, b, list(ps)] for p, s, b, ps in st["in_rows"]],
                               "out_rows": [[s, list(ix)] for s, ix in st["out_rows"]], "reads": st.get("reads", "inline")}
                           for t, st in sorted((statements or {}).items())}}


def circuit_bytes(doc: Mapping[str, Any]) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def window_of_record(doc: Mapping[str, Any], law: Mapping[str, Any], resolve: Callable[[str], Any]) -> CS.Window:
    """The verifier's window from ``circuit.json``'s document under its own ``law``, each template's Definition by
    ``resolve`` (descriptor id -> Definition, the verifier's own catalog), refused unless the population program, query,
    program digest and partition digest it rebuilds are the record's."""
    if doc.get("format") != CIRCUIT:
        raise ValueError(f"a window's circuit record is {CIRCUIT}, not {doc.get('format')!r}")
    defs = [(resolve(t), int(n)) for t, n in doc["templates"]]
    if [descriptor_id(fn) for fn, _ in defs] != [t for t, _ in doc["templates"]]:
        raise ValueError(f"the catalog resolves {[t for t, _ in doc['templates']]} to {[descriptor_id(fn) for fn, _ in defs]}")
    several = len(defs) > 1 or doc["query"] != PO.template_instance_query(defs[0][0])
    program, query = CS.population_program(defs, several=several)
    ms = tuple(CS.Member(m["template"], tuple(map(tuple, m["rows"])), tuple(map(tuple, m["outs"])), m["n"], m["base"],
                         tuple(map(tuple, m["tables"])), m["name"], tuple(map(tuple, m["schemas"])), tuple(m["link_units"]),
                         tuple(map(tuple, m["links"])), tuple((p, o, tuple(ws)) for p, o, ws in m["wired"]))
               for m in doc["members"])
    tables = tuple(CS.RowTable(t["engine_step"], t["op_path"], t["template"], t["words"], tuple(map(tuple, t["rows"])))
                   for t in doc["row_tables"])
    w = CS.Window(program, query, dict(law), doc["run"], ms, tables, doc["context"], doc["work_table"])
    got = {"query": dict(query), "program": w.partition["program"], "partition": w.partition_digest}
    if bad := sorted(k for k, v in got.items() if v != doc[k]):
        raise ValueError(f"the circuit record's {bad} are not the ones its templates make: {got}")
    return w


def of_records(window_dir: str | Path, law: Mapping[str, Any], resolve: Callable[[str], Any]) -> tuple[CS.Window, dict[str, Any]]:
    """The verifier's window from a window's records alone (`window_of_record` on its ``circuit.json``), refused unless the
    record is the one its registration binds (the window's ``circuit``) and names the registration's program, partition
    and run; -> (the window, the record)."""
    d = Path(window_dir)
    body, rec = (d / "circuit.json").read_bytes(), json.loads((d / "registration.json").read_text())
    bound = (rec.get("window") or {}).get("circuit")
    if hashlib.sha512(body).hexdigest() != bound:
        raise ValueError(f"{d / 'circuit.json'} is not the circuit record the registration binds ({bound})")
    doc = json.loads(body)
    if (doc.get("program"), doc.get("partition"), doc.get("run")) != (rec.get("program"), rec.get("partition"),
                                                                        (rec.get("window") or {}).get("run")):
        raise ValueError("the circuit record names another program, partition or run than its registration")
    return window_of_record(doc, law, resolve), doc


@dataclass(frozen=True)
class _Batch:
    inputs: list[np.ndarray]
    outputs: list[np.ndarray]
    words: list[np.ndarray]      # every port's (n, words) u16, parameters then return fields
    label: Any
    n: int


@dataclass
class Window:
    """One window of an integration's values: ``circuit``'s units under ``run`` (the record's run root, hex) and ``law`` (the
    proof service's law of the window, as its draw parses it), in members of at most ``cap`` units, with ``advice`` its
    compiler and advice (`advice_binding`) when the circuit depends on run-time choices. Each template's units are
    `record`ed in order, then `close` commits them and `write` writes and registers the window."""
    circuit: Circuit
    run: str
    law: Mapping[str, Any]
    cap: int
    advice: Mapping[str, Any] | None = None
    window: CS.Window = field(init=False)
    _batches: dict[str, list[_Batch]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        program, query = self.circuit.program()
        self.window = CS.Window(program, query, dict(self.law), self.run, members(self.circuit, self.cap))
        if self.advice is not None:
            self.advice = advice_binding(self.advice["compiler"], self.advice["advice"]["root"], self.advice["advice"]["bits"])

    def count(self, tid: str) -> int:
        """Template ``tid``'s units recorded so far."""
        return sum(b.n for b in self._batches.get(tid, ()))

    def record(self, template: Any, inputs: Sequence[Any], outputs: Sequence[Any], label: Any = None) -> range:
        """``template``'s next units: ``inputs`` one array per parameter, ``outputs`` one per return field (module docstring),
        all of one unit count, under the integration's opaque ``label`` (JSON). -> their indices among the template's units."""
        tid = descriptor_id(template)
        fn, total = self.circuit.template(tid), dict(zip(self.circuit.ids, (n for _, n in self.circuit.templates)))[tid]
        lay = layout(fn)
        if len(inputs) != len(lay.rows) or len(outputs) != len(lay.outs):
            raise ValueError(f"{tid}: {len(inputs)} inputs and {len(outputs)} outputs, not {len(lay.rows)} and {len(lay.outs)}")
        n, at = len(inputs[0]), self.count(tid)
        if n < 1 or at + n > total:
            raise ValueError(f"{tid}: {n} units at {at}, past its {total}")
        json.dumps(label)
        values = [np.ascontiguousarray(a) for a in [*inputs, *outputs]]
        ws = [words(a, w, n, c) for a, w, (_, c) in zip(values, lay.widths, lay.rows + lay.outs)]
        self._batches.setdefault(tid, []).append(_Batch(values[:len(inputs)], values[len(inputs):], ws, label, n))
        return range(at, at + n)

    def mismatches(self, kernel: str = "reference") -> dict[str, list[int]]:
        """Per template, the units whose recorded outputs are not its Definition's outputs on their recorded inputs
        (`evaluate_batch` under ``kernel``): what the proof of a drawn one would refuse."""
        out = {}
        for fn, _n in self.circuit.templates:
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
        missing = {tid: (self.count(tid), n) for tid, (_, n) in zip(self.circuit.ids, self.circuit.templates)
                   if self.count(tid) != n}
        if missing:
            raise ValueError(f"units recorded / in the circuit by template: {missing}; a window closes over all of them")
        base = self._bases()
        data = []
        for m in self.window.members:
            lay, got = layout(self.circuit.template(m.template)), self._batches[m.template]
            ports = [np.concatenate([b.words[g] for b in got]) for g in range(len(lay.rows) + len(lay.outs))]
            lo = m.base - base[m.template]
            data.append({"rows": {p: ports[g][lo:lo + m.n] for g, (p, _) in enumerate(lay.rows)},
                         "outs": {p: ports[len(lay.rows) + j][lo:lo + m.n] for j, (p, _) in enumerate(lay.outs)}})
        return CS.commit(self.window, data, workers=workers, salt_key=salt_key)

    def _bases(self) -> dict[str, int]:
        return {t: b for _k, t, b, _n in CS.template_partition(self.window.program, self.window.query).ranges}

    def labels(self) -> dict[str, Any]:
        """``index.json``: per member, each recorded run of its units with its label, ``[lo, hi, label]`` over global indices."""
        spans: dict[str, list[tuple[int, int, Any]]] = {}
        for tid, lo in self._bases().items():
            for b in self._batches.get(tid, ()):
                spans.setdefault(tid, []).append((lo, lo + b.n, b.label))
                lo += b.n
        return {"format": LABELS, "run": self.run,
                "members": [{"name": m.key, "labels": [[max(a, m.base), min(b, m.base + m.n), x]
                                                       for a, b, x in spans.get(m.template, ()) if a < m.base + m.n and b > m.base]}
                            for m in self.window.members]}

    def circuit_record(self) -> dict[str, Any]:
        """The window's ``circuit.json`` (`circuit_record`), each template's statement its bit rows (`unit_rows`)."""
        sts = {}
        for fn, _ in self.circuit.templates:
            i, o = unit_rows(fn)
            sts[descriptor_id(fn)] = {"in_rows": i, "out_rows": o, "reads": "inline"}
        return circuit_record(self.window, self.circuit.name, list(zip(self.circuit.ids, (n for _, n in self.circuit.templates))),
                              sts, self.circuit.external)

    def write(self, c: CS.Committed, out_dir: str | Path, provenance: Mapping[str, Any], *, registrar: Any,
              kind: str = "served") -> dict[str, Any]:
        """The window's files (`rows.write`, its ``index.json`` the labels) and ``circuit.json``, the registration core's record
        made by ``registrar`` (core's ``one_stage.registration``), its window naming ``provenance``, the circuit record's
        SHA-512 and the advice binding."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        body = circuit_bytes(self.circuit_record())
        (d / "circuit.json").write_bytes(body)
        bound = {**provenance, "circuit": hashlib.sha512(body).hexdigest(), **(self.advice or {})}
        return CS.write(c, d, bound, self.labels(), kind=kind, registrar=registrar)


class Integration(Protocol):
    """What an integration provides: its circuit, and a run that records every unit of it into a window (`Window.record`)
    before the window closes."""

    def circuit(self) -> Circuit: ...

    def run(self, window: Window) -> None: ...
