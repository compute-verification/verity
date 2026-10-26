"""What an integration hands Verity (Daniel, 24 Oct: integrations are frontends, and Verity is agnostic to them), one window
at a time, in four records:

1. **A circuit in proof units** (:class:`Circuit`): templates, each a complete binding of a Definition in Verity's catalog
   (`catalog`), with its unit count, in order. Their population program and template query (`rows.population_program`)
   partition it into proof units, and each unit's public layout is its Definition's bit rows (`layout`); ``external``
   names the ports that are the system's external I/O, ``wires`` the units that read another unit's output row whole
   (:class:`Wire`, one pattern for many units), and ``boundary`` its I/O layout as network-accounting's N1 states it. The
   window's ``circuit.json`` (`circuit_record`) holds the whole layout the verifier reads, stated per template and wire
   rather than per member, and the registration binds its SHA-512.
2. **A compiler and its advice** (optional, `advice_binding`): the registered compiler's name and digest (which carries the
   public cap on the advice's length; `compiler_digest`, over its code) with the commit that code is at, and the advice's
   hiding commitment (`verity.ml.commitments.advice`): its root and its length in bits, which is charged. Verity never
   reads the advice.
3. **The values** (:class:`Window`): each unit's values as computed (`Window.record`: one array per parameter and one per
   return field, a leading unit axis, each element its bits as an unsigned integer, `evaluation.evaluate_batch`'s
   convention), committed as ``hm96-sha512/row/v2`` rows in ``frame-v3-sha512`` trees, one tree per port under the
   window-domain rule (`verity.ml.commitments.window`; `Window.close`, `rows.commit`), and registered before any coin (`Window.write`: `rows.write`'s files,
   the registration core's record made by the ``registrar`` the caller passes, core's ``one_stage.registration``).
4. **On a draw, the openings**: a drawn member's private rows and salts as C-Flock's prover reads them
   (`flock_inputs.flock_input_files`), for the prover behind the firewall.

An integration may label each run of units it records with one opaque JSON value (``label``): the window's
``index.json``, whose SHA-512 the registration binds. Verity never reads a label.

The verifier's window comes from the records alone (`of_records`: ``circuit.json`` held to the registration's digest),
never from the integration's code. A unit's rows are its Definition's bit rows: one row per parameter, the little-endian
bit concatenation of its leaves, then one output row ``o<j>`` per return field, zero-padded to whole u16 words: C-Flock's
`UnitStager` rows (`unit_rows`), so the verifier composes the unit's circuit from the Definition alone. A Boolean-profile
port's leaves (one bit each) are read as their `boolean.word_type` words.

A wired port's row is committed under its writer's output row's salt, so the two rows' public strings are one, which
a5's wiring check holds before the draw (`wiring`). A wire reads a row whole: a row gathered from several writers' words
(RoPE's query from the Q coordinates, attention's K and V from earlier positions) is refused until circuits and proofs
choose its row form.

The circuit also declares what fills its other inputs (`Registered`, `Ingress`): rows of a registered parameter read at
public positions (`Read`), a registered table's root or domain id (`TableRead`), a table's row opened at a hidden key
(`Opening`), and the system's ingress at its step (`Ingress`). Every input row is wired, one of these, or free, and no row
is two of them (`check_inputs`); `input_count` counts the free rows' bits from the record alone, at the size of its
description. The window commits the first kind (`Window`'s ``registered``): a member whose units read a registered
parameter is a shared-row member, each read port a table of the registered rows its units read (`member_reads`), committed
under the registered value's salts (core's ``one_stage.registered``), so each row's public string is the registered one,
and its header names each row's position and path to the value's root; the verifier's ``--registered`` file of the member
comes from the record and the root it pinned alone (`registered_reads`), and Lean's registered reads open each row against
it. Tables, roots, openings and ingress are refused by the committer until it commits them.

That the circuit computes what the integration ran is completeness, never proved: the verifier accepts a window exactly
when its drawn units' committed outputs are their Definitions' outputs on their committed inputs (`Window.mismatches` is
the integration's own check before it commits).
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import json
import math
import pkgutil
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from verity.ml.circuits import boolean as B, partition_object as PO
from verity.ml.circuits.codec import canonical_json, descriptor_id
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments import advice as AC
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

from . import rows as CS

#: the element widths a parameter's or return field's leaves may have: a row is their little-endian bytes
PACKED = (8, 16, 32, 64)
CIRCUIT = "verity/window-circuit/v1"
LABELS = "verity/window-labels/v1"
#: the advice tree's binding tag (`advice_tree_binding`): its bytes are its first records', a later rename
ADVICE_SCHEME = "verity-vllm/advice-windows/advice/v1"
#: the advice's bytes as rows of this many bits, the last zero-padded (`advice_rows`)
ADVICE_ROW_BITS = 256
_DIGEST = re.compile("[0-9a-f]{128}")
#: a git commit id (SHA-1, or SHA-256 in a SHA-256 repository): a locator for code, never a commitment, which is the digest
_COMMIT = re.compile("[0-9a-f]{40}|[0-9a-f]{64}")
_ID = re.compile(r"([A-Za-z][A-Za-z0-9]*_v[0-9]+)(?:\{(.*)\})?", re.S)


def _verity(d: Any) -> bool:
    code = d.evaluate if isinstance(d, PrimitiveDefinition) else d.body
    return d.id in REGISTRY.defs and REGISTRY[d.id] is d and code.__module__.startswith("verity.")


def catalog(fn: Any) -> None:
    """Refuse (ValueError) a template that is not a complete binding of a Definition Verity registers from a ``verity`` module:
    an integration's own Definition never defines what a unit computes."""
    base = fn.defn if isinstance(fn, SpecializedDefinition) else fn
    if not isinstance(base, CompositeDefinition) or (isinstance(fn, SpecializedDefinition) and not fn.complete):
        raise ValueError(f"{getattr(fn, 'id', fn)!r}: a template is a complete binding of a composite Definition")
    if not _verity(base):
        raise ValueError(f"{base.id}: not a Definition Verity registers (defined in {base.body.__module__}); an integration "
                         "builds its units from Verity's Definitions and silicon semantics, never its own")


def _bindings(record: str) -> dict[str, Any]:
    """A descriptor id's binding record ``k=<canonical JSON>,...`` as its JSON values, in order."""
    dec, out, at = json.JSONDecoder(), {}, 0
    while at < len(record):
        eq = record.index("=", at)
        key = record[at:eq]
        if not key.isidentifier() or key in out:
            raise ValueError(f"binding record {record!r}: {key!r} is not a new static's name")
        out[key], at = dec.raw_decode(record, eq + 1)
        if at < len(record):
            if record[at] != ",":
                raise ValueError(f"binding record {record!r}: no ',' at {at}")
            at += 1
    return out


def _static(v: Any) -> Any:
    if isinstance(v, dict) and set(v) == {"fn"}:
        return resolve(v["fn"], template=False)
    if isinstance(v, dict) and set(v) == {"f64"}:
        return float.fromhex(v["f64"])
    if isinstance(v, dict) and set(v) == {"dict"}:
        return {k: _static(x) for k, x in v["dict"].items()}
    if isinstance(v, list):
        return tuple(_static(x) for x in v)
    return v


@functools.cache
def _load_catalog() -> None:
    """Import every module of `verity.ml.circuits.ops`, the catalog's: each registers its Definitions as it loads."""
    from verity.ml.circuits import ops

    for m in pkgutil.walk_packages(ops.__path__, f"{ops.__name__}."):
        if ".tests" not in m.name:
            importlib.import_module(m.name)


def resolve(tid: str, *, template: bool = True) -> Any:
    """The Definition descriptor id ``tid`` names in Verity's catalog (every Definition registered from a ``verity`` module):
    its base and binding record parsed (`codec.descriptor_id`'s form) and bound; a ``template`` refused unless `catalog`'s.
    Refused (ValueError) unless the Definition it binds is named exactly ``tid``."""
    _load_catalog()
    m = _ID.fullmatch(tid)
    if m is None or m[1] not in REGISTRY.defs or not _verity(REGISTRY[m[1]]):
        raise ValueError(f"{tid}: names no Definition in Verity's catalog")
    base = REGISTRY[m[1]]
    try:
        if m[2] is not None:
            fn = bind(base, **{k: _static(v) for k, v in _bindings(m[2]).items()})
        else:
            fn = bind(base) if template and isinstance(base, CompositeDefinition) else base
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"{tid}: {type(e).__name__}: {e}") from e
    if descriptor_id(fn) != tid:
        raise ValueError(f"{tid}: binds to {descriptor_id(fn)}, not to itself")
    if template:
        catalog(fn)
    return fn


def _fields(ret: Any) -> tuple[Any, ...]:
    return tuple(f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,)))


def _width(name: str, ty: Any) -> int:
    """A port's element width: its leaves', or for a Boolean-profile port (every leaf one bit) its `boolean.word_type`
    words', the same bits in the same order."""
    if ty.width_set() == {1}:
        ty = B.word_type(ty)
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


_WIRE_FIELDS = ("reader", "port", "writer", "out", "at", "radix", "base", "step")


@dataclass(frozen=True)
class Wire:
    """A wiring pattern: template ``reader``'s units ``at`` to ``at + prod(radix)`` read their ``port`` row whole as the
    ``out`` row of a unit of template ``writer``. Reader unit ``at + i`` is the digit vector d of i in mixed radix ``radix``
    (the first digit the slowest), and its writer is ``writer``'s unit ``base + sum(d[k] * step[k])``; each index counts
    its own template's units. A step of 0 is a fan-out: one writer row read by every unit along that digit."""
    reader: str
    port: str
    writer: str
    out: str
    at: int
    radix: tuple[int, ...]
    base: int
    step: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "radix", tuple(self.radix))
        object.__setattr__(self, "step", tuple(self.step))
        ints = [self.at, self.base, *self.radix, *self.step]
        if (not all(isinstance(x, str) for x in (self.reader, self.port, self.writer, self.out))
                or any(isinstance(x, bool) or not isinstance(x, int) for x in ints)
                or not self.radix or len(self.step) != len(self.radix) or min(self.radix) < 1 or self.at < 0):
            raise ValueError(f"{self.doc()}: a wire names its reader's port and its writer's output, and is a first reader unit, "
                             "at least one digit of radix 1 or more, a writer base and one integer step a digit")

    @property
    def units(self) -> int:
        return math.prod(self.radix)

    def span(self) -> tuple[int, int]:
        """The least and the greatest writer unit the wire reads."""
        reach = [s * (r - 1) for r, s in zip(self.radix, self.step)]
        return self.base + sum(min(0, x) for x in reach), self.base + sum(max(0, x) for x in reach)

    def writers(self, lo: int, n: int) -> np.ndarray:
        """The writer units (``writer``'s) of reader units ``lo`` to ``lo + n`` (``reader``'s), every one the wire's."""
        i = np.arange(lo - self.at, lo - self.at + n, dtype=np.int64)
        out = np.full(n, self.base, dtype=np.int64)
        for r, s in zip(reversed(self.radix), reversed(self.step)):
            out += (i % r) * s
            i //= r
        return out

    def doc(self) -> dict[str, Any]:
        return {"reader": self.reader, "port": self.port, "writer": self.writer, "out": self.out, "at": self.at,
                "radix": list(self.radix), "base": self.base, "step": list(self.step)}

    @classmethod
    def of(cls, d: Any) -> Wire:
        """A wire from its record (`doc`), refused unless it has exactly the wire's fields."""
        if (not isinstance(d, Mapping) or set(d) != set(_WIRE_FIELDS)
                or not all(isinstance(d[k], list) for k in ("radix", "step"))):
            raise ValueError(f"a wire's record is {{{', '.join(_WIRE_FIELDS)}}}, radix and step lists, not {d!r}")
        return cls(*(d[k] for k in _WIRE_FIELDS))


def check_wires(templates: Mapping[str, tuple[Layout, int]], wires: Sequence[Wire]) -> None:
    """Refuse a wire (ValueError) whose reader port or writer output is no port of its templates (``templates``: by descriptor
    id, its layout and units), whose units aren't among theirs, which reads a row of other words than its writer's, or which
    wires a reader unit's port twice. The work is per wire, never per unit."""
    spans: dict[tuple[str, str], list[tuple[int, int, Wire]]] = {}
    for w in wires:
        if w.reader not in templates or w.writer not in templates:
            raise ValueError(f"wire {w.doc()}: its reader and writer are templates of the circuit, {sorted(templates)}")
        (rl, rn), (wl, wn) = templates[w.reader], templates[w.writer]
        rows, outs = dict(rl.rows), dict(wl.outs)
        if w.port not in rows or w.out not in outs:
            raise ValueError(f"wire {w.doc()}: it reads one of {w.reader}'s parameters {sorted(rows)} as one of {w.writer}'s "
                             f"outputs {sorted(outs)}")
        if rows[w.port] != outs[w.out]:
            raise ValueError(f"wire {w.doc()}: {w.port} is {rows[w.port]} words and {w.out} {outs[w.out]}; a wire reads a row "
                             "whole, and a row gathered from several writers' words is no wire until circuits and proofs "
                             "choose its row form")
        lo, hi = w.span()
        if w.at + w.units > rn or lo < 0 or hi >= wn:
            raise ValueError(f"wire {w.doc()}: reader units [{w.at}, {w.at + w.units}) of {rn}, writer units [{lo}, {hi}] of "
                             f"{wn}; each among its template's")
        spans.setdefault((w.reader, w.port), []).append((w.at, w.at + w.units, w))
    for (tid, p), got in spans.items():
        got.sort(key=lambda s: s[0])
        for (_a, b, x), (c, _d, y) in zip(got, got[1:]):
            if c < b:
                raise ValueError(f"{tid} {p}: wires {x.doc()} and {y.doc()} both wire units from {c}")


def _ints(*xs: Any) -> bool:
    return all(type(x) is int for x in xs)


@dataclass(frozen=True)
class _Pattern:
    """The reader side of a pattern (`Read`): template ``reader``'s port ``port``, rows ``cut[0] .. cut[0] + C - 1``
    (``C = cut[1]``) of its cut (uncut, row 0 of 1). Reader index ``i < prod(radix)`` is unit ``at + i // C`` and row
    ``cut[0] + i % C``, and ``d`` its digits in mixed radix ``radix`` (the first digit the slowest; when ``C > 1`` the last
    digit is the row, of radix ``C``); its source is ``base + sum(d[k] * step[k])``, and a step of 0 a fan-out. The reader
    units it covers are ``[at, at + units / C)``."""
    reader: str
    port: str
    at: int
    radix: tuple[int, ...]
    base: int
    step: tuple[int, ...]
    cut: tuple[int, int] = (0, 1)
    #: the fields naming what the pattern reads, beside its reader port
    key_fields: ClassVar[tuple[str, ...]] = ()

    def key(self) -> tuple[Any, ...]:
        return tuple(getattr(self, k) for k in self.key_fields)

    def _check(self, names: Sequence[str], what: str) -> None:
        object.__setattr__(self, "radix", tuple(self.radix))
        object.__setattr__(self, "step", tuple(self.step))
        object.__setattr__(self, "cut", tuple(self.cut))
        if (not all(isinstance(x, str) for x in names) or not _ints(self.at, self.base, *self.radix, *self.step, *self.cut)
                or not self.radix or len(self.step) != len(self.radix) or min(self.radix) < 1 or self.at < 0
                or len(self.cut) != 2 or self.cut[0] < 0 or self.cut[1] < 1):
            raise ValueError(f"{self.doc()}: {what}, and is a first reader unit, at least one digit of radix 1 or more, a base, "
                             "one integer step a digit and its rows [c0, C] (C at least 1)")

    @property
    def units(self) -> int:
        return math.prod(self.radix)

    @property
    def reader_units(self) -> int:
        return self.units // self.cut[1]

    def span(self) -> tuple[int, int]:
        """The least and the greatest source the pattern reads."""
        reach = [s * (r - 1) for r, s in zip(self.radix, self.step)]
        return self.base + sum(min(0, x) for x in reach), self.base + sum(max(0, x) for x in reach)

    def source(self, i: np.ndarray) -> np.ndarray:
        """The sources of reader indices ``i``."""
        i = np.asarray(i, dtype=np.int64).copy()
        out = np.full(i.shape, self.base, dtype=np.int64)
        for r, s in zip(reversed(self.radix), reversed(self.step)):
            out += (i % r) * s
            i //= r
        return out

    def _doc(self) -> dict[str, Any]:
        return {"at": self.at, "radix": list(self.radix), "base": self.base, "step": list(self.step),
                **({"cut": list(self.cut)} if self.cut != (0, 1) else {})}


_READ_FIELDS = ("reader", "port", "param", "at", "radix", "base", "step")


def _of(cls: Any, d: Any, fields: Sequence[str], what: str) -> Any:
    if (not isinstance(d, Mapping) or not set(fields) <= set(d) <= {*fields, "cut"}
            or not all(isinstance(d[k], list) for k in ("radix", "step")) or ("cut" in d and not isinstance(d["cut"], list))):
        raise ValueError(f"a {what}'s record is {{{', '.join(fields)}}} and optionally its cut, its radix and step lists and its "
                         f"cut a list, not {d!r}")
    return cls(*(d[k] for k in fields), **({"cut": tuple(d["cut"])} if "cut" in d else {}))


@dataclass(frozen=True, init=False)
class Read(_Pattern):
    """A registered read at public positions (`_Pattern`): each reader row it covers is the row's words of the registered
    parameter ``param`` from its leaf ``base + sum(d[k] * step[k])`` on (16-bit leaves, the parameter's leaf order), the
    position the verifier derives from the record alone, as `one_stage.registered` reads a registered value's rows. A read
    of an output port ties the unit's output row to that row: each unit reads a row of its own, which the Lean verifier
    opens against the value's pinned root (a value the verifier pins from outside the prover, such as the output a
    network certifier relayed), and it fills no input."""
    param: str
    key_fields: ClassVar[tuple[str, ...]] = ("param",)

    def __init__(self, reader: str, port: str, param: str, at: int, radix: Sequence[int], base: int, step: Sequence[int],
                 cut: Sequence[int] = (0, 1)) -> None:
        for k, v in dict(reader=reader, port=port, param=param, at=at, radix=radix, base=base, step=step, cut=cut).items():
            object.__setattr__(self, k, v)
        self._check((reader, port, param), "a read names its reader's port and a registered parameter")

    def doc(self) -> dict[str, Any]:
        return {"reader": self.reader, "port": self.port, "param": self.param, **self._doc()}

    @classmethod
    def of(cls, d: Any) -> Read:
        return _of(cls, d, _READ_FIELDS, "read")


@dataclass(frozen=True)
class Table:
    """A registered table (`one_stage.registered` with ``words`` leaves a row): ``rows`` rows of the registered parameter
    ``param`` from its leaf ``offset``, committed one row a leaf so a row's rank is its key (`gates.embedding`)."""
    name: str
    param: str
    offset: int
    words: int
    rows: int

    def doc(self) -> dict[str, Any]:
        return {"name": self.name, "param": self.param, "offset": self.offset, "words": self.words, "rows": self.rows}


@dataclass(frozen=True)
class TableRead:
    """Template ``reader``'s units ``[at, at + units)`` read their ``port`` whole as table ``table``'s root or its domain
    id (``of``): a supplied value (`one_stage.registered`), never a free input."""
    reader: str
    port: str
    table: str
    of: str
    at: int
    units: int

    def doc(self) -> dict[str, Any]:
        return {"reader": self.reader, "port": self.port, "table": self.table, "of": self.of, "at": self.at, "units": self.units}


@dataclass(frozen=True)
class Opening:
    """Template ``reader``'s units ``[at, at + units)`` open table ``table``'s row at the hidden position their ``key`` port
    holds: ``ports`` are the opening (the row, its salt, its path), and ``verdict`` the output row whose bit is 1 exactly
    when they open to the table's root (`gates.embedding`'s ``ok``). On a verdict of 1 the opening is the table's row, salt
    and path at the key, under the binding of the row's hm96 leaf and frame-v3 tree: pinned, not free."""
    reader: str
    table: str
    key: str
    ports: tuple[str, ...]
    verdict: str
    at: int
    units: int

    def doc(self) -> dict[str, Any]:
        return {"reader": self.reader, "table": self.table, "key": self.key, "ports": list(self.ports), "verdict": self.verdict,
                "at": self.at, "units": self.units}


@dataclass(frozen=True)
class Registered:
    """What the registration fixes that units read: the registered root parameters by name with their leaves, the tables
    over them, and the reads (`Read`, `TableRead`, `Opening`)."""
    params: Mapping[str, int]
    tables: tuple[Table, ...] = ()
    reads: tuple[Read, ...] = ()
    roots: tuple[TableRead, ...] = ()
    openings: tuple[Opening, ...] = ()

    def doc(self) -> dict[str, Any]:
        return {"params": dict(sorted(self.params.items())), "tables": [t.doc() for t in self.tables],
                "reads": [r.doc() for r in self.reads], "roots": [r.doc() for r in self.roots],
                "openings": [o.doc() for o in self.openings]}

    @classmethod
    def of(cls, d: Any) -> Registered:
        try:
            if set(d) != {"params", "tables", "reads", "roots", "openings"}:
                raise ValueError(sorted(d))
            return cls({str(k): v for k, v in d["params"].items()}, tuple(Table(**t) for t in d["tables"]),
                       tuple(Read.of(r) for r in d["reads"]), tuple(TableRead(**r) for r in d["roots"]),
                       tuple(Opening(**{**o, "ports": tuple(o["ports"])}) for o in d["openings"]))
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            raise ValueError(f"a circuit's registered reads are {{params, tables, reads, roots, openings}}: {e}") from e


@dataclass(frozen=True)
class Ingress:
    """Template ``reader``'s units ``[at, at + units)`` read their ``port`` whole as the system's ingress at step ``step``
    (a request's slot, its prompt's ids and length, at its Program's anchor)."""
    reader: str
    port: str
    at: int
    units: int
    step: int

    def doc(self) -> dict[str, Any]:
        return {"reader": self.reader, "port": self.port, "at": self.at, "units": self.units, "step": self.step}


def _cut_ok(w: _Pattern, k: int, what: str) -> None:
    c0, C = w.cut
    if not (c0 + C <= k and (C == 1 or w.radix[-1] == C) and w.units % C == 0):
        raise ValueError(f"{what} {w.doc()}: rows [{c0}, {c0 + C}) of {w.port}'s {k}, its last digit {C} when C > 1, and C "
                         "dividing its reader rows")


def _disjoint(rects: Sequence[tuple[str, str, int, int, int, int, Any]], what: str) -> None:
    """Refuse two rectangles (reader units × rows) of one reader port that meet."""
    by: dict[tuple[str, str], list[tuple[int, int, int, int, Any]]] = {}
    for tid, p, a, b, r0, r1, d in rects:
        by.setdefault((tid, p), []).append((a, b, r0, r1, d))
    for (tid, p), got in by.items():
        got.sort(key=lambda s: s[0])
        live: list[tuple[int, int, int, int, Any]] = []
        for x in got:
            live = [y for y in live if y[1] > x[0]]
            for y in live:
                if y[2] < x[3] and x[2] < y[3]:
                    raise ValueError(f"{tid} {p}: {what} {y[4]} and {x[4]} both fill rows of units from {x[0]}")
            live.append(x)


def check_inputs(templates: Mapping[str, tuple[Layout, int]], cuts: Mapping[str, Mapping[str, int]], wires: Sequence[Wire],
                 registered: Registered | None, ingress: Sequence[Ingress]) -> None:
    """Refuse (ValueError) a registered read, table read, opening or ingress that names no port of its template, reads
    outside its units, its parameter or its table, or whose rows are another's: every input row is wired, registered,
    ingress or free, at most one of them. The work is per declaration, never per unit."""
    def port(tid: str, p: str, what: Any, outs: bool = False) -> tuple[int, int]:
        lay = templates[tid][0] if tid in templates else None
        ports = {} if lay is None else dict(lay.rows + (lay.outs if outs else ()))
        if p not in ports:
            raise ValueError(f"{what}: names no parameter of a template of the circuit" + (" nor its output" if outs else ""))
        return ports[p], int(cuts.get(tid, {}).get(p, 1))

    def units(tid: str, at: int, n: int, what: Any) -> None:
        if not _ints(at, n) or at < 0 or n < 1 or at + n > templates[tid][1]:
            raise ValueError(f"{what}: units [{at}, {at + n}) among its template's {templates[tid][1]}")

    rects = [(w.reader, w.port, w.at, w.at + w.units, 0, 1, w.doc()) for w in wires]
    reg = registered or Registered({})
    if not all(isinstance(v, int) and v > 0 for v in reg.params.values()):
        raise ValueError(f"registered parameters {dict(reg.params)}: each its leaves, at least one")
    tables = {}
    for t in reg.tables:
        if t.name in tables or t.param not in reg.params or not _ints(t.offset, t.words, t.rows) or t.offset < 0 \
                or t.words < 1 or t.rows < 1 or t.offset + t.words * t.rows > reg.params[t.param]:
            raise ValueError(f"table {t.doc()}: a new name, rows of a registered parameter's leaves")
        tables[t.name] = t
    for r in reg.reads:
        words, k = port(r.reader, r.port, r.doc(), outs=True)
        _cut_ok(r, k, "read")
        units(r.reader, r.at, r.reader_units, r.doc())
        lo, hi = r.span()
        if r.param not in reg.params or lo < 0 or hi + words // k > reg.params[r.param]:
            raise ValueError(f"read {r.doc()}: leaves [{lo}, {hi + words // k}) of a registered parameter {dict(reg.params)}")
        rects.append((r.reader, r.port, r.at, r.at + r.reader_units, r.cut[0], r.cut[0] + r.cut[1], r.doc()))
    for r in reg.roots:
        words, k = port(r.reader, r.port, r.doc())
        units(r.reader, r.at, r.units, r.doc())
        if r.table not in tables or r.of not in ("root", "domain") or words != 32:
            raise ValueError(f"table read {r.doc()}: a declared table's root or domain id, a 64-byte port")
        rects.append((r.reader, r.port, r.at, r.at + r.units, 0, k, r.doc()))
    for o in reg.openings:
        units(o.reader, o.at, o.units, o.doc())
        port(o.reader, o.key, o.doc())
        if o.table not in tables or o.verdict not in dict(templates[o.reader][0].outs) or not o.ports:
            raise ValueError(f"opening {o.doc()}: a declared table's row at its key, with a verdict output")
        for p in o.ports:
            rects.append((o.reader, p, o.at, o.at + o.units, 0, port(o.reader, p, o.doc())[1], o.doc()))
    for g in ingress:
        _words, k = port(g.reader, g.port, g.doc())
        units(g.reader, g.at, g.units, g.doc())
        if not _ints(g.step) or g.step < 0:
            raise ValueError(f"ingress {g.doc()}: at a step, a natural")
        rects.append((g.reader, g.port, g.at, g.at + g.units, 0, k, g.doc()))
    _disjoint(rects, "inputs")


def _committable(registered: Registered | None, ingress: Sequence[Ingress], what: str) -> None:
    """Refuse (ValueError) declarations the committer doesn't commit yet: a window commits registered reads at public
    positions, whole and uncut, and nothing else of them."""
    if ingress or (registered is not None and (registered.tables or registered.roots or registered.openings
                                               or any(r.cut != (0, 1) for r in registered.reads))):
        raise ValueError(f"{what}: a window commits uncut registered reads at public positions (`Read`); its tables, table "
                         "reads, openings, cut reads and ingress are recorded by their description until the committer "
                         "commits them")


@dataclass(frozen=True)
class Circuit:
    """An integration's circuit in proof units: ``templates``, each ``(template, units)`` in order, every template `catalog`'s
    and of its own descriptor id; ``external``, the ports (``<descriptor id>/<port>``) that are the system's external I/O;
    ``wires``, the units whose input rows are other units' output rows (`Wire`, `check_wires`); ``boundary``, its I/O
    layout (network-accounting's N1, ``verity/boundary/v1``), a JSON object carried as given; ``registered`` and
    ``ingress``, what fills its other inputs (`check_inputs`)."""
    name: str
    templates: tuple[tuple[Any, int], ...]
    external: tuple[str, ...] = ()
    wires: tuple[Wire, ...] = ()
    boundary: Mapping[str, Any] | None = None
    registered: Registered | None = None
    ingress: tuple[Ingress, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "templates", tuple((fn, int(n)) for fn, n in self.templates))
        object.__setattr__(self, "external", tuple(self.external))
        object.__setattr__(self, "wires", tuple(self.wires))
        object.__setattr__(self, "ingress", tuple(self.ingress))
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
        if not all(isinstance(w, Wire) for w in self.wires):
            raise ValueError(f"{self.name}: its wires are `Wire`s")
        check_wires(self.layouts, self.wires)
        check_inputs(self.layouts, {}, self.wires, self.registered, self.ingress)
        if self.boundary is not None:
            if not isinstance(self.boundary, Mapping):
                raise ValueError(f"{self.name}: its boundary is a JSON object, not {self.boundary!r}")
            object.__setattr__(self, "boundary", json.loads(json.dumps(self.boundary)))

    @property
    def ids(self) -> list[str]:
        return [descriptor_id(fn) for fn, _ in self.templates]

    @property
    def layouts(self) -> dict[str, tuple[Layout, int]]:
        """Each template's layout and units, by descriptor id."""
        return {tid: (layout(fn), n) for (fn, n), tid in zip(self.templates, self.ids)}

    @property
    def reads(self) -> tuple[Read, ...]:
        return self.registered.reads if self.registered is not None else ()

    def declared(self) -> dict[str, Any]:
        """The record's fields for its registered reads and ingress, each only when the circuit has one, so a circuit
        without them records as before."""
        return {**({"registered": self.registered.doc()} if self.registered is not None else {}),
                **({"ingress": [g.doc() for g in self.ingress]} if self.ingress else {})}

    def description(self) -> dict[str, Any]:
        """The record's fields `input_count` reads: its templates, wires and declarations, none of its program."""
        return json.loads(json.dumps({"templates": [[t, n] for t, (_, n) in zip(self.ids, self.templates)],
                                      "wires": [w.doc() for w in self.wires], **self.declared()}))

    def program(self) -> tuple[Any, dict[str, Any]]:
        """The verifier's population program and template query over the templates (`rows.population_program`)."""
        return CS.population_program(list(self.templates))

    def template(self, tid: str) -> Any:
        """The template of descriptor id ``tid``."""
        for fn, _ in self.templates:
            if descriptor_id(fn) == tid:
                return fn
        raise KeyError(f"{tid}: no template of {self.name}")


def read_shape(circuit: Circuit, value: str) -> dict[str, Any]:
    """Registered parameter ``value``'s rows as the circuit's reads fix them, the shape an outside committer (a network
    certifier's output value, `out_commit.OutShape`) commits it at: ``words``, the row width of the port every read of it
    names; ``rows``, its leaves over that; 16-bit leaves in ``hm96-sha512/row/v2`` rows (`unit_rows`). Refused
    (ValueError) when no read names it, two read it at two widths, or its leaves aren't whole rows."""
    lays = circuit.layouts
    widths = {dict(lays[r.reader][0].rows + lays[r.reader][0].outs)[r.port] for r in circuit.reads if r.param == value}
    leaves = (circuit.registered.params if circuit.registered is not None else {}).get(value)
    if not widths or leaves is None:
        raise ValueError(f"{circuit.name}: no registered read of {value!r}")
    if len(widths) != 1:
        raise ValueError(f"{circuit.name}: {value!r} is read at {sorted(widths)} words a row, not one width")
    (w,) = widths
    if leaves % w:
        raise ValueError(f"{circuit.name}: {value!r}'s {leaves} leaves aren't whole rows of {w} words")
    return {"value": value, "words": w, "rows": leaves // w, "word_bits": 16, "row_schema": V2}


def members(circuit: Circuit, cap: int) -> tuple[CS.Member, ...]:
    """The window's members: each template's units in the partition's order, in chunks ``<descriptor id>/c<j>`` of at most
    ``cap`` (one statement each), on the template's bit rows, each wired by the circuit's wires that cover it."""
    program, query = circuit.program()
    return _members(circuit.templates, program, query, cap, circuit.wires, circuit.reads)


def _reads_of(tid: str, lo: int, n: int, reads: Sequence[Read], rows: Mapping[str, int]) -> dict[str, tuple[str, list[int], np.ndarray]]:
    """Template ``tid``'s units ``[lo, lo + n)`` (its own indices), one member's, by each port a read fills: the parameter
    it reads, the registered rows its table holds (each a row of the port's words, distinct, in first-read order: core's
    ``registered.table``) and each unit's ref into them; ``rows``: the template's input rows' words, by port. A read covers
    whole members and reads whole registered rows: its base and steps are multiples of the port's words."""
    out = {}
    for r in reads:
        if r.reader != tid or not (r.at < lo + n and lo < r.at + r.reader_units):
            continue
        words = rows[r.port]
        if lo < r.at or lo + n > r.at + r.reader_units:
            raise ValueError(f"read {r.doc()} covers part of {tid}'s units [{lo}, {lo + n}); a read covers whole members")
        if r.cut != (0, 1) or r.base % words or any(s % words for s in r.step):
            raise ValueError(f"read {r.doc()}: an uncut read of whole registered rows, its base and steps multiples of "
                             f"{r.port}'s {words} words")
        order: dict[int, int] = {}
        refs = np.array([order.setdefault(int(p), len(order)) for p in r.source(np.arange(lo - r.at, lo - r.at + n)) // words],
                        dtype=np.int64)
        out[r.port] = (r.param, list(order), refs)
    return out


def member_reads(circuit: Circuit, m: CS.Member, base: int) -> dict[str, tuple[str, list[int], np.ndarray]]:
    """Member ``m``'s registered reads (`_reads_of`), its template's first unit the global index ``base``."""
    return _reads_of(m.template, m.base - base, m.n, circuit.reads, dict(m.rows + m.outs))


def _members(templates: Sequence[tuple[Any, int]], program: Any, query: Mapping[str, Any], cap: int,
             wires: Sequence[Wire], reads: Sequence[Read] = ()) -> tuple[CS.Member, ...]:
    """`members` of ``templates`` (``(Definition, units)``) under their population ``program`` and ``query``. A wire covers
    whole members: a member's port is wired on all its units or on none (`rows.Member.wired`). So does a registered read
    (`_reads_of`), and a member it covers is a shared-row member: each read port's table the registered rows its units
    read, each other input port's its units' own rows."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"a statement holds at least one unit, not {cap!r}")
    lays = {descriptor_id(fn): (layout(fn), n) for fn, n in templates}
    check_wires(lays, wires)
    ranges = CS.template_partition(program, query).ranges
    base = {tid: lo for _k, tid, lo, _n in ranges}
    out = []
    for _k, tid, lo, n in ranges:
        lay = lays[tid][0]
        for j, at in enumerate(range(0, n, cap)):
            m, wired = min(cap, n - at), []
            for w in (w for w in wires if w.reader == tid and at < w.at + w.units and w.at < at + m):
                if at < w.at or at + m > w.at + w.units:
                    raise ValueError(f"wire {w.doc()} covers part of {tid}/c{j}'s units [{at}, {at + m}); a wire covers whole "
                                     f"members, chunks of {cap} of their template's units")
                wired.append((w.port, w.out, tuple((base[w.writer] + w.writers(at, m)).tolist())))
            got = _reads_of(tid, at, m, reads, dict(lay.rows + lay.outs))
            tables = tuple((p, len(got[p][1]) if p in got else m) for p, _ in lay.rows) if got else ()
            out.append(CS.Member(tid, lay.rows, lay.outs, m, lo + at, tables, name=f"{tid}/c{j}", schemas=lay.schemas,
                                 wired=tuple(wired)))
    return tuple(out)


def advice_binding(compiler: Mapping[str, str], root: str, bits: int) -> dict[str, Any]:
    """The registration's binding of a circuit's compiler and advice: the compiler's ``name`` and ``digest``, with the
    ``commit`` its code is at when the integration gives one (where a verifier fetches code that hashes to the digest,
    `compiler_digest`), and the root and length in bits of the advice's hiding commitment (`verity.ml.commitments.advice`),
    each refused unless well-formed."""
    if not {"name", "digest"} <= set(compiler) <= {"name", "digest", "commit"} or not isinstance(compiler["name"], str) \
            or not _DIGEST.fullmatch(str(compiler["digest"])) or ("commit" in compiler and not _COMMIT.fullmatch(str(compiler["commit"]))):
        raise ValueError(f"a compiler is its name, its 64-byte digest in hex and optionally its git commit, not {dict(compiler)!r}")
    if not _DIGEST.fullmatch(str(root)):
        raise ValueError(f"the advice's root is 64 bytes, 128 hex digits, not {root!r}")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 0:
        raise ValueError(f"the advice's length is a whole number of bits, not {bits!r}")
    return {"compiler": {k: compiler[k] for k in ("name", "digest", "commit") if k in compiler}, "advice": {"root": root, "bits": bits}}


def code_files(root: str | Path, paths: Sequence[str]) -> dict[str, str]:
    """The files git lists under the repository ``paths`` at the work tree ``root`` (tracked, or untracked and not ignored, so
    build outputs don't count; ``*.md`` aside, which computes nothing), each by its path to the SHA-512 of its bytes. Refused
    where ``root`` is not a git work tree, or holds no file under ``paths``."""
    root = Path(root)
    p = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths], cwd=root,
                       capture_output=True)
    if p.returncode != 0:
        raise ValueError(f"{root} is not a git work tree: {p.stderr.decode(errors='replace').strip()[-200:]}")
    files = sorted({f for f in p.stdout.decode().split("\0") if f and not f.endswith(".md") and (root / f).is_file()})
    if not files:
        raise ValueError(f"no file under {list(paths)} at {root}")
    return {f: hashlib.sha512((root / f).read_bytes()).hexdigest() for f in files}


def compiler_digest(name: str, root: str | Path, code: Sequence[str]) -> str:
    """A registered compiler's digest: its name and its code (`code_files` under the repository paths ``code`` at the work tree
    ``root``). The integration computes it on the tree it runs from; the verifier recomputes it on its own checkout of the
    commit the registration names, over the paths its own table names, and runs the compiler only from that checkout."""
    return hashlib.sha512(canonical_json({"name": str(name), "code": code_files(root, code)})).hexdigest()


def commit_of(root: str | Path) -> str:
    """The commit the work tree ``root`` is at, in hex: the ``commit`` an integration registers its compiler under."""
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(root), capture_output=True, text=True)
    if p.returncode != 0 or not _COMMIT.fullmatch(p.stdout.strip()):
        raise ValueError(f"{root} is not a git work tree at a commit: {p.stderr.strip()[-200:]}")
    return p.stdout.strip()


def advice_tree_binding(compiler: Mapping[str, str]) -> bytes:
    """The advice tree's domain binding: `ADVICE_SCHEME` and the registered compiler's name and digest, so a root opens only
    under the compiler it names."""
    return hashlib.sha512(canonical_json({"scheme": ADVICE_SCHEME, "compiler": {"name": str(compiler["name"]),
                                                                                "digest": compiler["digest"]}})).digest()


def advice_rows(data: bytes) -> list[bytes]:
    """The advice's bytes as `ADVICE_ROW_BITS`-bit rows, the last zero-padded: as many rows as its public length fixes."""
    w = ADVICE_ROW_BITS // 8
    return [bytes(data[i:i + w]).ljust(w, b"\0") for i in range(0, max(len(data), 1), w)]


def commit_advice(compiler: Mapping[str, str], data: bytes, salts: Sequence[bytes] | None = None) -> AC.Registration:
    """The advice's hiding commitment (`verity.ml.commitments.advice`: `advice_rows` under `advice_tree_binding`), fresh salts
    unless ``salts`` is given."""
    return AC.register(advice_rows(data), ADVICE_ROW_BITS, advice_tree_binding(compiler), salts)


def advice_opening(data: bytes, reg: AC.Registration) -> dict[str, Any]:
    """The advice's opening, the prover's private file: its bytes and every row's salt. Handing it to the verifier stands in for
    the compile unit (the named gap)."""
    return {"scheme": ADVICE_SCHEME, "advice": bytes(data).hex(), "salts": [s.hex() for s in reg.salts]}


def opened_advice(compiler: Mapping[str, str], doc: Any) -> tuple[bytes, str]:
    """The verifier's side of `advice_opening`: the advice's bytes and the root they commit to under ``compiler`` with the
    opening's salts, which the registration's must be. A malformed opening is a ValueError."""
    if not isinstance(doc, Mapping) or doc.get("scheme") != ADVICE_SCHEME:
        raise ValueError(f"an advice opening is a {ADVICE_SCHEME} document")
    try:
        data, salts = bytes.fromhex(doc["advice"]), [bytes.fromhex(s) for s in doc["salts"]]
        return data, commit_advice(compiler, data, salts).root.hex()
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"an advice opening holds the advice and one salt a row, in hex: {e}") from e


def circuit_record(w: CS.Window, name: str, templates: Sequence[tuple[str, int]],
                   statements: Mapping[str, Mapping[str, Any]] | None = None, external: Sequence[str] = (), *,
                   cap: int | None = None, wires: Sequence[Wire] = (),
                   boundary: Mapping[str, Any] | None = None, declared: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``circuit.json``: everything the verifier's window is (`of_records`), none of its values. ``templates``: each
    ``(descriptor id, units)`` in the program's order; ``statements``: by template, how its unit's statement reads its rows
    (``in_rows``, ``out_rows``, ``reads``, as C-Flock's `UnitStager` takes them), for a template whose members are bit rows.
    The members are listed, or with ``cap`` stated: the verifier derives them from the templates, ``cap``, ``wires`` and
    registered reads (`members`), so the record grows with its templates and wires, not its units. ``boundary``: the
    circuit's I/O layout (`Circuit.boundary`), carried as given; ``declared``: its registered reads and ingress
    (`Circuit.declared`)."""
    if declared and cap is None:
        raise ValueError("a record that lists its members has no registered reads or ingress; they are stated with a cap")
    doc = {"format": CIRCUIT, "name": name, "templates": [[t, int(n)] for t, n in templates], "query": dict(w.query),
           "program": w.partition["program"], "partition": w.partition_digest, "run": w.run, "context": w.context,
           "work_table": w.work_table, "external": list(external),
           "row_tables": [{"segment": t.segment, "name": t.name, "template": t.template, "words": t.words,
                           "keys": [list(k) for k in t.keys], "label": [[k, v] for k, v in t.label]} for t in w.row_tables],
           "statements": {t: {"in_rows": [[p, s, b, list(ps)] for p, s, b, ps in st["in_rows"]],
                              "out_rows": [[s, list(ix)] for s, ix in st["out_rows"]], "reads": st.get("reads", "inline")}
                          for t, st in sorted((statements or {}).items())}}
    if cap is not None:
        doc |= {"cap": int(cap), "wires": [x.doc() for x in wires], **json.loads(json.dumps(dict(declared or {})))}
    elif wires:
        raise ValueError("a record that lists its members lists their wires with them; wires are stated with a cap")
    else:
        doc["members"] = [{"template": m.template, "name": m.name, "n": m.n, "base": m.base, "rows": [list(r) for r in m.rows],
                           "outs": [list(o) for o in m.outs], "schemas": [list(s) for s in m.schemas],
                           "tables": [list(t) for t in m.tables], "link_units": list(m.link_units),
                           "links": [list(x) for x in m.links], "wired": [[p, o, list(ws)] for p, o, ws in m.wired]}
                          for m in w.members]
    if boundary is not None:
        doc["boundary"] = json.loads(json.dumps(boundary))
    return doc


def templates_of(w: CS.Window) -> list[tuple[str, int]]:
    """``w``'s templates in its program's order, each ``(descriptor id, units)``: `circuit_record`'s ``templates``."""
    return [(t, n) for _k, t, _lo, n in CS.template_partition(w.program, w.query).ranges]


#: the circuit record's fields its window's run fixes rather than its circuit (the run root, the coins' context), which the
#: registration binds itself: a compiler's output is the record less these (`circuit_of`)
RUN_FIELDS = ("run", "context")


def circuit_of(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A circuit record less its `RUN_FIELDS`: what a registered compiler's output on the advice must be."""
    return {k: v for k, v in doc.items() if k not in RUN_FIELDS}


def circuit_bytes(doc: Mapping[str, Any]) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode() + b"\n"


#: a stated record's optional fields beside its cap and wires (`Circuit.declared`)
DECLARED = ("cuts", "registered", "ingress")


def _cuts(cuts: Any) -> dict[str, dict[str, int]]:
    if not isinstance(cuts, Mapping) or not all(isinstance(kv, Mapping) for kv in cuts.values()):
        raise ValueError(f"a circuit's cuts are {{template: {{port: k}}}}, not {cuts!r}")
    return {str(t): {str(p): k for p, k in sorted(kv.items())} for t, kv in sorted(cuts.items()) if kv}


def declarations(doc: Mapping[str, Any]) -> tuple[tuple[Wire, ...], dict[str, dict[str, int]], Registered | None,
                                                   tuple[Ingress, ...]]:
    """A stated record's wires, cuts, registered reads and ingress, each refused (ValueError) unless well-formed; none is
    checked against the templates here (`check_wires`, `check_inputs`)."""
    wires = tuple(Wire.of(x) for x in doc.get("wires", ()))
    cuts = _cuts(doc.get("cuts", {}))
    registered = Registered.of(doc["registered"]) if "registered" in doc else None
    try:
        ingress = tuple(Ingress(**g) for g in doc.get("ingress", ()))
    except TypeError as e:
        raise ValueError(f"a circuit's ingress is a list of {{reader, port, at, units, step}}: {e}") from e
    return wires, cuts, registered, ingress


def input_count(doc: Mapping[str, Any], resolve: Callable[[str], Any] = resolve) -> dict[str, int]:
    """A stated circuit record's inputs by what fills them, in bits, from its description alone (its templates, wires and
    declarations, `Circuit.description`; per template, wire and declaration, never per unit, and never its program): every
    unit's input rows (``inputs``), the rows its wires fill (``wired``, a wire's reader rows times its writer row's bits,
    `SampledProofs.Wires`' ``wiredBits``), its registered reads, table reads and openings (``registered``) and its ingress
    (``ingress``); ``free`` the rest, which only the prover chose. Refused unless the declarations are well-formed for the
    record's templates (`check_wires`, `check_inputs`), so no row counts twice."""
    if "wires" not in doc:
        raise ValueError("a circuit record whose members are listed is counted from its members, not its description")
    defs = [(resolve(t), int(n)) for t, n in doc["templates"]]
    lays = {descriptor_id(fn): (layout(fn), n) for fn, n in defs}
    wires, cuts, reg, ingress = declarations(doc)
    if cuts:
        raise ValueError(f"cuts {cuts}: a cut port's rows are counted once a committer commits them")
    check_wires(lays, wires)
    check_inputs(lays, cuts, wires, reg, ingress)

    def row_bits(tid: str, port: str) -> int:
        return 16 * dict(lays[tid][0].rows)[port] // int(cuts.get(tid, {}).get(port, 1))

    inputs = sum(16 * sum(w for _, w in lay.rows) * n for lay, n in lays.values())
    wired = sum(w.units * row_bits(w.reader, w.port) for w in wires)
    reg = reg or Registered({})
    registered = (sum(r.units * row_bits(r.reader, r.port) for r in reg.reads if r.port in dict(lays[r.reader][0].rows))
                  + sum(r.units * 16 * dict(lays[r.reader][0].rows)[r.port] for r in reg.roots)
                  + sum(o.units * 16 * dict(lays[o.reader][0].rows)[p] for o in reg.openings for p in o.ports))
    ingress_bits = sum(g.units * 16 * dict(lays[g.reader][0].rows)[g.port] for g in ingress)
    units = sum(n for _, n in lays.values())
    return {"units": units, "inputs": inputs, "wired": wired, "registered": registered, "ingress": ingress_bits,
            "free": inputs - wired - registered - ingress_bits}


def registered_reads(doc: Mapping[str, Any], w: CS.Window, pins: Mapping[str, Mapping[str, Any]], *,
                     registered: Any) -> dict[str, dict[str, Any]]:
    """The Lean verifier's ``--registered`` file of each member reading a registered parameter (``registered-reads/v1``, in
    the format of ``registered``, core's ``one_stage.registered``, by member key), from the stated record ``doc`` and its
    window ``w`` (`window_of_record`'s) alone, but for each parameter's entry from ``pins`` (``{param: entry}``: the value's
    ``root``, ``leaves``, ``words``, ``word_bits`` and ``domain`` as the verifier pinned them before the run): each read port
    reads its parameter's rows in the port's row format, and unit ``m.base + i`` the row its ``i``-th unit reads (`_reads_of`).
    Refused (ValueError) where a read's parameter has no pin, or a pin's rows are not the port's."""
    _wires, _cuts_, reg, _ingress = declarations(doc)
    if reg is None or not reg.reads:
        return {}
    base = {t: b for _k, t, b, _n in CS.template_partition(w.program, w.query).ranges}
    out: dict[str, dict[str, Any]] = {}
    for m in w.members:
        got = _reads_of(m.template, m.base - base[m.template], m.n, reg.reads, dict(m.rows + m.outs))
        if not got:
            continue
        ports = {}
        for p, (param, positions, refs) in sorted(got.items()):
            e = pins.get(param)
            words = dict(m.rows + m.outs)[p]
            if e is None:
                raise ValueError(f"{m.key} {p} reads registered parameter {param!r}, whose root the verifier didn't pin")
            if (e.get("words"), e.get("word_bits", registered.WORD_BITS)) != (words, registered.WORD_BITS) \
                    or e.get("value") != param:
                raise ValueError(f"{m.key} {p}: the pin of {param!r} is {e.get('value')!r} rows of {e.get('words')} "
                                 f"{e.get('word_bits', registered.WORD_BITS)}-bit leaves, not the port's {words} 16-bit words")
            ports[p] = {"value": param, "root": e["root"], "leaves": int(e["leaves"]), "words": words,
                        "word_bits": registered.WORD_BITS, "domain": e["domain"],
                        registered.SCHEMA_KEY: m.schema(p), "reads": [positions[int(i)] for i in refs],
                        registered.READS_FROM: m.base}
        out[m.key] = {"format": registered.READS_V1, "ports": ports}
    return out


def _frozen(x: Any) -> Any:
    """A JSON value with every list a tuple, as a row table's keys and labels are held."""
    return tuple(_frozen(v) for v in x) if isinstance(x, list) else x


def window_of_record(doc: Mapping[str, Any], law: Mapping[str, Any], resolve: Callable[[str], Any]) -> CS.Window:
    """The verifier's window from ``circuit.json``'s document under its own ``law``, each template's Definition by
    ``resolve`` (descriptor id -> Definition, the verifier's own catalog), its members the listed ones or, where the record
    states them, the ones its templates, ``cap`` and wires make (`members`); refused unless the population program, query,
    program digest and partition digest it rebuilds are the record's."""
    if doc.get("format") != CIRCUIT:
        raise ValueError(f"a window's circuit record is {CIRCUIT}, not {doc.get('format')!r}")
    defs = [(resolve(t), int(n)) for t, n in doc["templates"]]
    if [descriptor_id(fn) for fn, _ in defs] != [t for t, _ in doc["templates"]]:
        raise ValueError(f"the catalog resolves {[t for t, _ in doc['templates']]} to {[descriptor_id(fn) for fn, _ in defs]}")
    several = len(defs) > 1 or doc["query"] != PO.template_instance_query(defs[0][0])
    program, query = CS.population_program(defs, several=several)
    stated = "cap" in doc
    if ("members" in doc) == stated or ("wires" in doc) != stated or (not stated and set(DECLARED) & set(doc)):
        raise ValueError("a circuit record lists its members, or states them by its cap, wires and registered reads, one or "
                         "the other")
    if stated:
        if not isinstance(doc["wires"], list):
            raise ValueError(f"a circuit record's wires are a list, not {doc['wires']!r}")
        wires, cuts, registered, ingress = declarations(doc)
        if cuts:
            raise ValueError(f"cuts {cuts}: a window commits whole uncut rows")
        lays = {descriptor_id(fn): (layout(fn), n) for fn, n in defs}
        check_wires(lays, wires)
        check_inputs(lays, cuts, wires, registered, ingress)
        _committable(registered, ingress, "the circuit record")
        ms = _members(defs, program, query, doc["cap"], wires, registered.reads if registered is not None else ())
    else:
        ms = tuple(CS.Member(m["template"], tuple(map(tuple, m["rows"])), tuple(map(tuple, m["outs"])), m["n"], m["base"],
                             tuple(map(tuple, m["tables"])), m["name"], tuple(map(tuple, m["schemas"])), tuple(m["link_units"]),
                             tuple(map(tuple, m["links"])), tuple((p, o, tuple(ws)) for p, o, ws in m["wired"]))
                   for m in doc["members"])
    tables = tuple(CS.RowTable(t["segment"], t["name"], t["template"], t["words"], _frozen(t["keys"]),
                               tuple((k, _frozen(v)) for k, v in t["label"])) for t in doc["row_tables"])
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
class _Recorded:
    inputs: list[np.ndarray]
    outputs: list[np.ndarray]
    words: list[np.ndarray]      # every port's (n, words) u16, parameters then return fields
    label: Any
    n: int


@dataclass
class Window:
    """One window of an integration's values: ``circuit``'s units under ``run`` (the record's run root, hex) and ``law`` (the
    proof service's law of the window, as its draw parses it), in members of at most ``cap`` units, with ``advice`` its
    compiler and advice (`advice_binding`) when the circuit depends on run-time choices, and ``registered`` the values its
    registered reads read (`Read`), by parameter: each committed once, before the run, as core's
    ``one_stage.registered.Registered`` (rows of the reading port's words, its salts the prover's), which the verifier
    pinned by its entry. Each template's units are `record`ed in order, then `close` commits them and `write` writes and
    registers the window."""
    circuit: Circuit
    run: str
    law: Mapping[str, Any]
    cap: int
    advice: Mapping[str, Any] | None = None
    registered: Mapping[str, Any] | None = None
    work_table: Mapping[str, Any] | None = None
    window: CS.Window = field(init=False)
    _recorded: dict[str, list[_Recorded]] = field(init=False, default_factory=dict)
    _entries: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        _committable(self.circuit.registered, self.circuit.ingress, self.circuit.name)
        params = self.circuit.registered.params if self.circuit.registered is not None else {}
        for r in self.circuit.reads:
            v = (self.registered or {}).get(r.param)
            lay = layout(self.circuit.template(r.reader))
            words = dict(lay.rows + lay.outs)[r.port]
            if v is not None and (v.words != words or v.words * v.leaves != params[r.param]):
                raise ValueError(f"{r.param}: committed as {v.leaves} rows of {v.words} words, not the {params[r.param]} "
                                 f"leaves {r.port}'s rows of {words} words read")
        program, query = self.circuit.program()
        self.window = CS.Window(program, query, dict(self.law), self.run, members(self.circuit, self.cap),
                                work_table=None if self.work_table is None else dict(self.work_table))
        if self.advice is not None:
            self.advice = advice_binding(self.advice["compiler"], self.advice["advice"]["root"], self.advice["advice"]["bits"])

    def count(self, tid: str) -> int:
        """Template ``tid``'s units recorded so far."""
        return sum(b.n for b in self._recorded.get(tid, ()))

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
        self._recorded.setdefault(tid, []).append(_Recorded(values[:len(inputs)], values[len(inputs):], ws, label, n))
        return range(at, at + n)

    def mismatches(self, kernel: str = "reference") -> dict[str, list[int]]:
        """Per template, the units whose recorded outputs are not its Definition's outputs on their recorded inputs
        (`evaluate_batch` under ``kernel``): what the proof of a drawn one would refuse."""
        out = {}
        for fn, _n in self.circuit.templates:
            tid, bad, at = descriptor_id(fn), [], 0
            leaves = [len(f.leaf_widths()) for f in _fields(fn.signature()[1])]
            for b in self._recorded.get(tid, ()):
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
        base, cat = self._bases(), {}
        data, salts_of, self._entries = [], {}, {}
        for m in self.window.members:
            lay = layout(self.circuit.template(m.template))
            if m.template not in cat:
                got = self._recorded[m.template]
                cat[m.template] = [np.concatenate([b.words[g] for b in got]) for g in range(len(lay.rows) + len(lay.outs))]
            ports, lo = cat[m.template], m.base - base[m.template]
            d = {"rows": {p: ports[g][lo:lo + m.n] for g, (p, _) in enumerate(lay.rows)},
                 "outs": {p: ports[len(lay.rows) + j][lo:lo + m.n] for j, (p, _) in enumerate(lay.outs)}}
            reads = member_reads(self.circuit, m, base[m.template])
            if reads:
                d["refs"] = np.stack([reads[p][2] if p in reads else np.arange(m.n) for p, _ in lay.rows], axis=1)
                for p, (param, positions, refs) in reads.items():
                    if (v := (self.registered or {}).get(param)) is None:
                        raise ValueError(f"{m.key} {p} reads registered parameter {param!r}: give its committed value")
                    if p in d["outs"]:
                        # an output row stays the unit's own, committed under the value's salt, so a row other than
                        # the value's opens no pinned root
                        _rows, salts, entry = v.statement([positions[int(i)] for i in refs])
                        salts_of[m.port(p)] = b"".join(salts)
                        self._entries.setdefault(m.key, {})[p] = entry
                        continue
                    first = [int(np.flatnonzero(refs == j)[0]) for j in range(len(positions))]
                    rows = d["rows"][p][first]
                    if not np.array_equal(rows[refs], d["rows"][p]):
                        raise ValueError(f"{m.key} {p}: two units recorded two rows for one row of {param!r}")
                    _rows, salts, entry = v.statement(positions)
                    d["rows"][p] = rows
                    salts_of[m.port(p)] = b"".join(salts)
                    self._entries.setdefault(m.key, {})[p] = entry
            data.append(d)
        return CS.commit(self.window, data, workers=workers, salt_key=salt_key, salts_of=salts_of)

    def _bases(self) -> dict[str, int]:
        return {t: b for _k, t, b, _n in CS.template_partition(self.window.program, self.window.query).ranges}

    def labels(self) -> dict[str, Any]:
        """``index.json``: per member, each recorded run of its units with its label, ``[lo, hi, label]`` over global indices."""
        spans: dict[str, list[tuple[int, int, Any]]] = {}
        for tid, lo in self._bases().items():
            for b in self._recorded.get(tid, ()):
                spans.setdefault(tid, []).append((lo, lo + b.n, b.label))
                lo += b.n
        return {"format": LABELS, "run": self.run,
                "members": [{"name": m.key, "labels": [[max(a, m.base), min(b, m.base + m.n), x]
                                                       for a, b, x in spans.get(m.template, ()) if a < m.base + m.n and b > m.base]}
                            for m in self.window.members]}

    def circuit_record(self) -> dict[str, Any]:
        """The window's ``circuit.json`` (`circuit_record`), its members stated by its cap and the circuit's wires, each
        template's statement its bit rows (`unit_rows`)."""
        sts = {}
        for fn, _ in self.circuit.templates:
            i, o = unit_rows(fn)
            sts[descriptor_id(fn)] = {"in_rows": i, "out_rows": o, "reads": "inline"}
        return circuit_record(self.window, self.circuit.name, list(zip(self.circuit.ids, (n for _, n in self.circuit.templates))),
                              sts, self.circuit.external, cap=self.cap, wires=self.circuit.wires, boundary=self.circuit.boundary,
                              declared=self.circuit.declared())

    def write(self, c: CS.Committed, out_dir: str | Path, provenance: Mapping[str, Any], *, registrar: Any,
              kind: str = "served") -> dict[str, Any]:
        """The window's files (`rows.write`, its ``index.json`` the labels) and ``circuit.json``, the registration core's record
        made by ``registrar`` (core's ``one_stage.registration``), its window naming ``provenance``, the circuit record's
        SHA-512 and the advice binding; each member reading a registered value, its header naming each table row's position
        and path (`close`)."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        body = circuit_bytes(self.circuit_record())
        (d / "circuit.json").write_bytes(body)
        bound = {**provenance, "circuit": hashlib.sha512(body).hexdigest(), **(self.advice or {})}
        return CS.write(c, d, bound, self.labels(), kind=kind, registered=self._entries, registrar=registrar)


class Integration(Protocol):
    """What an integration provides: its circuit, and a run that records every unit of it into a window (`Window.record`)
    before the window closes."""

    def circuit(self) -> Circuit: ...

    def run(self, window: Window) -> None: ...
