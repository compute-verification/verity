"""What an integration hands Verity (integrations are frontends, and Verity is agnostic to them), one window
at a time, in four records:

1. **A circuit in proof blocks** (:class:`Circuit`): templates, each a complete binding of a Definition in Verity's catalog
   (`catalog`), with its unit count, in order. Their population program and template query (`rows.population_program`)
   partition it into proof blocks, and each unit's public layout is its Definition's bit rows (`layout`); ``external``
   names the ports that are the system's external I/O, ``wires`` the units that read another unit's output row whole
   (:class:`Wire`, one pattern for many units), and ``boundary`` its I/O layout as the network certifier states it. The
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

An integration may label each run of units it records (``label``): the window's ``index.json``, prover only, which the
registration binds by its hiding leaf (`rows.index_leaf`). A label is class structure, never a value. In a circuit with
blocks, a labelled run starts and ends where its template's units change inference block (their (slot, step), the
inference entries'), and its label names at most the run's ``step`` and ``slot`` and its ``form``, one of the window's
``forms`` (`LABEL_KEYS`, `Window.record`); the index writes an unlabelled stretch of runs as one run, so every bound it
shows is such a change, a member's or a template's. A circuit without blocks states no inference blocks, and its labels
are opaque JSON. Verity reads a label only to refuse it.

The verifier's window comes from the records alone (`of_records`: ``circuit.json`` held to the registration's digest),
never from the integration's code. A unit's rows are its Definition's bit rows: one row per parameter, the little-endian
bit concatenation of its leaves, then one output row ``o<j>`` per return field, zero-padded to whole u16 words: C-Flock's
`UnitStager` rows (`unit_rows`), so the verifier composes the unit's circuit from the Definition alone; an integration's
lowering refuses, with `check_proof_block`, a template whose rows C-Flock's composer would refuse (more than 256 rows, none
past 111 bytes, or past a C-Flock block of 2^27 bits). A Boolean-profile port's leaves (one bit each) are read as their
`boolean.word_type` words.

A wired port's row is committed under its writer's output row's salt, so the two rows' public strings are one, which
the wiring check holds on the drawn units (`wiring.check_drawn`). A wire reads a row whole: a row gathered from several
writers' words (RoPE's query from the Q coordinates, attention's K and V from earlier positions) is refused except in one
form, the wire cut (``cuts``, `SampledProofs.Wires`): a port read from several writers is cut ``k`` ways, parameter ``p``
is ``k`` rows ``p#j``, row ``j`` its outer elements ``[j·n/k, (j+1)·n/k)``, and a wire covers some of a port's rows
(`Wire.cut`). A unit that reads one row gathered from several writers' outputs reads it through a gather unit
(``AllGather_v1``) whose ``parts`` are cut.

The circuit also declares what fills its other inputs (`Registered`, `Ingress`): rows of a registered parameter read at
public positions (`Read`), a registered table's root or domain id (`TableRead`), a table's row opened at a hidden key
(`Opening`), and the system's ingress at its step (`Ingress`). Every input row is wired, one of these, or free, and no row
is two of them (`check_inputs`); `input_count` counts the free rows' bits from the record alone, at the size of its
description, a registered row free unless the verifier pinned that parameter's own root before the run and the window's
registration records the same root for it (``registered_roots``, `Window.write`; `held_params`). The window commits
the first three as registered reads at public positions (`Window`'s ``registered``, `committed_reads`): a member whose
units read a registered value is a shared-row member, each read port a table of the registered rows its units read,
committed under the registered value's salts (core's ``one_stage.registered``), so each row's public string is the
registered one, and its header names each row's position and path to the value's root; the verifier's ``--registered``
file of the member comes from the record and the roots it pinned alone (`registered_reads`), and Lean's registered reads
open each row against them. A table is a registered parameter of its own, every row of it; a table
read takes a supplied value (`supplied`), one row the verifier fills from its pin of the table's parameter, the root or
the domain id, under public salts; an opening's verdict reads the table's supplied 1, so a verdict of 0 opens no root,
and the opening's row, salt and path are its units' own values, which the verdict holds to the pinned root inside the
proof at the hidden key. Cut registered reads and ingress are refused by the committer until it commits them. A `Read`
may also name an output row ``o<j>``: that row's words are the parameter's leaves at the pattern's positions (a value a
server sends out, its response or a wire to another server). It fills no input row, and wires may read the row too.

The circuit's blocks (`Blocks`, ``verity/blocks/v1``, the record's ``blocks``) are stated per template and step, never per
unit: ``proof``, the partition into proof units (its units and partition digest, which the verifier holds to the record's,
and the planner's ``plan`` digest, carried and never read); ``inference``, the slot and step of every unit (`InferenceEntry`:
the entries tile each template's units exactly once, a slot digit's radix is the class's S and every step is below its T);
and network accounting's ``server``, ``server_time`` and ``transfers``, carried as given for its own check. Of those the
members read only the servers' unit ranges: a template's units are cut where a server's begin or end, so each statement's
units sit on one server. Blocks name units, ports, steps and row patterns, never a value.

That the circuit computes what the integration ran is completeness, never proved: the verifier accepts a window exactly
when its drawn units' committed outputs are their Definitions' outputs on their committed inputs (`Window.mismatches` is
the integration's own check before it commits).
"""
from __future__ import annotations

import bisect
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
from verity.ml.circuits.types import Array
from verity.ml.commitments import advice as AC
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2

from . import rows as CS

#: the element widths a parameter's or return field's leaves may have: a row is their little-endian bytes
PACKED = (8, 16, 32, 64)
CIRCUIT = "verity/window-circuit/v1"
LABELS = "verity/window-labels/v1"
#: what a label in a circuit with blocks may name: its run's step and slot, and its form (`Window.record`)
LABEL_KEYS = ("step", "slot", "form")
BLOCKS = "verity/blocks/v1"
#: what an inference entry's digit is: the unit's slot, its step past the entry's ``step``, or neither
AXES = ("slot", "step", "-")
#: network accounting's keys of the blocks, carried as given for its check; the members read the servers' unit ranges
NETWORK = ("server", "server_time", "transfers")
#: what a table's supplied values hold (`supplied`): its root, its domain id, and 1, what an opening's verdict reads
SUPPLIED = ("root", "domain", "opened")
#: the advice tree's binding tag (`advice_tree_binding`): its bytes are its first records', a later rename
ADVICE_SCHEME = "verity-vllm/advice-windows/advice/v1"
#: the advice's bytes as rows of this many bits, the last zero-padded (`advice_rows`)
ADVICE_ROW_BITS = 256
_DIGEST = re.compile("[0-9a-f]{128}")
#: a plan's digest, SHA-256 or SHA-512 in hex
_PLAN = re.compile("[0-9a-f]{64}|[0-9a-f]{128}")
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
    """Import every module of `verity.ml.circuits.ops`, the catalog's, the embedding's registered read
    (`commitments.gates.embedding`) and the streamed SHA-512 a checkpoint tie reads (`commitments.gates.sha512_stream`):
    each registers its Definitions as it loads."""
    from verity.ml.circuits import ops

    for m in pkgutil.walk_packages(ops.__path__, f"{ops.__name__}."):
        if ".tests" not in m.name:
            importlib.import_module(m.name)
    importlib.import_module("verity.ml.commitments.gates.embedding")
    importlib.import_module("verity.ml.commitments.gates.sha512_stream")


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
            fn = bind(base) if isinstance(base, CompositeDefinition) else base
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"{tid}: {type(e).__name__}: {e}") from e
    if descriptor_id(fn) != tid:
        raise ValueError(f"{tid}: binds to {descriptor_id(fn)}, not to itself")
    if template:
        catalog(fn)
    return fn


def _fields(ret: Any) -> tuple[Any, ...]:
    return tuple(f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,)))


def _bits(ty: Any) -> int:
    return sum(ty.leaf_widths())


def _width(name: str, ty: Any, out: bool = False) -> int:
    """A port's element width: its leaves', or for a Boolean-profile port (every leaf one bit) its `boolean.word_type`
    words', the same bits in the same order. A return field of fewer than 16 bits (an embedding read's ``ok``) is one u16
    word, its bits low and zero-padded as C-Flock's `UnitStager` pads every output field, recorded as a 16-bit integer."""
    if out and 0 < _bits(ty) < 16:
        return 16
    if ty.width_set() == {1}:
        ty = B.word_type(ty)
    widths = ty.width_set()
    if len(widths) != 1 or not widths <= set(PACKED) or sum(ty.leaf_widths()) % 16:
        raise ValueError(f"{name}: its leaves are {sorted(widths)} bits, {sum(ty.leaf_widths())} in all; a row is whole u16 "
                         f"words of leaves of one of {PACKED} bits")
    return next(iter(widths))


@dataclass(frozen=True)
class RowLayout:
    """A template's bit rows: ``rows`` each parameter's ``(name, words)``, ``outs`` each return field's ``(o<j>, words)``,
    each one's element width, and each parameter's outer length (its outermost array's, 1 when it is not an array), which
    a cut divides."""
    rows: tuple[tuple[str, int], ...]
    outs: tuple[tuple[str, int], ...]
    widths: tuple[int, ...]
    outer: tuple[int, ...] = ()

    @property
    def schemas(self) -> tuple[tuple[str, str], ...]:
        return tuple((p, V2) for p, _ in self.rows + self.outs)

    def cut(self, cuts: Mapping[str, int] | None = None) -> list[tuple[str, int, list[str]]]:
        """Each input row under ``cuts`` (port -> k), ``(name, words, elements)``: an uncut port is its row ``p``; one cut
        ``k`` ways is ``k`` rows ``p#j``, row ``j`` its outer elements ``p[r]`` for r in ``[j·n/k, (j+1)·n/k)``, as C-Flock's
        `class_statement.unit_ports` reads them."""
        out = []
        for (p, w), n in zip(self.rows, self.outer):
            k = int((cuts or {}).get(p, 1))
            if k == 1:
                out.append((p, w, [p]))
            else:
                out += [(row_name(p, j, k), w // k, [f"{p}[{r}]" for r in range(j * n // k, (j + 1) * n // k)]) for j in range(k)]
        return out


def row_name(port: str, j: int, k: int) -> str:
    """Row ``j`` of ``port`` cut ``k`` ways: the port itself when uncut."""
    return port if k == 1 else f"{port}#{j}"


def layout(fn: Any) -> RowLayout:
    """``fn``'s bit rows (module docstring); refused when a port is not whole u16 words of one ``PACKED`` width."""
    params, ret = fn.signature()
    rows = tuple((name, _bits(ty) // 16) for name, ty in params)
    outs = tuple((f"o{j}", -(-_bits(ty) // 16)) for j, ty in enumerate(_fields(ret)))
    if len({p for p, _ in rows + outs}) != len(rows) + len(outs):
        raise ValueError(f"{fn.id}: a parameter is named like an output row {[p for p, _ in outs]}")
    widths = tuple(_width(f"{fn.id} {name}", ty) for name, ty in params) + tuple(
        _width(f"{fn.id} {p}", ty, out=True) for (p, _), ty in zip(outs, _fields(ret)))
    return RowLayout(rows, outs, widths, tuple(ty.n if isinstance(ty, Array) else 1 for _, ty in params))


def unit_rows(fn: Any, cuts: Mapping[str, int] | None = None, seg: Sequence[str] = ()
              ) -> tuple[list[tuple[str, str, int, list[str]]], list[tuple[str, tuple[int, ...]]]]:
    """``fn``'s rows as C-Flock's `UnitStager` takes them, ``(in_rows, out_rows)``: one ``hm96-sha512/row/v2`` row per parameter,
    or per row of a cut one (`Layout.cut`), then one per return field; a parameter in ``seg`` (`seg_ports`) is one
    ``hm96-sha512/row-seg/v1`` row."""
    lay = layout(fn)
    return [(p, SEG if p in seg else V2, 16 * w, ps) for p, w, ps in lay.cut(cuts)], [(V2, (j,)) for j in range(len(lay.outs))]


#: C-Flock's composer's sizes (`verity_flock.circuit`, which ``verity`` may not import; the windows tests hold them to
#: its own): a C-Flock block of at most 2^K_MAX bits and at least 2^K_MIN, the sha512x3 and hm96 slots of 2^SLOT_LOG bits,
#: PER_SLOT compressions a sha512x3 slot, the mask's 2^MASK_LOG
K_MAX, K_MIN, SLOT_LOG, PER_SLOT, MASK_LOG = 27, 20, 18, 3, 14
#: the row bytes a row's last SHA-512 block holds beside the padding's 17: a longer row has a compression before it
LAST_BLOCK_BYTES = 111
#: the most rows a proof block may have: at one more, its hm96 range alone fills a C-Flock block of 2^K_MAX bits
MAX_ROWS = 1 << (K_MAX - SLOT_LOG - 1)


def rows_k_log(row_bytes: Sequence[int]) -> int:
    """The log of the C-Flock block that rows of ``row_bytes`` bytes alone need (C-Flock's composer's ranges for one unit:
    its hm96 slots, one a row and a power of two in all, the mask, then its sha512x3 slots); the unit's gates come on top."""
    comps = sum(-(-(b - LAST_BLOCK_BYTES) // 128) for b in row_bytes if b > LAST_BLOCK_BYTES)
    bits = ((1 << (len(row_bytes) - 1).bit_length()) + 1 + -(-comps // PER_SLOT)) << SLOT_LOG
    return max(K_MIN, (bits - 1).bit_length())


def check_proof_block(fn: Any, cuts: Mapping[str, int] | None = None) -> int:
    """Refuse (ValueError) a template whose proof block, its unit's rows under ``cuts`` (`unit_rows`), C-Flock's composer
    would refuse: more than `MAX_ROWS` rows, none past `LAST_BLOCK_BYTES` bytes (the sha512x3 range, which holds the pin,
    would have no slot), or past a C-Flock block of 2^`K_MAX` bits. Its message names the template, its cuts, its rows and
    the fix. Returns the rows' `rows_k_log`, the least the proof block's ``k_log`` can be."""
    lay = layout(fn)
    rows = [(p, 2 * w) for p, w, _ in lay.cut(cuts)] + [(p, 2 * w) for p, w in lay.outs]
    sizes: dict[str, list[int]] = {}
    for p, b in rows:
        sizes.setdefault(p.split("#")[0], []).append(b)
    name = descriptor_id(fn) + (f" cut {dict(cuts)}" if cuts else "") + ": " + ", ".join(
        f"{p} {len(bs)} rows of {bs[0]} B" if len(bs) > 1 else f"{p} {bs[0]} B" for p, bs in sizes.items())
    if len(rows) > MAX_ROWS:
        raise ValueError(f"{name}: {len(rows)} rows, past a proof block's {MAX_ROWS}, since their hm96 slots alone would fill "
                         f"a C-Flock block of 2^{K_MAX} bits: cut its ports fewer ways, or bind the Definition smaller")
    if all(b <= LAST_BLOCK_BYTES for _, b in rows):
        raise ValueError(f"{name}: no row past {LAST_BLOCK_BYTES} B, so none has a compression before its last SHA-512 block "
                         "and C-Flock's sha512x3 range, which holds the pin, has no slot: cut its ports fewer ways, or bind "
                         "the Definition larger")
    if (k := rows_k_log([b for _, b in rows])) > K_MAX:
        raise ValueError(f"{name}: its rows alone need a C-Flock block of 2^{k} bits, past 2^{K_MAX}: bind the Definition "
                         "smaller, so that each unit reads and writes fewer row bytes")
    return k


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


def _port_leaves(rows: np.ndarray, ty: Any) -> np.ndarray:
    """A port's ``(n, words)`` u16 rows as its ``(n, leaves)`` leaves of type ``ty``, as `evaluate_batch` takes them: a row
    is the little-endian bit concatenation of the leaves, so a 16-bit leaf is one word, and a 1-bit leaf (a Boolean-profile
    port's, whose rows are its `boolean.word_type` words) one bit."""
    b, w = np.ascontiguousarray(rows, dtype="<u2").view(np.uint8), ty.homogeneous_width()
    if w is not None and w % 8 == 0:
        return np.ascontiguousarray(b[:, :ty.leaves * w // 8]).view(f"<u{w // 8}")
    bits = np.unpackbits(b, axis=1, bitorder="little")
    if w == 1:
        return bits[:, :ty.leaves]
    at = np.cumsum([0, *ty.leaf_widths()])
    return np.stack([(bits[:, lo:hi].astype(np.uint64) << np.arange(hi - lo, dtype=np.uint64)).sum(axis=1)
                     for lo, hi in zip(at, at[1:])], axis=1)


def _port_rows(leaves: np.ndarray, ty: Any, count: int) -> np.ndarray:
    """`_port_leaves`' inverse: ``(n, leaves)`` of type ``ty`` as their ``(n, count)`` u16 rows, zero-padded."""
    w = ty.homogeneous_width()
    if w is not None and w % 8 == 0:
        b = np.ascontiguousarray(leaves.astype(f"<u{w // 8}")).view(np.uint8)
    elif w == 1:
        b = np.packbits(leaves.astype(np.uint8), axis=1, bitorder="little")
    else:
        a = leaves.astype(np.uint64)
        bits = [(a[:, e:e + 1] >> np.arange(x, dtype=np.uint64)) & np.uint64(1) for e, x in enumerate(ty.leaf_widths())]
        b = np.packbits(np.concatenate(bits, axis=1).astype(np.uint8), axis=1, bitorder="little")
    out = np.zeros((len(leaves), 2 * count), dtype=np.uint8)
    out[:, :b.shape[1]] = b
    return out.view("<u2")


def _ints(*xs: Any) -> bool:
    return all(type(x) is int for x in xs)


@dataclass(frozen=True)
class _Pattern:
    """The reader side of a pattern (`Wire`, `Read`): template ``reader``'s port ``port``, rows ``cut[0] .. cut[0] + C - 1``
    (``C = cut[1]``) of its cut (`Circuit.cuts`; uncut, row 0 of 1). Reader index ``i < prod(radix)`` is unit
    ``at + i // C`` and row ``cut[0] + i % C``, and ``d`` its digits in mixed radix ``radix`` (the first digit the slowest;
    when ``C > 1`` the last digit is the row, of radix ``C``); its source is ``base + sum(d[k] * step[k])``, and a step of 0
    a fan-out. The reader units it covers are ``[at, at + units / C)``."""
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


_WIRE_FIELDS = ("reader", "port", "writer", "out", "at", "radix", "base", "step")
_READ_FIELDS = ("reader", "port", "param", "at", "radix", "base", "step")


def _of(cls: Any, d: Any, fields: Sequence[str], what: str) -> Any:
    if (not isinstance(d, Mapping) or not set(fields) <= set(d) <= {*fields, "cut"}
            or not all(isinstance(d[k], list) for k in ("radix", "step")) or ("cut" in d and not isinstance(d["cut"], list))):
        raise ValueError(f"a {what}'s record is {{{', '.join(fields)}}} and optionally its cut, its radix and step lists and its "
                         f"cut a list, not {d!r}")
    return cls(*(d[k] for k in fields), **({"cut": tuple(d["cut"])} if "cut" in d else {}))


@dataclass(frozen=True, init=False)
class Wire(_Pattern):
    """A wiring pattern (`_Pattern`): each reader row it covers is the ``out`` row of unit ``base + sum(d[k] * step[k])`` of
    template ``writer``, each index counting its own template's units (the wire cut's form, `SampledProofs.Wires`). With no cut
    it reads the port's one row, a whole-port wire."""
    writer: str
    out: str
    key_fields: ClassVar[tuple[str, ...]] = ("writer", "out")

    def __init__(self, reader: str, port: str, writer: str, out: str, at: int, radix: Sequence[int], base: int,
                 step: Sequence[int], cut: Sequence[int] = (0, 1)) -> None:
        for k, v in dict(reader=reader, port=port, writer=writer, out=out, at=at, radix=radix, base=base, step=step,
                         cut=cut).items():
            object.__setattr__(self, k, v)
        self._check((reader, port, writer, out), "a wire names its reader's port and its writer's output")

    def writers(self, lo: int, n: int, row: int | None = None) -> np.ndarray:
        """The writer units (``writer``'s) of reader units ``lo`` to ``lo + n`` (``reader``'s) at reader row ``row`` (the
        wire's first by default), every one the wire's."""
        r = self.cut[0] if row is None else row
        return self.source((np.arange(lo, lo + n, dtype=np.int64) - self.at) * self.cut[1] + r - self.cut[0])

    def doc(self) -> dict[str, Any]:
        return {"reader": self.reader, "port": self.port, "writer": self.writer, "out": self.out, **self._doc()}

    @classmethod
    def of(cls, d: Any) -> Wire:
        """A wire from its record (`doc`), refused unless it has exactly the wire's fields (``cut`` optional)."""
        return _of(cls, d, _WIRE_FIELDS, "wire")


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


def coalesce(patterns: Sequence[Any]) -> list[Any]:
    """``patterns`` with every run of like patterns merged into one: patterns of one reader port, source, rows and digits
    whose reader units follow on (each ``at`` its predecessor's ``at + units / C``) and whose bases step by one constant
    ``D`` are one pattern with a new first digit (radix the run's length, step ``D``) and no digit of radix 1; merged again
    until none merges. The rows each reader row reads are the same, so a reader's description shrinks to its runs."""
    out = list(patterns)
    while True:
        out.sort(key=lambda w: (type(w).__name__, w.reader, w.port, w.key(), w.cut, w.radix, w.step, w.at))
        merged, i, changed = [], 0, False
        while i < len(out):
            w, j = out[i], i + 1
            while (j < len(out) and type(out[j]) is type(w) and (out[j].reader, out[j].port, out[j].key(), out[j].cut,
                                                                   out[j].radix, out[j].step) == (w.reader, w.port, w.key(),
                                                                                                  w.cut, w.radix, w.step)
                   and out[j].at == out[j - 1].at + w.reader_units
                   and (j == i + 1 or out[j].base - out[j - 1].base == out[i + 1].base - w.base)):
                j += 1
            if j - i > 1:
                digits = [(j - i, out[i + 1].base - w.base), *((r, s) for r, s in zip(w.radix, w.step) if r != 1)]
                fields = {**{k: getattr(w, k) for k in ("reader", "port", *w.key_fields)}, "at": w.at, "base": w.base,
                          "radix": tuple(r for r, _ in digits), "step": tuple(s for _, s in digits), "cut": w.cut}
                merged.append(type(w)(**fields))
                changed = True
            else:
                merged.append(w)
            i = j
        out = merged
        if not changed:
            return sorted(out, key=lambda w: (type(w).__name__, w.reader, w.port, w.at, w.cut))


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
    id (``of``): a supplied value (`one_stage.registered`), free unless the window's root for its parameter is the pinned
    root (`input_count`)."""
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
    and path at the key, under the binding of the row's hm96 leaf and frame-v3 tree: not free where the window's root for
    the table's parameter is the pinned root (`input_count`)."""
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


def check_wires(templates: Mapping[str, tuple[RowLayout, int]], wires: Sequence[Wire],
                cuts: Mapping[str, Mapping[str, int]] | None = None) -> None:
    """Refuse (ValueError) the circuit's cuts and wires as `SampledProofs.Wires.check` does, in its order (``templates``: by
    descriptor id, its layout and units): a cut naming no template's parameter or not dividing its outer length and words
    (``cuts``); then wire by wire, a reader or writer that is no template (``template``), a port no parameter of the reader
    or an output no output of the writer (``port``), rows not among the port's or a last digit that isn't ``C`` (``cut``),
    a reader row of other words than its writer's output row (``width``), reader or writer units outside their templates
    (``bounds``); then two wires wiring one reader row (``overlap``). The work is per wire, never per unit."""
    cuts = cuts or {}
    for tid, kv in cuts.items():
        lay = templates[tid][0] if tid in templates else None
        rows = dict(zip((p for p, _ in lay.rows), zip((w for _, w in lay.rows), lay.outer))) if lay else {}
        for p, k in (kv.items() if isinstance(kv, Mapping) else [(None, None)]):
            if p not in rows or not _ints(k) or k < 1 or rows[p][1] % k or rows[p][0] % k:
                raise ValueError(f"cuts: {tid} {p} cut {k!r} ways; a cut names a template's parameter and divides its outer "
                                 f"length and words {rows.get(p)}")
    for w in wires:
        if w.reader not in templates or w.writer not in templates:
            raise ValueError(f"wire {w.doc()}: its reader and writer are templates of the circuit, {sorted(templates)}")
        (rl, rn), (wl, wn) = templates[w.reader], templates[w.writer]
        rows, outs = dict(rl.rows), dict(wl.outs)
        if w.port not in rows or w.out not in outs:
            raise ValueError(f"wire {w.doc()}: it reads one of {w.reader}'s parameters {sorted(rows)} as one of {w.writer}'s "
                             f"outputs {sorted(outs)}")
        k = int(cuts.get(w.reader, {}).get(w.port, 1))
        _cut_ok(w, k, "wire")
        if rows[w.port] // k != outs[w.out]:
            raise ValueError(f"wire {w.doc()}: a row of {w.port} is {rows[w.port] // k} words and {w.out} {outs[w.out]}; a wire "
                             "reads its writer's output row whole, and a row gathered from several writers is cut or read "
                             "through a gather unit")
        lo, hi = w.span()
        if w.at + w.reader_units > rn or lo < 0 or hi >= wn:
            raise ValueError(f"wire {w.doc()}: reader units [{w.at}, {w.at + w.reader_units}) of {rn}, writer units [{lo}, {hi}] "
                             f"of {wn}; each among its template's")
    _disjoint([(w.reader, w.port, w.at, w.at + w.reader_units, w.cut[0], w.cut[0] + w.cut[1], w.doc()) for w in wires], "wires")


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


def check_inputs(templates: Mapping[str, tuple[RowLayout, int]], cuts: Mapping[str, Mapping[str, int]], wires: Sequence[Wire],
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

    rects = [(w.reader, w.port, w.at, w.at + w.reader_units, w.cut[0], w.cut[0] + w.cut[1], w.doc()) for w in wires]
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
    """Refuse (ValueError) declarations the committer doesn't commit: a cut registered read, an ingress, and a table that
    isn't its parameter's every row, since its reads take its parameter's root and domain (`supplied`)."""
    reg = registered or Registered({})
    if ingress or any(r.cut != (0, 1) for r in reg.reads):
        raise ValueError(f"{what}: a window commits uncut registered reads (`Read`), table reads and openings; its cut "
                         "reads and ingress are recorded by their description until the committer commits them")
    if bad := [t.doc() for t in reg.tables if t.offset or t.words * t.rows != reg.params[t.param]]:
        raise ValueError(f"{what}: tables {bad} are not their parameters' every row; a table's reads take its parameter's "
                         "root, so a table is a registered parameter of its own")


def digest_words(d: bytes) -> list[int]:
    """A 64-byte digest as a `Digest` port's 32 words (`gates.rowleaf.digest_bits`: its eight words big-endian, each least
    significant bit first, so each word's little-endian bytes)."""
    if len(d) != 64:
        raise ValueError(f"a digest is 64 bytes, not {len(d)}")
    return np.frombuffer(d, ">u8").astype("<u8").view("<u2").tolist()


def supplied_param(table: Table, of: str) -> str:
    """The supplied value a read of ``table``'s ``of`` (`SUPPLIED`) takes: ``<its parameter>/<of>``."""
    return f"{table.param}/{of}"


def committed_reads(registered: Registered | None) -> tuple[Read, ...]:
    """Every registered read a window commits (`Window.close`) and its verifier opens (`registered_reads`): the circuit's
    reads (`Read`); each table read, of its table's supplied root or domain id; and each opening's verdict, of its table's
    supplied 1 (an output row, so a verdict of 0 opens no root: `gates.embedding`'s ``ok``). A table read or verdict reads
    one row for all its units; an opening's other ports are its units' own rows, which its verdict holds to the table's
    pinned root inside the proof, its key hidden."""
    reg = registered or Registered({})
    tables = {t.name: t for t in reg.tables}
    out = list(reg.reads)
    out += [Read(r.reader, r.port, supplied_param(tables[r.table], r.of), r.at, (r.units,), 0, (0,)) for r in reg.roots]
    out += [Read(o.reader, o.verdict, supplied_param(tables[o.table], "opened"), o.at, (o.units,), 0, (0,))
            for o in reg.openings]
    return tuple(out)


def seg_ports(registered: Registered | None) -> dict[str, frozenset[str]]:
    """By template, the ports its registered reads (`Read`) fill with a declared table's parameter's rows, which are
    ``hm96-sha512/row-seg/v1`` leaves (`gates.embedding`), so the port is a row of that schema (`unit_rows`) where every
    other is ``hm96-sha512/row/v2``. Refused (ValueError) where one port reads a table's rows and another parameter's: a
    template's port has one row schema."""
    reg = registered or Registered({})
    tables = {t.param for t in reg.tables}
    by: dict[tuple[str, str], set[bool]] = {}
    for r in reg.reads:
        by.setdefault((r.reader, r.port), set()).add(r.param in tables)
    if bad := sorted(k for k, v in by.items() if len(v) > 1):
        raise ValueError(f"{bad}: a template's port reads a table's row-seg rows and another parameter's row/v2 rows; a port "
                         "has one row schema")
    out: dict[str, set[str]] = {}
    for (tid, p), v in by.items():
        if True in v:
            out.setdefault(tid, set()).add(p)
    return {tid: frozenset(ps) for tid, ps in out.items()}


def supplied(registered: Registered | None, entries: Mapping[str, Mapping[str, Any]], program: str, *,
             module: Any) -> dict[str, Any]:
    """The values a window's table reads and verdicts read (`committed_reads`), by name, each one row filled from the entry
    of its table's parameter alone (``entries``, core's ``Registered.entry()`` by parameter, as the verifier pinned it):
    the root's and the domain id's 64 bytes as a `Digest` port's 32 words (`digest_words`) and 1 as one word, each
    committed as core's ``registered.Registered`` (``module``, core's ``one_stage.registered`` as the caller passes it)
    under ``program`` (the window's partition digest) at its public salts (``supplied_salts``), so the prover and the
    verifier make the same value and the prover chooses none of it. A table whose parameter has no entry has none."""
    out = {}
    for t in (registered.tables if registered is not None else ()):
        if (e := entries.get(t.param)) is None:
            continue
        for of, row in zip(SUPPLIED, (digest_words(bytes.fromhex(e["root"])), digest_words(bytes.fromhex(e["domain"])), [1])):
            name = supplied_param(t, of)
            out[name] = module.Registered.commit(program, name, [row], module.supplied_salts(program, name, 1), schema=V2)
    return out


def _cuts(cuts: Any) -> dict[str, dict[str, int]]:
    if not isinstance(cuts, Mapping) or not all(isinstance(kv, Mapping) for kv in cuts.values()):
        raise ValueError(f"a circuit's cuts are {{template: {{port: k}}}}, not {cuts!r}")
    return {str(t): {str(p): k for p, k in sorted(kv.items())} for t, kv in sorted(cuts.items()) if kv}


_ENTRY_FIELDS = ("template", "at", "radix", "axes", "step")


@dataclass(frozen=True)
class InferenceEntry:
    """Template ``template``'s units ``[at, at + prod(radix))``: unit ``at + i`` has ``i``'s digits in mixed radix ``radix``
    (the first the slowest, as a `Wire`'s), ``axes`` names each digit (`AXES`), and the unit's inference block is (its slot
    digit, ``step`` + its step digit). One digit is the slot, at most one the step."""
    template: str
    at: int
    radix: tuple[int, ...]
    axes: tuple[str, ...]
    step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "radix", tuple(self.radix))
        object.__setattr__(self, "axes", tuple(self.axes))
        if (not isinstance(self.template, str) or not _ints(self.at, self.step, *self.radix) or self.at < 0 or self.step < 0
                or not self.radix or min(self.radix) < 1 or len(self.axes) != len(self.radix)
                or not all(a in AXES for a in self.axes) or self.axes.count("slot") != 1 or self.axes.count("step") > 1):
            raise ValueError(f"inference entry {self.doc()}: a template, a first unit and a step, naturals, and one axis of "
                             f"{AXES} a digit of radix 1 or more, one of them the slot and at most one the step")

    @property
    def units(self) -> int:
        return math.prod(self.radix)

    def radix_of(self, axis: str) -> int:
        """The radix of the entry's ``axis`` digit, 1 when it has none."""
        return self.radix[self.axes.index(axis)] if axis in self.axes else 1

    def digit(self, axis: str, i: np.ndarray) -> np.ndarray:
        """Units ``at + i``'s ``axis`` digits, 0 when the entry has none."""
        if axis not in self.axes:
            return np.zeros_like(i)
        k = self.axes.index(axis)
        return i // math.prod(self.radix[k + 1:]) % self.radix[k]

    def doc(self) -> dict[str, Any]:
        return {"template": self.template, "at": self.at, "radix": list(self.radix), "axes": list(self.axes), "step": self.step}

    @classmethod
    def of(cls, d: Any) -> InferenceEntry:
        if not isinstance(d, Mapping) or set(d) != set(_ENTRY_FIELDS) or not isinstance(d["radix"], list) \
                or not isinstance(d["axes"], list):
            raise ValueError(f"an inference entry is {{{', '.join(_ENTRY_FIELDS)}}}, its radix and axes lists, not {d!r}")
        return cls(*(d[k] for k in _ENTRY_FIELDS))


@dataclass(frozen=True)
class Blocks:
    """A circuit's blocks (module docstring): its class's ``slots`` (S) and ``steps`` (T), the ``inference`` entries,
    ``network`` accounting's keys (`NETWORK`, all or none) carried as given, and ``plan``, the planner's plan digest, carried
    and never read. Its record's ``proof`` is the circuit's own (`doc`)."""
    slots: int
    steps: int
    inference: tuple[InferenceEntry, ...]
    network: Mapping[str, Any] | None = None
    plan: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inference", tuple(self.inference))
        if not _ints(self.slots, self.steps) or self.slots < 1 or self.steps < 1:
            raise ValueError(f"blocks of {self.slots!r} slots and {self.steps!r} steps: a class's slots and steps, at least one")
        if not all(isinstance(e, InferenceEntry) for e in self.inference):
            raise ValueError("a circuit's inference entries are `InferenceEntry`s")
        if self.plan is not None and not (isinstance(self.plan, str) and _PLAN.fullmatch(self.plan)):
            raise ValueError(f"a plan's digest is SHA-256 or SHA-512 in hex, not {self.plan!r}")
        if self.network is not None:
            if not isinstance(self.network, Mapping) or set(self.network) != set(NETWORK):
                raise ValueError(f"network accounting's blocks are {list(NETWORK)}, all of them, not {self.network!r}")
            net = json.loads(json.dumps(dict(self.network)))
            if not all(isinstance(net[k], list) for k in NETWORK):
                raise ValueError(f"network accounting's {list(NETWORK)} are lists")
            object.__setattr__(self, "network", net)

    def check(self, units: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
        """Refuse (ValueError) entries that name no template of ``units`` (each template's unit count), don't tile each
        template's units exactly once, have a slot digit of another radix than ``slots`` or a step at or past ``steps``, and
        servers whose unit ranges aren't among their templates' units. -> each template's server ends (`server_ends`). The
        work is per entry and per server range, never per unit."""
        by: dict[str, list[InferenceEntry]] = {}
        for e in self.inference:
            if e.template not in units:
                raise ValueError(f"inference entry {e.doc()}: names no template of the circuit")
            if e.radix_of("slot") != self.slots:
                raise ValueError(f"inference entry {e.doc()}: its slot digit's radix is the class's {self.slots} slots")
            if e.step + e.radix_of("step") > self.steps:
                raise ValueError(f"inference entry {e.doc()}: steps [{e.step}, {e.step + e.radix_of('step')}), each below the "
                                 f"class's {self.steps}")
            by.setdefault(e.template, []).append(e)
        for tid, n in units.items():
            at = 0
            for e in sorted(by.get(tid, ()), key=lambda e: e.at):
                if e.at > at:
                    raise ValueError(f"{tid}: units [{at}, {e.at}) are in no inference entry; the entries tile each template's "
                                     "units exactly once")
                if e.at < at:
                    raise ValueError(f"{tid}: unit {e.at} is in two inference entries; the entries tile each template's units "
                                     "exactly once")
                at += e.units
            if at != n:
                raise ValueError(f"{tid}: inference entries tile units [0, {at}) of its {n}; they tile each template's units "
                                 "exactly once")
        return server_ends(self.network, units)

    @functools.cached_property
    def _by_template(self) -> dict[str, tuple[list[int], list[InferenceEntry]]]:
        by: dict[str, list[InferenceEntry]] = {}
        for e in sorted(self.inference, key=lambda e: e.at):
            by.setdefault(e.template, []).append(e)
        return {t: ([e.at for e in es], es) for t, es in by.items()}

    def slots_steps(self, template: str, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        """Template ``template``'s units ``[lo, hi)`` (which `check`'s tiling covers): each one's inference block, its slot
        and its step."""
        starts, es = self._by_template.get(template, ([], []))
        slot, step = np.empty(hi - lo, dtype=np.int64), np.empty(hi - lo, dtype=np.int64)
        k = max(bisect.bisect_right(starts, lo) - 1, 0)
        while k < len(es) and es[k].at < hi:
            e = es[k]
            a, b = max(lo, e.at), min(hi, e.at + e.units)
            i = np.arange(a - e.at, b - e.at, dtype=np.int64)
            slot[a - lo:b - lo], step[a - lo:b - lo] = e.digit("slot", i), e.step + e.digit("step", i)
            k += 1
        return slot, step

    def doc(self, units: int, partition: str) -> dict[str, Any]:
        """The record's ``blocks`` for a circuit of ``units`` units under the partition digest ``partition``."""
        return {"format": BLOCKS, "proof": {"units": int(units), "partition": partition, **({"plan": self.plan} if self.plan else {})},
                "inference": {"slots": self.slots, "steps": self.steps, "entries": [e.doc() for e in self.inference]},
                **(self.network or {})}

    @classmethod
    def of(cls, d: Any, units: Mapping[str, int], partition: str) -> tuple[Blocks, dict[str, tuple[int, ...]]]:
        """A record's ``blocks`` against its templates' unit counts ``units`` and its ``partition`` digest: its ``proof``
        their total and that digest, refused (ValueError) unless well-formed and `check`ed. -> (the blocks, the server ends)."""
        if not isinstance(d, Mapping) or d.get("format") != BLOCKS or not {"format", "proof", "inference"} <= set(d) \
                or not set(d) <= {"format", "proof", "inference", *NETWORK}:
            raise ValueError(f"a circuit's blocks are a {BLOCKS} object of its proof and inference, and network accounting's "
                             f"{list(NETWORK)}")
        proof, inf = d["proof"], d["inference"]
        if not isinstance(proof, Mapping) or not {"units", "partition"} <= set(proof) <= {"units", "partition", "plan"}:
            raise ValueError(f"a circuit's proof block is {{units, partition}} and optionally its plan, not {proof!r}")
        total = sum(units.values())
        if proof["units"] != total or isinstance(proof["units"], bool) or proof["partition"] != partition:
            raise ValueError(f"the proof block's units {proof['units']!r} and partition {proof['partition']!r} are not the "
                             f"record's, {total} and {partition}")
        if not isinstance(inf, Mapping) or set(inf) != {"slots", "steps", "entries"} or not isinstance(inf["entries"], list):
            raise ValueError(f"a circuit's inference block is {{slots, steps, entries}}, its entries a list, not {inf!r}")
        net = {k: d[k] for k in NETWORK if k in d}
        b = cls(inf["slots"], inf["steps"], tuple(InferenceEntry.of(e) for e in inf["entries"]), net or None, proof.get("plan"))
        return b, b.check(units)


def server_ends(network: Mapping[str, Any] | None, units: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    """Each template's unit indices where a server's units begin or end (the servers' ``units``, ``"all"`` or ``[template,
    at, n]`` ranges), refused (ValueError) where a range isn't among its template's units. That the servers partition the
    units is network accounting's check."""
    if not network:
        return {}
    servers, ends = network["server"], {}
    if not servers or not all(isinstance(s, Mapping) and "units" in s for s in servers):
        raise ValueError(f"a circuit's servers are objects with their units, at least one, not {servers!r}")
    for s in servers:
        if s["units"] == "all":
            continue
        if not isinstance(s["units"], list):
            raise ValueError(f"server {s.get('name')!r}: its units are \"all\" or [template, at, n] ranges")
        for r in s["units"]:
            if not (isinstance(r, list) and len(r) == 3 and r[0] in units and _ints(r[1], r[2]) and r[1] >= 0 and r[2] >= 1
                    and r[1] + r[2] <= units[r[0]]):
                raise ValueError(f"server {s.get('name')!r}: units {r!r} are not [template, at, n] among a template's units")
            ends.setdefault(r[0], set()).update((r[1], r[1] + r[2]))
    return {t: tuple(sorted(e)) for t, e in sorted(ends.items())}


@dataclass(frozen=True)
class Circuit:
    """An integration's circuit in proof blocks: ``templates``, each ``(template, units)`` in order, every template `catalog`'s
    and of its own descriptor id; ``external``, the ports (``<descriptor id>/<port>``) that are the system's external I/O;
    ``wires``, the units whose input rows are other units' output rows (`Wire`, `check_wires`); ``boundary``, its I/O
    layout (the network certifier's, ``verity/boundary/v1``), a JSON object carried as given; ``cuts``, by template, the
    ports read as several rows (`RowLayout.cut`); ``registered`` and ``ingress``, what fills its other inputs
    (`check_inputs`); ``blocks``, its proof, inference and server blocks (`Blocks`)."""
    name: str
    templates: tuple[tuple[Any, int], ...]
    external: tuple[str, ...] = ()
    wires: tuple[Wire, ...] = ()
    boundary: Mapping[str, Any] | None = None
    cuts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    registered: Registered | None = None
    ingress: tuple[Ingress, ...] = ()
    blocks: Blocks | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "templates", tuple((fn, int(n)) for fn, n in self.templates))
        object.__setattr__(self, "external", tuple(self.external))
        object.__setattr__(self, "wires", tuple(self.wires))
        object.__setattr__(self, "cuts", _cuts(self.cuts))
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
        check_wires(self.layouts, self.wires, self.cuts)
        check_inputs(self.layouts, self.cuts, self.wires, self.registered, self.ingress)
        if self.boundary is not None:
            if not isinstance(self.boundary, Mapping):
                raise ValueError(f"{self.name}: its boundary is a JSON object, not {self.boundary!r}")
            object.__setattr__(self, "boundary", json.loads(json.dumps(self.boundary)))
        if self.blocks is not None:
            if not isinstance(self.blocks, Blocks):
                raise ValueError(f"{self.name}: its blocks are `Blocks`")
            self.blocks.check(self.units)

    @property
    def units(self) -> dict[str, int]:
        """Each template's unit count, by descriptor id."""
        return {tid: n for tid, (_, n) in zip(self.ids, self.templates)}

    def server_ends(self) -> dict[str, tuple[int, ...]]:
        """Each template's unit indices where a server block begins or ends (`server_ends`)."""
        return server_ends(self.blocks.network if self.blocks else None, self.units)

    def blocks_record(self) -> dict[str, Any] | None:
        """The record's ``blocks`` (`Blocks.doc`), its proof block the circuit's units and partition digest, or None."""
        if self.blocks is None:
            return None
        program, query = self.program()
        return self.blocks.doc(sum(self.units.values()), PO.digest(PO.build(program, query)))

    @functools.cached_property
    def _ids(self) -> tuple[str, ...]:
        return tuple(descriptor_id(fn) for fn, _ in self.templates)

    @functools.cached_property
    def _by_id(self) -> dict[str, tuple[Any, RowLayout, int]]:
        """Each template, its layout and its units, by descriptor id, made once: a window looks one up a member."""
        return {tid: (fn, layout(fn), n) for (fn, n), tid in zip(self.templates, self._ids)}

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    @property
    def layouts(self) -> dict[str, tuple[RowLayout, int]]:
        """Each template's layout and units, by descriptor id."""
        return {tid: (lay, n) for tid, (_fn, lay, n) in self._by_id.items()}

    @property
    def reads(self) -> tuple[Read, ...]:
        return self.registered.reads if self.registered is not None else ()

    def declared(self) -> dict[str, Any]:
        """The record's fields for its cuts, registered reads and ingress, each only when the circuit has one, so a circuit
        without them records as before."""
        return {**({"cuts": self.cuts} if self.cuts else {}),
                **({"registered": self.registered.doc()} if self.registered is not None else {}),
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
        if (got := self._by_id.get(tid)) is None:
            raise KeyError(f"{tid}: no template of {self.name}")
        return got[0]


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


def members(circuit: Circuit, cap: int, built: tuple[Any, Mapping[str, Any]] | None = None) -> tuple[CS.Member, ...]:
    """The window's members: each template's units in the partition's order, cut where a wire's reader units begin or end and
    where a server block's units begin or end (`Circuit.server_ends`), in chunks ``<descriptor id>/c<j>`` of at most ``cap``
    (one statement each), on the template's bit rows under its cuts, each row wired by the circuit's wire that covers it,
    and each port a registered read fills a table of the rows it reads (`committed_reads`). ``built``: the circuit's
    population program and query (`Circuit.program`), when the caller has them."""
    program, query = built or circuit.program()
    return _members(circuit.templates, program, query, cap, circuit.wires, circuit.cuts, circuit.server_ends(),
                    committed_reads(circuit.registered), seg_ports(circuit.registered))


class _ReadIndex:
    """Registered reads by reader template, each template's units cut at its reads' ends into stretches, each with the reads
    that cover it (in the reads' order), so a member's reads (`_reads_of`) are a lookup and not a pass over every read: a
    window's reads grow with its positions and its members with its units, and the pass made the commit quadratic."""

    def __init__(self, reads: Sequence[Read]) -> None:
        by: dict[str, list[Read]] = {}
        for r in reads:
            by.setdefault(r.reader, []).append(r)
        self._by: dict[str, tuple[list[int], list[list[Read]], list[Read]]] = {}
        for tid, rs in by.items():
            ends = sorted({x for r in rs for x in (r.at, r.at + r.reader_units)})
            cover: list[list[Read]] = [[] for _ in ends[1:]]
            for r in rs:
                for j in range(bisect.bisect_left(ends, r.at), bisect.bisect_left(ends, r.at + r.reader_units)):
                    cover[j].append(r)
            self._by[tid] = (ends, cover, rs)

    def ends(self, tid: str) -> list[int]:
        """Where template ``tid``'s reads begin or end."""
        return self._by[tid][0] if tid in self._by else []

    def of(self, tid: str, lo: int, n: int) -> Sequence[Read]:
        """The reads that may meet template ``tid``'s units ``[lo, lo + n)``: those covering its stretch when it lies in one,
        else all of the template's, which `_reads_of` refuses where one covers part of it."""
        if tid not in self._by:
            return ()
        ends, cover, rs = self._by[tid]
        j = bisect.bisect_right(ends, lo) - 1
        if j == len(ends) - 1 or (j < 0 and lo + n <= ends[0]):
            return ()
        return cover[j] if j >= 0 and lo + n <= ends[j + 1] else rs


def _reads_of(tid: str, lo: int, n: int, reads: _ReadIndex, rows: Mapping[str, int]) -> dict[str, tuple[str, list[int], np.ndarray]]:
    """Template ``tid``'s units ``[lo, lo + n)`` (its own indices), one member's, by each port a read fills: the parameter
    it reads, the registered rows its table holds (each a row of the port's words, distinct, in first-read order: core's
    ``registered.table``) and each unit's ref into them; ``rows``: the template's input rows' words, by port. A read covers
    whole members and reads whole registered rows: its base and steps are multiples of the port's words."""
    out = {}
    for r in reads.of(tid, lo, n):
        if r.reader != tid or not (r.at < lo + n and lo < r.at + r.reader_units):
            continue
        if lo < r.at or lo + n > r.at + r.reader_units:
            raise ValueError(f"read {r.doc()} covers part of {tid}'s units [{lo}, {lo + n}); a read covers whole members")
        words = rows.get(r.port)
        if words is None or r.cut != (0, 1) or r.base % words or any(s % words for s in r.step):
            raise ValueError(f"read {r.doc()}: an uncut read of whole registered rows of an uncut port, its base and steps "
                             f"multiples of the port's words ({words})")
        order: dict[int, int] = {}
        refs = np.array([order.setdefault(int(p), len(order)) for p in r.source(np.arange(lo - r.at, lo - r.at + n)) // words],
                        dtype=np.int64)
        out[r.port] = (r.param, list(order), refs)
    return out


class _Drawn:
    """A window's registered reads as `Window.close` commits them, each (value, position) the window reads drawn once: by
    parameter, the positions its members read (sorted), their salts (`rows.value_salts`, one pass where the value's salts
    draw many) and their paths' hex. Each member's salts and header entry are core's ``Registered.statement``'s for its
    positions, which draws each position's salt again in pure Python and makes its row's ints for every member that reads
    it: the 4,32,4 class's members read 5,282,498 positions, 37,788 of them distinct."""

    def __init__(self, values: Mapping[str, Any], read: Mapping[str, Sequence[Sequence[int]]]) -> None:
        self._at: dict[str, tuple[Any, np.ndarray, np.ndarray, list[str]]] = {}
        for param, ps in read.items():
            v, at = values[param], np.unique(np.concatenate([np.asarray(p, dtype=np.int64) for p in ps]))
            ys = np.frombuffer(CS.value_salts(v, at), dtype=np.uint8).reshape(len(at), CS.SALT_BYTES)
            self._at[param] = (v, at, ys, [v.path(int(p)).hex() for p in at])

    def salts(self, param: str, positions: Sequence[int] | np.ndarray) -> bytes:
        """Rows ``positions`` of ``param``'s value: their salts concatenated."""
        _v, at, ys, _ = self._at[param]
        return ys[np.searchsorted(at, positions)].tobytes()

    def entry(self, param: str, positions: Sequence[int] | np.ndarray) -> dict[str, Any]:
        """Core's ``Registered.statement(positions)``'s header entry: ``{"value", "positions", "paths"}``."""
        v, at, _, paths = self._at[param]
        return {"value": v.value, "positions": [int(p) for p in positions],
                "paths": [paths[i] for i in np.searchsorted(at, positions)]}


def _members(templates: Sequence[tuple[Any, int]], program: Any, query: Mapping[str, Any], cap: int,
             wires: Sequence[Wire], cuts: Mapping[str, Mapping[str, int]] | None = None,
             servers: Mapping[str, Sequence[int]] | None = None, reads: Sequence[Read] = (),
             seg: Mapping[str, frozenset[str]] | None = None) -> tuple[CS.Member, ...]:
    """`members` of ``templates`` (``(Definition, units)``) under their population ``program`` and ``query``. A member's row
    is wired on all its units or on none (`rows.Member.wired`), so a template's units are cut at every wire's ends first, and
    a member's units are one server's, so they are cut at ``servers``' ends too (by template, `server_ends`). A registered
    read covers whole members (`_reads_of`), so they are cut at every read's ends, and a member it covers is a shared-row
    member: each read port's table the
    registered rows its units read, each other input row's its units' own rows. ``seg``: by template, its row-seg ports
    (`seg_ports`)."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"a statement holds at least one unit, not {cap!r}")
    cuts, index = cuts or {}, _ReadIndex(reads)
    lays = {descriptor_id(fn): (layout(fn), n) for fn, n in templates}
    check_wires(lays, wires, cuts)
    ranges = CS.template_partition(program, query).ranges
    base = {tid: lo for _k, tid, lo, _n in ranges}
    by: dict[str, list[Wire]] = {}
    for w in wires:
        by.setdefault(w.reader, []).append(w)
    out = []
    for _k, tid, lo, n in ranges:
        lay, kv = lays[tid][0], cuts.get(tid, {})
        rows = tuple((p, w) for p, w, _ in lay.cut(kv))
        schemas = tuple((p, SEG if p in (seg or {}).get(tid, ()) else V2) for p, _ in rows + lay.outs)
        mine = sorted(by.get(tid, ()), key=lambda w: w.at)
        ends = sorted({0, n, *(x for w in mine for x in (w.at, w.at + w.reader_units)), *(servers or {}).get(tid, ()),
                       *index.ends(tid)})
        live: list[Wire] = []
        nxt, j = 0, 0
        for a, b in zip(ends, ends[1:]):
            live = [w for w in live if w.at + w.reader_units > a]
            while nxt < len(mine) and mine[nxt].at <= a:
                live.append(mine[nxt])
                nxt += 1
            for at in range(a, b, cap):
                m = min(cap, b - at)
                wired = tuple((row_name(w.port, r, int(kv.get(w.port, 1))), w.out,
                               tuple((base[w.writer] + w.writers(at, m, r)).tolist()))
                              for w in live for r in range(w.cut[0], w.cut[0] + w.cut[1]))
                got = _reads_of(tid, at, m, index, dict(rows + lay.outs))
                tables = tuple((p, len(got[p][1]) if p in got else m) for p, _ in rows) if got else ()
                out.append(CS.Member(tid, rows, lay.outs, m, lo + at, tables, name=f"{tid}/c{j}", schemas=schemas,
                                     wired=wired))
                j += 1
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


def circuit_record(w: CS.Binding, name: str, templates: Sequence[tuple[str, int]],
                   statements: Mapping[str, Mapping[str, Any]] | None = None, external: Sequence[str] = (), *,
                   cap: int | None = None, wires: Sequence[Wire] = (),
                   boundary: Mapping[str, Any] | None = None, declared: Mapping[str, Any] | None = None,
                   blocks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``circuit.json``: everything the verifier's window is (`of_records`), none of its values. ``templates``: each
    ``(descriptor id, units)`` in the program's order; ``statements``: by template, how its unit's statement reads its rows
    (``in_rows``, ``out_rows``, ``reads``, as C-Flock's `UnitStager` takes them), for a template whose members are bit rows.
    The members are listed, or with ``cap`` stated: the verifier derives them from the templates, ``cap``, ``wires``, cuts
    and server blocks (`members`), so the record grows with its templates and wires, not its units. ``boundary``: the
    circuit's I/O layout (`Circuit.boundary`), carried as given; ``declared``: its cuts, registered reads and ingress
    (`Circuit.declared`); ``blocks``: its blocks (`Circuit.blocks_record`)."""
    if (declared or blocks is not None) and cap is None:
        raise ValueError("a record that lists its members has no cuts, registered reads, ingress or blocks; they are stated "
                         "with a cap")
    return {**_record(name, templates, statements, external, w.query, w.partition["program"], w.partition_digest, w.run,
                      w.context, w.work_table, w.row_tables),
            **(_stated(cap, wires, declared) if cap is not None else _listed(w, wires)),
            **({"boundary": json.loads(json.dumps(boundary))} if boundary is not None else {}),
            **({"blocks": json.loads(json.dumps(dict(blocks)))} if blocks is not None else {})}


def _record(name: str, templates: Sequence[tuple[str, int]], statements: Mapping[str, Mapping[str, Any]] | None,
            external: Sequence[str], query: Mapping[str, Any], program: Any, partition: str, run: str, context: str,
            work_table: Any, row_tables: Sequence[CS.RowTable]) -> dict[str, Any]:
    return {"format": CIRCUIT, "name": name, "templates": [[t, int(n)] for t, n in templates], "query": dict(query),
            "program": program, "partition": partition, "run": run, "context": context, "work_table": work_table,
            "external": list(external),
            "row_tables": [{"segment": t.segment, "name": t.name, "template": t.template, "words": t.words,
                            "keys": [list(k) for k in t.keys], "label": [[k, v] for k, v in t.label]} for t in row_tables],
            "statements": {t: {"in_rows": [[p, s, b, list(ps)] for p, s, b, ps in st["in_rows"]],
                               "out_rows": [[s, list(ix)] for s, ix in st["out_rows"]], "reads": st.get("reads", "inline")}
                           for t, st in sorted((statements or {}).items())}}


def _stated(cap: int, wires: Sequence[Wire], declared: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"cap": int(cap), "wires": [x.doc() for x in wires], **json.loads(json.dumps(dict(declared or {})))}


def _listed(w: CS.Binding, wires: Sequence[Wire]) -> dict[str, Any]:
    if wires:
        raise ValueError("a record that lists its members lists their wires with them; wires are stated with a cap")
    return {"members": [{"template": m.template, "name": m.name, "n": m.n, "base": m.base, "rows": [list(r) for r in m.rows],
                         "outs": [list(o) for o in m.outs], "schemas": [list(s) for s in m.schemas],
                         "tables": [list(t) for t in m.tables], "link_units": list(m.link_units),
                         "links": [list(x) for x in m.links], "wired": [[p, o, list(ws)] for p, o, ws in m.wired]}
                        for m in w.members]}


def descriptor_record(circuit: Circuit, cap: int) -> dict[str, Any]:
    """``circuit``'s ``circuit.json`` less its `RUN_FIELDS` (`circuit_of`), from its description alone: what a compiler emits
    before any unit runs. Its partition is the one `members` tiles, its members stated by ``cap``, its wires and its cuts, so
    building it is work per template and per wire, never per unit."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"a statement holds at least one unit, not {cap!r}")
    program, query = circuit.program()
    part = PO.build(program, query)
    sts, seg = {}, seg_ports(circuit.registered)
    for fn, tid in zip((f for f, _ in circuit.templates), circuit.ids):
        i, o = unit_rows(fn, circuit.cuts.get(tid), seg.get(tid, ()))
        sts[tid] = {"in_rows": i, "out_rows": o, "reads": "inline"}
    doc = {**_record(circuit.name, list(zip(circuit.ids, (n for _, n in circuit.templates))), sts, circuit.external, query,
                     part["program"], PO.digest(part), "", "", None, ()),
           **_stated(cap, circuit.wires, circuit.declared()),
           **({"boundary": circuit.boundary} if circuit.boundary is not None else {}),
           **({"blocks": circuit.blocks.doc(sum(circuit.units.values()), PO.digest(part))} if circuit.blocks else {})}
    return circuit_of(json.loads(circuit_bytes(doc)))


def templates_of(w: CS.Binding) -> list[tuple[str, int]]:
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


def held_params(pins: Mapping[str, str] | str | None, roots: Mapping[str, str] | None) -> frozenset[str]:
    """The registered parameters whose window root is the verifier's pin of that parameter, a pin naming its parameter
    alone: ``pins`` the roots the verifier fixed before the run and ``roots`` the window's, as its registration records
    them (`Window.write`), each ``{param: root}`` with a root hex digits, compared without case; a bare string ``pins`` is
    none. Refused (ValueError) in any other form, so a set of names never passes for pins."""
    if isinstance(pins, str):
        return frozenset()
    pins, roots = pins or {}, roots or {}
    for what, m in (("pins", pins), ("roots", roots)):
        if not isinstance(m, Mapping) or not all(isinstance(p, str) and isinstance(r, str) and re.fullmatch(r"[0-9a-fA-F]+", r)
                                                 for p, r in m.items()):
            raise ValueError(f"{what} are {{param: root}}, each root hex digits, not {m!r}: a registered read counts "
                             "registered only where the window's root for its parameter is the pinned root")
    return frozenset(p for p, r in pins.items() if r.lower() == str(roots.get(p, "")).lower())


def registered_roots(window: Mapping[str, Any]) -> dict[str, str]:
    """The roots a window's registration records for its registered parameters (its ``registered_roots``, `Window.write`:
    core's registration's window, or an integration's own window registration), by parameter; none where it records
    none. Refused (ValueError) unless each is a root of hex digits."""
    roots = window.get("registered_roots") or {}
    held_params({}, roots)
    return dict(roots)


def input_count(doc: Mapping[str, Any], resolve: Callable[[str], Any] = resolve, pins: Mapping[str, str] | str | None = None,
                roots: Mapping[str, str] | None = None) -> dict[str, int]:
    """A stated circuit record's inputs by what fills them, in bits, from its description alone (its templates, wires and
    declarations, `Circuit.description`; per template, wire and declaration, never per unit, and never its program): every
    unit's input rows (``inputs``), the rows its wires fill (``wired``, a wire's reader rows times its writer row's bits,
    `SampledProofs.Wires`' ``wiredBits``), its registered reads of input rows (a read of an output port fills none, `Read`),
    table reads and openings of a held parameter (``registered``) and its ingress (``ingress``); ``free`` the rest, which
    only the prover chose. ``pins`` are the verifier's own, never the record's: the roots it fixed before the run, by the
    names the record gives the registered parameters (``{param: root}``); ``roots`` the window's, as its registration
    records them (`Window.write`). A parameter is held where the window's root for it is the verifier's pin of it
    (`held_params`), and a pin names its parameter alone: a pin ``P`` holds no ``P/…`` (one of a registration of several
    parameters, a model's weights one a weight and row width), whose committed root nothing ties to ``P``'s. A read of
    any other parameter, or a table read or opening of a table over one, is free, since a root the prover commits after
    serving, or another root than the pinned one, constrains nothing; with no pins (a bare string is none) every
    registered read is free. Refused unless the declarations are well-formed for the record's templates (`check_wires`,
    `check_inputs`), so no row counts twice."""
    if "wires" not in doc:
        raise ValueError("a circuit record whose members are listed is counted from its members, not its description")
    defs = [(resolve(t), int(n)) for t, n in doc["templates"]]
    lays = {descriptor_id(fn): (layout(fn), n) for fn, n in defs}
    wires, cuts, reg, ingress = declarations(doc)
    check_wires(lays, wires, cuts)
    check_inputs(lays, cuts, wires, reg, ingress)

    def row_bits(tid: str, port: str) -> int:
        return 16 * dict(lays[tid][0].rows)[port] // int(cuts.get(tid, {}).get(port, 1))

    inputs = sum(16 * sum(w for _, w in lay.rows) * n for lay, n in lays.values())
    wired = sum(w.units * row_bits(w.reader, w.port) for w in wires)
    reg, held = reg or Registered({}), held_params(pins, roots)
    pinned = {t.name for t in reg.tables if t.param in held}
    registered = (sum(r.units * row_bits(r.reader, r.port) for r in reg.reads
                      if r.param in held and r.port in dict(lays[r.reader][0].rows))
                  + sum(r.units * 16 * dict(lays[r.reader][0].rows)[r.port] for r in reg.roots if r.table in pinned)
                  + sum(o.units * 16 * dict(lays[o.reader][0].rows)[p] for o in reg.openings if o.table in pinned
                        for p in o.ports))
    ingress_bits = sum(g.units * 16 * dict(lays[g.reader][0].rows)[g.port] for g in ingress)
    units = sum(n for _, n in lays.values())
    return {"units": units, "inputs": inputs, "wired": wired, "registered": registered, "ingress": ingress_bits,
            "free": inputs - wired - registered - ingress_bits}


def registered_reads(doc: Mapping[str, Any], w: CS.Binding, pins: Mapping[str, Mapping[str, Any]], *,
                     registered: Any) -> dict[str, dict[str, Any]]:
    """The Lean verifier's ``--registered`` file of each member reading a registered parameter (``registered-reads/v1``, in
    the format of ``registered``, core's ``one_stage.registered``, by member key), from the stated record ``doc`` and its
    window ``w`` (`window_of_record`'s) alone, but for each parameter's entry from ``pins`` (``{param: entry}``: the value's
    ``root``, ``leaves``, ``words``, ``word_bits`` and ``domain`` as the verifier pinned them before the run): each read
    port (`committed_reads`) reads its parameter's rows in the port's row format, and unit ``m.base + i`` the row its
    ``i``-th unit reads (`_reads_of`). A table's supplied values the verifier makes itself from its pin of the table's
    parameter (`supplied`, under ``w``'s partition digest). Refused (ValueError) where a read's parameter has no pin, or a
    pin's rows are not the port's."""
    _wires, _cuts_, reg, _ingress = declarations(doc)
    reads = committed_reads(reg)
    if not reads:
        return {}
    pins = {**pins, **{v: x.entry() for v, x in supplied(reg, pins, w.partition_digest, module=registered).items()}}
    base = {t: b for _k, t, b, _n in CS.template_partition(w.program, w.query).ranges}
    out: dict[str, dict[str, Any]] = {}
    index = _ReadIndex(reads)
    for m in w.members:
        got = _reads_of(m.template, m.base - base[m.template], m.n, index, dict(m.rows + m.outs))
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


def window_of_record(doc: Mapping[str, Any], law: Mapping[str, Any], resolve: Callable[[str], Any]) -> CS.Binding:
    """The verifier's window from ``circuit.json``'s document under its own ``law``, each template's Definition by
    ``resolve`` (descriptor id -> Definition, the verifier's own catalog), its members the listed ones or, where the record
    states them, the ones its templates, ``cap``, wires and server blocks make (`members`); refused unless the population
    program, query, program digest and partition digest it rebuilds are the record's, and its blocks (stated records only)
    are well-formed and its proof block is the record's (`Blocks.of`)."""
    if doc.get("format") != CIRCUIT:
        raise ValueError(f"a window's circuit record is {CIRCUIT}, not {doc.get('format')!r}")
    defs = [(resolve(t), int(n)) for t, n in doc["templates"]]
    if [descriptor_id(fn) for fn, _ in defs] != [t for t, _ in doc["templates"]]:
        raise ValueError(f"the catalog resolves {[t for t, _ in doc['templates']]} to {[descriptor_id(fn) for fn, _ in defs]}")
    several = len(defs) > 1 or doc["query"] != PO.template_instance_query(defs[0][0])
    program, query = CS.population_program(defs, several=several)
    stated = "cap" in doc
    if ("members" in doc) == stated or ("wires" in doc) != stated or (not stated and {*DECLARED, "blocks"} & set(doc)):
        raise ValueError("a circuit record lists its members, or states them by its cap, wires, cuts, registered reads and "
                         "blocks, one or the other")
    if stated:
        if not isinstance(doc["wires"], list):
            raise ValueError(f"a circuit record's wires are a list, not {doc['wires']!r}")
        wires, cuts, registered, ingress = declarations(doc)
        lays = {descriptor_id(fn): (layout(fn), n) for fn, n in defs}
        check_wires(lays, wires, cuts)
        check_inputs(lays, cuts, wires, registered, ingress)
        servers = Blocks.of(doc["blocks"], {t: n for t, (_, n) in lays.items()}, doc["partition"])[1] if "blocks" in doc else {}
        ms = _members(defs, program, query, doc["cap"], wires, cuts, servers,
                      committed_reads(registered), seg_ports(registered))
    else:
        ms = tuple(CS.Member(m["template"], tuple(map(tuple, m["rows"])), tuple(map(tuple, m["outs"])), m["n"], m["base"],
                             tuple(map(tuple, m["tables"])), m["name"], tuple(map(tuple, m["schemas"])), tuple(m["link_units"]),
                             tuple(map(tuple, m["links"])), tuple((p, o, tuple(ws)) for p, o, ws in m["wired"]))
                   for m in doc["members"])
    tables = tuple(CS.RowTable(t["segment"], t["name"], t["template"], t["words"], _frozen(t["keys"]),
                               tuple((k, _frozen(v)) for k, v in t["label"])) for t in doc["row_tables"])
    w = CS.Binding(program, query, dict(law), doc["run"], ms, tables, doc["context"], doc["work_table"])
    got = {"query": dict(query), "program": w.partition["program"], "partition": w.partition_digest}
    if bad := sorted(k for k, v in got.items() if v != doc[k]):
        raise ValueError(f"the circuit record's {bad} are not the ones its templates make: {got}")
    return w


def of_records(window_dir: str | Path, law: Mapping[str, Any], resolve: Callable[[str], Any]) -> tuple[CS.Binding, dict[str, Any]]:
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
    inputs: list[np.ndarray | None]   # None: by reference (`Window.record`)
    outputs: list[np.ndarray]
    words: list[np.ndarray | None]    # every port's (n, words) u16, parameters then return fields
    label: Any
    n: int
    at: int                           # its first unit among its template's


@dataclass
class Window:
    """One window of an integration's values: ``circuit``'s units under ``run`` (the record's run root, hex) and ``law`` (the
    proof service's law of the window, as its draw parses it), in members of at most ``cap`` units, with ``advice`` its
    compiler and advice (`advice_binding`) when the circuit depends on run-time choices, and ``registered`` the values its
    registered reads read (`committed_reads`), by parameter: each committed once, before the run, as core's
    ``one_stage.registered.Registered`` (rows of the reading port's words, its salts the prover's; a table's in its
    opening's row format), which the verifier pinned by its entry, and its tables' supplied values (`supply`); ``forms``,
    the forms of unit the integration's lowering has, which a label's ``form`` names (`record`). Each template's units are
    `record`ed in order, then `close` commits them and `write` writes and registers the window."""
    circuit: Circuit
    run: str
    law: Mapping[str, Any]
    cap: int
    advice: Mapping[str, Any] | None = None
    registered: Mapping[str, Any] | None = None
    work_table: Mapping[str, Any] | None = None
    forms: tuple[str, ...] = ()
    window: CS.Binding = field(init=False)
    _recorded: dict[str, list[_Recorded]] = field(init=False, default_factory=dict)
    _entries: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        _committable(self.circuit.registered, self.circuit.ingress, self.circuit.name)
        self.forms = tuple(self.forms)
        if not all(isinstance(f, str) for f in self.forms):
            raise ValueError(f"a window's forms are names, not {self.forms!r}")
        params = self.circuit.registered.params if self.circuit.registered is not None else {}
        lays = self.circuit.layouts
        for r in self.circuit.reads:
            v = (self.registered or {}).get(r.param)
            lay = lays[r.reader][0]
            words = dict(lay.rows + lay.outs)[r.port]
            if v is not None and (v.words != words or v.words * v.leaves != params[r.param]):
                raise ValueError(f"{r.param}: committed as {v.leaves} rows of {v.words} words, not the {params[r.param]} "
                                 f"leaves {r.port}'s rows of {words} words read")
        program, query = self.circuit.program()
        self.window = CS.Binding(program, query, dict(self.law), self.run, members(self.circuit, self.cap, (program, query)),
                                work_table=None if self.work_table is None else dict(self.work_table))
        if self.advice is not None:
            self.advice = advice_binding(self.advice["compiler"], self.advice["advice"]["root"], self.advice["advice"]["bits"])

    def supply(self, module: Any) -> None:
        """Add its tables' supplied values (`supplied`) to ``registered``, from its tables' parameters' committed values (each
        with core's ``entry()``) under the window's partition digest, ``module`` core's ``one_stage.registered`` as the
        caller passes it."""
        reg = dict(self.registered or {})
        entries = {p: v.entry() for p, v in reg.items() if p in {t.param for t in (self.circuit.registered or Registered({})).tables}}
        self.registered = {**reg, **supplied(self.circuit.registered, entries, self.window.partition_digest, module=module)}

    def count(self, tid: str) -> int:
        """Template ``tid``'s units recorded so far."""
        got = self._recorded.get(tid)
        return got[-1].at + got[-1].n if got else 0

    def record(self, template: Any, inputs: Sequence[Any], outputs: Sequence[Any], label: Any = None) -> range:
        """``template``'s next units: ``inputs`` one array per parameter, ``outputs`` one per return field (module docstring),
        all of one unit count, under the integration's ``label`` (JSON). -> their indices among the template's units.
        An input may be None, the same for all of a template's units: its rows are committed by reference (`close`), so a
        parameter every row of which is wired or a registered read is never held. In a circuit with blocks, a label is
        refused (ValueError) unless the run starts and ends where the template's units change inference block
        (`Blocks.slots_steps`) and the label names only `LABEL_KEYS`: the ``step`` and ``slot`` of all its units, ints,
        and its ``form``, one of the window's ``forms``."""
        tid = descriptor_id(template)
        fn = self.circuit.template(tid)
        lay, total = layout(fn), self.circuit._by_id[tid][2]
        if len(inputs) != len(lay.rows) or len(outputs) != len(lay.outs) or any(o is None for o in outputs):
            raise ValueError(f"{tid}: {len(inputs)} inputs and {len(outputs)} outputs (each an array), not {len(lay.rows)} and "
                             f"{len(lay.outs)}")
        n, at = len(outputs[0]), self.count(tid)
        if n < 1 or at + n > total:
            raise ValueError(f"{tid}: {n} units at {at}, past its {total}")
        json.dumps(label)
        if label is not None and self.circuit.blocks is not None:
            self._structural(tid, at, n, total, label)
        values = [None if a is None else np.ascontiguousarray(a) for a in [*inputs, *outputs]]
        ws = [None if a is None else words(a, w, n, c) for a, w, (_, c) in zip(values, lay.widths, lay.rows + lay.outs)]
        self._recorded.setdefault(tid, []).append(_Recorded(values[:len(inputs)], values[len(inputs):], ws, label, n, at))
        return range(at, at + n)

    def _structural(self, tid: str, at: int, n: int, total: int, label: Any) -> None:
        """Refuse (ValueError) ``label`` on ``tid``'s units ``[at, at + n)`` unless it is class structure (`record`)."""
        what = f"{tid} units [{at}, {at + n}) labelled {label!r}"
        if (not isinstance(label, Mapping) or not set(label) <= set(LABEL_KEYS)
                or not all(_ints(label[k]) for k in ("step", "slot") if k in label)
                or ("form" in label and not (isinstance(label["form"], str) and label["form"] in self.forms))):
            raise ValueError(f"{what}: a label is class structure, never a value: at most {list(LABEL_KEYS)}, the run's step "
                             f"and slot as ints and its form one of the window's forms {list(self.forms)}")
        lo, hi = max(at - 1, 0), min(at + n + 1, total)
        slot, step = self.circuit.blocks.slots_steps(tid, lo, hi)
        block = slot * self.circuit.blocks.steps + step
        first, last = at - lo, at - lo + n - 1
        if (at > 0 and block[first - 1] == block[first]) or (at + n < total and block[last + 1] == block[last]):
            raise ValueError(f"{what}: a labelled run starts and ends where its template's units change inference block, "
                             "(slot, step) (the circuit's inference entries); record each such stretch, or several whole, "
                             "as a run")
        for key, got in (("step", step), ("slot", slot)):
            seen = np.unique(got[first:last + 1]).tolist()
            if key in label and seen != [label[key]]:
                raise ValueError(f"{what}: its units are at {key} {seen}, not its label's {label[key]}")

    def mismatches(self, kernel: str = "reference") -> dict[str, list[int]]:
        """Per template, the units whose recorded outputs are not its Definition's outputs on their recorded inputs
        (`evaluate_batch` under ``kernel``): what the proof of a drawn one would refuse. The Definition reads each input row
        by its leaves' width (`_port_leaves`: a Boolean-profile port bit by bit), and each output row it gives is held to
        the committed row whole, its zero padding too."""
        out = {}
        for fn, _n in self.circuit.templates:
            tid, bad, at = descriptor_id(fn), [], 0
            (params, ret), lay = fn.signature(), layout(fn)
            for b in self._recorded.get(tid, ()):
                given = self.input_words(tid, at, b.n) if any(x is None for x in b.inputs) else b.words[:len(params)]
                want = np.asarray(evaluate_batch(fn, *(_port_leaves(r, ty) for r, (_, ty) in zip(given, params)),
                                                 kernel=kernel)).reshape(b.n, -1)
                wrong, k = np.zeros(b.n, dtype=bool), 0
                for ty, (_, count), got in zip(_fields(ret), lay.outs, b.words[len(params):]):
                    wrong |= (_port_rows(want[:, k:k + ty.leaves], ty, count) != got).any(axis=1)
                    k += ty.leaves
                bad += [at + int(i) for i in np.flatnonzero(wrong)]
                at += b.n
            if bad:
                out[tid] = bad
        return out

    def close(self, *, workers: int = 1, salt_key: bytes | None = None) -> CS.Committed:
        """Commit every unit's rows (`rows.commit`), refused unless every template's units are all recorded. ``salt_key``:
        tests only (`rows.commit`). A parameter recorded by reference (`record`) is committed as what it reads: a registered
        read's rows as its value's (`rows.ValueRows`), a wired row as its writer's output row (`rows.WIRED`); refused for a
        row that is neither. Each registered row the window reads has its salt drawn and its path made once (`_Drawn`)."""
        missing = {tid: (self.count(tid), n) for tid, (_, n) in zip(self.circuit.ids, self.circuit.templates)
                   if self.count(tid) != n}
        if missing:
            raise ValueError(f"units recorded / in the circuit by template: {missing}; a window closes over all of them")
        committed = _ReadIndex(committed_reads(self.circuit.registered))
        base, cat, by, values = self._bases(), {}, self.circuit._by_id, self.registered or {}
        read_by, wanted = [], {}
        for m in self.window.members:
            words = dict(m.rows + m.outs)
            read_by.append(reads := _reads_of(m.template, m.base - base[m.template], m.n, committed, words))
            for p, (param, positions, _refs) in reads.items():
                if (v := values.get(param)) is None:
                    raise ValueError(f"{m.key} {p} reads registered parameter {param!r}: give its committed value"
                                     + (" (`Window.supply`)" if param.rsplit("/", 1)[-1] in SUPPLIED else ""))
                if v.words != words[p]:
                    raise ValueError(f"{m.key} {p} reads {param!r}, committed as rows of {v.words} words, not the port's "
                                     f"{words[p]}")
                wanted.setdefault(param, []).append(positions)
        drawn = _Drawn(values, wanted)
        data, salts_of, self._entries = [], {}, {}
        for m, reads in zip(self.window.members, read_by, strict=True):
            lay = by[m.template][1]
            if m.template not in cat:
                cat[m.template] = self._words(m.template)
            ports, lo, kv = cat[m.template], m.base - base[m.template], self.circuit.cuts.get(m.template, {})
            d = {"rows": {}, "outs": {p: ports[len(lay.rows) + j][lo:lo + m.n] for j, (p, _) in enumerate(lay.outs)}}
            for g, (p, w) in enumerate(lay.rows):
                k = int(kv.get(p, 1))
                for j in range(k):
                    # row j of a port cut k ways is its outer elements [j·n/k, (j+1)·n/k): its words' j-th k-th (`RowLayout.cut`)
                    d["rows"][row_name(p, j, k)] = (None if ports[g] is None else ports[g][lo:lo + m.n] if k == 1 else
                                                    np.ascontiguousarray(ports[g][lo:lo + m.n, j * w // k:(j + 1) * w // k]))
            if reads:
                d["refs"] = np.stack([reads[p][2] if p in reads else np.arange(m.n) for p, _ in m.rows], axis=1)
                for p, (param, positions, refs) in reads.items():
                    if p in d["outs"]:
                        # an output row stays the unit's own, committed under the value's salt, so a row other than
                        # the value's opens no pinned root
                        at = np.asarray(positions, dtype=np.int64)[refs]
                        salts_of[m.port(p)] = drawn.salts(param, at)
                        self._entries.setdefault(m.key, {})[p] = drawn.entry(param, at)
                        continue
                    if d["rows"][p] is None:
                        d["rows"][p] = CS.ValueRows(values[param], tuple(int(x) for x in positions), drawn.salts(param, positions))
                        self._entries.setdefault(m.key, {})[p] = drawn.entry(param, positions)
                        continue
                    first = [int(np.flatnonzero(refs == j)[0]) for j in range(len(positions))]
                    rows = d["rows"][p][first]
                    if not np.array_equal(rows[refs], d["rows"][p]):
                        raise ValueError(f"{m.key} {p}: two units recorded two rows for one row of {param!r}")
                    d["rows"][p] = rows
                    salts_of[m.port(p)] = drawn.salts(param, positions)
                    self._entries.setdefault(m.key, {})[p] = drawn.entry(param, positions)
            wired = {q for q, _o, _ws in m.wired}
            for r, x in d["rows"].items():
                if x is None:
                    if r not in wired:
                        raise ValueError(f"{m.key} {r}: recorded by reference, but no wire or registered read fills it")
                    d["rows"][r] = CS.WIRED
            data.append(d)
        return CS.commit(self.window, data, workers=workers, salt_key=salt_key, salts_of=salts_of)

    def _words(self, tid: str) -> list[np.ndarray | None]:
        """Template ``tid``'s recorded words by port over all its units, None for a parameter recorded by reference."""
        got, out = self._recorded[tid], []
        for g in range(len(got[0].words)):
            ws = [b.words[g] for b in got]
            if any(w is None for w in ws) and not all(w is None for w in ws):
                raise ValueError(f"{tid}: parameter {g} recorded by reference for some units and in full for others")
            out.append(None if ws[0] is None else np.concatenate(ws))
        return out

    def input_words(self, tid: str, lo: int, n: int,
                    outputs: Callable[[str], Sequence[np.ndarray]] | None = None) -> list[np.ndarray]:
        """Template ``tid``'s units ``[lo, lo + n)`` (its own indices): each parameter's ``(n, words)`` u16 words as the window
        commits them (`close`): recorded in full, or by reference a registered read's value rows and a wired row its writer's
        output row, each writer's output words over all its template's units from ``outputs`` (by template), else as recorded."""
        lay, base = layout(self.circuit.template(tid)), self._bases()
        mine, kv = self._words(tid) if self._recorded.get(tid) else [None] * len(lay.rows), self.circuit.cuts.get(tid, {})
        committed, cache = _ReadIndex(committed_reads(self.circuit.registered)), {}
        starts = sorted((b, t) for t, b in base.items())
        firsts = np.array([b for b, _ in starts], dtype=np.int64)

        def out(t: str, port: str) -> np.ndarray:
            if (t, port) not in cache:
                lt = layout(self.circuit.template(t))
                j = [p for p, _ in lt.outs].index(port)
                cache[(t, port)] = (outputs(t) if outputs is not None else self._words(t)[len(lt.rows):])[j]
            return cache[(t, port)]

        res = []
        for g, (p, w) in enumerate(lay.rows):
            if mine[g] is not None:
                res.append(mine[g][lo:lo + n])
                continue
            k, a = int(kv.get(p, 1)), np.empty((n, w), dtype=np.uint16)
            for m in self.window.members:
                mlo = m.base - base[tid]
                if m.template != tid or mlo >= lo + n or mlo + m.n <= lo:
                    continue
                s, e = max(lo, mlo), min(lo + n, mlo + m.n)
                reads = _reads_of(tid, mlo, m.n, committed, dict(m.rows + m.outs))
                if p in reads:
                    param, positions, refs = reads[p]
                    v = self.registered[param]
                    rows = np.asarray(v.rows[np.asarray(positions)[refs[s - mlo:e - mlo]]] if isinstance(v.rows, np.ndarray)
                                      else [v.rows[positions[int(i)]] for i in refs[s - mlo:e - mlo]], dtype=np.uint16)
                    a[s - lo:e - lo] = rows.reshape(e - s, w)
                    continue
                wires = {q: (o, ws) for q, o, ws in m.wired}
                for j in range(k):
                    r = row_name(p, j, k)
                    if r not in wires:
                        raise ValueError(f"{m.key} {r}: recorded by reference, but no wire or registered read fills it")
                    o, ws = wires[r]
                    x = np.asarray(ws[s - mlo:e - mlo], dtype=np.int64)
                    at = np.searchsorted(firsts, x, side="right") - 1
                    for i in np.unique(at):
                        b, t = starts[int(i)]
                        a[np.flatnonzero(at == i) + (s - lo), j * w // k:(j + 1) * w // k] = out(t, o)[x[at == i] - b]
            res.append(a)
        return res

    def _bases(self) -> dict[str, int]:
        return {t: b for _k, t, b, _n in CS.template_partition(self.window.program, self.window.query).ranges}

    def labels(self) -> dict[str, Any]:
        """``index.json``: per member, each recorded run of its units with its label, ``[lo, hi, label]`` over global indices,
        an unlabelled stretch of runs one run (module docstring)."""
        spans: dict[str, list[tuple[int, int, Any]]] = {}
        for tid, lo in self._bases().items():
            for b in self._recorded.get(tid, ()):
                runs = spans.setdefault(tid, [])
                if b.label is None and runs and runs[-1][2] is None:
                    runs[-1] = (runs[-1][0], lo + b.n, None)
                else:
                    runs.append((lo, lo + b.n, b.label))
                lo += b.n
        ends = {t: [b for _a, b, _x in runs] for t, runs in spans.items()}

        def of(m: CS.Member) -> list[list[Any]]:
            runs, out = spans.get(m.template, []), []
            j = bisect.bisect_right(ends.get(m.template, []), m.base)
            while j < len(runs) and runs[j][0] < m.base + m.n:
                a, b, x = runs[j]
                out.append([max(a, m.base), min(b, m.base + m.n), x])
                j += 1
            return out

        return {"format": LABELS, "run": self.run, "members": [{"name": m.key, "labels": of(m)} for m in self.window.members]}

    def circuit_record(self) -> dict[str, Any]:
        """The window's ``circuit.json`` (`circuit_record`), its members stated by its cap and the circuit's wires, each
        template's statement its bit rows under its cuts and row-seg ports (`unit_rows`)."""
        sts, seg = {}, seg_ports(self.circuit.registered)
        for fn, tid in zip((f for f, _ in self.circuit.templates), self.circuit.ids):
            i, o = unit_rows(fn, self.circuit.cuts.get(tid), seg.get(tid, ()))
            sts[tid] = {"in_rows": i, "out_rows": o, "reads": "inline"}
        blocks = self.circuit.blocks
        return circuit_record(self.window, self.circuit.name, list(zip(self.circuit.ids, (n for _, n in self.circuit.templates))),
                              sts, self.circuit.external, cap=self.cap, wires=self.circuit.wires, boundary=self.circuit.boundary,
                              declared=self.circuit.declared(),
                              blocks=blocks.doc(sum(self.circuit.units.values()), self.window.partition_digest) if blocks else None)

    def write(self, c: CS.Committed, out_dir: str | Path, provenance: Mapping[str, Any], *, registrar: Any,
              kind: str = "served", index_salt: bytes | None = None) -> dict[str, Any]:
        """The window's files (`rows.write`, its prover-only ``index.json`` the labels, bound by their hiding leaf under
        ``index_salt``) and ``circuit.json``, the registration core's record made by ``registrar`` (core's
        ``one_stage.registration``), its window naming ``provenance``, the circuit record's SHA-512, the advice binding and
        each registered value's root by parameter (``registered_roots``, which `input_count` holds to the verifier's pins);
        each member reading a registered value, its header naming each table row's position and path (`close`), each path
        and each row it reads by reference written once for the window (`rows.write`)."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        body = circuit_bytes(self.circuit_record())
        (d / "circuit.json").write_bytes(body)
        roots = {p: v.root.hex() for p, v in sorted((self.registered or {}).items())}
        bound = {**provenance, "circuit": hashlib.sha512(body).hexdigest(), **(self.advice or {}),
                 **({"registered_roots": roots} if roots else {})}
        return CS.write(c, d, bound, self.labels(), kind=kind, registered=self._entries, registrar=registrar,
                        index_salt=index_salt)


class Integration(Protocol):
    """What an integration provides: its circuit, and a run that records every unit of it into a window (`Window.record`)
    before the window closes."""

    def circuit(self) -> Circuit: ...

    def run(self, window: Window) -> None: ...
