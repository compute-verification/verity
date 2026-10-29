"""`verity/partition/v2`: a private partition, registered as advice under a hidden root, and the public predicates it satisfies.

In v1 (`partition_object`) the partition is a query the verifier evaluates on a circuit it holds.  In v2 the circuit and its
partition are the prover's: the partition is *advice*, fixed-width rows committed before any coin (hm96-sha512 hiding rows in one
frame-v3-sha512 tree, `verity.ml.commitments.advice`), and the verifier holds only a statement naming the predicates the
advice must satisfy.  A proof shows them (the global predicates once per registration, the local ones with each drawn unit);
this module is the reference both statements are checked against.

**Statement** (public; `statement`, `validate`, `digest` = `SHA-512("verity/partition/v2\\0" || canon(statement))`)::

    {"format": "verity/partition/v2", "family": "<the public template's name>",
     "predicates": {"width": int | null, "recompute": "refuse" | "report", "max_gates": int | null, "max_in_bits": int | null,
                    "canonical": <a Q_word query> | null, "grid_bits": g},
     "units": n_b, "advice": {"rows": R_b, "root": "<frame-v3-sha512 root, hex>"}}

`n_b` and `R_b` are the unit count and the row count rounded up to the public grid (`grid`: at most g significant bits), so the
verifier learns a grid level of each, never the counts.  Indices n..n_b-1 are filler units: empty, trivially correct.

**Advice** (private; `build`, `rows`, `decode`).  The partition per Definition, not per gate:

* the *Call table*, one row per root Call k of the program (`partition_object.calls`): its root-body node, its cut-table entry, its
  members (1 for a call or primitive node, n for a batch) and `start_k`, the index of its first unit;
* the *cut table*, one entry per distinct Definition, in first-reach order (Calls in order, a separable entry's children depth
  first): `flat`, an owner per computing gate of its `cut.CallGraph` and the SHA-512 shape of each unit's circuit; or `parts`,
  for a separable Definition (`cut.separable`), per body node the child's entry (none for a child with no computing gate) and its
  members.  Units of a parts entry are its members' in node, then member, order, as `cut.DefinitionCut` numbers them.

Unit i of the program is (Call k, member, unit within the entry) with i = start_k + member * U + unit: in query mode the same
numbering as v1 (`Partition.locate`).  Rows are `ROW_BYTES` bytes, integers little-endian: row 0 the header (`MAGIC`, the
program's SHA-512, n, m, e, the rows used), rows 1..m the Call table (node, entry, members, start: 8 bytes each), then one
directory row per entry (the SHA-512 of its Definition's id, kind, computing gates, units, first payload row, payload rows, items:
a flat entry's shape classes or a parts entry's body nodes), then the payloads: a flat entry's owners (4 bytes each, 32 to a
row), its units' shape classes (the same; classes numbered in first-use order; neither for a one-unit entry, whose owners
and class are all 0) and the classes' SHA-512 shapes (2 to a row); a
parts entry's (child, members) pairs (8 bytes each, 8 to a row; `NONE` for no child).  Every unused byte is 0, and the rows past
the header's count up to R_b are all 0.

**Global predicates** (`check`; refusal codes in brackets), all of them over the program and the advice, once per registration:

* G1 Calls: the header names the program; the Call table lists its root Calls in order with their Definitions and member counts,
  each start the sum of the units before it, n their total [header-program, call-count, call-node, call-form, call-definition,
  call-members, call-start, units-total];
* G2 Partition: one entry per distinct Definition, each reached, in first-reach order; a flat entry's owner is total on its
  computing gates and uses every unit 0..U-1; a parts entry is of a separable Definition, node for node, with the members and
  children the body has [entry-duplicate, entry-unreached, entry-order, entry-definition, entry-kind, entry-gates, entry-units,
  owner-length, owner-dense, parts-shape, parts-child];
* G3 Boundary and G5 Recompute: each flat entry's cut on its graph (`cut.check_cut`, without the width rule): every computing
  gate in one unit, reads across units only through committed values, returned values committed; and the predicates' recompute
  rule [cut];
* G4 Width: every unit of every flat entry has at most `width` bits out (`WIDTH`, 16, by default), summed over its committed
  outputs (`cut.boundary_widths`).  The limit is literal: unlike `Q_word{X, W}`'s width rule there is no exception for one value
  of up to W bits, so a unit whose output is one 32-bit value is refused [unit-too-wide];
* G6 Count: n_b = grid(n); and, with a canonical query, no entry has more units than the query's cut of its Definition
  (`cut.evaluate_definition` at the query's X; its W bounds no count, and G4 is `width`'s alone) [units-bucket, units-canonical];
* G7 Shapes: each flat entry's classes and their shapes are its units' circuits' (`units.Activation.circuits`), within
  max_gates gates and max_in_bits input bits [shape, unit-too-large].

A width, a covering or a count that a consumer's weighting relies on must hold for every unit, faulty or not; only these global
predicates give that.  Instances of a Definition are laid out identically, so a check per entry is a check of every activation.

**Local predicates** (`check_unit`, with drawn unit i; `openings` lists the rows it reads):

* L1 Locate: the header, the Call rows a bisection reads, the directory rows and parts rows down the path, then the flat entry's
  owner rows and the unit's shape row [unit-outside, unit-locate]; a filler (n <= i < n_b) is empty.  The rows read depend on
  the Definition holding the unit, never on n;
* L2 Statement: the circuit of the unit, cut from the program at its place, is the entry's Definition and has the shape the
  entry records [unit-definition, unit-shape];
* L3 Bind: its inputs and outputs as root gates of the program, the positions its committed values are opened at (`ProofBlock`);
* L4 Correct: given the committed values (a trace of root gate -> value), its gates applied to its inputs give its
  outputs [unit-incorrect].
"""
from __future__ import annotations

import hashlib
from array import array
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import partition_object as PO
from .codec import canonical_json, program_digest
from .cut import (RECOMPUTE, WIDE_FAN_IN, WIRING, CallGraph, QueryInapplicable, boundary_widths, call_scope, check_cut,
                  evaluate_definition, separable, wide_operands)
from .partition import RECOMPUTE_RULES, CutVerdict
from .units import Activation, ProofBlockCircuit

FORMAT = "verity/partition/v2"
ROW_BYTES = 128
ROW_BITS = 8 * ROW_BYTES
MAGIC = b"verity/partition/v2\0"
#: the 32-byte binding of the advice tree's domain (`verity.ml.commitments.advice`)
BINDING = hashlib.sha512(b"verity/partition/v2/advice").digest()[:32]
NONE = (1 << 64) - 1
FLAT, PARTS = 1, 2
OWNERS_PER_ROW, SHAPES_PER_ROW, PAIRS_PER_ROW = ROW_BYTES // 4, ROW_BYTES // 64, ROW_BYTES // 16
GRID_BITS = 4
WIDTH = 16
#: the widest public size class of a proof block's bits out (the claims' `Claims.Sampling.unitClassMax`), provisional: at 16,
#: `UNIT_CLASSES` is the literal 16-bit limit alone
UNIT_CLASS_MAX = 262_144
#: the public size classes, the powers of two from 16 to `UNIT_CLASS_MAX`, ascending (the claims' `unitClasses`)
UNIT_CLASSES = tuple(16 << i for i in range((UNIT_CLASS_MAX // 16).bit_length()))
PARAM_MAX = PO.PARAM_MAX

_KEYS = {"format", "family", "predicates", "units", "advice"}
_PRED_KEYS = {"width", "recompute", "max_gates", "max_in_bits", "canonical", "grid_bits"}
_HEX512 = re.compile(r"^[0-9a-f]{128}$")
_MAX_LISTED = 64


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def grid(x: int, bits: int = GRID_BITS) -> int:
    """The least y >= x with at most `bits` significant bits: x itself below 2^bits, else x rounded up to a multiple of
    2^(bit_length(x) - bits).  y < x * (1 + 2^(1 - bits)), so fewer than one in 2^(bits - 1) of y's units are fillers."""
    if not (_is_int(x) and x >= 0 and _is_int(bits) and bits >= 1):
        raise ValueError(f"grid of a count >= 0 at bits >= 1, not {x!r}, {bits!r}")
    e = max(0, x.bit_length() - bits)
    return -(-x >> e) << e


def definition_digest(fn: Any) -> str:
    """The SHA-512 the advice names a Definition by: of its specialization id, `fn.id`."""
    return hashlib.sha512(b"verity/partition/v2/definition\0" + fn.id.encode()).hexdigest()


def unit_class(bits: int) -> int | None:
    """A unit's size class: the least of `UNIT_CLASSES` that holds its ``bits`` out (`cut.boundary_widths`), None past the widest."""
    return next((w for w in UNIT_CLASSES if bits <= w), None)


class MixedClasses(ValueError):
    """`build`'s refusal of a partition whose units span more than one size class: the statement's one `width` bounds every
    unit, so units of several classes need a width per class, which v2's predicates don't state.  ``spans`` maps each class
    (None past the widest) to the first unit in it, as (Definition id, unit)."""

    def __init__(self, spans: Mapping[int | None, tuple[str, int]]) -> None:
        self.spans = dict(spans)
        named = "; ".join(f"{'none' if w is None else w} bits: {d} unit {u}"
                          for w, (d, u) in sorted(self.spans.items(), key=lambda s: (s[0] is None, s[0] or 0)))
        super().__init__(f"a v2 private partition's units span {len(self.spans)} size classes ({named}): it holds one class until "
                         f"its predicates state a width per class, so cut it so that every unit's bits out fall in one of "
                         f"{UNIT_CLASSES[0]}..{UNIT_CLASSES[-1]}")


# --- the advice ------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    node: int
    entry: int
    members: int
    start: int


@dataclass(frozen=True)
class Entry:
    definition: str                                   # `definition_digest`
    kind: int                                         # FLAT or PARTS
    gates: int                                        # computing gates of one activation
    units: int
    owner: tuple[int, ...] = ()                       # FLAT: a unit per computing gate (`cut.CallGraph` order)
    shape: tuple[int, ...] = ()                       # FLAT: a class per unit, classes numbered in first-use order
    classes: tuple[str, ...] = ()                     # FLAT: `units.ProofBlockCircuit.shape` per class
    parts: tuple[tuple[int | None, int], ...] = ()    # PARTS: (child entry or None, members) per body node


@dataclass(frozen=True)
class Advice:
    program: str                                      # `codec.program_digest`
    calls: tuple[Call, ...]
    entries: tuple[Entry, ...]

    @property
    def units(self) -> int:
        if not self.calls:
            return 0
        c = self.calls[-1]
        return c.start + c.members * self.entries[c.entry].units


def _classes(digests: Iterable[str]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Per-unit shape digests as (a class per unit, the classes' digests in first-use order)."""
    index: dict[str, int] = {}
    shape = tuple(index.setdefault(d, len(index)) for d in digests)
    return shape, tuple(index)


def _members(node: Any) -> int:
    return node.n if node.form == "batch" else 1


def build(program: Any, q: Mapping[str, Any] | None = None, owners: Callable[[Any, CallGraph], Sequence[int]] | None = None) -> Advice:
    """The advice of `program`'s partition: the cuts of the query `q` (`Q_word` v1 or v2, `Q_call` excluded, or a template query:
    each instance one unit), or of `owners(fn, graph)` (a unit per computing gate of each root Call's Definition, cut flat, one cut
    per Definition).  Default: `Q_word{16, 32}` v1.  It refuses a partition whose units span more than one size class
    (`unit_class`, `MixedClasses`)."""
    advice, spans = _advice(program, q, owners)
    if len(spans) > 1:
        raise MixedClasses(spans)
    return advice


def _build_unchecked(program: Any, q: Mapping[str, Any] | None = None,
                     owners: Callable[[Any, CallGraph], Sequence[int]] | None = None) -> Advice:
    """`build` without its refusal of units of several size classes: advice past the builder's scope, as a prover may send it to
    `check`.  Tests alone call it (`TEST_ONLY` in the repository's `tests/test_boundaries.py`)."""
    return _advice(program, q, owners)[0]


def _advice(program: Any, q: Mapping[str, Any] | None,
            owners: Callable[[Any, CallGraph], Sequence[int]] | None) -> tuple[Advice, dict[int | None, tuple[str, int]]]:
    """`build`'s advice, and each size class of its units with the first unit in it (`MixedClasses.spans`)."""
    if q is not None and owners is not None:
        raise ValueError("a query or an owner function, not both")
    bad = PO.order_violation(program)
    if bad is not None:
        raise QueryInapplicable(f"{bad['definition']} node {bad['node']} names {bad['reference']}: not a parameter or an earlier node", bad)
    q = dict(q or PO.query()) if owners is None else None
    kind = (q["name"], q["version"]) if q else None
    template = kind in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES)
    if q is not None and not template and kind not in (("Q_word", 1), ("Q_word", 2)):
        raise ValueError(f"v2 builds Q_word or template-query advice, not {kind!r}")
    if template:
        PO.TemplatePartition(program, q)                 # its applicability: every root Call an instance of a listed template
    X = int(q["params"]["X"]) if q is not None and not template else 0
    memo: dict = {}
    graphs: dict[str, CallGraph] = {}
    acts: dict[str, Activation] = {}
    index: dict[str, int] = {}
    entries: list[Entry | None] = []
    spans: dict[int | None, tuple[str, int]] = {}

    def graph(fn: Any) -> CallGraph:
        g = graphs.get(fn.id)
        if g is None:
            g = graphs[fn.id] = CallGraph(fn)
        return g

    def flat(fn: Any, owner: Sequence[int], units: int | None = None) -> Entry:
        owner = tuple(int(u) for u in owner)
        a = acts.get(fn.id)
        if a is None:
            a = acts[fn.id] = Activation(fn)
        shape, classes = _classes(uc.shape for uc in a.circuits(owner))
        n = len(shape) if units is None else units
        for u, bits in enumerate(boundary_widths(graph(fn), owner, n)[0]):
            spans.setdefault(unit_class(bits), (fn.id, u))
        return Entry(definition_digest(fn), FLAT, graph(fn).n, n, owner, shape, classes)

    def add(fn: Any) -> int:
        if fn.id in index:
            return index[fn.id]
        i = index[fn.id] = len(entries)
        entries.append(None)
        if owners is not None:
            entries[i] = flat(fn, owners(fn, graph(fn)))
        elif template:
            entries[i] = flat(fn, [0] * graph(fn).n)
        else:
            cut = evaluate_definition(fn, X, memo)
            if cut.parts is None:
                entries[i] = flat(fn, cut.flat, cut.units)
            else:
                parts = tuple((add(c.fn) if c is not None else None, m) for c, m in cut.parts)
                entries[i] = Entry(definition_digest(fn), PARTS, cut.gates, cut.units, parts=parts)
        return i

    calls: list[Call] = []
    start = 0
    nodes = program.root.body.nodes
    for k, fn in PO.calls(program):
        form = nodes[k].form
        if form not in ("call", "primitive", "batch") or (q is not None and not template and form == "batch"):
            raise QueryInapplicable(f"root node {k} has form {form!r}", {"call": k, "form": form, "definition": fn.id})
        e = add(fn)
        m = _members(nodes[k])
        calls.append(Call(k, e, m, start))
        start += m * entries[e].units
    return Advice(program_digest(program), tuple(calls), tuple(entries)), spans


# --- rows ------------------------------------------------------------------------------------------------------------------------


def _u(x: int, n: int = 8) -> bytes:
    return int(x).to_bytes(n, "little")


def _row(*parts: bytes) -> bytes:
    b = b"".join(parts)
    if len(b) > ROW_BYTES:
        raise AssertionError("a row overflows")
    return b + bytes(ROW_BYTES - len(b))


def _flat_rows(gates: int, units: int, classes: int) -> tuple[int, int, int]:
    """A flat entry's owner, class and shape rows; a one-unit entry stores no owners and no classes (all 0)."""
    implied = units == 1
    return (0 if implied else _rows_of(gates, OWNERS_PER_ROW), 0 if implied else _rows_of(units, OWNERS_PER_ROW),
            _rows_of(classes, SHAPES_PER_ROW))


def _payload_rows(e: Entry) -> int:
    if e.kind == FLAT:
        return sum(_flat_rows(len(e.owner), e.units, len(e.classes)))
    return _rows_of(len(e.parts), PAIRS_PER_ROW)


def rows(advice: Advice) -> list[bytes]:
    """The advice as `ROW_BYTES`-byte rows, unpadded (`pad` rounds the count up to the grid)."""
    m, e = len(advice.calls), len(advice.entries)
    first = 1 + m + e
    starts, at = [], first
    for en in advice.entries:
        starts.append(at)
        at += _payload_rows(en)
    out = [_row(MAGIC, bytes.fromhex(advice.program), _u(advice.units), _u(m), _u(e), _u(at))]
    out += [_row(_u(c.node), _u(c.entry), _u(c.members), _u(c.start)) for c in advice.calls]
    out += [_row(bytes.fromhex(en.definition), _u(en.kind), _u(en.gates), _u(en.units), _u(s), _u(_payload_rows(en)),
                 _u(len(en.classes) if en.kind == FLAT else len(en.parts))) for en, s in zip(advice.entries, starts)]
    for en in advice.entries:
        if en.kind == FLAT and en.units == 1:
            if any(en.owner) or en.shape != (0,):
                raise ValueError(f"a one-unit entry's owners and class are 0, not {sorted(set(en.owner))}, {en.shape}")
        if en.kind == FLAT:
            for o in ((en.owner, en.shape) if en.units != 1 else ()):
                out += [_row(*(_u(u, 4) for u in o[j:j + OWNERS_PER_ROW])) for j in range(0, len(o), OWNERS_PER_ROW)]
            s = en.classes
            out += [_row(*(bytes.fromhex(x) for x in s[j:j + SHAPES_PER_ROW])) for j in range(0, len(s), SHAPES_PER_ROW)]
        else:
            p = en.parts
            out += [_row(*(_u(NONE if c is None else c) + _u(mm) for c, mm in p[j:j + PAIRS_PER_ROW]))
                    for j in range(0, len(p), PAIRS_PER_ROW)]
    assert len(out) == at
    return out


def pad(rs: Sequence[bytes], bits: int = GRID_BITS) -> list[bytes]:
    """`rs` with all-zero rows appended up to `grid(len(rs), bits)`."""
    return list(rs) + [bytes(ROW_BYTES)] * (grid(len(rs), bits) - len(rs))


class Malformed(ValueError):
    """Rows that are not an encoding of any advice (`rows-malformed`, with where)."""


def _int(b: bytes, lo: int, hi: int) -> int:
    return int.from_bytes(b[lo:hi], "little")


def _zero(b: bytes, lo: int, what: str) -> None:
    if any(b[lo:]):
        raise Malformed(f"{what}: bytes past {lo} are not zero")


class Reader:
    """Rows by index, recording every index read (the openings a local check needs)."""

    def __init__(self, rs: Sequence[bytes] | Mapping[int, bytes]) -> None:
        self._rows = rs
        self.read: set[int] = set()

    def __call__(self, i: int) -> bytes:
        try:
            r = self._rows[i]
        except (IndexError, KeyError):
            raise Malformed(f"row {i} is not there") from None
        if not isinstance(r, (bytes, bytearray)) or len(r) != ROW_BYTES:
            raise Malformed(f"row {i} is not {ROW_BYTES} bytes")
        self.read.add(i)
        return bytes(r)


def _header(read: Reader) -> tuple[str, int, int, int, int]:
    h = read(0)
    if h[:len(MAGIC)] != MAGIC:
        raise Malformed("row 0: not a verity/partition/v2 header")
    o = len(MAGIC)
    _zero(h, o + 96, "row 0")
    return h[o:o + 64].hex(), _int(h, o + 64, o + 72), _int(h, o + 72, o + 80), _int(h, o + 80, o + 88), _int(h, o + 88, o + 96)


def _call(read: Reader, k: int) -> Call:
    r = read(1 + k)
    _zero(r, 32, f"call row {k}")
    return Call(_int(r, 0, 8), _int(r, 8, 16), _int(r, 16, 24), _int(r, 24, 32))


def _directory(read: Reader, m: int, j: int) -> tuple[str, int, int, int, int, int, int]:
    """Directory row j: (definition digest, kind, gates, units, first payload row, payload rows, items), items a flat entry's
    shape classes or a parts entry's body nodes."""
    r = read(1 + m + j)
    _zero(r, 112, f"directory row {j}")
    kind = _int(r, 64, 72)
    if kind not in (FLAT, PARTS):
        raise Malformed(f"directory row {j}: kind {kind}")
    return (r[:64].hex(), kind, *(_int(r, x, x + 8) for x in range(72, 112, 8)))  # type: ignore[return-value]


def _words(read: Reader, j: int, at: int, count: int, t: int, what: str) -> list[int]:
    r = read(at + t)
    c = min(OWNERS_PER_ROW, count - t * OWNERS_PER_ROW)
    _zero(r, 4 * c, f"entry {j} {what} row {t}")
    return [_int(r, 4 * x, 4 * x + 4) for x in range(c)]


def _digests(read: Reader, j: int, at: int, count: int, t: int) -> list[str]:
    r = read(at + t)
    c = min(SHAPES_PER_ROW, count - t * SHAPES_PER_ROW)
    _zero(r, 64 * c, f"entry {j} class row {t}")
    return [r[64 * x:64 * x + 64].hex() for x in range(c)]


def _rows_of(count: int, per: int) -> int:
    return -(-count // per)


def _entry(read: Reader, m: int, j: int, total: int, unit: int | None = None) -> Entry:
    """Entry j from its directory and payload rows.  With `unit` (a unit of a flat entry) only that unit's class row and its
    class's digest row are read: the other units' classes come back -1 and the other digests empty."""
    d, kind, gates, units, start, n_rows, items = _directory(read, m, j)
    if start + n_rows > total:
        raise Malformed(f"entry {j}: payload rows {start}..{start + n_rows} past the {total} rows")
    if kind == PARTS:
        if _rows_of(items, PAIRS_PER_ROW) != n_rows:
            raise Malformed(f"entry {j}: {n_rows} payload rows for {items} parts")
        parts: list[tuple[int | None, int]] = []
        for t in range(n_rows):
            r = read(start + t)
            c = min(PAIRS_PER_ROW, items - t * PAIRS_PER_ROW)
            _zero(r, 16 * c, f"entry {j} parts row {t}")
            parts += [(None if (ch := _int(r, 16 * x, 16 * x + 8)) == NONE else ch, _int(r, 16 * x + 8, 16 * x + 16)) for x in range(c)]
        return Entry(d, PARTS, gates, units, parts=tuple(parts))
    o_rows, u_rows, c_rows = _flat_rows(gates, units, items)
    if o_rows + u_rows + c_rows != n_rows:
        raise Malformed(f"entry {j}: {n_rows} payload rows for {gates} owners, {units} units and {items} classes")
    at_shape, at_class = start + o_rows, start + o_rows + u_rows
    if units == 1:
        if not items == 1:
            raise Malformed(f"entry {j}: one unit of {items} classes")
        return Entry(d, FLAT, gates, 1, (0,) * gates, (0,), tuple(_digests(read, j, at_class, items, 0)))
    owner = [w for t in range(o_rows) for w in _words(read, j, start, gates, t, "owner")]
    if unit is None:
        shape = [w for t in range(u_rows) for w in _words(read, j, at_shape, units, t, "class")]
        classes = [x for t in range(c_rows) for x in _digests(read, j, at_class, items, t)]
    else:
        if not 0 <= unit < units:
            raise Malformed(f"entry {j}: no unit {unit} of {units}")
        t = unit // OWNERS_PER_ROW
        shape = [-1] * units
        shape[t * OWNERS_PER_ROW:t * OWNERS_PER_ROW + OWNERS_PER_ROW] = _words(read, j, at_shape, units, t, "class")
        k = shape[unit]
        if not k < items:
            raise Malformed(f"entry {j}: unit {unit} of class {k}, {items} classes")
        classes = [""] * items
        t = k // SHAPES_PER_ROW
        classes[t * SHAPES_PER_ROW:t * SHAPES_PER_ROW + SHAPES_PER_ROW] = _digests(read, j, at_class, items, t)
    return Entry(d, FLAT, gates, units, tuple(owner), tuple(shape), tuple(classes))


def decode(rs: Sequence[bytes]) -> Advice:
    """The advice the rows encode (`Malformed` when they encode none): the inverse of `rows`, the zero rows `pad` adds allowed."""
    read = Reader(rs)
    program, n, m, e, used = _header(read)
    if not 1 + m + e <= used <= len(rs):
        raise Malformed(f"row 0: {used} rows used, {m} calls and {e} entries, {len(rs)} rows")
    calls = tuple(_call(read, k) for k in range(m))
    entries = tuple(_entry(read, m, j, used) for j in range(e))
    for i in range(used, len(rs)):
        if any(read(i)):
            raise Malformed(f"row {i}: past the rows used and not zero")
    adv = Advice(program, calls, entries)
    if rows(adv) != list(rs[:used]):
        raise Malformed("the rows are not the canonical encoding of what they hold")
    if adv.units != n if calls else n != 0:
        raise Malformed(f"row 0: {n} units, the Call table {adv.units}")
    return adv


# --- the statement ---------------------------------------------------------------------------------------------------------------


def predicates(width: int | None = WIDTH, recompute: str = "refuse", max_gates: int | None = None,
               max_in_bits: int | None = None, canonical: Mapping[str, Any] | None = None, grid_bits: int = GRID_BITS) -> dict[str, Any]:
    """The public predicates: the limit on a unit's bits out or none, the recompute rule, the caps on a unit's circuit, the
    canonical query whose unit counts bound each entry's, and the grid."""
    return {"width": width, "recompute": recompute, "max_gates": max_gates,
            "max_in_bits": max_in_bits, "canonical": dict(canonical) if canonical is not None else None, "grid_bits": grid_bits}


def statement(family: str, preds: Mapping[str, Any], advice: Advice, root: bytes | str) -> dict[str, Any]:
    """The public statement of `advice` registered under `root`, its counts rounded up to the grid."""
    g = preds["grid_bits"]
    return {"format": FORMAT, "family": family, "predicates": dict(preds), "units": grid(advice.units, g),
            "advice": {"rows": len(pad(rows(advice), g)), "root": root.hex() if isinstance(root, bytes) else root}}


def _pos(x: Any) -> bool:
    return _is_int(x) and 0 < x <= (1 << 62)


def validate(st: Any) -> CutVerdict:
    """The statement's own form: exactly its keys; a non-empty family; predicates of exactly their keys and ranges (`canonical` a
    valid `Q_word` query of `partition_object.QUERIES`); the counts on the grid; a SHA-512 root."""
    bad: dict[str, list] = {}

    def fail(code: str, what: Any) -> None:
        bad.setdefault(code, []).append(what)

    if not isinstance(st, Mapping):
        return _verdict({"not-an-object": [type(st).__name__]})
    if set(st) != _KEYS:
        fail("keys", sorted(set(st) ^ _KEYS, key=repr))
    if st.get("format") != FORMAT:
        fail("format", st.get("format"))
    if not (isinstance(st.get("family"), str) and st["family"]):
        fail("family", st.get("family"))
    p = st.get("predicates")
    g = None
    if not isinstance(p, Mapping) or set(p) != _PRED_KEYS:
        fail("predicate-keys", sorted(set(p) ^ _PRED_KEYS, key=repr) if isinstance(p, Mapping) else p)
    else:
        w = p["width"]
        if w is not None and not (_is_int(w) and 0 < w <= PARAM_MAX):
            fail("predicates", {"width": w})
        if p["recompute"] not in RECOMPUTE_RULES:
            fail("predicates", {"recompute": p["recompute"]})
        for k in ("max_gates", "max_in_bits"):
            if p[k] is not None and not _pos(p[k]):
                fail("predicates", {k: p[k]})
        c = p["canonical"]
        if c is not None and not (isinstance(c, Mapping) and c.get("name") == "Q_word" and PO.validate(
                {"format": PO.FORMAT, "program": "0" * 128, "query": c}).ok):
            fail("predicates", {"canonical": c})
        if not (_is_int(p["grid_bits"]) and 1 <= p["grid_bits"] <= 64):
            fail("predicates", {"grid_bits": p["grid_bits"]})
        else:
            g = p["grid_bits"]
    for k, lo in (("units", 0),):
        v = st.get(k)
        if not (_is_int(v) and v >= lo) or (g is not None and grid(v, g) != v):
            fail("units", v)
    a = st.get("advice")
    if not (isinstance(a, Mapping) and set(a) == {"rows", "root"} and _is_int(a["rows"]) and a["rows"] >= 1
            and (g is None or grid(a["rows"], g) == a["rows"]) and isinstance(a["root"], str) and _HEX512.fullmatch(a["root"])):
        fail("advice", a)
    return _verdict(bad)


def digest(st: Mapping[str, Any]) -> str:
    """`SHA-512("verity/partition/v2\\0" || canon(statement))`, hex, of a valid statement."""
    v = validate(st)
    if not v.ok:
        raise ValueError(f"not a valid {FORMAT} statement: {v.codes}")
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical_json(st)).hexdigest()


def _verdict(bad: Mapping[str, list], detail: dict | None = None) -> CutVerdict:
    codes = tuple(k for k, v in bad.items() if v)
    d = dict(detail or {})
    for k in codes:
        d[k] = {"n": len(bad[k]), "gates": bad[k][:_MAX_LISTED]}
    return CutVerdict(not codes, codes, d)


# --- the global predicates -------------------------------------------------------------------------------------------------------


class _Graphs:
    def __init__(self) -> None:
        self.g: dict[str, CallGraph] = {}
        self.a: dict[str, Activation] = {}
        self.memo: dict = {}

    def graph(self, fn: Any) -> CallGraph:
        g = self.g.get(fn.id)
        if g is None:
            g = self.g[fn.id] = CallGraph(fn)
        return g

    def activation(self, fn: Any) -> Activation:
        a = self.a.get(fn.id)
        if a is None:
            a = self.a[fn.id] = Activation(fn)
        return a


def check(st: Mapping[str, Any], rs: Sequence[bytes], program: Any) -> CutVerdict:
    """G1-G7: the advice rows `rs` (all R_b of them, as registered) against the statement and the program.  The rows' root is the
    commitment's to check (`verity.ml.commitments.advice`); this is what the registration statement proves of them."""
    v = validate(st)
    if not v.ok:
        return v
    if len(rs) != st["advice"]["rows"]:
        return _verdict({"rows-count": [[len(rs), st["advice"]["rows"]]]})
    try:
        adv = decode(rs)
    except Malformed as e:
        return _verdict({"rows-malformed": [str(e)]})
    p = st["predicates"]
    bad: dict[str, list] = {}

    def fail(code: str, what: Any) -> None:
        bad.setdefault(code, []).append(what)

    order = PO.order_violation(program)
    if order is not None:
        return _verdict({"program-not-topological": [order]})
    G = _Graphs()
    # G1: the Call table against the program's root Calls
    if adv.program != program_digest(program):
        fail("header-program", adv.program)
    calls = PO.calls(program)
    nodes = program.root.body.nodes
    if len(calls) != len(adv.calls):
        fail("call-count", [len(adv.calls), len(calls)])
    fns: dict[int, Any] = {}
    reached: list[int] = []
    start = 0
    for (k, fn), c in zip(calls, adv.calls):
        if c.node != k:
            fail("call-node", [c.node, k])
        if nodes[k].form not in ("call", "primitive", "batch"):
            fail("call-form", [k, nodes[k].form])
        if c.members != _members(nodes[k]):
            fail("call-members", [k, c.members, _members(nodes[k])])
        if c.start != start:
            fail("call-start", [k, c.start, start])
        if not c.entry < len(adv.entries):
            fail("call-definition", [k, c.entry])
            continue
        _reach(c.entry, fn, adv, fns, reached, fail)
        start += c.members * adv.entries[c.entry].units
    if adv.units != start:
        fail("units-total", [adv.units, start])
    # G2: one entry per Definition, each reached, in first-reach order, of its Definition's structure
    seen: dict[str, int] = {}
    for j, en in enumerate(adv.entries):
        if en.definition in seen:
            fail("entry-duplicate", [j, seen[en.definition]])
        seen.setdefault(en.definition, j)
    unreached = [j for j in range(len(adv.entries)) if j not in fns]
    if unreached:
        fail("entry-unreached", unreached)
    if reached != sorted(reached) or reached != list(range(len(reached))):
        fail("entry-order", reached[:_MAX_LISTED])
    width = p["width"]
    canon = p["canonical"]
    for j, fn in sorted(fns.items()):
        en = adv.entries[j]
        if en.definition != definition_digest(fn):
            fail("entry-definition", [j, fn.id])
            continue
        if canon is not None and en.units > evaluate_definition(fn, int(canon["params"]["X"]), G.memo).units:
            fail("units-canonical", [j, fn.id, en.units])
        if en.kind == PARTS:
            if _check_parts(j, en, fn, adv, G, fail) and p["recompute"] == "refuse":
                # members are cut apart, so only a value two members compute crosses units, and only `refuse` refuses it
                owner = _flat_owner(j, adv)
                pairs, computing = _recomputed(fn)
                across = [[a, b] for a, b in pairs if owner[a] != owner[b]] if len(owner) == computing else []
                if len(owner) != computing:
                    fail("entry-gates", [j, len(owner), computing])
                elif across:
                    fail("cut", {"entry": j, "definition": fn.id, "codes": ["gate-recomputed"],
                                 "detail": {"gate-recomputed": {"n": len(across), "gates": across[:_MAX_LISTED]}}})
            continue
        g = G.graph(fn)
        if en.gates != g.n:
            fail("entry-gates", [j, en.gates, g.n])
        if len(en.owner) != g.n:
            fail("owner-length", [j, len(en.owner), g.n])
            continue
        if sorted(set(en.owner)) != list(range(en.units)):
            fail("owner-dense", [j, en.units])
            continue
        # G3, G5: the invariant and the recompute rule on the entry's graph
        r = check_cut(g, en.owner, None, None, p["recompute"])
        if not r.ok:
            fail("cut", {"entry": j, "definition": fn.id, "codes": list(r.codes), "detail": r.detail})
        # G4: the limit on every unit's bits out
        if width is not None:
            ob, og = boundary_widths(g, en.owner, en.units)
            wide = [[u, b, c] for u, (b, c) in enumerate(zip(ob, og)) if b > width]
            if wide:
                fail("unit-too-wide", {"entry": j, "definition": fn.id, "units": wide[:_MAX_LISTED], "n": len(wide)})
        # G7: the shapes, and the caps
        ucs = G.activation(fn).circuits(en.owner)
        want = _classes(uc.shape for uc in ucs)
        if (en.shape, en.classes) != want:
            wrong = [u for u, (a, b) in enumerate(zip(en.shape, want[0])) if a != b or en.classes[a:a + 1] != want[1][b:b + 1]]
            fail("shape", {"entry": j, "definition": fn.id, "units": wrong[:_MAX_LISTED], "n": len(wrong) + abs(len(en.shape) - len(ucs)),
                           "classes": [len(en.classes), len(want[1])]})
        big = [[u, len(uc.gates), sum(uc.in_widths)] for u, uc in enumerate(ucs)
               if (p["max_gates"] is not None and len(uc.gates) > p["max_gates"])
               or (p["max_in_bits"] is not None and sum(uc.in_widths) > p["max_in_bits"])]
        if big:
            fail("unit-too-large", {"entry": j, "definition": fn.id, "units": big[:_MAX_LISTED], "n": len(big)})
    # G6: the unit count on the grid
    if st["units"] != grid(adv.units, p["grid_bits"]):
        fail("units-bucket", [st["units"], adv.units])
    detail = {"units": adv.units, "calls": len(adv.calls), "entries": len(adv.entries), "rows": len(rs)}
    return _verdict(bad, detail)


def _reach(j: int, fn: Any, adv: Advice, fns: dict[int, Any], reached: list[int], fail: Callable[[str, Any], None]) -> None:
    """Assign Definition `fn` to entry `j` (and a parts entry's children to theirs), recording first reaches in order."""
    if j in fns:
        if fns[j].id != fn.id:
            fail("call-definition", [j, fns[j].id, fn.id])
        return
    fns[j] = fn
    reached.append(j)
    en = adv.entries[j]
    if en.kind != PARTS:
        return
    sep = separable(fn)
    if sep is None or len(sep) != len(en.parts):
        return                                         # refused by `_check_parts`
    for (child, _m), (callee, _n) in zip(en.parts, sep):
        if child is not None:
            if not child < len(adv.entries):
                fail("parts-child", [j, child])
                continue
            _reach(child, callee, adv, fns, reached, fail)


def _check_parts(j: int, en: Entry, fn: Any, adv: Advice, G: _Graphs, fail: Callable[[str, Any], None]) -> bool:
    """A parts entry against its Definition's body (`cut.separable`): node for node, members and children, units and gates."""
    sep = separable(fn)
    if sep is None:
        fail("entry-kind", [j, fn.id])
        return False
    if len(sep) != len(en.parts) or any(m != n for (_c, m), (_f, n) in zip(en.parts, sep)):
        fail("parts-shape", [j, fn.id])
        return False
    units = gates = 0
    for (child, m), (callee, _n) in zip(en.parts, sep):
        has = G.graph(callee).n > 0
        if (child is None) == has or (child is not None and not child < len(adv.entries)):
            fail("parts-child", [j, callee.id, child])
            return False
        if child is not None:
            units += m * adv.entries[child].units
            gates += m * adv.entries[child].gates
    ok = True
    if units != en.units:
        fail("entry-units", [j, en.units, units])
        ok = False
    if gates != en.gates:
        fail("entry-gates", [j, en.gates, gates])
        ok = False
    return ok


def _recomputed(fn: Any, wiring: frozenset[str] = WIRING) -> tuple[list[tuple[int, int]], int]:
    """`cut.CallGraph(fn).recomputed` and `.n` in one pass over the gates, keeping only a token per gate and the computing keys:
    the (gate, earlier gate) pairs, compact, by step 3 of `cut`'s algorithm, and the number of computing gates.  A key is its
    primitive's index and its operands' tokens packed as bytes; a token is a fresh id >= 0 per new value, and -1 - g for the
    input gate g, so equal tokens are equal values as `cut._Interner`'s are, with no table of the inputs (a Gemm's weights).  A separable body's graph holds every member's operands at once; this holds none of them, so a parts
    entry over a vocabulary-wide Gemm is checked in its gates' time."""
    c, call = call_scope(fn)
    lo, n = call.offset, fn.gates
    const, tok = bytearray(n), array("q", bytes(8 * n))
    prims: dict[str, int] = {}
    named: dict[bytes, int] = {}                   # a structure value's token
    first: dict[bytes, int] = {}                   # computing key -> the first gate with it
    out: list[tuple[int, int]] = []
    idx = array("q", bytes(8 * n))
    compact = fresh = 0

    def name(key: bytes) -> int:
        nonlocal fresh
        t = named.get(key)
        if t is None:
            t = named[key] = fresh
            fresh += 1
        return t

    for j in range(n):
        g = c.gate(lo + j)
        is_const = not g.prim.params
        is_free = is_const or g.prim.id in wiring
        wide = len(g.operands) > WIDE_FAN_IN
        ext = wide
        key = array("q", [prims.setdefault(g.prim.id, len(prims))])
        operands = 0
        all_const = True
        for o in (wide_operands(g, call) if wide else g.operands):
            if 0 <= o - lo < n:
                if o - lo >= j:
                    raise QueryInapplicable(f"{fn.id}: gate {j} reads gate {o - lo}, not before it (the layout is not topological)",
                                            {"definition": fn.id, "gate": j, "reads": o - lo, "why": "not topological"})
                key.append(tok[o - lo])
                all_const = all_const and bool(const[o - lo])
                operands += 1
            else:
                ext = True
                if not wide:
                    key.append(-1 - o)
                    all_const = False
                    operands += 1
        if not is_free and not ext and operands and all_const:
            is_const = is_free = True
        const[j] = is_const
        k = key.tobytes()
        if is_free:
            tok[j] = name(k)
            continue
        idx[j] = compact
        compact += 1
        f = None if wide else first.get(k)
        if f is not None:
            out.append((idx[j], idx[f]))
            tok[j] = tok[f]
        else:
            if not wide:
                first[k] = j
            tok[j] = fresh
            fresh += 1
    return out, compact


def _flat_owner(j: int, adv: Advice) -> list[int]:
    """Entry j's owner per computing gate of one activation, its parts' members' owners concatenated (`cut.DefinitionCut.owner`)."""
    en = adv.entries[j]
    if en.kind == FLAT:
        return list(en.owner)
    out: list[int] = []
    base = 0
    for child, m in en.parts:
        if child is None:
            continue
        sub, cu = _flat_owner(child, adv), adv.entries[child].units
        for _ in range(m):
            out += [u + base for u in sub]
            base += cu
    return out


# --- the local predicates --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofBlock:
    """Unit `index` at its place: root Call `call` (root-body node `node`), member `member`, the (body node, member) steps down to the
    flat Definition `definition` holding it, the unit `unit` of that entry, its circuit, and the root gates its outputs and inputs
    are (L3: the positions their committed values open at).  `filler`: an index in n..n_b-1, no gates."""

    index: int
    filler: bool = False
    call: int = -1
    node: int = -1
    member: int = 0
    path: tuple[tuple[int, int], ...] = ()
    entry: int = -1
    unit: int = -1
    definition: str = ""
    circuit: ProofBlockCircuit | None = None
    base: int = 0
    outputs: tuple[int, ...] = ()
    inputs: tuple[int, ...] = ()
    detail: dict = field(default_factory=dict)


def locate(st: Mapping[str, Any], read: Callable[[int], bytes], i: int) -> tuple[int, int, int, list[tuple[int, int]], int, int, Entry] | None:
    """L1 on the rows `read` yields: unit `i` -> (Call k, its root node, member, path of (body node, member), flat entry, unit
    within it, that entry), or None for a filler.  Reads the header, the Call rows of a bisection over the starts, and the
    directory and payload rows down to the flat entry."""
    if not (_is_int(i) and 0 <= i < st["units"]):
        raise IndexError(f"unit {i} outside 0..{st['units'] - 1}")
    _program, n, m, e, used = _header(read)
    if i >= n:
        return None
    lo, hi = 0, m - 1
    while lo < hi:                                      # the last Call whose start is <= i
        mid = (lo + hi + 1) // 2
        if _call(read, mid).start <= i:
            lo = mid
        else:
            hi = mid - 1
    c = _call(read, lo)
    if lo + 1 < m and not _call(read, lo + 1).start > i:
        raise Malformed(f"call rows {lo}, {lo + 1}: starts not increasing past unit {i}")
    j = c.entry
    if not j < e:
        raise Malformed(f"call row {lo}: entry {j} of {e}")
    units = _directory(read, m, j)[3]
    off = i - c.start
    if not 0 <= off < c.members * units:
        raise Malformed(f"unit {i}: not in call {lo}'s {c.members} x {units} units from {c.start}")
    member, u = divmod(off, units)
    path: list[tuple[int, int]] = []
    while True:
        kind = _directory(read, m, j)[1]
        if kind == FLAT:
            return lo, c.node, member, path, j, u, _entry(read, m, j, used, unit=u)
        for node, (child, mm) in enumerate(_entry(read, m, j, used).parts):
            if child is None:
                continue
            if not child < e:
                raise Malformed(f"entry {j}: child {child} of {e}")
            cu = _directory(read, m, child)[3]
            if u < cu * mm:
                path.append((node, u // cu))
                u %= cu
                j = child
                break
            u -= cu * mm
        else:
            raise Malformed(f"entry {j}: unit {u} past its parts")
        if len(path) > e:
            raise Malformed(f"unit {i}: a path through more than the {e} entries")


def _place(program: Any, node: int, member: int, path: Sequence[tuple[int, int]]) -> tuple[Any, int]:
    """The flat Definition at (root node, member, path) and the root gate of its first own gate."""
    fn = program.root.body.nodes[node].fn
    base = program.root.body.offsets[node] + member * fn.gates
    for k, mm in path:
        if not hasattr(fn, "body") or not 0 <= k < len(fn.body.nodes):
            raise Malformed(f"{fn.id}: no body node {k}")
        child = fn.body.nodes[k].fn
        base += fn.body.offsets[k] + mm * child.gates
        fn = child
    return fn, base


def unit(st: Mapping[str, Any], read: Callable[[int], bytes], i: int, program: Any, acts: dict | None = None) -> ProofBlock:
    """L1-L3 for unit `i`: where it is (from the rows), its circuit cut from the program there, and its outputs' and inputs' root
    gates.  `detail` holds the L2 failures (`unit-definition`, `unit-shape`)."""
    got = locate(st, read, i)
    if got is None:
        return ProofBlock(i, filler=True)
    k, node, member, path, j, u, en = got
    fn, base = _place(program, node, member, path)
    bad: dict[str, list] = {}
    if definition_digest(fn) != en.definition:
        bad["unit-definition"] = [[j, fn.id]]
        return ProofBlock(i, False, k, node, member, tuple(path), j, u, fn.id, None, base, detail=bad)
    acts = {} if acts is None else acts
    a = acts.get(fn.id)
    if a is None:
        a = acts[fn.id] = Activation(fn)
    if len(en.owner) != a.graph.n:
        raise Malformed(f"entry {j}: {len(en.owner)} owners for {a.graph.n} computing gates")
    uc = a.circuits(en.owner)[u]
    if uc.shape != en.classes[en.shape[u]]:
        bad["unit-shape"] = [[j, u]]
    root = _roots(program, a, base, uc)
    return ProofBlock(i, False, k, node, member, tuple(path), j, u, fn.id, uc, base, tuple(root[x] for x in uc.outputs),
                tuple(root[x] for x in uc.inputs), bad)


def _roots(program: Any, a: Activation, base: int, uc: ProofBlockCircuit) -> dict[int, int]:
    """Activation-local gate -> root gate, for the unit's gates and every operand they read."""
    out: dict[int, int] = {}
    circ = program.circuit
    for x in uc.gates:
        out[x] = base + x
        g, r = a.gate(x), circ.gate(base + x)
        if g.prim.id != r.prim.id or len(g.operands) != len(r.operands):
            raise AssertionError(f"{a.fn.id} gate {x} is not root gate {base + x}")
        for o, ro in zip(g.operands, r.operands):
            out.setdefault(a.local(o), ro)
    return out


def check_unit(st: Mapping[str, Any], read: Callable[[int], bytes], i: int, program: Any, values: Mapping[int, int],
               acts: dict | None = None) -> CutVerdict:
    """L1-L4 for drawn unit `i`: located and cut (`unit`), then its gates recomputed from the committed values `values` (root gate
    -> value) at its inputs and compared with them at its outputs (`unit-incorrect`: the root gates that differ)."""
    try:
        U = unit(st, read, i, program, acts)
    except Malformed as e:
        return _verdict({"unit-locate": [str(e)]})
    except IndexError as e:
        return _verdict({"unit-outside": [str(e)]})
    if U.filler:
        return _verdict({}, {"filler": True})
    bad = dict(U.detail)
    detail = {"call": U.call, "member": U.member, "path": [list(s) for s in U.path], "entry": U.entry, "unit": U.unit,
              "definition": U.definition, "outputs": list(U.outputs), "inputs": list(U.inputs)}
    if U.circuit is None or bad:
        return _verdict(bad, detail)
    a = (acts or {}).get(U.definition) or Activation(_place(program, U.node, U.member, U.path)[0])
    uc = U.circuit
    mine: dict[int, int] = {}
    root = _roots(program, a, U.base, uc)
    try:
        for x in uc.gates:
            g = a.gate(x)
            ops = [mine[k] if k in mine else values[root[k]] for k in (a.local(o) for o in g.operands)]
            lazy = getattr(g.prim, "evaluate_lazy", None)
            mine[x] = lazy(lambda t, _o=ops: _o[t]) if lazy is not None else g.prim.evaluate(*ops)
    except KeyError as e:
        return _verdict({"unit-incorrect": [{"missing": e.args[0]}]}, detail)
    wrong = [U.base + x for x in uc.outputs if values.get(U.base + x) != mine[x]]
    return _verdict({"unit-incorrect": wrong}, detail)


def openings(st: Mapping[str, Any], rs: Sequence[bytes], i: int) -> list[int]:
    """The rows unit `i`'s local check reads (L1), ascending: what the prover opens with it."""
    read = Reader(rs)
    locate(st, read, i)
    return sorted(read.read)


def from_v1(obj: Mapping[str, Any]) -> dict[str, Any]:
    """The predicates a v1 object's query implies: `Q_word{X, W}` v1 or v2 a limit of X bits out, its recompute rule and the query
    itself as the canonical bound; a template query no width limit.  W has no counterpart: a unit the query admits through its
    one-value exception (one value of more than X bits) fails G4."""
    q = obj["query"]
    if q["name"] == "Q_word":
        return predicates(q["params"]["X"], RECOMPUTE[q["version"]], canonical=q)
    if (q["name"], q["version"]) in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES):
        return predicates(None)
    raise ValueError(f"no v2 predicates for {q['name']!r} v{q['version']!r}")


__all__ = ["FORMAT", "ROW_BYTES", "ROW_BITS", "MAGIC", "BINDING", "NONE", "FLAT", "PARTS", "GRID_BITS", "WIDTH", "UNIT_CLASS_MAX",
           "UNIT_CLASSES", "Call", "Entry", "Advice", "Malformed", "MixedClasses", "Reader", "ProofBlock", "grid", "definition_digest",
           "unit_class", "build", "rows", "pad", "decode", "predicates",
           "statement", "validate", "digest", "check", "locate", "unit", "check_unit", "openings", "from_v1"]
