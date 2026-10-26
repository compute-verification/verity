"""I2's served window as one Circuit (VLLM-103's V3b, `verity.ml.windows.interface`): the window *class's* circuit, every
unit's output a committed row and every input row wired, registered or ingress.

A window class (`WindowClass`, N1's scan class: note:network-accounting/20261025T0830Z-report-net68-scan-router) has a
fixed number of slots a network timestep (ASK-122): ``lanes`` slots at each boundary, one every ``every`` scan buckets from the
window's start, ``slots`` in all, each of at most ``prompt`` prompt ids and ``outputs`` outputs. Slot 4b + j at the class of
record is boundary b's lane j, anchored at bucket 16b. Its circuit is one class Program a slot, whatever the traffic
(`class_circuit`): the Build's greedy request Program at one prompt row and P - 1 decode steps, P = prompt + outputs - 1
positions, every position computing the whole stack, the LM head and the token select, position p attending to keys 0..p.
So the circuit is a function of the class alone: which slots carry a request, arrival steps, prompt lengths, output lengths
and key counts are values, never structure. A slot's ingress row (`SLOT_ROW`) carries its occupancy word, its prompt's
length and its ids, the system's ingress at the slot's anchor. Position p's token is `io_shell.TokenFeed_v1{prompt, FIRST}`'s
while p is below the class's prompt cap (the prompt's id below its length, else the previous position's select; the slot it
carries starts as the row's), and past it the previous position's select. Each position's output is its select masked by
the slot's occupancy word (`io_shell.SlotMask_v1`), so an empty slot's outputs are `io_shell.EMPTY` wherever they are read.
A request decodes to the class's outputs, its selects past its finish masked by what reads them (external until N1's
Programs' tie, IO-TIE's, is in the circuit). `window_circuit(registered, compiled, cls, grain, placed=...)` refuses a window
the class's slots don't take and emits the class's circuit.

`class_circuit` walks the class Program's nodes in order and lowers each to units at ``grain``, once a slot: node j's units
of slot 0, then of slot 1 and so on, each allocation's slots side by side in its template (`_Builder.unit`):

- `Embedding_v1{V, H}` is the registered read `EmbeddingRead_v1{V, H}` (`commitments.gates.embedding`): its ``root`` and
  ``domain`` are the embedding table's (a `Table` over the weights record, read by `TableRead`), its ``row``, ``salt`` and
  ``path`` an `Opening` at the hidden key ``tok``, pinned by its verdict ``o1``. Its ``tok`` is its position's feed.
- `Gemm_v2{K, N}` is ⌈N / n⌉ slices `Gemm_v2{K, n}`, n = ``grain.gemm_words`` // K output columns a slice, each reading
  its weight rows from the weights record (`Read`) and its ``x`` whole.
- `RoPE_v1{NHEADS, D}` is its heads (`RoPEHead_v1{D}`), each reading its slice of ``x``.
- `Attention_v18{T, NH, KVH, D, BN, ...}` is `Attention_v19`'s cut (#4263, CIR-96): each head's Calls (`AttentionHead_v19`'s
  body, read once a T) are its units, its first visited key block `AttnBlockFirst_v1{D, NVIS}` (keys (nb - 1)·BN ... T - 1,
  nb = ⌈T / BN⌉), each full block below it `AttnBlock_v5` and its end `AttnHeadEnd_v1{D}`, one unit a head a Call, each
  block's running state its predecessor's rows. A block's keys and values are one row a KV group, which its group's heads
  read: one key's own row (RoPE's, or its value's GEMM slices gathered once, at its position), a full block's
  `AllGather_v1{BN, D}` (made once, when the block fills, and read by every later position) or the first block's
  `AllGather_v1{NVIS, D}`. A layer's cache is refused unless it is the keys its earlier positions' nodes read and one more.
- `TokenSelect_v4{V}` is its function at ``grain.select`` logits a chunk (its body's own form at another C): chunks
  `TokenSelectChunk_v2{C}`, each reading its C logits through one gather, and `TokenSelectCombine_v2{V, C}` over them:
  ``best`` through one gather of the chunks' maxima, ``idx`` cut a row a chunk, and ``x0`` (``logits[0]``) from one
  `Gemm_v2{K, 1}` unit over the LM head's input and its first weight row.
- `RMSNormTriton_v5`, `RMSNormFusedCuda_v2` and `SiluMul_v2` are one unit each.

A reader port that reads several writers' rows is cut (`interface.Layout.cut`), one cut per template and port: the cut ``d``
divides every use's row count, leaves its reader at most ``grain.rows`` rows in and out, and makes the fewest gather units,
a use's rows gathered ``m / d`` at a time by `AllGather_v1` units (whose ``parts`` are cut a row a part, so a gather of
``m / d`` parts holds ``m / d + 1`` rows with its output). Every wire and read is stated per node as a pattern (a node's
heads, KV groups and slices as digits of one) and `interface.coalesce`d, and each allocation's slots sit side by side in its
template, so the slots' copies of a pattern are one and the circuit's description grows with the class Program's nodes, not
its slots or units; so does the lowering's own state (a node's output rows are runs, a gather is keyed by its rows' runs).

The circuit's blocks (`interface.Blocks`) carry its inference block: a unit's slot is its slot (its class Program; the
blocks' slots are the class's S) and its step its position, T the class's positions. An embedding opens its position, the
stack, LM head, token select and mask after it are that position's, a slot's ingress row is its first position's, and a
gather is its reader's. An allocation's units are one entry, its slots' runs at one step side by side (the slot digit, then
the run), and like entries that follow on merge, at one step and then at consecutive steps (`_inference`).

Refused by name (`Unexpressible`): a window past its class's slots (more requests than slots, more anchored at a boundary
than its lanes, an anchor that is no boundary, a prompt past the cap, more outputs than the cap), a request whose last
output is past the window's last bucket (`ACROSS`), a request whose Program would read more than its prompt and the weights
(a stochastic request's seed, a prefix-cache hit's prefix, a TP rank's peers), a request a Commit window carries in or out,
byte ingress, a family this lowering doesn't know, a weight whose leaves aren't whole u16 words, a row form the cut rule
can't state and a unit of more than ``grain.rows`` rows. The circuit is completeness only: that it computes what serving ran
is held by the units' proofs on their committed rows, never by this lowering.
"""
from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Refs, runs
from verity.ml.circuits.types import Tuple
from verity.ml.commitments.gates import embedding as E
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.b1_tp2 import AllGather
from verity.ml.definitions.registry.dead_free import AttentionV19
from verity.ml.windows import interface as IF

from .build import Unexpressible
from .steps import RequestUnit, request_program

NAME = "verity-vllm/window"
WEIGHTS = "weights"
PROMPT = "prompt"
GREEDY = {"kind": "greedy"}
#: the Definition that reads a slot's ingress row, (occupancy, prompt length, ids). A placeholder for the row network-
#: accounting's BLOCKS-NET PR names (ASK-122: token ids in a committed row with a private occupancy word, tied to the
#: certified bytes by #4548's id form); the lowering reads a slot's row only through this
SLOT_ROW = IO.SlotRow
#: a request held across the window's end, which the class doesn't take: whether the network certifier's lanes hold one
#: across a window boundary is open (thread 1793030365.246499), and a slot that does needs a Program running into the next
#: window
ACROSS = "held across the window's end"


@dataclass(frozen=True)
class WindowClass:
    """A window class's public caps (N1's scan `Class`: its slots, its prompt ids and its KO) and its slots' timing (ASK-122:
    fixed slots a network timestep): ``slots`` class Programs a window, each of at most ``prompt`` prompt ids and
    ``outputs`` outputs, ``lanes`` of them anchored at each boundary, one every ``every`` scan buckets from the window's
    start (lanes 0: every slot at the window's one boundary). Slot ``lanes · b + j`` is boundary b's lane j, anchored at bucket
    ``every · b``, and the window is ``buckets`` = every · (slots / lanes) buckets."""
    slots: int
    prompt: int
    outputs: int
    lanes: int = 0
    every: int = 16

    def __post_init__(self) -> None:
        if not self.lanes:
            object.__setattr__(self, "lanes", self.slots)
        if not all(type(v) is int and v >= 1 for v in (self.slots, self.prompt, self.outputs, self.lanes, self.every)) \
                or self.slots % self.lanes:
            raise ValueError(f"a window class's slots, prompt, outputs, lanes and every are positive integers, its lanes "
                             f"dividing its slots, not {self}")

    @property
    def positions(self) -> int:
        """A class Program's positions: the longest prompt's, then every output but the last."""
        return self.prompt + self.outputs - 1

    @property
    def boundaries(self) -> tuple[int, ...]:
        """The buckets its slots are anchored at, one a boundary."""
        return tuple(self.every * b for b in range(self.slots // self.lanes))

    @property
    def buckets(self) -> int:
        return self.every * len(self.boundaries)

    def anchor(self, slot: int) -> int:
        """Slot ``slot``'s anchor: the bucket its ingress row is the system's ingress at."""
        return self.boundaries[slot // self.lanes]


#: the class of record, the certifier run of record's envelope at ASK-122's slots (network-accounting took vllm's pick):
#: 8 slots a 32-bucket window, 4 at each of buckets 0 and 16 (EVERY 16, D 2.7 s), each of at most 1,024 prompt ids and 64
#: outputs
CLASS_OF_RECORD = WindowClass(8, 1024, 64, lanes=4, every=16)


@dataclass(frozen=True)
class Grain:
    """The units' size (CIR-95's at Llama-3.2-1B by default): a GEMM slice reads ``gemm_words`` weight words (N = gemm_words
    // K output columns: `Gemm_v2{2048, 16}`, `Gemm_v2{8192, 4}`), no unit holds more than ``rows`` rows in and out (C-Flock's
    composer gives each row a unit reads or writes an hm96 slot, so a unit fits its 2^27-bit block with at most 256: a gather
    of at most 255 parts), and a token select is chunks of ``select`` logits (768: 167 chunks at V = 128,256, each chunk's
    row past one SHA-512 block)."""
    gemm_words: int = 32768
    rows: int = 256
    select: int = 768


_GRAIN = Grain()


@dataclass(frozen=True, slots=True)
class Row:
    """One unit's output row: template ``tid``'s unit ``unit``, its output ``out`` of ``words`` u16 words."""
    tid: str
    unit: int
    out: str
    words: int


@dataclass(slots=True)
class _Feed:
    """Units ``[at, at + count)`` of template ``tid`` read port ``port`` as ``rows``, in order, at their (slot, step)
    ``label``: each unit the same rows, or under ``lead`` (digits ``(radix, step)`` of the reader units, slowest first, the
    radices' product ``count``) unit ``at + i`` each row's writer unit plus ``i``'s digits times their steps."""
    tid: str
    port: str
    at: int
    count: int
    rows: tuple[Row, ...]
    label: tuple[int, int]
    lead: tuple[tuple[int, int], ...] = ()


def _runs_of(rows: Sequence[Row]) -> list[tuple[int, int, int]]:
    """``rows`` as runs ``(first, length, unit step)``, each of one writer's output at a constant unit step."""
    out, j = [], 0
    while j < len(rows):
        k = j + 1
        while k < len(rows) and (rows[k].tid, rows[k].out) == (rows[j].tid, rows[j].out) and \
                rows[k].unit - rows[k - 1].unit == rows[j + 1].unit - rows[j].unit:
            k += 1
        out.append((j, k - j, rows[j + 1].unit - rows[j].unit if k > j + 1 else 0))
        j = k
    return out


def _step(rowss: Sequence[Sequence[Row]]) -> int | None:
    """The unit step σ with ``rowss[i]`` each of ``rowss[0]``'s rows σ·i units on, None where there is none."""
    r0 = rowss[0]
    s = rowss[1][0].unit - r0[0].unit if len(rowss) > 1 and rowss[1] else 0
    for i, rs in enumerate(rowss):
        if len(rs) != len(r0) or any((a.tid, a.out, a.words, a.unit + s * i) != (b.tid, b.out, b.words, b.unit)
                                     for a, b in zip(r0, rs)):
            return None
    return s


@dataclass
class _Builder:
    grain: Grain
    programs: int = 1
    fns: dict[str, Any] = field(default_factory=dict)
    units: dict[str, int] = field(default_factory=dict)
    local: dict[tuple[int, str], int] = field(default_factory=dict)
    feeds: list[_Feed] = field(default_factory=list)
    cuts: dict[str, dict[str, int]] = field(default_factory=dict)
    wires: list[IF.Wire] = field(default_factory=list)
    reads: list[IF.Read] = field(default_factory=list)
    tables: dict[tuple[int, int, int], IF.Table] = field(default_factory=dict)
    roots: list[IF.TableRead] = field(default_factory=list)
    openings: list[IF.Opening] = field(default_factory=list)
    ingress: list[IF.Ingress] = field(default_factory=list)
    gathered: dict[tuple[tuple[str, str, int, int, int], ...], Row] = field(default_factory=dict)
    #: the (slot, step) the next units are of, and each template's runs of units by theirs, ``(at, count, slot, step)``
    label: tuple[int, int] | None = None
    spans: dict[str, list[tuple[int, int, int, int]]] = field(default_factory=dict)
    #: each attention Definition's head plan (`_Program._plan`), read once whatever its slots and layers
    plans: dict[str, list[tuple[Any, list[tuple[str, tuple[Any, ...]]]]]] = field(default_factory=dict)

    def unit(self, fn: Any, count: int = 1) -> tuple[str, int]:
        """``count`` new units of ``fn`` at the current label: its descriptor id and the first one's index. Every slot makes
        the same allocations in the same order, so a slot's allocation of a template whose earlier ones took ``base`` of its
        units takes units ``base · programs + slot · count`` on: each allocation's slots sit side by side, and the slots'
        copies of a wire or read coalesce into one pattern."""
        tid = descriptor_id(fn)
        if self.label is None:
            raise Unexpressible(f"{tid}: a unit of no slot's step")
        slot = self.label[0]
        self.fns.setdefault(tid, fn)
        base = self.local.get((slot, tid), 0)
        self.local[(slot, tid)] = base + count
        self.units[tid] = max(self.units.get(tid, 0), (base + count) * self.programs)
        at = base * self.programs + slot * count
        self.spans.setdefault(tid, []).append((at, count, *self.label))
        return tid, at

    def feed(self, tid: str, port: str, at: int, count: int, rows: Sequence[Row],
             lead: tuple[tuple[int, int], ...] = ()) -> None:
        if not rows:
            raise Unexpressible(f"{tid} {port}: reads no unit's rows")
        if lead and math.prod(r for r, _ in lead) != count:
            raise ValueError(f"{tid} {port}: reader digits {lead} for {count} units")
        self.feeds.append(_Feed(tid, port, at, count, tuple(rows), self.label, lead))

    def _gatherer(self, rows: Sequence[Row]) -> Any:
        w = {r.words for r in rows}
        if len(w) != 1 or len(rows) >= self.grain.rows:
            raise Unexpressible(f"a gather of {len(rows)} rows of {sorted(w)} words: rows of one width, at most "
                                f"{self.grain.rows - 1} (with its output, a unit's {self.grain.rows} rows)")
        return bind(AllGather, WORLD=len(rows), N=w.pop())

    def gather(self, rows: Sequence[Row]) -> Row:
        """One row holding ``rows`` in order: the row itself, or an `AllGather_v1{len(rows), w}` unit's (one a row sequence,
        keyed by its runs)."""
        rows = tuple(rows)
        if len(rows) == 1:
            return rows[0]
        key = tuple((rows[j].tid, rows[j].out, rows[j].unit, n, s) for j, n, s in _runs_of(rows))
        if key in self.gathered:
            return self.gathered[key]
        tid, u = self.unit(self._gatherer(rows))
        self.cuts.setdefault(tid, {})["parts"] = len(rows)
        self._wire(tid, "parts", u, 1, rows)
        out = self.gathered[key] = Row(tid, u, "o0", len(rows) * rows[0].words)
        return out

    def gathers(self, rowss: Sequence[Sequence[Row]]) -> tuple[Row, ...]:
        """One row a group of ``rowss`` (a KV group's rows of a key or a block): a group's one row as it is, else one
        `AllGather_v1` unit a group, allocated together and wired as one pattern where each group's rows are the first's
        shifted (`_step`), never shared."""
        if all(len(rs) == 1 for rs in rowss):
            return tuple(rs[0] for rs in rowss)
        if (s := _step(rowss)) is None:
            return tuple(self.gather(rs) for rs in rowss)
        rows, n = rowss[0], len(rowss)
        tid, u = self.unit(self._gatherer(rows), n)
        self.cuts.setdefault(tid, {})["parts"] = len(rows)
        self._wire(tid, "parts", u, n, rows, ((n, s),))
        return tuple(Row(tid, u + i, "o0", len(rows) * rows[0].words) for i in range(n))

    def read(self, tid: str, port: str, at: int, radix: Sequence[int], base: int, step: Sequence[int]) -> None:
        self.reads.append(IF.Read(tid, port, WEIGHTS, at=at, radix=tuple(radix), base=base, step=tuple(step)))

    def _wire(self, tid: str, port: str, at: int, count: int, rows: Sequence[Row],
              lead: tuple[tuple[int, int], ...] = ()) -> None:
        """Units ``[at, at + count)`` read ``rows`` (each shifted by ``lead``'s digits, `_Feed`) as their port's rows
        ``0 .. len(rows) - 1``: one pattern per run of rows from one writer's output at a constant unit step."""
        lead_ = [d for d in (lead or ((count, 0),)) if d[0] > 1]
        for j, C, unit_step in _runs_of(rows):
            r = rows[j]
            digits = lead_ + ([(C, unit_step)] if C > 1 else []) or [(1, 0)]
            self.wires.append(IF.Wire(tid, port, r.tid, r.out, at=at, radix=tuple(a for a, _ in digits), base=r.unit,
                                      step=tuple(s for _, s in digits), cut=(j, C) if len(rows) > 1 else (0, 1)))

    def settle(self) -> None:
        """Each feed's port cut (`_cut`) within the rows its reader has left, its rows gathered to the cut's (at its reader's
        label), and wired; refused where a unit would hold more than ``grain.rows`` rows in and out."""
        uses: dict[tuple[str, str], dict[int, int]] = {}
        for f in self.feeds:
            m = uses.setdefault((f.tid, f.port), {})
            m[len(f.rows)] = m.get(len(f.rows), 0) + 1
        for (tid, port), ms in uses.items():
            cuts = self.cuts.get(tid, {})
            d = _cut((tid, port), ms, self.grain.rows - _rows(self.fns[tid], cuts) + cuts.get(port, 1), self.grain.rows)
            if d > 1:
                self.cuts.setdefault(tid, {})[port] = d
        for f in self.feeds:
            d = self.cuts.get(f.tid, {}).get(f.port, 1)
            g, self.label = len(f.rows) // d, f.label
            if g > 1 and any(s for _, s in f.lead):
                raise Unexpressible(f"{f.tid} {f.port}: units each reading their own {len(f.rows)} rows through a cut of {d}, "
                                    "a gather a unit")
            self._wire(f.tid, f.port, f.at, f.count, [self.gather(f.rows[i * g:(i + 1) * g]) for i in range(d)], f.lead)
        for tid, fn in self.fns.items():
            if (n := _rows(fn, self.cuts.get(tid))) > self.grain.rows:
                raise Unexpressible(f"{fn.id}: {n} rows in and out under its cuts {self.cuts.get(tid)}, past a unit's "
                                    f"{self.grain.rows}")


def _rows(fn: Any, cuts: dict[str, int] | None) -> int:
    """A unit of ``fn``'s rows in and out under ``cuts``, each an hm96 slot C-Flock's composer gives it
    (`interface.unit_rows`)."""
    lay = IF.layout(fn)
    return len(lay.cut(cuts)) + len(lay.outs)


def _field(fn: Any, lo: int, hi: int) -> tuple[str, int]:
    """The output row of ``fn`` (its name and words) whose return leaves are exactly [lo, hi), refused where none is."""
    ret = fn.signature()[1]
    starts = [ret.field_offset(k) for k in range(len(ret.fields))] if isinstance(ret, Tuple) else [0]
    starts.append(ret.leaves)
    k = starts.index(lo) if lo in starts[:-1] else None
    if k is None or starts[k + 1] != hi:
        raise Unexpressible(f"{fn.id}: its return leaves [{lo}, {hi}) are not one of its output rows")
    return IF.layout(fn).outs[k]


def _cut(key: tuple[str, str], uses: dict[int, int], room: int, most: int) -> int:
    """The cut of a port its feeds read as ``uses`` (row count -> feeds): a divisor ``d`` of every count, at most ``room`` (the
    rows its reader has left) and leaving at most ``most - 1`` rows a gather (its output row is its last of ``most``), with
    the fewest gather units (``d`` for each feed of more than ``d`` rows)."""
    g = math.gcd(*uses)
    ok = [d for d in range(1, min(g, room) + 1) if g % d == 0 and max(uses) // d < most]
    if not ok:
        raise Unexpressible(f"{key[0]} {key[1]}: read as {sorted(uses)} rows, which no cut of at most {room} rows a port and "
                            f"{most - 1} a gather divides")
    return min(ok, key=lambda d: (sum(n * d for m, n in uses.items() if m > d), -d))


@dataclass
class _Keys:
    """A layer's keys (or values) as its attention nodes read them: ``segs`` the word runs its positions so far are, ``rows``
    each position's row a KV group, and ``blocks`` each full key block's gather a group."""
    segs: list[tuple[str, Any, int, int]] = field(default_factory=list)
    rows: list[tuple[Row, ...]] = field(default_factory=list)
    blocks: dict[int, tuple[Row, ...]] = field(default_factory=dict)


class _Program:
    """Slot ``slot``'s class Program lowered into ``b``'s units, its ingress row the system's ingress at the slot's anchor
    (`WindowClass.anchor`): each node's output as the rows that hold it, ``(lo, hi, row)`` its words [lo, hi) as ``row`` and
    the units after it, (hi - lo) / ``row.words`` rows of one output at consecutive units."""

    def __init__(self, b: _Builder, fn: Any, cls: WindowClass, name: str, slot: int) -> None:
        self.b, self.fn, self.cls, self.name, self.slot_index = b, fn, cls, name, slot
        self.anchor = cls.anchor(slot)
        params = [name for name, _ in self.fn.signature()[0]]
        if set(params) != {PROMPT, WEIGHTS}:
            raise Unexpressible(f"{name}: its inputs are {params}; the window's lowering takes a prompt and the weights record "
                                "(no seed, prefix-cache hit, carried state or peers)")
        self.params = params
        self.fields: list[tuple[int, int, int]] = []
        leaf = word = 0
        for field_name, ty in dict(self.fn.signature()[0])[WEIGHTS].fields:
            ws = ty.width_set()
            if len(ws) != 1 or next(iter(ws)) % 16:
                raise Unexpressible(f"{name}: weight {field_name}'s leaves are {sorted(ws)} bits, not one width of whole u16 words")
            self.fields.append((leaf, word, next(iter(ws)) // 16))
            leaf, word = leaf + ty.leaves, word + ty.leaves * next(iter(ws)) // 16
        self.fields.append((leaf, word, 0))
        self.weights = word
        self.outs: dict[int, list[tuple[int, int, Row]]] = {}
        self.words: dict[int, int] = {}
        self.gemms: dict[int, tuple[Any, tuple[Row, ...], int]] = {}
        self.keys: dict[tuple[str, tuple[str, Any, int, int]], _Keys] = {}
        self.position = 0
        self.slot: Row | None = None
        self.occ: Row | None = None

    def segments(self, refs: Refs) -> list[tuple[str, Any, int, int]]:
        """``refs`` as word ranges: ``("n", node, lo, hi)`` of a node's output or ``("p", name, lo, hi)`` of a parameter."""
        out = []
        for space, idx, lo, hi in runs(refs):
            if space == "n":
                w = self.words[idx]
                out.append(("n", idx, lo * w, hi * w))
            elif self.params[idx] == WEIGHTS:
                k = bisect.bisect_right([f[0] for f in self.fields], lo) - 1
                leaf, word, w = self.fields[k]
                if hi > self.fields[k + 1][0]:
                    raise Unexpressible(f"{self.name}: a weight read of leaves [{lo}, {hi}) spans two weights")
                out.append(("p", WEIGHTS, word + (lo - leaf) * w, word + (hi - leaf) * w))
            else:
                out.append(("p", self.params[idx], lo, hi))
        return out

    def rows(self, refs: Refs) -> tuple[Row, ...]:
        """The writer rows ``refs`` read, whole and in order; refused where they read part of a row or a parameter."""
        out: list[Row] = []
        for space, j, lo, hi in self.segments(refs):
            if space != "n":
                raise Unexpressible(f"{self.name}: a unit reads parameter {j} where a wire is due")
            for a, b, r in self.outs[j]:
                if b <= lo or a >= hi:
                    continue
                w = r.words
                i0, i1 = (max(a, lo) - a) // w, -(-(min(b, hi) - a) // w)
                if a + i0 * w < lo or a + i1 * w > hi:
                    k = i0 if a + i0 * w < lo else i1 - 1
                    raise Unexpressible(f"{self.name}: node {j}'s words [{lo}, {hi}) cut its row "
                                        f"{Row(r.tid, r.unit + k, r.out, w)} at [{a + k * w}, {a + (k + 1) * w})")
                out += [r] if b - a == w else [Row(r.tid, r.unit + i, r.out, w) for i in range(i0, i1)]
        return tuple(out)

    def _split(self, refs: Refs, n: int) -> list[tuple[Row, ...]]:
        """The rows each of ``refs``'s ``n`` equal word ranges reads (a node's heads or KV groups), refused where a row
        crosses from one range into the next."""
        rows = self.rows(refs)
        w, rest = divmod(sum(r.words for r in rows), n)
        out: list[tuple[Row, ...]] = []
        cur: list[Row] = []
        at = 0
        for r in rows:
            if rest or at % w + r.words > w:
                raise Unexpressible(f"{self.name}: row {r} crosses from one of {n} ranges of {w} words into the next")
            cur.append(r)
            at += r.words
            if at % w == 0:
                out.append(tuple(cur))
                cur = []
        return out

    def weight(self, refs: Refs, words: int) -> int:
        """The first word of the one run of the weights record ``refs`` read, ``words`` long."""
        segs = self.segments(refs)
        if len(segs) != 1 or segs[0][:2] != ("p", WEIGHTS) or segs[0][3] - segs[0][2] != words:
            raise Unexpressible(f"{self.name}: a weight read of {segs}, not one run of {words} weight words")
        return segs[0][2]

    def lower(self, j: int, node: Any) -> None:
        """Node ``j`` lowered into units, each of the Program's slot and the node's step: its position, which each
        embedding opens (`_token`)."""
        if node.form != "call":
            raise Unexpressible(f"{self.name}: node {j} is a {node.form}, not a call")
        ret = node.fn.signature()[1]
        ws = ret.width_set()
        if not ws <= {16, 32} or len(ws) != 1:
            raise Unexpressible(f"{self.name}: node {j} ({node.fn.id}) returns leaves of {sorted(ws)} bits")
        self.words[j] = next(iter(ws)) // 16
        family = node.fn.name
        lower = getattr(self, f"_{family}", None) if family in FAMILIES else None
        if lower is None:
            raise Unexpressible(f"{self.name}: node {j} is {node.fn.id}, a family the window's lowering "
                                f"doesn't take ({sorted(FAMILIES)})")
        step = self.position - (family != "Embedding")
        if step < 0:
            raise Unexpressible(f"{self.name}: node {j} ({node.fn.id}) comes before its first position's embedding")
        self.b.label = (self.slot_index, step)
        lower(j, node)

    def _Embedding(self, j: int, node: Any) -> None:
        V, H = node.fn.bindings["V"], node.fn.bindings["H"]
        tok, table = node.args
        tid, u = self.b.unit(E.read(V, H))
        self.b.feed(tid, "tok", u, 1, self._token(tok))
        lo = self.weight(table.refs, V * H)
        t = self.b.tables.setdefault((lo, H, V), IF.Table(f"embedding@{lo}", WEIGHTS, lo, H, V))
        self.b.roots += [IF.TableRead(tid, "root", t.name, "root", u, 1), IF.TableRead(tid, "domain", t.name, "domain", u, 1)]
        self.b.openings.append(IF.Opening(tid, t.name, "tok", ("row", "salt", "path"), "o1", u, 1))
        self.outs[j] = [(0, H, Row(tid, u, "o0", H))]

    def _token(self, tok: Any) -> tuple[Row, ...]:
        """The next position's token: its `TokenFeed_v1`'s below the class's prompt cap (the first's slot its slot's ingress row
        opened by `SLOT_ROW`, every other's its predecessor's, ``sel`` the previous position's select), past it the previous
        position's select."""
        p, NP = self.position, self.cls.prompt
        self.position += 1
        segs = self.segments(tok.refs)
        if (segs[0][:2] == ("p", PROMPT)) != (p == 0):
            raise Unexpressible(f"{self.name}: position {p}'s token reads {segs[0][:2]}; a class Program reads its one prompt row "
                                "at position 0 and every other position's token from its predecessor's select")
        if p >= NP:
            return self.rows(tok.refs)
        tid, u = self.b.unit(bind(IO.TokenFeed, NP=NP, FIRST=p == 0))
        if p == 0:
            rtid, ru = self.b.unit(bind(SLOT_ROW, NP=NP))
            self.b.ingress.append(IF.Ingress(rtid, "row", ru, 1, self.anchor))
            self.b.feed(tid, "slot", u, 1, (Row(rtid, ru, "o1", 2 * (NP + 1)),))
            self.occ = Row(rtid, ru, "o0", 2)
        else:
            self.b.feed(tid, "slot", u, 1, (self.slot,))
            self.b.feed(tid, "sel", u, 1, self.rows(tok.refs))
        self.slot = Row(tid, u, "o1", 2 * (NP + 1))
        return (Row(tid, u, "o0", 2),)

    def _one(self, j: int, node: Any) -> None:
        """A node that is one unit of its own Definition: each parameter a weight read or its writers' rows."""
        tid, u = self.b.unit(node.fn)
        lay = IF.layout(node.fn)
        for (p, words), arg in zip(lay.rows, node.args):
            segs = self.segments(arg.refs)
            if segs[0][:2] == ("p", WEIGHTS):
                self.b.read(tid, p, u, (1,), self.weight(arg.refs, words), (0,))
            else:
                self.b.feed(tid, p, u, 1, self.rows(arg.refs))
        at, pieces = 0, []
        for o, words in lay.outs:
            pieces.append((at, at + words, Row(tid, u, o, words)))
            at += words
        self.outs[j] = pieces

    _RMSNormTriton = _RMSNormFusedCuda = _SiluMul = _one

    def _Gemm(self, j: int, node: Any) -> None:
        b = node.fn.bindings
        K, N = b["K"], b["N"]
        n = max(1, min(N, self.b.grain.gemm_words // K))
        x, w = node.args
        xs, lo = self.rows(x.refs), self.weight(w.refs, N * K)
        full, rest = divmod(N, n)
        pieces = []
        for s0, m, count in ((0, n, full), (full * n, rest, int(rest > 0))):
            if not count:
                continue
            tid, u = self.b.unit(bind(node.fn.defn, **{**b, "N": m}), count)
            self.b.feed(tid, "x", u, count, xs)
            self.b.read(tid, "w", u, (count,), lo + s0 * K, (m * K,))
            pieces.append((s0, s0 + count * m, Row(tid, u, "o0", m)))
        self.outs[j] = pieces
        self.gemms[j] = (node.fn, xs, lo)

    def _heads(self, fn: Any, family: str, count: int) -> Any:
        nodes = fn.body.nodes
        head = nodes[0].fn
        if head.name != family or any(m.form != "batch" or descriptor_id(m.fn) != descriptor_id(head) for m in nodes) \
                or sum(m.n for m in nodes) != count:
            raise Unexpressible(f"{fn.id}: its body is not {count} heads of one {family} Definition")
        return head

    def _each(self, tid: str, port: str, at: int, rowss: Sequence[tuple[Row, ...]], rep: int = 1) -> None:
        """Units ``at + g·rep`` to ``at + (g + 1)·rep - 1`` read ``rowss[g]``: one feed where each group's rows are the first's
        shifted (`_step`), else a feed a group."""
        if (s := _step(rowss)) is None:
            for g, rows in enumerate(rowss):
                self.b.feed(tid, port, at + g * rep, rep, rows)
        else:
            self.b.feed(tid, port, at, len(rowss) * rep, rowss[0], ((len(rowss), s), (rep, 0)))

    def _RoPE(self, j: int, node: Any) -> None:
        H, D = node.fn.bindings["NHEADS"], node.fn.bindings["D"]
        head = self._heads(node.fn, "RoPEHead", H)
        x, cs = node.args
        tid, u = self.b.unit(head, H)
        self._each(tid, "x", u, self._split(x.refs, H))
        self.b.read(tid, "cs", u, (H,), self.weight(cs.refs, D), (0,))
        self.outs[j] = [(0, H * D, Row(tid, u, "o0", D))]

    def _plan(self, fn: Any, NH: int) -> list[tuple[Any, list[tuple[str, tuple[str, tuple[Any, ...]]]]]]:
        """``fn``'s heads as `Attention_v19`'s, read once a Definition (`_Builder.plans`): each Call of `AttentionHead_v19`'s
        body in order with what each of its ports reads, ``("q", ())``, ``("k" | "v", (a, b))`` (keys [a, b)) or
        ``("state", (i, out, words))`` (Call i's output row ``out``), the last Call's ``o0`` the head's output."""
        key = descriptor_id(fn)
        if key in self.b.plans:
            return self.b.plans[key]
        try:
            head = self._heads(bind(AttentionV19, **fn.bindings), "AttentionHead", NH)
            body = head.body
        except (TypeError, ValueError, KeyError) as e:
            raise Unexpressible(f"{fn.id}: its heads are not `AttentionHead_v19`'s ({e})") from None
        params, D = [p for p, _ in head.signature()[0]], fn.bindings["D"]
        plan = []
        for i, n in enumerate(body.nodes):
            ports = []
            for (port, _), a in zip(IF.layout(n.fn).rows, n.args):
                (space, idx, lo, hi), *more = runs(a.refs)
                name = params[idx] if space == "p" else None
                if more or n.form != "call":
                    src = None
                elif name == "q" and (lo, hi) == (0, D):
                    src = ("q", ())
                elif name in ("k", "v") and lo % D == hi % D == 0:
                    src = (name, (lo // D, hi // D))
                elif space == "n" and 0 <= idx < i:
                    src = ("state", (idx, *_field(body.nodes[idx].fn, lo, hi)))
                else:
                    src = None
                if src is None:
                    raise Unexpressible(f"{head.id}: Call {i} ({n.fn.id}) reads {runs(a.refs)} as {port}, not a head's query, "
                                        "a key range or an earlier Call's output row")
                ports.append((port, src))
            plan.append((n.fn, ports))
        last = body.nodes[-1].fn
        if runs(body.ret) != [("n", len(body.nodes) - 1, 0, last.signature()[1].leaves)] or \
                IF.layout(last).outs != (("o0", D),):
            raise Unexpressible(f"{head.id}: its output is not its last Call's one row of {D} words")
        self.b.plans[key] = plan
        return plan

    def _cache(self, port: str, c: Any, T: int, KVH: int) -> _Keys:
        """The layer's ``port`` (keys or values) at T keys: its earlier nodes' T - 1 and position T - 1's row a KV group
        (`_Builder.gathers`), refused unless ``c`` is those T - 1 and one more."""
        segs = self.segments(c.refs)
        ks = self.keys.setdefault((port, segs[0]), _Keys())
        if len(ks.rows) != T - 1 or segs[:len(ks.segs)] != ks.segs:
            raise Unexpressible(f"{self.name}: an attention's {port} of {T} keys is not its layer's {len(ks.rows)} earlier keys "
                                "and one more")
        n = c.refs.count // T
        ks.rows.append(self.b.gathers(self._split(c.refs.slice((T - 1) * n, T * n), KVH)))
        ks.segs = segs
        return ks

    def _block(self, ks: _Keys, a: int, b: int, BN: int) -> tuple[Row, ...]:
        """Keys [a, b)'s row a KV group: one key's own, a full block's gathers (made when it fills, then every later
        position's) or the first block's own."""
        if b - a == 1:
            return ks.rows[a]
        full = b - a == BN and a % BN == 0
        if full and a // BN in ks.blocks:
            return ks.blocks[a // BN]
        rows = self.b.gathers([tuple(r[g] for r in ks.rows[a:b]) for g in range(len(ks.rows[a]))])
        if full:
            ks.blocks[a // BN] = rows
        return rows

    def _Attention(self, j: int, node: Any) -> None:
        s = node.fn.bindings
        if node.fn.version not in (18, 19):
            raise Unexpressible(f"{node.fn.id}: the window's lowering cuts `Attention_v18`'s heads as v19's, not "
                                f"v{node.fn.version}'s")
        T, NH, KVH, D, BN = s["T"], s["NH"], s["KVH"], s["D"], s["BN"]
        plan = self._plan(node.fn, NH)
        q, kc, vc = node.args
        keys = {"k": self._cache("k", kc, T, KVH), "v": self._cache("v", vc, T, KVH)}
        qs = self._split(q.refs, NH)
        calls: list[tuple[str, int]] = []
        for fn, ports in plan:
            tid, u = self.b.unit(fn, NH)
            for port, (kind, x) in ports:
                if kind == "q":
                    self._each(tid, port, u, qs)
                elif kind == "state":
                    ctid, cu = calls[x[0]]
                    self.b.feed(tid, port, u, NH, (Row(ctid, cu, x[1], x[2]),), ((NH, 1),))
                else:
                    self._each(tid, port, u, [(r,) for r in self._block(keys[kind], *x, BN)], NH // KVH)
            calls.append((tid, u))
        self.outs[j] = [(0, NH * D, Row(tid, u, "o0", D))]

    def _TokenSelect(self, j: int, node: Any) -> None:
        V = node.fn.bindings["V"]
        nodes = node.fn.body.nodes
        chunk, combine = nodes[0].fn, nodes[-1].fn
        if chunk.name != "TokenSelectChunk" or combine.name != "TokenSelectCombine" or V < 2:
            raise Unexpressible(f"{node.fn.id}: its body is not chunks and one combine")
        C = min(self.b.grain.select, V)
        (logits,) = node.args
        parts = []
        for lo in range(0, V, C):
            tid, u = self.b.unit(bind(chunk.defn, **{**chunk.bindings, "C": min(C, V - lo)}))
            self.b.feed(tid, "x", u, 1, (self.b.gather(self.rows(logits.refs.slice(lo, min(lo + C, V)))),))
            parts.append((tid, u))
        ctid, cu = self.b.unit(bind(combine.defn, **{**combine.bindings, "V": V, "C": C}))
        self.b.feed(ctid, "best", cu, 1, (self.b.gather([Row(t, u, "o0", 1) for t, u in parts]),))
        self.b.feed(ctid, "idx", cu, 1, [Row(t, u, "o1", 2) for t, u in parts])
        (_, lm, lo, _), = self.segments(logits.refs.slice(0, 1))
        if lm not in self.gemms:
            raise Unexpressible(f"{node.fn.id}: its logits are node {lm}'s, not a GEMM's")
        fn, xs, wlo = self.gemms[lm]
        xtid, xu = self.b.unit(bind(fn.defn, **{**fn.bindings, "N": 1}))
        self.b.feed(xtid, "x", xu, 1, xs)
        self.b.read(xtid, "w", xu, (1,), wlo + lo * fn.bindings["K"], (0,))
        self.b.feed(ctid, "x0", cu, 1, (Row(xtid, xu, "o0", 1),))
        self.outs[j] = [(0, 2, Row(ctid, cu, "o0", 2))]
        if self.occ is None:
            raise Unexpressible(f"{node.fn.id}: a token select before its Program's first position's token")
        mtid, mu = self.b.unit(bind(IO.SlotMask))
        self.b.feed(mtid, "occ", mu, 1, (self.occ,))
        self.b.feed(mtid, "tok", mu, 1, self.outs[j][0][2:])


FAMILIES = ("Embedding", "RMSNormTriton", "RMSNormFusedCuda", "Gemm", "RoPE", "Attention", "SiluMul", "TokenSelect")


def _runs(xs: Sequence[Any], key: Any, at: Any, n: Any, make: Any) -> list[Any]:
    """``xs`` with each run of like entries (equal ``key``, consecutive units) merged by ``make(first, units)``."""
    out: list[Any] = []
    for x in sorted(xs, key=lambda x: (key(x), at(x))):
        if out and key(out[-1]) == key(x) and at(out[-1]) + n(out[-1]) == at(x):
            out[-1] = make(out[-1], n(out[-1]) + n(x))
        else:
            out.append(x)
    return out


def class_program(registered: Any, cls: WindowClass) -> Any:
    """The class's request Program (a `build.BuildResult`): the Build's greedy request at one prompt row and
    ``cls.positions`` - 1 decode steps, in the registered compiler's form."""
    return request_program(registered, RequestUnit(-1, "", 1, cls.positions - 1, GREEDY, None, 0))


def _follow(groups: list[tuple[int, tuple[tuple[int, str], ...], int]], axis: str,
            delta: int) -> list[tuple[int, tuple[tuple[int, str], ...], int]]:
    """``groups`` (first unit, digits as ``(radix, axis)``, step), consecutive, with each run of like ones (one digits, their
    steps ``delta`` apart) as one whose first digit is ``axis``, of the run's length."""
    out, i = [], 0
    while i < len(groups):
        at, digits, step = groups[i]
        j = i + 1
        while j < len(groups) and groups[j][1] == digits and groups[j][2] == step + (j - i) * delta:
            j += 1
        out.append((at, (((j - i, axis),) if j > i + 1 else ()) + digits, step))
        i = j
    return out


def _inference(spans: dict[str, list[tuple[int, int, int, int]]], slots: int) -> list[IF.InferenceEntry]:
    """Each template's units as inference entries (`interface.InferenceEntry`): its runs in unit order, an allocation's
    slots' runs of one length at one step side by side, are one entry (the slot, then the run), and like entries that follow
    on merge, at one step under a ``-`` digit, then at consecutive steps under a ``step`` digit. Refused where a template's
    units aren't such runs."""
    out = []
    for tid, spans_ in spans.items():
        merged: list[tuple[int, int, int, int]] = []
        for at, n, slot, step in sorted(spans_):
            if merged and merged[-1][2:] == (slot, step):
                merged[-1] = (merged[-1][0], merged[-1][1] + n, slot, step)
            else:
                merged.append((at, n, slot, step))
        groups = []
        for i in range(0, len(merged), slots):
            g = merged[i:i + slots]
            if [r[2] for r in g] != list(range(slots)) or len({(r[1], r[3]) for r in g}) != 1:
                raise Unexpressible(f"{tid}: units from {g[0][0]} are not {slots} slots' runs of one length at one step")
            groups.append((g[0][0], ((slots, "slot"),) + (((g[0][1], "-"),) if g[0][1] > 1 else ()), g[0][3]))
        for axis, delta in (("-", 0), ("step", 1)):
            groups = _follow(groups, axis, delta)
        out += [IF.InferenceEntry(tid, at, tuple(r for r, _ in d), tuple(a for _, a in d), step) for at, d, step in groups]
    return out


def class_circuit(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN,
                  name: str = NAME) -> IF.Circuit:
    """``cls``'s circuit (module docstring): the class Program lowered once a slot, node by node, slot k's ingress row at its
    anchor (`WindowClass.anchor`) and each allocation's slots side by side in its template; its blocks' slots are the slots.
    A window's circuit is every slot's; ``programs`` is the first slots only, a probe's census (`class_size`)."""
    programs = cls.slots if programs is None else programs
    if not 1 <= programs <= cls.slots:
        raise Unexpressible(f"{programs} Programs in a window of a class of {cls.slots} slots")
    b, fn = _Builder(grain, programs), class_program(registered, cls).program.fn
    progs = [_Program(b, fn, cls, f"class Program {k}", k) for k in range(programs)]
    for j, node in enumerate(fn.body.nodes):
        for r in progs:
            r.lower(j, node)
    for r in progs:
        if r.position != cls.positions:
            raise Unexpressible(f"{r.name}: {r.position} positions, not its class's {cls.positions}")
    weights = progs[0].weights
    b.settle()
    roots = _runs(b.roots, lambda x: (x.reader, x.port, x.table, x.of), lambda x: x.at, lambda x: x.units,
                  lambda x, n: IF.TableRead(x.reader, x.port, x.table, x.of, x.at, n))
    openings = _runs(b.openings, lambda x: (x.reader, x.table, x.key, x.ports, x.verdict), lambda x: x.at, lambda x: x.units,
                     lambda x, n: IF.Opening(x.reader, x.table, x.key, x.ports, x.verdict, x.at, n))
    ingress = _runs(b.ingress, lambda x: (x.reader, x.port, x.step), lambda x: x.at, lambda x: x.units,
                    lambda x, n: IF.Ingress(x.reader, x.port, x.at, n, x.step))
    reg = IF.Registered({WEIGHTS: weights}, tuple(b.tables.values()), tuple(IF.coalesce(b.reads)), tuple(roots),
                        tuple(openings))
    rows = {g.reader for g in b.ingress}
    masks = {f.tid for f in b.feeds if f.port == "occ"}
    return IF.Circuit(name, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()),
                      external=tuple(sorted({f"{t}/row" for t in rows} | {f"{t}/o0" for t in masks})),
                      wires=tuple(IF.coalesce(b.wires)), cuts=b.cuts, registered=reg, ingress=tuple(ingress),
                      blocks=IF.Blocks(programs, cls.positions, tuple(_inference(b.spans, programs))))


def window_circuit(registered: Any, compiled: Any, cls: WindowClass, grain: Grain = _GRAIN, name: str = NAME,
                   placed: Sequence[tuple[int, int | None]] | None = None) -> IF.Circuit:
    """``compiled``'s window (`steps.Compiled`, of ``registered``) as ``cls``'s circuit, every slot's class Program whatever
    its traffic, refused where the class's slots don't take the window: nothing of the window reaches the circuit.
    ``placed`` is each request's anchor bucket and the bucket of its last output in the window, in its request order, as the
    network certifier's scan placed it (the end None: unchecked); by default every request is at the window's first
    boundary, its end unchecked."""
    units = [u for u in compiled.requests if u is not None]
    if len(units) > cls.slots:
        raise Unexpressible(f"{len(units)} requests in a window of a class of {cls.slots} slots")
    placed = [(cls.boundaries[0], None)] * len(units) if placed is None else list(placed)
    if len(placed) != len(units):
        raise ValueError(f"{len(placed)} placements for the window's {len(units)} requests")
    anchored: dict[int, int] = {}
    for u, (anchor, end) in zip(units, placed):
        if anchor not in cls.boundaries:
            raise Unexpressible(f"request {u.request}: anchored at bucket {anchor}, none of its class's boundaries "
                                f"{list(cls.boundaries)}")
        if end is not None and end < anchor:
            raise Unexpressible(f"request {u.request}: its last output at bucket {end}, before its anchor {anchor}")
        if end is not None and end >= cls.buckets:
            raise Unexpressible(f"request {u.request}: anchored at bucket {anchor}, its last output at bucket {end}, past the "
                                f"window's {cls.buckets} buckets: {ACROSS}")
        anchored[anchor] = anchored.get(anchor, 0) + 1
    for anchor, n in sorted(anchored.items()):
        if n > cls.lanes:
            raise Unexpressible(f"{n} requests anchored at bucket {anchor}, past the {cls.lanes} slots a boundary has")
    for u in units:
        if u.carried_in or u.carried_out or u.C or u.parse is not None:
            raise Unexpressible(f"request {u.request}: carried across windows, a prefix-cache hit or byte ingress, which the "
                                "window's lowering doesn't take yet")
        if u.policy != GREEDY:
            raise Unexpressible(f"request {u.request}: a {u.policy.get('kind')} request, and the class Program is greedy (a "
                                "Program that reads a seed isn't lowered yet)")
        if u.LP > cls.prompt or u.T + 1 > cls.outputs:
            raise Unexpressible(f"request {u.request}: {u.LP} prompt ids and {u.T + 1} outputs, past its class's {cls.prompt} and "
                                f"{cls.outputs}")
    return class_circuit(registered, cls, None, grain, name)


__all__ = ["ACROSS", "CLASS_OF_RECORD", "FAMILIES", "Grain", "NAME", "Row", "SLOT_ROW", "WindowClass", "class_circuit",
           "class_program", "window_circuit"]
