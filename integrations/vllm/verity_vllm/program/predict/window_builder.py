"""The window circuit's units as `window_circuit` lowers a class Program into them: each template's units, allocated a slot
at a time with each allocation's slots side by side (`_Builder.unit`), the rows each unit's ports read (`_Feed`), the
gathers that make several writers' rows one (`AllGather_v1`, made once a run of rows), the weight reads, and `settle`:
each port's cut within its unit's rows and every feed wired as `interface.Wire` patterns, one a run of rows from one
writer's output at a constant unit step. A unit holds at most ``Grain.rows`` rows in and out, a gather's parts and its
output among them.
"""
from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry.b1_tp2 import AllGather
from verity.ml.windows import interface as IF

from .build import Unexpressible

WEIGHTS = "weights"


def weight_param(label: str, words: int) -> str:
    """The registered parameter of weight ``label`` (its name in the weights record) read in rows of ``words`` u16 words:
    one registration a weight and row width, so every read's base and steps are whole rows of it (`interface._reads_of`)
    and its pin is a root over the weight's own rows (`interface.registered_reads`). A pin of `WEIGHTS` pins them all
    (`interface.input_count`)."""
    return f"{WEIGHTS}/{label}@{words}"


@dataclass(frozen=True)
class Grain:
    """The units' size (CIR-95's at Llama-3.2-1B by default): a GEMM slice reads ``gemm_words`` weight words (N = gemm_words
    // K output columns: `Gemm_v2{2048, 16}`, `Gemm_v2{8192, 4}`; an s8 one twice the columns: `ScaledMmS8_v1{2048, 32}`,
    `ScaledMmS8_v1{8192, 8}`), no unit holds more than ``rows`` rows in and out (C-Flock's
    composer gives each row a unit reads or writes an hm96 slot, so a unit fits its 2^27-bit block with at most 256: a gather
    of at most 255 parts), and a token select is chunks of ``select`` logits (768: 167 chunks at V = 128,256, each chunk's
    row past one SHA-512 block)."""
    gemm_words: int = 32768
    rows: int = 256
    select: int = 768


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
    gathered: dict[tuple[tuple[str, str, int, int, int], ...], Row] = field(default_factory=dict)
    #: the (slot, step) the next units are of, and each template's runs of units by theirs, ``(at, count, slot, step)``
    label: tuple[int, int] | None = None
    spans: dict[str, list[tuple[int, int, int, int]]] = field(default_factory=dict)
    #: each attention Definition's head plan (`window_circuit._Program._plan`), read once whatever its slots and layers
    plans: dict[str, list[tuple[Any, list[tuple[str, tuple[Any, ...]]]]]] = field(default_factory=dict)
    #: the weights record's weights, ``(label, first word, words)`` in order, and the registered parameters read of them
    weights: tuple[tuple[str, int, int], ...] = ()
    params: dict[str, int] = field(default_factory=dict)

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

    def weight(self, lo: int, hi: int, words: int) -> tuple[str, int]:
        """The registered parameter of the one weight words [lo, hi) of the weights record lie in, read in rows of ``words``
        (`weight_param`), and ``lo`` within it; refused where they span two weights."""
        k = bisect.bisect_right([w[1] for w in self.weights], lo) - 1
        if k < 0 or hi > self.weights[k][1] + self.weights[k][2]:
            raise Unexpressible(f"a weight read of words [{lo}, {hi}) of the weights record spans two weights")
        label, first, n = self.weights[k]
        if n % words or (lo - first) % words:
            raise Unexpressible(f"weight {label} of {n} words read from word {lo - first} in rows of {words}: its registration "
                                "is whole rows of the port, every read on a row")
        name = weight_param(label, words)
        self.params[name] = n
        return name, lo - first

    def read(self, tid: str, port: str, at: int, radix: Sequence[int], base: int, step: Sequence[int]) -> None:
        """Units ``[at, at + Π radix)`` of ``tid`` read ``port`` from word ``base`` of the weights record on (``step`` words a
        digit), as rows of their weight's registered parameter at the port's width (`weight`)."""
        lay = IF.layout(self.fns[tid])
        words = dict(lay.rows + lay.outs)[port]
        param, lo = self.weight(base, base + words + sum((r - 1) * s for r, s in zip(radix, step)), words)
        self.reads.append(IF.Read(tid, port, param, at=at, radix=tuple(radix), base=lo, step=tuple(step)))

    def value(self, tid: str, port: str, at: int, param: str, row: int, rows: int) -> None:
        """Unit ``at`` of ``tid`` reads ``port`` as row ``row`` of the registered parameter ``param``, ``rows`` rows of the
        port's width (a per-window value, not the weights record)."""
        lay = IF.layout(self.fns[tid])
        words = dict(lay.rows)[port]
        if self.params.setdefault(param, rows * words) != rows * words:
            raise Unexpressible(f"{param}: read as {rows} rows of {words} words and as {self.params[param]} leaves")
        self.reads.append(IF.Read(tid, port, param, at=at, radix=(1,), base=row * words, step=(0,)))

    def table(self, lo: int, words: int, rows: int) -> IF.Table:
        """The registered table of ``rows`` rows of ``words`` from word ``lo`` of the weights record (an embedding's), one a
        weight and width."""
        param, off = self.weight(lo, lo + words * rows, words)
        return self.tables.setdefault((lo, words, rows), IF.Table(f"table/{param}+{off}", param, off, words, rows))

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
