"""The window circuit's units as `window_circuit` lowers a class Program into them: each template's units, allocated a slot
at a time with each allocation's slots side by side (`_Builder.unit`) and timed by their class Program node and their place
among the node's units that read one another (`_Builder.timed`), the rows each unit's ports read (`_Feed`), the registered
reads (the weights' and the scan state's), and `settle`: each port read as m writer rows cut m ways, a row of its reader a
writer row, and every feed wired as `interface.Wire` patterns, one a run of rows from one writer's output at a constant
unit step. No unit copies rows together: a reader reads each writer's row as it is. A unit holds at most ``Grain.rows``
rows in and out. Each template's units are the circuit's inference entries by their (slot, step) and time (`entries`),
each wire is placed inside one writer entry and one reader entry (`placed`), and the entries ranked by their units' time
are the circuit's wire-order certificate (`order`): every writer unit ranks below its reader unit.
"""
from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import descriptor_id
from verity.ml.windows import interface as IF
from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.weights_of_record import weight_param

from .build import Unexpressible

WEIGHTS = WR.PARAM


def table_param(label: str, words: int) -> str:
    """The registered parameter of weight ``label`` as a table of rows of ``words`` (an embedding's, `_Builder.table`):
    ``weight_param(label, words)/table``, a registration of its own, since a table's rows are ``hm96-sha512/row-seg/v1``
    leaves its opening reads in-circuit (`gates.embedding`) and a read's rows are ``hm96-sha512/row/v2`` leaves its unit
    takes by reference, so a tied embedding read at the table's width (the token select's ``x0``) is the other one."""
    return f"{weight_param(label, words)}/table"


@dataclass(frozen=True)
class Grain:
    """The units' size (Llama-3.2-1B's by default): a GEMM slice reads ``gemm_words`` weight words (N = gemm_words
    // K output columns: `Gemm_v2{2048, 16}`, `Gemm_v2{8192, 4}`; an s8 one twice the columns: `ScaledMmS8_v1{2048, 32}`,
    `ScaledMmS8_v1{8192, 8}`), no unit holds more than ``rows`` rows in and out (C-Flock's
    composer gives each row a unit reads or writes an hm96 slot, so a unit fits its 2^27-bit block with at most 256), a
    token select is units of ``select`` logits (2,672: 48 units at V = 128,256, each 167 groups of 16), and a fused norm
    cut at its residual add is parts of ``norm`` elements (256: 8 parts at N = 2,048, each x 64 rows of a `Gemm_v2{8192, 4}`)."""
    gemm_words: int = 32768
    rows: int = 256
    select: int = 2672
    norm: int = 256


def _spans(what: str, fields: Sequence[tuple[str, Any]]) -> list[tuple[int, int, int]]:
    """``fields`` laid end to end as u16 words: each one's ``(first leaf, first word, leaf bits)``, then ``(leaves, words, 0)``;
    refused where a field isn't leaves of one whole-byte width filling whole words (an s8 weight's are two a word)."""
    out, leaf, word = [], 0, 0
    for label, ty in fields:
        ws = ty.width_set()
        bits = next(iter(ws))
        if len(ws) != 1 or bits % 8 or ty.leaves * bits % 16:
            raise Unexpressible(f"{what} {label}'s {ty.leaves} leaves are {sorted(ws)} bits, not one byte width filling whole u16 "
                                "words")
        out.append((leaf, word, bits))
        leaf, word = leaf + ty.leaves, word + ty.leaves * bits // 16
    return out + [(leaf, word, 0)]


def _word(spans: Sequence[tuple[int, int, int]], leaf: int, what: str) -> int:
    """The u16 word leaf ``leaf`` of ``spans`` (`_spans`) starts, refused inside a word."""
    first, word, bits = spans[bisect.bisect_right([s[0] for s in spans], leaf) - 1]
    if (leaf - first) * bits % 16:
        raise Unexpressible(f"{what}: leaf {leaf} starts inside a u16 word of {bits}-bit leaves")
    return word + (leaf - first) * bits // 16


@dataclass(frozen=True, slots=True)
class Row:
    """One unit's output row: template ``tid``'s unit ``unit``, its output ``out`` of ``words`` u16 words."""
    tid: str
    unit: int
    out: str
    words: int


@dataclass(slots=True)
class _Feed:
    """Units ``[at, at + count)`` of template ``tid`` read port ``port`` as ``rows``, in order: each unit the same rows, or
    under ``lead`` (digits ``(radix, step)`` of the reader units, slowest first, the radices' product ``count``) unit
    ``at + i`` each row's writer unit plus ``i``'s digits times their steps."""
    tid: str
    port: str
    at: int
    count: int
    rows: tuple[Row, ...]
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


def _merge(xs: Sequence[Any], key: Any, make: Any) -> list[Any]:
    """``xs`` (each from unit ``at``, ``units`` of them) with each run of like ones (equal ``key``, consecutive units) merged
    by ``make(first, units)``."""
    out: list[Any] = []
    for x in sorted(xs, key=lambda x: (key(x), x.at)):
        if out and key(out[-1]) == key(x) and out[-1].at + out[-1].units == x.at:
            out[-1] = make(out[-1], out[-1].units + x.units)
        else:
            out.append(x)
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
    #: refuse at `settle` every unit C-Flock's composer wouldn't prove (`interface.check_proof_block`); off only for a
    #: probe's census and for tests of the lowering at a class too small to prove
    provable: bool = True
    fns: dict[str, Any] = field(default_factory=dict)
    units: dict[str, int] = field(default_factory=dict)
    local: dict[tuple[int, str], int] = field(default_factory=dict)
    feeds: list[_Feed] = field(default_factory=list)
    #: each template port's row count, its first feed's: the cut `settle` gives it
    read_rows: dict[tuple[str, str], int] = field(default_factory=dict)
    cuts: dict[str, dict[str, int]] = field(default_factory=dict)
    wires: list[IF.Wire] = field(default_factory=list)
    reads: list[IF.Read] = field(default_factory=list)
    tables: dict[tuple[int, int, int], IF.Table] = field(default_factory=dict)
    roots: list[IF.TableRead] = field(default_factory=list)
    openings: list[IF.TableOpening] = field(default_factory=list)
    #: the (slot, step) the next units are of, and each template's runs of units by theirs, ``(at, count, slot, step)``
    label: tuple[int, int] | None = None
    spans: dict[str, list[tuple[int, int, int, int]]] = field(default_factory=dict)
    #: the class Program node the next units are of, and each template's runs' times in `spans`' order, ``[node, sub]``:
    #: sub the run's place on the longest path through its node's runs that read one another (`timed`)
    node: int = 0
    times: dict[str, list[list[int]]] = field(default_factory=dict)
    made: list[tuple[str, int, int, list[int]]] = field(default_factory=list)
    timed_feeds: int = 0
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
        time = [self.node, 0]
        self.times.setdefault(tid, []).append(time)
        self.made.append((tid, at, count, time))
        return tid, at

    def timed(self) -> None:
        """Time the runs a slot's node just made: each one after every run of the node it reads, whatever order they were
        made in (an embedding read is made before the token unit it reads, a select's combine before its ``x0``)."""
        made, feeds = self.made, self.feeds[self.timed_feeds:]
        self.made, self.timed_feeds = [], len(self.feeds)

        def run(tid: str, unit: int) -> int | None:
            return next((k for k, (t, at, n, _) in enumerate(made) if t == tid and at <= unit < at + n), None)

        reads: dict[int, set[int]] = {}
        for f in feeds if len(made) > 1 else ():
            if (r := run(f.tid, f.at)) is not None:
                reads.setdefault(r, set()).update(w for x in {(x.tid, x.unit) for x in f.rows} if (w := run(*x)) is not None)
        sub = [0] * len(made)
        for _ in range(len(made) + 1):
            longer = [(r, sub[w] + 1) for r, ws in reads.items() for w in ws if sub[w] >= sub[r]]
            for r, s in longer:
                sub[r] = max(sub[r], s)
            if not longer:
                break
        else:
            raise Unexpressible(f"node {self.node}: its units read one another in a cycle")
        for (*_, time), s in zip(made, sub):
            time[1] = s

    def feed(self, tid: str, port: str, at: int, count: int, rows: Sequence[Row],
             lead: tuple[tuple[int, int], ...] = ()) -> None:
        if not rows:
            raise Unexpressible(f"{tid} {port}: reads no unit's rows")
        if lead and math.prod(r for r, _ in lead) != count:
            raise ValueError(f"{tid} {port}: reader digits {lead} for {count} units")
        self.feeds.append(_Feed(tid, port, at, count, tuple(rows), lead))
        self.read_rows.setdefault((tid, port), len(rows))

    def _weight(self, lo: int, hi: int, words: int) -> tuple[str, int, int]:
        """The one weight words [lo, hi) of the weights record lie in, read in rows of ``words``: its label, its words and
        ``lo`` within it; refused where they span two weights or miss its rows."""
        k = bisect.bisect_right([w[1] for w in self.weights], lo) - 1
        if k < 0 or hi > self.weights[k][1] + self.weights[k][2]:
            raise Unexpressible(f"a weight read of words [{lo}, {hi}) of the weights record spans two weights")
        label, first, n = self.weights[k]
        if n % words or (lo - first) % words:
            raise Unexpressible(f"weight {label} of {n} words read from word {lo - first} in rows of {words}: its registration "
                                "is whole rows of the port, every read on a row")
        return label, n, lo - first

    def weight(self, lo: int, hi: int, words: int) -> tuple[str, int]:
        """The registered parameter of the one weight words [lo, hi) of the weights record lie in, read in rows of ``words``
        (`weight_param`), and ``lo`` within it; refused where they span two weights."""
        label, n, off = self._weight(lo, hi, words)
        name = weight_param(label, words)
        self.params[name] = n
        return name, off

    def read(self, tid: str, port: str, at: int, radix: Sequence[int], base: int, step: Sequence[int],
             param: str | None = None) -> None:
        """Units ``[at, at + Π radix)`` of ``tid`` read ``port`` from word ``base`` of the weights record on (``step`` words a
        digit), as rows of their weight's registered parameter at the port's width (`weight`); from word ``base`` of
        registered parameter ``param`` when given."""
        if param is None:
            lay = IF.layout(self.fns[tid])
            words = dict(lay.rows + lay.outs)[port]
            param, base = self.weight(base, base + words + sum((r - 1) * s for r, s in zip(radix, step)), words)
        self.reads.append(IF.Read(tid, port, param, at=at, radix=tuple(radix), base=base, step=tuple(step)))

    def value(self, tid: str, port: str, at: int, param: str, row: int, rows: int, count: int = 1) -> None:
        """Units ``at`` to ``at + count - 1`` of ``tid`` read ``port`` as rows ``row`` to ``row + count - 1`` of the registered
        parameter ``param``, ``rows`` rows of the port's width (a per-window value, not the weights record); an output row
        ``port`` is held to its row (`IF.Read`)."""
        lay = IF.layout(self.fns[tid])
        words = dict(lay.rows + lay.outs)[port]
        if self.params.setdefault(param, rows * words) != rows * words:
            raise Unexpressible(f"{param}: read as {rows} rows of {words} words and as {self.params[param]} leaves")
        self.reads.append(IF.Read(tid, port, param, at=at, radix=(count,), base=row * words, step=(words if count > 1 else 0,)))

    def table(self, lo: int, words: int, rows: int) -> IF.Table:
        """The registered table of ``rows`` rows of ``words`` from word ``lo`` of the weights record (an embedding's), one a
        weight and width, its parameter its own (`table_param`)."""
        label, n, off = self._weight(lo, lo + words * rows, words)
        param = table_param(label, words)
        self.params[param] = n
        return self.tables.setdefault((lo, words, rows), IF.Table(f"table/{weight_param(label, words)}+{off}", param, off,
                                                                  words, rows))

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
        """Each port read as m writer rows cut m ways (`interface.Layout.cut`: row j its outer elements
        [j·n/m, (j+1)·n/m), each writer row one of them), and every feed wired. Refused at once, naming each
        (`Unexpressible`): a port read at two row counts (a port is cut one way), rows that aren't its outer elements cut
        that many ways, a unit of more than ``grain.rows`` rows in and out, and, unless not ``provable``, a unit C-Flock's
        composer wouldn't prove (`interface.check_proof_block`: past 256 rows, no row past 111 bytes, or rows past its
        block)."""
        counts: dict[tuple[str, str], set[int]] = {}
        widths: dict[tuple[str, str], set[int]] = {}
        for f in self.feeds:
            counts.setdefault((f.tid, f.port), set()).add(len(f.rows))
            widths.setdefault((f.tid, f.port), set()).update(r.words for r in f.rows)
        why = []
        for (tid, port), ms in counts.items():
            lay = IF.layout(self.fns[tid])
            k = [p for p, _ in lay.rows].index(port)
            words, n, m = lay.rows[k][1], lay.outer[k], max(ms)
            if len(ms) > 1:
                why.append(f"{tid} {port} read as {sorted(ms)} rows")
            elif n % m or words % m or widths[(tid, port)] != {words // m}:
                why.append(f"{tid} {port} ({n} elements, {words} words) read as {m} rows of "
                           f"{sorted(widths[(tid, port)])} words")
            elif m > 1:
                self.cuts.setdefault(tid, {})[port] = m
        why += [f"{fn.id}: {k} rows in and out, past a unit's {self.grain.rows}" for tid, fn in self.fns.items()
                if (k := _rows(fn, self.cuts.get(tid))) > self.grain.rows]
        if why:
            raise Unexpressible(f"{len(why)} templates the window can't hold, each port cut once a writer row: " + "; ".join(why))
        for tid, fn in self.fns.items() if self.provable else ():
            try:
                IF.check_proof_block(fn, self.cuts.get(tid))
            except ValueError as e:
                why.append(str(e))
        if why:
            raise Unexpressible(f"{len(why)} templates C-Flock's composer wouldn't prove: " + "; ".join(why))
        for f in self.feeds:
            self._wire(f.tid, f.port, f.at, f.count, f.rows, f.lead)

    def merged(self) -> tuple[list[IF.TableRead], list[IF.TableOpening]]:
        """The table reads and the openings, each run of like ones at consecutive units one."""
        return (_merge(self.roots, lambda x: (x.reader, x.port, x.table, x.of),
                       lambda x, n: IF.TableRead(x.reader, x.port, x.table, x.of, x.at, n)),
                _merge(self.openings, lambda x: (x.reader, x.table, x.key, x.ports, x.verdict),
                       lambda x, n: IF.TableOpening(x.reader, x.table, x.key, x.ports, x.verdict, x.at, n)))


def _rows(fn: Any, cuts: dict[str, int] | None) -> int:
    """A unit of ``fn``'s rows in and out under ``cuts``, each an hm96 slot C-Flock's composer gives it
    (`interface.unit_rows`)."""
    lay = IF.layout(fn)
    return len(lay.cut(cuts)) + len(lay.outs)


Time = tuple[int, int]
#: an entry before it is one: its first unit, its digits as ``(radix, axis)``, its step, its first unit's time and its time
#: a digit
_Group = tuple[int, tuple[tuple[int, str], ...], int, Time, tuple[Time, ...]]


def _minus(a: Time, b: Time) -> Time:
    return a[0] - b[0], a[1] - b[1]


def _follow(groups: list[_Group], axis: str, delta: int) -> list[_Group]:
    """``groups``, consecutive, with each run of like ones (one digits and time a digit, their steps ``delta`` apart and
    their times one step apart) as one whose first digit is ``axis``, of the run's length and its time that step."""
    out, i = [], 0
    while i < len(groups):
        at, digits, step, t, dt = groups[i]
        j = i + 1
        while j < len(groups) and groups[j][1:3] == (digits, step + (j - i) * delta) and groups[j][4] == dt \
                and _minus(groups[j][3], groups[j - 1][3]) == _minus(groups[i + 1][3], t):
            j += 1
        out.append((at, ((j - i, axis),) + digits, step, t, (_minus(groups[i + 1][3], t),) + dt) if j > i + 1 else groups[i])
        i = j
    return out


def entries(spans: dict[str, list[tuple[int, int, int, int]]], slots: int,
            times: dict[str, list[list[int]]] | None = None) -> list[tuple[IF.InferenceEntry, Time, tuple[Time, ...]]]:
    """Each template's units as inference entries (`interface.InferenceEntry`), each with its first unit's time and its
    time a digit (`_Builder.times`; one time for all when ``times`` is None): its runs in unit order, an allocation's slots'
    runs of one length, step and time side by side, are one entry (the slot, then the run, neither moving the time), and
    like entries that follow on, their times one step apart, merge: at one step under a ``-`` digit, again until none
    merges (a layer's two quantizers are a digit, its layers another), then at consecutive steps under a ``step`` digit.
    So each entry's time is affine in its digits. Refused where a template's units aren't such runs."""
    out = []
    for tid, spans_ in spans.items():
        merged: list[tuple[int, int, int, int, Time]] = []
        for (at, n, slot, step), t in sorted(zip(spans_, (tuple(t) for t in times[tid]) if times else [(0, 0)] * len(spans_))):
            if merged and merged[-1][2:] == (slot, step, t):
                merged[-1] = (merged[-1][0], merged[-1][1] + n, slot, step, t)
            else:
                merged.append((at, n, slot, step, t))
        groups: list[_Group] = []
        for i in range(0, len(merged), slots):
            g = merged[i:i + slots]
            if [r[2] for r in g] != list(range(slots)) or len({(r[1], r[3], r[4]) for r in g}) != 1:
                raise Unexpressible(f"{tid}: units from {g[0][0]} are not {slots} slots' runs of one length at one step and "
                                    "one time")
            digits = ((slots, "slot"),) + (((g[0][1], "-"),) if g[0][1] > 1 else ())
            groups.append((g[0][0], digits, g[0][3], g[0][4], ((0, 0),) * len(digits)))
        while len(nxt := _follow(groups, "-", 0)) < len(groups):
            groups = nxt
        groups = _follow(groups, "step", 1)
        out += [(IF.InferenceEntry(tid, at, tuple(r for r, _ in d), tuple(a for _, a in d), step), t, dt)
                for at, d, step, t, dt in groups]
    return out


#: the wire-order certificate's format (`order`)
ORDER = "verity/wire-order/v1"


def _places(radix: Sequence[int]) -> list[int]:
    """Each digit's place value in mixed radix ``radix``, the first digit the slowest."""
    out, p = [], 1
    for r in reversed(radix):
        out.append(p)
        p *= r
    return out[::-1]


def _sides(w: IF.Wire) -> tuple[tuple[str, int, list[int]], tuple[str, int, list[int]]]:
    """``w``'s writer side and reader side, each its template, its unit at digits 0 and each digit's unit stride: the
    writer's its steps, the reader's its place among the reader units, the cut digit's 0."""
    reader = _places(w.radix[:-1]) + [0] if w.cut[1] > 1 else _places(w.radix)
    return (w.writer, w.base, list(w.step)), (w.reader, w.at, reader)


def _place(o: int, strides: Sequence[int], radix: Sequence[int],
           e: IF.InferenceEntry) -> tuple[list[int], dict[int, tuple[int, int]]] | None:
    """Units ``o + Σ d_k·strides[k]`` (``d`` over ``radix``) in entry ``e``'s digits: ``o``'s digits, and each moving digit's
    position of ``e`` and multiple there (the position of radix past 1 with the greatest place dividing its stride); None
    where a digit has none."""
    places, out = _places(e.radix), {}
    for k, (r, t) in enumerate(zip(radix, strides)):
        if r > 1 and t:
            q = next((q for q, (R, p) in enumerate(zip(e.radix, places)) if R > 1 and t % p == 0), None)
            if q is None:
                return None
            out[k] = (q, t // places[q])
    return [(o - e.at) // p % R for R, p in zip(e.radix, places)], out


def _reach(digits: list[int], moves: dict[int, tuple[int, int]], radix: Sequence[int], q: int,
           but: int | None = None) -> tuple[int, int]:
    """The least and the greatest digit ``q`` reaches as the digits that move it (but ``but``) run over their radices."""
    ms = [(m, radix[k]) for k, (p, m) in moves.items() if p == q and k != but]
    return digits[q] + sum(min(0, m * (r - 1)) for m, r in ms), digits[q] + sum(max(0, m * (r - 1)) for m, r in ms)


def _misfit(w: IF.Wire, es: Sequence[IF.InferenceEntry]) -> tuple[int, int] | tuple[()] | None:
    """None where each of ``w``'s sides moves its entry's digits without a carry (`order`), else the refinement that
    fixes the first carry, ``(digit, f)`` (`_refined`: the digit's fast part the largest of its radix's divisors that
    fits), or ``()`` where none does."""
    cut = len(w.radix) - 1 if w.cut[1] > 1 else None
    for (_, o, strides), e in zip(_sides(w), es):
        got = _place(o, strides, w.radix, e)
        if got is None:
            return ()
        digits, moves = got
        reach = [_reach(digits, moves, w.radix, q) for q in range(len(e.radix))]
        q = next((q for q, (lo, hi) in enumerate(reach) if lo < 0 or hi > e.radix[q] - 1), None)
        if q is None:
            continue
        for k in sorted((k for k, (p, _) in moves.items() if p == q and k != cut),
                        key=lambda k: -abs(moves[k][1]) * (w.radix[k] - 1)):
            m, (lo, hi) = moves[k][1], _reach(digits, moves, w.radix, q, k)
            room = e.radix[q] - 1 - hi if m > 0 else lo
            f = max((f for f in range(2, w.radix[k]) if w.radix[k] % f == 0 and abs(m) * (f - 1) <= room), default=0)
            if f:
                return k, f
        return ()
    return None


def _wire(w: IF.Wire, radix: Sequence[int], step: Sequence[int], **at: Any) -> IF.Wire:
    """``w`` over digits ``radix`` and ``step`` instead, and ``at``'s other fields; its digits of radix 1 dropped."""
    d = [(r, s) for r, s in zip(radix, step) if r > 1] or [(1, 0)]
    f = {"at": w.at, "base": w.base, "cut": w.cut, **at}
    return IF.Wire(w.reader, w.port, w.writer, w.out, f["at"], tuple(r for r, _ in d), f["base"], tuple(s for _, s in d),
                   f["cut"])


def _refined(w: IF.Wire, k: int, f: int) -> IF.Wire:
    """``w`` with digit ``k`` (radix R, step t) as two, ``(R / f, t·f)`` then ``(f, t)``: the same rows."""
    return _wire(w, w.radix[:k] + (w.radix[k] // f, f) + w.radix[k + 1:], w.step[:k] + (w.step[k] * f, w.step[k]) + w.step[k + 1:])


def _entries_of(w: IF.Wire, find: Any) -> tuple[int, int] | None:
    """The one entry ``w``'s writer units lie in and the one its reader units do, None where either side's lie in two."""
    got = []
    for t, o, strides in _sides(w):
        ends = {find(t, o + sum(f(0, s * (r - 1)) for r, s in zip(w.radix, strides))) for f in (min, max)}
        if len(ends) > 1:
            return None
        got.append(ends.pop())
    return got[0], got[1]


def _values(w: IF.Wire, k: int, v: int, n: int) -> IF.Wire:
    """``w`` at values ``[v, v + n)`` of digit ``k``, every slower digit of radix 1."""
    return _wire(w, w.radix[:k] + (n,) + w.radix[k + 1:], w.step, at=w.at + v * _sides(w)[1][2][k], base=w.base + v * w.step[k])


def _pieces(w: IF.Wire, ents: Sequence[IF.InferenceEntry], find: Any) -> list[tuple[IF.Wire, int, int]]:
    """``w`` as wires of the same rows, each with its writer entry and reader entry (`placed`)."""
    while (got := _entries_of(w, find)) is not None:
        fix = _misfit(w, [ents[k] for k in got])
        if fix is None:
            return [(w, *got)]
        if not fix:
            break
        w = _refined(w, *fix)
    c0, C = w.cut
    rows = [_pieces(_wire(w, w.radix[:-1], w.step[:-1], base=w.base + v * w.step[-1], cut=(c0 + v, 1)), ents, find)
            for v in range(C)] if C > 1 else []
    k = next((k for k, r in enumerate(w.radix[:len(w.radix) - (C > 1)]) if r > 1), None)
    if k is None or rows and all(len(p) == 1 for p in rows):
        return [p for ps in rows for p in ps]
    runs: list[list[Any]] = []
    for v in range(w.radix[k]):
        key = _entries_of(_values(w, k, v, 1), find) if got is None else None
        if runs and key is not None and runs[-1][2] == key:
            runs[-1][1] += 1
        else:
            runs.append([v, 1, key])
    return [p for v, n, _ in runs for p in _pieces(_values(w, k, v, n), ents, find)]


def placed(wires: Sequence[IF.Wire], ents: Sequence[IF.InferenceEntry]) -> list[tuple[IF.Wire, int, int]]:
    """``wires`` as wires of the same rows, each inside one writer entry and one reader entry of ``ents`` (its units'
    entries, by index) and moving one digit of each a digit without a carry, in the order `interface.coalesce` gives:
    a digit that moves two of an entry's digits is refined into two (`_refined`), and a wire still across entries or out of
    their digits is cut, into its rows where that alone places it, else its slowest digit into runs of values whose units
    lie in one writer entry and one reader entry (out of its entries' digits, into its values), and placed again."""
    by: dict[str, tuple[list[int], list[int]]] = {}
    for k, e in sorted(enumerate(ents), key=lambda x: (x[1].template, x[1].at)):
        by.setdefault(e.template, ([], []))[0].append(e.at)
        by[e.template][1].append(k)

    def find(t: str, u: int) -> int:
        starts, ks = by.get(t, ([], []))
        j = bisect.bisect_right(starts, u) - 1
        if j < 0 or u >= ents[ks[j]].at + ents[ks[j]].units:
            raise Unexpressible(f"{t}: unit {u} is in no inference entry")
        return ks[j]

    out = [p for w in wires for p in _pieces(w, ents, find)]
    return sorted(out, key=lambda p: (p[0].reader, p[0].port, p[0].at, p[0].cut))


def order(pieces: Sequence[tuple[IF.Wire, int, int]], ents: Sequence[tuple[IF.InferenceEntry, Time, tuple[Time, ...]]]
          ) -> dict[str, Any]:
    """The circuit's wire-order certificate (`ORDER`) for its wires ``pieces`` (`placed`) and its entries ``ents``
    (`entries`): each entry a segment ranked by its units' time, node·S + sub with S past every sub, affine in its digits;
    each wire its writer segment and its reader segment. A wire is ordered where every writer unit's rank is below its
    reader unit's: each side's rank is affine in the wire's digits (`_place`), so its greatest rank(writer) - rank(reader)
    is a sum over the digits. Refused where a wire isn't ordered."""
    S = 1 + max((t[1] + sum(max(0, d[1] * (r - 1)) for d, r in zip(dt, e.radix)) for e, t, dt in ents), default=0)
    segs = [{"template": e.template, "at": e.at, "radix": list(e.radix), "c": t[0] * S + t[1],
             "r": [d[0] * S + d[1] for d in dt]} for e, t, dt in ents]
    for i, (w, *ks) in enumerate(pieces):
        K, a = [], []
        for (_, o, strides), k in zip(_sides(w), ks):
            got = _place(o, strides, w.radix, ents[k][0])
            if got is None:
                raise ValueError(f"wire {i} {w.doc()}: not placed in entry {k}")
            digits, moves = got
            K.append(segs[k]["c"] + sum(r * b for r, b in zip(segs[k]["r"], digits)))
            a.append([segs[k]["r"][moves[j][0]] * moves[j][1] if j in moves else 0 for j in range(len(w.radix))])
        reach = K[0] - K[1] + sum(max(0, (x - y) * (r - 1)) for x, y, r in zip(*a, w.radix))
        if reach >= 0:
            raise Unexpressible(f"wire {i} {w.doc()}: rank(writer) - rank(reader) reaches {reach} >= 0; its writer isn't made "
                                "before its reader")
    return {"format": ORDER, "segments": segs, "wires": [list(ks) for _, *ks in pieces]}
