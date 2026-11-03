"""The window circuit's units as `window_circuit` lowers a class Program into them: each template's units, allocated a slot
at a time with each allocation's slots side by side (`_Builder.unit`), the rows each unit's ports read (`_Feed`), the
registered reads (the weights' and the scan state's), and `settle`: each port read as m writer rows cut m ways, a row of
its reader a writer row, and every feed wired as `interface.Wire` patterns, one a run of rows from one writer's output at a
constant unit step. No unit copies rows together: a reader reads each writer's row as it is. A unit holds at most
``Grain.rows`` rows in and out. Each template's units are the circuit's inference entries by their (slot, step)
(`inference`).
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

from .build import Unexpressible

WEIGHTS = WR.PARAM


def weight_param(label: str, words: int) -> str:
    """The registered parameter of weight ``label`` (its name in the weights record) read in rows of ``words`` u16 words:
    one registration a weight and row width, so every read's base and steps are whole rows of it (`interface._reads_of`)
    and its pin is a root over the weight's own rows (`interface.registered_reads`), the committer's hiding root for it
    (`window_weights.weights_registration`); a pin of `WEIGHTS` pins none of them (`interface.input_count`)."""
    return f"{WEIGHTS}/{label}@{words}"


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
    openings: list[IF.Opening] = field(default_factory=list)
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

    def merged(self) -> tuple[list[IF.TableRead], list[IF.Opening]]:
        """The table reads and the openings, each run of like ones at consecutive units one."""
        return (_merge(self.roots, lambda x: (x.reader, x.port, x.table, x.of),
                       lambda x, n: IF.TableRead(x.reader, x.port, x.table, x.of, x.at, n)),
                _merge(self.openings, lambda x: (x.reader, x.table, x.key, x.ports, x.verdict),
                       lambda x, n: IF.Opening(x.reader, x.table, x.key, x.ports, x.verdict, x.at, n)))


def _rows(fn: Any, cuts: dict[str, int] | None) -> int:
    """A unit of ``fn``'s rows in and out under ``cuts``, each an hm96 slot C-Flock's composer gives it
    (`interface.unit_rows`)."""
    lay = IF.layout(fn)
    return len(lay.cut(cuts)) + len(lay.outs)


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


def inference(spans: dict[str, list[tuple[int, int, int, int]]], slots: int) -> list[IF.InferenceEntry]:
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
