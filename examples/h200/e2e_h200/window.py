"""The H200 window writer: where each proof block's input ports read their bytes in the step commitments, for
circuits' FAMILIES map (`plan.Map`).

**The layout.** `shared_layout` is FAMILIES' shared-row commit layout, one commit unit a row of at most
``row`` bytes (1 KiB), built from the map's groups (the embedding, each layer, the head):

- tiled: every table its readers share (the linears' coordinates, the embedding, the norm outputs, the residuals, softmax
  and PV), cut into rows of ``row`` bytes, the last one shorter. Producer unit i of a family writes a window of one row, and
  the gathers go;
- pooled within a group, at most two parts a row and at most ``row`` bytes: the RMS chunk sums and the amax words;
- folded: each quantizer chunk's codes and its table's f32 scale, one row of two parts;
- kept: K and V (``wk`` and ``wv``), a slot's one row each, since their positions are read at later steps.

At the frozen shape and 1 KiB it is 21,857 rows a slot under ``layout_digest`` `FROZEN_DIGEST`, and its
writers (`writers`) are circuits' writers artifact's. Under the ``residual`` fusion (`plan.FUSIONS`) the projections' rows
go: 18,529 rows and 24,008,260 B a slot, 1,099 writers. With ``norm`` as well each norm's output rows go too: 15,169 rows
and 20,569,300 B a slot, 1,204 writers. With ``pv`` too each layer's softmax rows go: 14,337 rows and 19,717,332 B a slot,
1,152 writers.

**The reads.** `Window.locate` gives a proof block's input port as `Piece`\\ s, its bytes theirs in order:

- `Segment`: bytes of one committed row, named by the step whose commitment holds it, the commitment's slot at that step
  (the batch row the sequence ran at), the row's index in the slot's layout, and the bytes within it;
- `Registered`: bytes of a row of a registered weight, in no step commitment: the tensor's little-endian bytes cut into rows
  of `Tensor.words` u16 words, the positions its registration opens. At the frozen shape a row of a linear is
  8,192 codes, so ``wrow{r}`` of coordinate n is its row n R + r whole and ``sw`` the scale's row n;
- `Run`: a run input of the step's Program, ``token`` (ingress) or ``n`` (position + 1, kind position);
- `Absent`: a key row at or past ``n``, never opened, which the Definition masks; assembled as zero bytes.

A proof block is unit ``index`` of a family (the map's op path, coords ``op``) at a step and slot; its ports are its
Definition's (the Program's: `PvRowPad_v2`'s ``p, v0 .. v31, n`` for PV). An attention block at step s on a slot at
position p reads key j < n = p + 1 at step s - p + j, in the row that sequence held then (`Replayer.history`'s rule). A
read opens the widest row holding its bytes and an output the narrowest, so the same calls work on today's
``families/v1`` layout (`commit.layout`), where a gather's row re-commits its parts' bytes.

`Window.bundle` gives the same pieces as the drawn-block bundle's ``locate`` returns them (a ``Read``'s
commit unit and rank, the port word it fills and its bytes in the row; ``n`` a ``Position``; an output that is one whole
row as ``(unit, rank, None)``). Binding a port's words to the segments of rows opened under the step roots is that bundle's
and C-Flock's statement's: the windows wiring reads a row whole, and a port here is pieces of rows.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .commit import _slots, fusion_tag
from .plan import Entry, Map, Table, Units
from .shape import PV_K, Shape, forms
from .units import sources
from e2e_h200 import served_weights as SW
from verity.ml.commitments import layout as L

ROW = 1024
KEPT = ("wk", "wv")
FROZEN_ROWS = 21_857
FROZEN_DIGEST = ("eb7937f19e2e8e3a0daf9f7b85f33d837b267a079cd0c5c60d3581e8c2b1675b"
                 "095d54fd36f2e8434c67734213533c0aba0e40d9f2601cc4307ca9474e403a03")
FROZEN_JSON_DIGEST = ("f8d6252c5933f22c8c276c911a7c22e84c7ea12d846d7de8f9ee90e312c68005"
                      "3d41a525a4350c921f9cf12b80fc0dc9d43d3034c8b7a33f83bc069c1e656a9f")
WRITERS_SCHEMA = "circuits/shared-row-bindings/v2"


@dataclass(frozen=True)
class Segment:
    """``nbytes`` bytes at ``offset`` of row ``row`` (`Window.unit_of` gives its commit unit and rank) of slot ``slot`` in
    step ``step``'s commitment."""

    step: int
    slot: int
    row: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Registered:
    """``nbytes`` bytes at ``offset`` of registered row ``row`` (its position) of the registered value ``value``."""

    value: str
    row: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Run:
    """The Program's run input ``name``: ``token`` or ``n``, one u16 word."""

    name: str


@dataclass(frozen=True)
class Absent:
    nbytes: int


Piece = Segment | Registered | Run | Absent


def _short(op: str) -> str:
    return op.split(".", 1)[1] if op[0] == "l" and op[1].isdigit() else op


def _kind(t: Table) -> str:
    op = _short(t.op)
    if op in KEPT:
        return "kept"
    if op.endswith("_sums"):
        return "sums"
    if op.endswith("_amax") or op == "argmax":
        return "amax"
    return "quant" if t.quant else "tiled"


def _sources(entries: list[Entry]) -> dict[str, int]:
    """Each table's first tensor in its group (a quantizer's scale is the next)."""
    out, n = {}, 0
    for t, _ in entries:
        out[t.op] = n
        n += 2 if t.quant else 1
    return out


def _shared_group(entries: list[Entry], row: int) -> L.Group:
    src = _sources(entries)
    sizes = [n for t, _ in entries for n in ([t.nbytes, 4] if t.quant else [t.nbytes])]
    rows: list[Any] = []
    pools: dict[str, list[list[tuple[int, int]]]] = defaultdict(list)
    for t, us in entries:
        s, k = src[t.op], _kind(t)
        if k == "tiled":
            rows += [L.Row((L.Part(s, o, min(row, t.nbytes - o)),)) for o in range(0, t.nbytes, row)]
        elif k == "kept":
            (g,) = [u for u in us if u.template == "AllGather_v1"]
            rows += [L.Row((L.Part(s, g.offset + i * g.stride, g.nbytes),)) for i in range(g.count)]
        elif k == "quant":
            (q,) = [u for u in us if u.scale or u.scale_only]
            rows += [L.Row((L.Part(s, q.offset + i * q.stride, q.stride), L.Part(s + 1, 0, 4))) for i in range(q.count)]
        else:
            if t.nbytes > row:
                raise ValueError(f"{t.op}: {t.nbytes} B doesn't pool into {row} B rows")
            pool = pools[k]
            if pool and len(pool[-1]) < 2 and sum(n for _, n in pool[-1]) + t.nbytes <= row:
                pool[-1].append((s, t.nbytes))
            else:
                pool.append([(s, t.nbytes)])
                rows.append(k)
    taken = {k: iter(v) for k, v in pools.items()}
    rows = [L.Row(tuple(L.Part(s, 0, n) for s, n in next(taken[r]))) if isinstance(r, str) else r for r in rows]
    return L.Group(tuple(sizes), tuple((r,) for r in rows))


def shared_layout(shape: Shape, row: int = ROW, fused: frozenset[str] = frozenset()) -> Any:
    """FAMILIES' shared-row layout at ``shape`` with the map's fusions ``fused``, one commit unit a row of at most ``row``
    bytes (a quantizer's row is its chunk's codes and the scale), as the `slots.CommitShape` that commits it; every layer's group
    is one `L.Group`."""
    S, _ = _slots()
    m = Map(shape, fused)
    layer = _shared_group(m.layer(0), row)
    groups = [layer if 1 <= i <= shape.layers else _shared_group(entries, row) for i, entries in enumerate(m.groups())]
    lay = S.CommitShape(f"e2e-h200/{shape.name}/c{shape.context}/families-shared-rows-{row}b{fusion_tag(m.fused)}/v1",
                   tuple(groups))
    check_shared(lay, m)
    return lay


def check_shared(layout: Any, m: Map) -> None:
    """`L.check_shared` on ``layout`` with every producer unit's output in each of its distinct groups (the layers' one group
    once): refuses a layout of the map's tables that isn't shared-row."""
    outputs, seen = [], set()
    for gi, (g, entries) in enumerate(zip(layout.groups, m.groups(), strict=True)):
        if id(g) not in seen:
            seen.add(id(g))
            src = _sources(entries)
            outputs += [(gi, src[t.op] + 1, 0, 4) if u.scale_only else (gi, src[t.op], u.offset + i * u.stride, u.nbytes)
                        for t, us in entries for u in us if u.template != "AllGather_v1" for i in range(u.count)]
    L.check_shared(L.Layout(layout.name, tuple(layout.groups)), outputs)


@dataclass(frozen=True)
class _Index:
    """One distinct group's rows by tensor: per tensor, its parts sorted by offset as (start, end, row in the group, offset
    in the row), and each unit's first row in the group."""

    parts: dict[int, np.ndarray]
    widest: dict[int, int]
    unit_rows: np.ndarray
    row_bytes: np.ndarray

    @classmethod
    def of(cls, group: Any) -> _Index:
        by: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        unit_rows, row_bytes, r = [], [], 0
        for u in group.units:
            unit_rows.append(r)
            for row in u:
                off = 0
                for p in row.parts:
                    by[p.source].append((p.offset, p.offset + p.nbytes, r, off))
                    off += p.nbytes
                row_bytes.append(off)
                r += 1
        parts = {s: np.array(sorted(v), np.int64) for s, v in by.items()}
        return cls(parts, {s: int((a[:, 1] - a[:, 0]).max()) for s, a in parts.items()}, np.array(unit_rows, np.int64),
                   np.array(row_bytes, np.int64))

    def cover(self, source: int, lo: int, hi: int, widest: bool) -> list[tuple[int, int, int]]:
        """Bytes ``lo .. hi`` of ``source`` as (row in the group, offset in the row, bytes), each from the widest (or the
        narrowest) row holding its first byte."""
        a, out = self.parts[source], []
        while lo < hi:
            k = int(np.searchsorted(a[:, 0], lo, side="right"))
            j = max(0, int(np.searchsorted(a[:, 0], lo - self.widest[source], side="left")))
            c = a[j:k]
            c = c[c[:, 1] > lo]
            if not len(c):
                raise ValueError(f"byte {lo} of tensor {source} is in no row")
            w = c[:, 1] - c[:, 0]
            s, e, r, off = (int(x) for x in c[int(np.argmax(w) if widest else np.argmin(w))])
            n = min(e, hi) - lo
            out.append((r, off + lo - s, n))
            lo += n
        return out


class Window:
    """Proof blocks' reads and outputs in the step commitments of a run at ``shape``: ``schedule[s]`` has ``slots`` and
    ``positions`` (`commit.StepTables`, `driver.StepResult`), batch row b of step s being sequence ``slots[b]`` at position
    ``positions[b]``; ``layout`` is `shared_layout` (by default) or any layout of the map's tables, ``fused`` the map's
    fusions."""

    def __init__(self, shape: Shape, schedule: Sequence[Any], layout: Any = None, fused: frozenset[str] = frozenset()) -> None:
        self.shape, self.schedule = shape, schedule
        self.layout = shared_layout(shape, fused=fused) if layout is None else layout
        self.map, self.f, self.src = Map(shape, fused), forms(shape), sources(shape)
        groups = self.map.groups()
        if len(groups) != len(self.layout.groups):
            raise ValueError(f"the layout has {len(self.layout.groups)} groups, the map {len(groups)}")
        self.units: dict[str, Units] = {}
        self.tables: dict[str, tuple[int, int, Table]] = {}
        for gi, entries in enumerate(groups):
            src = _sources(entries)
            for t, us in entries:
                self.tables[t.op] = (gi, src[t.op], t)
                self.units |= {u.name: u for u in us}
        self._index: dict[int, _Index] = {}
        self._row_base = self.layout.group_row_base
        self._unit_base = self.layout.unit_base
        if shape.tok_rows != "coordinate":
            raise ValueError(f"{shape.name}: an embedding block reads tok's coordinate rows, not {shape.tok_rows!r} rows")
        self._weights = {t.name: t for t in SW.tensors(shape)}

    # -- the layout

    def _ix(self, gi: int) -> _Index:
        g = self.layout.groups[gi]
        if id(g) not in self._index:
            self._index[id(g)] = _Index.of(g)
        return self._index[id(g)]

    def _segments(self, step: int, slot: int, op: str, lo: int, hi: int, widest: bool = True) -> tuple[Segment, ...]:
        gi, s, t = self.tables[op]
        if not 0 <= lo <= hi <= t.nbytes:
            raise ValueError(f"bytes {lo} .. {hi} are outside {op}'s {t.nbytes}")
        base = self._row_base[gi]
        return tuple(Segment(step, slot, base + r, o, n) for r, o, n in self._ix(gi).cover(s, lo, hi, widest))

    def _scale(self, step: int, slot: int, op: str, chunk: int) -> Segment:
        """Quantizer ``op``'s scale as committed with its chunk ``chunk``: the layout commits the scale once a chunk, the
        chunks' in order (beside the codes in a shared row, or in the chunk's own commit unit)."""
        gi, s, t = self.tables[op]
        a = self._ix(gi).parts[s + 1]
        if len(a) != t.nbytes // self.f.qchunk(t.nbytes) or (a[:, 1] - a[:, 0] != 4).any():
            raise ValueError(f"{op}'s layout doesn't commit its scale once a chunk")
        return Segment(step, slot, self._row_base[gi] + int(a[chunk, 2]), int(a[chunk, 3]), 4)

    def _registered(self, value: str, lo: int, hi: int) -> tuple[Registered, ...]:
        """Bytes ``lo .. hi`` of registered value ``value`` as pieces of its registered rows."""
        rb = 2 * self._weights[value].words
        return tuple(Registered(value, r, max(lo, r * rb) - r * rb, min(hi, (r + 1) * rb) - max(lo, r * rb))
                     for r in range(lo // rb, -(-hi // rb)))

    def unit_of(self, row: int) -> tuple[int, int]:
        """Row ``row`` of a slot's layout as (its commit unit, its rank in the unit)."""
        gi = int(np.searchsorted(self._row_base, row, side="right")) - 1
        ix = self._ix(gi)
        u = int(np.searchsorted(ix.unit_rows, row - self._row_base[gi], side="right")) - 1
        return self._unit_base[gi] + u, row - self._row_base[gi] - int(ix.unit_rows[u])

    def row_bytes(self, row: int) -> int:
        gi = int(np.searchsorted(self._row_base, row, side="right")) - 1
        return int(self._ix(gi).row_bytes[row - self._row_base[gi]])

    # -- the schedule

    def _slots_of(self, step: int) -> np.ndarray:
        st = self.schedule[step]
        return np.arange(len(st.positions)) if st.slots is None else np.asarray(st.slots)

    def position(self, step: int, slot: int) -> int:
        return int(self.schedule[step].positions[slot])

    def history(self, step: int, slot: int) -> list[tuple[int, int]]:
        """The (step, slot) of the commitment at which the sequence of ``slot`` at ``step`` ran each position 0 .. p."""
        q, p = int(self._slots_of(step)[slot]), self.position(step, slot)
        out = []
        for j in range(p + 1):
            s = step - p + j
            rows = np.flatnonzero(self._slots_of(s) == q) if s >= 0 else []
            if not len(rows) or self.position(s, int(rows[0])) != j:
                raise ValueError(f"sequence {q} is at position {p} at step {step} but didn't run position {j} at step {s}")
            out.append((s, int(rows[0])))
        return out

    # -- proof blocks

    def families(self) -> list[str]:
        """Every family with proof blocks, in the order the step writes them (a gather has none)."""
        return [u.name for g in self.map.groups() for _, us in g for u in us if u.template != "AllGather_v1"]

    def family(self, name: str) -> Units:
        u = self.units[name]
        if u.template == "AllGather_v1":
            raise ValueError(f"{name} is a gather: its table is the rows its parts wrote, and it has no proof block")
        return u

    def ports(self, family: str) -> tuple[str, ...]:
        u = self.family(family)
        t = u.template
        if t in ("ScaledMmS8CoordinateRows_v1", "ScaledMmS8ResidualCoordinateRows_v1"):
            r = self._r(u)
            ports = tuple(f"x{i}" for i in range(r)) + ("sx",) + tuple(f"wrow{i}" for i in range(r)) + ("sw",)
            return ports + ("r",) if t == "ScaledMmS8ResidualCoordinateRows_v1" else ports
        if t == "PvRow_v1":
            return ("p",) + tuple(f"v{j}" for j in range(PV_K)) + ("n",)
        if t == "QkSoftmaxPvPad_v1":
            return ("q", "k") + tuple(f"v{j}" for j in range(PV_K)) + ("n",)
        return {"EmbeddingCoordinate_v1": ("tok", "col", "pos"), "RmsSumSqChunks_v1": ("x",),
                "RmsNormOutSums_v1": ("x", "w", "s"), "RowAbsMax_v1": ("x",), "QuantChunkS8_v1": ("x", "amax"),
                "RmsNormAbsMax_v1": ("x", "w", "s"), "RmsNormQuantCodes_v1": ("x", "w", "s", "amax"),
                "QuantScaleS8_v1": ("amax",),
                "QkSoftmaxRowPad_v1": ("q", "k", "n"), "Bf16AddRow_v2": ("a", "b"), "Argmax_v2": ("logits",)}[t]

    def _r(self, u: Units) -> int:
        k = self.tables[self.src[u.name] + "_quant"][2].nbytes
        return k // self.f.qchunk(k)

    def _check(self, step: int, slot: int, u: Units, index: int) -> None:
        if not 0 <= step < len(self.schedule) or not 0 <= slot < len(self.schedule[step].positions):
            raise ValueError(f"step {step} has no slot {slot}")
        if not 0 <= index < u.count:
            raise ValueError(f"{u.name} has {u.count} units, not unit {index}")

    def locate(self, step: int, slot: int, family: str, index: int, port: str) -> tuple[Piece, ...]:
        """Input ``port`` of unit ``index`` of ``family`` at ``step`` and ``slot``: its bytes are the pieces', in order."""
        u = self.family(family)
        self._check(step, slot, u, index)
        if port not in self.ports(family):
            raise ValueError(f"{family}'s ports are {self.ports(family)}, not {port!r}")
        i, t, c = index, u.template, self.f.chunk
        seg = lambda op, lo, hi: self._segments(step, slot, op, lo, hi)  # noqa: E731
        whole = lambda op: seg(op, 0, self.tables[op][2].nbytes)  # noqa: E731
        if t == "EmbeddingCoordinate_v1":
            v, at = 2 * self.shape.vocab, 2 * (i * self.shape.context + self.position(step, slot))
            return {"tok": lambda: (Run("token"),), "col": lambda: self._registered("tok", v * i, v * (i + 1)),
                    "pos": lambda: self._registered("pos", at, at + 2)}[port]()
        if t == "RmsSumSqChunks_v1":
            n = 2 * c * (u.nbytes // 4)
            return seg(self.src[u.table[:-len("_sums")]], i * n, (i + 1) * n)
        if t == "RmsNormOutSums_v1":
            op = u.table[:-len("_norm")]
            return {"x": lambda: seg(self.src[op], 2 * c * i, 2 * c * (i + 1)),
                    "w": lambda: self._registered(op + "_norm", 2 * c * i, 2 * c * (i + 1)),
                    "s": lambda: whole(op + "_sums")}[port]()
        if t == "RowAbsMax_v1":
            op = u.table[:-len("_amax")]
            x = self.src[op + "_quant"]
            if x not in self.tables:
                return seg(u.table, 0, u.offset)
            parts, w = self.f.parts(self.tables[x][2].nbytes // 2)
            if u.name.endswith(".part"):
                return seg(x, 2 * w * i, 2 * w * (i + 1))
            if parts > 1:
                return seg(x, 2 * w * (parts - 1), 2 * w * parts) + seg(u.table, 0, 2 * (parts - 1))
            return whole(x)
        if t == "QuantChunkS8_v1":
            amax = u.table[:-len("_quant")] + "_amax"
            n = self.tables[amax][2].nbytes
            return {"x": lambda: seg(self.src[u.table], 2 * u.nbytes * i, 2 * u.nbytes * (i + 1)),
                    "amax": lambda: seg(amax, n - 2, n)}[port]()
        if t in ("RmsNormAbsMax_v1", "RmsNormQuantCodes_v1", "QuantScaleS8_v1"):
            op = u.table.rsplit("_", 1)[0]
            m = self.tables[op + "_amax"][2].nbytes
            if t == "QuantScaleS8_v1":
                return seg(op + "_amax", m - 2, m)
            norm = self.src[op + "_quant"][:-len("_norm")]
            n = 2 * (u.nbytes if t == "RmsNormQuantCodes_v1" else self.f.norm_parts(self.shape.width)[1])
            return {"x": lambda: seg(self.src[norm], n * i, n * (i + 1)),
                    "w": lambda: self._registered(norm + "_norm", n * i, n * (i + 1)),
                    "s": lambda: whole(norm + "_sums"), "amax": lambda: seg(op + "_amax", m - 2, m)}[port]()
        if t in ("ScaledMmS8CoordinateRows_v1", "ScaledMmS8ResidualCoordinateRows_v1"):
            qop = self.src[u.name] + "_quant"
            k = self.tables[qop][2].nbytes
            q = self.f.qchunk(k)
            if port == "r":
                return seg(self.src[u.table][0], 2 * i, 2 * (i + 1))
            if port == "sx":
                return (self._scale(step, slot, qop, 0),)
            if port == "sw":
                return self._registered(u.name + ".scale", 4 * i, 4 * (i + 1))
            r = int(port.removeprefix("wrow").removeprefix("x"))
            if port.startswith("wrow"):
                return self._registered(u.name, k * i + r * q, k * i + (r + 1) * q)
            return seg(qop, r * q, (r + 1) * q)
        if t in ("QkSoftmaxRowPad_v1", "PvRow_v1", "QkSoftmaxPvPad_v1"):
            layer, d = u.table.split(".")[0], self.shape.head_dim
            if port == "n":
                return (Run("n"),)
            if port == "q":
                return seg(layer + ".wq", 2 * d * i, 2 * d * (i + 1))
            if port == "p":
                return seg(layer + ".softmax", 2 * PV_K * i, 2 * PV_K * (i + 1))
            kv = layer + (".wk" if port == "k" else ".wv")
            hist = self.history(step, slot)
            keys = range(PV_K) if port == "k" else [int(port[1:])]
            return tuple(p for j in keys for p in (self._segments(*hist[j], kv, 0, 2 * d) if j < len(hist)
                                                    else (Absent(2 * d),)))
        if t == "Bf16AddRow_v2":
            a, b = self.src[u.table]
            return seg(a if port == "a" else b, u.nbytes * i, u.nbytes * (i + 1))
        if t == "Argmax_v2":
            return whole("head")
        raise ValueError(f"no reads for {t}")

    def outputs(self, step: int, slot: int, family: str, index: int) -> tuple[tuple[Segment, ...], ...]:
        """Unit ``index``'s output rows (``o0``, and a quantizer's scale ``o1``) where its step committed them."""
        u = self.family(family)
        self._check(step, slot, u, index)
        lo = u.offset + index * u.stride
        if u.scale_only:
            return ((self._scale(step, slot, u.table, index),),)
        codes = self._segments(step, slot, u.table, lo, lo + u.nbytes, widest=False)
        return (codes, (self._scale(step, slot, u.table, index),)) if u.scale else (codes,)

    def bundle(self, step: int, slot: int, family: str, index: int, port: str) -> Any:
        """Port ``port`` of the block as the drawn-block bundle's ``locate`` returns it (the bundle's stub):

        - an output row (``o0``, a quantizer's scale ``o1``) that is one whole committed row: ``(unit, rank, None)``;
        - otherwise a list of pieces, each a ``Read``'s fields ``{step, slot, unit, rank, at, start, nbytes}`` or a
          ``Position``'s ``{offset, at}`` (``n`` = position + 1: offset 1), ``at`` the port row's first word it fills and the
          words no piece covers zero (`Absent` keys);
        - ``None`` for a registered read (the weights: `locate`'s `Registered` pieces)."""
        if port in ("o0", "o1"):
            segs = self.outputs(step, slot, family, index)[int(port[1])]
            if len(segs) == 1 and segs[0].offset == 0 and segs[0].nbytes == self.row_bytes(segs[0].row):
                return self.unit_of(segs[0].row) + (None,)
            pieces: tuple[Piece, ...] = segs
        else:
            pieces = self.locate(step, slot, family, index, port)
        if all(isinstance(p, Registered) for p in pieces):
            return None
        out, at = [], 0
        for p in pieces:
            if isinstance(p, Segment):
                unit, rank = self.unit_of(p.row)
                out.append({"step": p.step, "slot": p.slot, "unit": unit, "rank": rank, "at": at, "start": p.offset,
                            "nbytes": p.nbytes})
            elif isinstance(p, Run) and p.name == "n":
                out.append({"offset": 1, "at": at})
            elif not isinstance(p, Absent):
                raise ValueError(f"{family}'s port {port} reads {p}, which the bundle has no piece for")
            at += (2 if isinstance(p, Run) else p.nbytes) // 2
        return out

    # -- values

    def tensors(self, st: Any) -> list[list[np.ndarray]]:
        """A kept step's tables (`commit.StepTables`) as the layout's groups of tensors, what `slots.commit` reads."""
        outs = []
        for g in self.map.groups():
            ts = []
            for t, _ in g:
                codes, word = st.tables[t.op]
                ts += [np.asarray(codes), np.asarray(word)] if t.quant else [np.asarray(codes)]
            outs.append(ts)
        return outs

    def slot_rows(self, st: Any, slot: int) -> list[bytes]:
        """Slot ``slot``'s committed rows under the layout, from a kept step's tables."""
        S, _ = _slots()
        return S.slot_rows(self.layout, self.tensors(st), slot)

    def run_inputs(self, tokens: Sequence[Any]) -> Callable[[int, int, str], int]:
        """``run`` for `assemble`: ``tokens[s][b]`` the byte row b of step s embedded, ``n`` its position + 1."""
        return lambda step, slot, name: int(tokens[step][slot]) if name == "token" else self.position(step, slot) + 1

    def assemble(self, pieces: Sequence[Piece], rows: Callable[[int, int], Sequence[bytes]],
                 weights: Callable[[str, int], bytes], run: Callable[[int, int, str], int], step: int, slot: int) -> bytes:
        """The bytes of ``pieces`` (a block at ``step`` and ``slot``): ``rows(step, slot)`` a slot's committed rows,
        ``weights(value, row)`` a registered row's bytes, ``run(step, slot, name)`` a run input."""
        out = []
        for p in pieces:
            if isinstance(p, Segment):
                row = rows(p.step, p.slot)[p.row]
                if p.offset + p.nbytes > len(row):
                    raise ValueError(f"{p} is outside its row's {len(row)} bytes")
                out.append(bytes(row[p.offset:p.offset + p.nbytes]))
            elif isinstance(p, Registered):
                out.append(bytes(weights(p.value, p.row)[p.offset:p.offset + p.nbytes]))
            elif isinstance(p, Run):
                out.append(int(run(step, slot, p.name)).to_bytes(2, "little"))
            else:
                out.append(bytes(p.nbytes))
        return b"".join(out)

    def inputs(self, step: int, slot: int, family: str, index: int, rows: Callable[[int, int], Sequence[bytes]],
               weights: Callable[[str, int], bytes], run: Callable[[int, int, str], int]) -> dict[str, bytes]:
        """Every input port of the block, assembled."""
        return {p: self.assemble(self.locate(step, slot, family, index, p), rows, weights, run, step, slot)
                for p in self.ports(family)}

    # -- the writers

    def writers(self) -> list[dict[str, Any]]:
        """Every producer family's window rule, in circuits' writers schema (`WRITERS_SCHEMA`): unit i writes bytes
        ``row_offset + window_bytes (i % G) ..`` of commit unit ``first + i // G`` (one commit unit a row: the shared
        layout's). Refuses a layout `check_shared` refuses."""
        check_shared(self.layout, self.map)
        out = []
        for name in self.families():
            u = self.units[name]
            gi, s, _ = self.tables[u.table]
            ix, base = self._ix(gi), self._row_base[gi]
            a = ix.parts[s]
            lo = u.offset + np.arange(u.count) * u.stride
            k = np.searchsorted(a[:, 0], lo, side="right") - 1
            row, off = base + a[k, 2], a[k, 3] + lo - a[k, 0]
            if u.scale_only:
                at = {int(r): int(o) for _, _, r, o in ix.parts[s + 1]}
                off = np.array([at[int(r)] for r in a[k, 2]], np.int64)
            g = int(np.sum(row == row[0]))
            i = np.arange(u.count)
            if not ((row == row[0] + i // g).all() and (off == off[0] + (i % g) * u.nbytes).all()):
                raise ValueError(f"{name}: its units' windows don't follow the writers rule")
            out.append({"family": name, "key": u.key, "units": u.count, "table": u.table, "first": int(row[0]), "G": g,
                        "window_bytes": u.nbytes, "row_offset": int(off[0]), "row_bytes": self.row_bytes(int(row[0])),
                        "scale": bool(u.scale)})
        return out


def writers_digest(writers: Sequence[Mapping[str, Any]]) -> str:
    """SHA-512 of the writers' canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha512(json.dumps(list(writers), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def weights_of(w: Any) -> Callable[[str, int], bytes]:
    """``weights`` for `assemble` from the seeded weights (`weights.Weights`): registered row ``row`` of a tensor."""

    def row(value: str, r: int) -> bytes:
        t = w.t[value]
        rb = 2 * t.words
        return w.stream(t.offset + r * rb, t.offset + (r + 1) * rb)

    return row
