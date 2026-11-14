"""One rank's request Program in the registered rank form (`steps.request_program`: dead-free, batched in class G, its
collectives' other partials its ``peers_*`` inputs) lowered node by node into a window of units (`window_builder._Builder`),
the frame each Definition family's lowering is added to (`FAMILIES`).

The frame walks the Program's nodes in order. A node whose family has a lowering is emitted as units at the node's step
(its scheduler step: a step ends at its token select, `SELECTS`); any other node emits nothing, and every row a lowered
reader would read of it is left a free input (`Free`), listed with its family and node, so a window settles from the first
family on and what is left is always listed (`NodeWindow.census`). Every other rank's partial (a ``peers_*`` input) and
a request input (the prompt) are free inputs the same way, listed by name; the census lists every ``peers_*`` input with
the nodes that read it, lowered or not.

A lowering is ``lower(w, j, node)``: it allocates units (`NodeWindow.unit`), feeds their ports from the rows the frame hands it
(`NodeWindow.take`, `NodeWindow.feed`), reads weights from the weights record (``w.b.read``, at `NodeWindow.weight`'s word),
and states the node's output as the rows that hold it (`NodeWindow.out`, `Piece`). A port whose rows are all wired is fed;
a port any of whose rows is free is left free whole (`NodeWindow.feed` records it), unless it is cut along its outer
elements and fed row by row (`NodeWindow.feed_rows`: a collective's own partial wired, the other ranks' rows free).

A batch node (``form == "batch"``, n members) is lowered as its members: member t of an argument batched on axis 0 is its
t-th slice, a shared one (axis None) the whole argument. A feed states a run of units at once by digits (`_Builder.feed`'s
``lead``) where each member's and each block's rows are the first's shifted by a constant unit step, else one feed a unit.

A reader that reads a row whole (`NodeWindow.take` with ``whole``) gets one row a member. Where its words are several
writer rows (a ``COORDS`` GEMM's coordinates, a ``COLS`` family's column blocks), the frame gathers them first, as
`pipeline.tp.grain` gathers a collective's coordinates: one `AllGather_v1{WORLD=n, N=c}` a member where the n writer rows of
c words and the gathered row fit a gather unit's `GATHER_ROWS` rows, else in levels, `AllGather_v1{WORLD=K/c, N=c}` into
rows of K words (K the most words of a whole number of writer rows within `GATHER_ROWS`, `grain.chunk` at c = 1) and then
those R/K rows of K words gathered the same way into the row of R words (`AllGather_v1{WORLD=R/K, N=K}` where they fit one
gather unit, so two levels in all). A gather is made once for the rows it reads, and its first level's
K-word rows stand for those writer rows in any later read (``most``), so a column block of a gathered row reads its chunks
rather than its coordinates. An `AllGather_v1` moves words and computes nothing, and has no rank static.

With ``stand_ins`` (`NodeWindow`), the frame lowers the same window and also keeps, for each port it leaves free, the
rows that port stands for as its lowering placed them (`NodeWindow.stood`), so a check on values can feed it the Program's
own words (`node_window_values`): where a take would be free it hands the lowering stand-in rows (`STAND_IN`: an unlowered
node's member's words, an input's words), a whole read of rows any of which stands in is their concatenation
(`NodeWindow.cats`) and no gather, and a feed of a stand-in row leaves its port free as a free take does. A lowering that
branches on a free take otherwise than by passing it to `NodeWindow.feed` may lower differently there, which the check
refuses by comparing the two windows' digests.

The window is completeness only: that it computes what serving ran is held by the units' proofs on their committed rows.
Refused by name (`Unexpressible`): a Program with no weights record, an input that is neither the prompt, the weights nor a
``peers_*`` partial, a lowered family's argument batched on an axis other than 0 or None, a read that cuts a writer's row,
a gather whose rows are not one width or of which no whole number within a gather unit divides the row, a port row-fed
at two cuts, and at `NodeWindow.circuit`, a row-fed port also fed whole or a row of a unit fed twice, then at
`_Builder.settle`, every template whose ports aren't cut a writer row or whose unit Flock's composer wouldn't prove, and
after it a row-fed port whose cut isn't its `feed_rows` cut.
"""
from __future__ import annotations

import bisect
import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.refs import Refs, runs
from verity.commitments.windows import interface as IF
from verity.definitions.registry import b1_tp2 as B1T

from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.query.program_view import spec_base, spec_statics

from .build import Unexpressible
from .window_builder import (
    WEIGHTS,
    Grain,
    Row,
    _Builder,
    _spans,
    _step,
    _word,
    inference,
)

NAME = "verity-vllm/node-window"
PEERS = "peers_"
PROMPT = "prompt"
#: the families whose node ends its scheduler step: the token selects (`serving_templates.STAGES`' rows)
SELECTS = frozenset(f.rsplit("_v", 1)[0] for f in ST.STAGES)
#: a gather unit's rows in and out at most: `pipeline.tp.grain.CHUNK_MAX` writer rows and the gathered row
GATHER_ROWS = 128
GATHER = "gather"
#: a stand-in row's template prefix (`NodeWindow` with ``stand_ins``): ``@n:<node>`` unit m out ``lo:hi``, words [lo, hi) of
#: member m of an unlowered node; ``@p:<input>`` out ``lo:hi``, an input's words; ``@cat`` unit i, `NodeWindow.cats`' i-th
STAND_IN = "@"
_GRAIN = Grain()


@dataclass(frozen=True, slots=True)
class Piece:
    """Words [lo, hi) of each member of a node's output, as rows of ``row``'s template and output, ``row.words`` each:
    member m's first is unit ``row.unit + m * member``, each next one ``step`` units on."""
    lo: int
    hi: int
    row: Row
    step: int = 1
    member: int = 1

    def rows(self, m: int, lo: int, hi: int, what: str) -> list[Row]:
        """Member ``m``'s rows holding its words [lo, hi) of this piece; refused where the range cuts a row."""
        a, b, w = max(lo, self.lo), min(hi, self.hi), self.row.words
        if a >= b:
            return []
        i0, i1 = (a - self.lo) // w, -(-(b - self.lo) // w)
        if self.lo + i0 * w != a or self.lo + i1 * w != b:
            raise Unexpressible(f"{what}: words [{lo}, {hi}) cut a row of {w} words of {self.row.tid} {self.row.out}")
        r = self.row
        return [Row(r.tid, r.unit + m * self.member + i * self.step, r.out, w) for i in range(i0, i1)]


@dataclass(frozen=True)
class Free:
    """Units [at, at + count) of template ``tid`` leave port ``port`` free: its rows are ``kind`` ``"peer"`` (another rank's
    partial, ``name`` its input), ``"node"`` (a node of ``family``, which has no lowering yet, ``name`` its index) or
    ``"param"`` (a request input, ``name`` it)."""
    tid: str
    port: str
    at: int
    count: int
    kind: str
    family: str
    name: str
    #: the port's rows it leaves, ``(first, rows, of)``: rows [first, first + rows) of the port cut ``of`` ways
    #: (`NodeWindow.feed_rows`); the whole port by default
    rows: tuple[int, int, int] = (0, 1, 1)


class _Free(Exception):
    def __init__(self, sources: Sequence[tuple[str, str, str]]) -> None:
        super().__init__(sources)
        self.sources = tuple(dict.fromkeys(sources))


@dataclass
class _Out:
    """A lowered node's output: ``members`` of ``words`` words each, as ``pieces``."""
    members: int
    words: int
    pieces: tuple[Piece, ...]


Lowering = Callable[["NodeWindow", int, Any], None]


class NodeWindow:
    """Program ``fn`` (one rank's request Program in the registered rank form) lowered into ``b``'s units by ``families``
    (module docstring): ``lower()`` walks its nodes, ``circuit()`` settles and emits the window, ``census()`` reports it.
    ``coord`` is the width of a ``COORDS`` GEMM's coordinates (1: the serving rows' coordinate, `serving_templates.coordinate`);
    ``stand_ins``, the module docstring's."""

    def __init__(self, fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN, name: str = NAME,
                 coord: int = 1, provable: bool = True, stand_ins: bool = False) -> None:
        self.fn, self.name, self.coord, self.stand_ins = fn, name, coord, stand_ins
        self.families = dict(FAMILIES if families is None else families)
        self.params = [p for p, _ in fn.signature()[0]]
        self._bits = {p: next(iter(t.width_set())) for p, t in fn.signature()[0] if p != WEIGHTS and len(t.width_set()) == 1}
        bad = [p for p in self.params if p not in (PROMPT, WEIGHTS) and not p.startswith(PEERS)]
        if WEIGHTS not in self.params or bad:
            raise Unexpressible(f"{name}: inputs {bad or self.params}; the node window takes the prompt, the weights record and "
                                f"{PEERS}* partials")
        self.b = _Builder(grain, 1, provable)
        weights = dict(fn.signature()[0])[WEIGHTS]
        labels = [label for label, _ in weights.fields]
        self.fields = _spans(f"{name}: weight", weights.fields)
        self.b.weights = tuple((label, s[1], e[1] - s[1]) for label, s, e in zip(labels, self.fields, self.fields[1:]))
        nodes = fn.body.nodes
        ends = [j for j, n in enumerate(nodes) if _family(n) in SELECTS]
        self.steps = max(1, len(ends))
        self._step_of = [min(bisect.bisect_left(ends, j), self.steps - 1) for j in range(len(nodes))]
        self.meta: dict[int, tuple[int, int, int, list[tuple[int, int, int]] | None]] = {}
        self.outs: dict[int, _Out | str] = {}
        self.free: list[Free] = []
        self.unlowered: dict[str, list[int]] = {}
        self.peers: dict[str, list[tuple[int, str]]] = {}
        #: each template's allocations, ``(first unit, units, family)`` (a gather's family `GATHER`)
        self.allocs: dict[str, list[tuple[int, int, str]]] = {}
        #: each gathered row by the writer rows it holds, and each first-level chunk by its first writer row
        self.wholes: dict[tuple[Row, ...], Row] = {}
        self.chunks: dict[Row, tuple[tuple[Row, ...], Row]] = {}
        #: each port `feed_rows` fed, by template and port: its cut and the rows it fed, ``(row, first unit, end unit, how)``
        self.row_fed: dict[tuple[str, str], tuple[int, list[tuple[int, int, int, str]]]] = {}
        #: each allocation's node, by template and first unit
        self.node_of: dict[tuple[str, int], int] = {}
        #: with ``stand_ins``: each port left free over a stand-in row, ``(tid, port, at, count, (first, rows, of), grid)``, its
        #: rows as `Free.rows` and ``grid`` one entry a unit; and each concatenation a stand-in ``@cat`` row is
        self.stood: list[tuple[str, str, int, int, tuple[int, int, int], list[tuple[Row, ...]]]] = []
        self.cats: list[tuple[Row, ...]] = []
        self._j = -1
        self.lowered = False

    # ---- walking the Program

    def lower(self) -> NodeWindow:
        """Every node in order, at its step: lowered by its family's lowering, else left (module docstring)."""
        for j, node in enumerate(self.fn.body.nodes):
            fam = _family(node)
            self._meta(j, node)
            for arg in node.args:
                for space, idx, _lo, _hi in runs(arg.refs):
                    if space == "p" and self.params[idx].startswith(PEERS):
                        self.peers.setdefault(self.params[idx], []).append((j, fam))
            lower = self.families.get(fam)
            if lower is None or node.form not in ("call", "batch"):
                self.unlowered.setdefault(fam, []).append(j)
                self.outs[j] = fam
                continue
            self.b.label = (0, self._step_of[j])
            self._j = j
            lower(self, j, node)
            if j not in self.outs:
                raise Unexpressible(f"{self.name}: node {j} ({node.fn.id}): its lowering states no output (`NodeWindow.out`)")
        self.lowered = True
        return self

    def _meta(self, j: int, node: Any) -> None:
        """Node ``j``'s members, a member's leaves and words, and its return fields' spans (`_spans`; None where they aren't
        whole u16 words, which only an unlowered node may have)."""
        n = node.n if node.form == "batch" else 1
        leaves = node.out_leaves // n if n else 0
        try:
            ret = node.fn.signature()[1]
            fields = [f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,))]
            spans = _spans(f"{self.name}: node {j} ({node.fn.id}) return", [(f"o{i}", t) for i, t in enumerate(fields)])
            words = spans[-1][1]
        except (Unexpressible, AttributeError, TypeError):
            spans, words = None, 0
        self.meta[j] = (n, leaves, words, spans)

    # ---- what a lowering calls

    def members(self, node: Any) -> int:
        return node.n if node.form == "batch" else 1

    def arg(self, node: Any, k: int, t: int) -> Refs:
        """Member ``t``'s argument ``k``: its slice where the batch maps it on axis 0, else the whole argument."""
        refs = node.args[k].refs
        if node.form != "batch" or node.axes[k] is None:
            return refs
        if node.axes[k] != 0:
            raise Unexpressible(f"{self.name}: {node.fn.id}'s argument {k} is batched on axis {node.axes[k]}, not 0")
        L = len(refs) // node.n
        return refs.slice(t * L, (t + 1) * L)

    def segments(self, refs: Refs) -> list[tuple[Any, ...]]:
        """``refs`` as word ranges: ``("n", j, lo, hi)`` of member-local words of one member of node ``j`` (its member in
        ``lo``'s place: ``("n", j, m, lo, hi)``), or ``("p", name, lo, hi)`` of an input, the weights record's in words."""
        out: list[Any] = []
        for space, idx, lo, hi in runs(refs):
            if space == "n":
                n, L, W, spans = self.meta[idx]
                m = lo // L
                while lo < hi:
                    e = min(hi, (m + 1) * L)
                    if spans is None:
                        out.append(("n", idx, m, -1, -1))
                    else:
                        what = f"{self.name}: node {idx}"
                        out.append(("n", idx, m, _word(spans, lo - m * L, what), _word(spans, e - m * L, what)))
                    lo, m = e, m + 1
            elif self.params[idx] == WEIGHTS:
                k = bisect.bisect_right([f[0] for f in self.fields], lo) - 1
                if hi > self.fields[k + 1][0]:
                    raise Unexpressible(f"{self.name}: a weight read of leaves [{lo}, {hi}) spans two weights")
                out.append(("p", WEIGHTS, _word(self.fields, lo, f"{self.name}: a weight read"),
                            _word(self.fields, hi, f"{self.name}: a weight read")))
            else:
                out.append(("p", self.params[idx], lo, hi))
        return out

    def rows(self, refs: Refs) -> tuple[Row, ...]:
        """The writer rows ``refs`` read, whole and in order; `_Free` naming what's free where any of them is."""
        out: list[Row] = []
        free: list[tuple[str, str, str]] = []
        standing = self.stand_ins
        for seg in self.segments(refs):
            if seg[0] == "p":
                name = seg[1]
                if name == WEIGHTS:
                    raise Unexpressible(f"{self.name}: a unit reads the weights record where a wire is due")
                free.append(("peer", PEERS.rstrip("_"), name) if name.startswith(PEERS) else ("param", name, name))
                bits = self._bits.get(name, 0)
                if standing and (seg[3] - seg[2]) * bits % 16 == 0 and seg[2] * bits % 16 == 0:
                    lo, hi = seg[2] * bits // 16, seg[3] * bits // 16
                    out.append(Row(f"{STAND_IN}p:{name}", 0, f"{lo}:{hi}", hi - lo))
                else:
                    standing = False
                continue
            _, j, m, lo, hi = seg
            o = self.outs[j]
            if isinstance(o, str):
                free.append(("node", o, str(j)))
                if standing and lo >= 0:
                    out.append(Row(f"{STAND_IN}n:{j}", m, f"{lo}:{hi}", hi - lo))
                else:
                    standing = False
                continue
            for p in o.pieces:
                out += p.rows(m, lo, hi, f"{self.name}: node {j}")
        if free and not standing:
            raise _Free(free)
        return tuple(out)

    def weight(self, refs: Refs, words: int) -> int:
        """The first word of the one run of the weights record ``refs`` read, ``words`` long."""
        segs = self.segments(refs)
        if len(segs) != 1 or segs[0][:2] != ("p", WEIGHTS) or segs[0][3] - segs[0][2] != words:
            raise Unexpressible(f"{self.name}: a weight read of {segs}, not one run of {words} weight words")
        return segs[0][2]

    def label(self, refs: Refs, words: int) -> str:
        """The label of the one weight ``refs`` read, ``words`` long."""
        lo = self.weight(refs, words)
        return self.b.weights[bisect.bisect_right([x[1] for x in self.b.weights], lo) - 1][0]

    def take(self, node: Any, k: int, lo: int | None = None, hi: int | None = None, *, whole: bool = False,
             most: int | None = None) -> list[tuple[Row, ...]] | _Free:
        """Each member's rows of argument ``k``'s leaves [lo, hi) (all of them by default): as the writers hold them; with
        ``whole``, one row a member, gathered where they are several (module docstring); with ``most``, the gathered
        chunks that stand for them, gathered whole where they are still more than ``most``. A `_Free` where any is free."""
        try:
            rowss = []
            for t in range(self.members(node)):
                refs = self.arg(node, k, t)
                rowss.append(self.rows(refs if lo is None else refs.slice(lo, hi)))
        except _Free as e:
            return e
        if whole:
            return [(r,) for r in self._whole(rowss)]
        if most is not None:
            subs = [self._chunked(r) for r in rowss]
            if all(len(s) <= most and len({x.words for x in s}) == 1 for s in subs):
                return subs
            return [(r,) for r in self._whole(rowss)]
        return rowss

    def feed(self, tid: str, port: str, at: int, grid: Sequence[tuple[Row, ...]] | _Free, shape: Sequence[int]) -> None:
        """Units ``at`` on, in the mixed radix ``shape`` (the first digit slowest), read ``grid``'s rows (unit order, one entry
        a unit): one feed where every digit shifts the first unit's rows by a constant step, else a feed a unit; left free
        (`Free`) where ``grid`` is."""
        count = math.prod(shape)
        if isinstance(grid, _Free):
            for kind, fam, name in grid.sources:
                self.free.append(Free(tid, port, at, count, kind, fam, name))
            return
        if self.stand_ins and any(_stands(rows) for rows in grid):
            self._stand(tid, port, at, count, (0, 1, 1), grid)
            return
        if len(grid) != count:
            raise ValueError(f"{tid} {port}: {len(grid)} entries for units of shape {tuple(shape)}")
        lead = _lead(grid, shape)
        if lead is not None:
            self.b.feed(tid, port, at, count, grid[0], lead)
            return
        for i, rows in enumerate(grid):
            self.b.feed(tid, port, at + i, 1, rows)

    def feed_rows(self, tid: str, port: str, at: int, grids: Sequence[Sequence[tuple[Row, ...]] | _Free],
                  shape: Sequence[int]) -> None:
        """Units ``at`` on (``shape``, as `feed`) read port ``port`` cut ``len(grids)`` ways, its row i from ``grids[i]`` (a
        `take` result, one row a unit): a wired row is wired, a free one left free (`Free`, its ``rows``), so a port whose
        other rows are other ranks' partials still has its own row wired. Refused at once: a port row-fed at another cut
        before; and at `circuit`, a port also fed whole and a row of a unit fed twice (`_check_rows`)."""
        k, count = len(grids), math.prod(shape)
        if k == 1:
            self.feed(tid, port, at, grids[0], shape)
            return
        lay = IF.layout(self.b.fns[tid])
        words, n = dict(lay.rows)[port], dict(zip((p for p, _ in lay.rows), lay.outer))[port]
        if words % k or n % k:
            raise Unexpressible(f"{tid} {port}: {n} elements, {words} words, cut {k} ways")
        cut, fed = self.row_fed.setdefault((tid, port), (k, []))
        if cut != k:
            raise Unexpressible(f"{tid} {port}: row-fed cut {k} ways, and {cut} ways before; a port is cut one way")
        self.b.cuts.setdefault(tid, {})[port] = k
        for i, grid in enumerate(grids):
            stood = self.stand_ins and not isinstance(grid, _Free) and any(_stands(rows) for rows in grid)
            fed.append((i, at, at + count, "free" if isinstance(grid, _Free) or stood else "wired"))
            if isinstance(grid, _Free):
                for kind, fam, name in grid.sources:
                    self.free.append(Free(tid, port, at, count, kind, fam, name, (i, 1, k)))
                continue
            if stood:
                self._stand(tid, port, at, count, (i, 1, k), grid)
                continue
            if len(grid) != count or any(len(r) != 1 or r[0].words != words // k for r in grid):
                raise Unexpressible(f"{tid} {port}: row {i} of {k} is {words // k} words, one writer row a unit")
            lead = _lead(grid, shape)
            runs = [(at, grid[0], lead)] if lead is not None else [(at + u, rows, ()) for u, rows in enumerate(grid)]
            for lo, rows, digits in runs:
                d = [x for x in digits if x[0] > 1] or [(1, 0)]
                r = rows[0]
                self.b.wires.append(IF.Wire(tid, port, r.tid, r.out, at=lo, radix=tuple(a for a, _ in d), base=r.unit,
                                            step=tuple(s for _, s in d), cut=(i, 1)))

    def out(self, j: int, node: Any, pieces: Sequence[Piece]) -> None:
        """Node ``j``'s output as ``pieces``, which tile a member's words."""
        n, _L, W, spans = self.meta[j]
        if spans is None:
            raise Unexpressible(f"{self.name}: node {j} ({node.fn.id}): its return isn't whole u16 words")
        at = 0
        for p in sorted(pieces, key=lambda p: p.lo):
            if p.lo != at or (p.hi - p.lo) % p.row.words:
                raise Unexpressible(f"{self.name}: node {j} ({node.fn.id}): pieces {pieces} don't tile its {W} words")
            at = p.hi
        if at != W:
            raise Unexpressible(f"{self.name}: node {j} ({node.fn.id}): pieces end at word {at}, not {W}")
        self.outs[j] = _Out(n, W, tuple(pieces))

    def unit(self, family: str, fn: Any, count: int) -> tuple[str, int]:
        tid, at = self.b.unit(fn, count)
        self.allocs.setdefault(tid, []).append((at, count, family))
        self.node_of[(tid, at)] = self._j
        return tid, at

    def _cat(self, rows: tuple[Row, ...]) -> Row:
        """A stand-in for one row holding ``rows`` in turn (`STAND_IN`)."""
        self.cats.append(rows)
        return Row(f"{STAND_IN}cat", len(self.cats) - 1, "o0", sum(r.words for r in rows))

    def _stand(self, tid: str, port: str, at: int, count: int, rows: tuple[int, int, int],
               grid: Sequence[tuple[Row, ...]]) -> None:
        """Leave a port free over ``grid``'s stand-in rows, as `feed` leaves one free over a free take, and keep the grid."""
        if len(grid) != count:
            raise ValueError(f"{tid} {port}: {len(grid)} entries for {count} units")
        for kind, fam, name in dict.fromkeys(s for rs in grid for s in self._sources(rs)):
            self.free.append(Free(tid, port, at, count, kind, fam, name, rows))
        self.stood.append((tid, port, at, count, rows, list(grid)))

    def _sources(self, rows: Sequence[Row]) -> list[tuple[str, str, str]]:
        """The free sources stand-in ``rows`` read, as `rows` names them."""
        out: list[tuple[str, str, str]] = []
        for r in rows:
            if r.tid == f"{STAND_IN}cat":
                out += self._sources(self.cats[r.unit])
            elif r.tid.startswith(f"{STAND_IN}n:"):
                j = int(r.tid[len(STAND_IN) + 2:])
                out.append(("node", str(self.outs[j]), str(j)))
            elif r.tid.startswith(f"{STAND_IN}p:"):
                name = r.tid[len(STAND_IN) + 2:]
                out.append(("peer", PEERS.rstrip("_"), name) if name.startswith(PEERS) else ("param", name, name))
        return out

    def family_of(self, tid: str, unit: int) -> str:
        """The family whose allocation holds unit ``unit`` of template ``tid``."""
        allocs = self.allocs[tid]
        return allocs[bisect.bisect_right([a[0] for a in allocs], unit) - 1][2]

    # ---- gathers

    def _chunked(self, rows: tuple[Row, ...]) -> tuple[Row, ...]:
        """``rows`` with each run that a gather's first level holds as one chunk replaced by that chunk."""
        out, i = [], 0
        while i < len(rows):
            hit = self.chunks.get(rows[i])
            if hit is not None and rows[i:i + len(hit[0])] == hit[0]:
                out.append(hit[1])
                i += len(hit[0])
            else:
                out.append(rows[i])
                i += 1
        return tuple(out)

    def _whole(self, rowss: list[tuple[Row, ...]]) -> list[Row]:
        """One row a member holding its rows ``rowss[t]`` (module docstring): the row itself where it is one, a gather made
        before for those rows, else a new gather for every member, at one step where the members' rows are the first's
        shifted (`_step`), else a member at a time."""
        if all(len(r) == 1 for r in rowss):
            return [r[0] for r in rowss]
        if self.stand_ins and any(_stands(rs) for rs in rowss):
            return [rs[0] if len(rs) == 1 else self._cat(rs) for rs in rowss]
        if all(r in self.wholes for r in rowss):
            return [self.wholes[r] for r in rowss]
        if _step(rowss) is None:
            return [self._gather([r])[0] for r in rowss]
        return self._gather(rowss)

    def _gather(self, rowss: list[tuple[Row, ...]]) -> list[Row]:
        rows, M = rowss[0], len(rowss)
        s = _step(rowss) or 0
        widths = {r.words for r in rows}
        if len(widths) != 1:
            raise Unexpressible(f"{self.name}: a whole read of rows of {sorted(widths)} words: a gather reads rows of one width")
        n, c = len(rows), widths.pop()
        R = n * c
        if n + 1 <= GATHER_ROWS:
            tid, at = self.unit(GATHER, bind(B1T.AllGather, WORLD=n, N=c), M)
            self.b.feed(tid, "parts", at, M, rows, ((M, s),))
            got = [Row(tid, at + t, "o0", R) for t in range(M)]
        else:
            K = max((d for d in range(c, R + 1, c) if R % d == 0 and d // c + 1 <= GATHER_ROWS), default=c)
            k, G = K // c, R // K
            if k == 1:
                raise Unexpressible(f"{self.name}: a whole read of {n} rows of {c} words: no whole number of them within a "
                                    f"gather unit's {GATHER_ROWS} rows divides the {R} words")
            t1, a1 = self.unit(GATHER, bind(B1T.AllGather, WORLD=k, N=c), G * M)
            groups = [rows[g * k:(g + 1) * k] for g in range(G)]
            if (gs := _step(groups)) is not None:
                self.b.feed(t1, "parts", a1, G * M, groups[0], ((G, gs), (M, s)))
            else:
                for g, grp in enumerate(groups):
                    self.b.feed(t1, "parts", a1 + g * M, M, grp, ((M, s),))
            for t, rs in enumerate(rowss):
                for g in range(G):
                    self.chunks[rs[g * k]] = (rs[g * k:(g + 1) * k], Row(t1, a1 + g * M + t, "o0", K))
            got = self._gather([tuple(Row(t1, a1 + g * M + t, "o0", K) for g in range(G)) for t in range(M)])
        for rs, r in zip(rowss, got):
            self.wholes[rs] = r
        return got

    # ---- the window

    def circuit(self) -> IF.Circuit:
        """The lowered units settled (`_Builder.settle`) and emitted as one Circuit: its registered reads the weights', its
        blocks one slot (the request) at the Program's steps, its free ports the ones `free` lists."""
        if not self.lowered:
            self.lower()
        b = self.b
        if not b.fns:
            raise Unexpressible(f"{self.name}: no family of the Program's has a lowering, so the window has no unit")
        self._check_rows()
        if self.stand_ins and (bad := sorted({f"{f.tid} {f.port}" for f in b.feeds if _stands(f.rows)})):
            raise Unexpressible(f"{self.name}: stand-in rows reached `_Builder.feed` for {bad}: a lowering fed a take to the "
                                "builder itself, which ``stand_ins`` can't follow")
        b.settle()
        if bad := [f"{tid} {port} {k}" for (tid, port), (k, _) in self.row_fed.items() if b.cuts.get(tid, {}).get(port) != k]:
            raise Unexpressible(f"{self.name}: {len(bad)} row-fed ports' cuts changed after `feed_rows`: " + "; ".join(bad))
        roots, openings = b.merged()
        reg = IF.Registered(dict(sorted(b.params.items())), tuple(b.tables.values()), tuple(IF.coalesce(b.reads)),
                            tuple(roots), tuple(openings))
        wires = tuple(IF.coalesce(b.wires))
        starts = {tid: [at for at, _n, _slot, _step in spans] for tid, spans in b.spans.items()}
        return IF.Circuit(self.name, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()), wires=wires, cuts=b.cuts,
                          registered=reg, blocks=IF.Blocks(1, self.steps, tuple(inference(b.spans, 1))),
                          order=IF.ordered(b.units, wires, starts) if wires else None)

    def _check_rows(self) -> None:
        """Refuse, naming each, a port `feed_rows` fed that is also fed whole (`_Builder.feed`, or left free whole by
        `feed`), and a row of a unit it fed twice (wired twice, free twice, or wired and free)."""
        whole = set(self.b.read_rows) | {(f.tid, f.port) for f in self.free if f.rows[2] == 1}
        why = [f"{tid} {port}: fed row by row, cut {k} ways, and whole" for (tid, port), (k, _) in self.row_fed.items()
               if (tid, port) in whole]
        for (tid, port), (_k, fed) in self.row_fed.items():
            end: dict[int, tuple[int, str]] = {}
            for i, lo, hi, how in sorted(fed):
                if i in end and lo < end[i][0]:
                    why.append(f"{tid} {port}: row {i} of units from {lo} fed twice ({end[i][1]}, then {how})")
                if i not in end or hi > end[i][0]:
                    end[i] = (hi, how)
        if why:
            raise Unexpressible(f"{self.name}: {len(why)} row-fed ports fed past their rows: " + "; ".join(why))

    def census(self, circuit: IF.Circuit) -> dict[str, Any]:
        """The window's digest (SHA-512 of its description); by the family that emitted them, its units, templates and wire
        patterns before `interface.coalesce` (by the reader's), and in all its coalesced wire patterns; its free input ports
        by the family their rows are of (ports, unit ports and their words); the nodes left unlowered by family; and each
        ``peers_*`` input with the nodes that read it."""
        lay = {tid: dict(IF.layout(fn).rows) for tid, fn in self.b.fns.items()}
        units: Counter[str] = Counter()
        templates: Counter[str] = Counter()
        wires: Counter[str] = Counter()
        for tid, allocs in self.allocs.items():
            for _at, n, fam in allocs:
                units[fam] += n
            for fam in {a[2] for a in allocs}:
                templates[fam] += 1
        for w in self.b.wires:
            wires[self.family_of(w.reader, w.at)] += 1
        free: dict[str, Counter[str]] = {}
        for f in self.free:
            key = "peers" if f.kind == "peer" else f.family
            c = free.setdefault(key, Counter())
            c["unit_ports"] += f.count
            c["words"] += f.count * lay[f.tid][f.port] * f.rows[1] // f.rows[2]
            c["ports"] += 1
        return {"digest": hashlib.sha512(IF.circuit_bytes(circuit.description())).hexdigest(),
                "units": dict(sorted(units.items())), "templates": dict(sorted(templates.items())),
                "wires": dict(sorted(wires.items())), "wire_patterns": len(circuit.wires),
                "free": {k: dict(v) for k, v in sorted(free.items())},
                "unlowered": {k: len(v) for k, v in sorted(self.unlowered.items())},
                "peers": {p: [f"node {j} {fam}" for j, fam in v] for p, v in sorted(self.peers.items(), key=lambda x: _peer(x[0]))},
                "steps": self.steps}


def _family(node: Any) -> str:
    return getattr(node.fn, "name", None) or node.fn.id.split("{")[0]


def _stands(rows: Sequence[Row]) -> bool:
    return any(r.tid.startswith(STAND_IN) for r in rows)


def _peer(name: str) -> int:
    return int(name[len(PEERS):]) if name[len(PEERS):].isdecimal() else -1


def _lead(grid: Sequence[tuple[Row, ...]], shape: Sequence[int]) -> tuple[tuple[int, int], ...] | None:
    """The digits ``(radix, step)`` with unit i's rows ``grid[0]``'s shifted by i's digits times their steps, None where no
    such steps hold."""
    first = grid[0]
    steps, stride = [], math.prod(shape)
    for r in shape:
        stride //= r
        steps.append(first and r > 1 and grid[stride][0].unit - first[0].unit or 0)
    for i, rows in enumerate(grid):
        off, rest, st = 0, i, math.prod(shape)
        for r, s in zip(shape, steps):
            st //= r
            off += (rest // st) * s
            rest %= st
        if len(rows) != len(first) or any((a.tid, a.out, a.words, a.unit + off) != (b.tid, b.out, b.words, b.unit)
                                          for a, b in zip(first, rows)):
            return None
    return tuple(zip(shape, steps))


def _part(spec: str, family: str, **statics: Any) -> Any:
    """The template of ``family`` among the ones a row of ``spec`` holds instances of (`serving_templates.unit_templates`)
    at ``statics``, bound."""
    for t, _n in ST.unit_templates(spec):
        if spec_base(t) == family and all(str(spec_statics(t).get(k)) == str(v) for k, v in statics.items()):
            return ST.template(t).fn
    raise Unexpressible(f"{spec}: no {family} at {statics} among its unit templates")


def block_fp8_linear(w: NodeWindow, j: int, node: Any) -> None:
    """A block-FP8 linear (`deepgemm.DeepGemmFp8Linear`, `deepgemm.FlashInferFp8Linear`) as its ``PARTS``: a member's
    quantiser (`Fp8GroupQuantUe8m0_v1{K, 128}`, `FlashInferFp8Quant1x128_v1{K}`) reading its row x whole, then its GEMM's
    N / c coordinates (`ScaledMmFp8BlockSharedScale_v1{K, c, 128}`, c = ``w.coord``), coordinate-major, each reading the
    quantiser's q and s whole, its c weight rows and its 128-row block's scale row. DeepGEMM's row returns (q, s, y) and
    FlashInfer's y (its q and s are the GEMM's mids)."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    K, N, c = node.fn.bindings["K"], node.fn.bindings["N"], w.coord
    G = ST.PART_STATICS[spec_base(spec)]["G"]
    if N % c or G % c or K % G:
        raise Unexpressible(f"{spec}: coordinates of {c} outputs don't divide its N = {N} or its {G}-row scale blocks")
    (qfam, _, _), (gfam, _, _) = ST.PARTS[spec_base(spec)]
    quant, gemm = _part(spec, qfam), _part(spec, gfam, N=c, G=G)
    x = w.take(node, 0, whole=True)
    tq, aq = w.unit(fam, quant, M)
    w.feed(tq, "x", aq, x, (M,))
    C = N // c
    tg, ag = w.unit(fam, gemm, C * M)
    w.b.feed(tg, "xq", ag, C * M, (Row(tq, aq, "o0", K // 2),), ((C, 0), (M, 1)))
    w.b.feed(tg, "sx", ag, C * M, (Row(tq, aq, "o1", K // 64),), ((C, 0), (M, 1)))
    lo = w.weight(node.args[1].refs, N * K // 2)
    w.b.read(tg, "w", ag, (C, M), lo, (c * K // 2, 0))
    blocks, rest = divmod(N, G)
    slo = w.weight(node.args[2].refs, -(-N // G) * K // 64)
    if blocks:
        w.b.read(tg, "ws", ag, (blocks, G // c, M), slo, (K // 64, 0, 0))
    if rest:
        w.b.read(tg, "ws", ag + blocks * (G // c) * M, (rest // c, M), slo + blocks * K // 64, (0, 0))
    y = Piece(0, N, Row(tg, ag, "o0", c), M, 1)
    if fam == "DeepGemmFp8Linear":
        q, s = K // 2, K // 64
        w.out(j, node, [Piece(0, q, Row(tq, aq, "o0", q)), Piece(q, q + s, Row(tq, aq, "o1", s)),
                        Piece(q + s, q + s + N, y.row, M, 1)])
    else:
        w.out(j, node, [y])


def _cols(N: int, most: int) -> int:
    """The widest halving N / 2^k of a norm's N of at most ``most`` columns (`serving_templates.COLS`)."""
    G = N
    while most < G and G % 2 == 0:
        G //= 2
    return G


def scale_norm(w: NodeWindow, j: int, node: Any) -> None:
    """An RMSNorm cut at its scale (``PARTS``: `DsaRMSNorm_v1`, `RMSNormCudaVector8_v1`, `RMSNormFusedCudaVector8_v1/v3`):
    a member's sum of squares reading x (and the fused one's residual) whole, then N / G apply units, block-major, each
    reading its G columns of x (and the residual) as the rows that hold them (a gather's chunks where there are any), its G
    weights and the scale, G the widest halving of N within ``grain.norm`` (`_cols`). A row outside every decoder layer
    (its weight's label, `serving_templates.in_layer`) keeps its ``FINAL`` cut: one unit of the row a member."""
    spec, fam, M = descriptor_id(node.fn), _family(node), w.members(node)
    base, N = spec_base(spec), node.fn.bindings["N"]
    fused = len(node.args) == 3
    wk = len(node.args) - 1
    ports = ("x", "res") if fused else ("x",)
    outs = ("o0", "o1") if fused else ("o0",)
    wlo, label = w.weight(node.args[wk].refs, N), w.label(node.args[wk].refs, N)
    xs = [w.take(node, k, whole=True) for k in range(wk)]
    if not ST.in_layer(label):
        if ST.FINAL.get(base) != ():
            raise Unexpressible(f"{spec}: a row outside every decoder layer ({label}) whose FINAL cut isn't the row itself")
        t, a = w.unit(fam, node.fn, M)
        for k, port in enumerate(ports):
            w.feed(t, port, a, xs[k], (M,))
        w.b.read(t, "w", a, (M,), wlo, (0,))
        w.out(j, node, [Piece(i * N, (i + 1) * N, Row(t, a, o, N), 1, 1) for i, o in enumerate(outs)])
        return
    (sfam, _, _), (afam, _, _) = ST.PARTS[base]
    G = _cols(N, w.b.grain.norm)
    sumsq, apply = _part(spec, sfam), _part(spec, afam, G=G)
    A = N // G
    cols: list[list[tuple[Row, ...]] | _Free] = []
    for k in range(wk):
        blocks = [w.take(node, k, a * G, (a + 1) * G, most=w.b.grain.rows // 4) for a in range(A)]
        free = [b for b in blocks if isinstance(b, _Free)]
        cols.append(free[0] if free else [rows for b in blocks if not isinstance(b, _Free) for rows in b])
    ts, as_ = w.unit(fam, sumsq, M)
    for k, port in enumerate(ports):
        w.feed(ts, port, as_, xs[k], (M,))
    ta, aa = w.unit(fam, apply, A * M)
    for k, port in enumerate(ports):
        w.feed(ta, port, aa, cols[k], (A, M))
    w.b.read(ta, "w", aa, (A, M), wlo, (G, 0))
    scale = dict(IF.layout(apply).rows)
    sport = "scale" if "scale" in scale else "r"
    w.b.feed(ta, sport, aa, A * M, (Row(ts, as_, "o0", scale[sport]),), ((A, 0), (M, 1)))
    w.out(j, node, [Piece(i * N, (i + 1) * N, Row(ta, aa, o, G), M, 1) for i, o in enumerate(outs)])


#: the lowerings by family (the node's Definition's name)
FAMILIES: dict[str, Lowering] = {
    "DeepGemmFp8Linear": block_fp8_linear,
    "FlashInferFp8Linear": block_fp8_linear,
    "DsaRMSNorm": scale_norm,
    "RMSNormCudaVector8": scale_norm,
    "RMSNormFusedCudaVector8": scale_norm,
}


def node_window(fn: Any, families: Mapping[str, Lowering] | None = None, grain: Grain = _GRAIN, name: str = NAME,
                coord: int = 1, provable: bool = True, stand_ins: bool = False) -> tuple[IF.Circuit, NodeWindow]:
    """Program ``fn``'s window (module docstring) and the frame that lowered it."""
    w = NodeWindow(fn, families, grain, name, coord, provable, stand_ins).lower()
    return w.circuit(), w


__all__ = ["FAMILIES", "Free", "GATHER_ROWS", "NAME", "NodeWindow", "Piece", "SELECTS", "STAND_IN", "block_fp8_linear",
           "node_window", "scale_norm"]
