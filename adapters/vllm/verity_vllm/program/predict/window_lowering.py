"""The class Program's lowering into a window circuit's units (`window_circuit`, whose module docstring says what each
family's node becomes): `_Program` is one slot's class Program, lowered node by node into the units of the circuit's
`window_builder._Builder`; ``FAMILIES`` are the families it takes, and ``CUTS`` the `Attention` versions it cuts by key block.
"""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.refs import Refs, runs
from verity.circuits.types import Tuple
from verity.commitments.gates import embedding as E
from verity.commitments.periods import interface as IF

from verity_vllm.definitions.registry import int8_exact as IX
from verity_vllm.definitions.registry import io_shell as IO
from verity_vllm.definitions.registry import rmsnorm_fused_split as NS
from verity_vllm.definitions.registry.attention_cuts import CUTS, heads
from verity_vllm.definitions.registry.dead_free import (
    AttnBlockFirstV2,
    TokenSelectGroupsV1,
)

from .build import Unexpressible
from .window_builder import WEIGHTS, Row, _Builder, _spans, _step, _word, weight_run
from .window_class import FixedLoad, PearlTables, Request, Tables, WindowClass
from .by_body import family as unit_family
from .window_cut import Linears, cut_linear

PROMPT = "prompt"
#: a token select's group of logits: a word of each `TokenSelectGroups_v1` unit's row of bests
GROUP = 16


def _field(fn: Any, lo: int, hi: int) -> tuple[str, int]:
    """The output row of ``fn`` (its name and words) whose return leaves are exactly [lo, hi), refused where none is."""
    ret = fn.signature()[1]
    starts = [ret.field_offset(k) for k in range(len(ret.fields))] if isinstance(ret, Tuple) else [0]
    starts.append(ret.leaves)
    k = starts.index(lo) if lo in starts[:-1] else None
    if k is None or starts[k + 1] != hi:
        raise Unexpressible(f"{fn.id}: its return leaves [{lo}, {hi}) are not one of its output rows")
    return IF.layout(fn).outs[k]


@dataclass
class _Keys:
    """A layer's keys (or values) as its attention nodes read them: ``segs`` the word runs its positions so far are, and
    ``rows`` each position's rows a KV group, as their writers hold them."""
    segs: list[tuple[str, Any, int, int]] = field(default_factory=list)
    rows: list[tuple[tuple[Row, ...], ...]] = field(default_factory=list)


class _Program:
    """Slot ``slot``'s class Program lowered into ``b``'s units, its request its lane's of the scan state row before its anchor
    (`WindowClass.anchor`, `WindowClass.lane`) through ``scan``, its lane's binding (`scan_state.bridge`), its outputs the scan's through
    ``entry`` (`io_shell.SlotEntry_v1`, its last position's `SelectEntry_v1`'s): each node's output as the rows that hold it,
    ``(lo, hi, row)`` its words [lo, hi) as ``row`` and the units after it, (hi - lo) / ``row.words`` rows of one output at
    consecutive units (none for the last position's select, which nothing reads). With ``tables`` it reads H1's tables
    (`Tables`), or Pearl+'s (`PearlTables`) cut at its linears, ``cut`` their layout (`window_cut.linears`). With ``at``,
    a served request's Program instead, at the request's anchor and lane and of its positions (`_dead`, `_token`,
    `_collect`)."""

    def __init__(self, b: _Builder, fn: Any, cls: WindowClass, name: str, slot: int, scan: Any, entry: Any,
                 tables: FixedLoad | None = None, cut: Linears | None = None, at: Request | None = None) -> None:
        self.b, self.fn, self.cls, self.name, self.slot_index = b, fn, cls, name, slot
        self.anchor, self.lane = (cls.anchor(slot), cls.lane(slot)) if at is None else (at.anchor, at.lane)
        self.positions = cls.positions if at is None else at.positions
        self.scan, self.entry = scan, entry
        self.served = at is not None
        self.L, self.O = (cls.prompt, cls.outputs) if at is None else (at.prompt, at.outputs)
        self.dead = self._dead() if self.served else frozenset()
        #: each position's token rows (`_token`)
        self.tok_rows: dict[int, tuple[Row, ...]] = {}
        self.tables, self.cut = tables, cut
        if isinstance(tables, PearlTables) != (cut is not None):
            raise ValueError(f"{name}: a cut's layout goes with Pearl+'s tables, and only with them")
        params = [name for name, _ in self.fn.signature()[0]]
        if set(params) != {PROMPT, WEIGHTS}:
            raise Unexpressible(f"{name}: its inputs are {params}; the window's lowering takes a prompt and the weights record "
                                "(no seed, prefix-cache hit, carried state or peers)")
        self.params = params
        labels = [label for label, _ in dict(self.fn.signature()[0])[WEIGHTS].fields]
        self.fields = _spans(f"{name}: weight", dict(self.fn.signature()[0])[WEIGHTS].fields)
        b.weights = b.weights or tuple((label, s[1], e[1] - s[1]) for label, s, e in zip(labels, self.fields, self.fields[1:]))
        self.outs: dict[int, list[tuple[int, int, Row]]] = {}
        self.words: dict[int, list[tuple[int, int, int]]] = {}
        self.gemms: dict[int, tuple[Any, tuple[Row, ...], int]] = {}
        #: each node's output leaves at which a node's argument starts or ends (`_slices`)
        self.read_at: dict[int, set[int]] | None = None
        self.keys: dict[tuple[str, tuple[str, Any, int, int]], _Keys] = {}
        self.position = 0
        self.slot: Row | None = None
        self.occ: Row | None = None
        self.head: Row | None = None
        self.collect: Row | None = None
        #: the units a node makes beyond its output rows: each position's `TokenFeed_v1` unit, None at the first (the scan
        #: unit's) and past the class's prompt limit; each token select's group units and ``x0`` unit
        self.tokens: list[tuple[str, int] | None] = []
        self.selects: dict[int, tuple[tuple[tuple[str, int], ...], tuple[str, int]]] = {}
        #: each fused norm cut at its residual add: its parts' sums, in order of AT
        self.norms: dict[int, tuple[Row, ...]] = {}
        #: each attention cut by key block: its cut's plan (`_plan`) and each Call's first unit, a unit a head
        self.heads: dict[int, tuple[Any, tuple[tuple[str, int], ...]]] = {}
        #: its lane's `ScanSlots_v5` unit (`window_circuit.lower` gives it), each position's `SelectCollect_v1` unit but the
        #: last and its `SelectEntry_v1` unit
        self.scan_unit: tuple[str, int] | None = None
        self.collects: list[tuple[str, int]] = []
        self.entry_unit: tuple[str, int] | None = None
        #: a served Program's last step's label, its entry's (`entry`)
        self.entry_label: tuple[int, int] | None = None
        #: with ``tables``: each position's int8 linears of K = tables.k so far, and each such quantizer's H1 leaf row
        #: (`Int8RowQuant_v5`'s ``o2``, the node's words whole), which no reader reads
        self.quants: list[int] = []
        self.packed: dict[int, Row] = {}
        #: with Pearl+'s tables: each position's linears cut so far, and each cut linear's A row and y row (`cut_linear`)
        self.linears: list[int] = []
        self.arows: dict[int, Row] = {}
        self.ys: dict[int, Row] = {}
        #: each linear's runs of units, ``(position, tid, at, count)``: the calls PoUW credits (`WindowCompiler.schedule`)
        self.calls: list[tuple[int, str, int, int]] = []

    def segments(self, refs: Refs) -> list[tuple[str, Any, int, int]]:
        """``refs`` as word ranges: ``("n", node, lo, hi)`` of a node's output or ``("p", name, lo, hi)`` of a parameter."""
        out = []
        for space, idx, lo, hi in runs(refs):
            if space == "n":
                spans = self.words[idx]
                out.append(("n", idx, _word(spans, lo, f"{self.name}: node {idx}"), _word(spans, hi, f"{self.name}: node {idx}")))
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
        return weight_run(self.name, self.segments(refs), words)

    def _dead(self) -> frozenset[int]:
        """A served Program's nodes no live node reads: the live ones are its selects from its prompt's last position on and
        what they read, its feeds below its prompt length reading no select (`_token`). So at a prompt position before the
        last the LM head, the select and the final norm are dead, and the last layer's query and all of it from its
        attention on with them (`lower` keeps that attention's keys its layer's)."""
        nodes = self.fn.body.nodes
        fams = [node_family(n.fn) for n in nodes]
        pos, p = [], -1
        for f in fams:
            p += f == "Embedding"
            pos.append(p)
        readers: list[list[int]] = [[] for _ in nodes]
        for k, n in enumerate(nodes):
            if fams[k] == "Embedding" and 1 <= pos[k] < self.L:
                continue
            for a in n.args:
                for space, idx, _lo, _hi in runs(a.refs):
                    if space == "n":
                        readers[idx].append(k)
        live: set[int] = set()
        for k in reversed(range(len(nodes))):
            if (fams[k] == "TokenSelect" and pos[k] >= self.L - 1) or any(r in live for r in readers[k]):
                live.add(k)
        return frozenset(range(len(nodes))) - live

    def kind(self, j: int, node: Any) -> tuple[Any, ...]:
        """What node ``j`` allocates in this Program: Programs of one kind at a node sit side by side (`_Builder.among`)."""
        family = node_family(node.fn)
        if j in self.dead:
            return ("collect",) if family == "TokenSelect" else ("dead",)
        if family == "Embedding":
            p = self.position
            return ("token", 0 if p == 0 else 1 if p < self.L else 2)
        if family == "TokenSelect":
            return ("entry", self.cls.outputs - self.O if self.served else None) if self.position == self.positions else \
                ("select",)
        return ("node",)

    def lower(self, j: int, node: Any) -> None:
        """Node ``j`` lowered into units, each of the Program's slot and the node's step: its position, which each
        embedding opens (`_token`); a dead node none, a dead select its position's collector alone (`_collect_alone`)."""
        if node.form != "call":
            raise Unexpressible(f"{self.name}: node {j} is a {node.form}, not a call")
        family = node_family(node.fn)
        if j in self.dead:
            self.b.label, self.b.node = (self.slot_index, self.position - 1), j
            if family == "TokenSelect":
                self._collect_alone()
            elif family == "Attention":
                T, KVH = node.fn.bindings["T"], node.fn.bindings["KVH"]
                self._cache("k", node.args[1], T, KVH)
                self._cache("v", node.args[2], T, KVH)
            return
        ret = node.fn.signature()[1]
        fields = [f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,))]
        self.words[j] = _spans(f"{self.name}: node {j} ({node.fn.id}) return", [(f"o{i}", t) for i, t in enumerate(fields)])
        lower = getattr(self, f"_{family}", None) if family is not None else None
        if lower is None:
            raise Unexpressible(f"{self.name}: node {j} is {node.fn.id}, a family the window's lowering "
                                f"doesn't take ({sorted(FAMILIES)})")
        step = self.position - (family != "Embedding")
        if step < 0:
            raise Unexpressible(f"{self.name}: node {j} ({node.fn.id}) comes before its first position's embedding")
        self.b.label, self.b.node = (self.slot_index, step), j
        lower(j, node)

    def _Embedding(self, j: int, node: Any) -> None:
        V, H = node.fn.bindings["V"], node.fn.bindings["H"]
        tok, table = node.args
        tid, u = self.b.unit(E.opening(V, H))
        self.b.feed(tid, "tok", u, 1, self._token(tok))
        t = self.b.table(self.weight(table.refs, V * H), H, V)
        self.b.roots += [IF.TableRead(tid, "root", t.name, "root", u, 1), IF.TableRead(tid, "domain", t.name, "domain", u, 1)]
        self.b.openings.append(IF.TableOpening(tid, t.name, "tok", ("row", "salt", "path"), "o1", u, 1))
        self.outs[j] = [(0, H, Row(tid, u, "o0", H))]

    def _token(self, tok: Any) -> tuple[Row, ...]:
        """The next position's token: the first's its lane's `ScanSlots_v5` unit's, of the scan state row at its anchor
        (with the slot it carries, its occupancy word and its head), below the class's prompt limit every other's
        `TokenFeed_v1`'s (its slot its predecessor's, ``sel`` the previous position's select), past it the previous select.
        A served Program's below its prompt length is `TokenFeed_v1{FIRST=True}`'s, its prompt's id, and past it the previous
        select. With ``tables``, that row is held to the position's row of its round's input-ids table (`Tables.at`)."""
        p, NP = self.position, self.cls.prompt
        self.position += 1
        space, idx, _lo, _hi = runs(tok.refs)[0]
        read = (space, idx if space == "n" else self.params[idx])
        if (read == ("p", PROMPT)) != (p == 0):
            raise Unexpressible(f"{self.name}: position {p}'s token reads {read}; a class Program reads its one prompt row "
                                "at position 0 and every other position's token from its predecessor's select")
        if p == 0:
            assert self.scan_unit is not None, f"{self.name}: its lane's scan unit before its first position"
            tid, u = self.scan_unit
            self.slot, self.occ, self.head = Row(tid, u, "o1", 2 * (NP + 1)), Row(tid, u, "o2", 2), Row(tid, u, "o3", 4)
            self.tokens.append(None)
            rows: tuple[Row, ...] = (Row(tid, u, "o0", 2),)
        elif p >= self.L:
            self.tokens.append(None)
            rows = self.rows(tok.refs)
        else:
            tid, u = self.b.unit(bind(IO.TokenFeedV2, NP=NP, FIRST=self.served))
            assert self.slot is not None, f"{self.name}: position {p}'s token before position 0's, which carries its slot"
            self.b.feed(tid, "slot", u, 1, (self.slot,))
            if not self.served:
                self.b.feed(tid, "sel", u, 1, self.rows(tok.refs))
            self.tokens.append((tid, u))
            self.slot = Row(tid, u, "o1", 2 * (NP + 1))
            rows = (Row(tid, u, "o0", 2),)
        self.tok_rows[p] = rows
        if isinstance(self.tables, Tables):
            if len(rows) != 1 or rows[0].words != 2:
                raise Unexpressible(f"{self.name}: position {p}'s token is rows {rows}, not one u32 row an input-ids row holds")
            s, i = self.tables.at(self.cls, self.slot_index, p)
            self.b.value(rows[0].tid, rows[0].out, rows[0].unit, self.tables.ids(s), i, self.tables.m)
            self.quants.append(0)
        elif self.cut is not None:
            self.linears.append(0)
        return rows

    def _one(self, j: int, node: Any, fn: Any = None) -> tuple[str, int]:
        """A node that is one unit of its own Definition, or of ``fn`` where given (one over the same leaves, its ports
        other shapes of them, its first outputs the node's return fields): each parameter a weight read or its writers'
        rows. -> the unit."""
        fn = node.fn if fn is None else fn
        tid, u = self.b.unit(fn)
        lay = IF.layout(fn)
        for (p, words), arg in zip(lay.rows, node.args):
            segs = self.segments(arg.refs)
            if segs[0][:2] == ("p", WEIGHTS):
                self.b.read(tid, p, u, (1,), self.weight(arg.refs, words), (0,))
            else:
                self.b.feed(tid, p, u, 1, self.rows(arg.refs))
        outs = lay.outs[:len(self.words[j]) - 1]
        ends = list(accumulate((w for _, w in outs), initial=0))
        self.outs[j] = [(a, e, Row(tid, u, o, w)) for (o, w), a, e in zip(outs, ends, ends[1:])]
        return tid, u

    _RMSNormTriton = _one

    def _RMSNormFusedCuda(self, j: int, node: Any) -> None:
        """`RMSNormFusedCuda_v2{N, EPS}`, one unit, or cut at its residual add where an earlier node reads its template's ``x``
        at another row count or one unit would pass ``grain.rows`` rows: N / C `RMSNormFusedCudaAdd_v2{N, C, AT, R}` units, C =
        ``grain.norm`` and R the rows their writers hold its x part in, and the tail `RMSNormFusedCudaTail_v1{N, EPS}`, its ``r`` cut j part AT = j's sum."""
        x, res, w = node.args
        (N, eps), xs, rs = (node.fn.bindings[k] for k in ("N", "EPS")), self.rows(x.refs), self.rows(res.refs)
        if self.b.read_rows.get((descriptor_id(node.fn), "x"), len(xs)) == len(xs) and \
                len(xs) + len(rs) + 3 <= self.b.grain.rows:
            self._one(j, node)
            return
        C = min(self.b.grain.norm, N)
        if N % C:
            raise Unexpressible(f"{node.fn.id}: its {N} elements aren't parts of the grain's {C}")
        parts = []
        for at in range(N // C):
            part = self.rows(x.refs.slice(at * C, (at + 1) * C))
            tid, u = self.b.unit(bind(NS.RMSNormFusedCudaAdd, N=N, C=C, AT=at, R=len(part)))
            self.b.feed(tid, "x", u, 1, part)
            self.b.feed(tid, "res", u, 1, rs)
            parts.append(Row(tid, u, "o0", C))
        tid, u = self.b.unit(bind(NS.RMSNormFusedCudaTail, N=N, EPS=eps))
        self.b.feed(tid, "r", u, 1, parts)
        self.b.read(tid, "w", u, (1,), self.weight(w.refs, N), (0,))
        self.outs[j] = [(0, N, Row(tid, u, "o0", N)), (N, 2 * N, Row(tid, u, "o1", N))]
        self.norms[j] = tuple(parts)

    def _Int8RowQuant(self, j: int, node: Any) -> None:
        """`Int8RowQuant_v1{K}` as one unit `Int8RowQuant_v3{K, R}`, its ``x`` the R rows its writers hold it in (one after a
        norm, a head's at o_proj), so each template's ``x`` is read at one row count; refused where R doesn't divide K. At
        the K of ``tables``, `Int8RowQuant_v5{K, R, ROUND}` at its position's round, whose ``o2`` is held to row c·m + i of
        its round's A table: c its index among its position's int8 linears of that K, i its position's row (`Tables.at`)."""
        R, K = len(self.rows(node.args[0].refs)), node.fn.bindings["K"]
        if K % R:
            raise Unexpressible(f"{node.fn.id}: its x is {R} writer rows, which don't divide its {K} values")
        t = self.tables
        if not isinstance(t, Tables) or t.k != K:
            self._one(j, node, bind(IX.Int8RowQuantV3, K=K, R=R))
            return
        s, i = t.at(self.cls, self.slot_index, self.position - 1)
        fn = bind(IX.Int8RowQuantV5, K=K, R=R, ROUND=s)
        tid, u = self._one(j, node, fn)
        self.packed[j] = Row(tid, u, "o2", dict(IF.layout(fn).outs)["o2"])
        self.b.value(tid, "o2", u, t.a(s), self.quants[-1] * t.m + i, t.calls * t.m)
        self.quants[-1] += 1

    def _SiluMul(self, j: int, node: Any) -> None:
        """`SiluMul_v2{I}` as P units `SiluMul_v2{I / P}`, unit k reading gate and up elements [k·I/P, (k+1)·I/P) as the
        gate-up GEMM's rows that hold them, P the fewest that leaves a unit within ``grain.rows`` rows."""
        n, (gu,) = node.fn.bindings["I"], node.args
        for P in (p for p in range(1, n + 1) if n % p == 0):
            w = n // P
            try:
                pieces = [self.rows(gu.refs.slice(k * w, (k + 1) * w)) + self.rows(gu.refs.slice(n + k * w, n + (k + 1) * w))
                          for k in range(P)]
            except Unexpressible:
                continue
            if max(map(len, pieces)) + 1 <= self.b.grain.rows:
                break
        else:
            raise Unexpressible(f"{node.fn.id}: no cut of its gate and up rows holds a unit within {self.b.grain.rows} rows")
        tid, u = self.b.unit(bind(node.fn.defn, I=w), P)
        self._each(tid, "gu", u, pieces)
        self.outs[j] = [(0, n, Row(tid, u, "o0", w))]

    def _slices(self, j: int, N: int, n: int) -> list[tuple[int, int, int]]:
        """Node ``j``'s N columns as slices of n, the last of N mod n, in runs ``(first column, columns a slice, slices)``:
        an allocation a run, a run ending where a node's argument starts or ends inside node ``j``'s output, so a reader of
        a range of each slot's slices (a head of a fused projection) reads one allocation."""
        if self.read_at is None:
            self.read_at = {}
            for nd in self.fn.body.nodes:
                for a in nd.args:
                    for space, idx, lo, hi in runs(a.refs):
                        if space == "n":
                            self.read_at.setdefault(idx, set()).update((lo, hi))
        full, rest = divmod(N, n)
        words = {_word(self.words[j], x, f"{self.name}: node {j}") for x in self.read_at.get(j, ())}
        cuts = [0, *sorted({c // n for c in words if 0 < c < full * n and c % n == 0}), full]
        return [(a * n, n, e - a) for a, e in zip(cuts, cuts[1:])] + ([(full * n, rest, 1)] if rest else [])

    def _Gemm(self, j: int, node: Any) -> None:
        if self.cut is not None and j not in self.cut.heads:
            cut_linear(self, j, node)
            return
        b = node.fn.bindings
        K, N = b["K"], b["N"]
        x, w = node.args
        xs, lo = self.rows(x.refs), self.weight(w.refs, N * K)
        n = max(1, min(N, self.b.grain.gemm_words // K))
        while n > 1 and any(self.b.read_rows.get((descriptor_id(bind(node.fn.defn, **{**b, "N": m})), "x"), len(xs)) != len(xs)
                            for m in (n, N % n) if m):
            n //= 2
        pieces = []
        for s0, m, count in self._slices(j, N, n):
            if not count:
                continue
            tid, u = self.b.unit(bind(node.fn.defn, **{**b, "N": m}), count)
            self.b.feed(tid, "x", u, count, xs)
            self.b.read(tid, "w", u, (count,), lo + s0 * K, (m * K,))
            pieces.append((s0, s0 + count * m, Row(tid, u, "o0", m)))
            self.calls.append((self.position - 1, tid, u, count))
        self.outs[j] = pieces
        self.gemms[j] = (node.fn, xs, lo)

    def _ScaledMmS8(self, j: int, node: Any) -> None:
        b = node.fn.bindings
        K, N = b["K"], b["N"]
        n = max(1, min(N, self.b.grain.gemm_words // (K // 2)))
        x, sx, w, sw = node.args
        xs, ss = self.rows(x.refs), self.rows(sx.refs)
        lo, slo = self.weight(w.refs, N * K // 2), self.weight(sw.refs, 2 * N)
        pieces = []
        for s0, m, count in self._slices(j, N, n):
            if not count:
                continue
            tid, u = self.b.unit(IX.scaled_mm_words(K, m), count)
            self.b.feed(tid, "x", u, count, xs)
            self.b.feed(tid, "sx", u, count, ss)
            self.b.read(tid, "w", u, (count,), lo + s0 * K // 2, (m * K // 2,))
            self.b.read(tid, "sw", u, (count,), slo + 2 * s0, (2 * m,))
            pieces.append((s0, s0 + count * m, Row(tid, u, "o0", m)))
            self.calls.append((self.position - 1, tid, u, count))
        self.outs[j] = pieces

    def _heads(self, fn: Any, family: str, count: int) -> Any:
        try:
            return heads(fn, family, count)
        except ValueError as e:
            raise Unexpressible(str(e)) from None

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
        """``fn``'s heads as its cut's (`CUTS`: `Attention_v19`'s or v20's), read once a Definition (`_Builder.plans`): each
        Call of the cut's head's body in order with what each of its ports reads, ``("q", ())``, ``("k" | "v", (a, b))`` (keys
        [a, b)) or ``("state", (i, out, words))`` (Call i's output row ``out``), the last Call's ``o0`` the head's output."""
        key = descriptor_id(fn)
        if key in self.b.plans:
            return self.b.plans[key]
        cut = CUTS[fn.version]
        try:
            head = self._heads(bind(cut, **fn.bindings), "AttentionHead", NH)
            body = head.body
        except (TypeError, ValueError, KeyError) as e:
            raise Unexpressible(f"{fn.id}: its heads are not `AttentionHead_v{cut.version}`'s ({e})") from None
        params, D = [p for p, _ in head.signature()[0]], fn.bindings["D"]
        plan = []
        for i, n in enumerate(body.nodes):
            ports = []
            for (port, _), a in zip(IF.layout(n.fn).rows, n.args):
                (space, idx, lo, hi), *more = runs(a.refs)
                name = params[idx] if space == "p" else None
                src: tuple[str, tuple] | None
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
        """The layer's ``port`` (keys or values) at T keys: its earlier nodes' T - 1 and position T - 1's rows a KV group,
        refused unless ``c`` is those T - 1 and one more."""
        segs = self.segments(c.refs)
        ks = self.keys.setdefault((port, segs[0]), _Keys())
        if len(ks.rows) != T - 1 or segs[:len(ks.segs)] != ks.segs:
            raise Unexpressible(f"{self.name}: an attention's {port} of {T} keys is not its layer's {len(ks.rows)} earlier keys "
                                "and one more")
        n = c.refs.count // T
        ks.rows.append(tuple(self._split(c.refs.slice((T - 1) * n, T * n), KVH)))
        ks.segs = segs
        return ks

    @staticmethod
    def _block(ks: _Keys, a: int, b: int) -> list[tuple[Row, ...]]:
        """Keys [a, b)'s rows a KV group, key by key: each key's rows as its writers hold them."""
        return [tuple(r for key in ks.rows[a:b] for r in key[g]) for g in range(len(ks.rows[a]))]

    def _Attention(self, j: int, node: Any) -> None:
        T, NH, KVH, D = (node.fn.bindings[k] for k in ("T", "NH", "KVH", "D"))
        if node.fn.version not in CUTS:
            self._whole_heads(j, node)
            return
        plan = self._plan(node.fn, NH)
        q, kc, vc = node.args
        keys = {"k": self._cache("k", kc, T, KVH), "v": self._cache("v", vc, T, KVH)}
        qs = self._split(q.refs, NH)
        calls: list[tuple[str, int]] = []
        plan = [(self._first(fn, ports, keys["v"]), ports) for fn, ports in plan]
        self._bound(T, *plan[0], keys, qs)
        for fn, ports in plan:
            tid, u = self.b.unit(fn, NH)
            for port, (kind, x) in ports:
                if kind == "q":
                    self._each(tid, port, u, qs)
                elif kind == "state":
                    ctid, cu = calls[x[0]]
                    self.b.feed(tid, port, u, NH, (Row(ctid, cu, x[1], x[2]),), ((NH, 1),))
                else:
                    self._each(tid, port, u, self._block(keys[kind], *x), NH // KVH)
            calls.append((tid, u))
        self.outs[j] = [(0, NH * D, Row(tid, u, "o0", D))]
        self.heads[j] = (plan, tuple(calls))

    def _bound(self, T: int, fn: Any, ports: Any, keys: dict[str, _Keys], qs: Sequence[tuple[Row, ...]]) -> None:
        """Refuse T keys where the first visited key block's unit, a head's rows in and out, passes ``grain.rows``, naming
        the most positions a Program holds at this model and grain: its fixed rows and its rows a key. A grain no key fits
        is `settle`'s to refuse, with every other unit past it."""
        n = [len(qs[0]) if kind == "q" else 1 if kind == "state" else len(self._block(keys[kind], *x)[0])
             for _port, (kind, x) in ports]
        a, b = next(x for _port, (kind, x) in ports if kind == "k")
        total, R = sum(n) + len(IF.layout(fn).outs), self.b.grain.rows
        if total <= R:
            return
        per = sum(len(self._block(keys[k], a, b)[0]) for k in ("k", "v")) // (b - a)
        fixed = total - (b - a) * per
        if fixed + per > R:
            return
        raise Unexpressible(f"{self.name}: position {T - 1}, past the {(R - fixed) // per} positions a Program holds at this model and "
                            f"grain: its first key block at {T} keys is {total} rows in and out ({fixed} and {per} a key), "
                            f"past a unit's {R}")

    def _first(self, fn: Any, ports: Any, vs: _Keys) -> Any:
        """``fn``, or for `AttnBlockFirst_v1` whose keys' values are each several rows, `AttnBlockFirst_v2` at VS their
        width, so its ``v`` is cut a row a value slice."""
        if (fn.name, fn.version) != ("AttnBlockFirst", 1):
            return fn
        (a, b), = (x for _port, (kind, x) in ports if kind == "v")
        if len(rows := self._block(vs, a, b)[0]) == b - a:
            return fn
        if len(width := {r.words for r in rows}) != 1:
            raise Unexpressible(f"{fn.id}: its keys' values are rows of {sorted(width)} words, not one slice width")
        return bind(AttnBlockFirstV2, **fn.bindings, VS=width.pop())

    def _whole_heads(self, j: int, node: Any) -> None:
        """An attention whose version has no per-key-block cut (not in `CUTS`): a unit a head of its own head Definition,
        reading its KV group's every key's rows (`_cache`'s), so a head past a unit's rows is refused at `settle`."""
        T, NH, KVH, D = (node.fn.bindings[k] for k in ("T", "NH", "KVH", "D"))
        head = self._heads(node.fn, "AttentionHead", NH)
        q, kc, vc = node.args
        tid, u = self.b.unit(head, NH)
        self._each(tid, "q", u, self._split(q.refs, NH))
        for port, c in (("k", kc), ("v", vc)):
            self._each(tid, port, u, self._block(self._cache(port, c, T, KVH), 0, T), NH // KVH)
        self.outs[j] = [(0, NH * D, Row(tid, u, "o0", D))]

    def _TokenSelect(self, j: int, node: Any) -> None:
        V, nodes = node.fn.bindings["V"], node.fn.body.nodes
        chunk, combine = nodes[0].fn, nodes[-1].fn
        if chunk.name != "TokenSelectChunk" or (combine.name, combine.version) != ("TokenSelectCombine", 2) or V < 2:
            raise Unexpressible(f"{node.fn.id}: its body is not chunks and one `TokenSelectCombine_v2`")
        C = min(self.b.grain.select, V)
        if V % C or (C % GROUP and C < V):
            raise Unexpressible(f"{node.fn.id}: {V} logits aren't units of {C}, each groups of {GROUP}: bind the grain's "
                                f"select to a divisor of V that is a multiple of {GROUP}")
        (logits,), n, parts = node.args, -(-C // GROUP), []
        if self.occ is None or self.head is None:
            raise Unexpressible(f"{node.fn.id}: a token select before its Program's first position's token")
        for lo in range(0, V, C):
            tid, u = self.b.unit(bind(TokenSelectGroupsV1, C=C, G=GROUP))
            self.b.feed(tid, "x", u, 1, self.rows(logits.refs.slice(lo, lo + C)))
            parts.append((tid, u))
        ctid, cu = self._collect(V, self.head, self.occ)
        self.b.feed(ctid, "best", cu, 1, [Row(t, u, "o0", n) for t, u in parts])
        self.b.feed(ctid, "idx", cu, 1, [Row(t, u, "o1", 2 * n) for t, u in parts])
        (_, lm, lo, _), = self.segments(logits.refs.slice(0, 1))
        if lm not in self.gemms:
            raise Unexpressible(f"{node.fn.id}: its logits are node {lm}'s, not a GEMM's")
        fn, xs, wlo = self.gemms[lm]
        xtid, xu = self.b.unit(bind(fn.defn, **{**fn.bindings, "N": 1}))
        self.b.feed(xtid, "x", xu, 1, xs)
        self.b.read(xtid, "w", xu, (1,), wlo + lo * fn.bindings["K"], (0,))
        self.b.feed(ctid, "x0", cu, 1, (Row(xtid, xu, "o0", 1),))
        if (ctid, cu) != self.entry_unit:
            self.outs[j] = [(0, 2, Row(ctid, cu, "o0", 2))]
        self.selects[j] = (tuple(parts), (xtid, xu))

    def _collect(self, V: int, head: Row, occ: Row) -> tuple[str, int]:
        """The position's combine unit, the slot's outputs collected on with its select (`io_shell.SelectCollect_v1`, its
        state the next position's ``state``: the first's reads its lane's head), and at the last position the slot's entry
        too (`io_shell.SelectEntry_v1`, reading its occupancy word), whose select is no row: its one output is the entry. With
        H1's tables, a select that is a later round's token (`_token`) is collected at that round (`io_shell.SelectCollect_v2`).
        A served Program's last select is collected on too, then KO - O more collectors shift its O outputs to the front of
        the state, and its entry is `io_shell.SlotEntry_v1{NP=1}` (``entry``, `place_entry`), which fits only where the
        scan's prompt length is its own."""
        p, KO = self.position - 1, self.cls.outputs
        at = {"V": V, "C": GROUP, "KO": KO, "FIRST": p == 0}
        e = self.entry.bindings
        last = p == self.positions - 1
        if last and not self.served:
            fn = bind(IO.SelectEntry, **at, IDB=e["IDB"], NP=e["NP"], STOP=e["STOP"])
        elif isinstance(self.tables, Tables) and p + 1 >= self.cls.prompt:
            fn = bind(IO.SelectCollectV2, **at, ROUND=self.tables.at(self.cls, self.slot_index, p + 1)[0])
        else:
            fn = bind(IO.SelectCollect, **at)
        ctid, cu = self.b.unit(fn)
        state = head if p == 0 else self.collect
        assert state is not None, f"{self.name}: position {p}'s select before position {p - 1}'s, which collects on to it"
        self.b.feed(ctid, "head" if p == 0 else "state", cu, 1, (state,))
        if last and not self.served:
            self.b.feed(ctid, "occ", cu, 1, (occ,))
            self.entry_unit = (ctid, cu)
            return ctid, cu
        self.collect = Row(ctid, cu, "o1", 2 * (KO + 2))
        self.collects.append((ctid, cu))
        if last:
            tok = (Row(ctid, cu, "o0", 2),)
            for _ in range(KO - self.O):
                self._collect_alone(tok)
            self.entry_label = self.b.label
        return ctid, cu

    def place_entry(self, tid: str, u: int) -> None:
        """A served Program's entry, unit ``u`` of the window's entries (`io_shell.SlotEntry_v1{NP=1}`), on its occupancy
        word and its last collector's state."""
        if self.occ is None or self.collect is None or self.entry_label is None:
            raise Unexpressible(f"{self.name}: an entry before its last position's select")
        self.b.feed(tid, "occ", u, 1, (self.occ,))
        self.b.feed(tid, "state", u, 1, (self.collect,))
        self.entry_unit = (tid, u)

    def place_empty(self, scan: tuple[str, int], collect: tuple[str, int], entry: tuple[str, int]) -> None:
        """An empty lane's collector (`io_shell.OutCollect_v1{FIRST}`, unit ``collect``) on the head and first token of its
        lane's `ScanSlots_v5` unit ``scan``, and its entry (unit ``entry`` of the window's entries) on that unit's occupancy
        word and the collector's state: units of this Program's slot, so no wire leaves it."""
        (tid, u), (ct, cu), (et, eu) = scan, collect, entry
        self.b.feed(ct, "head", cu, 1, (Row(tid, u, "o3", 4),))
        self.b.feed(ct, "tok", cu, 1, (Row(tid, u, "o0", 2),))
        self.b.feed(et, "occ", eu, 1, (Row(tid, u, "o2", 2),))
        self.b.feed(et, "state", eu, 1, (Row(ct, cu, "o0", 2 * (self.cls.outputs + 2)),))

    def _collect_alone(self, tok: tuple[Row, ...] | None = None) -> None:
        """A collector with no select (`io_shell.OutCollect_v1`): at a prompt position before the last, whose select is
        dead, its token ``tok`` the position's; after a served Program's last select, its pads."""
        p, KO = self.position - 1, self.cls.outputs
        first = p == 0 and not self.collects
        if self.head is None or (not first and self.collect is None):
            raise Unexpressible(f"{self.name}: position {p}'s collector before its Program's first position's token")
        ctid, cu = self.b.unit(bind(IO.OutCollectV2, KO=KO, FIRST=first))
        self.b.feed(ctid, "head" if first else "state", cu, 1, (self.head if first else self.collect,))
        self.b.feed(ctid, "tok", cu, 1, self.tok_rows[p] if tok is None else tok)
        self.collect = Row(ctid, cu, "o0", 2 * (KO + 2))
        self.collects.append((ctid, cu))


FAMILIES = ("Embedding", "RMSNormTriton", "RMSNormFusedCuda", "Gemm", "Int8RowQuant", "ScaledMmS8", "RoPE", "Attention", "SiluMul",
            "TokenSelect")


def node_family(fn: Any) -> str | None:
    """``fn``'s family among `FAMILIES` by `by_body.family`: the lowering takes a unit by its family's name."""
    return unit_family(fn, FAMILIES, "the window lowers a unit by its family's name")
