"""I2's served window as one Circuit (VLLM-103's V3b, `verity.ml.windows.interface`): every request Program `compile_steps`
compiled for the window, flattened into proof units of the catalog's Definitions, every unit's output a committed row and
every input row wired, registered or ingress.

`window_circuit(compiled, grain)` walks each request Program's nodes in order and lowers each to units at ``grain``:

- `Embedding_v1{V, H}` is the registered read `EmbeddingRead_v1{V, H}` (`commitments.gates.embedding`): its ``root`` and
  ``domain`` are the embedding table's (a `Table` over the weights record, read by `TableRead`), its ``row``, ``salt`` and
  ``path`` an `Opening` at the hidden key ``tok``, pinned by its verdict ``o1``. A prompt position's ``tok`` is the system's
  ingress at the step whose row first reaches it (`Ingress`); a decode position's is the previous position's token select.
- `Gemm_v2{K, N}` is ⌈N / n⌉ slices `Gemm_v2{K, n}`, n = ``grain.gemm_words`` // K output columns a slice, each reading
  its weight rows from the weights record (`Read`) and its ``x`` whole.
- `RoPE_v1{NHEADS, D}` and `Attention_v18{T, NH, KVH, D}` are their heads (`RoPEHead_v1{D}`, `AttentionHead_v18{T}`).
  A head's keys and values are one row of an `AllGather_v1{T, D}` over every key position's row (T at most `KEYS`), each
  position's row gathered first when it spans several writers (the K and V slices).
- `TokenSelect_v4{V}` is its function at ``grain.select`` logits a chunk (its body's own form at another C): chunks
  `TokenSelectChunk_v2{C}`, each reading its C logits through one gather, and `TokenSelectCombine_v2{V, C}` over them:
  ``best`` through one gather of the chunks' maxima, ``idx`` cut a row a chunk, and ``x0`` (``logits[0]``) from one
  `Gemm_v2{K, 1}` unit over the LM head's input and its first weight row.
- `RMSNormTriton_v5`, `RMSNormFusedCuda_v2` and `SiluMul_v2` are one unit each.

A reader port that reads several writers' rows is cut (`interface.Layout.cut`), one cut per template and port: the cut ``d``
divides every use's row count, holds at most ``grain.rows`` rows, and makes the fewest gather units, a use's rows gathered
``m / d`` at a time by `AllGather_v1` units (whose ``parts`` are cut a row a part). Every wire and read is stated per node as
a pattern and `interface.coalesce`d, so the circuit's description grows with the Programs' nodes, not their units.

Refused by name (`Unexpressible`): a request Program reading more than its prompt and the weights (a stochastic request's
seed, a prefix-cache hit's prefix, a TP rank's peers), a family this lowering doesn't know, a request a Commit window carries
in or out, byte ingress, a weight whose leaves aren't whole u16 words, and a row form the cut rule can't state. The circuit is completeness only: that it computes what serving ran is held by the units' proofs
on their committed rows, never by this lowering.
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
from verity.ml.commitments.gates import embedding as E
from verity.ml.definitions.registry.b1_tp2 import AllGather
from verity.ml.windows import interface as IF

from .build import Unexpressible

NAME = "verity-vllm/window"
WEIGHTS = "weights"
PROMPT = "prompt"


#: the most keys an attention head reads through one `AllGather_v1{T, D}`, until v19's per-key-block cut is wired
#: (`AttentionHead_v19`, CIR-96)
KEYS = 136


@dataclass(frozen=True)
class Grain:
    """The units' size (CIR-95's at Llama-3.2-1B by default): a GEMM slice reads ``gemm_words`` weight words (N = gemm_words
    // K output columns: `Gemm_v2{2048, 16}`, `Gemm_v2{8192, 4}`), no unit reads a port as more than ``rows`` rows nor
    gathers more than ``rows``, and a token select is chunks of ``select`` logits (768: 167 chunks at V = 128,256, each
    chunk's row past one SHA-512 block)."""
    gemm_words: int = 32768
    rows: int = 256
    select: int = 768


@dataclass(frozen=True)
class Row:
    """One unit's output row: template ``tid``'s unit ``unit``, its output ``out`` of ``words`` u16 words."""
    tid: str
    unit: int
    out: str
    words: int


@dataclass
class _Feed:
    """Units ``[at, at + count)`` of template ``tid`` all read port ``port`` as ``rows``, in order."""
    tid: str
    port: str
    at: int
    count: int
    rows: tuple[Row, ...]


@dataclass
class _Builder:
    grain: Grain
    fns: dict[str, Any] = field(default_factory=dict)
    units: dict[str, int] = field(default_factory=dict)
    feeds: list[_Feed] = field(default_factory=list)
    cuts: dict[str, dict[str, int]] = field(default_factory=dict)
    wires: list[IF.Wire] = field(default_factory=list)
    reads: list[IF.Read] = field(default_factory=list)
    tables: dict[tuple[int, int, int], IF.Table] = field(default_factory=dict)
    roots: list[IF.TableRead] = field(default_factory=list)
    openings: list[IF.Opening] = field(default_factory=list)
    ingress: list[IF.Ingress] = field(default_factory=list)
    gathered: dict[tuple[Row, ...], Row] = field(default_factory=dict)

    def unit(self, fn: Any, count: int = 1) -> tuple[str, int]:
        """``count`` new units of ``fn``: its descriptor id and the first one's index."""
        tid = descriptor_id(fn)
        self.fns.setdefault(tid, fn)
        at = self.units.get(tid, 0)
        self.units[tid] = at + count
        return tid, at

    def feed(self, tid: str, port: str, at: int, count: int, rows: Sequence[Row]) -> None:
        if not rows:
            raise Unexpressible(f"{tid} {port}: reads no unit's rows")
        self.feeds.append(_Feed(tid, port, at, count, tuple(rows)))

    def gather(self, rows: Sequence[Row]) -> Row:
        """One row holding ``rows`` in order: the row itself, or an `AllGather_v1{len(rows), w}` unit's (one a row sequence)."""
        rows = tuple(rows)
        if len(rows) == 1:
            return rows[0]
        if rows in self.gathered:
            return self.gathered[rows]
        w = {r.words for r in rows}
        if len(w) != 1 or len(rows) > self.grain.rows:
            raise Unexpressible(f"a gather of {len(rows)} rows of {sorted(w)} words: rows of one width, at most {self.grain.rows}")
        tid, u = self.unit(bind(AllGather, WORLD=len(rows), N=w.pop()))
        self.cuts.setdefault(tid, {})["parts"] = len(rows)
        self._wire(tid, "parts", u, 1, rows)
        out = self.gathered[rows] = Row(tid, u, "o0", len(rows) * rows[0].words)
        return out

    def read(self, tid: str, port: str, at: int, radix: Sequence[int], base: int, step: Sequence[int]) -> None:
        self.reads.append(IF.Read(tid, port, WEIGHTS, at=at, radix=tuple(radix), base=base, step=tuple(step)))

    def _wire(self, tid: str, port: str, at: int, count: int, rows: Sequence[Row]) -> None:
        """Units ``[at, at + count)`` read ``rows`` as their port's rows ``0 .. len(rows) - 1``: one pattern per run of rows from
        one writer's output at a constant unit step."""
        j = 0
        while j < len(rows):
            k = j + 1
            while k < len(rows) and (rows[k].tid, rows[k].out) == (rows[j].tid, rows[j].out) and \
                    rows[k].unit - rows[k - 1].unit == rows[j + 1].unit - rows[j].unit:
                k += 1
            C, r = k - j, rows[j]
            digits = ([(count, 0)] if count > 1 else []) + ([(C, rows[j + 1].unit - r.unit)] if C > 1 else []) or [(1, 0)]
            self.wires.append(IF.Wire(tid, port, r.tid, r.out, at=at, radix=tuple(a for a, _ in digits), base=r.unit,
                                      step=tuple(s for _, s in digits), cut=(j, C) if len(rows) > 1 else (0, 1)))
            j = k

    def settle(self) -> None:
        """Each feed's port cut (`_cut`), its rows gathered to the cut's, and wired."""
        uses: dict[tuple[str, str], dict[int, int]] = {}
        for f in self.feeds:
            m = uses.setdefault((f.tid, f.port), {})
            m[len(f.rows)] = m.get(len(f.rows), 0) + 1
        cut = {k: _cut(k, ms, self.grain.rows) for k, ms in uses.items()}
        for (tid, port), d in cut.items():
            if d > 1:
                self.cuts.setdefault(tid, {})[port] = d
        for f in self.feeds:
            d = cut[f.tid, f.port]
            g = len(f.rows) // d
            self._wire(f.tid, f.port, f.at, f.count, [self.gather(f.rows[i * g:(i + 1) * g]) for i in range(d)])


def _cut(key: tuple[str, str], uses: dict[int, int], most: int) -> int:
    """The cut of a port its feeds read as ``uses`` (row count -> feeds): a divisor ``d`` of every count, at most ``most`` and
    leaving at most ``most`` rows a gather, with the fewest gather units (``d`` for each feed of more than ``d`` rows)."""
    g = math.gcd(*uses)
    ok = [d for d in range(1, min(g, most) + 1) if g % d == 0 and max(uses) // d <= most]
    if not ok:
        raise Unexpressible(f"{key[0]} {key[1]}: read as {sorted(uses)} rows, which no cut of at most {most} rows a port and a "
                            "gather divides")
    return min(ok, key=lambda d: (sum(n * d for m, n in uses.items() if m > d), -d))


class _Request:
    """One request Program's nodes lowered into ``b``'s units: each node's output as the rows that hold it."""

    def __init__(self, b: _Builder, unit: Any, program: Any, steps: dict[int, int]) -> None:
        self.b, self.unit, self.fn, self.steps = b, unit, program.fn, steps
        params = [name for name, _ in self.fn.signature()[0]]
        if set(params) != {PROMPT, WEIGHTS}:
            raise Unexpressible(f"request {unit.request}: its Program's inputs are {params}; the window's lowering takes a "
                                "prompt and the weights record (no prefix-cache hit, carried state or peers)")
        self.params = params
        self.fields: list[tuple[int, int, int]] = []
        leaf = word = 0
        for name, ty in dict(self.fn.signature()[0])[WEIGHTS].fields:
            ws = ty.width_set()
            if len(ws) != 1 or next(iter(ws)) % 16:
                raise Unexpressible(f"request {unit.request}: weight {name}'s leaves are {sorted(ws)} bits, not one width of whole "
                                    "u16 words")
            self.fields.append((leaf, word, next(iter(ws)) // 16))
            leaf, word = leaf + ty.leaves, word + ty.leaves * next(iter(ws)) // 16
        self.fields.append((leaf, word, 0))
        self.weights = word
        self.outs: dict[int, list[tuple[int, int, Row]]] = {}
        self.words: dict[int, int] = {}
        self.gemms: dict[int, tuple[Any, tuple[Row, ...], int]] = {}

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
                    raise Unexpressible(f"request {self.unit.request}: a weight read of leaves [{lo}, {hi}) spans two weights")
                out.append(("p", WEIGHTS, word + (lo - leaf) * w, word + (hi - leaf) * w))
            else:
                out.append(("p", self.params[idx], lo, hi))
        return out

    def rows(self, refs: Refs) -> tuple[Row, ...]:
        """The writer rows ``refs`` read, whole and in order; refused where they read part of a row or a parameter."""
        out: list[Row] = []
        for space, j, lo, hi in self.segments(refs):
            if space != "n":
                raise Unexpressible(f"request {self.unit.request}: a unit reads parameter {j} where a wire is due")
            for a, b, r in self.outs[j]:
                if b <= lo or a >= hi:
                    continue
                if a < lo or b > hi:
                    raise Unexpressible(f"request {self.unit.request}: node {j}'s words [{lo}, {hi}) cut its row {r} at [{a}, {b})")
                out.append(r)
        return tuple(out)

    def weight(self, refs: Refs, words: int) -> int:
        """The first word of the one run of the weights record ``refs`` read, ``words`` long."""
        segs = self.segments(refs)
        if len(segs) != 1 or segs[0][:2] != ("p", WEIGHTS) or segs[0][3] - segs[0][2] != words:
            raise Unexpressible(f"request {self.unit.request}: a weight read of {segs}, not one run of {words} weight words")
        return segs[0][2]

    def lower(self) -> None:
        for j, node in enumerate(self.fn.body.nodes):
            if node.form != "call":
                raise Unexpressible(f"request {self.unit.request}: node {j} is a {node.form}, not a call")
            ret = node.fn.signature()[1]
            ws = ret.width_set()
            if not ws <= {16, 32} or len(ws) != 1:
                raise Unexpressible(f"request {self.unit.request}: node {j} ({node.fn.id}) returns leaves of {sorted(ws)} bits")
            self.words[j] = next(iter(ws)) // 16
            family = node.fn.name
            lower = getattr(self, f"_{family}", None) if family in FAMILIES else None
            if lower is None:
                raise Unexpressible(f"request {self.unit.request}: node {j} is {node.fn.id}, a family the window's lowering "
                                    f"doesn't take ({sorted(FAMILIES)})")
            lower(j, node)

    def _Embedding(self, j: int, node: Any) -> None:
        V, H = node.fn.bindings["V"], node.fn.bindings["H"]
        tok, table = node.args
        tid, u = self.b.unit(E.read(V, H))
        segs = self.segments(tok.refs)
        if segs[0][:2] == ("p", PROMPT):
            pos = segs[0][2]
            self.b.ingress.append(IF.Ingress(tid, "tok", u, 1, self.steps[pos]))
        else:
            self.b.feed(tid, "tok", u, 1, self.rows(tok.refs))
        lo = self.weight(table.refs, V * H)
        t = self.b.tables.setdefault((lo, H, V), IF.Table(f"embedding@{lo}", WEIGHTS, lo, H, V))
        self.b.roots += [IF.TableRead(tid, "root", t.name, "root", u, 1), IF.TableRead(tid, "domain", t.name, "domain", u, 1)]
        self.b.openings.append(IF.Opening(tid, t.name, "tok", ("row", "salt", "path"), "o1", u, 1))
        self.outs[j] = [(0, H, Row(tid, u, "o0", H))]

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
            pieces += [(s0 + i * m, s0 + (i + 1) * m, Row(tid, u + i, "o0", m)) for i in range(count)]
        self.outs[j] = pieces
        self.gemms[j] = (node.fn, xs, lo)

    def _heads(self, node: Any, family: str, count: int) -> Any:
        nodes = node.fn.body.nodes
        fn = nodes[0].fn
        if fn.name != family or any(m.form != "batch" or descriptor_id(m.fn) != descriptor_id(fn) for m in nodes) \
                or sum(m.n for m in nodes) != count:
            raise Unexpressible(f"{node.fn.id}: its body is not {count} heads of one {family} Definition")
        return fn

    def _RoPE(self, j: int, node: Any) -> None:
        H, D = node.fn.bindings["NHEADS"], node.fn.bindings["D"]
        head = self._heads(node, "RoPEHead", H)
        x, cs = node.args
        tid, u = self.b.unit(head, H)
        for h in range(H):
            self.b.feed(tid, "x", u + h, 1, self.rows(x.refs.slice(h * D, (h + 1) * D)))
        self.b.read(tid, "cs", u, (H,), self.weight(cs.refs, D), (0,))
        self.outs[j] = [(h * D, (h + 1) * D, Row(tid, u + h, "o0", D)) for h in range(H)]

    def _Attention(self, j: int, node: Any) -> None:
        s = node.fn.bindings
        T, NH, KVH, D = s["T"], s["NH"], s["KVH"], s["D"]
        if T > KEYS:
            raise Unexpressible(f"{node.fn.id}: a head of {T} keys, past the {KEYS} one `AllGather_v1` reads until v19's "
                                "per-key-block cut is wired")
        head = self._heads(node, "AttentionHead", NH)
        q, kc, vc = node.args
        tid, u = self.b.unit(head, NH)
        for h in range(NH):
            g = h // (NH // KVH)
            self.b.feed(tid, "q", u + h, 1, self.rows(q.refs.slice(h * D, (h + 1) * D)))
            for port, c in (("k", kc), ("v", vc)):
                keys = [self.b.gather(self.rows(c.refs.slice((t * KVH + g) * D, (t * KVH + g + 1) * D))) for t in range(T)]
                self.b.feed(tid, port, u + h, 1, (self.b.gather(keys),))
        self.outs[j] = [(h * D, (h + 1) * D, Row(tid, u + h, "o0", D)) for h in range(NH)]

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


def _steps(compiled: Any) -> dict[int, dict[int, int]]:
    """Per request, each position's step: the step whose row first reaches it."""
    out: dict[int, dict[int, int]] = {}
    for s, rows in enumerate(compiled.steps):
        for r in rows:
            for p in range(r.start, r.start + r.n):
                out.setdefault(r.request, {}).setdefault(p, s)
    return out


def window_circuit(compiled: Any, grain: Grain = Grain(), name: str = NAME) -> IF.Circuit:
    """``compiled``'s window (`steps.Compiled`) as one Circuit (module docstring), its requests in order."""
    b, steps = _Builder(grain), _steps(compiled)
    units = [u for u in compiled.requests if u is not None]
    if not units:
        raise Unexpressible("a window with no request compiles to no circuit")
    weights = set()
    for u in units:
        if u.carried_in or u.carried_out or u.C or u.parse is not None:
            raise Unexpressible(f"request {u.request}: carried across windows, a prefix-cache hit or byte ingress, which the "
                                "window's lowering doesn't take yet")
        r = _Request(b, u, compiled.programs[u.digest], steps.get(u.request, {}))
        r.lower()
        weights.add(r.weights)
    if len(weights) != 1:
        raise Unexpressible(f"the window's requests read weight records of {sorted(weights)} words")
    b.settle()
    roots = _runs(b.roots, lambda x: (x.reader, x.port, x.table, x.of), lambda x: x.at, lambda x: x.units,
                  lambda x, n: IF.TableRead(x.reader, x.port, x.table, x.of, x.at, n))
    openings = _runs(b.openings, lambda x: (x.reader, x.table, x.key, x.ports, x.verdict), lambda x: x.at, lambda x: x.units,
                     lambda x, n: IF.Opening(x.reader, x.table, x.key, x.ports, x.verdict, x.at, n))
    ingress = _runs(b.ingress, lambda x: (x.reader, x.port, x.step), lambda x: x.at, lambda x: x.units,
                    lambda x, n: IF.Ingress(x.reader, x.port, x.at, n, x.step))
    reg = IF.Registered({WEIGHTS: weights.pop()}, tuple(b.tables.values()), tuple(IF.coalesce(b.reads)), tuple(roots),
                        tuple(openings))
    emb = {g.reader for g in b.ingress}
    sel = {f.tid for f in b.feeds if f.port == "x0"}
    return IF.Circuit(name, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()),
                      external=tuple(sorted({f"{t}/tok" for t in emb} | {f"{t}/o0" for t in sel})),
                      wires=tuple(IF.coalesce(b.wires)), cuts=b.cuts, registered=reg, ingress=tuple(ingress))


__all__ = ["FAMILIES", "Grain", "NAME", "Row", "window_circuit"]
