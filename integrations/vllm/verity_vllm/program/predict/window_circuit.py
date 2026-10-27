"""I2's served window as one Circuit (VLLM-103's V3b, `verity.ml.windows.interface`): the window *class's* circuit, every
unit's output a committed row and every input row wired or registered.

A window class (`WindowClass`, N1's scan class: note:network-accounting/20261025T0830Z-report-net68-scan-router) has a
fixed number of slots a network timestep (ASK-122): ``lanes`` slots at each boundary, one every ``every`` scan buckets from the
window's start, ``slots`` in all, each of at most ``prompt`` prompt ids and ``outputs`` outputs. Slot 4b + j at the class of
record is boundary b's lane j, anchored at bucket 16b. Its circuit is one class Program a slot, whatever the traffic
(`class_circuit`): the Build's greedy request Program at one prompt row and P - 1 decode steps, P = prompt + outputs - 1
positions, every position computing the whole stack, the LM head and the token select, position p attending to keys 0..p.
So the circuit is a function of the class alone: which slots carry a request, arrival steps, prompt lengths, output lengths
and key counts are values, never structure. A slot's row (`SLOT_ROW`'s input, row k of the window's slot rows `SLOT_ROWS`,
a per-window registered value) carries its occupancy word, its prompt's length and its ids, the system's ingress at the
slot's anchor. Position p's token is `io_shell.TokenFeed_v1{prompt, FIRST}`'s
while p is below the class's prompt cap (the prompt's id below its length, else the previous position's select; the slot it
carries starts as the row's), and past it the previous position's select. Each position's output is its select masked by
the slot's occupancy word (`io_shell.SlotMask_v1`), so an empty slot's outputs are `io_shell.EMPTY` wherever they are read.
A request decodes to the class's outputs, its selects past its finish masked by what reads them (external until N1's
Programs' tie, IO-TIE's, is in the circuit). `window_circuit(registered, compiled, cls, grain, placed=...)` refuses a window
the class's slots don't take and emits the class's circuit.

`class_circuit` walks the class Program's nodes in order and lowers each to units at ``grain``, once a slot: node j's units
of slot 0, then of slot 1 and so on, each allocation's slots side by side in its template (`window_builder`, which holds
the units, their feeds, gathers and weight reads, and wires them):

- `Embedding_v1{V, H}` is the registered read `EmbeddingRead_v1{V, H}` (`commitments.gates.embedding`): its ``root`` and
  ``domain`` are the embedding table's (a `Table` over the weights record, read by `TableRead`), its ``row``, ``salt`` and
  ``path`` an `Opening` at the hidden key ``tok``, pinned by its verdict ``o1``. Its ``tok`` is its position's feed.
- `Gemm_v2{K, N}` is ⌈N / n⌉ slices `Gemm_v2{K, n}`, n = ``grain.gemm_words`` // K output columns a slice, each reading
  its weight rows from the weights record (`Read`) and its ``x`` whole.
- An int8 linear (`registry.int8_exact`) is its quantizer `Int8RowQuant_v1{K}`, one unit, and its product `ScaledMmS8_v1{K, N}`,
  ⌈N / n⌉ slices `ScaledMmS8_v1{K, n}` at the same weight words a slice (n = ``grain.gemm_words`` // (K / 2): an s8 weight row
  is K / 2 words), each reading its s8 weight rows and their f32 row scales from the weights record and the quantizer's codes
  ``x`` and scale ``sx`` whole. Words are laid out by each field's leaf width (`_spans`): a weight or node output whose leaves
  aren't one byte width filling whole u16 words, and a read that starts inside a word, are refused.
- `RoPE_v1{NHEADS, D}` is its heads (`RoPEHead_v1{D}`), each reading its slice of ``x``.
- `Attention_v18{T, NH, KVH, D, BN, ...}` is `Attention_v19`'s cut (#4263, CIR-96): each head's Calls (`AttentionHead_v19`'s
  body, read once a T) are its units, its first visited key block `AttnBlockFirst_v1{D, NVIS}` (keys (nb - 1)·BN ... T - 1,
  nb = ⌈T / BN⌉), each full block below it `AttnBlock_v5` and its end `AttnHeadEnd_v1{D}`, one unit a head a Call, each
  block's running state its predecessor's rows. A block's keys and values are one row a KV group, which its group's heads
  read: one key's own row (RoPE's, or its value's GEMM slices gathered once, at its position), a full block's
  `AllGather_v1{BN, D}` (made once, when the block fills, and read by every later position) or the first block's
  `AllGather_v1{NVIS, D}`. A layer's cache is refused unless it is the keys its earlier positions' nodes read and one more.
- Any other attention version (FA3's `Attention_v4` on sm_90) is its heads, a unit each of its own head Definition
  (`AttentionHead_v4{T}`), while T is at most `KEYS`: a head's keys and values are one row a KV group, an
  `AllGather_v1{T, D}` over every key's own row, and past `KEYS` keys it is refused.
- `TokenSelect_v4{V}` is its function at ``grain.select`` logits a chunk (its body's own form at another C): chunks
  `TokenSelectChunk_v2{C}`, each reading its C logits through one gather, and `TokenSelectCombine_v2{V, C}` over them:
  ``best`` through one gather of the chunks' maxima, ``idx`` cut a row a chunk, and ``x0`` (``logits[0]``) from one
  `Gemm_v2{K, 1}` unit over the LM head's input and its first weight row.
- `RMSNormTriton_v5`, `RMSNormFusedCuda_v2` and `SiluMul_v2` are one unit each.

The weights record is never one registered parameter: each weight a port reads is its own at the port's row width
(`window_builder.weight_param`: ``weights/<label>@<words>``), its reads' bases and steps whole rows of it, so each is pinned
as a root over its own rows and the window's members take its rows as registered rows (`interface.members`). An embedding's
table is a parameter of its own (`window_builder.table_param`: ``weights/<label>@<words>/table``, its rows ``row-seg``
leaves), so a tied embedding is registered three times: as its table, as the LM head's rows and as ``x0``'s rows at the
table's width. A pin of `WEIGHTS` counts them all pinned (`interface.input_count`), but nothing in the circuit ties their
roots to one another or to the weights record's root: the verifier pins each by name, its root and domain id
(`interface.registered_reads`).

A reader port that reads several writers' rows is cut (`interface.Layout.cut`), one cut per template and port: the cut ``d``
divides every use's row count, leaves its reader at most ``grain.rows`` rows in and out, and makes the fewest gather units,
a use's rows gathered ``m / d`` at a time by `AllGather_v1` units (whose ``parts`` are cut a row a part, so a gather of
``m / d`` parts holds ``m / d + 1`` rows with its output). Every wire and read is stated per node as a pattern (a node's
heads, KV groups and slices as digits of one) and `interface.coalesce`d, and each allocation's slots sit side by side in its
template, so the slots' copies of a pattern are one and the circuit's description grows with the class Program's nodes, not
its slots or units; so does the lowering's own state (a node's output rows are runs, a gather is keyed by its rows' runs).

The circuit's blocks (`interface.Blocks`) carry its inference block: a unit's slot is its slot (its class Program; the
blocks' slots are the class's S) and its step its position, T the class's positions. An embedding opens its position, the
stack, LM head, token select and mask after it are that position's, a slot's row is its first position's, and a
gather is its reader's. An allocation's units are one entry, its slots' runs at one step side by side (the slot digit, then
the run), and like entries that follow on merge, at one step and then at consecutive steps (`_inference`).

Refused by name (`Unexpressible`): a window past its class's slots (more requests than slots, more anchored at a boundary
than its lanes, an anchor that is no boundary, a prompt past the cap, more outputs than the cap), a request whose last
output is past the window's last bucket (`ACROSS`), a request whose Program would read more than its prompt and the weights
(a stochastic request's seed, a prefix-cache hit's prefix, a TP rank's peers), a request a Commit window carries in or out,
byte ingress, a family this lowering doesn't know, a head past `KEYS` keys of an attention version with no per-key-block
cut, a weight whose leaves aren't whole u16 words, a row form the cut rule can't state and a unit of more than ``grain.rows``
rows. The circuit is completeness only: that it computes what serving ran
is held by the units' proofs on their committed rows, never by this lowering.
"""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Refs, runs
from verity.ml.circuits.types import Tuple
from verity.ml.commitments.gates import embedding as E
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.dead_free import AttentionV19
from verity.ml.windows import interface as IF

from .build import Unexpressible
from .steps import RequestUnit, request_program
from .window_builder import WEIGHTS, Grain, Row, _Builder, _step

NAME = "verity-vllm/window"
PROMPT = "prompt"
GREEDY = {"kind": "greedy"}
#: the Definition that reads a slot's row, (occupancy, prompt length, ids). A placeholder for the row network-
#: accounting's BLOCKS-NET PR names (ASK-122: token ids in a committed row with a private occupancy word, tied to the
#: certified bytes by #4548's id form); the lowering reads a slot's row only through this
SLOT_ROW = IO.SlotRow
#: the window's slot rows, a per-window registered value: row k is slot k's `SLOT_ROW` row, committed under the window's
#: run root before its draw. The circuit doesn't tie them to the certified bytes yet (the scan registration's state row,
#: #4853, does), so the audit counts their reads free
SLOT_ROWS = "slot-rows"
#: a request held across the window's end, which the class doesn't take: whether the network certifier's lanes hold one
#: across a window boundary is open (thread 1793030365.246499), and a slot that does needs a Program running into the next
#: window
ACROSS = "held across the window's end"
#: the most keys a head reads through one `AllGather_v1{T, D}` where its attention's version has no per-key-block cut (its
#: gather's T parts and output within a unit's 256 rows, with room for its reader's cut)
KEYS = 136


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
        """Slot ``slot``'s anchor: the bucket its row is the system's ingress at."""
        return self.boundaries[slot // self.lanes]


#: the class of record, the certifier run of record's envelope at ASK-122's slots (network-accounting took vllm's pick):
#: 8 slots a 32-bucket window, 4 at each of buckets 0 and 16 (EVERY 16, D 2.7 s), each of at most 1,024 prompt ids and 64
#: outputs
CLASS_OF_RECORD = WindowClass(8, 1024, 64, lanes=4, every=16)


_GRAIN = Grain()


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
    """A layer's keys (or values) as its attention nodes read them: ``segs`` the word runs its positions so far are, ``rows``
    each position's row a KV group, and ``blocks`` each full key block's gather a group."""
    segs: list[tuple[str, Any, int, int]] = field(default_factory=list)
    rows: list[tuple[Row, ...]] = field(default_factory=list)
    blocks: dict[int, tuple[Row, ...]] = field(default_factory=dict)


class _Program:
    """Slot ``slot``'s class Program lowered into ``b``'s units, its row of `SLOT_ROWS` the system's ingress at the slot's anchor
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
        labels = [label for label, _ in dict(self.fn.signature()[0])[WEIGHTS].fields]
        self.fields = _spans(f"{name}: weight", dict(self.fn.signature()[0])[WEIGHTS].fields)
        b.weights = b.weights or tuple((label, s[1], e[1] - s[1]) for label, s, e in zip(labels, self.fields, self.fields[1:]))
        self.outs: dict[int, list[tuple[int, int, Row]]] = {}
        self.words: dict[int, list[tuple[int, int, int]]] = {}
        self.gemms: dict[int, tuple[Any, tuple[Row, ...], int]] = {}
        self.keys: dict[tuple[str, tuple[str, Any, int, int]], _Keys] = {}
        self.position = 0
        self.slot: Row | None = None
        self.occ: Row | None = None
        #: the units a node makes beyond its output rows: each position's `TokenFeed_v1` unit and (at position 0) its
        #: `SLOT_ROW` unit, None past the class's prompt cap; each token select's chunks, ``x0`` unit and `IO.SlotMask` unit
        self.tokens: list[tuple[tuple[str, int] | None, tuple[str, int] | None]] = []
        self.selects: dict[int, tuple[tuple[tuple[str, int], ...], tuple[str, int], tuple[str, int]]] = {}

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
        fields = [f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,))]
        self.words[j] = _spans(f"{self.name}: node {j} ({node.fn.id}) return", [(f"o{i}", t) for i, t in enumerate(fields)])
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
        t = self.b.table(self.weight(table.refs, V * H), H, V)
        self.b.roots += [IF.TableRead(tid, "root", t.name, "root", u, 1), IF.TableRead(tid, "domain", t.name, "domain", u, 1)]
        self.b.openings.append(IF.Opening(tid, t.name, "tok", ("row", "salt", "path"), "o1", u, 1))
        self.outs[j] = [(0, H, Row(tid, u, "o0", H))]

    def _token(self, tok: Any) -> tuple[Row, ...]:
        """The next position's token: its `TokenFeed_v1`'s below the class's prompt cap (the first's slot its slot's row
        opened by `SLOT_ROW`, every other's its predecessor's, ``sel`` the previous position's select), past it the previous
        position's select."""
        p, NP = self.position, self.cls.prompt
        self.position += 1
        segs = self.segments(tok.refs)
        if (segs[0][:2] == ("p", PROMPT)) != (p == 0):
            raise Unexpressible(f"{self.name}: position {p}'s token reads {segs[0][:2]}; a class Program reads its one prompt row "
                                "at position 0 and every other position's token from its predecessor's select")
        if p >= NP:
            self.tokens.append((None, None))
            return self.rows(tok.refs)
        tid, u = self.b.unit(bind(IO.TokenFeed, NP=NP, FIRST=p == 0))
        if p == 0:
            rtid, ru = self.b.unit(bind(SLOT_ROW, NP=NP))
            self.b.value(rtid, "row", ru, SLOT_ROWS, self.slot_index, self.cls.slots)
            self.b.feed(tid, "slot", u, 1, (Row(rtid, ru, "o1", 2 * (NP + 1)),))
            self.occ = Row(rtid, ru, "o0", 2)
            self.tokens.append(((tid, u), (rtid, ru)))
        else:
            self.b.feed(tid, "slot", u, 1, (self.slot,))
            self.b.feed(tid, "sel", u, 1, self.rows(tok.refs))
            self.tokens.append(((tid, u), None))
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

    _RMSNormTriton = _RMSNormFusedCuda = _SiluMul = _Int8RowQuant = _one

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

    def _ScaledMmS8(self, j: int, node: Any) -> None:
        b = node.fn.bindings
        K, N = b["K"], b["N"]
        n = max(1, min(N, self.b.grain.gemm_words // (K // 2)))
        x, sx, w, sw = node.args
        xs, ss = self.rows(x.refs), self.rows(sx.refs)
        lo, slo = self.weight(w.refs, N * K // 2), self.weight(sw.refs, 2 * N)
        full, rest = divmod(N, n)
        pieces = []
        for s0, m, count in ((0, n, full), (full * n, rest, int(rest > 0))):
            if not count:
                continue
            tid, u = self.b.unit(bind(node.fn.defn, **{**b, "N": m}), count)
            self.b.feed(tid, "x", u, count, xs)
            self.b.feed(tid, "sx", u, count, ss)
            self.b.read(tid, "w", u, (count,), lo + s0 * K // 2, (m * K // 2,))
            self.b.read(tid, "sw", u, (count,), slo + 2 * s0, (2 * m,))
            pieces.append((s0, s0 + count * m, Row(tid, u, "o0", m)))
        self.outs[j] = pieces

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
        T, NH, KVH, D, BN = s["T"], s["NH"], s["KVH"], s["D"], s["BN"]
        if node.fn.version not in (18, 19):
            self._whole_heads(j, node)
            return
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

    def _whole_heads(self, j: int, node: Any) -> None:
        """An attention whose version has no per-key-block cut (FA3's `Attention_v4`): a unit a head of its own head
        Definition, its keys and values each one row a KV group, an `AllGather_v1{T, D}` over every key's row (`_cache`'s),
        refused past `KEYS` keys."""
        s = node.fn.bindings
        T, NH, KVH, D = s["T"], s["NH"], s["KVH"], s["D"]
        if T > KEYS:
            raise Unexpressible(f"{node.fn.id}: a head of {T} keys, past the {KEYS} one `AllGather_v1` reads; the window's "
                                f"lowering cuts only `Attention_v18`'s and v19's heads by key block, not v{node.fn.version}'s")
        head = self._heads(node.fn, "AttentionHead", NH)
        q, kc, vc = node.args
        tid, u = self.b.unit(head, NH)
        self._each(tid, "q", u, self._split(q.refs, NH))
        for port, c in (("k", kc), ("v", vc)):
            rows = self._cache(port, c, T, KVH).rows
            self._each(tid, port, u, [(r,) for r in self.b.gathers([tuple(r[g] for r in rows) for g in range(KVH)])],
                       NH // KVH)
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
        self.selects[j] = (tuple(parts), (xtid, xu), (mtid, mu))


FAMILIES = ("Embedding", "RMSNormTriton", "RMSNormFusedCuda", "Gemm", "Int8RowQuant", "ScaledMmS8", "RoPE", "Attention", "SiluMul",
            "TokenSelect")


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


@dataclass(frozen=True)
class Lowered:
    """A class circuit with what its lowering knows (`window_rows` fills its units from it): the class Program's function, each
    slot's lowered Program (node j's output as the rows that hold it, ``outs[j]``) and each template's runs of units by their
    (slot, step), ``(at, count, slot, step)``."""
    circuit: IF.Circuit
    fn: Any
    programs: tuple[_Program, ...]
    spans: dict[str, list[tuple[int, int, int, int]]]


def class_circuit(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN,
                  name: str = NAME) -> IF.Circuit:
    """``cls``'s circuit (module docstring): the class Program lowered once a slot, node by node, slot k reading
    row k of `SLOT_ROWS` (the system's ingress at its anchor, `WindowClass.anchor`) and each allocation's slots side by side in its
    template; its blocks' slots are the slots.
    A window's circuit is every slot's; ``programs`` is the first slots only, a probe's census (`class_size`)."""
    return lower(registered, cls, programs, grain, name).circuit


def lower(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN, name: str = NAME) -> Lowered:
    """`class_circuit`'s circuit with its lowering (`Lowered`)."""
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
    b.settle()
    roots = _runs(b.roots, lambda x: (x.reader, x.port, x.table, x.of), lambda x: x.at, lambda x: x.units,
                  lambda x, n: IF.TableRead(x.reader, x.port, x.table, x.of, x.at, n))
    openings = _runs(b.openings, lambda x: (x.reader, x.table, x.key, x.ports, x.verdict), lambda x: x.at, lambda x: x.units,
                     lambda x, n: IF.Opening(x.reader, x.table, x.key, x.ports, x.verdict, x.at, n))
    reg = IF.Registered(dict(sorted(b.params.items())), tuple(b.tables.values()), tuple(IF.coalesce(b.reads)), tuple(roots),
                        tuple(openings))
    rows = {r.reader for r in b.reads if r.param == SLOT_ROWS}
    masks = {f.tid for f in b.feeds if f.port == "occ"}
    circuit = IF.Circuit(name, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()),
                         external=tuple(sorted({f"{t}/row" for t in rows} | {f"{t}/o0" for t in masks})),
                         wires=tuple(IF.coalesce(b.wires)), cuts=b.cuts, registered=reg,
                         blocks=IF.Blocks(programs, cls.positions, tuple(_inference(b.spans, programs))))
    return Lowered(circuit, fn, tuple(progs), b.spans)


def window_circuit(registered: Any, compiled: Any, cls: WindowClass, grain: Grain = _GRAIN, name: str = NAME,
                   placed: Sequence[tuple[int, int | None]] | None = None) -> IF.Circuit:
    """``compiled``'s window (`steps.Compiled`, of ``registered``) as ``cls``'s circuit, every slot's class Program whatever
    its traffic, refused where the class's slots don't take the window: nothing of the window reaches the circuit.
    ``placed`` is each request's anchor bucket and the bucket of its last output in the window, in its request order, as the
    network certifier's scan placed it (the end None: unchecked); by default every request is at the window's first
    boundary, its end unchecked."""
    takes(compiled, cls, placed)
    return class_circuit(registered, cls, None, grain, name)


def takes(compiled: Any, cls: WindowClass, placed: Sequence[tuple[int, int | None]] | None = None) -> None:
    """Refuse a window ``cls``'s slots don't take (`window_circuit`)."""
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


__all__ = ["ACROSS", "CLASS_OF_RECORD", "FAMILIES", "Grain", "KEYS", "Lowered", "NAME", "Row", "SLOT_ROW", "SLOT_ROWS", "WindowClass",
           "class_circuit", "class_program", "lower", "takes", "window_circuit"]
