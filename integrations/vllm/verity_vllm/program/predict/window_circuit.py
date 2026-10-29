"""I2's served window as one Circuit (`verity.ml.windows.interface`): the window *class's* circuit, every
unit's output a committed row and every input row wired or registered.

A window class (`WindowClass`, N1's scan class) has a fixed number of slots a network timestep: ``lanes`` slots at each
boundary, one every ``every`` scan buckets from the window's start, ``slots`` in all, each of at most ``prompt`` prompt ids and
``outputs`` outputs. Slots are lane-major: slot 2j + b at the class of record is boundary b's lane j, anchored at bucket 16b.
Its circuit is one class Program a slot, whatever the traffic (`class_circuit`): the Build's greedy request Program at one
prompt row and P - 1 decode steps, P = prompt + outputs - 1 positions, every position computing the whole stack, the LM head and
the token select, position p attending to keys 0..p. So the circuit is a function of the class alone: which slots carry a
request, arrival steps, prompt lengths, output lengths and key counts are values, never structure.

A slot's request comes from the network certifier's scan: boundary b's Programs read the scan's state row before it
(`scan_state.SCAN_STATE`, registered) through `boolean_io_shell.ScanSlots_v2` (`scan_state.bridge`), which gives each lane
its first position's token (its prompt's first id) and the slot it carries on, its occupancy word and its head (its prompt's
length and ``max_tokens``). Every later position's token is `io_shell.TokenFeed_v1{prompt, FIRST=False}`'s below the class's
prompt cap (the prompt's id below its length, else the previous position's select), and past it the previous position's
select. Each select is collected (`io_shell.OutCollect_v1`, from the prompt's last position on). The last position makes the
slot's entry of the scan's Programs' row (`io_shell.SlotEntry_v1`): its output count (up to ``max_tokens`` and the model's
first stop id, that one kept, `stop_ids`) as a u32, then its ids cut to the scan's 17 bits, all zero for an empty slot (the
entry reads the occupancy word, so no select is masked on the way); a request the class can't hold (a prompt past ``prompt``
ids, or outputs that don't end within ``outputs``) has ``outputs`` + 1 and no ids, which the scan refuses at the anchor and the
egress never renders. The state rows and the entries are the circuit's external ports. A slot's Program
hands all its outputs to the scan at its anchor, which carries them across windows (`SC_GROUP`), so no Program runs into the
next window. `window_circuit(registered, compiled, cls, grain, placed=...)` refuses a window the class's slots don't take and
emits its circuit.

`class_circuit` walks the class Program's nodes in order and lowers each to units at ``grain``, once a slot: node j's units
of slot 0, then of slot 1 and so on, each allocation's slots side by side in its template (`window_builder`, which holds
the units, their feeds and weight reads, and wires them). Every unit reads its writers' rows as they hold them:

- `Embedding_v1{V, H}` is the registered read `EmbeddingRead_v1{V, H}` (`commitments.gates.embedding`): its ``root`` and
  ``domain`` are the embedding table's (a `Table` over the weights record, read by `TableRead`), its ``row``, ``salt`` and
  ``path`` an `Opening` at the hidden key ``tok``, pinned by its verdict ``o1``. Its ``tok`` is its position's feed.
- `Gemm_v2{K, N}` is ⌈N / n⌉ slices `Gemm_v2{K, n}`, each reading its weight rows from the weights record (`Read`) and
  its ``x`` as its writers' rows, n = ``grain.gemm_words`` // K columns, halved while an earlier node reads that slice
  template's ``x`` at another row count (a port is cut one way, and o_proj's ``x`` is a row a head).
- An int8 linear (`registry.int8_exact`) is its quantizer `Int8RowQuant_v1{K}`, one unit (its body's two, scale and codes,
  where an earlier node reads its template's ``x`` at another row count: o_proj's), and its product `ScaledMmS8_v1{K, N}`,
  ⌈N / n⌉ slices `ScaledMmS8_v1{K, n}` at the same weight words a slice (n = ``grain.gemm_words`` // (K / 2): an s8 weight row
  is K / 2 words), each reading its s8 weight rows and their f32 row scales from the weights record and the quantizer's codes
  ``x`` and scale ``sx`` whole. Words are laid out by each field's leaf width (`_spans`): a weight or node output whose leaves
  aren't one byte width filling whole u16 words, and a read that starts inside a word, are refused.
- `RoPE_v1{NHEADS, D}` is its heads (`RoPEHead_v1{D}`), each reading its slice of ``x``.
- `Attention_v18{T, NH, KVH, D, BN, ...}` is `Attention_v19`'s cut: each head's Calls (`AttentionHead_v19`'s
  body, read once a T) are its units, its first visited key block `AttnBlockFirst_v1{D, NVIS}` (keys (nb - 1)·BN ... T - 1,
  nb = ⌈T / BN⌉), each full block below it `AttnBlock_v5` and its end `AttnHeadEnd_v1{D}`, one unit a head a Call, each
  block's running state its predecessor's rows. A block's keys and values a KV group are its keys' rows as their
  writers hold them (a key's RoPE head's row, a value's GEMM slices), key by key, which its group's heads read. A layer's
  cache is refused unless it is the keys its earlier positions' nodes read and one more.
- FA3's `Attention_v4` (sm_90) is `Attention_v20`'s cut the same way: its first visited block
  `AttnBlockFirst_v1` (FA3's first iteration is FA2's word for word), each full block below it
  `AttnBlock_v4{FIRST=False, CHECK = b >= MASKED_FROM}` and its end `AttnHeadEnd_v1{D, INV=Fa3InvSum_v1}`. `CUTS` names
  the versions cut by key block and the cut of each.
- Any other attention version (`Attention_v2`, as `fa3_construction = "guard-every-block"` binds) is its heads, a unit
  each of its own head Definition (`AttentionHead_v2{T}`), reading its KV group's every key's rows.
- `TokenSelect_v4{V}` is its function at groups of `GROUP` logits: V / C units `TokenSelectGroups_v1{C, GROUP}`, C =
  ``grain.select``, each reading its C logits as the LM head's rows and writing its groups' bests as a row and their indices
  as another, then `TokenSelectCombine_v2{V, GROUP}` over those rows and ``x0`` (``logits[0]``), from one `Gemm_v2{K, 1}`
  unit over the LM head's input and its first weight row. A V that isn't whole units of whole groups is refused.
- `SiluMul_v2{I}` is P units `SiluMul_v2{I / P}`, each reading its gate and up elements as the gate-up GEMM's rows, P the
  fewest that leaves a unit within ``grain.rows`` rows; `RMSNormTriton_v5` and `RMSNormFusedCuda_v2` are one unit each.

The weights record is never one registered parameter: each weight a port reads is its own at the port's row width
(`window_builder.weight_param`: ``weights/<label>@<words>``), its reads' bases and steps whole rows of it, so each is pinned
as a root over its own rows, by its own name, and the window's members take its rows as registered rows
(`interface.members`). An embedding's table is a parameter of its own (`window_builder.table_param`:
``weights/<label>@<words>/table``, its rows ``row-seg`` leaves), so a tied embedding is registered three times: as its
table, as the LM head's rows and as ``x0``'s rows at the table's width. Nothing in the circuit ties their roots to one
another or to the checkpoint: the verifier pins each by name, its root and domain id (`interface.registered_reads`), at the
committer's registration (`window_weights`).

A reader port that reads m writer rows is cut m ways (`interface.Layout.cut`, `window_builder._Builder.settle`), a row of
it a writer row, so no unit copies rows together: a template's port is read at one row count, its rows its outer elements
cut that many ways. Every wire and read is stated per node as a pattern (a node's heads, KV groups and slices as digits of
one) and `interface.coalesce`d, and each allocation's slots sit side by side in its template, so the slots' copies of a
pattern are one and the circuit's description grows with the class Program's nodes, not its slots or units; so does the
lowering's own state (a node's output rows are runs).

The circuit's blocks (`interface.Blocks`) carry its inference block: a unit's slot is its slot (its class Program; the blocks'
slots are the class's S) and its step its position, T the class's positions. An embedding opens its position, the stack, LM head,
token select and collector after it are that position's, a slot's `ScanSlots_v2` unit its first position's and
its entry its last's. An allocation's units are one entry, its slots' runs at one step side by side (the slot
digit, then the run), and like entries that follow on merge, at one step and then at consecutive steps (`_inference`).

Refused by name (`Unexpressible`): a window past its class's slots (more requests than slots, more anchored at a boundary than
its lanes, an anchor that is no boundary, a prompt past the cap, more outputs than the cap), a request whose Program would read
more than its prompt and the weights (a stochastic request's seed, a prefix-cache hit's prefix, a TP rank's peers), a request
a Commit window carries in or out, byte ingress, a family this lowering doesn't know, a weight whose leaves aren't whole u16
words, a read that cuts a writer's row, and at `settle`, every one named at once, a template's port read at two row counts
or as rows that aren't its outer elements cut that many ways, a unit of more than ``grain.rows`` rows, and then a unit
Flock's composer wouldn't prove (`interface.check_proof_block`). The circuit is completeness only: that it computes what
serving ran is held by the units' proofs on their committed rows, never by this lowering.

Its weight reads, the embedding's table reads and its openings count registered (`interface.input_count`) only where the
verifier pinned that weight's own parameter, ``weights/<label>@<words>``, before the run, at the root the window's
registration records for it; anywhere else they are free. The pins are the hiding roots the committer registered once per
model and key, never roots the verifier computes (`window_weights`). Its scan state reads are held otherwise: the window
commits the scan's state rows it reads as `scan_state.SCAN_STATE` (`ingress_scan.window_state`, `window_rows.state_value`)
and the scan's own state ports read that root under its domain, so the scan's chain and the window's reads are one tree.
No pin holds them, so `interface.input_count` counts them free.
"""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import accumulate, product
from typing import Any

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.refs import Refs, runs
from verity.ml.circuits.types import Tuple
from verity.ml.commitments.gates import embedding as E
from verity.ml.definitions.registry import int8_exact as IX
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.dead_free import AttentionV19, TokenSelectGroupsV1
from verity.ml.definitions.registry.fa3_check_inf import AttentionV20
from verity.ml.windows import interface as IF

from . import scan_state as SS
from .build import Unexpressible
from .steps import RequestUnit, request_program
from .window_builder import WEIGHTS, Grain, Row, _Builder, _step

NAME = "verity-vllm/window"
PROMPT = "prompt"
GREEDY = {"kind": "greedy"}
#: the `Attention` versions cut by key block, each by the version of its function whose heads' steps are Calls: FA2's v18
#: by v19, FA3's v4 by v20
CUTS = {18: AttentionV19, 19: AttentionV19, 4: AttentionV20, 20: AttentionV20}
#: a token select's group of logits: a word of each `TokenSelectGroups_v1` unit's row of bests
GROUP = 16


@dataclass(frozen=True)
class WindowClass:
    """A window class's public caps (N1's scan `Class`: its slots, its prompt ids and its KO) and its slots' timing (fixed
    slots a network timestep): ``slots`` class Programs a window, each of at most ``prompt`` prompt ids and
    ``outputs`` outputs, ``lanes`` of them anchored at each boundary, one every ``every`` scan buckets from the window's
    start (lanes 0: every slot at the window's one boundary). Slots are lane-major: slot ``nb · j + b`` is boundary b's lane j
    (nb = slots / lanes boundaries), anchored at bucket ``every · b``, and the window is ``buckets`` = every · nb buckets. Its
    `scan` is the network certifier's scan class its slots are."""
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
        """Slot ``slot``'s anchor: the bucket whose scan state row before it the slot's Program reads."""
        return self.boundaries[slot % len(self.boundaries)]

    def lane(self, slot: int) -> int:
        """Slot ``slot``'s lane at its boundary: its slot of the scan's state row."""
        return slot // len(self.boundaries)

    @property
    def scan(self) -> Any:
        """The network certifier's scan class at its lanes, outputs and timestep (`scan_state.scan_class`)."""
        return SS.scan_class(self.lanes, self.outputs, self.every)


#: the class of record, the network certifier's envelope at these fixed slots:
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
    """A layer's keys (or values) as its attention nodes read them: ``segs`` the word runs its positions so far are, and
    ``rows`` each position's rows a KV group, as their writers hold them."""
    segs: list[tuple[str, Any, int, int]] = field(default_factory=list)
    rows: list[tuple[tuple[Row, ...], ...]] = field(default_factory=list)


class _Program:
    """Slot ``slot``'s class Program lowered into ``b``'s units, its request its lane's of the scan state row before its anchor
    (`WindowClass.anchor`, `WindowClass.lane`) through ``scan`` (`scan_state.bridge`), its outputs the scan's through
    ``entry`` (`io_shell.SlotEntry_v1`): each node's output as the rows that hold it, ``(lo, hi, row)`` its words [lo, hi) as
    ``row`` and the units after it, (hi - lo) / ``row.words`` rows of one output at consecutive units."""

    def __init__(self, b: _Builder, fn: Any, cls: WindowClass, name: str, slot: int, scan: Any, entry: Any) -> None:
        self.b, self.fn, self.cls, self.name, self.slot_index = b, fn, cls, name, slot
        self.anchor, self.lane, self.scan, self.entry = cls.anchor(slot), cls.lane(slot), scan, entry
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
        self.head: Row | None = None
        self.collect: Row | None = None
        #: the units a node makes beyond its output rows: each position's `TokenFeed_v1` unit, None at the first (the scan
        #: unit's) and past the class's prompt cap; each token select's group units and ``x0`` unit
        self.tokens: list[tuple[str, int] | None] = []
        self.selects: dict[int, tuple[tuple[tuple[str, int], ...], tuple[str, int]]] = {}
        #: each attention cut by key block: its cut's plan (`_plan`) and each Call's first unit, a unit a head
        self.heads: dict[int, tuple[Any, tuple[tuple[str, int], ...]]] = {}
        #: the slot's `ScanSlots_v2` unit, each position's `OutCollect_v1` unit and its `SlotEntry_v1` unit
        self.scan_unit: tuple[str, int] | None = None
        self.collects: list[tuple[str, int]] = []
        self.entry_unit: tuple[str, int] | None = None

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
        """The next position's token: the first's its lane's of the scan state row the slot's `ScanSlots_v2` reads at its
        anchor (with the slot it carries, its occupancy word and its head), below the class's prompt cap every other's
        `TokenFeed_v1`'s (its slot its predecessor's, ``sel`` the previous position's select), past it the previous select."""
        p, NP = self.position, self.cls.prompt
        self.position += 1
        segs = self.segments(tok.refs)
        if (segs[0][:2] == ("p", PROMPT)) != (p == 0):
            raise Unexpressible(f"{self.name}: position {p}'s token reads {segs[0][:2]}; a class Program reads its one prompt row "
                                "at position 0 and every other position's token from its predecessor's select")
        if p == 0:
            tid, u = self.b.unit(self.scan)
            self.b.read(tid, "state", u, (1,), self.anchor * self.scan.bindings["NS"], (0,), SS.SCAN_STATE)
            self.scan_unit, o = (tid, u), 4 * self.lane
            self.slot, self.occ, self.head = (Row(tid, u, f"o{o + 1}", 2 * (NP + 1)), Row(tid, u, f"o{o + 2}", 2),
                                              Row(tid, u, f"o{o + 3}", 4))
            self.tokens.append(None)
            return (Row(tid, u, f"o{o}", 2),)
        if p >= NP:
            self.tokens.append(None)
            return self.rows(tok.refs)
        tid, u = self.b.unit(bind(IO.TokenFeed, NP=NP, FIRST=False))
        self.b.feed(tid, "slot", u, 1, (self.slot,))
        self.b.feed(tid, "sel", u, 1, self.rows(tok.refs))
        self.tokens.append((tid, u))
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
        ends = list(accumulate((w for _, w in lay.outs), initial=0))
        self.outs[j] = [(a, e, Row(tid, u, o, w)) for (o, w), a, e in zip(lay.outs, ends, ends[1:])]

    _RMSNormTriton = _RMSNormFusedCuda = _one

    def _Int8RowQuant(self, j: int, node: Any) -> None:
        """`Int8RowQuant_v1{K}`, one unit, or where an earlier node reads its template's ``x`` at another row count, its body's
        two: `Int8RowScale_v1{K}` (its scale) and `Int8StaticQuant_v1{K}` (its codes under it), each reading ``x``."""
        xs, K, (codes, scale) = self.rows(node.args[0].refs), node.fn.bindings["K"], IF.layout(node.fn).outs
        if self.b.read_rows.get((descriptor_id(node.fn), "x"), len(xs)) == len(xs):
            return self._one(j, node)
        (st, su), (qt, qu) = self.b.unit(bind(IX.Int8RowScale, K=K)), self.b.unit(bind(IX.Int8StaticQuant, K=K))
        s = Row(st, su, "o0", scale[1])
        self.b.feed(st, "x", su, 1, xs)
        self.b.feed(qt, "x", qu, 1, xs)
        self.b.feed(qt, "s", qu, 1, (s,))
        self.outs[j] = [(0, codes[1], Row(qt, qu, "o0", codes[1])), (codes[1], codes[1] + scale[1], s)]

    def _SiluMul(self, j: int, node: Any) -> None:
        """`SiluMul_v2{I}` as P units `SiluMul_v2{I / P}`, unit k reading gate and up elements [k·I/P, (k+1)·I/P) as the
        gate-up GEMM's rows that hold them, P the fewest that leaves a unit within ``grain.rows`` rows."""
        I, (gu,) = node.fn.bindings["I"], node.args
        for P in (p for p in range(1, I + 1) if I % p == 0):
            w = I // P
            try:
                pieces = [self.rows(gu.refs.slice(k * w, (k + 1) * w)) + self.rows(gu.refs.slice(I + k * w, I + (k + 1) * w))
                          for k in range(P)]
            except Unexpressible:
                continue
            if max(map(len, pieces)) + 1 <= self.b.grain.rows:
                break
        else:
            raise Unexpressible(f"{node.fn.id}: no cut of its gate and up rows holds a unit within {self.b.grain.rows} rows")
        tid, u = self.b.unit(bind(node.fn.defn, I=w), P)
        self._each(tid, "gu", u, pieces)
        self.outs[j] = [(0, I, Row(tid, u, "o0", w))]

    def _Gemm(self, j: int, node: Any) -> None:
        b = node.fn.bindings
        K, N = b["K"], b["N"]
        x, w = node.args
        xs, lo = self.rows(x.refs), self.weight(w.refs, N * K)
        n = max(1, min(N, self.b.grain.gemm_words // K))
        while n > 1 and any(self.b.read_rows.get((descriptor_id(bind(node.fn.defn, **{**b, "N": m})), "x"), len(xs)) != len(xs)
                            for m in (n, N % n) if m):
            n //= 2
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
        if chunk.name != "TokenSelectChunk" or combine.name != "TokenSelectCombine" or V < 2:
            raise Unexpressible(f"{node.fn.id}: its body is not chunks and one combine")
        C = min(self.b.grain.select, V)
        if V % C or (C % GROUP and C < V):
            raise Unexpressible(f"{node.fn.id}: {V} logits aren't units of {C}, each groups of {GROUP}: bind the grain's "
                                f"select to a divisor of V that is a multiple of {GROUP}")
        (logits,), n, parts = node.args, -(-C // GROUP), []
        for lo in range(0, V, C):
            tid, u = self.b.unit(bind(TokenSelectGroupsV1, C=C, G=GROUP))
            self.b.feed(tid, "x", u, 1, self.rows(logits.refs.slice(lo, lo + C)))
            parts.append((tid, u))
        ctid, cu = self.b.unit(bind(combine.defn, **{**combine.bindings, "V": V, "C": GROUP}))
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
        self.outs[j] = [(0, 2, Row(ctid, cu, "o0", 2))]
        if self.occ is None or self.head is None:
            raise Unexpressible(f"{node.fn.id}: a token select before its Program's first position's token")
        self.selects[j] = (tuple(parts), (xtid, xu))
        self._collect(self.position - 1, Row(ctid, cu, "o0", 2))

    def _collect(self, p: int, tok: Row) -> None:
        """Position p's select ``tok`` into the slot's `io_shell.OutCollect_v1` chain, and at its last its entry."""
        KO = self.cls.outputs
        ctid, cu = self.b.unit(bind(IO.OutCollect, KO=KO, FIRST=p == 0))
        self.b.feed(ctid, "head" if p == 0 else "state", cu, 1, (self.head if p == 0 else self.collect,))
        self.b.feed(ctid, "tok", cu, 1, (tok,))
        self.collect = Row(ctid, cu, "o0", 2 * (KO + 2))
        self.collects.append((ctid, cu))
        if p == self.cls.positions - 1:
            etid, eu = self.b.unit(self.entry)
            self.b.feed(etid, "occ", eu, 1, (self.occ,))
            self.b.feed(etid, "state", eu, 1, (self.collect,))
            self.entry_unit = (etid, eu)


FAMILIES = ("Embedding", "RMSNormTriton", "RMSNormFusedCuda", "Gemm", "Int8RowQuant", "ScaledMmS8", "RoPE", "Attention", "SiluMul",
            "TokenSelect")


def class_program(registered: Any, cls: WindowClass) -> Any:
    """The class's request Program (a `build.BuildResult`): the Build's greedy request at one prompt row and
    ``cls.positions`` - 1 decode steps, in the registered compiler's form."""
    return request_program(registered, RequestUnit(-1, "", 1, cls.positions - 1, GREEDY, None, 0))


def stop_ids(registered: Any) -> tuple[int, ...]:
    """The registered model's stop ids, its config's ``eos_token_id`` (an id, a list or none): outputs end at the first, kept."""
    e = registered.phi.hf_config.get("eos_token_id")
    return tuple(sorted({int(x) for x in ([] if e is None else [e] if isinstance(e, int) else e)}))


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
                  name: str = NAME, provable: bool = True) -> IF.Circuit:
    """``cls``'s circuit (module docstring): the class Program lowered once a slot, node by node, slot k reading its lane of the
    scan state row before its anchor (`WindowClass.anchor`), each allocation's slots side by side in its template; its blocks'
    slots are the slots. Its registered parameters are the weights (`window_builder.weight_param`) and ``SCAN_STATE``'s
    ``cls.buckets`` rows, row a the scan's state row before bucket a (`ingress_scan.window_state`). A window's circuit is every
    slot's; ``programs`` is the first slots only, a probe's census (`class_size`). Each unit is a proof block Flock's composer
    proves (`_Builder.settle`) unless not ``provable`` (a probe's, or a test's of too small a class)."""
    return lower(registered, cls, programs, grain, name, provable=provable).circuit


def lower(registered: Any, cls: WindowClass, programs: int | None = None, grain: Grain = _GRAIN, name: str = NAME,
          wired: bool = True, provable: bool = True) -> Lowered:
    """`class_circuit`'s circuit with its lowering (`Lowered`); unless ``wired``, its units and registered reads alone, no
    wire or cut, whatever `settle` would refuse (the weights a registration commits, `window_weights`)."""
    programs = cls.slots if programs is None else programs
    if not 1 <= programs <= cls.slots:
        raise Unexpressible(f"{programs} Programs in a window of a class of {cls.slots} slots")
    try:
        scan = SS.bridge(cls.scan, cls.prompt)
    except ValueError as e:
        raise Unexpressible(f"{cls}: its slots aren't a scan state row's lanes ({e})") from None
    entry = bind(IO.SlotEntry, KO=cls.outputs, IDB=cls.scan.IDB, NP=cls.prompt, STOP=stop_ids(registered))
    b, fn = _Builder(grain, programs, provable), class_program(registered, cls).program.fn
    progs = [_Program(b, fn, cls, f"class Program {k}", k, scan, entry) for k in range(programs)]
    for (j, node), r in product(enumerate(fn.body.nodes), progs):
        r.lower(j, node)
    for r in progs:
        if r.position != cls.positions:
            raise Unexpressible(f"{r.name}: {r.position} positions, not its class's {cls.positions}")
    if wired:
        b.settle()
    roots, openings = b.merged()
    params = {**b.params, SS.SCAN_STATE: cls.buckets * scan.bindings["NS"]}
    reg = IF.Registered(dict(sorted(params.items())), tuple(b.tables.values()), tuple(IF.coalesce(b.reads)), tuple(roots),
                        tuple(openings))
    circuit = IF.Circuit(name, tuple((fn, b.units[tid]) for tid, fn in b.fns.items()),
                         external=(f"{descriptor_id(scan)}/state", f"{descriptor_id(entry)}/o0"),
                         wires=tuple(IF.coalesce(b.wires)), cuts=b.cuts, registered=reg,
                         blocks=IF.Blocks(programs, cls.positions, tuple(_inference(b.spans, programs))))
    return Lowered(circuit, fn, tuple(progs), b.spans)


def window_circuit(registered: Any, compiled: Any, cls: WindowClass, grain: Grain = _GRAIN, name: str = NAME,
                   placed: Sequence[tuple[int, int | None]] | None = None, provable: bool = True) -> IF.Circuit:
    """``compiled``'s window (`steps.Compiled`, of ``registered``) as ``cls``'s circuit, every slot's class Program whatever
    its traffic, refused where the class's slots don't take the window: nothing of the window reaches the circuit.
    ``placed`` is each request's anchor bucket and the bucket of its last output, in its request order, as the network
    certifier's scan placed it (the end None: unchecked); by default every request is at the window's first boundary, its
    end unchecked. A last output past the window's end is the scan's to carry (module docstring)."""
    takes(compiled, cls, placed)
    return class_circuit(registered, cls, None, grain, name, provable)


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


__all__ = ["CLASS_OF_RECORD", "CUTS", "FAMILIES", "Grain", "Lowered", "NAME", "Row", "WindowClass", "class_circuit",
           "class_program", "lower", "stop_ids", "takes", "window_circuit"]
