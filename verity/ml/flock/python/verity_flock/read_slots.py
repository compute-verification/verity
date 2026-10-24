"""A unit's table reads as C-Flock lookup slots (PRF-139): the unit's circuit split at its reads of the pinned MUFU tables, by
read depth, into the unit and tail stages wired to ``lookup:<table>`` slots (PROTOCOL.md §16.5), the last stage returning the
outputs. A slot type's matrix is folded once per statement however many slots of it there are, so a unit's reads no longer
put their tables' set bits in its own matrix, as `ir_lower._Read`'s inline rows do (about 1e8 entries an EX2 read). Every
piece is a form the verifiers already take, inside the proved scope (`ProvedScope.scopeOkI` with `wiresAcyclic`): tail
stages, lookup slots, instance-local wires, and outputs a tail stage returns (`circuit.compose`'s ``tail``).

The split (:func:`split`): an input bit has depth 0, an AND its operands' greatest depth, a slot read's value bits 1 + its
index's greatest depth, and any other read (a table no slot holds) stays inline at its index's depth. Stage s holds the ANDs
and inline reads of depth s, computes the index of every read of depth s + 1, and receives each earlier bit it reads from the
stage that computes it and each slot read's value from its slot. Stage 0 is the unit, on the unit's own leaf ports; the
deepest stage returns the output words. Assertions go to the stage of their deepest bit."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from . import gf2
from . import ir_lower as IL
from . import tail as TL

#: the tables a lookup slot reads (`circuit.compose`'s META ``lookups``), each by its index width
SLOT_TABLES = {t: TL.TABLE_BITS[t] for t in IL.TABLES}


def _net(s: int) -> str:
    return "unit" if s == 0 else f"stage{s - 1}"


def _run(lay: IL.Layout, ins: Sequence[Sequence[int]], T: int, flip: tuple[int, int] | None = None) -> tuple[list[list[int]], int]:
    """`ir_lower.evaluate_layout` on bit-sliced ports of any width: ``ins[j]`` is input port j's bits (each a T-bit int, one bit
    per lane). Returns each output port's bits and the mask of lanes whose every row holds."""
    mask = (1 << T) - 1
    z = [0] * lay.useful
    for at, width, bits in zip(lay.in_port_cols, lay.in_bits, ins, strict=True):
        assert len(bits) == width, "an input port's bits are not its width"
        z[at:at + width] = bits
    z[lay.const] = mask

    def ev(r):
        s = 0
        for k in IL._cols_of(r):
            s ^= z[k]
        return s

    for i in range(lay.in_words * IL.WORD, lay.useful):
        if i != lay.const:
            z[i] = ev(lay.rows_a[i]) & ev(lay.rows_b[i])
    if flip is not None:
        z[flip[0]] ^= 1 << flip[1]
    ok = mask
    for i in range(lay.useful):
        ok &= ~((ev(lay.rows_a[i]) & ev(lay.rows_b[i])) ^ z[i]) & mask
    return [z[at:at + nb] for at, nb in zip(lay.out_port_cols, lay.out_bits)], ok


def _read(table: str, idx: Sequence[int], T: int) -> list[int]:
    """A lookup slot's value on every lane: the table word at the index bits ``idx`` (bit-sliced), its low ``VALUE_BITS``."""
    words = np.asarray(gf2.table(table), dtype=np.uint32)
    at = IL.unslice_words(idx, T).astype(np.int64)
    return IL.slice_words(words[at] & np.uint32((1 << TL.VALUE_BITS) - 1), TL.VALUE_BITS)


class ReadSlots(IL.Lowering):
    """A unit lowered with its slot reads split out (:func:`split`): ``layout`` is the unit's (stage 0) and ``tail`` the
    `tail.Tail` of the later stages, the lookup slots and the wires, which `circuit.compose` writes as they are. `evaluate`
    runs the unit, the slots and the stages in order and returns the output words the last stage returns."""

    evaluates_rets = True

    def __init__(self, name, definition, units, layout, tail, unit_outs: list, reads: list, asserts: int):
        super().__init__(name, definition, units, None, layout)
        self.tail = tail
        self.unit_outs = unit_outs
        #: per slot read, in slot order: its tag (the read's index in the unit circuit), table and the stage computing its index
        self.reads = reads
        self.asserts = asserts

    @property
    def domain(self) -> str:
        return "finite-only" if self.asserts else "total"

    @property
    def cut(self) -> dict:
        return {"words": len(self.units.cut), "read_slots": [[t, _net(s)] for _g, t, s in self.reads]}

    @property
    def counts(self) -> dict:
        """The unit's and the stages' rows summed (`ir_lower.Layout.counts`' keys), the lookup slots and the stages."""
        lays = [self.layout, *(st.layout for st in self.tail.stages)]
        c = {k: sum(lay.counts[k] for lay in lays) for k in ("ands", "asserts", "rows", "nnz_a", "nnz_b", "reads", "read_rows")}
        return {**c, "unit_ands": self.layout.counts["ands"], "stages": len(self.tail.stages), "lookups": len(self.reads)}

    def evaluate(self, flat_inputs: np.ndarray, cuts: np.ndarray | None = None, flip: tuple[int, int] | None = None):
        """``(outputs (n, output words), (n, 0), lane mask of instances whose unit, stages and reads all hold)``; ``flip`` flips a
        unit row's bit (`ir_lower.evaluate_layout`'s)."""
        if cuts is not None:
            raise NotImplementedError(f"{self.name}: the stages compute every cut word; none is given")
        x = np.asarray(flat_inputs, dtype=np.uint64)
        n = x.shape[0]
        ins = [IL.slice_words(w, b) for w, b in zip(self.unit_inputs(x), self.layout.in_bits)]
        outs, ok = _run(self.layout, ins, n * self.units.n, flip)
        produced = {0: dict(zip(self.unit_outs, outs))}
        values: dict[int, list[int]] = {}

        def read(s: int) -> None:
            for g, t, at in self.reads:
                if at == s:
                    values[g] = _read(t, produced[s][("index", g)], n)

        read(0)
        for s, st in enumerate(self.tail.stages, 1):
            ins = [produced[tag[1]][("to", s)] if tag[0] == "from" else values[tag[1]] for tag in st.ins]
            outs, ok_s = _run(st.layout, ins, n)
            ok &= ok_s
            produced[s] = dict(zip(st.outs, outs))
            read(s)
        last = produced[len(self.tail.stages)]
        res = np.zeros((n, len(self.units.tail_rets)), dtype=np.uint64)
        for o in range(res.shape[1]):
            res[:, o] = IL.unslice_words(last[("ret", o)], n)
        return res, np.zeros((n, 0), dtype=np.uint64), ok


def _pad(bits: int) -> int:
    """The rows `ir_lower.layout` gives one port group of ``bits`` bits: a power of two 128-bit words, none for no port."""
    return IL.ceil_pow2(-(-bits // IL.WORD)) * IL.WORD if bits else 0


def net_rows(ands: int, asserts: int, in_bits: int, out_bits: int) -> int:
    """A net's ``useful`` rows as `ir_lower.layout` lays out a `split` net (one input group, one output group): the input
    group, the ANDs and inline read rows, the assertions, the output group aligned to its own size, and the constant row."""
    i, o = _pad(in_bits), _pad(out_bits)
    body = i + ands + asserts
    return (-(-body // o) * o if o else body) + o + 1


def net_log(rows: int) -> int:
    """The least slot log a net of ``rows`` useful rows takes in `circuit.compose` (`circuit._slot_log`, and a net may not fill
    its slot)."""
    return max(7, rows.bit_length())


def ands_cap(log: int, asserts: int, in_bits: int, out_bits: int) -> int:
    """The most ANDs (and inline read rows) a net with these ports and assertions holds in a slot of ``2^log``: the largest
    ``a`` with ``net_log(net_rows(a, ...)) <= log``; at most 0 when its ports alone don't fit."""
    i, o = _pad(in_bits), _pad(out_bits)
    room = (1 << log) - 2 - o
    return (room // o * o if o else room) - i - asserts


class _Plan:
    """`split`'s placement of ``C``'s live gates in nets under per-net AND caps (``caps[s]``, the last one for every later
    net; None: no cap): each net's ANDs (``ands``) and ``(ANDs and inline read rows, assertions, input bits, output bits)``
    (``nets``), which `net_rows` turns into its layout's rows."""

    def __init__(self, C: gf2.Circuit, live, slot: Mapping[int, str], in_names, out_names, caps: Sequence[int] | None,
                 read_rows) -> None:
        kind, data, lookups = C.kind, C.data, C.lookups
        self.caps = None if caps is None else list(caps)
        depth = self.depth = [0] * len(kind)
        rd = self.rd = {}
        inline = self.inline = {}
        need = self.need = {}        # stage -> the earlier stages' bits it reads
        need_lk = self.need_lk = {}  # stage -> the slot reads whose values it reads
        inline_bits: dict[int, set[int]] = {}

        def fd(f) -> int:
            return max((depth[u] for u in f.s), default=0)

        def note(s: int, f) -> None:
            for u in f.s:
                if kind[u] == "lookup" and data[u][0] in slot:
                    need_lk.setdefault(s, set()).add(data[u][0])
                elif depth[u] < s:
                    need.setdefault(s, set()).add(u)

        fill: list[int] = []
        last = None if caps is None else caps[-1]
        for v in range(len(kind)):
            if v not in live or kind[v] == "in":
                continue
            if kind[v] == "and":
                a, b = data[v]
                s = max(fd(a), fd(b))
                if caps is not None:
                    while s < len(fill) and fill[s] >= (caps[s] if s < len(caps) else last):
                        s += 1
                    fill.extend([0] * (s + 1 - len(fill)))
                    fill[s] += 1
                depth[v] = s
                note(s, a)
                note(s, b)
                continue
            g = data[v][0]
            idx = lookups[g][1]
            if g in slot:
                if g not in rd:
                    rd[g] = 1 + max((fd(f) for f in idx), default=0)
                    for f in idx:
                        note(rd[g] - 1, f)
                depth[v] = rd[g]
            else:
                if g not in inline:
                    inline[g] = max((fd(f) for f in idx), default=0)
                    for f in idx:
                        note(inline[g], f)
                inline_bits.setdefault(g, set()).add(data[v][1])
                depth[v] = inline[g]
        D = self.D = max(depth[v] for v in live)
        self.a_at = [(nm, f, fd(f)) for nm, f in C.asserts]
        for _nm, f, s in self.a_at:
            note(s, f)
        for nm in out_names:
            for f in C.outputs[nm]:
                note(D, f)

        ands = self.ands = [0] * (D + 1)
        for v in live:
            if kind[v] == "and":
                ands[depth[v]] += 1
        rows = list(ands)
        for g, s in inline.items():
            rows[s] += read_rows(g, inline_bits[g])
        asserts = [0] * (D + 1)
        for _nm, _f, s in self.a_at:
            asserts[s] += 1
        ins = [sum(len(C.inputs[nm]) for nm in in_names)] + [0] * D
        outs = [0] * (D + 1)
        for s, us in need.items():
            ins[s] += len(us)
            for u in us:
                outs[depth[u]] += 1
        for s, gs in need_lk.items():
            ins[s] += TL.VALUE_BITS * len(gs)
        for g, s in rd.items():
            outs[s - 1] += len(lookups[g][1])
        outs[D] += sum(len(C.outputs[nm]) for nm in out_names)
        self.nets = list(zip(rows, asserts, ins, outs))
        self.logs = [net_log(net_rows(*n)) for n in self.nets]
        #: the nets' slots' cells, which `circuit.compose` places in the block beside the statement's fixed ranges
        self.cells = sum(1 << L for L in self.logs)


#: how many placements `split`'s ``stage_ands="auto"`` tries a slot size, each capping a net by the last one's ports
AUTO_ROUNDS = 6


def _auto(plan) -> _Plan:
    """The placement whose nets' slots take the fewest cells (then the fewest nets): no cap, then caps that fit every net in
    ``2^L`` for L one below the largest net's slot, then one below that, while it shrinks. A cap is `ands_cap` of the net's
    ports, assertions and inline read rows in the last placement, so it tightens as the cut moves the exports, and never
    loosens. A cap that would leave a net no AND gives L up."""
    best = plan(None)
    L = max(best.logs)
    while L > 7:
        L -= 1
        got, last, caps = None, best, None
        for _ in range(AUTO_ROUNDS):
            new = [ands_cap(L, a_ + r - n, i, o) for n, (r, a_, i, o) in zip(last.ands, last.nets)]
            if caps is not None:
                new = [min(c, caps[k] if k < len(caps) else caps[-1]) for k, c in enumerate(new)]
                new += caps[len(new):]
            caps = new
            if min(caps) < 1:
                break
            last = plan(caps)
            if max(last.logs) <= L:
                got = last
                break
        if got is None or (got.cells, len(got.nets)) >= (best.cells, len(best.nets)):
            break
        best = got
    return best


def split(C: gf2.Circuit, in_names: Sequence[str], out_names: Sequence[str], name: str, definition, units: IL.Units,
          slots: Mapping[str, int] = SLOT_TABLES, stage_ands: int | str | None = "auto") -> ReadSlots | None:
    """``C`` (a unit circuit on input ports ``in_names`` and 16-bit output ports ``out_names``, ``units`` its one unit's leaf
    ports) split at its slot reads (the module docstring), or None when it reads no table a slot holds (``slots``: table ->
    index width; a read's value must be at most `tail.VALUE_BITS` wide). ``C`` is only read.

    ``stage_ands`` caps a net's ANDs, each AND going to the first net at or after its operands' with room: more stages, each
    nearer a power of two, since every net's slot is one (a unit whose reads cut it into a few large nets can need a block
    twice the flat unit's). An int is the cap on every net, None no cap, and ``"auto"`` (`_auto`) the per-net caps whose nets'
    slots take the fewest cells, from each net's ports, exports and assertions (`ands_cap`); the statement's fixed ranges
    (compressions, lookups, mask) are the same whatever the cut, so the fewest cells is the smallest block. The chosen caps
    are the result's ``caps``."""
    kind, data, lookups = C.kind, C.data, C.lookups
    slot = {g: t for g, (t, idx, w) in enumerate(lookups) if slots.get(t) == len(idx) and w <= TL.VALUE_BITS}
    live = C.live()
    if not any(kind[v] == "lookup" and data[v][0] in slot for v in live):
        return None
    if any(kind[v] == "hint" for v in live):
        raise NotImplementedError(f"{name}: a hint in a unit whose reads go to lookup slots")
    if any(len(C.outputs[nm]) != 16 for nm in out_names) or units.n != 1:
        raise ValueError(f"{name}: one unit an instance, returning 16-bit output words")
    if isinstance(stage_ands, int) and stage_ands < 1:
        raise ValueError(f"{name}: stage_ands {stage_ands}: at least one AND a net")
    if not (stage_ands is None or stage_ands == "auto" or isinstance(stage_ands, int)):
        raise ValueError(f"{name}: stage_ands {stage_ands!r}: an int, None or 'auto'")

    sized: dict[tuple, int] = {}

    def read_rows(g: int, bits: set[int]) -> int:
        t, idx, _w = lookups[g]
        key = (t, len(idx), tuple(sorted(bits)))
        if key not in sized:
            sized[key] = IL._Read(C, g, sorted(bits), 0).rows
        return sized[key]

    def plan(caps):
        return _Plan(C, live, slot, in_names, out_names, caps, read_rows)

    P = _auto(plan) if stage_ands == "auto" else plan(None if stage_ands is None else [stage_ands])
    depth, rd, need, need_lk, a_at, D = P.depth, P.rd, P.need, P.need_lk, P.a_at, P.D
    reads = sorted(rd)
    # from_[s][p]: the bits stage s receives from stage p, in order
    from_ = {s: {} for s in range(D + 1)}
    for s, us in need.items():
        for u in sorted(us):
            from_[s].setdefault(depth[u], []).append(u)

    cls = type(C)
    stages = [cls() for _ in range(D + 1)]
    own: list = [None] * len(kind)
    imp: list[dict[int, gf2.L]] = [{} for _ in range(D + 1)]
    lk: list[dict[int, list]] = [{} for _ in range(D + 1)]
    tags_in: list[list] = [[] for _ in range(D + 1)]
    for nm in in_names:
        for u, w in zip(C.inputs[nm], stages[0].input(nm, len(C.inputs[nm]))):
            (b,) = tuple(u.s)
            own[b] = w
    for s in range(1, D + 1):
        S = stages[s]
        for p in sorted(from_[s]):
            for u, w in zip(from_[s][p], S.input(f"in{len(tags_in[s])}", len(from_[s][p]))):
                imp[s][u] = w
            tags_in[s].append(("from", p))
        for g in sorted(need_lk.get(s, ())):
            lk[s][g] = S.input(f"in{len(tags_in[s])}", TL.VALUE_BITS)
            tags_in[s].append(("lookup", g))
    if need_lk.get(0):
        raise AssertionError("the unit reads a slot's value")

    def form(s: int, f) -> gf2.L:
        acc: set[int] = set()
        c = f.c
        for u in f.s:
            if kind[u] == "lookup" and data[u][0] in slot:
                x = lk[s][data[u][0]][data[u][1]]
            elif depth[u] == s:
                x = own[u]
            else:
                x = imp[s][u]
            acc.symmetric_difference_update(x.s)
            c ^= x.c
        return gf2.L(frozenset(acc), c)

    made: dict[int, list] = {}
    for v in range(len(kind)):
        if v not in live or kind[v] == "in":
            continue
        s = depth[v]
        if kind[v] == "and":
            a, b = data[v]
            own[v] = stages[s].AND(form(s, a), form(s, b))
            continue
        g, j = data[v]
        if g in slot:
            continue
        if g not in made:
            t, idx, w = lookups[g]
            made[g] = stages[s].lookup(t, [form(s, f) for f in idx], w)
        own[v] = made[g][j]
    for nm, f, s in a_at:
        stages[s].assert_zero(nm, form(s, f))

    tags_out: list[list] = [[] for _ in range(D + 1)]

    def put(s: int, tag, bits) -> None:
        stages[s].output(f"out{len(tags_out[s])}", list(bits))
        tags_out[s].append(tag)

    for s in range(D + 1):
        for g in reads:
            if rd[g] - 1 == s:
                put(s, ("index", g), [form(s, f) for f in lookups[g][1]])
        for t in range(s + 1, D + 1):
            if s in from_[t]:
                put(s, ("to", t), [own[u] for u in from_[t][s]])
    for o, nm in enumerate(out_names):
        put(D, ("ret", o), [form(D, f) for f in C.outputs[nm]])
    own.clear()
    made.clear()
    for d in imp:
        d.clear()

    lays = []
    for s in range(D + 1):
        ins = list(in_names) if s == 0 else [f"in{j}" for j in range(len(tags_in[s]))]
        lays.append(IL.layout(stages[s], ins, [f"out{j}" for j in range(len(tags_out[s]))]))
        stages[s] = None

    local: dict[int, int] = {}
    seen: dict[str, int] = {}
    for g in reads:
        local[g] = seen.get(slot[g], 0)
        seen[slot[g]] = local[g] + 1
    wires = []
    for s in range(1, D + 1):
        for j, tag in enumerate(tags_in[s]):
            src = ([_net(tag[1]), 0, tags_out[tag[1]].index(("to", s))] if tag[0] == "from"
                   else [f"lookup:{slot[tag[1]]}", local[tag[1]], 0])
            wires.append([_net(s), 0, j, *src])
    for g in reads:
        wires.append([f"lookup:{slot[g]}", local[g], 0, _net(rd[g] - 1), 0, tags_out[rd[g] - 1].index(("index", g))])

    W = len(out_names)
    tail = TL.Tail([TL.Stage(None, tags_in[s], tags_out[s], lays[s].in_bits, lays[s].out_bits, layout=lays[s]) for s in range(1, D + 1)],
                   [(slot[g], rd[g] - 2) for g in reads], [], wires=wires, slotted_rets=(16 * W) & (16 * W - 1) != 0)
    U = IL.Units(target=units.target, in_widths=list(units.in_widths), out_widths=lays[0].out_bits, in_src=units.in_src,
                 out_src=[[("cut", k) for k in range(len(tags_out[0]))]], signature=units.signature, gates=[], arg_gates=[],
                 ret_gates=[], cut=list(range(len(tags_out[0]))), public=list(units.public), tail_rets=list(range(W)))
    low = ReadSlots(name, definition, U, lays[0], tail, tags_out[0], [(g, slot[g], rd[g] - 1) for g in reads], len(C.asserts))
    low.caps = P.caps
    return low
