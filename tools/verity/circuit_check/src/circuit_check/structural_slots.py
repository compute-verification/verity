"""``circuit-check --structural DIR TARGET`` where DIR holds a split (`verity_flock.struct_slots`, ``struct-slots/v1``: a
``flock-ir-tmpl`` template a net and ``slots.json``): its nets, composed through its lookup slots by its wires, held to the
Definition TARGET's flat lowering as `structural` holds one template. circuit-check imports nothing of Flock: the split, its
format's name, its slot tables' widths and their words come from the reader its module's sidecar defines
(`targets.struct_slots_split`).

The nets are composed by the wires alone, as the statement composes them: the unit's input ports are the Definition's input
leaves; each later net's input port is the one port a wire into it names, of an earlier net or of a ``lookup:<table>``
slot, at its width; a slot's index is the one earlier net's port its wire names, at the table's index width; and the
outputs are the last net's ports tagged ``ret``, in order. Each net is read as `structural` reads a template. A slot's word
is uninterpreted on both sides (`structural.Canon.READ`: one node a bit per table and index form), and the flat lowering
reads each read of a slot table that way (`structural.lowering_bits` with ``slots``), so a syntactic verdict says the split
computes the lowering's function of its inputs and the tables' words, whatever the tables hold. The semantic fallback reads
the tables. Refused by name, besides what `structural` refuses of a net: a wire that isn't six fields, its nets named and
its indices and ports ints; a slot none of the split's lookups is, or whose index is a slot's word rather than an earlier
net's output port (so no wire loops); an input port no wire fills or two do, a wire into the unit or into a port that isn't
there, a wire from a port that isn't there or from a net that isn't earlier, a width that disagrees, and port tags that
aren't the last net's ports."""
from __future__ import annotations

import operator
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import checks as CK
from . import structural as ST
from . import targets as TG


class Split:
    """A split as `structural._check`'s side: its counts, its inputs, its canonical output bits and its evaluation."""

    def __init__(self, sp: Any, slots: dict[str, tuple[int, int]], words: Callable[[str], Sequence[int]]) -> None:
        if not sp.nets:
            raise ST.Refused("malformed", "a split of no nets")
        self.sp, self.slots, self.words = sp, slots, words
        self.names = [("unit" if s == 0 else f"stage{s - 1}") for s in range(len(sp.nets))]
        self.ports = [ST.ports(t.groups) for t in sp.nets]
        self.inputs = sum(map(len, self.ports[0][0]))
        self.into = self._wires()

    def _wires(self) -> dict[tuple, tuple]:
        """Each wire checked before any net is composed: an index or a port is an int; a slot end is one of the split's
        lookups, port 0; a slot's index is a net's output port, never a slot's word, so no wire loops; a stage's input port
        is a port of the stage, wired once, from a net's output port or a slot's word; and every slot has an index."""
        net_of = {nm: s for s, nm in enumerate(self.names)}
        count: dict[str, int] = {}
        for lk in self.sp.lookups if isinstance(self.sp.lookups, list) else [None]:
            if not (isinstance(lk, tuple) and len(lk) == 2 and isinstance(lk[0], str) and lk[0] in self.slots):
                raise ST.Refused("wires", f"slot {lk} is not (table, net) of a slot table {sorted(self.slots)}")
            count[lk[0]] = count.get(lk[0], 0) + 1

        def slot(net: str, i: int, p: int, what: str) -> bool:
            if not net.startswith("lookup:"):
                return False
            if not 0 <= i < count.get(net.removeprefix("lookup:"), 0) or p != 0:
                raise ST.Refused("wires", f"{what} {net}[{i}] port {p}: no slot's {'index' if what == 'into' else 'word'} "
                                          f"(the split's lookups are {count})")
            return True

        def out_port(net: str, i: int, p: int, what: str) -> None:
            t = net_of.get(net)
            if t is None or i != 0 or not 0 <= p < len(self.ports[t][1]):
                raise ST.Refused("wires", f"{what} reads {net}[{i}] port {p}: no net's output port")

        into: dict[tuple, tuple] = {}
        for w in self.sp.wires if isinstance(self.sp.wires, list) else [None]:
            if not (isinstance(w, list) and len(w) == 6 and isinstance(w[0], str) and isinstance(w[3], str)
                    and all(type(x) is int for x in (w[1], w[2], w[4], w[5]))):
                raise ST.Refused("wires", f"wire {w} is not [dst net, dst index, dst port, src net, src index, src port], "
                                          f"nets named and indices and ports ints")
            dst, di, dp, src, si, spt = w
            if slot(dst, di, dp, "into"):
                if src.startswith("lookup:"):
                    raise ST.Refused("wires", f"{dst}[{di}]'s index is {src}[{si}]'s word: a slot's index is an earlier "
                                              f"net's output port")
                out_port(src, si, spt, f"{dst}[{di}]'s index")
            else:
                s = net_of.get(dst)
                if not s or di != 0 or not 0 <= dp < len(self.ports[s][0]):
                    raise ST.Refused("wires", f"a wire into {dst}[{di}] port {dp}: no input port of a stage")
                if not slot(src, si, spt, f"{dst} port {dp} reads"):
                    out_port(src, si, spt, f"{dst} port {dp}")
            if (dst, di, dp) in into:
                raise ST.Refused("wires", f"{dst}[{di}] port {dp} is wired twice")
            into[(dst, di, dp)] = (src, si, spt)
        unwired = [f"lookup:{t}[{i}]" for t, n in count.items() for i in range(n) if (f"lookup:{t}", i, 0) not in into]
        if unwired:
            raise ST.Refused("wires", f"{unwired[0]}'s index has no wire")
        return into

    def counts(self) -> dict:
        nets = self.sp.nets
        return {"nets": len(nets), "lookups": len(self.sp.lookups), "wires": len(self.sp.wires),
                "defs": sum(len(t.defs) for t in nets), "template_rows": sum(len(d.rows) for t in nets for d in t.defs),
                "template_calls": sum(len(d.calls) for t in nets for d in t.defs)}

    def _compose(self, inputs: list, one: int, AND, READ) -> tuple[list, dict]:
        """The last net's ``ret`` ports' bits, each net walked (`structural._walk`) on the bits its wires give it, a slot's
        word ``READ(table, its index's bits)``; and what it took."""
        names, into = self.names, self.into
        net_of = {nm: s for s, nm in enumerate(names)}
        outs: list[list[list]] = []
        words: dict[tuple[str, Any], list] = {}
        total = {"rows": 0, "evaluated_rows": 0, "ands": 0, "copies": 0}
        nets = []

        def port(src: tuple, s: int) -> list:
            """The bits of ``src`` (a checked wire's source) for net ``s``: a slot's index is a net's port (`_wires`), so
            this recurses at most once."""
            sn, si, spt = src
            if sn.startswith("lookup:"):
                table = sn.removeprefix("lookup:")
                if (table, si) not in words:
                    idx = port(into[(sn, si, 0)], s)
                    if len(idx) != self.slots[table][0]:
                        raise ST.Refused("wires", f"{sn}[{si}]'s index is {len(idx)} bits, the table's "
                                                  f"{self.slots[table][0]}")
                    words[(table, si)] = READ(table, idx)
                return words[(table, si)]
            t = net_of[sn]
            if t >= s:
                raise ST.Refused("wires", f"{names[s]} reads {sn}, no earlier net")
            return outs[t][spt]

        for s, net in enumerate(self.sp.nets):
            in_ports, out_ports, _pad = self.ports[s]
            x = list(inputs) if s == 0 else []
            for p, cols in enumerate(in_ports if s else ()):
                src = into.get((names[s], 0, p))
                if src is None:
                    raise ST.Refused("wires", f"{names[s]} input port {p} has no wire")
                bits = port(src, s)
                if len(bits) != len(cols):
                    raise ST.Refused("wires", f"{names[s]} input port {p} is {len(cols)} bits, its wire's {len(bits)}")
                x += bits
            val, rec = ST._walk(net, x, one, AND)
            held = [c for c in rec["pad"] if val[c]]
            if held:
                raise ST.Refused("ports", f"{names[s]}'s column {held[0]} is in a port group's words but no port's bit, "
                                          f"and is not zero")
            outs.append([[val[c] or 0 for c in cols] for cols in out_ports])
            for k in total:
                total[k] += rec[k]
            nets.append({"net": names[s], "rows": rec["rows"], "evaluated_rows": rec["evaluated_rows"]})
        last = len(outs) - 1
        tags = self.sp.outs[last] if isinstance(self.sp.outs, list) and len(self.sp.outs) == len(outs) else None
        if not isinstance(tags, list) or len(tags) != len(outs[last]):
            raise ST.Refused("ports", f"{names[last]} has {len(outs[last])} output ports, tagged {tags}")
        rets = [p for p, tag in enumerate(tags) if isinstance(tag, tuple) and tag[:1] == ("ret",)]
        if [tags[p][1:] for p in rets] != [(j,) for j in range(len(rets))] or any(type(tags[p][1]) is not int for p in rets):
            raise ST.Refused("ports", f"{names[last]}'s ret ports are not the outputs in order")
        mine = [b for p in rets for b in outs[last][p]]
        return mine, {**total, "input_widths": [len(p) for p in self.ports[0][0]], "output_bits": len(mine),
                      "slot_words": len(words), "nets": nets}

    def bits(self, K: ST.Canon) -> tuple[list[int], dict]:
        mine, rec = self._compose(K.inputs, 1, K.AND, lambda t, idx: K.READ(t, idx, self.slots[t][1]))
        return [b or ST.ZERO for b in mine], rec

    def evaluate(self, lanes: list[int], T: int) -> list[int]:
        n = (T + 7) // 8

        def read(table: str, idx: list[int]) -> list[int]:
            bits = np.stack([np.unpackbits(np.frombuffer(b.to_bytes(n, "little"), np.uint8), bitorder="little")[:T]
                             for b in idx]).astype(np.int64)
            w = np.asarray(self.words(table), np.uint32)[(bits << np.arange(len(idx))[:, None]).sum(axis=0)]
            return [int.from_bytes(np.packbits(((w >> j) & 1).astype(np.uint8), bitorder="little").tobytes(), "little")
                    for j in range(self.slots[table][1])]

        return self._compose(lanes, (1 << T) - 1, operator.and_, read)[0]


def check(target: Any, path: Path, *, exhaustive_bits: int = ST.EXHAUSTIVE_BITS, vectors: int = ST.VECTORS,
          seed: int = CK.Options.seed, label: str | None = None) -> dict:
    """The split in the directory ``path``, read by the sidecar that defines its reader (`targets.struct_slots_split`), held
    to ``target``'s flat lowering, as `structural.check` holds a template."""
    from verity.circuits.evaluation.reference import resolve

    t = resolve(target)
    read, form, slots, words = TG.struct_slots_split(str(path))

    def side() -> Split:
        try:
            sp = read()
        except (OSError, KeyError, TypeError, AssertionError, IndexError, ValueError) as e:
            raise ST.Refused("malformed", f"not a {form} directory: {type(e).__name__} {e}") from None
        return Split(sp, slots, words)

    return ST._check(side, lambda: ST._lowered(t), label or t.id, {"definition": t.id, "split": str(path)},
                     ("the Definition's", "input leaves"), exhaustive_bits, vectors, seed)


__all__ = ["Split", "check"]
