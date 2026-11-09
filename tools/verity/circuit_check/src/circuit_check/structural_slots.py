"""``circuit-check --structural DIR TARGET`` where DIR holds a split (`verity_flock.struct_slots`, ``struct-slots/v1``: a
``flock-ir-tmpl`` template a net and ``slots.json``): its nets, composed through its lookup slots by its wires, held to the
Definition TARGET's flat lowering as `structural` holds one template.

The nets are composed by the wires alone, as the statement composes them: the unit's input ports are the Definition's input
leaves; each later net's input port is the one port a wire into it names, of an earlier net or of a ``lookup:<table>``
slot, at its width; a slot's index is the one earlier net's port its wire names, at the table's index width; and the
outputs are the last net's ports tagged ``ret``, in order. Each net is read as `structural` reads a template. A slot's word
is uninterpreted on both sides (`structural.Canon.READ`: one node a bit per table and index form), and the flat lowering
reads each read of a slot table that way (`structural.lowering_bits` with ``slots``), so a syntactic verdict says the split
computes the lowering's function of its inputs and the tables' words, whatever the tables hold. The semantic fallback reads
the tables. Refused by name, besides what `structural` refuses of a net: an input port no wire fills or two do, a wire into
the unit or into a port that isn't there, a wire from a port that isn't there or from a net that isn't earlier, a width
that disagrees, and port tags that aren't the last net's ports."""
from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import numpy as np

from . import checks as CK
from . import structural as ST


class Split:
    """A split as `structural._check`'s side: its counts, its inputs, its canonical output bits and its evaluation."""

    def __init__(self, sp: Any, slots: dict[str, tuple[int, int]]) -> None:
        if not sp.nets:
            raise ST.Refused("malformed", "a split of no nets")
        self.sp, self.slots = sp, slots
        self.names = [("unit" if s == 0 else f"stage{s - 1}") for s in range(len(sp.nets))]
        self.ports = [ST.ports(t.groups) for t in sp.nets]
        self.inputs = sum(map(len, self.ports[0][0]))
        self.into = self._wires()

    def _wires(self) -> dict[tuple, tuple]:
        net_of = {nm: s for s, nm in enumerate(self.names)}
        into: dict[tuple, tuple] = {}
        for w in self.sp.wires:
            if len(w) != 6:
                raise ST.Refused("wires", f"wire {w} is not [dst net, dst index, dst port, src net, src index, src port]")
            dst, di, dp, src, si, spt = w
            s = net_of.get(dst)
            if isinstance(dst, str) and dst.startswith("lookup:"):
                if dst.removeprefix("lookup:") not in self.slots or dp != 0:
                    raise ST.Refused("wires", f"a wire into {dst}[{di}] port {dp}: no slot's index")
            elif not s or di != 0 or not 0 <= dp < len(self.ports[s][0]):
                raise ST.Refused("wires", f"a wire into {dst}[{di}] port {dp}: no input port of a stage")
            if (dst, di, dp) in into:
                raise ST.Refused("wires", f"{dst}[{di}] port {dp} is wired twice")
            into[(dst, di, dp)] = (src, si, spt)
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
            sn, si, spt = src
            if isinstance(sn, str) and sn.startswith("lookup:"):
                table = sn.removeprefix("lookup:")
                if table not in self.slots or spt != 0:
                    raise ST.Refused("wires", f"{names[s]} reads {sn}[{si}] port {spt}: no slot's word")
                if (table, si) not in words:
                    ix = into.get((sn, si, 0))
                    if ix is None:
                        raise ST.Refused("wires", f"{sn}[{si}]'s index has no wire")
                    idx = port(ix, s)
                    if len(idx) != self.slots[table][0]:
                        raise ST.Refused("wires", f"{sn}[{si}]'s index is {len(idx)} bits, the table's "
                                                  f"{self.slots[table][0]}")
                    words[(table, si)] = READ(table, idx)
                return words[(table, si)]
            t = net_of.get(sn)
            if t is None or t >= s or si != 0:
                raise ST.Refused("wires", f"{names[s]} reads {sn}[{si}], no earlier net")
            if not 0 <= spt < len(outs[t]):
                raise ST.Refused("wires", f"{names[s]} reads {sn} port {spt}, which {sn} hasn't")
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
        tags = self.sp.outs[last]
        if len(tags) != len(outs[last]):
            raise ST.Refused("ports", f"{names[last]} has {len(outs[last])} output ports, {len(tags)} tagged")
        rets = [p for p, tag in enumerate(tags) if tag and tag[0] == "ret"]
        if [tags[p][1] for p in rets] != list(range(len(rets))):
            raise ST.Refused("ports", f"{names[last]}'s ret ports are not the outputs in order")
        mine = [b for p in rets for b in outs[last][p]]
        return mine, {**total, "input_widths": [len(p) for p in self.ports[0][0]], "output_bits": len(mine),
                      "slot_words": len(words), "nets": nets}

    def bits(self, K: ST.Canon) -> tuple[list[int], dict]:
        mine, rec = self._compose(K.inputs, 1, K.AND, lambda t, idx: K.READ(t, idx, self.slots[t][1]))
        return [b or ST.ZERO for b in mine], rec

    def evaluate(self, lanes: list[int], T: int) -> list[int]:
        from verity_flock import gf2

        n = (T + 7) // 8

        def read(table: str, idx: list[int]) -> list[int]:
            bits = np.stack([np.unpackbits(np.frombuffer(b.to_bytes(n, "little"), np.uint8), bitorder="little")[:T]
                             for b in idx]).astype(np.int64)
            w = np.asarray(gf2.table(table), np.uint32)[(bits << np.arange(len(idx))[:, None]).sum(axis=0)]
            return [int.from_bytes(np.packbits(((w >> j) & 1).astype(np.uint8), bitorder="little").tobytes(), "little")
                    for j in range(self.slots[table][1])]

        return self._compose(lanes, (1 << T) - 1, operator.and_, read)[0]


def check(target: Any, path: Path, *, exhaustive_bits: int = ST.EXHAUSTIVE_BITS, vectors: int = ST.VECTORS,
          seed: int = CK.Options.seed, label: str | None = None) -> dict:
    """The split in the directory ``path`` (`verity_flock.struct_slots.read`) held to ``target``'s flat lowering, as
    `structural.check` holds a template."""
    from verity_flock import struct_slots as SS

    from verity.ml.circuits.evaluation.reference import resolve

    t = resolve(target)

    def side() -> Split:
        try:
            sp = SS.read(Path(path))
        except (OSError, KeyError, TypeError, AssertionError, IndexError, ValueError) as e:
            raise ST.Refused("malformed", f"not a {SS.FORMAT} directory: {type(e).__name__} {e}") from None
        return Split(sp, SS.SLOTS)

    return ST._check(side, lambda: ST._lowered(t), label or t.id, {"definition": t.id, "split": str(path)},
                     ("the Definition's", "input leaves"), exhaustive_bits, vectors, seed)


__all__ = ["Split", "check"]
