"""A unit whose circuit reads pinned tables (`tail_pieces.table_read`: MUFU.EX2, RSQ, SQRT and RCP), split at its reads into
the general instance form the soundness proof covers under hidden outputs (`Flock.ProvedScope.scopeOkI` with
`wiresAcyclic`): every read is a lookup slot, which the prover and the verifier build from the pinned table
(``live/src/lookup.rs``, ``Flock/Lookup.lean``), in place of the read's inline rows (`ir_lower._Read`: about ``31 * 2^22``
nonzeros a read of a 2^23-word table). The lincheck folds a slot type once however many slots it has, so the reads of one
table cost that table's nonzeros once.

A committed bit's depth is the number of reads on its deepest path from the inputs. The unit computes the bits of depth 0;
stage ``s`` takes the values of the reads at depth ``s`` and hands the reads at depth ``s + 1`` their indices; the last stage
returns the outputs. An AND of depth ``d >= 1`` is in the earliest stage that reads it, at or past ``d``: as late as it can
go, so a stage holds what the next reads' indices need and the rest waits for the stage that uses it (the products a read's
value weighs, in attention, wait for the normalization), which keeps each stage type's power-of-two slot from doubling. A bit
a later stage reads is a cut word of the unit (a depth-0 bit; :data:`PORT_BITS` to a port, grouped by the stages that read
them) or is handed on stage to stage (``pass`` ports, grouped by the last stage that reads them). The gates are the unit
circuit's own, each in one piece.
`circuit.compose` takes the split as its tail (``tail``), as it does `multi_unit`'s."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import gf2
from . import ir_lower as IL
from . import tail as TL

#: bits of a cut or pass port
PORT_BITS = 32
VALUE_MASK = (1 << TL.VALUE_BITS) - 1


@dataclass
class SplitLowering(IL.Lowering):
    """The split's unit (``layout``, its depth-0 part; ``units``' cut words its outputs) and its stages and reads (``tail``).
    :meth:`evaluate` runs the unit, then each stage on its cut words, the previous stage's ports and its reads' values."""

    tail: TL.Tail | None = None
    n_out: int = 0
    stage_asserts: int = 0

    @property
    def domain(self) -> str:
        return "finite-only" if self.layout.counts["asserts"] or self.stage_asserts else "total"

    @property
    def cut(self) -> dict | None:
        U, lay = self.units, self.layout
        return {"words": len(U.cut), "split": "lookup-slots", "in": [[] for _ in U.in_src],
                "out": [[[lay.out_port_cols[j] - lay.out_col * IL.WORD, p] for j, (k, p) in enumerate(src) if k == "cut"]
                        for src in U.out_src]}

    def evaluate(self, flat_inputs: np.ndarray, cuts: np.ndarray | None = None, flip: tuple[int, int] | None = None):
        """``(outputs (n, output words), the unit's cut words (n, ports), lane mask of instances whose every slot is
        satisfied)``."""
        if cuts is not None or flip is not None:
            raise NotImplementedError(f"{self.name}: a lookup split takes no cut words and flips no bit")
        flat = np.asarray(flat_inputs, dtype=np.uint64)
        n = flat.shape[0]
        outs, ok = IL.evaluate_layout(self.layout, self.unit_inputs(flat), n)
        cut = np.zeros((n, len(self.units.cut)), dtype=np.uint64)
        for j, (_k, p) in enumerate(self.units.out_src[0]):
            cut[:, p] = np.asarray(outs[j], dtype=np.uint64)
        res = np.zeros((n, self.n_out), dtype=np.uint64)
        prev: dict = {}
        for st in self.tail.stages:
            ports = [cut[:, tag[1]] if tag[0] == "cut" else prev[tag] for tag in st.ins]
            o, ok_s = IL.evaluate_layout(TL.stage_layout(st), ports, n)
            ok &= ok_s
            prev = {}
            for tag, w in zip(st.outs, o):
                w = np.asarray(w, dtype=np.uint64)
                if tag[0] == "index":
                    words = np.asarray(gf2.table(self.tail.lookups[tag[1]][0]), dtype=np.uint64)
                    prev[("lookup", tag[1])] = words[w.astype(np.int64)] & np.uint64(VALUE_MASK)
                elif tag[0] == "ret":
                    res[:, tag[1]] = w
                else:
                    prev[tag] = w
        return res, cut, ok


def _refuse_reads(C: gf2.Circuit, name: str) -> None:
    for t, idx, width in C.lookups:
        if t not in IL.TABLES or len(idx) != TL.TABLE_BITS.get(t) or width != TL.VALUE_BITS:
            raise NotImplementedError(f"{name}: a read of {t} ({len(idx)} index bits, {width} value bits) has no lookup slot: "
                                      f"the slots are the pinned tables {sorted(IL.TABLES)} at {TL.VALUE_BITS} value bits")


def split(C: gf2.Circuit, name: str, definition, n_in: int, n_out: int) -> SplitLowering:
    """``C`` (inputs ``in0 ..``, ``n_in`` u16 words; outputs ``out0 ..``, ``n_out`` u16 words) split at its reads. ``C``
    becomes the unit's circuit and is laid out and dropped; each stage is laid out as it is built and its circuit dropped."""
    _refuse_reads(C, name)
    kind, data = C.kind, C.data
    if "hint" in kind:
        raise NotImplementedError(f"{name}: a hint has no place in a lookup split")
    live = C.live()
    N = len(kind)
    depth = [0] * N
    read_depth: dict[int, int] = {}
    read_bits: dict[int, dict[int, int]] = defaultdict(dict)   # read -> value bit -> its committed bit
    for v in range(N):
        if v not in live:
            continue
        if kind[v] == "and":
            a, b = data[v]
            depth[v] = max((depth[x] for f in (a, b) for x in f.s), default=0)
        elif kind[v] == "lookup":
            g, j = data[v]
            if g not in read_depth:
                read_depth[g] = 1 + max((depth[x] for f in C.lookups[g][1] for x in f.s), default=0)
            depth[v] = read_depth[g]
            read_bits[g][j] = v
    if not read_depth:
        raise ValueError(f"{name}: no live table read to split at")
    top = max(read_depth.values())
    outs = [C.outputs[f"out{o}"] for o in range(n_out)]
    asserts = [(nm, w, max((depth[x] for x in w.s), default=0)) for nm, w in C.asserts]

    # where each bit is: the unit (0) for depth 0, a read's value at its depth, an AND at the earliest stage that reads it
    first = [top + 1] * N
    for o in outs:
        for w in o:
            for x in w.s:
                first[x] = min(first[x], top)
    for _nm, w, at in asserts:
        for x in w.s:
            first[x] = min(first[x], at)
    for g, d in read_depth.items():
        for f in C.lookups[g][1]:
            for x in f.s:
                first[x] = min(first[x], d - 1)
    place = list(depth)
    for v in reversed(range(N)):
        if v in live and kind[v] == "and":
            if depth[v]:
                place[v] = first[v] if first[v] <= top else depth[v]
            a, b = data[v]
            for f in (a, b):
                for x in f.s:
                    first[x] = min(first[x], place[v])

    last = [-1] * N                      # the last stage past its own that reads a bit of a stage
    cut_readers: dict[int, set] = defaultdict(set)   # a bit of the unit -> the stages that read it
    by_stage: dict[int, list[int]] = defaultdict(list)

    def need(x: int, t: int) -> None:
        if place[x] == 0:
            cut_readers[x].add(t)
        elif t > place[x] and t > last[x]:
            last[x] = t

    for v in range(N):
        if v in live and kind[v] == "and" and place[v]:
            by_stage[place[v]].append(v)
            a, b = data[v]
            for f in (a, b):
                for x in f.s:
                    need(x, place[v])
    for g, d in read_depth.items():
        for f in C.lookups[g][1]:
            for x in f.s:
                need(x, d - 1)
    for o in outs:
        for w in o:
            for x in w.s:
                need(x, top)
    unit_asserts, stage_asserts = [], defaultdict(list)
    for nm, w, at in asserts:
        if at:
            stage_asserts[at].append((nm, w))
            for x in w.s:
                need(x, at)
        else:
            unit_asserts.append((nm, w))

    # ports: the unit's cut words by the stages that read them, each stage's pass ports by the last stage that reads them
    groups: dict[tuple, list[int]] = defaultdict(list)
    for x in sorted(cut_readers):
        groups[tuple(sorted(cut_readers[x]))].append(x)
    cut_bits, cut_of_stage = [], defaultdict(list)
    for key in sorted(groups):
        bits = groups[key]
        for i in range(0, len(bits), PORT_BITS):
            for t in key:
                cut_of_stage[t].append(len(cut_bits))
            cut_bits.append(bits[i:i + PORT_BITS])
    pass_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for v in range(N):
        if last[v] > place[v] >= 1:
            pass_groups[(place[v], last[v])].append(v)
    pass_bits, new_pass = [], defaultdict(list)
    for (s, lst) in sorted(pass_groups):
        bits = pass_groups[(s, lst)]
        for i in range(0, len(bits), PORT_BITS):
            new_pass[s].append(len(pass_bits))
            pass_bits.append((lst, bits[i:i + PORT_BITS]))
    reads_at: dict[int, list[int]] = defaultdict(list)
    for g in sorted(read_depth):
        reads_at[read_depth[g]].append(g)
    position: dict[int, int] = {}
    lookups = []
    for s in range(top):
        for g in reads_at[s + 1]:
            position[g] = len(lookups)
            lookups.append((C.lookups[g][0], s))

    stages, carried, n_stage_asserts = [], [], 0
    for t in range(top + 1):
        Ct = type(C)()
        st = TL.Stage(Ct)
        m: dict[int, int] = {}

        def take(tag, bits, width=None):
            ws = Ct.input(f"in{len(st.ins)}", width or len(bits))
            st.ins.append(tag)
            st.in_widths.append(len(ws))
            for x, w in zip(bits, ws):
                if x is not None:
                    (m[x],) = tuple(w.s)

        for k in cut_of_stage[t]:
            take(("cut", k), cut_bits[k])
        for pid in carried:
            take(("pass", pid), pass_bits[pid][1])
        for g in reads_at[t]:
            take(("lookup", position[g]), [read_bits[g].get(j) for j in range(TL.VALUE_BITS)], TL.VALUE_BITS)

        def rf(f):
            return gf2.L(tuple(m[x] for x in f.s), f.c)

        for v in by_stage[t]:
            a, b = data[v]
            (m[v],) = tuple(Ct._new("and", (rf(a), rf(b))).s)
        for nm, w in stage_asserts[t]:
            Ct.asserts.append((nm, rf(w)))
            n_stage_asserts += 1
        ports = [(("pass", pid), [gf2.L((m[x],), 0) for x in pass_bits[pid][1]]) for pid in carried if pass_bits[pid][0] > t]
        ports += [(("pass", pid), [gf2.L((m[x],), 0) for x in pass_bits[pid][1]]) for pid in new_pass[t]]
        ports += [(("index", position[g]), [rf(f) for f in C.lookups[g][1]]) for g in reads_at[t + 1]]
        if t == top:
            ports += [(("ret", o), [rf(w) for w in outs[o]]) for o in range(n_out)]
        for j, (tag, forms) in enumerate(ports):
            Ct.output(f"out{j}", forms)
            st.outs.append(tag)
            st.out_widths.append(len(forms))
        TL.stage_layout(st)
        st.circuit = None
        stages.append(st)
        carried = [pid for pid in carried if pass_bits[pid][0] > t] + new_pass[t]

    C.outputs = {f"out{k}": [gf2.L((x,), 0) for x in bits] for k, bits in enumerate(cut_bits)}
    C.asserts = unit_asserts
    lay = IL.layout(C, [f"in{j}" for j in range(n_in)], [f"out{k}" for k in range(len(cut_bits))])
    units = IL.Units(target=None, in_widths=[16] * n_in, out_widths=[len(b) for b in cut_bits],
                     in_src=[[("leaf", p) for p in range(n_in)]], out_src=[[("cut", k) for k in range(len(cut_bits))]],
                     signature=(definition.id,), gates=[], arg_gates=[], ret_gates=[], cut=list(range(len(cut_bits))),
                     tail_rets=list(range(n_out)))
    tail = TL.Tail(stages, lookups, [], list(range(n_out)))
    return SplitLowering(name, definition, units, None, lay, tail=tail, n_out=n_out, stage_asserts=n_stage_asserts)


def counts(low: SplitLowering) -> dict:
    """Each slot type's ANDs and A, B nonzeros (the lincheck folds each type once) and its slots per VU; a lookup type's
    nonzeros are its product rows' (`lookup.rs`: one A entry per set bit of the table, below ``out_bits``)."""
    def one(lay, per_vu):
        c = lay.counts
        return {"ands": c["ands"], "nnz": c["nnz_a"] + c["nnz_b"], "rows": lay.useful, "per_vu": per_vu}

    out = {"unit": one(low.layout, low.units.n)}
    for i, st in enumerate(low.tail.stages):
        out[f"stage{i}"] = one(TL.stage_layout(st), st.count)
    per: dict[str, int] = defaultdict(int)
    for t, _s in low.tail.lookups:
        per[t] += 1
    for t, c in sorted(per.items()):
        words = np.asarray(gf2.table(t), dtype=np.uint32)
        nnz = sum(int(np.count_nonzero((words >> np.uint32(j)) & np.uint32(1))) for j in range(TL.VALUE_BITS))
        out[f"lookup:{t}"] = {"product_nnz": nnz, "per_vu": c}
    out["nnz"] = sum(v.get("nnz", 0) + v.get("product_nnz", 0) for v in out.values())
    return out
