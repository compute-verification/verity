"""``verity/flock-circuit/types``: the circuit statement with its unit as circuit types, not rows. A statement carries the types,
the partition and the layouts. The verifier derives the unit's rows itself (`Flock.Derive`, the soundness lane's), and the
prover gets them as data beside the statement (:attr:`Typed.unit_rows`, :mod:`derive`'s mirror), so staging needs no Lean.

It is :func:`circuit.compose`'s statement with two changes:
- **META** names the new statement (`FORMAT`, `STATEMENT`) and gains `types` (each circuit type's content by its digest),
  `layouts`, `unit_layout`, `unit_type`, `policy` and `unit_in` (the unit's constant-bound inputs with their binding
  times; none yet). Every other key is unchanged, `unit_sha256` included: the SHA-256 of the lowering's unit text, the
  derived rows followed by its `LEAVES` (and `CUT`) lines, as before.
- **The unit's `CIRCUIT` section is gone.** The row hashing's sections stay, as the block level does.

**A flat class** (no tail) keeps its block as it is: its unit is one flat type, whose rows are the old unit section byte
for byte (checked on every compose).

**A template with a tail** (attention, GEMM) is typed as one instance type (:func:`instance_type`): the tail stages' gates
are its own rows, its units calls of the unit type, and its MUFU reads reads of the library tables. Its layout places the
unit calls and the reads (generated `table/v2` layouts) in block ranges of their own, per instance, beside its own region
and the row hashing. So META drops the flat statement's wiring keys (:data:`WIRING_KEYS`: leaves, wires, leaf cuts, the
tail's lookups) and recomputes `ranges`, `per_vu`, `k_log` and `vus_per_block`. The instance type's inputs are the
instance's input leaves in order, and its outputs its output leaves: the rows' words, bit by bit.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from verity.ml.circuits import library

from . import circuit as C
from . import circuit_types as CT
from . import derive as D
from . import ir_lower as IL
from . import layouts as LY

FORMAT = "flock-circuit/types"
STATEMENT = "verity/flock-circuit/types"
POLICY = "calls/v1"


@dataclass
class Typed:
    """A typed statement, and the prover's copy of the unit's rows (``flock-ir-unit/v2``), derived from its types. For a
    template with a tail, ``rows`` also holds each placed layout's rows (by digest) and the unit's Δ (``delta``)."""
    text: str
    unit_rows: str
    types: dict
    layouts: list
    rows: dict | None = None
    tables: dict | None = None
    derivation: object = None


def unit_type(low: IL.Lowering) -> CT.CircuitType:
    """The unit circuit as a flat circuit type, its ports ``in0 ..`` and ``out0 ..`` one word each."""
    U, circ = low.units, low.circuit
    ins = [b for j in range(len(U.in_widths)) for b in circ.inputs[f"in{j}"]]
    outs = [f for j in range(len(U.out_widths)) for f in circ.outputs[f"out{j}"]]
    return CT.from_gf2(circ, ins, outs, [(1, w) for w in U.in_widths], [(1, w) for w in U.out_widths], library.PROGRAM, {})


def _split(groups) -> tuple[int, ...]:
    """The port indices where a layout's second and later port groups start."""
    return tuple(itertools.accumulate(len(bits) for _, _, bits in groups))[:-1]


def unit_layout(low: IL.Lowering, t: CT.CircuitType) -> LY.Layout:
    """The unit's layout: placed by the block (no range), its port groups as the lowering laid them out."""
    return LY.Layout(t.digest(), None, _split(low.layout.in_groups), _split(low.layout.out_groups), ("derive",), (), ())


def sections(text: str) -> tuple[str, dict, list[tuple[str, str]]]:
    """A circuit file's first line, its META, and its ``CIRCUIT`` sections in order as (name, text)."""
    lines = text.split("\n")
    head, meta = lines[0], json.loads(lines[1][len("META "):])
    out, i = [], 2
    while i < len(lines) and lines[i]:
        _, name, count = lines[i].split(" ")
        n = int(count)
        out.append((name, "\n".join(lines[i + 1:i + 1 + n]) + "\n"))
        i += 1 + n
    return head, meta, out


@dataclass
class Instance:
    """A template instance as one circuit type: its units as calls of the unit type, its tail stages' gates as its own
    ANDs, and its MUFU reads as reads of the library tables."""
    root: CT.CircuitType
    unit: CT.CircuitType
    tables: list  # (name, sha512, index bits, value bits), in the root's first-use order


def _xor(forms, c=0):
    s: set[int] = set()
    for fs, fc in forms:
        s ^= set(fs)
        c ^= fc
    return tuple(sorted(s)), c


def instance_type(low: IL.Lowering, tail) -> Instance:
    """The instance (one template instance) of a template with a tail as a circuit type. Its inputs are the instance's
    input leaves and its outputs its output leaves, in order. Items are emitted as they become ready, since a stage both
    feeds units and reads them (GEMM's first stage makes the initial accumulator and finishes the last): a stage gate
    once its operands are, a unit call once its cut words are, a read once its index is. Items no output reads are
    dropped."""
    from . import tail as TL

    U = low.units
    unit_t = unit_type(low)
    lv = low.leaves
    in_ports, out_ports = lv["in_ports"], lv["out_ports"]
    leaf_wires, w = [], 0
    for count, bits in in_ports:
        for _ in range(count):
            leaf_wires.append(list(range(w, w + bits)))
            w += bits
    n = w
    items: list = []
    tables: list = []
    zero = ((), 0)

    def fit(forms, width):
        return (list(forms) + [zero] * width)[:width]

    cut: dict[int, list] = {k: [((x,), 0) for x in leaf_wires[p]] for k, p in U.public}
    out_leaf: dict[int, list] = {}
    import heapq

    maps = [dict() for _ in tail.stages]
    got = [set() for _ in tail.stages]
    # each stage's live ANDs by dependency count: an AND is emitted once every bit its forms read is mapped
    waiting, readers, ready = [], [], []
    for st in tail.stages:
        C_ = st.circuit
        wt, rd = {}, {}
        for v in sorted(x for x in C_.live() if C_.kind[x] == "and"):
            a, b = C_.data[v]
            deps = set(a.s) | set(b.s)
            wt[v] = len(deps)
            for d in deps:
                rd.setdefault(d, []).append(v)
        waiting.append(wt)
        readers.append(rd)
        ready.append([v for v, c in wt.items() if c == 0])
        heapq.heapify(ready[-1])
    passed: dict[tuple, list] = {}
    index: dict[int, list] = {}
    value: dict[int, list] = {}
    called, read = set(), set()

    def mapped(si, f):
        m = maps[si]
        if any(v not in m for v in f.s):
            return None
        return _xor([m[v] for v in f.s], f.c)

    def bind(si, v, f):
        maps[si][v] = f
        for r in readers[si].get(v, ()):
            waiting[si][r] -= 1
            if waiting[si][r] == 0:
                heapq.heappush(ready[si], r)

    def source(si, tag, width):
        if tag[0] == "cut":
            return fit(cut[tag[1]], width) if tag[1] in cut else None
        if tag[0] in ("pass", "ctx"):
            return passed.get((si - 1, tag))
        return value.get(tag[1])

    progress = True
    while progress:
        progress = False
        for si, st in enumerate(tail.stages):
            C_ = st.circuit
            for idx, tag in enumerate(st.ins):
                bits = C_.inputs[f"in{idx}"]
                (v0,) = tuple(bits[0].s)
                if v0 in maps[si]:
                    continue
                forms = source(si, tag, st.in_widths[idx])
                if forms is not None:
                    for b, f in zip(bits, forms):
                        (v,) = tuple(b.s)
                        bind(si, v, f)
                    progress = True
            while ready[si]:
                v = heapq.heappop(ready[si])
                a, b = C_.data[v]
                items.append((CT.AND, mapped(si, a), mapped(si, b)))
                bind(si, v, ((n,), 0))
                n += 1
                progress = True
            for idx, tag in enumerate(st.outs):
                if idx in got[si]:
                    continue
                forms = [mapped(si, f) for f in C_.outputs[f"out{idx}"]]
                if any(f is None for f in forms):
                    continue
                got[si].add(idx)
                progress = True
                if tag[0] in ("pass", "ctx"):
                    passed[(si, tag)] = forms
                elif tag[0] == "index":
                    index[tag[1]] = forms
                elif tag[0] == "cut":
                    cut[tag[1]] = forms
                else:
                    out_leaf[tag[1]] = forms
        for j, (table, _si) in enumerate(tail.lookups):
            if j in read or j not in index:
                continue
            ent = (library.table(table).sha512, TL.TABLE_BITS[table], TL.VALUE_BITS)
            if ent not in tables:
                tables.append(ent)
            items.append((CT.READ, tables.index(ent), tuple(index[j])))
            value[j] = [((n + i,), 0) for i in range(ent[2])]
            n += ent[2]
            read.add(j)
            progress = True
        for u in range(U.n):
            if u in called:
                continue
            args = []
            for j, (kind, k) in enumerate(U.in_src[u]):
                width = U.in_widths[j]
                if kind == "leaf":
                    args.append(fit([((x,), 0) for x in leaf_wires[k]], width))
                elif kind == "zero":
                    args.append([zero] * width)
                elif k in cut:
                    args.append(fit(cut[k], width))
                else:
                    args = None
                    break
            if args is None:
                continue
            items.append((CT.CALL, 0, tuple(f for a in args for f in a)))
            for j, (kind, k) in enumerate(U.out_src[u]):
                forms = [((n + i,), 0) for i in range(U.out_widths[j])]
                n += U.out_widths[j]
                (cut if kind == "cut" else out_leaf)[k] = forms
            called.add(u)
            progress = True
    total_out = sum(c for c, _ in out_ports)
    if len(called) < U.n or len(read) < len(tail.lookups) or len(out_leaf) < total_out:
        raise ValueError(f"{low.name}: the instance's units, stages and reads don't order")
    outputs = [f for p, (_, bits) in zip(range(total_out), [(0, b) for c, b in out_ports for _ in range(c)])
               for f in fit(out_leaf[p], bits)]
    root = _prune(CT.CircuitType(library.PROGRAM, tuple((c, b) for c, b in in_ports), tuple((c, b) for c, b in out_ports),
                                 (unit_t.digest(),), tuple(tables), tuple(items), tuple(outputs)), unit_t)
    names = {library.table(name).sha512: name for name in {t for t, _ in tail.lookups}}
    return Instance(root, unit_t, [(names[s], s, ib, vb) for s, ib, vb in root.tables])


def _prune(t: CT.CircuitType, callee: CT.CircuitType) -> CT.CircuitType:
    """``t`` without the ANDs and reads no output reaches, wires renumbered and tables listed by first use."""
    widths = [1 if it[0] == CT.AND else callee.out_bits if it[0] == CT.CALL else t.tables[it[1]][2] for it in t.items]
    starts, at = [], t.in_bits
    for w in widths:
        starts.append(at)
        at += w
    used = {w for f in t.outputs for w in f[0]}
    keep = [False] * len(t.items)
    for i in range(len(t.items) - 1, -1, -1):
        it = t.items[i]
        if it[0] == CT.CALL or any(w in used for w in range(starts[i], starts[i] + widths[i])):
            keep[i] = True
            used.update(w for f in CT._item_forms(it) for w in f[0])
    remap = {w: w for w in range(t.in_bits)}
    at, items, tables = t.in_bits, [], []
    for i, it in enumerate(t.items):
        if not keep[i]:
            continue
        for d in range(widths[i]):
            remap[starts[i] + d] = at + d
        at += widths[i]

    def form(f):
        return (tuple(sorted(remap[w] for w in f[0])), f[1])

    for i, it in enumerate(t.items):
        if not keep[i]:
            continue
        if it[0] == CT.AND:
            items.append((CT.AND, form(it[1]), form(it[2])))
        elif it[0] == CT.CALL:
            items.append((CT.CALL, 0, tuple(form(f) for f in it[2])))
        else:
            ent = t.tables[it[1]]
            if ent not in tables:
                tables.append(ent)
            items.append((CT.READ, tables.index(ent), tuple(form(f) for f in it[2])))
    return CT.CircuitType(t.label, t.ins, t.outs, t.calls, tuple(tables), tuple(items), tuple(form(f) for f in t.outputs))


#: a placed read's low decoder bits by index width (`table/v2`, the constant design's read fixes)
GEN_LO = {23: 14, 24: 14}
#: the flat statement's META keys that describe its block wiring; a typed statement derives that wiring instead
WIRING_KEYS = ("unit_log", "units_per_vu", "leaves_in", "leaves_out", "wires", "leaf_cuts", "out_net", "out_net_ports",
               "lookups", "per_vu", "ranges", "k_log", "vus_per_block")


def compose(low: IL.Lowering, sub, **kw) -> Typed:
    """The typed statement (``kw`` as :func:`circuit.compose` takes them): a flat class's unit as one type, or a template
    with a tail as its instance type (:func:`instance_type`)."""
    U = low.units
    if U.cut or U.tail_rets:
        return _compose_tail(low, sub, **kw)
    return _compose_flat(low, sub, **kw)


def _compose_tail(low: IL.Lowering, sub, vus_per_block: int | None = None, **kw) -> Typed:
    """A template with a tail. Its instance type is the unit: the tail stages' gates are its own rows, its units calls of
    the unit type placed in their block range, its MUFU reads placed ``table/v2`` layouts in theirs. The block holds, per
    instance, the row hashing, the instance type's own region, and its placed parts; the wiring among them is derived."""
    import numpy as np

    from . import gf2
    from . import tail as TL

    text = C.compose(low, sub, vus_per_block=vus_per_block, **kw)
    _, meta, secs = sections(text)
    inst = instance_type(low, TL.lower_tail(low, kw.get("forged", False)))
    R, Ut = inst.root, inst.unit
    tables = {sha: np.asarray(gf2.table(name), dtype=np.uint32) for name, sha, _, _ in inst.tables}
    info = {sha: LY.table_info(ws) for sha, ws in tables.items()}
    ul = meta["unit_log"]
    layouts = [LY.Layout(Ut.digest(), ul, _split(low.layout.in_groups), _split(low.layout.out_groups), ("derive",), (), ())]
    gen = {}
    for _name, sha, ib, vb in inst.tables:
        lo = GEN_LO[ib]
        rows = LY.read_rows_placed(ib, lo, LY.k_of(sha, vb, info[sha]))
        layouts.append(LY.Layout(None, C._slot_log(rows), (), (), ("gen", sha, ib, vb, lo), (), ()))
        gen[sha] = len(layouts) - 1
    calls, reads, slot = [], [], {}
    for i, it in enumerate(R.items):
        if it[0] == CT.AND:
            continue
        j = 0 if it[0] == CT.CALL else gen[R.tables[it[1]][0]]
        (calls if it[0] == CT.CALL else reads).append((i, j, slot.get(j, 0)))
        slot[j] = slot.get(j, 0) + 1
    layouts.append(LY.Layout(R.digest(), None, tuple(range(1, len(R.ins))), tuple(range(1, len(R.outs))), ("derive",),
                             tuple(calls), tuple(reads)))
    unit = len(layouts) - 1
    types = {R.digest(): R, Ut.digest(): Ut}
    dv = D.Derivation(types, layouts, unit, tables)
    if D.netlist(dv.physical(0), low.name) != dict(secs)["unit"]:
        raise AssertionError(f"{low.name}: derive does not reproduce the unit's rows")
    own = dv.unit.own
    root_log = C._slot_log(own)
    if own == 1 << root_log:
        raise ValueError(f"{low.name}: the instance type's own region fills its slot; a zero leaf needs a forced-zero row")
    ports = [C._port(nm, w) for nm, w, _ in meta["ports"]]
    n_sha, _ = C._row_slots(ports)
    n_hm = C._pow2(len(ports))
    parts = {layouts[j].digest(): (layouts[j].range_log, n) for j, n in slot.items()}

    def fits(G: int):
        items = [("unit", C.SHA_LOG, G * n_sha, {"net": C.SHA_NET, "packed": True}), ("unit", C.HM_LOG, G * n_hm, {"net": C.HM_NET}),
                 ("unit", root_log, G, {"net": "root"})]
        items += [("unit", sl, G * n, {"net": d, "packed": True}) for d, (sl, n) in parts.items()]
        items.append(("mask", C.MASK_LOG, 1, {"net": "mask"}))
        return C._ranges(items)

    if vus_per_block is None:
        G, k1 = 1, fits(1)[1]
        while 2 * G * max(n for _, n in parts.values()) <= 4096 and fits(2 * G)[1] <= max(k1, 22):
            G *= 2
    else:
        G = vus_per_block
    ranges, k_log = fits(G)
    if k_log > C.K_MAX:
        raise ValueError(f"{low.name}: an instance needs a block of 2^{k_log} bits (at most 2^{C.K_MAX})")
    for k in WIRING_KEYS:
        meta.pop(k, None)
    meta.update(format=FORMAT, statement=STATEMENT, types={d: x.content() for d, x in types.items()},
                layouts=[x.to_json() for x in layouts], unit_layout=unit, unit_type=R.digest(), policy=POLICY, unit_in=[],
                ranges=ranges, k_log=k_log, vus_per_block=G,
                per_vu={C.SHA_NET: n_sha, C.HM_NET: n_hm, "root": 1, **{d: n for d, (_, n) in parts.items()}})
    head = f"{FORMAT} {low.name}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"
    body = "".join(f"CIRCUIT {nm} {s.count(chr(10))}\n{s}" for nm, s in secs if nm in (C.SHA_NET, C.HM_NET))
    rows = {"root": D.netlist(dv.unit, low.name), "delta": D.delta_text(dv),
            **{layouts[j].digest(): D.netlist(dv.physical(j), layouts[j].digest()[:16]) for j in slot}}
    return Typed(head + body, rows["root"], types, layouts, rows, tables, dv)


def _compose_flat(low: IL.Lowering, sub, **kw) -> Typed:
    """The typed statement for a flat class (``kw`` as :func:`circuit.compose` takes them)."""
    text = C.compose(low, sub, **kw)
    _, meta, secs = sections(text)
    t = unit_type(low)
    types, layouts = {t.digest(): t}, [unit_layout(low, t)]
    dv = D.Derivation(types, layouts, 0, {})
    rows = D.netlist(dv.unit, low.name)
    old = dict(secs)["unit"]
    if rows != old:
        raise AssertionError(f"{low.name}: derive does not reproduce the unit's rows")
    meta.update(format=FORMAT, statement=STATEMENT, types={d: x.content() for d, x in types.items()},
                layouts=[x.to_json() for x in layouts], unit_layout=0, unit_type=t.digest(), policy=POLICY, unit_in=[])
    head = f"{FORMAT} {low.name}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"
    body = "".join(f"CIRCUIT {nm} {s.count(chr(10))}\n{s}" for nm, s in secs if nm != "unit")
    return Typed(head + body, rows, types, layouts, {"unit": rows}, {}, dv)


def stage(low: IL.Lowering, input_set, lo: int, hi: int, out_dir, public_only: bool = False, salt_key: bytes | None = None,
          indices: list | None = None, partition: str | None = None, program: str | None = None, share_rows: bool = False):
    """:func:`circuit.stage` for the typed statement. ``<out_dir>/circuit.txt`` is the statement. ``rows/`` is the prover's
    copy of the rows the verifier derives: ``unit.txt`` for a flat class; for a template with a tail, ``root.txt``,
    ``<layout digest>.txt`` per placed layout and ``delta.txt``. ``tables/`` holds the pinned tables its reads need, and
    ``circuit-forged.txt`` (a tail's constants x 4) is the negatives' statement. The instance files are
    :func:`circuit.write`'s, naming the typed statement."""
    from pathlib import Path

    if public_only and salt_key is None:
        raise ValueError("a verifier stages its own public file only from the prover's staging salt key (salts are otherwise fresh)")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    statics = C.set_statics(input_set)
    typed = compose(low, input_set.subcircuit, partition=partition, program=program, statics=statics)
    (out / "circuit.txt").write_text(typed.text)
    rows = out / "rows"
    rows.mkdir(exist_ok=True)
    for name, text in typed.rows.items():
        (rows / f"{name}.txt").write_text(text)
    if low.units.cut:
        IL.write_tables(out / "tables")
        if not (out / "circuit-forged.txt").exists():
            forged = compose(low, input_set.subcircuit, forged=True, partition=partition, program=program, statics=statics)
            (out / "circuit-forged.txt").write_text(forged.text)
    inst, pub = out / f"inst-{hi - lo}.bin", out / f"pub-{hi - lo}.bin"
    C.write(typed.text, input_set, lo, hi, None if public_only else inst, pub, salt_key, indices, share_rows=share_rows)
    return pub if public_only else inst


__all__ = ["FORMAT", "GEN_LO", "Instance", "POLICY", "STATEMENT", "Typed", "compose", "instance_type", "sections", "stage",
           "unit_layout", "unit_type"]
