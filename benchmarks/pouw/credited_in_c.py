"""A credited linear of C whose nodes write its y, through the door.

C is one linear y = x · Wᵀ over M rows of K and N columns whose nodes are its grid's units of `PearlPlusTileCoreWords_v5`
(`pearl_plus_credited.tiles`), each reading its ports as rows of the values C holds: x's and W's rows (``a``, ``b``), the
prover's certificates (``cert``) and the public row (`pearl_plus_credited.PUBLIC`: ``salt``, ``root``, ``unit``,
``rows``, ``col``), registered under public salts; each deriving its noise inside from the salt and the root, and each
writing its tile of y (``o1``), which
a reader of the linear takes by wire: the residual add of each row of y and the residual stream's row (``r``,
`Bf16AddRow_v1`), C's output. y is no value C holds. No unit fits one proof block whole, so C is laid
(`verity.circuits.laying.lay`) under the composer's limit (`fits`, `verity_flock.circuit.unit_block`'s rule on a block's
ports and ANDs): each unit opens into its noise's seed, lines and Gram rows and its wired core
(`pearl_plus_credited.cut`), each one proof block, and each tile is credited over all its blocks. A drawn core's outputs are
registered as the record its reader holds them to: its digest and y under hiding salts, its ``ok`` (one u16) as 1 under
public salts. The noise is seeded by the
epoch's salt and W's registered root: the stager forms its tiles under them, and the verifier's side fills the public row
from the salt, W's registered root and each tile's place alone (`public_values`), no noise value among them, and its
registered roots from that row, which every staged file's roots must equal; the door doesn't yet bind the salt and the
root to the supplied functions (`pouw.pearlplus.salt/v1`, `registered.root/v1`).
``stage`` evaluates C on an honest linear, places each laid block in its tile and each tile's amount
(`scheme.laid_tiles` over the laid C's strata and its digest, recorded before the coin, `tile_macs` on the tile's
statics; a tile below the served floor is a test instance's, `scheme.test_instance`, and credits nothing), draws one
block uniformly of every block of the laid C (``--steer TEMPLATE``: of that template's blocks only, recorded as steered),
printing the population by Definition and each template's fan-out (the readers of a block's outputs), cuts it
(`restrict`), lowers it by the door (`one_circuit.lower` under C's `one_circuit.cuts`, with the door's check of the block
in C), and stages its proof. Each of its output fields is committed as its template's field tree (``KEY/oJ``, one row a unit of the template) under
hiding salts (a wired core's ``ok`` under public salts, the record its reader holds it to) and opened at its leaf; each
registered parameter row it reads is read through its own argument (`registered.reads` of its template). It finds each
of its input rows' writers in C: a row C wires from more than one writer unit is no one writer's output field, which
Lean's registered reads refuse (`Flock.Registered.treeReads`, ``FIELD_WHY``) unless the lowering cuts it into one row a
writer (``p#j``, `Flock.OneCircuit.io`), and the stage stops there when it does not (``WIRED-ROWS``, exit 4). A row one
writer's field feeds, whole and in order (`wired_rows`, which refuses any other row by name, ``FIELD``), is read by path
in that field's one tree (``KEY/oJ``, `Flock.Registered.fieldTree`), which the stage commits under hiding salts from the
writer template's units' outputs, each root node of the laid C evaluated on C's values (`node_outputs`, `field_tree`),
and opens at the writer's leaf (`tree_leaf`) in the drawn block's statement; then it stages the block for
``flock-circuit``.
``door`` hands each proof to `proofs.verify` with the verifier's own program, partition, draw and registered file, and
nothing of Pearl+'s. `credited_in_c.sh` builds, proves and verifies; run it on a node."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCHEME = "pearl-c-h100-v1-h3"
#: the values C holds that the prover commits; every other value is the public row the verifier fills itself
PRIVATE = ("x", "w", "cert", "r")


def _peak_gib(who: int = resource.RUSAGE_SELF) -> float:
    return round(resource.getrusage(who).ru_maxrss / 2**20, 3)


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def fits(ins: tuple[int, ...], outs: tuple[int, ...], ands: int) -> bool:
    """The composer's limit on a block (`verity.circuits.laying`'s ``fits``): `verity_flock.circuit.unit_block`'s rule on
    its ports (one bit row each, whole u16 words) and ANDs, at one unit a block, within 2^`K_MAX`."""
    from verity_flock import circuit as FC
    ports = [FC.Port(f"p{i}", -(-b // 16), 16 * -(-b // 16)) for i, b in enumerate((*ins, *outs))]
    useful = FC._flat_useful(ands, sum(p.n_bits for p in ports[:len(ins)]), sum(p.n_bits for p in ports[len(ins):]))
    return FC._block_ranges(FC._row_slots(ports)[0], FC._pow2(len(ports)), FC._slot_log(useful + 1), 1)[1] <= FC.K_MAX


def writers(program, query, u: int) -> dict[str, list[int]]:
    """Unit ``u``'s parameters that C wires from other units, each with its writer units in order."""
    from verity.circuits import partition_object as PO
    tp = PO.TemplatePartition(program, query)
    by = {g: v for v in range(tp.population) if v != u for g in tp.outputs(v)}
    gates, at, out = tp.inputs(u), 0, {}
    for name, t in program.root.body.nodes[tp.locate(u)[0]].fn.signature()[0]:
        ws = list(dict.fromkeys(by[g] for g in gates[at:at + t.leaves] if g in by))
        at += t.leaves
        if ws:
            out[name] = ws
    return out


def fan_out(program, query, of: list[str]) -> dict[str, dict[str, int]]:
    """Each template's fan-out: over its blocks (``of``, each block's template), the fewest and most other blocks that
    read a gate one of them writes, and the blocks that write to none."""
    from verity.circuits import partition_object as PO
    tp = PO.TemplatePartition(program, query)
    by = {g: v for v in range(tp.population) for g in tp.outputs(v)}
    readers: list[set[int]] = [set() for _ in range(tp.population)]
    for u in range(tp.population):
        for w in {by[g] for g in tp.inputs(u) if g in by} - {u}:
            readers[w].add(u)
    out: dict[str, dict[str, int]] = {}
    for u, tid in enumerate(of):
        n, e = len(readers[u]), out.setdefault(tid, {"min": len(readers[u]), "max": 0, "none": 0})
        e["min"], e["max"], e["none"] = min(e["min"], n), max(e["max"], n), e["none"] + (n == 0)
    return out


def value_bits(t, value) -> list[int]:
    """A value of type ``t``'s bits, low bit of each element first."""
    from verity.circuits import boolean as BL
    words, widths = _flat(value), BL.elements(t)
    assert len(words) == len(widths), (len(words), len(widths))
    return [(w >> j) & 1 for w, wd in zip(words, widths) for j in range(wd)]


def words16(bits: list[int]) -> list[int]:
    bits = bits + [0] * (-len(bits) % 16)
    return [sum(bits[16 * w + j] << j for j in range(16)) for w in range(len(bits) // 16)]


@dataclasses.dataclass(frozen=True)
class Linear:
    """C: ``program`` over its values, ``tiles`` its units in order, ``types`` each value's row type, ``reads`` each
    unit's ports as (value, row), and ``m`` and ``n`` the linear's rows and columns."""

    program: object
    tiles: list
    types: dict
    reads: list
    m: int
    n: int


def linear(k: int, m: int, n: int, tile: tuple[int, int], reader: bool = True) -> Linear:
    """C over ``m`` rows of depth ``k`` and ``n`` columns at ``tile``: its grid's units, and with ``reader`` the residual
    add of each row of y (`Bf16AddRow_v1`), which takes the row by wire from the units that wrote it; without, C's
    output is its units' (digest, y, ok), y the record a later reader is held to."""
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.circuits.ops.prims import BF16
    from verity.circuits.ops.rows import Bf16AddRow
    from verity.circuits.program import Program
    from verity.circuits.refs import Coll, concat, tuple_of
    from verity.circuits.types import Array
    from verity.protocols.run import schemes
    PC = schemes.catalog("pearl-plus", "credited")

    tm, tn = tile
    ts = PC.tiles(SCHEME, k, m, n, tile)
    if len({t.real for t in ts}) != 1:
        raise ValueError(f"every row tile of M = {m} at TM = {tm} has the same real rows, so C's tiles are one template: "
                         "M below TM or a multiple of it")
    port, ret = ts[0].core.signature()
    port = dict(port)
    rt, ct = len({t.rows for t in ts}), n // tn
    row = {"x": port["a"], "w": port["b"], "cert": port["cert"], "r": Array(n, BF16), "rows": port["rows"],
           "col": port["col"]}
    count = {"x": rt, "w": ct, "cert": len(ts), "r": m, "rows": rt, "col": ct}
    names = ("x", "w", "cert") + (("r",) if reader else ()) + PC.PUBLIC
    params = tuple((v, Array(count[v], row[v])) if v in count else (v, port[v]) for v in names)
    types = {v: row.get(v, t) for v, t in params}
    reads = []
    for i, t in enumerate(ts):
        r = t.rows[0] // tm
        reads.append({"a": ("x", r), "b": ("w", t.col), "cert": ("cert", i), "salt": ("salt", 0), "root": ("root", 0),
                      "unit": ("unit", 0), "rows": ("rows", r), "col": ("col", t.col)})
    add = bind(Bf16AddRow, N=n)

    def body(B, S, *xs):
        held = dict(zip((v for v, _ in params), xs))
        outs = {}
        for t, rd in zip(ts, reads):
            args = []
            for p, pt in t.core.signature()[0]:
                v, j = rd[p]
                got = held[v][j] if v in count else held[v]
                assert got.type == pt, (p, got.type, pt)
                args.append(got)
            outs[t.rows[0], t.col] = B.call(t.core, *args)
        tiles_out = Coll(Array(len(ts), ret), concat([outs[t.rows[0], t.col].refs for t in ts]))
        if not reader:
            return tiles_out
        z = []
        for i in range(m):
            r0 = i - i % tm
            y = Coll(Array(n, BF16), concat([outs[r0, c][1][i - r0].refs for c in range(ct)]))
            z.append(B.call(add, y, held["r"][i]))
        return tuple_of(tiles_out, Coll(Array(m, Array(n, BF16)), concat([o.refs for o in z])))

    from verity.circuits.types import Tuple
    out = Tuple(Array(len(ts), ret), Array(m, Array(n, BF16))) if reader else Array(len(ts), ret)
    root = CompositeDefinition("PearlPlusCreditedLinear", 2, (), lambda S: (params, out), body, register=False)
    return Linear(Program(root, registered=[v for v, _ in params]), ts, types, reads, m, n)


def public_values(ln: Linear, k: int, unit: int, salt: bytes, root: bytes) -> dict:
    """The public row of call ``unit`` as C's values (`pearl_plus_credited.public` of each tile): ``salt``, ``root`` and
    ``unit`` one row each, ``rows`` a row tile's, ``col`` a column tile's."""
    from verity.protocols.run import schemes
    PC = schemes.catalog("pearl-plus", "credited")

    pub: dict = {"rows": {}, "col": {}}
    for t in ln.tiles:
        p = PC.public(SCHEME, k, t, unit, salt, root)
        pub["rows"][t.rows[0] // len(t.rows)], pub["col"][t.col] = p["rows"], p["col"]
        shared = {v: p[v] for v in ("salt", "root", "unit")}
    return {**shared, **{v: [pub[v][r] for r in range(len(pub[v]))] for v in ("rows", "col")}}


def honest(k: int, m: int, n: int, ln: Linear, unit: int, seed: str, salt: bytes, root_of) -> tuple[dict, list]:
    """An honest linear: x and W BF16 (widened to FP32), each tile's certificates under the epoch's ``salt`` and W's
    registered root (``root_of`` of W's values), the residual stream's rows and the public row, every value as C holds
    it, a row a port's read (x a row tile's rows, W a column tile's); and beside them the linear's served y
    (`PearlC.checked`'s U unscaled, BF16, `pearl_c_u.y`), M rows of N words, which C's tiles must write."""
    from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
    from verity.protocols.accounting.compute.pouw.prover.circuit import inert_bound
    from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
    from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_u as CU

    rng = random.Random(seed)

    def bf16_row() -> list[int]:
        return [int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) & 0xFFFF0000 for _ in range(k)]

    tm, tn = len(ln.tiles[0].rows), len(ln.tiles[0].cols)
    x = [bf16_row() if i < m else [0] * k for i in range(-(-m // tm) * tm)]
    w = [bf16_row() for _ in range(n)]
    r = [[int(np.float32(rng.uniform(-4, 4)).view(np.uint32)) >> 16 for _ in range(n)] for _ in range(m)]
    w_values = [[[v >> 16 for v in w[j]] for j in range(c * tn, (c + 1) * tn)] for c in range(n // tn)]
    root = root_of(w_values)
    s = pouw_schemes.get(SCHEME, k_min=k, tile=(tm, tn))
    dev = s.device.name
    _d, ng = inert_bound._promoted(dev, k)
    y = [[0] * n for _ in range(m)]
    certs = []
    for t in ln.tiles:
        ctx = C.Context(salt, C.Matmul(unit, C.Shape(len(x), k, n), 0), bytes(64), root)
        a, b = [x[i] for i in t.rows], [w[j] for j in t.cols]
        ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r_) for r_ in a], [C.f32_row(r_) for r_ in b], C.Tile(t.rows, t.cols))
        checked = s.checked(ctx, ops)
        certs.append([[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], dev))[2 * z:2 * z + 2]
                        for z in range(ng - 1)] for j in range(tn)] for i in range(tm)])
        for i in range(t.real):
            for j, c in enumerate(t.cols):
                y[t.rows[i]][c] = CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j])
    row_tiles = sorted({t.rows: t.real for t in ln.tiles}.items())
    held = {"x": [[x[i] for i in rs] for rs, _ in row_tiles], "w": w_values, "cert": certs, "r": r}
    names = [v for v, _ in ln.program.fn.signature()[0]]
    return {v: held[v] for v in held if v in names} | public_values(ln, k, unit, salt, root), y


def reference(ln: Linear, index: int, vals: dict, unit: int) -> list[int]:
    """`pearl_plus_tile.reference`'s (digest, y, ok) of unit ``index`` on ``vals``: the digest's 64 bytes, its real rows'
    TN y words, then ok."""
    from verity.protocols.run import schemes
    PT = schemes.catalog("pearl-plus", "tile")
    t = ln.tiles[index]
    r, s = t.rows[0] // len(t.rows), t.core.bindings
    b = [[v << 16 for v in row] for row in vals["w"][t.col]]
    salt, root = (b"".join(w.to_bytes(2, "little") for w in vals[v]) for v in ("salt", "root"))
    d, y, ok = PT.reference(s["DEV"], s["K"], s["N"], vals["x"][r], b, vals["cert"][index], salt, root,
                            unit, list(t.rows), list(t.cols), s["RHO"], s["RHOR"], t.real)
    return [*(d or bytes(64)), *(w for row in (y or [[0] * len(t.cols)] * t.real) for w in row), ok]


MANY = ("x", "w", "cert", "r", "rows", "col")


def rows_of(ln: Linear, values: dict) -> dict[str, list[list[int]]]:
    """Each value's registered rows (a many-row value's rows, a one-row value its one), each its leaves packed into the
    value's registered width (`registered.width`, `registered.bit_words`)."""
    from verity.protocols.proof.spot.verifier.one_stage import registered as REG
    out = {}
    for v, x in values.items():
        w, b = REG.width(ln.program, v), ln.types[v].leaf_widths()[0]
        out[v] = [REG.bit_words(_flat(r), b, w) for r in (x if v in MANY else [x])]
    return out


def register(program, rows: dict, supplied: tuple[str, ...]) -> dict:
    """Each value's registered rows (`Registered.commit`) in the circuit's ``sha512/row/v2`` port rows; the ``supplied``
    ones under their public salts (`supplied_salts`), so their roots are a function of content the verifier computes."""
    from verity.circuits.codec import program_digest
    from verity.commitments import registered_value as RV
    from verity.protocols.proof.spot.verifier.one_stage import registered as REG
    prog = program_digest(program)
    out = {}
    for v, rs in rows.items():
        out[v] = RV.Registered.commit(prog, v, rs, RV.supplied_salts(prog, v, len(rs)) if v in supplied else None,
                                      word_bits=REG.width(program, v), schema=RV.SCHEMA_HM96_SHA512_ROW_V2)
    return out


def output(program, value: str, rows: list[list[int]], supplied: bool):
    """An output field registered as the record a later reader holds it to (``KEY/oJ``, `Flock.Registered.fieldTree`), one
    ``sha512/row/v2`` row a unit; ``supplied``, under public salts: a field whose value the statement fixes."""
    from verity.circuits.codec import program_digest
    from verity.commitments import registered_value as RV
    prog = program_digest(program)
    return RV.Registered.commit(prog, value, rows, RV.supplied_salts(prog, value, len(rows)) if supplied else None,
                                word_bits=16, schema=RV.SCHEMA_HM96_SHA512_ROW_V2)


def order_runs(nodes: list[tuple[str, int, list[tuple]]]) -> list[tuple[str, int, int]]:
    """A window's ``order`` from C's root units, ``(template, first unit, each member's writer fields)`` a root node in C's
    order: a range a run of consecutive units of one template whose members read the same writers' fields, since the
    window's Program lays each range as one batch (`population.wired_population_program`) and an edge reads one writer
    field at every member (`edges._part_wires`). A batch whose members read different fields of one writer (eight line
    units each one of the seed's index fields) is so as many ranges as it reads fields."""
    order: list[tuple[str, int, int]] = []
    last = None
    for tid, lo, reads in nodes:
        for mm, r in enumerate(reads):
            if order and order[-1][0] == tid and order[-1][1] + order[-1][2] == lo + mm and last == r:
                order[-1] = (tid, order[-1][1], order[-1][2] + 1)
            else:
                order.append((tid, lo + mm, 1))
            last = r
    return order


def provable(fns: dict, cuts: dict) -> None:
    """Refuse by name (``WINDOW-BLOCK``, `interface.check_proof_block`'s reason) a window whose template's proof block, its
    rows under its cuts, Flock's composer would refuse: a reader whose every row is within the last SHA-512 block, as
    `Bf16AddRow_v1{N=1}`'s, would otherwise be a member the audit cannot lower."""
    from verity.commitments.periods import interface as IF
    for t, fn in fns.items():
        try:
            IF.check_proof_block(fn, cuts.get(t))
        except ValueError as e:
            raise SystemExit(f"WINDOW-BLOCK: {e}") from None


def _field_starts(t) -> list[int]:
    """Where each output field (`verity.circuits.edges`' fields: a Tuple's or Record's members, else the whole) of a
    return type ``t`` starts, in leaves, and its end."""
    from verity.circuits.types import RecordType, Tuple
    fs = list(t.fields) if isinstance(t, Tuple) else [f for _, f in t.fields] if isinstance(t, RecordType) else [t]
    out = [0]
    for f in fs:
        out.append(out[-1] + f.leaves)
    return out


def node_outputs(program, values: dict, skip: set[str], at: dict | None = None) -> dict[int, np.ndarray]:
    """Each root node of the laid ``program`` but its inputs and the nodes of ``skip``'s templates, evaluated on C's
    ``values`` one node at a time (`evaluate_batch` of the node's Definition on the leaves each member reads, by C's
    gates): its members' outputs, one row a member; ``at``, when given, filled with every gate's value."""
    from verity.circuits.cut import member_refs
    from verity.circuits.evaluation import evaluate_batch
    from verity.circuits.codec import descriptor_id
    from verity.circuits.layout import resolve
    from verity.circuits.program import is_input
    from verity.protocols.proof.spot.verifier.one_stage import registered as REG
    scope, at = program.circuit.root_scope, {} if at is None else at
    for v, x in values.items():
        gates = REG.parameter_gates(program, v)
        words = _flat(x)
        assert len(gates) == len(words), (v, len(gates), len(words))
        at.update(zip(gates, words))
    out = {}
    for k, nd in enumerate(program.root.body.nodes):
        if is_input(nd.fn) or descriptor_id(nd.fn) in skip:
            continue
        n = nd.n if nd.form == "batch" else 1
        axes = nd.axes if nd.form == "batch" else (None,) * len(nd.args)
        args = []
        for arg, ax in zip(nd.args, axes):
            rows = []
            for m in range(n):
                rs = member_refs(arg, ax, m) if nd.form == "batch" else arg.refs
                rows.append([at[resolve(scope, rs[j])] for j in range(rs.count)])
            args.append(np.asarray(rows, dtype=np.int64))
        got = np.asarray(evaluate_batch(nd.fn, *args)).reshape(n, -1)
        per = got.shape[1]
        for m in range(n):
            for j in range(per):
                at[resolve(scope, ("n", k, m * per + j))] = int(got[m, j])
        out[k] = got
    return out


def wired_rows(program, node: int, member: int, port_rows: list[str], cut: dict) -> dict[str, tuple[int, int, int]]:
    """Each input row of in(S) of the reader unit (root ``node``, its ``member``) that C wires from another unit: its
    writer's root node, member and output field, the row being that field whole and in order, read from the member's own
    operands (`cut.member_refs`). A row that is part of a field, or of more than one, is refused by name (``FIELD``):
    Lean's registered reads read one field's tree."""
    from verity.circuits.cut import member_refs
    from verity.circuits.program import is_input
    nd = program.root.body.nodes[node]
    sig = dict(nd.fn.signature()[0])
    args = {p: member_refs(a, ax, member)
            for p, a, ax in zip(sig, nd.args, nd.axes if nd.form == "batch" else (None,) * len(nd.args))}
    nodes, out = program.root.body.nodes, {}
    for name in port_rows:
        p, _, j = name.partition("#")
        if p not in args:
            continue
        rows, ln = cut.get(p, 1), sig[p].leaves
        lo, hi = (int(j) * ln // rows, (int(j) + 1) * ln // rows) if j else (0, ln)
        refs = args[p]
        got = [tuple(refs[i]) for i in range(lo, hi)]
        if not any(r[0] == "n" and r[1] != node and not is_input(nodes[r[1]].fn) for r in got):
            continue
        kw = got[0][1]
        wn = nodes[kw]
        starts = _field_starts(wn.fn.signature()[1])
        per = starts[-1]
        m, at = divmod(got[0][2], per)
        if at not in starts[:-1]:
            raise SystemExit(f"FIELD: row {name} of the reader at node {node} starts at leaf {at} of its writer's output, "
                             "no field's first")
        f = starts.index(at)
        want = [("n", kw, m * per + x) for x in range(starts[f], starts[f + 1])]
        if got != want:
            raise SystemExit(f"FIELD: row {name} of the reader at node {node} is no one output field of node {kw}, whole "
                             "and in order")
        out[name] = (kw, m, f)
    return out


def field_tree(program, outs: dict[int, np.ndarray], kw: int, f: int, words: int) -> tuple[str, list[list[int]]]:
    """Output field ``f`` of the template of root node ``kw`` as the one tree a reader opens (`Flock.Registered.fieldTree`,
    ``KEY/oJ``): one row a unit of the template, in the root body's order (`HmLinks.treeLeaf`), the field's bits in
    16-bit words, ``words`` of them."""
    from verity.circuits.codec import descriptor_id
    from verity.circuits.types import RecordType, Tuple
    nodes = program.root.body.nodes
    t = descriptor_id(nodes[kw].fn)
    ret = nodes[kw].fn.signature()[1]
    ft = (list(ret.fields) if isinstance(ret, Tuple) else [x for _, x in ret.fields] if isinstance(ret, RecordType)
          else [ret])[f]
    starts, widths = _field_starts(ret), ft.leaf_widths()
    rows = []
    for k, nd in enumerate(nodes):
        if k in outs and descriptor_id(nd.fn) == t:
            for m in range(outs[k].shape[0]):
                leaves = outs[k][m, starts[f]:starts[f + 1]].tolist()
                bits = [(w >> j) & 1 for w, wd in zip(leaves, widths) for j in range(wd)]
                row = words16(bits)
                assert len(row) == words, (t, f, len(row), words)
                rows.append(row)
    return f"{t}/o{f}", rows


def tree_leaf(program, kw: int, m: int) -> int:
    """Unit (node ``kw``, member ``m``)'s row in its template's field trees (`HmLinks.treeLeaf`)."""
    from verity.circuits.codec import descriptor_id
    nodes = program.root.body.nodes
    t = descriptor_id(nodes[kw].fn)
    return sum(nd.n if nd.form == "batch" else 1 for nd in nodes[:kw] if descriptor_id(nd.fn) == t) + m


def reads(program, partition, rows: dict, roots: dict[str, str], outs: dict, template: str, ports=None) -> dict:
    """The verifier's ``--registered`` file of ``template``'s statements over its program and the partition's query,
    position-free (`registered.reads_file`), with each committed field of ``outs`` (an output row's or a wired input
    row's)."""
    from verity.commitments import registered_value as RV
    from verity.protocols.proof.spot.verifier.one_stage import registered as REG
    f = REG.reads_file(program, partition["query"], {v: len(rs[0]) for v, rs in rows.items()}, roots, ports=ports,
                       schemas={v: RV.SCHEMA_HM96_SHA512_ROW_V2 for v in rows}, positions=False, template=template)
    for p, out in outs.items():
        f["ports"][p] = {"value": out.value, "root": out.root.hex(), "leaves": out.leaves, "words": out.words,
                         "word_bits": 16, "domain": out.domain.domain_id.hex(), RV.SCHEMA_KEY: out.schema}
    return f


def stage(a) -> int:
    k, tm, tn, n, m = map(int, a.size.split(","))
    from verity.circuits import laying as LY
    from verity.circuits import partition_object as PO
    from verity.circuits import partition_query as PQ
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_chunks, descriptor_id
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate_batch
    from verity.circuits.program import input_id
    from verity.circuits.ops.rows import Bf16AddRow
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.proof.spot.verifier.one_stage import registered as REG
    from verity.protocols.accounting.compute.pouw import scheme as PS
    from verity.protocols.run import schemes
    from verity.commitments import registered_value as RV
    from verity_flock import circuit as FC
    PC = schemes.catalog("pearl-plus", "credited")

    rec: dict = {"size": {"K": k, "TM": tm, "TN": tn, "N": n, "M": m}, "scheme": SCHEME, "phases": {}}
    t = time.perf_counter()

    def phase(name: str, **extra) -> None:
        nonlocal t
        now = time.perf_counter()
        rec["phases"][name] = {"wall_s": round(now - t, 2), "peak_gib": _peak_gib(), **extra}
        print(f"PHASE {name} {json.dumps(rec['phases'][name])}", flush=True)
        t = now

    ln = linear(k, m, n, (tm, tn), a.reader)
    units = len(ln.tiles)
    rec["reader"] = a.reader
    phase("program", units=units)
    program = LY.lay(ln.program, fits)
    query = LY.query(program)
    m_ = LY.Measure(program, query)
    population = len(m_.units)
    laid = [descriptor_id(nd.fn) for nd in program.root.body.nodes if input_id(nd.fn.id) is None]
    cut = [descriptor_id(u) for t in ln.tiles for u, _n in PC.cut(t.core)]
    cut += [descriptor_id(bind(Bf16AddRow, N=n))] * m if a.reader else []
    if laid != cut:
        at = next((i for i, (x, y) in enumerate(zip(laid, cut)) if x != y), min(len(laid), len(cut)))
        print("LAIDCUT " + json.dumps({"laid": len(laid), "cut": len(cut), "first_difference": at,
                                       "laid_from": laid[at:at + 4], "cut_from": cut[at:at + 4],
                                       "laid_by": dict(Counter(laid)), "cut_by": dict(Counter(cut))}), flush=True)
    core = bind(PC.PearlPlusTileCoreWired, **ln.tiles[0].core.bindings)
    blocks = {}
    for u in range(population):
        ins, outs, ands = m_.block(u, u + 1)
        tid = descriptor_id(program.root.body.nodes[m_.units[u][0]].fn)
        b = blocks.setdefault(tid, {"blocks": 0, "ands": ands, "ins_bits": sum(ins), "outs_bits": sum(outs)})
        b["blocks"] += 1
        assert fits(ins, outs, ands) and b["ands"] == ands, (u, tid)
    rec["laid"] = {"by": "verity.circuits.laying.lay under the composer's limit (fits)", "population": population,
                   "strata": blocks}
    print("LAID " + json.dumps(rec["laid"]), flush=True)
    part = PO.build(program, query, [[(u, u + 1) for u in range(population)]])
    c = b"".join(descriptor_chunks(program))
    rec["laid"]["digest"] = hashlib.sha512(c).hexdigest()
    phase("lay", descriptor_bytes=len(c), population=population)

    strata = PQ.strata(program, part["query"])
    tiles = PS.laid_tiles(ln.program, fits, strata, rec["laid"]["digest"])
    rec["amounts"] = {"strata": [[tid, rs] for tid, rs in strata],
                      "tiles": [{"index": tl.index, "gates": tl.gates, "amount": tl.amount, "blocks": len(tl.blocks),
                                 "kinds": dict(tl.kinds), "test_instance": tl.test} for tl in tiles]}
    print("AMOUNTS " + json.dumps(rec["amounts"]), flush=True)
    assert [tl.amount for tl in tiles] == [PS.tile_macs(tl.core.bindings) for tl in ln.tiles]
    test = next((tl for tl in tiles if tl.test), None)
    rec["credit"] = {"credits": "nothing" if test else "each tile its amount, when every stratum it has a block in "
                                                         "passed (credited_tiles)",
                     "refusal": f"{PS.TEST_INSTANCE}: tile {test.index} {test.test}" if test else None}
    print("CREDIT " + json.dumps(rec["credit"]), flush=True)
    if a.until == "amounts":
        return 0

    coin = os.urandom(64)
    of = [""] * population
    for tid, rs in strata:
        for lo, hi in rs:
            of[lo:hi] = [tid] * (hi - lo)
    pool = [u for u in range(population) if a.steer is None or of[u].startswith(a.steer)]
    if not pool:
        raise SystemExit(f"no-block: no block of the laid C is of {a.steer}")
    block_no = pool[int.from_bytes(hashlib.sha512(coin).digest(), "little") % len(pool)]
    held_by = [tl.index for tl in tiles if block_no in tl.blocks]
    index = held_by[0] if held_by else None
    drawn_tid = of[block_no]
    is_core = drawn_tid == descriptor_id(core)
    key = PO.template_id(drawn_tid)
    population_by = {tid: of.count(tid) for tid in dict.fromkeys(of)}
    rec["draw"] = {"coin": coin.hex(), "block": block_no, "template": drawn_tid, "tile": index, "steered": a.steer,
                   "of": (f"one block uniformly of the {len(pool)} blocks of {a.steer} (steered)" if a.steer else
                          f"one block uniformly of all {population} blocks of the laid C"),
                   "population": population_by, "fan_out": fan_out(program, query, of)}
    print("DRAW " + json.dumps(rec["draw"]), flush=True)
    blk = OC.block(c, part, block_no)
    b = R.restrict(c, part, blk)
    phase("restrict", block_units=blk, block_bytes=len(b.descriptor))

    params = [v for v, _ in program.fn.signature()[0]]
    unit = 7
    salt = random.Random("credited-in-c/salt").randbytes(32)
    supplied = tuple(v for v in params if v not in PRIVATE)
    w_reg: dict = {}

    def root_of(w: list) -> bytes:
        w_reg.update(register(program, rows_of(ln, {"w": w}), supplied))
        return w_reg["w"].root

    good, served = honest(k, m, n, ln, unit, "credited-in-c", salt, root_of)
    ti = 0 if index is None else index
    drawn = ln.tiles[ti]
    tr, tn = drawn.real, len(drawn.cols)
    nd, ny = 512, tr * tn

    def run(values: dict) -> tuple[list[list[int]], list[list[int]]]:
        """C's outputs: each unit's leaves (its digest's 512 bits, its y's TR · TN words, its ok), and the reader's rows."""
        flat = evaluate_batch(ln.program.fn, *(np.asarray([_flat(values[v])], dtype=np.int64) for v in params))
        flat = flat.reshape(-1).tolist()
        per = nd + ny + 1
        assert len(flat) == per * units + (m * n if a.reader else 0), (len(flat), per, units, m, n)
        z = flat[per * units:]
        return [flat[per * u:per * (u + 1)] for u in range(units)], [z[n * i:n * (i + 1)] for i in range(m)]

    def words(g: list[int]) -> list[int]:
        return [sum(g[8 * i + j] << j for j in range(8)) for i in range(64)] + g[nd:]

    def y_of(g: list[int]) -> list[int]:
        return g[nd:nd + ny]

    got_h, z_h = run(good)
    y_rows = [[0] * n for _ in range(m)]
    for tile, g in zip(ln.tiles, got_h):
        for i in range(tile.real):
            y_rows[tile.rows[i]][tile.col * tn:(tile.col + 1) * tn] = y_of(g)[i * tn:(i + 1) * tn]
    add = evaluate_batch(bind(Bf16AddRow, N=n), np.asarray(y_rows, dtype=np.int64),
                         np.asarray(good["r"] if a.reader else [[0] * n] * m, dtype=np.int64))
    rec["evaluate"] = {"of": "C on its words (`evaluate_batch` of its root before laying, whose gates the laid C keeps): "
                              "each unit's (digest, y, ok) and the reader's rows",
                       "tile": {"index": ti, "rows": drawn.rows, "real": drawn.real, "amount": tiles[ti].amount,
                                "credited": 0 if tiles[ti].test else tiles[ti].amount},
                       "honest_ok": [g[-1] for g in got_h],
                       "y": y_rows, "y_is_the_served_y": y_rows == served,
                       "reader_reads_y_by_wire": z_h == add.reshape(m, n).tolist() if a.reader else None,
                       "tile_reference_agrees": words(got_h[ti]) == reference(ln, ti, good, unit)}
    phase("evaluate")
    print("EVALUATE " + json.dumps(rec["evaluate"]), flush=True)
    assert rec["evaluate"]["tile_reference_agrees"], rec["evaluate"]
    assert rec["evaluate"]["y_is_the_served_y"] and rec["evaluate"]["reader_reads_y_by_wire"] in (True, None), rec["evaluate"]
    assert rec["evaluate"]["honest_ok"] == [1] * units

    def fields_of(g: list[int]) -> dict[str, list[int]]:
        """A unit's output fields' bits, each leaf low bit first: o0 the digest, o1 y's words, o2 ok's word."""
        return {"o0": g[:nd], "o1": [(w >> j) & 1 for w in y_of(g) for j in range(16)],
                "o2": [(g[-1] >> j) & 1 for j in range(16)]}

    claimed = [fields_of(g) for g in got_h]

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        cut = OC.cuts(c, part)
        text = OC.lower(b.descriptor, part, lambda: a.lower, cut=cut, formed=OC.formed_in(c, part, blk)).decode()
    except (OC.NotDerived, OC.Refused) as e:
        rec["lower"] = {"refused": f"{type(e).__name__}: {e}", "query": part["query"]["name"]}
        print("LOWER-REFUSED " + json.dumps(rec["lower"]), flush=True)
        (a.out / "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 3
    meta = FC.meta_of(text)
    rec["lower"] = {"by": "the door's one_circuit.lower (flock-lower-check compose)", "wall_s": round(time.perf_counter() - t0, 2),
                    "peak_gib_children": _peak_gib(resource.RUSAGE_CHILDREN), "bytes": len(text), "k_log": meta.get("k_log"),
                    "sha512": hashlib.sha512(text.encode()).hexdigest(), "ports": [q[:2] for q in meta.get("ports", [])],
                    "cut": cut.get(drawn_tid, {})}
    print("LOWER " + json.dumps(rec["lower"]), flush=True)
    phase("lower")

    widths = dict((q[0], q[1]) for q in meta["ports"])
    node, member = m_.units[blk[0][0]]
    leaf = tree_leaf(program, node, member)
    assert not is_core or leaf == ti, (leaf, ti)
    out_ports = [f"o{f}" for f in range(len(_field_starts(program.root.body.nodes[node].fn.signature()[1])) - 1)]
    if missing := [p for p in out_ports if p not in widths]:
        raise SystemExit(f"OUT-ROWS: output fields {missing} of {drawn_tid} are no one row each of the lowered block's "
                         f"ports ({sorted(widths)}): a field is committed as one tree, one row a unit")
    if is_core:
        rec["ok_row"] = {"port": "o2", "words": widths.get("o2"), "bits": 16 * widths["o2"] if "o2" in widths else None}
        print("OK-ROW " + json.dumps(rec["ok_row"]), flush=True)
    wired = writers(program, part["query"], blk[0][0])
    rows = [q[0] for q in meta["ports"]]
    many = {p: len(ws) for p, ws in wired.items() if len(ws) > 1 and p in rows}
    rec["wired"] = {p: {"writers": len(ws), "templates": sorted({descriptor_id(program.root.body.nodes[m_.units[w][0]].fn)
                                                                 for w in ws}), "row": p in rows}
                    for p, ws in wired.items()}
    print("WIRED " + json.dumps(rec["wired"]), flush=True)
    if many:
        rec["wired_rows"] = {
            "refused": f"the drawn block's input rows {sorted(many)} are each wired from more than one writer unit "
                       f"({many}), and the door's lowering (one_circuit.lower under one_circuit.cuts) left each one row: "
                       "a row that is no one writer's output field, whole, is refused by Lean's registered reads "
                       "(Flock.Registered.treeReads, FIELD_WHY)"}
        print("WIRED-ROWS " + json.dumps(rec["wired_rows"]), flush=True)
        (a.out / "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 4
    fed = wired_rows(program, node, member, rows, rec["lower"]["cut"])
    trees = sorted({(kw, f) for kw, _m, f in fed.values()})
    rec["writer_rows"] = {"of": "each input row of in(S) C wires from a writer is that writer's output field whole, read "
                                "by path in the field's one tree (KEY/oJ, one row a unit of the writer's template), "
                                "committed under hiding salts",
                          "rows": len(fed),
                          "trees": sorted({f"{descriptor_id(program.root.body.nodes[kw].fn)}/o{f}" for kw, f in trees})}
    print("WRITER-ROWS " + json.dumps(rec["writer_rows"]), flush=True)
    skip = {descriptor_id(core), descriptor_id(bind(Bf16AddRow, N=n))} - ({drawn_tid} if not is_core else set())
    verifier_row = public_values(ln, k, unit, salt, w_reg["w"].root)
    verifier_roots = {v: r.root.hex() for v, r in register(program, rows_of(ln, verifier_row), supplied).items()}
    rec["public_row"] = {"of": "the verifier's: `public_values` from the salt, W's registered root and the tiles' places "
                         "alone, no noise value", "ports": list(PC.PUBLIC),
                         "salt": salt.hex(), "w_root": w_reg["w"].root.hex(), "supplied": list(supplied)}
    rec["registered"] = {}
    tag, vals = "honest", good
    d = a.out / tag
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "circuit.txt").write_text(text)
    rows = rows_of(ln, vals)
    fresh = register(program, {v: r for v, r in rows.items() if v != "w"}, supplied)
    regs = {v: w_reg[v] if v == "w" else fresh[v] for v in rows}
    assert all(regs[v].root.hex() == verifier_roots[v] for v in supplied)
    roots = {v: r.root.hex() for v, r in regs.items()}
    made = node_outputs(program, vals, skip)
    if is_core:
        fields = {p: [f[p] for f in claimed] for p in out_ports}
        out_rows = {p: [words16(f) + [0] * (widths[p] - len(words16(f))) for f in fs] for p, fs in fields.items()}
    else:
        out_rows = {p: field_tree(program, made, node, f, widths[p])[1] for f, p in enumerate(out_ports)}
    outs = {p: output(program, f"{key}/{p}", out_rows[p], is_core and p == "o2") for p in out_ports}
    words_of = {(kw, f): widths[p] for p, (kw, _m, f) in fed.items()}
    committed = {}
    for kw, f in trees:
        name, trows = field_tree(program, made, kw, f, words_of[kw, f])
        committed[kw, f] = output(program, name, trows, False)
    held = {p: committed[kw, f] for p, (kw, _m, f) in fed.items()}
    own = reads(program, part, rows, roots, outs | held, drawn_tid)
    (d / "registered.json").write_text(json.dumps(own))
    regd = REG.reads(program, part["query"], {v: len(rs[0]) for v, rs in rows.items()},
                     {v: RV.SCHEMA_HM96_SHA512_ROW_V2 for v in rows}, template=drawn_tid)
    at = {p: (e["value"], e["rows"][blk[0][0]]) for p, e in regd.items() if e["rows"][blk[0][0]] is not None}
    tables, salts, registered = {}, {}, {}
    for p, (v, j) in at.items():
        rs, ss, entry = regs[v].statement([j])
        tables[p] = np.asarray([RV.row_words(r, regs[v].word_bits, RV.SCHEMA_HM96_SHA512_ROW_V2) for r in rs],
                               dtype=np.uint16)
        salts[p], registered[p] = ss, entry
    for p, (kw, m, f) in fed.items():
        rs, ss, entry = held[p].statement([tree_leaf(program, kw, m)])
        tables[p] = np.asarray([RV.row_words(r, 16, RV.SCHEMA_HM96_SHA512_ROW_V2) for r in rs], dtype=np.uint16)
        salts[p], registered[p] = ss, entry
    for p, o in outs.items():
        _rs, ss, entry = o.statement([leaf])
        salts[p], registered[p] = ss, entry
    outs_h = {p: np.asarray([out_rows[p][leaf]], dtype=np.uint16) for p in out_ports}
    iset = SimpleNamespace(name=f"credited-in-c-{tag}",
                           subcircuit=SimpleNamespace(id=program.root.body.nodes[node].fn.id, inputs=[], outputs=[]))
    FC.write(text, iset, 0, 1, d / "inst-1.bin", d / "pub-1.bin", indices=[blk[0][0]], tables=tables,
             refs={p: [0] for p in tables}, share_rows=True, salts=salts, registered=registered, outs=outs_h)
    (d / "unit.tmpl").write_text(FC.template_of(text))
    (d / "partition.json").write_text(json.dumps(part))
    (d / "program.json").write_bytes(c)
    (d / "draw.json").write_text(json.dumps({"units": [block_no]}))
    rec["registered"][tag] = {v: r.root.hex()[:16] for v, r in regs.items()}
    phase("stage")
    rec["peak_gib"] = _peak_gib()
    (a.out / "stage.json").write_text(json.dumps(rec, indent=1, default=str))
    print("STAGE " + json.dumps({x: rec[x] for x in ("size", "draw", "evaluate", "registered")}, default=str), flush=True)
    return 0


def door(a) -> int:
    """`proofs.verify` on each staged proof with the verifier's own program, partition, draw and registered file, its
    lowering the door's (`one_circuit.lower`), and no ``inputs``: the door holds nothing of Pearl+'s."""
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    os.environ[PF.TMPL_SWITCH] = "1"
    PF.executable = lambda: Path(a.lean)
    PF.lower_executable = lambda: Path(a.lower)
    restrict, memo = OC.restrict, {}

    def once(c: bytes, partition, units, decoded=None) -> bytes:
        key = (hashlib.sha512(c).hexdigest(), json.dumps(partition, sort_keys=True), json.dumps([list(u) for u in units]))
        if key not in memo:
            memo[key] = restrict(c, partition, units, decoded)
        return memo[key]

    OC.restrict = once
    h = a.out / "honest"
    program, partition = (h / "program.json").read_bytes(), json.loads((h / "partition.json").read_text())
    draw_ = json.loads((h / "draw.json").read_text())

    def call(tag: str, reads_name: str) -> dict:
        d = a.out / tag
        t0 = time.perf_counter()
        found, refusal = PF.verify(PF.Proof(d / "circuit.txt", d / "pub-1.bin", tuple(map(Path, a.session.get(tag, ())))),
                                   program=program, partition=partition, draw=draw_,
                                   registered=json.loads((d / reads_name).read_text()))
        rec = {"proof": tag, "registered": reads_name, "wall_s": round(time.perf_counter() - t0, 2), "peak_gib": _peak_gib(),
               "refusal": refusal, "findings": [dataclasses.asdict(f) for f in found or []]}
        print("DOOR " + json.dumps(rec, default=str), flush=True)
        return rec

    honest = call("honest", "registered.json")
    staged = json.loads((a.out / "stage.json").read_text())
    drawn, credit = staged["draw"], staged["credit"]
    credits = f"credits nothing ({credit['refusal']})" if credit["refusal"] else f"credits {credit['credits']}"
    how = f"steered to {drawn['steered']}" if drawn["steered"] else "drawn over every block of C"
    if honest["refusal"] is None and honest["findings"] and all(f["accepted"] for f in honest["findings"]):
        if drawn["template"].startswith("PearlPlusTileCoreWired"):
            print(f"ACCEPT noise supplied, not derived; {credits}: the door accepts block {draw_['units']} of C, {how}, "
                  "a wired core, whose eb, fa, fb and g are wired rows held to their writers' committed output trees; "
                  "the blocks that derive them from the salt and W's root (the seed, the lines, the Gram rows) are not "
                  "proved in this run. It reads its unit, rows and col from the public row the verifier registered "
                  "itself; its digest, the tile leaf, is bound to no transcript root, and the salt and the root to no "
                  "supplied function (pouw.pearlplus.salt/v1, registered.root/v1)", flush=True)
        else:
            print(f"ACCEPT {credits}: the door accepts block {draw_['units']} of C, {how}, a {drawn['template']}, its "
                  "output fields committed as its template's field trees and opened at its leaf, its registered rows "
                  "read through its own arguments and its wired rows held to their writers' committed trees; the salt "
                  "and W's root are bound to no supplied function (pouw.pearlplus.salt/v1, registered.root/v1)",
                  flush=True)
    return 0


def window(a) -> int:
    """C's laid units as a window (`interface.Window`) for ``records_audit.py audit``, as `one_linear.py window` writes
    its own: each root unit of the laid C a unit of its template, in C's order (``order``); each port C feeds from other
    units a wire (`interface.Wire`) of each of its rows from one writer's output field, whole, cut into those rows where
    there are several (``cuts``); each port C feeds from its parameters a run input the unit commits (``external``),
    the public row among them. C is without the reader (`linear`'s ``reader``): y is its output, since the residual add
    is a proof block Flock's composer refuses below 56 columns (`provable`). Refused by name where a port reads part of
    an output field or both a parameter and a unit
    (``WINDOW-PORT``)."""
    k, tm, tn, n, m = map(int, a.size.split(","))
    from verity.circuits import laying as LY
    from verity.circuits.codec import descriptor_id
    from verity.circuits.cut import member_refs
    from verity.circuits.layout import resolve
    from verity.circuits.program import is_input
    from verity.commitments import docket_digest as DD
    from verity.commitments.periods import interface as IF
    from verity.protocols.proof.spot.prover import docket as DK
    from verity.protocols.proof.spot.verifier import service as S

    t0, out = time.perf_counter(), Path(a.out)
    ln = linear(k, m, n, (tm, tn), reader=False)
    program = LY.lay(ln.program, fits)
    params = [v for v, _ in program.fn.signature()[0]]
    supplied = tuple(v for v in params if v not in PRIVATE)
    w_reg: dict = {}

    def root_of(w: list) -> bytes:
        w_reg.update(register(program, rows_of(ln, {"w": w}), supplied))
        return w_reg["w"].root

    good, _served = honest(k, m, n, ln, 7, "credited-in-c", random.Random("credited-in-c/salt").randbytes(32), root_of)
    at: dict = {}
    outs = node_outputs(program, good, set(), at)
    scope, nodes = program.circuit.root_scope, program.root.body.nodes
    fns, units, base = {}, {}, {}
    for kk, nd in enumerate(nodes):
        if is_input(nd.fn):
            continue
        tid = descriptor_id(nd.fn)
        fns.setdefault(tid, nd.fn)
        base[kk] = units.get(tid, 0)
        units[tid] = base[kk] + (nd.n if nd.form == "batch" else 1)
    wires, cuts, external, reads = [], {}, set(), {}
    inputs: dict[int, list] = {}
    for kk, nd in enumerate(nodes):
        if kk not in base:
            continue
        tid, count = descriptor_id(nd.fn), outs[kk].shape[0]
        axes = nd.axes if nd.form == "batch" else (None,) * len(nd.args)
        inputs[kk], reads[kk] = [], [[] for _ in range(count)]
        for (p, ty), arg, ax in zip(nd.fn.signature()[0], nd.args, axes):
            rows, fed = [], set()
            for mm in range(count):
                rs = member_refs(arg, ax, mm) if nd.form == "batch" else arg.refs
                got = [tuple(rs[j]) for j in range(rs.count)]
                wired = [r[0] == "n" and not is_input(nodes[r[1]].fn) for r in got]
                if any(wired) != all(wired):
                    raise SystemExit(f"WINDOW-PORT: {tid} port {p} reads both a parameter and a unit")
                fed.add(all(wired))
                if not all(wired):
                    rows.append([at[resolve(scope, r)] for r in got])
                    continue
                pieces = []
                for r in got:
                    wn = nodes[r[1]]
                    starts = _field_starts(wn.fn.signature()[1])
                    mw, off = divmod(r[2], starts[-1])
                    f = next(i for i in range(len(starts) - 1) if starts[i] <= off < starts[i + 1])
                    if pieces and pieces[-1][:3] == (r[1], mw, f) and pieces[-1][3] + 1 == off:
                        pieces[-1] = (r[1], mw, f, off)
                    elif off == starts[f]:
                        pieces.append((r[1], mw, f, off))
                    else:
                        raise SystemExit(f"WINDOW-PORT: {tid} port {p} reads part of an output field of node {r[1]}")
                for kw, mw, f, last in pieces:
                    if last + 1 != _field_starts(nodes[kw].fn.signature()[1])[f + 1]:
                        raise SystemExit(f"WINDOW-PORT: {tid} port {p} reads part of an output field of node {kw}")
                c = len(pieces)
                reads[kk][mm] += [(p, kw, f) for kw, _mw, f, _last in pieces]
                if cuts.setdefault(tid, {}).setdefault(p, c) != c:
                    raise SystemExit(f"WINDOW-PORT: {tid} port {p} reads {c} writers' fields at one unit and "
                                     f"{cuts[tid][p]} at another")
                for j, (kw, mw, f, _last) in enumerate(pieces):
                    wires.append(IF.Wire(tid, p, descriptor_id(nodes[kw].fn), f"o{f}", base[kk] + mm, (1,), base[kw] + mw,
                                         (0,), (j, 1) if c > 1 else (0, 1)))
            if fed == {False}:
                external.add(f"{tid}/{p}")
                inputs[kk].append(np.asarray(rows, dtype=np.int64))
            else:
                inputs[kk].append(None)
    cuts = {t: {p: c for p, c in cs.items() if c > 1} for t, cs in cuts.items()}
    cuts = {t: cs for t, cs in cuts.items() if cs}
    provable(fns, cuts)
    order = order_runs([(descriptor_id(nodes[kk].fn), base[kk], [tuple(r) for r in reads[kk]]) for kk in base])
    circuit = IF.Circuit("credited_in_c", tuple((fns[t], c) for t, c in units.items()), external=tuple(sorted(external)),
                         wires=tuple(IF.coalesce(wires)), cuts=cuts, order=tuple(order))
    wprog, wquery = circuit.program()
    law = S.params_sampling_design(a.rule, wprog, wquery, None)[0]
    win = IF.Window(circuit, "ab" * 64, law, 1)

    def as_rows(leaves: np.ndarray, ty, port: str, lay) -> np.ndarray:
        words = dict(lay.rows + lay.outs)[port]
        width = dict(zip((q for q, _ in lay.rows + lay.outs), lay.widths))[port]
        return IF.rows_of_leaves(leaves, ty, words).view(f"<u{width // 8}")

    for kk, nd in enumerate(nodes):
        if kk not in base:
            continue
        lay, (sig, ret) = IF.layout(nd.fn), nd.fn.signature()
        ins = [None if x is None else as_rows(x, ty, p, lay) for x, (p, ty) in zip(inputs[kk], sig)]
        starts = _field_starts(ret)
        fields = [as_rows(outs[kk][:, starts[f]:starts[f + 1]], ty, f"o{f}", lay) for f, ty in enumerate(IF._fields(ret))]
        win.record(nd.fn, ins, fields)
    if bad := win.mismatches():
        print("MISMATCHED " + json.dumps({t: us[:8] for t, us in bad.items()}), flush=True)
    win.supply()
    out.mkdir(parents=True, exist_ok=True)
    win.write(win.close(), out, {"credited_in_c": "window"}, registrar=DK)
    (out / "pins.json").write_text("[]")
    print("STEP " + json.dumps({"step": "window", "size": {"K": k, "TM": tm, "TN": tn, "N": n, "M": m},
                                "wall_s": round(time.perf_counter() - t0, 1), "peak_gib": _peak_gib(), "units": units,
                                "external": sorted(external), "cuts": cuts, "wires": len(circuit.wires), "law": law,
                                "circuit_sha512": hashlib.sha512((out / "circuit.json").read_bytes()).hexdigest(),
                                "docket_digest": DD.digest(json.loads((out / "registration.json").read_text()))},
                               default=str), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    wn = sub.add_parser("window")
    wn.add_argument("--size", default="256,1,1,1,2", help="K,TM,TN,N,M")
    wn.add_argument("--out", type=Path, required=True)
    wn.add_argument("--rule", required=True, help="the sampling design over the window's proof blocks")
    st = sub.add_parser("stage")
    st.add_argument("--size", default="256,1,1,1,2", help="K,TM,TN,N,M")
    st.add_argument("--lower", required=True, help="flock-lower-check, Lean's lowering")
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--no-reader", dest="reader", action="store_false",
                    help="C without the reader: its units' y is C's output, the record a later reader is held to")
    st.add_argument("--steer", default=None, metavar="TEMPLATE",
                    help="draw only among the blocks whose template's descriptor id starts with TEMPLATE, recorded as "
                         "steered")
    st.add_argument("--until", choices=("amounts",), default=None,
                    help="stop once the laid C's tiles and their amounts are stated (LAID, AMOUNTS, CREDIT)")
    dr = sub.add_parser("door")
    dr.add_argument("--lower", required=True)
    dr.add_argument("--lean", required=True, help="verity-verify-proof")
    dr.add_argument("--out", type=Path, required=True)
    dr.add_argument("--session", nargs=2, action="append", metavar=("PROOF", "DIR"), required=True)
    a = ap.parse_args()
    if a.cmd == "door":
        a.session = {t: [s for u, s in a.session if u == t] for t, _ in a.session}
        return door(a)
    if a.cmd == "window":
        return window(a)
    return stage(a)


if __name__ == "__main__":
    sys.exit(main())
