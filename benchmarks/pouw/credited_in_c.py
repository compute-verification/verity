"""A credited linear of C whose nodes write its y, through the door.

C is one linear y = x · Wᵀ over M rows of K and N columns whose nodes are its grid's units of `PearlPlusTileCoreWords_v4`
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
``stage`` evaluates C on an honest linear and stages a dishonest proof of the drawn tile (``--bad``: ``y``, its claimed y
a word other than the one it computes; ``cert``, a certificate word other than the prover's; ``filler``, a nonzero word
in its filler row; each claiming the honest outputs), places each laid block in its tile and each tile's amount
(`scheme.laid_tiles` over the laid C's strata, `tile_macs` on the tile's statics), draws one block of the wired cores'
stratum, cuts it (`restrict`), lowers it by the door (`one_circuit.lower` under C's `one_circuit.cuts`), and finds each
of its input rows' writers in C: a row C wires from more than one writer unit (the core's fa and fb, K lines; g, 32 Gram
rows) is no one writer's output field, which Lean's registered reads refuse (`Flock.Registered.treeReads`,
``FIELD_WHY``) unless the lowering cuts it into one row a writer (``p#j``, `Flock.OneCircuit.io`), and the stage stops
there when it does not (``WIRED-ROWS``, exit 4); a row one writer's field feeds is the writer's committed output, which
this stage does not commit, so it stops there too (``WRITER-ROWS``, exit 4); otherwise it stages the block for
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
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC

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
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC

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
    from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_u as CU

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
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
    t = ln.tiles[index]
    r, s = t.rows[0] // len(t.rows), t.core.bindings
    b = [[v << 16 for v in row] for row in vals["w"][t.col]]
    d, y, ok = PT.reference(s["DEV"], s["K"], s["N"], vals["x"][r], b, vals["cert"][index], vals["salt"], vals["root"],
                            unit, list(t.rows), list(t.cols), s["RHO"], s["RHOR"], t.real)
    return [*(d or bytes(64)), *(w for row in (y or [[0] * len(t.cols)] * t.real) for w in row), ok]


def tamper(ln: Linear, index: int, vals: dict, bad: str) -> dict:
    """``vals`` with unit ``index``'s tile made false: ``cert``, its first certificate word's low bit flipped; ``filler``,
    its last row, a filler row, with word 3 set to 1.0; ``y``, unchanged (the forgery is in the outputs it claims)."""
    t = ln.tiles[index]
    r, out = t.rows[0] // len(t.rows), json.loads(json.dumps({v: vals[v] for v in ("x", "cert")}))
    if bad == "cert":
        out["cert"][index][0][0][0][0] ^= 1
    elif bad == "filler":
        if t.real == len(t.rows):
            raise ValueError(f"unit {index}'s tile has no filler row: {t.real} of {len(t.rows)} rows are real")
        out["x"][r][-1][3] = 0x3F800000
    elif bad != "y":
        raise ValueError(f"no tamper {bad!r}")
    return {**vals, **out}


MANY = ("x", "w", "cert", "r", "rows", "col")


def rows_of(ln: Linear, values: dict) -> dict[str, list[list[int]]]:
    """Each value's registered rows (a many-row value's rows, a one-row value its one), each its leaves packed into the
    value's registered width (`registered.width`, `registered.bit_words`)."""
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG
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
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG
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


def reads(program, partition, rows: dict, roots: dict[str, str], outs: dict) -> dict:
    """The verifier's ``--registered`` file over its program and the partition's query, position-free
    (`registered.reads_file`), with each output field of ``outs``."""
    from verity.commitments import registered_value as RV
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG
    f = REG.reads_file(program, partition["query"], {v: len(rs[0]) for v, rs in rows.items()}, roots,
                       schemas={v: RV.SCHEMA_HM96_SHA512_ROW_V2 for v in rows}, positions=False)
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
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.resource_accounting.compute.pouw import scheme as PS
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
    from verity.commitments import registered_value as RV
    from verity_flock import circuit as FC

    rec: dict = {"size": {"K": k, "TM": tm, "TN": tn, "N": n, "M": m}, "bad": a.bad, "scheme": SCHEME, "phases": {}}
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
    assert laid == cut + ([descriptor_id(bind(Bf16AddRow, N=n))] * m if a.reader else []), (laid[:12], cut[:12])
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
    key = PO.template_id(core)
    phase("lay", descriptor_bytes=len(c), population=population)

    strata = PQ.strata(program, part["query"])
    tiles = PS.laid_tiles(ln.program, program, strata)
    rec["amounts"] = {"strata": [[tid, rs] for tid, rs in strata],
                      "tiles": [{"index": tl.index, "gates": tl.gates, "amount": tl.amount, "blocks": len(tl.blocks),
                                 "kinds": dict(tl.kinds)} for tl in tiles]}
    print("AMOUNTS " + json.dumps(rec["amounts"]), flush=True)
    assert [tl.amount for tl in tiles] == [PS.tile_macs(tl.core.bindings) for tl in ln.tiles]

    coin = os.urandom(64)
    cores = [u for tid, rs in strata if tid == descriptor_id(core) for lo, hi in rs for u in range(lo, hi)]
    if not cores:
        raise SystemExit(f"no-core-block: the laying opened {descriptor_id(core)}, so no block of the laid C is a wired "
                         "core for this stage to stage")
    block_no = cores[int.from_bytes(hashlib.sha512(coin).digest(), "little") % len(cores)]
    (index,) = [tl.index for tl in tiles if block_no in tl.blocks]
    rec["draw"] = {"coin": coin.hex(), "block": block_no, "tile": index, "amount": tiles[index].amount,
                   "of": "one block uniformly of the wired cores' stratum, the block this stage stages; its tile is "
                         "credited its amount only when every stratum it has a block in passed (credited_tiles)"}
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
    drawn = ln.tiles[index]
    bad = tamper(ln, index, good, a.bad)
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

    (got_h, z_h), (got_d, _z_d) = run(good), run(bad)
    got = {"honest": got_h, "dishonest": got_d}
    want = {tag: reference(ln, index, vals, unit) for tag, vals in (("honest", good), ("dishonest", bad))}
    y_rows = [[0] * n for _ in range(m)]
    for tile, g in zip(ln.tiles, got_h):
        for i in range(tile.real):
            y_rows[tile.rows[i]][tile.col * tn:(tile.col + 1) * tn] = y_of(g)[i * tn:(i + 1) * tn]
    add = evaluate_batch(bind(Bf16AddRow, N=n), np.asarray(y_rows, dtype=np.int64),
                         np.asarray(good["r"] if a.reader else [[0] * n] * m, dtype=np.int64))
    rec["evaluate"] = {"of": "C on its words (`evaluate_batch` of its root before laying, whose gates the laid C keeps): "
                              "each unit's (digest, y, ok) and the reader's rows",
                       "drawn": {"rows": drawn.rows, "real": drawn.real, "credited": tiles[index].amount},
                       "honest_ok": [g[-1] for g in got_h], "dishonest_ok": [g[-1] for g in got_d],
                       "y": y_rows, "y_is_the_served_y": y_rows == served,
                       "reader_reads_y_by_wire": z_h == add.reshape(m, n).tolist() if a.reader else None,
                       "drawn_reference_agrees": {tag: words(got[tag][index]) == want[tag] for tag in got}}
    phase("evaluate")
    print("EVALUATE " + json.dumps(rec["evaluate"]), flush=True)
    assert all(rec["evaluate"]["drawn_reference_agrees"].values()), rec["evaluate"]
    assert rec["evaluate"]["y_is_the_served_y"] and rec["evaluate"]["reader_reads_y_by_wire"] in (True, None), rec["evaluate"]
    assert rec["evaluate"]["honest_ok"] == [1] * units
    assert rec["evaluate"]["dishonest_ok"][index] == (1 if a.bad == "y" else 0), rec["evaluate"]

    def fields_of(g: list[int]) -> dict[str, list[int]]:
        """A unit's output fields' bits, each leaf low bit first: o0 the digest, o1 y's words, o2 ok's word."""
        return {"o0": g[:nd], "o1": [(w >> j) & 1 for w in y_of(g) for j in range(16)],
                "o2": [(g[-1] >> j) & 1 for j in range(16)]}

    claimed = {"honest": [fields_of(g) for g in got_h], "dishonest": [fields_of(g) for g in got_h]}
    if a.bad == "y":
        claimed["dishonest"][index]["o1"][15] ^= 1

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        cut = OC.cuts(c, part)
        text = OC.lower(b.descriptor, part, lambda: a.lower, cut=cut).decode()
    except (OC.NotDerived, OC.Refused) as e:
        rec["lower"] = {"refused": f"{type(e).__name__}: {e}", "query": part["query"]["name"]}
        print("LOWER-REFUSED " + json.dumps(rec["lower"]), flush=True)
        (a.out / "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 3
    meta = FC.meta_of(text)
    rec["lower"] = {"by": "the door's one_circuit.lower (flock-lower-check compose)", "wall_s": round(time.perf_counter() - t0, 2),
                    "peak_gib_children": _peak_gib(resource.RUSAGE_CHILDREN), "bytes": len(text), "k_log": meta.get("k_log"),
                    "sha512": hashlib.sha512(text.encode()).hexdigest(), "ports": [q[:2] for q in meta.get("ports", [])],
                    "cut": cut.get(descriptor_id(core), {})}
    print("LOWER " + json.dumps(rec["lower"]), flush=True)
    phase("lower")

    widths = dict((q[0], q[1]) for q in meta["ports"])
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
    if fed := {p: [r for r in rows if r == p or r.startswith(p + "#")] for p in wired}:
        rec["writer_rows"] = {
            "rows": {p: len(rs) for p, rs in fed.items()},
            "refused": "the drawn block reads these rows each from one writer unit's output field (in(S)'s cut, "
                       "one_circuit.cuts), which the door takes from each writer's committed output, as the period's "
                       "members' rows (flock_inputs); this stage registers only C's registered values and the drawn "
                       "block's own outputs (registered.reads_file), so it commits no writer's output and stages no "
                       "proof"}
        print("WRITER-ROWS " + json.dumps(rec["writer_rows"]), flush=True)
        (a.out / "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 4
    verifier_row = public_values(ln, k, unit, salt, w_reg["w"].root)
    verifier_roots = {v: r.root.hex() for v, r in register(program, rows_of(ln, verifier_row), supplied).items()}
    rec["public_row"] = {"of": "the verifier's: `public_values` from the salt, W's registered root and the tiles' places "
                         "alone, no noise value", "ports": list(PC.PUBLIC),
                         "salt": salt.hex(), "w_root": w_reg["w"].root.hex(), "supplied": list(supplied)}
    other = PC.public(SCHEME, k, drawn, unit + 1, good["salt"], good["root"])
    wrong_unit = rows_of(ln, {"unit": other["unit"]})
    rec["registered"] = {}
    for tag, vals in (("honest", good), ("dishonest", bad)):
        d = a.out / tag
        (d / "tables").mkdir(parents=True, exist_ok=True)
        (d / "circuit.txt").write_text(text)
        rows = rows_of(ln, vals)
        fresh = register(program, {v: r for v, r in rows.items() if v != "w"}, supplied)
        regs = {v: w_reg[v] if v == "w" else fresh[v] for v in rows}
        assert all(regs[v].root.hex() == verifier_roots[v] for v in supplied)
        roots = {v: r.root.hex() for v, r in regs.items()}
        fields = {p: [f[p] for f in claimed[tag]] for p in ("o0", "o1", "o2")}
        out_rows = {p: [words16(f) + [0] * (widths[p] - len(words16(f))) for f in fs] for p, fs in fields.items()}
        outs = {"o0": output(program, f"{key}/o0", out_rows["o0"], False),
                "o1": output(program, f"{key}/o1", out_rows["o1"], False),
                "o2": output(program, f"{key}/o2", out_rows["o2"], True)}
        (d / "registered.json").write_text(json.dumps(reads(program, part, rows, roots, outs)))
        if tag == "honest":
            bad_roots = roots | {"unit": register(program, wrong_unit, supplied)["unit"].root.hex()}
            (d / "registered-wrong.json").write_text(json.dumps(reads(program, part, rows, bad_roots, outs)))
            zero = [list(r) for r in out_rows["o2"]]
            zero[index][0] = 0
            not_ok = outs | {"o2": output(program, f"{key}/o2", zero, True)}
            (d / "registered-not-ok.json").write_text(json.dumps(reads(program, part, rows, roots, not_ok)))
        at = {p: r for p, r in ln.reads[index].items() if p in dict(core.signature()[0])}
        tables, salts, registered = {}, {}, {}
        for p, (v, j) in at.items():
            rs, ss, entry = regs[v].statement([j])
            tables[p] = np.asarray([RV.row_words(r, regs[v].word_bits, RV.SCHEMA_HM96_SHA512_ROW_V2) for r in rs],
                                   dtype=np.uint16)
            salts[p], registered[p] = ss, entry
        for p, o in outs.items():
            _rs, ss, entry = o.statement([index])
            salts[p], registered[p] = ss, entry
        outs_h = {p: np.asarray([words16(fields[p][index])], dtype=np.uint16) for p in fields}
        iset = SimpleNamespace(name=f"credited-in-c-{tag}", subcircuit=SimpleNamespace(id=core.id, inputs=[], outputs=[]))
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
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
    os.environ[PF.TMPL_SWITCH] = "1"
    PF.executable = lambda: Path(a.lean)
    PF.lower_executable = lambda: Path(a.lower)
    restrict, memo = OC.restrict, {}

    def once(c: bytes, partition, units) -> bytes:
        key = (hashlib.sha512(c).hexdigest(), json.dumps(partition, sort_keys=True), json.dumps([list(u) for u in units]))
        if key not in memo:
            memo[key] = restrict(c, partition, units)
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
    if "dishonest" in a.session:
        call("dishonest", "registered.json")
    for name in ("registered-wrong.json", "registered-not-ok.json"):
        call("honest", name)
    if honest["refusal"] is None and honest["findings"] and all(f["accepted"] for f in honest["findings"]):
        print(f"ACCEPT the door accepts drawn block {draw_['units']} of C, a unit of {partition['query']['params']['template']}"
              ": the tile with its noise derived inside from the salt and W's root, read with its unit, rows and col from the "
              "public row the verifier registered itself; its digest, the tile leaf, is bound to no transcript root, and the "
              "salt and the root to no supplied function (pouw.pearlplus.salt/v1, registered.root/v1)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    st = sub.add_parser("stage")
    st.add_argument("--size", default="256,1,1,1,2", help="K,TM,TN,N,M")
    st.add_argument("--lower", required=True, help="flock-lower-check, Lean's lowering")
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--no-reader", dest="reader", action="store_false",
                    help="C without the reader: its units' y is C's output, the record a later reader is held to")
    st.add_argument("--bad", choices=("y", "cert", "filler"), default="y",
                    help="the dishonest proof of the drawn tile: its claimed y a word other than the one it writes, a "
                         "certificate word not the prover's, or a word of its filler row nonzero")
    dr = sub.add_parser("door")
    dr.add_argument("--lower", required=True)
    dr.add_argument("--lean", required=True, help="verity-verify-proof")
    dr.add_argument("--out", type=Path, required=True)
    dr.add_argument("--session", nargs=2, action="append", metavar=("PROOF", "DIR"), required=True)
    a = ap.parse_args()
    if a.cmd == "door":
        a.session = {t: [s for u, s in a.session if u == t] for t, _ in a.session}
        return door(a)
    return stage(a)


if __name__ == "__main__":
    sys.exit(main())
