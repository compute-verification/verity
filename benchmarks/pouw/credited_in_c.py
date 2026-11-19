"""A credited linear of C with Pearl+'s check as its nodes, through the door.

C is one linear y = x · Wᵀ over M rows of K and N columns whose nodes are its grid's units of `PearlPlusTileCoreWords_v1`
(`pearl_plus_credited.tiles`), each reading its ports as rows of the values C holds: x's and W's rows (``a``, ``b``), the
linear's one y (``y``), the prover's certificates (``cert``) and the public row (`pearl_plus_credited.PUBLIC`: ``salt``,
``root``, ``unit``, ``rows``, ``col``, ``eb``, ``fa``, ``fb``, ``g``), registered under public salts; and each unit's ``ok``
registered as 1 the same way. Here the stager computes the public row (`pearl_plus_credited.public`, from its own salt and
a root of its own, not W's registered root) and writes the verifier's registered file from it: that the verifier computes
each field itself is `Registered.supplied`'s, not this run's. ``stage``
evaluates C on an honest linear and on one whose drawn tile is false (``--bad``: its y word's sign flipped, or a
nonzero word in its filler row), places the units' amounts
(`resource_accounting.scheme.credited` over C's strata, `PEARL_PLUS.blocks`' cost on the template's statics), draws one
credited block, cuts it (`restrict`), lowers it by the door (`one_circuit.lower`) and stages it for ``flock-circuit``.
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
PRIVATE = ("x", "w", "y", "cert")


def _peak_gib(who: int = resource.RUSAGE_SELF) -> float:
    return round(resource.getrusage(who).ru_maxrss / 2**20, 3)


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


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
    """C: ``program`` over its values, ``tiles`` its units in order, ``types`` each value's row type and ``reads`` each
    unit's ports as (value, row)."""

    program: object
    tiles: list
    types: dict
    reads: list


def linear(k: int, m: int, n: int, tile: tuple[int, int]) -> Linear:
    from verity.circuits.defs import CompositeDefinition
    from verity.circuits.program import Program
    from verity.circuits.refs import Coll, concat
    from verity.circuits.types import Array
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC

    tm, tn = tile
    ts = PC.tiles(SCHEME, k, m, n, tile)
    if len({t.real for t in ts}) != 1:
        raise ValueError(f"every row tile of M = {m} at TM = {tm} has the same real rows, so y's rows are one type: "
                         "M below TM or a multiple of it")
    port, ret = ts[0].core.signature()
    port = dict(port)
    rt, ct = len({t.rows for t in ts}), n // tn
    row = {"x": port["a"], "w": port["b"], "y": port["y"], "cert": port["cert"], "rows": port["rows"],
           "col": port["col"], "eb": port["eb"]}
    count = {"x": rt, "w": ct, "y": rt, "cert": len(ts), "rows": rt, "col": ct, "eb": ct}
    params = tuple((v, Array(count[v], row[v])) if v in count else (v, port[v])
                   for v in ("x", "w", "y", "cert", "salt", "root", "unit", "rows", "col", "eb", "fa", "fb", "g"))
    types = {v: row.get(v, t) for v, t in params}
    reads = []
    for i, t in enumerate(ts):
        r = t.rows[0] // tm
        reads.append({"a": ("x", r), "b": ("w", t.col), "y": ("y", r), "cert": ("cert", i), "salt": ("salt", 0),
                      "root": ("root", 0), "unit": ("unit", 0), "rows": ("rows", r), "col": ("col", t.col),
                      "eb": ("eb", t.col), "fa": ("fa", 0), "fb": ("fb", 0), "g": ("g", 0)})

    def body(B, S, *xs):
        held = dict(zip((v for v, _ in params), xs))
        outs = []
        for t, rd in zip(ts, reads):
            args = []
            for p, pt in t.core.signature()[0]:
                v, j = rd[p]
                got = held[v][j] if v in count else held[v]
                assert got.type == pt, (p, got.type, pt)
                args.append(got)
            outs.append(B.call(t.core, *args))
        return Coll(Array(len(ts), ret), concat([o.refs for o in outs]))

    root = CompositeDefinition("PearlPlusCreditedLinear", 1, (), lambda S: (params, Array(len(ts), ret)), body,
                               register=False)
    return Linear(Program(root, registered=[v for v, _ in params]), ts, types, reads)


def honest(k: int, m: int, n: int, ln: Linear, unit: int, seed: str) -> dict:
    """An honest linear: x and W BF16 (widened to FP32), the epoch's salt and the weights' root, each tile's certificates
    and the served y (`PearlC.checked`'s U unscaled, BF16), and the public row; every value as C holds it, a row a port's
    read (x and y a row tile's rows, W a column tile's)."""
    from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
    from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_u as CU

    rng = random.Random(seed)

    def bf16_row() -> list[int]:
        return [int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) & 0xFFFF0000 for _ in range(k)]

    tm, tn = len(ln.tiles[0].rows), len(ln.tiles[0].cols)
    x = [bf16_row() if i < m else [0] * k for i in range(-(-m // tm) * tm)]
    w = [bf16_row() for _ in range(n)]
    salt, root = rng.randbytes(32), rng.randbytes(64)
    s = pouw_schemes.get(SCHEME, k_min=k, tile=(tm, tn))
    dev = s.device.name
    _d, ng = inert_bound._promoted(dev, k)
    y = [[0] * n for _ in range(m)]
    certs = []
    pub = {"rows": {}, "col": {}, "eb": {}}
    for t in ln.tiles:
        ctx = C.Context(salt, C.Matmul(unit, C.Shape(len(x), k, n), 0), bytes(64), root)
        a, b = [x[i] for i in t.rows], [w[j] for j in t.cols]
        ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r) for r in a], [C.f32_row(r) for r in b], C.Tile(t.rows, t.cols))
        checked = s.checked(ctx, ops)
        certs.append([[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], dev))[2 * z:2 * z + 2]
                        for z in range(ng - 1)] for j in range(tn)] for i in range(tm)])
        for i in range(t.real):
            for j, c in enumerate(t.cols):
                y[t.rows[i]][c] = CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) << 16
        p = PC.public(SCHEME, k, t, unit, salt, root)
        pub["rows"][t.rows[0] // tm], pub["col"][t.col], pub["eb"][t.col] = p["rows"], p["col"], p["eb"]
        shared = {v: p[v] for v in ("salt", "root", "unit", "fa", "fb", "g")}
    row_tiles = sorted({t.rows: t.real for t in ln.tiles}.items())
    values = {"x": [[x[i] for i in rs] for rs, _ in row_tiles],
              "w": [[[v >> 16 for v in w[j]] for j in range(c * tn, (c + 1) * tn)] for c in range(n // tn)],
              "y": [[y[i] for i in rs[:real]] for rs, real in row_tiles], "cert": certs, **shared,
              **{v: [pub[v][r] for r in range(len(pub[v]))] for v in ("rows", "col", "eb")}}
    return values


def reference(ln: Linear, index: int, vals: dict, unit: int) -> list[int]:
    """`pearl_plus_tile.reference`'s (digest, ok) of unit ``index`` on ``vals``, the digest's 64 bytes then ok."""
    from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
    t = ln.tiles[index]
    r, s = t.rows[0] // len(t.rows), t.core.bindings
    b = [[v << 16 for v in row] for row in vals["w"][t.col]]
    d, ok = PT.reference(s["DEV"], s["K"], s["N"], vals["x"][r], b, vals["y"][r], vals["cert"][index], vals["salt"],
                         vals["root"], unit, list(t.rows), list(t.cols), t.col, s["RHO"], s["RHOR"])
    return [*(d or bytes(64)), ok]


def tamper(ln: Linear, index: int, vals: dict, bad: str) -> dict:
    """``vals`` with unit ``index``'s tile made false: ``y``, its first real row's first word of its column tile with the
    sign flipped; ``filler``, its last row, a filler row, with word 3 set to 1.0."""
    t = ln.tiles[index]
    r, out = t.rows[0] // len(t.rows), json.loads(json.dumps({v: vals[v] for v in ("x", "y")}))
    if bad == "y":
        out["y"][r][0][t.cols[0]] ^= 1 << 31
    elif t.real < len(t.rows):
        out["x"][r][-1][3] = 0x3F800000
    else:
        raise ValueError(f"unit {index}'s tile has no filler row: {t.real} of {len(t.rows)} rows are real")
    return {**vals, **out}


MANY = ("x", "w", "y", "cert", "rows", "col", "eb")


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
    from verity.circuits import partition_object as PO
    from verity.circuits import partition_query as PQ
    from verity.circuits import restrict as R
    from verity.circuits.codec import descriptor_chunks
    from verity.circuits.evaluation import evaluate_batch
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.resource_accounting import scheme as RS
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

    ln = linear(k, m, n, (tm, tn))
    program, units = ln.program, len(ln.tiles)
    core = ln.tiles[0].core
    part = PO.build(program, PO.template_instance_query(core), [(u, u + 1) for u in range(units)])
    c = b"".join(descriptor_chunks(program))
    key = part["query"]["params"]["template"]
    phase("program", descriptor_bytes=len(c), units=units)

    strata = PQ.strata(program, part["query"])
    credited = sorted(RS.credited([PS.PEARL_PLUS], strata))
    decoded = R.program(c)
    (block,) = PS.PEARL_PLUS.blocks
    amounts = {u: block.cost(decoded.defs[tid].bindings) for tid, rs in strata for lo, hi in rs for u in range(lo, hi)
               if u in credited}
    rec["amounts"] = {"strata": [[tid, rs] for tid, rs in strata], "credited": credited, "amount": amounts,
                      "tile_macs": [PS.tile_macs(tl.core.bindings) for tl in ln.tiles]}
    print("AMOUNTS " + json.dumps(rec["amounts"]), flush=True)
    assert credited == list(range(units)) and [amounts[u] for u in credited] == rec["amounts"]["tile_macs"]

    coin = os.urandom(64)
    index = credited[int.from_bytes(hashlib.sha512(coin).digest(), "little") % len(credited)]
    rec["draw"] = {"coin": coin.hex(), "block": index, "amount": amounts[index], "of": "one block uniformly of "
                   "the credited ones, whose amounts are equal"}
    print("DRAW " + json.dumps(rec["draw"]), flush=True)
    blk = OC.block(c, part, index)
    b = R.restrict(c, part, blk)
    phase("restrict", block_units=blk, block_bytes=len(b.descriptor), bit_gates=int(b.bits.size))

    unit = 7
    good = honest(k, m, n, ln, unit, "credited-in-c")
    drawn = ln.tiles[index]
    bad = tamper(ln, index, good, a.bad)

    params = [v for v, _ in program.fn.signature()[0]]

    def run(values: dict) -> list[list[int]]:
        flat = evaluate_batch(program.fn, *(np.asarray([_flat(values[v])], dtype=np.int64) for v in params))
        flat = flat.reshape(-1).tolist()
        n_out = len(flat) // units
        return [flat[n_out * u:n_out * (u + 1)] for u in range(units)]

    def words(g: list[int]) -> list[int]:
        return [sum(g[8 * i + j] << j for j in range(8)) for i in range(64)] + [g[512]]

    got = {"honest": run(good), "dishonest": run(bad)}
    want = {tag: reference(ln, index, vals, unit) for tag, vals in (("honest", good), ("dishonest", bad))}
    rec["evaluate"] = {"of": "C on its words (`evaluate_batch` of its root): each unit's (digest, ok)",
                       "drawn": {"rows": drawn.rows, "real": drawn.real, "credited": amounts[index]},
                       "honest_ok": [g[512] for g in got["honest"]], "dishonest_ok": [g[512] for g in got["dishonest"]],
                       "drawn_reference_agrees": {tag: words(got[tag][index]) == want[tag] for tag in got}}
    phase("evaluate")
    print("EVALUATE " + json.dumps(rec["evaluate"]), flush=True)
    assert all(rec["evaluate"]["drawn_reference_agrees"].values()), rec["evaluate"]
    assert rec["evaluate"]["honest_ok"] == [1] * units and rec["evaluate"]["dishonest_ok"][index] == 0

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    text = OC.lower(b.descriptor, part, lambda: a.lower).decode()
    meta = FC.meta_of(text)
    rec["lower"] = {"by": "the door's one_circuit.lower (flock-lower-check compose)", "wall_s": round(time.perf_counter() - t0, 2),
                    "peak_gib_children": _peak_gib(resource.RUSAGE_CHILDREN), "bytes": len(text),
                    "sha512": hashlib.sha512(text.encode()).hexdigest(), "ports": [q[0] for q in meta.get("ports", [])]}
    print("LOWER " + json.dumps(rec["lower"]), flush=True)
    phase("lower")

    widths = dict((q[0], q[1]) for q in meta["ports"])
    supplied = tuple(v for v in params if v not in PRIVATE)
    pub_rows = rows_of(ln, good)
    verifier_roots = {v: r.root.hex() for v, r in register(program, {v: pub_rows[v] for v in supplied}, supplied).items()}
    other = PC.public(SCHEME, k, drawn, unit + 1, good["salt"], good["root"])
    wrong_unit = rows_of(ln, {"unit": other["unit"]})
    rec["registered"] = {}
    for tag, vals in (("honest", good), ("dishonest", bad)):
        d = a.out / tag
        (d / "tables").mkdir(parents=True, exist_ok=True)
        (d / "circuit.txt").write_text(text)
        rows = rows_of(ln, vals)
        regs = register(program, rows, supplied)
        assert all(regs[v].root.hex() == verifier_roots[v] for v in supplied)
        roots = {v: r.root.hex() for v, r in regs.items()}
        fields = {"o0": [g[:512] for g in got["honest"]], "o1": [g[512:] for g in got["honest"]]}
        out_rows = {p: [words16(f) + [0] * (widths[p] - len(words16(f))) for f in fs] for p, fs in fields.items()}
        outs = {"o0": output(program, f"{key}/o0", out_rows["o0"], False),
                "o1": output(program, f"{key}/o1", out_rows["o1"], True)}
        (d / "registered.json").write_text(json.dumps(reads(program, part, rows, roots, outs)))
        if tag == "honest":
            bad_roots = roots | {"unit": register(program, wrong_unit, supplied)["unit"].root.hex()}
            (d / "registered-wrong.json").write_text(json.dumps(reads(program, part, rows, bad_roots, outs)))
            zero = [list(r) for r in out_rows["o1"]]
            zero[index][0] = 0
            not_ok = outs | {"o1": output(program, f"{key}/o1", zero, True)}
            (d / "registered-not-ok.json").write_text(json.dumps(reads(program, part, rows, roots, not_ok)))
        at = ln.reads[index]
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
        (d / "draw.json").write_text(json.dumps({"units": [index]}))
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
        print(f"ACCEPT the door accepts drawn block {draw_['units']} of C, a unit of {partition['query']['params']['template']}",
              flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    st = sub.add_parser("stage")
    st.add_argument("--size", default="256,1,1,1,2", help="K,TM,TN,N,M")
    st.add_argument("--lower", required=True, help="flock-lower-check, Lean's lowering")
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--bad", choices=("y", "filler"), default="y",
                    help="the dishonest linear: the drawn tile's y word flipped, or a word of its filler row nonzero")
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
