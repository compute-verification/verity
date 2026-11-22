"""Pearl+'s tile in the served circuit, through the door.

The served circuit is WIDE's (TINY at hidden size 256, `test_steps.WIDE`) at class 1,1,1 and the test grain
(`test_window_circuit.GRAIN`, its GEMM unit K · TN words), lowered by `window_circuit.class_circuit` with one change: unit 0
of layer 0's qkv projection, `Gemm_v2{K=256,N=TN}` over columns 0 to TN − 1, is `PearlPlusTileCoreWords_v6`
(`pearl_plus_credited.served_product` at TM 1). It reads its ``a`` by wire from the writer that unit read x from (the first
norm), its ``b`` as that unit read ``w`` (row 0 of ``weights/model.layers.0.self_attn.qkv_proj.weight@<K·TN>``), its
``cert`` and the public row (`pearl_plus_credited.PUBLIC`) as registered values of the epoch (``pearlplus/<port>``), and it
writes columns 0 to TN − 1 of the linear's y as its ``o1``, which every reader of those columns reads by wire. Every other
unit is the served lowering's. At TN 4 the door's lowering refuses the core block (Lean's compose: 2^29 bits, past
2^27), and at TN 1 the served lowering refuses the grain (GEMM units of 256 words).

The tile is laid (`verity.circuits.laying`'s ``_laid`` on the tile's root node alone, under `fits`, which is
`credited_in_c.fits` within 2^`K_MAX`: Python's estimate of a block, which Lean's compose does not match) until each
of its units is one proof block; every other unit of C is one block as lowered. ``stage`` reports the laid tile's blocks
and their sizes (``LAID``), samples one of the core's blocks (``--steer``), cuts it out (`restrict`), lowers it by the door
(`one_circuit.lower` under `one_circuit.cuts`) and stages two proofs of it: ``honest``, on the served values (x the first
norm on token 7's embedding row, the registered weight row under the committer's registration, the prover's certificates
under the epoch's salt and that row's registered root), and ``flipped``, the same with bit 7 of word 5 of the weight row
the block reads flipped, its statement's registered read still the registration's. Each staged value is evaluated in the
block's cone only (`cone`): the units C wires into the block, from the parameters and the norm's input row.
``door`` puts each proved session through `proofs.verify`. `credited_served.sh` builds, proves and verifies; run it on a
node."""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tests/adapters/vllm"))
import credited_in_c as CI  # noqa: E402

SCHEME = "pearl-c-h100-v1-h3"
QKV = "model.layers.0.self_attn.qkv_proj.weight"
NORM = "model.layers.0.input_layernorm.weight"
K, N, TN = 256, 512, 4
#: the test grain's words a GEMM unit (`window_builder.Grain.gemm_words`) and Flock's largest block, 2^K_MAX bits
#: (`Flock/HmRow.lean` MAX_K_LOG), which the laying holds each block to
GEMM_WORDS, K_MAX = 1024, 27
TOKEN, UNIT = 7, 0
VALUE = "pearlplus/"
PUBLIC_VALUES = tuple(VALUE + p for p in ("salt", "root", "unit", "rows", "col"))


def served(tmp: Path, tile_fn) -> SimpleNamespace:
    """WIDE's checkpoint and registration, and its class 1,1,1 circuit with ``tile_fn`` in place of unit 0 of layer 0's
    qkv projection (module docstring)."""
    from tests.program.predict import test_steps as S
    from tests.program.predict import test_window_circuit as TW
    from tests.program.predict import test_pearl_plus_served_linear as SL
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.program.predict import window_lowering as WL
    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.window_builder import Row
    from verity_vllm.program.predict.window_circuit import WindowClass, class_circuit
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import bind

    grain = dataclasses.replace(TW.GRAIN, gemm_words=GEMM_WORDS)
    reg = S._registered(S.WIDE)
    d, man = SL._checkpoint(tmp)
    registration, rpath = TW._register(tmp / "committer", reg, d, checkpoints_manifest=man, grain=grain)
    values = WW.weight_values(reg, WW.record_words(reg, d, man), rpath, grain, keep_rows=True)
    ck = WR.CheckpointIndex(d, checkpoints_manifest=man, model="org/m", revision="r")
    placed: dict = {}
    orig = WL._Program._Gemm

    def gemm(self, j, node):
        b = node.fn.bindings
        Kn, Nn = b["K"], b["N"]
        if placed or self.cut is not None or Kn != K:
            return orig(self, j, node)
        x, w = node.args
        lo = self.weight(w.refs, Nn * Kn)
        if self.b._weight(lo, lo + Nn * Kn, Kn)[0] != QKV:
            return orig(self, j, node)
        xs = self.rows(x.refs)
        n = max(1, min(Nn, self.b.grain.gemm_words // Kn))
        assert n == TN, n
        pieces = []
        for s0, m, count in self._slices(j, Nn, n):
            if not count:
                continue
            if s0 == 0:
                assert m == TN, m
                tid, u = self.b.unit(tile_fn, 1)
                self.b.feed(tid, "a", u, 1, xs)
                self.b.read(tid, "b", u, (1,), lo, (TN * Kn,))
                for p in ("cert", "salt", "root", "unit", "rows", "col"):
                    self.b.value(tid, p, u, VALUE + p, 0, 1)
                pieces.append((0, TN, Row(tid, u, "o1", TN)))
                placed["tile"] = (tid, u, xs)
                s0, count = s0 + m, count - 1
                if not count:
                    continue
            tid, u = self.b.unit(bind(node.fn.defn, **{**b, "N": m}), count)
            self.b.feed(tid, "x", u, count, xs)
            self.b.read(tid, "w", u, (count,), lo + s0 * Kn, (m * Kn,))
            pieces.append((s0, s0 + count * m, Row(tid, u, "o0", m)))
            self.calls.append((self.position - 1, tid, u, count))
        self.outs[j] = pieces
        self.gemms[j] = (node.fn, xs, lo)

    WL._Program._Gemm = gemm
    try:
        circuit = class_circuit(reg, WindowClass(1, 1, 1), None, grain)
    finally:
        WL._Program._Gemm = orig
    if "tile" not in placed:
        raise SystemExit("NO-PLACE: no Gemm_v2 of K = 256 reads layer 0's qkv projection in WIDE's class 1,1,1 circuit")
    from verity.commitments import registered_value as RV
    from verity.commitments.periods import interface as IF
    ports = dict(tile_fn.signature()[0])
    wide = {VALUE + p: {IF.FORMAT_KEYS[0]: RV.SCHEMA_HM96_SHA512_ROW_V2, IF.FORMAT_KEYS[1]: 32}
            for p in ("cert", "salt", "root", "unit", "rows", "col") if ports[p].leaf_widths()[0] == 32}
    reg_ = circuit.registered
    circuit = dataclasses.replace(circuit, registered=dataclasses.replace(reg_, formats={**dict(reg_.formats), **wide}))
    tid = descriptor_id(tile_fn)
    wires = [{"reader": w.reader[:40], "port": w.port, "writer": w.writer[:40], "out": w.out, "at": w.at, "base": w.base,
              "radix": list(w.radix), "step": list(w.step), "cut": list(w.cut)}
             for w in circuit.wires if tid in (w.reader, w.writer)]
    return SimpleNamespace(reg=reg, registration=registration, rpath=rpath, values=values, ck=ck, circuit=circuit,
                           tile=placed["tile"], wires=wires)


def fits(ins: tuple[int, ...], outs: tuple[int, ...], ands: int) -> bool:
    """`credited_in_c.fits` within 2^`K_MAX`, the largest block Lean's compose takes."""
    from verity_flock import circuit as FC
    ports = [FC.Port(f"p{i}", -(-b // 16), 16 * -(-b // 16)) for i, b in enumerate((*ins, *outs))]
    useful = FC._flat_useful(ands, sum(p.n_bits for p in ports[:len(ins)]), sum(p.n_bits for p in ports[len(ins):]))
    return FC._block_ranges(FC._row_slots(ports)[0], FC._pow2(len(ports)), FC._slot_log(useful + 1), 1)[1] <= K_MAX


def cert_of(tile, a, b, salt: bytes, root: bytes) -> list:
    """The prover's inert-work certificates of the tile's one row against its TN columns (`test_pearl_plus_served_linear._cert`
    at this TN)."""
    from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
    from verity.protocols.accounting.compute.pouw.prover.circuit import inert_bound
    from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
    s = pouw_schemes.get(SCHEME, k_min=K, tile=(1, TN))
    dev = s.device.name
    _d, ng = inert_bound._promoted(dev, K)
    ctx = C.Context(salt, C.Matmul(UNIT, C.Shape(1, K, N), 0), bytes(64), root)
    ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r) for r in a], [C.f32_row(r) for r in b], C.Tile(tile.rows, tile.cols))
    assert ops is not None, "the served x and W rows don't encode under Pearl+'s scheme"
    return [[[list(inert_bound.certificates(ops.a.codes[0], ops.b.codes[j], dev))[2 * z:2 * z + 2] for z in range(ng - 1)]
             for j in range(TN)]]


def lay_tile(program, tile_ids: set[str], force: set[str] = frozenset()):
    """``program`` with the tile's units (``tile_ids``: the tile and its cut's units) opened (`laying._laid`) until each
    fits one block (`credited_in_c.fits`), and every unit of ``force`` opened whatever `fits` says (its estimate takes
    a block Lean's compose refuses); every other unit as it is. Refused by name where another unit is past the limit
    (``UNFIT``)."""
    from verity.circuits import laying as LY
    from verity.circuits.codec import descriptor_id
    while True:
        m = LY.Measure(program, LY.query(program))
        over = sorted({m.units[u][0] for u in range(len(m.units)) if not fits(*m.block(u, u + 1))})
        nodes = program.root.body.nodes
        mine = {k for k in over if descriptor_id(nodes[k].fn) in tile_ids}
        mine |= {k for k, nd in enumerate(nodes) if descriptor_id(nd.fn) in force}
        if other := sorted({descriptor_id(nodes[k].fn) for k in over} - tile_ids):
            raise SystemExit(f"UNFIT: served units past the composer's limit, not the tile's: {other[:6]}")
        if not mine:
            return program, m
        program = LY._laid(program, mine)


class Cone:
    """Root units of a laid C evaluated one member at a time on demand, from preset gate values (`at`): a gate not
    preset is its writer member's output, that member evaluated on its own arguments' gates first."""

    def __init__(self, program) -> None:
        from verity.circuits.layout import resolve
        self.p, self.scope, self.at = program, program.circuit.root_scope, {}
        self.nodes = program.root.body.nodes
        self.base = [resolve(self.scope, ("n", k, 0)) for k in range(len(self.nodes))]
        self.order = sorted(range(len(self.nodes)), key=lambda k: self.base[k])
        self.starts = [self.base[k] for k in self.order]
        self.done: set = set()
        self.out: dict = {}

    def per(self, k: int) -> int:
        return self.nodes[k].fn.signature()[1].leaves

    def writer(self, g: int) -> tuple[int, int]:
        k = self.order[bisect.bisect_right(self.starts, g) - 1]
        return k, (g - self.base[k]) // self.per(k)

    def arg_gates(self, k: int, m: int) -> list[list[int]]:
        from verity.circuits.cut import member_refs
        from verity.circuits.layout import resolve
        nd = self.nodes[k]
        axes = nd.axes if nd.form == "batch" else (None,) * len(nd.args)
        out = []
        for arg, ax in zip(nd.args, axes):
            rs = member_refs(arg, ax, m) if nd.form == "batch" else arg.refs
            out.append([resolve(self.scope, rs[j]) for j in range(len(rs))])
        return out

    def member(self, k: int, m: int) -> list[int]:
        from verity.circuits.codec import descriptor_id
        from verity.circuits.cut import member_refs
        from verity.circuits.evaluation import evaluate_batch
        from verity.circuits.layout import resolve
        from verity.circuits.program import is_input
        if (k, m) not in self.done:
            nd = self.nodes[k]
            if is_input(nd.fn):
                raise SystemExit(f"CONE: parameter node {k} ({nd.fn.id}) is read and was not preset")
            params = nd.fn.signature()[0]
            axes = nd.axes if nd.form == "batch" else (None,) * len(nd.args)
            args = []
            for j, (arg, ax) in enumerate(zip(nd.args, axes)):
                rs = member_refs(arg, ax, m) if nd.form == "batch" else arg.refs
                want = params[j][1].leaves
                if len(rs) != want:
                    raise SystemExit(f"CONE: node {k} member {m} argument {j}: {len(rs)} references of {want} leaves "
                                     f"({arg.refs!r}, axis {ax}, {arg.type})")
                row = []
                for r in rs:
                    g = resolve(self.scope, r)
                    if g in self.at:
                        row.append(self.at[g])
                        continue
                    if r[0] != "n":
                        raise SystemExit(f"CONE: node {k} reads {r}, not a root node's leaf")
                    wk, wm = r[1], r[2] // self.per(r[1])
                    if (wk, wm) == (k, m):
                        raise SystemExit(f"CONE: node {k} reads its own output")
                    row.append(self.member(wk, wm)[r[2] % self.per(wk)])
                args.append(np.asarray([row], dtype=np.int64))
            t = time.perf_counter()
            got = np.asarray(evaluate_batch(nd.fn, *args)).reshape(-1).tolist()
            print(f"CONE {descriptor_id(nd.fn)[:48]} node {k} member {m}: {len(got)} leaves, "
                  f"{time.perf_counter() - t:.1f} s", flush=True)
            self.out[(k, m)] = got
            self.done.add((k, m))
        return self.out[(k, m)]

def units_count(program, tid: str) -> int:
    from verity.circuits.codec import descriptor_id
    return sum(nd.n if nd.form == "batch" else 1 for nd in program.root.body.nodes if descriptor_id(nd.fn) == tid)


def stage(a) -> int:
    from verity.circuits import partition_object as PO
    from verity.circuits import partition_query as PQ
    from verity.circuits import restrict as R
    from verity.circuits import laying as LY
    from verity.circuits.codec import descriptor_chunks, descriptor_id
    from verity.circuits.program import input_id
    from verity.commitments import registered_value as RV
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.proof.spot.verifier.one_stage import registered as REG
    from verity.protocols.run import schemes
    from verity_flock import circuit as FC
    PC = schemes.catalog("pearl-plus", "credited")
    PT = schemes.catalog("pearl-plus", "tile")

    rec: dict = {"scheme": SCHEME, "class": "1,1,1", "model": "WIDE (TINY at hidden 256)", "tile": [1, TN],
                 "gemm_words": GEMM_WORDS, "k_max": K_MAX, "phases": {}}
    t = time.perf_counter()

    def phase(name: str, **extra) -> None:
        nonlocal t
        now = time.perf_counter()
        rec["phases"][name] = {"wall_s": round(now - t, 2), "peak_gib": CI._peak_gib(), **extra}
        print(f"PHASE {name} {json.dumps(rec['phases'][name])}", flush=True)
        t = now

    tile_fn = PC.served_product(SCHEME, K, 1, TN, N)
    force: set[str] = set()
    if a.core == "v4":
        tile_fn = PC.bind(PC.PearlPlusTileCoreWordsV7, **tile_fn.bindings)
        force = {descriptor_id(PC.bind(PC.PearlPlusTileCoreWiredV4, **tile_fn.bindings))}
    tid = descriptor_id(tile_fn)
    tmp = Path(a.out) / "served"
    tmp.mkdir(parents=True, exist_ok=True)
    sv = served(tmp, tile_fn)
    by = {descriptor_id(fn): n for fn, n in sv.circuit.templates}
    rec["served"] = {"templates": {t_[:60]: n for t_, n in by.items()}, "units": sum(by.values()), "tile": tid,
                     "tile_wires": sv.wires}
    print("SERVED " + json.dumps(rec["served"]), flush=True)
    program0, query0 = sv.circuit.program()
    phase("served", units=sum(by.values()))

    cut = [descriptor_id(u) for u, n_ in PC.cut(tile_fn) for _ in range(n_)]
    program, m_ = lay_tile(program0, {tid, *cut, *force}, force)
    query = LY.query(program)
    m_ = LY.Measure(program, query)
    population = len(m_.units)
    laid = [descriptor_id(nd.fn) for nd in program.root.body.nodes if input_id(nd.fn.id) is None]
    core = next((descriptor_id(u) for u, _n in PC.cut(tile_fn) if u.name.startswith("PearlPlusTileCoreWired")), None)
    blocks: dict = {}
    for u in range(population):
        ins, outs, ands = m_.block(u, u + 1)
        t_ = descriptor_id(program.root.body.nodes[m_.units[u][0]].fn)
        if t_ not in cut:
            continue
        b = blocks.setdefault(t_, {"blocks": 0, "ands": ands, "ins_bits": sum(ins), "outs_bits": sum(outs),
                                   "ins_rows": len(ins), "outs_rows": len(outs)})
        b["blocks"] += 1
        assert b["ands"] == ands, (u, t_)
    rec["laid"] = {"by": f"laying._laid on the tile alone, each block within 2^{K_MAX} bits (fits)", "population": population,
                   "served_units": sum(by.values()) - 1, "tile_blocks": sum(b["blocks"] for b in blocks.values()),
                   "tile_ands": sum(b["blocks"] * b["ands"] for b in blocks.values()),
                   "by_template": blocks,
                   "cut_agrees": {t_: b["blocks"] for t_, b in blocks.items()} == dict(Counter(cut)),
                   "tile_root_nodes": sum(1 for t_ in laid if t_ in cut)}
    print("LAID " + json.dumps(rec["laid"]), flush=True)
    part = PO.build(program, query, [[(u, u + 1) for u in range(population)]])
    c = b"".join(descriptor_chunks(program))
    phase("lay", descriptor_bytes=len(c), population=population)
    if a.until == "laid":
        Path(a.out, "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 0

    strata = PQ.strata(program, part["query"])
    of = [""] * population
    for t_, rs in strata:
        for lo, hi in rs:
            of[lo:hi] = [t_] * (hi - lo)
    pool = [u for u in range(population) if of[u].startswith(a.steer)]
    if not pool:
        raise SystemExit(f"no-block: no block of the laid C is of {a.steer}")
    coin = os.urandom(64)
    block_no = pool[int.from_bytes(hashlib.sha512(coin).digest(), "little") % len(pool)]
    drawn_tid = of[block_no]
    is_core = drawn_tid == core
    key = PO.template_id(drawn_tid)
    rec["draw"] = {"coin": coin.hex(), "block": block_no, "template": drawn_tid, "steered": a.steer,
                   "of": f"one block uniformly of the {len(pool)} blocks of {a.steer} (steered)",
                   "population": dict(Counter(of))}
    print("DRAW " + json.dumps({k_: v for k_, v in rec["draw"].items() if k_ != "population"}), flush=True)
    blk = OC.block(c, part, block_no)
    b = R.restrict(c, part, blk)
    phase("restrict", block_units=blk, block_bytes=len(b.descriptor))
    try:
        cuts = OC.cuts(c, part)
        text = OC.lower(b.descriptor, part, lambda: a.lower, cut=cuts, formed=OC.formed_in(c, part, blk)).decode()
    except (OC.NotDerived, OC.Refused) as e:
        rec["lower"] = {"refused": f"{type(e).__name__}: {e}"}
        print("LOWER-REFUSED " + json.dumps(rec["lower"]), flush=True)
        Path(a.out, "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 3
    meta = FC.meta_of(text)
    rec["lower"] = {"bytes": len(text), "k_log": meta.get("k_log"), "ports": [q[:2] for q in meta.get("ports", [])],
                    "cut": cuts.get(drawn_tid, {}), "sha512": hashlib.sha512(text.encode()).hexdigest()}
    print("LOWER " + json.dumps(rec["lower"]), flush=True)
    phase("lower")
    if a.until == "lower":
        Path(a.out, "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 0

    widths = dict((q[0], q[1]) for q in meta["ports"])
    node, member = m_.units[blk[0][0]]
    if (leaf := CI.tree_leaf(program, node, member)) != 0 or units_count(program, drawn_tid) != 1:
        raise SystemExit(f"ONE-UNIT: the staged output trees are one row, and block {block_no} is leaf {leaf} of its "
                         "template's units")
    out_ports = [f"o{f}" for f in range(len(CI._field_starts(program.root.body.nodes[node].fn.signature()[1])) - 1)]
    rows_meta = [q[0] for q in meta["ports"]]
    wired = CI.writers(program, part["query"], blk[0][0])
    many = {p: len(ws) for p, ws in wired.items() if len(ws) > 1 and p in rows_meta}
    if many:
        rec["wired_rows"] = {"refused": f"rows {sorted(many)} each wired from more than one writer ({many})"}
        print("WIRED-ROWS " + json.dumps(rec["wired_rows"]), flush=True)
        Path(a.out, "stage.json").write_text(json.dumps(rec, indent=1, default=str))
        return 4
    fed = CI.wired_rows(program, node, member, rows_meta, rec["lower"]["cut"])
    rec["writer_rows"] = {p: [descriptor_id(program.root.body.nodes[kw].fn)[:48], mw, f] for p, (kw, mw, f) in fed.items()}
    print("WRITER-ROWS " + json.dumps(rec["writer_rows"]), flush=True)

    # The epoch's values: the registered weights, the norm's input row, the certificates and the public row.
    params = dict(tile_fn.signature()[0])
    pv = sv.values
    wparam = next(r.param for r in sv.circuit.reads if r.reader == tid and r.port == "b")
    nparam = next(r.param for r in sv.circuit.reads if r.param.startswith(f"weights/{NORM}@"))
    row = [int(v) for v in pv[wparam].rows[UNIT]]
    w16 = [row[j * K:(j + 1) * K] for j in range(TN)]
    emb = sv.ck.array("model.embed_tokens.weight").view(np.uint16)[TOKEN].tolist()
    salt = random.Random("credited-served/salt").randbytes(32)
    root = pv[wparam].root
    (tile,) = [t_ for t_ in PC.tiles(SCHEME, K, 1, N, tile=(1, TN)) if t_.col == UNIT]
    pub = PC.public(SCHEME, K, tile, UNIT, salt, root)

    cone = Cone(program)
    for v in (wparam, nparam):
        gates = REG.parameter_gates(program, v)
        words = [int(x) for r in pv[v].rows for x in r]
        assert len(gates) == len(words), (v, len(gates), len(words))
        cone.at.update(zip(gates, words))
    from verity.circuits.evaluation import evaluate
    (norm_fn,) = [fn for fn, _n in sv.circuit.templates if fn.id.startswith("RMSNormTriton_v5")]
    x = [int(v) for v in evaluate(norm_fn, emb, [int(v) for r in pv[nparam].rows for v in r])]
    if "a" in fed:
        kw_a, mw_a, _f = fed["a"]
        norm_args = cone.arg_gates(kw_a, mw_a)
        x_gates = next(gs for gs in norm_args if not set(gs) <= set(cone.at))
        assert len(x_gates) == K, len(x_gates)
        cone.at.update(zip(x_gates, emb))
        assert cone.member(kw_a, mw_a) == x
    a_rows = [[v << 16 for v in x]]
    b_rows = [[v << 16 for v in r] for r in w16]
    cert = cert_of(tile, a_rows, b_rows, salt, root)
    vals = {VALUE + "cert": cert, **{VALUE + p: pub[p] for p in ("salt", "root", "unit", "rows", "col")}}
    supplied = PUBLIC_VALUES
    ptype = {VALUE + p: params[p] for p in ("cert", "salt", "root", "unit", "rows", "col")}

    def value_rows(v: str, x) -> list[list[int]]:
        return [REG.bit_words(CI._flat(x), ptype[v].leaf_widths()[0], REG.width(program, v))]

    regs = {v: CI.register(program, {v: value_rows(v, x_)}, supplied)[v] for v, x_ in vals.items()}
    for v, x_ in vals.items():
        gates = REG.parameter_gates(program, v)
        words = CI._flat(x_)
        assert len(gates) == len(words), (v, len(gates), len(words))
        cone.at.update(zip(gates, words))
    s_ = tile_fn.bindings
    ref = PT.reference(s_["DEV"], K, N, a_rows, b_rows, cert, salt, root, UNIT, list(tile.rows), list(tile.cols),
                       s_["RHO"], s_["RHOR"], tile.real)
    phase("values")

    def run_core(b_words: list[int] | None) -> list[int]:
        if b_words is not None:
            for g, w in zip(REG.parameter_gates(program, wparam)[:TN * K], b_words):
                cone.at[g] = w
            cone.done.discard((node, member))
        return cone.member(node, member)

    got = run_core(None)
    nd_ = 512
    ret_d = program.root.body.nodes[node].fn.signature()[1]
    starts_d = CI._field_starts(ret_d)
    widths_d = ret_d.leaf_widths()
    if is_core:
        digest = bytes(sum(got[8 * i + j] << j for j in range(8)) for i in range(64))
        rec["evaluate"] = {"ok": got[-1], "y": got[nd_:nd_ + TN], "reference_ok": ref[2],
                           "reference_agrees": (digest, got[nd_:nd_ + TN], got[-1]) == (ref[0], ref[1][0], ref[2]),
                           "x_is_the_norm_on_token": TOKEN}
    else:
        rec["evaluate"] = {"block": drawn_tid, "leaves": len(got), "of": "the block's template on its cone's values"}
    print("EVALUATE " + json.dumps(rec["evaluate"]), flush=True)
    phase("evaluate")
    assert not is_core or (rec["evaluate"]["reference_agrees"] and got[-1] == 1), rec["evaluate"]

    def fields_of(g: list[int]) -> dict[str, list[int]]:
        if is_core:
            return {"o0": g[:nd_], "o1": [(w >> j) & 1 for w in g[nd_:nd_ + TN] for j in range(16)],
                    "o2": [(g[-1] >> j) & 1 for j in range(16)]}
        return {f"o{f}": [(w >> j) & 1 for w, wd in zip(g[lo:hi], widths_d[lo:hi]) for j in range(wd)]
                for f, (lo, hi) in enumerate(zip(starts_d, starts_d[1:]))}

    nodes = program.root.body.nodes

    def units_of(t_: str) -> int:
        return sum(nd.n if nd.form == "batch" else 1 for nd in nodes if descriptor_id(nd.fn) == t_)

    def tree(kw: int, f: int, words: int) -> tuple[str, list[list[int]]]:
        """Field ``f`` of the template of node ``kw`` as its one tree: a row a unit, the evaluated members' rows their
        outputs, every other row zero (the block opens only its writers' leaves)."""
        from verity.circuits.types import RecordType, Tuple
        t_ = descriptor_id(nodes[kw].fn)
        ret = nodes[kw].fn.signature()[1]
        ft = (list(ret.fields) if isinstance(ret, Tuple) else [x_ for _, x_ in ret.fields] if isinstance(ret, RecordType)
              else [ret])[f]
        starts, wd = CI._field_starts(ret), ft.leaf_widths()
        rows = [[0] * words for _ in range(units_of(t_))]
        for (k2, m2) in cone.done:
            if descriptor_id(nodes[k2].fn) == t_:
                leaves = cone.member(k2, m2)[starts[f]:starts[f + 1]]
                r = CI.words16([(w >> j) & 1 for w, x_ in zip(leaves, wd) for j in range(x_)])
                assert len(r) == words, (t_, f, len(r), words)
                rows[CI.tree_leaf(program, k2, m2)] = r
        return f"{t_}/o{f}", rows

    trees = sorted({(kw, f) for kw, _m, f in fed.values()})
    words_of = {(kw, f): widths[p] for p, (kw, _m, f) in fed.items()}
    committed = {}
    for kw, f in trees:
        name, trows = tree(kw, f, words_of[kw, f])
        committed[kw, f] = CI.output(program, name, trows, False)
    held = {p: committed[kw, f] for p, (kw, _m, f) in fed.items()}
    reg_rows = {wparam: [list(r) for r in pv[wparam].rows], **{v: value_rows(v, x_) for v, x_ in vals.items()}}
    domains = {wparam: pv[wparam].domain.domain_id.hex()}
    roots = {wparam: root.hex(), **{v: r.root.hex() for v, r in regs.items()}}
    file = REG.reads_file(program, part["query"], {v: len(rs[0]) for v, rs in reg_rows.items()}, roots,
                          domains=domains, schemas={v: RV.SCHEMA_HM96_SHA512_ROW_V2 for v in reg_rows},
                          positions=False, template=drawn_tid)
    regd = REG.reads(program, part["query"], {v: len(rs[0]) for v, rs in reg_rows.items()},
                     {v: RV.SCHEMA_HM96_SHA512_ROW_V2 for v in reg_rows}, template=drawn_tid)
    at = {p: (e["value"], e["rows"][blk[0][0]]) for p, e in regd.items() if e["rows"][blk[0][0]] is not None}
    rec["reads"] = {p: [v, j] for p, (v, j) in at.items()}
    print("READS " + json.dumps(rec["reads"]), flush=True)
    allregs = {wparam: pv[wparam], **regs}
    from verity.circuits.program import RUN
    names = [n for n, _t in program.root.signature()[0]]
    rec["reads_bound"] = {p: [v, names.index(v), program.bindings[names.index(v)]] for p, (v, _j) in at.items()}
    print("READS-BOUND " + json.dumps(rec["reads_bound"]), flush=True)

    def write(tag: str, g: list[int], table_override: dict) -> None:
        d = Path(a.out) / tag
        (d / "tables").mkdir(parents=True, exist_ok=True)
        (d / "circuit.txt").write_text(text)
        claimed = fields_of(g)
        out_rows = {p: [CI.words16(claimed[p]) + [0] * (widths[p] - len(CI.words16(claimed[p])))] for p in out_ports}
        outs = {p: CI.output(program, f"{key}/{p}", out_rows[p], is_core and p == "o2") for p in out_ports}
        f = json.loads(json.dumps(file))
        for p, (v, _j) in at.items():
            w = names.index(v)
            if program.bindings[w] == RUN:
                f["ports"][p]["value"] = REG.fresh_value(w)
        for p, o in (outs | held).items():
            f["ports"][p] = {"value": o.value, "root": o.root.hex(), "leaves": o.leaves, "words": o.words,
                             "word_bits": 16, "domain": o.domain.domain_id.hex(), RV.SCHEMA_KEY: o.schema}
        (d / "registered.json").write_text(json.dumps(f))
        tables, salts, registered = {}, {}, {}
        for p, (v, j) in at.items():
            rs, ss, entry = allregs[v].statement([j])
            tables[p] = np.asarray([RV.row_words(r, allregs[v].word_bits, RV.SCHEMA_HM96_SHA512_ROW_V2) for r in rs],
                                   dtype=np.uint16)
            salts[p], registered[p] = ss, entry
        for p, rw in table_override.items():
            tables[p] = rw
        for p, (kw, mw, ff) in fed.items():
            rs, ss, entry = held[p].statement([CI.tree_leaf(program, kw, mw)])
            tables[p] = np.asarray([RV.row_words(r, 16, RV.SCHEMA_HM96_SHA512_ROW_V2) for r in rs], dtype=np.uint16)
            salts[p], registered[p] = ss, entry
        for p, o in outs.items():
            _rs, ss, entry = o.statement([0])
            salts[p], registered[p] = ss, entry
        outs_h = {p: np.asarray([out_rows[p][0]], dtype=np.uint16) for p in out_ports}
        iset = SimpleNamespace(name=f"credited-served-{tag}",
                               subcircuit=SimpleNamespace(id=nodes[node].fn.id, inputs=[], outputs=[]))
        FC.write(text, iset, 0, 1, d / "inst-1.bin", d / "pub-1.bin", indices=[blk[0][0]], tables=tables,
                 refs={p: [0] for p in tables}, share_rows=True, salts=salts, registered=registered, outs=outs_h)
        (d / "unit.tmpl").write_text(FC.template_of(text))
        (d / "partition.json").write_text(json.dumps(part))
        (d / "program.json").write_bytes(c)
        (d / "draw.json").write_text(json.dumps({"units": [block_no]}))

    write("honest", got, {})
    if any(v == wparam for v, _j in at.values()):
        flip_row = list(row)
        flip_row[5] ^= 1 << 7
        g2 = run_core(flip_row)
        b_port = next(p for p, (v, _j) in at.items() if v == wparam)
        honest_b = allregs[wparam].statement([at[b_port][1]])[0]
        tb = np.asarray([RV.row_words(r, allregs[wparam].word_bits, RV.SCHEMA_HM96_SHA512_ROW_V2) for r in honest_b],
                        dtype=np.uint16)
        if tb.shape != (1, len(row)) or tb[0].tolist() != row:
            raise SystemExit(f"FLIP: the table row of {b_port} ({tb.shape}) is not the weight row's {len(row)} words")
        fb = tb.copy()
        fb[0, 5] ^= 1 << 7
        write("flipped", g2, {b_port: fb})
        rec["flipped"] = {"word": 5, "bit": 7, "port": b_port, "ok": g2[-1], "y": g2[nd_:nd_ + TN]}
    else:
        rec["flipped"] = {"staged": False, "why": f"{drawn_tid} reads no weight row"}
    print("FLIPPED " + json.dumps(rec["flipped"]), flush=True)
    phase("stage")
    Path(a.out, "stage.json").write_text(json.dumps(rec, indent=1, default=str))
    print("STAGE " + json.dumps({x_: rec[x_] for x_ in ("laid", "draw", "evaluate", "flipped")}, default=str)[:4000],
          flush=True)
    return 0


def door(a) -> int:
    """`proofs.verify` on each staged proof with the verifier's own program, partition, draw and registered file."""
    import dataclasses
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    os.environ[PF.TMPL_SWITCH] = "1"
    PF.executable = lambda: Path(a.lean)
    PF.lower_executable = lambda: Path(a.lower)
    for tag in a.session:
        d = Path(a.out) / tag
        t0 = time.perf_counter()
        found, refusal = PF.verify(PF.Proof(d / "circuit.txt", d / "pub-1.bin", tuple(map(Path, a.session[tag]))),
                                   program=(d / "program.json").read_bytes(),
                                   partition=json.loads((d / "partition.json").read_text()),
                                   draw=json.loads((d / "draw.json").read_text()),
                                   registered=json.loads((d / "registered.json").read_text()))
        print("DOOR " + json.dumps({"proof": tag, "wall_s": round(time.perf_counter() - t0, 2), "refusal": refusal,
                                    "findings": [dataclasses.asdict(f) for f in found or []]}, default=str), flush=True)
    return 0


def main() -> int:
    global TN, GEMM_WORDS, K_MAX
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    st = sub.add_parser("stage")
    st.add_argument("--lower", default=None, help="flock-lower-check, Lean's lowering")
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--steer", default="PearlPlusTileCoreWired")
    st.add_argument("--until", choices=("laid", "lower"), default=None)
    st.add_argument("--tn", type=int, default=TN, help="the tile's columns, the grain's GEMM unit's (K · TN words)")
    st.add_argument("--k-max", type=int, default=K_MAX)
    st.add_argument("--core", choices=("v3", "v4"), default="v3",
                    help="v4: the tile `PearlPlusTileCoreWords_v7`, its core `PearlPlusTileCoreWired_v4` opened into TN + 1 units")
    dr = sub.add_parser("door")
    dr.add_argument("--lower", required=True)
    dr.add_argument("--lean", required=True)
    dr.add_argument("--out", type=Path, required=True)
    dr.add_argument("--session", nargs=2, action="append", metavar=("PROOF", "DIR"), required=True)
    a = ap.parse_args()
    if a.cmd == "stage":
        TN, GEMM_WORDS, K_MAX = a.tn, K * a.tn, a.k_max
    if a.cmd == "door":
        a.session = {t_: [s for u, s in a.session if u == t_] for t_, _ in a.session}
        return door(a)
    return stage(a)


if __name__ == "__main__":
    sys.exit(main())
