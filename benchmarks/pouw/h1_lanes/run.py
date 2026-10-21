"""run.py prepare|summarize: one served H1 leaf (32 × 32, at a small K: 384, the least whose combine statement has a row of
more than one SHA-512 block, which C-Flock's circuit needs) proved as its seeds, its 32 lanes and its combine
(`pouw.circuit.h1lane`), one C-Flock --zk statement each, the seed pair and the lanes' partials carried from statement to
statement as produced values (`one_stage.produced`), and the negatives of the produced route.

* **The leaf's Program** is `h1lane.program` over `h1window.h1_tile(K, (32, 32))` (H1_TILE v1 or v3), its query
  `h1lane.query`: unit 0 the seeds (`PouwUnitSeeds_v2` on the leaf's salt, position and roots), units 1 .. 32 the lanes,
  unit 33 the combine, one template each, so one statement each; every statement is bound to the leaf's partition, and the
  Lean verifier derives each instance's unit from its own copy of the Program (``--program``, ``--partition``).
* **The registered values** (`registered.Registered.commit`, ``row/v2`` rows of 16-bit leaves, under the leaf Program's
  digest): ``a`` (32 rows: int8 codes and s_a), ``b`` (32 rows: codes and s_w), ``zy`` (the leaf's one z/y record) and
  ``receipt`` (the leaf's CP-W receipt, `h1tile.reference`). STAND-IN: in a served window these are the window's A and z
  tables, the weights' b/<K> and the window's receipt tree (#3085's `h1window`); here the leaf's own registration record
  (`registration.record`, law subset of every unit) stands for the window's, and its digest D is what the produced
  values' domain binds. The roots every seed reads are (SHA-512 of root(a) under `ROOTS_TAG`, root(b)), as h1_prove's.
* **The produced values**, committed after the draw (STAND-IN: the draw of every unit, its digest `draw_digest`): the seed
  pair (``PouwUnitSeeds_v2``, one row: [seed_j, σ_R] on bits, 64 u16 words) and ``partial`` (32 rows, lane l's CP-W
  partial at row l, 2 (KS + m) u32 words), each `produced.commit`'s tree under D; their record is `produced.record`'s,
  which `produced.check` holds to the verifier's own entries (``produced_check``).
* **The statements** (`h1unit`'s forms): the seeds' input row ``pub`` (salt || position || roots, public: the verifier
  computes its ``b || c`` under `h1unit.pub_salt`) and output ``o0`` (the seed pair, opened at row 0 of its produced
  tree); lane l's inputs ``a0..a3``, ``b0..b7`` (its rows of a and b), ``zy`` (the record), ``seeds`` (row 0 of the seed
  pair) and outputs ``o0`` (its partial, opened at row l of ``partial``) and ``o1`` (ok, the known row 1 under
  `ok_salt`); the combine's inputs ``p0..p31`` (rows 0 .. 31 of ``partial``) and ``receipt`` and output ``o0`` (ok, the
  known row 1). Each statement's ``--registered`` file is the verifier's own (`registered.reads_file`,
  `produced.reads_ports`) restricted to its circuit's ports, and its public file the verifier's (`h1unit.verifier_pub`)
  from the prover's openings and the verifier's own positions and known rows.
* **The negatives**, each one statement proved and checked like the others:
  ``swap-lane3`` the partial tree with rows 3 and 4 swapped (its record well-formed): lane 3's output is row 4's, which
  does not open the root at row 3, where the verifier reads it (REJECT); ``swap-combine`` the combine on that tree, which
  opens and XORs to the receipt, since XOR is order-free: the swap is refused at the lanes (ACCEPT, by design);
  ``other-leaf`` the combine with p5 another leaf's partial, opened in that leaf's tree (its registration's domain):
  REJECT; ``other-leaf-in-tree`` the other leaf's partial committed at row 5 of this leaf's tree: it opens, and the
  combine's ok is 0 (REJECT); ``receipt-mismatch`` a registered receipt one bit off the lanes' XOR, with the produced
  values recommitted under its registration: the combine's ok is 0 (REJECT).

``prepare`` writes WORK/plan.tsv (``S tag dir`` a session; ``C tag dir pub registered expect`` a Lean check),
WORK/plan.json and WORK/prepare.json; ``summarize`` writes OUT/summary.json from the Lean verdicts, and exits 1 unless every
check's verdict is the expected one and the honest statements' reported roots are the produced record's
(`produced.opened`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "h1_prove"))

import h1unit as U  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.core.protocols.pouw.circuit import h1lane as L  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as H  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import produced as PR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as R  # noqa: E402

RC = (32, 32)
SCHEME = "frame-v3-sha512/hm96-sha512"
ROOTS_TAG = b"verity/pouw/h1-lanes/roots/v0\0"
DRAW_TAG = b"verity/pouw/h1-lanes/stand-in-draw/v0\0"
SALT_TAG = b"verity/pouw/h1-lanes/stand-in-salt/v0\0"
SEEDS, PARTIAL = L.PRODUCED_SEEDS, L.PRODUCED
REGISTERED = ("a", "b", "zy", "receipt")
#: every check's expected verdict
EXPECT = {"swap-lane3": "REJECT", "swap-combine": "ACCEPT", "other-leaf": "REJECT", "other-leaf-in-tree": "REJECT",
          "receipt-mismatch": "REJECT"}


def tile(k: int):
    return HW.h1_tile(k, RC, H.PouwH1Tile3 if U.tile() == "v3" else None)


def f32(x) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32)


@dataclass(frozen=True)
class Leaf:
    a: list
    b: list
    sa: list
    sw: list
    zy: tuple
    salt: bytes
    position: int


def leaf_data(seed: int, k: int, salt: bytes, position: int) -> Leaf:
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (RC[0], k)).astype(np.int8).view(np.uint8)
    b = rng.integers(-128, 128, (RC[1], k)).astype(np.int8).view(np.uint8)
    sa = f32(rng.uniform(1e-3, 0.1, RC[0]) / 127)
    sw = f32(rng.uniform(1e-3, 0.5, RC[1]) / 127)
    a_, b_, sa_, sw_ = a.tolist(), b.tolist(), [int(x) for x in sa], [int(x) for x in sw]
    return Leaf(a_, b_, sa_, sw_, Z.records(a_, b_, sa_, sw_), salt, position)


def code_words(codes: list, scales: list) -> np.ndarray:
    return U.row_words(np.asarray(codes, np.uint8).view(np.int8), np.asarray(scales, np.uint32).view(np.float32))


def u32_words(words) -> list[int]:
    return [int(v) for v in np.ascontiguousarray(np.asarray(words, np.uint16).astype("<u2")).view("<u4")]


def bit_words(bits: list[int]) -> list[int]:
    return np.packbits(np.asarray(bits, np.uint8), bitorder="little").view("<u2").astype(np.uint16).tolist()


def draw_digest(registration: str, population: int) -> str:
    """STAND-IN for the draw's digest: the live coin draws every unit of the leaf."""
    return hashlib.sha512(DRAW_TAG + canonical_json({"registration": registration, "units": list(range(population))})).hexdigest()


@dataclass(frozen=True)
class World:
    """One registration and the produced values after its draw: the verifier's view of a leaf, honest or not."""

    name: str
    leaf: Leaf
    roots: tuple
    regs: dict
    registration: str
    draw: str
    trees: dict
    record: dict


def register(prog: str, leaf: Leaf) -> tuple[dict, tuple, list[int]]:
    regs = {"a": G.Registered.commit(prog, "a", code_words(leaf.a, leaf.sa).tolist(), None, G.WORD_BITS, V2),
            "b": G.Registered.commit(prog, "b", code_words(leaf.b, leaf.sw).tolist(), None, G.WORD_BITS, V2),
            "zy": G.Registered.commit(prog, "zy", [U.record_words(*leaf.zy).tolist()], None, G.WORD_BITS, V2)}
    roots = (hashlib.sha512(ROOTS_TAG + regs["a"].root).digest(), regs["b"].root)
    z, receipt = H.reference(leaf.a, leaf.b, leaf.salt, leaf.position, roots, stream=U.tile() == "v3")
    assert [v & 0xFFFFFFFF for v in z] == [v & 0xFFFFFFFF for v in leaf.zy[0]], "the record's z is the tile's"
    regs["receipt"] = receipt_reg(prog, receipt)
    return regs, roots, list(receipt)


def receipt_reg(prog: str, receipt: list[int], flip: int | None = None) -> G.Registered:
    """The registered receipt, with bit ``flip`` of it flipped when given."""
    receipt = list(receipt)
    if flip is not None:
        receipt[flip // 32] ^= 1 << (flip % 32)
    return G.Registered.commit(prog, "receipt", [U.receipt_words(receipt).tolist()], None, G.WORD_BITS, V2)


def seeds_row(leaf: Leaf, roots: tuple) -> list[int]:
    bits = SD.leaves(SD.seed(leaf.salt, leaf.position, *roots), True) + SD.leaves(
        SD.receipt_seed(leaf.salt, leaf.position, *roots), True)
    return bit_words(bits)


def partials(leaf: Leaf, roots: tuple, lay) -> list[list[int]]:
    out = []
    for ln in range(lay.lanes):
        rows, cols = lay.rows(ln), lay.cols(ln)
        p, ok = L.lane_reference([leaf.a[i] for i in rows], [leaf.b[j] for j in cols], [leaf.sa[i] for i in rows],
                                 [leaf.sw[j] for j in cols], leaf.zy, leaf.salt, leaf.position, roots, lay, ln)
        assert ok == 1
        out.append(U.receipt_words(p).tolist())
    return out


def world(name: str, ctx, leaf: Leaf, regs: dict, roots: tuple, partial_rows: list) -> World:
    rec = R.record(ctx.program, ctx.query, {"law": "subset", "k": ctx.population}, SCHEME, [],
                   {"kind": "stand-in", "harness": "h1_lanes", "leaf": leaf.position}, partition=ctx.part,
                   registered=[regs[v].entry() for v in REGISTERED])
    d = R.digest(rec)
    draw = draw_digest(d, ctx.population)
    trees = {SEEDS: PR.commit(ctx.prog, SEEDS, [seeds_row(leaf, roots)], d, schema=V2),
             PARTIAL: PR.commit(ctx.prog, PARTIAL, partial_rows, d, schema=V2)}
    return World(name, leaf, roots, regs, d, draw, trees, PR.record(d, draw, [trees[SEEDS].entry(), trees[PARTIAL].entry()]))


# -- the verifier's reads ---------------------------------------------------------------------------------------------------

def reads_template(ctx, w: World) -> dict[str, dict]:
    """The verifier's ``--registered`` ports of the whole leaf (`registered.reads_file`, `produced.reads_ports`), split by
    statement kind: ``seeds``, ``lane`` and ``combine``, each only its circuit's ports."""
    vals = {"a": ctx.k // 2 + 2, "b": ctx.k // 2 + 2, "zy": 3 * RC[0] * RC[1], "receipt": ctx.wp}
    ports = {**{f"a[{i}]": f"a{i}" for i in range(L.LANE_R)}, **{f"b[{j}]": f"b{j}" for j in range(L.LANE_C)}}
    g = G.reads_file(ctx.program, ctx.query, vals, {v: w.regs[v].root.hex() for v in vals}, ports=ports,
                     schemas={v: V2 for v in vals})
    s = PR.reads_ports(ctx.program, ctx.query, {SEEDS: 64}, {SEEDS: w.trees[SEEDS].root.hex()}, w.registration,
                       ports={SEEDS: "o0"}, schemas={SEEDS: V2})
    p = PR.reads_ports(ctx.program, ctx.query, {PARTIAL: ctx.wp}, {PARTIAL: w.trees[PARTIAL].root.hex()}, w.registration,
                       ports={PARTIAL: "o0", **{f"p[{ln}]": L.partial_port(ln) for ln in range(ctx.lanes)}},
                       schemas={PARTIAL: V2})
    gp = g["ports"]
    lane = {**{f"a{i}": gp[f"a{i}"] for i in range(L.LANE_R)}, **{f"b{j}": gp[f"b{j}"] for j in range(L.LANE_C)},
            "zy": gp["zy"], "seeds": s["seeds"], "o0": p["o0"]}
    combine = {**{L.partial_port(ln): p[L.partial_port(ln)] for ln in range(ctx.lanes)}, "receipt": gp["receipt"]}
    return {k: {"format": g["format"], "ports": v} for k, v in (("seeds", {"o0": s["o0"]}), ("lane", lane),
                                                                 ("combine", combine))}


def reads_of(ctx, w: World) -> dict[str, dict]:
    """`reads_template`'s files for world ``w``: the template's ports with ``w``'s roots and domains."""
    own = {v: (t.root.hex(), t.entry()["domain"]) for v, t in {**w.regs, **w.trees}.items()}
    return {k: {**f, "ports": {p: {**e, "root": own[e["value"]][0], "domain": own[e["value"]][1]}
                               for p, e in f["ports"].items()}} for k, f in ctx.reads_t.items()}


# -- a statement ------------------------------------------------------------------------------------------------------------

def opening(tree: G.Registered, position: int, salt_row: int | None = None) -> dict:
    """What the prover opens for row ``position`` of ``tree``: the row it holds there and its salt and ``b || c`` (those of
    row ``salt_row`` when it opens another row in that row's place), and the path at ``position``."""
    r = position if salt_row is None else salt_row
    return {"value": tree.value, "row": list(tree.rows[r]), "salt": tree.salts[r], "bc": tree.bcs[r], "position": position,
            "path": tree.path(position).hex()}


def known(words, salt: bytes) -> dict:
    return {"row": [int(x) for x in words], "salt": salt}


def stager(ctx, kind: str, ln: int | None):
    from verity_flock import class_statement as CS

    bindp = {"partition": P.digest(ctx.part), "program": ctx.part["program"]}
    circuit = ctx.work / "circuits" / (kind if ln is None else f"lane{ln}") / "circuit.txt"
    if kind == "seeds":
        return CS.UnitStager(bind(SD.PouwUnitSeedsBits), "unit/pouw-h1-seeds", [("pub", V2, U.PUB_BITS, list(U.PUBLIC))],
                             [(V2, (0,))], circuit=circuit, bind=bindp)
    if kind == "lane":
        ab = 16 * (ctx.k // 2 + 2)
        rows = ([(f"a{i}", V2, ab, [f"a[{i}]"]) for i in range(L.LANE_R)] + [(f"b{j}", V2, ab, [f"b[{j}]"]) for j in range(L.LANE_C)]
                + [("zy", V2, 48 * RC[0] * RC[1], ["zy"]), ("seeds", V2, 1024, ["seeds"])])
        return CS.UnitStager(L.lane(ctx.tile, ln), f"unit/pouw-h1-lane{ln}", rows, [(V2, (0,)), (V2, (1,))], circuit=circuit,
                             bind=bindp)
    bits = 16 * ctx.wp
    rows = [(L.partial_port(i), V2, bits, [f"p[{i}]"]) for i in range(ctx.lanes)] + [("receipt", V2, bits, ["receipt"])]
    return CS.UnitStager(L.combine(ctx.lay), "unit/pouw-h1-combine", rows, [(V2, (0,))], circuit=circuit, bind=bindp)


def expected(ctx, kind: str, ln: int | None, w: World, ins: dict) -> dict[str, list[int]]:
    """The statement's outputs on its opened inputs, on words: the seed pair; lane ``ln``'s (partial, ok)
    (`h1lane.lane_reference`); the combine's ok (`h1lane.combine_reference`)."""
    lf = w.leaf
    if kind == "seeds":
        return {"o0": seeds_row(lf, w.roots)}
    if kind == "lane":
        rows, cols = ctx.lay.rows(ln), ctx.lay.cols(ln)
        p, ok = L.lane_reference([lf.a[i] for i in rows], [lf.b[j] for j in cols], [lf.sa[i] for i in rows],
                                 [lf.sw[j] for j in cols], lf.zy, lf.salt, lf.position, w.roots, ctx.lay, ln)
        return {"o0": U.receipt_words(p).tolist(), "o1": [ok]}
    ps = [u32_words(ins[L.partial_port(i)]["row"]) for i in range(ctx.lanes)]
    return {"o0": [L.combine_reference(ps, u32_words(ins["receipt"]["row"]))]}


def statement(ctx, st, kind: str, ln: int | None, tag: str, w: World, ins: dict, outs: dict) -> dict:
    """Stage one statement on ``st`` and write the verifier's files: ``ins`` and ``outs`` each port's opening (`opening`)
    or known row (`known`); the verifier reads ``w``'s registration and produced roots."""
    unit = 0 if kind == "seeds" else (1 + ln if kind == "lane" else 1 + ctx.lanes)
    reads = reads_of(ctx, w)["seeds" if kind == "seeds" else kind]
    want = expected(ctx, kind, ln, w, ins)

    ir = type(st).reference_check.__get__(st)

    def check(rows_, got):
        ir(rows_, got)
        for p, v in want.items():
            if not np.array_equal(np.asarray(got[p][0], np.uint16), np.asarray(v, np.uint16)):
                raise AssertionError(f"{tag}: the lowered circuit's {p} is not the word reference's")

    st.reference_check = check
    rows = {p: np.asarray([o["row"]], np.uint16) for p, o in ins.items()}
    reg = {p: {"value": o["value"], "positions": [o["position"]], "paths": [o["path"]]}
           for p, o in {**ins, **outs}.items() if "value" in o}
    t0 = time.time()
    d = ctx.work / "st" / tag
    got = st.stage(rows, d, os.urandom(32), salts={p: [o["salt"]] for p, o in {**ins, **outs}.items()}, indices=[unit],
                   set_name=f"h1-lanes-{tag}", tables=rows, refs={p: [0] for p in rows}, registered=reg)
    (d / "circuit.txt").unlink(missing_ok=True)
    (d / "circuit.txt").symlink_to(st.circuit)
    (d / "pin.txt").write_text(st.pin)
    ports = {p.name: p for p in st.ports}
    committed = {p: U.commit_strings(ports[p], np.asarray(got["outs"][p]), [o["salt"]])[0] for p, o in outs.items()}
    committed.update({p: U.commit_strings(ports[p], rows[p], [o["salt"]])[0] for p, o in ins.items()})
    mine = {}
    for p, o in {**ins, **outs}.items():
        mine[p] = o["bc"] if "bc" in o else U.commit_strings(ports[p], np.asarray([o["row"]], np.uint16), [o["salt"]])[0]
    vreads = [{p: (o["value"], reads["ports"][p]["reads"][unit], o["path"]) for p, o in {**ins, **outs}.items() if "value" in o}]
    if set(vreads[0]) != set(reads["ports"]):
        raise AssertionError(f"{tag}: the opened ports {sorted(vreads[0])} are not the verifier's {sorted(reads['ports'])}")
    pub = U.verifier_pub({"meta": st.meta, "pin": st.pin, "class": st.class_sha512}, f"h1-lanes-{tag}", st.sub.id, [mine],
                         [unit], vreads)
    vd = ctx.work / "verifier"
    (vd / f"{tag}.pub.bin").write_bytes(pub)
    (vd / f"{tag}.registered.json").write_text(json.dumps(reads))
    return {"tag": tag, "kind": kind, "lane": ln, "unit": unit, "dir": str(d), "pub": str(vd / f"{tag}.pub.bin"),
            "registered": str(vd / f"{tag}.registered.json"), "ands": got["ands"], "circuit_sha512": got["circuit_sha512"],
            "outs": {p: np.asarray(v[0]).tolist() for p, v in got["outs"].items()},
            "binding": {p: committed[p] == mine[p] for p in mine}, "seconds": round(time.time() - t0, 1)}


# -- the plan ---------------------------------------------------------------------------------------------------------------

CTX = None


def lane_ins(ctx, w: World, ln: int, seeds_tree=None) -> dict:
    t = w.regs
    ins = {f"a{i}": opening(t["a"], r) for i, r in enumerate(ctx.lay.rows(ln))}
    ins.update({f"b{j}": opening(t["b"], c) for j, c in enumerate(ctx.lay.cols(ln))})
    ins.update({"zy": opening(t["zy"], 0), "seeds": opening(seeds_tree or w.trees[SEEDS], 0)})
    return ins


def ok_out(ctx, unit: int) -> dict:
    return known(U.OK, U.ok_salt(f"{ctx.name}#{unit}", ctx.position))


def circuit_jobs(name: str) -> list[dict]:
    """Every statement on circuit ``name`` (``seeds``, ``lane<l>``, ``combine`` or ``combine-k4096``, the last only built
    and counted)."""
    ctx = CTX
    A = ctx.worlds["honest"]
    if name == "combine-k4096":
        from verity_flock import class_statement as CS

        lay = H.Layout(RC[0], RC[1], 4096)
        bits = 16 * 4 * lay.sets
        st = CS.UnitStager(L.combine(lay), "unit/pouw-h1-combine-k4096",
                           [(L.partial_port(i), V2, bits, [f"p[{i}]"]) for i in range(lay.lanes)]
                           + [("receipt", V2, bits, ["receipt"])], [(V2, (0,))])
        return [{"tag": name, "ands": st.low.layout.counts["ands"], "W": 2 * lay.sets}]
    kind = "lane" if name.startswith("lane") else name
    ln = int(name[4:]) if kind == "lane" else None
    st = stager(ctx, kind, ln)
    out = []
    if kind == "seeds":
        lf = A.leaf
        pub = known(U.pub_words(lf.salt, lf.position, A.roots), U.pub_salt(ctx.name, lf.position))
        out.append(statement(ctx, st, kind, None, "seeds", A, {"pub": pub}, {"o0": opening(A.trees[SEEDS], 0)}))
    elif kind == "lane":
        unit = 1 + ln
        out.append(statement(ctx, st, kind, ln, f"lane{ln}", A, lane_ins(ctx, A, ln),
                             {"o0": opening(A.trees[PARTIAL], ln), "o1": ok_out(ctx, unit)}))
        if ln == 3:
            S = ctx.worlds["swap"]
            out.append(statement(ctx, st, kind, ln, "swap-lane3", S, lane_ins(ctx, S, ln),
                                 {"o0": opening(S.trees[PARTIAL], 3, salt_row=4), "o1": ok_out(ctx, unit)}))
    else:
        unit = 1 + ctx.lanes

        def ins(w: World, p5=None) -> dict:
            got = {L.partial_port(i): opening(w.trees[PARTIAL], i) for i in range(ctx.lanes)}
            if p5 is not None:
                got[L.partial_port(5)] = p5
            return {**got, "receipt": opening(w.regs["receipt"], 0)}

        B = ctx.worlds["other"]
        for tag, w, p5 in (("combine", A, None), ("swap-combine", ctx.worlds["swap"], None),
                           ("other-leaf", A, opening(B.trees[PARTIAL], 5)), ("other-leaf-in-tree", ctx.worlds["mixed"], None),
                           ("receipt-mismatch", ctx.worlds["receipt"], None)):
            out.append(statement(ctx, st, kind, None, tag, w, ins(w, p5), {"o0": ok_out(ctx, unit)}))
    return out


@dataclass
class Ctx:
    work: Path
    name: str
    k: int
    position: int
    tile: object = None
    lay: object = None
    lanes: int = 0
    wp: int = 0
    program: object = None
    query: dict | None = None
    part: dict | None = None
    prog: str = ""
    population: int = 0
    worlds: dict | None = None
    reads_t: dict | None = None


def prepare(args) -> None:
    global CTX
    t0 = time.time()
    work = Path(args.work)
    for d in ("verifier", "st", "circuits"):
        (work / d).mkdir(parents=True, exist_ok=True)
    ctx = Ctx(work, args.name, args.k, args.position)
    ctx.tile = tile(args.k)
    ctx.lay = L.layout(ctx.tile)
    ctx.lanes, ctx.wp = ctx.lay.lanes, 4 * ctx.lay.sets
    if 2 * ctx.wp + 17 <= 128:
        raise SystemExit(f"K={args.k}: a partial row of {16 * ctx.wp} bits is one SHA-512 block, and every row of the combine's "
                         f"statement is one, but C-Flock needs a row with a compression before its last block: K >= 384")
    ctx.program, ctx.query = L.program(ctx.tile), L.query(ctx.tile)
    ctx.part = P.partition(ctx.program, ctx.query)
    ctx.prog, ctx.population = P.program_digest(ctx.program), P.units(ctx.program, ctx.query)
    (work / "verifier" / "program.json").write_bytes(canonical_json(encode_program(ctx.program)))
    (work / "verifier" / "partition.json").write_text(json.dumps(ctx.part))
    print(f"program: {ctx.population} units, digest {ctx.prog[:16]}, {time.time() - t0:.0f} s", flush=True)

    salt = hashlib.sha512(SALT_TAG + args.name.encode()).digest()
    la, lb = leaf_data(args.seed, args.k, salt, args.position), leaf_data(args.seed + 1, args.k, salt, args.position + 1)
    regs, roots, receipt = register(ctx.prog, la)
    pa = partials(la, roots, ctx.lay)
    A = world("honest", ctx, la, regs, roots, pa)
    regs_b, roots_b, _ = register(ctx.prog, lb)
    B = world("other", ctx, lb, regs_b, roots_b, partials(lb, roots_b, ctx.lay))
    swapped = list(pa)
    swapped[3], swapped[4] = pa[4], pa[3]
    mixed = list(pa)
    mixed[5] = list(B.trees[PARTIAL].rows[5])
    ctx.worlds = {"honest": A, "other": B,
                  "swap": replace(world("swap", ctx, la, regs, roots, swapped), name="swap"),
                  "mixed": world("mixed", ctx, la, regs, roots, mixed),
                  "receipt": world("receipt", ctx, la, {**regs, "receipt": receipt_reg(ctx.prog, receipt, flip=3 * 32 + 17)}, roots, pa)}
    for nm in ("swap", "mixed"):   # the honest registration: only the produced trees differ
        w = ctx.worlds[nm]
        assert w.registration == A.registration
    mine = PR.entries(ctx.program, ctx.query, {SEEDS: 64, PARTIAL: ctx.wp}, A.registration, {SEEDS: V2, PARTIAL: V2})
    produced_check = {nm: list(PR.check(w.record, A.registration, A.draw, mine)) for nm, w in ctx.worlds.items()}
    ctx.reads_t = reads_template(ctx, A)
    print(f"registered and produced: {time.time() - t0:.0f} s; produced records' codes against the honest registration: "
          f"{produced_check}", flush=True)
    CTX = ctx

    names = ["seeds", "combine", "combine-k4096"] + [f"lane{ln}" for ln in range(ctx.lanes)]
    with mp.get_context("fork").Pool(args.jobs) as pool:
        recs = [r for rs in pool.imap_unordered(circuit_jobs, names) for r in rs]
    by = {r["tag"]: r for r in recs}
    k4096 = by.pop("combine-k4096")
    honest = ["seeds"] + [f"lane{ln}" for ln in range(ctx.lanes)] + ["combine"]
    checks = honest + list(EXPECT)
    with open(work / "plan.tsv", "w") as f:
        for t in checks:
            f.write(f"S\t{t}\t{by[t]['dir']}\n")
        for t in checks:
            f.write(f"C\t{t}\t{by[t]['dir']}\t{by[t]['pub']}\t{by[t]['registered']}\t{EXPECT.get(t, 'ACCEPT')}\n")
    plan = {"checks": {t: {**by[t], "expect": EXPECT.get(t, "ACCEPT")} for t in checks}, "honest": honest}
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    lane_ands = sorted({by[f"lane{ln}"]["ands"] for ln in range(ctx.lanes)})
    summary = {"k": args.k, "tile": U.tile(), "units": ctx.population, "program_sha512": ctx.prog,
               "partition_sha512": P.digest(ctx.part), "registration": A.registration, "draw": A.draw,
               "produced": A.record, "produced_check": produced_check,
               "ands": {"seeds": by["seeds"]["ands"], "lane": lane_ands, "combine": by["combine"]["ands"],
                        "combine_k4096": k4096["ands"], "combine_k4096_W": k4096["W"]},
               "binding": {t: by[t]["binding"] for t in checks},
               "outs": {t: by[t]["outs"] for t in checks},
               "seconds": round(time.time() - t0, 1)}
    (work / "prepare.json").write_text(json.dumps(summary, indent=1))
    print("PREPARED " + json.dumps({x: summary[x] for x in ("k", "tile", "units", "ands", "produced_check", "seconds")}),
          flush=True)


def _verdict(p: Path) -> dict | None:
    if not p.exists():
        return None
    v = [json.loads(x[8:]) for x in p.read_text(errors="replace").splitlines() if x.startswith("VERDICT ")]
    if not v:
        return None
    return {**v[0], "accepted": all(x.get("accepted") for x in v), "sessions": len(v),
            "why": next((x.get("why", "") for x in v if not x.get("accepted")), "")}


def _time(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def summarize(args) -> int:
    work, out = Path(args.work), Path(args.out)
    plan, prep = json.loads((work / "plan.json").read_text()), json.loads((work / "prepare.json").read_text())
    rows, opened = {}, {}
    for tag, c in plan["checks"].items():
        v = _verdict(out / "lean" / f"{tag}.out")
        err = out / "lean" / f"{tag}.err"
        word = "NONE" if v is None else ("ACCEPT" if v["accepted"] else "REJECT")
        rows[tag] = {"expect": c["expect"], "lean": word, "as_expected": word == c["expect"],
                     "why": (v or {}).get("why") or (err.read_text(errors="replace").strip().splitlines()[-1:] if err.exists() else ""),
                     "ands": c["ands"], "prove": _time(out / tag / "prove.time"), "lean_time": _time(out / "lean" / f"{tag}.time"),
                     "registered": (v or {}).get("registered")}
        if tag in plan["honest"] and v and v["accepted"]:
            for e in (v.get("registered") or {}).values():
                if opened.setdefault(e["value"], e["root"]) != e["root"]:
                    opened[e["value"]] = None
    honest_record = prep["produced"]
    opened_codes = list(PR.opened(honest_record, opened))
    ok = all(r["as_expected"] for r in rows.values()) and not opened_codes and not prep["produced_check"]["honest"]
    summary = {"ok": ok, "k": prep["k"], "tile": prep["tile"], "units": prep["units"], "ands": prep["ands"],
               "produced_check": prep["produced_check"], "opened": opened, "opened_codes": opened_codes,
               "checks": rows, "registration": prep["registration"], "draw": prep["draw"], "produced": honest_record,
               "binding": prep["binding"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for tag, r in rows.items():
        print(f"{tag:20s} {r['lean']:7s} expect {r['expect']:7s} {'ok' if r['as_expected'] else 'MISMATCH'}  "
              f"{str(r['why'])[:160]}")
    print(f"opened roots against the produced record: {opened_codes or 'ok'}; ANDs {prep['ands']}; all as expected: {ok}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "summarize"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=384)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--position", type=int, default=5)
    ap.add_argument("--name", default="h1-lanes-local")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a)
    else:
        sys.exit(summarize(a))


if __name__ == "__main__":
    main()
