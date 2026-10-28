"""j10r.py prepare|summarize: `run.py`'s protocol for a drawn J10-R leaf: one leaf of a one-call window over J10-R's
`PouwH1Tile_v5` (16 × 16 by default, ``H1L_RC``) proved as its statements as `pouw.leaf_lanes`' J10-R leaf stages them
(``j10r``): each lane in `weights.chunks`(K) chunks (`PouwH1LaneZyChunk_v2`, the whole lane `PouwH1LaneZy_v2` at
n = 1), then `PouwH1Combine_v2`, which folds every lane's and chunk's ok row into the leaf's one ok. Each statement is
`h1window.leaf_program`'s at its K and chunk, staged on `h1zy.statement_rows` and checked by the Lean verifier against
the verifier's own `--registered` file (`leaf_lanes.registered_file` at `leaf_lanes.verifier_entries` of every
after-draw value, `leaf_lanes.produced_values`).

* **The registered values**, as `run.py`'s: ``a`` and ``b`` (each row its `h1zy.chunk_leaves`: n leaves a row, row i's
  chunk c at n i + c), ``zy`` and ``receipt`` (`h1zy.reference` over v5 at the unit's ``call`` and ``blocks``,
  `h1window.j10r_inputs`, which is the lanes' partials' XOR).
* **The after-draw values** (`one_stage.produced`, under the registration's digest): ``partials`` (row l lane l's
  `h1zy.lane_partial`), each ``carry<c>`` (row l what lane l's chunk c hands chunk c + 1, `h1zy.lane_chunk_carries`)
  and ``oks`` (row i statement i's ok word, 1), the row every lane and chunk statement's output ``o0`` opens: so those
  oks are in the trace, and only the combine's ``o0`` is a known row (the one 1 of `h1unit.ok_salt`).
* **The negatives**: ``lane1-chunk0-carry-swapped`` and ``lane1-chunk1-carry-swapped``: a ``carry0`` tree with rows 1
  and 2 swapped, so lane 1's chunk 0 computes a carry its ``carry_out`` row isn't and its chunk 1 continues from lane
  2's: each ok is 0 against its ``oks`` row's 1 (REJECT). ``combine-ok-row-zero``: an ``oks`` tree whose row of lane
  1's last chunk is 0, so the combine's ok is 0 against its known 1 (REJECT); ``lane1-last-ok-row-zero``, that chunk on
  that tree, its ok 1 against the row's 0 (REJECT). At n = 1 only the ok-row ones.

`pouw-leaf-own-window` (`PouwCheck.LeafOwn`) states v1's statements only, so this runner has no leaf-own check;
`summarize` exits 1 unless every check's verdict is the expected one, the honest statements' reported roots are the
produced record's (`produced.opened`) and their opened rows its openings (`produced.shared`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run as RN  # noqa: E402
from run import U  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.core.protocols.pouw import leaf_lanes as LL  # noqa: E402
from verity.core.protocols.pouw import weights as W  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as H  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import produced as PR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as R  # noqa: E402

RC = tuple(int(x) for x in os.environ.get("H1L_RC", "16x16").split("x"))
RN.RC = RC
REGISTERED = RN.REGISTERED
STEPS = RN.STEPS
#: the lane and chunk the negatives are of, and the lane its carry is swapped with
LN, SWAP = 1, 2


def expects(n: int) -> dict[str, str]:
    """Every negative check's expected verdict at n chunks a lane."""
    ok = {"combine-ok-row-zero": "REJECT", "lane1-last-ok-row-zero": "REJECT"}
    return ok if n == 1 else {"lane1-chunk0-carry-swapped": "REJECT", "lane1-chunk1-carry-swapped": "REJECT", **ok}


def chunk_rows(words: np.ndarray, n: int) -> list[list[int]]:
    """Whole rows' u16 words (`h1unit.row_words`: K / 2 code words, then the scale's 2) as `h1zy.chunk_leaves`' leaves:
    row i's chunk c at n i + c, its K / 2n code words then the row's scale."""
    kc = (words.shape[1] - 2) // n
    return [[int(x) for x in (*w[c * kc:(c + 1) * kc], *w[-2:])] for w in words for c in range(n)]


@dataclass
class World:
    """One registration and the after-draw values over it: the verifier's view of the leaf, honest or not."""

    name: str
    leaf: RN.Leaf
    roots: tuple
    regs: dict
    registration: str
    draw: str
    trees: dict
    record: dict
    reg_record: dict
    partials: list
    carries: list = field(default_factory=list)
    oks: list = field(default_factory=list)


@dataclass
class Ctx:
    work: Path
    name: str
    k: int
    n: int
    position: int
    win: object = None
    j10r: tuple = ()
    bound: dict | None = None
    reg: RN.Bound | None = None
    prog: str = ""
    stagers: dict | None = None
    worlds: dict | None = None
    place: dict | None = None
    values: dict | None = None


CTX: Ctx | None = None


def statement_key(ctx: Ctx, ln: int | None, ch: int | None) -> tuple:
    """The statement as `leaf_lanes` keys it: the combine, a whole lane (n = 1), or a lane's chunk."""
    if ln is None:
        return (LL.COMBINE, None)
    return (LL.LANE, ln) if ctx.n == 1 else (LL.CHUNK, ln, ch)


def derived(ctx: Ctx, kind: str, ch: int | None) -> RN.Bound:
    """``kind``'s statement (a chunk's ``ch``) as a served window's verifier derives it (`h1window.leaf_program`)."""
    program, query = HW.leaf_program(ctx.win, ctx.k, kind, ch)
    part = P.partition(program, query)
    if P.units(program, query) != 1:
        raise AssertionError(f"{kind} {ch}: the statement's Program is one unit")
    return RN.Bound(HW.leaf_unit(ctx.win, ctx.k, kind, ch), program, query, part)


def circuit_name(kind: str, ch: int | None) -> str:
    return kind if ch is None else f"{kind}{ch}"


def register(ctx: Ctx, leaf: RN.Leaf) -> tuple[dict, tuple, list[int]]:
    """The leaf's registered values in their window tables (``ctx.place``), the roots its seeds read and its receipt."""
    def commit(v, rows):
        return G.Registered.commit(ctx.prog, ctx.place[v][0], rows, None, G.WORD_BITS, V2)

    regs = {LL.A: commit(LL.A, chunk_rows(RN.code_words(leaf.a, leaf.sa), ctx.n)),
            LL.B: commit(LL.B, chunk_rows(RN.code_words(leaf.b, leaf.sw), ctx.n)),
            LL.ZY: commit(LL.ZY, [U.record_words(*leaf.zy).tolist()])}
    roots = (hashlib.sha512(RN.ROOTS_TAG + regs[LL.A].root).digest(), regs[LL.B].root)
    receipt, ok = Z.reference(leaf.a, leaf.b, leaf.sa, leaf.sw, leaf.zy, leaf.salt, leaf.position, roots, j10r=ctx.j10r)
    assert ok == 1, "the leaf's record is its tile's"
    regs[LL.RECEIPT] = RN.receipt_reg(ctx.prog, receipt, ctx.place)
    return regs, roots, list(receipt)


def after_draw(ctx: Ctx, leaf: RN.Leaf, roots: tuple) -> tuple[list, list]:
    """The leaf's honest partials (u32 words a lane) and carries (``carries[c][l]``, lane l's chunk c's carry out)."""
    partials, carries = [], [[] for _ in range(ctx.n - 1)]
    for ln in range(LL.lanes(*RC)):
        a, b, sa, sw = RN.lane_rows(leaf, ln)
        partials.append(Z.lane_partial(a, b, leaf.admit, leaf.salt, leaf.position, roots, ln, RC, ctx.j10r))
        chain = Z.lane_chunk_carries(a, b, sa, sw, leaf.salt, leaf.position, roots, ln, RC, ctx.n, ctx.j10r)
        assert ctx.n == 1 or [v & 0xFFFFFFFF for v in chain[-1]["words"]] == partials[-1], "the chain is the lane's"
        for c in range(ctx.n - 1):
            carries[c].append(chain[c])
    return partials, carries


def world(ctx: Ctx, name: str, leaf: RN.Leaf, regs: dict, roots: tuple, partials: list, carries: list,
          oks: list) -> World:
    rec = R.record(ctx.reg.program, ctx.reg.query, {"law": "subset", "k": 1}, RN.SCHEME, [],
                   {"kind": "stand-in", "harness": "h1_lanes/j10r", "leaf": leaf.position}, partition=ctx.reg.part,
                   registered=[regs[v].entry() for v in REGISTERED])
    d = R.digest(rec)
    draw = RN.draw_digest(d, 1)
    rows = {LL.PARTIALS: [U.receipt_words(p).tolist() for p in partials],
            **{LL.carry(c): [U.receipt_words(Z.carry_words(x)).tolist() for x in carries[c]] for c in range(ctx.n - 1)},
            LL.OKS: [[int(o)] for o in oks]}
    for v, (n_rows, words) in ctx.values.items():
        assert len(rows[v]) == n_rows and all(len(r) == words for r in rows[v]), f"{v}: {n_rows} rows of {words} words"
    trees = {v: PR.commit(ctx.prog, v, rs, d, schema=V2) for v, rs in rows.items()}
    ops = {**{v: PR.rows_of(t) for v, t in trees.items()}, **{v: PR.rows_of(regs[v]) for v in REGISTERED}}
    return World(name, leaf, roots, regs, d, draw, trees, PR.record(d, draw, [t.entry() for t in trees.values()], ops),
                 rec, partials, carries, oks)


def known_rows(ctx: Ctx, w: World, kind: str, ln: int | None, ch: int | None, index: int) -> dict:
    """The statement's known ports (`h1window.leaf_known_rows`, unit 0's call and blocks), each ``(value, row, salt)``:
    lane l's ``pub`` in every chunk, ``admit`` in its last, and the combine's ``o0``."""
    lf = w.leaf
    rows = HW.leaf_known_rows(ctx.win, kind, ln, ch, salt=lf.salt, position=lf.position, roots=w.roots, admit=lf.admit,
                              u=0)
    salts = {LL.OK: U.ok_salt(f"{ctx.name}#{index}", ctx.position), LL.PUB: U.pub_salt(f"{ctx.name}#lane{ln}", lf.position),
             LL.ADMIT: U.pub_salt(f"{ctx.name}#admit", lf.position)}
    return {p: (v, row, salts[p]) for p, (v, row) in rows.items()}


def expected(ctx: Ctx, w: World, kind: str, ln: int | None, ch: int | None) -> list[int]:
    """The statement's ok on the world's rows, on words (`h1zy.lane_reference`, `lane_chunk_reference`,
    `combine_reference` with the ok rows)."""
    lf = w.leaf
    if kind == LL.COMBINE:
        return [Z.combine_reference(w.partials, RN.u32_words(w.regs[LL.RECEIPT].rows[0]), w.oks)]
    a, b, sa, sw = RN.lane_rows(lf, ln)
    if ctx.n == 1:
        return [Z.lane_reference(a, b, sa, sw, lf.admit, lf.zy, w.partials[ln], lf.salt, lf.position, w.roots, ln, RC,
                                 ctx.j10r)]
    kc, last = ctx.k // ctx.n, ch == ctx.n - 1
    cut = slice(ch * kc, ch * kc + kc)
    return [Z.lane_chunk_reference([r[cut] for r in a], [r[cut] for r in b], sa, sw, lf.admit if last else None,
                                   lf.zy if last else None, w.partials[ln] if last else None,
                                   w.carries[ch - 1][ln] if ch else None, None if last else w.carries[ch][ln], lf.salt,
                                   lf.position, w.roots, ln, RC, ctx.n, ch, ctx.j10r)]


def stager(ctx: Ctx, kind: str, ch: int | None):
    """The statement's stager on `h1zy.statement_rows` (each input port a ``row/v2`` row of its bits, `statement_ports`),
    its output ``o0`` one ``row/v2`` row."""
    b = ctx.bound[(kind, ch)]
    bits = {p: sum(x for _f, _at, x in fs) for p, fs in Z.statement_ports(b.defn)}
    rows = [(p, V2, bits[p], ps) for p, ps in Z.statement_rows(b.defn)]
    name = circuit_name(kind, ch)
    circuit = ctx.work / "circuits" / name / "circuit.txt"
    return U.stager(b.defn, f"unit/pouw-h1-{name}", rows, [(V2, (0,))], U.layout(), circuit=circuit, bind=b.bind)


def statement(ctx: Ctx, tag: str, w: World, ln: int | None, ch: int | None) -> dict:
    """Stage one statement and write the verifier's files: every read port opened at its row of the world's registered
    or after-draw value (`leaf_lanes.reads`, ``o0`` of a lane or chunk the statement's ``oks`` row), every known port at
    the verifier's own row (`known_rows`, `produced.known_port`); the verifier's file is `leaf_lanes.registered_file` at
    the world's records' entries and roots alone. The lowered ok is held to the word reference (`expected`)."""
    key = statement_key(ctx, ln, ch)
    kind, chunk = key[0], (key[2] if len(key) > 2 else None)
    st = ctx.stagers[(kind, chunk)]
    index = LL.statements(*RC, j10r=ctx.k).index(key)
    ent, roots = LL.verifier_entries(w.reg_record, w.record, LL.lanes(*RC), 2 * LL.words(ctx.k), values=ctx.values)
    kr = known_rows(ctx, w, kind, ln, chunk, index)
    vk = {p: PR.known_port(ctx.prog, v, row, salt, w.registration, schema=V2) for p, (v, row, salt) in kr.items()}
    reads_of = LL.reads(*RC, kind, ln, chunk, j10r=ctx.k)
    reads = LL.registered_file(reads_of, ent, roots, {v: V2 for v in ent}, {p: kp for p, (kp, _) in vk.items()},
                               place=ctx.place)
    opened = {p: RN.opening(w.regs[v] if v in w.regs else w.trees[v], row) for p, (v, row) in reads_of.items()}
    known = {p: RN.known_opening(ctx, w, v, row, salt) for p, (v, row, salt) in kr.items()}
    ins = {p: o for p, o in {**opened, **known}.items() if p != LL.OK}
    outs = {p: o for p, o in {**opened, **known}.items() if p == LL.OK}
    want = expected(ctx, w, kind, ln, chunk)

    def check(_rows, got):
        if not np.array_equal(np.asarray(got["o0"][0], np.uint16), np.asarray(want, np.uint16)):
            raise AssertionError(f"{tag}: the lowered circuit's ok is not the word reference's {want}")

    st.reference_check = check
    rows = {p: np.asarray([o["row"]], np.uint16) for p, o in ins.items()}
    reg = {p: {"value": o["value"], "positions": [o["position"]], "paths": [o["path"]]} for p, o in {**ins, **outs}.items()}
    t0 = time.time()
    d = ctx.work / "st" / tag
    got = st.stage(rows, d, os.urandom(32), salts={p: [o["salt"]] for p, o in {**ins, **outs}.items()}, indices=[0],
                   set_name=f"h1-lanes-{tag}", tables=rows, refs={p: [0] for p in rows}, registered=reg)
    (d / "circuit.txt").unlink(missing_ok=True)
    (d / "circuit.txt").symlink_to(st.circuit)
    (d / "pin.txt").write_text(st.pin)
    ports = {p.name: p for p in st.ports}
    committed = {p: U.commit_strings(ports[p], rows[p], [o["salt"]])[0] for p, o in ins.items()}
    mine = {p: o["bc"] for p, o in {**ins, **outs}.items()}
    vreads = [{p: (o["value"], reads["ports"][p]["reads"][0], o["path"]) for p, o in {**ins, **outs}.items()}]
    if set(vreads[0]) != set(reads["ports"]) or set(reads["ports"]) != LL.own(*RC, kind, ln, chunk, j10r=ctx.k):
        raise AssertionError(f"{tag}: the opened ports {sorted(vreads[0])} are not the verifier's {sorted(reads['ports'])}")
    pub = U.verifier_pub({"meta": st.meta, "pin": st.pin, "class": st.class_sha512}, f"h1-lanes-{tag}", st.sub.id, [mine],
                         [0], vreads)
    vd = ctx.work / "verifier"
    (vd / f"{tag}.pub.bin").write_bytes(pub)
    (vd / f"{tag}.registered.json").write_text(json.dumps(reads))
    name = circuit_name(kind, chunk)
    return {"tag": tag, "kind": kind, "lane": ln, "chunk": chunk, "index": index, "world": w.name,
            "stmt": "combine" if ln is None else str(ln), "dir": str(d), "pub": str(vd / f"{tag}.pub.bin"),
            "registered": str(vd / f"{tag}.registered.json"), "program": str(vd / f"{name}.program.json"),
            "partition": str(vd / f"{name}.partition.json"), "ands": got["ands"],
            "circuit_sha512": got["circuit_sha512"], "statement": st.meta["statement"],
            "ok": np.asarray(got["outs"]["o0"][0]).tolist(), "want": want,
            "known": {kr[p][0]: [list(op)] for p, (_, op) in vk.items()},
            "binding": {p: committed[p] == mine[p] for p in ins}, "seconds": round(time.time() - t0, 1)}


def jobs(ctx: Ctx) -> list[tuple]:
    """Every statement: ``(tag, lane, chunk, world)``; the combine's lane None."""
    n = ctx.n
    honest = [(f"lane{ln}" + (f"-chunk{c}" if n > 1 else ""), ln, c, "honest")
              for ln in range(LL.lanes(*RC)) for c in range(n)] + [("combine", None, None, "honest")]
    neg = [("combine-ok-row-zero", None, None, "okzero"), ("lane1-last-ok-row-zero", LN, n - 1, "okzero")]
    if n > 1:
        neg = [("lane1-chunk0-carry-swapped", LN, 0, "carry"), ("lane1-chunk1-carry-swapped", LN, 1, "carry"), *neg]
    return honest + neg


def run_job(job: tuple) -> dict:
    tag, ln, ch, wname = job
    return statement(CTX, tag, CTX.worlds[wname], ln, ch if CTX.n > 1 else None)


def prepare(args) -> None:
    global CTX
    t0 = time.time()
    work = Path(args.work)
    for d in ("verifier", "st", "circuits"):
        (work / d).mkdir(parents=True, exist_ok=True)
    n = W.chunks(args.k)
    ctx = Ctx(work, args.name, args.k, n, args.position)
    t = HW.h1_tile(args.k, RC, H.PouwH1Tile5)
    ctx.win = HW.program((("w", args.k, RC[1]),), (("w", 0),), RC[0], RC, unit=lambda _k: Z.unit(t))
    if not HW.is_j10r(ctx.win):
        raise AssertionError("the window is J10-R's")
    calls, blocks = HW.j10r_inputs(ctx.win)
    ctx.j10r = (calls[0], tuple(blocks[0]))
    kinds = [(LL.COMBINE, None)] + ([(LL.LANE, None)] if n == 1 else [(LL.CHUNK, c) for c in range(n)])
    ctx.bound = {kc: derived(ctx, *kc) for kc in kinds}
    ctx.reg = ctx.bound[kinds[1]]
    ctx.place = HW.place(ctx.win, 0, STEPS)
    ctx.values = LL.produced_values(*RC, args.k, j10r=True)
    if any(base for _t, base in ctx.place.values()):
        raise AssertionError(f"the one-call window's unit 0 reads its values from row 0, not {ctx.place}")
    vd = work / "verifier"
    (vd / "window.json").write_bytes(canonical_json(encode_program(ctx.win)))
    (vd / "rounds.json").write_text(json.dumps([{"round": s} for s in range(STEPS)]))
    ctx.prog = P.program_digest(ctx.reg.program)
    for (kind, ch), b in ctx.bound.items():
        name = circuit_name(kind, ch)
        (vd / f"{name}.program.json").write_bytes(canonical_json(encode_program(b.program)))
        (vd / f"{name}.partition.json").write_text(json.dumps(b.part))
    programs = {circuit_name(*kc): P.program_digest(b.program) for kc, b in ctx.bound.items()}
    print(f"J10-R leaf {RC[0]} × {RC[1]} at K = {args.k}: {n} chunks a lane, call {ctx.j10r[0]} blocks {ctx.j10r[1]}; "
          f"programs {json.dumps({x: v[:16] for x, v in programs.items()})}, {time.time() - t0:.0f} s", flush=True)

    salt = hashlib.sha512(RN.SALT_TAG + args.name.encode()).digest()
    leaf = RN.leaf_data(args.seed, args.k, salt, args.position)
    regs, roots, receipt = register(ctx, leaf)
    partials, carries = after_draw(ctx, leaf, roots)
    xor = [0] * len(receipt)
    for p in partials:
        xor = [x ^ v for x, v in zip(xor, p)]
    assert xor == [v & 0xFFFFFFFF for v in receipt], "the honest partials XOR to v5's receipt"
    oks = [1] * (LL.lanes(*RC) * n)
    A = world(ctx, "honest", leaf, regs, roots, partials, carries, oks)
    zero = list(oks)
    zero[LL.statements(*RC, j10r=args.k).index(statement_key(ctx, LN, n - 1 if n > 1 else None))] = 0
    ctx.worlds = {"honest": A, "okzero": world(ctx, "okzero", leaf, regs, roots, partials, carries, zero)}
    if n > 1:
        swapped = [list(c) for c in carries]
        swapped[0][LN], swapped[0][SWAP] = carries[0][SWAP], carries[0][LN]
        ctx.worlds["carry"] = world(ctx, "carry", leaf, regs, roots, partials, swapped, oks)
    for nm, w in ctx.worlds.items():
        assert w.registration == A.registration, "only the after-draw values differ"
        (vd / f"{nm}.registration.json").write_text(json.dumps(w.reg_record))
        (vd / f"{nm}.produced.json").write_text(json.dumps(w.record))
    mine = {v: PR.entry(ctx.prog, v, rows, words, A.registration) for v, (rows, words) in ctx.values.items()}
    produced_check = {nm: list(PR.check(w.record, A.registration, A.draw, mine)) for nm, w in ctx.worlds.items()}
    print(f"registered and produced ({sorted(ctx.values)}): {time.time() - t0:.0f} s; produced records' codes against the "
          f"honest registration: {produced_check}", flush=True)
    ctx.stagers = {kc: stager(ctx, *kc) for kc in kinds}
    circuits = {circuit_name(*kc): {"statement": st.meta["statement"], "pin": st.pin, "class": st.class_sha512,
                                    "ands": st.low.layout.counts["ands"], "bytes": st.circuit.stat().st_size}
                for kc, st in ctx.stagers.items()}
    print(f"circuits: {json.dumps(circuits)}, {time.time() - t0:.0f} s", flush=True)
    CTX = ctx

    js = jobs(ctx)
    with mp.get_context("fork").Pool(args.jobs) as pool:
        recs = list(pool.imap_unordered(run_job, js))
    by = {r["tag"]: r for r in recs}
    expect = expects(n)
    with open(work / "plan.tsv", "w") as f:
        for tg, *_ in js:
            f.write(f"S\t{tg}\t{by[tg]['dir']}\n")
        for tg, *_ in js:
            r = by[tg]
            f.write(f"C\t{tg}\t{r['dir']}\t{r['pub']}\t{r['registered']}\t{r['program']}\t{r['partition']}\t"
                    f"{expect.get(tg, 'ACCEPT')}\t{vd / (r['world'] + '.registration.json')}\t"
                    f"{vd / (r['world'] + '.produced.json')}\t{vd / 'window.json'}\t{vd / 'rounds.json'}\t0\t{r['stmt']}\t"
                    f"{r['statement']}\n")
    honest = [jb[0] for jb in js if jb[0] not in expect]
    plan = {"checks": {tg: {**by[tg], "expect": expect.get(tg, "ACCEPT")} for tg, *_ in js}, "honest": honest}
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    summary = {"k": args.k, "rc": list(RC), "chunks": n, "tile": "PouwH1Tile_v5", "lanes": LL.lanes(*RC),
               "statements": len(LL.statements(*RC, j10r=args.k)), "j10r": [ctx.j10r[0], list(ctx.j10r[1])],
               "programs": programs, "place": ctx.place, "values": ctx.values, "registration": A.registration,
               "draw": A.draw, "produced": A.record, "produced_check": produced_check, "circuits": circuits,
               "ok": {tg: by[tg]["ok"] for tg in by}, "want": {tg: by[tg]["want"] for tg in by},
               "binding": {tg: by[tg]["binding"] for tg in by}, "seconds": round(time.time() - t0, 1)}
    (work / "prepare.json").write_text(json.dumps(summary, indent=1))
    print("PREPARED " + json.dumps({x: summary[x] for x in ("k", "rc", "chunks", "statements", "ok", "produced_check",
                                                            "seconds")}), flush=True)


def summarize(args) -> int:
    work, out = Path(args.work), Path(args.out)
    plan, prep = json.loads((work / "plan.json").read_text()), json.loads((work / "prepare.json").read_text())
    rows, opened, verdicts, known = {}, {}, {}, {}
    for tag, c in plan["checks"].items():
        v = RN._verdict(out / "lean" / f"{tag}.out")
        err = out / "lean" / f"{tag}.err"
        word = "NONE" if v is None else "ACCEPT" if v["accepted"] else "REJECT"
        why = (v or {}).get("why") or (err.read_text(errors="replace").strip().splitlines()[-1:] if err.exists() else "")
        rows[tag] = {"expect": c["expect"], "lean": word, "as_expected": word == c["expect"], "why": why,
                     "statement": c.get("statement"), "kind": c["kind"], "lane": c["lane"], "chunk": c["chunk"],
                     "world": c["world"], "ok": c["ok"], "want": c["want"], "ands": c["ands"],
                     "prove": RN._time(out / tag / "prove.time"), "lean_time": RN._time(out / "lean" / f"{tag}.time"),
                     "registered": (v or {}).get("registered")}
        if tag in plan["honest"] and word == "ACCEPT":
            verdicts[tag] = {**v, "of_record": True}
            known.update(c["known"])
            for e in (v.get("registered") or {}).values():
                if opened.setdefault(e["value"], e["root"]) != e["root"]:
                    opened[e["value"]] = None
    record = prep["produced"]
    opened_codes = list(PR.opened(record, opened))
    shared_codes = list(PR.shared(PR.wanted(record, known), verdicts))
    ok = (all(r["as_expected"] for r in rows.values()) and not opened_codes and not shared_codes
          and len(verdicts) == len(plan["honest"]) and not any(prep["produced_check"].values()))
    summary = {"ok": ok, "k": prep["k"], "rc": prep["rc"], "chunks": prep["chunks"], "statements": prep["statements"],
               "circuits": prep["circuits"], "lean_accepts": sum(r["lean"] == "ACCEPT" for r in rows.values()),
               "produced_check": prep["produced_check"], "opened": opened, "opened_codes": opened_codes,
               "shared_codes": shared_codes, "checks": rows, "registration": prep["registration"], "draw": prep["draw"],
               "produced": record, "binding": prep["binding"], "leaf_own": "not run: pouw-leaf-own-window states v1's "
               "statements only"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for tag, r in rows.items():
        print(f"{tag:28s} {r['lean']:7s} expect {r['expect']:7s} ok={r['ok']} {'ok' if r['as_expected'] else 'MISMATCH'}  "
              f"{str(r['why'])[:160]}")
    print(f"J10-R leaf {prep['rc']} at K = {prep['k']} in {prep['chunks']} chunks: {prep['statements']} statements; opened "
          f"roots against the produced record: {opened_codes or 'ok'}; opened rows against its openings: "
          f"{shared_codes or 'ok'}; ANDs {json.dumps({x: c['ands'] for x, c in prep['circuits'].items()})}; Lean "
          f"ACCEPTs {summary['lean_accepts']}; all as expected: {ok}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "summarize"))
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=4608)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--position", type=int, default=5)
    ap.add_argument("--name", default="h1-lanes-j10r-local")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a)
    else:
        sys.exit(summarize(a))


if __name__ == "__main__":
    main()
