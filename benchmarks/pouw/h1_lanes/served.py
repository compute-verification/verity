"""served.py extract|prepare|summarize: one unit a served H1 window's own draw selected (`h1_prove/served.py`, core's
`credit.drawn` on the window's signed draw coin), proved as its 32 lanes (`h1zy.PouwH1LaneZy_v1`, option A, run.py's lane
statement) from the window's committed values, each lane one C-Flock --zk statement whose rows open the window's own
registered tables at the rows the verifier reads them.

* **extract** (the H1 venv: it unpickles the prover's saved window, whose classes import torch). From the window's record
  (``pouw-h1-calls.json``: its manifest checked by `pouw_h1_challenger.manifest_reason`, so the served trees' entries
  are the ones its ``calls_root`` binds), the drawn units (``--drawn``, `h1_prove/served.py`'s), the period's
  descriptor and the prover's private directory (``--prover``, `pouw_h1_prover`: its salt key, each call's kept values
  and the saved window ``state/`` `pouw_h1_handoff.FILE`): the verifier's own reads of the drawn unit's call
  (`credit.registered_file` at the record's roots, the window's root, steps and Program salt), and for each of its data
  ports (the unit's 32 A rows, its 32 B rows and its z/y record; ``pub`` and ``o1`` are the verifier's own rows, which no
  lane reads) the prover's opening of that row in the tree the window registered, each held to the entry's root and
  domain and opened by `registered.verify_read`. A tree that keeps no rows (`pouw_h1_registered.Rows`) has the row
  rebuilt from the kept values (`pouw_h1_kept.kept_row`), and every row is held to the tree's ``b || c`` at its
  position. The openings (rows and salts, private) go to ``--out``, never under the run's ``out/``; ``--public`` gets
  the unit, the entries and the checks.
* **prepare** (hidden_zk's venv): the unit's leaf from those rows, the lane statement's Program (`h1window.leaf_program`
  of the window Program rebuilt from ``--descriptor`` and held to its bytes, run.py's `derived`; without one, run.py's
  one-call window: each statement is unit 0 of it), the lanes' partials committed after the draw
  (`produced`, under a registration record of the lane Program that names the window's three entries), and each lane
  staged with its ports read through ``--registered`` at the window's rows: ``a<s>`` row M·j + 32 rb + i of
  ``h1/a/<K>@<s>`` (j the call's place among its step's calls of K), ``b<t>`` row b0 + 32 cb + j of ``h1/b/<K>``,
  ``zy`` row t of ``h1/zy@<c>`` (the window's roots and
  domains, `credit.registered_file`'s), ``partials`` row l of the produced tree, and ``pub``, ``admit`` and ``o0`` known
  rows (run.py's `known_rows`). The lane's public values are conventions of this harness, named in its report: salt
  `pouw_h1_challenger.program_salt_of` of the window's commitment, position the drawn unit u, roots (root of the call's
  A table, root of ``h1/b/<K>``), admit bit i 1 exactly when slot 32 rb + i is one of the call's served rows.
  The negatives, each proved and checked like the lanes: ``tampered-zy`` (lane 0 given the record with cell 2's z low
  bit flipped, a cell lane 0 doesn't check, so its ok is still 1, under the honest salt and path, its public
  commitment the tampered row's, as the proof's: the row opens no root, REJECT) and ``other-call-a`` (lane 0 with the
  verifier's ``a`` ports at the A table of a call of K in another step, its root and domain, the prover opening its
  own: REJECT).
* **summarize**: OUT/summary.json from the Lean verdicts; exit 1 unless every check is as expected and every honest
  lane's opened roots are the window's.
* **smoke**: a stand-in window at a small K in extract's form (`smoke`), the harness's own check.

No combine: a served window commits no CP-W receipt for it to check; the 32 lanes' ok bits cover every cell of the unit's
record (z = A Bᵀ exactly, y = its dequant).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "h1_prove"))
sys.path.insert(0, str(HERE))

import h1unit as U  # noqa: E402
import run as L  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.core.protocols.pouw import leaf_lanes as LL  # noqa: E402
from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import produced as PR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as R  # noqa: E402

RC = L.RC
EXPECT = {"tampered-zy": "REJECT", "other-call-a": "REJECT"}
#: the record word flipped in ``tampered-zy``: cell 2's z, its low half (cell 2, row 0 column 2, is lane 1's)
TAMPER_WORD = 4


# -- extract ----------------------------------------------------------------------------------------------------------------

def _descriptor(path: str) -> bytes:
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def _window(path: str):
    """The served window's Program rebuilt from its descriptor (`credit.layout_of`'s layout, `credit.shape_of`'s prefix),
    each K's template held to the descriptor's by its descriptor id (which names every binding, so the 870 MB descriptor
    is parsed, never re-encoded), and its digest: the lane statement is `h1window.leaf_program`'s of it, which reads only
    the template at K, as a served window's verifier derives it."""
    from verity.core.protocols.pouw import credit as CR
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity.ml.circuits.codec import descriptor_id

    raw = _descriptor(path)
    layout, why = CR.layout_of(raw)
    if why is not None:
        raise SystemExit(f"REFUSED: {why}")
    shape = CR.shape_of(raw)
    win = HW.program([tuple(x) for x in layout["linears"]], [tuple(x) for x in layout["calls"]], layout["m"],
                     tuple(layout["tile"]), shape["prefix"])
    got = {int(k): descriptor_id(t) for k, t in win.fn.bindings["TILES"]}
    want = {int(k): t["fn"] for k, t in shape["tiles"].items()}
    if got != want:
        raise SystemExit(f"the window Program rebuilt from the descriptor's layout has other templates: K "
                         f"{sorted(k for k in got.keys() | want.keys() if got.get(k) != want.get(k))}")
    return win, hashlib.sha512(raw).hexdigest()


def _k_calls(layout: dict, k: int) -> list[int]:
    ks = {name: kk for name, kk, _n in layout["linears"]}
    return [c for c, (lin, _blk) in enumerate(layout["calls"]) if ks[lin] == k]


def _data_port(port: str) -> bool:
    """A port of a call's statement that a lane reads: `credit.registered_file`'s A, B and z/y reads."""
    return port.startswith(("a_k", "b_k")) or port == "zy"


def extract(a) -> None:
    import pickle

    from verity.core.protocols.pouw import credit as CR
    from verity_vllm.protocols.pouw.h1 import challenger as PHC
    from verity_vllm.protocols.pouw.h1 import handoff as HO
    from verity_vllm.protocols.pouw.h1 import prover as PV
    from verity_vllm.protocols.pouw.h1 import kept as KP

    t0 = time.time()
    doc = json.loads(Path(a.calls).read_text())
    drawn = json.loads(Path(a.drawn).read_text())
    commits = doc["commitment"]
    root = commits["session"]["root"]
    steps, psalt = int(commits["steps"]), PHC.program_salt_of(commits).hex()
    if drawn["window"] != root:
        raise SystemExit(f"the drawn units are window {drawn['window'][:16]}...'s, not {root[:16]}...")
    if (why := PHC.manifest_reason(doc)) is not None:
        raise SystemExit(f"REFUSED: {why}")
    raw = _descriptor(a.descriptor)
    program = hashlib.sha512(raw).hexdigest()
    if program != drawn["circuit"]:
        raise SystemExit("the descriptor is not the circuit the window's period committed")
    layout, why = CR.layout_of(raw)
    if why is not None:
        raise SystemExit(f"REFUSED: {why}")
    shape = CR.shape_of(raw)
    entries = commits["served"]["entries"]
    roots: dict[str, list[str]] = {}
    for e in entries:
        roots.setdefault(e["value"], []).append(e["root"])
    mine = sorted((u for u in drawn["units"] if u["k"] == a.k), key=lambda u: u["unit"])
    if not mine:
        raise SystemExit(f"the draw selected no unit at K = {a.k}: {sorted({u['k'] for u in drawn['units']})}")
    unit = mine[0]
    u, c = unit["unit"], unit["call"]
    reg, why = CR.registered_file(program, layout, shape, c, roots, root, steps, psalt)
    if why is not None:
        raise SystemExit(f"REFUSED: {why}")
    a_port = f"a_k{a.k}_0"
    reg2 = None
    for c2 in (x for x in _k_calls(layout, a.k) if x != c):
        reg2, why = CR.registered_file(program, layout, shape, c2, roots, root, steps, psalt)
        if why is not None:
            raise SystemExit(f"REFUSED: {why}")
        if reg2["ports"][a_port]["value"] != reg["ports"][a_port]["value"]:
            break
    else:
        raise SystemExit(f"no call of K = {a.k} in another step: other-call-a has no other A table to read")
    print(f"unit {u} (call {c}, rb {unit['row_block']}, cb {unit['col_block']}); verifier's reads built, "
          f"{time.time() - t0:.0f} s", flush=True)

    prover = Path(a.prover)
    with open(prover / PV.STATE / HO.FILE, "rb") as f:
        st = pickle.load(f)
    _kind, served, _layers = st["state"]["served"]
    trees = served["registered"]
    print(f"state loaded: {len(trees)} trees, {time.time() - t0:.0f} s", flush=True)
    if served["root"] != root or served["program_digest"] != program:
        raise SystemExit("the saved window is not the record's: another root or Program")
    ports, checks = {}, {}
    for port, p in reg["ports"].items():
        if not _data_port(port):
            continue
        row, tree = p["reads"][u], trees[p["value"]]
        try:
            words = list(tree.rows[row])
        except LookupError:
            words = KP.kept_row(prover, st["state"], doc["calls"], p["value"], row)
        ok = {"root": tree.root.hex() == p["root"], "domain": tree.domain.domain_id.hex() == p["domain"],
              "shape": (tree.leaves, tree.words) == (p["leaves"], p["words"]) and tree.schema == V2}
        path, bc = tree.path(row), tree.bcs[row]
        ok["opens"] = G.verify_read(tree.domain, tree.root, row, bc, path, V2)
        ok["row"] = G.Registered.commit(tree.program, tree.value, [words], [tree.salts[row]], tree.word_bits, V2,
                                        tree.window, tree.produced).bcs[0] == bc
        checks[port] = ok
        ports[port] = {"entry": {x: v for x, v in p.items() if x != "reads"},
                       "opening": {"value": p["value"], "row": words, "salt": tree.salts[row].hex(),
                                   "bc": bc.hex(), "position": row, "path": path.hex()}}
    bad = {p: c_ for p, c_ in checks.items() if not all(c_.values())}
    if bad:
        raise SystemExit(f"the saved trees don't open the verifier's reads: {dict(list(bad.items())[:4])}")
    other = {x: v for x, v in reg2["ports"][a_port].items() if x != "reads"}
    call = next(x for x in doc["calls"] if x["index"] == c)
    out = {"window": root, "program": program, "draw_root": commits["draw"]["root"], "k": a.k, "unit": unit,
           "salt": PHC.program_salt_of(commits).hex(), "served_rows": int(call["served"]), "ports": ports,
           "other": {"call": c2, "entry": other}}
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(out, f)
    pub = {**{x: out[x] for x in ("window", "program", "draw_root", "k", "unit", "served_rows", "other")},
           "drawn": len(drawn["units"]), "drawn_at_k": [x["unit"] for x in mine],
           "entries": {p: v["entry"] for p, v in ports.items()},
           "reads": {p: v["opening"]["position"] for p, v in ports.items()}, "checks": checks,
           "seconds": round(time.time() - t0, 1)}
    Path(a.public).write_text(json.dumps(pub, indent=1))
    print("EXTRACTED " + json.dumps({x: pub[x] for x in ("unit", "served_rows", "drawn", "seconds")}), flush=True)


def smoke(a) -> None:
    """A stand-in window at ``--k`` in `extract`'s form: its A, B and z/y tables of random rows, committed as a served
    window's (the record's tables under a random window root and Program digest, B under none), unit 3 of a 64 × 64 call
    drawn: the harness's own check, never a served result."""
    rng = np.random.default_rng(a.seed)
    k, rb, cb, t, u = a.k, 1, 1, 3, 7
    root, program = os.urandom(64).hex(), os.urandom(64).hex()
    ca, cbw = (rng.integers(-128, 128, (64, k)).astype(np.int8) for _ in range(2))
    sa, sw = ((rng.uniform(1e-3, hi, 64) / 127).astype("<f4") for hi in (0.1, 0.5))

    def tree(value, rows, window=root):
        return G.Registered.commit(program, value, rows, schema=V2, window=window)

    A, A2 = tree(f"h1/a/{k}@3", U.row_words(ca, sa).tolist()), tree(f"h1/a/{k}@5", U.row_words(ca[::-1], sa).tolist())
    B = tree(f"h1/b/{k}", U.row_words(cbw, sw).tolist(), None)
    recs = []
    for x in range(4):
        r_, c_ = divmod(x, 2)
        rows, cols = slice(32 * r_, 32 * r_ + 32), slice(32 * c_, 32 * c_ + 32)
        z, y = Z.records(ca[rows].view(np.uint8).tolist(), cbw[cols].view(np.uint8).tolist(),
                         sa[rows].view(np.uint32).tolist(), sw[cols].view(np.uint32).tolist())
        recs.append(U.record_words(np.asarray(z, np.int64), np.asarray(y)).tolist())
    ZY = tree("h1/zy@3", recs)

    def entry(tr):
        return {**tr.entry(), "word_bits": tr.word_bits, **G.schema_entry(V2)}

    def port(tr, row):
        return {"entry": entry(tr), "opening": {"value": tr.value, "row": list(tr.rows[row]), "salt": tr.salts[row].hex(),
                                                "bc": tr.bcs[row].hex(), "position": row, "path": tr.path(row).hex()}}

    ports = {f"a_k{k}_{i}": port(A, 32 * rb + i) for i in range(32)}
    ports.update({f"b_k{k}_{j}": port(B, 32 * cb + j) for j in range(32)})
    ports["zy"] = port(ZY, t)
    op = {"window": root, "program": program, "draw_root": os.urandom(64).hex(), "k": k,
          "unit": {"unit": u, "call": 3, "row_block": rb, "col_block": cb, "k": k, "n": 64, "z_leaf": t, "stand_in": True},
          "salt": os.urandom(64).hex(), "served_rows": 50, "ports": ports, "other": {"call": 5, "entry": entry(A2)}}
    Path(a.out).write_text(json.dumps(op))


# -- prepare ----------------------------------------------------------------------------------------------------------------

@dataclass
class Served:
    """The verifier's view of the drawn unit: its leaf, the roots its seeds read, the produced partials and the window's
    ports (``ports``: port name -> its entry and the prover's opening)."""

    leaf: L.Leaf
    roots: tuple
    registration: str
    draw: str
    tree: G.Registered
    record: dict
    ports: dict
    values: dict


def _opening(o: dict) -> dict:
    return {**o, "salt": bytes.fromhex(o["salt"]), "bc": bytes.fromhex(o["bc"])}


def leaf_of(op: dict, k: int, salt: bytes, position: int, rb: int, served_rows: int) -> L.Leaf:
    a, sa, b, sw = [], [], [], []
    for i in range(RC[0]):
        codes, s = U.split_row(op["ports"][f"a_k{k}_{i}"]["opening"]["row"])
        a.append(codes)
        sa.append(s)
    for j in range(RC[1]):
        codes, s = U.split_row(op["ports"][f"b_k{k}_{j}"]["opening"]["row"])
        b.append(codes)
        sw.append(s)
    zy = U.split_record(op["ports"]["zy"]["opening"]["row"], RC[0] * RC[1])
    admit = [int(RC[0] * rb + i < served_rows) for i in range(RC[0])]
    return L.Leaf(a, b, sa, sw, zy, admit, salt, position)


def lane_reads(k: int, ln: int) -> dict[str, str]:
    """Lane ``ln``'s window ports by its statement's port name."""
    out = {f"a{s}": f"a_k{k}_{i}" for s, i in enumerate(LL.lane_rows(*RC, ln))}
    out.update({f"b{t}": f"b_k{k}_{j}" for t, j in enumerate(LL.lane_cols(*RC, ln))})
    return {**out, LL.ZY: "zy"}


def expected(sw: Served, ln: int, ins: dict) -> list[int]:
    lf = sw.leaf
    a, b, sa, sw_ = L.lane_rows(lf, ln)
    zy = U.split_record(ins[LL.ZY]["row"], RC[0] * RC[1])
    return [Z.lane_reference(a, b, sa, sw_, lf.admit, zy, L.u32_words(ins[LL.PARTIALS]["row"]), lf.salt, lf.position,
                             sw.roots, ln, RC)]


def statement(ctx, ln: int, tag: str, sw: Served, ins: dict, entries: dict) -> dict:
    """Stage lane ``ln`` as ``tag`` and write the verifier's files: ``ins`` each input port's opening; ``entries`` each
    read port's ``(value, entry, row)`` as the verifier holds it (the window's, or the produced partials')."""
    st = ctx.stagers[LL.LANE]
    kr = L.known_rows(ctx, sw, LL.LANE, ln, ln)
    vk = {p: PR.known_port(ctx.prog, v, row, salt, sw.registration, schema=V2) for p, (v, row, salt) in kr.items()}
    reads = {"format": G.READS, "ports": {**{p: PR.port(v, e, e["root"], [row], V2) for p, (v, e, row) in entries.items()},
                                          **{p: kp for p, (kp, _) in vk.items()}}}
    want = expected(sw, ln, ins)
    v_ok, row_ok, salt_ok = kr[LL.OK]
    outs = {LL.OK: L.known_opening(ctx, sw, v_ok, row_ok, salt_ok)}

    def check(_rows, got):
        if not np.array_equal(np.asarray(got["o0"][0], np.uint16), np.asarray(want, np.uint16)):
            raise AssertionError(f"{tag}: the lowered circuit's ok is not the word reference's")

    st.reference_check = check
    rows = {p: np.asarray([o["row"]], np.uint16) for p, o in ins.items()}
    opened = {**ins, **outs}
    reg = {p: {"value": o["value"], "positions": [o["position"]], "paths": [o["path"]]} for p, o in opened.items()
           if "value" in o}
    t0 = time.time()
    d = ctx.work / "st" / tag
    got = st.stage(rows, d, os.urandom(32), salts={p: [o["salt"]] for p, o in opened.items()}, indices=[0],
                   set_name=f"h1-served-{tag}", tables=rows, refs={p: [0] for p in rows}, registered=reg)
    (d / "circuit.txt").unlink(missing_ok=True)
    (d / "circuit.txt").symlink_to(st.circuit)
    (d / "pin.txt").write_text(st.pin)
    ports = {p.name: p for p in st.ports}
    committed = {p: U.commit_strings(ports[p], rows[p], [o["salt"]])[0] for p, o in ins.items()}
    mine = {p: o["bc"] if "bc" in o else U.commit_strings(ports[p], np.asarray([o["row"]], np.uint16), [o["salt"]])[0]
            for p, o in opened.items()}
    vreads = [{p: (o["value"], reads["ports"][p]["reads"][0], o["path"]) for p, o in opened.items() if "value" in o}]
    if set(vreads[0]) != set(reads["ports"]) or set(reads["ports"]) != LL.own(*RC, LL.LANE, ln):
        raise AssertionError(f"{tag}: the opened ports {sorted(vreads[0])} are not the verifier's {sorted(reads['ports'])}")
    pub = U.verifier_pub({"meta": st.meta, "pin": st.pin, "class": st.class_sha512}, f"h1-served-{tag}", st.sub.id, [mine],
                         [0], vreads)
    vd = ctx.work / "verifier"
    (vd / f"{tag}.pub.bin").write_bytes(pub)
    (vd / f"{tag}.registered.json").write_text(json.dumps(reads))
    return {"tag": tag, "lane": ln, "dir": str(d), "pub": str(vd / f"{tag}.pub.bin"),
            "registered": str(vd / f"{tag}.registered.json"), "program": str(vd / "lane.program.json"),
            "partition": str(vd / "lane.partition.json"), "ands": got["ands"], "circuit_sha512": got["circuit_sha512"],
            "ok": np.asarray(got["outs"]["o0"][0]).tolist(), "want": want,
            "binding": {p: committed[p] == mine[p] for p in ins}, "seconds": round(time.time() - t0, 1)}


CTX = None


def lane_ins(ctx, sw: Served, ln: int) -> tuple[dict, dict]:
    """Lane ``ln``'s honest openings and the verifier's read entries."""
    ins, entries = {}, {}
    for port, wp in lane_reads(ctx.k, ln).items():
        p = sw.ports[wp]
        ins[port] = _opening(p["opening"])
        entries[port] = (p["entry"]["value"], p["entry"], p["opening"]["position"])
    ins[LL.PARTIALS] = L.opening(sw.tree, ln)
    entries[LL.PARTIALS] = (LL.PARTIALS, {**sw.values[LL.PARTIALS], "root": sw.record["produced"][0]["root"]}, ln)
    kr = L.known_rows(ctx, sw, LL.LANE, ln, ln)
    for p in (LL.ADMIT, LL.PUB):
        v, row, salt = kr[p]
        ins[p] = L.known_opening(ctx, sw, v, row, salt)
    return ins, entries


def run_job(job: tuple) -> dict:
    ctx = CTX
    tag, ln, neg = job
    sw = ctx.worlds["served"]
    ins, entries = lane_ins(ctx, sw, ln)
    if neg == "tampered-zy":
        row = list(ins[LL.ZY]["row"])
        row[TAMPER_WORD] ^= 1
        ins[LL.ZY] = {**{x: v for x, v in ins[LL.ZY].items() if x != "bc"}, "row": row}
    elif neg == "other-call-a":
        other = sw.values["other-a"]
        for s in range(LL.LANE_ROWS):
            _v, _e, row = entries[f"a{s}"]
            entries[f"a{s}"] = (other["value"], other, row)
            ins[f"a{s}"] = {**ins[f"a{s}"], "value": other["value"]}
    return statement(ctx, ln, tag, sw, ins, entries)


def prepare(a) -> None:
    global CTX
    t0 = time.time()
    work = Path(a.work)
    for d in ("verifier", "st", "circuits"):
        (work / d).mkdir(parents=True, exist_ok=True)
    k = a.k if a.wait else int(json.loads(Path(a.opened).read_text())["k"])
    ctx = L.Ctx(work, a.name, k, 0)
    t = L.tile(k)
    ctx.wp = 2 * LL.words(k)
    win, circuit = _window(a.descriptor) if a.descriptor else (L.window(t, k), None)
    ctx.lane = ctx.reg = L.derived(win, k, LL.LANE)
    ctx.prog = P.program_digest(ctx.lane.program)
    (work / "verifier" / "lane.program.json").write_bytes(canonical_json(encode_program(ctx.lane.program)))
    (work / "verifier" / "lane.partition.json").write_text(json.dumps(ctx.lane.part))
    print(f"lane Program {ctx.prog[:16]}, {time.time() - t0:.0f} s", flush=True)
    # the lane circuit depends on K alone: lowered (and pinned) before the openings, which --wait lets arrive meanwhile
    ctx.stagers = {LL.LANE: L.stager(ctx, LL.LANE)}
    st = ctx.stagers[LL.LANE]
    st.pin, st.class_sha512  # noqa: B018
    print(f"lane circuit {st.low.layout.counts['ands']} ANDs, {time.time() - t0:.0f} s", flush=True)
    while a.wait and not Path(a.opened).exists():
        if time.time() - t0 > a.wait:
            raise SystemExit(f"no openings at {a.opened} after {a.wait} s")
        time.sleep(10)
    op = json.loads(Path(a.opened).read_text())
    if int(op["k"]) != k:
        raise SystemExit(f"the openings are at K={op['k']}, the lane circuit at K={k}")
    if circuit is not None and op["program"] != circuit:
        raise SystemExit(f"the openings are window Program {op['program'][:16]}'s, the lane statement {circuit[:16]}'s")
    unit = op["unit"]
    u = ctx.position = int(unit["unit"])
    print(f"openings of {op['window'][:16]} unit {u}, {time.time() - t0:.0f} s", flush=True)

    salt = bytes.fromhex(op["salt"])
    a_root = bytes.fromhex(op["ports"][f"a_k{k}_0"]["entry"]["root"])
    b_root = bytes.fromhex(op["ports"][f"b_k{k}_0"]["entry"]["root"])
    leaf = leaf_of(op, k, salt, u, int(unit["row_block"]), int(op["served_rows"]))
    z, y = Z.records(leaf.a, leaf.b, leaf.sa, leaf.sw)
    zbad = [i for i in range(len(z)) if z[i] != (leaf.zy[0][i] & Z.M32)]
    ybad = [i for i in range(len(y)) if y[i] != (leaf.zy[1][i] & 0xFFFF)]
    unit_ok = not zbad and not ybad
    if not unit_ok:
        raise SystemExit(f"the unit's committed record is not its committed rows' product and dequant: z differs at "
                         f"{len(zbad)} cells {zbad[:8]}, y at {len(ybad)} {ybad[:8]}")
    print(f"the unit's committed record checks on words (all {len(z)} cells), {time.time() - t0:.0f} s", flush=True)

    win = {v: op["ports"][p]["entry"] for v, p in ((LL.A, f"a_k{k}_0"), (LL.B, f"b_k{k}_0"), (LL.ZY, "zy"))}
    rec = R.record(ctx.reg.program, ctx.reg.query, {"law": "subset", "k": 1}, L.SCHEME, [],
                   {"kind": "served-window", "window": op["window"], "draw_root": op["draw_root"], "unit": u,
                    "call": unit["call"]}, partition=ctx.reg.part, registered=list(win.values()))
    d = R.digest(rec)
    pa = L.partials(leaf, (a_root, b_root))
    # the partials' salts come from a private opening salt, so prepares of disjoint lane sets (on several nodes) commit
    # the one produced tree the unit's lanes read
    zs = bytes.fromhex(op["ports"]["zy"]["opening"]["salt"])
    seed = hashlib.sha512(b"h1-lanes/partials-salts/v1" + zs).digest()
    psalts = [b"".join(hashlib.sha512(seed + i.to_bytes(4, "big") + j.to_bytes(2, "big")).digest()
                       for j in range(-(-len(zs) // 64)))[:len(zs)] for i in range(len(pa))]
    tree = PR.commit(ctx.prog, LL.PARTIALS, pa, d, psalts, schema=V2)
    ops = {LL.PARTIALS: PR.rows_of(tree)}
    for p in op["ports"].values():
        ops.setdefault(p["entry"]["value"], []).append((p["opening"]["position"], p["opening"]["bc"]))
    record = PR.record(d, op["draw_root"], [tree.entry()], ops)
    produced = PR.entry(ctx.prog, LL.PARTIALS, LL.lanes(*RC), ctx.wp, d)
    produced_check = list(PR.check(record, d, op["draw_root"], {LL.PARTIALS: produced}))
    if produced_check:
        raise SystemExit(f"the produced record fails its own check: {produced_check}")
    sw = Served(leaf, (a_root, b_root), d, op["draw_root"], tree, record, op["ports"],
                {LL.PARTIALS: produced, "other-a": op["other"]["entry"]})
    ctx.worlds = {"served": sw}
    print(f"registration {d[:16]}, partials {tree.root.hex()[:16]}, {time.time() - t0:.0f} s", flush=True)
    CTX = ctx

    lanes = [ln for ln in range(LL.lanes(*RC))] if a.lanes == "all" else [int(x) for x in a.lanes.split(",")]
    js = [(f"lane{ln}", ln, None) for ln in lanes] + [(tg, 0, tg) for tg in EXPECT if tg not in a.skip.split(",")]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        recs = list(pool.imap_unordered(run_job, js))
    by = {r["tag"]: r for r in recs}
    wrong = {tg: (r["ok"], r["want"]) for tg, r in by.items() if r["ok"] != [1]}
    if wrong:
        raise SystemExit(f"a statement's ok is not 1: {wrong}")
    with open(work / "plan.tsv", "w") as f:
        for tg, _ln, _neg in js:
            f.write(f"S\t{tg}\t{by[tg]['dir']}\n")
        for tg, _ln, _neg in js:
            r = by[tg]
            f.write(f"C\t{tg}\t{r['dir']}\t{r['pub']}\t{r['registered']}\t{r['program']}\t{r['partition']}\t"
                    f"{EXPECT.get(tg, 'ACCEPT')}\n")
    honest = [tg for tg, _ln, neg in js if neg is None]
    plan = {"checks": {tg: {**by[tg], "expect": EXPECT.get(tg, "ACCEPT")} for tg, _l, _n in js}, "honest": honest}
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    summary = {"k": k, "tile": U.tile(), "unit": unit, "window": op["window"], "draw_root": op["draw_root"],
               "lane_program": ctx.prog, "registration": d, "produced": record["produced"], "unit_ok_on_words": unit_ok,
               "roots": {"a": a_root.hex(), "b": b_root.hex(), "zy": op["ports"]["zy"]["entry"]["root"]},
               "lanes": lanes, "ands": by[js[0][0]]["ands"], "ok": {tg: by[tg]["ok"] for tg in by},
               "binding": {tg: by[tg]["binding"] for tg in by}, "seconds": round(time.time() - t0, 1)}
    (work / "prepare.json").write_text(json.dumps(summary, indent=1))
    print("PREPARED " + json.dumps({x: summary[x] for x in ("k", "unit", "ands", "seconds")}), flush=True)


# -- summarize --------------------------------------------------------------------------------------------------------------

def summarize(a) -> int:
    work, out = Path(a.work), Path(a.out)
    plan, prep = json.loads((work / "plan.json").read_text()), json.loads((work / "prepare.json").read_text())
    rows, opened = {}, {}
    for tag, c in plan["checks"].items():
        v = L._verdict(out / "lean" / f"{tag}.out")
        err = out / "lean" / f"{tag}.err"
        word = "NONE" if v is None else ("ACCEPT" if v["accepted"] else "REJECT")
        why = (v or {}).get("why") or (err.read_text(errors="replace").strip().splitlines()[-1:] if err.exists() else "")
        rows[tag] = {"expect": c["expect"], "lean": word, "as_expected": word == c["expect"], "why": why, "ok": c["ok"],
                     "ands": c["ands"], "prove": L._time(out / tag / "prove.time"),
                     "lean_time": L._time(out / "lean" / f"{tag}.time"), "registered": (v or {}).get("registered")}
        if tag in plan["honest"] and v and v["accepted"]:
            for e in (v.get("registered") or {}).values():
                opened.setdefault(e["value"], set()).add(e["root"])
    window = {prep["roots"]["a"], prep["roots"]["b"], prep["roots"]["zy"], prep["produced"][0]["root"]}
    roots_ok = bool(opened) and all(len(r) == 1 and next(iter(r)) in window for v, r in opened.items()
                                    if not v.startswith(("pub", "ok", LL.ADMIT)))
    ok = all(r["as_expected"] for r in rows.values()) and roots_ok
    summary = {"ok": ok, "k": prep["k"], "unit": prep["unit"], "window": prep["window"], "draw_root": prep["draw_root"],
               "roots": prep["roots"], "opened": {v: sorted(r) for v, r in opened.items()}, "roots_ok": roots_ok,
               "lane_program": prep["lane_program"], "registration": prep["registration"], "produced": prep["produced"],
               "ands": prep["ands"], "checks": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for tag, r in rows.items():
        print(f"{tag:14s} {r['lean']:7s} expect {r['expect']:7s} ok={r['ok']} {'ok' if r['as_expected'] else 'MISMATCH'}  "
              f"{str(r['why'])[:160]}")
    print(f"honest lanes' opened roots are the window's: {roots_ok}; all as expected: {ok}")
    return 0 if ok else 1


SHARED = ("k", "unit", "window", "draw_root", "roots", "lane_program", "registration", "produced", "ands")


def merge(a) -> int:
    """The summaries of jobs that each proved a share of one unit's lanes (H1L_LANES), as one: every share read the same
    window, draw, unit, registration and produced tree, each lane and negative was checked once, every lane accepted and
    both negatives were refused."""
    parts = [(p, json.loads(Path(p).read_text())) for p in a.summaries]
    head, problems, checks, opened = parts[0][1], [], {}, {}
    for p, s in parts:
        problems += [f"{p}: {x} differs from {parts[0][0]}'s" for x in SHARED if s[x] != head[x]]
        for tag, r in s["checks"].items():
            if tag in checks:
                problems.append(f"{tag} checked twice ({checks[tag]['from']} and {p})")
            checks[tag] = {**r, "from": p}
        for v, rs in s["opened"].items():
            opened.setdefault(v, set()).update(rs)
    n = LL.lanes(*RC)
    lanes = [f"lane{ln}" for ln in range(n)]
    problems += [f"{t} not checked" for t in lanes + sorted(EXPECT) if t not in checks]
    problems += [f"{t}: Lean {r['lean']}, expected {r['expect']}" for t, r in checks.items() if not r["as_expected"]]
    problems += [f"{v} opened at {len(r)} roots" for v, r in opened.items() if len(r) != 1]
    problems += [f"{p}: its honest lanes' roots aren't the window's" for p, s in parts if not s["roots_ok"]]
    merged = {"ok": not problems, "problems": problems, "parts": [p for p, _ in parts],
              **{x: head[x] for x in SHARED}, "opened": {v: sorted(r) for v, r in opened.items()},
              "accepted": sum(checks[t]["lean"] == "ACCEPT" for t in lanes if t in checks),
              "refused": {t: checks[t]["why"] for t in EXPECT if t in checks}, "checks": checks}
    Path(a.out).write_text(json.dumps(merged, indent=1))
    print(f"{merged['accepted']}/{n} lanes accepted; negatives {merged['refused']}; "
          f"{'ok' if merged['ok'] else 'NOT ok: ' + '; '.join(problems[:12])}")
    return 0 if merged["ok"] else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("extract", "smoke", "prepare", "summarize", "merge"))
    ap.add_argument("summaries", nargs="*")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--calls")
    ap.add_argument("--drawn")
    ap.add_argument("--descriptor")
    ap.add_argument("--prover", help="extract: the window's prover directory (pouw_h1_prover)")
    ap.add_argument("--public")
    ap.add_argument("--opened")
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--lanes", default="all")
    ap.add_argument("--skip", default="")
    ap.add_argument("--wait", type=int, default=0, help="prepare: wait up to this many seconds for --opened (K from --k)")
    ap.add_argument("--name", default="h1-served-local")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a)
    elif a.cmd == "smoke":
        smoke(a)
    elif a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "merge":
        sys.exit(merge(a))
    else:
        sys.exit(summarize(a))


if __name__ == "__main__":
    main()
