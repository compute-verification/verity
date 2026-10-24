"""warden-zk: the warden's cell rows read by the proof system. A toy unit over one cell's bits, its port registered
against the warden's signed egress root (`warden_rows`), proved by C-Flock `prove --zk` behind the Lean firewall and
held by the Lean verifier's registered reads (`flock-verify verify --zk --registered`), through the one-stage audit as
N2-ZK's driver proves N2's head checks.

    python benchmarks/network_traces/warden_zk.py prove --run DIR --bin flock-circuit --lean flock-verify --out DIR
      [--units 16] [--dry]

* **The run.** A warden-serve run's directory (`warden_serve.py rehearse`'s): `deployment.json`, the records, and the
  cell rows the warden committed and signed while serving: `cells.jsonl` (each link-window's registration) and
  `cells.salts` (the rows' salts, witness, which the warden hands the prover alone).
* **The registration.** The egress link-window whose busiest connection has the most cells: its registration read by
  the verifier (`warden_rows.check_registration`, under the deployment's key and session and the schedule's grid and
  parameters), and the warden's opening (the record's rows under the salts) held to its root.
* **The unit** (`WardenCellIdentity{B}`): one cell's 8·B bits in (`cell`), the same bits out (`o0`), each through two
  NOTs (the partition refuses a template with no gate; no AND). n units, unit i the connection's i-th row of the
  window, its data cells first (`positions`); `reads` is that wiring, opened to the verifier as the units' wiring is.
* **Members.** One statement of n units each, a shared-row public file whose registered ports are the verifier's own
  (`--registered`, `warden_rows.port` at the warden's root):
  - ``sender``: ``o0`` registered, under the warden's salts and paths: the sender's unit opens its output as the cells;
  - ``receiver``: ``cell`` registered: the receiver's check opens the same rows as its input;
  - ``unsigned``: ``o0`` under the prover's own tree over the window's rows, unit 0's cell one byte off: rows the
    warden didn't sign (and the prover's registration of its own root, under the warden's certificate, is refused by
    `check_registration`);
  - ``altered``: ``o0`` under the warden's salts and paths, unit 0's input cell one bit off, so its output differs from
    the warden's cell.
  Each is registered, every unit drawn by the verifier's Lean (`flock-verify draw --stratified n`), proved `--zk`
  behind the firewall (``FC_FIREWALL=1``, ``FC_FIREWALL_LEAN``) and held by Lean `verify --zk --partition --program
  --registered`. ``sender`` and ``receiver`` are to be accepted, the other two refused by Lean's registered reads.
* **What the verifier receives**, scanned for every 16-byte piece of every opened cell and every salt of the
  statement (`a8.received`) but the pieces the public circuit holds and the empty slot's (zero bytes, the schedule's):
  none is to be found; the prover's instance file, which holds them, is the control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ONE_STAGE = HERE.parent / "one_stage"
sys.path.insert(0, str(HERE))

import warden_serve as ws  # noqa: E402
from verity.ml.circuits import boolean as BL  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.ml.circuits.types import Array  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA  # noqa: E402
from verity.core.protocols.certifier.certificate import Certificate  # noqa: E402
from verity.core.protocols.certifier.warden import END, WardenParams  # noqa: E402

MEMBERS = {"sender": ("o0", True), "receiver": ("cell", True), "unsigned": ("o0", False), "altered": ("o0", False)}


def log(msg: str) -> None:
    print(f"WZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the unit ---------------------------------------------------------------------------------------------------------

def _signature(S):
    return (("cell", Array(8 * S.B, BL.BIT)),), Array(8 * S.B, BL.BIT)


def _body(B, S, cell):
    return B.batch(BL.NOT, B.batch(BL.NOT, cell, axes=(0,)), axes=(0,))


WardenCellIdentity = CompositeDefinition(
    "WardenCellIdentity", 1, ("B",), _signature, _body, conformance="exact-model-tested", register=False,
    doc="WardenCellIdentity{B}: one warden cell's 8·B bits, each through two NOTs: the toy unit whose port opens a "
        "cell row of the warden's signed root (warden_zk.py)")


# --- the run ----------------------------------------------------------------------------------------------------------

def attestation(run: Path, n: int) -> SimpleNamespace:
    """The egress link-window to read, its registration as the verifier reads it, the warden's opening, and the rows
    the n units open (the busiest connection's, data cells first)."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    key, session = ws.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    link, direction = ws.CERTIFIED[ws.EGRESS]
    records = {r.window: r for r in ws.read_records(run, params)[ws.EGRESS]}
    cells = {(d["registration"]["window"]): (d, salts) for d, salts in ws.read_cells(run)
             if "registration" in d and d["registration"]["direction"] == direction.value}
    best = None
    for w, record in sorted(records.items()):
        for q, rows in WR.connection_rows(record, params.grid).items():
            if len(rows) >= n and w in cells and (best is None or len(rows) > len(best[2])):
                best = (w, q, rows)
    if best is None:
        raise SystemExit(f"no egress link-window of {run} has a connection of {n} rows and a registration")
    w, q, rows = best
    doc, salts = cells[w]
    head, dom = WR.check_registration(doc["registration"], key, session, params.grid, params.digest, link, direction, w)
    opening = WR.Cells.commit(link, direction, w, WR.window_rows(records[w], params.grid), params.digest, session, salts)
    if opening.root != head.root:
        raise SystemExit(f"window {w}: the warden's rows under its salts aren't its registered root")
    data = [p for p in rows if opening.rows[p][0] in (DATA, END)]
    positions = (data + [p for p in rows if p not in data])[:n]
    return SimpleNamespace(dep=dep, params=params, key=key, session=session, link=link, direction=direction, window=w,
                           connection=q, registration=doc["registration"], cost=doc["cost"], head=head, dom=dom,
                           opening=opening, positions=positions, data_rows=len(data))


def flip(row: bytes, byte: int = -1, bit: int = 0) -> bytes:
    b = bytearray(row)
    b[byte] ^= 1 << bit
    return bytes(b)


def member_rows(name: str, at: SimpleNamespace) -> SimpleNamespace:
    """A member's input rows, its salts for the registered port, the header's `registered` entry and, for ``unsigned``,
    the prover's own tree."""
    pos, op = at.positions, at.opening
    cell = [op.rows[p] for p in pos]
    own = None
    if name == "unsigned":
        rows = list(op.rows)
        rows[pos[0]] = flip(rows[pos[0]])
        own = WR.Cells.commit(at.link, at.direction, at.window, rows, op.parameters, op.session)
        cell[0] = rows[pos[0]]
    elif name == "altered":
        cell[0] = flip(cell[0], bit=1)
    src = own or op
    _words, salts, entry = src.statement(pos)
    return SimpleNamespace(cell=cell, salts=salts, entry=entry, own=own)


def opened(at: SimpleNamespace, mr: SimpleNamespace) -> list[bool]:
    """The host's read of a member's registered port, as Lean's registered reads make it: whether each unit's row,
    committed under the member's salt, climbs its path to the warden's signed root at the unit's position."""
    return [WR.opens(at.dom, at.head.root, p, WR.bc(r, s), bytes.fromhex(path))
            for p, r, s, path in zip(at.positions, mr.cell, mr.salts, mr.entry["paths"])]


# --- the audit (N2-ZK's flow on one-stage's a0, a7 and a8) --------------------------------------------------------------

def _one_stage():
    sys.path.insert(0, str(ONE_STAGE))
    import a0
    import a7
    import a8
    import numpy as np
    from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_flock import class_statement as CS

    return SimpleNamespace(a0=a0, a7=a7, a8=a8, np=np, D=D, P=P, R=R, V=V, canonical_json=canonical_json,
                           encode_program=encode_program, V2=V2, CS=CS)


def stager(M, B: int, n: int, out: Path) -> SimpleNamespace:
    """The statement program and partition (one template, n units) and the unit's stager."""
    defn = bind(WardenCellIdentity, B=B)
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/warden-cell", [("cell", M.V2, 8 * B, ["cell"])], [(M.V2, (0,))],
                         circuit=out / "circuit.txt", bind=bound)
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, program=program, query=query, n=n, base=base, st=st, out=out,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, sg: SimpleNamespace, at: SimpleNamespace, name: str) -> dict:
    """Member `name`'s statement: its input rows, a shared-row file (each instance its own table row), its registered
    port's rows under its salts and paths (`member_rows`), every other row under a fresh salt; and the verifier's own
    `--registered` file at the warden's root."""
    t0 = time.time()
    st, n, out = sg.st, sg.n, sg.out / f"stage-{name}"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {k_log} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    port, _ = MEMBERS[name]
    mr = member_rows(name, at)
    words = M.np.asarray([WR.words(r) for r in mr.cell], dtype=M.np.uint16)
    fresh = [os.urandom(WR.SALT_BYTES) for _ in range(n)]
    salts = {"cell": fresh, "o0": mr.salts} if port == "o0" else {"cell": mr.salts, "o0": fresh}
    set_name = f"{st.iset.name}-{name}"
    got = st.stage({"cell": words}, out, os.urandom(32), salts=salts, indices=range(sg.base, sg.base + n),
                   set_name=set_name, tables={"cell": words}, refs={"cell": list(range(n))}, registered={port: mr.entry})
    outs = M.np.asarray(got["outs"]["o0"], dtype=M.np.uint16)
    if not M.np.array_equal(outs, words):
        raise SystemExit(f"{name}: the identity's outputs are not its inputs")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    reads = WR.reads_file({port: WR.port(at.head, at.positions, sg.base)})
    (out / "registered.json").write_text(json.dumps(reads, sort_keys=True))
    predicted = opened(at, mr)
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name,
         "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"],
         "ports": st.ports, "port": port, "cells": mr.cell, "own": mr.own, "partition_file": sg.partition_file,
         "program_file": sg.program_file, "registered_file": out / "registered.json", "predicted": predicted,
         "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")


def firewall(M, m: dict) -> dict:
    """A firewalled prover's environment (`FC_FIREWALL=1`, its Lean frame named by ``FC_FIREWALL_LEAN``)."""
    if not os.environ.get("FC_FIREWALL_LEAN"):
        raise SystemExit("FC_FIREWALL_LEAN names no Lean firewall frame (flock-verify's flock-firewall)")
    return {"FC_FIREWALL": "1", **M.a0.firewall_env(m)}


def audit_member(M, a, sg, m: dict, at: SimpleNamespace, out: Path, port: int) -> dict:
    """Member m registered, drawn by the verifier's Lean (every unit), proved --zk behind the firewall and held by Lean
    with the verifier's own registered reads. Writes what the verifier received under ``out / "verifier"``."""
    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    n, name = m["n"], m["name"]
    want, roots = {}, []
    fv = m["header"]["frame_v3"]
    for p_, w in a7.frame_ports(m["set"], m["meta"], n).items():
        want[f"{name}/{p_}"] = w
        roots.append({"port": f"{name}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"], "domain": fv["domain_ids"][p_]})
    leaf_layer = hashlib.sha512(M.canonical_json({name: a0.sha512_file(m["dir"] / f"pub-{n}.bin")})).hexdigest()
    law = M.D.parse_law(f"stratified:{n}")
    win = {"kind": "stand-in", "of": f"one cell of each of {n} rows of the warden's egress link-window {at.window} "
                                      "(warden_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": f"registered: port {m['port']} reads {at.head.value} at the warden's signed root"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    receipt = R.receipt(rec, "one-stage-e2e verifier (warden_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False}
    p = a0.run([a.lean, "draw", "--program", str(sg.program_file), "--partition", str(sg.partition_file), "--stratified",
                str(n)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + n)
    sdir = out / "session"
    record, live, proc = M.a8.session(a, m, sdir, local, port, env=firewall(M, m))
    draw_t = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw_t, {"population": n}, {"law": "subset", "k": len(local)})
    verdicts = {f"m0-verifier/{name}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                        and (live or {}).get("roots") == fv["roots"] and v.ok,
                                        "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}}
    lv = a0.lean_verify(a.lean, m, n, sdir, zk=True)
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program --registered"}
    proved = a0.proved_units(m["header"], live or {})
    timing = {x: (live or {}).get(x) for x in TIMING}
    statement = {name: {"session_s": proc["wall_s"], **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest",
                                                                                          "sigma")},
                        "timing": timing, "process": proc, "lean_verify_s": lv.get("verify_s")}}
    sessions = [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), (live or {}).get("instances", 0), statement,
                            verdicts, a.delta, sessions=sessions, served=sorted(m["base"] + u for u in draw_t.get("units", [])),
                            session_roots={f"{name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    (out / "warden-registration.json").write_text(json.dumps(at.registration, indent=1, sort_keys=True))
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt",
              m["dir"] / f"pub-{n}.bin", m["registered_file"], sdir / "draw.json", out / "audit.json",
              out / "warden-registration.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    return {**res, "accepted": bool(result.get("accepted")),
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()}, "why": {x: y.get("why") for x, y in verdicts.items()},
            "registered": lv.get("registered"), "timing": statement[name], "lean_verify_s": lv.get("verify_s")}


def blobs_of(m: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each opened cell and every salt of the statement, all of
    which the prover's instance file holds."""
    blobs = {f"cell {i}": c for i, c in enumerate(m["cells"])}
    blobs.update({f"salt {p} {i}": s for p, ss in m["salts"].items() for i, s in enumerate(ss)})
    return blobs, set(blobs)


def public_pieces(a8, blobs: dict[str, bytes], held: set[str], public: bytes) -> tuple[dict[str, bytes], set[str], list[str]]:
    """`blobs` cut into the scan's aligned pieces (`a8.needles`'s), each distinct piece once under its first name, less
    the pieces `public` holds raw or as hex: the staged circuit, written from the Definition before any instance, and
    the empty slot's zero bytes, which the schedule makes public."""
    pieces: dict[str, bytes] = {}
    seen: set[bytes] = set()
    for name, b in blobs.items():
        for o in range(0, len(b) - a8.PIECE + 1, a8.PIECE):
            x = b[o:o + a8.PIECE]
            if x not in seen:
                seen.add(x)
                pieces[f"{name} @{o}"] = x
    out = {pieces[k] for k in a8.find(public, a8.needles(pieces))}
    kept = {k: x for k, x in pieces.items() if x not in out}
    return kept, {k for k in kept if k.rsplit(" @", 1)[0] in held}, sorted(x.hex() for x in out)


def expected(name: str, res: dict) -> bool:
    """The members to be accepted are; the others are refused by Lean's registered reads."""
    lean = res["verdicts"].get(f"lean-verify/{name}")
    if MEMBERS[name][1]:
        return res["accepted"] and lean is True
    return not res["accepted"] and lean is False and "registered reads" in (res["why"].get(f"lean-verify/{name}") or "")


def prove(a) -> int:
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    at = attestation(run, a.units)
    B, N = at.params.cell_bytes, WR.rows_of(at.params.grid)
    summary: dict = {"run": str(run), "link_window": at.head.value, "connection": at.connection, "rows": N,
                     "cell_bytes": B, "positions": at.positions, "data_rows": at.data_rows,
                     "warden_cost": at.cost, "registration": at.registration}
    summary["warden_costs"] = ws.check_cells(run, at.dep, at.params, ws.read_records(run, at.params))[0]
    own = member_rows("unsigned", at).own
    c = at.registration["certificate"]
    forged = WR.registration(own.head, Certificate(bytes.fromhex(c["key"]), at.link, at.direction, at.window,
                                                   bytes.fromhex(c["digest"]), bytes.fromhex(c["signature"])),
                             at.params.grid)
    try:
        WR.check_registration(forged, at.key, at.session, at.params.grid, at.params.digest, at.link, at.direction,
                              at.window)
        summary["unsigned_registration"] = "accepted"
    except ValueError as e:
        summary["unsigned_registration"] = f"refused: {e}"
    log(f"{at.head.value}: {N} rows of {B} bytes, root {at.head.root.hex()[:16]}, connection {at.connection}'s rows "
        f"{at.positions} ({at.data_rows} data); the warden's cost {at.cost}; the prover's own root under the warden's "
        f"certificate: {summary['unsigned_registration']}")
    M = _one_stage()
    sg = stager(M, B, a.units, out / "stager")
    summary["stager"] = {"lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"], "vus_per_block": sg.st.meta["vus_per_block"],
                         "pin": sg.st.pin, "base": sg.base}
    log(f"stager: {sg.defn.id}, {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
    staged = {}
    for name in MEMBERS:
        m = staged[name] = stage(M, sg, at, name)
        summary.setdefault("members", {})[name] = {"port": m["port"], "ands": m["ands"], "stage_s": m["stage_s"],
                                                   "predicted_opens": m["predicted"]}
        log(f"staged {name}: port {m['port']} registered, {m['ands']} ANDs, rows that open the warden's root (host) "
            f"{sum(m['predicted'])}/{len(m['predicted'])}")
        if all(m["predicted"]) != MEMBERS[name][1]:
            raise SystemExit(f"{name}: the host's read of its rows {m['predicted']} is not the member's")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    ok = summary["unsigned_registration"].startswith("refused")
    for j, (name, m) in enumerate(staged.items()):
        res = audit_member(M, av, sg, m, at, out / name, a.port + j)
        circuit = (m["dir"] / "circuit.txt").read_bytes() + bytes(WR.SALT_BYTES)
        blobs, held, public = public_pieces(M.a8, *blobs_of(m), circuit)
        res["received"] = {**M.a8.received(out / name / "verifier", m["dir"] / f"inst-{m['n']}.bin", blobs, held),
                           "public_pieces": public}
        good = expected(name, res) and not res["received"]["verifier_found"] and res["received"]["control"]["holds_all"]
        ok &= good
        summary["members"][name].update(res)
        log(f"member {name}: verdicts {res['verdicts']}, audit {res['audit']}, Lean: {res['why']}, verifier received "
            f"{res['received']['verifier_bytes']} bytes, witness pieces found there: "
            f"{res['received']['verifier_found'] or 'none'} (of {len(blobs)}; {len(public)} public, left out), control "
            f"{res['received']['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="warden_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="the toy unit's ports registered against the warden's egress root, proved --zk")
    p.add_argument("--run", required=True, type=Path, help="a warden-serve run's directory")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--units", type=int, default=16)
    p.add_argument("--port", type=int, default=7781)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the statements and the host's read of them: nothing proved")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
