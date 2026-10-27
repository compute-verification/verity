"""warden-zk: the warden's bucket rows read by the proof system. A toy unit over one row's bits, its port registered
against the warden's signed egress root (`warden_rows`), proved by C-Flock `prove --zk` behind the Lean firewall and
held by the Lean verifier's registered reads (`flock-verify verify --zk --registered`), through the one-stage audit as
`n2_zk.py` proves N2's head checks.

    python benchmarks/network_traces/warden_zk.py prove --run DIR --bin flock-circuit --lean flock-verify --out DIR
      [--units 16] [--dry]

* **The run.** A warden-serve run's directory (`warden_serve.py rehearse`'s): `deployment.json`, the records, and the
  rows the warden committed and signed while serving, a row a bucket: `cells.jsonl` (each link-window's registration)
  and `cells.salts` (the rows' salts and the wiring's, witness, which the warden hands the prover alone).
* **The registration.** An egress link-window's: its registration read by the verifier
  (`warden_rows.check_registration`, under the deployment's key and session and the schedule), and the warden's
  opening (the record's rows and wiring under the salts) held to its head.
* **The unit** (`WardenRowIdentity{B}`, B a row's S·B bytes): one bucket row's 8·B bits in (`row`), the same bits out
  (`o0`), each through two NOTs (the partition refuses a template with no gate; no AND).
* **The units' wiring is the Program's** (`warden_rows.Units`), fixed from public structure alone, never the cells'
  bytes: the connection admitted in a link-window (its first row past the window's first, or window 0) with the most
  cells, at least D·r + n, and unit i reading its egress cell i, its cell D·r + i (the D buckets before are its
  delay's pads, the schedule's). Each is at its bucket's row, the cell at its slot's words (`warden_rows.Wiring.place`,
  with the warden's wiring, opened to the verifier: the named gap). The verifier's port names the units' digest in
  its value (`warden_rows.port`), so a statement made for any other units' wiring is refused by Lean's registered
  reads, even at the same rows.
* **Members.** One statement of n units each, a shared-row public file whose registered ports are the verifier's own
  (`--registered`, `warden_rows.port` at the warden's root):
  - ``sender``: ``o0`` registered, under the warden's salts and paths: the sender's unit opens its output as the rows;
  - ``receiver``: ``row`` registered: the receiver's check opens the same rows as its input;
  - ``unsigned``: ``o0`` under the prover's own tree over the window's rows, unit 0's cell one byte off: rows the
    warden didn't sign (and the prover's registration of its own root, under the warden's certificate, is refused by
    `check_registration`);
  - ``altered``: ``o0`` under the warden's salts and paths, unit 0's input row one bit off in its cell, so its output
    differs from the warden's row;
  - ``pad``: ``o0`` under the warden's salts and paths, made for units whose unit 0 reads the connection's cell 0, a
    pad of its delay, where the Program's reads its egress cell 0: the header names the pad's units' wiring;
  - ``pad-named``: ``pad``'s rows under the Program's value, which is public: unit 0's row isn't the Program's;
  - ``swapped``: made for units 0 and 1 reading each other's cells, in one row when r > 1, so its rows are the
    Program's: the header names the swapped units' wiring.
  Each is registered, every unit drawn by the verifier's Lean (`flock-verify draw --stratified n`), proved `--zk`
  behind the firewall (``FC_FIREWALL=1``, ``FC_FIREWALL_LEAN``) and held by Lean `verify --zk --partition --program
  --registered`. ``sender`` and ``receiver`` are to be accepted, the others refused by Lean's registered reads.
* **What the verifier receives**, scanned for every 16-byte piece of every opened row and every salt of the
  statement (`a8.received`) but the pieces the public circuit holds, the empty slot's (zero bytes, the schedule's)
  and the pads' (`pad_cells`, (PAD, q) and zeros, the wiring's): none is to be found; the prover's instance file,
  which holds them, is the control.
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

import leak_scan as LS  # noqa: E402
import warden_serve as ws  # noqa: E402
from verity.ml.circuits import boolean as BL  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.ml.circuits.types import Array  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA, PAD  # noqa: E402
from verity.core.protocols.certifier.certificate import Certificate  # noqa: E402
from verity.core.protocols.certifier.warden import END, WardenParams, cell  # noqa: E402

MEMBERS = {"sender": ("o0", True), "receiver": ("row", True), "unsigned": ("o0", False), "altered": ("o0", False),
           "pad": ("o0", False), "pad-named": ("o0", False), "swapped": ("o0", False)}
#: what Lean's registered reads, and the host's read of them, refuse each refused member by
WHY = {"unsigned": "does not open", "altered": "does not open", "pad": "not another value",
       "pad-named": "reads position", "swapped": "not another value"}


def log(msg: str) -> None:
    print(f"WZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the unit ---------------------------------------------------------------------------------------------------------

def _signature(S):
    return (("row", Array(8 * S.B, BL.BIT)),), Array(8 * S.B, BL.BIT)


def _body(B, S, row):
    return B.batch(BL.NOT, B.batch(BL.NOT, row, axes=(0,)), axes=(0,))


WardenRowIdentity = CompositeDefinition(
    "WardenRowIdentity", 1, ("B",), _signature, _body, conformance="exact-model-tested", register=False,
    doc="WardenRowIdentity{B}: one warden bucket row's 8·B bits, each through two NOTs: the toy unit whose port opens "
        "a row of the warden's signed root (warden_zk.py)")


# --- the run ----------------------------------------------------------------------------------------------------------

def attestation(run: Path, n: int) -> SimpleNamespace:
    """The egress link-window to read, its registration as the verifier reads it, the warden's opening, and the
    Program's units' wiring (`units`, the module docstring's), from public structure alone: `reads` its (connection,
    cell) each, the row and slot the warden's wiring places each at (`positions`, `slots`), and how many of them hold
    the server's bytes (`data_rows`, reported, never read to choose)."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    key, session = ws.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    link, direction = ws.CERTIFIED[ws.EGRESS]
    records = {r.window: r for r in ws.read_records(run, params)[ws.EGRESS]}
    cells = {(d["registration"]["window"]): (d, salts, wiring_salt) for d, salts, wiring_salt in ws.read_cells(run)
             if "registration" in d and d["registration"]["direction"] == direction.value}
    pads = params.delay * params.slots
    best = None
    for w, record in sorted(records.items()):
        for q, first, k in WR.wiring(record, params).connections:
            if w in cells and (w == 0 or first > 0) and k >= pads + n and (best is None or k > best[2]):
                best = (w, q, k)
    if best is None:
        raise SystemExit(f"no egress link-window of {run} has a registration and a connection admitted in it with "
                         f"D·r + n = {pads} + {n} cells")
    w, q, k = best
    doc, salts, wiring_salt = cells[w]
    head, dom = WR.check_registration(doc["registration"], key, session, params, link, direction, w)
    opening = WR.Cells.of(link, direction, records[w], params, session, salts, wiring_salt)
    if opening.head != head:
        raise SystemExit(f"window {w}: the warden's rows and wiring under its salts aren't its registered head")
    units = WR.Units.span(q, pads, n)
    placed = opening.wiring.place(params, units.cells)
    B = params.cell_bytes
    return SimpleNamespace(dep=dep, params=params, key=key, session=session, link=link, direction=direction, window=w,
                           connection=q, registration=doc["registration"], cost=doc["cost"], head=head, dom=dom,
                           opening=opening, units=units, pads=pads, reads=list(units.cells),
                           positions=[t for t, _ in placed], slots=[j for _, j in placed],
                           data_rows=sum(opening.rows[t][j * B] in (DATA, END) for t, j in placed))


def cell_of(at: SimpleNamespace, row: bytes, i: int) -> bytes:
    """Unit i's cell in `row`: its slot's B bytes."""
    B = at.params.cell_bytes
    return row[at.slots[i] * B:(at.slots[i] + 1) * B]


def flip(row: bytes, byte: int = -1, bit: int = 0) -> bytes:
    b = bytearray(row)
    b[byte] ^= 1 << bit
    return bytes(b)


def member_units(name: str, at: SimpleNamespace) -> WR.Units:
    """The units' wiring a member's statement is made for: the Program's, but ``pad``'s and ``pad-named``'s, whose
    unit 0 reads the connection's cell 0, a pad of its delay, and ``swapped``'s, whose units 0 and 1 read each other's
    cells."""
    c = at.units.cells
    if name in ("pad", "pad-named"):
        return WR.Units(((at.connection, 0),) + c[1:])
    if name == "swapped":
        return WR.Units((c[1], c[0]) + c[2:])
    return at.units


def member_rows(name: str, at: SimpleNamespace) -> SimpleNamespace:
    """A member's units' wiring, the rows they read (`positions`), its input rows, its salts for the registered port,
    the header's `registered` entry and, for ``unsigned``, the prover's own tree. The flips are in unit 0's cell (its
    slot's last byte). ``pad-named``'s entry names the Program's value."""
    units = member_units(name, at)
    op, last = at.opening, (at.slots[0] + 1) * at.params.cell_bytes - 1
    pos = [t for t, _ in op.wiring.place(at.params, units.cells)]
    own = None
    if name == "unsigned":
        rows = list(op.rows)
        rows[pos[0]] = flip(rows[pos[0]], byte=last)
        own = WR.Cells.commit(at.link, at.direction, at.window, rows, op.wiring, op.params, op.session)
    src = own or op
    unit_rows = [src.rows[p] for p in pos]
    if name == "altered":
        unit_rows[0] = flip(unit_rows[0], byte=last, bit=1)
    _words, salts, entry = src.statement(pos, units)
    if name == "pad-named":
        entry = entry | {"value": at.units.value(at.head)}
    return SimpleNamespace(units=units, positions=pos, rows=unit_rows, salts=salts, entry=entry, own=own)


def opened(at: SimpleNamespace, mr: SimpleNamespace) -> list[bool]:
    """Whether each of a member's rows, committed under its salt, climbs its path to the warden's signed root at the
    position the member's entry names."""
    return [WR.opens(at.dom, at.head.root, p, WR.bc(r, s), bytes.fromhex(path))
            for p, r, s, path in zip(mr.entry["positions"], mr.rows, mr.salts, mr.entry["paths"])]


def refused(at: SimpleNamespace, mr: SimpleNamespace, reads_from: int = 0) -> str | None:
    """The host's read of a member's registered port against the verifier's (`warden_rows.port` of the Program's
    units), as Lean's registered reads make it (`warden_rows.refusal`): None when they accept, else why not."""
    port = WR.port(at.head, at.opening.wiring, at.params, at.units, reads_from)
    return WR.refusal(port, at.dom, mr.entry, [WR.bc(r, s) for r, s in zip(mr.rows, mr.salts)])


# --- the audit (`n2_zk`'s flow on one-stage's a0, a7 and a8) ------------------------------------------------------------

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
    defn = bind(WardenRowIdentity, B=B)
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/warden-row", [("row", M.V2, 8 * B, ["row"])], [(M.V2, (0,))],
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
    words = M.np.asarray([WR.words(r) for r in mr.rows], dtype=M.np.uint16)
    fresh = [os.urandom(WR.SALT_BYTES) for _ in range(n)]
    salts = {"row": fresh, "o0": mr.salts} if port == "o0" else {"row": mr.salts, "o0": fresh}
    set_name = f"{st.iset.name}-{name}"
    got = st.stage({"row": words}, out, os.urandom(32), salts=salts, indices=range(sg.base, sg.base + n),
                   set_name=set_name, tables={"row": words}, refs={"row": list(range(n))}, registered={port: mr.entry})
    outs = M.np.asarray(got["outs"]["o0"], dtype=M.np.uint16)
    if not M.np.array_equal(outs, words):
        raise SystemExit(f"{name}: the identity's outputs are not its inputs")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    reads = WR.reads_file({port: WR.port(at.head, at.opening.wiring, at.params, at.units, sg.base)})
    (out / "registered.json").write_text(json.dumps(reads, sort_keys=True))
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name,
         "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"],
         "ports": st.ports, "port": port, "rows": mr.rows, "own": mr.own, "partition_file": sg.partition_file,
         "program_file": sg.program_file, "registered_file": out / "registered.json", "opens": opened(at, mr),
         "predicted": refused(at, mr, sg.base), "units": mr.units.digest().hex(), "stage_s": round(time.time() - t0, 1)}
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
    win = {"kind": "stand-in", "of": f"the bucket rows of {n} cells of connection {at.connection} in the warden's "
                                      f"egress link-window {at.window} (warden_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "units": {"cells": [list(c) for c in at.units.cells], "digest": at.units.digest().hex()},
           "link": f"registered: port {m['port']} reads {at.units.value(at.head)} at the warden's signed root"}
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
    """The witness pieces the verifier must not receive: each opened row and every salt of the statement, all of
    which the prover's instance file holds."""
    blobs = {f"row {i}": c for i, c in enumerate(m["rows"])}
    blobs.update({f"salt {p} {i}": s for p, ss in m["salts"].items() for i, s in enumerate(ss)})
    return blobs, set(blobs)


def pad_cells(at: SimpleNamespace) -> bytes:
    """Each connection's pad as the warden writes it, (PAD, q) and zeros: fixed by the ordinal, which the wiring
    (opened to the verifier) names, so a 16-byte piece of it matches any small integer among zeros."""
    return b"".join(cell(PAD, q, b"", at.params.cell_bytes) for q, _, _ in at.opening.wiring.connections)


#: The scan's one cutter (`leak_scan.public_pieces`: each row cut whole, less the pieces `public` holds: the staged
#: circuit, the empty slot's zero bytes, which the schedule makes public, and the pads, `pad_cells`).
public_pieces = LS.public_pieces


def expected(name: str, res: dict) -> bool:
    """The members to be accepted are; the others are refused by Lean's registered reads, each by its `WHY`."""
    lean = res["verdicts"].get(f"lean-verify/{name}")
    if MEMBERS[name][1]:
        return res["accepted"] and lean is True
    why = res["why"].get(f"lean-verify/{name}") or ""
    return not res["accepted"] and lean is False and "registered reads" in why and WHY[name] in why


def prove(a) -> int:
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    at = attestation(run, a.units)
    if not at.pads or a.units < 2:
        raise SystemExit("pad reads a pad of the delay and swapped two units: D ≥ 1 and --units ≥ 2")
    B, N = WR.row_bytes(at.params), WR.rows_of(at.params)
    summary: dict = {"run": str(run), "link_window": at.head.value, "value": at.units.value(at.head),
                     "connection": at.connection, "rows": N, "row_bytes": B, "cell_bytes": at.params.cell_bytes,
                     "pads": at.pads, "reads": at.reads, "positions": at.positions, "slots": at.slots,
                     "data_rows": at.data_rows, "units_bytes": len(at.units.encode()), "warden_cost": at.cost,
                     "registration": at.registration}
    summary["warden_costs"] = ws.check_cells(run, at.dep, at.params, ws.read_records(run, at.params))[0]
    own = member_rows("unsigned", at).own
    c = at.registration["certificate"]
    forged = WR.registration(own.head, Certificate(bytes.fromhex(c["key"]), at.link, at.direction, at.window,
                                                   bytes.fromhex(c["digest"]), bytes.fromhex(c["signature"])),
                             at.params)
    try:
        WR.check_registration(forged, at.key, at.session, at.params, at.link, at.direction, at.window)
        summary["unsigned_registration"] = "accepted"
    except ValueError as e:
        summary["unsigned_registration"] = f"refused: {e}"
    log(f"{at.head.value}: {N} rows of {B} bytes, root {at.head.root.hex()[:16]}; the Program's units read connection "
        f"{at.connection}'s cells {at.pads} … {at.pads + a.units - 1} (units {at.units.digest().hex()[:16]}) at rows "
        f"{at.positions}, slots {at.slots} ({at.data_rows} data); the warden's cost {at.cost}; the prover's own root "
        f"under the warden's certificate: {summary['unsigned_registration']}")
    M = _one_stage()
    sg = stager(M, B, a.units, out / "stager")
    summary["stager"] = {"lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"], "vus_per_block": sg.st.meta["vus_per_block"],
                         "pin": sg.st.pin, "base": sg.base}
    log(f"stager: {sg.defn.id}, {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
    staged = {}
    for name in MEMBERS:
        m = staged[name] = stage(M, sg, at, name)
        summary.setdefault("members", {})[name] = {"port": m["port"], "ands": m["ands"], "stage_s": m["stage_s"],
                                                   "units": m["units"], "predicted_opens": m["opens"],
                                                   "predicted": m["predicted"] or "accepted"}
        log(f"staged {name}: port {m['port']} registered, {m['ands']} ANDs, units {m['units'][:16]}, rows that open "
            f"the warden's root (host) {sum(m['opens'])}/{len(m['opens'])}; the host's read: "
            f"{m['predicted'] or 'accepted'}")
        if (m["predicted"] is None) != MEMBERS[name][1] or WHY.get(name, "") not in (m["predicted"] or ""):
            raise SystemExit(f"{name}: the host's read of its statement ({m['predicted'] or 'accepted'}) is not the "
                             "member's")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    ok = summary["unsigned_registration"].startswith("refused")
    for j, (name, m) in enumerate(staged.items()):
        res = audit_member(M, av, sg, m, at, out / name, a.port + j)
        known = (m["dir"] / "circuit.txt").read_bytes() + bytes(WR.SALT_BYTES) + pad_cells(at)
        blobs, held, public = public_pieces(M.a8, *blobs_of(m), known)
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
