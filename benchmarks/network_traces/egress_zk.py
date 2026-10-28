"""egress-zk: N2's egress bucket in zero knowledge, the generator form. Every bucket of one egress link-window computed
by its own `EgressBucket_v1` unit, its computed row, output ``o1``, registered against the warden's signed root
(`warden_rows.bucket_port`: unit base + t reads row t), proved by C-Flock `prove --zk` behind the Lean firewall and held
by the Lean verifier's registered reads (`flock-verify verify --zk --registered`); each unit's one public output, its
16-bit ``o0`` (1 when the state renders a row), opened to the verifier. Standalone: no Spec holds these units yet
(`verity.core.protocols.certifier.egress`).

    python benchmarks/network_traces/egress_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]

* **The link** (`PARAMS`, `NR`, `K`): a fake upstream's responses served through the warden's own egress
  (`egress_traffic.busy`, the egress tests'), each link-window's rows committed by the warden and its head signed by a
  test-size LMS key (`warden_rows.attest`). The window read is the one with the most `DATA` cells; its registration
  is read by the verifier (`warden_rows.check_registration`) before its port is built from the head it names.
* **The units** (`EgressBucket_v1{B=256,R=8,r=2,NR=4,K=16,IB=18,D=2}`, circuit-check's pinned shape), one a bucket, T
  of them: each reads row ``state`` (the egress view of the slot state the scan's ingress step wrote at its bucket,
  `egress_traffic.states`; a hidden row under a fresh salt here, where the scan registers it), and computes ``o0``, 1
  when the state renders a row, and ``o1``, the row, committed under the salt of the warden's row t and registered as
  a read of it, so the proof opens the warden's row only when the row the state renders is it.
* **Members.** ``honest``: accepted, every ``o0`` 1. ``lying``: one rendered output id of one bucket's state changed:
  the unit computes a row the warden didn't sign, so its ``o1`` opens no registered row: refused by Lean ("does not
  open"). ``moved``: units t and t + 1 given each other's state, their positions and salts swapped to match, so each
  ``o1`` opens at the other unit's position: refused by Lean ("reads position"). Each is registered, every unit drawn
  by the verifier's Lean (`flock-verify draw --stratified n`), proved ``--zk`` behind the firewall (``FC_FIREWALL=1``)
  and held by Lean `verify --zk --partition --program --registered`; the verifier opens each held unit's ``o0`` under
  the salt the prover sends for it.
* **What the verifier receives** is scanned for every 16-byte piece of every unit's state and row and every salt but
  ``o0``'s (`a8.received`), less the pieces public anyway: the staged circuit, zero bytes, and the pad and `END`
  cells of the window's ordinals (a piece of them is a small integer among zeros). The prover's instance file, which
  holds them all, is the control; it lays the ``state`` table's 204-byte rows back to back, each followed by the next
  rather than by zeros, so a state's zero-padded last chunk isn't held there (`leak_scan.public_pieces`' ``fields``).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import warden_zk as WZ  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity.core.protocols.certifier import egress as EG  # noqa: E402
from verity.core.protocols.certifier import io_tie as IT, warden as W, warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsSigner  # noqa: E402

_TRAFFIC = HERE.parents[1] / "verity/core/protocols/certifier/tests/egress_traffic.py"
_spec = importlib.util.spec_from_file_location("egress_traffic", _TRAFFIC)
traffic = sys.modules["egress_traffic"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traffic)

PARAMS = W.WardenParams(cell_bytes=256, slots=2, delay=2, lanes=4, window=6)
NR, K = 4, 16
SHAPE = traffic.shape(PARAMS, NR, K)
LINK, SESSION = "responses", bytes(range(32))
#: the fake upstream: (seed, connections, longest response) of `egress_traffic.busy`, over `WINDOWS` link-windows
SEED, CONNECTIONS, LONGEST, WINDOWS = 6, 14, 15, 6
#: each member's outcome: accepted, or refused by Lean's registered reads
MEMBERS = {"honest": "accepted", "lying": "does not open", "moved": "reads position"}


def log(msg: str) -> None:
    print(f"EZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the link ---------------------------------------------------------------------------------------------------------

def _data(row: bytes) -> int:
    return sum(row[s * SHAPE.B] == DATA for s in range(SHAPE.R))


def link(out: Path) -> SimpleNamespace:
    """The link-window to prove: the warden's opening and its signed registration, read back by the verifier (the
    head and domain the port is built on), and each bucket's state (`egress_traffic.states`)."""
    served = traffic.busy(SEED, PARAMS, CONNECTIONS, LONGEST, WINDOWS)
    T = PARAMS.window
    states, rows = traffic.states(served, NR, K), traffic.rows(served)
    w = max(range(len(served.records)), key=lambda w: sum(_data(r) for r in rows[w * T:w * T + T]))
    out.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"egress-zk").digest()),
                              out / "signer-state.json")
    cells, cert = WR.attest(Issuer(signer, SESSION), served.records[w], PARAMS, Direction.EGRESS)
    if cert is None:
        raise SystemExit("the test signer refused the link-window")
    registration = WR.registration(cells.head, cert, PARAMS)
    head, dom = WR.check_registration(registration, signer.public_key(), SESSION, PARAMS, LINK, Direction.EGRESS, w)
    window = states[w * T:w * T + T]
    if [EG.rendered(SHAPE, s) for s in window] != list(cells.rows):
        raise SystemExit(f"window {w}: the states don't render the warden's rows")
    ordinals = sorted({s.q for st in window for s in st.slots if s.live})
    return SimpleNamespace(window=w, states=window, cells=cells, registration=registration, head=head, dom=dom,
                           ordinals=ordinals)


def _changed_state(at: SimpleNamespace) -> tuple[int, EG.State, dict]:
    """The first bucket with a `DATA` cell, its state with that cell's output id changed."""
    for t, st in enumerate(at.states):
        live = [EG.Request(s.q, s.admitted, s.n) if s.live else None for s in st.slots]
        records, _ = EG.bucket_map(st.b, live, SHAPE.R, SHAPE.r, SHAPE.D)
        for m in records:
            if m.e < K:
                ids = list(st.slots[m.j].ids)
                ids[m.e] = (ids[m.e] + 1) % (1 << SHAPE.IB)
                slots = list(st.slots)
                slots[m.j] = slots[m.j]._replace(ids=tuple(ids))
                return t, st._replace(slots=tuple(slots)), {"bucket": t, "slot": m.j, "output": m.e}
    raise SystemExit("the window renders no DATA cell")


def member_rows(name: str, at: SimpleNamespace) -> SimpleNamespace:
    """A member's units' state, in unit order, the row each computes (`egress.outputs`, its ``o1``), the salts its
    ``o1`` rows are committed under, the header's `registered` entry (the warden's positions and paths for the rows
    it names), and each unit's ``o0``."""
    cells, T = at.cells, PARAMS.window
    states, order, changed = list(at.states), list(range(T)), {}
    if name == "lying":
        t, states[t], changed = _changed_state(at)
    if name == "moved":
        t = next(t for t in range(T - 1) if cells.rows[t] != cells.rows[t + 1])
        order[t], order[t + 1] = t + 1, t
        changed = {"units": [t, t + 1]}
    states = [states[b] for b in order]
    _words, salts, entry = cells.statement(order)
    got = [EG.outputs(SHAPE, s) for s in states]
    return SimpleNamespace(states=states, rows=[r for _o, r in got], words=[EG.state_words(SHAPE, s) for s in states],
                           salts=salts, entry=entry, order=order, changed=changed, verdicts=[o for o, _r in got])


def refused(at: SimpleNamespace, mr: SimpleNamespace, reads_from: int) -> str | None:
    """The host's read of a member's registered ``o1`` port against the verifier's own (`warden_rows.bucket_port`),
    as Lean's registered reads make it (`warden_rows.refusal`): None when they accept, else why not."""
    port = WR.bucket_port(at.head, PARAMS, reads_from)
    return WR.refusal(port, at.dom, mr.entry, [WR.bc(r, s) for r, s in zip(mr.rows, mr.salts)])


# --- the statement ----------------------------------------------------------------------------------------------------

def stager(M, n: int, out: Path) -> SimpleNamespace:
    """The statement program and partition (one template, a unit a bucket) and the unit's stager: row ``state`` in,
    rows ``o0`` and ``o1`` out."""
    sh = SHAPE
    defn = bind(EG.EgressBucket, B=sh.B, R=sh.R, r=sh.r, NR=sh.NR, K=sh.K, IB=sh.IB, D=sh.D)
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/egress-bucket", [("state", M.V2, sh.state_bits, ["state"])],
                         [(M.V2, (0,)), (M.V2, (1,))], circuit=out / "circuit.txt", bind=bound)
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, program=program, query=query, n=n, base=base, st=st, out=out,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def opened(M, m: dict, units) -> list[int | None]:
    """The verifier's side of each unit's verdict: the word the prover sends, opened under the salt it sends against
    the ``o0`` commitment the public file holds; None where it doesn't open or the unit isn't held."""
    port = next(p for p in m["ports"] if p.name == "o0")
    return [w if i in units and M.a7.opens(port, [w], m["salts"]["o0"][i], verdict_bcs(m)[i]) else None
            for i, w in enumerate(m["verdicts"])]


def verdict_bcs(m: dict) -> list[bytes]:
    """Each unit's ``o0`` b‖c from the public file: a shared-row file holds them port-major (each port's table rows in
    turn, `shared_rows` counting them), and unit i's ``o0`` row is its table row i."""
    from verity_flock.circuit import BC_BYTES as w

    head, body = (m["dir"] / f"pub-{m['n']}.bin").read_bytes().split(b"\n", 1)
    rows = json.loads(head)["shared_rows"]
    at = sum(rows[p.name] for p in m["ports"][:[p.name for p in m["ports"]].index("o0")])
    return [body[(at + i) * w:(at + i + 1) * w] for i in range(m["n"])]


def stage(M, sg: SimpleNamespace, at: SimpleNamespace, name: str) -> dict:
    """Member `name`'s statement: its units' ``state`` rows in a shared-row file (each instance its own table row),
    ``state`` and ``o0`` under fresh salts and ``o1`` under the salts and paths of its registered entry; and the
    verifier's own `--registered` file at the warden's root. The units' outputs are the reference's."""
    t0 = time.time()
    st, n, out = sg.st, sg.n, sg.out / f"stage-{name}"
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= st.meta["k_log"] + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {st.meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX}")
    mr = member_rows(name, at)
    words = M.np.asarray(mr.words, dtype=M.np.uint16)
    set_name = f"{st.iset.name}-{name}"
    fresh = {p: [os.urandom(WR.SALT_BYTES) for _ in range(n)] for p in ("state", "o0")}
    got = st.stage({"state": words}, out, os.urandom(32), salts={**fresh, "o1": mr.salts},
                   indices=range(sg.base, sg.base + n), set_name=set_name, tables={"state": words},
                   refs={"state": list(range(n))}, registered={"o1": mr.entry})
    o0 = [int(x[0]) for x in M.np.asarray(got["outs"]["o0"])]
    if o0 != mr.verdicts:
        raise SystemExit(f"{name}: the units' o0 {o0} are not the reference's {mr.verdicts}")
    if [M.np.asarray(x, dtype="<u2").tobytes() for x in M.np.asarray(got["outs"]["o1"])] != mr.rows:
        raise SystemExit(f"{name}: a unit's o1 row is not the reference's")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    reads = WR.reads_file({"o1": WR.bucket_port(at.head, PARAMS, sg.base)})
    (out / "registered.json").write_text(json.dumps(reads, sort_keys=True))
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512,
         "set": set_name, "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
         "salts": got["salts"], "ports": st.ports, "port": "o1", "rows": mr.rows, "words": mr.words,
         "verdicts": mr.verdicts, "partition_file": sg.partition_file, "program_file": sg.program_file,
         "registered_file": out / "registered.json", "predicted": refused(at, mr, sg.base), "changed": mr.changed,
         "order": mr.order, "stage_s": round(time.time() - t0, 1)}
    if opened(M, m, range(n)) != mr.verdicts:
        raise SystemExit(f"{name}: a unit's verdict doesn't open against its public o0 commitment")
    m["header"] = M.a0.public_header(out, n)
    return m


# --- the audit (warden_zk's, on a bucket port) ------------------------------------------------------------------------

def audit_member(M, a, sg, m: dict, at: SimpleNamespace, out: Path, port: int) -> dict:
    """Member m registered, drawn by the verifier's Lean (every unit), proved --zk behind the firewall and held by Lean
    with the verifier's own registered reads; then each held unit's verdict opened. Writes what the verifier received
    under ``out / "verifier"``."""
    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    n, name = m["n"], m["name"]
    want, roots = {}, []
    fv = m["header"]["frame_v3"]
    for p_, w in a7.frame_ports(m["set"], m["meta"], n).items():
        want[f"{name}/{p_}"] = w
        roots.append({"port": f"{name}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"],
                      "domain": fv["domain_ids"][p_]})
    leaf_layer = hashlib.sha512(M.canonical_json({name: a0.sha512_file(m["dir"] / f"pub-{n}.bin")})).hexdigest()
    law = M.D.parse_law(f"stratified:{n}")
    win = {"kind": "stand-in", "of": f"the {n} egress bucket rows of the warden's link-window {at.window} "
                                      "(egress_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": f"registered: port row reads {at.head.value} at the warden's signed root, unit base + t row t"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want,
                                      leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    receipt = R.receipt(rec, "one-stage-e2e verifier (egress_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False, "verdicts": {}, "why": {}, "opened": []}
    p = a0.run([a.lean, "draw", "--program", str(sg.program_file), "--partition", str(sg.partition_file),
                "--stratified", str(n)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + n)
    sdir = out / "session"
    record, live, proc = M.a8.session(a, m, sdir, local, port, env=WZ.firewall(M, m))
    draw_t = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw_t, {"population": n}, {"law": "subset", "k": len(local)})
    verdicts = {f"m0-verifier/{name}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                        and (live or {}).get("roots") == fv["roots"] and v.ok,
                                        "why": ((record or {}).get("verdict") or {}).get("why"),
                                        "draw_check": v.to_json()}}
    lv = a0.lean_verify(a.lean, m, n, sdir, zk=True)
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program "
                                                     "--registered"}
    proved = a0.proved_units(m["header"], live or {})
    timing = {x: (live or {}).get(x) for x in WZ.TIMING}
    statement = {name: {"session_s": proc["wall_s"], **{x: (live or {}).get(x) for x in ("circuit_sha512",
                                                                                          "statement_digest",
                                                                                          "sigma")},
                        "timing": timing, "process": proc, "lean_verify_s": lv.get("verify_s")}}
    sessions = [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), (live or {}).get("instances", 0),
                            statement, verdicts, a.delta, sessions=sessions,
                            served=sorted(m["base"] + u for u in draw_t.get("units", [])),
                            session_roots={f"{name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    held = {u - m["base"] for u in proved} if result.get("accepted") else set()
    words = opened(M, m, held)
    sent = [{"unit": m["base"] + i, "verdict": w, "salt/o0": m["salts"]["o0"][i].hex()} for i, w in enumerate(words)
            if i in held]
    (out / "verdicts.json").write_text(json.dumps(sent, indent=1))
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    (out / "warden-registration.json").write_text(json.dumps(at.registration, indent=1, sort_keys=True))
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt",
              m["dir"] / f"pub-{n}.bin", m["registered_file"], sdir / "draw.json", out / "audit.json",
              out / "warden-registration.json", out / "verdicts.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    window_ok = bool(result.get("accepted")) and len(held) == n and all(w == EG.ACCEPT for w in words)
    return {**res, "accepted": bool(result.get("accepted")), "window_accepted": window_ok, "opened": words,
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()},
            "why": {x: y.get("why") for x, y in verdicts.items()}, "registered": lv.get("registered"),
            "timing": statement[name], "lean_verify_s": lv.get("verify_s")}


def blobs_of(m: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each unit's state and row (its ``o1``) and every salt of the
    statement but ``o0``'s, all of which the prover's instance file holds."""
    blobs = {f"row {i}": r for i, r in enumerate(m["rows"])}
    blobs.update({f"state {i}": b"".join(w.to_bytes(2, "little") for w in ws) for i, ws in enumerate(m["words"])})
    blobs.update({f"salt {p} {i}": s for p, ss in m["salts"].items() if p != "o0" for i, s in enumerate(ss)})
    return blobs, set(blobs)


def known(sg: SimpleNamespace, at: SimpleNamespace) -> bytes:
    """What the scan leaves out (the module docstring's): the circuit, zeros, and the window's pads and ENDs."""
    B = PARAMS.cell_bytes
    cells = [IT.pad_cell(B, q) + IT.end_cell(q, B) for q in [0, *at.ordinals]]
    return (sg.out / "circuit.txt").read_bytes() + bytes(WR.SALT_BYTES) + b"".join(cells)


def expected(name: str, res: dict) -> bool:
    """The honest member's window is accepted; the others are refused by Lean's registered reads, each by its
    `MEMBERS` reason."""
    lean = res["verdicts"].get(f"lean-verify/{name}")
    if MEMBERS[name] == "accepted":
        return res["window_accepted"] and lean is True
    why = res["why"].get(f"lean-verify/{name}") or ""
    return not res["accepted"] and lean is False and "registered reads" in why and MEMBERS[name] in why


def predicted(name: str, m: dict) -> bool:
    """The host's read of a member's statement before any proof: its registered ``o1`` port, and every unit's ``o0``
    1 (each member's states render rows)."""
    if set(m["verdicts"]) != {EG.ACCEPT}:
        return False
    if MEMBERS[name] == "accepted":
        return m["predicted"] is None
    return MEMBERS[name] in (m["predicted"] or "")


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    at = link(out / "warden")
    T, n = PARAMS.window, PARAMS.window
    rows = list(at.cells.rows)
    summary: dict = {"params": {"cell_bytes": PARAMS.cell_bytes, "slots": PARAMS.slots, "delay": PARAMS.delay,
                                "lanes": PARAMS.lanes, "window": T, "R": SHAPE.R, "NR": NR, "K": K, "IB": SHAPE.IB,
                                "state_bits": SHAPE.state_bits},
                     "traffic": {"seed": SEED, "connections": CONNECTIONS, "longest": LONGEST, "windows": WINDOWS},
                     "link_window": at.head.value, "window": at.window, "rows": T, "row_bytes": WR.row_bytes(PARAMS),
                     "cells": {"data": sum(_data(r) for r in rows),
                               "end": sum(r[s * SHAPE.B] == W.END for r in rows for s in range(SHAPE.R))},
                     "ordinals": at.ordinals, "registration": at.registration}
    log(f"{at.head.value}: {T} rows of {WR.row_bytes(PARAMS)} bytes, root {at.head.root.hex()[:16]}, cells "
        f"{summary['cells']}, ordinals {at.ordinals}; registration read under the warden's key")
    M = WZ._one_stage()
    sg = stager(M, n, out / "stager")
    summary["stager"] = {"unit": sg.defn.id, "lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"],
                         "vus_per_block": sg.st.meta["vus_per_block"], "pin": sg.st.pin, "base": sg.base}
    log(f"stager: {sg.defn.id}, {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
    staged = {}
    for name in MEMBERS:
        m = staged[name] = stage(M, sg, at, name)
        summary.setdefault("members", {})[name] = {"ands": m["ands"], "stage_s": m["stage_s"], "changed": m["changed"],
                                                   "order": m["order"], "verdicts": m["verdicts"],
                                                   "predicted": m["predicted"] or "accepted"}
        log(f"staged {name}: {m['ands']} ANDs a unit, {m['changed'] or 'unchanged'}, o0 {m['verdicts']}; the "
            f"host's read of the o1 port: {m['predicted'] or 'accepted'}")
        if not predicted(name, m):
            raise SystemExit(f"{name}: the host's read of its statement is not the member's ({MEMBERS[name]})")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    ok = True
    for j, (name, m) in enumerate(staged.items()):
        res = audit_member(M, av, sg, m, at, out / name, a.port + j)
        blobs, held, public = WZ.public_pieces(M.a8, *blobs_of(m), known(sg, at), fields=True)
        res["received"] = {**M.a8.received(out / name / "verifier", m["dir"] / f"inst-{m['n']}.bin", blobs, held),
                           "public_pieces": public}
        good = expected(name, res) and not res["received"]["verifier_found"] and res["received"]["control"]["holds_all"]
        ok &= good
        summary["members"][name].update(res)
        log(f"member {name}: proofs {res['verdicts']}, audit {res.get('audit')}, opened verdicts {res['opened']}, "
            f"window {'accepted' if res['window_accepted'] else 'refused'}, Lean: {res['why']}, verifier received "
            f"{res['received']['verifier_bytes']} bytes, witness pieces found there: "
            f"{res['received']['verifier_found'] or 'none'} (of {len(blobs)}; {len(public)} public, left out), control "
            f"{res['received']['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="egress_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="an egress link-window's EgressBucket units, the o1 port registered, proved --zk")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7791)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the statements and the host's read of them: nothing proved")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
