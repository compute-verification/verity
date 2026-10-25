"""egress-zk: N2's egress encoder in zero knowledge. Every bucket of one egress link-window rendered by its own
`EgressBucket_v1` unit, its row ``o0`` registered against the warden's signed root (`warden_rows.bucket_port`: unit
base + t reads row t), proved by C-Flock `prove --zk` behind the Lean firewall and held by the Lean verifier's
registered reads (`flock-verify verify --zk --registered`), as `warden_zk.py` proves its toy unit. Standalone: no
Spec holds these units yet (`verity.core.protocols.certifier.egress`).

    python benchmarks/network_traces/egress_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]

* **The link** (`PARAMS`, `NR`, `K`): a fake upstream's responses served through the warden's own egress
  (`egress_traffic.busy`, the egress tests'), each link-window's rows committed by the warden and its head signed by a
  test-size LMS key (`warden_rows.attest`). The window read is the one with the most `DATA` records; its registration
  is read by the verifier (`warden_rows.check_registration`) before its port is built from the head it names.
* **The units** (`EgressBucket_v1{B=256,R=8,NR=16,K=16}`, circuit-check's pinned shape), one a bucket, T of them:
  each reads one row ``wit`` (its map, routed entries and ready bits, witness) and writes ``o0``, its bucket's row,
  registered, and ``o1``, presence. The routed entries are `EgressRoute`'s on the prover's switches, here by its
  reference (`egress.routed`); the router isn't in this statement.
* **Members.** ``honest``: ``o0`` under the warden's salts and paths: accepted. ``poisoned``: one `DATA` record of a
  bucket naming the next output, so its slot renders `POISON`·B and its row opens no row the warden signed: refused
  ("does not open"). ``moved``: units t and t + 1 given each other's inputs, their rows' positions and salts swapped to
  match, so each row opens, at the other unit's position: refused ("reads position"). Each is registered, every unit
  drawn by the verifier's Lean (`flock-verify draw --stratified n`), proved ``--zk`` behind the firewall
  (``FC_FIREWALL=1``) and held by Lean `verify --zk --partition --program --registered`.
* **What the verifier receives** is scanned for every 16-byte piece of every unit's input and output row and every
  salt (`a8.received`) but the pieces public anyway: the staged circuit, zero bytes, the poison's, and the pad and
  `END` cells of the window's ordinals (a piece of them is a small integer among zeros). The prover's instance file,
  which holds them all, is the control.
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
from verity.core.protocols.certifier import egress as EG, io_tie as IT, warden as W, warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsSigner  # noqa: E402

_TRAFFIC = HERE.parents[1] / "verity/core/protocols/certifier/tests/egress_traffic.py"
_spec = importlib.util.spec_from_file_location("egress_traffic", _TRAFFIC)
traffic = sys.modules["egress_traffic"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traffic)

PARAMS = W.WardenParams(cell_bytes=256, slots=2, delay=2, lanes=4, window=6)
NR, K = 16, 16
LINK, SESSION = "responses", bytes(range(32))
#: the fake upstream: (seed, connections, longest response) of `egress_traffic.busy`, over `WINDOWS` link-windows
SEED, CONNECTIONS, LONGEST, WINDOWS = 6, 14, 15, 6
MEMBERS = {"honest": True, "poisoned": False, "moved": False}
#: what Lean's registered reads, and the host's read of them, refuse each refused member by
WHY = {"poisoned": "does not open", "moved": "reads position"}


def log(msg: str) -> None:
    print(f"EZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the link ---------------------------------------------------------------------------------------------------------

def link(out: Path) -> SimpleNamespace:
    """The link-window to prove: the warden's opening and its signed registration, read back by the verifier (the
    head and domain the port is built on), and the encoder's inputs (`egress_traffic.window`)."""
    served = traffic.busy(SEED, PARAMS, CONNECTIONS, LONGEST, WINDOWS)
    wins = [traffic.window(served, w, NR, K) for w in range(len(served.records))]
    w = max(range(len(wins)), key=lambda w: sum(m.e < K for ms in wins[w].maps for m in ms))
    win = wins[w]
    out.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"egress-zk").digest()),
                              out / "signer-state.json")
    cells, cert = WR.attest(Issuer(signer, SESSION), served.records[w], PARAMS, Direction.EGRESS)
    if cert is None:
        raise SystemExit("the test signer refused the link-window")
    registration = WR.registration(cells.head, cert, PARAMS)
    head, dom = WR.check_registration(registration, signer.public_key(), SESSION, PARAMS, LINK, Direction.EGRESS, w)
    rendered = [traffic.row(win, t) for t in range(PARAMS.window)]
    if rendered != list(cells.rows):
        raise SystemExit(f"window {w}: the encoder's reference rows are not the warden's")
    return SimpleNamespace(window=w, win=win, cells=cells, registration=registration, head=head, dom=dom,
                           ordinals=sorted({s.q for s in win.slots if s is not None}))


def _wit(inputs: list[list[int]]) -> list[int]:
    """A unit's arguments, concatenated in the signature's order, as the row's u16 words (bit i of word k its bit
    16k + i)."""
    bits = [b for arg in inputs for b in arg]
    bits += [0] * (-len(bits) % 16)
    return [sum(b << i for i, b in enumerate(bits[k:k + 16])) for k in range(0, len(bits), 16)]


def member_rows(name: str, at: SimpleNamespace) -> SimpleNamespace:
    """A member's units' inputs and output rows, the salts its registered ``o0`` rows are committed under, and the
    header's `registered` entry (the warden's positions and paths for the rows it names)."""
    win, cells, T = at.win, at.cells, PARAMS.window
    maps = [list(m) for m in win.maps]
    order = list(range(T))
    changed: dict = {}
    if name == "poisoned":
        t, s = next((t, s) for t in range(T) for s, m in enumerate(maps[t]) if m.e < K)
        maps[t][s] = maps[t][s]._replace(e=(maps[t][s].e + 1) % K)
        changed = {"bucket": t, "slot": s, "e": [win.maps[t][s].e, maps[t][s].e]}
    if name == "moved":
        t = next(t for t in range(T - 1) if cells.rows[t] != cells.rows[t + 1])
        order[t], order[t + 1] = t + 1, t
        changed = {"units": [t, t + 1]}
    inputs = [EG.bucket_arguments(NR, maps[b], win.routed[b], win.ready[b]) for b in order]
    rows = [EG.bucket_row(PARAMS.cell_bytes, NR, K, maps[b], win.routed[b], win.ready[b]) for b in order]
    _words, salts, entry = cells.statement(order)
    return SimpleNamespace(inputs=inputs, wit=[_wit(x) for x in inputs], rows=rows, salts=salts, entry=entry,
                           order=order, changed=changed)


def refused(at: SimpleNamespace, mr: SimpleNamespace, reads_from: int) -> str | None:
    """The host's read of a member's registered port against the verifier's own (`warden_rows.bucket_port`), as
    Lean's registered reads make it (`warden_rows.refusal`): None when they accept, else why not."""
    port = WR.bucket_port(at.head, PARAMS, reads_from)
    return WR.refusal(port, at.dom, mr.entry, [WR.bc(r, s) for r, s in zip(mr.rows, mr.salts)])


# --- the statement ----------------------------------------------------------------------------------------------------

def stager(M, n: int, out: Path) -> SimpleNamespace:
    """The statement program and partition (one template, a unit a bucket) and the unit's stager: row ``wit`` in, rows
    ``o0`` and ``o1`` out."""
    defn = bind(EG.EgressBucket, B=PARAMS.cell_bytes, R=PARAMS.grid.allocation, NR=NR, K=K)
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    R = PARAMS.grid.allocation
    bits = R * 3 * 32 + R * 2 * 32 + 16 * EG.ready_words(NR)
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/egress-bucket", [("wit", M.V2, bits, ["map", "routed", "ready"])],
                         [(M.V2, (0,)), (M.V2, (1,))], circuit=out / "circuit.txt", bind=bound)
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, program=program, query=query, n=n, base=base, st=st, out=out,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, sg: SimpleNamespace, at: SimpleNamespace, name: str) -> dict:
    """Member `name`'s statement: its units' ``wit`` rows in a shared-row file (each instance its own table row), the
    ``o0`` rows under the salts and paths of its registered entry, the rest under fresh salts; and the verifier's own
    `--registered` file at the warden's root."""
    t0 = time.time()
    st, n, out = sg.st, sg.n, sg.out / f"stage-{name}"
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= st.meta["k_log"] + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {st.meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX}")
    mr = member_rows(name, at)
    wit = M.np.asarray(mr.wit, dtype=M.np.uint16)
    set_name = f"{st.iset.name}-{name}"
    got = st.stage({"wit": wit}, out, os.urandom(32), salts={"wit": [os.urandom(WR.SALT_BYTES) for _ in range(n)],
                                                             "o0": mr.salts},
                   indices=range(sg.base, sg.base + n), set_name=set_name, tables={"wit": wit},
                   refs={"wit": list(range(n))}, registered={"o0": mr.entry})
    o0 = M.np.asarray(got["outs"]["o0"], dtype=M.np.uint16)
    if not M.np.array_equal(o0, M.np.asarray([WR.words(r) for r in mr.rows], dtype=M.np.uint16)):
        raise SystemExit(f"{name}: the units' o0 rows are not the encoder's reference rows")
    if not (M.np.asarray(got["outs"]["o1"]) == 1).all():
        raise SystemExit(f"{name}: a unit's presence word is not 1")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    reads = WR.reads_file({"o0": WR.bucket_port(at.head, PARAMS, sg.base)})
    (out / "registered.json").write_text(json.dumps(reads, sort_keys=True))
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512,
         "set": set_name, "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
         "salts": got["salts"], "ports": st.ports, "port": "o0", "rows": mr.rows, "wit": mr.wit,
         "partition_file": sg.partition_file, "program_file": sg.program_file,
         "registered_file": out / "registered.json", "predicted": refused(at, mr, sg.base), "changed": mr.changed,
         "order": mr.order, "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


# --- the audit (warden_zk's, on a bucket port) ------------------------------------------------------------------------

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
        roots.append({"port": f"{name}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"],
                      "domain": fv["domain_ids"][p_]})
    leaf_layer = hashlib.sha512(M.canonical_json({name: a0.sha512_file(m["dir"] / f"pub-{n}.bin")})).hexdigest()
    law = M.D.parse_law(f"stratified:{n}")
    win = {"kind": "stand-in", "of": f"the {n} egress bucket rows of the warden's link-window {at.window} "
                                      "(egress_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": f"registered: port o0 reads {at.head.value} at the warden's signed root, unit base + t row t"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want,
                                      leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    receipt = R.receipt(rec, "one-stage-e2e verifier (egress_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False, "verdicts": {}, "why": {}}
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
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()},
            "why": {x: y.get("why") for x, y in verdicts.items()}, "registered": lv.get("registered"),
            "timing": statement[name], "lean_verify_s": lv.get("verify_s")}


def blobs_of(m: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each unit's input and output row and every salt of the
    statement, all of which the prover's instance file holds."""
    blobs = {f"row {i}": r for i, r in enumerate(m["rows"])}
    blobs.update({f"wit {i}": b"".join(w.to_bytes(2, "little") for w in ws) for i, ws in enumerate(m["wit"])})
    blobs.update({f"salt {p} {i}": s for p, ss in m["salts"].items() for i, s in enumerate(ss)})
    return blobs, set(blobs)


def known(sg: SimpleNamespace, at: SimpleNamespace) -> bytes:
    """What the scan leaves out (the module docstring's): the circuit, zeros, the poison, and the window's pads and
    ENDs."""
    B = PARAMS.cell_bytes
    cells = [IT.pad_cell(B, q) + IT.end_cell(q, B) for q in [0, *at.ordinals]]
    return ((sg.out / "circuit.txt").read_bytes() + bytes(WR.SALT_BYTES) + bytes([EG.POISON]) * WR.SALT_BYTES
            + b"".join(cells))


def expected(name: str, res: dict) -> bool:
    """The member to be accepted is; the others are refused by Lean's registered reads, each by its `WHY`."""
    lean = res["verdicts"].get(f"lean-verify/{name}")
    if MEMBERS[name]:
        return res["accepted"] and lean is True
    why = res["why"].get(f"lean-verify/{name}") or ""
    return not res["accepted"] and lean is False and "registered reads" in why and WHY[name] in why


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    at = link(out / "warden")
    T, n = PARAMS.window, PARAMS.window
    summary: dict = {"params": {"cell_bytes": PARAMS.cell_bytes, "slots": PARAMS.slots, "delay": PARAMS.delay,
                                "lanes": PARAMS.lanes, "window": T, "R": PARAMS.grid.allocation, "NR": NR, "K": K},
                     "traffic": {"seed": SEED, "connections": CONNECTIONS, "longest": LONGEST, "windows": WINDOWS},
                     "link_window": at.head.value, "window": at.window, "rows": T, "row_bytes": WR.row_bytes(PARAMS),
                     "records": {"data": sum(m.e < K for ms in at.win.maps for m in ms),
                                 "end": sum(m.e == IT.END_ENTRY for ms in at.win.maps for m in ms),
                                 "held": sum(m.e != IT.EMPTY for ms in at.win.maps for m in ms)},
                     "ordinals": at.ordinals, "registration": at.registration}
    log(f"{at.head.value}: {T} rows of {WR.row_bytes(PARAMS)} bytes, root {at.head.root.hex()[:16]}, records "
        f"{summary['records']}, ordinals {at.ordinals}; registration read under the warden's key")
    M = WZ._one_stage()
    sg = stager(M, n, out / "stager")
    summary["stager"] = {"unit": sg.defn.id, "lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"],
                         "vus_per_block": sg.st.meta["vus_per_block"], "pin": sg.st.pin, "base": sg.base}
    log(f"stager: {sg.defn.id}, {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
    staged = {}
    for name in MEMBERS:
        m = staged[name] = stage(M, sg, at, name)
        summary.setdefault("members", {})[name] = {"ands": m["ands"], "stage_s": m["stage_s"], "changed": m["changed"],
                                                   "order": m["order"], "predicted": m["predicted"] or "accepted"}
        log(f"staged {name}: {m['ands']} ANDs a unit, {m['changed'] or 'unchanged'}; the host's read: "
            f"{m['predicted'] or 'accepted'}")
        if (m["predicted"] is None) != MEMBERS[name] or WHY.get(name, "") not in (m["predicted"] or ""):
            raise SystemExit(f"{name}: the host's read of its statement ({m['predicted'] or 'accepted'}) is not the "
                             "member's")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    ok = True
    for j, (name, m) in enumerate(staged.items()):
        res = audit_member(M, av, sg, m, at, out / name, a.port + j)
        blobs, held, public = WZ.public_pieces(M.a8, *blobs_of(m), known(sg, at))
        res["received"] = {**M.a8.received(out / name / "verifier", m["dir"] / f"inst-{m['n']}.bin", blobs, held),
                           "public_pieces": public}
        good = expected(name, res) and not res["received"]["verifier_found"] and res["received"]["control"]["holds_all"]
        ok &= good
        summary["members"][name].update(res)
        log(f"member {name}: verdicts {res['verdicts']}, audit {res.get('audit')}, Lean: {res['why']}, verifier "
            f"received {res['received']['verifier_bytes']} bytes, witness pieces found there: "
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
    p = sub.add_parser("prove", help="an egress link-window's EgressBucket units, o0 registered, proved --zk")
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
