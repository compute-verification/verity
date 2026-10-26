"""transfer-zk: a server block's proof units read against the network certifier's transfer rows (`certifier.transfer_rows`):
a TINY server block, n units of ``ScaledMmS8Coordinate_v1{K=128}`` on one uplink, every input row a registered read of a
value the network certifier committed and signed as it crossed the server's boundary and every output row a row of the
response it signed. The verifier pins each root from its certificate alone (`certifier.blocks`), C-Flock `prove --zk`
proves the statement behind the Lean firewall, and the Lean verifier's registered reads hold it
(`flock-verify verify --zk --registered`), through the one-stage audit as warden_zk.py's.

    python benchmarks/network_traces/transfer_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--units 8]
      [--dry]

* **The circuit** (`circuit`): one template, n units, unit i a W8A8 linear's output coordinate i on one token: ``x`` the
  token's s8 row and ``sx`` its scale (the request's rows, external ports), ``wrow`` channel i's s8 weights and ``sw``
  its scale (the weights' load), ``o0`` the coordinate (the response's rows, an external port). Every input row is a
  registered read (`VALUES`), unit i reading row i of its value.
* **The blocks** (`blocks`, `verity/blocks/v1`): one server on uplink `UPLINK`, its server-time block window `WINDOW`
  of n steps, unit i's rows at step i; the loads ``w`` and ``sw`` (ingress, the server session's window 0), the
  request's ``request`` and ``request_scale`` (ingress) and the ``response`` (egress, every unit's ``o0``).
* **The network certifier**, in process here: its own LMS key (the deployed N = 64, W = 1, on 2^6 leaves) in its own
  session. It commits each value as it crosses (`transfer_rows.attest`) under its own salts, which it hands the prover
  alone, and publishes the registrations (`transfer_rows.registration`).
* **The verifier** checks the blocks against the circuit record and pins each ingress value's root from the
  registrations under the certifier's key alone (`blocks.pins`), so the interface's ``input_count`` under those pins
  has no free input (and with none every input is free). Its own ``--registered`` file is each value's port at the
  certifier's signed root (`transfer_rows.port`).
* **Members.** One statement of n units each, every port registered:
  - ``honest``: every row under the certifier's salts and paths, to be accepted;
  - ``forged``: the response the certifier signed is one byte off unit 0's output (a server that sent bytes its proven
    computation didn't produce: row 0's low byte), and o0 is under that response's salts and paths, to be refused at
    that read (`WHY`);
  - ``unsigned``: o0 under the prover's own commitment of the units' outputs (its own salts and tree), against the
    certifier's root, to be refused at that read. The prover's registration of its own root under the certifier's
    certificate is refused by `transfer_rows.check_registration` and by `blocks.certified`.
  Each is drawn by the verifier's Lean (every unit, `flock-verify draw --stratified n`), proved ``--zk`` behind the
  firewall (``FC_FIREWALL=1``, ``FC_FIREWALL_LEAN``) and held by Lean ``verify --zk --partition --program --registered``.
* **What the verifier receives**, scanned for every 16-byte piece of every x and wrow row and every salt of the
  statement (`a8.received`) but the pieces the staged circuit holds: none is to be found, and the prover's instance file,
  which holds them, is the control. The sx, sw and o0 rows are 4 and 2 bytes, below the scan's floor (`leak_scan.WHOLE`),
  and are held by the same proof.
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
import numpy as np  # noqa: E402
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.evaluation import evaluate_batch  # noqa: E402
from verity.ml.circuits.ops import gemm as GM  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity.ml.windows import interface as IF  # noqa: E402
from verity.core.protocols.certifier import blocks as BK, transfer_rows as TR, warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.certificate import Certificate, Direction, Issuer, LmsSigner  # noqa: E402

K = 128
COORD = bind(GM.ScaledMmS8Coordinate, K=K, **GM.GEMM_SM120_S8)
TID = descriptor_id(COORD)
UPLINK, WINDOW = "s0/uplink", 1
#: the deployed scheme (N = 64, W = 1) on a tree of 2^6 leaves
PARAMS = lms.Params(64, 1, 6)
#: each port's value, its kind and the link-window it crosses in (a load's the server session's, 0)
VALUES = {"x": ("request", "request", WINDOW), "sx": ("request_scale", "request", WINDOW), "wrow": ("w", "load", 0),
          "sw": ("sw", "load", 0), "o0": ("response", "response", WINDOW)}
INS, PORTS = ("x", "sx", "wrow", "sw"), ("x", "sx", "wrow", "sw", "o0")
MEMBERS = {"honest": True, "forged": False, "unsigned": False}
#: what Lean's registered reads, and the host's read of them, refuse each refused member by, at port o0
WHY = {"forged": "does not open", "unsigned": "does not open"}


def head(heads: dict, p: str) -> TR.TransferHead:
    """Port ``p``'s value's head among `blocks.certified`'s, which are by value and direction."""
    value, kind, _window = VALUES[p]
    return heads[value, TR.KINDS[kind][0]]


def log(msg: str) -> None:
    print(f"TZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the circuit and its blocks ---------------------------------------------------------------------------------------

def circuit(n: int) -> IF.Circuit:
    lay = dict(IF.layout(COORD).rows)
    reads = tuple(IF.Read(TID, p, VALUES[p][0], 0, (n,), 0, (lay[p],)) for p in INS)
    return IF.Circuit("transfer-zk", ((COORD, n),), external=(f"{TID}/x", f"{TID}/sx", f"{TID}/o0"),
                      registered=IF.Registered({VALUES[p][0]: n * lay[p] for p in INS}, reads=reads))


def blocks(n: int) -> dict:
    steps = {"base": 0, "radix": [n], "step": [1]}

    def transfer(p: str, d: str) -> dict:
        value, kind, window = VALUES[p]
        return {"server": "s0", "dir": d, "kind": kind, "value": value, "window": window,
                "t": None if kind == "load" else steps}

    out = {**transfer("o0", "out"), "leaves": n,
           "writes": [{"writer": TID, "out": "o0", "at": 0, "radix": [n], "base": 0, "step": [1]}]}
    return {"format": BK.FORMAT, "server": [{"name": "s0", "uplink": UPLINK, "units": "all"}],
            "server_time": {"window": WINDOW, "steps": n}, "transfers": [*(transfer(p, "in") for p in INS), out]}


def values(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """n coordinates' rows, each port's ``(n, words)`` u16 words: seeded s8 rows and positive f32 scales, and the
    Definition's outputs on them."""
    rng = np.random.default_rng(seed)
    x, w = (rng.integers(-128, 128, (n, K)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127 + 1e-3).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    o0 = np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16).reshape(n, 1)
    lay = dict(IF.layout(COORD).rows)
    words = {p: IF.words(a, 8 if p in ("x", "wrow") else 32, n, lay[p]) for p, a in zip(INS, ins)}
    return {**words, "o0": o0}


# --- the network certifier and the verifier ---------------------------------------------------------------------------

def certifier(directory: Path) -> SimpleNamespace:
    """The network certifier on `UPLINK`: a fresh LMS key and session, and its issuer."""
    directory.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(PARAMS, os.urandom(64)), directory / "signer-state.json")
    session = os.urandom(32)
    return SimpleNamespace(issuer=Issuer(signer, session), key=signer.public_key(), session=session)


def attest(cert: SimpleNamespace, program: str, value: str, rows) -> tuple[TR.Transfer, dict]:
    """`value`'s rows as they crossed `UPLINK`, committed and signed: the certifier's opening (witness) and its
    registration (public)."""
    p = next(p for p, v in VALUES.items() if v[0] == value)
    _v, kind, window = VALUES[p]
    direction = Direction.EGRESS if p == "o0" else Direction.INGRESS
    t, c = TR.attest(cert.issuer, UPLINK, direction, window, kind, program, value, [list(map(int, r)) for r in rows])
    if c is None:
        raise SystemExit(f"{value}: the network certifier's signer refused: {cert.issuer.refused}")
    return t, TR.registration(t.head, c)


def attestation(n: int, out: Path) -> SimpleNamespace:
    """The circuit record and its blocks; the units' rows; the network certifier's transfers and registrations, the
    honest response's and the forged one's; and the prover's own commitment of its outputs."""
    c = circuit(n)
    doc = IF.descriptor_record(c, cap=n)
    words = values(n)
    cert = certifier(out / "certifier")
    tr, regs = {}, {}
    for p in PORTS:
        v = VALUES[p][0]
        tr[v], regs[v] = attest(cert, doc["program"], v, words[p])
    forged_rows = words["o0"].copy()
    forged_rows[0, 0] ^= 1
    forged, forged_reg = attest(cert, doc["program"], "response", forged_rows)
    own = TR.Transfer.commit(UPLINK, Direction.EGRESS, WINDOW, "response", doc["program"], "response",
                             [list(map(int, r)) for r in words["o0"]])
    return SimpleNamespace(n=n, circuit=c, doc=doc, blocks=blocks(n), words=words, cert=cert, transfers=tr,
                           registrations=regs, forged=forged, forged_registration=forged_reg, own=own)


def layouts(doc: dict) -> dict:
    """Each template's bit rows, the verifier's own from its catalog."""
    return {t: IF.layout(IF.resolve(t)) for t, _n in doc["templates"]}


def verifier(at: SimpleNamespace, regs: list[dict]) -> SimpleNamespace:
    """The verifier's read of the registrations it is handed: the blocks checked and every transfer's head from its
    certificate alone (`blocks.certified`), the pins, the free inputs with and without them, and its own port of each
    value at the certified root."""
    certifiers = {UPLINK: (at.cert.key, at.cert.session)}
    heads = BK.certified(at.doc, at.blocks, layouts(at.doc), regs, certifiers)
    pins = BK.pins(at.doc, at.blocks, layouts(at.doc), regs, certifiers)
    ports = {p: TR.port(head(heads, p), list(range(at.n))) for p in PORTS}
    return SimpleNamespace(heads=heads, pins=pins, ports=ports, free=IF.input_count(at.doc, pins=pins)["free"],
                           free_unpinned=IF.input_count(at.doc)["free"])


def presented(name: str, at: SimpleNamespace) -> list[dict]:
    """The registrations the network certifier published for member `name`'s window: ``forged``'s response is the
    forged bytes it saw."""
    regs = dict(at.registrations)
    if name == "forged":
        regs["response"] = at.forged_registration
    return list(regs.values())


def refusals(at: SimpleNamespace) -> dict[str, str]:
    """The verifier's blocks read refusing what isn't the certifier's for this response: none, the prover's own root
    under the certifier's certificate, and the certifier's certificate of the same rows in the next window, named this
    one's."""
    out = {}
    c = at.registrations["response"]["certificate"]
    own = TR.registration(at.own.head, Certificate(bytes.fromhex(c["key"]), UPLINK, Direction.EGRESS, WINDOW,
                                                   bytes.fromhex(c["digest"]), bytes.fromhex(c["signature"])))
    t, nxt = TR.attest(at.cert.issuer, UPLINK, Direction.EGRESS, WINDOW + 1, "response", at.doc["program"], "response",
                       [list(map(int, r)) for r in at.words["o0"]])
    later = TR.registration(t.head, nxt) | {"window": WINDOW}
    rest = [r for v, r in at.registrations.items() if v != "response"]
    for case, regs in {"no-certificate": rest, "unsigned": [*rest, own], "another-window": [*rest, later]}.items():
        try:
            verifier(at, regs)
            out[case] = "accepted"
        except ValueError as e:
            out[case] = f"refused: {e}"
    return out


def member_ports(name: str, at: SimpleNamespace) -> dict[str, tuple[list[bytes], dict]]:
    """Each port's salts and header entry for member `name`'s statement: the certifier's opening of its value, but
    ``forged``'s o0 under the forged response's and ``unsigned``'s under the prover's own commitment."""
    out = {}
    for p in PORTS:
        t = at.transfers[VALUES[p][0]]
        if p == "o0" and name == "forged":
            t = at.forged
        elif p == "o0" and name == "unsigned":
            t = at.own
        _rows, salts, entry = t.statement(range(at.n))
        out[p] = (salts, entry)
    return out


def refused(at: SimpleNamespace, v: SimpleNamespace, mp: dict, base: int) -> dict[str, str]:
    """The host's read of each registered port against the verifier's, as Lean's registered reads make it
    (`warden_rows.refusal`): the ports it refuses, with why."""
    out = {}
    for p in PORTS:
        h = head(v.heads, p)
        salts, entry = mp[p]
        why = WR.refusal(TR.port(h, list(range(at.n)), base), TR.domain(at.doc["program"], h.value, h.words, h.rows),
                         entry, [TR.bc(list(map(int, r)), s) for r, s in zip(at.words[p], salts)])
        if why is not None:
            out[p] = why
    return out


# --- the audit (warden_zk's flow on one-stage's a0, a7 and a8) ----------------------------------------------------------

def _one_stage():
    sys.path.insert(0, str(ONE_STAGE))
    import a0
    import a7
    import a8
    from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity_flock import class_statement as CS

    return SimpleNamespace(a0=a0, a7=a7, a8=a8, D=D, P=P, R=R, V=V, canonical_json=canonical_json,
                           encode_program=encode_program, CS=CS)


def stager(M, at: SimpleNamespace, out: Path) -> SimpleNamespace:
    """The statement program and partition (the circuit's, one template of n units) and the unit's stager."""
    program, query = at.circuit.program()
    part = M.P.partition(program, query)
    if part["program"] != at.doc["program"]:
        raise SystemExit("the stager's program is not the circuit record's")
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    t = time.time()
    in_rows, out_rows = IF.unit_rows(COORD)
    st = M.CS.UnitStager(COORD, "unit/transfer", in_rows, out_rows, circuit=out / "circuit.txt", bind=bound)
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(COORD) and cnt == at.n
    return SimpleNamespace(defn=COORD, program=program, query=query, n=at.n, base=base, st=st, out=out,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, sg: SimpleNamespace, at: SimpleNamespace, name: str) -> dict:
    """Member `name`'s statement: the units' rows, a shared-row file (each instance its own table row), every port
    registered under its value's salts and paths (`member_ports`); and the verifier's own `--registered` file at the
    certifier's roots."""
    t0 = time.time()
    st, n, out = sg.st, sg.n, sg.out / f"stage-{name}"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {k_log} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    v = verifier(at, presented(name, at))
    mp = member_ports(name, at)
    rows = {p: at.words[p] for p in INS}
    got = st.stage(rows, out, os.urandom(32), salts={p: mp[p][0] for p in PORTS},
                   indices=range(sg.base, sg.base + n), set_name=f"{st.iset.name}-{name}", tables=rows,
                   refs={p: list(range(n)) for p in INS}, registered={p: mp[p][1] for p in PORTS})
    if not np.array_equal(np.asarray(got["outs"]["o0"], dtype=np.uint16), at.words["o0"]):
        raise SystemExit(f"{name}: the units' outputs are not the Definition's")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    reads = WR.reads_file({p: TR.port(head(v.heads, p), list(range(n)), sg.base) for p in PORTS})
    (out / "registered.json").write_text(json.dumps(reads, sort_keys=True))
    (out / "transfer-registrations.json").write_text(json.dumps(presented(name, at), indent=1, sort_keys=True))
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512,
         "set": f"{st.iset.name}-{name}", "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"],
         "unit_nnz": got["unit_nnz"], "salts": got["salts"], "ports": st.ports, "partition_file": sg.partition_file,
         "program_file": sg.program_file, "registered_file": out / "registered.json",
         "registrations_file": out / "transfer-registrations.json", "predicted": refused(at, v, mp, sg.base),
         "pins": v.pins, "free": v.free, "free_unpinned": v.free_unpinned, "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")


def firewall(M, m: dict) -> dict:
    """A firewalled prover's environment (`FC_FIREWALL=1`, its Lean frame named by ``FC_FIREWALL_LEAN``)."""
    if not os.environ.get("FC_FIREWALL_LEAN"):
        raise SystemExit("FC_FIREWALL_LEAN names no Lean firewall frame (flock-verify's flock-firewall)")
    return {"FC_FIREWALL": "1", **M.a0.firewall_env(m)}


def audit_member(M, a, sg, m: dict, out: Path, port: int) -> dict:
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
    win = {"kind": "stand-in", "of": f"a TINY server block of {n} W8A8 coordinates on {UPLINK}, every row a transfer "
                                      "row the network certifier signed (transfer_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": {p: f"registered: port {p} reads {VALUES[p][0]} at the network certifier's signed root" for p in PORTS}}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want),
              "transfers": {"ok": m["free"] == 0, "free": m["free"], "free_unpinned": m["free_unpinned"]}}
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False, "verdicts": {}, "why": {}}
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
    receipt = R.receipt(rec, "one-stage-e2e verifier (transfer_zk.py)")
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), (live or {}).get("instances", 0), statement,
                            verdicts, a.delta, sessions=sessions, served=sorted(m["base"] + u for u in draw_t.get("units", [])),
                            session_roots={f"{name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt",
              m["dir"] / f"pub-{n}.bin", m["registered_file"], m["registrations_file"], sdir / "draw.json",
              out / "audit.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    return {**res, "accepted": bool(result.get("accepted")),
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()}, "why": {x: y.get("why") for x, y in verdicts.items()},
            "registered": lv.get("registered"), "timing": statement[name], "lean_verify_s": lv.get("verify_s")}


def blobs_of(at: SimpleNamespace, m: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: every x and wrow row and every salt of the statement, all of
    which the prover's instance file holds."""
    blobs = {f"{p} {i}": r.astype("<u2").tobytes() for p in ("x", "wrow") for i, r in enumerate(at.words[p])}
    blobs.update({f"salt {p} {i}": s for p, ss in m["salts"].items() for i, s in enumerate(ss)})
    return blobs, set(blobs)


def expected(name: str, res: dict) -> bool:
    """The member to be accepted is; the others are refused by Lean's registered reads at port o0, each by its `WHY`."""
    lean = res["verdicts"].get(f"lean-verify/{name}")
    if MEMBERS[name]:
        return res["accepted"] and lean is True
    why = res["why"].get(f"lean-verify/{name}") or ""
    return not res["accepted"] and lean is False and "registered reads" in why and "o0" in why and WHY[name] in why


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    at = attestation(a.units, out)
    v = verifier(at, presented("honest", at))
    summary: dict = {"template": TID, "units": at.n, "uplink": UPLINK, "window": WINDOW, "blocks": at.blocks,
                     "circuit": at.doc, "registrations": at.registrations, "forged_registration": at.forged_registration,
                     "pins": v.pins, "free": v.free, "free_unpinned": v.free_unpinned, "refusals": refusals(at)}
    log(f"{TID} × {at.n} on {UPLINK}: the certifier signed {sorted(at.registrations)}; the verifier pins "
        f"{sorted(v.pins)}, free {v.free} bits (with no pins {v.free_unpinned}); refusals {summary['refusals']}")
    if v.free != 0 or v.free_unpinned == 0 or not all(r.startswith("refused") for r in summary["refusals"].values()):
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        raise SystemExit("the verifier's blocks read is not free 0 only with the certified pins")
    M = _one_stage()
    sg = stager(M, at, out / "stager")
    summary["stager"] = {"lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"], "vus_per_block": sg.st.meta["vus_per_block"],
                         "pin": sg.st.pin, "base": sg.base}
    log(f"stager: {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
    staged = {}
    for name in MEMBERS:
        m = staged[name] = stage(M, sg, at, name)
        summary.setdefault("members", {})[name] = {"ands": m["ands"], "stage_s": m["stage_s"], "free": m["free"],
                                                   "predicted": m["predicted"] or "accepted"}
        log(f"staged {name}: {m['ands']} ANDs, every port registered; the host's read: {m['predicted'] or 'accepted'}")
        bad = set(m["predicted"]) - {"o0"} or (not m["predicted"]) != MEMBERS[name] or \
            WHY.get(name, "") not in m["predicted"].get("o0", "")
        if bad:
            raise SystemExit(f"{name}: the host's read of its statement ({m['predicted'] or 'accepted'}) is not the member's")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    ok = True
    for j, (name, m) in enumerate(staged.items()):
        res = audit_member(M, av, sg, m, out / name, a.port + j)
        if "verdicts" in res and res["verdicts"]:
            blobs, held, public = LS.public_pieces(M.a8, *blobs_of(at, m), (m["dir"] / "circuit.txt").read_bytes())
            res["received"] = {**M.a8.received(out / name / "verifier", m["dir"] / f"inst-{m['n']}.bin", blobs, held),
                               "public_pieces": public}
        good = expected(name, res) and "received" in res and not res["received"]["verifier_found"] \
            and res["received"]["control"]["holds_all"]
        ok &= good
        summary["members"][name].update(res)
        r = res.get("received", {})
        log(f"member {name}: verdicts {res['verdicts']}, audit {res.get('audit')}, Lean: {res['why']}, verifier received "
            f"{r.get('verifier_bytes')} bytes, witness pieces found there: {r.get('verifier_found') or 'none'} (of "
            f"{r.get('blobs')}), control {r.get('control', {}).get('blobs_found')}; {'as expected' if good else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="transfer_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="a server block's units registered against the network certifier's roots, proved --zk")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--units", type=int, default=8)
    p.add_argument("--port", type=int, default=7791)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the statements and the host's read of them: nothing proved")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
