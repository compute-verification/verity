"""The served rows held to their Programs' statics (`served_statics.ServedStaticsTie_v1`) on a sealed served window's real
rows, proved in zero knowledge: the window's P / G blocks, each a unit whose committed served rows are its leaf of the
served-rows registration, read through the Lean verifier's registered reads, by C-Flock `prove --zk` behind the Lean
firewall and held by Lean `verify --zk --registered`, through N2-ZK's audit path (`served_tie_zk`'s stager and scan); and
the cost.

    python benchmarks/prove_path/served_statics_zk.py cost [--out DIR]
    python benchmarks/prove_path/served_statics_zk.py prove --serve-run DIR --window K --g G [--per N]
      --bin flock-circuit --lean flock-verify --out DIR [--dry]

* **cost**: a unit's ANDs (`served_statics.size`) and C-Flock block (``zk_block``) at the compile slice's class, the
  sealed run 9c10's and the served class, at every G dividing P, and the largest G whose unit fits C-Flock's 2^K_MAX. The
  unit hashes its two rows, the block's statics and its served rows, and no more.
* **prove**: the window and its steps opened and bound natively (`served_tie_window.bindings`), and the Programs' statics
  checked against each step's committed positions (``statics_held``). The Programs are a stand-in: the honest rows' (start,
  n) in batch order (`served_statics.program_of`), as compile(advice) lays out a step whose advice's rows they are. One
  served-rows registration (`served_rows.register`, a row each block of G steps, fresh salts) holds the honest window as
  window 0 and each case as a window of its own, all under the run root; every unit reads its leaf, and a record whose G
  isn't the statics' is refused (`same_g`). Statements of one template:
  - ``window-i``: the honest window's P / G blocks, each once, the idle ones too (Selection All), at most ``--per`` a
    statement;
  - ``cases``: the block holding each case's edit against the Programs the honest window ran (``KINDS``): a prefill's
    start one on (refused at ``pos``), a prefill one short and the prefill row dropped (``count``), another request's row
    in place of a decode's (``pos``), and a decode row's request id changed, which the tie accepts (it is
    ``IoTieRequest``'s);
  - ``unregistered``: honest blocks whose rows are committed under fresh salts, not the registration's: the tie accepts
    them, and Lean's registered reads refuse the statement.

  Each statement fills with honest blocks to `--zk`'s least shape. The verifier opens each unit's ``pub`` row to its own
  copy of the Programs' statics and its verdict ``o0``; Lean holds each ``srb`` row's b‖c to the leaf its own reads file
  names (`served_rows.reads_port`), and the host predicts the same read (`served_rows.leaf_ok`). A block passes when it is
  proved and its verdict is ``ACCEPT``. The received-pieces scan is `leak_scan.scan`
  (``benchmarks/network_traces/leak_scan.py``), every zero-knowledge driver's, and a row shorter than its piece (``srb``
  at G = 1 in the slice's class, 8 bytes) is found whole (`leak_scan.find_whole`).
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
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "benchmarks/network_traces")]

import leak_scan as LS  # noqa: E402
import served_tie_window as STW  # noqa: E402
import served_tie_zk as STZ  # noqa: E402
from verity.core.protocols.certifier import served_rows as SR, served_statics as SS, served_tie as ST  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as REG  # noqa: E402

CLASSES = {k: {s: c[s] for s in ("R", "P", "IW", "L", "T")} for k, c in STZ.CLASSES.items()}
OPENED = ("o0", "pub")
KINDS = ("honest", "start", "short", "dropped", "id", "other")
ACCEPTED = ("honest", "id")
log = STZ.log


def zk_block(rows: list[tuple[str, int]], ands: int) -> dict:
    """A unit's C-Flock block as `circuit.compose` places it for one VU (`served_tie_zk.zk_block`'s, on these rows)."""
    from verity_flock import circuit as FC

    ports = [FC.Port(r, -(-b // 16), b) for r, b in rows]
    n_sha, _ = FC._row_slots(ports)
    ul = FC._slot_log(ands + sum(b for _r, b in rows))
    _r, k = FC._ranges([("unit", FC.SHA_LOG, n_sha, {"packed": True}), ("unit", FC.HM_LOG, FC._pow2(len(ports)), {}),
                        ("unit", ul, 1, {}), ("mask", FC.MASK_LOG, 1, {})])
    return {"k_log": k, "fits": k <= FC.K_MAX, "sha_slots": n_sha, "unit_log": ul,
            "row_bytes": {r: b // 8 for r, b in rows}}


def cost(a) -> int:
    out = {}
    for name, c in CLASSES.items():
        t = time.time()
        step = SS.step_size(**c)["ands"]
        by = {}
        for G in [g for g in range(c["P"], 0, -1) if c["P"] % g == 0 and (g & (g - 1) == 0 or g == c["P"])]:
            st = {**SS.statics(SimpleNamespace(**c), G=G)}
            unit = SS.size(**st)["ands"]
            by[G] = {"units": c["P"] // G, "unit_ands": unit, "window_ands": c["P"] // G * unit,
                     "zk_block": zk_block(SS.rows(SimpleNamespace(**st)), unit)}
        fit = [G for G, r in by.items() if r["zk_block"]["fits"]]
        out[name] = {"statics": c, "step_ands": step, "by_G": by, "largest_G_fitting": max(fit) if fit else None,
                     "seconds": round(time.time() - t, 1)}
        log(f"{name}: {json.dumps(out[name])}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "cost.json").write_text(json.dumps(out, indent=1))
    return 0


# --- the units -----------------------------------------------------------------------------------------------------------

def table(window: list) -> tuple[dict[str, list], dict]:
    """`served_tie_window.cases`' windows and the prefill row dropped (REDTEAM phase 10's shape: a row the step's Program
    computes, registered as none), with the steps they change."""
    cases, edits = STW.cases(window)
    pre, k, _row = edits["prefill"]
    dropped = [tuple(r for i, r in enumerate(s) if i != k) if u == pre else s for u, s in enumerate(window)]
    wins = {**{n: w for n, (w, _by) in cases.items()}, "dropped": dropped}
    step = {"start": pre, "short": pre, "dropped": pre, "id": edits["decode"][0], "other": edits["decode"][0]}
    return {n: wins[n] for n in KINDS}, {**edits, "steps": step}


def same_g(S, St: SimpleNamespace, record: dict) -> int:
    """The served-rows record's G, as `served_rows.leaf` reads it (P when absent), refused unless it is the G of the
    statics the unit is built with: its leaves are blocks of the record's G steps, and the unit reads St.G of them."""
    if (g := record.get("G", S.P)) != St.G:
        raise SystemExit(f"the served-rows record's G is {g} but the unit's statics' is {St.G}: a leaf would be a block of "
                         f"{g} steps read as {St.G}; register the window at the unit's G (served_rows.register(..., "
                         f"G={St.G}))")
    return g


def plan(S, window: list, programs: list, run_root: bytes, g: int, ids=None) -> SimpleNamespace:
    """The registration (``KINDS``' windows in turn, a row each block of g steps) and the units: the honest window's every
    block, then each case's block holding its edit, each its leaf, the block's served rows and the Programs' statics."""
    St = SimpleNamespace(**SS.statics(S, G=g))
    wins, edits = table(window)
    served = SR.register(S, list(wins.values()), run_root, G=g)
    rec = served.record()
    U = S.P // same_g(S, St, rec)
    units = {"window": [], "cases": []}
    for c, (name, win) in enumerate(wins.items()):
        for j in range(U) if name == "honest" else [edits["steps"][name] // g]:
            leaf = SR.leaf(S, rec, c, j)
            srb = SR.block_bits(S, win, g, j)
            if SR._row(srb)[0] != served.registration.rows[leaf]:
                raise SystemExit(f"{name} block {j}: its rows are not leaf {leaf} of the registration")
            x = SS.case(St, srb, programs[j * g:(j + 1) * g])
            units["window" if name == "honest" else "cases"].append(SimpleNamespace(
                kind=name, j=j, leaf=leaf, rows=x.rows, flags=x.flags, verdict=x.verdict,
                ids=None if ids is None else ids(win), words={r: STZ._words(v) for r, v in x.rows.items()}))
    return SimpleNamespace(S=S, St=St, st=SS.statics(S, G=g), served=served, record=rec, units=units, edits=edits)


def units_of(a) -> SimpleNamespace:
    x = STW.load(a.serve_run, a.window)
    S = STW.statics_of(x)
    port = x.ctx["served_window"]["step_input_port"]
    StR = SimpleNamespace(**ST.statics(S, IB=port["IB"], PB=port["PB"]))
    ok, steps, native = STW.bindings(x, S, StR)
    (window,) = STW._session(x.priv)
    window = list(window) + [None] * (len(steps) - len(window))
    ok["statics_held"] = STW.statics_held(window, steps)
    seqs = {int(r["arrival"]): list(r["prompt_ids"]) + list(r["streamed_ids"]) for r in x.reqs if r.get("arrival") is not None}
    d = plan(S, window, SS.program_of(window), x.run.run_root, a.g, ids=lambda w: STW.ids_ok(StR, w, steps, seqs))
    return SimpleNamespace(**vars(d), x=x, ok=ok, native=native, steps=steps, window=window, run_root=x.run.run_root)


# --- the statement -----------------------------------------------------------------------------------------------------

def shared_bcs(path: Path, ports: list) -> list[dict[str, bytes]]:
    """Each instance's b‖c by port from a shared-row public file (`circuit.write`: its header line, each port's table rows'
    b‖c, port-major in table order, then every instance's refs, u32 little-endian, instance-major, ports in order)."""
    import numpy as np

    head, body = path.read_bytes().split(b"\n", 1)
    h, w = json.loads(head), 128
    n, at = h["instances"], [0]
    for p in ports:
        at.append(at[-1] + int(h["shared_rows"][p.name]))
    refs = np.frombuffer(body[at[-1] * w:at[-1] * w + 4 * n * len(ports)], dtype="<u4").reshape(n, len(ports))
    return [{p.name: body[(at[q] + int(refs[i, q])) * w:(at[q] + int(refs[i, q]) + 1) * w] for q, p in enumerate(ports)}
            for i in range(n)]


def stager(M, defn, St, n: int, out: Path) -> SimpleNamespace:
    rows = [(r, M.V2, b, [r]) for r, b in SS.rows(St)]
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/served-statics", rows, [(M.V2, (0,))], circuit=out / "circuit.txt",
                         bind={"partition": M.P.digest(part), "program": part["program"]})
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, program=program, query=query, n=n, base=base, st=st, out=out,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, sg: SimpleNamespace, name: str, units: list, d: SimpleNamespace, salt_key: bytes, registered: bool) -> dict:
    """A statement: each unit its own table row of both ports; ``srb``'s rows its leaves', under the registration's salts
    (``registered``) or fresh ones, each with its leaf's position and path; and the verifier's own reads file, of a
    record at the unit's G (`same_g`)."""
    same_g(d.S, d.St, d.record)
    t0 = time.time()
    st, n, out = sg.st, len(units), sg.out / "stage"
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= st.meta["k_log"] + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {st.meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    reg = d.served.registration
    data = {r: M.np.asarray([u.words[r] for u in units], dtype=M.np.uint16) for r, _b in SS.rows(d.St)}
    salts = {"srb": [reg.salts[u.leaf] if registered else os.urandom(len(reg.salts[u.leaf])) for u in units]}
    entry = {"value": SR.SCHEMA, "positions": [u.leaf for u in units], "paths": [b"".join(reg.open(u.leaf).path).hex() for u in units]}
    set_name = f"{st.iset.name}-{name}"
    got = st.stage(data, out, salt_key, salts=salts, indices=range(sg.base, sg.base + n), set_name=set_name, tables=data,
                   refs={r: list(range(n)) for r in data}, registered={"srb": entry})
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    reads = {"format": REG.READS_V1, "ports": {"srb": SR.reads_port(d.record, d.run_root, [u.leaf for u in units], sg.base)}}
    (out / "registered.json").write_text(json.dumps(reads, sort_keys=True))
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name,
         "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"],
         "ports": st.ports, "oks": [int(v) for v in M.np.asarray(got["outs"]["o0"])[:, 0]], "nbl": nbl,
         "partition_file": sg.partition_file, "program_file": sg.program_file, "registered_file": out / "registered.json",
         "registered": registered, "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


def audit(M, N2, a, sg, m: dict, units: list, d: SimpleNamespace, out: Path, port: int) -> dict:
    """The statement registered, every unit drawn by the verifier's Lean, proved --zk behind the firewall and held by Lean
    with the verifier's own registered reads; then per unit the verifier's side: ``pub`` opened to its own copy, the
    verdict read, and ``srb``'s read predicted (`served_rows.leaf_ok`). Writes what the verifier received under
    ``out / "verifier"``."""
    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    n, name = m["n"], m["name"]
    want, roots = {}, []
    fv = m["header"]["frame_v3"]
    for p_, wd in a7.frame_ports(m["set"], m["meta"], n).items():
        want[f"{name}/{p_}"] = wd
        roots.append({"port": f"{name}/{p_}", "root": fv["roots"][p_], "leaves": wd["leaves"], "domain": fv["domain_ids"][p_]})
    leaf_layer = hashlib.sha512(M.canonical_json({name: a0.sha512_file(m["dir"] / f"pub-{n}.bin")})).hexdigest()
    law = M.D.parse_law(f"stratified:{n}")
    win = {"kind": "stand-in", "of": f"the served rows held to their Programs' statics on a sealed served window "
                                      f"({a.serve_run.name} w{a.window})",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "served_statics: each unit's srb row is its leaf of the served-rows registration under the run root "
                   "(registered reads); its pub row, opened, the Programs' statics"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False}
    p = a0.run([a.lean, "draw", "--program", str(sg.program_file), "--partition", str(sg.partition_file), "--stratified",
                str(n)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + n)
    sdir = out / "session"
    record, live, proc = M.a8.session(a, m, sdir, local, port, env=N2.zk_server_env(M, m))
    draw_t = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw_t, {"population": n}, {"law": "subset", "k": len(local)})
    verdicts = {f"m0-verifier/{name}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                        and (live or {}).get("roots") == fv["roots"] and v.ok,
                                        "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}}
    lv = a0.lean_verify(a.lean, m, n, sdir, zk=True)
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program --registered"}
    proved = a0.proved_units(m["header"], live or {})
    timing = {k: (live or {}).get(k) for k in N2.TIMING}
    statement = {name: {"session_s": proc["wall_s"], **{k: (live or {}).get(k) for k in ("circuit_sha512", "statement_digest",
                                                                                         "sigma")},
                        "timing": timing, "process": proc, "lean_verify_s": lv.get("verify_s")}}
    sessions = [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, R.receipt(rec, "served-statics verifier (served_statics_zk.py)"), lean_draw, rec["law"],
                            sorted(proved), (live or {}).get("instances", 0), statement, verdicts, a.delta, sessions=sessions,
                            served=sorted(m["base"] + u for u in draw_t.get("units", [])),
                            session_roots={f"{name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    held = set(proved) if result.get("accepted") else set()
    port_ = {x.name: x for x in m["ports"]}
    bcs = shared_bcs(m["dir"] / f"pub-{n}.bin", m["ports"])
    reg = d.served.registration
    rows, sent = [], []
    for i, u in enumerate(units):
        word = m["oks"][i]
        opened = ((m["base"] + i) in held and a7.opens(port_["pub"], u.words["pub"], m["salts"]["pub"][i], bcs[i]["pub"])
                  and a7.opens(port_["o0"], [word], m["salts"]["o0"][i], bcs[i]["o0"]))
        srb_ok = SR.leaf_ok(d.record, d.run_root, u.leaf, bcs[i]["srb"], reg.open(u.leaf).path)
        verdict = "malformed" if not opened else "accept" if word == SS.ACCEPT and srb_ok else "reject"
        rows.append({"unit": m["base"] + i, "kind": u.kind, "block": u.j, "leaf": u.leaf, "verdict_word": word,
                     "reference": u.verdict, "flags": u.flags, "srb_read": srb_ok, "ids": u.ids, "verdict": verdict})
        sent.append({"unit": m["base"] + i, "verdict": word, **{f"salt/{r}": m["salts"][r][i].hex() for r in OPENED}})
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt",
              m["dir"] / f"pub-{n}.bin", m["registered_file"], sdir / "draw.json", out / "audit.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    proofs = (live or {}).get("proof_bytes") or []
    return {**res, "accepted": bool(result.get("accepted")), "units": rows,
            "audit": {k: result.get(k) for k in ("accepted", "complete", "wrong_units")},
            "verdicts": {k: y.get("accepted") for k, y in verdicts.items()}, "why": {k: y.get("why") for k, y in verdicts.items()},
            "registered": lv.get("registered"), "timing": statement[name], "lean_verify_s": lv.get("verify_s"),
            "proof_bytes": proofs,
            "per_unit": {"prove_total_s": N2._per(timing.get("prove_total_s"), n), "proof_bytes": N2._per(sum(proofs) if proofs else None, n),
                         "lean_verify_s": N2._per(lv.get("verify_s"), n)}}


def blobs_of(units: list, salts: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each unit's served rows as the instance file holds them (u16
    little-endian) and their salts, which it holds too."""
    blobs = {}
    for i, u in enumerate(units):
        if any(u.words["srb"]):
            blobs[f"srb {i}"] = b"".join(x.to_bytes(2, "little") for x in u.words["srb"])
        for r, ss in salts.items():
            if r not in OPENED:
                blobs[f"{r} salt {i}"] = ss[i]
    return blobs, set(blobs)


def received(a8, vdir: Path, control: Path, blobs: dict[str, bytes], held: set[str], circuit: bytes) -> dict:
    """What the verifier received (`vdir`) scanned for `blobs`, by `leak_scan`'s rule: each row of at least a piece cut
    whole, less the pieces the staged `circuit` holds (`leak_scan.public_pieces`, found by `a8.received`), and each
    shorter row found whole (`leak_scan.find_whole`, as ``whole_rows``); the `control` (the instance file) is to hold
    every one in `held`."""
    short = {k: x for k, x in blobs.items() if len(x) < a8.PIECE}
    pieces, kept, public = LS.public_pieces(a8, {k: x for k, x in blobs.items() if k not in short}, held, circuit)
    rec = {**a8.received(vdir, control, pieces, kept), "public_pieces": len(public)}
    rec.update(LS.scan(rec, pieces, vdir))
    rec["whole_rows"] = LS.find_whole(rec, short, vdir, control, held & set(short))
    return rec


def expected(name: str, registered: bool, res: dict, units: list) -> bool:
    """A registered statement is accepted, each unit's verdict its kind's; an unregistered one is refused by Lean's
    registered reads."""
    lean = res.get("verdicts", {}).get(f"lean-verify/{name}")
    if not registered:
        return not res["accepted"] and lean is False and "registered reads" in (res.get("why", {}).get(f"lean-verify/{name}") or "")
    want = ["accept" if u.kind in ACCEPTED else "reject" for u in units]
    return res["accepted"] and all(res["verdicts"].values()) and [r["verdict"] for r in res.get("units", [])] == want


def prove(a) -> int:
    t0 = time.time()
    d = units_of(a)
    honest, cases = d.units["window"], d.units["cases"]
    summary: dict = {"window": a.window, "run": str(a.serve_run), "statics": d.st, "bindings": d.ok, "native": d.native,
                     "edits": d.edits, "N": [s.N for s in d.steps], "registration": d.record,
                     "size": {"unit": SS.size(**d.st)["ands"], "step": SS.step_size(**d.st)["ands"]},
                     "zk_block": zk_block(SS.rows(d.St), SS.size(**d.st)["ands"]),
                     "reference": {"window": [u.verdict for u in honest],
                                   "cases": [{"kind": u.kind, "block": u.j, "leaf": u.leaf, "verdict": u.verdict,
                                              "flags": u.flags, "ids": u.ids} for u in cases]}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_ok = (all(d.ok.values()) and all(u.verdict == SS.ACCEPT for u in honest)
              and all((u.verdict == SS.ACCEPT) == (u.kind in ACCEPTED) for u in cases))
    summary["reference_ok"] = ref_ok
    log(f"window w{a.window}: {len(d.steps)} steps, bindings {d.ok}, {len(honest)} blocks of G = {a.g} "
        f"({summary['size']['unit']} ANDs, k_log {summary['zk_block']['k_log']}), cases {summary['reference']['cases']}, "
        f"{round(time.time() - t0, 1)} s")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry or not ref_ok:
        return 0 if ref_ok else 1

    N2 = __import__("n2_zk")
    M = N2._one_stage()
    defn = SS.unit(**d.st)
    secret = os.urandom(32)
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta, points=[], arms=["fw"], cold=1,
                         serve_run=a.serve_run, window=a.window)
    per = a.per or len(honest)
    groups = [(f"window-{i // per}", honest[i:i + per], True) for i in range(0, len(honest), per)]
    groups += [("cases", cases, True), ("unregistered", honest[:2], False)]
    summary["statements"] = {name: [(u.kind, u.j, u.leaf) for u in us] for name, us, _r in groups}
    ok, meta = True, None
    for k, (name, us, registered) in enumerate(groups):
        units = STZ.fill(us, honest, STZ.population(meta, len(us))) if meta else us
        sg = stager(M, defn, d.St, len(units), out / f"stager-{name}")
        if meta is None and STZ.population(sg.st.meta, len(us)) > len(units):
            units = STZ.fill(us, honest, STZ.population(sg.st.meta, len(us)))
            sg = stager(M, defn, d.St, len(units), out / f"stager-{name}")
        meta = sg.st.meta
        log(f"{name}: lowered in {sg.lower_s} s, k_log {meta['k_log']}, vus_per_block {meta['vus_per_block']}, "
            f"{len(units)} units")
        m = stage(M, sg, name, units, d, hashlib.sha512(secret + name.encode()).digest()[:32], registered)
        if m["oks"] != [u.verdict for u in units]:
            raise SystemExit(f"{name}: the staged verdict words {m['oks']} are not the reference's")
        log(f"{name}: staged {m['n']} units, {m['ands']} ANDs a unit, k_log {meta['k_log']}, m {meta['k_log'] + m['nbl']}, "
            f"{'registered' if registered else 'fresh'} srb salts, {m['stage_s']} s")
        res = audit(M, N2, av, sg, m, units, d, out / name, a.port + 20 * k)
        blobs, held = blobs_of(units, m["salts"])
        res["received"] = rec = received(M.a8, out / name / "verifier", m["dir"] / f"inst-{m['n']}.bin", blobs, held,
                                         (m["dir"] / "circuit.txt").read_bytes())
        wr = rec["whole_rows"]
        res["staged"] = {"n": m["n"], "ands": m["ands"], "k_log": meta["k_log"], "nbl": m["nbl"], "pin": m["pin"],
                         "lower_s": sg.lower_s, "stage_s": m["stage_s"], "registered": registered}
        res["as_expected"] = (expected(name, registered, res, units) and not rec["leaked"] and rec["control"]["holds_all"]
                              and not wr["leaked"] and wr["control"]["holds_all"])
        ok &= res["as_expected"]
        summary[name] = res
        log(f"{name}: proofs {res.get('verdicts')}, why {res.get('why')}, audit {res.get('audit')}, verdicts "
            f"{[(r['kind'], r['block'], r['verdict_word'], r['srb_read'], r['verdict']) for r in res.get('units', [])][:12]}..., "
            f"prove {res.get('timing', {}).get('timing', {}).get('prove_total_s')} s, Lean {res.get('lean_verify_s')} s, "
            f"proof {sum(res.get('proof_bytes') or [0])} B, witness pieces found {rec['leaked'] or 'none'} "
            f"(of {len(blobs)} rows), structured {rec['structured'] or 'none'}; rows shorter than a piece found whole "
            f"{wr['leaked'] or 'none'} (of {wr['rows']}), structured {wr['structured'] or 'none'}; "
            f"{'as expected' if res['as_expected'] else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = ok
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="served_statics_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cost", help="a unit's ANDs and C-Flock block at the slice's, 9c10's and the served class, by G")
    c.add_argument("--out")
    p = sub.add_parser("prove", help="a sealed window's blocks proved --zk and held by Lean with its registered reads")
    p.add_argument("--serve-run", type=Path, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--g", type=int, required=True, help="steps a block")
    p.add_argument("--per", type=int, default=0, help="units a statement (the window's blocks in statements of at most PER)")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7891)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the window, its units and their references only")
    a = ap.parse_args(argv)
    return {"cost": cost, "prove": prove}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
