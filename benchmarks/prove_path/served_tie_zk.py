"""The routed served-rows tie (`served_tie.ServedRowsTie_v2`) on a sealed served window's real rows, proved in zero
knowledge: the window's P / G units by C-Flock `prove --zk` behind the Lean firewall and held by Lean `verify --zk`, through
N2-ZK's audit path (`benchmarks/network_traces/n2_zk.py`: its stager, firewall and received-pieces scan); and the cost.

    python benchmarks/prove_path/served_tie_zk.py cost [--out DIR]
    python benchmarks/prove_path/served_tie_zk.py build --class slice|9c10|served --g G [--out DIR]
    python benchmarks/prove_path/served_tie_zk.py prove --serve-run DIR --window K [--g G] [--blocks all|executed] [--per N]
      --bin flock-circuit --lean flock-verify --out DIR [--dry]
    python benchmarks/prove_path/served_tie_zk.py rescan --serve-run DIR --window K [--g G] [--blocks ...] [--per N]
      --from PROVE_OUT --out DIR

* **cost**: a step's and a unit's ANDs (`served_tie.step_size`, `size`) at the compile slice's class, at the sealed run
  9c10's (R 8, P 256, IW 16, L 512, T 2048) and at the served class (T 8192, R 256, L 131072), each beside the all-pairs
  form's count it replaces (``all_pairs``), and a unit's C-Flock block (``zk_block``): its rows' in-circuit hashing, not
  the tie's ANDs, sets it, since every unit hashes the window's whole ``srv`` row and each step's T·PB bytes of positions.
* **build**: a class's unit at G steps built as a Definition, and each Definition it calls, with seconds and peak RSS.
* **prove**: the window and the rows as `served_tie_window` opens and binds them natively (``bindings``). Statements of
  one template, block j of G steps a unit:
  - ``window-i``: the honest window's blocks, each once (Selection All: every unit drawn), at most ``--per`` a statement;
    ``--blocks executed`` takes only the blocks holding the window's executed steps;
  - ``cases``: the start, short, id and other cases' blocks (`served_tie_window.cases`), each case's rows registered
    consistently under the run root, and honest blocks to the statement's least shape.
  The verifier opens each unit's ``pub`` row to its own copy (the block and the steps' N) and its verdict ``o0``, and checks
  each unit's ``srv`` b‖c to be its registration's row w (`served_rows.leaf_ok`). A window passes when each of its blocks
  is proved and its verdict is ``ACCEPT``. The positions chunks are bound natively, opened under the run root
  (``bindings``): the Commit's leaves are its position-leaf digests under its hiding rule, not row leaves, so their b‖c is
  not yet a proof row's.
  The received-pieces scan (`leak_scan.public_pieces`, `a8.received`) fails a statement on a ``leaked`` piece, whatever its
  row or its bytes, under the rule every zero-knowledge driver's scan follows (`leak_scan.scan`,
  ``benchmarks/network_traces/leak_scan.py``): a find counts only inside one field of C-Flock's proof format
  (`redigest.layout`) or one record of a round message, and is excused only by what the statement fixes. The staged circuit's pieces are left
  out before the scan (`public_pieces`).
* **rescan**: a finished ``prove``'s statements scanned again (`leak_scan.rescan`): the verifier's files searched afresh
  for the pieces recomputed from the sealed run, under the current cutter and rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "benchmarks/network_traces")]

import leak_scan as LS  # noqa: E402
import served_tie_window as STW  # noqa: E402
from verity.core.protocols.certifier import served_rows as SR, served_tie as ST  # noqa: E402

CLASSES = {
    "slice": dict(R=2, P=7, IW=6, L=4096, T=256, IB=8, PB=8),
    "9c10": dict(R=8, P=256, IW=16, L=512, T=2048, IB=8, PB=8),
    "served": dict(R=256, P=256, IW=16, L=131072, T=8192, IB=8, PB=8),
}
GS = {"slice": (7, 1), "9c10": (256, 8, 1), "served": (256, 16, 1)}
OPENED = ("o0", "pub")
NEGATIVES = ("start", "short", "id", "other")


def log(msg: str) -> None:
    print(f"STZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


ALL_PAIRS_MEASURED = {"9c10": 649_079}


def all_pairs(name: str, c: dict) -> tuple[int, str]:
    """ServedRowsTie_v1's ANDs a step, the form this replaces: its traced count where measured, else its docstring's
    estimate T·(29·R + 8·PB − W)."""
    if name in ALL_PAIRS_MEASURED:
        return ALL_PAIRS_MEASURED[name], "measured"
    LW, TW = max(c["L"].bit_length(), 1), max(c["T"].bit_length(), 1)
    W = max(LW, TW + max(c["R"].bit_length(), 1)) + 2
    return c["T"] * (29 * c["R"] + 8 * c["PB"] - W), "estimate"


def zk_block(S, ands: int) -> dict:
    """A unit's C-Flock block as `circuit.compose` places it for one VU: its rows' SHA-512 slots (packed) and hm96 slots,
    its unit's slot, the mask. The unit's slot is estimated from its ANDs and input bits (the lowered unit's own count
    sets it)."""
    from verity_flock import circuit as FC

    bits = [b for _r, b in ST.rows(S)]
    ports = [FC.Port(r, -(-b // 16), b) for r, b in ST.rows(S)]
    n_sha, _ = FC._row_slots(ports)
    ul = FC._slot_log(ands + sum(bits))
    _r, k = FC._ranges([("unit", FC.SHA_LOG, n_sha, {"packed": True}), ("unit", FC.HM_LOG, FC._pow2(len(ports)), {}),
                        ("unit", ul, 1, {}), ("mask", FC.MASK_LOG, 1, {})])
    return {"k_log": k, "fits": k <= FC.K_MAX, "sha_slots": n_sha, "unit_log": ul, "row_bytes": sum(bits) // 8,
            "srv_bytes": dict(ST.rows(S))["srv"] // 8}


def cost(a) -> int:
    out = {}
    for name, c in CLASSES.items():
        t = time.time()
        step = ST.step_size(**c)["ands"]
        rows = {}
        for G in GS[name]:
            st = {**c, "G": G}
            S = SimpleNamespace(**st)
            Z = ST.widths(S)
            unit = ST.size(**st)["ands"]
            cells = unit + sum(n for _r, n in ST.rows(S))
            rows[G] = {"units": Z.U, "unit_ands": unit, "unit_input_bits": cells - unit, "window_ands": Z.U * unit,
                       "zk_block": zk_block(S, unit)}
        ap, how = all_pairs(name, c)
        out[name] = {"statics": c, "step_ands": step, "per_token": round(step / c["T"], 1), "window_ands": c["P"] * step,
                     "all_pairs_step": ap, "all_pairs_window": c["P"] * ap, "all_pairs_is": how,
                     "ratio": round(ap / step, 2), "by_G": rows,
                     "widths": vars(ST.widths(SimpleNamespace(**c, G=c["P"]))),
                     "seconds": round(time.time() - t, 1)}
        log(f"{name}: {json.dumps(out[name])}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "cost.json").write_text(json.dumps(out, indent=1))
    return 0


def build(a) -> int:
    """One class's unit at G steps built as a Definition, with every Definition it calls once: each one's gates, the
    seconds its body took and the process's peak RSS after it."""
    import resource

    st = {**CLASSES[a.cls], "G": a.g}
    t0, seen, rows = time.time(), set(), []

    def walk(d) -> None:
        if d.id in seen:
            return
        seen.add(d.id)
        t = time.time()
        body = d.body
        rows.append({"definition": d.id, "gates": d.gates, "seconds": round(time.time() - t, 1),
                     "peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)})
        log(f"{a.cls} G {a.g}: {rows[-1]}")
        for n in body.nodes:
            if n.form == "call":
                walk(n.fn)

    walk(ST.unit(**st))
    out = {"class": a.cls, "statics": st, "unit_ands": ST.size(**st)["ands"], "definitions": rows,
           "seconds": round(time.time() - t0, 1), "peak_gb": max(r["peak_gb"] for r in rows)}
    log(f"{a.cls} G {a.g}: built in {out['seconds']} s at {out['peak_gb']} GB peak")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"build-{a.cls}-G{a.g}.json").write_text(json.dumps(out, indent=1))
    return 0


# --- the proof ---------------------------------------------------------------------------------------------------------

def units_of(a) -> SimpleNamespace:
    """The window, its steps, each case's registration, and both statements' units: kind, block j, rows, reference."""
    x = STW.load(a.serve_run, a.window)
    S = STW.statics_of(x)
    port = x.ctx["served_window"]["step_input_port"]
    st = ST.statics(S, IB=port["IB"], PB=port["PB"], G=a.g)
    St = SimpleNamespace(**st)
    ok, steps, native = STW.bindings(x, S, St)
    (window,) = STW._session(x.priv)
    window = list(window) + [None] * (len(steps) - len(window))
    seqs = {int(r["arrival"]): list(r["prompt_ids"]) + list(r["streamed_ids"]) for r in x.reqs if r.get("arrival") is not None}
    table, edits = STW.cases(window)
    run_root = x.run.run_root
    port_steps = [(s.N, s.chunks) for s in steps]
    regs, units = {}, {"window": [], "cases": []}
    for name, (win, by) in table.items():
        if name == "honest":
            reg = SR.register(S, STW._session(x.priv), run_root, salts=[bytes.fromhex(s) for s in x.priv["salts"]])
        else:
            reg = SR.register(S, [win], run_root)
        regs[name] = reg
        bits = SR.window_bits(S, win)
        if name == "honest":
            blocks = range(ST.widths(St).U if a.blocks == "all" else -(-len(steps) // a.g))
        else:
            u = edits["prefill" if name in ("start", "short") else "decode"][0]
            blocks = [u // a.g]
        for j in blocks:
            r = ST.unit_rows(St, bits, port_steps, j)
            flags, ref = ST.check(St, r), ST.reference(St, r)
            if flags != ref:
                raise SystemExit(f"{name} block {j}: the circuit's reference {ref} isn't the statement's {flags}")
            units["window" if name == "honest" else "cases"].append(SimpleNamespace(
                kind=name, j=j, rows=r, flags=flags, verdict=ST.verdict(flags), by=by, reg=reg,
                ids=STW.ids_ok(St, win, steps, seqs), words={k: _words(v) for k, v in r.items()}))
    return SimpleNamespace(x=x, S=S, st=st, St=St, ok=ok, native=native, steps=steps, window=window, edits=edits, regs=regs,
                           units=units, run_root=run_root)


def _words(bits: list[int]) -> list[int]:
    return [sum(int(bits[16 * k + j]) << j for j in range(16)) for k in range(len(bits) // 16)]


def stager(M, defn, St, n: int, out: Path) -> SimpleNamespace:
    """A statement's program and partition (one template, n units) and its stager, the unit lowered by its builder."""
    rows = [(r, M.V2, b, [r]) for r, b in ST.rows(St)]
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/served-tie", rows, [(M.V2, (0,))], circuit=out / "circuit.txt",
                         bind={"partition": M.P.digest(part), "program": part["program"]})
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, rows=rows, program=program, query=query, n=n, base=base, st=st, out=out, part=part,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, sg: SimpleNamespace, name: str, units: list, salt_key: bytes, salts: dict) -> dict:
    t0 = time.time()
    st, n, out = sg.st, len(units), sg.out / "stage"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {k_log} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    data = {r: M.np.asarray([u.words[r] for u in units], dtype=M.np.uint16) for r, _b in ST.rows(units[0].St)}
    set_name = f"{st.iset.name}-{name}"
    got = st.stage(data, out, salt_key, salts=salts, indices=range(sg.base, sg.base + n), set_name=set_name)
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name,
         "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"],
         "ports": st.ports, "oks": [int(v) for v in M.np.asarray(got["outs"]["o0"])[:, 0]], "nbl": nbl,
         "partition_file": sg.partition_file, "program_file": sg.program_file, "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


def audit(M, N2, a, sg, m: dict, units: list, w: int, run_root: bytes, out: Path, port: int) -> dict:
    """The statement registered, every unit drawn by the verifier's Lean, proved --zk behind the firewall and held by Lean;
    then per unit the verifier's side: ``pub`` opened to its own copy, the verdict read, ``srv``'s b‖c held to its
    registration's row w (`served_rows.leaf_ok`). Writes what the verifier received under ``out / "verifier"``."""
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
    win = {"kind": "stand-in", "of": f"the served rows' tie on a sealed served window ({a.serve_run.name} w{w})",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "served_tie: each unit's srv row is its served-rows registration's row w; its positions are the step "
                   "streams' port chunks, opened under the run root"}
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
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program"}
    proved = a0.proved_units(m["header"], live or {})
    timing = {k: (live or {}).get(k) for k in N2.TIMING}
    statement = {name: {"session_s": proc["wall_s"], **{k: (live or {}).get(k) for k in ("circuit_sha512", "statement_digest",
                                                                                         "sigma")},
                        "timing": timing, "process": proc, "lean_verify_s": lv.get("verify_s")}}
    sessions = [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, R.receipt(rec, "served-tie verifier (served_tie_zk.py)"), lean_draw, rec["law"],
                            sorted(proved), (live or {}).get("instances", 0), statement, verdicts, a.delta, sessions=sessions,
                            served=sorted(m["base"] + u for u in draw_t.get("units", [])),
                            session_roots={f"{name}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    held = set(proved) if result.get("accepted") else set()
    port_ = {x.name: x for x in m["ports"]}
    bcs = a7.public_bcs(m["dir"] / f"pub-{n}.bin", m["ports"])
    rows, sent = [], []
    for i, u in enumerate(units):
        word = m["oks"][i]
        opened = ((m["base"] + i) in held and a7.opens(port_["pub"], u.words["pub"], m["salts"]["pub"][i], bcs[i]["pub"])
                  and a7.opens(port_["o0"], [word], m["salts"]["o0"][i], bcs[i]["o0"]))
        reg = u.reg.registration
        srv_ok = SR.leaf_ok(u.reg.record(), run_root, 0, bcs[i]["srv"], reg.open(0).path) and bcs[i]["srv"] == u.reg.bc(0)
        verdict = "malformed" if not opened else "accept" if word == ST.ACCEPT and srv_ok else "reject"
        rows.append({"unit": m["base"] + i, "kind": u.kind, "block": u.j, "verdict_word": word, "reference": u.verdict,
                     "flags": u.flags, "srv_bound": srv_ok, "ids": u.ids, "verdict": verdict})
        sent.append({"unit": m["base"] + i, "verdict": word, **{f"salt/{r}": m["salts"][r][i].hex() for r in OPENED}})
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt",
              m["dir"] / f"pub-{n}.bin", sdir / "draw.json", out / "audit.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    proofs = (live or {}).get("proof_bytes") or []
    return {**res, "accepted": bool(result.get("accepted")), "units": rows,
            "audit": {k: result.get(k) for k in ("accepted", "complete", "wrong_units")},
            "verdicts": {k: y.get("accepted") for k, y in verdicts.items()}, "why": {k: y.get("why") for k, y in verdicts.items()},
            "timing": statement[name], "lean_verify_s": lv.get("verify_s"), "proof_bytes": proofs,
            "per_unit": {"prove_total_s": N2._per(timing.get("prove_total_s"), n), "proof_bytes": N2._per(sum(proofs) if proofs else None, n),
                         "lean_verify_s": N2._per(lv.get("verify_s"), n)}}


def blobs_of(units: list, salts: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each unit's positions chunks and switches as the instance file
    holds them (u16 little-endian), and every witness row's salt; the rows are what the control holds."""
    blobs, held = {}, set()
    for i, u in enumerate(units):
        for r, ws in u.words.items():
            if r.startswith(("pos_", "sw_")) and any(ws):
                blobs[f"{r} {i}"] = b"".join(x.to_bytes(2, "little") for x in ws)
                held.add(f"{r} {i}")
        for r, ss in salts.items():
            if r not in OPENED:
                blobs[f"{r} salt {i}"] = ss[i]
    return blobs, held


def population(meta: dict, n: int) -> int:
    """The fewest units, at least n, whose statement fits `--zk`'s shape (k_log + nbl ≥ 25, `stage`)."""
    b = 25 - meta["k_log"]
    return n if b <= 3 else max(n, meta["vus_per_block"] * ((1 << (b - 1)) + 1))


def fill(units: list, honest: list, least: int) -> list:
    """`units`, then honest blocks in turn until there are `least`."""
    out = list(units)
    while len(out) < least:
        out.append(honest[(len(out) - len(units)) % len(honest)])
    return out


def prove(a) -> int:
    t0 = time.time()
    d = units_of(a)
    for u in d.units["window"] + d.units["cases"]:
        u.St = d.St
    summary: dict = {"window": a.window, "run": str(a.serve_run), "statics": d.st, "bindings": d.ok, "native": d.native,
                     "edits": d.edits, "N": [s.N for s in d.steps],
                     "size": {"unit": ST.size(**d.st)["ands"], "step": ST.step_size(**d.st)["ands"]},
                     "reference": {"window": [u.verdict for u in d.units["window"]],
                                   "cases": [{"kind": u.kind, "block": u.j, "verdict": u.verdict, "flags": u.flags, "ids": u.ids}
                                             for u in d.units["cases"]]}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    expect = {"honest": ST.ACCEPT, "id": ST.ACCEPT}
    ref_ok = all(d.ok.values()) and all(u.verdict == ST.ACCEPT for u in d.units["window"]) and all(
        (u.verdict == ST.ACCEPT) == (u.kind in expect) for u in d.units["cases"])
    summary["reference_ok"] = ref_ok
    log(f"window w{a.window}: {len(d.steps)} steps, N {[s.N for s in d.steps if s.N]}, bindings {d.ok}, units "
        f"{len(d.units['window'])} of G = {a.g} ({summary['size']['unit']} ANDs, step {summary['size']['step']}), cases "
        f"{summary['reference']['cases']}, {round(time.time() - t0, 1)} s")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry or not ref_ok:
        return 0 if ref_ok else 1

    N2 = __import__("n2_zk")
    M = N2._one_stage()
    defn = ST.unit(**d.st)
    secret = os.urandom(32)
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta, points=[], arms=["fw"], cold=1, serve_run=a.serve_run)
    ok = True
    honest, groups = d.units["window"], statements(d, a.per)
    summary["statements"] = {name: [(u.kind, u.j) for u in us] for name, us in groups}
    meta = None
    for k, (name, us) in enumerate(groups):
        units = fill(us, honest, population(meta, len(us))) if meta else us
        sg = stager(M, defn, d.St, len(units), out / f"stager-{name}")
        if meta is None and population(sg.st.meta, len(us)) > len(units):
            units = fill(us, honest, population(sg.st.meta, len(us)))
            sg = stager(M, defn, d.St, len(units), out / f"stager-{name}")
        meta = sg.st.meta
        log(f"{name}: lowered in {sg.lower_s} s, k_log {meta['k_log']}, vus_per_block {meta['vus_per_block']}, "
            f"{len(units)} units")
        salts = {"srv": [u.reg.registration.salts[0] for u in units]}
        m = stage(M, sg, name, units, hashlib.sha512(secret + name.encode()).digest()[:32], salts)
        refs = [u.verdict for u in units]
        if m["oks"] != refs:
            raise SystemExit(f"{name}: the staged verdict words {m['oks']} are not the reference's {refs}")
        log(f"{name}: staged {m['n']} units, {m['ands']} ANDs a unit, k_log {m['meta']['k_log']}, m {m['meta']['k_log'] + m['nbl']}, "
            f"{m['stage_s']} s")
        res = audit(M, N2, av, sg, m, units, a.window, d.run_root, out / name, a.port + 20 * k)
        blobs, held, public = LS.public_pieces(M.a8, *blobs_of(units, m["salts"]), (m["dir"] / "circuit.txt").read_bytes())
        res["received"] = {**M.a8.received(out / name / "verifier", m["dir"] / f"inst-{m['n']}.bin", blobs, held),
                           "public_pieces": len(public)}
        res["received"].update(LS.scan(res["received"], blobs, out / name / "verifier"))
        res["staged"] = {"n": m["n"], "ands": m["ands"], "k_log": m["meta"]["k_log"], "nbl": m["nbl"], "pin": m["pin"],
                         "lower_s": sg.lower_s, "stage_s": m["stage_s"]}
        want = ["accept" if u.kind in expect else "reject" for u in units]
        got = [r["verdict"] for r in res.get("units", [])]
        res["as_expected"] = (res["accepted"] and all(res["verdicts"].values()) and got == want
                              and not res["received"]["leaked"] and res["received"]["control"]["holds_all"])
        ok &= res["as_expected"]
        summary[name] = res
        log(f"{name}: proofs {res['verdicts']}, audit {res['audit']}, verdicts {[(r['kind'], r['block'], r['verdict_word'], r['srv_bound'], r['verdict']) for r in res.get('units', [])][:12]}..., "
            f"prove {res['timing']['timing'].get('prove_total_s')} s, Lean {res['lean_verify_s']} s, proof {sum(res['proof_bytes'] or [0])} B, "
            f"witness pieces found {res['received']['leaked'] or 'none'} (of {len(blobs)}), structured ones "
            f"{res['received']['structured'] or 'none'}; "
            f"{'as expected' if res['as_expected'] else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = ok
    summary["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def statements(d: SimpleNamespace, per: int) -> list[tuple[str, list]]:
    """The honest window's blocks in statements of at most `per` units (all in one when 0), then the cases'."""
    honest = d.units["window"]
    per = per or len(honest)
    return [(f"window-{i // per}", honest[i:i + per]) for i in range(0, len(honest), per)] + [("cases", d.units["cases"])]


def rescan(a) -> int:
    """`prove`'s statements under `--from` scanned again (`leak_scan.rescan`): the verifier's files each one kept searched
    afresh for its pieces, recomputed from the sealed run as `prove` cut them (its units, then the honest blocks it filled
    the statement with, and their ``srv`` salts, less the pieces the circuit holds: the kept one, else the statement
    lowered again and held to the run's pin). A found salt `prove` drew isn't recomputed, so its summary's find of it
    stays and reads leaked."""
    d = units_of(a)
    N2 = __import__("n2_zk")
    M = N2._one_stage()
    src, out = Path(getattr(a, "from")), Path(a.out)
    summary = json.loads((src / "summary.json").read_text())
    honest, ok, res = d.units["window"], True, {}
    for name, us in statements(d, a.per):
        if name not in summary:
            continue
        if [[u.kind, u.j] for u in us] != [list(x) for x in summary["statements"][name]]:
            raise SystemExit(f"{name}: these units aren't the run's {summary['statements'][name]}")
        units, vdir = fill(us, honest, summary[name]["staged"]["n"]), src / name / "verifier"
        rows, held = blobs_of(units, {"srv": [u.reg.registration.salts[0] for u in units]})
        if (vdir / "circuit.txt").is_file():
            circuit = (vdir / "circuit.txt").read_bytes()
        else:
            with tempfile.TemporaryDirectory() as tmp:
                sg = stager(M, ST.unit(**d.st), d.St, len(units), Path(tmp))
                if sg.st.pin != summary[name]["staged"]["pin"]:
                    raise SystemExit(f"{name}: the statement lowered again isn't the run's (pin {sg.st.pin[:16]}, the run's "
                                     f"{summary[name]['staged']['pin'][:16]}): rescan it with the commit that proved it")
                circuit = (Path(tmp) / "circuit.txt").read_bytes()
        blobs = LS.public_pieces(M.a8, rows, held, circuit)[0]
        got = summary[name]["received"]
        res[name] = r = {"pieces": len(blobs), **LS.rescan(got, blobs, vdir, a8=M.a8, rows=rows)}
        ok &= not r["leaked"]
        log(f"{name}: {len(blobs)} pieces, found {sorted({*r['leaked'], *r['structured']}) or 'none'} (the run found "
            f"{got['verifier_found'] or 'none'}), leaked {r['leaked'] or 'none'} (was {r['was'] or 'none'}), "
            f"structured {r['structured'] or 'none'}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rescan.json").write_text(json.dumps({"from": str(src), "ok": ok, "statements": res}, indent=1, sort_keys=True))
    log("rescan: " + ("clean" if ok else "LEAKED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="served_tie_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cost", help="the routed tie's ANDs at the slice's, 9c10's and the served class")
    c.add_argument("--out")
    b = sub.add_parser("build", help="one class's unit and the Definitions it calls built: seconds and peak RSS")
    b.add_argument("--class", dest="cls", choices=tuple(CLASSES), required=True)
    b.add_argument("--g", type=int, required=True, help="steps a unit")
    b.add_argument("--out")
    p = sub.add_parser("prove", help="a sealed window's units proved --zk and held by Lean")
    p.add_argument("--serve-run", type=Path, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--g", type=int, default=1, help="steps a unit")
    p.add_argument("--blocks", choices=("all", "executed"), default="all",
                   help="the window's blocks proved: all P / G, or those holding an executed step")
    p.add_argument("--per", type=int, default=0, help="units a statement (the window's blocks in statements of at most PER)")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7791)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the window, its units and their references only")
    r = sub.add_parser("rescan", help="a finished prove's statements scanned again from its summary and verifier files")
    r.add_argument("--serve-run", type=Path, required=True)
    r.add_argument("--window", type=int, required=True)
    r.add_argument("--g", type=int, default=1, help="steps a unit")
    r.add_argument("--blocks", choices=("all", "executed"), default="all")
    r.add_argument("--per", type=int, default=0)
    r.add_argument("--from", required=True, help="the prove's --out")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"cost": cost, "build": build, "prove": prove, "rescan": rescan}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
