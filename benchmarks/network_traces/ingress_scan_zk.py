"""The scan's ingress bucket unit (`ingress_scan.IngressScanBucket`) in zero knowledge at w63's class: buckets of a
warden run's ingress record, each one unit, proved by C-Flock `prove --zk` behind the Lean firewall and held by Lean
`verify --zk`, through the one-stage audit as `warden_zk.py` proves its rows.

    python benchmarks/network_traces/ingress_scan_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]

* **The class** (`CLASS`): w63's, B 512, r 4, 4 lanes, C 47 cells a request, IDB 17, Programs started every 10
  buckets and the 12-bucket ingress bound (note:network-accounting/20261025T0830Z-report-net68-scan-router), about
  1.37M ANDs a unit; the warden at `PARAMS`, which the class serves (`Class.serves`: D 21).
* **The run.** The warden itself, driven tick by tick on an injected clock (`test_network_certifier_warden`'s
  harness), carries five warden-form id requests (`PROMPTS`, `ARRIVALS`): 1,500 prompt ids from bucket 0, 600 and
  40 from bucket 2, 300 from bucket 12, and 100 from bucket 15, which waits in the queue until request 1's lane frees
  past its egress and takes request 1's slot, free past its anchor (bucket 20). Its ingress record's rows
  (`warden_rows.window_rows`) are scanned by the reference from the empty state (`ingress_scan.scan`).
* **The statement**: one template of n units, each a bucket's (row, S_{t−1}, S_t) on rows ``row`` (4,096 words),
  ``sin`` and ``sout`` (9,065 words each), and its output ``o0``, the ok bit. The units are the buckets in `HONEST`,
  then the negatives, each refused by the reference: ``swap`` (bucket 2's first two blocks swapped), ``end`` (request
  1's first data cell made `END`), ``digit`` (a digit of a continuation cell's prompt id changed), ``ids`` (a prompt
  id bit of S_t off) and ``held`` (the reuse bucket with request 1's slot still held, so the new request may not take
  it), then the rest of the window's buckets in order, as many as the statement needs to fit `--zk`'s shape.
* **The verifier** draws every unit (Lean `draw --stratified n`), the prover proves them `--zk` behind the firewall
  (``FC_FIREWALL=1``), Lean `verify --zk --archive --partition --program` holds the session, and the verifier opens
  each unit's ``o0`` under the salt the prover sends: 1 on the honest buckets, 0 on the negatives.
* **Standalone**: the rows are committed under fresh salts, not yet registered reads (the warden's signed row, the
  scan's own state rows); the scan's state chain is the reference's here.
* **What the verifier receives** is scanned for every piece of every unit's three rows and every salt (`a8.received`)
  less the pieces the public circuit holds, the pads (`warden.cell(PAD, q)`, fixed by the ordinal) and the sparse ones
  (at most one nonzero byte: a small integer in a zero field, which a proof's encoding holds whatever the witness); the
  prover's instance file, which holds them, is the control.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ONE_STAGE = HERE.parent / "one_stage"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "verity/core/protocols/certifier/tests"))

import n2_zk as N2  # noqa: E402
from test_network_certifier_warden import EOF, Server, drive  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.core.protocols.certifier import ingress_scan as IS  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import HEADER, id_request  # noqa: E402
from verity.core.protocols.certifier.warden import DATA, END, PAD, WardenParams, cell, cells  # noqa: E402

B = 512
CLASS = IS.Class(B=B, r=4, lanes=4, C=47, IDB=17, EVERY=10, WITHIN=12)
PARAMS = WardenParams(cell_bytes=B, slots=4, bucket_ms=100, delay=21, lanes=4, queue=4, window=40, ingress_cells=47,
                      ingress_buckets=12, egress_cells=64)
PROMPTS = ([(7 * k + 11) % 128_256 for k in range(1500)], [100_000 + k for k in range(600)],
           [131_071 - 3 * k for k in range(40)], [(k * k) % 128_256 for k in range(300)], [5_000 + k for k in range(100)])
ARRIVALS = {0: [0], 2: [1, 2], 12: [3], 15: [4]}
#: the buckets proved first (then the reuse bucket): request 1's first, buckets with a new and a continuing block,
#: request 1's END and the anchor
HONEST = (0, 2, 3, 8, 12, 20)
NEGATIVES = ("swap", "end", "digit", "ids", "held")
ROWS = ("row", "sin", "sout")
TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")


def log(msg: str) -> None:
    print(f"ISZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _request(i: int, prompt: list[int]) -> bytes:
    return id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=prompt, max_tokens=16 + i, seed=7 * i,
                      temperature=0.7, top_p=0.95, frame_bytes=B, lead=0)


def run() -> SimpleNamespace:
    """The warden's run at `PARAMS`, its ingress rows and the reference's scan of them."""
    CLASS.serves(PARAMS)
    reqs = [_request(i, p) for i, p in enumerate(PROMPTS)]
    arrivals = {t: [cells(reqs[i], B) for i in ids] for t, ids in ARRIVALS.items()}
    warden, _ = asyncio.run(drive(Server([[(1, b"a" * 20), (2, EOF)] for _ in reqs]), arrivals, params=PARAMS))
    rows = WR.window_rows(next(r for r in warden.records if r.link == "requests"), PARAMS)
    states = IS.scan(CLASS, rows)
    if states is None:
        raise SystemExit("the reference refuses the warden's own ingress rows")
    states = [IS.empty(CLASS)] + states
    admitted: dict[int, int] = {}
    for st in states:
        for s in st.slots:
            if s.live:
                admitted.setdefault(s.q, s.admitted)
    reuse = admitted[5]
    for q, prompt in enumerate(PROMPTS, 1):
        a = CLASS.anchor(admitted[q], admitted[q] % CLASS.EVERY)
        slot = next((s for s in states[a].slots if s.live and s.q == q), None) if a < len(states) else None
        if slot is not None and not (slot.done and slot.fields == len(prompt)
                                     and [slot.ids[IS.position(CLASS, p)] for p in range(len(prompt))] == prompt):
            raise SystemExit(f"request {q}'s slot at its anchor {a} isn't its whole request")
    return SimpleNamespace(rows=rows, states=states, bits=[IS.encode(CLASS, s) for s in states], admitted=admitted,
                           reuse=reuse, pads=b"".join(cell(PAD, q, b"", B) for q in admitted))


def _blocks(row: bytes) -> list[list[bytes]]:
    r = CLASS.r
    return [[row[(p * r + i) * B:(p * r + i + 1) * B] for i in range(r)] for p in range(CLASS.lanes)]


def _per(v, n: int):
    return None if v is None else round(v / n, 3)


def _join(blocks) -> bytes:
    return b"".join(c for blk in blocks for c in blk)


def negative(x: SimpleNamespace, kind: str) -> tuple[int, bytes, list[int], list[int]]:
    """(bucket, row, sin, sout) of a negative kind (the module docstring's)."""
    rows, bits = x.rows, x.bits
    if kind == "swap":
        blk = _blocks(rows[2])
        return 2, _join([blk[1], blk[0], *blk[2:]]), bits[2], bits[3]
    t, p, i = next((t, p, i) for t, row in enumerate(rows) for p, b in enumerate(_blocks(row)) for i, c in enumerate(b)
                   if c[0] == DATA)
    if kind == "end":
        blk = _blocks(rows[t])
        blk[p][i] = bytes([END]) + blk[p][i][1:]
        return t, _join(blk), bits[t], bits[t + 1]
    if kind == "digit":
        blk, s = _blocks(rows[3]), x.states[3].slots[0]
        if not (blk[0][0][0] == DATA and s.q == 1 and s.cells > 0):
            raise SystemExit("bucket 3's first cell isn't a continuation of request 1")
        c = bytearray(blk[0][0])
        at = HEADER.size + 11 * 3 + 10
        c[at] = ord("0") + (c[at] - ord("0") + 1) % 10
        blk[0][0] = bytes(c)
        return 3, _join(blk), bits[3], bits[4]
    if kind == "ids":
        sout = list(bits[4])
        sout[IS.layout(CLASS)["0.ids"][0] + 3] ^= 1
        return 3, rows[3], bits[3], sout
    if kind == "held":
        t = x.reuse
        st = x.states[t]
        slots = list(st.slots)
        j = next(j for j, s in enumerate(slots) if s.q == 1)
        slots[j] = replace(slots[j], live=1, anchor=t + 1)
        return t, rows[t], IS.encode(CLASS, replace(st, slots=slots)), bits[t + 1]
    raise ValueError(kind)


def cases(x: SimpleNamespace, n: int) -> list[SimpleNamespace]:
    """The statement's n units: `HONEST`'s buckets, the negatives, then the window's other buckets in order."""
    honest = HONEST + (x.reuse,)
    out = [SimpleNamespace(kind=f"bucket {t}", bucket=t, row=x.rows[t], sin=x.bits[t], sout=x.bits[t + 1], want=1)
           for t in honest]
    for k in NEGATIVES:
        t, row, sin, sout = negative(x, k)
        out.append(SimpleNamespace(kind=k, bucket=t, row=row, sin=sin, sout=sout, want=0))
    rest = [t for t in range(len(x.rows)) if t not in honest]
    while len(out) < n:
        t = rest[(len(out) - len(honest) - len(NEGATIVES)) % len(rest)]
        out.append(SimpleNamespace(kind=f"bucket {t}", bucket=t, row=x.rows[t], sin=x.bits[t], sout=x.bits[t + 1],
                                   want=1))
    for u in out[:n]:
        got = IS.reference(CLASS, u.row, u.sin, u.sout)
        if got != u.want:
            raise SystemExit(f"{u.kind}: the reference's ok is {got}, not {u.want}")
    return out[:n]


def words(u: SimpleNamespace) -> dict[str, bytes]:
    """A unit's rows as the stager takes them: little-endian bits in u16 words."""
    import numpy as np

    pack = lambda bits: np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little").tobytes()  # noqa: E731
    return {"row": u.row, "sin": pack(u.sin), "sout": pack(u.sout)}


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


def stager(M, n: int, out: Path) -> SimpleNamespace:
    """The statement program and partition (one template, n units) and the unit's stager."""
    defn = bind(IS.IngressScanBucket, B=CLASS.B, r=CLASS.r, lanes=CLASS.lanes, C=CLASS.C, IDB=CLASS.IDB,
                EVERY=CLASS.EVERY, WITHIN=CLASS.WITHIN)
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    ns = IS.state_bits(CLASS)
    t = time.time()
    st = M.CS.UnitStager(defn, "unit/ingress-scan", [("row", M.V2, 8 * CLASS.S * B, ["row"]), ("sin", M.V2, ns, ["sin"]),
                                                     ("sout", M.V2, ns, ["sout"])], [(M.V2, (0,))],
                         circuit=out / "circuit.txt", bind=bound)
    st.pin
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, program=program, query=query, n=n, base=base, st=st, out=out,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def population(meta: dict, n: int) -> int:
    """The fewest units, at least n, whose statement fits `--zk`'s shape (k_log + nbl ≥ 25, nbl ≥ 3)."""
    b = 25 - meta["k_log"]
    return n if b <= 3 else max(n, meta["vus_per_block"] * ((1 << (b - 1)) + 1))


def stage(M, sg: SimpleNamespace, units: list[SimpleNamespace]) -> dict:
    t0 = time.time()
    st, n, out = sg.st, sg.n, sg.out / "stage"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= M.a7.M_MAX:
        raise SystemExit(f"m = {k_log} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    ws = [words(u) for u in units]
    data = {r: M.np.stack([M.np.frombuffer(w[r], dtype="<u2") for w in ws]).astype(M.np.uint16) for r in ROWS}
    set_name = f"{st.iset.name}-w63"
    got = st.stage(data, out, os.urandom(32), indices=range(sg.base, sg.base + n), set_name=set_name)
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    m = {"tag": "w63", "name": "w63", "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512,
         "set": set_name, "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
         "salts": got["salts"], "ports": st.ports, "oks": [int(v) for v in M.np.asarray(got["outs"]["o0"])[:, 0]],
         "words": ws, "partition_file": sg.partition_file, "program_file": sg.program_file, "k_log": k_log, "nbl": nbl,
         "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


def audit(M, a, sg, m: dict, units: list[SimpleNamespace], out: Path, port: int) -> dict:
    """The statement registered, every unit drawn by the verifier's Lean, proved --zk behind the firewall and held by
    Lean; then each unit's ``o0`` opened under the salt the prover sends. Writes what the verifier received under
    ``out / "verifier"``."""
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
    win = {"kind": "stand-in", "of": f"{n} buckets of a warden run's ingress record at w63's class (ingress_scan_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "standalone: each unit's row, sin and sout rows under fresh salts, not yet registered reads"}
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
    if not os.environ.get("FC_FIREWALL_LEAN"):
        raise SystemExit("FC_FIREWALL_LEAN names no Lean firewall frame (flock-verify's flock-firewall)")
    record, live, proc = M.a8.session(a, m, sdir, local, port, env={"FC_FIREWALL": "1", **a0.firewall_env(m)})
    draw_t = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw_t, {"population": n}, {"law": "subset", "k": len(local)})
    verdicts = {f"m0-verifier/{name}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                        and (live or {}).get("roots") == fv["roots"] and v.ok,
                                        "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}}
    lv = a0.lean_verify(a.lean, m, n, sdir, zk=True)
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program"}
    proved = a0.proved_units(m["header"], live or {})
    timing = {k: (live or {}).get(k) for k in TIMING}
    statement = {name: {"session_s": proc["wall_s"], **{k: (live or {}).get(k) for k in ("circuit_sha512", "statement_digest",
                                                                                         "sigma")},
                        "timing": timing, "process": proc, "lean_verify_s": lv.get("verify_s")}}
    sessions = [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, R.receipt(rec, "one-stage-e2e verifier (ingress_scan_zk.py)"), lean_draw, rec["law"],
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
        opened = (m["base"] + i) in held and a7.opens(port_["o0"], [word], m["salts"]["o0"][i], bcs[i]["o0"])
        rows.append({"unit": m["base"] + i, "kind": u.kind, "bucket": u.bucket, "ok": word, "want": u.want,
                     "verdict": "malformed" if not opened else "accept" if word == 1 else "reject"})
        sent.append({"unit": m["base"] + i, "o0": word, "salt/o0": m["salts"]["o0"][i].hex()})
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
            "timing": statement[name], "lean_verify_s": lv.get("verify_s"),
            "per_unit": {"prove_total_s": _per(timing.get("prove_total_s"), n), "e2e_s": _per(timing.get("e2e_s"), n),
                         "proof_bytes": _per(sum(proofs) if proofs else None, n),
                         "lean_verify_s": _per(lv.get("verify_s"), n)}}


def blobs_of(m: dict) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: every unit's rows and every salt of the statement."""
    blobs = {f"{r} {i}": w[r] for i, w in enumerate(m["words"]) for r in ROWS}
    blobs.update({f"salt {p} {i}": s for p, ss in m["salts"].items() for i, s in enumerate(ss)})
    return blobs, set(blobs)


def sparse(pieces: dict[str, bytes], held: set[str]) -> tuple[dict[str, bytes], set[str], int]:
    out = {k for k, x in pieces.items() if sum(map(bool, x)) <= 1}
    return {k: x for k, x in pieces.items() if k not in out}, held - out, len(out)


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    x = run()
    summary: dict = {"class": vars(CLASS), "params": vars(PARAMS), "state_bits": IS.state_bits(CLASS),
                     "admitted": x.admitted, "reuse": x.reuse,
                     "anchors": {q: CLASS.anchor(s, s % CLASS.EVERY) for q, s in x.admitted.items()},
                     "run_s": round(time.time() - t, 1)}
    log(f"run: {len(x.rows)} buckets, admitted {x.admitted}, anchors {summary['anchors']}, request 5 reuses a slot at "
        f"bucket {x.reuse}, the reference scans every bucket ({summary['run_s']} s)")
    if a.dry:
        units = cases(x, len(HONEST) + 1 + len(NEGATIVES))
        summary["reference"] = [(u.kind, u.bucket, u.want) for u in units]
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        log(f"dry: {summary['reference']}")
        return 0
    M = _one_stage()
    k = len(HONEST) + 1 + len(NEGATIVES)
    probe = stager(M, k, out / "probe")
    n = population(probe.st.meta, k)
    log(f"probe: k_log {probe.st.meta['k_log']}, vus_per_block {probe.st.meta['vus_per_block']}, lowered in "
        f"{probe.lower_s} s; {n} units")
    sg = probe if n == k else stager(M, n, out / "stager")
    summary["stager"] = {"lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"], "vus_per_block": sg.st.meta["vus_per_block"],
                         "pin": sg.st.pin, "n": n, "base": sg.base}
    units = cases(x, n)
    m = stage(M, sg, units)
    if m["oks"] != [u.want for u in units]:
        raise SystemExit(f"the staged ok bits {m['oks']} are not the reference's {[u.want for u in units]}")
    summary["staged"] = {"n": n, "ands": m["ands"], "unit_nnz": m["unit_nnz"], "k_log": m["k_log"], "nbl": m["nbl"],
                         "m": m["k_log"] + m["nbl"], "stage_s": m["stage_s"], "pin": m["pin"], "oks": m["oks"],
                         "kinds": [u.kind for u in units]}
    log(f"staged: {n} units, {m['ands']} ANDs a unit, m = {m['k_log']} + {m['nbl']}, ok bits {m['oks']} (the "
        f"reference's), {m['stage_s']} s")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    res = audit(M, av, sg, m, units, out / "w63", a.port or N2._free_port())
    known = (m["dir"] / "circuit.txt").read_bytes() + bytes(64) + x.pads
    blobs, held, public = N2.public_pieces(M.a8, *blobs_of(m), known)
    blobs, held, nsparse = sparse(blobs, held)
    res["received"] = {**M.a8.received(out / "w63" / "verifier", m["dir"] / f"inst-{n}.bin", blobs, held),
                       "public_pieces": len(public), "sparse_pieces": nsparse}
    summary["audit"] = res
    rec = res["received"]
    got = [r["verdict"] for r in res.get("units", [])]
    want = ["accept" if u.want else "reject" for u in units]
    ok = (res["accepted"] and all(res["verdicts"].values()) and got == want and not rec["verifier_found"]
          and rec["control"]["holds_all"])
    log(f"proofs {res['verdicts']}, audit {res['audit']}, Lean {res['why']}, units "
        f"{[(r['kind'], r['ok'], r['verdict']) for r in res.get('units', [])[:k]]}, per unit {res['per_unit']}, "
        f"verifier received {rec['verifier_bytes']} bytes, witness pieces found there: {rec['verifier_found'] or 'none'} "
        f"(of {len(blobs)}; {len(public)} public, {nsparse} sparse left out), control {rec['control']['blobs_found']}; "
        f"{'as expected' if ok else 'UNEXPECTED'}")
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ingress_scan_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="the w63 class's bucket units proved --zk and held by Lean")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=0, help="the session's port (0: one the OS has free)")
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the run, the units and the reference only: nothing lowered")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    sys.exit(main())
