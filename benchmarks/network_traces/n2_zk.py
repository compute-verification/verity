"""N2's head checks proved in zero knowledge beside N2's verdict of record, on a certifier-serve run: each link-window's
head checks (`head_check.NetworkHeadCheck_v2`, the gated part of N2's unit, `service.LinkWindows`) proved by C-Flock
`prove --zk` behind the Lean firewall and held by the Lean verifier, through the one-stage audit as
`benchmarks/one_stage/a8.py` proves its openings. The stream's verdict of record is still N2's host verify, which reads
the opening (`link_windows`), and the proved template is the row adapter, not the Program N2 registers.

    python benchmarks/network_traces/n2_zk.py virtual --out DIR [--requests 40] [--tokens 4]
    python benchmarks/network_traces/n2_zk.py prove --run DIR [--extra DIR] --bin flock-circuit --lean flock-verify
      --out DIR [--points 1] [--dry]

* **The run.** A certifier-serve run's directory (`certifier_serve.py rehearse`'s per strategy, or `virtual`'s): its
  `plan.json`, `calibration.json`, `deployment.json` and `certified.jsonl`. `virtual` makes one on an injected clock
  against a stub endpoint that answers at once (certifier-serve's tests' run), honest and `extra`, with fresh keys.
* **Instances** (`link_windows`). The run's `LinkWindows`, rebuilt as `certifier_serve.check` builds it; per link-window
  its certified records committed (`LinkWindows.commit`, fresh salts), registered on the link's stream, selected
  (Selection All) and each end's head check taken under the registered root (`LinkWindows.head_checks`).
* **Rows** (`NetworkHeadCheckRows`). C-Flock's statement takes u16 rows, and the head check's parameters are no whole
  number of u16 words (`direction` is one bit, and an odd L makes the head an odd number of bytes). The adapter takes
  two rows, ``pub`` (the head check's public inputs in its signature's order: the registered root, the link-window's
  domain, the link, the end, the window, the end's grid, its profile's row and the certificate's signed digest) and
  ``wit`` (the head, its leaf's salt, its two siblings and the certifier's salt), each zero-padded to whole words,
  slices them into `NetworkHeadCheck_v2{L, T, COUNTS}`'s arguments and calls it unchanged: its one output, ``o0``, is
  the head check's. Nothing reads the padding. `rows_of` encodes an instance's rows from `head_check.arguments`.
* **Members.** One statement per (round, end), a template of n units, n the run's windows: ``honest`` both ends;
  ``tampered``, the egress end of the honest records with window 0's head one count off (one frame more in its first
  bucket that has room), committed by the prover under its own root, its certificate still the original's, so its
  counts and its signed digest refuse it; and with ``--extra``, the egress end of certifier-serve's ``extra`` run (the
  shim wrote one event more, the certifier certified it), which only its counts refuse. Each member is registered,
  every unit drawn by the verifier's Lean (`flock-verify draw --stratified n`), proved `prove --zk` behind the firewall
  (``FC_FIREWALL=1``, ``FC_FIREWALL_LEAN``) and held by Lean `verify --zk --partition --program`.
* **What the verifier opens.** Of each unit its ``pub`` row, to its own copy (`public_words`, from the link's public
  fields, the registered root and the certificate's signed digest), and its verdict row ``o0``, under the salts the
  prover sends for those two rows; never ``wit``. A link-window's head checks pass when both ends' units open with
  verdict 1. What the verifier received is scanned for every piece of every head, head salt, sibling and certifier salt
  (`a8.received`) but the pieces the public circuit holds (`public_pieces`); the prover's instance file, which holds the
  heads, is the control.
* **Cost** (``--points``): sessions on the honest members whose verifier draws its first k units, behind the firewall
  (`a8.cost`), for the prover's time and proof bytes per unit.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import sys
import tempfile
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ONE_STAGE = HERE.parent / "one_stage"
sys.path.insert(0, str(HERE))

import certifier_serve as cs  # noqa: E402
from verity.ml.circuits import boolean as BL  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.ml.circuits.types import Array  # noqa: E402
from verity.core.protocols.certifier import head_check as HC  # noqa: E402
from verity.core.protocols.certifier.certificate import CertifiedRecord, Direction, Issuer  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402

PUBLIC = ("record", "record_domain", "link", "direction", "window", "grid", "profile", "signed")
ROWS = ("pub", "wit")
ENDS = ("egress", "ingress")
OPENED = ("o0", "pub")


def log(msg: str) -> None:
    print(f"N2ZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the rows ---------------------------------------------------------------------------------------------------------

def row_bits(fn) -> dict[str, int]:
    """Each row's bits for the head check `fn`: its parameters' (``pub`` the public ones), padded to whole u16 words."""
    params, _ = fn.signature()
    bits = {r: sum(t.leaves for n, t in params if (n in PUBLIC) == (r == "pub")) for r in ROWS}
    return {r: 16 * -(-b // 16) for r, b in bits.items()}


def adapter(name: str, inner) -> CompositeDefinition:
    """A Definition on two bit rows, ``pub`` and ``wit`` (`row_bits`), that slices them into the arguments of
    ``inner(L, T, COUNTS)`` (a Definition of the head check's signature) and returns its output."""

    def signature(S):
        fn = inner(S.L, S.T, S.COUNTS)
        bits = row_bits(fn)
        return tuple((r, Array(bits[r], BL.BIT)) for r in ROWS), fn.signature()[1]

    def body(B, S, pub, wit):
        fn = inner(S.L, S.T, S.COUNTS)
        refs, at, args = {"pub": HC._refs(pub), "wit": HC._refs(wit)}, dict.fromkeys(ROWS, 0), []
        for n, t in fn.signature()[0]:
            r = "pub" if n in PUBLIC else "wit"
            args.append(HC._coll(t, refs[r][at[r]:at[r] + t.leaves]))
            at[r] += t.leaves
        return B.call(fn, *args)

    return CompositeDefinition(name, 1, ("L", "T", "COUNTS"), signature, body, conformance="exact-model-tested",
                               register=False, doc=f"{name}: the head check on two u16-word rows, pub (its public inputs) "
                                                   "and wit (its witness), each zero-padded; the padding is read by nothing")


NetworkHeadCheckRows = adapter("NetworkHeadCheckRows", lambda L, T, counts: HC.unit(L, T, bool(counts)))


def unit(key: tuple[int, int, int]):
    """The adapter around `head_check.unit(L, T, COUNTS)`."""
    L, T, counts = key
    return bind(NetworkHeadCheckRows, L=L, T=T, COUNTS=int(counts))


def _words(bits: list[int]) -> list[int]:
    return [sum(b << j for j, b in enumerate(bits[k:k + 16])) for k in range(0, len(bits), 16)]


def rows_of(inst: HC.Instance) -> dict[str, list[int]]:
    """The instance's ``pub`` and ``wit`` rows as u16 words: its arguments' bits (`head_check.arguments`) in order,
    little-endian, zero-padded."""
    params, _ = inst.unit.signature()
    bits: dict[str, list[int]] = {r: [] for r in ROWS}
    for (n, _t), a in zip(params, HC.arguments(inst)):
        bits["pub" if n in PUBLIC else "wit"] += a
    width = row_bits(inst.unit)
    return {r: _words(b + [0] * (width[r] - len(b))) for r, b in bits.items()}


def public_words(lw: LinkWindows, window: int, direction: int, root: bytes, signed: bytes) -> list[int]:
    """The ``pub`` row the verifier derives itself: the registered `root`, the link-window's domain, the link, the end,
    the window, the end's grid and its profile's row, and `signed`, the certificate's signed digest."""
    L, T, _ = lw.head_key(direction)
    profile = lw.occupancy[direction]
    blank = HC.Instance(root, lw.domain(window).domain_id, lw.link.encode("utf-8"), direction, window, lw.grid(direction),
                        None if profile is None else tuple(profile[window]), signed, bytes(HC.Shape(L, T).HL),
                        bytes(HC.SALT.leaves // 8), (bytes(64), bytes(64)), bytes(64), T)
    return rows_of(blank)["pub"]


# --- the run's link-windows -------------------------------------------------------------------------------------------

def link(run: Path) -> tuple[LinkWindows, dict]:
    """The run's link and its plan, as `certifier_serve.check` builds them."""
    plan, report, dep = (json.loads((run / f).read_text()) for f in ("plan.json", "calibration.json", "deployment.json"))
    setup = cs.ar.Setup(report)
    profile = plan.get("occupancy") or {}
    lw = LinkWindows(dep["links"]["n2"], setup.egress_params, setup.sync_set, cs.read_public(dep["certifier_key"]),
                     dep["run_id"], bytes.fromhex(dep["session"]), ingress=setup.ingress_params,
                     proof_deadline=cs.PROOF_DEADLINE_S, occupancy=(profile.get("egress"), profile.get("ingress")),
                     served=cs.served_profile(plan))
    return lw, plan


def link_windows(run: Path, edit=None, stream: bool = True, clock=time.monotonic) -> SimpleNamespace:
    """Every link-window of the run with both certified records: its opening (fresh salts) and each end's head check
    under its root. With `stream`, on N2's stream as certifier-serve's check takes it: registered, selected (Selection
    All), and its outcome of record decided before the next registration, which the stream requires, by N2's verifier of
    record on the host (`LinkWindows.outcome`, with the Lean audit), its verdict kept as ``host``; without, each root is
    the commitment's own (`LinkWindows.root`). ``edit(lw, w, pair)``: the (egress, ingress) records the prover commits in
    place of the certified ones."""
    lw, plan = link(run)
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    st = lw.stream(budget=plan["windows"], delta=cs.DELTA, challenger="n2-zk", coins=cs._no_coins, clock=clock) \
        if stream else None
    certified, refused = cs.read_certified(run / "certified.jsonl")
    out, previous = [], None
    for w in range(plan["windows"]):
        pair = certified.get((Direction.EGRESS, w)), certified.get((Direction.INGRESS, w))
        if None in pair:
            break
        if edit is not None:
            pair = edit(lw, w, pair)
        opening = lw.commit(w, *pair)
        x = SimpleNamespace(window=w, opening=opening, digest=None, drawn=None, host=None)
        if st is None:
            x.root = lw.root(opening)
        else:
            reg = lw.register(st, lw.registration(opening, previous))
            previous = x.digest = reg.digest
            draw = st.select(reg)
            x.root, x.drawn = reg.record["roots"][0]["root"], dict(draw.draw)
            outcome = st.deliver(draw, [lw.session(opening, draw)]) or lw.outcome(st, draw, opening, syncs)
            x.host = {"accepted": outcome.accepted, "codes": list(outcome.verdict.codes), "public": dict(outcome.public)}
        x.checks = lw.head_checks(opening, bytes.fromhex(x.root))
        out.append(x)
    return SimpleNamespace(lw=lw, plan=plan, windows=out, issuer_refused=refused,
                           network=lw.network_verdict(st, plan["windows"]) if st is not None else None)


def one_count(lw: LinkWindows, w: int, pair: tuple[CertifiedRecord, CertifiedRecord]):
    """Window 0's egress head with one frame more in its first bucket that has frames and room (else its first with
    room), the certifier's salt and certificate unchanged: what a prover that changed one count commits."""
    if w != 0:
        return pair
    egress, ingress = pair
    counts = list(egress.head.counts)
    r = egress.head.allocation
    t = next((t for t, n in enumerate(counts) if 0 < n < r), None)
    t = t if t is not None else next(t for t, n in enumerate(counts) if n < r)
    counts[t] += 1
    return CertifiedRecord(replace(egress.head, counts=tuple(counts)), egress.salt, egress.certificate), ingress


# --- a run on an injected clock ---------------------------------------------------------------------------------------

async def _endpoint(reader, writer):
    """A streaming completions endpoint that answers at once: `max_tokens` chunk events and `[DONE]`, chunked."""
    import re

    head = await reader.readuntil(b"\r\n\r\n")
    body = json.loads(await reader.readexactly(int(re.search(rb"content-length: (\d+)", head, re.I).group(1))))
    events = [b'data: {"choices":[{"text":" t%d"}]}\n\n' % k for k in range(body["max_tokens"])] + [b"data: [DONE]\n\n"]
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                 + b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in events) + b"0\r\n\r\n")
    await writer.drain()
    writer.close()


def _virtual_run(run: Path, plan: dict, report: dict, url: str, upstream: tuple, certifier, challenger, session: bytes,
                 strategy: str) -> None:
    setup = cs.ar.Setup(report)
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "calibration.json").write_text(json.dumps(report))
    shim = plan if strategy == "honest" else cs.ar.adversary_plan(plan, setup, strategy)

    async def after(proxies, clock):
        record = cs.deployment(proxies[0], setup, plan["windows"], session=session, n2_link=cs.N2_LINK,
                               link=("127.0.0.1", 0), reveal=("127.0.0.1", 0), origin_unix=0.0, url=url)
        (run / "deployment.json").write_text(json.dumps(record, indent=1))

    publish = cs.CertifyingPublisher(run, setup, Issuer(certifier, session), cs.N2_LINK, threaded=False)
    cs.ar.run_virtual(shim, report, run, upstream, salt=True, publish=publish, after=after,
                      proxy_args={"challenger_key": challenger.public_key(), "signer": certifier})


def virtual(out: Path, requests: int = 40, tokens: int = 4, strategies=("honest", "extra"), height: int = 6) -> dict:
    """certifier-serve's run on an injected clock against `_endpoint`, one per strategy under `out`, with fresh LMS keys
    of `height` in a scratch directory that is removed. Returns the plan."""
    keys = Path(tempfile.mkdtemp(prefix="n2-zk-keys-"))
    challenger = cs.signer(keys / "challenger", height)
    session = secrets.token_bytes(32)

    async def go():
        endpoint = await asyncio.start_server(_endpoint, "127.0.0.1", 0)
        upstream = endpoint.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        probed = await asyncio.to_thread(cs.probe, url, cs.request_body("stub", tokens), 2, clock=lambda: 0.0)
        report, plan = cs.served_plan(probed, requests)
        for strategy in strategies:
            certifier = cs.signer(keys / strategy, height)
            try:
                await asyncio.to_thread(_virtual_run, out / strategy, plan, report, url, upstream, certifier, challenger,
                                        session, strategy)
            finally:
                certifier.close()
        endpoint.close()
        return plan

    try:
        return asyncio.run(go())
    finally:
        challenger.close()
        import shutil

        shutil.rmtree(keys)


# --- staging, the sessions and the verifier ---------------------------------------------------------------------------

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


def builders() -> tuple:
    """Give the traced parts the head check calls that carry no builder (`head_check._part`'s and
    `gates.lms._leaf_definition`'s; `frame_v3._part` and `boolean.trace.definition` keep one) the builder their trace was
    taken from, so C-Flock lowers each activation by it, commits included (`ir_lower._boolean_activation`, which calls
    ``build(C, S, *e)`` for a part with statics and ``build(C, *e)`` without). Gate by gate their linear forms grow without
    bound: one SHA-512 compression passes 1 GB, the unit 108 GB in 3.5 minutes. Returns the parts."""
    from verity.ml.commitments.gates import lms
    from verity.ml.commitments.gates.sha512 import compress_gadget

    parts = ((lms.Sha512Compress, lambda C, *e: compress_gadget(C, list(e[:8]), list(e[8:24]))),
             (lms.LmsEqual, lms._build_equal), (HC.NetworkHeadFields, HC._build_fields),
             (HC.NetworkHeadCounts, HC._build_counts))
    for d, build in parts:
        if getattr(d, "build", None) is None:
            d.build = build
    return tuple(d for d, _ in parts)


def stager(M, key: tuple[int, int, int], end: str, n: int, out: Path) -> SimpleNamespace:
    """One end's statement program and partition (one template, n units) and its stager (the unit lowered once, its
    traced parts by their builders, `builders`)."""
    import gc

    builders()
    defn = unit(key)
    bits = row_bits(HC.unit(key[0], key[1], bool(key[2])))
    rows = [(r, M.V2, bits[r], [r]) for r in ROWS]
    program = M.P.mixed_population_program([(defn, n)])
    query = M.P.template_instances_query([defn])
    part = M.P.partition(program, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(program)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    t, was = time.time(), gc.isenabled()
    gc.disable()
    try:
        st = M.CS.UnitStager(defn, f"unit/n2-head-{end}", rows, [(M.V2, (0,))], circuit=out / "circuit.txt", bind=bound)
        st.pin
    finally:
        if was:
            gc.enable()
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, key=key, end=end, rows=rows, program=program, query=query, n=n, base=base, st=st,
                           out=out, lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, sg: SimpleNamespace, insts: list[HC.Instance], name: str, salt_key: bytes) -> dict:
    """One member's statement on its end's stager: instance i's rows (`rows_of`), every row under a salt from the
    prover's ``salt_key``."""
    t0 = time.time()
    st, n, out = sg.st, len(insts), sg.out / f"stage-{name}"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {k_log} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    words = [rows_of(i) for i in insts]
    data = {r: M.np.asarray([w[r] for w in words], dtype=M.np.uint16) for r in ROWS}
    set_name = f"{st.iset.name}-{name}"
    got = st.stage(data, out, salt_key, indices=range(sg.base, sg.base + n), set_name=set_name)
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    m = {"tag": name, "name": name, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name,
         "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"],
         "ports": st.ports, "oks": [int(x) for x in M.np.asarray(got["outs"]["o0"])[:, 0]],
         "partition_file": sg.partition_file, "program_file": sg.program_file, "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    return m


TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")


def audit_member(M, a, sg, m: dict, rd: SimpleNamespace, d: int, out: Path, port: int) -> dict:
    """Member m registered, drawn by the verifier's Lean (every unit), proved --zk behind the firewall and held by Lean;
    then the verifier's side: each unit's ``pub`` row opened to its own copy (`public_words`) and its verdict row read.
    Writes what the verifier received under ``out / "verifier"``."""
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
    win = {"kind": "stand-in", "of": f"N2's {ENDS[d]} head checks of the link-windows of a certifier-serve run (n2_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "head_check: each unit's public record input is the link-window's registered N2 root"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    receipt = R.receipt(rec, "one-stage-e2e verifier (n2_zk.py)")
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
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program"}
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

    held = set(proved) if result.get("accepted") else set()
    port_ = {x.name: x for x in m["ports"]}
    bcs = a7.public_bcs(m["dir"] / f"pub-{n}.bin", m["ports"])
    units, sent = [], []
    for i, x in enumerate(rd.windows):
        record_ = x.opening.records[d]
        mine = public_words(rd.lw, x.window, d, bytes.fromhex(x.root), record_.certificate.digest)
        word = m["oks"][i]
        opened = ((m["base"] + i) in held and a7.opens(port_["pub"], mine, m["salts"]["pub"][i], bcs[i]["pub"])
                  and a7.opens(port_["o0"], [word], m["salts"]["o0"][i], bcs[i]["o0"]))
        units.append({"window": x.window, "unit": m["base"] + i, "verdict": "malformed" if not opened
                      else "accept" if word == 1 else "reject"})
        sent.append({"unit": m["base"] + i, "window": x.window, "verdict": word,
                     **{f"salt/{r}": m["salts"][r][i].hex() for r in OPENED}})

    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    for f in (out / "registration.json", sg.partition_file, sg.program_file, m["dir"] / "circuit.txt",
              m["dir"] / f"pub-{n}.bin", sdir / "draw.json", out / "audit.json"):
        os.link(f, vdir / f.name)
    for f in sorted((sdir / "sessions").rglob("*")):
        if f.is_file():
            os.link(f, vdir / f"session-{f.parent.name}-{f.name}")
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    (vdir / "n2.json").write_text(json.dumps([{"window": x.window, "registration": x.digest, "root": x.root,
                                               "signed": x.opening.records[d].certificate.digest.hex()}
                                              for x in rd.windows], indent=1))
    proofs = (live or {}).get("proof_bytes") or []
    return {**res, "accepted": bool(result.get("accepted")), "units": units,
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()}, "why": {x: y.get("why") for x, y in verdicts.items()},
            "timing": statement[name], "lean_verify_s": lv.get("verify_s"),
            "per_unit": {"prove_total_s": _per(timing.get("prove_total_s"), n), "e2e_s": _per(timing.get("e2e_s"), n),
                         "proof_bytes": _per(sum(proofs) if proofs else None, n),
                         "lean_verify_s": _per(lv.get("verify_s"), n)},
            "opened_rows": list(OPENED)}


def _per(x, n: int):
    return None if x is None else round(x / n, 3)


def firewall(M, m: dict) -> dict:
    """A firewalled prover's environment (`FC_FIREWALL=1`, its Lean frame named by ``FC_FIREWALL_LEAN``)."""
    if not os.environ.get("FC_FIREWALL_LEAN"):
        raise SystemExit("FC_FIREWALL_LEAN names no Lean firewall frame (flock-verify's flock-firewall)")
    return {"FC_FIREWALL": "1", **M.a0.firewall_env(m)}


def expected(name: str, failed: list[list[str]]) -> bool:
    """Whether each link-window's failed head-check parts are the round's: none when honest; tampered, window 0's
    counts and signed digest only; extra, the counts alone on at least one window."""
    if name == "honest":
        return not any(failed)
    if name == "tampered":
        return failed[0] == ["counts", "signed"] and not any(failed[1:])
    return all(f in ([], ["counts"]) for f in failed) and ["counts"] in failed


def blobs_of(rd: SimpleNamespace, d: int) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each head, its leaf's salt, its siblings and the certifier's
    salt; the heads are what the prover's instance file holds as they are."""
    blobs, held = {}, set()
    for x in rd.windows:
        inst = x.checks[d]
        blobs[f"head {x.window}"] = inst.head
        held.add(f"head {x.window}")
        blobs[f"head salt {x.window}"] = inst.salt
        blobs[f"certifier salt {x.window}"] = inst.salt_c
        for k, s in enumerate(inst.path):
            blobs[f"sibling {k} {x.window}"] = s
    return blobs, held


def public_pieces(a8, blobs: dict[str, bytes], held: set[str], public: bytes) -> tuple[dict[str, bytes], set[str], list[str]]:
    """`blobs` cut into the scan's aligned pieces (`a8.needles`'s), each distinct piece once under its first name
    (``"<blob> @<offset>"``), less the pieces `public` holds raw or as hex. `public` is the staged circuit, which the
    stager writes from the Definition before any instance, so what it holds the verifier has whatever the witness: a head's
    empty buckets are sixteen zero bytes, which the circuit holds too. Returns the pieces to scan for, the held ones (pieces
    of a blob in `held`) and the excluded pieces' hex."""
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


def prove(a) -> int:
    M = _one_stage()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    secret = os.urandom(32)
    summary: dict = {"run": str(a.run), "extra": a.extra and str(a.extra)}
    t = time.time()
    rounds = {"honest": (link_windows(Path(a.run)), ENDS), "tampered": (link_windows(Path(a.run), edit=one_count), ("egress",))}
    if a.extra:
        rounds["extra"] = (link_windows(Path(a.extra)), ("egress",))
    lw, n = rounds["honest"][0].lw, len(rounds["honest"][0].windows)
    summary["link"] = {"name": lw.link, "windows": n, "keys": {e: lw.head_key(d) for d, e in enumerate(ENDS)},
                       "profile": rounds["honest"][0].plan["occupancy"]["kind"],
                       "planned_windows": rounds["honest"][0].plan["windows"]}
    summary["instances_s"] = round(time.time() - t, 1)
    summary["host"] = {name: {"windows": [{"window": x.window, "registration": x.digest, **(x.host or {})} for x in rd.windows],
                              "network": rd.network} for name, (rd, _) in rounds.items()}
    for name, h in summary["host"].items():
        log(f"{name}: N2's host verdict of record per link-window "
            f"{[(x.get('accepted'), x.get('codes')) for x in h['windows']]}, network {(h['network'] or {}).get('accepted')}")
    members = {}
    for name, (rd, ends) in rounds.items():
        if len(rd.windows) != n:
            raise SystemExit(f"{name}: {len(rd.windows)} link-windows with both records, the honest run's {n}")
        for d, end in enumerate(ENDS):
            if end not in ends:
                continue
            insts = [x.checks[d] for x in rd.windows]
            if any(i is None for i in insts):
                raise SystemExit(f"{name} {end}: a link-window the unit takes no instance of: "
                                 f"{[x.window for x, i in zip(rd.windows, insts) if i is None]}")
            parts = [HC.reference_parts(i) for i in insts]
            members[f"{name}-{end}"] = SimpleNamespace(round=name, d=d, end=end, rd=rd, insts=insts, parts=parts,
                                                       ref=[int(all(p.values())) for p in parts])
            log(f"{name} {end}: {n} link-windows, the head check's reference {members[f'{name}-{end}'].ref}, failed parts "
                f"{[[k for k, ok in p.items() if not ok] for p in parts]}")
    for k, mb in members.items():
        failed = [[x for x, ok in p.items() if not ok] for p in mb.parts]
        if not expected(mb.round, failed):
            raise SystemExit(f"{k}: the head checks' failed parts {failed} are not the {mb.round} round's")
    summary["reference"] = {k: {"ref": mb.ref, "failed": [[x for x, ok in p.items() if not ok] for p in mb.parts]}
                            for k, mb in members.items()}
    if a.dry:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        log("dry: done")
        return 0

    salt_key = hashlib.sha512(secret + b"row salts").digest()[:32]
    stagers, staged = {}, {}
    for k, mb in members.items():
        if mb.end not in stagers:
            stagers[mb.end] = sg = stager(M, lw.head_key(mb.d), mb.end, n, out / f"stager-{mb.end}")
            summary.setdefault("stagers", {})[mb.end] = {"key": sg.key, "lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"],
                                                         "vus_per_block": sg.st.meta["vus_per_block"], "pin": sg.st.pin,
                                                         "rows": {r: b for r, _s, b, _p in sg.rows}}
            log(f"stager {mb.end}: {sg.defn.id}, {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
        m = stage(M, stagers[mb.end], mb.insts, k, salt_key)
        if m["oks"] != mb.ref:
            raise SystemExit(f"{k}: the staged verdict words {m['oks']} are not the reference's {mb.ref}")
        staged[k] = m
        summary.setdefault("members", {})[k] = {"n": n, "ands": m["ands"], "unit_nnz": m["unit_nnz"], "stage_s": m["stage_s"],
                                                "k_log": m["meta"]["k_log"], "pin": m["pin"], "verdict_words": m["oks"]}
        log(f"staged {k}: {n} units, {m['ands']} ANDs, k_log {m['meta']['k_log']}, verdicts {m['oks']}, {m['stage_s']} s")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta, points=a.points, arms=["fw"], cold=1)
    ok = True
    for j, (k, mb) in enumerate(members.items()):
        m, sg = staged[k], stagers[mb.end]
        res = audit_member(M, av, sg, m, mb.rd, mb.d, out / k, a.port + j)
        blobs, held, public = public_pieces(M.a8, *blobs_of(mb.rd, mb.d), (m["dir"] / "circuit.txt").read_bytes())
        res["received"] = {**M.a8.received(out / k / "verifier", m["dir"] / f"inst-{n}.bin", blobs, held),
                           "public_pieces": public}
        summary["members"][k].update(res)
        verdicts = [u["verdict"] for u in res.get("units", [])]
        good = (res["accepted"] and all(res["verdicts"].values()) and verdicts == ["accept" if r else "reject" for r in mb.ref]
                and not res["received"]["verifier_found"] and res["received"]["control"]["holds_all"])
        ok &= good
        log(f"member {k}: proofs {res['verdicts']}, audit {res['audit']}, units {verdicts}, per unit {res['per_unit']}, "
            f"verifier received {res['received']['verifier_bytes']} bytes, witness pieces found there: "
            f"{res['received']['verifier_found'] or 'none'} (of {len(blobs)}; {len(public)} the public circuit holds, "
            f"left out), control {res['received']['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    table = {}
    for name in rounds:
        for i in range(n):
            row = {mb.end: summary["members"][k]["units"][i]["verdict"] for k, mb in members.items()
                   if mb.round == name and summary["members"][k].get("units")}
            table.setdefault(name, []).append({"window": i, **row, "head_checks": "pass" if row and all(
                v == "accept" for v in row.values()) else "fail"})
    summary["windows"] = table
    log("windows: " + json.dumps(table))

    if a.points:
        summary["cost"] = {}
        for end in ENDS:
            k = f"honest-{end}"
            if k in staged:
                rows = M.a8.cost(av, staged[k], out / "cost" / end, a.port + 20 + 10 * ENDS.index(end))
                summary["cost"][end] = rows
                ok &= all(r["accepted"] for r in rows) and all(r["lean"]["accepted"] for r in rows if "lean" in r)
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="n2_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("virtual", help="certifier-serve's honest and extra runs on an injected clock, stub endpoint")
    v.add_argument("--out", required=True, type=Path)
    v.add_argument("--requests", type=int, default=40)
    v.add_argument("--tokens", type=int, default=4)
    p = sub.add_parser("prove", help="the run's head checks proved --zk and held by Lean")
    p.add_argument("--run", required=True, type=Path, help="the honest run's directory")
    p.add_argument("--extra", type=Path, help="certifier-serve's extra run of the same plan")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7681)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--points", default="", help="cost points: the k drawn units of each session, comma-separated")
    p.add_argument("--dry", action="store_true", help="the instances and their references only: nothing lowered or proved")
    a = ap.parse_args(argv)
    if a.command == "virtual":
        plan = virtual(a.out, a.requests, a.tokens)
        log(f"virtual: {plan['windows']} windows, served {plan['served']}, under {a.out}")
        return 0
    a.points = [int(x) for x in a.points.split(",") if x]
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
