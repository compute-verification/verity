"""N2's advised unit proved in zero knowledge: each link-window of an advised link, its two head checks and the advised
counts on the counts those heads hold (`service.LinkWindows(advised=…)`'s unit, `NetworkAdvisedLinkWindow_v1`, less
its certificates' check, which has no gates), proved by C-Flock `prove --zk` behind the Lean firewall and held by the
Lean verifier, through the one-stage audit as `n2_zk.py` proves the head checks alone. The stream's verdict of record
is still N2's host verify, which reads the opening and the advice.

    python benchmarks/network_traces/n2_advised_zk.py prove --bin flock-circuit --lean flock-verify --out DIR
      [--seed 7] [--points 1] [--dry]

* **The link** (`simulated`). The advised tests' simulated vLLM serving (`advised_traffic.serve`: varied prompt and
  output lengths, chunked prefill, prefix-cache hits, a preemption resumed from a hit, the async row past a finish, a
  step's rows out of request order, frames carried across a window and a response released in two windows) at
  `SHAPE`: T = 512 buckets on both ends, the default-budget root's size class (`advised.circuit-check.py`: T = 512,
  R_in = 4, salted), egress R = 3, four windows and a burst of requests across each of the 0/1 and 2/3 boundaries, the
  certifier's own frame in every window. Its records are certified under a fresh LMS key and certifier session.
* **Members.** One statement per member, a template of one unit per window: ``honest``, the session's advice and
  records; ``moved``, one kept output's release bucket one later (`advised_traffic.moved`); ``short``, one request's
  output length one short (`advised_traffic.with_output_len`); ``extra``, one frame more handed to the egress certifier
  in its first busy bucket, the egress records certified anew. The advice is public and the heads are witness: moved
  and short change the view the verifier computes, extra the egress heads the prover commits. Each negative is the
  first of its kind that `advised.view` takes in every window, so the verifier has every unit's public row, and that
  the advised counts refuse in some window: the proof's verdict row refuses it, with both head checks passing. Each
  window goes through N2's stream as `n2_zk.link_windows` takes it, its host verdict of record kept as ``host``.
* **Rows** (`NetworkAdvisedRows`). Two u16 rows, as `n2_zk`'s adapter takes them: ``pub``, each end's head check's
  public inputs in its signature's order (the registered root, the link-window's domain, the link, the end, the
  window, the end's grid and the certificate's signed digest), then the window's view of the advice (`advised.view`:
  each bucket's released frames, the request-frame floor and the window's received frames); ``wit``, each end's head,
  its leaf's salt, its two siblings and the certifier's salt. Each is zero-padded to whole words, and nothing reads the
  padding. The adapter calls each end's head check (`head_check.unit(L, T, 0)`), cuts each head's counts as
  `service._checks` does, and calls the advised counts (`advised.unit(T, RIN, OWN)`) on the ingress and egress counts
  and the view; its one output, ``o0``, is the AND of the three.
* **What the verifier opens.** Of each unit its ``pub`` row, to its own copy (`public_words`, from the link's public
  fields, the registered root, both certificates' signed digests and its own view of the public advice), and its
  verdict row ``o0``; never ``wit``. What it received is scanned for every piece of both ends' heads, head salts,
  siblings and certifier salts (`n2_zk.blobs_of`, `a8.received`) but those the public circuit holds.
* **Cost** (``--points``, 1 by default): `a8.cost` sessions on the honest member whose verifier draws its first k
  units, behind the firewall: at k = 1 one link-window proved alone, its prover's seconds, its proof's bytes (the
  prover's LIVE line) and Lean `verify --zk`'s seconds, the per-window price.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TRAFFIC = HERE.parents[1] / "verity" / "core" / "protocols" / "certifier" / "tests" / "advised_traffic.py"
sys.path.insert(0, str(HERE))

import certifier_serve as cs  # noqa: E402
import n2_zk as Z  # noqa: E402
from n2_zk import ENDS, OPENED, ROWS  # noqa: E402
from verity.ml.circuits import boolean as BL  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.ml.circuits.refs import Coll, concat  # noqa: E402
from verity.ml.circuits.types import Array  # noqa: E402
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, parameters_digest  # noqa: E402
from verity.core.protocols.certifier.grid import GridParams  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402


def _traffic():
    spec = importlib.util.spec_from_file_location("advised_traffic", TRAFFIC)
    mod = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(mod)
    return mod


AT = _traffic()
T, WINDOWS, OWN, SEED = 512, 4, 1, 7
SHAPE = AT.Shape(T, WINDOWS, GridParams(T, AT.R, 1), GridParams(T, 4, 1),
                 tuple(range(T - 8, T + 8)) + tuple(range(3 * T - 8, 3 * T + 8)))
MEMBERS = ("honest", "moved", "short", "extra")
STATICS = ("L", "T", "RIN", "OWN")


def log(msg: str) -> None:
    print(f"N2ADV {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# --- the unit on two rows ---------------------------------------------------------------------------------------------

def params(S) -> list[tuple[str, object, str]]:
    """The unit's inputs in order, each (name, type, row): each end's head check's parameters (``pub`` its public
    ones, `n2_zk.PUBLIC`), then the advised counts' public inputs, the window's view (``pub``)."""
    out = [(f"{end}_{n}", t, "pub" if n in Z.PUBLIC else "wit") for end in ENDS
           for n, t in HC.unit(S.L, S.T, False).signature()[0]]
    return out + [(n, t, "pub") for n, t in AD.unit(S.T, S.RIN, S.OWN).signature()[0][2:]]


def row_bits(S) -> dict[str, int]:
    """Each row's bits: its inputs', padded to whole u16 words."""
    bits = {r: sum(t.leaves for _, t, row in params(S) if row == r) for r in ROWS}
    return {r: 16 * -(-b // 16) for r, b in bits.items()}


def _sig(S):
    bits = row_bits(S)
    return tuple((r, Array(bits[r], BL.BIT)) for r in ROWS), BL.BIT


def _body(B, S, pub, wit):
    refs, at = {"pub": HC._refs(pub), "wit": HC._refs(wit)}, dict.fromkeys(ROWS, 0)
    args = []
    for _, t, r in params(S):
        args.append(HC._coll(t, refs[r][at[r]:at[r] + t.leaves]))
        at[r] += t.leaves
    fn, sh = HC.unit(S.L, S.T, False), HC.Shape(S.L, S.T)
    names = [n for n, _ in fn.signature()[0]]
    oks, counts = [], []
    for d in range(len(ENDS)):
        end = args[d * len(names):(d + 1) * len(names)]
        oks.append(B.call(fn, *end))
        head = HC._refs(end[names.index("head")])
        counts.append(HC._coll(Array(S.T, AD.COUNT), head[8 * sh.CNT:8 * sh.ROOT]))
    oks.append(B.call(AD.unit(S.T, S.RIN, S.OWN), counts[1], counts[0], *args[len(ENDS) * len(names):]))
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BL.BIT), bits))


NetworkAdvisedRows = CompositeDefinition(
    "NetworkAdvisedRows", 1, STATICS, _sig, _body, conformance="exact-model-tested", register=False,
    doc="N2's advised unit less its certificates' check on two u16-word rows, pub (both head checks' public inputs and "
        "the window's view of the advice) and wit (both heads' witness), each zero-padded; the padding is read by "
        "nothing")


def key_of(lw: LinkWindows) -> tuple[int, int, int, int]:
    """The link's (L, T, RIN, OWN): its heads' (L, T), with no occupancy profile, and its advised counts' (RIN, OWN)."""
    keys = {lw.head_key(d) for d in range(len(ENDS))}
    T_, RIN, OWN_ = lw.advised_key()
    if keys != {(len(lw.link.encode("utf-8")), T_, 0)}:
        raise ValueError(f"an advised link's heads are of one size class with no profile, not {sorted(keys)}")
    return keys.pop()[0], T_, RIN, OWN_


def unit(key: tuple[int, int, int, int]):
    return bind(NetworkAdvisedRows, **dict(zip(STATICS, key)))


def rows_of(key: tuple[int, int, int, int], checks, view: AD.WindowView) -> dict[str, list[int]]:
    """A link-window's ``pub`` and ``wit`` rows as u16 words: each end's head check's arguments
    (`head_check.arguments`) and the view's (`advised.arguments`), little-endian, zero-padded."""
    bits: dict[str, list[int]] = {r: [] for r in ROWS}
    for inst in checks:
        for (n, _t), a in zip(inst.unit.signature()[0], HC.arguments(inst)):
            bits["pub" if n in Z.PUBLIC else "wit"] += a
    for a in AD.arguments((), (), view)[2:]:
        bits["pub"] += a
    width = row_bits(SimpleNamespace(**dict(zip(STATICS, key))))
    return {r: Z._words(b + [0] * (width[r] - len(b))) for r, b in bits.items()}


def public_words(lw: LinkWindows, window: int, root: bytes, signed: tuple[bytes, bytes], view: AD.WindowView) -> list[int]:
    """The ``pub`` row the verifier derives itself: for each end the registered `root`, the link-window's domain, the
    link, the end, the window and the end's grid, and `signed`'s, the end's certificate's signed digest; then its own
    `view` of the public advice."""
    key = key_of(lw)
    L, T_ = key[:2]
    blank = [HC.Instance(root, lw.domain(window).domain_id, lw.link.encode("utf-8"), d, window, lw.grid(d), None,
                         signed[d], bytes(HC.Shape(L, T_).HL), bytes(HC.SALT.leaves // 8), (bytes(64), bytes(64)),
                         bytes(64), T_) for d in range(len(ENDS))]
    return rows_of(key, blank, view)["pub"]


def builders() -> tuple:
    """`n2_zk.builders`, and the advised counts' traced parts given the builders their traces were taken from."""
    parts = ((AD.NetworkAdvisedOwn, AD._build_own), (AD.NetworkAdvisedBucket, AD._build_bucket),
             (AD.NetworkAdvisedEnd, AD._build_end))
    for d, build in parts:
        if getattr(d, "build", None) is None:
            d.build = build
    return Z.builders() + tuple(d for d, _ in parts)


# --- the link and its members -----------------------------------------------------------------------------------------

def _view(lw: LinkWindows, link: AD.LinkAdvice, w: int) -> AD.WindowView | None:
    try:
        return AD.view(link, w, lw.params.buckets, lw.params.allocation, AT.SYNCS, lw.sync_set,
                       lw.advised.request_frames)
    except ValueError:
        return None


def advised_refs(lw: LinkWindows, link: AD.LinkAdvice, egress, ingress) -> list[bool | None]:
    """The advised counts' reference on each window's records (raw, as the certifiers close them) and its view of
    `link`; None where `advised.view` refuses the window."""
    out = []
    for w, (e, i) in enumerate(zip(AT.counts(egress), AT.counts(ingress))):
        v = _view(lw, link, w)
        out.append(None if v is None else AD.reference(*lw.advised_key(), i, e, v.released, v.floor, v.received))
    return out


def negatives(lw: LinkWindows, s, shape=SHAPE) -> dict[str, tuple[AD.LinkAdvice, list, dict]]:
    """The three negatives on session `s`, each (its advice, its egress records, what was changed): the first of its
    kind that every window's view takes and the advised counts refuse in some window."""
    link, a = s.link, s.link.advice

    def refused(off, egress=s.egress) -> bool:
        got = advised_refs(lw, off, egress, s.ingress)
        return None not in got and not all(got)

    out = {}
    for k in range(len(a.release)):
        try:
            off = AT.moved(link, k, 1)
        except ADV.MalformedAdvice:
            continue
        if refused(off):
            out["moved"] = (off, s.egress, {"output": k, "from": a.release[k], "to": a.release[k] + 1})
            break
    for j, q in enumerate(a.requests):
        if q.output_len >= 2 and not AT.feeds_a_hit(a, j):
            off = AT.with_output_len(link, j, -1, shape)
            if refused(off):
                out["short"] = (off, s.egress, {"request": j, "output_len": q.output_len, "to": q.output_len - 1})
                break
    handed = list(s.handed)
    t = next(t for t, n in enumerate(handed) if n)
    handed[t] += 1
    egress = AT.egress_records(handed, shape)
    if refused(link, egress):
        out["extra"] = (link, egress, {"bucket": t, "handed": handed[t]})
    missing = [m for m in MEMBERS[1:] if m not in out]
    if missing:
        raise SystemExit(f"the session has no {missing} the views take and the advised counts refuse")
    return out


def simulated(seed: int = SEED, shape=SHAPE, own: int = OWN, height: int = 6) -> SimpleNamespace:
    """The advised tests' session `seed` at `shape`: the link (an advised profile of its windows, the tests' framing
    and `own`), and each member's advice and certified (egress, ingress) records per window, under a fresh LMS key of
    `height` (in a scratch directory that is removed) and a fresh certifier session."""
    s = AT.serve(seed, own, shape)
    keys = Path(tempfile.mkdtemp(prefix="n2-advised-keys-"))
    session = secrets.token_bytes(32)
    try:
        certifier = cs.signer(keys / "certifier", height)
        try:
            issuer = Issuer(certifier, session)
            lw = LinkWindows(AT.LINK, shape.egress, AT.SYNC_SET, certifier.public_key(), "n2-advised-zk", session,
                             ingress=shape.ingress, proof_deadline=cs.PROOF_DEADLINE_S,
                             advised=AD.AdvisedProfile(shape.windows, AT.FRAMING, own))
            ingress = [issuer.certify(r, Direction.INGRESS, shape.ingress, parameters_digest(shape.ingress))
                       for r in s.ingress]

            def egress(records):
                return [issuer.certify(r, Direction.EGRESS, shape.egress, parameters_digest(shape.egress, AT.SYNC_SET))
                        for r in records]

            honest = egress(s.egress)
            members = {"honest": (s.link, list(zip(honest, ingress)), {})}
            for name, (link, records, what) in negatives(lw, s, shape).items():
                members[name] = (link, list(zip(honest if records is s.egress else egress(records), ingress)), what)
        finally:
            certifier.close()
    finally:
        shutil.rmtree(keys)
    a = s.link.advice
    traffic = {"seed": seed, "requests": len(a.requests), "outputs": len(a.release), "steps": len(a.steps),
               "arrive": s.arrive, "received": list(s.link.received), "own": own}
    return SimpleNamespace(lw=lw, session=s, members=members, traffic=traffic)


def link_windows(lw: LinkWindows, link: AD.LinkAdvice, pairs, stream: bool = True, clock=time.monotonic) -> SimpleNamespace:
    """Every link-window of the member: its opening (fresh salts), its view of the advice and each end's head check
    under its root; with `stream`, on N2's stream as `n2_zk.link_windows` takes it, its outcome of record decided by N2's
    verifier of record on the host with the advice (`LinkWindows.outcome`), kept as ``host``."""
    st = lw.stream(budget=len(pairs), delta=cs.DELTA, challenger="n2-advised-zk", coins=cs._no_coins, clock=clock) \
        if stream else None
    out, previous = [], None
    for w, pair in enumerate(pairs):
        opening = lw.commit(w, *pair)
        x = SimpleNamespace(window=w, opening=opening, digest=None, drawn=None, host=None, view=_view(lw, link, w))
        if x.view is None:
            raise SystemExit(f"window {w}: the advice has no view of it, so no unit's public row")
        if st is None:
            x.root = lw.root(opening)
        else:
            reg = lw.register(st, lw.registration(opening, previous))
            previous = x.digest = reg.digest
            draw = st.select(reg)
            x.root, x.drawn = reg.record["roots"][0]["root"], dict(draw.draw)
            outcome = st.deliver(draw, [lw.session(opening, draw)]) or lw.outcome(st, draw, opening, AT.SYNCS,
                                                                                  advice=link)
            x.host = {"accepted": outcome.accepted, "codes": list(outcome.verdict.codes), "public": dict(outcome.public)}
        x.checks = lw.head_checks(opening, bytes.fromhex(x.root))
        if any(c is None for c in x.checks):
            raise SystemExit(f"window {w}: a head the unit takes no instance of")
        egress, ingress = (r.head.counts for r in opening.records)
        x.parts = {**{end: HC.reference_parts(c) for end, c in zip(ENDS, x.checks)},
                   "advised": AD.reference(*lw.advised_key(), ingress, egress, x.view.released, x.view.floor,
                                           x.view.received)}
        x.ref = int(all(all(x.parts[end].values()) for end in ENDS) and x.parts["advised"])
        out.append(x)
    return SimpleNamespace(lw=lw, link=link, windows=out,
                           network=lw.network_verdict(st, len(pairs)) if st is not None else None)


def failed(x) -> list[str]:
    """The window's failed parts: an end's head-check part as ``<end>/<part>``, and ``advised``."""
    out = [f"{end}/{k}" for end in ENDS for k, ok in x.parts[end].items() if not ok]
    return out + ([] if x.parts["advised"] else ["advised"])


def expected(name: str, fails: list[list[str]]) -> bool:
    """Whether each window's failed parts are the member's: none when honest; else the advised counts alone, in at
    least one window."""
    if name == "honest":
        return not any(fails)
    return all(f in ([], ["advised"]) for f in fails) and ["advised"] in fails


def blobs_of(rd: SimpleNamespace) -> tuple[dict[str, bytes], set[str]]:
    """`n2_zk.blobs_of` of both ends, each piece's name led by its end."""
    blobs, held = {}, set()
    for d, end in enumerate(ENDS):
        b, h = Z.blobs_of(rd, d)
        blobs |= {f"{end} {k}": v for k, v in b.items()}
        held |= {f"{end} {k}" for k in h}
    return blobs, held


# --- staging, the sessions and the verifier ---------------------------------------------------------------------------

def stager(M, key: tuple[int, int, int, int], n: int, out: Path) -> SimpleNamespace:
    """The statement program and partition (one template, n units) and its stager (the unit lowered once, its traced
    parts by their builders, `builders`)."""
    import gc

    builders()
    defn, bits = unit(key), row_bits(SimpleNamespace(**dict(zip(STATICS, key))))
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
        st = M.CS.UnitStager(defn, "unit/n2-advised", rows, [(M.V2, (0,))], circuit=out / "circuit.txt", bind=bound)
        st.pin
    finally:
        if was:
            gc.enable()
    (tid, base, cnt), = M.P.ranges(program, query)
    assert tid == M.P.template_id(defn) and cnt == n
    return SimpleNamespace(defn=defn, key=key, rows=rows, program=program, query=query, n=n, base=base, st=st, out=out,
                           lower_s=round(time.time() - t, 1), partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, sg: SimpleNamespace, rd: SimpleNamespace, name: str, salt_key: bytes) -> dict:
    """One member's statement on the stager: window i's rows (`rows_of`), every row under a salt from ``salt_key``."""
    t0 = time.time()
    st, n, out = sg.st, len(rd.windows), sg.out / f"stage-{name}"
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= k_log + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{name}: m = {k_log} + {nbl} is outside 25 .. {M.a7.M_MAX} (--zk and flock-circuit's shape)")
    words = [rows_of(sg.key, x.checks, x.view) for x in rd.windows]
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


def audit_member(M, a, sg, m: dict, rd: SimpleNamespace, out: Path, port: int) -> dict:
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
    win = {"kind": "stand-in", "of": "N2's advised units of the link-windows of an advised link (n2_advised_zk.py)",
           "members": {name: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + n]}},
           "link": "each unit's public record input is the link-window's registered N2 root, and its view the "
                   "verifier's own advised.view of the public advice"}
    rec = R.record(sg.program, sg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, sg.program, sg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              f"statement/{name}": a7.statement_checks(m, rec, want)}
    receipt = R.receipt(rec, "one-stage-e2e verifier (n2_advised_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False}
    p = a0.run([a.lean, "draw", "--program", str(sg.program_file), "--partition", str(sg.partition_file), "--stratified",
                str(n)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + n)
    sdir = out / "session"
    record, live, proc = M.a8.session(a, m, sdir, local, port, env=Z.firewall(M, m))
    draw_t = (record or {}).get("unit_draw") or {}
    v = V.check_draw(draw_t, {"population": n}, {"law": "subset", "k": len(local)})
    verdicts = {f"m0-verifier/{name}": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                        and (live or {}).get("roots") == fv["roots"] and v.ok,
                                        "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}}
    lv = a0.lean_verify(a.lean, m, n, sdir, zk=True)
    verdicts[f"lean-verify/{name}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition --program"}
    proved = a0.proved_units(m["header"], live or {})
    timing = {x: (live or {}).get(x) for x in Z.TIMING}
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
        signed = tuple(r.certificate.digest for r in x.opening.records)
        mine = public_words(rd.lw, x.window, bytes.fromhex(x.root), signed, _view(rd.lw, rd.link, x.window))
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
                                               "signed": [r.certificate.digest.hex() for r in x.opening.records]}
                                              for x in rd.windows], indent=1))
    (vdir / "advice.json").write_text(json.dumps({"advice": ADV.to_json(rd.link.advice), "received": list(rd.link.received),
                                                  "digest": AD.digest(rd.link)}, sort_keys=True))
    proofs = (live or {}).get("proof_bytes") or []
    return {**res, "accepted": bool(result.get("accepted")), "units": units,
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()}, "why": {x: y.get("why") for x, y in verdicts.items()},
            "timing": statement[name], "lean_verify_s": lv.get("verify_s"),
            "session_per_unit": {"prove_total_s": Z._per(timing.get("prove_total_s"), n),
                                 "proof_bytes": Z._per(sum(proofs) if proofs else None, n),
                                 "lean_verify_s": Z._per(lv.get("verify_s"), n)},
            "opened_rows": list(OPENED)}


def _live(sdir: Path) -> dict:
    """The prover's LIVE line in a session directory's ``prove.log``, or {}."""
    log_ = sdir / "prove.log"
    lines = log_.read_text().splitlines() if log_.exists() else []
    return next((json.loads(x[5:]) for x in lines if x.startswith("LIVE\t")), {})


def cost(M, av, m: dict, out: Path, port: int) -> list[dict]:
    """`a8.cost` on member m, each row with its prover's proof bytes and wire bytes (its LIVE line)."""
    rows = M.a8.cost(av, m, out, port)
    for row in rows:
        live = _live(out / f"{row['arm']}-k{row['k']}-r{row['rep']}")
        proofs = live.get("proof_bytes") or []
        row |= {"proof_bytes": sum(proofs) if proofs else None, "proofs": proofs,
                **{x: live.get(x) for x in ("wire_sent", "wire_recv", "nbl", "ands_per_vu")}}
    return rows


def prove(a) -> int:
    M = Z._one_stage() if not a.dry else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    sim = simulated(a.seed)
    lw, key = sim.lw, key_of(sim.lw)
    summary: dict = {"link": {"name": lw.link, "windows": WINDOWS, "key": dict(zip(STATICS, key)), "unit": lw.unit.id,
                              "egress": asdict(SHAPE.egress), "ingress": asdict(SHAPE.ingress)},
                     "traffic": sim.traffic, "changed": {k: w for k, (_, _, w) in sim.members.items()}}
    rounds = {name: link_windows(lw, link, pairs) for name, (link, pairs, _) in sim.members.items()}
    summary["instances_s"] = round(time.time() - t, 1)
    n = WINDOWS
    for name, rd in rounds.items():
        fails = [failed(x) for x in rd.windows]
        summary.setdefault("reference", {})[name] = {"ref": [x.ref for x in rd.windows], "failed": fails}
        summary.setdefault("host", {})[name] = {"windows": [{"window": x.window, "registration": x.digest, **x.host}
                                                            for x in rd.windows], "network": rd.network}
        log(f"{name}: N2's host verdict of record per link-window {[(x.host['accepted'], x.host['codes']) for x in rd.windows]}, "
            f"network {rd.network['accepted']}; the unit's reference {[x.ref for x in rd.windows]}, failed parts {fails}")
        if not expected(name, fails):
            raise SystemExit(f"{name}: the failed parts {fails} are not the {name} member's")
        if [x.host["accepted"] for x in rd.windows] != [bool(x.ref) for x in rd.windows]:
            raise SystemExit(f"{name}: the host verdicts aren't the unit's reference")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0

    salt_key = hashlib.sha512(os.urandom(32) + b"row salts").digest()[:32]
    sg = stager(M, key, n, out / "stager")
    summary["stager"] = {"unit": sg.defn.id, "lower_s": sg.lower_s, "k_log": sg.st.meta["k_log"],
                         "vus_per_block": sg.st.meta["vus_per_block"], "pin": sg.st.pin,
                         "rows": {r: b for r, _s, b, _p in sg.rows}}
    log(f"stager: {sg.defn.id}, {sg.lower_s} s, k_log {sg.st.meta['k_log']}, pin {sg.st.pin[:16]}")
    staged = {}
    for name, rd in rounds.items():
        m = stage(M, sg, rd, name, salt_key)
        ref = [x.ref for x in rd.windows]
        if m["oks"] != ref:
            raise SystemExit(f"{name}: the staged verdict words {m['oks']} are not the reference's {ref}")
        staged[name] = m
        summary.setdefault("members", {})[name] = {"n": n, "ands": m["ands"], "unit_nnz": m["unit_nnz"],
                                                   "stage_s": m["stage_s"], "k_log": m["meta"]["k_log"], "pin": m["pin"],
                                                   "verdict_words": m["oks"]}
        log(f"staged {name}: {n} units, {m['ands']} ANDs, k_log {m['meta']['k_log']}, verdicts {m['oks']}, {m['stage_s']} s")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta, points=a.points, arms=["fw"], cold=1)
    ok = True
    for j, (name, rd) in enumerate(rounds.items()):
        m = staged[name]
        res = audit_member(M, av, sg, m, rd, out / name, a.port + j)
        blobs, held, public = Z.public_pieces(M.a8, *blobs_of(rd), (m["dir"] / "circuit.txt").read_bytes())
        res["received"] = {**M.a8.received(out / name / "verifier", m["dir"] / f"inst-{n}.bin", blobs, held),
                           "public_pieces": public}
        summary["members"][name].update(res)
        verdicts = [u["verdict"] for u in res.get("units", [])]
        good = (res["accepted"] and all(res["verdicts"].values())
                and verdicts == ["accept" if x.ref else "reject" for x in rd.windows]
                and not res["received"]["verifier_found"] and res["received"]["control"]["holds_all"])
        ok &= good
        log(f"member {name}: proofs {res['verdicts']}, audit {res['audit']}, units {verdicts}, verifier received "
            f"{res['received']['verifier_bytes']} bytes, witness pieces found there: "
            f"{res['received']['verifier_found'] or 'none'} (of {len(blobs)}; {len(public)} the public circuit holds, "
            f"left out), control {res['received']['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["windows"] = {name: [{"window": u["window"], "proof": u["verdict"], "host": x.host["accepted"]}
                                 for u, x in zip(summary["members"][name].get("units", []), rd.windows)]
                          for name, rd in rounds.items()}
    log("windows: " + json.dumps(summary["windows"]))

    if a.points:
        rows = cost(M, av, staged["honest"], out / "cost", a.port + 20)
        summary["cost"] = rows
        one = next((r for r in rows if r["k"] == 1), None)
        if one is not None:
            summary["per_window"] = {"prove_total_s": one.get("prove_total_s"), "e2e_s": one.get("e2e_s"),
                                     "session_wall_s": one.get("wall_s"), "proof_bytes": one.get("proof_bytes"),
                                     "lean_verify_s": (one.get("lean") or {}).get("verify_s"),
                                     "max_child_rss_gb": one.get("max_child_rss_gb"), "of": "a session drawing one unit"}
            log(f"per window (one unit drawn alone): {summary['per_window']}")
        ok &= all(r["accepted"] for r in rows) and all(r["lean"]["accepted"] for r in rows if "lean" in r)
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="n2_advised_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="the advised link's units proved --zk and held by Lean")
    p.add_argument("--seed", type=int, default=SEED, help="the advised tests' session")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7881)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--points", default="1", help="cost points: the k drawn units of each session, comma-separated")
    p.add_argument("--dry", action="store_true", help="the instances, host verdicts and references only: nothing "
                                                      "lowered or proved")
    a = ap.parse_args(argv)
    a.points = [int(x) for x in a.points.split(",") if x]
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
