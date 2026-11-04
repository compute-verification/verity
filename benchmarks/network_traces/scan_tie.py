"""scan-tie: the egress tie (`scan_service.EgressTie`) on C-Flock's prover. An execution phase of a network certifier
link (scan_zk's `link`: the synthetic run or a warden_serve run's, at a setup's class) is one registration of the egress
tie's stream; every proof block is sampled, each of its four statements (``first``, ``ingress`` with the phase's
`IngressScanBucket` blocks but the last, ``last``, ``egress``) is staged on the verifier's own registered reads, proved
by Flock `prove --zk` behind the ZK server, delivered, and decided by `EgressTie.verify` (C-Flock's Lean verifier with
the verifier's registered reads); `EgressTie.tie` reads the verdict off the stream's outcomes.

    python benchmarks/network_traces/scan_tie.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]
        [--class w63|w63k4|test] [--warden-run RUN [--outputs FILE]] [--port P] [--proof-deadline S]

* **The rounds** (`ROUNDS`): ``honest``, the network certifier's rows as it signed them, accepted; ``tampered``, the
  phase's egress with one token's id changed in one `DATA` cell and the phase re-signed (scan_zk's `_tamper`: a
  network certifier that saw one token the trace doesn't render), refused at the egress statement, whose ``o1`` opens
  no row of the signed root. Each round is a stream of its own on the verifier's OS coins.
* **The trace** is a stand-in for the compute audit's registration (`COMPUTE`): this driver commits the scan's
  `SCAN_STATE` and each lane's slot entries itself, as `G.Registered` rows of the trace's chain under fresh salts. What
  it shows is the tie to a trace so registered, not that the compute audit registered it. The chain's outputs are the
  ids the network certifier's egress carries, so the trace renders that egress whatever it is; with ``--outputs`` (a
  JSON object of each connection's committed output ids by its stream ordinal, as `warden_chain.io`'s records number
  them) they are the compute's, and where those don't render the network certifier's rows (scan_zk `link`'s
  ``unrendered``: a relay that rewrote or added an id) the honest round is refused at the egress statement too
  (`expected`).
* **The prover's trees**: ``scan/view`` and ``scan/last`` (S_{T−1}), each under fresh salts.
* **What the verifier receives** (`verifier_dir`) is scanned as scan_zk scans its rounds (`scan_zk.received`) for
  every proof block's input and output rows but the public ``o0`` and every salt but ``o0``'s, less what is public
  anyway (`scan_zk.known`), the prover's instance files the control: a round is as expected only when none of it
  leaked there (`scan_zk.unseen`; a find whose every placement straddles fields, ``across``, isn't a leak).
* ``--dry`` stages every statement and checks it on the host (each block's outputs the references', every o0 1),
  registers and samples, and proves nothing.
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
sys.path.insert(0, str(HERE))

import scan_zk as SZ  # noqa: E402
import warden_zk as WZ  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity.core.protocols.certifier import ingress_scan as IS, scan_service as SC, warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsSigner  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF, registered as G  # noqa: E402

#: the stand-in compute audit's Program, whose registration names the trace's trees (this driver commits them)
COMPUTE = hashlib.sha512(b"scan_tie.py: the stand-in compute audit's program").hexdigest()
CHALLENGER = "scan_tie.py on the verifier's OS coins"
#: each round's expected verdict: accepted, or refused naming exactly these statements
ROUNDS = {"honest": (True, ()), "tampered": (False, ("egress",))}


def expected(at: SimpleNamespace, name: str) -> tuple[bool, tuple[str, ...]]:
    """Round `name`'s expected verdict: `ROUNDS`', but the honest round is refused at the egress statement where the
    committed outputs don't render the network certifier's rows (``at.unrendered``)."""
    return ROUNDS["tampered"] if name == "honest" and at.unrendered else ROUNDS[name]


def log(msg: str) -> None:
    print(f"SCTIE {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def one_link(at: SimpleNamespace, out: Path) -> SimpleNamespace:
    """`at` with both directions on one link, as the network certifier names its link's: a synthetic run's sides
    (scan_traffic's records name the ingress link ``requests`` and the egress ``responses``) recommitted under the
    egress side's link with fresh salts, their heads signed by a test-size key of the driver's own and read back
    (`warden_rows.check_registration`), and the tampered window recomputed on them (scan_zk's `_tamper`). A run whose
    sides are one link's is `at` itself."""
    link = at.sides["egress"][0].head.link
    if at.sides["ingress"][0].head.link == link:
        return at
    out.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"scan-tie").digest()),
                              out / "tie-signer-state.json")
    issuer, sides = Issuer(signer, at.session), {}
    for side, d in (("ingress", Direction.INGRESS), ("egress", Direction.EGRESS)):
        c = at.sides[side][0].cells
        cells = WR.Cells.commit(link, d, c.window, c.rows, c.wiring, c.params, c.session)
        cert = issuer.sign(link, d, c.window, cells.head.digest())
        if cert is None:
            raise SystemExit("the driver's signer refused an execution phase")
        registration = WR.registration(cells.head, cert, c.params)
        head, dom = WR.check_registration(registration, signer.public_key(), c.session, c.params, link, d, c.window)
        sides[side] = [SimpleNamespace(cells=cells, registration=registration, head=head, dom=dom)]
    got = SimpleNamespace(sides=sides, issuer=issuer, key=signer.public_key(), key_of="the driver's own")
    tampered = SZ._tamper(got, at.chain.egress, at.T, at.setup.cls.B, at.setup.cls.egress.R)
    return SimpleNamespace(**{**vars(at), "sides": sides, "tampered": tampered,
                              "source": f"{at.source}, both directions recommitted as link {link}'s"})


def trace(tie: SC.EgressTie, ch) -> SimpleNamespace:
    """The stand-in compute audit's trees of the chain `ch`: its `SCAN_STATE` (T leaves, leaf t S_{t−1}) and lane j's
    slot entries (nb leaves), and the `Trace` the egress tie reads off their registration."""
    cls, T = tie.cls, tie.T
    state = G.Registered.commit(COMPUTE, IS.SCAN_STATE, [SC.words(s) for s in ch.states[:T]], None, G.WORD_BITS,
                                SC.SCHEMA)
    progs = [G.Registered.commit(COMPUTE, SC.value(f"prog{j}"),
                                 [SC.words(IS.prog_rows(cls, ch.progs[b * cls.EVERY])[j]) for b in range(tie.nb)],
                                 None, G.WORD_BITS, SC.SCHEMA) for j in tie.lanes]
    return SimpleNamespace(state=state, progs=progs,
                           trace=SC.Trace(SC.Tree.of(state), tuple(SC.Tree.of(p) for p in progs)))


def prover_trees(tie: SC.EgressTie, heads: SC.Heads, ch) -> tuple[G.Registered, G.Registered]:
    """The prover's ``scan/view`` and ``scan/last`` (S_{T−1}, the chain's state after bucket T − 1), under fresh
    salts, registered before any coin."""
    T = tie.T
    return tie.view_tree(heads, ch.views[:T]), tie.last_tree(heads, ch.states[T])


def stagers(M, tie: SC.EgressTie, out: Path) -> dict[str, SimpleNamespace]:
    """Each template's stager, bound to the egress tie's partition and Program, every output its own row; each
    template's range in the partition held to its members' proof blocks."""
    part = tie.partition
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(tie.program)
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    ranges = M.P.ranges(tie._program, tie.query)
    got = {}
    for tp, (tid, base, cnt) in zip(SC.TEMPLATES, ranges, strict=True):
        units = sorted(u for m in tie.members if m.template == tp for u in m.units)
        if (tid, base, cnt) != (M.P.template_id(tie.templates[tp]), units[0], len(units)) or \
                units != list(range(base, base + cnt)):
            raise SystemExit(f"the partition's range of {tp} is ({base}, {cnt}), not its members' proof blocks")
        ins, outs = SZ.ports_of(tie.cls)[tp]
        sdir = out / tp
        sdir.mkdir(exist_ok=True)
        t = time.time()
        st = M.CS.UnitStager(tie.templates[tp], f"unit/scan-tie-{tp}", [(r, M.V2, b, [r]) for r, b in ins],
                             [(M.V2, (j,)) for j in range(1 + len(outs))], circuit=sdir / "circuit.txt", bind=bound)
        _ = st.pin
        got[tp] = SimpleNamespace(st=st, ins=ins, outs=outs, out=sdir, lower_s=round(time.time() - t, 1),
                                  partition_file=out / "partition.json", program_file=out / "program.json")
    return got


def stage(M, tie: SC.EgressTie, sg: SimpleNamespace, member: SC.Member, heads: SC.Heads, cells: SimpleNamespace,
          tr: SimpleNamespace, view: G.Registered, last: G.Registered, out: Path) -> dict:
    """Member `member`'s statement: each input port's rows the registered rows its read opens (a slot-entries port one
    table row a boundary, `registered.table`), every row, input or output, under the salts and paths of the registered
    row the verifier's read of it opens (`scan_service`'s statement), and the verifier's own `--registered` file
    (`EgressTie.registered`). Its outputs are the circuit's on those rows: each output row must be the tree's row it is
    read as, but the egress ``o1`` of a round whose network certifier's rows aren't the trace's."""
    t0 = time.time()
    st, bs, n, every = sg.st, list(member.buckets), len(member.buckets), tie.cls.EVERY
    known = tie.known_tree(heads)
    words, tables, refs, salts, entries, want = {}, {}, {}, {}, {}, {}

    def read(p: str, reg, positions: list[int]) -> None:
        table, refs[p] = G.table(positions)
        rows, salts[p], entries[p] = reg.statement(table)
        tables[p] = M.np.asarray(rows, dtype=M.np.uint16)
        words[p] = tables[p][refs[p]]

    def made(p: str, reg, positions: list[int], rows=None) -> None:
        got, salts[p], entries[p] = reg.statement(positions)
        want[p] = got if rows is None else rows

    if member.template == "egress":
        read("state", view, bs)
        made("o1", cells.egress, bs)
    else:
        read("row", cells.ingress, bs)
        if member.template == "ingress":
            read("sin", tr.state, bs)
        for j, p in enumerate(tr.progs):
            read(f"prog{j}", p, [b // every for b in bs])
        if member.name == "last":
            made("o1", last, [0])
        else:
            made("o1", tr.state, [b + 1 for b in bs])
        made("o2", view, bs)
    made("o0", known, list(member.units))
    if sorted(words) != sorted(r for r, _b in sg.ins):
        raise SystemExit(f"{member.name}: the staged inputs {sorted(words)} aren't the template's")
    out.mkdir(parents=True, exist_ok=True)
    got = st.stage(words, out, os.urandom(32), salts=salts, indices=range(member.base, member.base + n),
                   set_name=f"unit-scan-tie-{member.name}", tables=tables, refs=refs, registered=entries)
    outs = {p: [[int(w) for w in row] for row in M.np.asarray(got["outs"][p])] for p in want}
    off = {p: [k for k, (a, b) in enumerate(zip(outs[p], want[p])) if list(a) != list(b)] for p in want}
    rows = [*tables.items(), *((p, col) for p, col in outs.items() if p != "o0")]
    blobs = {f"{member.name} {p} {i}": M.np.asarray(r, dtype="<u2").tobytes() for p, col in rows
             for i, r in enumerate(col)}
    blobs.update({f"{member.name} salt {p} {i}": s for p, ss in salts.items() if p != "o0" for i, s in enumerate(ss)})
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    reads = tie.registered(member, heads, tr.trace, view.root, last.root)
    (out / "registered.json").write_text(json.dumps(reads, sort_keys=True))
    return {"name": member.name, "template": member.template, "dir": out, "n": n, "base": member.base, "pin": st.pin,
            "ports": st.ports, "blobs": blobs, "partition_file": sg.partition_file, "program_file": sg.program_file,
            "registered_file": out / "registered.json", "ands": got["ands"], "k_log": st.meta["k_log"],
            "off": {p: [member.base + k for k in ks] for p, ks in off.items() if ks},
            "stage_s": round(time.time() - t0, 1)}


def host_read(name: str, staged: dict[str, dict], refused: bool) -> list[str]:
    """Where a round's staged statements aren't what its expected verdict needs: every output the tree's row it is read
    as, but the egress o1 of a round expected refused, which must differ at one proof block or more."""
    bad = []
    for k, m in staged.items():
        off = dict(m["off"])
        if refused and m["template"] == "egress":
            if not off.pop("o1", None):
                bad.append(f"{k}: the {name} round's egress o1 is the signed rows'")
        if off:
            bad.append(f"{k}: outputs {off} aren't the rows their reads open")
    return bad


def tie_of(out: Path, setup, warden_run: Path | None = None, proof_deadline: float = 6 * 3600.0,
           outputs: dict[int, list[int]] | None = None) -> SimpleNamespace:
    """The execution phase (`one_link` of scan_zk's `link`, its chain of `outputs` when given), its egress tie and the
    stand-in trace."""
    at = one_link(SZ.link(out / "warden", warden_run, setup, 1, outputs=outputs), out / "warden")
    if at.void or at.tampered is None:
        raise SystemExit(f"the run's execution phase is void (FAIL buckets {at.void}): the egress tie proves a served "
                         "one")
    tie = SC.EgressTie(at.sides["ingress"][0].head.link, at.params, at.session, setup.cls,
                       proof_deadline=proof_deadline)
    return SimpleNamespace(at=at, tie=tie, tr=trace(tie, at.chain))


def stage_round(M, tie: SC.EgressTie, at: SimpleNamespace, tr: SimpleNamespace, sgs: dict, name: str, out: Path,
                delta: float) -> SimpleNamespace:
    """Round `name` on a stream of its own, registered (the network certifier's heads, the trace's roots and the
    prover's ``scan/view`` and ``scan/last``), every proof block sampled and each statement staged and read on the
    host."""
    egress = at.tampered if name == "tampered" else at.sides["egress"][0]
    cells = SimpleNamespace(ingress=at.sides["ingress"][0].cells, egress=egress.cells)
    heads = SC.Heads(at.sides["ingress"][0].head, egress.head)
    view, last = prover_trees(tie, heads, at.chain)
    stream = tie.stream(budget=1, delta=delta, challenger=CHALLENGER)
    record = tie.registration(heads, tr.trace, view.root, last.root)
    reg = tie.register(stream, record, heads, tr.trace)
    draw = stream.select(reg)
    if draw.units != list(range(tie.N)):
        raise SystemExit(f"{name}: the stream sampled {len(draw.units)} of {tie.N} proof blocks, not every one")
    staged = {m.name: stage(M, tie, sgs[m.template], m, heads, cells, tr, view, last, out / f"stage-{m.name}")
              for m in tie.members}
    res = {"registration": reg.digest, "sampled": len(draw.units),
           "staged": {k: {x: m[x] for x in ("n", "base", "ands", "k_log", "off", "stage_s")}
                      for k, m in staged.items()}}
    bad = host_read(name, staged, not expected(at, name)[0])
    if bad:
        raise SystemExit(f"{name}: the host's read of the staged statements isn't the round's: {bad}")
    log(f"{name}: registered {reg.digest[:16]}, every one of {tie.N} proof blocks sampled; staged "
        + "; ".join(f"{k} {m['n']} proof blocks of {m['ands']} ANDs" for k, m in staged.items()))
    return SimpleNamespace(stream=stream, draw=draw, heads=heads, staged=staged, res=res, record=record,
                           registrations={"ingress": at.sides["ingress"][0].registration,
                                          "egress": egress.registration},
                           public=SZ.known(list(staged.values()), at))


def verifier_dir(sr: SimpleNamespace, sgs: dict, out: Path) -> Path:
    """What the verifier received in round `sr`: the registration, the network certifier's signed registrations of
    the execution phase, the partition and Program, and each statement's circuit, public file, registered reads and
    session files."""
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    (vdir / "registration.json").write_text(json.dumps(sr.record, indent=1, sort_keys=True))
    for side, doc in sr.registrations.items():
        (vdir / f"warden-registration-{side}.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    sg = next(iter(sgs.values()))
    for f in (sg.partition_file, sg.program_file):
        os.link(f, vdir / f.name)
    for m in sr.staged.values():
        for f in (m["dir"] / "circuit.txt", m["dir"] / f"pub-{m['n']}.bin", m["registered_file"]):
            os.link(f, vdir / f"{m['name']}-{f.name}")
        for f in sorted((out / f"session-{m['name']}" / "sessions").rglob("*")):
            if f.is_file():
                os.link(f, vdir / f"{m['name']}-session-{f.parent.name}-{f.name}")
    return vdir


def prove_round(M, a, tie: SC.EgressTie, at: SimpleNamespace, tr: SimpleNamespace, sgs: dict, name: str,
                out: Path, port: int) -> dict:
    """Round `name` staged (`stage_round`) and, unless ``--dry``, each statement proved by one session behind the ZK
    server, delivered, and decided by the verifier (`EgressTie.outcome`)."""
    sr = stage_round(M, tie, at, tr, sgs, name, out, a.delta)
    stream, draw, heads, staged, res = sr.stream, sr.draw, sr.heads, sr.staged, sr.res
    if a.dry:
        return res
    av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
    proofs, timing = {}, {}
    for j, m in enumerate(tie.members):
        md, sdir = staged[m.name], out / f"session-{m.name}"
        _record, live, proc = M.a8.session(av, md, sdir, list(range(md["n"])), port + j if port else SZ._free_port(),
                                           env=WZ.firewall(M, md))
        proofs[m.name] = PF.Proof(md["dir"] / "circuit.txt", md["dir"] / f"pub-{md['n']}.bin",
                                  tuple(sorted(p.parent for p in (sdir / "sessions").glob("*/session.json"))))
        timing[m.name] = {"session": proc, **{x: (live or {}).get(x) for x in SZ.TIMING}}
        log(f"{name}/{m.name}: prover {'accepted' if (live or {}).get('accepted') else 'refused'} "
            f"{(live or {}).get('why') or ''}, prove {timing[m.name].get('prove_total_s')} s, session {proc['wall_s']} s")
    delivered = SC.Delivered(proofs)
    late = stream.deliver(draw, delivered.sessions(tie.members))
    circuits = {tp: PF.file_sha512(sg.out / "circuit.txt") for tp, sg in sgs.items()}
    t = time.time()
    outcome = tie.outcome(stream, draw, delivered, circuits=circuits, heads=heads, trace=tr.trace)
    verdict = stream.records[-1]["audit"]["verdicts"][SC.VERIFIER]
    tied = tie.tie(stream)
    verify_s = round(time.time() - t, 1)
    vdir = verifier_dir(sr, sgs, out)
    inst = out / "instances.bin"
    inst.write_bytes(b"".join((m["dir"] / f"inst-{m['n']}.bin").read_bytes() for m in staged.values()))
    blobs = {k: x for m in staged.values() for k, x in m["blobs"].items()}
    rec = SZ.received(M.a8, vdir, inst, blobs, set(blobs), sr.public)
    inst.unlink()
    want_ok, want_refused = expected(at, name)
    res.update({"late": None if late is None else str(late), "verify_s": verify_s,
                "outcome": {"accepted": outcome.accepted, "public": outcome.public},
                "verdict": {x: verdict.get(x) for x in ("accepted", "decided_by", "refused", "digest", "public")},
                "tie": {"link": tied.link, "accepted": list(tied.accepted)}, "timing": timing, "received": rec,
                "expected": outcome.accepted is want_ok and tied.accepted == (want_ok,)
                and tuple(sorted(verdict.get("refused") or {})) == want_refused and SZ.unseen(rec)})
    log(f"{name}: verdict {'accepted' if outcome.accepted else 'refused'} {verdict.get('refused') or ''}, "
        f"tie {tied.accepted}; the verifier received {rec['verifier_bytes']} bytes, {SZ.finds(rec)}, control holds "
        f"all {rec['control']['holds_all']}; {'as expected' if res['expected'] else 'UNEXPECTED'}")
    return res


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup = SZ.SETUPS[a.setup]
    outputs = None
    if a.outputs is not None:
        try:
            outputs = {int(q): [int(x) for x in xs] for q, xs in json.loads(a.outputs.read_text()).items()}
        except (OSError, ValueError, TypeError, AttributeError) as e:
            raise SystemExit(f"--outputs {a.outputs}: not a JSON object of id lists by stream ordinal: {e}")
    t = tie_of(out, setup, a.warden_run, a.proof_deadline, outputs)
    at, tie, tr = t.at, t.tie, t.tr
    summary = {"source": at.source, "setup": setup.name, "link": tie.link, "T": tie.T, "N": tie.N,
               "members": [{"name": m.name, "template": m.template, "base": m.base, "n": len(m.units)}
                           for m in tie.members],
               "spec": tie.spec.digest(), "program": tie.partition["program"],
               "trace": {"of": "the stand-in compute audit's registration (scan_tie.py commits it, COMPUTE)",
                         "outputs": "the network certifier's egress ids" if outputs is None else str(a.outputs),
                         "unrendered": at.unrendered,
                         "state": tr.state.root.hex(), "progs": [p.root.hex() for p in tr.progs]},
               "tampered": {"bucket": at.tampered.bucket, "key": at.tampered.key_of}, "dry": bool(a.dry), "rounds": {}}
    M = WZ._one_stage()
    sgs = stagers(M, tie, out / "stagers")
    for m in tie.members:
        meta = sgs[m.template].st.meta
        nbl = max(3, (-(-len(m.units) // meta["vus_per_block"]) - 1).bit_length())
        if not 25 <= meta["k_log"] + nbl <= M.a7.M_MAX:
            raise SystemExit(f"{m.name}: m = {meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX}, the --zk prover's")
    summary["stagers"] = {tp: {"lower_s": g.lower_s, "k_log": g.st.meta["k_log"], "pin": g.st.pin} for tp, g in sgs.items()}
    log(f"{at.source}: link {tie.link}, {setup.name}'s class, T {tie.T}: {tie.N} proof blocks in "
        f"{[(m.name, len(m.units)) for m in tie.members]}; stagers {summary['stagers']}")
    ok = True
    for j, name in enumerate(ROUNDS):
        res = prove_round(M, a, tie, at, tr, sgs, name, out / name, a.port and a.port + 4 * j)
        summary["rounds"][name] = res
        ok &= bool(a.dry) or res["expected"]
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan_tie.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="an execution phase's egress tie, every proof block proved --zk and decided "
                                     "by EgressTie.verify")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=0, help="round i's statement j listens on PORT + 4i + j; 0 (the "
                                                       "default): a free port for each session")
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--proof-deadline", type=float, default=6 * 3600.0,
                   help="the Spec's proof deadline in seconds from the sample (default 6 hours)")
    p.add_argument("--dry", action="store_true", help="stage, register and sample every round: nothing proved")
    p.add_argument("--class", dest="setup", choices=sorted(SZ.SETUPS), default=SZ.W63K4.name)
    p.add_argument("--warden-run", type=Path, help="a warden_serve run's directory (as scan_zk.py's): its execution "
                                                   "phase 0 in place of the synthetic run's")
    p.add_argument("--outputs", type=Path, help="the compute's committed output ids by stream ordinal (JSON), the "
                                                "trace's in place of the network certifier's egress ids")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
