"""N2's verdict of record from zero-knowledge proofs of N2's registered unit, on a certifier-serve run: each link-window
committed, registered on the link's own stream (`service.LinkWindows.stream`, the proof system's `Stream`), selected
(Selection All: its one unit, no coins), its unit (`NetworkLinkWindow_v5`, both ends' head checks on their pinned gate
ids) proved by C-Flock `prove --zk` behind the Lean firewall, delivered, and decided by `LinkWindows.outcome`: the
verifier's own Lean `verify --zk` of the proof, the proof's public row opened to the verifier's own copy and its output
row read. No host check reads the opening; `--debug-clear` adds the in-the-clear comparison to each verdict, which
decides nothing.

    python benchmarks/network_traces/n2_zk.py virtual --out DIR [--requests 40] [--tokens 4]
    python benchmarks/network_traces/n2_zk.py prove --run DIR [--extra DIR] [--late DIR] --bin flock-circuit
      --lean flock-verify --out DIR [--jobs 1] [--gpu] [--debug-clear] [--dry]
    python benchmarks/network_traces/n2_zk.py prove --warden-run DIR --bin flock-circuit --lean flock-verify --out DIR
      [--draw K]

* **The runs.** A certifier-serve run's directory (`certifier_serve.py rehearse`'s per strategy, or `virtual`'s): its
  `plan.json`, `calibration.json`, `deployment.json` and `certified.jsonl`. `virtual` makes them on an injected clock
  against a stub endpoint that answers at once (certifier-serve's tests' run), honest, extra and late, with fresh keys.
  Or, with ``--warden-run``, an active warden's run (`warden_serve.py`'s, or the gate's ``warden-rehearse``'s), read by
  `warden_chain`: its own certified records under its key and session, on the warden link (`service.WardenProfile`:
  both heads at the warden's grid and parameters digest, no sync declared or charged), a certifier rehearsal's run
  refused by name; its members are honest and tampered, the tampered head failing its signed digest alone.
* **Members.** One stream each, over all of the run's link-windows: ``honest``; ``tampered``, the honest records with
  window 0's egress head one count off (`one_count`), committed by the prover under its own root, the certificates still
  the certifier's; with ``--extra``, certifier-serve's ``extra`` run (the shim wrote one event more and the certifier
  certified it); with ``--late``, its ``late`` run. The prover's references (`head_check.reference_parts`, and the unit's
  output word by the gates' references, `LinkWindows.outputs`) say beforehand which windows each member's proofs refuse
  (`expected`).
* **The period** (``--draw K``): each member's windows sampled rather than all proved: every window's salted root
  committed, then K of them drawn on the verifier's own coins (`LinkWindows.period_draw`), and only those
  registered, proved and decided; the network verdict is the sampled one, its ``wrong_units`` at the
  stream's δ, on the Lean verifier of record's draw. ``tampered`` then changes every window (`every_count`), which any
  draw refuses, and with K < n ``other_subset`` proves the honest records on another K-subset (`other_subset`), every
  window accepted, which the network verdict refuses by name (``period-draw``).
* **The statement** (`stager`, `stage`). One circuit, `LinkWindows.proof_unit` lowered once (the registered unit with each
  gate the Definition its pinned digest names), bound to the registered partition and Program. Each link-window is a
  statement of its own with one instance, unit 0: rows ``pub`` and ``wit`` (`LinkWindows.layout`, `LinkWindows.row`) and
  the output row ``o0``, the verdict and deadline bits. The lowering evaluates every instance at once, held to the prover's
  reference and to the IR reference; each statement's salts come from a key of its own, and the verifier's opening of its
  ``pub`` and ``o0`` rows is checked on the staged public file before anything is proved.
* **A round** (`member`), per link-window in order: register, select, one live `prove --zk` session behind the firewall on
  the statement (the server serving the stream's draw), deliver, and the outcome of record. Per window: prove (the
  session's wall time, the prover's ``prove_total_s``, proof bytes) and verify (the verifier of record's run, Lean
  included). Then the link's network verdict (`LinkWindows.network_verdict`). The members' rounds run one after
  another: outer proofs go through the firewall one at a time (`ONE_SESSION`), so ``--jobs`` is 1 and more is refused.
* **What the verifier receives**, per link-window, kept under ``<member>/verifier/`` (`RECEIVED`): the registration record
  (the salted root, nothing of the certificates) and the draw; the two certificates' public fields, from the certifier,
  the auditor's device; the declared clock syncs; from the prover, the statement's public file (its header and each row's
  hiding commitment), its session files, and the salts of the ``pub`` and ``o0`` rows with the output word; and the
  outcome record. Its own: the circuit file's SHA-512 (its lowering, which prover and verifier share here), the Program's
  descriptor and the partition object. Never the opening (records, heads, leaf salts, siblings, the certifier's salt)
  nor the ``wit`` row's salt. Every file there is scanned for every piece of every head, leaf salt, sibling and
  certifier salt of the member (`a8.received`) but the pieces the public circuit or a window's ``pub`` row holds
  (`public_pieces`), which the verifier has whatever the witness, and a member fails on a ``leaked`` piece under the
  rule every zero-knowledge driver's scan follows (`leak_scan.scan`); the prover's instance files, which hold the heads,
  are the control.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import gc
import hashlib
import json
import os
import secrets
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ONE_STAGE = HERE.parent / "one_stage"
sys.path.insert(0, str(HERE))

import certifier_serve as cs  # noqa: E402
import leak_scan as LS  # noqa: E402
from verity.core.protocols.certifier import head_check as HC  # noqa: E402
from verity.core.protocols.certifier import service as SVC  # noqa: E402
from verity.core.protocols.certifier.certificate import CertifiedRecord, Direction, Issuer  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402
from verity.core.protocols.sampled_proofs import service as S  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402

ENDS = ("egress", "ingress")
#: the sampled period's negative (``--draw K``, K < n): the prover proves another K-subset than the draw of record's
OTHER = "other_subset"
ROUNDS = ("honest", "tampered", "extra", "late")
#: AGENTS.md's design rule, which `one_session` holds every driver's ``--jobs`` to
ONE_SESSION = "outer proofs go through the firewall one at a time (AGENTS.md, 'Live coins only')"
#: what the verifier receives per link-window, by file (`member`)
RECEIVED = {"registration.json": "the registration record", "draw.json": "the stream's draw",
            "certificates.json": "the certificates' public fields (the certifier's)", "syncs.json": "the declared syncs",
            "pub-1.bin": "the statement's public file", "session-*": "the session's files",
            "delivered.json": "the pub and o0 rows' salts and the output word", "outcome.json": "the outcome record"}


def log(msg: str) -> None:
    print(f"N2ZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


#: the verifier of record's own Lean runs (`proofs.verify`'s findings and refusal), by the proof's public file, so each
#: window records what decided it: a rejected outcome's public outputs are the service's, which say nothing
LEAN_RUNS: dict[str, tuple] = {}


def _record_lean_runs(PF) -> None:
    verify = PF.verify

    def recorded(proof, **kw):
        out = verify(proof, **kw)
        LEAN_RUNS[str(proof.public)] = out
        return out

    if not hasattr(verify, "recorded"):
        recorded.recorded = True
        PF.verify = recorded


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


def warden_link(run: Path) -> tuple[LinkWindows, dict, dict, list]:
    """A warden run's link (`warden_chain.link`: the active warden's own records, its key and session, no sync), the
    plan it stands for (its windows, no syncs), its certified records by (direction, window) and the signer's
    refusals. A certifier rehearsal's run is refused by name (`warden_chain.PROXY`)."""
    import warden_chain as WC

    lw, dep, _ = WC.link(run)
    got, refused = WC.pairs(run, dep)
    certified = {(d, w): r for w, pair in enumerate(got) for d, r in zip((Direction.EGRESS, Direction.INGRESS), pair)}
    return lw, {"windows": dep["windows"], "syncs": [], "warden_run": dep["run_id"]}, certified, refused


def link_windows(run: Path, edit=None, clock=None, warden: bool = False) -> SimpleNamespace:
    """Every link-window of the run with both certified records, as the prover holds it before it registers: its opening
    (fresh salts), the root it commits (`LinkWindows.root`), the unit's arguments under that root (None where the unit
    takes no instance), each end's head check's failed parts (`head_check.reference_parts`) and the unit's output word by
    the gates' references (`LinkWindows.outputs`); and the certifier's certificates, which the verifier receives from
    the certifier. ``edit(lw, w, pair)``: the records the prover commits in place of the certified ones. ``expect``: the
    verdict the verifier of record should reach on an honest proof of the opening. With ``warden``, ``run`` is an active
    warden's run (`warden_link`), its records and certificates the warden's own and no sync declared.

    With ``clock``, for a report on runs nothing is proved for (network_e2e), also on N2's stream as certifier-serve's
    check takes it: each link-window registered (``digest``), selected (Selection All) and decided before the next by
    `LinkWindows.unproved_outcome`, in the clear (``host``), and the link's network verdict (``network``)."""
    if warden:
        lw, plan, certified, refused = warden_link(run)
    else:
        lw, plan = link(run)
        certified, refused = cs.read_certified(run / "certified.jsonl")
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    out = []
    for w in range(plan["windows"]):
        pair = certified.get((Direction.EGRESS, w)), certified.get((Direction.INGRESS, w))
        if None in pair:
            break
        certificates = tuple(r.certificate for r in pair)
        if edit is not None:
            pair = edit(lw, w, pair)
        opening = lw.commit(w, *pair)
        root = bytes.fromhex(lw.root(opening))
        insts = lw.head_checks(opening, root)
        args = lw.arguments(opening, root, syncs)
        word = lw.outputs(opening, root, syncs) if args is not None else None
        others = lw.certified(w, certificates) and (lw.warden is not None
                                                    or (w < len(syncs) and syncs[w] in lw.sync_set))
        out.append(SimpleNamespace(
            window=w, opening=opening, root=root, insts=insts, args=args, word=word, certificates=certificates,
            failed={e: None if i is None else [k for k, ok in HC.reference_parts(i).items() if not ok]
                    for e, i in zip(ENDS, insts)},
            expect=int(word == 0b11 and others), statement=None))
    rd = SimpleNamespace(lw=lw, plan=plan, syncs=syncs, windows=out, issuer_refused=refused)
    if clock is not None:
        st = lw.stream(budget=plan["windows"], delta=cs.DELTA, challenger="n2-zk", coins=cs._no_coins, clock=clock)
        previous = None
        for x in out:
            reg = lw.register(st, lw.registration(x.opening, previous))
            previous = x.digest = reg.digest
            draw = st.select(reg)
            outcome = st.deliver(draw, [lw.session(x.opening, draw)]) or lw.unproved_outcome(st, draw, x.opening, syncs)
            x.host = {"accepted": outcome.accepted, "codes": list(outcome.verdict.codes), "public": dict(outcome.public)}
        rd.network = lw.network_verdict(st, plan["windows"])
    return rd


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


def every_count(lw: LinkWindows, w: int, pair: tuple[CertifiedRecord, CertifiedRecord]):
    """Every window's egress head with one count more (`one_count`'s change, in each window): the tampered member of a
    sampled period (``--draw``), which refuses it whichever windows it draws."""
    return one_count(lw, 0, pair)


def expected(name: str, rd: SimpleNamespace) -> bool:
    """Whether the member's references are its round's: every unit takes an instance, and its verdict bit is 1 exactly
    where no part fails; honest and other_subset, none fails and every window accepts; tampered, window 0's egress head
    (every window's, ``rd.every``) fails its counts and its signed digest only (its signed digest alone on the advice
    profile, whose egress counts no profile holds, and on a warden link, whose counts are its schedule's);
    extra, egress heads fail their counts alone, on one window or more; late, egress heads
    fail their fields and counts, on one window or more. The ingress heads fail nothing, and every negative member
    refuses one window or more."""
    xs = rd.windows
    if not xs or any(x.word is None for x in xs) or any(bool(x.word & 1) != (not x.failed["egress"] and not
                                                                            x.failed["ingress"]) for x in xs):
        return False
    if any(x.failed["ingress"] for x in xs):
        return False
    eg = [x.failed["egress"] for x in xs]
    if name in ("honest", OTHER):
        return all(x.expect for x in xs)
    if name == "tampered":
        signed = (rd.plan.get("occupancy") or {}).get("kind") == cs.ADVICE or rd.lw.warden is not None
        want = ["signed"] if signed else ["counts", "signed"]
        return eg[0] == want and all(f == (want if getattr(rd, "every", False) else []) for f in eg[1:])
    allowed = {"extra": ([], ["counts"]), "late": ([], ["fields", "counts"])}[name]
    return all(f in allowed for f in eg) and allowed[1] in eg


def blobs_of(rd: SimpleNamespace) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: each head, its leaf's salt, its siblings, the certifier's salt,
    and the records' leaf salts; the heads are what the prover's instance files hold as they are."""
    blobs, held = {}, set()
    for x in rd.windows:
        for d, (end, inst) in enumerate(zip(ENDS, x.insts)):
            if inst is None:
                continue
            blobs[f"head {end} {x.window}"] = inst.head
            held.add(f"head {end} {x.window}")
            blobs[f"head salt {end} {x.window}"] = inst.salt
            blobs[f"certifier salt {end} {x.window}"] = inst.salt_c
            blobs[f"record salt {end} {x.window}"] = x.opening.salts[d]
            for k, s in enumerate(inst.path):
                blobs[f"sibling {k} {end} {x.window}"] = s
    return blobs, held


#: The scan's one cutter (`leak_scan.public_pieces`: each row cut whole, less the pieces the staged circuit holds, such
#: as a head's empty buckets, sixteen zero bytes), under the name the drivers that import this module call. The held
#: blobs are fields of the ``wit`` row (``fields``): a head of 157 + L + 4T bytes ends off a 16-byte boundary,
#: and the instance holds its salt after it, never its zero-padded last chunk.
public_pieces = functools.partial(LS.public_pieces, fields=True)


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


def virtual(out: Path, requests: int = 40, tokens: int = 4, strategies=("honest", "extra", "late"),
            height: int = 6) -> dict:
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
        shutil.rmtree(keys)


# --- the statement ----------------------------------------------------------------------------------------------------

def _one_stage():
    sys.path.insert(0, str(ONE_STAGE))
    import a0
    import a8
    import numpy as np
    from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS

    return SimpleNamespace(a0=a0, a8=a8, np=np, P=P, PF=PF, V2=V2, FC=FC, CS=CS)


def stager(M, lw: LinkWindows, out: Path) -> SimpleNamespace:
    """The one circuit: `LinkWindows.proof_unit` lowered once on rows ``pub`` and ``wit`` with output row ``o0``, bound
    to the registered partition and Program, which are written beside it as the verifier's own. Its ports are
    `service.PORTS`, each the length the verifier opens (`LinkWindows.bits`)."""
    rows = [(r, M.V2, lw.bits(r), list(lw.layout[r])) for r in SVC.ROWS]
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(lw.partition, sort_keys=True))
    (out / "program.json").write_bytes(lw.program)
    bound = {"partition": M.P.digest(lw.partition), "program": lw.partition["program"]}
    t, was = time.time(), gc.isenabled()
    gc.disable()
    try:
        st = M.CS.UnitStager(lw.proof_unit, "unit/n2-link-window", rows, [(M.V2, (0,))], circuit=out / "circuit.txt",
                             bind=bound)
        st.pin
    finally:
        if was:
            gc.enable()
    got = [(p.name, p.n_bits) for p in st.ports]
    if got != [(p, lw.bits(p)) for p in SVC.PORTS] or st.out_words != [1]:
        raise SystemExit(f"the statement's rows {got} are not the ones the verifier opens "
                         f"{[(p, lw.bits(p)) for p in SVC.PORTS]}")
    k_log = st.meta["k_log"]
    if not 25 <= k_log + 3 <= 35:
        raise SystemExit(f"m = {k_log} + 3 is outside 25 .. 35 (--zk and flock-circuit's shape)")
    return SimpleNamespace(st=st, lw=lw, out=out, circuit=out / "circuit.txt", partition_file=out / "partition.json",
                           program_file=out / "program.json", lower_s=round(time.time() - t, 1), k_log=k_log,
                           circuit_sha512=M.PF.file_sha512(out / "circuit.txt"))


def stage(M, sg: SimpleNamespace, members: dict[str, SimpleNamespace], secret: bytes, out: Path) -> dict:
    """Every link-window's statement on the one circuit, one instance each, unit 0. The lowering evaluates all of their
    rows at once (`Lowering.evaluate`), which must satisfy every unit and give the prover's output words, held to the IR
    reference in one batch (`UnitStager.reference_check`); then each statement's files are written as `UnitStager.stage`
    writes them (`circuit.write`), under salts from a key of the statement's own (`circuit.stage_salts` derives them from
    the key and the set's content, which every statement shares). The verifier's opening of each staged ``pub`` and
    ``o0`` row is checked on its public file. Sets each window's ``statement``."""
    np, st = M.np, sg.st
    items = [(name, x) for name, mb in members.items() for x in mb.windows]
    rows = {r: np.stack([np.frombuffer(members[name].lw.row(x.args, r), "<u2") for name, x in items])
            for r in SVC.ROWS}
    t = time.time()
    flat = np.concatenate([rows[r] for r in st.row_ports], axis=1)
    got, _cut, ok = st.low.evaluate(flat.astype(np.uint64))
    if int(ok) != (1 << len(items)) - 1:
        raise SystemExit("a unit is unsatisfied on its rows")
    o0 = got.astype(np.uint16)[:, :1]
    words = [int(w) for w in o0[:, 0]]
    if words != [x.word for _, x in items]:
        raise SystemExit(f"the lowering's output words {words} are not the prover's references {[x.word for _, x in items]}")
    eval_s, t = round(time.time() - t, 1), time.time()
    st.reference_check(rows, {"o0": o0})
    ref_s, t = round(time.time() - t, 1), time.time()
    for k, (name, x) in enumerate(items):
        lw = members[name].lw
        d = out / name / f"w{x.window:03d}" / "statement"
        iset = SimpleNamespace(**{**vars(st.iset), "name": f"{st.iset.name}-{name}-w{x.window}"})
        key = hashlib.sha512(secret + f"row salts/{name}/{x.window}".encode()).digest()[:32]
        salts = M.FC.stage_salts(iset, 0, 1, st.ports, key)
        d.mkdir(parents=True, exist_ok=True)
        M.FC.write(st.text, iset, 0, 1, d / "inst-1.bin", d / "pub-1.bin", key,
                   rows={r: rows[r][k:k + 1] for r in SVC.ROWS}, outs={"o0": o0[k:k + 1]}, salts=salts,
                   public_words={}, digests=(st.pin, st.class_sha512), indices=[0], meta=st.meta)
        (d / "tables").mkdir(exist_ok=True)
        os.link(sg.circuit, d / "circuit.txt")
        bcs = SVC._commitments(d / "pub-1.bin", 0)
        mine = lw.public_arguments(x.window, x.root, x.certificates, members[name].syncs)
        if not (bcs and mine is not None and lw.row(mine, "pub") == lw.row(x.args, "pub")
                and SVC._opens(lw.row(mine, "pub"), lw.bits("pub"), salts["pub"][0], bcs["pub"])
                and SVC._opens(SVC.output_row(words[k]), SVC.OUT_BITS, salts["o0"][0], bcs["o0"])):
            raise SystemExit(f"{name} window {x.window}: the staged pub or o0 row doesn't open to the verifier's own")
        x.statement = SimpleNamespace(dir=d, salts={"pub": salts["pub"][0], "o0": salts["o0"][0]}, output=words[k])
    return {"statements": len(items), "evaluate_s": eval_s, "reference_s": ref_s, "write_s": round(time.time() - t, 1)}


def firewall(M, sg: SimpleNamespace) -> dict:
    """A firewalled prover's environment (`FC_FIREWALL=1`, its Lean frame named by ``FC_FIREWALL_LEAN``) on the
    statement's partition and Program."""
    if not os.environ.get("FC_FIREWALL_LEAN"):
        raise SystemExit("FC_FIREWALL_LEAN names no Lean firewall frame (flock-verify's flock-firewall)")
    return {"FC_FIREWALL": "1", **M.a0.firewall_env({"partition_file": sg.partition_file,
                                                     "program_file": sg.program_file})}


def one_session(jobs: int) -> None:
    """A driver's ``--jobs``: 1, its members proved one after another, each session behind the firewall alone
    (`ONE_SESSION`); any other count is refused before anything runs."""
    if jobs != 1:
        raise SystemExit(f"--jobs {jobs}: {ONE_SESSION}, so a prover runs one firewalled session at a time: --jobs 1")


# --- the rounds -------------------------------------------------------------------------------------------------------

def _certificate(c) -> dict:
    return {k: (v.hex() if isinstance(v, bytes) else getattr(v, "code", v)) for k, v in asdict(c).items()}


def _free_port() -> int:
    """A loopback port the OS has free now. A pool's slots share a machine, so a fixed port collides across runs."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def other_subset(drawn: dict) -> list[int]:
    """A K-subset of the period's windows that isn't the draw's: the first K the draw didn't hold, then drawn ones
    from the start where it left fewer than K (none other at K = n)."""
    held = set(drawn["units"])
    return sorted(([w for w in range(drawn["population"]) if w not in held] + drawn["units"])[:drawn["k"]])


def member(M, a, sg: SimpleNamespace, name: str, rd: SimpleNamespace, out: Path, port: int,
           coins=None, forge=None) -> dict:
    """The member's link-windows on its own stream, in order: register, select, prove the statement live behind the
    firewall, deliver, and decide on the proof (`LinkWindows.outcome`); then its network verdict. Writes what the
    verifier receives under ``out / "verifier"`` (`RECEIVED`), each file prefixed by its window. Window j's session
    listens on ``port + j``, or with ``port`` 0 on a port the OS picks free.

    With ``a.draw`` K (> 0), the period is sampled (`LinkWindows.period_draw`): every window's salted root is
    committed first, then `coins` draw K of the windows, and only those are registered, proved and
    decided; the draw is ``period.json`` beside what the verifier receives, and the network verdict is the sampled one,
    with its ``wrong_units`` at the stream's δ (``a.delta``, by default `certifier_serve.DELTA`), on the draw of
    record: the Lean verifier of record's (``--lean``) draw on the coins the period draw read (`service.lean_draw`),
    which the windows proved must be. Coins with a ``drawn`` (the challenger's, `a5_coins.RankCoins`) end their turn
    on the session once the draw has read them, and the result's ``segment`` is what they released. ``forge``, a
    negative, is the prover's: of the period draw, the K windows it registers and proves instead, under a draw object
    listing them (`other_subset`), which is what ``period.json`` then holds."""
    lw = rd.lw
    k = getattr(a, "draw", 0) or None
    stream = lw.stream(budget=rd.plan["windows"], delta=getattr(a, "delta", None) or cs.DELTA, challenger="n2-zk",
                       coins=cs._no_coins, clock=time.monotonic)
    gpu = getattr(a, "gpu", False)
    # the firewall rechecks a witness the host computed, so a device prover takes the host's, as n2_tie_zk's does
    env = {**firewall(M, sg), **({"FC_HOST_WITNESS": "1"} if gpu else {})}
    av = SimpleNamespace(bin=a.bin, lease=[], gpu=gpu)
    vdir = out / "verifier"
    vdir.mkdir(parents=True, exist_ok=True)
    drawn, windows, segment = None, rd.windows, None
    if k is not None:
        try:
            drawn = lw.period_draw(stream, [x.root.hex() for x in rd.windows], k, coins)
        finally:
            if callable(getattr(coins, "drawn", None)):
                segment = coins.drawn()
        log(f"{name}: drew windows {drawn['units']} ({drawn['k']} of {drawn['population']}) on "
            f"{len(drawn['read']) // 2} coin bytes after the period's commitment {drawn['commitment'][:16]}")
        if forge is not None:
            drawn = {**drawn, "units": forge(drawn)}
            log(f"{name}: the prover proves windows {drawn['units']} instead")
        (vdir / "period.json").write_text(json.dumps(drawn, indent=1, sort_keys=True))
        windows = [rd.windows[u] for u in drawn["units"]]
    previous, rows = None, []
    for j, x in enumerate(windows):
        wdir, pre = out / f"w{x.window:03d}", f"w{x.window:03d}-"
        reg = lw.register(stream, lw.registration(x.opening, previous), drawn)
        previous = reg.digest
        if reg.record["roots"][0]["root"] != x.root.hex():
            raise SystemExit(f"{name} window {x.window}: the registered root is not the statement's")
        draw = stream.select(reg)
        if list(draw.units) != [0]:
            raise SystemExit(f"{name} window {x.window}: the stream drew {list(draw.units)}, not the one unit")
        sdir = wdir / "session"
        m = {"dir": x.statement.dir, "n": 1, "pin": sg.st.pin}
        record, live, proc = M.a8.session(av, m, sdir, [0], port + j if port else _free_port(), env=env)
        sessions = tuple(sorted(p.parent for p in (sdir / "sessions").glob("*/session.json")))
        delivered = SVC.Delivered(M.PF.Proof(sg.circuit, x.statement.dir / "pub-1.bin", sessions), x.statement.salts,
                                  x.statement.output)
        late = stream.deliver(draw, delivered.sessions())
        t = time.time()
        outcome = late or lw.outcome(stream, draw, x.certificates, rd.syncs, delivered,
                                     circuit_sha512=sg.circuit_sha512, opening=x.opening if a.debug_clear else None)
        verify_s = round(time.time() - t, 1)
        body = stream.records[-1]
        verdict = (body.get("audit") or {}).get("verdicts", {}).get(SVC.VERIFIER) or {}
        findings, refused = LEAN_RUNS.get(str(delivered.proof.public), (None, "no Lean run"))
        args = lw.public_arguments(x.window, x.root, x.certificates, rd.syncs)
        proof = {"lean_accepted": refused is None and bool(findings) and all(f.accepted for f in findings),
                 "refused": refused, "proved": refused is None and lw.proved(findings or [])
                 and all(list(f.units) == [0] for f in findings),
                 "opened_word": None if args is None else SVC._opened(delivered, 0, lw.row(args, "pub"), lw.bits("pub")),
                 "certified": lw.certified(x.window, x.certificates)}
        files = {"registration.json": reg.record, "draw.json": dict(draw.draw),
                 "certificates.json": [_certificate(c) for c in x.certificates],
                 "syncs.json": [[s.delta, str(s.rho)] for s in rd.syncs], "outcome.json": body,
                 "delivered.json": {"salts": {r: s.hex() for r, s in x.statement.salts.items()},
                                    "output": x.statement.output}}
        for f, v in files.items():
            (vdir / f"{pre}{f}").write_text(json.dumps(v, indent=1, sort_keys=True, default=str))
        os.link(x.statement.dir / "pub-1.bin", vdir / f"{pre}pub-1.bin")
        for f in sorted((sdir / "sessions").rglob("*")):
            if f.is_file():
                os.link(f, vdir / f"{pre}session-{f.parent.name}-{f.name}")
        proofs = live.get("proof_bytes") or []
        row = {"window": x.window, "registration": reg.digest, "accepted": outcome.accepted,
               "public": verdict.get("public"), "service_public": dict(outcome.public), "late": late is not None,
               "decided_by": verdict.get("decided_by"), "proof": proof,
               "expect": x.expect, "word": x.word, "failed": x.failed,
               "prove": {"session_s": proc["wall_s"], "prove_total_s": live.get("prove_total_s"),
                         "e2e_s": live.get("e2e_s"), "proof_bytes": sum(proofs) if proofs else None,
                         "k_log": live.get("k_log"), "prover_accepted": live.get("accepted"),
                         "max_child_rss_gb": proc["max_child_rss_gb"], "cpu_s": proc["cpu_s"]},
               "verify_s": verify_s, "server_verdict": ((record or {}).get("verdict") or {}).get("accepted"),
               **({"clear": verdict["clear"]} if "clear" in verdict else {})}
        row["as_expected"] = (row["accepted"] == bool(x.expect) and not row["late"] and row["decided_by"] == "proof"
                              and proof["proved"] and proof["opened_word"] == x.word and proof["certified"]
                              and row["public"] == {"verdict": x.expect, "deadline": x.word >> 1 & 1})
        rows.append(row)
        log(f"{name} window {x.window}: {'accept' if outcome.accepted else 'reject'} public {row['public']}"
            f"{' LATE' if late else ''}, Lean {'proved' if proof['proved'] else 'NOT proved'}, output word "
            f"{proof['opened_word']}, prove {row['prove']['session_s']} s (prove_total {row['prove']['prove_total_s']}"
            f" s, {row['prove']['proof_bytes']} B), verify {verify_s} s; "
            f"{'as expected' if row['as_expected'] else 'UNEXPECTED'}")
    network = lw.network_verdict(stream, rd.plan["windows"], drawn)
    if drawn is not None:
        sel = network["selection"]
        log(f"{name}: the draw of record ({sel['by']}) is windows {sel['units']}; proved "
            f"{[r['window'] for r in rows]}; " + (f"REFUSED by name: {network['why']}" if network["why"] else
                                                   "the windows proved are the draw of record's"))
    return {"windows": rows, "network": network, **({} if drawn is None else {"period": drawn}),
            **({} if segment is None else {"segment": segment})}


def _member_scan(M, sg, rd, out: Path, received=None) -> dict:
    """The scan of what the verifier received for every window's witness pieces, leaving out the pieces the public
    circuit or a received window's ``pub`` row holds; `received`, the windows the verifier got (a sampled period's
    drawn ones), by default every one. The control (the prover's instance files) is to hold every held piece but a
    head's zero-padded last chunk, which `public_pieces` (``fields``) leaves out of the held ones since a head inside
    its row never shows it (``padded_tails`` counts them); the scan of what the verifier received still looks for
    those."""
    rows = b"".join(rd.lw.row(x.args, "pub") for x in rd.windows
                    if x.args is not None and (received is None or x.window in received))
    raw, held_blobs = blobs_of(rd)
    blobs, held, public = public_pieces(M.a8, raw, held_blobs, sg.circuit.read_bytes() + rows)
    tails = {f"{k} @{len(b) - len(b) % M.a8.PIECE}" for k, b in raw.items() if k in held_blobs and len(b) % M.a8.PIECE}
    control = out / "prover-instances.bin"
    with open(control, "wb") as f:
        for x in rd.windows:
            f.write((x.statement.dir / "inst-1.bin").read_bytes())
    got = M.a8.received(out / "verifier", control, blobs, held)
    return {**got, **LS.scan(got, blobs, out / "verifier"), "public_pieces": public, "files": len(got["files"]),
            "padded_tails": len(tails & blobs.keys())}


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    warden = a.warden_run is not None
    run = Path(a.warden_run if warden else a.run)
    summary: dict = {"run": str(run), "extra": a.extra and str(a.extra), "late": a.late and str(a.late),
                     "debug_clear": a.debug_clear, "gpu": a.gpu, "received": RECEIVED}
    t = time.time()
    members = {"honest": link_windows(run, warden=warden),
               "tampered": link_windows(run, edit=every_count if a.draw else one_count, warden=warden)}
    members["tampered"].every = bool(a.draw)
    n = len(members["honest"].windows)
    if a.draw and not 1 <= a.draw <= n:
        raise SystemExit(f"--draw {a.draw}: a period draw takes 1 to {n} of the link's {n} windows")
    if a.draw and a.draw < n:
        members[OTHER] = link_windows(run, warden=warden)
    summary["draw"] = ({"selection": "all", "k": n, "population": n, "coins": None} if not a.draw else {
        "selection": "sampled", "law": f"subset:{a.draw}", "k": a.draw, "population": n, "delta": cs.DELTA,
        "coins": {"source": "verifier", "by": "the verifier's own: the operating system's bytes through the Lean "
                  "verifier of record (sampled_proofs verifier.coins), read only after the period's commitment; not "
                  "the auditor's challenger's"}})
    if warden:
        summary["warden_run"] = members["honest"].plan["warden_run"]
        summary["warden"] = {"syncs": "none: the warden runs no shaper clock, so a window declares no sync and the "
                                      "link's charge is 0",
                             "counts": "the warden's schedule's (service.WardenProfile), not a profile's"}
    for name in ("extra", "late"):
        if getattr(a, name):
            members[name] = link_windows(Path(getattr(a, name)))
    members = {k: v for k, v in members.items() if k in a.rounds or k == OTHER}
    lw0 = next(iter(members.values())).lw
    summary["link"] = {"name": lw0.link, "unit": lw0.unit.id, "keys": {e: lw0.head_key(d) for d, e in enumerate(ENDS)},
                       "program": lw0.partition["program"], "partition": P.digest(lw0.partition),
                       "gates": [g.id for g in lw0.gates()], "pinned": all(HC.pinned(g) for g in lw0.gates()),
                       "windows": {k: len(v.windows) for k, v in members.items()},
                       "planned_windows": {k: v.plan["windows"] for k, v in members.items()}}
    summary["instances_s"] = round(time.time() - t, 1)
    for name, rd in members.items():
        if rd.lw.partition != lw0.partition or rd.lw.program != lw0.program:
            raise SystemExit(f"{name}: its link's registered unit is not the honest run's")
        summary.setdefault("members", {})[name] = {
            "reference": [{"window": x.window, "word": x.word, "expect": x.expect, "failed": x.failed} for x in rd.windows],
            "issuer_refused": rd.issuer_refused}
        log(f"{name}: {len(rd.windows)} link-windows, words {[x.word for x in rd.windows]}, expected verdicts "
            f"{[x.expect for x in rd.windows]}, failed parts {[x.failed for x in rd.windows if any(x.failed.values())]}")
        if not expected(name, rd):
            raise SystemExit(f"{name}: the references are not the {name} round's")
    if not summary["link"]["pinned"]:
        raise SystemExit(f"a gate names no pinned circuit digest: {summary['link']['gates']}")
    if a.dry:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        log("dry: done")
        return 0

    M = _one_stage()
    M.PF.executable = lambda: Path(a.lean)
    _record_lean_runs(M.PF)
    sg = stager(M, lw0, out / "stager")
    summary["stager"] = {"unit": lw0.proof_unit.id, "lower_s": sg.lower_s, "k_log": sg.k_log, "pin": sg.st.pin,
                         "circuit_sha512": sg.circuit_sha512, "bits": {p: lw0.bits(p) for p in SVC.PORTS},
                         "ands": sg.st.low.layout.counts["ands"]}
    log(f"stager: {lw0.proof_unit.id}, {sg.lower_s} s, k_log {sg.k_log}, {summary['stager']['ands']} ANDs, "
        f"pin {sg.st.pin[:16]}")
    summary["staging"] = stage(M, sg, members, os.urandom(32), out)
    log(f"staged: {summary['staging']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    def go(k: int, name: str) -> tuple[str, dict]:
        res = member(M, a, sg, name, members[name], out / name, a.port + 100 * k if a.port else 0,
                     coins=S.V.coins if a.draw else None, forge=other_subset if name == OTHER else None)
        res["received"] = _member_scan(M, sg, members[name], out / name,
                                       set(res["period"]["units"]) if "period" in res else None)
        return name, res

    done = dict(go(k, name) for k, name in enumerate(members))
    ok = True
    for name, res in done.items():
        rec, net = res["received"], res["network"]
        good = (all(r["as_expected"] for r in res["windows"]) and net["accepted"] == (name == "honest")
                and set(net.get("why") or ()) == ({SVC.PERIOD_DRAWN} if name == OTHER else set())
                and not rec["leaked"] and rec["control"]["holds_all"])
        ok &= good
        summary["members"][name].update(res, ok=good)
        if "wrong_units" in net:
            sel, wu = net["selection"], net["wrong_units"]
            summary.setdefault("sampled", {})[name] = {
                "accepted": net["accepted"], "proved": [r["window"] for r in res["windows"]], "why": net["why"],
                "draw": {"law": f"subset:{sel['k']}", "coins": "verifier", "units": sel["units"],
                         "of": sel["population"], "commitment": sel["commitment"], "by": sel["by"],
                         "coin_bytes": len(res["period"]["read"]) // 2},
                "audit": {"accepted": net["accepted"], "wrong_units": wu},
                "charge": {"delta": wu["delta"], "wrong_units": wu, "epsilon": wu["epsilon"]} if net["accepted"]
                else None}
            log(f"member {name}: the draw of record ({sel['by']}) {sel['units']}, {sel['k']} of {sel['population']}: "
                f"at most {wu['bound']} wrong, epsilon {wu['epsilon']:.4f} at delta {wu['delta']:.3g}"
                + (f"; refused by name {sorted(net['why'])}" if net["why"] else ""))
        log(f"member {name}: network {'accept' if res['network']['accepted'] else 'refuse'} (refused windows "
            f"{res['network']['refused']}), windows {[int(r['accepted']) for r in res['windows']]}, verifier received "
            f"{rec['verifier_bytes']} bytes in {rec['files']} files, witness pieces found there: "
            f"{rec['verifier_found'] or 'none'} (of {rec['blobs']} blobs; {len(rec['public_pieces'])} pieces the public "
            f"circuit holds left out), leaked {rec['leaked'] or 'none'}, structured {rec['structured'] or 'none'}, "
            f"control {rec['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="n2_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("virtual", help="certifier-serve's honest, extra and late runs on an injected clock, stub endpoint")
    v.add_argument("--out", required=True, type=Path)
    v.add_argument("--requests", type=int, default=40)
    v.add_argument("--tokens", type=int, default=4)
    p = sub.add_parser("prove", help="each link-window's unit proved --zk and decided by N2's verifier of record")
    run = p.add_mutually_exclusive_group(required=True)
    run.add_argument("--run", type=Path, help="the honest run's directory")
    run.add_argument("--warden-run", type=Path, help="an active warden's run (warden_serve.py's, or the gate's "
                     "warden-rehearse's): its own certified records, no certifier rehearsal (warden_chain)")
    p.add_argument("--extra", type=Path, help="certifier-serve's extra run of the same plan")
    p.add_argument("--late", type=Path, help="certifier-serve's late run of the same plan")
    p.add_argument("--rounds", default=",".join(ROUNDS), help="the members to run, comma-separated")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=0,
                   help="member k's window j listens on PORT + 100k + j; 0 (the default): a free port for each session")
    p.add_argument("--jobs", type=int, default=1, help="1: the members proved one after another (ONE_SESSION); more "
                                                       "is refused")
    p.add_argument("--debug-clear", action="store_true",
                   help="each verdict also carries the in-the-clear check on the opening, which decides nothing")
    p.add_argument("--gpu", action="store_true", help="each session's prover on the GPU (`flock-circuit prove --gpu`)")
    p.add_argument("--draw", type=int, default=0,
                   help="K: each member's period sampled, K of its windows drawn on the verifier's coins after every "
                        "window's root is committed, and only those proved (LinkWindows.period_draw); tampered then "
                        "changes every window, and other_subset proves another K-subset; 0 (the default): every window")
    p.add_argument("--dry", action="store_true", help="the instances and their references only: nothing lowered or proved")
    a = ap.parse_args(argv)
    if a.command == "virtual":
        plan = virtual(a.out, a.requests, a.tokens)
        log(f"virtual: {plan['windows']} windows, served {plan['served']}, under {a.out}")
        return 0
    one_session(a.jobs)
    a.rounds = tuple(x for x in a.rounds.split(",") if x)
    if a.warden_run is not None and (a.extra or a.late):
        raise SystemExit("--extra and --late are certifier-serve's runs: a warden run's members are honest and "
                         "tampered")
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
