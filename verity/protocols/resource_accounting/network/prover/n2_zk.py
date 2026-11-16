"""N2's verdict of record from zero-knowledge proofs of N2's registered proof block, on an active network attester's
run: each execution phase committed, registered on the link's own stream (`service.TransferBlocks.stream`, the proof
system's `Stream`), selected (Selection All: its one proof block, no coins), its proof block (`NetworkTransferBlocks_v5`,
both ends' head checks on their pinned gate ids) proved by Flock `prove --zk` behind the Lean ZK server, delivered, and
decided by `TransferBlocks.outcome`: the verifier's own Lean `verify --zk` of the proof, the proof's public row opened to
the verifier's own copy and its output row read. No host check reads the opening; `--debug-clear`, a run by hand's,
adds the in-the-clear comparison to each verdict, which decides nothing and goes to ``--witness`` alone (it needs one),
the public summary keeping the flag and the outcome the verifier receives carrying none of it.
The public summary (``summary.json``) holds only the keys its list of released_keys.json declares; the run's path,
the device, timings and memory, the references' failed parts and the scan's finds go to ``--witness``.
The printed log is the summary's split too (`log`, `wlog`, `sides`): stdout carries only public lines, a failure as its
name alone; the process's stdout and stderr, which every child it starts inherits, its tracebacks and failures' text and
the witness lines go to ``n2.log`` beside ``--witness``, and without ``--witness`` nowhere.

    python benchmarks/network_traces/zk_drivers.py n2_zk prove --warden-run DIR --bin flock-circuit --lean verity-verify-proof --out DIR
      [--jobs 1] [--gpu] [--debug-clear] [--dry] [--draw K] [--witness FILE]

* **The run** (``--warden-run``): an active network attester's run (`network_attester_serve.py`'s, or the gate's
  ``warden-rehearse``'s), read by `network_attester_chain`: its own certified records under its key and session, on the network
  attester link (`service.NetworkAttesterProfile`: both heads at the network attester's grid and parameters digest, no sync
  declared or charged), a certifier rehearsal's run refused by name. Which execution phases its signer refused follows
  what the node sent, so the public summary of a run whose signer refused any holds each member's
  ``certifier_refused`` and ``ok`` alone, nothing per window, and the members whole go to ``--witness``.
* **Members.** One stream each, over all of the run's execution phases: ``honest``; ``tampered``, the honest records
  with window 0's egress head one count off (`one_count`), committed by the prover under its own root, the certificates
  still the network attester's, its head failing its signed digest alone. The prover's references
  (`head_check.reference_parts`, and the proof block's output word by the gates' references, `TransferBlocks.outputs`) say
  beforehand which windows each member's proofs refuse (`expected`).
* **The period** (``--draw K``): each member's windows sampled rather than all proved: every window's salted root
  committed, then K of them sampled on the verifier's own coins (`TransferBlocks.period_draw`), and only those
  registered, proved and decided; the network verdict is the sampled one, its ``wrong_units`` at the
  stream's δ, on the Lean verifier's sample. ``tampered`` then changes every window (`every_count`), which any
  sample refuses, and with K < n ``other_subset`` proves the honest records on another K-subset (`other_subset`), every
  window accepted, which the network verdict refuses by name (``period-draw``).
* **The statement** (`stager`, `stage`). One circuit, `TransferBlocks.proof_block` lowered once (the registered unit with each
  gate the Definition its pinned digest names), bound to the registered partition and Program. Each execution phase is a
  statement of its own with one instance, proof block 0: rows ``pub`` and ``wit`` (`TransferBlocks.layout`,
  `TransferBlocks.row`) and the output row ``o0``, the verdict and deadline bits. The lowering evaluates every instance at
  once, held to the prover's reference and to the IR reference; each statement's salts come from a key of its own, and
  the verifier's opening of its ``pub`` and ``o0`` rows is checked on the staged public file before anything is proved.
* **A round** (`member`), per execution phase in order: register, select, one live `prove --zk` session behind the ZK
  server on the statement (the server serving the stream's sample), deliver, and the outcome of record. Per window:
  prove (the session's wall time, the prover's ``prove_total_s``, proof bytes) and verify (the verifier's run,
  Lean included). Then the link's network verdict (`TransferBlocks.network_verdict`). The members' rounds run one after
  another: outer proofs go through the ZK server one at a time (`ONE_SESSION`), so ``--jobs`` is 1 and more is refused.
* **What the verifier receives**, per execution phase, kept under ``<member>/verifier/`` (`RECEIVED`): the registration
  record (the salted root, nothing of the certificates) and the sample; the two certificates' public fields, from the certifier,
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
import contextlib
import functools
import gc
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from verity.protocols.resource_accounting.network.prover import leak_scan as LS, proof_drivers
from verity.protocols.resource_accounting.network.prover import network_attester_chain as WC
from verity.protocols.resource_accounting.network.verifier import head_check as HC
from verity.protocols.resource_accounting.network.verifier import service as SVC
from verity.protocols.resource_accounting.network.verifier.certificate import CertifiedRecord, Direction
from verity.protocols.resource_accounting.network.verifier.schedule import ClockSync
from verity.protocols.resource_accounting.network.verifier.service import TransferBlocks
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import partition as P

ENDS = ("egress", "ingress")
#: the stream's δ, by default
DELTA = 2.0 ** -40
#: the sampled period's negative (``--draw K``, K < n): the prover proves another K-subset than the sample of record's
OTHER = "other_subset"
ROUNDS = ("honest", "tampered")
#: AGENTS.md's design rule, which `one_session` holds every driver's ``--jobs`` to
ONE_SESSION = "outer proofs go through the ZK server one at a time (AGENTS.md, 'Live coins only')"
#: what the verifier receives per execution phase, by file (`member`)
RECEIVED = {"registration.json": "the registration record", "draw.json": "the stream's sample",
            "certificates.json": "the certificates' public fields (the certifier's)", "syncs.json": "the declared syncs",
            "pub-1.bin": "the statement's public file", "session-*": "the session's files",
            "delivered.json": "the pub and o0 rows' salts and the output word", "outcome.json": "the outcome record"}


#: where `log` writes inside `sides`: the stdout the process had, which the chain releases as ``n2.log``
PUBLIC = None


def log(msg: str) -> None:
    print(f"N2ZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", file=PUBLIC or sys.stdout, flush=True)


@contextlib.contextmanager
def sides(witness: Path | None):
    """The process's stdout and stderr, which every child it starts inherits and where a traceback, a stray print or a
    child's message lands, sent to ``n2.log`` beside ``witness``, or to nowhere without one; `log` writes to the
    stdout the process had."""
    global PUBLIC
    sys.stdout.flush()
    sys.stderr.flush()
    if witness is not None:
        witness.parent.mkdir(parents=True, exist_ok=True)
    sink = os.open(witness.with_name("n2.log") if witness is not None else os.devnull,
                   os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    saved, streams = (os.dup(1), os.dup(2)), (sys.stdout, sys.stderr)
    PUBLIC = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    sys.stdout = sys.stderr = os.fdopen(sink, "w", buffering=1)
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = streams
        PUBLIC.close()
        PUBLIC = None
        for fd, kept in zip((1, 2), saved):
            os.dup2(kept, fd)
            os.close(kept)


def wlog(a, msg: str) -> None:
    """A line of what the public summary leaves out: appended to ``n2.log`` beside ``--witness``, and dropped without
    one."""
    if a.witness is None:
        return
    a.witness.parent.mkdir(parents=True, exist_ok=True)
    with open(a.witness.with_name("n2.log"), "a") as f:
        f.write(f"N2ZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}\n")


def plog(a, msg: str) -> None:
    """A line of public values that says something per execution phase: public, but on the witness side when the
    signer refused any phase, as the public summary then holds nothing per window."""
    (wlog if getattr(a, "refused_any", False) else lambda _a, m: log(m))(a, msg)


def _no_coins(n: int) -> bytes:
    raise AssertionError("Selection All samples every execution phase and reads no coins")


#: the verifier's own Lean runs (`proofs.verify`'s findings and refusal), by the proof's public file, so each
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


def _without_clear(body: dict) -> dict:
    """The outcome record as the verifier receives it: without `--debug-clear`'s in-the-clear check, which is witness
    (the window's ``clear``)."""
    verdict = ((body.get("audit") or {}).get("verdicts") or {}).get(SVC.VERIFIER) or {}
    if "clear" not in verdict:
        return body
    body = json.loads(json.dumps(body, default=str))
    del body["audit"]["verdicts"][SVC.VERIFIER]["clear"]
    return body


# --- the run's execution phases ---------------------------------------------------------------------------------------

def network_attester_link(run: Path) -> tuple[TransferBlocks, dict, dict, list]:
    """A network attester run's link (`network_attester_chain.link`: the active network attester's own records, its key and
    session, no sync), the plan it stands for (its windows, no syncs), its certified records by (direction, window) and
    the signer's refusals. A certifier rehearsal's run is refused by name (`network_attester_chain.PROXY`)."""
    lw, dep, _ = WC.link(run)
    got, refused = WC.pairs(run, dep)
    certified = {(d, w): r for w, pair in enumerate(got) for d, r in zip((Direction.EGRESS, Direction.INGRESS), pair)}
    return lw, {"windows": dep["windows"], "syncs": [], "warden_run": dep["run_id"]}, certified, refused


def phases(run: Path, edit=None) -> SimpleNamespace:
    """Every execution phase of the active network attester's run (`network_attester_link`) with both certified records, as the
    prover holds it before it registers: its opening (fresh salts), the root it commits (`TransferBlocks.root`), the proof
    block's arguments under that root (None where the proof block takes no instance), each end's head check's failed
    parts (`head_check.reference_parts`) and the proof block's output word by the gates' references
    (`TransferBlocks.outputs`); and the network attester's certificates, which the verifier receives from it, no sync
    declared. ``edit(lw, w, pair)``: the records the prover commits in place of the certified ones. ``expect``: the
    verdict the verifier should reach on an honest proof of the opening."""
    lw, plan, certified, refused = network_attester_link(run)
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
        others = lw.certified(w, certificates) and (lw.network_attester is not None
                                                    or (w < len(syncs) and syncs[w] in lw.sync_set))
        out.append(SimpleNamespace(
            window=w, opening=opening, root=root, insts=insts, args=args, word=word, certificates=certificates,
            failed={e: None if i is None else [k for k, ok in HC.reference_parts(i).items() if not ok]
                    for e, i in zip(ENDS, insts)},
            expect=int(word == 0b11 and others), statement=None))
    return SimpleNamespace(lw=lw, plan=plan, syncs=syncs, windows=out, issuer_refused=refused)


def one_count(lw: TransferBlocks, w: int, pair: tuple[CertifiedRecord, CertifiedRecord]):
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


def every_count(lw: TransferBlocks, w: int, pair: tuple[CertifiedRecord, CertifiedRecord]):
    """Every window's egress head with one count more (`one_count`'s change, in each window): the tampered member of a
    sampled period (``--draw``), which refuses it whichever windows it samples."""
    return one_count(lw, 0, pair)


def expected(name: str, rd: SimpleNamespace) -> bool:
    """Whether the member's references are its round's: every proof block takes an instance, and its verdict bit is 1
    exactly where no part fails; honest and other_subset, none fails and every window accepts; tampered, window 0's egress
    head (every window's, ``rd.every``) fails its signed digest alone, the network attester link's counts being its schedule's.
    The ingress heads fail nothing."""
    xs = rd.windows
    if not xs or any(x.word is None for x in xs) or any(bool(x.word & 1) != (not x.failed["egress"] and not
                                                                            x.failed["ingress"]) for x in xs):
        return False
    if any(x.failed["ingress"] for x in xs):
        return False
    eg = [x.failed["egress"] for x in xs]
    if name in ("honest", OTHER):
        return all(x.expect for x in xs)
    want = ["signed"]
    return eg[0] == want and all(f == (want if getattr(rd, "every", False) else []) for f in eg[1:])


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


# --- the statement ----------------------------------------------------------------------------------------------------

def _one_stage():
    import numpy as np
    from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
    from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

    d = proof_drivers.drivers()
    return SimpleNamespace(a0=d.a0, a8=d.a8, np=np, P=P, PF=PF, V2=V2, FC=d.circuit, CS=d.class_statement)


def stager(M, lw: TransferBlocks, out: Path) -> SimpleNamespace:
    """The one circuit: `TransferBlocks.proof_block` lowered once on rows ``pub`` and ``wit`` with output row ``o0``, bound
    to the registered partition and Program, which are written beside it as the verifier's own. Its ports are
    `service.PORTS`, each the length the verifier opens (`TransferBlocks.bits`)."""
    rows = [(r, M.V2, lw.bits(r), list(lw.layout[r])) for r in SVC.ROWS]
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(lw.partition, sort_keys=True))
    (out / "program.json").write_bytes(lw.program)
    bound = {"partition": M.P.digest(lw.partition), "program": lw.partition["program"]}
    t, was = time.time(), gc.isenabled()
    gc.disable()
    try:
        st = M.CS.UnitStager(lw.proof_block, "unit/n2-link-window", rows, [(M.V2, (0,))], circuit=out / "circuit.txt",
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
    """Every execution phase's statement on the one circuit, one instance each, proof block 0. The lowering evaluates
    all of their rows at once (`Lowering.evaluate`), which must satisfy every proof block and give the prover's output
    words, held to the IR reference in one batch (`UnitStager.reference_check`); then each statement's files are written
    as `UnitStager.stage` writes them (`circuit.write`), under salts from a key of the statement's own
    (`circuit.stage_salts` derives them from the key and the set's content, which every statement shares). The
    verifier's opening of each staged ``pub`` and ``o0`` row is checked on its public file. Sets each window's
    ``statement``."""
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


def zk_server_env(M, sg: SimpleNamespace) -> dict:
    """A zk_served prover's environment (`FC_ZK_SERVER=1`, its Lean frame named by ``FC_ZK_SERVER_LEAN``) on the
    statement's partition and Program."""
    if not os.environ.get("FC_ZK_SERVER_LEAN"):
        raise SystemExit("FC_ZK_SERVER_LEAN names no Lean ZK server frame (verity-verify-proof's flock-zk-server)")
    return {"FC_ZK_SERVER": "1", **M.a0.zk_server_env({"partition_file": sg.partition_file,
                                                     "program_file": sg.program_file})}


def one_session(jobs: int) -> None:
    """A driver's ``--jobs``: 1, its members proved one after another, each session behind the ZK server alone
    (`ONE_SESSION`); any other count is refused before anything runs."""
    if jobs != 1:
        raise SystemExit(f"--jobs {jobs}: {ONE_SESSION}, so a prover runs one ZK server session at a time: --jobs 1")


# --- the rounds -------------------------------------------------------------------------------------------------------

def _certificate(c) -> dict:
    return {k: (v.hex() if isinstance(v, bytes) else getattr(v, "code", v)) for k, v in asdict(c).items()}


def _free_port() -> int:
    """A loopback port the OS has free now. A pool's slots share a machine, so a fixed port collides across runs."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def other_subset(sampled: dict) -> list[int]:
    """A K-subset of the period's windows that isn't the sample's: the first K the sample didn't hold, then sampled ones
    from the start where it left fewer than K (none other at K = n)."""
    held = set(sampled["units"])
    return sorted(([w for w in range(sampled["population"]) if w not in held] + sampled["units"])[:sampled["k"]])


def member(M, a, sg: SimpleNamespace, name: str, rd: SimpleNamespace, out: Path, port: int,
           coins=None, forge=None) -> dict:
    """The member's execution phases on its own stream, in order: register, select, prove the statement live behind the
    ZK server, deliver, and decide on the proof (`TransferBlocks.outcome`); then its network verdict. Writes what the
    verifier receives under ``out / "verifier"`` (`RECEIVED`), each file prefixed by its window. Window j's session
    listens on ``port + j``, or with ``port`` 0 on a port the OS picks free.

    With ``a.draw`` K (> 0), the period is sampled (`TransferBlocks.period_draw`): every window's salted root is
    committed first, then `coins` sample K of the windows, and only those are registered, proved and
    decided; the sample is ``period.json`` beside what the verifier receives, and the network verdict is the sampled one,
    with its ``wrong_units`` at the stream's δ (``a.delta``, by default `DELTA`), on the sample of
    record: the Lean verifier's (``--lean``) sample on the coins the period flip read (`service.lean_draw`),
    which the windows proved must be. Coins with a ``drawn`` (the challenger's, `a5_coins.RankCoins`) end their turn
    on the session once the sample has read them, and the result's ``segment`` is what they released. ``forge``, a
    negative, is the prover's: of the period sample, the K windows it registers and proves instead, under a sample
    object listing them (`other_subset`), which is what ``period.json`` then holds."""
    lw = rd.lw
    k = getattr(a, "draw", 0) or None
    stream = lw.stream(budget=rd.plan["windows"], delta=getattr(a, "delta", None) or DELTA, challenger="n2-zk",
                       coins=_no_coins, clock=time.monotonic)
    gpu = getattr(a, "gpu", False)
    # the ZK server rechecks a witness the host computed, so a device prover takes the host's, as n2_tie_zk's does
    env = {**zk_server_env(M, sg), **({"FC_HOST_WITNESS": "1"} if gpu else {})}
    av = SimpleNamespace(bin=a.bin, lease=[], gpu=gpu)
    vdir = out / "verifier"
    vdir.mkdir(parents=True, exist_ok=True)
    sampled, windows, segment = None, rd.windows, None
    if k is not None:
        try:
            sampled = lw.period_draw(stream, [x.root.hex() for x in rd.windows], k, coins)
        finally:
            if callable(getattr(coins, "drawn", None)):
                segment = coins.drawn()
        plog(a, f"{name}: sampled windows {sampled['units']} ({sampled['k']} of {sampled['population']}) on "
             f"{len(sampled['read']) // 2} coin bytes after the period's commitment {sampled['commitment'][:16]}")
        if forge is not None:
            sampled = {**sampled, "units": forge(sampled)}
            plog(a, f"{name}: the prover proves windows {sampled['units']} instead")
        (vdir / "period.json").write_text(json.dumps(sampled, indent=1, sort_keys=True))
        windows = [rd.windows[u] for u in sampled["units"]]
    previous, rows = None, []
    for j, x in enumerate(windows):
        wdir, pre = out / f"w{x.window:03d}", f"w{x.window:03d}-"
        reg = lw.register(stream, lw.registration(x.opening, previous), sampled)
        previous = reg.digest
        if reg.record["roots"][0]["root"] != x.root.hex():
            raise SystemExit(f"{name} window {x.window}: the registered root is not the statement's")
        sample = stream.select(reg)
        if list(sample.units) != [0]:
            raise SystemExit(f"{name} window {x.window}: the stream sampled {list(sample.units)}, not the one unit")
        sdir = wdir / "session"
        m = {"dir": x.statement.dir, "n": 1, "pin": sg.st.pin}
        record, live, proc = M.a8.session(av, m, sdir, [0], port + j if port else _free_port(), env=env)
        sessions = tuple(sorted(p.parent for p in (sdir / "sessions").glob("*/session.json")))
        delivered = SVC.Delivered(M.PF.Proof(sg.circuit, x.statement.dir / "pub-1.bin", sessions), x.statement.salts,
                                  x.statement.output)
        late = stream.deliver(sample, delivered.sessions())
        t = time.time()
        outcome = late or lw.outcome(stream, sample, x.certificates, rd.syncs, delivered,
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
        files = {"registration.json": reg.record, "draw.json": dict(sample.draw),
                 "certificates.json": [_certificate(c) for c in x.certificates],
                 "syncs.json": [[s.delta, str(s.rho)] for s in rd.syncs], "outcome.json": _without_clear(body),
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
        plog(a, f"{name} window {x.window}: {'accept' if outcome.accepted else 'reject'} public {row['public']}"
             f"{' LATE' if late else ''}, Lean {'proved' if proof['proved'] else 'NOT proved'}, output word "
             f"{proof['opened_word']}; {'as expected' if row['as_expected'] else 'UNEXPECTED'}")
        wlog(a, f"{name} window {x.window}: prove {row['prove']['session_s']} s (prove_total "
             f"{row['prove']['prove_total_s']} s, {row['prove']['proof_bytes']} B), verify {verify_s} s")
    network = lw.network_verdict(stream, rd.plan["windows"], sampled)
    if sampled is not None:
        sel = network["selection"]
        plog(a, f"{name}: the sample of record ({sel['by']}) is windows {sel['units']}; proved "
             f"{[r['window'] for r in rows]}; " + (f"REFUSED by name: {network['why']}" if network["why"] else
                                                    "the windows proved are the sample of record's"))
    return {"windows": rows, "network": network, **({} if sampled is None else {"period": sampled}),
            **({} if segment is None else {"segment": segment})}


def _member_scan(M, sg, rd, out: Path, received=None) -> dict:
    """The scan of what the verifier received for every window's witness pieces, leaving out the pieces the public
    circuit or a received window's ``pub`` row holds; `received`, the windows the verifier got (a sampled period's
    sampled ones), by default every one. The control (the prover's instance files) is to hold every held piece but a
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
    run = Path(a.network_certifier_run)
    summary: dict = {"run": str(run), "debug_clear": a.debug_clear, "gpu": a.gpu, "received": RECEIVED}
    # what the public summary leaves to --witness: the run's path, the prover's device, timings and memory, values in
    # the clear, the prover's own reference parts and scan files, and which execution phases the signer refused
    kept = ("run", "gpu", "instances_s", "staging/evaluate_s", "staging/reference_s", "staging/write_s",
            "stager/lower_s", "members/*/issuer_refused", "members/*/reference/*/failed", "members/*/windows/*/failed",
            "members/*/windows/*/prove", "members/*/windows/*/verify_s", "members/*/windows/*/clear",
            "members/*/received/verifier_bytes", "members/*/received/verifier_found",
            "members/*/received/public_pieces", "members/*/received/placed", "members/*/received/was",
            "members/*/received/structured", "members/*/received/control/bytes",
            "members/*/received/control/blobs_found", "members/*/received/control/missing")

    def take(doc, segs: list[str], into: dict) -> None:
        for k in (list(doc) if isinstance(doc, dict) else range(len(doc))) if segs[0] == "*" else [segs[0]]:
            if isinstance(doc, dict) and k not in doc:
                continue
            if len(segs) == 1:
                into[str(k)] = doc.pop(k)
            elif isinstance(doc[k], (dict, list)):
                take(doc[k], segs[1:], sub := into.setdefault(str(k), {}))
                if not sub:
                    del into[str(k)]

    def write(summary: dict) -> None:
        """The public summary, and the rest to --witness (a list's entries by their index). A run whose signer refused
        an execution phase is public as each member's flag and its ok alone, so nothing public says where."""
        full = json.loads(json.dumps(summary, default=str))
        public, private = json.loads(json.dumps(full)), {}
        for path in kept:
            take(public, path.split("/"), private)
        if any(m.get("certifier_refused") for m in public.get("members", {}).values()):
            private |= {"members": full["members"], "link": {"windows": full["link"]["windows"]},
                        **({"sampled": public.pop("sampled")} if "sampled" in public else {})}
            public["members"] = {k: {"certifier_refused": True, **({"ok": m["ok"]} if "ok" in m else {})}
                                 for k, m in public["members"].items()}
            public["link"].pop("windows", None)
        (out / "summary.json").write_text(json.dumps(public, indent=1, sort_keys=True))
        if a.witness is not None:
            a.witness.parent.mkdir(parents=True, exist_ok=True)
            a.witness.write_text(json.dumps(private, indent=1, sort_keys=True))

    t = time.time()
    members = {"honest": phases(run), "tampered": phases(run, edit=every_count if a.draw else one_count)}
    members["tampered"].every = bool(a.draw)
    n = len(members["honest"].windows)
    if a.draw and not 1 <= a.draw <= n:
        raise SystemExit(f"--draw {a.draw}: a period sample takes 1 to {n} of the link's {n} windows")
    if a.draw and a.draw < n:
        members[OTHER] = phases(run)
    summary["draw"] = ({"selection": "all", "k": n, "population": n, "coins": None} if not a.draw else {
        "selection": "sampled", "law": f"subset:{a.draw}", "k": a.draw, "population": n, "delta": DELTA,
        "coins": {"source": "verifier", "by": "the verifier's own: the operating system's bytes through the Lean "
                  "verifier of record (spot verifier.coins), read only after the period's commitment; not "
                  "the auditor's challenger's"}})
    summary["warden_run"] = members["honest"].plan["warden_run"]
    summary["warden"] = {"syncs": "none: the network certifier runs no shaper clock, so a window declares no sync and the link's "
                                  "charge is 0",
                         "counts": "the network certifier's schedule's (service.NetworkAttesterProfile), not a profile's"}
    members = {k: v for k, v in members.items() if k in a.rounds or k == OTHER}
    lw0 = next(iter(members.values())).lw
    summary["link"] = {"name": lw0.link, "unit": lw0.unit.id, "keys": {e: lw0.head_key(d) for d, e in enumerate(ENDS)},
                       "program": lw0.partition["program"], "partition": P.digest(lw0.partition),
                       "gates": [g.id for g in lw0.gates()], "pinned": all(HC.pinned(g) for g in lw0.gates()),
                       "windows": {k: len(v.windows) for k, v in members.items()},
                       "planned_windows": {k: v.plan["windows"] for k, v in members.items()}}
    summary["instances_s"] = round(time.time() - t, 1)
    a.refused_any = any(rd.issuer_refused for rd in members.values())
    for name, rd in members.items():
        if rd.lw.partition != lw0.partition or rd.lw.program != lw0.program:
            raise SystemExit(f"{name}: its link's registered unit is not the honest run's")
        summary.setdefault("members", {})[name] = {
            "reference": [{"window": x.window, "word": x.word, "expect": x.expect, "failed": x.failed} for x in rd.windows],
            "certifier_refused": bool(rd.issuer_refused), "issuer_refused": rd.issuer_refused}
        plog(a, f"{name}: {len(rd.windows)} execution phases, words {[x.word for x in rd.windows]}, expected "
             f"verdicts {[x.expect for x in rd.windows]}")
        wlog(a, f"{name}: failed parts {[x.failed for x in rd.windows if any(x.failed.values())]}")
        if not expected(name, rd):
            raise SystemExit(f"{name}: the references are not the {name} round's")
    if not summary["link"]["pinned"]:
        raise SystemExit(f"a gate names no pinned circuit digest: {summary['link']['gates']}")
    if a.dry:
        write(summary)
        log("dry: done")
        return 0

    M = _one_stage()
    M.PF.executable = lambda: Path(a.lean)
    _record_lean_runs(M.PF)
    sg = stager(M, lw0, out / "stager")
    summary["stager"] = {"unit": lw0.proof_block.id, "lower_s": sg.lower_s, "k_log": sg.k_log, "pin": sg.st.pin,
                         "circuit_sha512": sg.circuit_sha512, "bits": {p: lw0.bits(p) for p in SVC.PORTS},
                         "ands": sg.st.low.layout.counts["ands"]}
    log(f"stager: {lw0.proof_block.id}, k_log {sg.k_log}, {summary['stager']['ands']} ANDs, pin {sg.st.pin[:16]}")
    wlog(a, f"stager: lowered in {sg.lower_s} s")
    summary["staging"] = stage(M, sg, members, os.urandom(32), out)
    log(f"staged: {summary['staging']['statements']} statements")
    wlog(a, f"staged: {summary['staging']}")
    write(summary)

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
            plog(a, f"member {name}: the sample of record ({sel['by']}) {sel['units']}, {sel['k']} of "
                 f"{sel['population']}: at most {wu['bound']} wrong, epsilon {wu['epsilon']:.4f} at delta "
                 f"{wu['delta']:.3g}" + (f"; refused by name {sorted(net['why'])}" if net["why"] else ""))
        plog(a, f"member {name}: network {'accept' if res['network']['accepted'] else 'refuse'} (refused windows "
             f"{res['network']['refused']}), windows {[int(r['accepted']) for r in res['windows']]}")
        plog(a, f"member {name}: the verifier received {rec['files']} files, scanned for {rec['blobs']} witness "
             f"pieces: leaked {rec['leaked'] or 'none'}, control holds all {rec['control']['holds_all']}; "
             f"{'as expected' if good else 'UNEXPECTED'}")
        wlog(a, f"member {name}: verifier received {rec['verifier_bytes']} bytes, witness pieces found there: "
             f"{rec['verifier_found'] or 'none'} ({len(rec['public_pieces'])} pieces the public circuit holds left "
             f"out), structured {rec['structured'] or 'none'}, control {rec['control']['blobs_found']}")
    summary["ok"] = ok
    write(summary)
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="n2_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="each execution phase's proof block proved --zk and decided by N2's verifier of "
                                     "record")
    p.add_argument("--warden-run", dest="network_certifier_run", type=Path, required=True,
                   help="an active network certifier's run (network_attester_serve.py's, or the gate's warden-rehearse's): "
                        "its own certified records, no certifier rehearsal (network_attester_chain)")
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
                   help="K: each member's period sampled, K of its windows sampled on the verifier's coins after every "
                        "window's root is committed, and only those proved (TransferBlocks.period_draw); tampered then "
                        "changes every window, and other_subset proves another K-subset; 0 (the default): every window")
    p.add_argument("--dry", action="store_true", help="the instances and their references only: nothing lowered or proved")
    p.add_argument("--witness", type=Path, help="where what the public summary leaves out is written, on the chain's "
                                                "private side: the timings, the values in the clear, and each "
                                                "member's execution phases its signer refused")
    a = ap.parse_args(argv)
    one_session(a.jobs)
    a.rounds = tuple(x for x in a.rounds.split(",") if x)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    if a.debug_clear and a.witness is None:
        raise SystemExit("--debug-clear writes the in-the-clear check to the witness side alone: it needs --witness")
    if a.witness is not None and Path(a.out).resolve() in a.witness.resolve().parents:
        raise SystemExit("--witness is under --out, which the chain releases: put it on the private side")
    with sides(a.witness):
        if a.witness is None:
            log("no --witness asked for: what the public summary leaves out is not written (witness lines, stderr and "
                "every child's output)")
        try:
            return prove(a)
        except BaseException as e:  # noqa: BLE001 - any failure's text, an interrupt's too, stays off stdout
            if isinstance(e, SystemExit) and e.code in (0, None):
                return 0
            traceback.print_exc()
            log(f"failed: {type(e).__name__}")
            return e.code if isinstance(e, SystemExit) and isinstance(e.code, int) else 1


if __name__ == "__main__":
    raise SystemExit(main())
