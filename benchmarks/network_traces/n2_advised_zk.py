"""N2's advised unit proved in zero knowledge and decided on the proof: each link-window of an advised link
(`service.LinkWindows(advised=…)`, its unit `NetworkAdvisedLinkWindow_v2`: both ends' head checks and the advised
counts on the counts those heads hold) committed, registered on the link's own stream, proved by C-Flock `prove --zk`
behind the Lean firewall, delivered, and decided by N2's verifier of record with the link's advice
(`LinkWindows.outcome(advice=…)`), through `n2_zk`'s statement and rounds (`n2_zk.stager`, `stage`, `member`).

    python benchmarks/network_traces/n2_advised_zk.py prove --bin flock-circuit --lean flock-verify --out DIR
      [--seed 7 | --e2e RUN [--windows N]] [--members honest,moved,...] [--jobs 2] [--debug-clear] [--dry]

* **The link** (`simulated`). The advised tests' simulated vLLM serving (`advised_traffic.serve`: varied prompt and
  output lengths, chunked prefill, prefix-cache hits, a preemption resumed from a hit, the async row past a finish, a
  step's rows out of request order, frames carried across a window and a response released in two windows) at
  `SHAPE`: T = 512 buckets on both ends, the default-budget root's size class (`advised.circuit-check.py`: T = 512,
  R_in = 4, salted), egress R = 3, four windows and a burst of requests across each of the 0/1 and 2/3 boundaries, the
  certifier's own frame in every window. Its records are certified under a fresh LMS key and certifier session. The
  tie is IoTieRequest's verdict, from the verifier's own tie source (`service.LinkWindows.ties`), which the proof
  service doesn't have yet, so the verifier of record refuses every advised link (`advised`, fail closed); here the
  link's tie source is the tests' stand-in (`advised_traffic.StandInTie`) on the link's certified prompts, the
  verifier's own and never the prover's, so this prices the unit and its proof, not IoTieRequest's. Its clock
  source is the verifier of record's: the certifier signs each window's sync (`Issuer.clock`), and the link takes
  those clock certificates (`service.LinkWindows.clock_certificates`, `advised.SignedClocks`).
* **A served run's link** (`e2e`, ``--e2e RUN``): network-e2e's output directory of a run on the advice profile
  (``report.json``, the honest rehearsal's records under ``rehearsal/honest``, the endpoint's advice under
  ``endpoint/advice``), N2's link of its honest rehearsal (``served``: T = 60 on both ends, the egress allocation the
  calibration chose, R_in = 1, the certifier's salt heading every window) under the run's own certifier key, syncs and
  id-request framing (`active.id_request_frames`). Its advice is the Commit windows' I1 advice that the report bound to
  the run and rebuilt from its certified I/O, joined into one (`joined`), its requests in the certified arrival order,
  each kept output released in the bucket the certified I/O says its frame went out, and ``received`` the certified
  ingress heads' request frames; the tie the same stand-in, on the certified prompts. Its clock source
  (`served_clocks`) is the run's certifier's clock certificates (``clocks.jsonl``), the verifier of record's. A run
  that predates them falls back to `CertifierClocks`, the syncs its certifier's proxy logged (``proxy.json.gz``),
  unsigned and labelled a stand-in. It is never the plan the prover declared. A run with neither file is refused, and
  so is one whose clock source gives fewer windows' syncs than are proved. The link's first ``--windows`` windows are
  proved, by default every one.
* **The tie** on either link is the stand-in, and the summary says so (``tie``): IoTieRequest is pinned at one size
  class, `TIE_PINNED`, and neither link's ingress is in it. The summary's ``clocks`` names the clock source.
* **Members.** One stream each, over the link's windows (``--members``: some of them): ``honest``, the session's
  advice and records; ``moved``, one kept output's release bucket one later (`advised_traffic.moved`); ``short``, one
  request's output length one short (`advised_traffic.with_output_len`); on the simulated link ``extra``, one frame
  more handed to the egress certifier in its first busy bucket, the egress records certified anew; and on the served
  link ``tampered``, window 0's egress head with one count more and its certificate unchanged (`n2_zk.one_count`),
  since the run's certifier key isn't the driver's. The advice is public and the heads are witness: moved and short
  change the view the verifier computes, extra and tampered the egress heads the prover commits. Each advice negative
  and extra is the first of its kind that `advised.view` takes in every window, so the unit takes every window's
  instance, and that the advised counts refuse in some window: the proof's verdict bit refuses it, with both head
  checks passing. Tampered's window 0 fails the egress head check's signed digest.
* **The statement and the rounds** are `n2_zk`'s: one circuit, `LinkWindows.proof_unit` lowered once (each gate the
  Definition its pinned digest names), and each link-window a statement of one instance on rows ``pub`` and ``wit``
  (`LinkWindows.layout`; ``pub`` ends with the window's view of the advice, `service.ADVICE`) and the output row
  ``o0``, the verdict and deadline bits. A round registers, selects, proves live behind the firewall, delivers, and
  decides on the proof: the verifier's own Lean run, the ``pub`` row opened only to its own copy (its own view of the
  public advice) and the ``o0`` row read. Per window: prove (the session's wall time, the prover's ``prove_total_s``,
  proof bytes) and verify (the verifier of record's run, Lean included): the per-window price.
* **What the verifier receives** is `n2_zk.RECEIVED` per window and the advice (``advice.json``, with no tie),
  scanned for every piece of both ends' heads, head salts, siblings and certifier salts but those the public circuit
  or a window's ``pub`` row holds: on an advised link an accepted window's egress counts are its view's public
  ``released``.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TRAFFIC = HERE.parents[1] / "verity" / "core" / "protocols" / "certifier" / "tests" / "advised_traffic.py"
sys.path.insert(0, str(HERE))

import certifier_serve as cs  # noqa: E402
import n2_zk as Z  # noqa: E402
from n2_zk import ENDS  # noqa: E402
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC  # noqa: E402
from verity.core.protocols.certifier import service as SVC  # noqa: E402
from verity.core.protocols.certifier.active import id_request_frames  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, parameters_digest  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402
from verity.core.protocols.certifier.grid import GridParams  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402


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
E2E_MEMBERS = ("honest", "moved", "short", "tampered")
#: IoTieRequest's one pinned size class (B, T, R, OWN): the only one whose ingress tie can be proved
TIE_PINNED = (512, 4, 2, 1)


def log(msg: str) -> None:
    print(f"N2ADV {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def builders() -> tuple:
    """`n2_zk.builders`, and the advised counts' traced parts given the builders their traces were taken from."""
    parts = ((AD.NetworkAdvisedOwn, AD._build_own), (AD.NetworkAdvisedBucket, AD._build_bucket),
             (AD.NetworkAdvisedEnd, AD._build_end))
    for d, build in parts:
        if getattr(d, "build", None) is None:
            d.build = build
    return Z.builders() + tuple(d for d, _ in parts)


# --- the link and its members -----------------------------------------------------------------------------------------

def advised_refs(lw: LinkWindows, link: AD.LinkAdvice, egress, ingress, syncs=AT.SYNCS) -> list[bool | None]:
    """The advised counts' reference on each window's (egress, ingress) counts and the verifier's view of `link`
    under `syncs` (`LinkWindows._view`, under its tie source); None where the view is refused."""
    out = []
    for w, (e, i) in enumerate(zip(egress, ingress)):
        v = lw._view(w, syncs, link)
        out.append(None if v is None else AD.reference(*lw.advised_key(), i, e, v.released, v.floor, v.received))
    return out


@dataclass(frozen=True)
class CertifierClocks:
    """A stand-in clock source (`advised.ClockSource`, `LinkWindows(clocks=…)`) for a served run that predates the
    certifier's clock certificates (``clocks.jsonl``). It holds the syncs the run's certifier ran its windows under, as
    its proxy logged them (``proxy.json.gz``'s ``syncs``, its shaper's), for `link` alone. It reads nothing the prover
    declares (the plan's ``syncs``), but nothing signs it, and it is read from the run's directory."""

    link: str
    ran: tuple[ClockSync, ...]

    @classmethod
    def of(cls, run: Path, link: str) -> CertifierClocks:
        f = run / "proxy.json.gz"
        if not f.is_file():
            raise SystemExit(f"{run}: neither clocks.jsonl, the certifier's clock certificates, nor proxy.json.gz, "
                             "its log of the syncs it ran, so no clock source")
        with gzip.open(f, "rt") as fh:
            return cls(link, tuple(ClockSync(d, Fraction(r)) for d, r in json.load(fh)["syncs"]))

    def syncs(self, link: str) -> tuple[ClockSync, ...]:
        return self.ran if link == self.link else ()


def served_clocks(run: Path, link: str) -> dict:
    """A served run's clock source, as `LinkWindows` takes it. It is the certifier's clock certificates
    (``clocks.jsonl``, `certifier_serve.read_clocks`), the verifier of record's, when the run has them. Otherwise it is
    `CertifierClocks`, the certifier's unsigned log, a stand-in."""
    if (run / "clocks.jsonl").is_file():
        return {"clock_certificates": tuple(cs.read_clocks(run / "clocks.jsonl"))}
    return {"clocks": CertifierClocks.of(run, link)}


def clocks_label(lw: LinkWindows, declared) -> dict:
    """The summary's ``clocks``: where the link's syncs come from, the syncs its clock source gives, window by window,
    and whether they are the ones declared."""
    if lw.clock_certificates is not None:
        by = {"source": "signed", "by": "the certifier's clock certificates (clocks.jsonl), advised.SignedClocks "
                                        "under the link's certifier key and session"}
    elif isinstance(lw.clocks, CertifierClocks):
        by = {"source": "stand-in", "by": "CertifierClocks: the certifier's proxy log (proxy.json.gz's syncs, its "
                                          "shaper's), unsigned, for a run that predates clocks.jsonl"}
    else:
        by = {"source": "stand-in", "by": type(lw.clocks).__name__}
    got = list(lw.clock_source.syncs(lw.link)) if lw.clock_source is not None else []
    return {**by, "syncs": [[s.delta, str(s.rho)] for s in got], "as_declared": got == list(declared)}


def standin_tie(ingress: GridParams, own: int) -> dict:
    """What the link's tie is: the stand-in, and why not the proved ingress tie."""
    key = (ingress.frame_bytes, ingress.buckets, ingress.allocation, own)
    return {"source": "stand-in", "by": "advised_traffic.StandInTie: IoTieRequest's reference in the clear on the "
                                        "certified prompt lengths, the verifier's own tie source",
            "ingress": key, "why": f"IoTieRequest is pinned only at (B, T, R, OWN) = {TIE_PINNED}, and this link's "
                                   f"ingress is {key}"}


def short(lw: LinkWindows, link: AD.LinkAdvice, egress, ingress, syncs, windows: int):
    """The first request of two kept outputs or more, whose KV no hit reads, with its output length one short, which
    every window's view takes and the advised counts refuse in one of the first `windows` windows: (its advice, what
    was changed), or None."""
    a = link.advice
    for j, q in enumerate(a.requests):
        if q.output_len < 2 or AT.feeds_a_hit(a, j):
            continue
        try:
            off = AT.with_output_len(link, j, -1)
        except ADV.MalformedAdvice:
            continue
        got = advised_refs(lw, off, egress, ingress, syncs)
        if None not in got and not all(got[:windows]):
            return off, {"request": j, "output_len": q.output_len, "to": q.output_len - 1}
    return None


def moved(lw: LinkWindows, link: AD.LinkAdvice, egress, ingress, syncs, below: int):
    """The first kept output, released before bucket `below`, whose release one bucket later every window's view
    takes and the advised counts refuse in some window: (its advice, what was changed), or None."""
    a = link.advice
    for k, b in enumerate(a.release):
        if b + 1 >= below:
            continue
        try:
            off = AT.moved(link, k, 1)
        except ADV.MalformedAdvice:
            continue
        got = advised_refs(lw, off, egress, ingress, syncs)
        if None not in got and not all(got):
            return off, {"output": k, "from": b, "to": b + 1}
    return None


def negatives(lw: LinkWindows, s, shape=SHAPE) -> dict[str, tuple[AD.LinkAdvice, list, dict]]:
    """The three negatives on session `s`, each (its advice, its egress records, what was changed): the first of its
    kind that every window's view takes and the advised counts refuse in some window."""
    link, a = s.link, s.link.advice

    def refused(off, egress=s.egress) -> bool:
        got = advised_refs(lw, off, AT.counts(egress), AT.counts(s.ingress))
        return None not in got and not all(got)

    out = {}
    m = moved(lw, link, AT.counts(s.egress), AT.counts(s.ingress), AT.SYNCS, shape.windows * shape.T)
    if m is not None:
        out["moved"] = (m[0], s.egress, m[1])
    m = short(lw, link, AT.counts(s.egress), AT.counts(s.ingress), AT.SYNCS, shape.windows)
    if m is not None:
        out["short"] = (m[0], s.egress, m[1])
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
            clocks = [issuer.clock(AT.LINK, w, sync) for w, sync in enumerate(AT.SYNCS[:shape.windows])]
            if None in clocks:
                raise SystemExit(f"the certifier refused to sign a window's sync: {issuer.refused}")
            lw = LinkWindows(AT.LINK, shape.egress, AT.SYNC_SET, certifier.public_key(), "n2-advised-zk", session,
                             ingress=shape.ingress, proof_deadline=cs.PROOF_DEADLINE_S,
                             advised=AD.AdvisedProfile(shape.windows, AT.FRAMING, own),
                             ties=AT.StandInTie(s.certified, AT.LINK, shape.windows), clock_certificates=tuple(clocks))
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
    return SimpleNamespace(lw=lw, session=s, members=members, traffic=traffic, syncs=AT.SYNCS, windows=shape.windows,
                           egress=shape.egress, ingress=shape.ingress, tie=standin_tie(shape.ingress, own),
                           clocks=clocks_label(lw, AT.SYNCS[:shape.windows]))


def joined(advices: list[ADV.Advice], frames: dict[str, list[int]]) -> ADV.Advice:
    """One advice of the link from its Commit windows' (`advices`, in order): their requests and steps in turn, each
    window's request indices after the windows' before it, and kept output λ of the request with id i released in
    bucket ``frames[i][λ]``."""
    requests, steps = [], []
    for a in advices:
        off = len(requests)
        requests += a.requests
        steps += [ADV.StepAdvice(tuple(replace(r, request=r.request + off,
                                               sources=tuple(replace(s, request=s.request + off) for s in r.sources))
                                       for r in st.rows)) for st in a.steps]
    probe = ADV.Advice(tuple(requests), tuple(steps), (0,) * sum(q.output_len for q in requests))
    release = tuple(frames[requests[o.request].id][o.index] for o in ADV.outputs(probe))
    return ADV.Advice(tuple(requests), tuple(steps), release)


def e2e(run: Path, windows: int = 0) -> SimpleNamespace:
    """network-e2e's run `run` on the advice profile as an advised link (the module docstring, "A served run's link"),
    its first `windows` link-windows (0: every one): the link and its members (`E2E_MEMBERS`) on its certified records,
    each (advice, [(egress, ingress)] per window, what was changed)."""
    honest = run / "rehearsal" / "honest"
    plan, report, dep = (json.loads((honest / f).read_text()) for f in ("plan.json", "calibration.json",
                                                                         "deployment.json"))
    if (plan.get("occupancy") or {}).get("kind") != "advice":
        raise SystemExit(f"{honest}: not a run on the advice profile")
    bits = json.loads((run / "report.json").read_text())["chain"]["bits"]
    bound = [c["window"] for c in bits["commit_windows"] if c["bound"] and c["rebuilt"]]
    if not bound or not bits["bound"] or not bits["rebuilt"]:
        raise SystemExit(f"{run}: the report binds no advice to the run and rebuilds none from its certified I/O")
    io = json.loads((honest / "io.json").read_text())["requests"]
    if any(len(r["frames"]) != len(r["output"]) for r in io):
        raise SystemExit("io.json's frames aren't one per kept output")
    frames = {r["user"]: [b for _, b in r["frames"]] for r in io}
    advice = joined([ADV.from_json(json.loads((run / "endpoint" / "advice" / f"w{w}" / "private" / "advice.json")
                                              .read_text())) for w in bound], frames)
    if [q.id for q in advice.requests] != [r["user"] for r in io] or \
            any(q.output_len != len(frames[q.id]) for q in advice.requests):
        raise SystemExit("the bound advice's requests aren't the certified ones in arrival order, each its outputs")
    setup = cs.ar.Setup(report)
    n, B, own = plan["windows"], setup.ingress_params.frame_bytes, 1
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    framing = tuple(id_request_frames(p, B) for p in range(1, plan["varied"]["prompt_tokens"][1] + 1))
    link_name = dep["links"]["n2"]
    lw = LinkWindows(link_name, setup.egress_params, setup.sync_set, cs.read_public(dep["certifier_key"]),
                     dep["run_id"], bytes.fromhex(dep["session"]), ingress=setup.ingress_params,
                     proof_deadline=cs.PROOF_DEADLINE_S, advised=AD.AdvisedProfile(n, framing, own),
                     ties=AT.StandInTie(tuple(len(r["prompt"]) for r in io), link_name, n),
                     **served_clocks(honest, link_name))
    windows = windows or n
    if len(got := lw.clock_source.syncs(link_name)) < windows:
        raise SystemExit(f"{honest}: the link's clock source gives the syncs of its first {len(got)} windows, not "
                         f"{windows}, so the verifier refuses every window after")
    certified, _ = cs.read_certified(honest / "certified.jsonl")
    pairs = [(certified[(Direction.EGRESS, w)], certified[(Direction.INGRESS, w)]) for w in range(n)]
    egress, ingress = ([list(p[d].head.counts) for p in pairs] for d in (0, 1))
    link = AD.LinkAdvice(advice, tuple(sum(c) - own for c in ingress))
    members = {"honest": (link, pairs[:windows], {})}
    m = moved(lw, link, egress, ingress, syncs, windows * setup.egress_params.buckets)
    if m is None:
        raise SystemExit(f"the run's first {windows} windows have no moved release the views take and the advised "
                         "counts refuse")
    members["moved"] = (m[0], pairs[:windows], m[1])
    m = short(lw, link, egress, ingress, syncs, windows)
    if m is None:
        raise SystemExit(f"the run's first {windows} windows have no output length one short the views take and the "
                         "advised counts refuse")
    members["short"] = (m[0], pairs[:windows], m[1])
    members["tampered"] = (link, [Z.one_count(lw, w, p) for w, p in enumerate(pairs[:windows])],
                           {"window": 0, "egress": "one count more, its certificate unchanged"})
    traffic = {"run": str(run), "commit_windows": bound, "requests": len(advice.requests),
               "outputs": len(advice.release), "steps": len(advice.steps), "v0_bits": ADV.total_bits(advice),
               "received": list(link.received), "own": own, "max_released": max(max(c) for c in egress)}
    return SimpleNamespace(lw=lw, session=None, members=members, traffic=traffic, syncs=syncs, windows=windows,
                           egress=setup.egress_params, ingress=setup.ingress_params,
                           tie=standin_tie(setup.ingress_params, own), clocks=clocks_label(lw, syncs))


def link_windows(lw: LinkWindows, link: AD.LinkAdvice, pairs, syncs=AT.SYNCS) -> SimpleNamespace:
    """Every link-window of the member as the prover holds it before it registers, as `n2_zk.link_windows` holds a
    run's: its opening (fresh salts), the root it commits, the unit's arguments under that root on the advice
    (`LinkWindows.arguments`), each end's head check's failed parts and the advised counts' (``advised``,
    `advised_check`), the unit's output word by the gates' references (`LinkWindows.outputs`) and the certificates;
    ``advice`` the link's, which the verifier of record reads. Every window's unit takes an instance (`negatives`)."""
    out = []
    for w, pair in enumerate(pairs):
        certificates = tuple(r.certificate for r in pair)
        opening = lw.commit(w, *pair)
        root = bytes.fromhex(lw.root(opening))
        insts = lw.head_checks(opening, root)
        args = lw.arguments(opening, root, syncs, link)
        if args is None:
            raise SystemExit(f"window {w}: the unit takes no instance of a head or of the advice's view")
        word = lw.outputs(opening, root, syncs, link)
        fails = {e: [k for k, ok in HC.reference_parts(i).items() if not ok] for e, i in zip(ENDS, insts)}
        fails["advised"] = [] if lw.advised_check(opening, syncs, link) else ["counts"]
        others = lw.certified(w, certificates) and syncs[w] in lw.sync_set
        out.append(SimpleNamespace(window=w, opening=opening, root=root, insts=insts, args=args, word=word,
                                   certificates=certificates, failed=fails, expect=int(word == 0b11 and others),
                                   statement=None))
    return SimpleNamespace(lw=lw, plan={"windows": len(pairs)}, syncs=syncs, windows=out, issuer_refused=[],
                           advice=link)


def failed(x) -> list[str]:
    """The window's failed parts: an end's head-check part as ``<end>/<part>``, and ``advised``."""
    return [f"{end}/{k}" for end in ENDS for k in x.failed[end]] + (["advised"] if x.failed["advised"] else [])


def expected(name: str, fails: list[list[str]]) -> bool:
    """Whether each window's failed parts are the member's: none when honest; tampered, window 0's egress head its
    signed digest (and the advised counts, which read the count it changed) and no other window; else the advised
    counts alone, in at least one window."""
    if name == "honest":
        return not any(fails)
    if name == "tampered":
        return "egress/signed" in fails[0] and set(fails[0]) <= {"egress/signed", "advised"} and not any(fails[1:])
    return all(f in ([], ["advised"]) for f in fails) and ["advised"] in fails


# --- the proofs -------------------------------------------------------------------------------------------------------

def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    sim = e2e(Path(a.e2e), a.windows) if a.e2e else simulated(a.seed)
    lw = sim.lw
    chosen = a.members.split(",") if a.members else list(sim.members)
    if not set(chosen) <= set(sim.members):
        raise SystemExit(f"--members {a.members}: the link's members are {list(sim.members)}")
    sim.members = {k: v for k, v in sim.members.items() if k in chosen}
    rounds = {name: link_windows(lw, link, pairs, sim.syncs) for name, (link, pairs, _) in sim.members.items()}
    log(f"tie: {sim.tie}; clocks: {sim.clocks}")
    summary: dict = {"tie": sim.tie, "clocks": sim.clocks,
                     "link": {"name": lw.link, "windows": sim.windows, "unit": lw.unit.id, "advised": lw.advised_key(),
                              "keys": {e: lw.head_key(d) for d, e in enumerate(ENDS)},
                              "program": lw.partition["program"], "partition": P.digest(lw.partition),
                              "gates": [g.id for g in lw.gates()], "pinned": all(HC.pinned(g) for g in lw.gates()),
                              "bits": {p: lw.bits(p) for p in SVC.PORTS}, "egress": asdict(sim.egress),
                              "ingress": asdict(sim.ingress)},
                     "traffic": sim.traffic, "changed": {k: w for k, (_, _, w) in sim.members.items()},
                     "advice": {name: AD.digest(rd.advice) for name, rd in rounds.items()},
                     "debug_clear": a.debug_clear, "received": {**Z.RECEIVED, "advice.json": "the link's advice"}}
    summary["instances_s"] = round(time.time() - t, 1)
    for name, rd in rounds.items():
        fails = [failed(x) for x in rd.windows]
        summary.setdefault("members", {})[name] = {
            "reference": [{"window": x.window, "word": x.word, "expect": x.expect, "failed": f}
                          for x, f in zip(rd.windows, fails)]}
        log(f"{name}: words {[x.word for x in rd.windows]}, expected verdicts {[x.expect for x in rd.windows]}, "
            f"failed parts {fails}")
        if not expected(name, fails):
            raise SystemExit(f"{name}: the failed parts {fails} are not the {name} member's")
        if [x.expect for x in rd.windows] != [int(not f) for f in fails]:
            raise SystemExit(f"{name}: the expected verdicts aren't the references'")
    if not summary["link"]["pinned"]:
        raise SystemExit(f"a gate names no pinned circuit digest: {summary['link']['gates']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0

    M = Z._one_stage()
    M.PF.executable = lambda: Path(a.lean)
    Z._record_lean_runs(M.PF)
    builders()
    sg = Z.stager(M, lw, out / "stager")
    summary["stager"] = {"unit": lw.proof_unit.id, "lower_s": sg.lower_s, "k_log": sg.k_log, "pin": sg.st.pin,
                         "circuit_sha512": sg.circuit_sha512, "bits": {p: lw.bits(p) for p in SVC.PORTS},
                         "ands": sg.st.low.layout.counts["ands"]}
    log(f"stager: {lw.proof_unit.id}, {sg.lower_s} s, k_log {sg.k_log}, {summary['stager']['ands']} ANDs, "
        f"pin {sg.st.pin[:16]}")
    summary["staging"] = Z.stage(M, sg, rounds, os.urandom(32), out)
    log(f"staged: {summary['staging']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))

    def go(k: int, name: str) -> tuple[str, dict]:
        rd = rounds[name]
        res = Z.member(M, a, sg, name, rd, out / name, a.port + 100 * k if a.port else 0)
        (out / name / "verifier" / "advice.json").write_text(json.dumps(
            {"advice": ADV.to_json(rd.advice.advice), "received": list(rd.advice.received),
             "digest": AD.digest(rd.advice)}, sort_keys=True))
        res["received"] = Z._member_scan(M, sg, rd, out / name)
        return name, res

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        done = dict(pool.map(lambda kv: go(*kv), enumerate(rounds)))
    ok = True
    for name, res in done.items():
        rec = res["received"]
        good = (all(r["as_expected"] for r in res["windows"]) and res["network"]["accepted"] == (name == "honest")
                and (res["network"]["accepted"] is False or res["network"]["advice"] == AD.digest(rounds[name].advice))
                and not rec["verifier_found"] and rec["control"]["holds_all"])
        ok &= good
        summary["members"][name].update(res, ok=good)
        log(f"member {name}: network {'accept' if res['network']['accepted'] else 'refuse'} (refused windows "
            f"{res['network']['refused']}), windows {[int(r['accepted']) for r in res['windows']]}, verifier received "
            f"{rec['verifier_bytes']} bytes in {rec['files']} files, witness pieces found there: "
            f"{rec['verifier_found'] or 'none'} (of {rec['blobs']} blobs; {len(rec['public_pieces'])} pieces the "
            f"public circuit or rows hold left out), control {rec['control']['blobs_found']}; "
            f"{'as expected' if good else 'UNEXPECTED'}")
    summary["windows"] = {name: [int(r["accepted"]) for r in res["windows"]] for name, res in done.items()}
    costs = [r for res in done.values() for r in res["windows"]]
    summary["per_window"] = {
        "of": f"each of the {len(costs)} link-windows, one unit proved alone and decided by the verifier of record",
        **{k: [r["prove"][k] for r in costs] for k in ("session_s", "prove_total_s", "e2e_s", "proof_bytes",
                                                       "max_child_rss_gb")},
        "verify_s": [r["verify_s"] for r in costs]}
    log(f"windows {summary['windows']}; per window: {summary['per_window']}")
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="n2_advised_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="each advised link-window's unit proved --zk and decided by N2's verifier of "
                                     "record")
    p.add_argument("--seed", type=int, default=SEED, help="the advised tests' session")
    p.add_argument("--e2e", help="network-e2e's output directory of a run on the advice profile: its honest "
                                 "rehearsal's link, in place of the simulated one")
    p.add_argument("--windows", type=int, default=0, help="with --e2e: the link's first windows proved (0, the "
                                                          "default: every one)")
    p.add_argument("--members", help="comma-separated: the members proved (default every member of the link)")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=0, help="member k's window j listens on PORT + 100k + j; 0 (the "
                                                       "default): a free port for each session")
    p.add_argument("--jobs", type=int, default=2, help="members proved at once")
    p.add_argument("--debug-clear", action="store_true",
                   help="each verdict also carries the in-the-clear check on the opening, which decides nothing")
    p.add_argument("--dry", action="store_true", help="the instances and their references only: nothing lowered or "
                                                      "proved")
    a = ap.parse_args(argv)
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
