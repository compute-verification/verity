"""N2's advised unit proved in zero knowledge and decided on the proof: each link-window of an advised link
(`service.LinkWindows(advised=…)`, its unit `NetworkAdvisedLinkWindow_v2`: both ends' head checks and the advised
counts on the counts those heads hold) committed, registered on the link's own stream, proved by C-Flock `prove --zk`
behind the Lean firewall, delivered, and decided by N2's verifier of record with the link's advice
(`LinkWindows.outcome(advice=…)`), through `n2_zk`'s statement and rounds (`n2_zk.stager`, `stage`, `member`).

    python benchmarks/network_traces/n2_advised_zk.py prove --bin flock-circuit --lean flock-verify --out DIR
      [--seed 7] [--jobs 2] [--debug-clear] [--dry]

* **The link** (`simulated`). The advised tests' simulated vLLM serving (`advised_traffic.serve`: varied prompt and
  output lengths, chunked prefill, prefix-cache hits, a preemption resumed from a hit, the async row past a finish, a
  step's rows out of request order, frames carried across a window and a response released in two windows) at
  `SHAPE`: T = 512 buckets on both ends, the default-budget root's size class (`advised.circuit-check.py`: T = 512,
  R_in = 4, salted), egress R = 3, four windows and a burst of requests across each of the 0/1 and 2/3 boundaries, the
  certifier's own frame in every window. Its records are certified under a fresh LMS key and certifier session, and
  every advice carries the link's certified prompt-length tie.
* **Members.** One stream each, over the link's four windows: ``honest``, the session's advice and records;
  ``moved``, one kept output's release bucket one later (`advised_traffic.moved`); ``short``, one request's output
  length one short (`advised_traffic.with_output_len`); ``extra``, one frame more handed to the egress certifier in its
  first busy bucket, the egress records certified anew. The advice is public and the heads are witness: moved and
  short change the view the verifier computes, extra the egress heads the prover commits. Each negative is the first
  of its kind that `advised.view` takes in every window, so the unit takes every window's instance, and that the
  advised counts refuse in some window: the proof's verdict bit refuses it, with both head checks passing.
* **The statement and the rounds** are `n2_zk`'s: one circuit, `LinkWindows.proof_unit` lowered once (each gate the
  Definition its pinned digest names), and each link-window a statement of one instance on rows ``pub`` and ``wit``
  (`LinkWindows.layout`; ``pub`` ends with the window's view of the advice, `service.ADVICE`) and the output row
  ``o0``, the verdict and deadline bits. A round registers, selects, proves live behind the firewall, delivers, and
  decides on the proof: the verifier's own Lean run, the ``pub`` row opened only to its own copy (its own view of the
  public advice) and the ``o0`` row read. Per window: prove (the session's wall time, the prover's ``prove_total_s``,
  proof bytes) and verify (the verifier of record's run, Lean included): the per-window price.
* **What the verifier receives** is `n2_zk.RECEIVED` per window and the advice (``advice.json``), scanned for every
  piece of both ends' heads, head salts, siblings and certifier salts but those the public circuit or a window's
  ``pub`` row holds: on an advised link an accepted window's egress counts are its view's public ``released``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
from verity.core.protocols.certifier.certificate import Direction, Issuer, parameters_digest  # noqa: E402
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


def link_windows(lw: LinkWindows, link: AD.LinkAdvice, pairs) -> SimpleNamespace:
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
        args = lw.arguments(opening, root, AT.SYNCS, link)
        if args is None:
            raise SystemExit(f"window {w}: the unit takes no instance of a head or of the advice's view")
        word = lw.outputs(opening, root, AT.SYNCS, link)
        fails = {e: [k for k, ok in HC.reference_parts(i).items() if not ok] for e, i in zip(ENDS, insts)}
        fails["advised"] = [] if lw.advised_check(opening, AT.SYNCS, link) else ["counts"]
        others = lw.certified(w, certificates) and AT.SYNCS[w] in lw.sync_set
        out.append(SimpleNamespace(window=w, opening=opening, root=root, insts=insts, args=args, word=word,
                                   certificates=certificates, failed=fails, expect=int(word == 0b11 and others),
                                   statement=None))
    return SimpleNamespace(lw=lw, plan={"windows": len(pairs)}, syncs=AT.SYNCS, windows=out, issuer_refused=[],
                           advice=link)


def failed(x) -> list[str]:
    """The window's failed parts: an end's head-check part as ``<end>/<part>``, and ``advised``."""
    return [f"{end}/{k}" for end in ENDS for k in x.failed[end]] + (["advised"] if x.failed["advised"] else [])


def expected(name: str, fails: list[list[str]]) -> bool:
    """Whether each window's failed parts are the member's: none when honest; else the advised counts alone, in at
    least one window."""
    if name == "honest":
        return not any(fails)
    return all(f in ([], ["advised"]) for f in fails) and ["advised"] in fails


# --- the proofs -------------------------------------------------------------------------------------------------------

def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    sim = simulated(a.seed)
    lw = sim.lw
    rounds = {name: link_windows(lw, link, pairs) for name, (link, pairs, _) in sim.members.items()}
    summary: dict = {"link": {"name": lw.link, "windows": WINDOWS, "unit": lw.unit.id, "advised": lw.advised_key(),
                              "keys": {e: lw.head_key(d) for d, e in enumerate(ENDS)},
                              "program": lw.partition["program"], "partition": P.digest(lw.partition),
                              "gates": [g.id for g in lw.gates()], "pinned": all(HC.pinned(g) for g in lw.gates()),
                              "bits": {p: lw.bits(p) for p in SVC.PORTS}, "egress": asdict(SHAPE.egress),
                              "ingress": asdict(SHAPE.ingress)},
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
        res = Z.member(M, a, sg, name, rd, out / name, a.port + 100 * k)
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
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7881)
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
