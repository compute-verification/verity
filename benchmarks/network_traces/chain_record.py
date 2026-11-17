"""chain-record: the network chain's run of record, from the gate's fetched tarballs to one report
(`chain_record.sh` runs it on a CPU machine). Each step reads what the step before it kept; none drives an endpoint.

    python benchmarks/network_traces/chain_record.py custody --gates DIR --out FILE
    python benchmarks/network_traces/chain_record.py certs --warden-run DIR [--deployed] --out FILE [--witness FILE]
    python benchmarks/network_traces/chain_record.py endpoint --warden-run DIR --endpoint DIR --eos IDS --out FILE
    python benchmarks/network_traces/chain_record.py report --custody FILE --certs FILE [--endpoint REPORT]
      [--n2 SUMMARY ...] [--scan SUMMARY] [--tie SUMMARY] [--job FILE] [--audit WINDOW ...]
      [--period-answer FILE [--period-case FILE]] --out DIR
    python benchmarks/network_traces/chain_record.py leak --warden-run DIR [--scan SUMMARY] --out FILE

* **custody**: each gate run under ``--gates`` (a directory holding the gate's ``end.json``, as the gate's ``fetch``
  writes it) holds the files its ``end.json`` names, each with the SHA-512 it records. In the clear, of public
  files; it proves nothing about what the gate ran.
* **certs** (``--warden-run``): the network attester's own offline check rerun on its witness, and the public
  ``certificates.jsonl`` held to the certified lines (`network_attester_chain.checked`). ``--deployed`` refuses a key that
  isn't LMS at `lms.DEPLOYED`. Which execution phases were certified or refused, and how many problems the check
  found, are the step's private side (`network_attester_chain.WITNESS_KEYS`), written to ``--witness`` and never to ``--out``.
* **report**: one JSON and one Markdown report, per step its verdict, its run ids, its ε at δ and what it computed in
  the clear, and the period's terms for `NetTiming.Guarantees.MonitoringPeriodLeakAccepted` summed by
  `monitoringPeriodLeak_le_sum`. ``--endpoint`` is network-e2e's ``--no-prove`` report on the run and the endpoint's
  records: the endpoint's advice bound to the run and the I/O tie, in the clear, required on the advice profile. N2's
  summary is `n2_zk.py prove`'s (Selection All: every execution phase's one unit proved) or a sampled sample's
  (``CR_N2_CMD``), read by the same fields; on the advice profile, whose egress counts no N2 unit holds, N2 isn't
  run. Scan's is `scan_zk.py prove`'s, a stand-in on its own synthetic network attester run until its ``--warden-run``
  takes the gate's network attester rows, accepted only when every round's verdict is the one scan_zk expects and no
  witness piece reached the verifier (`scan_step`). A scan whose honest round's registration names a tied value (its
  ``rests_on_period_check``, the slot entries ``prog``) is ``accepted on the period check`` (`RESTS`), never a bare
  accept: its acceptance rests on the network period check, which refuses scan_zk's stand-in block, and so does the
  report's. A step on a stand-in (N2 on ``--job``'s ``stand_in``, scan without a network attester
  run) never makes the report accepted. Without ``--n2`` the report is the scan's part of the record alone (`SCAN_ONLY`,
  its ``scope``), and its verdict is ``scan accepted``, ``scan accepted on the period check`` or ``scan refused``: N2
  isn't required, and the rest is held as before. On the certifier-only path (``--job``'s ``path``,
  `CERTIFIER_ONLY`) there are no network attester rows: the scan isn't
  run or required, and the report's ``stubbed`` says what the path stands on (`STUBBED`, `stubbed`). Its negatives are
  N2's members, the rehearsal's own check on each run it must refuse, and the I/O tie's verdict on each tampered run
  that check lets through. d1's K (`pooled_k`) counts the served units' compute audits (``--audit``: each window's sample
  and free inputs, `free_inputs`) and the network's own terms (`network_terms`), at each δ. The report writes
  no block record's certifier entry, and its ``block`` says why (`block_entry`): with ``--period-answer``
  (certifier/v0's ``network-period check`` answer on the period, a ``certifier.answer.json``, its case
  ``certifier.case.json`` beside it or ``--period-case``) held to its case's commitment (`period_binding`), the
  reason is `ROWS_FORM`.
* **The network attester path** (``--job``'s ``path``, `NETWORK_CERTIFIER`): every step reads an active network attester's own
  run (the gate's ``warden-rehearse``, or `network_attester_serve.py`'s directory; `network_attester_chain`) and no certifier rehearsal's,
  whose input is refused by name (`network_attester_chain.PROXY`). ``certs --warden-run`` reruns the network attester's own
  offline check on its witness and holds the public ``certificates.jsonl`` to the certified lines
  (`network_attester_chain.certificates`; no clock certificate: the network attester signs none). That check requires the
  session's certified start, so a gate run made by a device that certifies no start is refused here. ``endpoint`` binds
  the endpoint's own advice to the certified I/O read off the network attester's cells (`network_attester_endpoint`); where the
  endpoint's records stay on the served node, that node runs it (`served_endpoint`, `served_session.py endpoint`) and
  the chain reads its report. N2 is `n2_zk.py prove --warden-run`'s, on the network attester link; the I/O tie is
  `n2_tie_zk.py prove --class warden`'s, proved (``--tie``, `tie_step`); the scan is `scan_zk.py prove
  --warden-run`'s. Each is held to the network attester run the certificates step read (`on_network_attester`), every one is
  required, and the report names what the path stands on (`NETWORK_CERTIFIER_STUBBED`, or `NETWORK_CERTIFIER_SERVED_STUBBED` on the served node's
  endpoint report).
* **leak**: K per monitoring period (week, month, year) on the certifier's gate run, pooled at δ = 1% and 10⁻⁶ and in the
  per-period form, each beside its share of one copy of the weights (`leak`), computed by `leak_bounds`' scaling by
  links: one link at the run's rate, and as many of it summed as the census's reference datacenter and the world
  have machines. The network attester's term at a scope's own load is `verity bench certifier_term`.
"""
from __future__ import annotations

import argparse
import decimal
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(Path(__file__).resolve().parents[2]))  # the root, for benchmarks.network_traces

SCHEMA = "verity-network-chain/record-report/v1"
DELTAS = (0.5, 2.0 ** -20)
ADVICE = "advice"  # a rehearsal's profile whose egress counts derive from the endpoint's advice

CLEAR = {
    "custody": "the SHA-512 of each fetched file against the gate's end.json (public files)",
    "endpoint": "at the auditor, on the endpoint's records and the run's certified I/O: each Commit window's advice "
                "opened to its registered root and rebuilt from the certified I/O, and the I/O tie (IoTieRequest's "
                "reference, network-e2e's stand-ins for the network certifier's rows and the Commit's port) on the honest run",
    "n2": "at the verifier: the registration records, the sample, the certificates' public fields, the clock syncs, the "
          "proof's pub row opened to the verifier's own copy and the o0 output word; never a record, head, leaf "
          "salt, sibling or the certifier's salt (n2_zk's RECEIVED, scanned for every piece)",
    "scan": "at the verifier: each unit's o0 output opened from the salts the prover sends; the I/O tie "
            "(IoTieRequest) stays in the clear, outside the proofs",
    "lean": "nothing: each proof is decided by the verifier's own Lean `verity-verify-proof verify --zk` run",
    "warden_certificates": "at the auditor, on the network certifier's witness: the network certifier's own offline check "
                           "(network_attester_serve.check: each record's certificate and its rows' registration under the "
                           "network certifier's key and session, the audit and the client's view), and the public "
                           "certificates.jsonl held to the certified lines",
    "warden_endpoint": "at the auditor, on the endpoint's records and the network certifier's certified I/O (network_attester_chain.io, off "
                       "the records' cells): each Commit window's advice opened to its registered root and rebuilt "
                       "from the certified I/O",
    "warden_endpoint_served": "on the served node, the prover's, whose records hold request values and stay there: "
                              "each Commit window's advice opened to its registered root and rebuilt from the network certifier's "
                              "certified I/O sent to it (served_session.py endpoint); the chain reads its report",
    "tie": "at the verifier: each ingress bucket unit's pub row opened to the verifier's own copy (the advice's prompt "
           "lengths and the network certifier's release) and its o0 verdict bit; never a cell or its salt (n2_tie_zk's RECEIVED, "
           "scanned for every piece)",
}
#: the certifier-only path (``--job``'s ``path``): the records are the certifier rehearsal's alone
CERTIFIER_ONLY = "certifier-only"
#: a report given no N2 summary (``report`` without ``--n2``, `chain_record.sh`'s ``CR_SCAN_ONLY=1``): the scan's
#: part of the record, whose N2 runs in jobs of its own; its verdict is ``scan accepted`` (or on the period check,
#: `RESTS`) or ``scan refused``, never the whole chain's ``accepted``
SCAN_ONLY = "scan only, N2 not given"
#: the scan step's verdict, and the report's, when the scan's honest round accepted with tied values R8 let through
#: (its registration verdict's ``rests_on_period_check``): no bare accept, since its acceptance rests on a period check;
#: nor a K, since `block_entry` writes none (`ROWS_FORM`)
RESTS = "accepted on the period check"
RESTS_WHY = ("the honest round's registration names {tied} tied: its acceptance rests on the network period check "
             "(NetworkVerifier.Period's wired), which ties their rows to the audit that registered them and refuses "
             "scan_zk's stand-in block (period_case.UNWIRED)")
#: what a report on the certifier-only path stands on that isn't the chain's own
STUBBED = (
    "the records come from the certifier-only path (a rehearsal's proxy in front of the endpoint), not the "
    "network certifier's own cells as one device, so there are no network certifier rows and the scan isn't run",
    "the output tie (IoTieRequest) is computed in the clear at the auditor, outside the proofs",
    "the weights' load isn't certified: nothing here shows the endpoint served the registered weights")
#: the network attester path (``--job``'s ``path``): every step reads an active network attester's own run
#: (`network_attester_chain`)
NETWORK_CERTIFIER = "warden"
#: what a report on the network attester path stands on that isn't the chain's own
NETWORK_CERTIFIER_STUBBED = (
    "the endpoint's advice is bound to the network certifier's certified I/O and rebuilt from it in the clear at the auditor, "
    "and the output tie to the Commit's port isn't proved",
    "the weights' load isn't certified: nothing here shows the endpoint served the registered weights")
#: an endpoint report's ``at`` when its step ran on the served node (`served_session.py endpoint`)
SERVED_NODE = "the served node"
#: `NETWORK_CERTIFIER_STUBBED` when the endpoint's report is the served node's
NETWORK_CERTIFIER_SERVED_STUBBED = (
    "the endpoint's advice is bound to the network certifier's certified I/O and rebuilt from it in the clear on the served node, "
    "the prover's, so the chain's endpoint step is the prover's report of that check, and the output tie to the "
    "Commit's port isn't proved", NETWORK_CERTIFIER_STUBBED[1])
#: the rehearsal's strategies its own check accepts: token's changed bytes only the tie refuses
REHEARSAL_ACCEPTS = ("honest", "token")
#: the scan's kinds of round whose window is accepted (scan_zk.ROUNDS), one a round an execution phase: the toy
#: block's is proved like the honest one's
SCAN_ACCEPTS = ("honest", "block")
#: a proof block's outputs, at most 16 bits: an incorrect proof block's leak
OUT_BITS = 16
#: what `pooled_k`'s K rests on until its two per-window inputs are discharged
POOLED_CONDITION = ("NetTiming.Guarantees.PooledLeakAccepted is conditional: its per-window inputs hesc "
                     "(proofs' SampledKSLive) and hleak (the live SampledLeakAccepted) are not discharged")
#: why a report without ``--period-answer`` writes no block entry
NO_PERIOD_ANSWER = "no --period-answer: no certifier/v0 answer on the period"
#: why a report writes no block entry for any answer the period check gives
ROWS_FORM = ("the rows form's execution phases are the scan's registrations, which this chain does not hold, so no K is "
             "written")
#: certifier/v0's commitment digest: tagged_sha512(PERIOD_TAG, the commitment's bytes) (`NetworkCheck.Period`'s)
PERIOD_TAG = "verity-network-certifier/period/v0"


def _json(path) -> dict | None:
    p = Path(path) if path else None
    return json.loads(p.read_text()) if p and p.is_file() else None


def subset_bound(n: int, k: int, delta: float) -> int | None:
    """The most incorrect proof blocks an accepted uniform sample of k of n proof blocks hides except with probability
    δ: the integrity profile's count bound (`Stratified.count_bound`), the largest b with C(n − b, k) / C(n, k) ≥ δ.
    Selection All (k = n) is 0."""
    if n <= 0 or not 0 < k <= n:
        return None
    from verity.protocols.proofs.spot.verifier.profile import Stratified

    return Stratified.uniform(n, k).count_bound(delta)


def eps_at_delta(n: int, k: int, law: str) -> dict:
    return {"law": law, "units": n, "drawn": k,
            "incorrect_at_delta": {f"{d:.3g}": subset_bound(n, k, d) for d in DELTAS},
            "per_proof": "each proof's soundness error, ε_ks + ε_bind (KSLinOn's κ + √(γ·Pr B) + Pr[Bext]), "
                         "symbolic: Claims.SamplingGeneric.Compiled.KS is a hypothesis until "
                         "Flock.Guarantees.SampledKSLive"}


# --- custody ----------------------------------------------------------------------------------------------------------

def custody(gates: Path) -> dict:
    runs = []
    for end in sorted(gates.rglob("end.json")):
        root, doc = end.parent, json.loads(end.read_text())
        files = doc.get("files") or {}
        bad = {}
        for rel, want in files.items():
            f = root / "out" / rel
            got = hashlib.sha512(f.read_bytes()).hexdigest() if f.is_file() else None
            if got != want:
                bad[rel] = "missing" if got is None else "sha512 differs"
        # a gate's fetch names no peer and no lane in the run's request (`auditor.REQUEST_WITHHELD`); a copy kept
        # from a gate that still did loses the two here
        request = {k: v for k, v in (_json(root / "request.json") or {}).items() if k not in ("peer", "by")}
        summary = _json(root / "out" / "summary.json") or {}
        runs.append({"gate_run": root.name, "dir": str(root.relative_to(gates)), "ended": doc.get("ended"),
                     "files": len(files), "mismatches": bad,
                     "request": request, "lms_height": summary.get("lms_height"), "windows": summary.get("windows"),
                     "profile": summary.get("profile"),
                     "strategies": {s: (r or {}).get("verdict") for s, r in (doc.get("runs") or {}).items()}})
    ok = bool(runs) and all(r["files"] and not r["mismatches"] for r in runs)
    return {"verdict": "accepted" if ok else "refused", "gate_runs": runs, "in_the_clear": CLEAR["custody"]}


# --- certificates -----------------------------------------------------------------------------------------------------

# --- report -----------------------------------------------------------------------------------------------------------

def endpoint_step(e: dict | None) -> dict:
    """network-e2e's ``--no-prove`` report on the run and the endpoint's records: the endpoint's own advice bound to the
    run's certified I/O and rebuilt from it (`chain.served_advice`), and the I/O tie's verdict on the honest run
    (`chain.tie`), both in the clear. The tie decides a tampered run only where the rehearsal's own check let it
    through (no ``own_code`` in the report's ``negatives``: token), and must refuse it at exactly the execution phases
    its changes fall in (``ok``); a run the check refused (late, extra) is the rehearsal's negative, not the tie's."""
    if e is None:
        return {"verdict": "not run", "why": "no endpoint records (CR_ENDPOINT)", "in_the_clear": CLEAR["endpoint"]}
    chain = e.get("chain") or {}
    b, t = chain.get("bits") or {}, chain.get("tie") or {}
    honest = (t.get("runs") or {}).get("honest") or {}
    windows = b.get("commit_windows") or []
    advice = {"bound": b.get("bound"), "rebuilt": b.get("rebuilt"), "why": b.get("bound_why") or b.get("why"),
              "commit_windows": len(windows), "bound_windows": sum(bool(c.get("bound") and c.get("rebuilt"))
                                                                   for c in windows)}
    tie = {k: honest.get(k) for k in ("verdict", "ok", "codes", "requests", "outputs", "placed", "unplaced", "why")}
    ok = bool(advice["bound"] and advice["rebuilt"]) and honest.get("verdict") == "accepted"
    checked = {n for n, g in (e.get("negatives") or {}).items() if g.get("own_code")}
    negatives = {name: r.get("verdict") == "refused" and r.get("ok") is True
                 for name, r in (t.get("runs") or {}).items() if name != "honest" and name not in checked
                 and r.get("verdict") in ("accepted", "refused")}
    return {"verdict": "accepted" if ok else "refused", "advice": advice, "tie": tie | {"ran": t.get("ran"),
            "stand_ins": t.get("stand_ins")}, "negatives": negatives, "advice_bits": advice_bits(b),
            "in_the_clear": CLEAR["endpoint"]}


def advice_bits(b: dict) -> dict:
    """The advice's bits over the run's windows (`chain.bits`): v0's encoding, and the residual's (what the advice says
    beyond what the walk derives from the certified I/O), None unless the advice was rebuilt from it."""
    runs = [w for w in b.get("windows") or [] if w.get("in_run")]
    derived = [w.get("derived_bits") for w in runs]
    return {"v0_total": b.get("v0_total"), "T": b.get("T"),
            "derived": sum(derived) if runs and None not in derived and b.get("rebuilt") else None}


#: `verity-vllm advice`, as network-e2e runs it
ADVICE_CLI = [sys.executable, "-m", "verity_vllm.pipeline.cli", "advice"]


def network_attester_endpoint(run: Path, ep: Path, eos: list[int]) -> dict:
    """The endpoint's own advice for a network attester run (`served_endpoint`), on the certified I/O read off its
    cells (`network_attester_chain.io`)."""
    from benchmarks.network_traces import network_attester_chain as WC

    return served_endpoint(WC.deployment(run)[0], WC.io(run), ep, eos)


def served_endpoint(dep: dict, io: dict, ep: Path, eos: list[int]) -> dict:
    """The endpoint's own advice for the network attester run of ``dep`` (its ``deployment.json``), as network-e2e's
    `endpoint_advice` makes it for a certifier rehearsal's: `verity-vllm advice` over the endpoint's records ``ep``
    (network-e2e's ``chain.ENDPOINT``) into ``ep/advice`` at the network attester's bucket unless an index is there,
    then ``chain.served_advice`` on the certified I/O ``io`` (`network_attester_chain.io`'s) over the network attester's
    execution phases, with the model's EOS ids ``eos`` (a network attester run records none)."""
    import subprocess

    sys.path.insert(0, str(HERE.parent / "network_e2e"))
    import chain

    T = dep["params"]["window"]
    window = {"origin_unix": dep["origin_unix"], "windows": dep["windows"],
              "window_s": T * dep["params"]["bucket_ms"] / 1000}
    out = {"warden_run": dep["run_id"], "endpoint": str(ep), "io": {k: v for k, v in io.items() if k != "records"}}
    adv = ep / "advice"
    if not (adv / "windows.json").exists():
        p = subprocess.run([*ADVICE_CLI, str(ep), "--out", str(adv), "--tau", repr(window["window_s"] / T)],
                           capture_output=True, text=True, check=False)
        (ep / "advice.log").write_text(p.stdout + p.stderr)
        if not (adv / "windows.json").exists():
            return out | {"chain": {"bits": {"bound": False, "bound_why": f"verity-vllm advice made no index (exit "
                                             f"{p.returncode}: {(p.stdout + p.stderr)[-300:].strip()})"}}}
    try:
        info, bits = chain.served_advice(ep, adv, window, io["records"], T, eos)
    except (OSError, KeyError, ValueError) as err:
        return out | {"chain": {"bits": {"bound": False, "bound_why": f"the endpoint's advice: {err!r}"}}}
    return out | {"advice": info, "chain": {"bits": bits}}


def network_attester_endpoint_step(e: dict | None) -> dict:
    """`network_attester_endpoint`'s report: accepted when the network attester's certified I/O reads whole off its cells and the
    endpoint's own advice is bound to it and rebuilt from it, in the clear."""
    if e is None:
        return {"verdict": "not run", "why": "no endpoint records (CR_ENDPOINT)",
                "in_the_clear": CLEAR["warden_endpoint"]}
    b, io = (e.get("chain") or {}).get("bits") or {}, e.get("io") or {}
    windows = b.get("commit_windows") or []
    advice = {"bound": b.get("bound"), "rebuilt": b.get("rebuilt"), "why": b.get("bound_why") or b.get("why"),
              "commit_windows": len(windows), "bound_windows": sum(bool(c.get("bound") and c.get("rebuilt"))
                                                                   for c in windows)}
    why = (f"the network certifier's certified I/O doesn't read whole off its cells: {io.get('violations')}"
           if not io.get("accepted") else None if advice["bound"] and advice["rebuilt"]
           else advice["why"] or "the endpoint's advice isn't bound to the certified I/O or rebuilt from it")
    return {"verdict": "refused" if why else "accepted", "why": why, "warden_run": e.get("warden_run"), "io": io,
            "advice": advice, "advice_bits": advice_bits(b), "at": e.get("at"),
            "in_the_clear": CLEAR["warden_endpoint_served" if e.get("at") == SERVED_NODE else "warden_endpoint"]}


def tie_step(t: dict | None, job: dict | None = None) -> dict:
    """`n2_tie_zk.py prove --class warden`'s summary: accepted when its ``ok`` holds (every execution phase's verdict
    its reference's, the tie's on the link, no witness piece leaked), its honest member tied in every execution phase it
    proved and its forged member refused in one or more."""
    job = job or {}
    rc, why = (job.get("rc") or {}).get("tie"), (job.get("why") or {}).get("tie")
    if t is None or rc not in (None, "0") and not t.get("tie"):
        return {"verdict": "not run" if t is None and rc in (None, "2") else "failed", "why": why,
                "in_the_clear": CLEAR["tie"]}
    tie, link = t.get("tie") or {}, t.get("link") or {}
    honest, forged = tie.get("honest") or [], tie.get("forged") or []
    units = len(honest) * int(link.get("T") or 0)
    why = ("dry: nothing proved" if not tie else (why or "n2_tie_zk's ok is not true") if t.get("ok") is not True
           else None if honest and all(honest) and forged and not all(forged)
           else f"the tie's verdicts aren't honest's all tied and forged's refused: {tie}")
    return {"verdict": "dry" if not tie else "refused" if why else "accepted", "why": why, "ok": t.get("ok"),
            "warden_run": t.get("warden_run"), "class": t.get("class"), "unit": link.get("unit"), "tie": tie,
            "negatives": {"forged": bool(forged) and not all(forged)},
            "eps_at_delta": eps_at_delta(units, units, "all"), "in_the_clear": CLEAR["tie"]}


def on_network_attester(step: dict, doc: dict | None, run_id: str | None, what: str) -> dict:
    """A network-certifier-path step held to the network attester run the certificates step read (``run_id``): refused
    by name (`network_attester_chain.PROXY`) when its input is a certifier rehearsal's (it names no active network attester run),
    and when it is another network attester run's."""
    from benchmarks.network_traces.network_attester_chain import PROXY

    if doc is None or step.get("verdict") in ("not run", "not given"):
        return step
    got = doc.get("warden_run")
    why = (f"{what} names no network certifier run: {PROXY}" if not got
           else f"{what} is on network certifier run {got}, not the certificates step's {run_id}" if got != run_id else None)
    return step | {"verdict": "refused", "why": why} if why else step


def escaped(name: str, m: dict, s: dict) -> dict | None:
    """A negative member the sampled sample accepted because none of its incorrect windows (those its reference expects
    refused) was among the windows sampled: the sample's sampling error, which its ``wrong_units`` at δ bounds, not a
    failure of the run. None for honest, a member refused, or a period not sampled."""
    net = m.get("network") or {}
    units = (((s.get("sampled") or {}).get(name) or {}).get("draw") or {}).get("units") or (m.get("period") or {}).get(
        "units")
    bad = [r["window"] for r in m.get("reference") or [] if not r.get("expect")]
    if name == "honest" or not net.get("accepted") or units is None or set(bad) & set(units):
        return None
    return {"incorrect_windows": bad, "drawn": sorted(units)}


def unexpected(name: str, m: dict, escape: dict | None = None) -> list[str]:
    """What keeps an N2 driver's member from its ``ok``: the conditions `n2_zk` holds it to, but
    a negative's acceptance when it escaped the sample (`escaped`). A witness piece counts when the scan reads it
    ``leaked`` (`leak_scan.classify`); a ``structured`` find is listed, not counted, as the drivers decide."""
    wins, net, rec = m.get("windows") or [], m.get("network") or {}, m.get("received") or {}
    return ([f"windows {[w.get('window') for w in wins if not w.get('as_expected')]} not as expected"]
            * any(not w.get("as_expected") for w in wins)
            + [f"network {'accepted' if net.get('accepted') else 'refused'}"]
            * (net.get("accepted") != (name == "honest") and escape is None)
            + [f"witness pieces leaked to the verifier: {rec.get('leaked')}"] * bool(rec.get("leaked"))
            + [f"the control (the prover's own {(rec.get('control') or {}).get('file')}) doesn't hold all "
               f"{(rec.get('control') or {}).get('of')} pieces, so the scan isn't shown to find a leak"] * ((rec.get("control") or {}).get("holds_all") is False))


def n2_step(s: dict | None, certs_doc: dict, job: dict | None = None) -> dict:
    """N2's step from its public summary `s` alone: the timings and the scan's ``structured`` finds stay in the
    private side ``n2_zk.py prove --witness`` writes, and the report, which is released, holds none of them."""
    job = job or {}
    why, stand_in = (job.get("why") or {}).get("n2"), (job.get("stand_in") or {}).get("n2")
    rc = (job.get("rc") or {}).get("n2")
    if s is None or (rc not in (None, "0") and not any((m.get("windows") or [])
                                                        for m in (s.get("members") or {}).values())):
        return {"verdict": "not run" if s is None and rc in (None, "2") else "failed",
                "why": why, "stand_in": stand_in, "in_the_clear": CLEAR["n2"]}
    members = {}
    for name, m in (s.get("members") or {}).items():
        wins, escape = m.get("windows") or [], escaped(name, m, s)
        members[name] = {"network_accepted": (m.get("network") or {}).get("accepted"),
                         "unexpected": unexpected(name, m, escape) if wins else None, "escaped_the_draw": escape,
                         "drawn": sorted(((s.get("sampled") or {}).get(name) or {}).get("draw", {}).get("units") or [])
                         or None, "refused_by_name": ((s.get("sampled") or {}).get(name) or {}).get("why") or None,
                         "refused_windows": (m.get("network") or {}).get("refused"),
                         "windows": len(wins), "accepted_windows": sum(bool(w.get("accepted")) for w in wins),
                         "lean_proved": sum(bool((w.get("proof") or {}).get("proved")) for w in wins),
                         "as_expected": sum(bool(w.get("as_expected")) for w in wins),
                         "leaked": (m.get("received") or {}).get("leaked"),
                         "ok": m.get("ok")}
    honest = members.get("honest") or {}
    sample = s.get("draw") if isinstance(s.get("draw"), dict) else {}
    n = (sample.get("population") or honest.get("windows")
         or (s.get("link") or {}).get("planned_windows", {}).get("honest") or 0)
    sampled = sample.get("k", s.get("drawn", n))
    law = sample.get("law") or s.get("law", "all")
    proved = (bool(members) and all(v["windows"] and not v["unexpected"] for v in members.values())
              and (s.get("ok") is True and rc in (None, "0") or any(v["escaped_the_draw"] for v in members.values())))
    if not proved and honest.get("windows"):
        why = "; ".join(f"{k}: {', '.join(v['unexpected'])}" for k, v in members.items() if v.get("unexpected")) or why
    elif proved:
        gone = [k for k, v in members.items() if v["escaped_the_draw"]]
        why = (f"the driver's ok is false only for {', '.join(gone)}: accepted with none of its incorrect windows "
               "sampled, within the sample's bound" if gone else None)
    return {"verdict": ("accepted" if honest.get("network_accepted") else "refused") if proved else
            ("dry" if "members" in s and not honest.get("windows") else "failed"), "why": why, "stand_in": stand_in,
            "ok": s.get("ok"), "unit": (s.get("link") or {}).get("unit"), "members": members,
            "negatives": {k: "escaped the sample" if v["escaped_the_draw"] else v["network_accepted"] is False
                          for k, v in members.items() if k != "honest"},
            "challenger": s.get("challenger"),
            "clocks": s.get("clocks") or {"source": "declared", "by": "plan.json's syncs (n2_zk); the certificates step "
                                                                      "found them equal to the signed ones: "
                                                                      f"{certs_doc.get('signed_are_declared')}"},
            "eps_at_delta": eps_at_delta(n, sampled, law), "stager": s.get("stager"), "in_the_clear": CLEAR["n2"]}


def scan_step(s: dict | None, network_certifier_run: str | None, job: dict | None = None) -> dict:
    """`scan_zk.py prove`'s summary: accepted when every honest and block round's window is accepted (`SCAN_ACCEPTS`,
    a round's kind `scan_zk.kind_of`'s), every other round's (chain, tampered, unread, stale) refused, no witness piece
    was found at the verifier, and scan_zk's own ``ok`` holds; `RESTS` in place of accepted when an honest round's
    registration verdict let tied values through (``rests_on_period_check``, which the step carries)."""
    from verity.protocols.resource_accounting.network.prover.scan_zk import kind_of
    job = job or {}
    if s is None or (job.get("rc") or {}).get("scan") not in (None, "0"):
        return {"verdict": "not run" if s is None and not (job.get("rc") or {}).get("scan") else "failed",
                "why": scan_exit(s, job), "in_the_clear": CLEAR["scan"]}
    rounds = {}
    for name, r in (s.get("rounds") or {}).items():
        a = r.get("audit") or {}
        rounds[name] = {"accepted": a.get("accepted"), "window_accepted": a.get("window_accepted"),
                        "verdicts": a.get("verdicts"), "why": a.get("why"),
                        "wrong_units": (a.get("audit") or {}).get("wrong_units"), "ok": r.get("ok"),
                        "leaked": (a.get("received") or {}).get("verifier_found") or [],
                        "rests_on_period_check": a.get("rests_on_period_check") or []}
    honest = [v for n, v in rounds.items() if kind_of(n) == "honest"]
    wrong = ([f"{n}: window {'accepted' if v.get('window_accepted') else 'refused'}" for n, v in rounds.items()
              if bool(v.get("window_accepted")) != (kind_of(n) in SCAN_ACCEPTS)]
             + [f"{n}: witness pieces found at the verifier: {v['leaked']}" for n, v in rounds.items() if v["leaked"]]
             + ["scan_zk's ok is not true"] * (s.get("ok") is not True))
    units = 2 * int(s.get("T") or 0) + 1
    k = (s.get("draw") or {}).get("honest") if isinstance(s.get("draw"), dict) else None
    tied = sorted({x for v in honest for x in v.get("rests_on_period_check") or []})
    verdict = ("dry" if not rounds else "refused" if not honest or not all(v.get("window_accepted") for v in honest)
               else "failed" if wrong else RESTS if tied else "accepted")
    return {"verdict": verdict, "rests_on_period_check": tied,
            "why": "; ".join(wrong) or (RESTS_WHY.format(tied=tied) if verdict == RESTS else
                                        (job.get("why") or {}).get("scan")), "ok": s.get("ok"), "rounds": rounds,
            "negatives": {k: not v.get("window_accepted") for k, v in rounds.items() if kind_of(k) not in SCAN_ACCEPTS},
            "stand_in": None if network_certifier_run else "scan_zk's own synthetic network certifier run (scan_traffic), not the gate's "
                                                "rows: its --warden-run takes those",
            "warden_run": network_certifier_run, "eps_at_delta": eps_at_delta(units, int(k or s.get("drawn") or units),
                                                                    f"stratified:{k}" if k else s.get("law") or "all"),
            "in_the_clear": CLEAR["scan"]}


def scan_exit(s: dict | None, job: dict) -> str | None:
    """Why the scan step didn't run or failed: the job's own line for it and, when `scan_zk.py prove` exited nonzero,
    from its summary each round's witness pieces found at the verifier and its own ``ok``."""
    rc, said = (job.get("rc") or {}).get("scan"), (job.get("why") or {}).get("scan")
    if rc in (None, "0"):
        return said
    found = {n: ((r.get("audit") or {}).get("received") or {}).get("verifier_found")
             for n, r in ((s or {}).get("rounds") or {}).items()}
    return "; ".join([*([said] if said else []), f"scan_zk exited {rc}",
                      *(f"{n}: witness pieces found at the verifier: {v}" for n, v in found.items() if v),
                      *(["scan_zk's ok is not true"] if s is not None and s.get("ok") is not True else [])])


#: the value a window circuit registers the scan's state rows as, which no pin holds: the verifier fixes it before no run,
#: so its reads stay free (`NetworkCheck.Period.stateValue`)
SCAN_STATE = "verity/certifier/scan-state/v1"


def input_terms(circuit: dict, pins: dict | None = None, roots: dict | None = None) -> dict:
    """A compute audit window's input bits by what fills them, from its ``circuit.json`` alone (its templates, its
    statements' ``in_rows`` and its declarations; work per template, wire and declaration, never per unit, and no
    Definition looked up), as `NetworkCheck.Circuit.countsOf` counts them (`interface.input_count`):

    * ``inputs``: every unit's input rows;
    * ``wired``: each wire's reader rows times one row of its port, a cut port's row its words over its cuts
      (`Spot.Wires`' ``wiredBits``);
    * ``registered``: each registered read of a held parameter, its reader rows, and each table read and opening of a
      table over one, its ports' rows. A parameter is held where the verifier's pin of it (``pins``, ``{param: root}``,
      the verifier's own, never the record's) is the root the window's registration records for it (``roots``, its
      ``registered_roots``; `interface.held_params`); a read of any other parameter stays free, and with no pins every
      one does;
    * ``free``: the rest, ``#(freeInputs \\ reg)``. A port ``external`` names as the system's I/O stays free, as
      ``countsOf`` leaves it: the prover chooses its values as it does any other unwired input.

    Refused (ValueError): a record that lists its members, a statement row of no one port, a wire or declaration outside
    its port's units or rows, two filling one row (Lean's ``apart``), and a pin of `SCAN_STATE`."""
    from verity.commitments.periods import interface as IF

    if "members" in circuit or "wires" not in circuit:
        raise ValueError("a window circuit is counted from its stated record (its templates, statements and wires), as "
                         "the period check counts it; a record that lists its members is refused")
    if SCAN_STATE in (pins if isinstance(pins, dict) else {}):
        raise ValueError(f"a pin of {SCAN_STATE}: the scan's state is no value the verifier fixes before the run, so its "
                         "reads count free")
    units = {t: int(n) for t, n in circuit["templates"]}
    rows: dict[tuple[str, str], list[int]] = {}
    for t in units:
        for name, _schema, bits, ports in circuit["statements"][t]["in_rows"]:
            port, cut, j = name.partition("#")
            got = rows.setdefault((t, port), [])
            if (ports != [name] if not cut else not all(p.startswith(f"{port}[") for p in ports) or j != str(len(got))):
                raise ValueError(f"{t}: statement row {name!r} of ports {ports} is no one port's row, nor row j of a cut "
                                 "port's rows p#0, p#1, …")
            got.append(int(bits))
    wires, cuts, reg, _ingress = IF.declarations(circuit)
    for (t, port), rs in rows.items():
        if len(rs) != int(cuts.get(t, {}).get(port, 1)) or len(set(rs)) != 1:
            raise ValueError(f"{t} {port}: {len(rs)} rows of {sorted(set(rs))} bits, not its cuts' "
                             f"{cuts.get(t, {}).get(port, 1)} rows of one width")
    held = IF.held_params(pins, roots)
    rects: dict[tuple[str, str], list[tuple[int, int, int, int, str]]] = {}

    def fill(tid: str, port: str, at: int, n: int, r0: int, r1: int, what: str, bits: int) -> int:
        if (tid, port) not in rows or not (0 <= at and at + n <= units[tid] and 0 <= r0 < r1 <= len(rows[tid, port])):
            raise ValueError(f"{what}: units [{at}, {at + n}) and rows [{r0}, {r1}) of no input port of the circuit's")
        for a2, b2, s2, e2, w2 in rects.setdefault((tid, port), []):
            if a2 < at + n and at < b2 and s2 < r1 and r0 < e2:
                raise ValueError(f"{tid} {port}: {w2} and {what} both fill rows of units from {max(at, a2)}")
        rects[tid, port].append((at, at + n, r0, r1, what))
        return bits

    wired = sum(fill(w.reader, w.port, w.at, w.reader_units, w.cut[0], w.cut[0] + w.cut[1], f"wire {w.doc()}",
                     w.units * rows.get((w.reader, w.port), [0])[0]) for w in wires)
    registered = 0
    if reg is not None:
        pinned = {x.name for x in reg.tables if x.param in held}
        for r in reg.reads:
            if (r.reader, r.port) in rows:
                registered += fill(r.reader, r.port, r.at, r.reader_units, r.cut[0], r.cut[0] + r.cut[1],
                                   f"registered read {r.doc()}",
                                   r.units * rows[r.reader, r.port][0] if r.param in held else 0)
        for x in reg.roots:
            k = len(rows.get((x.reader, x.port), ()))
            registered += fill(x.reader, x.port, x.at, x.units, 0, max(k, 1), f"table read {x.doc()}",
                               x.units * sum(rows.get((x.reader, x.port), ())) if x.table in pinned else 0)
        for o in reg.openings:
            for p in o.ports:
                k = len(rows.get((o.reader, p), ()))
                registered += fill(o.reader, p, o.at, o.units, 0, max(k, 1), f"opening {o.doc()} {p}",
                                   o.units * sum(rows.get((o.reader, p), ())) if o.table in pinned else 0)
    inputs = sum(units[t] * sum(rs) for (t, _p), rs in rows.items())
    return {"inputs": inputs, "wired": wired, "registered": registered, "free": inputs - wired - registered}


def free_inputs(circuit: dict, pins: dict | None = None, roots: dict | None = None) -> int:
    """``#(freeInputs \\ reg)`` of a compute audit's window from its ``circuit.json``: every proof block's input-row
    bits less those its wires and the held registered reads at the verifier's ``pins`` fill (`input_terms`)."""
    return input_terms(circuit, pins, roots)["free"]


def registered_bits(circuit: dict, pins: dict | None = None, roots: dict | None = None) -> int:
    """The bits of a window circuit's input rows its registration registers at the verifier's ``pins``, as
    `NetworkCheck.Circuit.countsOf` counts them (`input_terms`' ``registered``)."""
    return input_terms(circuit, pins, roots)["registered"]


def audit_of(d: Path, pins: dict | None = None) -> dict:
    """A compute audit's window as one-stage registers it (``registration.json`` and ``circuit.json``): its proof
    blocks, the proof blocks its sampling design samples (a subset design's ``k``, a stratified design's ``Σ k``, as
    `NetworkCheck.Period` counts them), the
    least rate it samples a unit at (``q``: ``k/n``, or the least stratum's ``k/n``) and its free inputs with the terms
    taken out of them (`input_terms`, the registered rows at the verifier's ``pins`` the registration's window records as
    its ``registered_roots``). Its jitter term is the link's steps over the period, which `d1_terms` charges. A window
    holding a pin must bind ``circuit.json`` by its SHA-512 (``window.circuit``), as `NetworkCheck.Period.auditCheck`
    refuses a count on a circuit its registration doesn't bind."""
    from verity.commitments.periods import interface as IF
    reg = json.loads((d / "registration.json").read_text())
    body = (d / "circuit.json").read_bytes()
    window = reg.get("window") if isinstance(reg.get("window"), dict) else {}
    try:
        roots = IF.registered_roots(window)
        circuit = json.loads(body)
        terms = input_terms(circuit, pins, roots)
    except ValueError as e:
        raise SystemExit(f"--audit {d}: {e}") from e
    if terms["registered"] and window.get("circuit") != hashlib.sha512(body).hexdigest():
        raise SystemExit(f"--audit {d}: its registration's window binds another circuit record than circuit.json, so "
                         "the registered rows counted there are none the registration registers")
    law, n = reg["law"], reg["population"]
    if law.get("law") == "subset":
        sampled, q = law["k"], Fraction(law["k"], n)
    elif law.get("law") == "stratified" and law.get("strata"):
        sizes = [sum(hi - lo for lo, hi in s["units"]) for s in law["strata"]]
        sampled, q = sum(s["k"] for s in law["strata"]), min(Fraction(s["k"], m) for s, m in zip(law["strata"], sizes))
        if q == 0 or sum(sizes) != n:
            raise SystemExit(f"--audit {d}: a stratified sampling design whose strata cover its {n} units, each sampled, not {law}")
    else:
        raise SystemExit(f"--audit {d}: K's q is a subset or stratified sampling design's, not {law}")
    return {"window": str(d), "registration": hashlib.sha512(json.dumps(reg, sort_keys=True).encode()).hexdigest(),
            "units": n, "drawn": sampled, "q": str(q), "free": terms["free"], "inputs": terms,
            "held": sorted(IF.held_params(pins, roots))}


def power_bits(m: int, f: int, exact_below: int = 1 << 16) -> int:
    """``(m^f - 1).bit_length()``, the bits naming one of ``m^f`` sets, computed exactly without the power where it
    would be large: ``f·log2 m`` for ``m`` a power of two, else ``⌊f·log2 m⌋ + 1``, the floor read off a 60-digit
    logarithm, and where that logarithm can't tell the floor (``f·log2 m`` within ``1e-30`` of an integer) the larger
    candidate, so never fewer bits than the power has. A power of under ``exact_below`` bits is computed."""
    if m < 1 or f < 0:
        raise ValueError(f"the sets of {m}^{f}: m ≥ 1, f ≥ 0")
    if m == 1 or f == 0:
        return 0
    if m & (m - 1) == 0:
        return (m.bit_length() - 1) * f
    if f * m.bit_length() < exact_below:
        return (m ** f - 1).bit_length()
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        x = decimal.Decimal(f) * decimal.Decimal(m).ln() / decimal.Decimal(2).ln()
        slack = decimal.Decimal("1e-30")
        return int((x + slack).to_integral_value(decimal.ROUND_FLOOR)) + 1


def pooled_k(audits: list[dict], delta: Fraction, network: dict) -> dict:
    """`NetTiming.Guarantees.PooledLeakAccepted`'s K over one monitoring period: the incorrect proof blocks paid once
    over the period, ``Fmax = ⌈ln(1/δ)/q⌉`` with ``q`` the least ``k_i/n_i`` of its compute audits (a stratified audit's
    least stratum's, `audit_of`'s ``q``), so K = the network's own terms (``network``: the advice's and the syncs' bits,
    and the cells') + ``Fmax·16 + ⌈log2 (N + 1)^Fmax⌉ + Σ_i (free_i + jitter_i)``, ``N = Σ_i n_i``, except with
    probability ``δ + Σ_i (e_i + e'_i)``. `PooledLeakAccepted` is conditional on its per-window inputs, so this K is
    cited as conditional, and its jitter term as `leak_bounds.JITTER` (``jitter_term``)."""
    import leak_bounds as LB
    q = min(Fraction(a["q"]) if "q" in a else Fraction(a["drawn"], a["units"]) for a in audits)
    fmax = math.ceil(math.log(1 / float(delta)) / float(q) * (1 + 1e-12))
    n = sum(a["units"] for a in audits)
    terms = {**network, "faulty_bits": OUT_BITS * fmax, "faulty_sets": power_bits(n + 1, fmax),
             "free": sum(a["free"] for a in audits), "jitter": sum(a["jitter"] for a in audits)}
    k = sum(terms.values())
    return {"delta": str(delta), "q": str(q), "Fmax": fmax, "units": n, "terms": terms, "K": k,
            "share": k / (OUT_BITS * n), "eps": f"{delta} + Σ_i (e_i + e'_i) over {len(audits)} audits",
            "conditional": POOLED_CONDITION, "jitter_term": LB.JITTER}


def network_terms(cert: dict, endpoint: dict) -> dict:
    """The network's own terms of K, charged once a period: the advice's residual bits (`advice_bits`; its v0 bits
    when it wasn't rebuilt from the certified I/O), each window's declared sync, log2 #Σ_sync,
    and no cell: no connection passes the certifier but the link, whose counts the advice fixes."""
    bits, syncs = endpoint.get("advice_bits") or {}, cert.get("sync_bits")
    advice = bits.get("derived") if bits.get("derived") is not None else bits.get("v0_total")
    return {"advice": advice, "syncs": None if syncs is None else math.ceil((cert.get("windows") or 0) * syncs),
            "cells": 0}


def d1_terms(cert: dict, n2: dict, scan: dict, custody_doc: dict, endpoint: dict | None = None,
             audits: list[dict] | None = None) -> dict:
    import leak_bounds as LB
    # each window certified is the certificates step's accept alone: which ones its signer refused is the private side's
    windows = cert.get("windows") or 0
    honest = (n2.get("members") or {}).get("honest") or {}
    acc = [cert.get("verdict") == "accepted" and bool(honest.get("network_accepted"))] * windows
    network = network_terms(cert, endpoint or {})
    steps = windows * (cert.get("window_buckets") or 0)
    if None in network.values():
        k = {"open": f"the network's terms aren't all computed: {network}"}
    elif not audits:
        k = {"open": "no compute audit's window (--audit: the served units' registration and circuit), whose free "
                     "inputs and sample the K counts", "network": network}
    elif not steps:
        k = {"open": "the certificates step records no execution phase's egress buckets (window_buckets), the steps over "
                     "which K's jitter term charges each audit", "network": network}
    else:
        audits = [a | {"steps": steps, "jitter": LB.jitter_bits(steps)} for a in audits]
        k = {f"{d:.3g}": pooled_k(audits, Fraction(d), network) for d in DELTAS} | {"audits": audits}
    return {"theorem": "NetTiming.Guarantees.MonitoringPeriodLeakAccepted, KS as hypothesis",
            "K": k,
            "sum": "NetTiming.monitoringPeriodLeak_le_sum: Pr[every window accepted ∧ total > a + Σ_w b_w] ≤ Σ_w ε_w",
            "windows": windows, "acc": acc, "all_accepted": bool(acc) and all(acc),
            "eps_w": "δ + ε_ks + ε_bind per window (KSLinOn), plus the sample's term at δ (eps_at_delta)",
            "sum_eps": f"{windows}·(δ + ε_ks + ε_bind)",
            "a": "log2 #Ss + Σ_c log2(cap_c + 1), charged once a period",
            "b_w": "#(freeInputs_w \\ reg_w) + log2 #incorrectSets_w(δ) + β_w + k_w·log2 Dj",
            "lms_height": [g.get("lms_height") for g in custody_doc.get("gate_runs", [])],
            "uncovered": [
                "the I/O tie (IoTieRequest) is checked in the clear, outside the proofs",
                "Claims.SamplingGeneric.Compiled.KS is a hypothesis until Flock.Guarantees.SampledKSLive restates it "
                "live",
                "hdata (the network certifiers' arrivals are the framed egress of the certifier's observations) and Carries "
                "are run conditions, not yet named assumptions",
                "B_reg: the trace's values at reg are the registration's",
                "#(freeInputs \\ reg) is not bounded by the advice bits until V3's padding: b_w is vacuous",
                "no carry across windows: each window's terms stand alone"]}


# --- leak: K per monitoring period, on the network attester run ------------------------------------------------------

#: what a monitoring period's count runs over, each machine one link at the run's rate: one link
#: (`leak_bounds.per_link`), and as many of it as the census's reference datacenter and the world have machines
#: (`leak_bounds.at_link_rate`)
SCOPES = ("link", "datacenter", "world")


def leak(run: Path, scan: dict | None = None, weights=None) -> dict:
    """K, the bits the network attester's accepted transcripts can carry about the secret, per monitoring period on
    its gate run ``run`` (a fetched copy: its ``request.json``, ``end.json`` and ``deployment.json``), for the scan's class
    (`scan_zk.CLASS`) and its sample of every proof block, by `leak_bounds`' scaling by links (`SCOPES`): one link
    certifying the run's own request rate (`leak_bounds.per_link`), and as many identical links summed
    (`leak_bounds.at_link_rate`) as the census's reference datacenter has machines (`leak_bounds.datacenter_links`)
    and as the world fleet has (`leak_bounds.world_links`), whatever those scopes serve. Each count is in both forms,
    the per-period form (`MonitoringPeriodLeakAccepted`'s count) and the pooled one (`PooledLeakAccepted`'s at q = 1), in both
    readings of the sampling block's free inputs (`leak_bounds.READINGS`), and beside ε, its scans' δ summed, and its
    share of ``weights`` (`leak_bounds.WEIGHTS`). The run's request rate is the requests its request asked for over
    its execution phases: the end carries no count of the traffic, and the gate takes an end as accepted only when
    its check counted the request's own number (`auditor.network_attester_outcome`). So K is counted only on an accepted run
    of the device's own client; a relay session's request asks for no number, its clients being the peer's, and
    nothing holds its requests to one."""
    import leak_bounds as LB
    from verity.protocols.resource_accounting.network.verifier import scan_answer as SA
    from verity.protocols.resource_accounting.network.prover import scan_zk as SZ
    from verity.protocols.resource_accounting.network.prover import ingress_scan as IS
    from verity.protocols.resource_accounting.network.verifier import network_attester as W
    weights = LB.WEIGHTS if weights is None else weights
    dep = _json(next((p for p in (run / "out" / "deployment.json", run / "deployment.json") if p.is_file()), None))
    v = (_json(run / "end.json") or {}).get("verdict") or {}
    if not dep or not v.get("windows"):
        raise SystemExit(f"leak: {run} has no deployment.json, or no end.json verdict with its windows")
    asked = (_json(run / "request.json") or {}).get("requests")
    if type(asked) is not int or asked < 1:
        raise SystemExit(f"leak: {run}'s request.json asks for no number of requests (a relay session's clients are "
                         "the peer's), so nothing holds its requests to a count and K has no rate")
    if v.get("accepted") is not True:
        raise SystemExit(f"leak: {run}'s end isn't accepted, and the gate holds only an accepted end's requests to "
                         "the number its request asked for")
    params, cls = W.NetworkAttesterParams(**dep["params"]), SZ.CLASS
    cls.serves(params)
    T, units = params.window, 2 * params.window + 1
    sampled = (scan or {}).get("draw", {}).get("honest") if isinstance((scan or {}).get("draw"), dict) else units
    if sampled != units:
        raise SystemExit(f"leak: the scan sampled {sampled} of {units} proof blocks; K's terms for incorrect proof blocks "
                         f"are 0 only with every proof block sampled (scan_zk --draw {units})")
    rate = asked / v["windows"]
    window_s = T * params.bucket_ms / 1000
    at = {"T": T, "tau": Fraction(params.bucket_ms, 1000), "egress_cells": params.egress_cells}
    per_s = Fraction(asked, v["windows"]) / (T * at["tau"])
    link = LB.link(cls, per_s, **at)
    links = {"link": 1, "datacenter": LB.datacenter_links(), "world": LB.world_links()}

    def bound(scope: str, period: str, reading: str, delta: Fraction | None = None) -> LB.Bound:
        if scope == "link":
            return LB.per_link(cls, per_s, period, reading, delta, **at, weights=weights)
        return LB.at_link_rate(link, links[scope], period, reading, delta, weights=weights)

    rows = []
    for period, s in LB.PERIODS.items():
        for scope in SCOPES:
            lw = links[scope] * s / window_s
            for reading in LB.READINGS:
                pooled = {f"{float(d):g}": bound(scope, period, reading, d) for d in LB.DELTAS}
                per = bound(scope, period, reading)
                rows.append({"period": period, "scope": scope, "links": links[scope], "link_windows": lw,
                             "requests": rate * lw, "prog": reading, "pooled": {d: b.bits for d, b in pooled.items()},
                             "pooled_share": {d: b.share for d, b in pooled.items()},
                             "per_period": per.bits, "per_period_share": per.share, "eps": str(per.eps),
                             "per_link_window": link.bits(link.block_s, reading)})
    free = {reading: {"link_window": f["block"], "request": f["request"]}
            for reading, f in ((r, link.free(r)) for r in LB.READINGS)}
    split = "; ".join(f"the {scope}'s F = {LB.fmax(d / links[scope])} at δ = {float(d):g}"
                      for scope in SCOPES[1:] for d in LB.DELTAS)
    return {"warden_run": (_json(run / "end.json") or {}).get("run") or run.name, "params": dep["params"],
            "class": {k: getattr(cls, k) for k in IS.STATICS},
            "widths": link.widths | {"entries": link.entries, "egress_row": 8 * cls.egress.R * cls.B},
            "draw": {"units": units, "drawn": sampled},
            "run": {"link_windows": v["windows"], "requests": asked, "requests_per_link_window": rate,
                    "link_window_s": window_s, "accepted": v.get("accepted")},
            "terms": {"schedules_per_request": link.cell, "cells_per_request": link.cell,
                      "cap_per_request": link.cap, "free": free, "faulty": 0,
                      "jitter_per_block": link.jitter, "jitter_bound": SA.JITTER_BOUND, "jitter_term": LB.JITTER,
                      "b": link.b, "fmax": {f"{float(d):g}": LB.fmax(d) for d in LB.DELTAS},
                      "scan_delta": str(link.scan_delta)},
            "links": links, "weights": weights._asdict(), "rows": rows, "theorems": dict(LB.STATEMENTS),
            "conditions": [
                "the generator reading of the scan's statement (each proof block's registered rows its boundary "
                "values): a claim, not yet a locked guarantee",
                "each connection's cap is its lane's egress_cells, which the network certifier holds a lane to and "
                "NetworkVerifier.Period doesn't check; the schedules and the fail cells are charged at that cap, "
                "⌈log2(cap + 1)⌉ each a request, as its rows form charges them",
                f"the jitter term is T bits a sampling block, one a bucket: {LB.JITTER}",
                "the scan proves sampling block 0 of the network certifier run; a monitoring period's terms are its "
                "per-sampling-block and per-request terms at the run's request rate",
                "the datacenter's rows sum as many identical links as the census's reference datacenter has machines, "
                "and the world's as many as the world fleet has (leak_bounds.at_link_rate, its scaling by links), "
                f"each link pooled at δ / links ({split}) and paying its own incorrect proof blocks: our "
                "composition, which no Lean statement covers. Each link certifies the run's rate; a scope's requests "
                "past its links' rate are uncertified, and no row counts them, so these rows are not the scope's K "
                "(leak_bounds.table counts the network certifier's terms over every request a scope serves)",
                "ε is the scans' δ summed over the period's sampling blocks and the links (NetworkVerifier.Period's "
                "eps, m·δ), each scan at the scan's δ",
                "the loop's (an untied compute audit's) free inputs don't enter K: its δ goes into JointPeriodLive's "
                "sum",
                "the scan's prog is a compute audit's output (NetworkVerifier.Period's progTied), so its slot entries' "
                "bits are the tied audit's free inputs, which Lean's rows-form K charges to the sampling block's term "
                "(tiedAudits), as this K does",
                f"a share is of one copy of the weights, {weights.label}, whatever the links"]}


def leak_markdown(doc: dict) -> str:
    import leak_bounds as LB
    d1, d2 = (f"{float(d):g}" for d in LB.DELTAS)
    bits = lambda x: f"{x / 10 ** (e := math.floor(math.log10(x))):.3g}·10^{e}" if x >= 1e6 else f"{x:,.0f}"
    share = lambda x: f"{100 * x:.3g}%" if x < 1 else f"{x:.3g}×" if x < 100 else f"{x:,.0f}×"
    lines = []
    for reading, title in LB.READINGS.items():
        lines += [f"**{title}**, bits, the share of one copy of the weights ({doc['weights']['label']}) and ε, the "
                  f"scans' δ summed (the jitter term one bit a bucket, {LB.JITTER}):", "",
                  "| period | links | K pooled, δ = 1% | K pooled, δ = 1e-6 | K per period | ε | a sampling block |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['period']} | {r['scope']}, {r['links']:,} | {bits(r['pooled'][d1])} "
                  f"({share(r['pooled_share'][d1])}) | {bits(r['pooled'][d2])} ({share(r['pooled_share'][d2])}) | "
                  f"{bits(r['per_period'])} ({share(r['per_period_share'])}) | {float(Fraction(r['eps'])):.3g} | "
                  f"{r['per_link_window']:,.0f} |"
                  for r in doc["rows"] if r["prog"] == reading]
        lines.append("")
    return "\n".join(lines)


def merged_n2(paths: list[str] | None) -> dict | None:
    """N2's summaries of one period's members proved in several runs (``--n2 A B …``, each run's sample on its own
    challenger session, every one of the same population and K): one summary, its members and sampled samples their
    union, ``ok`` only when every run's is, and each run's challenger session kept."""
    docs = [d for d in map(_json, [paths] if isinstance(paths, str) else paths or []) if d]
    if len(docs) < 2:
        return docs[0] if docs else None
    samples = {json.dumps({k: (d.get("draw") or {}).get(k) for k in ("law", "k", "population")}) for d in docs}
    if len(samples) > 1:
        raise SystemExit(f"--n2: the runs' samples differ: {sorted(samples)}")
    return {**docs[0], "members": {k: v for d in docs for k, v in (d.get("members") or {}).items()},
            "sampled": {k: v for d in docs for k, v in (d.get("sampled") or {}).items()},
            "challenger": [d.get("challenger") for d in docs], "ok": all(d.get("ok") is True for d in docs),
            "runs": len(docs)}


def stubbed(job: dict, custody_doc: dict, endpoint: dict | None = None) -> list[str]:
    """What a certifier-only report stands on that isn't the chain's own: `STUBBED`, and where its gate runs ran and
    what served them when their ``request.json`` says (demo_chain.sh's); a network-certifier-path report's,
    `NETWORK_CERTIFIER_STUBBED`, or `NETWORK_CERTIFIER_SERVED_STUBBED` when its endpoint step (``endpoint``) ran on the served node."""
    if job.get("path") == NETWORK_CERTIFIER:
        return list(NETWORK_CERTIFIER_SERVED_STUBBED if (endpoint or {}).get("at") == SERVED_NODE else NETWORK_CERTIFIER_STUBBED)
    if job.get("path") != CERTIFIER_ONLY:
        return []
    out = list(STUBBED)
    for g in custody_doc.get("gate_runs") or []:
        r = g.get("request") or {}
        if "off-host" in str(r.get("where") or ""):
            out.append(f"{g['gate_run']} ran off-host ({r['where']}), so no gate session or gate custody stands behind it")
        if r.get("engine"):
            out.append(f"{g['gate_run']}'s endpoint is demo_endpoint.py's {r['engine']} engine, not the H200's own "
                       "serving loop")
    return out


def period_binding(answer: dict, case: dict) -> str | None:
    """Why certifier/v0's period ``answer`` isn't ``case``'s, else None: the answer commits to its case's commitment
    alone (`PERIOD_TAG`'s digest)."""
    from verity.commitments.identity import tagged_sha512

    try:
        raw = bytes.fromhex(case["commitment"])
    except (KeyError, TypeError, ValueError) as e:
        return f"the case doesn't decode: {type(e).__name__}: {e}"
    if tagged_sha512(PERIOD_TAG, raw).hex() != answer.get("commitment"):
        return "the case isn't the answer's: its commitment's digest is another"
    return None


def block_entry(answer: dict | None, case: dict | None, verdict: str) -> dict:
    """Why the report writes no block record's certifier entry for certifier/v0's period ``answer`` on ``case``: none
    without an answer, an accepted chain and the answer's own case (`period_binding`), and none for any answer the
    period check gives (`ROWS_FORM`)."""
    why = (NO_PERIOD_ANSWER if answer is None else
           f"the chain is {verdict}: only an accepted chain carries a K" if verdict != "accepted" else
           "no case: what the answer commits to can't be held to the chain" if case is None else
           period_binding(answer, case))
    return {"written": False, "why": why or ROWS_FORM}


def period_block(a, verdict: str) -> dict:
    """`block_entry` on the report's ``--period-answer`` and its case (``--period-case``, by default ``X.case.json``
    beside ``X.answer.json``)."""
    pa = getattr(a, "period_answer", None)
    if pa and not Path(pa).is_file():
        return {"written": False, "why": f"--period-answer {pa}: not a file"}
    pc = getattr(a, "period_case", None) or (str(pa).removesuffix(".answer.json") + ".case.json" if pa else None)
    if pa and not Path(pc).is_file():
        return {"written": False, "why": f"no case at {pc} (--period-case): what the answer commits to can't be held "
                                         "to the chain"}
    return block_entry(_json(pa), _json(pc), verdict)


def report(a) -> dict:
    cust, cert = _json(a.custody) or {}, _json(a.certs) or {}
    job = _json(a.job) or {}
    certifier_only = job.get("path") == CERTIFIER_ONLY
    network_attester = job.get("path") == NETWORK_CERTIFIER
    scan_only = a.n2 is None and not certifier_only and not network_attester
    endpoint = (network_attester_endpoint_step if network_attester else endpoint_step)(_json(a.endpoint))
    merged = None if scan_only else merged_n2(a.n2)
    n2 = ({"verdict": "not given", "why": SCAN_ONLY, "in_the_clear": CLEAR["n2"]} if scan_only
          else n2_step(merged, cert, job))
    scan = ({"verdict": "not run", "why": STUBBED[0], "in_the_clear": CLEAR["scan"]} if certifier_only
            else scan_step(_json(a.scan), a.network_certifier_run, job))
    certificates = dict(cert)
    tie = None
    if network_attester:
        run_id, tie_doc = cert.get("warden_run"), _json(getattr(a, "tie", None))
        certificates = on_network_attester(certificates, _json(a.certs), run_id, "the certificates step's input")
        endpoint = on_network_attester(endpoint, _json(a.endpoint), run_id, "the endpoint's report")
        n2 = on_network_attester(n2, merged, run_id, "N2's summary") | {
            "clocks": {"source": "none", "by": "the network certifier link declares no sync and signs no clock (n2_zk "
                                               "--warden-run)"}}
        tie = on_network_attester(tie_step(tie_doc, job), tie_doc, run_id, "the tie's summary")
        scan = on_network_attester(scan, _json(a.scan), run_id, "the scan's summary")
    rests = scan.get("verdict") == RESTS
    proved = ((scan_only or n2.get("verdict") == "accepted")
              and (certifier_only or scan.get("verdict") in ("accepted", RESTS))
              and (tie is None or tie.get("verdict") == "accepted"))
    steps = {"custody": cust, "certificates": certificates, "endpoint": endpoint, **({"tie": tie} if network_attester else {}),
             "n2": n2, "scan": scan, "lean": {"verdict": (RESTS if rests else "accepted") if proved else (
                                        scan if scan_only else n2).get("verdict"),
                                    "binary": job.get("lean"), "in_the_clear": CLEAR["lean"],
                                    "n2_windows_proved": ((n2.get("members") or {}).get("honest") or {}).get(
                                        "lean_proved"),
                                    "scan_rounds": {k: v.get("verdicts") for k, v in scan.get("rounds", {}).items()}}}
    required = (("custody", "certificates", "endpoint", "tie", "n2", "scan") if network_attester else
                ("custody", "certificates", *(("endpoint",) if job.get("profile") == ADVICE else ()),
                 *(() if scan_only else ("n2",)), *(() if certifier_only else ("scan",))))
    accepted = (all(steps[k].get("verdict") in ("accepted", RESTS) for k in required)
                and not any(steps[k].get("stand_in") for k in ("n2", "scan")))
    rehearsal = {f"rehearsal/{g['gate_run']}/{s}": v == "refused" for g in cust.get("gate_runs") or []
                 for s, v in (g.get("strategies") or {}).items() if s not in REHEARSAL_ACCEPTS}
    negatives = {**rehearsal, **{f"tie/{k}": v for k, v in endpoint.get("negatives", {}).items()},
                 **{f"tie/{k}": v for k, v in (tie or {}).get("negatives", {}).items()},
                 **{f"n2/{k}": v for k, v in n2.get("negatives", {}).items()},
                 **{f"scan/{k}": v for k, v in scan.get("negatives", {}).items()}}
    audit_dirs = [Path(d) for d in getattr(a, "audit", None) or []]
    pins = getattr(a, "pins", None)
    if pins is not None and not Path(pins).is_file():
        raise SystemExit(f"--pins {pins}: no such file")
    audits = [audit_of(d, _json(pins)) for d in audit_dirs]
    verdict = ("scan " if scan_only else "") + ((RESTS if rests else "accepted") if accepted else "refused")
    why = ({k: ": ".join(str(x) for x in (steps[k].get("verdict") or "no verdict", steps[k].get("why")) if x)
            for k in required if steps[k].get("verdict") != "accepted"}
           | {k: f"stand-in: {steps[k]['stand_in']}" for k in ("n2", "scan") if steps[k].get("stand_in")})
    return {"schema": SCHEMA, "job": job, "verdict": verdict, "path": job.get("path"),
            "scope": SCAN_ONLY if scan_only else None, "stubbed": stubbed(job, cust, endpoint),
            "negatives_refused": negatives, "steps": steps,
            "d1": d1_terms(cert, n2, scan, cust, endpoint, audits),
            "block": period_block(a, verdict), "why": why or None}


def markdown(r: dict) -> str:
    j, s = r["job"], r["steps"]
    lines = [f"# Network chain run {j.get('run', '?')}", "",
             f"Tree {j.get('tree', '?')}, gates {', '.join(j.get('gates', [])) or '?'}. Verdict: **{r['verdict']}**"
             + (f" on the {r['path']} path" if r.get("path") else "") + (f", {r['scope']}." if r.get("scope") else "."),
             ""]
    if r.get("stubbed"):
        lines += ["Stubbed:", *[f"- {x}" for x in r["stubbed"]], ""]
    lines += ["| step | verdict | ε at δ | in the clear |", "|---|---|---|---|"]
    shown = [k for k in ("custody", "certificates", "endpoint", "tie", "n2", "scan", "lean") if k in s]
    for k in shown:
        e = s[k].get("eps_at_delta") or {}
        eps = (f"{e['law']}, {e['drawn']} of {e['units']}: incorrect ≤ {e['incorrect_at_delta']}" if e else "")
        lines.append(f"| {k} | {s[k].get('verdict')} | {eps} | {s[k].get('in_the_clear', '')} |")
    notes = [f"- {k}: {s[k][f]}" for k in ("endpoint", "tie", "n2", "scan") if k in s for f in ("why", "stand_in")
             if s[k].get(f)]
    ep = s["endpoint"]
    if ep.get("advice"):
        notes.append(f"- endpoint: advice {ep['advice']}; " + (f"I/O tie on honest {ep['tie']}" if "tie" in ep else
                                                                f"certified I/O {ep.get('io')}"))
    if s.get("tie", {}).get("tie"):
        notes.append(f"- tie: execution phases tied by member {s['tie']['tie']}")
    lines += ["", *notes, "", f"Negatives refused: {r['negatives_refused']}", "", "## D1", ""]
    d = r["d1"]
    lines += [f"- {d['theorem']}; {d['sum']}", f"- windows {d['windows']}, all accepted {d['all_accepted']}",
              f"- Σ ε_w = {d['sum_eps']}"]
    k = d.get("K") or {}
    if "open" in k:
        lines.append(f"- K open: {k['open']}" + (f"; the network's terms {k['network']}" if k.get("network") else ""))
    for delta, h in k.items():
        if isinstance(h, dict) and "K" in h:
            lines.append(f"- K at δ = {delta}: {h['K']} bits, {h['share']:.3g}× the {h['units']} units' "
                         f"{OUT_BITS * h['units']} output bits; Fmax {h['Fmax']} at q = {h['q']}, terms {h['terms']}, "
                         f"the jitter term {h['jitter_term']}; {h['conditional']}")
    lines += ["- uncovered:"] + [f"  - {u}" for u in d["uncovered"]]
    lines += ["", "## Block record", "", f"- no certifier entry: {(r.get('block') or {}).get('why')}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("custody", help="each gate run's files against its end.json")
    c.add_argument("--gates", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("certs", help="the client's kept certificates, verified")
    v.add_argument("--warden-run", dest="network_certifier_run", required=True, type=Path,
                   help="the network certifier path: an active network certifier's run (network_attester_chain)")
    v.add_argument("--deployed", action="store_true", help="refuse a key that isn't LMS at lms.DEPLOYED")
    v.add_argument("--out", required=True, type=Path)
    v.add_argument("--witness", type=Path, help="where the step's private side goes (network_attester_chain.WITNESS_KEYS): the "
                                                "chain's, never the record")
    e = sub.add_parser("endpoint",
                       help="the network certifier path: the endpoint's own advice bound to the network certifier's certified I/O")
    e.add_argument("--warden-run", dest="network_certifier_run", required=True, type=Path)
    e.add_argument("--endpoint", required=True, type=Path, help="the endpoint's records (network-e2e chain.ENDPOINT)")
    e.add_argument("--eos", required=True,
                   help="the model's EOS token ids, comma-separated (a network certifier run records none)")
    e.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("report", help="one report over the steps' outputs")
    for f in ("custody", "certs", "endpoint", "scan", "job"):
        r.add_argument(f"--{f}")
    r.add_argument("--tie", help="the network certifier path: n2_tie_zk.py prove --class warden's summary")
    r.add_argument("--n2", nargs="*", help="N2's summary, or one per run of a period's members (merged_n2); without "
                                           "it, the scan's part alone (SCAN_ONLY)")
    r.add_argument("--warden-run", dest="network_certifier_run")
    r.add_argument("--audit", nargs="*", help="the served proof blocks' compute audit windows (registration.json and "
                                              "circuit.json each), whose sample and free inputs d1's K counts")
    r.add_argument("--pins", help="the verifier's own pins, a JSON {param: root} of the roots it fixed before the run: "
                                  "an --audit window's reads of a parameter whose registered root is its pin leave its "
                                  "free inputs (chain_record.registered_bits); with none, every registered read counts")
    r.add_argument("--period-answer", help="certifier/v0's network-period check answer on the period "
                                           "(certifier.answer.json)")
    r.add_argument("--period-case", help="the answer's case (default X.case.json beside X.answer.json), held to the "
                                         "answer's commitment")
    r.add_argument("--out", required=True, type=Path)
    k = sub.add_parser("leak", help="K per monitoring period on the network certifier run, for the scan's class and sample")
    k.add_argument("--warden-run", dest="network_certifier_run", required=True, type=Path,
                   help="the fetched network certifier gate run (its end.json)")
    k.add_argument("--scan", help="the scan's summary on that run (its sample); default every proof block sampled")
    k.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.command == "custody":
        doc = custody(a.gates)
    elif a.command == "certs":
        from benchmarks.network_traces import network_attester_chain as WC
        doc, private = WC.checked(a.network_certifier_run, a.deployed)
        doc |= {"in_the_clear": CLEAR["warden_certificates"]}
        if a.witness is not None:
            a.witness.parent.mkdir(parents=True, exist_ok=True)
            a.witness.write_text(json.dumps(private, indent=1, sort_keys=True))
    elif a.command == "endpoint":
        doc = network_attester_endpoint(a.network_certifier_run, a.endpoint, [int(x) for x in a.eos.split(",")])
        doc["verdict"] = network_attester_endpoint_step(doc)["verdict"]
        a.out.parent.mkdir(parents=True, exist_ok=True)
    elif a.command == "leak":
        doc = leak(a.network_certifier_run, _json(a.scan))
        a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
        a.out.with_suffix(".md").write_text(leak_markdown(doc))
        print(leak_markdown(doc))
        return 0
    else:
        doc = report(a)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "report.md").write_text(markdown(doc))
        a.out = a.out / "report.json"
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: doc.get(k) for k in ("verdict", "scope", "why", "negatives_refused") if k in doc}))
    return 0 if doc.get("verdict") in ("accepted", "scan accepted", RESTS, f"scan {RESTS}") else 1


if __name__ == "__main__":
    sys.exit(main())
