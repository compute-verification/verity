"""chain-record: the network chain's run of record, from the gate's fetched tarballs to one report
(`chain_record.sh` runs it on a CPU machine). Each step reads what the step before it kept; none drives an endpoint.

    python benchmarks/network_traces/chain_record.py custody --gates DIR --out FILE
    python benchmarks/network_traces/chain_record.py certs --run DIR [--client DIR] [--deployed] --out FILE
    python benchmarks/network_traces/chain_record.py certs --warden-run DIR [--deployed] --out FILE
    python benchmarks/network_traces/chain_record.py endpoint --warden-run DIR --endpoint DIR --eos IDS --out FILE
    python benchmarks/network_traces/chain_record.py report --custody FILE --certs FILE [--endpoint REPORT]
      [--n2 SUMMARY ...] [--scan SUMMARY] [--tie SUMMARY] [--job FILE] [--audit WINDOW ...]
      [--period-answer FILE [--period-case FILE] --server NODE] --out DIR
    python benchmarks/network_traces/chain_record.py leak --warden-run DIR [--scan SUMMARY] --out FILE

* **custody**: each gate run under ``--gates`` (a directory holding the gate's ``end.json``, as the gate's ``fetch``
  writes it) holds the files its ``end.json`` names, each with the SHA-512 it records. In the clear, of public
  files; it proves nothing about what the gate ran.
* **certs**: the client's kept files (``--client``, by default the certifier rehearsal's own directory ``--run``):
  ``certificates.jsonl`` and ``clocks.jsonl``, each certificate's public part (`certifier_serve.keep_certificates`),
  held again to the certifier's key and session in ``--run``'s ``deployment.json`` by the client's own
  `check_certificates`. Each window's two certified records (``--run``'s ``certified.jsonl``, their heads and salts)
  are held by the host audit (`certifier_serve._certified`, N2's audit without its head checks) under the syncs the
  certifier signed (``--run``'s salted ``clocks.jsonl``, `advised.SignedClocks`), never the plan's, and their
  certificates, and the clocks', must be exactly the client's. ``--deployed`` refuses a key that isn't LMS at
  `lms.DEPLOYED`.
* **report**: one JSON and one Markdown report, per step its verdict, its run ids, its ε at δ and what it computed in
  the clear, and the period's terms for `NetTiming.Guarantees.PeriodLeakAccepted` summed by
  `periodLeak_le_sum`. ``--endpoint`` is network-e2e's ``--no-prove`` report on the run and the endpoint's
  records: the endpoint's advice bound to the run and the I/O tie, in the clear, required on the advice profile. N2's
  summary is `n2_advised_zk.py prove --e2e`'s on the advice profile and `n2_zk.py prove`'s on the others (Selection
  All: every link-window's one unit proved) or a sampled draw's (``CR_N2_CMD``), read by the same fields. Scan's is
  `scan_zk.py prove`'s, a stand-in on its own synthetic warden run until its ``--warden-run`` takes the gate's warden
  rows, accepted only when every round's verdict is the one scan_zk expects and no witness piece reached the
  verifier (`scan_step`). A step on a stand-in (N2 on ``--job``'s ``stand_in``, scan without a warden run) never makes
  the report accepted. Without ``--n2`` the report is the scan's part of the record alone (`SCAN_ONLY`, its
  ``scope``), and its verdict is ``scan accepted`` or ``scan refused``: N2 isn't required, and the rest is held as
  before. On the certifier-only path (``--job``'s ``path``, `CERTIFIER_ONLY`) there are no warden rows: the scan
  isn't run or required, and the report's ``stubbed`` says what the path stands on (`STUBBED`, `stubbed`). Its
  negatives are N2's members, the rehearsal's own check on each run it must refuse, and the I/O tie's verdict on each
  tampered run that check lets through. d1's K (`horizon_k`) counts the served units' compute audits (``--audit``: each
  window's draw and free inputs, `free_inputs`) and the network's own terms (`network_terms`), at each δ. With
  ``--period-answer`` (certifier/v0's ``network-period check`` answer on the period, a ``certifier.answer.json``,
  its case ``certifier.case.json`` beside it or ``--period-case``) and ``--server``, an accepted report also writes
  ``block.json``: the block record's certifier entry (``legs.certifier`` of ``verity/block-record/v1``),
  `block_certifier.block_leg` of that answer, once the answer is held to this chain (`period_binding`) and is
  ``network-period check``'s own on its case (`decide_period`; `block_entry`).
* **The warden path** (``--job``'s ``path``, `WARDEN`): every step reads an active warden's own run (the gate's
  ``warden-rehearse``, or `warden_serve.py`'s directory; `warden_chain`) and no certifier rehearsal's, whose input is
  refused by name (`warden_chain.PROXY`). ``certs --warden-run`` reruns the warden's own offline check on its witness
  and holds the public ``certificates.jsonl`` to the certified lines (`warden_chain.certificates`; no clock
  certificate: the warden signs none). ``endpoint`` binds the endpoint's own advice to the certified I/O read off the
  warden's cells (`warden_endpoint`); where the endpoint's records stay on the served node, that node runs it
  (`served_endpoint`, `served_session.py endpoint`) and the chain reads its report. N2 is `n2_zk.py prove
  --warden-run`'s, on the warden link; the I/O tie is
  `n2_tie_zk.py prove --class warden`'s, proved (``--tie``, `tie_step`); the scan is `scan_zk.py prove
  --warden-run`'s. Each is held to the warden run the certificates step read (`on_warden`), every one is required,
  and the report names what the path stands on (`WARDEN_STUBBED`, or `WARDEN_SERVED_STUBBED` on the served node's
  endpoint report).
* **leak**: K per monitoring period (week, month, year; one machine and 1,000) on the warden gate run, pooled at
  δ = 1% and 10⁻⁶ and in the per-period form, each beside its share of Llama-3.2-1B's BF16 weights (`leak`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

from verity.core.protocols.pous import wipe_monitoring

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCHEMA = "verity-network-chain/record-report/v1"
DELTAS = (0.5, 2.0 ** -20)
ADVICE = "advice"  # certifier_serve.ADVICE: the profile whose counts N2's advised unit decides on the endpoint's advice

CLEAR = {
    "custody": "the SHA-512 of each fetched file against the gate's end.json (public files)",
    "endpoint": "at the auditor, on the endpoint's records and the run's certified I/O: each Commit window's advice "
                "opened to its registered root and rebuilt from the certified I/O, and the I/O tie (IoTieRequest's "
                "reference, network-e2e's stand-ins for the warden's rows and the Commit's port) on the honest run",
    "certificates": "at the client, on its kept certificates (their public parts): each certificate's and clock "
                    "certificate's LMS signature under the certifier's key and session (check_certificates); and the "
                    "host audit of the rehearsal's certified heads and salted clock openings (witness a rehearsal "
                    "publishes), whose certificates must be exactly the client's",
    "n2": "at the verifier: the registration records, the draw, the certificates' public fields, the clock syncs, the "
          "proof's pub row opened to the verifier's own copy and the o0 output word; never a record, head, leaf "
          "salt, sibling or the certifier's salt (n2_zk's RECEIVED, scanned for every piece)",
    "scan": "at the verifier: each unit's o0 output opened from the salts the prover sends; the I/O tie "
            "(IoTieRequest) stays in the clear, outside the proofs",
    "lean": "nothing: each proof is decided by the verifier's own Lean `flock-verify verify --zk` run",
    "warden_certificates": "at the auditor, on the warden's witness: the warden's own offline check "
                           "(warden_serve.check: each record's certificate and its rows' registration under the "
                           "warden's key and session, the audit and the client's view), and the public "
                           "certificates.jsonl held to the certified lines",
    "warden_endpoint": "at the auditor, on the endpoint's records and the warden's certified I/O (warden_chain.io, off "
                       "the records' cells): each Commit window's advice opened to its registered root and rebuilt "
                       "from the certified I/O",
    "warden_endpoint_served": "on the served node, the prover's, whose records hold request values and stay there: "
                              "each Commit window's advice opened to its registered root and rebuilt from the warden's "
                              "certified I/O sent to it (served_session.py endpoint); the chain reads its report",
    "tie": "at the verifier: each ingress bucket unit's pub row opened to the verifier's own copy (the advice's prompt "
           "lengths and the warden's release) and its o0 verdict bit; never a cell or its salt (n2_tie_zk's RECEIVED, "
           "scanned for every piece)",
}
#: the certifier-only path (``--job``'s ``path``): the records are the certifier rehearsal's alone
CERTIFIER_ONLY = "certifier-only"
#: a report given no N2 summary (``report`` without ``--n2``, `chain_record.sh`'s ``CR_SCAN_ONLY=1``): the scan's
#: part of the record, whose N2 runs in jobs of its own; its verdict is ``scan accepted`` or ``scan refused``, never the
#: whole chain's ``accepted``
SCAN_ONLY = "scan only, N2 not given"
#: what a report on the certifier-only path stands on that isn't the chain's own
STUBBED = (
    "the records come from the certifier-only path (certifier_serve.py's proxy in front of the endpoint), not the "
    "network certifier's own cells as one device, so there are no warden rows and the scan isn't run",
    "the output tie (IoTieRequest) is computed in the clear at the auditor, outside the proofs",
    "the weights' load isn't certified: nothing here shows the endpoint served the registered weights")
#: the warden path (``--job``'s ``path``): every step reads an active warden's own run (`warden_chain`)
WARDEN = "warden"
#: what a report on the warden path stands on that isn't the chain's own
WARDEN_STUBBED = (
    "the endpoint's advice is bound to the warden's certified I/O and rebuilt from it in the clear at the auditor, "
    "and the output tie to the Commit's port isn't proved",
    "the weights' load isn't certified: nothing here shows the endpoint served the registered weights")
#: an endpoint report's ``at`` when its step ran on the served node (`served_session.py endpoint`)
SERVED_NODE = "the served node"
#: `WARDEN_STUBBED` when the endpoint's report is the served node's
WARDEN_SERVED_STUBBED = (
    "the endpoint's advice is bound to the warden's certified I/O and rebuilt from it in the clear on the served node, "
    "the prover's, so the chain's endpoint step is the prover's report of that check, and the output tie to the "
    "Commit's port isn't proved", WARDEN_STUBBED[1])
#: the rehearsal's strategies its own check accepts (certifier_serve.ACCEPTED): token's changed bytes only the tie refuses
REHEARSAL_ACCEPTS = ("honest", "token")
#: a proof block's outputs, at most 16 bits: a faulty proof block's leak
OUT_BITS = 16
#: K's jitter term, 0 at D_j = 1: J (`NetTiming.JitterBounded`) assumes it, not met there until a run shows the written
#: buckets tell a receiver no more than an idle one. The warden hands each bucket to a pacer that releases it at a time
#: fixed before the bucket is made (`warden.Warden(paced=…)`), as the served proxy does (`active.Proxy(paced=…)`)
JITTER = "assumed (J)"
#: what `horizon_k`'s K rests on until its two per-window inputs are discharged
HORIZON_CONDITION = ("NetTiming.Guarantees.HorizonLeakAccepted is conditional: its per-window inputs hesc "
                     "(proofs' SampledKSLive) and hleak (the live SampledLeakAccepted) are not discharged")
#: why a report without ``--period-answer`` writes no block entry
NO_PERIOD_ANSWER = ("no --period-answer: certifier/v0's case needs the period's compute audit (its registration and "
                    "answer) beside each link-window's N2 registration, and this report has neither")
#: certifier/v0's commitment digest: tagged_sha512(PERIOD_TAG, the commitment's bytes) (`NetworkCheck.Period`'s)
PERIOD_TAG = "verity-network-certifier/period/v0"


def _json(path) -> dict | None:
    p = Path(path) if path else None
    return json.loads(p.read_text()) if p and p.is_file() else None


def subset_bound(n: int, k: int, delta: float) -> int | None:
    """The most wrong units an accepted uniform sample of k of n units hides except with probability δ: the integrity
    profile's count bound (`Stratified.count_bound`), the largest b with C(n − b, k) / C(n, k) ≥ δ. Selection All
    (k = n) is 0."""
    if n <= 0 or not 0 < k <= n:
        return None
    from verity.core.protocols.profile import Stratified

    return Stratified.uniform(n, k).count_bound(delta)


def eps_at_delta(n: int, k: int, law: str) -> dict:
    return {"law": law, "units": n, "drawn": k,
            "wrong_units_at_delta": {f"{d:.3g}": subset_bound(n, k, d) for d in DELTAS},
            "per_proof": "each proof's soundness error, ε_ks + ε_bind (KSLinOn's κ + √(γ·Pr B) + Pr[Bext]), "
                         "symbolic: Claims.Sampling.KS is a hypothesis until Flock.Guarantees.SampledKSLive"}


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
        request = _json(root / "request.json") or {}
        summary = _json(root / "out" / "summary.json") or {}
        runs.append({"gate_run": root.name, "dir": str(root.relative_to(gates)), "ended": doc.get("ended"),
                     "ended_utc": doc.get("ended_utc"), "files": len(files), "mismatches": bad,
                     "request": request, "lms_height": summary.get("lms_height"), "windows": summary.get("windows"),
                     "profile": summary.get("profile"),
                     "strategies": {s: (r or {}).get("verdict") for s, r in (doc.get("runs") or {}).items()}})
    ok = bool(runs) and all(r["files"] and not r["mismatches"] for r in runs)
    return {"verdict": "accepted" if ok else "refused", "gate_runs": runs, "in_the_clear": CLEAR["custody"]}


# --- certificates -----------------------------------------------------------------------------------------------------

def certs(run: Path, client: Path, deployed: bool = False) -> dict:
    import certifier_serve as cs
    from verity.ml.crypto import lms
    from verity.core.protocols.certifier import advised as AD

    plan, report, dep = (json.loads((run / f).read_text()) for f in ("plan.json", "calibration.json",
                                                                    "deployment.json"))
    setup, key = cs.ar.Setup(report), cs.read_public(dep["certifier_key"])
    session, windows = bytes.fromhex(dep["session"]), plan["windows"]
    profile = plan.get("occupancy") or {}
    lw = cs.LinkWindows(dep["links"]["n2"], setup.egress_params, setup.sync_set, key, dep["run_id"], session,
                        ingress=setup.ingress_params, proof_deadline=cs.PROOF_DEADLINE_S,
                        occupancy=(profile.get("egress"), profile.get("ingress")), served=cs.served_profile(plan))
    kept, kept_clocks = client / "certificates.jsonl", client / "clocks.jsonl"
    records = next((d / "certified.jsonl" for d in (run, client) if (d / "certified.jsonl").is_file()), None)
    clocks_file = run / "clocks.jsonl"
    out: dict = {"client": str(client), "kept": kept.name if kept.is_file() else None,
                 "kept_clocks": kept_clocks.name if kept_clocks.is_file() else None,
                 "records_file": records and records.name,
                 "clocks_file": clocks_file.name if clocks_file.is_file() else None, "windows": windows,
                 "key": {"params": list(dep["certifier_key"]["params"]), "key_id": dep["certifier_key"].get("key_id"),
                         "public": key.public.hex(),
                         "deployed": tuple(dep["certifier_key"]["params"]) == (lms.DEPLOYED.N, lms.DEPLOYED.W,
                                                                               lms.DEPLOYED.H)},
                 "session": dep["session"], "link": lw.link, "sync_bits": lw.charge,
                 "in_the_clear": CLEAR["certificates"]}
    if records is None or out["clocks_file"] is None or not kept.is_file() or not kept_clocks.is_file():
        return out | {"verdict": "refused", "why": f"{client}: the client kept no certificates.jsonl or clocks.jsonl, "
                                                   f"or {run} holds no certified.jsonl or clocks.jsonl"}
    held = client_certificates(dep, windows, kept, kept_clocks)
    certified, refused = cs.read_certified(records)
    clocks = cs.read_clocks(clocks_file)
    public = lambda c: json.dumps(cs.certificate_json(c), sort_keys=True)
    mine = {(d.value, w): public(r.certificate) for (d, w), r in certified.items()}
    out["client_check"] = {k: v for k, v in held.items() if k not in ("by_window", "clocks")} | {
        "records_are_the_clients": mine == held["by_window"],
        "clocks_are_the_clients": sorted(public(c.certificate) for c in clocks) == sorted(held["clocks"])}
    signed = list(AD.SignedClocks(key, session, tuple(clocks)).syncs(lw.link))
    declared = [cs.ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    rows = []
    for w in range(windows):
        pair = certified.get((cs.Direction.EGRESS, w)), certified.get((cs.Direction.INGRESS, w))
        if None in pair:
            rows.append({"window": w, "certified": False, "why": "no certified record"})
        elif w >= len(signed):
            rows.append({"window": w, "certified": False, "why": "no clock certificate verifies for the window"})
        else:
            rows.append({"window": w, "certified": bool(cs._certified(lw, lw.commit(w, *pair), signed))})
    out |= {"clock_certificates": len(clocks), "clocks_signed": len(signed),
            "signed_are_declared": signed == declared[:len(signed)] and len(signed) == len(declared),
            "issuer_refused": refused, "certified_windows": sum(r["certified"] for r in rows), "rows": rows}
    cc = out["client_check"]
    client_ok = (not cc["refused"] and cc["records_are_the_clients"] and cc["clocks_are_the_clients"]
                 and cc["certificates"] == 2 * windows)
    ok = (out["certified_windows"] == windows and out["clocks_signed"] >= windows and not refused and client_ok
          and (out["key"]["deployed"] or not deployed))
    why = None if ok else ("the key isn't LMS at lms.DEPLOYED" if deployed and not out["key"]["deployed"]
                           else "the client's kept certificates don't verify or aren't the records'" if not client_ok
                           else "a window's records or clock don't verify")
    return out | {"verdict": "accepted" if ok else "refused", "why": why}


def client_certificates(dep: dict, windows: int, kept: Path, kept_clocks: Path) -> dict:
    """The client's kept files (`certifier_serve.keep_certificates`: each certificate's public part with its session,
    no head, sync or salt) held again to the deployment's key and session, window by window, by the client's own
    `check_certificates` on the answer they make up."""
    import certifier_serve as cs

    lines = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    certs, clocks = lines(kept), lines(kept_clocks)
    strip = lambda d: {k: v for k, v in d.items() if k != "session"}
    sessions = {d["session"] for d in certs + clocks if "session" in d}
    refused, seen, by_window, kept_clock = [], {}, {}, []
    for w in range(windows):
        answer = {"window": w, "session": dep["session"],
                  "certificates": [strip(d) for d in certs if "unsigned" not in d and d["window"] == w],
                  "clocks": [strip(d) for d in clocks if d["window"] == w],
                  "unsigned": [d["unsigned"] for d in certs if "unsigned" in d and d["unsigned"][2] == w]}
        got, synced, unsigned, no = cs.check_certificates(dep, w, answer, seen)
        refused += no + [f"window {w}: the signer refused {u}" for u in unsigned]
        for c in got:
            by_window[(c.direction.value, c.window)] = json.dumps(cs.certificate_json(c), sort_keys=True)
        kept_clock += [json.dumps(cs.certificate_json(k), sort_keys=True) for k in synced]
    if sessions - {dep["session"]}:
        refused.append(f"certificates of another session: {sorted(sessions - {dep['session']})}")
    return {"certificates": len(by_window), "clock_certificates": len(kept_clock), "refused": refused,
            "by_window": by_window, "clocks": kept_clock}


# --- report -----------------------------------------------------------------------------------------------------------

def endpoint_step(e: dict | None) -> dict:
    """network-e2e's ``--no-prove`` report on the run and the endpoint's records: the endpoint's own advice bound to the
    run's certified I/O and rebuilt from it (`chain.served_advice`), and the I/O tie's verdict on the honest run
    (`chain.tie`), both in the clear. The tie decides a tampered run only where the rehearsal's own check let it
    through (no ``own_code`` in the report's ``negatives``: token), and must refuse it at exactly the link-windows its
    changes fall in (``ok``); a run the check refused (late, extra) is the rehearsal's negative, not the tie's."""
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


def warden_endpoint(run: Path, ep: Path, eos: list[int]) -> dict:
    """The endpoint's own advice for a warden run (`served_endpoint`), on the certified I/O read off its cells
    (`warden_chain.io`)."""
    import warden_chain as WC

    return served_endpoint(WC.deployment(run)[0], WC.io(run), ep, eos)


def served_endpoint(dep: dict, io: dict, ep: Path, eos: list[int]) -> dict:
    """The endpoint's own advice for the warden run of ``dep`` (its ``deployment.json``), as network-e2e's
    `endpoint_advice` makes it for a certifier rehearsal's: `verity-vllm advice` over the endpoint's records ``ep``
    (network-e2e's ``chain.ENDPOINT``) into ``ep/advice`` at the warden's bucket unless an index is there, then
    ``chain.served_advice`` on the certified I/O ``io`` (`warden_chain.io`'s) over the warden's link-windows, with the
    model's EOS ids ``eos`` (a warden run records none)."""
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


def warden_endpoint_step(e: dict | None) -> dict:
    """`warden_endpoint`'s report: accepted when the warden's certified I/O reads whole off its cells and the
    endpoint's own advice is bound to it and rebuilt from it, in the clear."""
    if e is None:
        return {"verdict": "not run", "why": "no endpoint records (CR_ENDPOINT)",
                "in_the_clear": CLEAR["warden_endpoint"]}
    b, io = (e.get("chain") or {}).get("bits") or {}, e.get("io") or {}
    windows = b.get("commit_windows") or []
    advice = {"bound": b.get("bound"), "rebuilt": b.get("rebuilt"), "why": b.get("bound_why") or b.get("why"),
              "commit_windows": len(windows), "bound_windows": sum(bool(c.get("bound") and c.get("rebuilt"))
                                                                   for c in windows)}
    why = (f"the warden's certified I/O doesn't read whole off its cells: {io.get('violations')}"
           if not io.get("accepted") else None if advice["bound"] and advice["rebuilt"]
           else advice["why"] or "the endpoint's advice isn't bound to the certified I/O or rebuilt from it")
    return {"verdict": "refused" if why else "accepted", "why": why, "warden_run": e.get("warden_run"), "io": io,
            "advice": advice, "advice_bits": advice_bits(b), "at": e.get("at"),
            "in_the_clear": CLEAR["warden_endpoint_served" if e.get("at") == SERVED_NODE else "warden_endpoint"]}


def tie_step(t: dict | None, job: dict | None = None) -> dict:
    """`n2_tie_zk.py prove --class warden`'s summary: accepted when its ``ok`` holds (every link-window's verdict its
    reference's, the tie's on the link, no witness piece leaked), its honest member tied in every link-window it
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


def on_warden(step: dict, doc: dict | None, run_id: str | None, what: str) -> dict:
    """A warden-path step held to the warden run the certificates step read (``run_id``): refused by name
    (`warden_chain.PROXY`) when its input is a certifier rehearsal's (it names no warden run), and when it is another
    warden run's."""
    from warden_chain import PROXY

    if doc is None or step.get("verdict") in ("not run", "not given"):
        return step
    got = doc.get("warden_run")
    why = (f"{what} names no warden run: {PROXY}" if not got
           else f"{what} is on warden run {got}, not the certificates step's {run_id}" if got != run_id else None)
    return step | {"verdict": "refused", "why": why} if why else step


def escaped(name: str, m: dict, s: dict) -> dict | None:
    """A negative member the sampled draw accepted because none of its faulty windows (those its reference expects
    refused) was among the windows drawn: the draw's sampling error, which its ``wrong_units`` at δ bounds, not a
    failure of the run. None for honest, a member refused, or a period not sampled."""
    net = m.get("network") or {}
    units = (((s.get("sampled") or {}).get(name) or {}).get("draw") or {}).get("units") or (m.get("period") or {}).get(
        "units")
    bad = [r["window"] for r in m.get("reference") or [] if not r.get("expect")]
    if name == "honest" or not net.get("accepted") or units is None or set(bad) & set(units):
        return None
    return {"faulty_windows": bad, "drawn": sorted(units)}


def unexpected(name: str, m: dict, escape: dict | None = None) -> list[str]:
    """What keeps an N2 driver's member from its ``ok``: the conditions `n2_zk` and `n2_advised_zk` hold it to, but
    a negative's acceptance when it escaped the draw (`escaped`). A witness piece counts when the scan reads it
    ``leaked`` (`leak_scan.classify`); a ``structured`` find is listed, not counted, as the drivers decide."""
    wins, net, rec = m.get("windows") or [], m.get("network") or {}, m.get("received") or {}
    return ([f"windows {[w.get('window') for w in wins if not w.get('as_expected')]} not as expected"]
            * any(not w.get("as_expected") for w in wins)
            + [f"network {'accepted' if net.get('accepted') else 'refused'}"]
            * (net.get("accepted") != (name == "honest") and escape is None)
            + [f"witness pieces leaked to the verifier: {rec.get('leaked')}"] * bool(rec.get("leaked"))
            + [f"the control (the prover's own {(rec.get('control') or {}).get('file')}) holds "
               f"{(rec.get('control') or {}).get('blobs_found')} of {(rec.get('control') or {}).get('of')} pieces, "
               "so the scan isn't shown to find a leak"] * ((rec.get("control") or {}).get("holds_all") is False))


def n2_step(s: dict | None, certs_doc: dict, job: dict | None = None) -> dict:
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
                         "structured": (m.get("received") or {}).get("structured"), "ok": m.get("ok"),
                         "prove_s": round(sum((w.get("prove") or {}).get("session_s") or 0 for w in wins), 1),
                         "verify_s": round(sum(w.get("verify_s") or 0 for w in wins), 1)}
    honest = members.get("honest") or {}
    draw = s.get("draw") if isinstance(s.get("draw"), dict) else {}
    n = (draw.get("population") or honest.get("windows")
         or (s.get("link") or {}).get("planned_windows", {}).get("honest") or 0)
    drawn = draw.get("k", s.get("drawn", n))
    law = draw.get("law") or s.get("law", "all")
    proved = (bool(members) and all(v["windows"] and not v["unexpected"] for v in members.values())
              and (s.get("ok") is True and rc in (None, "0") or any(v["escaped_the_draw"] for v in members.values())))
    if not proved and honest.get("windows"):
        why = "; ".join(f"{k}: {', '.join(v['unexpected'])}" for k, v in members.items() if v.get("unexpected")) or why
    elif proved:
        gone = [k for k, v in members.items() if v["escaped_the_draw"]]
        why = (f"the driver's ok is false only for {', '.join(gone)}: accepted with none of its faulty windows drawn, "
               "within the draw's bound" if gone else None)
    return {"verdict": ("accepted" if honest.get("network_accepted") else "refused") if proved else
            ("dry" if "members" in s and not honest.get("windows") else "failed"), "why": why, "stand_in": stand_in,
            "ok": s.get("ok"), "unit": (s.get("link") or {}).get("unit"), "members": members,
            "negatives": {k: "escaped the draw" if v["escaped_the_draw"] else v["network_accepted"] is False
                          for k, v in members.items() if k != "honest"},
            "challenger": s.get("challenger"),
            "clocks": s.get("clocks") or {"source": "declared", "by": "plan.json's syncs (n2_zk); the certificates step "
                                                                      "found them equal to the signed ones: "
                                                                      f"{certs_doc.get('signed_are_declared')}"},
            "eps_at_delta": eps_at_delta(n, drawn, law), "stager": s.get("stager"), "in_the_clear": CLEAR["n2"]}


def scan_step(s: dict | None, warden_run: str | None, job: dict | None = None) -> dict:
    """`scan_zk.py prove`'s summary: accepted when the honest round's window is accepted, every other round's (chain,
    forged) refused, no witness piece was found at the verifier, and scan_zk's own ``ok`` holds."""
    job = job or {}
    if s is None or (job.get("rc") or {}).get("scan") not in (None, "0"):
        return {"verdict": "not run" if s is None and not (job.get("rc") or {}).get("scan") else "failed",
                "why": (job.get("why") or {}).get("scan"), "in_the_clear": CLEAR["scan"]}
    rounds = {}
    for name, r in (s.get("rounds") or {}).items():
        a = r.get("audit") or {}
        rounds[name] = {"accepted": a.get("accepted"), "window_accepted": a.get("window_accepted"),
                        "verdicts": a.get("verdicts"), "why": a.get("why"),
                        "wrong_units": (a.get("audit") or {}).get("wrong_units"), "ok": r.get("ok"),
                        "leaked": (a.get("received") or {}).get("verifier_found") or []}
    honest = rounds.get("honest") or {}
    wrong = ([f"{n}: window {'accepted' if v.get('window_accepted') else 'refused'}" for n, v in rounds.items()
              if bool(v.get("window_accepted")) != (n == "honest")]
             + [f"{n}: witness pieces found at the verifier: {v['leaked']}" for n, v in rounds.items() if v["leaked"]]
             + ["scan_zk's ok is not true"] * (s.get("ok") is not True))
    units = 2 * int(s.get("T") or 0)
    k = (s.get("draw") or {}).get("honest") if isinstance(s.get("draw"), dict) else None
    return {"verdict": ("dry" if not rounds else "refused" if not honest.get("window_accepted")
                        else "failed" if wrong else "accepted"),
            "why": "; ".join(wrong) or (job.get("why") or {}).get("scan"), "ok": s.get("ok"), "rounds": rounds,
            "negatives": {k: not v.get("window_accepted") for k, v in rounds.items() if k != "honest"},
            "stand_in": None if warden_run else "scan_zk's own synthetic warden run (scan_traffic), not the gate's "
                                                "rows: its --warden-run takes those",
            "warden_run": warden_run, "eps_at_delta": eps_at_delta(units, int(k or s.get("drawn") or units),
                                                                    f"stratified:{k}" if k else s.get("law") or "all"),
            "in_the_clear": CLEAR["scan"]}


def free_inputs(circuit: dict) -> int:
    """``#(freeInputs \\ reg)`` of a compute audit's window from its ``circuit.json``: every unit's input-row bits (each
    template's units times its statement's ``in_rows`` widths), less each unit's rows whose every port a wire feeds from
    another unit's output (``wires``) or ``external`` names as the system's I/O. A row the registration's anchors
    register still counts, an overcount."""
    from verity.ml.windows.interface import Wire

    units = {t: n for t, n in circuit["templates"]}
    covered: dict[tuple[str, str], set[int] | None] = {}
    for e in circuit["external"]:
        t, port = e.rsplit("/", 1)
        if t not in units:
            raise ValueError(f"external port {e!r} is no template's")
        covered[t, port] = None
    for w in map(Wire.of, circuit["wires"]):
        if w.reader not in units:
            raise ValueError(f"a wire's reader {w.reader!r} is no template")
        fed = covered.setdefault((w.reader, w.port), set())
        if fed is not None:
            fed.update(range(w.at, w.at + w.units))
    bits = 0
    for t, n in units.items():
        for _, _, width, ports in circuit["statements"][t]["in_rows"]:
            fed = [covered.get((t, p), set()) for p in ports]
            gone = 0 if not fed else n if all(f is None for f in fed) else len(set(range(n)).intersection(
                *(set(range(n)) if f is None else f for f in fed)))
            bits += (n - gone) * width
    return bits


def audit_of(d: Path) -> dict:
    """A compute audit's window as one-stage registers it (``registration.json`` and ``circuit.json``): its units, the
    units its law draws (a subset law's ``k``, a stratified law's ``Σ k``, as `NetworkCheck.Period` counts them), the
    least rate it draws a unit at (``q``: ``k/n``, or the least stratum's ``k/n``) and its free inputs
    (`free_inputs`); its jitter term 0 at D_j = 1, `JITTER`: N2 holds the egress counts exactly, and J the offsets
    within a bucket."""
    reg = json.loads((d / "registration.json").read_text())
    law, n = reg["law"], reg["population"]
    if law.get("law") == "subset":
        drawn, q = law["k"], Fraction(law["k"], n)
    elif law.get("law") == "stratified" and law.get("strata"):
        sizes = [sum(hi - lo for lo, hi in s["units"]) for s in law["strata"]]
        drawn, q = sum(s["k"] for s in law["strata"]), min(Fraction(s["k"], m) for s, m in zip(law["strata"], sizes))
        if q == 0 or sum(sizes) != n:
            raise SystemExit(f"--audit {d}: a stratified law whose strata cover its {n} units, each drawn, not {law}")
    else:
        raise SystemExit(f"--audit {d}: K's q is a subset or stratified law's, not {law}")
    return {"window": str(d), "registration": hashlib.sha512(json.dumps(reg, sort_keys=True).encode()).hexdigest(),
            "units": n, "drawn": drawn, "q": str(q),
            "free": free_inputs(json.loads((d / "circuit.json").read_text())), "jitter": 0}


def horizon_k(audits: list[dict], delta: Fraction, network: dict) -> dict:
    """`NetTiming.Guarantees.HorizonLeakAccepted`'s K over one monitoring period: the faulty proof blocks paid once over
    the period, ``Fmax = ⌈ln(1/δ)/q⌉`` with ``q`` the least ``k_i/n_i`` of its compute audits (a stratified audit's
    least stratum's, `audit_of`'s ``q``), so K = the network's own terms (``network``: the advice's and the syncs' bits,
    and the cells') + ``Fmax·16 + ⌈log2 (N + 1)^Fmax⌉ + Σ_i (free_i + jitter_i)``, ``N = Σ_i n_i``, except with
    probability ``δ + Σ_i (e_i + e'_i)``. `HorizonLeakAccepted` is conditional on its per-window inputs, so this K is
    cited as conditional, and its jitter term as `JITTER` (``jitter_term``)."""
    q = min(Fraction(a["q"]) if "q" in a else Fraction(a["drawn"], a["units"]) for a in audits)
    fmax = math.ceil(math.log(1 / float(delta)) / float(q) * (1 + 1e-12))
    n = sum(a["units"] for a in audits)
    terms = {**network, "faulty_bits": OUT_BITS * fmax, "faulty_sets": ((n + 1) ** fmax - 1).bit_length(),
             "free": sum(a["free"] for a in audits), "jitter": sum(a["jitter"] for a in audits)}
    k = sum(terms.values())
    return {"delta": str(delta), "q": str(q), "Fmax": fmax, "units": n, "terms": terms, "K": k,
            "share": k / (OUT_BITS * n), "eps": f"{delta} + Σ_i (e_i + e'_i) over {len(audits)} audits",
            "conditional": HORIZON_CONDITION, "jitter_term": JITTER}


def network_terms(cert: dict, endpoint: dict) -> dict:
    """The network's own terms of K, charged once a period: the advice's residual bits (`advice_bits`; its v0 bits
    when it wasn't rebuilt from the certified I/O), each window's declared sync, log2 #Σ_sync (`LinkWindows.charge`),
    and no cell: no connection passes the certifier but the link, whose counts the advice fixes."""
    bits, syncs = endpoint.get("advice_bits") or {}, cert.get("sync_bits")
    advice = bits.get("derived") if bits.get("derived") is not None else bits.get("v0_total")
    return {"advice": advice, "syncs": None if syncs is None else math.ceil(len(cert.get("rows") or []) * syncs),
            "cells": 0}


def d1_terms(cert: dict, n2: dict, scan: dict, custody_doc: dict, endpoint: dict | None = None,
             audits: list[dict] | None = None) -> dict:
    rows = cert.get("rows") or []
    honest = (n2.get("members") or {}).get("honest") or {}
    acc = [bool(r.get("certified")) and bool(honest.get("network_accepted")) for r in rows]
    network = network_terms(cert, endpoint or {})
    if None in network.values():
        k = {"open": f"the network's terms aren't all computed: {network}"}
    elif not audits:
        k = {"open": "no compute audit's window (--audit: the served units' registration and circuit), whose free "
                     "inputs and draw the K counts", "network": network}
    else:
        k = {f"{d:.3g}": horizon_k(audits, Fraction(d), network) for d in DELTAS} | {"audits": audits}
    return {"theorem": "NetTiming.Guarantees.PeriodLeakAccepted, KS as hypothesis",
            "K": k,
            "sum": "NetTiming.periodLeak_le_sum: Pr[every window accepted ∧ total > a + Σ_w b_w] ≤ Σ_w ε_w",
            "windows": len(rows), "acc": acc, "all_accepted": bool(acc) and all(acc),
            "eps_w": "δ + ε_ks + ε_bind per window (KSLinOn), plus the draw's term at δ (eps_at_delta)",
            "sum_eps": f"{len(rows)}·(δ + ε_ks + ε_bind)",
            "a": "log2 #Ss + Σ_c log2(cap_c + 1), charged once a period",
            "b_w": "#(freeInputs_w \\ reg_w) + log2 #faultySets_w(δ) + β_w + k_w·log2 Dj",
            "lms_height": [g.get("lms_height") for g in custody_doc.get("gate_runs", [])],
            "uncovered": [
                "the I/O tie (IoTieRequest) is checked in the clear, outside the proofs",
                "Claims.Sampling.KS is a hypothesis until Flock.Guarantees.SampledKSLive restates it live",
                "hdata (the wardens' arrivals are the framed egress of the certifier's observations) and Carries "
                "are run conditions, not yet named assumptions",
                "B_reg: the trace's values at reg are the registration's",
                "#(freeInputs \\ reg) is not bounded by the advice bits until V3's padding: b_w is vacuous",
                "no carry across windows: each window's terms stand alone"]}


# --- leak: K per monitoring period, on the warden run ------------------------------------------------------------------

#: core's monitoring periods, seconds by name, and its reporting deltas, as floats (`wipe_monitoring`)
PERIODS = dict(wipe_monitoring.PERIODS)
LEAK_DELTAS = tuple(map(float, wipe_monitoring.DELTAS))
#: the machines a period's count runs over (the census's convention)
MACHINES = (1, 1000)
#: Llama-3.2-1B's parameters (128,256 × 2,048 tied embedding, 16 layers of 60,821,504, the final norm), BF16
WEIGHTS_BITS = 1_235_814_400 * 16


def leak(run: Path, scan: dict | None = None, weights_bits: int = WEIGHTS_BITS) -> dict:
    """K, the bits the network certifier's accepted transcripts can carry about the secret, per monitoring period on
    the warden run ``run`` (a fetched gate run: its ``end.json`` and ``deployment.json``), for the scan's class
    (`scan_zk.CLASS`) and its draw of every unit. The per-period form is `NetTiming.Guarantees.PeriodLeakAccepted`'s
    count, the pooled form `HorizonLeakAccepted`'s at q = 1 (Fmax = ⌈ln(1/δ)⌉), each term in the
    generator reading of the scan's statement: log2 #Ss = 0, the fail cell's
    position log2(egress_cells + 1) a connection, the scan's ``prog`` rows its free inputs (T·NP a link-window as
    registered, an entry's 33 + KO·IDB bits a request if ``prog`` were an input only at its anchored entries,
    `ingress_scan.entry_layout`), no faulty term with
    every unit drawn, b = NS + NV, and Dj = 1. The period's link-windows and requests scale the run's own."""
    import scan_zk as SZ
    from verity.core.protocols.certifier import ingress_scan as IS, warden as W
    dep = _json(next((p for p in (run / "out" / "deployment.json", run / "deployment.json") if p.is_file()), None))
    v = (_json(run / "end.json") or {}).get("verdict") or {}
    if not dep or not v.get("windows") or not v.get("requests"):
        raise SystemExit(f"leak: {run} has no deployment.json, or no end.json verdict with its windows and requests")
    params, cls = W.WardenParams(**dep["params"]), SZ.CLASS
    cls.serves(params)
    T, units = params.window, 2 * params.window
    drawn = (scan or {}).get("draw", {}).get("honest") if isinstance((scan or {}).get("draw"), dict) else units
    if drawn != units:
        raise SystemExit(f"leak: the scan drew {drawn} of {units} units; K's faulty terms are 0 only with every unit "
                         f"drawn (scan_zk --draw {units})")
    NS, NP, NV = IS.state_bits(cls), IS.prog_bits(cls), IS.view_bits(cls)
    cap = math.log2(params.egress_cells + 1)
    anchored = sum(w for _, w in IS.entry_layout(cls))
    rate = v["requests"] / v["windows"]
    window_s = T * params.bucket_ms / 1000
    free = {"registered": {"link_window": T * NP, "request": 0}, "anchors": {"link_window": 0, "request": anchored}}

    def count(link_windows: float, delta: float | None, reading: str) -> float:
        requests = rate * link_windows
        k = requests * cap + link_windows * free[reading]["link_window"] + requests * free[reading]["request"]
        if delta is None:
            return k
        fmax = math.ceil(math.log(1 / delta))
        return k + fmax * math.log2(link_windows * units + 1) + (NS + NV) * fmax

    rows = []
    for period, seconds in PERIODS.items():
        for m in MACHINES:
            lw = m * seconds / window_s
            for reading in free:
                pooled = {f"{d:g}": count(lw, d, reading) for d in LEAK_DELTAS}
                per = count(lw, None, reading)
                rows.append({"period": period, "machines": m, "link_windows": lw, "requests": rate * lw,
                             "prog": reading, "pooled": pooled,
                             "pooled_share": {d: k / weights_bits for d, k in pooled.items()},
                             "per_period": per, "per_period_share": per / weights_bits,
                             "per_link_window": count(1, None, reading)})
    return {"warden_run": (_json(run / "end.json") or {}).get("run") or run.name, "params": dep["params"],
            "class": {k: getattr(cls, k) for k in IS.STATICS},
            "widths": {"NS": NS, "NP": NP, "NV": NV, "egress_row": 8 * cls.egress.R * cls.B},
            "draw": {"units": units, "drawn": drawn},
            "run": {"link_windows": v["windows"], "requests": v["requests"], "requests_per_link_window": rate,
                    "link_window_s": window_s, "accepted": v.get("accepted")},
            "terms": {"log2_Ss": 0, "cap_per_request": cap, "free": free, "faulty": 0, "jitter": 0,
                      "jitter_term": JITTER, "b": NS + NV,
                      "fmax": {f"{d:g}": math.ceil(math.log(1 / d)) for d in LEAK_DELTAS}},
            "weights_bits": weights_bits, "rows": rows,
            "theorems": {"per_period": "NetTiming.Guarantees.PeriodLeakAccepted",
                         "pooled": "HorizonLeakAccepted, conditional: hesc and hleak await SampledKSLive "
                                   "and P2"},
            "conditions": [
                "the generator reading of the scan's statement (each unit's registered rows its boundary values): "
                "a claim, with lean and Daniel",
                "log2 #Ss = 0 only once the schedule's digest binds each connection's admission bucket and lane, "
                "and the server can't move an admission (not met)",
                f"the jitter term is 0 at D_j = 1, {JITTER}, not met until a run shows the paced warden's written "
                "buckets tell a receiver no more than an idle one",
                "the scan proves link-window 0 of the warden run; the period's terms are its per-link-window and "
                "per-request terms at the run's request rate",
                "the loop's (compute audit's) free inputs don't enter K: its δ goes into JointPeriodLive's sum"]}


#: each reading of the scan's free inputs, as the report labels it
READINGS = {"registered": "`prog` as registered today (the record's K)",
            "anchors": "`prog` only at its anchored entries (conditional on the generator-form lowering)"}


def leak_markdown(doc: dict) -> str:
    d1, d2 = (f"{d:g}" for d in LEAK_DELTAS)
    bits = lambda x: f"{x / 10 ** (e := math.floor(math.log10(x))):.3g}·10^{e}" if x >= 1e6 else f"{x:,.0f}"
    share = lambda x: f"{100 * x:.3g}%" if x < 1 else f"{x:.3g}×" if x < 100 else f"{x:,.0f}×"
    lines = []
    for reading, title in READINGS.items():
        lines += [f"**{title}**, bits, and the share of the weights (the jitter term 0, {JITTER}):", "",
                  "| period | machines | K pooled, δ = 1% | K pooled, δ = 1e-6 | K per period | a link-window |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['period']} | {r['machines']:,} | {bits(r['pooled'][d1])} ({share(r['pooled_share'][d1])}) | "
                  f"{bits(r['pooled'][d2])} ({share(r['pooled_share'][d2])}) | {bits(r['per_period'])} "
                  f"({share(r['per_period_share'])}) | {r['per_link_window']:,.0f} |"
                  for r in doc["rows"] if r["prog"] == reading]
        lines.append("")
    return "\n".join(lines)


def merged_n2(paths: list[str] | None) -> dict | None:
    """N2's summaries of one period's members proved in several runs (``--n2 A B …``, each run's draw on its own
    challenger session, every one of the same population and K): one summary, its members and sampled draws their
    union, ``ok`` only when every run's is, and each run's challenger session kept."""
    docs = [d for d in map(_json, [paths] if isinstance(paths, str) else paths or []) if d]
    if len(docs) < 2:
        return docs[0] if docs else None
    draws = {json.dumps({k: (d.get("draw") or {}).get(k) for k in ("law", "k", "population")}) for d in docs}
    if len(draws) > 1:
        raise SystemExit(f"--n2: the runs' draws differ: {sorted(draws)}")
    return {**docs[0], "members": {k: v for d in docs for k, v in (d.get("members") or {}).items()},
            "sampled": {k: v for d in docs for k, v in (d.get("sampled") or {}).items()},
            "challenger": [d.get("challenger") for d in docs], "ok": all(d.get("ok") is True for d in docs),
            "runs": len(docs)}


def stubbed(job: dict, custody_doc: dict, endpoint: dict | None = None) -> list[str]:
    """What a certifier-only report stands on that isn't the chain's own: `STUBBED`, and where its gate runs ran and
    what served them when their ``request.json`` says (demo_chain.sh's); a warden-path report's, `WARDEN_STUBBED`, or
    `WARDEN_SERVED_STUBBED` when its endpoint step (``endpoint``) ran on the served node."""
    if job.get("path") == WARDEN:
        return list(WARDEN_SERVED_STUBBED if (endpoint or {}).get("at") == SERVED_NODE else WARDEN_STUBBED)
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


def decide_period(case: dict) -> dict:
    """``network-period check`` on ``case``: certifier/v0's answer by its verifier (`NetworkCheck.Period`, the
    executable ``VERITY_NETWORK_PERIOD_EXE`` names, which chain_record.sh builds), as `block_certifier.check` decides a
    period."""
    from block_certifier import check

    exe = os.environ.get("VERITY_NETWORK_PERIOD_EXE")
    if not exe:
        raise RuntimeError("no VERITY_NETWORK_PERIOD_EXE: no network-period executable to re-decide the case")
    return check(exe, case)


def period_binding(answer: dict, case: dict, *, cert: dict, n2: dict[int, str], audits: list[dict]) -> str | None:
    """Why certifier/v0's period ``answer`` on ``case`` isn't this chain's, else None. The answer commits to its case's
    commitment alone (`PERIOD_TAG`'s digest of each compute audit's and each link-window's one-stage registration, a
    link-window's naming its index), so each link-window it registers must be one the chain certified (``cert``'s
    rows) and N2 registered, under N2's own registration (``n2``: the honest member's digest by link-window), and its
    audits must be this report's ``--audit`` registrations (``audits``). The case's parameters and evidence are outside
    that digest: its link, certifier session and key must be ``cert``'s, and `block_entry` re-decides the case. The
    rows form's link-windows are the scan's registrations, which no step here holds, so that form is refused."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R
    from verity.ml.commitments.identity import tagged_sha512

    try:
        raw, params = bytes.fromhex(case["commitment"]), case["params"]
        c = json.loads(raw)
    except (KeyError, TypeError, ValueError) as e:
        return f"the case doesn't decode: {type(e).__name__}: {e}"
    if tagged_sha512(PERIOD_TAG, raw).hex() != answer.get("commitment"):
        return "the case isn't the answer's: its commitment's digest is another"
    if "sync_set" not in params:
        return "a rows-form answer: its link-windows are the scan's registrations, which this chain doesn't hold"
    mine = {"link": cert.get("link"), "session": cert.get("session"), "key": (cert.get("key") or {}).get("public")}
    other = [k for k, v in mine.items() if params.get(k) != v]
    if other:
        return f"the case's {', '.join(other)}: not the chain's, so another certifier session's period"
    try:
        regs = [(int(r["window"]["window"]), r) for r in c["windows"]]
    except (KeyError, TypeError, ValueError) as e:
        return f"the commitment's link-windows don't name their index: {type(e).__name__}: {e}"
    certified = {r["window"] for r in cert.get("rows") or [] if r.get("certified")}
    missing = [w for w, _ in regs if w not in certified or w not in n2]
    if missing:
        return (f"the answer covers link-windows {missing} the chain doesn't have: it certified {sorted(certified)} "
                f"and N2 registered {sorted(n2)}")
    foreign = [w for w, r in regs if R.digest(r) != n2[w]]
    if foreign:
        return f"link-windows {foreign}: the answer's registrations aren't N2's on this chain (another period's or run's)"
    if sorted(R.digest(r) for r in c["audits"]) != sorted(R.digest(r) for r in audits):
        return (f"the answer's {len(c['audits'])} compute audits aren't this report's {len(audits)} --audit "
                "registrations")
    return None


def block_entry(answer: dict | None, case: dict | None, verdict: str, *, server: str | None, runs: list[str],
                chain: dict) -> dict:
    """The block record's certifier entry: `block_certifier.block_leg` of certifier/v0's period ``answer`` on
    ``server``, from ``runs``. Written only when the chain itself is accepted, the answer is held to it (`period_binding` on ``chain``:
    its certificates, N2's registrations and the ``--audit`` registrations) and ``network-period check`` gives exactly
    that answer on its ``case`` (`decide_period`), so neither a refused chain nor another period's answer carries a K;
    else why not, `block_leg`'s refusal included."""
    from block_certifier import block_leg

    why = (NO_PERIOD_ANSWER if answer is None else
           f"the chain is {verdict}: only an accepted chain carries a K" if verdict != "accepted" else
           "no --server: the node the block ran on" if not server else
           "no case: what the answer commits to can't be held to the chain" if case is None else
           period_binding(answer, case, **chain))
    if why:
        return {"written": False, "why": why}
    try:
        again = decide_period(case)
    except (OSError, RuntimeError) as e:
        return {"written": False, "why": f"network-period check on the case: {e}"}
    if again != answer:
        return {"written": False, "why": f"network-period check on the case answers {again.get('verdict')} with K "
                f"{(again.get('public') or {}).get('K')}, not the given answer: it isn't the verifier's on this case"}
    try:
        return {"written": True, "entry": block_leg(answer, server=server, runs=runs, source="real")}
    except ValueError as e:
        return {"written": False, "why": f"block_leg: {e}"}


def period_block(a, verdict: str, runs: list[str], *, cert: dict, n2: dict | None, audit_dirs: list[Path]) -> dict:
    """`block_entry` on the report's ``--period-answer``, its case (``--period-case``, by default ``X.case.json``
    beside ``X.answer.json``) and ``--server``."""
    pa = getattr(a, "period_answer", None)
    if pa and not Path(pa).is_file():
        return {"written": False, "why": f"--period-answer {pa}: not a file"}
    pc = getattr(a, "period_case", None) or (str(pa).removesuffix(".answer.json") + ".case.json" if pa else None)
    if pa and not Path(pc).is_file():
        return {"written": False, "why": f"no case at {pc} (--period-case): what the answer commits to can't be held "
                                         "to the chain"}
    honest = ((n2 or {}).get("members") or {}).get("honest") or {}
    chain = {"cert": cert, "n2": {int(w["window"]): w.get("registration") for w in honest.get("windows") or []
                                  if "window" in w},
             "audits": [json.loads((d / "registration.json").read_text()) for d in audit_dirs]}
    return block_entry(_json(pa), _json(pc), verdict, server=getattr(a, "server", None), runs=runs, chain=chain)


def report(a) -> dict:
    cust, cert = _json(a.custody) or {}, _json(a.certs) or {}
    job = _json(a.job) or {}
    certifier_only = job.get("path") == CERTIFIER_ONLY
    warden = job.get("path") == WARDEN
    scan_only = a.n2 is None and not certifier_only and not warden
    endpoint = (warden_endpoint_step if warden else endpoint_step)(_json(a.endpoint))
    merged = None if scan_only else merged_n2(a.n2)
    n2 = ({"verdict": "not given", "why": SCAN_ONLY, "in_the_clear": CLEAR["n2"]} if scan_only
          else n2_step(merged, cert, job))
    scan = ({"verdict": "not run", "why": STUBBED[0], "in_the_clear": CLEAR["scan"]} if certifier_only
            else scan_step(_json(a.scan), a.warden_run, job))
    certificates = {k: v for k, v in cert.items() if k != "rows"}
    tie = None
    if warden:
        run_id, tie_doc = cert.get("warden_run"), _json(getattr(a, "tie", None))
        certificates = on_warden(certificates, _json(a.certs), run_id, "the certificates step's input")
        endpoint = on_warden(endpoint, _json(a.endpoint), run_id, "the endpoint's report")
        n2 = on_warden(n2, merged, run_id, "N2's summary") | {
            "clocks": {"source": "none", "by": "the warden link declares no sync and signs no clock (n2_zk "
                                               "--warden-run)"}}
        tie = on_warden(tie_step(tie_doc, job), tie_doc, run_id, "the tie's summary")
        scan = on_warden(scan, _json(a.scan), run_id, "the scan's summary")
    proved = ((scan_only or n2.get("verdict") == "accepted")
              and (certifier_only or scan.get("verdict") == "accepted")
              and (tie is None or tie.get("verdict") == "accepted"))
    steps = {"custody": cust, "certificates": certificates, "endpoint": endpoint, **({"tie": tie} if warden else {}),
             "n2": n2, "scan": scan, "lean": {"verdict": "accepted" if proved else (scan if scan_only else n2).get(
                                        "verdict"),
                                    "binary": job.get("lean"), "in_the_clear": CLEAR["lean"],
                                    "n2_windows_proved": ((n2.get("members") or {}).get("honest") or {}).get(
                                        "lean_proved"),
                                    "scan_rounds": {k: v.get("verdicts") for k, v in scan.get("rounds", {}).items()}}}
    required = (("custody", "certificates", "endpoint", "tie", "n2", "scan") if warden else
                ("custody", "certificates", *(("endpoint",) if job.get("profile") == ADVICE else ()),
                 *(() if scan_only else ("n2",)), *(() if certifier_only else ("scan",))))
    accepted = (all(steps[k].get("verdict") == "accepted" for k in required)
                and not any(steps[k].get("stand_in") for k in ("n2", "scan")))
    rehearsal = {f"rehearsal/{g['gate_run']}/{s}": v == "refused" for g in cust.get("gate_runs") or []
                 for s, v in (g.get("strategies") or {}).items() if s not in REHEARSAL_ACCEPTS}
    negatives = {**rehearsal, **{f"tie/{k}": v for k, v in endpoint.get("negatives", {}).items()},
                 **{f"tie/{k}": v for k, v in (tie or {}).get("negatives", {}).items()},
                 **{f"n2/{k}": v for k, v in n2.get("negatives", {}).items()},
                 **{f"scan/{k}": v for k, v in scan.get("negatives", {}).items()}}
    audit_dirs = [Path(d) for d in getattr(a, "audit", None) or []]
    audits = [audit_of(d) for d in audit_dirs]
    verdict = ("scan " if scan_only else "") + ("accepted" if accepted else "refused")
    runs = list(dict.fromkeys(r for r in (job.get("run"), *(g.get("gate_run") for g in cust.get("gate_runs") or []))
                              if r))
    return {"schema": SCHEMA, "job": job, "verdict": verdict, "path": job.get("path"),
            "scope": SCAN_ONLY if scan_only else None, "stubbed": stubbed(job, cust, endpoint),
            "negatives_refused": negatives, "steps": steps,
            "d1": d1_terms(cert, n2, scan, cust, endpoint, audits),
            "block": period_block(a, verdict, runs, cert=cert, n2=merged, audit_dirs=audit_dirs)}


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
        eps = (f"{e['law']}, {e['drawn']} of {e['units']}: wrong ≤ {e['wrong_units_at_delta']}" if e else "")
        lines.append(f"| {k} | {s[k].get('verdict')} | {eps} | {s[k].get('in_the_clear', '')} |")
    notes = [f"- {k}: {s[k][f]}" for k in ("endpoint", "tie", "n2", "scan") if k in s for f in ("why", "stand_in")
             if s[k].get(f)]
    ep = s["endpoint"]
    if ep.get("advice"):
        notes.append(f"- endpoint: advice {ep['advice']}; " + (f"I/O tie on honest {ep['tie']}" if "tie" in ep else
                                                                f"certified I/O {ep.get('io')}"))
    if s.get("tie", {}).get("tie"):
        notes.append(f"- tie: link-windows tied by member {s['tie']['tie']}")
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
                         f"the jitter term {JITTER}; {h['conditional']}")
    lines += ["- uncovered:"] + [f"  - {u}" for u in d["uncovered"]]
    b = r.get("block") or {}
    if b.get("written"):
        e, q = b["entry"], b["entry"]["quantity"]
        at = "; ".join(f"at δ = {delta}: " + (f"K = {x['value']}" if x["holds"] else "not established")
                       for delta, x in e["at"].items())
        lines += ["", "## Block record", "", f"- certifier entry on {e['server']}: K = {q['value']} bits over "
                  f"{q['windows']} link-windows, error {e['error']} ({e['error_from']}); {at}; runs {', '.join(e['runs'])}"]
    else:
        lines += ["", "## Block record", "", f"- no certifier entry: {b.get('why')}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("custody", help="each gate run's files against its end.json")
    c.add_argument("--gates", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("certs", help="the client's kept certificates and clock certificates, verified")
    which = v.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", type=Path, help="the certifier rehearsal's honest directory")
    which.add_argument("--warden-run", type=Path, help="the warden path: an active warden's run (warden_chain)")
    v.add_argument("--client", type=Path, help="the client's kept files (default --run)")
    v.add_argument("--deployed", action="store_true", help="refuse a key that isn't LMS at lms.DEPLOYED")
    v.add_argument("--out", required=True, type=Path)
    e = sub.add_parser("endpoint",
                       help="the warden path: the endpoint's own advice bound to the warden's certified I/O")
    e.add_argument("--warden-run", required=True, type=Path)
    e.add_argument("--endpoint", required=True, type=Path, help="the endpoint's records (network-e2e chain.ENDPOINT)")
    e.add_argument("--eos", required=True,
                   help="the model's EOS token ids, comma-separated (a warden run records none)")
    e.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("report", help="one report over the steps' outputs")
    for f in ("custody", "certs", "endpoint", "scan", "job"):
        r.add_argument(f"--{f}")
    r.add_argument("--tie", help="the warden path: n2_tie_zk.py prove --class warden's summary")
    r.add_argument("--n2", nargs="*", help="N2's summary, or one per run of a period's members (merged_n2); without "
                                           "it, the scan's part alone (SCAN_ONLY)")
    r.add_argument("--warden-run")
    r.add_argument("--audit", nargs="*", help="the served units' compute audit windows (registration.json and "
                                              "circuit.json each), whose draw and free inputs d1's K counts")
    r.add_argument("--period-answer", help="certifier/v0's network-period check answer on the period "
                                           "(certifier.answer.json), the block record's certifier entry's")
    r.add_argument("--period-case", help="the answer's case (default X.case.json beside X.answer.json), held to this "
                                         "chain and re-decided by network-period check")
    r.add_argument("--server", help="the node the block ran on (the block record's server)")
    r.add_argument("--out", required=True, type=Path)
    k = sub.add_parser("leak", help="K per monitoring period on the warden run, for the scan's class and draw")
    k.add_argument("--warden-run", required=True, type=Path, help="the fetched warden gate run (its end.json)")
    k.add_argument("--scan", help="the scan's summary on that run (its draw); default every unit drawn")
    k.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.command == "custody":
        doc = custody(a.gates)
    elif a.command == "certs" and a.warden_run:
        import warden_chain as WC
        doc = WC.certificates(a.warden_run, a.deployed) | {"in_the_clear": CLEAR["warden_certificates"]}
    elif a.command == "certs":
        doc = certs(a.run, a.client or a.run, a.deployed)
    elif a.command == "endpoint":
        doc = warden_endpoint(a.warden_run, a.endpoint, [int(x) for x in a.eos.split(",")])
        doc["verdict"] = warden_endpoint_step(doc)["verdict"]
        a.out.parent.mkdir(parents=True, exist_ok=True)
    elif a.command == "leak":
        doc = leak(a.warden_run, _json(a.scan))
        a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
        a.out.with_suffix(".md").write_text(leak_markdown(doc))
        print(leak_markdown(doc))
        return 0
    else:
        doc = report(a)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "report.md").write_text(markdown(doc))
        if doc["block"]["written"]:
            (a.out / "block.json").write_text(json.dumps(doc["block"]["entry"], indent=1, sort_keys=True))
        a.out = a.out / "report.json"
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: doc.get(k) for k in ("verdict", "scope", "why", "negatives_refused") if k in doc}))
    return 0 if doc.get("verdict") in ("accepted", "scan accepted") else 1


if __name__ == "__main__":
    sys.exit(main())
