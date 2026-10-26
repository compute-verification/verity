"""chain-record: the network chain's run of record, from the gate's fetched tarballs to one report
(`chain_record.sh` runs it on a CPU machine). Each step reads what the step before it kept; none drives an endpoint.

    python benchmarks/network_traces/chain_record.py custody --gates DIR --out FILE
    python benchmarks/network_traces/chain_record.py certs --run DIR [--client DIR] [--deployed] --out FILE
    python benchmarks/network_traces/chain_record.py report --custody FILE --certs FILE [--endpoint REPORT]
      [--n2 SUMMARY ...] [--scan SUMMARY] [--job FILE] --out DIR
    python benchmarks/network_traces/chain_record.py leak --warden-run DIR [--scan SUMMARY] --out FILE

* **custody**: each gate run under ``--gates`` (a directory holding the gate's ``end.json``, as `vy-challenger-gate
  fetch` writes it) holds the files its ``end.json`` names, each with the SHA-512 it records. In the clear, of public
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
  the clear, and the period's terms for `NetTiming.Guarantees.PeriodLeakAccepted` (#4326) summed by
  `periodLeak_le_sum` (#4324). ``--endpoint`` is network-e2e's ``--no-prove`` report on the run and the endpoint's
  records: the endpoint's advice bound to the run and the I/O tie, in the clear, required on the advice profile. N2's
  summary is `n2_advised_zk.py prove --e2e`'s on the advice profile and `n2_zk.py prove`'s on the others (Selection
  All: every link-window's one unit proved) or, once it lands, N2-ZK's sampled draw's, read by the same fields. Scan's
  is `scan_zk.py prove`'s, a stand-in on its own synthetic warden run until IO-TIE's ``--warden-run`` takes the gate's
  warden rows. A step on a stand-in (N2 on ``--job``'s ``stand_in``, scan without a warden run) never makes the report
  accepted.
* **leak**: K per monitoring period (week, month, year; one machine and 1,000) on the warden gate run, pooled at
  δ = 1% and 10⁻⁶ and in the per-period form, each beside its share of Llama-3.2-1B's BF16 weights (`leak`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

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
            "(IoTieRequest) stays in the clear tonight, outside the proofs",
    "lean": "nothing: each proof is decided by the verifier's own Lean `flock-verify verify --zk` run",
}


def _json(path) -> dict | None:
    p = Path(path) if path else None
    return json.loads(p.read_text()) if p and p.is_file() else None


def subset_bound(n: int, k: int, delta: float) -> int | None:
    """The fewest wrong units b such that an accepted draw of k of n units, uniform without replacement, hides more
    than b except with probability δ: the least b with C(n − b − 1, k) / C(n, k) ≤ δ. Selection All (k = n) is 0."""
    if n <= 0 or not 0 < k <= n:
        return None
    total = math.comb(n, k)
    return next(b for b in range(n) if math.comb(n - b - 1, k) <= delta * total)


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
                     "profile": summary.get("profile")})
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
                         "deployed": tuple(dep["certifier_key"]["params"]) == (lms.DEPLOYED.N, lms.DEPLOYED.W,
                                                                               lms.DEPLOYED.H)},
                 "link": lw.link, "in_the_clear": CLEAR["certificates"]}
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
    (`chain.tie`), both in the clear."""
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
    return {"verdict": "accepted" if ok else "refused", "advice": advice, "tie": tie | {"ran": t.get("ran"),
            "stand_ins": t.get("stand_ins")}, "in_the_clear": CLEAR["endpoint"]}


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
    a negative's acceptance when it escaped the draw (`escaped`)."""
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
    job = job or {}
    if s is None or (job.get("rc") or {}).get("scan") not in (None, "0"):
        return {"verdict": "not run" if s is None and not (job.get("rc") or {}).get("scan") else "failed",
                "why": (job.get("why") or {}).get("scan"), "in_the_clear": CLEAR["scan"]}
    rounds = {}
    for name, r in (s.get("rounds") or {}).items():
        a = r.get("audit") or {}
        rounds[name] = {"accepted": a.get("accepted"), "window_accepted": a.get("window_accepted"),
                        "verdicts": a.get("verdicts"), "why": a.get("why"),
                        "wrong_units": (a.get("audit") or {}).get("wrong_units"), "ok": r.get("ok")}
    honest = rounds.get("honest") or {}
    units = 2 * int(s.get("T") or 0)
    k = (s.get("draw") or {}).get("honest") if isinstance(s.get("draw"), dict) else None
    return {"verdict": ("accepted" if honest.get("window_accepted") else "refused") if rounds else "dry",
            "ok": s.get("ok"), "rounds": rounds,
            "negatives": {k: not v.get("window_accepted") for k, v in rounds.items() if k != "honest"},
            "stand_in": None if warden_run else "scan_zk's own synthetic warden run (scan_traffic), not the gate's "
                                                "rows: IO-TIE's --warden-run takes those",
            "warden_run": warden_run, "eps_at_delta": eps_at_delta(units, int(k or s.get("drawn") or units),
                                                                    f"stratified:{k}" if k else s.get("law") or "all"),
            "in_the_clear": CLEAR["scan"]}


def d1_terms(cert: dict, n2: dict, scan: dict, custody_doc: dict) -> dict:
    rows = cert.get("rows") or []
    honest = (n2.get("members") or {}).get("honest") or {}
    acc = [bool(r.get("certified")) and bool(honest.get("network_accepted")) for r in rows]
    return {"theorem": "NetTiming.Guarantees.PeriodLeakAccepted (#4326), KS as hypothesis",
            "sum": "NetTiming.periodLeak_le_sum (#4324): Pr[every window accepted ∧ total > a + Σ_w b_w] ≤ Σ_w ε_w",
            "windows": len(rows), "acc": acc, "all_accepted": bool(acc) and all(acc),
            "eps_w": "δ + ε_ks + ε_bind per window (KSLinOn), plus the draw's term at δ (eps_at_delta)",
            "sum_eps": f"{len(rows)}·(δ + ε_ks + ε_bind)",
            "a": "log2 #Ss + Σ_c log2(cap_c + 1), charged once a period",
            "b_w": "#(freeInputs_w \\ reg_w) + log2 #faultySets_w(δ) + β_w + k_w·log2 Dj",
            "lms_height": [g.get("lms_height") for g in custody_doc.get("gate_runs", [])],
            "uncovered": [
                "the I/O tie (IoTieRequest) is checked in the clear tonight, outside the proofs",
                "Claims.Sampling.KS is a hypothesis until Flock.Guarantees.SampledKSLive restates it live",
                "hdata (the wardens' arrivals are the framed egress of the certifier's observations) and Carries "
                "are run conditions, not yet named assumptions",
                "B_reg: the trace's values at reg are the registration's (proofs' #4253)",
                "#(freeInputs \\ reg) is not bounded by the advice bits until V3's padding (NET-68): b_w is vacuous",
                "no carry across windows: each window's terms stand alone"]}


# --- leak: K per monitoring period, on the warden run ------------------------------------------------------------------

#: monitoring periods in minutes, and the machines a period's count runs over (memory-accounting's census convention)
PERIODS = {"week": 10_080, "month": 43_200, "year": 525_600}
MACHINES = (1, 1000)
LEAK_DELTAS = (0.01, 1e-6)
#: Llama-3.2-1B's parameters (128,256 × 2,048 tied embedding, 16 layers of 60,821,504, the final norm), BF16
WEIGHTS_BITS = 1_235_814_400 * 16


def leak(run: Path, scan: dict | None = None, weights_bits: int = WEIGHTS_BITS) -> dict:
    """K, the bits the network certifier's accepted transcripts can carry about the secret, per monitoring period on
    the warden run ``run`` (a fetched gate run: its ``end.json`` and ``deployment.json``), for the scan's class
    (`scan_zk.CLASS`) and its draw of every unit. The per-period form is `NetTiming.Guarantees.PeriodLeakAccepted`'s
    count, the pooled form `HorizonLeakAccepted`'s at q = 1 (Fmax = ⌈ln(1/δ)⌉), each term in note
    network-accounting/20261025T2250Z-finding-warden-form-k's generator reading: log2 #Ss = 0, the fail cell's
    position log2(egress_cells + 1) a connection, the scan's ``prog`` rows its free inputs (T·NP a link-window as
    registered, 32 + KO·IDB a request if ``prog`` were an input only at its anchored entries), no faulty term with
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
    anchored = 32 + cls.KO * cls.IDB
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
    for period, minutes in PERIODS.items():
        for m in MACHINES:
            lw = m * minutes * 60 / window_s
            for reading in free:
                pooled = {f"{d:g}": count(lw, d, reading) for d in LEAK_DELTAS}
                per = count(lw, None, reading)
                rows.append({"period": period, "machines": m, "link_windows": lw, "requests": rate * lw,
                             "prog": reading, "pooled": pooled,
                             "pooled_share": {d: k / weights_bits for d, k in pooled.items()},
                             "per_period": per, "per_period_share": per / weights_bits,
                             "per_link_window": count(1, None, reading)})
    return {"warden_run": (_json(run / "end.json") or {}).get("run") or run.name, "params": dep["params"], "class": {k: getattr(cls, k) for k in ("B", "r", "lanes", "C", "IDB", "EVERY",
                                                                             "WITHIN", "D", "KO")},
            "widths": {"NS": NS, "NP": NP, "NV": NV, "egress_row": 8 * cls.egress.R * cls.B},
            "draw": {"units": units, "drawn": drawn},
            "run": {"link_windows": v["windows"], "requests": v["requests"], "requests_per_link_window": rate,
                    "link_window_s": window_s, "accepted": v.get("accepted")},
            "terms": {"log2_Ss": 0, "cap_per_request": cap, "free": free, "faulty": 0, "jitter": 0,
                      "b": NS + NV, "fmax": {f"{d:g}": math.ceil(math.log(1 / d)) for d in LEAK_DELTAS}},
            "weights_bits": weights_bits, "rows": rows,
            "theorems": {"per_period": "NetTiming.Guarantees.PeriodLeakAccepted (#4326, #4349, #4373, #4370)",
                         "pooled": "HorizonLeakAccepted (#4367), conditional: hesc and hleak await SampledKSLive "
                                   "and P2"},
            "conditions": [
                "the generator reading of the scan's statement (each unit's registered rows its boundary values): "
                "a claim, with lean and Daniel",
                "log2 #Ss = 0 only once the schedule's digest binds each connection's admission bucket and lane, "
                "and the server can't move an admission (not met today)",
                "the scan proves link-window 0 of the warden run; the period's terms are its per-link-window and "
                "per-request terms at the run's request rate",
                "the loop's (compute audit's) free inputs don't enter K: its δ goes into JointPeriod's sum"]}


#: each reading of the scan's free inputs, as the report labels it
READINGS = {"registered": "`prog` as registered today (the record's K)",
            "anchors": "`prog` only at its anchored entries (conditional on the generator-form lowering, LEAN-49)"}


def leak_markdown(doc: dict) -> str:
    d1, d2 = (f"{d:g}" for d in LEAK_DELTAS)
    bits = lambda x: f"{x / 10 ** (e := math.floor(math.log10(x))):.3g}·10^{e}" if x >= 1e6 else f"{x:,.0f}"
    share = lambda x: f"{100 * x:.3g}%" if x < 1 else f"{x:.3g}×" if x < 100 else f"{x:,.0f}×"
    lines = []
    for reading, title in READINGS.items():
        lines += [f"**{title}**, bits, and the share of the weights:", "",
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


def report(a) -> dict:
    cust, cert = _json(a.custody) or {}, _json(a.certs) or {}
    job = _json(a.job) or {}
    endpoint = endpoint_step(_json(a.endpoint))
    n2 = n2_step(merged_n2(a.n2), cert, job)
    scan = scan_step(_json(a.scan), a.warden_run, job)
    steps = {"custody": cust, "certificates": {k: v for k, v in cert.items() if k != "rows"}, "endpoint": endpoint,
             "n2": n2, "scan": scan, "lean": {"verdict": "accepted" if n2.get("verdict") == "accepted"
                                    and scan.get("verdict") == "accepted" else n2.get("verdict"),
                                    "binary": job.get("lean"), "in_the_clear": CLEAR["lean"],
                                    "n2_windows_proved": ((n2.get("members") or {}).get("honest") or {}).get(
                                        "lean_proved"),
                                    "scan_rounds": {k: v.get("verdicts") for k, v in scan.get("rounds", {}).items()}}}
    required = ("custody", "certificates", *(("endpoint",) if job.get("profile") == ADVICE else ()), "n2", "scan")
    accepted = (all(steps[k].get("verdict") == "accepted" for k in required)
                and not any(steps[k].get("stand_in") for k in ("n2", "scan")))
    negatives = {**{f"n2/{k}": v for k, v in n2.get("negatives", {}).items()},
                 **{f"scan/{k}": v for k, v in scan.get("negatives", {}).items()}}
    return {"schema": SCHEMA, "job": job, "verdict": "accepted" if accepted else "refused",
            "negatives_refused": negatives, "steps": steps, "d1": d1_terms(cert, n2, scan, cust)}


def markdown(r: dict) -> str:
    j, s = r["job"], r["steps"]
    lines = [f"# Network chain run {j.get('run', '?')}", "",
             f"Tree {j.get('tree', '?')}, gates {', '.join(j.get('gates', [])) or '?'}. Verdict: **{r['verdict']}**.",
             "", "| step | verdict | ε at δ | in the clear |", "|---|---|---|---|"]
    for k in ("custody", "certificates", "endpoint", "n2", "scan", "lean"):
        e = s[k].get("eps_at_delta") or {}
        eps = (f"{e['law']}, {e['drawn']} of {e['units']}: wrong ≤ {e['wrong_units_at_delta']}" if e else "")
        lines.append(f"| {k} | {s[k].get('verdict')} | {eps} | {s[k].get('in_the_clear', '')} |")
    notes = [f"- {k}: {s[k][f]}" for k in ("endpoint", "n2", "scan") for f in ("why", "stand_in") if s[k].get(f)]
    ep = s["endpoint"]
    if ep.get("advice"):
        notes.append(f"- endpoint: advice {ep['advice']}; I/O tie on honest {ep['tie']}")
    lines += ["", *notes, "", f"Negatives refused: {r['negatives_refused']}", "", "## D1", ""]
    d = r["d1"]
    lines += [f"- {d['theorem']}; {d['sum']}", f"- windows {d['windows']}, all accepted {d['all_accepted']}",
              f"- Σ ε_w = {d['sum_eps']}", "- uncovered:"] + [f"  - {u}" for u in d["uncovered"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("custody", help="each gate run's files against its end.json")
    c.add_argument("--gates", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("certs", help="the client's kept certificates and clock certificates, verified")
    v.add_argument("--run", required=True, type=Path, help="the certifier rehearsal's honest directory")
    v.add_argument("--client", type=Path, help="the client's kept files (default --run)")
    v.add_argument("--deployed", action="store_true", help="refuse a certifier key that isn't LMS at lms.DEPLOYED")
    v.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("report", help="one report over the steps' outputs")
    for f in ("custody", "certs", "endpoint", "scan", "job"):
        r.add_argument(f"--{f}")
    r.add_argument("--n2", nargs="*", help="N2's summary, or one per run of a period's members (merged_n2)")
    r.add_argument("--warden-run")
    r.add_argument("--out", required=True, type=Path)
    k = sub.add_parser("leak", help="K per monitoring period on the warden run, for the scan's class and draw")
    k.add_argument("--warden-run", required=True, type=Path, help="the fetched warden gate run (its end.json)")
    k.add_argument("--scan", help="the scan's summary on that run (its draw); default every unit drawn")
    k.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.command == "custody":
        doc = custody(a.gates)
    elif a.command == "certs":
        doc = certs(a.run, a.client or a.run, a.deployed)
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
        a.out = a.out / "report.json"
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: doc.get(k) for k in ("verdict", "why", "negatives_refused") if k in doc}))
    return 0 if doc.get("verdict") == "accepted" else 1


if __name__ == "__main__":
    sys.exit(main())
