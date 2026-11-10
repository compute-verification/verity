"""onsite: the onsite demo's network certifier path at each zone's boundary of the mock datacenter, and each zone's
verdict and charge in a record labelled simulated. Nothing here is served: the endpoint is the stand-in
`demo_stub.py`, and the zones, their network certifiers and the fabric are network namespaces of one host.

Each zone runs the onsite demo's own code, unforked (`benchmarks/network_traces/certifier_piece.py`, on the
active-warden path):
- **the endpoint**: `demo_stub.py serve` in the zone's namespace, on its uplink's address at `PORT`, keeping its
  records (the serve dir) in KEEP/<zone>/serve;
- **the tunnel**: `certifier_piece.relay` in the zone's certifier namespace, from that namespace's loopback `PORT` to
  the endpoint, standing in for the gate's ssh tunnel to the node's port;
- **the front**: `certifier_piece.py front PORT SERVE_DIR OUT/<zone>/front --gate local` in the certifier namespace.
  This tree's gate runs its scan rehearsal there, its network certifier (`warden_serve.py rehearse`) and client with it,
  so every answer reaches the network certifier across the zone's uplink. The gate's fetch is kept under the serve dir,
  and OUT/<zone>/front gets the front's public outcome (`session.json`, `bundle.tgz`);
- **the piece**: `certifier_piece.py piece SERVE_DIR WINDOW OUT/<zone>/piece`, the verdict path. It audits the kept
  run's records as rows (`network-rows`), runs the chain on the kept run (certs, endpoint, N2, tie and the scan at the
  run's class), then the period check (`network-period`), whose answer is the zone's verdict. WINDOW is the endpoint's
  Commit period holding the front's requests (`window`). The network certifier's key and session are the auditor's
  record's (``--certifier-key``), the lab's own: its local gate's copy of the run (`auditor_record`).

`front` runs every zone at once on a lab that is up in its shape and idle (`mockdc.py up`), and writes OUT/front.json.
`piece` copies the front's outcome from ``--front`` (the front's OUT) into its own OUT, runs each zone's piece in turn
and writes the record; it needs no namespace, but its chain needs hidden_zk's builds (`chain_record.sh`), so it runs
from a clone in the network slots. `record` reads only what the front and the piece
released, and writes OUT/record.json and OUT/record.md: each zone's verdicts and its bound's terms in
`PeriodLeakAccepted`'s symbols (`TERMS`), per monitoring period, for each zone, the lab's zones summed, the census's
datacenter and the world, labelled simulated with the premises the lab leaves unmet (`SIMULATED`).

    python onsite.py front --out DIR --keep DIR [--blocks 4 ...]
    python onsite.py piece --out DIR --keep DIR [--front DIR] [--zone b0 ...] [--lean EXE] [CR_K=V ...]
    python onsite.py record --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mockdc  # noqa: E402
import graph  # noqa: E402
import records  # noqa: E402
import certifier_piece as CP  # noqa: E402
import chain_record as CR  # noqa: E402
import leak_bounds as LB  # noqa: E402
from verity.core.protocols.certifier import ingress_scan as IS  # noqa: E402

TRACES = mockdc.ROOT / "benchmarks" / "network_traces"
#: the endpoint's port in its zone, and the front's on its certifier's loopback
PORT = CP.SS.PORT
#: the stand-in's vocabulary, at most a byte; its EOS id is the vocabulary's size, which no answer holds
VOCAB = 256
#: the scan class a front's gate runs at when its request names none: the gate's class of record
GATE_CLASS = "w63"
#: the reading of the sampling block's free inputs the record's K is in (`leak_bounds.READINGS`)
READING = "registered"
#: `NetTiming.Guarantees.PeriodLeakAccepted`'s terms, in the order the bound adds them
TERMS = ("k_sched", "k_fail", "ell_a", "k_skip", "jitter")
BOUND = "k_exfil(Y) ≤ k_sched + k_fail + Σ_j (ℓ_a,j + k_skip,j + T_j·log2 d_jit)"
#: δ_net's terms beside the period check's eps, each once an execution phase, as `PeriodLeakAccepted` bounds a phase's
#: failure by δ + ε_ks + ε_bind + ε_tap: the knowledge error (`Claims.Sampling.KS`, its hypothesis), the binding error
#: (`Claims.Sampling.Bext`) and the collisions with the network certifier's openings (`NetTiming.Btap`). No check
#: computes them, so δ_net has no number.
ERRORS = ("ε_ks", "ε_bind", "ε_tap")
#: the period check's two holds of a tied compute audit that K's ℓ_a,j waits on where a phase's scan is sampled in part
#: and the phase is charged its tied audits' counted free inputs; until both are Lean refusals, K charged so is
#: conditional
OPEN_CHECKS = ("free counted off the tied audit's registered descriptor",
               "each phase's window circuit held to the registered class circuit")
#: what the lab leaves unmet of the onsite demo's placement, each a reason the record is simulated and never served
SIMULATED = (
    "one host: every zone, its network certifier and the fabric are network namespaces of one Linux host "
    "(mockdc.py up), with netem's delays on the fabric",
    "the endpoint is demo_stub.py, a stand-in: its answers are SHA-512's ids, not a model's, and it commits no circuit",
    "trusted-host unmet: each network certifier ran in its zone's certifier namespace on the prover's host (the "
    "front's --gate local), no auditor's device",
    "the tunnel is a relay from the certifier namespace's loopback to the zone's endpoint, standing in for the gate's "
    "ssh tunnel to the node",
    "the gate's client runs in the certifier namespace, as the onsite gate's does on its host, so no request crosses "
    "the fabric",
    "the chain's checks and draws, the period check and their proofs' coins ran on the lab's host (the piece's "
    "placement), the verifier's own coins",
    "no compute audit (V3c) of the served window: the scan's prog is tied to no audit, so the period check refuses the "
    "honest case at the wiring",
)


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def dirs(args) -> tuple[Path, Path | None]:
    """OUT, published by the run, and KEEP, private (0700), outside OUT."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    keep = getattr(args, "keep", None)
    if keep is None:
        return out, None
    keep = Path(keep).resolve()
    if keep == out or out in keep.parents:
        raise SystemExit(f"--keep {keep} is inside --out, which the run publishes: give a directory outside it")
    keep.mkdir(mode=0o700, parents=True, exist_ok=True)
    keep.chmod(0o700)
    return out, keep


# ---------------------------------------------------------------------------------------------------------------- front

def tunnel(args) -> None:
    """The lab's tunnel, in a certifier's namespace: `certifier_piece.relay` from its loopback to the zone's endpoint,
    printing one line once it listens, until SIGTERM ends it."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    CP.relay(args.listen, args.to, host=args.host, ready=lambda: print(json.dumps({"listening": args.listen}),
                                                                        flush=True))


def start(ns: str, argv: list[str], log: Path, *, piped: bool = True) -> subprocess.Popen:
    """``argv`` in namespace ``ns`` as the invoking user (`mockdc.netns_exec`), its stderr in ``log``, and its stdout
    piped (`ready` reads its first line) or in ``log`` too."""
    with open(log, "w") as err:
        return subprocess.Popen(mockdc.netns_exec(ns, argv), stdout=subprocess.PIPE if piped else err,
                                stderr=err if piped else subprocess.STDOUT, text=True)


def ready(p: subprocess.Popen, what: str, log: Path) -> dict:
    line = p.stdout.readline()
    if not line:
        raise SystemExit(f"{what} exited before it was ready ({p.wait()}): {log.read_text()[-800:]}")
    return json.loads(line)


def gate_outcome(session: dict | None) -> dict:
    """The front's public outcome (its ``session.json``), as the record reads it."""
    if session is None:
        return {"accepted": None, "why": "the front wrote no session.json"}
    if "refused" in session:
        return {"accepted": False, "why": session["refused"]}
    v = session.get("verdict") or {}
    return {"accepted": session.get("rc") == 0 and v.get("accepted") is True, "rc": session.get("rc"),
            "ended": session.get("ended"), "run": session.get("run"), "windows": v.get("windows"),
            "requests": v.get("requests"), "problems": v.get("problems"), "failed": v.get("failed"),
            "refused": v.get("refused"), "request": session.get("request")}


def front(args) -> int:
    dc = mockdc.lab(args)
    plan = mockdc.plan_of(args, dc)
    if diff := mockdc.differences(plan):
        raise SystemExit(f"{'; '.join(diff)}: {mockdc.RESHAPE}")
    if busy := {ns: pids for ns in mockdc.namespaces() if (pids := mockdc.sudo(["ip", "netns", "pids", ns]).stdout.split())}:
        raise SystemExit(f"the lab is busy, processes {busy} run in its namespaces: wait until its run ends")
    out, keep = dirs(args)
    py, zones = sys.executable, {}
    for b in (b.name for b in dc.blocks):
        _, cert = records.uplink(dc, b)
        zones[b] = {"zone": graph.netns_name(b), "certifier": graph.netns_name(cert), "address": plan.address(b, cert)}
    procs, fronts = {}, {}
    try:
        for b, z in zones.items():
            private = keep / b
            private.mkdir(mode=0o700, exist_ok=True)
            serve = private / "serve"
            serve.mkdir(exist_ok=True)
            endpoint = start(z["zone"], [py, "-B", "-u", str(TRACES / "demo_stub.py"), "serve", "--host", z["address"],
                                         "--port", str(PORT), "--prompt-ids", str(args.prompt_ids), "--tokens",
                                         str(args.tokens), "--vocab", str(VOCAB), "--out", str(serve)],
                             private / "endpoint.log")
            procs[b] = [endpoint]
            z["endpoint"] = ready(endpoint, f"{b}'s endpoint", private / "endpoint.log")["listening"]
            relay = start(z["certifier"], [py, "-B", "-u", str(Path(__file__).resolve()), "tunnel", "--listen",
                                           str(PORT), "--host", z["address"], "--to", str(PORT)], private / "tunnel.log")
            procs[b].append(relay)
            ready(relay, f"{b}'s tunnel", private / "tunnel.log")
        for b, z in zones.items():
            (out / b).mkdir(exist_ok=True)
            argv = [py, "-B", "-u", str(TRACES / "certifier_piece.py"), "front", str(PORT), str(keep / b / "serve"),
                    str(out / b / "front"), "--gate", "local", "--prompts", str(args.prompts), "--prompt-ids",
                    str(args.prompt_ids), "--tokens", str(args.tokens), "--vocab", str(VOCAB), "--height",
                    str(args.height), *(["--scan-class", args.scan_class] if args.scan_class else [])]
            fronts[b] = start(z["certifier"], argv, keep / b / "front.log", piped=False)
        print(f"fronts started in {', '.join(z['certifier'] for z in zones.values())}", flush=True)
        for b, p in fronts.items():
            zones[b]["exit"] = {"front": p.wait()}
    finally:
        # The endpoint first: at SIGTERM it closes its open Commit period, which the piece reads.
        for b, z in zones.items():
            for ns, p in zip((z["zone"], z["certifier"]), procs.get(b, ())):
                mockdc.terminate(ns)
                z.setdefault("exit", {})["endpoint" if ns == z["zone"] else "tunnel"] = p.wait()
        for p in fronts.values():
            p.wait()
    host = socket.gethostname()
    for b, z in zones.items():
        z["gate"] = gate_outcome(_json(out / b / "front" / "session.json"))
        z["endpoint_summary"] = (keep / b / "serve" / "summary.json").is_file()
        g = z["gate"]
        print(f"{b}: front exit {z['exit']['front']}, the gate's check "
              + ("ACCEPTED" if g["accepted"] else f"NOT accepted ({g.get('why') or g.get('failed')})")
              + (f": {g['requests']} requests in {g['windows']} phase(s), {g['problems']} problems"
                 if g.get("requests") is not None else ""), flush=True)
    doc = {"label": "simulated", "simulated": list(SIMULATED), "host": host, "declaration": dc.name, "zones": zones}
    (out / "front.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if all(z["exit"]["front"] == 0 and z["gate"]["accepted"] for z in zones.values()) else 1


# ---------------------------------------------------------------------------------------------------------------- piece

def window(serve_dir: Path) -> int:
    """The endpoint's Commit period holding the front's requests: every request the endpoint served, since the front's
    group is its one load, in one window."""
    held = Counter(w for r in CP._lines(serve_dir / "requests.jsonl") for w in r.get("windows") or ())
    if len(held) != 1:
        raise SystemExit(f"{serve_dir}: the front's requests lie in {len(held)} Commit periods, not one")
    return next(iter(held))


def scan_class(session: dict | None) -> str:
    """The class the front's gate ran its scan at: its request's ``--scan-class``, else the gate's class of record."""
    words = shlex.split((session or {}).get("request") or "")
    return words[words.index("--scan-class") + 1] if "--scan-class" in words else GATE_CLASS


def auditor_record(serve_dir: Path, into: Path) -> Path | None:
    """The auditor's record of the kept run's deployment (`warden_chain.RECORD`, the piece's ``--certifier-key``)
    written to ``into``. The lab is its own auditor, so the record is its local gate's own copy of the run (the front's
    ``--gate local``: this tree's gate on a scratch host under the serve dir), as a served run's auditor reads the
    gate's (``CERTIFIER_KEY``). None, said on stdout, where the serve dir keeps not one run, the gate not one copy of it
    or no readable one, and the piece then refuses."""
    import warden_chain as WC
    try:
        run = CP.kept_run(serve_dir)
    except CP.NotRun as e:
        print(f"{serve_dir}: no auditor's record of the run's deployment ({e}), so the piece refuses", flush=True)
        return None
    gates = {p.parent.parent for p in (serve_dir / CP.PRIVATE / CP.SS.LOCAL_GATE).rglob(f"{run}/out/deployment.json")}
    if len(gates) != 1:
        print(f"{serve_dir}: {len(gates)} copies of run {run} at its local gate, not one: no auditor's record of its "
              "deployment, so the piece refuses", flush=True)
        return None
    try:
        into.write_text(json.dumps(WC.own_record(gates.pop())))
    except (SystemExit, OSError, ValueError) as e:
        print(f"{serve_dir}: run {run}'s deployment at its local gate: {type(e).__name__}: {e}: no auditor's record of "
              "it, so the piece refuses", flush=True)
        return None
    return into


def piece(args) -> int:
    out, keep = dirs(args)
    given = Path(args.front).resolve() if args.front else out
    if (given / "front.json").is_file() and given != out:
        shutil.copyfile(given / "front.json", out / "front.json")
        for d in given.glob("*/front"):
            shutil.copytree(d, out / d.parent.name / "front", dirs_exist_ok=True)
    if not (out / "front.json").is_file():
        raise SystemExit(f"no front.json in {given}: give --front the front's --out")
    zones = args.zone or sorted(p.name for p in keep.iterdir() if (p / "serve").is_dir())
    if not zones:
        raise SystemExit(f"no zone's serve dir under {keep}: run `onsite.py front` with this --keep first")
    codes = {}
    for b in zones:
        serve, d = keep / b / "serve", out / b / "piece"
        d.mkdir(parents=True, exist_ok=True)
        cls = scan_class(_json(out / b / "front" / "session.json"))
        key = auditor_record(serve, keep / b / "certifier-key.json")
        argv = [sys.executable, "-B", "-u", str(TRACES / "certifier_piece.py"), "piece", str(serve),
                str(window(serve)), str(d), *(["--certifier-key", str(key)] if key else []),
                *(["--lean", args.lean] if args.lean else []), f"CR_SCAN_ARGS=--class {cls}", *args.set]
        print(f"{b}: the piece at the scan class {cls}", flush=True)
        with open(d / "piece.log", "w") as log:
            codes[b] = subprocess.run(argv, cwd=mockdc.ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{b}: piece exit {codes[b]}, {json.dumps(period_outcome(d))[:600]}", flush=True)
    (out / "piece.json").write_text(json.dumps({"exit": codes}, indent=1) + "\n")
    record(args)
    return 0 if all(c == 0 for c in codes.values()) else 1


# ---------------------------------------------------------------------------------------------------------------- record

def period_outcome(d: Path) -> dict | None:
    """The piece's answer in DIR ``d`` (its verdict and reason), its refusal to run, or None where it didn't run."""
    if (a := _json(d / "answer.json")) is not None:
        return {"verdict": a.get("verdict"), "reason": a.get("reason"), "protocol": a.get("protocol")}
    if (d / CP.REFUSED).is_file():
        return {"verdict": "refused", "reason": (d / CP.REFUSED).read_text().strip()}
    return None


def chain_outcome(d: Path) -> dict | None:
    """The chain's public verdicts (`certifier_piece.PUBLIC`'s part of its report), or None where none was released."""
    if (r := _json(d / "chain" / "report" / "report.json")) is None:
        return None
    return {"verdict": r.get("verdict"), "path": r.get("path"), "scope": r.get("scope"),
            "stubbed": r.get("stubbed") or [], "negatives_refused": r.get("negatives_refused") or {},
            "steps": {k: {"verdict": s.get("verdict"), **({"eps_at_delta": s["eps_at_delta"]}
                                                           if "eps_at_delta" in s else {})}
                      for k, s in (r.get("steps") or {}).items()}}


def warden_run(bundle: Path, into: Path) -> Path:
    """The front's public bundle unpacked into ``into``: its gate run's directory (``end.json``, ``out/``)."""
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(into, filter="data")
    (end,) = into.glob("*/end.json")
    return end.parent


def load(k: dict) -> LB.LinkLoad:
    """The zone's link as `chain_record.leak` counted it: its class, certified rate, T, τ and egress cells."""
    p, run = k["params"], k["run"]
    tau = Fraction(p["bucket_ms"], 1000)
    rate = Fraction(run["requests"], run["link_windows"]) / (p["window"] * tau)
    return LB.link(IS.Class(**k["class"]), rate, T=p["window"], tau=tau, egress_cells=p["egress_cells"])


def delta_net(eps: str, phases: float) -> list[dict]:
    """δ_net over ``phases`` execution phases, term by term: the period check's eps, only its δ part, and each of
    `ERRORS` once a phase. Never eps alone."""
    return [{"symbol": "eps", "value": str(eps)}, *({"symbol": s, "times": phases} for s in ERRORS)]


def terms(k: dict, period: str) -> dict:
    """One zone's bound over ``period`` in `PeriodLeakAccepted`'s terms, from `chain_record.leak`'s doc ``k``, beside the
    pooled form's at each δ, whose incorrect proof blocks are its k_skip."""
    t = k["terms"]
    (row,) = [r for r in k["rows"] if (r["period"], r["scope"], r["prog"]) == (period, "link", READING)]
    m, requests = row["link_windows"], row["requests"]
    got = {"m": m, "requests": requests, "k_sched": m * t["schedules_per_block"],
           "k_fail": m * t["cells_per_block"], "ell_a": m * t["free_per_block"][READING],
           "k_skip": 0, "jitter": m * t["jitter_per_block"]}
    if not math.isclose(sum(got[x] for x in TERMS), row["per_period"], rel_tol=1e-12):
        raise SystemExit(f"{period}: the terms sum to {sum(got[x] for x in TERMS)}, not leak's {row['per_period']}")
    return got | {"bound": row["per_period"], "share": row["per_period_share"], "eps": row["eps"],
                  "delta_net": delta_net(row["eps"], m), "vacuous": Fraction(row["eps"]) >= 1,
                  "pooled": {d: {"bound": b, "k_skip": b - row["per_period"], "share": row["pooled_share"][d]}
                             for d, b in row["pooled"].items()}}


def zone_record(zd: Path) -> dict:
    """A zone's verdicts and K, from what its front and its piece released in ``zd``. K is `PeriodLeakAccepted`'s
    bound at the zone's run, established only where the period check accepted (``established``); elsewhere it is the
    bound the zone's load would have had."""
    session = _json(zd / "front" / "session.json")
    entry = {"gate": gate_outcome(session), "chain": chain_outcome(zd / "piece"), "period": period_outcome(zd / "piece")}
    entry["verdict"] = (entry["period"] or {}).get("verdict") or "not run"
    if not (zd / "front" / "bundle.tgz").is_file() or not entry["gate"]["accepted"]:
        return entry | {"K": None}
    with tempfile.TemporaryDirectory() as tmp:
        k = CR.leak(warden_run(zd / "front" / "bundle.tgz", Path(tmp)))
    T = k["params"]["window"]
    entry["K"] = {
        "established": entry["verdict"] == "accept", "statement": LB.STATEMENTS["per_period"], "bound": BOUND,
        "conditional_on": list(OPEN_CHECKS), "reading": READING, "class": k["class"],
        "scan_class": scan_class(session), "params": k["params"], "run": k["run"],
        "per_phase": {"connections": k["terms"]["connections_per_block"], "cell": k["terms"]["cell"],
                      "k_sched": k["terms"]["schedules_per_block"], "k_fail": k["terms"]["cells_per_block"],
                      "ell_a": k["terms"]["free_per_block"][READING], "k_skip": 0, "T_j": T,
                      "tau_ms": k["params"]["bucket_ms"], "phase_s": k["run"]["link_window_s"],
                      "d_jit": k["terms"]["jitter_bound"], "jitter": k["terms"]["jitter_per_block"],
                      "K": next(r["per_link_window"] for r in k["rows"] if r["prog"] == READING)},
        "periods": {p: terms(k, p) for p in LB.PERIODS},
        "scopes": [{key: r[key] for key in ("period", "scope", "links", "link_windows", "per_period",
                                            "per_period_share", "eps", "pooled", "pooled_share")}
                   | {"delta_net": delta_net(r["eps"], r["link_windows"])}
                   for r in k["rows"] if r["prog"] == READING],
        "load": k}
    return entry


def lab_sum(zones: dict[str, dict]) -> dict | None:
    """The lab's zones summed (`leak_bounds.datacenter` on each zone's link): our composition, as the census's."""
    links = [load(z["K"]["load"]) for z in zones.values() if z["K"]]
    if not links:
        return None
    phases = {p: sum(z["K"]["periods"][p]["m"] for z in zones.values() if z["K"]) for p in LB.PERIODS}
    return {p: {"per_period": (per := LB.datacenter(links, p, READING)).bits, "share": per.share, "m": phases[p],
                "eps": str(per.eps), "delta_net": delta_net(per.eps, phases[p]), "vacuous": per.eps >= 1,
                "pooled": {f"{float(d):g}": LB.datacenter(links, p, READING, d).bits for d in LB.DELTAS}}
            for p in LB.PERIODS} | {"links": len(links), "of": len(zones),
                                    "established": all(z["K"] and z["K"]["established"] for z in zones.values())}


#: what each scope's figures rest on
BASIS = {
    "zone": "PeriodLeakAccepted's terms at the zone's own run: its gate's check covered one execution phase, so a "
            "period is that phase's load scaled to the period's m phases",
    "lab": "the lab's zones summed, each at its own run's load: our composition (leak_bounds.datacenter), which no Lean "
           "statement covers, scaled to the period",
    "datacenter": "the census's reference datacenter, every link at a zone's load: our composition, scaled",
    "world": "the adversary resource model's compute in links, every link at a zone's load: our composition, scaled",
}


def notes() -> list[str]:
    return [
        "PeriodLeakAccepted bounds a zone only over a period its period check accepted: a zone whose check answered "
        "anything else has no K established, and its figures are the bound its run's load would have had; the lab's "
        "sum is established only where every zone's is, and the datacenter's and the world's never are, their links "
        "being no run's",
        "k_sched and k_fail each charge ⌈log2(cap + 1)⌉ for every connection the schedule can carry (Schedule.caps: "
        "egress_cells for each of lanes·T_j connections a phase), whatever the requests served, since the adversary "
        "picks how many connections it opens. Both count the cell counts: PeriodLeakAccepted's terms as proved, a "
        "known overcount, and a bound charging each bit once would be tighter",
        "ℓ_a,j is phase j's free-input term in the period check's answer (scans[j]'s free and entries), not the "
        "realized advice length |a_j|, which is private and not in this record. The lab ties no audit (no V3c), so "
        "the record reads it as registered (leak_bounds): the scan's NR·T_j/EVERY slot entries, NE bits each, which "
        "the period check charges once a phase whose scan has every proof block drawn",
        "every K here is conditional. Charged as the scan's slot entries once, ℓ_a,j bounds what the link carries only "
        "on a statement not yet in the lock (the count PeriodLeakAccepted bounds at most 2^K at the check's K) and on "
        "the scan's egress views being tied to its state, which no check reads. A phase whose scan is sampled in part "
        "is charged its tied audits' counted free inputs instead, which holds only once two checks are Lean refusals "
        "in the period check (" + "; ".join(OPEN_CHECKS) + "). No advice cap is charged for what is open: the link "
        "figure charges the advice through the entries, and the advice's padded length is in no term of K",
        "k_skip,j is 0 in the per-period form, every proof block being drawn; the pooled form's k_skip at δ is "
        "F·log2(N + 1) + b·F (leak_bounds), and that form is conditional (hesc and hleak await SampledKSLive and P2)",
        f"the jitter is T_j·log2 d_jit at d_jit = {LB.JITTER_BOUND}, T_j bits a phase: JitterBounded holds only at "
        f"d_jit = {LB.JITTER_BOUND}, since at d_jit = 1 the next bucket's host-state channel breaks it",
        "no sync term on the active-warden path",
        "δ_net = eps + m·(ε_ks + ε_bind + ε_tap) over a row's m phases (PeriodLeakAccepted's failure a phase, "
        "δ + ε_ks + ε_bind + ε_tap): eps, the period check's, is only its δ part (m·δ, each phase's scan at its δ), and no check computes ε_ks, ε_bind or ε_tap, "
        "so δ_net has no number here; a row whose eps alone reaches 1 is vacuous, a bound that holds with no "
        "probability",
        "each zone's figures scale one phase's run to the period; nothing over a week, a month or a year was run",
    ]


def fmt(x: float) -> str:
    if x >= 1e6:
        e = math.floor(math.log10(x))
        return f"{x / 10 ** e:.3g}e{e}"
    return f"{x:,.0f}"


def delta(eps: str) -> str:
    """eps, δ_net's δ part, as the tables print it: vacuous where it alone reaches 1, where the bound holds with no
    probability."""
    e = Fraction(eps)
    return f"{float(e):.3g}" + (" (vacuous)" if e >= 1 else "")


def held(established: bool) -> str:
    return "yes" if established else "no"


def short(reason: str, n: int = 160) -> str:
    """``reason`` cut at a word within ``n`` characters (record.json keeps it whole)."""
    return reason if len(reason) <= n else reason[:n].rsplit(" ", 1)[0] + " …"


def markdown(doc: dict) -> str:
    zones = doc["zones"]
    lines = ["# The onsite network path on the mock datacenter: SIMULATED, never served", "",
             f"Lab: `{doc['declaration']}` on `{doc['host']}`. Simulated because:", "",
             *(f"- {s}" for s in doc["simulated"]), "", "## Verdicts", "",
             "| zone | gate's check | chain | period check (the zone's verdict) |", "|---|---|---|---|"]
    for b, z in zones.items():
        g, c, p = z["gate"], z["chain"], z["period"]
        gate = ("accepted" if g["accepted"] else f"not accepted: {g.get('why') or g.get('failed')}") + (
            f", {g['requests']} requests in {g['windows']} phase(s)" if g.get("requests") is not None else "")
        chain = "not run" if c is None else c["verdict"] + " (" + ", ".join(f"{k} {s['verdict']}"
                                                                          for k, s in c["steps"].items()) + ")"
        period = "not run" if p is None else f"{p['verdict']}: {short(p.get('reason') or '')}"
        lines.append(f"| {b} | {gate} | {chain} | {period} |")
    chains = [z["chain"] for z in zones.values() if z["chain"]]
    stubbed = list(dict.fromkeys(s for c in chains for s in c["stubbed"]))
    negatives = sorted({k for c in chains for k, v in c["negatives_refused"].items() if v})
    missed = sorted({f"{b}: {k}" for b, z in zones.items() if z["chain"]
                     for k, v in z["chain"]["negatives_refused"].items() if not v})
    if chains:
        lines += ["", "What the chain's path stubs, by its report:", "", *(f"- {s}" for s in stubbed or ["nothing"]), "",
                  f"Its tampered negatives refused: {', '.join(negatives) or 'none'}"
                  + (f"; NOT refused: {', '.join(missed)}" if missed else "") + "."]
    accepted = [b for b, z in zones.items() if z["K"] and z["K"]["established"]]
    lines += ["", "## K, the bits a zone's server can send beyond its committed program", "",
              f"`{LB.STATEMENTS['per_period']}`: {BOUND}, in the registered reading, shares of one copy of the weights "
              f"({LB.WEIGHTS.label}). It bounds a zone only over a period its period check accepted: K is established "
              f"for {', '.join(accepted) or 'no zone'} here. A zone's figures without that accept are the bound its "
              "run's load would have had, had its period check accepted.", "",
              "Every K here is conditional: it charges ℓ_a,j as the scan's slot entries once, as the period check "
              "does where every proof block of the scan is drawn, which bounds what the link carries only on a "
              "statement not yet in the lock; a phase whose scan is sampled in part is charged its tied audits' "
              f"counted free inputs instead, which holds only once the period check holds {OPEN_CHECKS[0]} and "
              f"{OPEN_CHECKS[1]} in Lean. No advice cap is charged for what is open: the link figure charges the "
              "advice through the entries.", "",
              "Each bound holds except at δ_net = eps + m·(ε_ks + ε_bind + ε_tap) over its row's m phases. The tables "
              "give eps, the period check's δ part of δ_net; no check computes ε_ks, ε_bind or ε_tap, so δ_net has "
              "no number here.", ""]
    for b, z in zones.items():
        if not z["K"]:
            lines.append(f"{b}: no K (the gate's check didn't accept)")
            continue
        k, w = z["K"], z["K"]["per_phase"]
        why = {"not run": "the period check didn't run", "refused": "the piece refused to run"}.get(
            z["verdict"], f"the period check answered {z['verdict']}")
        status = "established" if k["established"] else f"not established: {why}"
        lines += [f"**{b}** ({status}) at scan class {k['scan_class']}, "
                  f"{k['run']['requests_per_link_window']:g} requests served in a phase of T_j = {w['T_j']} buckets "
                  f"of τ = {w['tau_ms']} ms ({w['phase_s']:g} s). Its terms a phase, whatever the requests served: "
                  f"k_sched = k_fail = {w['k_sched']:,} ({w['connections']} connections the schedule can carry, "
                  f"{w['cell']} bits each), ℓ_a,j = {w['ell_a']:,} (the scan's slot entries, conditional), k_skip,j = 0, "
                  f"T_j·log2 d_jit = {w['jitter']} "
                  f"(d_jit = {w['d_jit']}); K a phase = {w['K']:,.0f}.", "",
                  f"| period | m ({w['phase_s']:g} s phases) | k_sched | k_fail | Σ_j ℓ_a,j | Σ_j T_j·log2 d_jit | "
                  "K per period (share) | eps (δ part) | K pooled, δ = 1e-6 (k_skip) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p, t in k["periods"].items():
            pooled = t["pooled"]["1e-06"]
            lines.append(f"| {p} | {fmt(t['m'])} | {fmt(t['k_sched'])} | {fmt(t['k_fail'])} | {fmt(t['ell_a'])} | "
                         f"{fmt(t['jitter'])} | {fmt(t['bound'])} ({t['share']:.3g}) | {delta(t['eps'])} | "
                         f"{fmt(pooled['bound'])} ({fmt(pooled['k_skip'])}) |")
        lines.append("")
    if doc["lab"]:
        first = next(z["K"] for z in zones.values() if z["K"])
        lab = held(doc["lab"]["established"])
        w = first["per_phase"]
        lines += ["## Over all compute", "",
                  f"m counts the phases over the period, summed over the scope's links, each phase T_j = {w['T_j']} "
                  f"buckets of τ = {w['tau_ms']} ms ({w['phase_s']:g} s), the lab's tick. K a phase reads no τ, so at "
                  "a longer tick a period holds fewer phases and its K and eps shrink in proportion.", "",
                  "| period | scope | links | m | established | K per period (share) | eps (δ part) | "
                  "K pooled, δ = 1e-6 | basis |", "|---|---|---|---|---|---|---|---|---|"]
        for p in LB.PERIODS:
            for b, z in zones.items():
                if z["K"]:
                    t = z["K"]["periods"][p]
                    lines.append(f"| {p} | zone {b} | 1 | {fmt(t['m'])} | {held(z['K']['established'])} | "
                                 f"{fmt(t['bound'])} ({t['share']:.3g}) | {delta(t['eps'])} | "
                                 f"{fmt(t['pooled']['1e-06']['bound'])} | zone |")
            s = doc["lab"][p]
            lines.append(f"| {p} | the lab | {doc['lab']['links']} | {fmt(s['m'])} | {lab} | {fmt(s['per_period'])} "
                         f"({s['share']:.3g}) | {delta(s['eps'])} | {fmt(s['pooled']['1e-06'])} | lab |")
            for r in first["scopes"]:
                if r["period"] == p and r["scope"] in ("datacenter", "world"):
                    lines.append(f"| {p} | {r['scope']} | {r['links']:,} | {fmt(r['link_windows'])} | no (scaled) | "
                                 f"{fmt(r['per_period'])} ({r['per_period_share']:.3g}) | {delta(r['eps'])} | "
                                 f"{fmt(r['pooled']['1e-06'])} | {r['scope']} |")
        lines += ["", *(f"- **{k}**: {v}" for k, v in BASIS.items())]
    lines += ["", "## Notes", "", *(f"- {n}" for n in doc["notes"]), ""]
    return "\n".join(lines)


def record(args) -> int:
    out, _ = dirs(args)
    front_doc = _json(out / "front.json")
    if front_doc is None:
        raise SystemExit(f"no {out / 'front.json'}: run `onsite.py front --out {out}` first")
    zones = {b: zone_record(out / b) for b in front_doc["zones"]}
    loads = {repr(load(z["K"]["load"])) for z in zones.values() if z["K"]}
    doc = {"label": "simulated", "served": False, "simulated": list(SIMULATED), "host": front_doc["host"],
           "declaration": front_doc["declaration"],
           "placement": {b: {k: z[k] for k in ("zone", "certifier", "address")} for b, z in front_doc["zones"].items()},
           "zones": zones, "lab": lab_sum(zones), "basis": BASIS, "weights": LB.WEIGHTS._asdict(),
           "one_load": len(loads) == 1, "notes": notes()}
    if len(loads) > 1:
        doc["notes"].append("the zones' loads differ, so the datacenter's and the world's rows are the first zone's")
    for z in zones.values():
        if z["K"]:
            z["K"].pop("load")
    (out / "record.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (out / "record.md").write_text(markdown(doc))
    print(markdown(doc))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("front")
    mockdc.shape_options(f)
    f.add_argument("--out", required=True, help="published: each zone's front's public outcome, front.json")
    f.add_argument("--keep", required=True, help="private, outside --out: each zone's serve dir and logs")
    f.add_argument("--prompts", type=int, default=CP.PROMPTS)
    f.add_argument("--prompt-ids", type=int, default=CP.PROMPT_IDS)
    f.add_argument("--tokens", type=int, default=CP.TOKENS)
    f.add_argument("--height", type=int, default=CP.HEIGHT)
    f.add_argument("--scan-class", help=f"the gate's scan class (the front's default, {CP.V3C_SCAN_CLASS})")
    f.set_defaults(fn=front)
    p = sub.add_parser("piece")
    p.add_argument("--out", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--front", help="the front's --out, whose front.json and zones' front/ it copies to --out (default: "
                   "--out itself)")
    p.add_argument("--zone", action="append", default=[], help="a zone to run (default: every zone kept)")
    p.add_argument("--lean", help="network-period, built (else the piece builds it)")
    p.add_argument("set", nargs="*", help="CR_K=V, to each piece's chain")
    p.set_defaults(fn=piece)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=record)
    t = sub.add_parser("tunnel")
    t.add_argument("--listen", type=int, required=True)
    t.add_argument("--host", required=True)
    t.add_argument("--to", type=int, required=True)
    t.set_defaults(fn=tunnel)
    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
