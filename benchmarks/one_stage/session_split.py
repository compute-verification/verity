"""A warm session's parts at several sample sizes, from one prover process: SPOT's proof server
(`verity_flock.proof_server.ProofServer`) on one staged statement of n blocks, each session proving k of them (a uniform
subset of the population, as the verifier serves a sample), so that a statement's fixed seconds and a block's come from the
same warm process.

    python benchmarks/one_stage/session_split.py --bin flock-circuit --stage Z/<type>/c<n>/stage --out DIR --device D \\
        [--draws 64,128,256] [--rounds 4] [--provers 1] [--ahead 0] [--outstanding 1] [--deadline EPOCH]

* **Cold:** each prover's first session, at the largest k, starts it and its verifier, which load the circuit and build the
  population's statement (the logs' ``statement ... built in`` lines, in ``builds``); its parts are the cold session's.
* **Warm-up:** each other k's first session on each prover builds that sample size's statement (the statement cache) and
  isn't timed.
* **Timed:** ``--rounds`` rounds of every k, ascending then descending, so that a drift in the node's load falls on every k
  alike; K sessions each (``--provers``). With ``--outstanding 1`` and ``--ahead 0`` each session runs alone, so its parts
  are its own: the prover's ``witness_s``, ``prove_total_s`` (its device phases) and ``wait_e2e_s`` (its wait on the live
  verifier inside the session), and ``rest_s``, the remainder of ``e2e_s``.
* **Measured:** per k each part's median over the timed sessions, and each part fitted as ``fixed + per_block k`` (least
  squares); each process's peak RSS and the provers' and verifiers' summed peaks (every 0.5 s); the device's peak memory
  and its busy share over the timed sessions; with ``--outstanding`` above 1, statements a second between the timed
  sessions' filings by their verifiers, the first and last K left out.

Writes DIR/split.json and DIR/sessions.jsonl beside the server's logs (no circuit, prover file or proof). As in
`server_rate`, a run with a refused session, a process that exited nonzero or a prover out of device memory is void: its
reasons in ``void``, a ``VOID`` line, and exit 1.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import server_rate as SR  # noqa: E402
from verity_flock.proof_server import ProofServer, Statement  # noqa: E402

PARTS = ("e2e_s", "witness_s", "witness_prebuilt_s", "prove_total_s", "wait_e2e_s", "wait_s", "rest_s")
BUILT = re.compile(r"^statement (\S+) instances=(\d+) .* built in ([\d.]+)s", re.MULTILINE)


def rest(live: dict) -> float | None:
    """``e2e_s`` less the witness, the device's phases and the wait on the live verifier inside the session."""
    if not all(isinstance(live.get(k), (int, float)) for k in ("e2e_s", "prove_total_s", "wait_e2e_s")):
        return None
    return live["e2e_s"] - (live.get("witness_s") or 0.0) - live["prove_total_s"] - live["wait_e2e_s"]


def fit(points: list[tuple[int, float]]) -> dict | None:
    """``y = fixed + per_block k`` by least squares, with the residuals' standard errors (None below two distinct k)."""
    if len({k for k, _ in points}) < 2:
        return None
    X = np.array([[1.0, k] for k, _ in points])
    y = np.array([v for _, v in points])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(points) - 2
    se = [None, None]
    if dof > 0:
        s2 = float(((y - X @ beta) ** 2).sum()) / dof
        se = [round(float(x), 5) for x in np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))]
    return {"fixed_s": round(float(beta[0]), 4), "per_block_s": round(float(beta[1]), 6), "fixed_se": se[0],
            "per_block_se": se[1], "sessions": len(points)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--stage", required=True, help="a staged statement's directory (stage.py's OUT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", required=True, help="the GPU's index or UUID (gpu-lease's CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--draws", default="64,128,256", help="sample sizes k, each at most the statement's n")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--provers", type=int, default=1)
    ap.add_argument("--ahead", type=int, default=0)
    ap.add_argument("--outstanding", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=0, help="UTC epoch seconds after which no session is submitted")
    a = ap.parse_args()
    d, out = Path(a.stage), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = json.loads((d / "stage.json").read_text())
    n, K = stage["n"], a.provers
    ks = sorted({int(x) for x in a.draws.split(",")})
    if ks[-1] > n or ks[0] < 1:
        raise SystemExit(f"session_split: --draws {a.draws} must lie in 1..{n}, the statement's blocks")
    st = Statement(pin=(d / "pin.txt").read_text().strip(), circuit=d / "circuit.txt", tables=d / "tables",
                   instances=d / "inst.bin", public=d / "pub.bin")
    rng = random.Random(f"session-split/{st.pin}")

    def draw(k: int) -> dict:
        return {"law": "subset", "population": n, "k": k, "units": sorted(rng.sample(range(n), k)) if k < n else list(range(n))}

    order = [ks[-1]] * K + [k for k in reversed(ks[:-1]) for _ in range(K)]
    phases = ["cold"] * K + ["warmup"] * (len(order) - K)
    for r in range(a.rounds):
        for k in (ks if r % 2 == 0 else ks[::-1]):
            order += [k] * K
            phases += ["timed"] * K
    rec: dict = {"stage": stage, "n": n, "draws": ks, "provers": K, "verify_ahead": a.ahead, "outstanding": a.outstanding,
                 "rounds": a.rounds, "device": a.device, "pin": st.pin}
    room, lock, done = threading.Semaphore(a.outstanding), threading.Lock(), []
    gpu, t0, cpu0 = SR.Device(a.device), time.monotonic(), SR.throttled()
    wall0 = time.time() - t0
    ps = ProofServer(a.bin, out, devices=[a.device], provers=K, verify_ahead=a.ahead, verdicts=1,
                     env={"FC_FIREWALL": "0", "FC_HOST_WITNESS": None, "FC_PHASE_TIMES": "1"})

    def stop(sig, _frame) -> None:
        # the server's processes run in sessions of their own, so a lease's SIGTERM to this one alone would leave them on the GPU
        for name, pid in ps.pids.items():
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                print(f"session_split: {name} (pid {pid}) had already exited at signal {sig}", file=sys.stderr, flush=True)
        os._exit(128 + sig)

    was = signal.signal(signal.SIGTERM, stop)
    host = SR.Host(ps)

    def wait(s, i: int, k: int, phase: str, t_sub: float) -> None:
        try:
            s.result()
            err = None
        except Exception as e:  # noqa: BLE001 -- recorded with the session, the rest go on
            err = str(e)[:600]
        with lock:
            done.append({"i": i, "k": k, "phase": phase, "s": s, "error": err, "t_sub": t_sub, "t": time.monotonic(),
                         "load1": os.getloadavg()[0]})
        room.release()

    threads: list[threading.Thread] = []
    stopped = None
    try:
        for i, (k, phase) in enumerate(zip(order, phases)):
            # each phase starts once the one before has ended, so no timed session shares the device with a cold one
            if i and phases[i - 1] != phase:
                for t in threads:
                    t.join()
                rec[f"{phases[i - 1]}_end_s"] = round(time.monotonic() - t0, 1)
            if a.deadline and time.time() > a.deadline:
                stopped = f"past the deadline at session {i} of {len(order)}"
                break
            room.acquire()
            s = ps.submit(st, draw(k), f"k{k}/{i}")
            threads.append(threading.Thread(target=wait, args=(s, i, k, phase, time.monotonic()), daemon=True))
            threads[-1].start()
        for t in threads:
            t.join()
    finally:
        rec["exits"] = ps.close()
        signal.signal(signal.SIGTERM, was)
        rec["host"] = host.stop()
        gpu.stop()
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    rec["stopped"] = stopped
    done.sort(key=lambda r: r["i"])
    bad = [r for r in done if r["error"] or not r["s"].accepted]
    rec |= {"sessions": len(done), "accepted": len(done) - len(bad), "refused_n": len(bad),
            "refused": [{"label": r["s"].label, "error": r["error"], "why": (r["s"].live or {}).get("why")} for r in bad][:8]}
    for r in done:
        r["live"] = dict(r["s"].live or {})
        r["live"]["rest_s"] = rest(r["live"])
        r["filed"] = r["s"].dir.stat().st_mtime - wall0 if r["s"].dir is not None and r["s"].dir.exists() else None
    timed = [r for r in done if r["phase"] == "timed" and not r["error"] and r["s"].accepted]
    med = lambda xs: round(statistics.median(xs), 4) if xs else None  # noqa: E731
    num = lambda r, p: r["live"].get(p) if isinstance(r["live"].get(p), (int, float)) else None  # noqa: E731
    rec["cold"] = [{"label": r["s"].label, "prover": r["s"].prover, "k": r["k"], "seconds": r["s"].seconds,
                    **{p: num(r, p) for p in PARTS}} for r in done if r["phase"] == "cold"]
    rec["warmup"] = [{"label": r["s"].label, "prover": r["s"].prover, "k": r["k"], "seconds": r["s"].seconds,
                      **{p: num(r, p) for p in PARTS}} for r in done if r["phase"] == "warmup"]
    rec["median"] = {str(k): {"sessions": sum(1 for r in timed if r["k"] == k),
                              "seconds": med([r["s"].seconds for r in timed if r["k"] == k and r["s"].seconds]),
                              **{p: med([v for r in timed if r["k"] == k and (v := num(r, p)) is not None]) for p in PARTS}}
                     for k in ks}
    rec["fit"] = ({p: fit([(r["k"], v) for r in timed if (v := num(r, p)) is not None]) for p in PARTS}
                  | {"seconds": fit([(r["k"], r["s"].seconds) for r in timed if r["s"].seconds])})
    t_first = min((r["t_sub"] for r in timed), default=None)
    rec["busy"] = gpu.busy(t_first, max(r["t"] for r in timed)) if timed else None
    rec["device_peak_mib"] = gpu.peak_mib()
    rec["throttled"] = {k: SR.throttled().get(k, 0) - cpu0.get(k, 0) for k in ("nr_throttled", "throttled_usec")}
    filed = sorted(r["filed"] for r in timed if r["filed"] is not None)
    if a.outstanding > 1 and len(filed) > 2 * K + 2:
        w = filed[K - 1:len(filed) - K]
        rec["statements_per_s"] = round((len(w) - 1) / (w[-1] - w[0]), 4) if w[-1] > w[0] else None
    rec["builds"] = [{"log": f.stem, "statement": m[0], "instances": int(m[1]), "built_s": float(m[2])}
                     for f in sorted(out.glob("*.log")) for m in BUILT.findall(f.read_text(errors="replace"))]
    with open(out / "sessions.jsonl", "w") as f:
        for r in done:
            live = {k: v for k, v in r["live"].items() if k not in ("verdict", "buckets", "roots", "unit_draw", "identity")}
            f.write(json.dumps({"i": r["i"], "k": r["k"], "phase": r["phase"], "prover": r["s"].prover,
                                "submitted": round(r["t_sub"] - t0, 3), "ended": round(r["t"] - t0, 3),
                                "filed": None if r["filed"] is None else round(r["filed"] - t0, 3), "seconds": r["s"].seconds,
                                "load1": round(r["load1"], 2), "error": r["error"], "live": live}, default=str) + "\n")
    with open(out / "device.tsv", "w") as f:
        f.writelines(f"{x - t0:.3f}\t{u:g}\t{m:g}\n" for x, u, m in gpu.samples)
    if why := SR.void_reasons(len(bad), len(done), rec["exits"], out):
        rec |= {"void": why, "fit": None, "median": None, "statements_per_s": None}
    (out / "split.json").write_text(json.dumps(rec, indent=1, default=str))
    for k, m in (rec["median"] or {}).items():
        print(f"SPLIT k={k} " + json.dumps(m), flush=True)
    print("FIT " + json.dumps(rec["fit"]), flush=True)
    print("HOST " + json.dumps({"host": rec["host"], "device_peak_mib": rec["device_peak_mib"], "cold": rec["cold"],
                                "builds": rec["builds"], "statements_per_s": rec.get("statements_per_s")}), flush=True)
    if rec.get("void"):
        print("VOID " + "; ".join(rec["void"]), flush=True)
        return 1
    return 0 if not stopped else 1


if __name__ == "__main__":
    sys.exit(main())
