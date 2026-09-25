"""The TABLES.md batch sweep for ``flock-pure``: one ``result.json`` per point, stamped with its ``sweep`` and ``protocol``.

    python -m verity_flock.bench --relation bf16-hopper --bin target/release/flock-pure --out-dir D \\
        [--start 1024] [--max 65536] [--runs 5] [--verifier HOST:PORT] [--procs N]

Each point writes its instance set range (``verity_flock.instances``), starts a live verifier on loopback (unless
``--verifier`` names one that serves the same range), and runs ``flock-pure prove --warm 1 --runs R --dump``: one untimed
session, then ``R`` timed sessions, each committing its batch afresh (row digests and the three frame-v3 roots) before
proving. A session's time is its prover's end to end: commitment, rows, both reps with their coin waits; the verifier's
final replay is not in it. Doubling stops when two successive doublings together raise the median throughput by less
than 2 % or a point fails; the plateau is the point with the highest median throughput. Every point's proofs, session
records and the verifier's index stay in its directory; the plateau's ``result.json`` is also ``$RESEARCH_RUN_DIR/result.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import instances, lowering
from .backend import FLOCK_FRAME_V3_BLAKE3, lowering_version

RULE = ("double total VUs from start at fixed per-proof settings until two successive doublings together raise the median "
        "throughput by < 2 % or a point fails; plateau = the point with the highest median throughput (kb/TABLES.md)")
PROFILES = {"bf16-hopper": "bf16-hopper-mma-draft/2026-09-22", "fp8-ada": "fp8-ada-mma-draft/2026-09-22",
            "fp8-hopper": "fp8-hopper-wgmma-draft/2026-09-22"}


def _guard():
    instances._ligero()
    from ligero import timing_guard
    return timing_guard


def _quota() -> int:
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _lines(path: Path, tag: str) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if line.startswith(tag + "\t"):
            out.append(json.loads(line.split("\t", 1)[1]))
    return out


def _wait_serving(log: Path, proc: subprocess.Popen, timeout: float = 600) -> None:
    t = time.time()
    while time.time() - t < timeout:
        if log.exists() and "SERVING" in log.read_text():
            return
        if proc.poll() is not None:
            raise RuntimeError(f"verifier exited: {log.read_text()[-500:]}")
        time.sleep(0.5)
    raise RuntimeError("verifier did not start")


_PHASES = {"witgen": "t.witness", "compact q": "t.encoding_commitment", "commit": "t.encoding_commitment",
           "open": "t.encoding_commitment", "bind statement": "t.arithmetic", "s_hat_v_ab fold": "t.arithmetic",
           "boolean zerocheck": "t.arithmetic", "wiring GKR": "t.arithmetic", "claim assembly": "t.arithmetic",
           "witness buffer give-back": "t.witness"}


def trace_buckets(err: Path) -> list[dict[str, float]]:
    """Per prove (one rep), seconds per bucket from Flock's ``PCS_TRACE`` lines (``[prove_union] <phase>...: X ms``)."""
    out, cur = [], {}
    for line in err.read_text(errors="replace").splitlines():
        if not line.startswith("  [prove_union] "):
            continue
        body = line[len("  [prove_union] "):]
        if body.startswith("TOTAL"):
            cur["total"] = float(body.split()[1]) / 1e3
            out.append(cur)
            cur = {}
            continue
        name, _, rest = body.partition(":")
        try:
            ms = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        for k, b in _PHASES.items():
            if name.startswith(k):
                cur[b] = cur.get(b, 0.0) + ms / 1e3
                break
    return out


def point(a, total: int, d: Path, net: Path) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = instances.write(d / "instances.bin", a.relation, total, a.procs, a.cache)
    gen_s = time.time() - t0
    pin = lowering.PINS[a.relation]
    common = ["--instances", str(d / "instances.bin"), "--netlist", str(net)]
    srv = None
    addr = a.verifier
    if not addr:
        addr = "127.0.0.1:%d" % (7400 + total.bit_length())
        log = d / "serve.log"
        srv = subprocess.Popen(["nice", "-n", "5", a.bin, "serve", "--listen", addr, "--out", str(d / "sessions"), "--pin", pin, *common],
                               stdout=open(log, "w"), stderr=subprocess.STDOUT)
        _wait_serving(log, srv)
    try:
        return _prove(a, total, d, common, addr, srv, h, gen_s)
    finally:
        if srv is not None and srv.poll() is None:
            srv.terminate()
            srv.wait()


def _prove(a, total, d, common, addr, srv, h, gen_s) -> dict:
    tg = _guard()
    guard = tg.Guard().start()
    t1 = time.time()
    timer = ["/usr/bin/time", "-v", "-o", str(d / "prove.time")] if Path("/usr/bin/time").exists() else []
    rc = subprocess.call([*timer, a.bin, "prove", "--verifier", addr, "--warm", "1",
                          "--runs", str(a.runs), "--dump", str(d / "proofs"), *common],
                         stdout=open(d / "prove.txt", "w"), stderr=open(d / "prove.err", "w"), env={**os.environ, "PCS_TRACE": "1"})
    wall = time.time() - t1
    contention = guard.stop()
    if srv is not None:
        time.sleep(1)
        srv.terminate()
        srv.wait()
    live = _lines(d / "prove.txt", "LIVE") if (d / "prove.txt").exists() else []
    timed = [r for r in live if not r.get("warm")]
    ok = rc == 0 and len(timed) == a.runs and all(r.get("accepted") is True for r in live)
    rss = None
    if (d / "prove.time").exists():
        for line in (d / "prove.time").read_text().splitlines():
            if "Maximum resident set size" in line:
                rss = int(line.split(":")[1]) * 1024
    tr = trace_buckets(d / "prove.err") if (d / "prove.err").exists() else []
    if len(tr) == 2 * len(live):
        for i, r in enumerate(live):
            r["buckets"] = {b: tr[2 * i].get(b, 0.0) + tr[2 * i + 1].get(b, 0.0) for b in set(_PHASES.values())}
    return {"total": total, "rc": rc, "ok": ok, "live": live, "timed": timed, "header": h, "gen_s": gen_s, "wall_s": wall,
            "contention": contention, "peak_rss": rss, "dir": d.name}


def result(a, p: dict, sweep: dict) -> dict:
    from verity_numerical.bench import contract

    tg = _guard()
    timed = p["timed"]
    med = lambda k: statistics.median(r[k] for r in timed)  # noqa: E731
    e2e = [r["e2e_s"] for r in timed]
    t_e2e, t_commit = statistics.median(e2e), med("commit_s")
    t_prove = t_e2e - t_commit
    net_wait = max(0.0, statistics.median(r["wait_s"] - r["verdict"].get("handle_s", 0.0) for r in timed))
    n = p["total"]
    profile = PROFILES[a.relation]
    rec = FLOCK_FRAME_V3_BLAKE3.record()
    first = timed[0]
    hw = contract.probe_hardware()
    fp = contract.fingerprint(
        profile=profile, domain="finite", K=instances.K, B=n, proof_class="NON_ZK_PROOF", authentication="included-hash",
        zk_mode="none",
        security={"target": -128, "achieved_log2": -195.5, "profile": "flock-128-r2", "reps": 2, "per_rep_log2": -97.8,
                  "coins": "live verifier (flock-live R1-R8)", "accountant": rec["soundness"]["accountant"]},
        hardware=hw,
        software={"backend": {"name": "flock-pure", "configuration": rec["configuration"], "commit": a.commit,
                              "lowering_sha256": lowering.PINS[a.relation], "lowering_version": lowering_version(),
                              "flock": "succinctlabs/flock b684b12 + backends/flock/flock-link-b684b12.patch"},
                  "toolchain": {"rustc": a.rustc, "python": platform.python_version()}},
        preprocessing="none: the circuit shape and the unit netlist are built once per process (fixed.setup_seconds); every "
                      "timed session commits its batch and builds its rows afresh",
        retention="retained", instances=p["header"]["instances"],
        scheme="frame-v3/blake3-keyed/row/v2", statement="verity/flock-pure/v1", relation=a.relation,
        roots=first["roots"], sigma=first["sigma"], dense_m=first["dense_m"], units_per_vu=first["units"],
        n_blake3=first["n_blake3"], n_units=first["n_units"],
        rtt_method="net.rtt_ms = (prover coin wait - verifier handling time) / round trips, per timed session, median",
        verifier={"placement": "loopback" if not a.verifier else a.verifier,
                                                                         "process": "flock-pure serve (separate process)"})
    ms = [contract.measurement("t.total", t_prove), contract.measurement("vu.count", n),
          contract.measurement("vu.seconds_per_vu", t_prove / n), contract.measurement("vu.per_second", n / t_prove),
          *contract.rate_measurements(t_prove, n, instances.K, profile),
          {"name": "commit.seconds", "value": t_commit, "unit": "s"},
          {"name": "e2e.seconds", "value": t_e2e, "unit": "s"},
          {"name": "e2e.vu_per_second", "value": n / t_e2e, "unit": "1/s"},
          {"name": "e2e.seconds_min", "value": min(e2e), "unit": "s"}, {"name": "e2e.seconds_max", "value": max(e2e), "unit": "s"},
          {"name": "rows.seconds", "value": med("rows_s"), "unit": "s"},
          {"name": "prove.rep0_seconds", "value": statistics.median(r["prove_s"][0] for r in timed), "unit": "s"},
          {"name": "prove.rep1_seconds", "value": statistics.median(r["prove_s"][1] for r in timed), "unit": "s"},
          contract.measurement("net.wait_seconds", med("wait_s")), contract.measurement("net.messages", int(first["calls"])),
          contract.measurement("net.bytes_out", int(first["wire_sent"])), contract.measurement("net.bytes_in", int(first["wire_recv"])),
          contract.measurement("rounds.protocol", int(first["verdict"]["rounds"])),
          contract.measurement("rounds.sequential_depth", int(first["calls"])),
          {"name": "live.verifier_round_seconds", "value": statistics.median(r["verdict"].get("handle_s", 0.0) for r in timed), "unit": "s"},
          {"name": "live.network_wait_seconds", "value": net_wait, "unit": "s"},
          contract.measurement("net.rtt_ms", 1e3 * net_wait / max(1, int(first["calls"]))),
          {"name": "live.prover_compute_seconds", "value": t_prove - med("wait_s"), "unit": "s"},
          contract.measurement("transcript.bytes", int(first["wire_sent"]) + int(first["wire_recv"])),
          contract.measurement("verify.wall_s", statistics.median(r["verdict"]["verify_s"] for r in timed)),
          {"name": "proof.bytes", "value": sum(first["proof_bytes"]), "unit": "B"},
          contract.measurement("fixed.setup_seconds", p["gen_s"])]
    if all("buckets" in r for r in timed):
        for b in sorted(set(_PHASES.values())):
            v = statistics.median(r["buckets"][b] + (r["rows_s"] if b == "t.witness" else 0.0) for r in timed)
            ms.append(contract.measurement(b, v))
    if p["peak_rss"]:
        ms.append(contract.measurement("mem.peak_host_bytes", p["peak_rss"]))
    doc = {"schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID"), "workload_fingerprint": fp,
           "validation": {"status": "passed" if p["ok"] else "failed", "checker": "flock-pure serve (live verifier, R1-R8 replay)",
                          "detail": f"{sum(1 for r in p['live'] if r.get('accepted'))}/{len(p['live'])} sessions accepted",
                          "evidence": [f"{p['dir']}/sessions/index.jsonl"]},
           "measurements": ms, "measurement_files": [f"{p['dir']}/prove.txt"], "artifacts": [],
           "sweep": sweep, "contention": p["contention"],
           "protocol": {"warm": True, "runs": len(timed), "statistic": "median", "contended": p["contention"].get("contended"),
                        "warm_rule": "one full untimed session (prove --warm 1) before the timed sessions, same process"},
           "runs": [{"e2e_s": r["e2e_s"], "commit_s": r["commit_s"], "prove_s": r["prove_s"], "wait_s": r["wait_s"],
                     "dump": r.get("dump")} for r in timed]}
    problems = contract.validate(doc)
    doc["validation"]["contract_problems"] = problems
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verity_flock.bench")
    ap.add_argument("--relation", required=True, choices=sorted(lowering.PIPES))
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=int, default=1024)
    ap.add_argument("--max", type=int, default=1 << 16)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--verifier", default="")
    ap.add_argument("--procs", type=int, default=_quota())
    ap.add_argument("--cache", default=None)
    ap.add_argument("--commit", default=os.environ.get("RESEARCH_SOURCE_COMMIT", "unknown"))
    a = ap.parse_args(argv)
    a.rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = out / f"net-{a.relation}.txt"
    net.write_text(lowering.netlist(a.relation))
    if hashlib.sha256(net.read_bytes()).hexdigest() != lowering.PINS[a.relation]:
        print("the lowering is not the pinned one", file=sys.stderr)
        return 2
    sid = "sweep-" + hashlib.sha256(json.dumps([a.relation, a.start, a.runs, time.time()]).encode()).hexdigest()[:12]
    points = []
    total = a.start
    while total <= a.max:
        print(f"=== {time.strftime('%H:%M:%SZ', time.gmtime())} point {len(points)} total {total}", flush=True)
        try:
            p = point(a, total, out / f"p{len(points)}-{total}", net)
        except Exception as e:  # noqa: BLE001  (a failed point ends the sweep)
            p = {"total": total, "ok": False, "err": str(e), "timed": [], "dir": f"p{len(points)}-{total}"}
        tp = (total / statistics.median(r["e2e_s"] for r in p["timed"])) if p["ok"] else None
        p["vu_per_second"] = tp
        points.append(p)
        print(f"    ok={p['ok']} rc={p.get('rc')} vu/s={tp} wall={p.get('wall_s')} contended={p.get('contention', {}).get('contended')} "
              f"rss={p.get('peak_rss')} err={p.get('err')}", flush=True)
        tps = [q["vu_per_second"] for q in points if q["vu_per_second"]]
        if tp is None or (len(tps) >= 3 and tps[-1] < 1.02 * tps[-3]):
            break
        total *= 2
    ok = [q for q in points if q["vu_per_second"]]
    if not ok:
        return 1
    best = max(ok, key=lambda q: (q["vu_per_second"], -q["total"]))
    tps = [q["vu_per_second"] for q in ok]
    stop = "a point failed" if points[-1]["vu_per_second"] is None else (
        "< 2 % over two doublings" if len(tps) >= 3 and tps[-1] < 1.02 * tps[-3] else "reached --max")
    for q in ok:
        sweep = {"id": sid, "point": points.index(q), "plateau": q is best, "total_vus": q["total"], "rule": RULE, "stopped": stop,
                 "points": [[x["total"], x["vu_per_second"]] for x in points]}
        doc = result(a, q, sweep)
        (out / q["dir"] / "result.json").write_text(json.dumps(doc, indent=1, default=str))
        if q is best and os.environ.get("RESEARCH_RUN_DIR"):
            Path(os.environ["RESEARCH_RUN_DIR"], "result.json").write_text(json.dumps(doc, indent=1, default=str))
    summary = {"id": sid, "relation": a.relation, "rule": RULE, "stopped": stop, "plateau_total_vus": best["total"],
               "points": [{k: x.get(k) for k in ("total", "ok", "vu_per_second", "wall_s", "peak_rss", "dir", "err")} for x in points]}
    (out / "sweep.json").write_text(json.dumps(summary, indent=1))
    print("SWEEP\t" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
