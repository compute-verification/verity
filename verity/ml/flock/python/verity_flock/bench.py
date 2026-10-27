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
import math
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
from .backend import BACKENDS, FLOCK_FRAME_V3_BLAKE3, lowering_version

RULE = ("double total VUs from start at fixed per-proof settings until two successive doublings together raise the median "
        "throughput by < 2 % or a point fails; plateau = the point with the highest median throughput (kb/TABLES.md)")
PROFILES = {"bf16-hopper": "bf16-hopper-mma-draft/2026-09-22", "fp8-ada": "fp8-ada-mma-draft/2026-09-22",
            "fp8-hopper": "fp8-hopper-wgmma-draft/2026-09-22", "bf16-ampere": "first-campaign-target/2026-09-21",
            "bf16-hopper-wgmma": "bf16-hopper-mma-draft/2026-09-22", "fp4-nvf4": "nvfp4-sm120-mma-draft/2026-09-22"}


def _guard():
    from . import timing_guard
    return timing_guard


_OURS: set[int] = set()
#: when our prover subprocesses exited (monotonic): NVML can list an exited context for a while, under a pid we can't match
_EXITED: list[float] = []
EXIT_GRACE_S = 10.0


def _sampler():
    """The timing guard's sample with our own prover subprocess taken out: it holds the GPU context, not this process, and
    NVML may list it under another pid namespace, so one listed process per running prover of ours is ours."""
    s = _guard().sample()
    if s.get("pids") is not None:
        mine = [p for p in s["pids"] if int(p) in _OURS] if s["pids"] else []
        rest = [p for p in s["pids"] if p not in mine]
        now = time.monotonic()
        lingering = sum(1 for t in _EXITED if now - t < EXIT_GRACE_S)
        drop = max(0, len(_OURS) + lingering - len(mine))
        s["pids"] = rest[drop:]
        s["n_other"] = len(s["pids"])
    return s


def _quota() -> int:
    """The cgroup's CPU quota in whole cores (v2 ``cpu.max``, else v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``), else the
    host's CPU count."""
    cg = Path("/sys/fs/cgroup")
    try:
        q, p = (cg / "cpu.max").read_text().split()
    except (OSError, ValueError):
        try:
            q, p = ((cg / "cpu" / f).read_text().strip() for f in ("cpu.cfs_quota_us", "cpu.cfs_period_us"))
        except OSError:
            q = p = "max"
    try:
        if q not in ("max", "-1"):
            return max(1, int(int(q) / int(p)))
    except ValueError:
        pass
    return os.cpu_count() or 1


def _lines(path: Path, tag: str) -> list[dict]:
    """The tagged JSON lines. A LIVE record's ``wait_s`` is the transport's wait read after ``Finish``, so it includes the
    verifier's end-of-session replay (``verdict.verify_s``), which ``e2e_s`` does not: that is taken out here (the raw figure
    stays as ``wait_s_raw``)."""
    out = []
    for line in path.read_text().splitlines():
        if line.startswith(tag + "\t"):
            r = json.loads(line.split("\t", 1)[1], strict=False)
            if tag == "LIVE" and "wait_s" in r and isinstance(r.get("verdict"), dict) and r["verdict"].get("verify_s") is not None:
                r["wait_s_raw"] = r["wait_s"]
                r["wait_s"] = max(0.0, r["wait_s"] - r["verdict"]["verify_s"])
            out.append(r)
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


def subbatches(total: int, per: int) -> list[tuple[int, int]]:
    return [(lo, min(total, lo + per)) for lo in range(0, total, per)]


def _probe(addr: str, timeout: float = 1800) -> None:
    """Wait until the verifier for the next statement listens, then use up the one connection its `--sessions` count
    reserves for this probe (the previous server may still hold the port for a moment after its last session)."""
    import socket
    time.sleep(3)
    host, port = addr.rsplit(":", 1)
    t = time.time()
    while time.time() - t < timeout:
        try:
            socket.create_connection((host, int(port)), timeout=5).close()
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"no verifier at {addr}")


PER_PROOF_LOG2 = -195.44   # flock-128-r2 whole-proof bound per proof (flock-pure-block/v2's)


def tcp_rtt(target: str, n: int = 30) -> dict | None:
    """TCP connect time to ``target`` (the verifier pod's sshd, not its session port: a connect there would use up a
    session): median and min over ``n`` connects, in ms."""
    import socket
    if not target:
        return None
    host, port = target.rsplit(":", 1)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            socket.create_connection((host, int(port)), timeout=5).close()
        except OSError:
            continue
        ts.append((time.perf_counter() - t) * 1e3)
        time.sleep(0.02)
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "n": len(ts), "target": target} if ts else None


def loopback_probe(a, q: dict, out: Path, net: Path) -> dict:
    """The same binary and batch as the plateau point, verifier on 127.0.0.1: per-round coin wait, median over sessions."""
    d = out / q["dir"]
    pin = lowering.PINS[a.relation]
    per = []
    for i, _ in enumerate(q["subbatches"]):
        common = ["--instances", str(d / f"instances-s{i}.bin"), "--netlist", str(net)]
        log = d / f"loopback-serve-s{i}.log"
        srv = subprocess.Popen([a.bin, "serve", "--listen", "127.0.0.1:7391", "--out", str(d / f"loopback-sessions-s{i}"), "--pin", pin,
                                *common], stdout=open(log, "w"), stderr=subprocess.STDOUT)
        try:
            _wait_serving(log, srv)
            subprocess.call([a.bin, "prove", "--verifier", "127.0.0.1:7391", "--warm", "1", "--runs", str(a.runs), *a.prove_arg, *common],
                            stdout=open(d / f"loopback-prove-s{i}.txt", "w"), stderr=open(d / f"loopback-prove-s{i}.err", "w"))
        finally:
            srv.terminate()
            srv.wait()
        for r in _lines(d / f"loopback-prove-s{i}.txt", "LIVE"):
            if not r.get("warm") and r.get("accepted") and r.get("calls"):
                per.append(r["wait_s"] / r["calls"])
    return {"seconds_per_round": statistics.median(per) if per else None, "sessions": len(per),
            "method": f"loopback probe in this run: {len(per)} timed sessions ({a.runs} per sub-batch) of the plateau point's "
                      f"instance files with the same binary against `serve` on 127.0.0.1, median per-round coin wait ((wait_s - verify_s) / calls)"}


def _merge(parts: list[list[dict]]) -> list[dict]:
    """One record per run over the sub-batches (sequential proofs, one in flight): times, rounds and bytes add up."""
    out = []
    for runs in zip(*parts):
        m = {"accepted": all(r.get("accepted") is True for r in runs), "warm": runs[0].get("warm"), "run": runs[0].get("run"),
             "subbatches": len(runs)}
        for k in ("e2e_s", "commit_s", "rows_s", "wait_s", "calls", "wire_sent", "wire_recv"):
            m[k] = sum(r[k] for r in runs)
        m["prove_s"] = [sum(r["prove_s"][i] for r in runs) for i in range(2)]
        m["verdict"] = {k: sum(r["verdict"].get(k, 0) or 0 for r in runs) for k in ("rounds", "verify_s", "handle_s")}
        m["proof_bytes"] = [sum(r["proof_bytes"][i] for r in runs) for i in range(2)]
        for k in ("roots", "sigma", "dense_m", "rtt_ms", "rtt_method", "statement", "statement_digest", "prover", "open_rtt_ms", "domain_digests", "scheme"):
            m[k] = [r.get(k) for r in runs] if len(runs) > 1 else runs[0].get(k)
        m["units"] = runs[0].get("units")
        m["n_blake3"] = sum(r.get("n_blake3") or 0 for r in runs)
        m["n_sha256"] = sum(r.get("n_sha256") or 0 for r in runs)
        m["n_units"] = sum(r.get("n_units") or 0 for r in runs)
        if all(isinstance(r.get("buckets"), list) for r in runs):
            m["buckets"] = {b: sum(rep.get(b, 0.0) for r in runs for rep in r["buckets"]) for b in ("t.witness", "t.encoding_commitment", "t.arithmetic")}
        elif all(isinstance(r.get("buckets"), dict) for r in runs):
            m["buckets"] = {b: sum(r["buckets"].get(b, 0.0) for r in runs) for b in ("t.witness", "t.encoding_commitment", "t.arithmetic")}
        out.append(m)
    return out


def write_instances(a, path: Path, total: int, lo: int, hi: int) -> dict:
    """One sub-batch's instance file: a spine input set (``--set``), the 8-bit synthetic stream at ``--k``, or the relation's
    own set."""
    if a.set:
        return instances.write_set(path, a.relation, a.set, lo, hi, a.leaf, a.procs, a.y_model)
    if a.k:
        return instances.write_synth(path, a.relation, a.k, lo, hi, a.leaf, a.procs)
    return instances.write(path, a.relation, total, a.procs, a.cache, lo, hi, scheme=a.scheme, leaf=a.leaf)


def point(a, total: int, d: Path, net: Path) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    pin = lowering.PINS[a.relation]
    subs = subbatches(total, a.per_proof)
    headers, gen_s = [], 0.0
    for i, (lo, hi) in enumerate(subs):
        t0 = time.time()
        headers.append(write_instances(a, d / f"instances-s{i}.bin", total, lo, hi))
        gen_s += time.time() - t0
    tg = _guard()
    guard = tg.Guard(sampler=_sampler).start()
    t1 = time.time()
    parts, rcs, rss = [], [], None
    for i, _ in enumerate(subs):
        common = ["--instances", str(d / f"instances-s{i}.bin"), "--netlist", str(net)]
        srv, addr = None, a.verifier
        if addr:
            _probe(addr)
        else:
            addr = "127.0.0.1:%d" % (7400 + total.bit_length())
            log = d / f"serve-s{i}.log"
            srv = subprocess.Popen(["nice", "-n", "5", a.bin, "serve", "--listen", addr, "--out", str(d / f"sessions-s{i}"), "--pin", pin,
                                    *common], stdout=open(log, "w"), stderr=subprocess.STDOUT)
            _wait_serving(log, srv)
        try:
            timer = ["/usr/bin/time", "-v", "-o", str(d / f"prove-s{i}.time")] if Path("/usr/bin/time").exists() else []
            pr = subprocess.Popen([*timer, a.bin, "prove", "--verifier", addr, "--warm", "1", "--runs", str(a.runs),
                                   "--dump", str(d / f"proofs-s{i}"), *a.prove_arg, *common],
                                  stdout=open(d / f"prove-s{i}.txt", "w"), stderr=open(d / f"prove-s{i}.err", "w"),
                                  env={**os.environ, "PCS_TRACE": "1"})
            _OURS.add(pr.pid)
            try:
                rcs.append(pr.wait())
                time.sleep(2)  # NVML lists an exited context for a moment: still ours
            finally:
                _OURS.discard(pr.pid)
        finally:
            if srv is not None:
                time.sleep(1)
                srv.terminate()
                srv.wait()
        live = _lines(d / f"prove-s{i}.txt", "LIVE") if (d / f"prove-s{i}.txt").exists() else []
        if not any("buckets" in r for r in live):
            tr = trace_buckets(d / f"prove-s{i}.err") if (d / f"prove-s{i}.err").exists() else []
            if len(tr) == 2 * len(live):
                for j, r in enumerate(live):
                    r["buckets"] = {b: tr[2 * j].get(b, 0.0) + tr[2 * j + 1].get(b, 0.0) for b in set(_PHASES.values())}
        parts.append(live)
        if (d / f"prove-s{i}.time").exists():
            for line in (d / f"prove-s{i}.time").read_text().splitlines():
                if "Maximum resident set size" in line:
                    rss = max(rss or 0, int(line.split(":")[1]) * 1024)
        if rcs[-1] != 0 or len(live) != a.runs + 1:
            break
    wall = time.time() - t1
    contention = guard.stop()
    ok = (len(parts) == len(subs) and all(rc == 0 for rc in rcs) and all(len(x) == a.runs + 1 for x in parts)
          and all(r.get("accepted") is True and "e2e_s" in r for x in parts for r in x))
    live = _merge(parts) if ok else []
    timed = [r for r in live if not r.get("warm")]
    ok = ok and len(timed) == a.runs and all(r["accepted"] for r in live)
    h = dict(headers[0])
    h["instances"] = dict(h["instances"], range=[0, total])
    if a.set:
        h["instances"]["content_digest"] = json.loads((Path(a.set) / "manifest.json").read_text()).get("content_digest")
    return {"total": total, "rc": max(rcs) if rcs else None, "ok": ok, "live": live, "timed": timed, "header": h, "gen_s": gen_s,
            "wall_s": wall, "contention": contention, "peak_rss": rss, "dir": d.name, "subbatches": subs}


def result(a, p: dict, sweep: dict) -> dict:
    from verity_numerical.bench import contract

    tg = _guard()
    timed = p["timed"]
    med = lambda k: statistics.median(r[k] for r in timed)  # noqa: E731
    e2e = [r["e2e_s"] for r in timed]
    t_e2e, t_commit = statistics.median(e2e), med("commit_s")
    t_prove = t_e2e - t_commit
    net_wait = max(0.0, statistics.median(r["wait_s"] - r["verdict"].get("handle_s", 0.0) for r in timed))
    rtts = [x for r in timed for x in (r["open_rtt_ms"] if isinstance(r.get("open_rtt_ms"), list) else [r.get("open_rtt_ms")])
            if isinstance(x, (int, float)) and x == x]
    open_rtt = statistics.median(rtts) if rtts else None
    n = p["total"]
    profile = PROFILES[a.relation]
    binary = Path(a.bin).name
    backend = next((b for b in BACKENDS if b.binary == binary), FLOCK_FRAME_V3_BLAKE3)
    rec = backend.record()
    first = timed[0]
    hw = contract.probe_hardware()
    fp = contract.fingerprint(
        profile=profile, domain=lowering.PIPES[a.relation].domain, K=p["header"]["k"], B=n, proof_class="NON_ZK_PROOF", authentication="included-hash",
        zk_mode="none",
        security={"target": -128, "achieved_log2": PER_PROOF_LOG2 + math.log2(len(p["subbatches"])), "per_proof_log2": PER_PROOF_LOG2,
                  "proofs": len(p["subbatches"]), "union": "union bound over the batch's sequential sub-batch proofs",
                  "profile": "flock-128-r2", "reps": 2, "per_rep_log2": -97.8,
                  "coins": "live verifier (flock-live R1-R8)", "accountant": rec["soundness"]["accountant"]},
        hardware=hw,
        software={"backend": {"name": binary, "configuration": rec["configuration"], "commit": a.commit,
                              "statement": first.get("statement") or "verity/flock-pure/v1", "prover": first.get("prover") or "cpu-union",
                              "prove_args": a.prove_arg,
                              "lowering_sha256": lowering.PINS[a.relation], "lowering_version": lowering_version(),
                              "flock": "succinctlabs/flock b684b12 + verity/ml/flock/flock-link-b684b12.patch"},
                  "toolchain": {"rustc": a.rustc, "python": platform.python_version()}},
        preprocessing="none: the circuit shape and the unit netlist are built once per process (fixed.setup_seconds); every "
                      "timed session commits its batch and builds its rows afresh",
        retention="retained", instances=p["header"]["instances"], N_subbatches=len(p["subbatches"]),
        batching={"total": n, "per_proof": a.per_proof, "in_flight": 1, "subbatches": [list(x) for x in p["subbatches"]]},
        scheme=(f"frame-v3/{p['header']['schemas']['a']}" if a.scheme == "frame-v3" else backend.scheme), statement=first.get("statement") or "verity/flock-pure/v1", relation=a.relation,
        roots=first["roots"], sigma=first["sigma"], dense_m=first["dense_m"], units_per_vu=first["units"],
        n_blake3=first["n_blake3"], n_units=first["n_units"], **({"n_sha256": first["n_sha256"]} if first.get("n_sha256") else {}),
        **({"domain_digests": first["domain_digests"]} if first.get("domain_digests") else {}),
        rtt_method="net.rtt_ms = (prover coin wait - verifier handling time) / round trips, per timed session, median",
        verifier={"placement": "loopback" if not a.verifier else a.verifier,
                                                                         "process": "flock-pure serve (separate process)"})
    ms = [contract.measurement("t.total", t_prove), contract.measurement("t.total_live", t_prove), contract.measurement("vu.count", n),
          contract.measurement("vu.seconds_per_vu", t_prove / n), contract.measurement("vu.per_second", n / t_prove),
          *contract.rate_measurements(t_prove, n, p["header"]["k"], profile),
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
          {"name": "live.coin_wait_per_round_ms", "value": 1e3 * net_wait / max(1, int(first["calls"])), "unit": "ms"},
          *([contract.measurement("net.rtt_ms", open_rtt)] if open_rtt is not None else []),
          *([{"name": "live.tcp_connect_rtt_ms", "value": a.tcp["median_ms"], "unit": "ms"}] if getattr(a, "tcp", None) else []),
          *([{"name": "live.loopback_round_seconds", "value": p["loopback"]["seconds_per_round"], "unit": "s"}]
            if p.get("loopback", {}).get("seconds_per_round") else []),
          {"name": "live.prover_compute_seconds", "value": t_prove - med("wait_s"), "unit": "s"},
          contract.measurement("transcript.bytes", int(first["wire_sent"]) + int(first["wire_recv"])),
          contract.measurement("verify.wall_s", statistics.median(r["verdict"]["verify_s"] for r in timed)),
          {"name": "proof.bytes", "value": sum(first["proof_bytes"]), "unit": "B"},
          contract.measurement("fixed.setup_seconds", p["gen_s"])]
    if all(isinstance(r.get("buckets"), dict) for r in timed):
        bk = {b: statistics.median(r["buckets"].get(b, 0.0) for r in timed) for b in ("t.witness", "t.encoding_commitment", "t.arithmetic")}
        if sum(bk.values()) + med("rows_s") <= t_prove:
            bk["t.witness"] += med("rows_s")
        for b, v in bk.items():
            ms.append(contract.measurement(b, v))
    if p["peak_rss"]:
        ms.append(contract.measurement("mem.peak_host_bytes", p["peak_rss"]))
    doc = {"schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID"), "workload_fingerprint": fp,
           "validation": {"status": "passed" if p["ok"] else "failed", "checker": "flock-pure serve (live verifier, R1-R8 replay)",
                          "detail": f"{sum(1 for r in p['live'] if r.get('accepted'))}/{len(p['live'])} sessions accepted",
                          "evidence": [f"{p['dir']}/sessions/index.jsonl"]},
           "measurements": ms, "measurement_files": [], "artifacts": [],
           "sweep": sweep, "contention": p["contention"],
           "interaction": {"rtt_method": ("net.rtt_ms = round trip of a 64-byte Ping request on the open session connection to the "
                                          "verifier (answered Ok by the verifier's server, no session state), 30 probes right after "
                                          "connecting and before Hello, in every timed session; median over all of them") if open_rtt is not None else None,
                           "tcp_connect_method": (f"live.tcp_connect_rtt_ms = TCP connect time to the verifier pod's sshd ({a.tcp['target']}), "
                                                  f"median of {a.tcp['n']}, before the sweep") if getattr(a, "tcp", None) else None,
                           "coin_wait_method": "live.coin_wait_per_round_ms = (the prover's coin wait - the verifier's round-handling time) / "
                                               "round trips, median over the timed sessions; verifier " + ("on loopback" if not a.verifier else a.verifier),
                           "loopback_method": p.get("loopback", {}).get("method"),
                           "hello_rtt_ms": statistics.median(x for r in timed for x in (r["rtt_ms"] if isinstance(r.get("rtt_ms"), list) else [r.get("rtt_ms")]) if x is not None)
                           if any(r.get("rtt_ms") is not None for r in timed) else None,
                           "hello_rtt_method": "the session's Hello request/response time (flock-pure-gpu rtt_ms), median",
                           "verifier_operator": a.operator or None,
                           "t_total_includes_wait": True,
                           "t_total_note": "live coins: t.total and the buckets include the prover's coin waits; prover compute alone "
                                           "is live.prover_compute_seconds = t.total - net.wait_seconds"},
           "protocol": {"warm": True, "runs": len(timed), "statistic": "median", "contended": p["contention"].get("contended"),
                        "warm_rule": "one full untimed session (prove --warm 1) before the timed sessions, same process"},
           "runs": [{"e2e_s": r["e2e_s"], "commit_s": r["commit_s"], "prove_s": r["prove_s"], "wait_s": r["wait_s"],
                     "dump": r.get("dump")} for r in timed]}
    problems = contract.validate(doc)
    doc["validation"]["contract_problems"] = problems
    return doc


def serve_plan(a, plan: list[int], out: Path, net: Path) -> int:
    """The verifier pod's side of a remote sweep: its own instance files for every point and sub-batch (regenerated here,
    never the prover's), then one `serve --sessions` per sub-batch in the prover's order (probe + warm + runs)."""
    pin = lowering.PINS[a.relation]
    jobs = []
    for i, total in enumerate(plan):
        d = out / f"p{i}-{total}"
        d.mkdir(parents=True, exist_ok=True)
        for j, (lo, hi) in enumerate(subbatches(total, a.per_proof)):
            write_instances(a, d / f"instances-s{j}.bin", total, lo, hi)
            jobs.append((d, j))
    print(f"PLAN\t{json.dumps({'points': plan, 'servers': len(jobs)})}", flush=True)
    for d, j in jobs:
        rc = subprocess.call([a.bin, "serve", "--listen", a.listen, "--out", str(d / f"sessions-s{j}"), "--pin", pin,
                              "--instances", str(d / f"instances-s{j}.bin"), "--netlist", str(net), "--sessions", str(a.runs + 2),
                              "--operator", a.operator], stdout=open(d / f"serve-s{j}.log", "w"), stderr=subprocess.STDOUT)
        print(f"SERVED\t{d.name}/s{j} rc={rc}", flush=True)
    return 0


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
    ap.add_argument("--per-proof", type=int, default=8192, help="VUs per proof; larger batches are sequential sub-batches")
    ap.add_argument("--prove-arg", action="append", default=[], help="extra argument for `prove` (e.g. --gpu)")
    ap.add_argument("--points", default="", help="explicit batch sizes (both sides of a remote verifier must agree)")
    ap.add_argument("--serve-plan", action="store_true", help="verifier side: serve every point's sub-batches in order")
    ap.add_argument("--listen", default="0.0.0.0:7400")
    ap.add_argument("--operator", default="")
    ap.add_argument("--rtt-target", default="", help="host:port on the verifier pod to time TCP connects to (its sshd)")
    ap.add_argument("--loopback", action="store_true", help="after the sweep, a loopback probe at the plateau point")
    ap.add_argument("--scheme", choices=["frame-v3", "vllm-v1"], default="frame-v3",
                    help="the commitment scheme the instance files' roots are under (the binary must prove that statement)")
    ap.add_argument("--leaf", default="blake3", choices=sorted(instances.SCHEMES), help="frame-v3 row-leaf schema of trees a and b")
    ap.add_argument("--set", default="", help="a spine input set directory (x.u16 / w.u16 / y.u16); batches never exceed it")
    ap.add_argument("--y-model", action="store_true", help="with --set: y is the relation's chain output (a set captured on other silicon)")
    ap.add_argument("--k", type=int, default=0, help="an 8-bit relation's synthetic E4M3 stream at this K")
    a = ap.parse_args(argv)
    a.rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = out / f"net-{a.relation}.txt"
    net.write_text(lowering.netlist(a.relation))
    if hashlib.sha256(net.read_bytes()).hexdigest() != lowering.PINS[a.relation]:
        print("the lowering is not the pinned one", file=sys.stderr)
        return 2
    plan = [int(x) for x in a.points.split()] if a.points else None
    if a.serve_plan:
        return serve_plan(a, plan, out, net)
    a.tcp = tcp_rtt(a.rtt_target)
    print(f"RTT\t{json.dumps(a.tcp)}", flush=True)
    sid = "sweep-" + hashlib.sha256(json.dumps([a.relation, a.start, a.runs, time.time()]).encode()).hexdigest()[:12]
    points = []
    total = plan[0] if plan else a.start
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
        if tp is None or (not plan and len(tps) >= 3 and tps[-1] < 1.02 * tps[-3]):
            break
        if plan:
            if len(points) == len(plan):
                break
            total = plan[len(points)]
        else:
            total *= 2
    ok = [q for q in points if q["vu_per_second"]]
    if not ok:
        return 1
    clean = [q for q in ok if not q["contention"].get("contended")] or ok
    best = max(clean, key=lambda q: (q["vu_per_second"], -q["total"]))  # a contended point is never the plateau
    tps = [q["vu_per_second"] for q in ok]
    done_at = next((i for i in range(2, len(tps)) if tps[i] < 1.02 * tps[i - 2]), None)
    stop = "a point failed" if points[-1]["vu_per_second"] is None else (
        f"< 2 % over two doublings at {ok[done_at]['total']}" if done_at is not None else "reached the last planned point / --max")
    if a.loopback:
        best["loopback"] = loopback_probe(a, best, out, net)
        print(f"LOOPBACK\t{json.dumps(best['loopback'])}", flush=True)
    for q in ok:
        sweep = {"id": sid, "point": points.index(q), "plateau": q is best, "total_vus": q["total"], "rule": RULE, "stopped": stop,
                 "points": [[x["total"], x["vu_per_second"]] for x in points]}
        doc = result(a, q, sweep)
        (out / q["dir"] / "result.json").write_text(json.dumps(doc, indent=1, default=str))
        if q is best and os.environ.get("RESEARCH_RUN_DIR"):
            rd = Path(os.environ["RESEARCH_RUN_DIR"]).resolve()
            pd = (out / q["dir"]).resolve()
            if pd.is_relative_to(rd):
                doc["artifacts"] = [str(pd.relative_to(rd))]
            Path(rd, "result.json").write_text(json.dumps(doc, indent=1, default=str))
    summary = {"id": sid, "relation": a.relation, "rule": RULE, "stopped": stop, "plateau_total_vus": best["total"],
               "points": [{k: x.get(k) for k in ("total", "ok", "vu_per_second", "wall_s", "peak_rss", "dir", "err")} for x in points]}
    (out / "sweep.json").write_text(json.dumps(summary, indent=1))
    print("SWEEP\t" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
