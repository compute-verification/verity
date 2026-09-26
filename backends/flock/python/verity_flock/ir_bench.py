"""The TABLES sweep for ``flock-ir-frame`` (``verity/flock-ir-frame/v1``): one ``result.json`` per point, stamped with its
``sweep`` and ``protocol``, as ``verity_flock.bench`` does for the GEMM cells (whose guard, probes and helpers this reuses).

    python -m verity_flock.ir_bench --template rope-head --input-set DIR --bin target/release/flock-ir-frame --out-dir D \\
        --points "64 128 256 512 1024" [--per-proof N] [--runs 5] [--verifier HOST:PORT] [--rtt-target HOST:PORT] [--loopback]
    python -m verity_flock.ir_bench ... --serve-plan --listen 0.0.0.0:7400        the verifier pod's side

A point is the first ``total`` instances of the input set, proved in sequential sub-batches of ``--per-proof``. Each side stages
its own instance files from its own copy of the input set (``verity_flock.ir_frame.stage``; the verifier never reads the
prover's). The prover runs ``flock-ir-frame prove --warm 1 --runs R --dump`` per sub-batch against the verifier (loopback
unless ``--verifier``). The plateau is the point with the highest median throughput over the planned points (a point that
fails ends the sweep); ``--loopback`` re-runs the plateau point against a verifier on 127.0.0.1 for the per-round probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .bench import PER_PROOF_LOG2, _OURS, _guard, _lines, _probe, _quota, _sampler, _wait_serving, subbatches, tcp_rtt

RULE = "the planned points (instances of the input set, sequential sub-batches of --per-proof); plateau = the point with the highest median throughput"
STATEMENT = "verity/flock-ir-frame/v3"
SCHEME = "frame-v3/blake3-keyed"


def relation(template: str) -> str:
    return f"irf:{template}"


def _lowering(template: str, input_set):
    from verity_numerical.bench import lowerings

    mod = lowerings.registry("C-Flock").module(template)
    if hasattr(mod, "lowering_for_set"):
        return mod.lowering_for_set(input_set)
    return mod.frame_lowering(input_set.subcircuit)


def frame_module(template: str):
    """The statement's staging module: ``ir_sampling`` (``verity/flock-ir-sampling/v1``) for the sampling template, else
    ``ir_frame``. Each has ``STATEMENT``, ``stage(low, input_set, lo, hi, dir)`` and ``serve_args(path)``."""
    from . import ir_frame, ir_sampling

    return ir_sampling if template == ir_sampling.TEMPLATE else ir_frame


def stage_point(a, low, input_set, total: int, d: Path) -> tuple[list[tuple[int, int]], float]:
    d.mkdir(parents=True, exist_ok=True)
    subs = subbatches(total, a.per_proof)
    t0 = time.time()
    for i, (lo, hi) in enumerate(subs):
        p = frame_module(a.template).stage(low, input_set, lo, hi, d / f"stage-s{i}")
        os.replace(p, d / f"instances-s{i}.bin")
    return subs, time.time() - t0


def _sum_buckets(r: dict) -> dict:
    out = {"t.witness": 0.0, "t.encoding_commitment": 0.0, "t.arithmetic": 0.0}
    for rep in r.get("buckets") or []:
        if "t.total" in rep:
            out["t.witness"] += rep.get("t.witness_host", 0.0) + rep.get("t.witness", 0.0)
            out["t.encoding_commitment"] += rep.get("t.encoding_commitment", 0.0)
            out["t.arithmetic"] += rep.get("t.arithmetic", 0.0)
        else:
            out["t.witness"] += rep.get("t.witness", 0.0)
            out["t.arithmetic"] += rep.get("t.prove", 0.0)
    return out


def _merge(parts: list[list[dict]]) -> list[dict]:
    out = []
    for runs in zip(*parts):
        m = {"accepted": all(r.get("accepted") is True for r in runs), "warm": runs[0].get("warm"), "run": runs[0].get("run"),
             "subbatches": len(runs)}
        for k in ("e2e_s", "rows_s", "wait_s", "calls", "wire_sent", "wire_recv"):
            m[k] = sum(r[k] for r in runs)
        m["prove_s"] = [sum(r["prove_s"][i] for r in runs) for i in range(2)]
        m["verdict"] = {k: sum(r["verdict"].get(k, 0) or 0 for r in runs) for k in ("rounds", "verify_s", "handle_s")}
        m["proof_bytes"] = [sum(r["proof_bytes"][i] for r in runs) for i in range(2)]
        for k in ("roots", "sigma", "dense_m", "statement", "statement_digest", "prover", "netlist_sha256", "public_sha256", "open_rtt_ms"):
            m[k] = [r.get(k) for r in runs] if len(runs) > 1 else runs[0].get(k)
        for k in ("n_blake3", "n_units", "instances"):
            m[k] = sum(r.get(k) or 0 for r in runs)
        bk = [_sum_buckets(r) for r in runs]
        m["buckets"] = {b: sum(x[b] for x in bk) for b in bk[0]}
        out.append(m)
    return out


def point(a, low, input_set, total: int, d: Path, net: Path, pin: str) -> dict:
    subs, gen_s = stage_point(a, low, input_set, total, d)
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
                                    *common, *a.tables, *frame_module(a.template).serve_args(d / f"instances-s{i}.bin")],
                                   stdout=open(log, "w"), stderr=subprocess.STDOUT)
            _wait_serving(log, srv)
        try:
            timer = ["/usr/bin/time", "-v", "-o", str(d / f"prove-s{i}.time")] if Path("/usr/bin/time").exists() else []
            pr = subprocess.Popen([*timer, a.bin, "prove", "--verifier", addr, "--warm", "1", "--runs", str(a.runs),
                                   "--dump", str(d / f"proofs-s{i}"), *a.prove_arg, *common, *a.tables],
                                  stdout=open(d / f"prove-s{i}.txt", "w"), stderr=open(d / f"prove-s{i}.err", "w"))
            _OURS.add(pr.pid)
            try:
                rcs.append(pr.wait())
                time.sleep(2)
            finally:
                _OURS.discard(pr.pid)
        finally:
            if srv is not None:
                time.sleep(1)
                srv.terminate()
                srv.wait()
        live = _lines(d / f"prove-s{i}.txt", "LIVE") if (d / f"prove-s{i}.txt").exists() else []
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
    ok = ok and len(timed) == a.runs
    return {"total": total, "rc": max(rcs) if rcs else None, "ok": ok, "live": live, "timed": timed, "gen_s": gen_s, "wall_s": wall,
            "contention": contention, "peak_rss": rss, "dir": d.name, "subbatches": subs}


def loopback_probe(a, q: dict, out: Path, net: Path, pin: str) -> dict:
    d = out / q["dir"]
    per = []
    for i, _ in enumerate(q["subbatches"]):
        common = ["--instances", str(d / f"instances-s{i}.bin"), "--netlist", str(net)]
        log = d / f"loopback-serve-s{i}.log"
        srv = subprocess.Popen([a.bin, "serve", "--listen", "127.0.0.1:7391", "--out", str(d / f"loopback-sessions-s{i}"), "--pin", pin,
                                *common, *a.tables, *frame_module(a.template).serve_args(d / f"instances-s{i}.bin")],
                               stdout=open(log, "w"), stderr=subprocess.STDOUT)
        try:
            _wait_serving(log, srv)
            subprocess.call([a.bin, "prove", "--verifier", "127.0.0.1:7391", "--warm", "1", "--runs", str(a.runs), *a.prove_arg, *common, *a.tables],
                            stdout=open(d / f"loopback-prove-s{i}.txt", "w"), stderr=open(d / f"loopback-prove-s{i}.err", "w"))
        finally:
            srv.terminate()
            srv.wait()
        for r in _lines(d / f"loopback-prove-s{i}.txt", "LIVE"):
            if not r.get("warm") and r.get("accepted") and r.get("calls"):
                per.append(r["wait_s"] / r["calls"])
    return {"seconds_per_round": statistics.median(per) if per else None, "sessions": len(per),
            "method": f"loopback probe in this run: {len(per)} timed sessions ({a.runs} per sub-batch) of the plateau point's instance "
                      f"files with the same binary against `serve` on 127.0.0.1, median per-round coin wait (wait_s / calls)"}


def result(a, input_set, low, p: dict, sweep: dict, pin: str) -> dict:
    from verity_numerical.bench import contract

    timed = p["timed"]
    med = lambda k: statistics.median(r[k] for r in timed)  # noqa: E731
    e2e = [r["e2e_s"] for r in timed]
    t_e2e = statistics.median(e2e)
    net_wait = max(0.0, statistics.median(r["wait_s"] - r["verdict"].get("handle_s", 0.0) for r in timed))
    rtts = [x for r in timed for x in (r["open_rtt_ms"] if isinstance(r.get("open_rtt_ms"), list) else [r.get("open_rtt_ms")])
            if isinstance(x, (int, float)) and x == x]
    open_rtt = statistics.median(rtts) if rtts else None
    n = p["total"]
    first = timed[0]
    sub = input_set.subcircuit
    statement = frame_module(a.template).STATEMENT
    fp = contract.fingerprint(
        profile=None, domain="finite", K=None, B=n, proof_class="NON_ZK_PROOF", authentication="included-hash", zk_mode="none",
        security={"target": -128, "achieved_log2": PER_PROOF_LOG2 + math.log2(len(p["subbatches"])), "per_proof_log2": PER_PROOF_LOG2,
                  "proofs": len(p["subbatches"]), "union": "union bound over the batch's sequential sub-batch proofs",
                  "profile": "flock-128-r2", "reps": 2, "per_rep_log2": -97.8, "coins": "live verifier (flock-live R1-R8)"},
        hardware=contract.probe_hardware(),
        software={"backend": {"name": Path(a.bin).name, "configuration": f"C-interactive/{Path(a.bin).name}", "commit": a.commit,
                              "statement": statement, "prover": first.get("prover"), "prove_args": a.prove_arg, "lowering_sha256": pin,
                              "unit": low.name, "flock": "succinctlabs/flock b684b12 + backends/flock/flock-link-b684b12.patch"},
                  "toolchain": {"rustc": a.rustc, "python": platform.python_version()}},
        preprocessing="none: the circuit shape and the unit netlist are built once per process (fixed.setup_seconds: staging); every "
                      "timed session hashes its batch's rows afresh (commit.seconds) before proving",
        retention="retained", instances=input_set.ref(0, n, a.art), N_subbatches=len(p["subbatches"]),
        batching={"total": n, "per_proof": a.per_proof, "in_flight": 1, "subbatches": [list(x) for x in p["subbatches"]]},
        scheme=SCHEME, statement=statement, relation=relation(a.template), subcircuit=sub.id, subcircuit_template=sub.template.id,
        roots=first["roots"], sigma=first["sigma"], dense_m=first["dense_m"], units_per_instance=low.units.n,
        n_blake3=first["n_blake3"], n_units=first["n_units"],
        rtt_method="net.rtt_ms = 64-byte Ping round trips on the open session connection before Hello, per timed session, median",
        verifier={"placement": "loopback" if not a.verifier else a.verifier, "process": f"{Path(a.bin).name} serve (separate process)"})
    ms = [contract.measurement("t.total", t_e2e), contract.measurement("t.total_live", t_e2e), contract.measurement("vu.count", n),
          contract.measurement("vu.seconds_per_vu", t_e2e / n), contract.measurement("vu.per_second", n / t_e2e),
          {"name": "commit.seconds", "value": med("rows_s"), "unit": "s"},
          {"name": "e2e.seconds", "value": t_e2e, "unit": "s"}, {"name": "e2e.vu_per_second", "value": n / t_e2e, "unit": "1/s"},
          {"name": "e2e.seconds_min", "value": min(e2e), "unit": "s"}, {"name": "e2e.seconds_max", "value": max(e2e), "unit": "s"},
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
          {"name": "live.prover_compute_seconds", "value": t_e2e - med("wait_s"), "unit": "s"},
          contract.measurement("transcript.bytes", int(first["wire_sent"]) + int(first["wire_recv"])),
          contract.measurement("verify.wall_s", statistics.median(r["verdict"]["verify_s"] for r in timed)),
          {"name": "proof.bytes", "value": sum(first["proof_bytes"]), "unit": "B"},
          contract.measurement("fixed.setup_seconds", p["gen_s"])]
    bk = {b: statistics.median(r["buckets"][b] for r in timed) for b in ("t.witness", "t.encoding_commitment", "t.arithmetic")}
    if sum(bk.values()) <= t_e2e:
        ms += [contract.measurement(b, v) for b, v in bk.items()]
    if p["peak_rss"]:
        ms.append(contract.measurement("mem.peak_host_bytes", p["peak_rss"]))
    doc = {"schema": contract.RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID"), "workload_fingerprint": fp,
           "validation": {"status": "passed" if p["ok"] else "failed", "checker": f"{Path(a.bin).name} serve (live verifier, R1-R8 replay)",
                          "detail": f"{sum(1 for r in p['live'] if r.get('accepted'))}/{len(p['live'])} sessions accepted",
                          "evidence": [f"{p['dir']}/sessions/index.jsonl"]},
           "measurements": ms, "measurement_files": [], "artifacts": [], "sweep": sweep, "contention": p["contention"],
           "interaction": {"rtt_method": ("net.rtt_ms = round trip of a 64-byte Ping request on the open session connection to the "
                                          "verifier (answered Ok by the verifier's server, no session state), 30 probes right after "
                                          "connecting and before Hello, in every timed session; median over all of them") if open_rtt is not None else None,
                           "tcp_connect_method": (f"live.tcp_connect_rtt_ms = TCP connect time to the verifier pod's sshd ({a.tcp['target']}), "
                                                  f"median of {a.tcp['n']}, before the sweep") if getattr(a, "tcp", None) else None,
                           "coin_wait_method": "live.coin_wait_per_round_ms = (the prover's coin wait - the verifier's round-handling time) / "
                                               "round trips, median over the timed sessions; verifier " + ("on loopback" if not a.verifier else a.verifier),
                           "loopback_method": p.get("loopback", {}).get("method"), "verifier_operator": a.operator or None,
                           "t_total_includes_wait": True,
                           "t_total_note": "live coins: t.total and the buckets include the prover's coin waits; prover compute alone is "
                                           "live.prover_compute_seconds = t.total - net.wait_seconds"},
           "protocol": {"warm": True, "runs": len(timed), "statistic": "median", "contended": p["contention"].get("contended"),
                        "warm_rule": "one full untimed session (prove --warm 1) before the timed sessions, same process"},
           "runs": [{"e2e_s": r["e2e_s"], "rows_s": r["rows_s"], "prove_s": r["prove_s"], "wait_s": r["wait_s"]} for r in timed]}
    doc["validation"]["contract_problems"] = contract.validate(doc)
    return doc


def serve_plan(a, low, input_set, plan: list[int], out: Path, net: Path, pin: str) -> int:
    """The verifier pod's side: its own instance files for every point and sub-batch (staged here from its own input set, never
    the prover's), then one ``serve --sessions`` per sub-batch in the prover's order (probe + warm + runs)."""
    jobs = []
    for i, total in enumerate(plan):
        d = out / f"p{i}-{total}"
        subs, _ = stage_point(a, low, input_set, total, d)
        jobs += [(d, j) for j in range(len(subs))]
    print(f"PLAN\t{json.dumps({'points': plan, 'servers': len(jobs)})}", flush=True)
    for d, j in jobs:
        rc = subprocess.call([a.bin, "serve", "--listen", a.listen, "--out", str(d / f"sessions-s{j}"), "--pin", pin,
                              "--instances", str(d / f"instances-s{j}.bin"), "--netlist", str(net), "--sessions", str(a.runs + 2),
                              "--operator", a.operator, *a.tables, *frame_module(a.template).serve_args(d / f"instances-s{j}.bin")],
                             stdout=open(d / f"serve-s{j}.log", "w"), stderr=subprocess.STDOUT)
        print(f"SERVED\t{d.name}/s{j} rc={rc}", flush=True)
    return 0


def main(argv=None) -> int:
    from verity_numerical.bench.input_sets import InputSet

    ap = argparse.ArgumentParser(prog="verity_flock.ir_bench")
    ap.add_argument("--template", required=True)
    ap.add_argument("--input-set", required=True)
    ap.add_argument("--art", default=None, help="the input set's art: id, recorded in the fingerprint")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--points", required=True, help="instance counts (both sides of a remote verifier must agree)")
    ap.add_argument("--per-proof", type=int, default=1 << 30)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--verifier", default="")
    ap.add_argument("--procs", type=int, default=_quota())
    ap.add_argument("--commit", default=os.environ.get("RESEARCH_SOURCE_COMMIT", "unknown"))
    ap.add_argument("--prove-arg", action="append", default=[])
    ap.add_argument("--serve-plan", action="store_true")
    ap.add_argument("--listen", default="0.0.0.0:7400")
    ap.add_argument("--operator", default="")
    ap.add_argument("--rtt-target", default="")
    ap.add_argument("--loopback", action="store_true")
    a = ap.parse_args(argv)
    a.rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    input_set = InputSet.open(a.input_set)
    if input_set.subcircuit.template.id != a.template:
        print(f"input set {input_set.name} is not a {a.template} set", file=sys.stderr)
        return 2
    low = _lowering(a.template, input_set)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = out / "net.txt"
    net.write_text(low.text)
    pin = hashlib.sha256(net.read_bytes()).hexdigest()
    a.tables = []
    if low.units.cut:  # the cut templates' tails, and the sampling statement's native DivFullRcp
        from . import ir_lower

        ir_lower.write_tables(out / "tables")
        a.tables = ["--tables", str(out / "tables")]
    plan = [int(x) for x in a.points.split()]
    if max(plan) > input_set.n:
        print(f"point {max(plan)} exceeds the input set's {input_set.n} instances", file=sys.stderr)
        return 2
    if a.serve_plan:
        return serve_plan(a, low, input_set, plan, out, net, pin)
    a.tcp = tcp_rtt(a.rtt_target)
    print(f"RTT\t{json.dumps(a.tcp)}", flush=True)
    sid = "sweep-" + hashlib.sha256(json.dumps([a.template, input_set.content_digest, a.runs, time.time()]).encode()).hexdigest()[:12]
    points = []
    for total in plan:
        print(f"=== {time.strftime('%H:%M:%SZ', time.gmtime())} point {len(points)} total {total}", flush=True)
        try:
            p = point(a, low, input_set, total, out / f"p{len(points)}-{total}", net, pin)
        except Exception as e:  # noqa: BLE001  (a failed point ends the sweep)
            p = {"total": total, "ok": False, "err": str(e), "timed": [], "dir": f"p{len(points)}-{total}", "contention": {}}
        p["vu_per_second"] = (total / statistics.median(r["e2e_s"] for r in p["timed"])) if p["ok"] else None
        points.append(p)
        print(f"    ok={p['ok']} rc={p.get('rc')} inst/s={p['vu_per_second']} wall={p.get('wall_s')} contended="
              f"{p.get('contention', {}).get('contended')} rss={p.get('peak_rss')} err={p.get('err')}", flush=True)
        if not p["ok"]:
            break
    ok = [q for q in points if q["vu_per_second"]]
    if not ok:
        return 1
    clean = [q for q in ok if not q["contention"].get("contended")] or ok
    best = max(clean, key=lambda q: (q["vu_per_second"], -q["total"]))
    stop = "a point failed" if points[-1]["vu_per_second"] is None else "reached the last planned point"
    if a.loopback:
        best["loopback"] = loopback_probe(a, best, out, net, pin)
        print(f"LOOPBACK\t{json.dumps(best['loopback'])}", flush=True)
    for q in ok:
        sweep = {"id": sid, "point": points.index(q), "plateau": q is best, "total_vus": q["total"], "rule": RULE, "stopped": stop,
                 "points": [[x["total"], x["vu_per_second"]] for x in points]}
        doc = result(a, input_set, low, q, sweep, pin)
        (out / q["dir"] / "result.json").write_text(json.dumps(doc, indent=1, default=str))
        if q is best and os.environ.get("RESEARCH_RUN_DIR"):
            rd = Path(os.environ["RESEARCH_RUN_DIR"]).resolve()
            pd = (out / q["dir"]).resolve()
            if pd.is_relative_to(rd):
                doc["artifacts"] = [str(pd.relative_to(rd))]
            Path(rd, "result.json").write_text(json.dumps(doc, indent=1, default=str))
    summary = {"id": sid, "template": a.template, "rule": RULE, "stopped": stop, "plateau_total": best["total"],
               "points": [{k: x.get(k) for k in ("total", "ok", "vu_per_second", "wall_s", "peak_rss", "dir", "err")} for x in points]}
    (out / "sweep.json").write_text(json.dumps(summary, indent=1))
    print("SWEEP\t" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
