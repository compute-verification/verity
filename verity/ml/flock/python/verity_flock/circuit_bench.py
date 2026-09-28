"""The TABLES sweep for ``flock-circuit`` (``verity/flock-circuit``, or with ``--typed`` the typed statement
``verity/flock-circuit/types``): one ``result.json`` per point, as ``verity_flock.ir_bench``
does for ``flock-ir-frame`` (whose guard, probes and helpers, from ``verity_flock.bench``, this reuses).

    python -m verity_flock.circuit_bench --template rope-head --input-set DIR --bin target/release/flock-circuit --out-dir D \\
        --points "256 1024" [--per-proof N] [--runs 5] [--verifier HOST:PORT] [--rtt-target HOST:PORT] [--loopback] [--typed]
    python -m verity_flock.circuit_bench ... --serve-plan --listen 0.0.0.0:7501        the verifier pod's side

A point is the first ``total`` instances of the input set, proved in sequential sub-batches of ``--per-proof``. The prover stages
its instance files (rows included); the verifier stages only its own public files (roots, row digests, outputs; no row) from its
own copy of the input set, and the composite circuit, whose sha256 is the pin, from the lowering. The plateau is the point with
the highest median throughput over the planned points; ``--loopback`` re-runs it against a verifier on 127.0.0.1. A result
names the pinned circuit's statement, counts ANDs as the prover does on either statement (tensor-core units + tail,
:data:`ANDS_COUNTED`), and names the verifier that accepted its sessions.
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

from . import circuit as FC
from . import typed_statement as TS
from .bench import PER_PROOF_LOG2, PROFILES, _OURS, _guard, _lines, _probe, _quota, _sampler, _wait_serving, subbatches, tcp_rtt

RULE = "the planned points (instances of the input set, sequential sub-batches of --per-proof); plateau = the point with the highest median throughput"


def relation(template: str) -> str:
    return f"circuit:{template}"


#: Both pods stage from one salt key, derived from the set's content digest, so the verifier pod stages its own public files: the
#: salts are public in a benchmark, which measures proving, not the serving side's hiding (the checking process never reads a
#: salt: its file has none).
STAGE_SALTS = "public benchmark key: sha256('verity/flock-circuit/bench-stage-salts' || content_digest)"


def salt_key(input_set) -> bytes:
    return hashlib.sha256(b"verity/flock-circuit/bench-stage-salts\0" + bytes.fromhex(input_set.content_digest)).digest()


def stage_point(a, low, input_set, total: int, d: Path, public: bool) -> tuple[list[tuple[int, int]], float]:
    d.mkdir(parents=True, exist_ok=True)
    subs = subbatches(total, a.per_proof)
    t0 = time.time()
    stage = TS.stage if a.typed else FC.stage
    for i, (lo, hi) in enumerate(subs):
        p = stage(low, input_set, lo, hi, d / f"stage-s{i}", public_only=public, salt_key=salt_key(input_set))
        os.replace(p, d / f"instances-s{i}.bin")
    return subs, time.time() - t0


#: what ``ands_per_instance`` counts, on the flat and the typed statement alike
ANDS_COUNTED = "tensor-core units + tail: the instance's AND items as the prover counts them (the rows' SHA-512 and hm96 proved, not counted)"


def ands(first: dict) -> int:
    """AND items per instance, as the prover reports them (``ands_per_vu``): its tensor-core units and its tail."""
    v = first["ands_per_vu"]
    return v[0] if isinstance(v, list) else v


def rows(first: dict) -> int:
    """Committed rows per instance, as the prover reports them (``rows_per_vu``): every circuit but the rows'."""
    v = first["rows_per_vu"]
    return v[0] if isinstance(v, list) else v


def _merge(parts: list[list[dict]]) -> list[dict]:
    out = []
    for runs in zip(*parts):
        m = {"accepted": all(r.get("accepted") is True for r in runs), "warm": runs[0].get("warm"), "run": runs[0].get("run"), "subbatches": len(runs)}
        for k in ("e2e_s", "witness_s", "wait_s", "wait_e2e_s", "calls", "wire_sent", "wire_recv"):
            m[k] = sum(r[k] for r in runs)
        m["prove_s"] = [sum(r["prove_s"][i] for r in runs) for i in range(2)]
        m["verdict"] = {k: sum(r["verdict"].get(k, 0) or 0 for r in runs) for k in ("rounds", "verify_s", "handle_s")}
        m["proof_bytes"] = [sum(r["proof_bytes"][i] for r in runs) for i in range(2)]
        for k in ("roots", "sigma", "dense_m", "k_log", "statement", "statement_digest", "prover", "circuit_sha512", "public_sha512", "unit_draw", "open_rtt_ms", "identity",
                  "ands_per_vu", "rows_per_vu"):
            m[k] = [r.get(k) for r in runs] if len(runs) > 1 else runs[0].get(k)
        m["instances"] = sum(r.get("instances") or 0 for r in runs)
        b = {"t.witness": 0.0, "t.encoding_commitment": 0.0, "t.arithmetic": 0.0}
        for r in runs:
            b["t.witness"] += r["witness_s"]
            for rep in r.get("buckets") or []:
                b["t.witness"] += rep.get("t.witness", 0.0)
                b["t.encoding_commitment"] += rep.get("t.encoding_commitment", 0.0)
                b["t.arithmetic"] += rep.get("t.arithmetic", rep.get("t.prove", 0.0))
        m["buckets"] = b
        out.append(m)
    return out


def point(a, low, input_set, total: int, d: Path, net: Path, pin: str) -> dict:
    subs, gen_s = stage_point(a, low, input_set, total, d, public=False)
    if not a.verifier:
        stage_point(a, low, input_set, total, d / "verifier", public=True)
    tg = _guard()
    guard = tg.Guard(sampler=_sampler).start()
    t1 = time.time()
    parts, rcs, rss = [], [], None
    for i, _ in enumerate(subs):
        srv, addr = None, a.verifier
        if addr:
            _probe(addr)
        else:
            addr = "127.0.0.1:%d" % (7500 + total.bit_length())
            log = d / f"serve-s{i}.log"
            srv = subprocess.Popen(["nice", "-n", "5", a.bin, "serve", "--listen", addr, "--out", str(d / f"sessions-s{i}"), "--pin", pin,
                                    "--instances", str(d / "verifier" / f"instances-s{i}.bin"), "--circuit", str(net), *a.tables],
                                   stdout=open(log, "w"), stderr=subprocess.STDOUT)
            _wait_serving(log, srv)
        try:
            timer = ["/usr/bin/time", "-v", "-o", str(d / f"prove-s{i}.time")] if Path("/usr/bin/time").exists() else []
            pr = subprocess.Popen([*timer, a.bin, "prove", "--verifier", addr, "--warm", "1", "--runs", str(a.runs), "--dump", str(d / f"proofs-s{i}"),
                                   *a.prove_arg, "--instances", str(d / f"instances-s{i}.bin"), "--circuit", str(net), *a.tables],
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
    stage_point(a, a.low, a.input_set, q["total"], d / "verifier", public=True)
    per = []
    for i, _ in enumerate(q["subbatches"]):
        log = d / f"loopback-serve-s{i}.log"
        srv = subprocess.Popen([a.bin, "serve", "--listen", "127.0.0.1:7591", "--out", str(d / f"loopback-sessions-s{i}"), "--pin", pin,
                                "--instances", str(d / "verifier" / f"instances-s{i}.bin"), "--circuit", str(net), *a.tables],
                               stdout=open(log, "w"), stderr=subprocess.STDOUT)
        try:
            _wait_serving(log, srv)
            subprocess.call([a.bin, "prove", "--verifier", "127.0.0.1:7591", "--warm", "1", "--runs", str(a.runs), *a.prove_arg,
                             "--instances", str(d / f"instances-s{i}.bin"), "--circuit", str(net), *a.tables],
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
    and_count = ands(first) * n
    # a GEMM coordinate's result names its campaign target and K, as flock-pure's cells do (bench.PROFILES)
    gemm = sub.template.id == "gemm-coordinate"
    fp = contract.fingerprint(
        profile=PROFILES["bf16-ampere"] if gemm else None, domain=low.domain, K=int(sub.params["K"]) if gemm else None, B=n, proof_class="NON_ZK_PROOF", authentication="included-hash", zk_mode="none",
        security={"target": -128, "achieved_log2": PER_PROOF_LOG2 + math.log2(len(p["subbatches"])), "per_proof_log2": PER_PROOF_LOG2,
                  "proofs": len(p["subbatches"]), "union": "union bound over the batch's sequential sub-batch proofs",
                  "profile": "flock-128-r2", "reps": 2, "per_rep_log2": -97.8, "commitments": "one root per proof, bound by both reps (R1)",
                  "coins": FC.coins(p["live"])},
        hardware=contract.probe_hardware(),
        software={"backend": {"name": Path(a.bin).name, "configuration": f"C-interactive/{Path(a.bin).name}", "commit": a.commit,
                              "statement": a.meta["statement"], "prover": first.get("prover"), "prove_args": a.prove_arg, "lowering_sha256": pin,
                              "circuit_format": a.meta["format"], "unit": low.name, "unit_sha256": low.sha256, "identity": first.get("identity"),
                              "flock": "succinctlabs/flock b684b12 + verity/ml/flock/flock-link-b684b12.patch + cuda_circuit_patch.py"},
                  "toolchain": {"rustc": a.rustc, "python": platform.python_version()}},
        preprocessing="none: the circuit shape and the composite circuit are built once per process (fixed.setup_seconds: staging); every "
                      "timed session hashes its rows and evaluates every unit afresh (the witness bucket) before proving",
        retention="retained", instances=input_set.ref(0, n, a.art), N_subbatches=len(p["subbatches"]),
        batching={"total": n, "per_proof": a.per_proof, "in_flight": 1, "subbatches": [list(x) for x in p["subbatches"]]},
        scheme=FC.SCHEME, statement=a.meta["statement"], relation=relation(a.template), subcircuit=sub.id, subcircuit_template=sub.template.id,
        roots=first["roots"], sigma=first["sigma"], dense_m=first["dense_m"], units_per_instance=low.units.n, ands_per_instance=ands(first),
        ands_counted=ANDS_COUNTED, rows_per_instance=rows(first),
        private="every word but the frame-v3 roots, the row digests, the declared outputs and the pinned layout",
        rtt_method="net.rtt_ms = 64-byte Ping round trips on the open session connection before Hello, per timed session, median",
        verifier={"placement": "loopback" if not a.verifier else a.verifier, "process": f"{Path(a.bin).name} serve (separate process)",
                  "accepted_by": f"{Path(a.bin).name} serve: the Rust live verifier (flock-live R1-R8); no Lean replay in this run",
                  "instance_file": "its own public file (no rows, no salts)", "stage_salts": STAGE_SALTS})
    ms = [contract.measurement("t.total", t_e2e), contract.measurement("t.total_live", t_e2e), contract.measurement("vu.count", n),
          contract.measurement("vu.seconds_per_vu", t_e2e / n), contract.measurement("vu.per_second", n / t_e2e),
          {"name": "ands.count", "value": and_count, "unit": "count"}, {"name": "ands.per_second", "value": and_count / t_e2e, "unit": "1/s"},
          {"name": "e2e.seconds", "value": t_e2e, "unit": "s"}, {"name": "e2e.vu_per_second", "value": n / t_e2e, "unit": "1/s"},
          {"name": "e2e.seconds_min", "value": min(e2e), "unit": "s"}, {"name": "e2e.seconds_max", "value": max(e2e), "unit": "s"},
          {"name": "witness.host_seconds", "value": med("witness_s"), "unit": "s"},
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
          {"name": "live.prover_compute_seconds", "value": statistics.median(r["e2e_s"] - r["wait_e2e_s"] for r in timed), "unit": "s"},
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
                                           "live.prover_compute_seconds = e2e_s - wait_e2e_s per timed session (the waits inside the "
                                           "e2e window), median; net.wait_seconds also counts the proof uploads and the verdict after it"},
           "protocol": {"warm": True, "runs": len(timed), "statistic": "median", "contended": p["contention"].get("contended"),
                        "warm_rule": "one full untimed session (prove --warm 1) before the timed sessions, same process"},
           "runs": [{"e2e_s": r["e2e_s"], "witness_s": r["witness_s"], "prove_s": r["prove_s"], "wait_s": r["wait_s"],
                     "wait_e2e_s": r["wait_e2e_s"]} for r in timed]}
    doc["validation"]["contract_problems"] = contract.validate(doc)
    return doc


def serve_plan(a, low, input_set, plan: list[int], out: Path, net: Path, pin: str) -> int:
    """The verifier pod's side: its own public files for every point and sub-batch (from its own input set, never the prover's),
    then one ``serve --sessions`` per sub-batch in the prover's order (probe + warm + runs)."""
    jobs = []
    for i, total in enumerate(plan):
        d = out / f"p{i}-{total}"
        subs, _ = stage_point(a, low, input_set, total, d, public=True)
        jobs += [(d, j) for j in range(len(subs))]
    print(f"PLAN\t{json.dumps({'points': plan, 'servers': len(jobs)})}", flush=True)
    for d, j in jobs:
        rc = subprocess.call([a.bin, "serve", "--listen", a.listen, "--out", str(d / f"sessions-s{j}"), "--pin", pin,
                              "--instances", str(d / f"instances-s{j}.bin"), "--circuit", str(net), "--sessions", str(a.runs + 2),
                              "--operator", a.operator, *a.tables], stdout=open(d / f"serve-s{j}.log", "w"), stderr=subprocess.STDOUT)
        print(f"SERVED\t{d.name}/s{j} rc={rc}", flush=True)
    return 0


def main(argv=None) -> int:
    from verity_numerical.bench.input_sets import InputSet

    ap = argparse.ArgumentParser(prog="verity_flock.circuit_bench")
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
    ap.add_argument("--listen", default="0.0.0.0:7501")
    ap.add_argument("--operator", default="")
    ap.add_argument("--rtt-target", default="")
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--typed", action="store_true", help="stage and prove the typed statement (verity/flock-circuit/types)")
    a = ap.parse_args(argv)
    a.rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    input_set = InputSet.open(a.input_set)
    if input_set.subcircuit.template.id != a.template:
        print(f"input set {input_set.name} is not a {a.template} set", file=sys.stderr)
        return 2
    low = FC.lowering_for_set(input_set)
    a.low, a.input_set = low, input_set
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = out / "circuit.txt"
    statics = FC.set_statics(input_set)
    net.write_text(TS.compose(low, input_set.subcircuit, statics=statics).text if a.typed else FC.compose(low, input_set.subcircuit, statics=statics))
    pin = hashlib.sha512(net.read_bytes()).hexdigest()
    a.meta = json.loads(net.read_text().split("\n", 2)[1][len("META "):])
    a.tables = []
    if low.units.cut:
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
        print(f"    ok={p['ok']} rc={p.get('rc')} inst/s={p['vu_per_second']} and/s={p['vu_per_second'] and p['vu_per_second'] * ands(p['timed'][0])} "
              f"wall={p.get('wall_s')} contended={p.get('contention', {}).get('contended')} rss={p.get('peak_rss')} err={p.get('err')}", flush=True)
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
