"""P2 v1's scale sweep on a shared GPU node: honest acceptance and forward overhead over weight sets, T and repetitions, one
cell per `research run`, so a preempted run loses one cell.

    python benchmarks/pous/p2_v1/sweep.py cell SEED --work DIR [--audits 500] [--control-audits 50]
    python benchmarks/pous/p2_v1/sweep.py aggregate RUN_DIR... --out TABLE.json [--markdown TABLE.md]

`cell` is a `research keeper` chunk's command (`{seed}` for SEED) or a queued run's: SEED names the cell (`cell_of`: every T
of the first weight set at repetition 0, then the next weight sets, then the next repetition, so a curve exists early). Inside
the run's GPU lease it picks three idle whole cores on the leased GPU's NUMA node from the CPUs it may use (`pick_cores`: the
verifier's, the responder's and the sampler's; their SMT siblings stay unused, so a sibling's busy share is a neighbour's),
samples every CPU's busy share, the load average and every GPU at 1 Hz into `out/neighbors.jsonl`, and runs `pod.sh
IMPL=p2-v1-gpu` with the loop at T and the honest audit under forwards at T, its outputs in `out/` (`--declared-output
'out/*'`), after `prepare` has built the venv, CGBN and the library the cells share in `--work`. The setup (W, C, the salt) is the harness's deterministic one, so a repetition differs only in its audits' secret
draws and in its timing.

`aggregate` reads the cells' run dirs (`research fetch --all RUN --on NODE`): per weight set and T, the decode overhead (median
and range over repetitions) and the honest audits accepted out of run with a 95% Clopper-Pearson interval, at the run's limit
(`Δ` plus its calibrated loopback allowance, which the spec caps at 354 µs). Each late honest answer is attributed from its
own timings and the sampler's second (`attribute`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WEIGHT_SETS = ("qwen2.5-0.5b/linears+head", "qwen2.5-7b/layer0+head", "llama3.1-8b/layer0+head")
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
DELTA_NS = 500_000
RTT_CAP_NS = 354_000
BUSY = 50.0
"""A sibling thread or NUMA node at least this busy (percent, over the sampler's second) is load from a neighbour."""


def cell_of(seed: int) -> dict[str, Any]:
    per = len(WEIGHT_SETS) * len(TOKENS)
    r = seed % per
    return {"seed": seed, "rep": seed // per, "weight_set": WEIGHT_SETS[r // len(TOKENS)], "tokens": TOKENS[r % len(TOKENS)]}


def _cpu_times() -> dict[int, tuple[int, int]]:
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            v = list(map(int, f[1:]))
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out


def _busy(a: dict[int, tuple[int, int]], b: dict[int, tuple[int, int]]) -> dict[int, float]:
    return {c: round(100 * (1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0])), 1) for c in b if c in a}


def _cpulist(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _gpu() -> dict[str, Any]:
    """The lease's GPU (GPU_LEASE_UUID, else CUDA_VISIBLE_DEVICES' first): its index, UUID, bus id and NUMA node."""
    want = (os.environ.get("GPU_LEASE_UUID") or os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0]
    rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    for row in rows:
        index, uuid, bus = (x.strip() for x in row.split(","))
        if want in (index, uuid) or (want and uuid.startswith(want)):
            node = Path(f"/sys/bus/pci/devices/{bus[-12:].lower()}/numa_node")
            numa = int(node.read_text()) if node.exists() else -1
            return {"index": int(index), "uuid": uuid, "bus_id": bus, "numa": numa}
    return {"index": None, "uuid": want or None, "bus_id": None, "numa": -1}


def pick_cores(allowed: set[int], numa_cpus: set[int], busy: dict[int, float], count: int = 3) -> list[list[int]]:
    """`count` whole cores (every SMT sibling allowed), the least busy first, on the GPU's NUMA node when it has enough."""
    cores: dict[tuple[int, ...], float] = {}
    for c in sorted(allowed):
        sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        threads = tuple(sorted(_cpulist(sib.read_text()))) if sib.exists() else (c,)
        if set(threads) <= allowed:
            cores[threads] = sum(busy.get(t, 100.0) for t in threads)
    near = {k: v for k, v in cores.items() if set(k) <= numa_cpus}
    pool = near if len(near) >= count else cores
    return [list(k) for k in sorted(pool, key=lambda k: (pool[k], k))[:count]]


def sample(path: Path, watch: list[int]) -> None:
    """Every second until killed: every CPU's busy share, the watched CPUs', the load average and every GPU's load."""
    out = path.open("a", buffering=1)
    prev = _cpu_times()
    while True:
        time.sleep(1.0)
        cur = _cpu_times()
        busy = _busy(prev, cur)
        prev = cur
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        gpus = [dict(zip(("i", "util", "mem_mib", "power_w", "sm_mhz"), (x.strip() for x in line.split(","))))
                for line in q.strip().splitlines()]
        n = max(busy) + 1 if busy else 0
        out.write(json.dumps({"t": time.time(), "loadavg": Path("/proc/loadavg").read_text().split()[:3],
                              "busy": [busy.get(c) for c in range(n)], "watch": {str(c): busy.get(c) for c in watch},
                              "gpus": gpus}) + "\n")


def prepare(work: str, pod: list[str], env: dict[str, str], out: Path) -> int:
    """The cells share WORK's venv, CGBN and builds: the first cell to get here builds them with a gates-only run on one
    segment (its outputs in `out/prepare/`) while the others wait on WORK's lock, and a marker keyed by this tree's pod.sh,
    p2v1.cu and Makefile lets every later cell skip it."""
    import fcntl
    import hashlib

    key = hashlib.sha256(b"".join((REPO / p).read_bytes() for p in ("benchmarks/pous/pod.sh", "benchmarks/pous/p2_v1/p2v1.cu",
                                                                     "benchmarks/pous/p2_v1/Makefile"))).hexdigest()[:16]
    Path(work).mkdir(parents=True, exist_ok=True)
    marker = Path(work) / f"prepared-{key}"
    with open(Path(work) / "prepare.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return 0
        (out / "prepare").mkdir(exist_ok=True)
        args = {**env, "OUT": str(out / "prepare"), "GATES_ONLY": "1", "MAX_SEGMENTS": "1", "AUDITS": "0"}
        rc = subprocess.run([*pod, *(f"{k}={v}" for k, v in args.items())], cwd=REPO, env={**os.environ, "RESEARCH_RUN_DIR": ""}
                            ).returncode
        if rc == 0:
            marker.write_text(json.dumps({"run": os.environ.get("RESEARCH_RUN_ID"), "at": time.time()}) + "\n")
        return rc


def run_cell(a: argparse.Namespace) -> int:
    run_dir = Path(os.environ["RESEARCH_RUN_DIR"])
    out = run_dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    c = cell_of(a.seed)
    gpu = _gpu()
    allowed = os.sched_getaffinity(0)
    numa_file = Path(f"/sys/devices/system/node/node{gpu['numa']}/cpulist")
    numa_cpus = _cpulist(numa_file.read_text()) if gpu["numa"] >= 0 and numa_file.exists() else set(allowed)
    t0 = _cpu_times()
    time.sleep(2.0)
    busy0 = _busy(t0, _cpu_times())
    cores = pick_cores(set(allowed), numa_cpus, busy0)
    if len(cores) < 3:
        print(f"sweep: fewer than 3 whole cores in the allowed CPUs {sorted(allowed)}", file=sys.stderr)
        return 3
    (vcore, rcore, score) = cores
    doc = {**c, "audits": a.audits, "control_audits": a.control_audits, "gpu": gpu, "allowed_cpus": sorted(allowed),
           "numa_cpus": sorted(numa_cpus), "verifier_core": vcore, "responder_core": rcore, "sampler_core": score,
           "busy_at_start": {str(t): busy0.get(t) for t in vcore + rcore + score},
           "loadavg_at_start": Path("/proc/loadavg").read_text().split()[:3], "host": os.uname().nodename,
           "started_unix": time.time()}
    (out / "cell.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("seed", "rep", "weight_set", "tokens", "gpu", "verifier_core", "responder_core")}),
          flush=True)
    sampler = subprocess.Popen(["taskset", "-c", str(score[0]), sys.executable, __file__, "sample", str(out / "neighbors.jsonl"),
                                "--watch", ",".join(map(str, vcore + rcore))])
    env = {"IMPL": "p2-v1-gpu", "WS": c["weight_set"], "HW": a.hardware, "AUDITS": str(a.audits),
           "CONTROL_AUDITS": str(a.control_audits), "TOKENS": str(c["tokens"]), "LOAD_TOKENS": str(c["tokens"]), "OUT": str(out),
           "WORK": a.work, "VCPU": str(vcore[0]), "RCPU": str(rcore[0]), "WORKERS": "2"}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    pod = ["bash", str(REPO / "benchmarks" / "pous" / "pod.sh")]
    rc = 1
    try:
        if (rc := prepare(a.work, pod, env, out)) != 0:
            return rc
        rc = subprocess.run([*pod, *(f"{k}={v}" for k, v in env.items())], cwd=REPO).returncode
    finally:
        sampler.terminate()
        (out / "bench" / "leaves-p2.bin").unlink(missing_ok=True)
    return cell_rc(rc, out / "bench" / "bench-result.json")


def cell_rc(rc: int, result: Path) -> int:
    """A late honest answer is what the sweep measures, not a failed cell: pod.sh's rc unless `timed_audit` is the only gate
    that failed, so a keeper counts only broken cells toward its pause."""
    if rc == 0 or not result.exists():
        return rc
    gates = json.loads(result.read_text()).get("gates") or {}
    return 0 if "timed_audit" in gates and all(v.get("passed") for g, v in gates.items() if g != "timed_audit") else rc


# --- aggregate --------------------------------------------------------------------------------------------------------------

def _binom_tail_ge(x: int, n: int, p: float) -> float:
    """P(X >= x) for X ~ Binomial(n, p)."""
    if x <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq for k in range(x, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """The exact two-sided `1 - alpha` interval for a binomial proportion, `x` successes of `n`."""
    if n == 0:
        return 0.0, 1.0

    def solve(f, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if x == 0 else (alpha / 2) ** (1 / n) if x == n else solve(lambda p: _binom_tail_ge(x, n, p), alpha / 2)
    upper = 1.0 if x == n else solve(lambda p: -(1 - _binom_tail_ge(x + 1, n, p)), -alpha / 2)
    return lower, upper


def _neighbour_at(samples: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    """The sampler's line whose second covers `t` (the first line at or after it, within 2 s)."""
    after = [s for s in samples if s["t"] >= t]
    return min(after, key=lambda s: s["t"]) if after and after[0]["t"] - t <= 2.0 else None


def attribute(answer: dict[str, Any], limit_ns: int, host_median_ns: float, sample: dict[str, Any] | None,
              cell: dict[str, Any]) -> dict[str, Any]:
    """A late honest answer's cause, from its own split and the sampler's second:
    - `device`: the responder's own time (`server_ns`, the block's copy off the device under the run's forwards included)
      leaves no room for the run's median host part: the honest prover's GPU, not a neighbour;
    - `load`: the host part (verifier, loopback, scheduling) made it late, and a sibling of the verifier's or responder's core
      or the GPU's NUMA node was at least `BUSY`% busy that second;
    - `host, unattributed`: the host part made it late with no neighbour seen, or no sample covers it."""
    host_ns = answer["elapsed_ns"] - answer["server_ns"]
    ours = set(cell["verifier_core"][:1] + cell["responder_core"][:1])
    siblings = [t for t in cell["verifier_core"] + cell["responder_core"] if t not in ours]
    sib_busy = numa_busy = None
    if sample is not None:
        busy = sample["busy"]
        sib_busy = max((busy[t] for t in siblings if t < len(busy) and busy[t] is not None), default=None)
        numa = [busy[t] for t in cell["numa_cpus"] if t < len(busy) and busy[t] is not None]
        numa_busy = round(statistics.fmean(numa), 1) if numa else None
    if answer["server_ns"] + host_median_ns > limit_ns:
        cause = "device"
    elif sample is not None and ((sib_busy or 0) >= BUSY or (numa_busy or 0) >= BUSY):
        cause = "load"
    else:
        cause = "host, unattributed"
    return {"cause": cause, "elapsed_ns": answer["elapsed_ns"], "server_ns": answer["server_ns"], "device_ns": answer["device_ns"],
            "host_ns": host_ns, "over_limit_ns": answer["elapsed_ns"] - limit_ns, "sibling_busy_pct": sib_busy,
            "numa_busy_pct": numa_busy, "loadavg": sample["loadavg"] if sample else None}


def read_cell(run_dir: Path) -> dict[str, Any] | None:
    out = run_dir / "out"
    cell_path, bench_path = out / "cell.json", out / "bench" / "bench-result.json"
    if not cell_path.exists():
        return None
    cell = json.loads(cell_path.read_text())
    row: dict[str, Any] = {"run": run_dir.name, **{k: cell[k] for k in ("seed", "rep", "weight_set", "tokens")},
                           "gpu": cell["gpu"].get("index"), "complete": False}
    if not bench_path.exists():
        return row
    doc = json.loads(bench_path.read_text())
    m = doc.get("metrics") or {}
    honest = m.get("audit") or {}
    if not honest.get("records"):
        row["gates"] = {g: v.get("passed") for g, v in (doc.get("gates") or {}).items()}
        return row
    t = str(cell["tokens"])
    loop = m["loop"]
    limit, k = honest["limit_ns"], honest["k"]
    samples = []
    if (out / "neighbors.jsonl").exists():
        samples = [json.loads(line) for line in (out / "neighbors.jsonl").read_text().splitlines() if line.strip()]
    answers = [x for rec in honest["records"] for x in rec["answers"]]
    host_median = statistics.median(x["elapsed_ns"] - x["server_ns"] for x in answers)
    late = []
    for rec in honest["records"]:
        for x in rec["answers"]:
            if not x["on_time"]:
                late.append({"audit": rec["counter"], **attribute(x, limit, host_median,
                                                                  _neighbour_at(samples, rec.get("started_unix", 0.0)), cell)})
    cap_limit = DELTA_NS + RTT_CAP_NS
    full = [rec for rec in honest["records"] if len(rec["answers"]) == k]
    at_cap_lower = sum(all(x["correct"] and x["elapsed_ns"] <= cap_limit for x in rec["answers"]) for rec in full)
    at_cap_upper = sum(all(x["correct"] and x["elapsed_ns"] <= cap_limit for x in rec["answers"]) for rec in honest["records"])
    gates = {g: v.get("passed") for g, v in doc["gates"].items()}
    row.update(complete=True, overhead=loop["overhead"][t], plain_s=loop["plain_s"][t], pous_s=loop["pous_s"][t],
               decode_s=loop["decode_s"], graphed=loop["pous_graphed"][t].get("graphed"), launch_bound=loop["launch_bound"][t],
               limit_ns=limit, rtt_allowance_ns=honest["rtt_allowance_ns"],
               allowance_within_cap=honest["rtt_allowance_ns"] <= RTT_CAP_NS, audits=honest["audits"],
               accepted=honest["audits_accepted"], answers=honest["answers"], late=honest["late"], wrong=honest["wrong"],
               forwards_completed=honest.get("forwards_completed"), elapsed_s=honest["elapsed_s"],
               accepted_at_cap_limit=[at_cap_lower, at_cap_upper], late_answers=late, gates=gates,
               controls={"value_accepted": (m.get("value_control") or {}).get("audits_accepted"),
                         "late_accepted": (m.get("late_control") or {}).get("audits_accepted"),
                         "recompute_accepted": (m.get("recompute_control") or {}).get("audits_accepted"),
                         "recompute_margin_max_s": ((m.get("recompute_control") or {}).get("device_margin_s") or {}).get("max")},
               neighbours={"loadavg_start": cell.get("loadavg_at_start"),
                           "numa_busy_mean_pct": round(statistics.fmean(
                               statistics.fmean(s["busy"][c] for c in cell["numa_cpus"] if c < len(s["busy"]) and s["busy"][c] is not None)
                               for s in samples), 1) if samples else None})
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r["complete"]]
    table = []
    for ws in WEIGHT_SETS:
        for t in TOKENS:
            cells = [r for r in done if r["weight_set"] == ws and r["tokens"] == t]
            if not cells:
                continue
            x, n = sum(r["accepted"] for r in cells), sum(r["audits"] for r in cells)
            lo, hi = clopper_pearson(x, n)
            ov = [r["overhead"] for r in cells]
            causes: dict[str, int] = {}
            for r in cells:
                for a in r["late_answers"]:
                    causes[a["cause"]] = causes.get(a["cause"], 0) + 1
            table.append({"weight_set": ws, "tokens": t, "reps": len(cells), "runs": [r["run"] for r in cells],
                          "overhead_median": statistics.median(ov), "overhead_min": min(ov), "overhead_max": max(ov),
                          "accepted": x, "audits": n, "cp95": [lo, hi], "late_answers": sum(r["late"] for r in cells),
                          "late_by_cause": causes, "wrong": sum(r["wrong"] for r in cells),
                          "rtt_allowance_us_median": statistics.median(r["rtt_allowance_ns"] for r in cells) / 1e3,
                          "allowance_within_cap": all(r["allowance_within_cap"] for r in cells),
                          "accepted_at_cap_limit": [sum(r["accepted_at_cap_limit"][0] for r in cells),
                                                    sum(r["accepted_at_cap_limit"][1] for r in cells)],
                          "gates_failed": sorted({g for r in cells for g, ok in r["gates"].items() if not ok})})
    pooled = {}
    for ws in WEIGHT_SETS + ("all",):
        cells = [r for r in done if ws in ("all", r["weight_set"])]
        x, n = sum(r["accepted"] for r in cells), sum(r["audits"] for r in cells)
        pooled[ws] = {"cells": len(cells), "accepted": x, "audits": n, "cp95": list(clopper_pearson(x, n)) if n else None}
    return {"schema": "verity-pous/p2-sweep-table/v0", "delta_ns": DELTA_NS, "rtt_cap_ns": RTT_CAP_NS, "table": table,
            "pooled": pooled, "cells": rows, "incomplete": [r["run"] for r in rows if not r["complete"]]}


def markdown(doc: dict[str, Any]) -> str:
    lines = ["| weight set | T | reps | overhead median (min-max) | honest accepted | 95% CP | late (by cause) | allowance µs |",
             "|---|---:|---:|---|---:|---|---|---:|"]
    for r in doc["table"]:
        causes = ", ".join(f"{k}: {v}" for k, v in sorted(r["late_by_cause"].items())) or "-"
        lines.append(f"| {r['weight_set']} | {r['tokens']} | {r['reps']} | {r['overhead_median']:.2f}x "
                     f"({r['overhead_min']:.2f}-{r['overhead_max']:.2f}) | {r['accepted']}/{r['audits']} | "
                     f"[{r['cp95'][0]:.4f}, {r['cp95'][1]:.4f}] | {r['late_answers']} ({causes}) | {r['rtt_allowance_us_median']:.0f} |")
    lines += ["", "| pooled | cells | honest accepted | 95% CP |", "|---|---:|---:|---|"]
    for ws, p in doc["pooled"].items():
        cp = f"[{p['cp95'][0]:.5f}, {p['cp95'][1]:.5f}]" if p["cp95"] else "-"
        lines.append(f"| {ws} | {p['cells']} | {p['accepted']}/{p['audits']} | {cp} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("seed", type=int)
    c.add_argument("--work", required=True, help="pod.sh's WORK: the venvs, CGBN and builds the cells share")
    c.add_argument("--audits", type=int, default=500)
    c.add_argument("--control-audits", type=int, default=50)
    c.add_argument("--hardware", default="rtx-pro-6000-server/bf16")
    s = sub.add_parser("sample")
    s.add_argument("path")
    s.add_argument("--watch", default="")
    g = sub.add_parser("aggregate")
    g.add_argument("runs", nargs="+")
    g.add_argument("--out", required=True)
    g.add_argument("--markdown")
    a = ap.parse_args(argv)
    if a.cmd == "cell":
        return run_cell(a)
    if a.cmd == "sample":
        sample(Path(a.path), [int(x) for x in a.watch.split(",") if x])
        return 0
    rows = [r for r in (read_cell(Path(p)) for p in a.runs) if r is not None]
    doc = aggregate(rows)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    if a.markdown:
        Path(a.markdown).write_text(markdown(doc))
    print(markdown(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
