"""kernels-maxtput's results: maxtput.py's OUT/maxtput.json as the run's result, and one result per arm as a hash-loop workload row.

On the node (job.sh), after the driver:

    python3 benchmarks/hash/maxtput/row.py OUT --bench-rc RC [--pin-rc RC] --to RESULT

writes RESULT (research/result/v0.1, every arm's measures, each with its ``arm``) and OUT/arms/<arm>.json, the same measures per arm
(no ``arm`` field), for publishing. On the VM, after the run:

    python3 benchmarks/hash/maxtput/row.py --publish RUN... --row 11 --hypothesis TEXT [--workload maxtput] [--branch B]

puts each per-arm result into the store (``--kind bench-result/v1``) and labels it campaign=hash-loop, sweep=iteration, row, arm=
<workload>:<arm> (plot.py draws it as workload <workload>, line <arm>), label, source, hardware; the run is labeled the same, with
arm=<workload>.

Measures, over the timed runs (phase ``measure``); a commit arm's ratios are against the stock engine of the same cycle (each
cycle's arms run back to back on one card), averaged over cycles, each cycle's in ``runs``:

  throughput_overhead_pct  100 (1 - out tok/s_arm / out tok/s_stock) in the steady window (queue never empty): the share lost
  tpot_overhead_pct        100 (TPOT p50_arm / TPOT p50_stock - 1)
  whole_throughput_overhead_pct, requests_overhead_pct, ttft_p50_overhead_pct   the same over the whole run (finalize included),
                           requests/s, TTFT p50
  output_tokens_per_s, requests_per_s, scheduled_tokens_per_s, whole_output_tokens_per_s, ttft_{p50,p99}_ms, tpot_{p50,p99,mean}_ms,
  gpu_util_mean_pct, gpu_util_min_pct, waiting_min, running_mean, kv_usage_mean, finalize_s   the absolutes (means over runs)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HARDWARE = {"vy-dc-h200": "H200", "vy-dc-b200": "B200", "vy-nebius-2": "RTX PRO 6000"}
BRANCH = "cursor/kernels-maxtput-15ad"
ABS = (("output_tokens_per_s", "output_tokens_per_s", 1, "tok/s"), ("requests_per_s", "requests_per_s", 1, "req/s"),
       ("scheduled_tokens_per_s", "scheduled_tokens_per_s", 1, "tok/s"),
       ("whole_output_tokens_per_s", "whole_output_tokens_per_s", 1, "tok/s"),
       ("ttft_p50_ms", "ttft_p50_s", 1e3, "ms"), ("ttft_p99_ms", "ttft_p99_s", 1e3, "ms"),
       ("tpot_p50_ms", "tpot_p50_s", 1e3, "ms"), ("tpot_p99_ms", "tpot_p99_s", 1e3, "ms"), ("tpot_mean_ms", "tpot_mean_s", 1e3, "ms"),
       ("waiting_min", "waiting_min", 1, "requests"), ("running_mean", "running_mean", 1, "requests"),
       ("kv_usage_mean", "kv_usage_mean", 1, "fraction"))
#: (name, measure, sign): sign -1 is a share lost (throughput), +1 a rise (time)
RATIOS = (("throughput_overhead_pct", "output_tokens_per_s", -1), ("tpot_overhead_pct", "tpot_p50_s", 1),
          ("whole_throughput_overhead_pct", "whole_output_tokens_per_s", -1), ("requests_overhead_pct", "requests_per_s", -1),
          ("ttft_p50_overhead_pct", "ttft_p50_s", 1))


def research(*args: str) -> str:
    return subprocess.run(["research", *args], check=True, capture_output=True, text=True).stdout


def timed(doc: dict[str, Any]) -> dict[str, dict[int, list[dict]]]:
    """arm -> cycle -> its timed runs."""
    out: dict[str, dict[int, list[dict]]] = {}
    for e in doc["engines"]:
        if e.get("error"):
            continue
        out.setdefault(e["arm"], {}).setdefault(e["cycle"], []).extend(r for r in e["runs"] if r["phase"] == "measure")
    return out


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def arm_measures(by: dict[str, dict[int, list[dict]]], arm: str) -> list[dict[str, Any]]:
    runs = [r for c in sorted(by.get(arm, {})) for r in by[arm][c]]
    if not runs:
        return []
    out = []
    for name, key, scale, unit in ABS:
        vals = [r["measures"].get(key) for r in runs]
        v = mean(vals)
        if v is not None:
            out.append({"name": name, "value": round(v * scale, 4), "unit": unit,
                        "runs": [round(x * scale, 4) if x is not None else None for x in vals]})
    for name, key in (("gpu_util_mean_pct", "util_mean_pct"), ("gpu_util_min_pct", "util_min_pct"), ("sm_clock_mean_mhz", "sm_clock_mean_mhz")):
        vals = [(r.get("gpu") or {}).get(key) for r in runs]
        v = mean(vals)
        if v is not None:
            out.append({"name": name, "value": round(v, 2), "unit": "%" if name.endswith("pct") else "MHz", "runs": vals})
    out.append({"name": "finalize_s", "value": round(mean([r["finalize_s"] for r in runs]), 4), "unit": "s",
                "runs": [round(r["finalize_s"], 4) for r in runs]})
    out.append({"name": "window_s", "value": round(mean([r["measures"]["window"]["seconds"] for r in runs]), 3), "unit": "s"})
    if arm != "stock":
        for name, key, sign in RATIOS:
            per = []
            for c, rs in sorted(by[arm].items()):
                base = mean([r["measures"].get(key) for r in by.get("stock", {}).get(c, [])])
                x = mean([r["measures"].get(key) for r in rs])
                if base and x is not None:
                    per.append(100 * (1 - x / base) if sign < 0 else 100 * (x / base - 1))
            if per:
                out.append({"name": name, "value": round(statistics.fmean(per), 3), "unit": "%", "runs": [round(p, 3) for p in per]})
    return out


def build(a: argparse.Namespace) -> int:
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR", a.out.parent))
    src = a.out / "maxtput.json"
    errors: list[str] = []
    doc: dict[str, Any] = json.loads(src.read_text()) if src.is_file() else {}
    if not doc:
        errors.append(f"no {src}")
    gates = doc.get("gates") or []
    failed = [g for g in gates if not g.get("passed")]
    if failed:
        errors.append("gates failed: " + "; ".join(f"{g['arm']}/{g['cycle']} {g['kind']}" for g in failed[:6]))
    pin: dict[str, Any] = {"state": "unpinned"}
    if os.environ.get("PIN_BUILD"):
        rc = a.pin_rc if (a.out / "pin_check.txt").is_file() else None
        pin = {"state": "pinned" if rc == 0 else "pin-failed", "build": os.environ.get("PIN_BUILD_ART", os.environ["PIN_BUILD"]),
               "check_rc": rc}
        if rc != 0:
            errors.append(f"served-pin check rc {rc}: the kernels are not the Build's")
    if a.bench_rc:
        errors.append(f"bench rc {a.bench_rc}")
    by = timed(doc) if doc else {}
    ident = next((e["identity"] for e in doc.get("engines", ()) if e.get("identity")), {})
    clocks = {w: (a.out / f"clocks.{w}.csv").read_text().strip().splitlines()[-1]
              for w in ("before", "after") if (a.out / f"clocks.{w}.csv").is_file()}
    fp = {"bench": "hash/maxtput", "model": doc.get("model"), "arms": doc.get("arms"), "options": doc.get("options"),
          "load": doc.get("load"), "clocks": clocks, "pin": pin,
          "engine": {k: ident.get(k) for k in ("vllm_version", "gpu", "model", "revision", "max_num_seqs", "max_num_batched_tokens",
                                               "max_model_len", "enable_prefix_caching", "async_scheduling", "gpu_memory_utilization",
                                               "kv_cache_tokens", "kv_bytes", "block_size")}}

    def result(meas: list[dict], arms_failed: list[dict]) -> dict[str, Any]:
        errs = errors if not arms_failed else errors + [f"{g['arm']}/{g['cycle']} {g['kind']}" for g in arms_failed]
        status = "passed" if doc and not errs and meas else "failed"
        return {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "workload_fingerprint": fp,
                "validation": {"status": status, "checker": "maxtput gates (each engine's arm gate, one KV pool, tokens equal to stock's)",
                               "detail": "; ".join(errs) or "gates passed", "evidence": []},
                "measurements": meas, "measurement_files": [str(src.relative_to(run_dir))] if src.is_file() else []}

    every = [{**m, "arm": arm} for arm in doc.get("arms", []) for m in arm_measures(by, arm)]
    whole = result(every, [])
    (a.out / "arms").mkdir(parents=True, exist_ok=True)
    for arm in doc.get("arms", []):
        mine = [g for g in failed if g["arm"] in (arm, "stock")]
        one = result(arm_measures(by, arm), mine)
        one["validation"]["detail"] += " (this arm's and stock's)"
        (a.out / "arms" / f"{arm}.json").write_text(json.dumps({**one, "arm": arm}, indent=1) + "\n")
    tmp = a.to.with_name(a.to.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(whole, indent=1) + "\n")
    os.replace(tmp, a.to)
    for m in every:
        if m["name"] in ("output_tokens_per_s", "requests_per_s", "tpot_p50_ms", "ttft_p50_ms", "gpu_util_mean_pct", "waiting_min")\
                or m["name"].endswith("overhead_pct"):
            print(f"{m['arm']:>14} {m['name']:<30} {m['value']} {m.get('runs', '')}")
    print(f"validation {whole['validation']['status']}: {whole['validation']['detail']}")
    return 0 if whole["validation"]["status"] == "passed" else 1


def publish(a: argparse.Namespace) -> int:
    rc = 0
    for run in a.publish:
        subprocess.run(["research", "fetch", run], capture_output=True, text=True)
        att = json.loads(research("data", "show", run, "--json"))
        machine = (att.get("execution") or {}).get("machine")
        commit = (att.get("source") or {}).get("commit")
        files = (att.get("outputs") or {}).get("run_files")
        if machine not in HARDWARE or not commit or not files:
            print(f"{run}: machine {machine!r}, commit {commit!r}, run_files {files!r}: nothing to publish", file=sys.stderr)
            rc = 1
            continue
        common = [("campaign", "hash-loop"), ("sweep", a.sweep), ("row", str(a.row)), ("label", a.hypothesis),
                  ("source", f"{a.branch}@{commit}"), ("hardware", HARDWARE[machine])]
        with tempfile.TemporaryDirectory() as d:
            research("data", "fetch", files, "--path", "out/arms/*.json", "--to", d)
            docs = sorted(Path(d).glob("out/arms/*.json"))
            if not docs:
                print(f"{run}: no out/arms/*.json in {files}", file=sys.stderr)
                rc = 1
            for p in docs:
                one = json.loads(p.read_text())
                put = json.loads(research("data", "put", "--kind", "bench-result/v1", "--meta", f"@{p}", "--file", str(p),
                                          "--preserve", "--json"))
                art = put.get("id") or put.get("art") or put.get("artifact")
                for k, v in [*common, ("arm", f"{a.workload}:{one['arm']}")]:
                    research("data", "label", art, k, v, "--by", a.by, "--ref", run)
                head = {m["name"]: m["value"] for m in one["measurements"]}
                print(f"{run} {HARDWARE[machine]:<13} {a.workload}:{one['arm']:<14} {art} {one['validation']['status']} "
                      f"out tok/s {head.get('output_tokens_per_s')} tput ovh {head.get('throughput_overhead_pct')} % "
                      f"TPOT ovh {head.get('tpot_overhead_pct')} %")
        for k, v in [*common, ("arm", a.workload)]:
            research("data", "label", run, k, v, "--by", a.by, "--ref", run)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--bench-rc", type=int, default=0)
    ap.add_argument("--pin-rc", type=int, default=None)
    ap.add_argument("--to", type=Path, default=None)
    ap.add_argument("--publish", nargs="+", metavar="RUN")
    ap.add_argument("--row", default="11")
    ap.add_argument("--hypothesis", default="max throughput: native compiled vLLM at saturation against the Commit's arms")
    ap.add_argument("--workload", default="maxtput", help="the `arm` label's workload: maxtput (Qwen3-8B), maxtput-llama31-8b")
    ap.add_argument("--sweep", default="iteration")
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--by", default="hash")
    a = ap.parse_args()
    if a.publish:
        return publish(a)
    if a.out is None:
        ap.error("OUT, or --publish RUN...")
    a.to = a.to or Path(os.environ.get("RESEARCH_RUN_DIR", a.out.parent)) / "result.json"
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
