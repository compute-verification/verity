"""kernels-maxtput's results: maxtput.py's OUT/maxtput.json as the run's result, and one result per arm as a hash-loop workload row.

On the node (job.sh), after the driver:

    python3 benchmarks/hash/maxtput/row.py OUT --bench-rc RC [--pin-rc RC] --to RESULT

writes RESULT (research/result/v0.1, every arm's measures, each with its ``arm``) and OUT/arms/<arm>.json, the same measures per arm
(no ``arm`` field). On the VM, after the run:

    python3 benchmarks/hash/maxtput/row.py --publish RUN... --row 11 --hypothesis TEXT [--workload maxtput] [--branch B]

waits (up to 30 min) for the run's custody on the store's remote, then rebuilds each arm's result from it (out/maxtput.json, and its
result's pin state and bench rc), puts it into the
store (``--kind bench-result/v1``, the result as its meta, as a run's own result) and labels it campaign=hash-loop, sweep=iteration, row, arm=
<workload>:<arm> (plot.py draws it as workload <workload>, line <arm>), label, source, hardware; the run is labeled the same, with
arm=<workload>. ``row.py --table [--row 11]`` prints every published maxtput* arm of the row from the store, one line per run and arm.

Measures, over the timed runs (phase ``measure``); a commit arm's ratios are against the stock engine of the same cycle (each
cycle's arms run back to back on one card), averaged over cycles, each cycle's in ``runs``:

  throughput_overhead_pct  100 (1 - out tok/s_arm / out tok/s_stock) in the steady window (queue never empty): the share lost
  tpot_overhead_pct        100 (TPOT p50_arm / TPOT p50_stock - 1)
  whole_throughput_overhead_pct, requests_overhead_pct, ttft_p50_overhead_pct   the same over the whole run (finalize included),
                           requests/s, TTFT p50
  output_tokens_per_s, requests_per_s, scheduled_tokens_per_s, whole_output_tokens_per_s, ttft_{p50,p99}_ms, tpot_{p50,p99,mean}_ms,
  gpu_util_mean_pct, gpu_util_min_pct, waiting_min, running_mean, kv_usage_mean, finalize_s   the absolutes (means over runs)

Every request arrives at t=0, so TTFT counts from there: under the burst it is the queue wait plus the first token, and --table
heads it so.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
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


def results(doc: dict[str, Any], pin: dict[str, Any], bench_rc: int, run_id: str, clocks: dict[str, str],
            files: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The run's result (every arm's measures, each with its ``arm``) and one result per arm (no ``arm`` field), from maxtput.json."""
    errors: list[str] = [] if doc else ["no maxtput.json"]
    failed = [g for g in doc.get("gates") or [] if not g.get("passed")]
    if failed:
        errors.append("gates failed: " + "; ".join(f"{g['arm']}/{g['cycle']} {g['kind']}" for g in failed[:6]))
    if pin.get("state") == "pin-failed":
        errors.append(f"served-pin check rc {pin.get('check_rc')}: the kernels are not the Build's")
    if bench_rc:
        errors.append(f"bench rc {bench_rc}")
    by = timed(doc) if doc else {}
    ident = next((e["identity"] for e in doc.get("engines", ()) if e.get("identity")), {})
    fp = {"bench": "hash/maxtput", "model": doc.get("model"), "arms": doc.get("arms"), "options": doc.get("options"),
          "load": doc.get("load"), "clocks": clocks, "pin": pin, "bench_rc": bench_rc,
          "engine": {k: ident.get(k) for k in ("vllm_version", "gpu", "model", "revision", "max_num_seqs", "max_num_batched_tokens",
                                               "max_model_len", "enable_prefix_caching", "async_scheduling", "gpu_memory_utilization",
                                               "kv_cache_tokens", "kv_bytes", "block_size", "attention_backend", "attention_impls")}}

    def result(meas: list[dict], arms_failed: list[dict]) -> dict[str, Any]:
        errs = errors if not arms_failed else errors + [f"{g['arm']}/{g['cycle']} {g['kind']}" for g in arms_failed]
        status = "passed" if doc and not errs and meas else "failed"
        return {"schema": "research/result/v0.1", "run_id": run_id, "workload_fingerprint": fp,
                "validation": {"status": status, "checker": "maxtput gates (each engine's arm gate, one KV pool, tokens equal to stock's)",
                               "detail": "; ".join(errs) or "gates passed", "evidence": []},
                "measurements": meas, "measurement_files": files}

    whole = result([{**m, "arm": arm} for arm in doc.get("arms", []) for m in arm_measures(by, arm)], [])
    per = {}
    for arm in doc.get("arms", []):
        one = result(arm_measures(by, arm), [g for g in failed if g["arm"] in (arm, "stock")])
        one["validation"]["detail"] += " (this arm's and stock's)"
        per[arm] = {**one, "arm": arm}
    return whole, per


def build(a: argparse.Namespace) -> int:
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR", a.out.parent))
    src = a.out / "maxtput.json"
    doc: dict[str, Any] = json.loads(src.read_text()) if src.is_file() else {}
    pin: dict[str, Any] = {"state": "unpinned"}
    if os.environ.get("PIN_BUILD"):
        rc = a.pin_rc if (a.out / "pin_check.txt").is_file() else None
        pin = {"state": "pinned" if rc == 0 else "pin-failed", "build": os.environ.get("PIN_BUILD_ART", os.environ["PIN_BUILD"]),
               "check_rc": rc}
    clocks = {w: (a.out / f"clocks.{w}.csv").read_text().strip().splitlines()[-1]
              for w in ("before", "after") if (a.out / f"clocks.{w}.csv").is_file()}
    whole, per = results(doc, pin, a.bench_rc, os.environ.get("RESEARCH_RUN_ID", "local"), clocks,
                         [str(src.relative_to(run_dir))] if src.is_file() else [])
    every = whole["measurements"]
    (a.out / "arms").mkdir(parents=True, exist_ok=True)
    for arm, one in per.items():
        (a.out / "arms" / f"{arm}.json").write_text(json.dumps(one, indent=1) + "\n")
    tmp = a.to.with_name(a.to.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(whole, indent=1) + "\n")
    os.replace(tmp, a.to)
    for m in every:
        if m["name"] in ("output_tokens_per_s", "requests_per_s", "tpot_p50_ms", "ttft_p50_ms", "gpu_util_mean_pct", "waiting_min")\
                or m["name"].endswith("overhead_pct"):
            print(f"{m['arm']:>14} {m['name']:<30} {m['value']} {m.get('runs', '')}")
    print(f"validation {whole['validation']['status']}: {whole['validation']['detail']}")
    return 0 if whole["validation"]["status"] == "passed" else 1


def preserved(run: str, tries: int = 30) -> bool:
    """Custody reaches the store's remote a few minutes after `research run --wait` returns; until then the attempt isn't there.
    `research data custody` reads the local run dir, which only `research fetch` makes on a VM that didn't launch the run."""
    for i in range(tries):
        subprocess.run(["research", "fetch", run], capture_output=True, text=True)
        if subprocess.run(["research", "data", "custody", run], capture_output=True, text=True).returncode == 0:
            return True
        if i + 1 < tries:
            time.sleep(60)
    return False


def publish(a: argparse.Namespace) -> int:
    rc = 0
    for run in a.publish:
        if not preserved(run):
            print(f"{run}: no custody on the store's remote yet (`research data custody {run}`): nothing published", file=sys.stderr)
            rc = 1
            continue
        att = json.loads(research("data", "show", run, "--json"))
        machine = (att.get("execution") or {}).get("machine")
        commit = (att.get("source") or {}).get("commit")
        outs = att.get("outputs") or {}
        files, res = outs.get("run_files"), outs.get("result")
        if machine not in HARDWARE or not commit or not files or not res:
            print(f"{run}: machine {machine!r}, commit {commit!r}, run_files {files!r}, result {res!r}: nothing to publish", file=sys.stderr)
            rc = 1
            continue
        common = [("campaign", "hash-loop"), ("sweep", a.sweep), ("row", str(a.row)), ("label", a.hypothesis),
                  ("source", f"{a.branch}@{commit}"), ("hardware", HARDWARE[machine])]
        man = json.loads(research("data", "show", res, "--json"))
        whole = (man.get("manifest") or man).get("meta") or {}
        wf = whole.get("workload_fingerprint") or {}
        bench_rc = wf.get("bench_rc")
        if bench_rc is None:
            got = re.search(r"bench rc (\d+)", (whole.get("validation") or {}).get("detail", ""))
            bench_rc = int(got.group(1)) if got else 0
        with tempfile.TemporaryDirectory() as d:
            research("data", "fetch", files, "--path", "out/maxtput.json", "--to", d)
            doc = json.loads((Path(d) / "out" / "maxtput.json").read_text())
            _, per = results(doc, wf.get("pin") or {"state": "unpinned"}, bench_rc, run, wf.get("clocks") or {},
                             whole.get("measurement_files") or [])
            for arm, one in per.items():
                p = Path(d) / f"{arm}.json"
                p.write_text(json.dumps(one) + "\n")
                put = json.loads(research("data", "put", "--kind", "bench-result/v1", "--meta", f"@{p}", "--preserve", "--json"))
                art = put["id"]
                for k, v in [*common, ("arm", f"{a.workload}:{one['arm']}")]:
                    research("data", "label", art, k, v, "--by", a.by, "--ref", run)
                head = {m["name"]: m["value"] for m in one["measurements"]}
                print(f"{run} {HARDWARE[machine]:<13} {a.workload}:{one['arm']:<14} {art} {one['validation']['status']} "
                      f"out tok/s {head.get('output_tokens_per_s')} tput ovh {head.get('throughput_overhead_pct')} % "
                      f"TPOT ovh {head.get('tpot_overhead_pct')} %")
        for k, v in [*common, ("arm", a.workload)]:
            research("data", "label", run, k, v, "--by", a.by, "--ref", run)
    return rc


ARM_ORDER = ("stock", "sha512", "sha512-layer", "blake3")
COLUMNS = (("out tok/s", "output_tokens_per_s", "{:,.0f}"), ("lost %", "throughput_overhead_pct", "{:.1f}"),
           ("req/s", "requests_per_s", "{:.1f}"), ("queue wait + TTFT p50 s", "ttft_p50_ms", "{:.1f}", 1e-3),
           ("queue wait + TTFT p99 s", "ttft_p99_ms", "{:.1f}", 1e-3), ("TPOT p50 ms", "tpot_p50_ms", "{:.1f}"),
           ("TPOT p99 ms", "tpot_p99_ms", "{:.1f}"), ("TPOT rise %", "tpot_overhead_pct", "{:.1f}"),
           ("waiting min", "waiting_min", "{:.0f}"), ("running mean", "running_mean", "{:.0f}"), ("KV use", "kv_usage_mean", "{:.3f}"),
           ("GPU util %", "gpu_util_mean_pct", "{:.1f}"))


def table(a: argparse.Namespace) -> int:
    """Every published maxtput* arm of row ROW as a markdown table, from the store (its results and their labels)."""
    lines = []
    for x in json.loads(research("data", "select", "--kind", "bench-result/v1", "--label", "campaign=hash-loop",
                                 "--label", f"row={a.row}", "--json")):
        got = sorted(json.loads(research("data", "labels", x["id"], "--json")), key=lambda e: e.get("ts", ""))
        labels = {e["key"]: e["value"] for e in got}
        workload, _, arm = labels.get("arm", "").partition(":")
        if not workload.startswith("maxtput") or labels.get("kernel_outcome") == "voided":
            continue
        meta = x["manifest"]["meta"]
        fp = meta.get("workload_fingerprint") or {}
        eng = fp.get("engine") or {}
        head = {m["name"]: m["value"] for m in meta["measurements"]}
        cells = [labels.get("hardware", "?"), eng.get("model", fp.get("model", "?")), f"{workload}:{arm}", meta["run_id"],
                 (fp.get("pin") or {}).get("state", "?"), ",".join(eng.get("attention_impls") or []),
                 f"{eng.get('kv_bytes', 0) / 1e9:.1f}", meta["validation"]["status"]]
        for _, key, form, *scale in COLUMNS:
            v = head.get(key)
            cells.append("" if v is None else form.format(v * (scale[0] if scale else 1)))
        lines.append((labels.get("hardware", ""), workload, ARM_ORDER.index(arm) if arm in ARM_ORDER else 9, meta["run_id"], cells))
    heads = ["GPU", "model", "workload:arm", "run", "pin", "attention", "KV GB", "validation", *(c[0] for c in COLUMNS)]
    print("| " + " | ".join(heads) + " |\n|" + "---|" * len(heads))
    for *_, cells in sorted(lines):
        print("| " + " | ".join(cells) + " |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--bench-rc", type=int, default=0)
    ap.add_argument("--pin-rc", type=int, default=None)
    ap.add_argument("--to", type=Path, default=None)
    ap.add_argument("--publish", nargs="+", metavar="RUN")
    ap.add_argument("--table", action="store_true", help="print row ROW's published maxtput* arms as a markdown table")
    ap.add_argument("--row", default="11")
    ap.add_argument("--hypothesis", default="max throughput: native compiled vLLM at saturation against the Commit's arms")
    ap.add_argument("--workload", default="maxtput", help="the `arm` label's workload: maxtput (Qwen3-8B), maxtput-llama31-8b")
    ap.add_argument("--sweep", default="iteration")
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--by", default="hash")
    a = ap.parse_args()
    if a.publish:
        return publish(a)
    if a.table:
        return table(a)
    if a.out is None:
        ap.error("OUT, --publish RUN... or --table")
    a.to = a.to or Path(os.environ.get("RESEARCH_RUN_DIR", a.out.parent)) / "result.json"
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
