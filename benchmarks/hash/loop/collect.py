"""One hash-loop run's measures, per arm, from served_commit's outputs: `$OUT/bench/result.json` (verity-bench's interleaved
reps, ratios to stock with their 95% intervals) and `$OUT/summary.<arm>.json` (nsys_summary.py --each: per run, the mean prefill
and decode step). Writes `$RESEARCH_RUN_DIR/result.json` (research/result/v0.1); every measurement names its `arm`, so the run's
one result artifact holds every arm, and plot.py reads them back from the evidence store.

    python3 benchmarks/hash/loop/collect.py OUT [--reps 3] [--row N] [--hypothesis TEXT] [--bench-rc RC] [--result PATH]
                                                [--arm-names bench-arm=arm,...]

An arm's name is ARM's for the bench arm, or --arm-names' ($ARM_NAMES; a new arm such as commit+pouw needs only that), else the
bench arm's own id. Measures (stock's step times are stock-prof's, the same engine with the NVTX ranges only):
  tpot_overhead_pct        100 (TPOT_arm / TPOT_stock - 1), the harness's interleaved ratio (headline)
  throughput_overhead_pct  100 (1 - tokens/s_arm / tokens/s_stock), the share of stock's throughput lost (headline)
  ttft_overhead_pct        100 (TTFT_arm / TTFT_stock - 1)
  tpot_ms, tokens_per_s    the absolute means
  {prefill,decode}_step_ms     mean engine-step span (nsys exec range to the next), over the timed runs
  {prefill,decode}_serve_ms    GPU time of the model's own kernels per step
  {prefill,decode}_hash_gpu_ms GPU time of the hash kernels (leaves + tree) launched per step
  {prefill,decode}_serve_rise_pct  100 (serve_arm / serve_stock - 1): the model's kernels slowed by sharing the GPU
  noise_tpot_cv_pct        (stock only) the coefficient of variation of stock's TPOT over its interleaved reps
  first_run_ms             wall time of the arm's first profiled run (the gate run, the same steps): no plan is learned yet, so every
                           step is a learning step, as every step is under live serving, where a batch's shape never repeats; the
                           timed runs replay its requests and hit its plans
  first_run_rise_pct       100 (first_run_arm / first_run_stock - 1)
  first_run_copy_gb        GB copied D2H plus H2D in that run (memcpy.DtoH.GB + memcpy.HtoD.GB)
  serve_run_ms, hash_run_ms, hash_corun_ms  per timed run: GPU time of the model's kernels, of the hash kernels, and the time both ran
  serve_lost_per_corun_ms  (serve_run_arm - serve_run_stock) / hash_corun_arm: model-kernel GPU time lost per ms of co-running hash
                           (circuits' interference measure, note:circuits-commit-serve-interference)
The result's workload_fingerprint.pin is PIN_BUILD's served-pin check (pinned: the arms ran under the Build of record's compile
artifact and its check found the same winners; unpinned: no PIN_BUILD, as on B200 and RTX PRO 6000 today).
The step measures need a one-point sweep: nsys_summary's runs carry no sweep point, so a wider sweep reports the ratios only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ARM = {"stock": "stock", "stock-prof": "stock", "served-commit-rows": "sha512", "served-commit-rows-blake3": "blake3"}
for _kv in filter(None, os.environ.get("ARM_NAMES", "").split(",")):
    ARM[_kv.split("=", 1)[0]] = _kv.split("=", 1)[1]
RATIOS = (("tpot_overhead_pct", "time_per_output_token_s", 1), ("throughput_overhead_pct", "tokens_per_s", -1),
          ("ttft_overhead_pct", "time_to_first_token_s", 1))


def pct(ratio: float, sign: int) -> float:
    return round(100 * (ratio - 1) if sign > 0 else 100 * (1 - ratio), 3)


def bench_measures(res: dict) -> list[dict]:
    out = []
    point = next(p for p in res["points"] if p["point"].get("headline")) if any(p["point"].get("headline") for p in res["points"]) \
        else res["points"][0]
    for bench_arm, a in point["arms"].items():
        if bench_arm == "stock-prof":
            continue
        arm, m = ARM.get(bench_arm, bench_arm), a["metrics"]
        out.append({"name": "tpot_ms", "arm": arm, "value": round(1e3 * m["time_per_output_token_s"]["mean"], 5), "unit": "ms"})
        out.append({"name": "tokens_per_s", "arm": arm, "value": round(m["tokens_per_s"]["mean"], 3), "unit": "1/s"})
        if bench_arm == res.get("baseline", "stock"):
            reps = [r["metrics"]["time_per_output_token_s"] for r in a.get("reps", ())]
            cv = 100 * statistics.stdev(reps) / statistics.fmean(reps) if len(reps) > 1 else None
            out.append({"name": "noise_tpot_cv_pct", "arm": arm, "value": None if cv is None else round(cv, 3), "unit": "%", "n": len(reps)})
            continue
        for name, key, sign in RATIOS:
            r = m[key]["ratio_to_stock"]
            out.append({"name": name, "arm": arm, "value": pct(r["mean"], sign), "unit": "%", "n": r.get("n", 1)})
            if r.get("ci95"):
                out[-1]["ci95"] = sorted(pct(x, sign) for x in r["ci95"])
    return out


def step_means(summary: list[dict], reps: int) -> dict[str, float]:
    """Mean over the last `reps` runs (the timed ones; the earlier ranges are the gate and warm-up) of each step measure."""
    runs = summary[-reps:]
    out = {}
    for phase in ("prefill", "decode"):
        rows = [r.get(f"{phase}_mean") or {} for r in runs]
        if not all(rows):
            continue
        mean = lambda f: round(statistics.fmean(f(r) for r in rows), 4)  # noqa: E731
        out[f"{phase}_step_ms"] = mean(lambda r: r.get("step_ms", 0.0))
        out[f"{phase}_serve_ms"] = mean(lambda r: r.get("serve", 0.0))
        out[f"{phase}_hash_gpu_ms"] = mean(lambda r: r.get("hash.leaves", 0.0) + r.get("hash.tree", 0.0))
    for key, name in (("gpu_busy_serve_ms", "serve_run_ms"), ("gpu_busy_hash_ms", "hash_run_ms"), ("hash_overlap_serve_ms", "hash_corun_ms")):
        if all(key in r for r in runs):
            out[name] = round(statistics.fmean(r[key] for r in runs), 3)
    first = summary[0] if len(summary) > reps else {}
    if "run_wall_ms" in first:
        out["first_run_ms"] = round(first["run_wall_ms"], 3)
        cat = first.get("gpu_ms_by_category", {})
        out["first_run_copy_gb"] = round(sum(cat.get(k, (0.0,))[0] for k in ("memcpy.DtoH.GB", "memcpy.HtoD.GB")), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--row", default="")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--bench-rc", type=int, default=0)
    ap.add_argument("--result", type=Path, default=None)
    ap.add_argument("--arm-names", default="")
    ap.add_argument("--pin-rc", type=int, default=None, help="prof.sh's `pin check rc` (with PIN_BUILD)")
    a = ap.parse_args()
    ARM.update(kv.split("=", 1) for kv in filter(None, a.arm_names.split(",")))
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR", a.out.parent))
    res_path = a.out / "bench" / "result.json"
    meas: list[dict] = []
    errors: list[str] = []
    gates_ok = False
    res: dict = {}
    if res_path.is_file():
        res = json.loads(res_path.read_text())
        meas += bench_measures(res)
        gates_ok = bool(res.get("gates")) and all(g.get("passed") for g in res["gates"])
    else:
        errors.append(f"no {res_path}")
    steps = {}
    one_point = len(res.get("points", ())) == 1
    for f in sorted(a.out.glob("summary.*.json")) if one_point else ():
        bench_arm = f.name[len("summary."):-len(".json")]
        steps[ARM.get(bench_arm, bench_arm)] = step_means(json.loads(f.read_text()), a.reps)
    base = steps.get("stock", {})
    for arm, s in steps.items():
        for k, v in s.items():
            meas.append({"name": k, "arm": arm, "value": v, "unit": "GB" if k.endswith("_gb") else "ms"})
        if arm != "stock" and base.get("first_run_ms") and "first_run_ms" in s:
            rise = round(100 * (s["first_run_ms"] / base["first_run_ms"] - 1), 3)
            meas.append({"name": "first_run_rise_pct", "arm": arm, "value": rise, "unit": "%"})
        if arm != "stock" and base.get("serve_run_ms") and s.get("hash_corun_ms"):
            lost = (s["serve_run_ms"] - base["serve_run_ms"]) / s["hash_corun_ms"]
            meas.append({"name": "serve_lost_per_corun_ms", "arm": arm, "value": round(lost, 4), "unit": "ms/ms"})
        if arm != "stock":
            for phase in ("prefill", "decode"):
                b, x = base.get(f"{phase}_serve_ms"), s.get(f"{phase}_serve_ms")
                if b and x is not None:
                    meas.append({"name": f"{phase}_serve_rise_pct", "arm": arm, "value": round(100 * (x / b - 1), 3), "unit": "%"})
    clocks = {}
    for when in ("before", "after"):
        p = a.out / f"clocks.{when}.csv"
        if p.is_file():
            clocks[when] = p.read_text().strip().splitlines()[-1]
    pin_txt = a.out / "pin_check.txt"
    pin = {"state": "unpinned"}
    if os.environ.get("PIN_BUILD"):
        rc = a.pin_rc if pin_txt.is_file() else None
        pin = {"state": "pinned" if rc == 0 else "pin-failed", "build": os.environ.get("PIN_BUILD_ART", os.environ["PIN_BUILD"]),
               "check_rc": rc}
        if rc != 0:
            errors.append(f"served-pin check rc {rc}: the kernels are not the Build's")
    ident = res.get("identity", {})
    status = "passed" if gates_ok and a.bench_rc == 0 and not errors else "failed"
    doc = {
        "schema": "research/result/v0.1",
        "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
        "workload_fingerprint": {"bench": "served_commit/prof.sh", "spec": ident.get("preset", {}).get("id"), "sweep": ident.get("sweep"),
                                 "git_commit": ident.get("git_commit"), "vllm_version": ident.get("vllm_version"),
                                 "arms": [x["id"] for x in ident.get("arms", [])], "reps": a.reps, "row": a.row,
                                 "hypothesis": a.hypothesis, "clocks": clocks, "pin": pin},
        "validation": {"status": status, "checker": "verity-bench gates (token-for-token vs stock, commit-arm gate)",
                       "detail": "; ".join(errors) or f"bench rc {a.bench_rc}, gates {'passed' if gates_ok else 'failed'}", "evidence": []},
        "measurements": meas,
        "measurement_files": [str(p.relative_to(run_dir)) for p in [res_path, *sorted(a.out.glob("summary.*.json"))] if p.is_file()],
    }
    dst = a.result or run_dir / "result.json"
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, dst)
    for m in meas:
        print(f"{m['arm']:>7} {m['name']:<26} {m['value']}{' ' + str(m['ci95']) if 'ci95' in m else ''}")
    print(f"validation {status}: {doc['validation']['detail']}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
