"""kernels-maxtput's results: maxtput.py's OUT/maxtput.json as the run's result, and one result per arm as a hash-loop workload row.

On the node (job.sh), after the driver:

    python3 benchmarks/hash/maxtput/row.py OUT --bench-rc RC [--pin-rc RC] --to RESULT

writes RESULT (research/result/v0.1, every arm's measures, each with its ``arm``) and OUT/arms/<arm>.json, the same measures per arm
(no ``arm`` field). On the VM, after the run:

    python3 benchmarks/hash/maxtput/row.py --publish RUN... --row 11 [--native RUN] [--hypothesis TEXT] [--workload maxtput] [--branch B]

waits (up to 30 min) for the run's custody on the store's remote, then rebuilds each arm's result from it (out/maxtput.json, and its
result's pin state and bench rc), adds its loss against the native reference run (--native, default the run's own stock), puts it
into the store (``--kind bench-result/v1``, the result as its meta, as a run's own result) and labels it campaign=hash-loop,
sweep=iteration, row, arm=<workload>:<arm> (plot.py draws it as workload <workload>, line <arm>), label, source, hardware; the run is
labeled the same, with arm=<workload>; workload and label default to the run's own labels. An earlier result of the same run and
arm in this store's catalog is labeled superseded_by the new one (so refresh the catalog first on a fresh VM:
`research data refresh`). ``row.py --table [--row 11]`` prints every published maxtput* arm of the row from the store, one line
per run and arm, superseded ones left out.

Measures, over the timed runs (phase ``measure``); a commit arm's ratios are against the stock engine of the same cycle (each
cycle's arms run back to back on one card), averaged over cycles, each cycle's in ``runs``:

  throughput_overhead_pct  100 (1 - out tok/s_arm / out tok/s_stock) in the steady window (queue never empty): the share lost
  tpot_overhead_pct        100 (TPOT p50_arm / TPOT p50_stock - 1)
  native_loss_pct          100 (1 - out tok/s_arm / out tok/s of the native reference's stock), both steady-window means over
                           cycles, with native_output_tokens_per_s (each names the reference run in ``ref``; the result's
                           workload_fingerprint.native_ref has its card, model, pin, KV pool and attention). Published results
                           only, on every arm but the reference's own stock. The native reference is stock at vLLM's own KV
                           sizing and attention backend: the run's own stock when it is that, else a stock-only run on the
                           same card and model (the RTX PRO 6000's at 0.9 of its memory, beside rows whose pool the Commit
                           shrinks; FlashInfer's beside a B200 row pinned to FLASH_ATTN)
  whole_throughput_overhead_pct, requests_overhead_pct, ttft_p50_overhead_pct   the same over the whole run (finalize included),
                           requests/s, TTFT p50
  output_tokens_per_s, requests_per_s, scheduled_tokens_per_s, whole_output_tokens_per_s, ttft_{p50,p99}_ms, tpot_{p50,p99,mean}_ms,
  gpu_util_mean_pct, gpu_util_min_pct, waiting_min, running_mean, kv_usage_mean, finalize_s   the absolutes (means over runs)
  vllm_bench_output_tokens_per_s, vllm_bench_requests_per_s, driver_vs_vllm_bench_pct   with XCHECK (job.sh): `vllm bench
                           throughput`'s whole-run figures (its two runs, before and after the driver) and 100 (the driver's
                           stock whole_output_tokens_per_s / vLLM's - 1), in the run's result only

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
HYPOTHESIS = "max throughput: native compiled vLLM at saturation against the Commit's arms"
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


def vllm_bench(log: Path) -> dict[str, float] | None:
    """`vllm bench throughput`'s summary line (over its whole `llm.generate`): requests/s, total tokens/s, output tokens/s."""
    got = re.search(r"^Throughput: ([\d.]+) requests/s, ([\d.]+) total tokens/s, ([\d.]+) output tokens/s",
                    log.read_text(errors="replace"), re.M) if log.is_file() else None
    return dict(zip(("requests_per_s", "total_tokens_per_s", "output_tokens_per_s"), map(float, got.groups()))) if got else None


def xcheck(whole: dict[str, Any], out: Path) -> None:
    """job.sh's XCHECK: vLLM's own entry point beside the driver's stock engine, both over the whole run on one fixed load."""
    bench = [b for w in ("before", "after") if (b := vllm_bench(out / f"vllm-bench-{w}.log"))]
    if len(bench) < 2:
        whole["validation"]["status"] = "failed"
        whole["validation"]["detail"] += f"; vllm bench throughput reported {len(bench)} of 2 runs (out/vllm-bench-*.log)"
    if not bench:
        return
    ref = mean([b["output_tokens_per_s"] for b in bench])
    ms = whole["measurements"]
    ms.append({"name": "vllm_bench_output_tokens_per_s", "value": round(ref, 4), "unit": "tok/s", "arm": "vllm-bench",
               "runs": [b["output_tokens_per_s"] for b in bench]})
    ms.append({"name": "vllm_bench_requests_per_s", "value": round(mean([b["requests_per_s"] for b in bench]), 4), "unit": "req/s",
               "arm": "vllm-bench", "runs": [b["requests_per_s"] for b in bench]})
    driver = next((m["value"] for m in ms if m["arm"] == "stock" and m["name"] == "whole_output_tokens_per_s"), None)
    if driver:
        ms.append({"name": "driver_vs_vllm_bench_pct", "value": round(100 * (driver / ref - 1), 3), "unit": "%", "arm": "stock"})


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
    if os.environ.get("XCHECK"):
        xcheck(whole, a.out)
    every = whole["measurements"]
    (a.out / "arms").mkdir(parents=True, exist_ok=True)
    for arm, one in per.items():
        (a.out / "arms" / f"{arm}.json").write_text(json.dumps(one, indent=1) + "\n")
    tmp = a.to.with_name(a.to.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(whole, indent=1) + "\n")
    os.replace(tmp, a.to)
    for m in every:
        if m["name"] in ("output_tokens_per_s", "requests_per_s", "tpot_p50_ms", "ttft_p50_ms", "gpu_util_mean_pct", "waiting_min",
                         "whole_output_tokens_per_s") or m["name"].endswith("overhead_pct") or "vllm_bench" in m["name"]:
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


def stored(run: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """A run's attempt, its result's meta and its out/maxtput.json, from the store."""
    att = json.loads(research("data", "show", run, "--json"))
    outs = att.get("outputs") or {}
    files, res = outs.get("run_files"), outs.get("result")
    if not files or not res:
        raise LookupError(f"{run}: run_files {files!r}, result {res!r}")
    man = json.loads(research("data", "show", res, "--json"))
    with tempfile.TemporaryDirectory() as d:
        research("data", "fetch", files, "--path", "out/maxtput.json", "--to", d)
        doc = json.loads((Path(d) / "out" / "maxtput.json").read_text())
    return att, (man.get("manifest") or man).get("meta") or {}, doc


def reference(run: str, att: dict[str, Any], whole: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    """A run's stock engine as a native reference: its steady-window out tok/s (mean over cycles), card, model, pin, KV and attention."""
    wf = whole.get("workload_fingerprint") or {}
    eng = wf.get("engine") or {}
    head = {m["name"]: m["value"] for m in arm_measures(timed(doc), "stock")}
    return {"run": run, "machine": (att.get("execution") or {}).get("machine"), "model": doc.get("model"),
            "output_tokens_per_s": head.get("output_tokens_per_s"), "pin": (wf.get("pin") or {}).get("state", "unpinned"),
            "kv_bytes": eng.get("kv_bytes"), "attention_impls": eng.get("attention_impls")}


def latest(target: str) -> dict[str, str]:
    got = sorted(json.loads(research("data", "labels", target, "--json")), key=lambda e: e.get("ts", ""))
    return {e["key"]: str(e["value"]) for e in got}


def supersede(run: str, arm_label: str, art: str, by: str) -> list[str]:
    """Label every earlier result of ``run``'s ``arm_label`` in this store's catalog superseded_by ``art``."""
    old = []
    for x in research("data", "select", "--kind", "bench-result/v1", "--label", "campaign=hash-loop",
                      "--label", f"arm={arm_label}", "--where", f"meta.run_id={run}", "--ids").split():
        if x != art and not latest(x).get("superseded_by"):
            research("data", "label", x, "superseded_by", art, "--by", by, "--ref", run)
            old.append(x)
    return old


def publish(a: argparse.Namespace) -> int:
    rc = 0
    ref = None
    if a.native:
        ref = reference(a.native, *stored(a.native))
    for run in a.publish:
        if not preserved(run):
            print(f"{run}: no custody on the store's remote yet (`research data custody {run}`): nothing published", file=sys.stderr)
            rc = 1
            continue
        try:
            att, whole, doc = stored(run)
        except LookupError as e:
            print(f"{e}: nothing to publish", file=sys.stderr)
            rc = 1
            continue
        machine = (att.get("execution") or {}).get("machine")
        commit = (att.get("source") or {}).get("commit")
        if machine not in HARDWARE or not commit:
            print(f"{run}: machine {machine!r}, commit {commit!r}: nothing to publish", file=sys.stderr)
            rc = 1
            continue
        native = ref or reference(run, att, whole, doc)
        if (native["machine"], native["model"]) != (machine, doc.get("model")) or not native["output_tokens_per_s"]:
            print(f"{run}: native reference {native['run']} is {native['model']} on {native['machine']} with stock "
                  f"{native['output_tokens_per_s']} out tok/s; this run is {doc.get('model')} on {machine}: nothing published",
                  file=sys.stderr)
            rc = 1
            continue
        had = latest(run)
        workload = a.workload or had.get("arm") or "maxtput"
        common = [("campaign", "hash-loop"), ("sweep", a.sweep), ("row", str(a.row)),
                  ("label", a.hypothesis or had.get("label") or HYPOTHESIS),
                  ("source", f"{a.branch}@{commit}"), ("hardware", HARDWARE[machine])]
        wf = whole.get("workload_fingerprint") or {}
        bench_rc = wf.get("bench_rc")
        if bench_rc is None:
            got = re.search(r"bench rc (\d+)", (whole.get("validation") or {}).get("detail", ""))
            bench_rc = int(got.group(1)) if got else 0
        _, per = results(doc, wf.get("pin") or {"state": "unpinned"}, bench_rc, run, wf.get("clocks") or {},
                         whole.get("measurement_files") or [])
        with tempfile.TemporaryDirectory() as d:
            for arm, one in per.items():
                head = {m["name"]: m["value"] for m in one["measurements"]}
                out = head.get("output_tokens_per_s")
                if out and not (native["run"] == run and arm == "stock"):
                    one["measurements"] += [
                        {"name": "native_output_tokens_per_s", "value": native["output_tokens_per_s"], "unit": "tok/s",
                         "ref": native["run"]},
                        {"name": "native_loss_pct", "value": round(100 * (1 - out / native["output_tokens_per_s"]), 3),
                         "unit": "%", "ref": native["run"]}]
                    head["native_loss_pct"] = one["measurements"][-1]["value"]
                one["workload_fingerprint"] = {**one["workload_fingerprint"], "native_ref": native}
                p = Path(d) / f"{arm}.json"
                p.write_text(json.dumps(one) + "\n")
                put = json.loads(research("data", "put", "--kind", "bench-result/v1", "--meta", f"@{p}", "--preserve", "--json"))
                art = put["id"]
                for k, v in [*common, ("arm", f"{workload}:{one['arm']}")]:
                    research("data", "label", art, k, v, "--by", a.by, "--ref", run)
                old = supersede(run, f"{workload}:{one['arm']}", art, a.by)
                print(f"{run} {HARDWARE[machine]:<13} {workload}:{one['arm']:<14} {art} {one['validation']['status']} "
                      f"out tok/s {out} tput ovh {head.get('throughput_overhead_pct')} % "
                      f"lost vs native {head.get('native_loss_pct')} % ({native['run']}) TPOT ovh {head.get('tpot_overhead_pct')} %"
                      + (f"; supersedes {', '.join(old)}" if old else ""))
        for k, v in [*common, ("arm", workload)]:
            research("data", "label", run, k, v, "--by", a.by, "--ref", run)
    return rc


ARM_ORDER = ("stock", "sha512", "sha512-layer", "blake3")
COLUMNS = (("out tok/s", "output_tokens_per_s", "{:,.0f}"), ("lost % (same cycle)", "throughput_overhead_pct", "{:.1f}"),
           ("lost % vs native", "native_loss_pct", "{:.1f}"), ("req/s", "requests_per_s", "{:.1f}"), ("queue wait + TTFT p50 s", "ttft_p50_ms", "{:.1f}", 1e-3),
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
        if not workload.startswith("maxtput") or labels.get("kernel_outcome") == "voided" or labels.get("superseded_by"):
            continue
        meta = x["manifest"]["meta"]
        fp = meta.get("workload_fingerprint") or {}
        eng = fp.get("engine") or {}
        head = {m["name"]: m["value"] for m in meta["measurements"]}
        nat = fp.get("native_ref") or {}
        cells = [labels.get("hardware", "?"), eng.get("model", fp.get("model", "?")), f"{workload}:{arm}", meta["run_id"],
                 (fp.get("pin") or {}).get("state", "?"), ",".join(eng.get("attention_impls") or []),
                 f"{eng.get('kv_bytes', 0) / 1e9:.1f}", meta["validation"]["status"],
                 "" if not nat else "itself" if nat.get("run") == meta["run_id"] else f"{nat.get('run')} ({nat.get('pin')}, "
                 f"{(nat.get('kv_bytes') or 0) / 1e9:.1f} GB, {','.join(nat.get('attention_impls') or [])})"]
        for _, key, form, *scale in COLUMNS:
            v = head.get(key)
            cells.append("" if v is None else form.format(v * (scale[0] if scale else 1)))
        lines.append((labels.get("hardware", ""), workload, ARM_ORDER.index(arm) if arm in ARM_ORDER else 9, meta["run_id"], cells))
    heads = ["GPU", "model", "workload:arm", "run", "pin", "attention", "KV GB", "validation", "native reference",
             *(c[0] for c in COLUMNS)]
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
    ap.add_argument("--hypothesis", default=None, help=f"the `label` (default: the run's own, else {HYPOTHESIS!r})")
    ap.add_argument("--workload", default=None,
                    help="the `arm` label's workload: maxtput (Qwen3-8B), maxtput-llama31-8b (default: the run's own, else maxtput)")
    ap.add_argument("--native", metavar="RUN", default=None,
                    help="the native reference run (stock at vLLM's own KV sizing and attention backend, on the same card and "
                         "model; pinned when the run is, where a Build of that backend exists): default each run's own stock")
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
