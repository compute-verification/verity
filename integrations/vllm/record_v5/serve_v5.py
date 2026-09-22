#!/usr/bin/env python
"""[A9 contract 2026-09-10] The original execution: 3-field production (inputs, outputs, seed -- decision 1) in the matched-C serving
configuration, and the stock-vLLM comparison on the SAME requests (R19).  GPU pod; the vLLM venv's python (``--py-vllm``).

Two passes of ``verity_vllm_adapter.production`` (L1's serving module, unchanged), separately identified:

* ``adapter/production/`` -- **matched C**: ``VLLM_BATCH_INVARIANT=1``, ``--enforce-eager``, greedy, prefix caching as the module records it
  (the flag is in its manifest), ``--logprobs 2``; its recorded pass is THE original execution (class 2: execution + collection); its
  ``timings.json`` (recording on vs off, ``--reps`` each, same engine) is the collection overhead inside the matched configuration;
* ``adapter/stock-baseline/`` -- **stock vLLM**: no ``VLLM_BATCH_INVARIANT``, no ``--enforce-eager``, same requests, same module (its
  recording-off passes are the stock serving time; class 1 is native/trusted-replay -- this is the serving-side native baseline). Its
  outputs are NOT used for any commitment; the per-request token agreement with the matched pass is reported (agreement is not
  equivalence to C; what establishes equivalence is stated in the report: conformance vectors / holdout captures, R19).

The workload: the B0 workload's prompts (``e2e/workload-B0-small.json``) truncated to ``prompt_len`` for shapes <= 32, or synthetic
seeded random prompts (D4's 6.5K practice) for longer shapes; ``max_tokens = STEPS + 1`` slots; 4 requests.  Written to ``REC/workload.json``.

    python -m record_v5.serve_v5 --config run.json --record REC --py-vllm PY [--reps 3] [--skip-stock]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from record_v5 import POC, ROOT, deps, freeze
from record_v5 import config as cfgmod
from record_v5 import workload_v5 as wl5

deps.install()

WHEEL_SHA = "7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3"
VLLM_COMMIT = "d9105ea80"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())



def engine_versions(py: str) -> dict[str, Any]:
    """The serving environment's actual versions (recorded, and compared with the VLLM_COMMIT constant by the reader)."""
    code = "import json,sys,vllm,torch; print(json.dumps({'python': sys.version.split()[0], 'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'executable': sys.executable}))"
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300)
        v = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else {"error": out.stderr[-500:]}
    except Exception as e:  # noqa: BLE001
        v = {"error": f"{type(e).__name__}: {e}"}
    v["vllm_commit_expected"] = VLLM_COMMIT
    v["vllm_commit_matches"] = VLLM_COMMIT in str(v.get("vllm", ""))
    return v


def run_production(py: str, workload: Path, out: Path, *, matched: bool, reps: int, log: Path, adapter_rev: str, case: str = "B0") -> float:
    env = {**os.environ, "PYTHONPATH": str(POC)}
    cmd = [py, "-m", "verity_vllm_adapter.production", "--case", case, "--checkpoints", str(POC / "manifests" / "checkpoints.json"), "--workload", str(workload),
           "--out", str(out), "--adapter-revision", adapter_rev, "--wheel-sha256", WHEEL_SHA, "--vllm-commit", VLLM_COMMIT, "--logprobs", "2", "--reps", str(reps)]
    if matched:
        env["VLLM_BATCH_INVARIANT"] = "1"
        cmd.append("--enforce-eager")
    else:
        env.pop("VLLM_BATCH_INVARIANT", None)
    t0 = time.perf_counter()
    with log.open("w") as f:
        f.write(f"# {utc()} $ {'VLLM_BATCH_INVARIANT=1 ' if matched else ''}{' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(POC)).returncode
    if rc != 0:
        raise SystemExit(f"[serve_v5] production ({'matched' if matched else 'stock'}) exited {rc}; see {log}")
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--py-vllm", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-stock", action="store_true", help="skip the stock-vLLM comparison pass (R19 then UNAVAILABLE)")
    ap.add_argument("--hardware", default=os.environ.get("VERITOR_HARDWARE", "unknown GPU (set VERITOR_HARDWARE)"))
    a = ap.parse_args(argv)
    cfg = cfgmod.load(a.config)
    rec = Path(a.record)
    rec.mkdir(parents=True, exist_ok=True)
    freeze.check(rec, "serve")
    (rec / "adapter").mkdir(exist_ok=True)
    (rec / "logs").mkdir(exist_ok=True)
    if not (rec / "workload.json").exists():
        raise SystemExit(f"{rec / 'workload.json'} missing: the auditor's request rows are written by instance_v5 (epoch.inputs) BEFORE serving; serve_v5 never builds them")
    wl = json.loads((rec / "workload.json").read_text())
    if [r["request_id"] for r in wl["requests"]] != [r["request_id"] for r in wl5.make_workload(cfg)["requests"]]:
        raise SystemExit("REC/workload.json does not match the run config's workload (request ids differ)")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "a9-runner"
    case = str((cfg.raw.get("workload") or {}).get("registry_case", "B0"))  # [E1 night 2026-09-12] the adapter case of this model (profile.CASES); default = TINY
    prod = rec / "adapter" / "production"
    if prod.exists():
        raise SystemExit(f"{prod} exists: a record's original execution is served once")
    wall_m = run_production(a.py_vllm, rec / "workload.json", prod, matched=True, reps=a.reps, log=rec / "logs" / "serve-matched.log", adapter_rev=f"contract/a9-runner@{rev}", case=case)
    tm = json.loads((prod / "timings.json").read_text())
    rows = [json.loads(l) for l in (prod / "production.jsonl").read_text().splitlines() if l.strip()]
    rec_pass = tm["recorded_pass"]
    off = [t["wall_s"] for t in tm["recording_off"]]
    on = [t["wall_s"] for t in tm["recording_on"]]
    serving = {
        "schema": "veritor/a9/serving-v5/v1", "utc": utc(), "hardware": a.hardware, "requests": len(rows), "workload": cfg.workload,
        "engine_versions": engine_versions(a.py_vllm),
        "matched": {"dir": str(prod), "engine": "vLLM " + VLLM_COMMIT + " wheel " + WHEEL_SHA[:12] + ", VLLM_BATCH_INVARIANT=1, enforce_eager, greedy", "wall_s_total": round(wall_m, 1),
                    "engine_build_s": tm.get("engine_build_s"), "recorded_pass_wall_s": rec_pass.get("wall_s"), "recording_off_wall_s": off, "recording_on_wall_s": on,
                    "production_overhead_ratio": tm.get("production_overhead_ratio"), "prefix_cache_resets_succeeded": tm.get("prefix_cache_resets_succeeded"),
                    "capture_fields": ["prompt_token_ids (inputs)", "output_token_ids (outputs)", "seed / sampling_params"], "label": "MEASURED"},
        "stock": None,
        "agreement": None,
        "equivalence_note": "batch invariance and token agreement are not proof of equivalence to C; equivalence is established only by the conformance vectors / holdout captures of the "
                            "matched configuration against the registered semantic profile (R19), which this run does not re-run",
    }
    if not a.skip_stock:
        stock = rec / "adapter" / "stock-baseline"
        wall_s = run_production(a.py_vllm, rec / "workload.json", stock, matched=False, reps=a.reps, log=rec / "logs" / "serve-stock.log", adapter_rev=f"contract/a9-runner@{rev}-stock", case=case)
        ts = json.loads((stock / "timings.json").read_text())
        srows = {r["request_id"]: r for r in (json.loads(l) for l in (stock / "production.jsonl").read_text().splitlines() if l.strip())}
        agree = []
        for r in rows:
            s = srows.get(r["request_id"])
            a_out, s_out = list(r.get("output_token_ids") or []), list((s or {}).get("output_token_ids") or [])
            n = sum(1 for x, y in zip(a_out, s_out) if x == y)
            agree.append({"request_id": r["request_id"], "matched_len": len(a_out), "stock_len": len(s_out), "identical": a_out == s_out, "prefix_agree": next((i for i, (x, y) in enumerate(zip(a_out, s_out)) if x != y), min(len(a_out), len(s_out))), "equal_positions": n})
        soff = [t["wall_s"] for t in ts["recording_off"]]
        serving["stock"] = {"dir": str(stock), "engine": "vLLM " + VLLM_COMMIT + " stock kernels (no VLLM_BATCH_INVARIANT, no enforce_eager)", "wall_s_total": round(wall_s, 1),
                            "engine_build_s": ts.get("engine_build_s"), "recording_off_wall_s": soff, "recording_on_wall_s": [t["wall_s"] for t in ts["recording_on"]], "label": "MEASURED",
                            "note": "comparison only; nothing from this pass is committed"}
        serving["agreement"] = {"requests_identical": sum(1 for x in agree if x["identical"]), "requests": len(agree), "per_request": agree}
        m_off = sum(off) / len(off) if off else None
        s_off = sum(soff) / len(soff) if soff else None
        serving["overhead"] = {"matched_recording_off_mean_s": m_off, "stock_recording_off_mean_s": s_off, "matched_over_stock_ratio": (m_off / s_off) if m_off and s_off else None,
                               "matched_recording_on_mean_s": sum(on) / len(on) if on else None,
                               "attribution": "matched/stock (recording off) = batch-invariant kernels + eager execution; matched on/off = capture of inputs/outputs/seed (+ boundary records); "
                                              "engine build excluded from both; same 4 requests, fresh prefill each pass"}
    (rec / "serving_v5.json").write_text(json.dumps(serving, indent=1) + "\n")
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "original execution + collection (matched C: recorded pass, batch-invariant eager, inputs/outputs/seed captured)", "class": "execution",
                            "seconds": rec_pass.get("wall_s"), "hardware": a.hardware, "batch": f"{len(rows)} requests", "label": "MEASURED",
                            "note": f"class 2; engine build {tm.get('engine_build_s')} s and warm-up excluded; recording on/off x{a.reps}: {on} / {off}"}) + "\n")
        if serving["stock"]:
            f.write(json.dumps({"stage": "native baseline: stock vLLM, same requests, recording off (mean of reps)", "class": "native",
                                "seconds": serving["overhead"]["stock_recording_off_mean_s"], "hardware": a.hardware, "batch": f"{len(rows)} requests", "label": "MEASURED",
                                "note": "class 1 (native / trusted-replay baseline), serving side; the replay-side native baseline is the replay's forward time (class 4 decomposition)"}) + "\n")
        f.write(json.dumps({"stage": "serve wall (matched production incl. engine build, warm-up, reps)", "class": "wall_only", "seconds": round(wall_m, 1), "hardware": a.hardware,
                            "batch": f"{len(rows)} requests", "label": "MEASURED", "note": "not a cost class; the whole-process wall for scheduling"}) + "\n")
    print(f"[serve_v5] matched recorded pass {rec_pass.get('wall_s')} s; on/off {on}/{off}; stock {serving['stock'] and serving['stock']['recording_off_wall_s']}; "
          f"agreement {serving['agreement'] and serving['agreement']['requests_identical']}/{len(rows)} -> {rec / 'serving_v5.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
