"""M0: the observer overhead ladder on the frozen workload (DESIGN §3 / §8).

Rungs (3 repetitions each, same engine, same request each time):
  a  eager baseline (no observer; timing wrappers only)
  b  empty TorchDispatchMode that just calls func
  c  recording mode, no re-entry, no Triton hook, no module hooks (events built, buffer discarded)
  d  + re-entry into vllm.unified_attention_with_output / vllm.unified_kv_cache_update
  e  + Triton launch hook
  f  + module naming hooks
  g  full observer writing the log to disk (gzip)
  h  full observer + snapshots of steps 0 and 8

Reports wall time per engine step (mean over the run's steps, then over the repetitions) and total run wall time;
prints a markdown table and writes DIR/m0.json.

    ... /workspace/venv-cu129/bin/python -m verity_capture.bench.m0_overhead --out /vol/cp/l1/m0 [--reps 3] [--rungs a,b,...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from verity_vllm_adapter import profile as prof  # noqa: F401

prof.apply_env()

from verity_capture import vllm_adapter as va  # noqa: E402

RUNGS: dict[str, tuple[str, dict]] = {
    "a": ("eager baseline (no observer)", {"mode": "none", "write_log": False}),
    "b": ("empty TorchDispatchMode", {"mode": "empty", "write_log": False}),
    "c": ("recording, no re-entry/Triton/module hooks", {"mode": "record", "reenter": False, "triton": False, "module_hooks": False, "write_log": False}),
    "d": ("+ re-entry", {"mode": "record", "reenter": True, "triton": False, "module_hooks": False, "write_log": False}),
    "e": ("+ Triton hook", {"mode": "record", "reenter": True, "triton": True, "module_hooks": False, "write_log": False}),
    "f": ("+ module naming hooks", {"mode": "record", "reenter": True, "triton": True, "module_hooks": True, "write_log": False}),
    "g": ("full observer, log to disk", {"mode": "record", "reenter": True, "triton": True, "module_hooks": True, "write_log": True}),
    "h": ("full observer + snapshots {0,8}", {"mode": "record", "reenter": True, "triton": True, "module_hooks": True, "write_log": True,
                                             "snapshot_steps": frozenset({0, 8})}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="verity_capture/profiles/workload_32x16_1req.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--case", default=None, help="checkpoint role (B0 / B1); default: the workload file's, else B0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload)
    eng = va.build_engine(args.checkpoints, max_num_seqs=args.max_num_seqs, case=va.case_of(wl, args.case))
    print(f"[m0] engine built in {eng.build_s:.1f}s (case {eng.case}, {eng.cp['repo']})", flush=True)
    (out / "versions.json").write_text(json.dumps(va.versions_doc(eng), indent=1) + "\n")

    # warm-up: two uninstrumented passes (Triton JIT / allocator / cuBLAS handles)
    for i in range(2):
        _, tw = va.run_requests(eng, va.throwaway_requests(eng, wl["requests"], seed=100 + i), wl["sampling"])
        print(f"[m0] warm-up {i}: {tw['wall_s']:.3f}s, {tw['engine_steps']} steps", flush=True)

    header = va.make_header(eng, wl, adapter_revision="m0", extra_notes={"argv": sys.argv})
    results: dict[str, dict] = {}
    tokens_ref: list[int] | None = None
    for r in args.rungs.split(","):
        r = r.strip()
        if r not in RUNGS:
            continue
        label, kw = RUNGS[r]
        reps = []
        for rep in range(args.reps):
            cfg = va.ObserverConfig(**kw)
            log_path = None
            if cfg.write_log:
                log_path = out / f"m0_{r}.jsonl.gz"
            if cfg.snapshot_steps:
                cfg.snapshot_dir = out / f"m0_{r}_snapshots"
            cap = va.Capture(eng, header=header if cfg.mode == "record" else None, log_path=log_path, config=cfg, log=lambda *a, **k: None)
            cap.install()
            t0 = time.perf_counter()
            toks, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
            wall = time.perf_counter() - t0
            stats = cap.close()
            gen = next(iter(toks.values()))
            if tokens_ref is None:
                tokens_ref = gen
            steps = cap.step_times
            fw = [s["forward_s"] for s in steps]
            sp = [s["sample_s"] for s in steps]
            ws = [s["wall_s"] for s in steps]
            reps.append({"rep": rep, "wall_s": wall, "engine_wall_s": timing.get("wall_s"), "engine_steps": timing.get("engine_steps"),
                         "observed_steps": len(steps), "step_wall_mean_s": statistics.fmean(ws) if ws else None,
                         "step_wall_prefill_s": ws[0] if ws else None,
                         "step_wall_decode_mean_s": statistics.fmean(ws[1:]) if len(ws) > 1 else None,
                         "forward_mean_s": statistics.fmean(fw) if fw else None, "sample_mean_s": statistics.fmean(sp) if sp else None,
                         "events": stats["events_written"] if cfg.write_log else
                         (stats["recorder"].get("dispatch", 0) * 2 + stats["recorder"].get("triton", 0)),
                         "recorded_calls": stats["recorder"].get("dispatch", 0), "triton_launches": stats["recorder"].get("triton", 0),
                         "snapshot_bytes": stats.get("snapshot_bytes", 0), "tokens_equal_first_rung": gen == tokens_ref,
                         "errors": stats["errors"][:3]})
            print(f"[m0] {r} rep{rep}: wall {wall:.3f}s steps {len(steps)} step-mean {reps[-1]['step_wall_mean_s']:.4f}s "
                  f"decode-mean {reps[-1]['step_wall_decode_mean_s']} events {reps[-1]['events']} tokens_eq {gen == tokens_ref}", flush=True)
        results[r] = {"label": label, "config": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in kw.items()}, "reps": reps}

    # table
    base = results.get("a")
    base_step = statistics.fmean(x["step_wall_mean_s"] for x in base["reps"]) if base else None
    base_wall = statistics.fmean(x["wall_s"] for x in base["reps"]) if base else None
    lines = ["| rung | what | step wall mean (ms) | decode step mean (ms) | prefill step (ms) | total wall (s) | x baseline (step) | events (recorded calls / Triton launches) |",
             "|---|---|---|---|---|---|---|---|"]
    for r, res in results.items():
        st = statistics.fmean(x["step_wall_mean_s"] for x in res["reps"])
        dec = statistics.fmean(x["step_wall_decode_mean_s"] for x in res["reps"] if x["step_wall_decode_mean_s"] is not None)
        pre = statistics.fmean(x["step_wall_prefill_s"] for x in res["reps"])
        wall = statistics.fmean(x["wall_s"] for x in res["reps"])
        ratio = f"{st / base_step:.2f}" if base_step else "-"
        x0 = res["reps"][0]
        ev = f"{x0['events']} ({x0['recorded_calls']} / {x0['triton_launches']})" if x0["recorded_calls"] else "-"
        lines.append(f"| {r} | {res['label']} | {st * 1e3:.2f} | {dec * 1e3:.2f} | {pre * 1e3:.2f} | {wall:.3f} | {ratio} | {ev} |")
    table = "\n".join(lines)
    print(table, flush=True)
    doc = {"schema": "verity-capture/m0/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reps": args.reps,
           "workload": args.workload, "results": results, "table_md": table, "baseline_step_wall_s": base_step,
           "baseline_total_wall_s": base_wall, "versions": va.versions_doc(eng)}
    (out / "m0.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (out / "m0.md").write_text(table + "\n")
    print(f"[m0] wrote {out / 'm0.json'}", flush=True)


if __name__ == "__main__":
    main()
