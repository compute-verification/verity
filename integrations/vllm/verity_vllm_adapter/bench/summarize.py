"""Turn the raw throughput rows of a bench matrix into Table 4 rows + ratios (CPU only, JSON only).

Reads ``<bench_root>/<model>/<config>/{rows.jsonl,engine.json,kernels.json}`` (as written by
``throughput.py`` / ``pod/run_matrix.sh``) and writes into the experiment directory:

* ``table4.jsonl``: one contract §9 Table 4 row per model x config x shape. Field mapping for these
  (throughput) rows, since Table 4 was designed for replay cost: ``partition_id`` =
  ``throughput:<model>/<in>x<out>`` (there is no partition); ``instrumentation_mode`` = the
  configuration id and its flags; ``production_overhead_ratio`` = **mean steady-state wall time of
  this configuration / that of C0 on the same model and shape** (C0 = 1.0; §16.7's "cost of the
  fallback"); ``peak_replay_memory.device_gb`` = peak torch-allocated device memory of the run;
  ``restrictions`` = the flags carried, the EFFECTIVE execution mode and the kernel families that
  ran; ``spend`` = the run's wall time (build + warm-up + passes + profile) at the pod's hourly
  price. Everything not measured is null, never 0.
* ``timings.json`` (§12: raw measurements retained): per model x config x shape the per-rep
  tokens/s, latencies, achieved concurrency, memory, plus warm-up and build times, and the
  ratio table.
* ``ratios.md``: the Markdown table (C1/C0, C2/C1, C2/C0 as time-cost ratios, i.e. >1 = slower).

    python -m verity_vllm_adapter.bench.summarize --bench-root results/<exp>/bench --experiment-id <exp> \
        --case H100COST --profile-id <12hex> --adapter-revision <sha> --matrix-log results/<exp>/bench/matrix.log
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from verity_vllm import rows as R

CONFIG_ORDER = ["C0", "C1", "C1v2", "C2"]
MODEL_ORDER = ["Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"]


def load_dir(d: Path) -> dict[str, Any] | None:
    if not (d / "rows.jsonl").exists() or not (d / "engine.json").exists():
        return None
    rows = [json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()]
    engine = json.loads((d / "engine.json").read_text())
    kernels = json.loads((d / "kernels.json").read_text()) if (d / "kernels.json").exists() else None
    # re-derive the memory/compile figures from the retained log lines (the parser learned the
    # batch-invariant phrasing of the compile-time line after the first runs were recorded)
    if engine.get("effective_log_lines"):
        from verity_vllm_adapter.bench.throughput import parse_memory_lines

        engine["engine_memory"] = parse_memory_lines(engine["effective_log_lines"])
    return {"rows": rows, "engine": engine, "kernels": kernels}


def run_seconds(matrix_log: Path | None) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    if matrix_log is None or not matrix_log.exists():
        return out
    for l in matrix_log.read_text().splitlines():
        m = re.search(r"end (\S+)/(\S+) rc=0 in (\d+)s", l)
        if m:
            out[(m.group(1), m.group(2))] = int(m.group(3))
    return out


def agg(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def shape_key(r: dict[str, Any]) -> str:
    return f"{r['shape']['input_len']}x{r['shape']['output_len']}"


def kernel_families_short(kern: dict[str, Any] | None) -> list[str]:
    if not kern:
        return []
    fams = kern.get("families", {})
    keep = [f for f in fams if f.split(".")[0] in ("gemm", "attention", "rmsnorm", "activation", "rope", "inductor", "softmax", "bmm", "mean")]
    return [f"{f}={fams[f]['share_of_device_time']:.3f}" for f in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--case", default="H100COST")
    ap.add_argument("--profile-id", required=True)
    ap.add_argument("--worker", default="W4d")
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--matrix-log", default=None)
    ap.add_argument("--usd-per-hour", type=float, default=3.49)
    ap.add_argument("--out", default=None, help="experiment dir (default: parent of bench-root)")
    args = ap.parse_args()

    root = Path(args.bench_root)
    out = Path(args.out) if args.out else root.parent
    secs = run_seconds(Path(args.matrix_log) if args.matrix_log else root / "matrix.log")
    common = R.Common(args.experiment_id, args.case, args.profile_id, args.worker, args.adapter_revision)

    data: dict[str, dict[str, dict[str, Any]]] = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            d = load_dir(cdir)
            if d:
                data.setdefault(mdir.name, {})[cdir.name] = d
    models = sorted(data, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))

    timings: dict[str, Any] = {"schema": "verity-vllm/bench/timings/v1", "experiment_id": args.experiment_id, "usd_per_hour": args.usd_per_hour, "models": {}}
    table4: list[dict[str, Any]] = []
    md: list[str] = []
    for model in models:
        cfgs = sorted(data[model], key=lambda c: (CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 99, c))
        per_cfg: dict[str, Any] = {}
        for cfg in cfgs:
            d = data[model][cfg]
            steady = [r for r in d["rows"] if r["phase"] == "steady"]
            warm = [r for r in d["rows"] if r["phase"] == "warmup"]
            eng = d["engine"]
            shapes: dict[str, Any] = {}
            for sk in sorted({shape_key(r) for r in steady}, key=lambda s: int(s.split("x")[0])):
                rs = [r for r in steady if shape_key(r) == sk]
                shapes[sk] = {
                    "reps": len(rs),
                    "wall_s": agg([r["wall_s"] for r in rs]),
                    "generated_tokens_per_s": agg([r["generated_tokens_per_s"] for r in rs]),
                    "prompt_tokens_per_s": agg([r["prompt_tokens_per_s"] for r in rs]),
                    "total_tokens_per_s": agg([r["total_tokens_per_s"] for r in rs]),
                    "requests_per_s": agg([r["requests_per_s"] for r in rs]),
                    "e2e_p50_s": agg([r["latency_s"]["e2e"]["p50"] for r in rs]),
                    "e2e_p95_s": agg([r["latency_s"]["e2e"]["p95"] for r in rs]),
                    "ttft_p50_s": agg([r["latency_s"]["ttft"]["p50"] for r in rs]),
                    "ttft_p95_s": agg([r["latency_s"]["ttft"]["p95"] for r in rs]),
                    "tpot_p50_ms": agg([r["latency_s"]["tpot_ms"]["p50"] for r in rs]),
                    "tpot_p95_ms": agg([r["latency_s"]["tpot_ms"]["p95"] for r in rs]),
                    "concurrency_cap": rs[0]["concurrency"].get("cap") or rs[0]["scheduler"]["max_num_seqs"],
                    "concurrency_peak_achieved": max(r["concurrency"]["peak_achieved"] for r in rs),
                    "concurrency_mean_achieved": agg([r["concurrency"]["mean_achieved"] for r in rs]),
                    "engine_steps": agg([r["engine_steps"] for r in rs]),
                    "preemptions": sum(r["concurrency"]["preemptions"] for r in rs),
                    "cached_tokens": sum(r["cached_tokens"] for r in rs),
                    # per rep, with W4c's field names (verity_vllm_adapter/diagnostics/throughput.py, schema
                    # verity-vllm/measure/throughput/v1) as aliases so the two devices' passes compare field by field
                    "per_rep": [{"rep": r["rep"], "wall_s": r["wall_s"], "engine_steps": r["engine_steps"],
                                 "generated_tokens": r["generated_tokens"], "prompt_tokens": r["prompt_tokens"],
                                 "generated_tokens_per_s": r["generated_tokens_per_s"], "prompt_tokens_per_s": r["prompt_tokens_per_s"],
                                 "total_tokens_per_s": r["total_tokens_per_s"], "requests_per_s": r["requests_per_s"],
                                 "latency_e2e_s": r["latency_s"]["e2e"], "ttft_s": r["latency_s"]["ttft"],
                                 "tpot_s": {k: (v / 1000.0 if v is not None else None) for k, v in r["latency_s"]["tpot_ms"].items()},
                                 "concurrency": {"running_max": r["concurrency"]["peak_achieved"], "running_mean_nonzero": r["concurrency"]["mean_achieved"],
                                                 "preemptions": r["concurrency"]["preemptions"]},
                                 "gpu_memory": {"torch_max_allocated_gb": r["gpu_memory"]["torch_max_allocated_gb"],
                                                "torch_max_reserved_gb": r["gpu_memory"]["torch_reserved_gb"],
                                                "nvidia_smi_used_peak_mib": r["gpu_memory"]["nvidia_smi_used_mib"]},
                                 "cached_tokens": r["cached_tokens"]} for r in rs],
                }
            eff = eng["config"]["effective"]
            per_cfg[cfg] = {
                "label": eng["config"]["label"],
                "env_flags": eng["config"]["env_flags"],
                "harness_env": eng["config"]["harness_env"],
                "engine_kwargs": eng["config"]["engine_kwargs"],
                "effective": eff,
                "engine_build_s": eng["engine_build_s"],
                "engine_memory": eng.get("engine_memory"),
                "warmup": {**eng["warmup"], "generated_tokens_per_s": warm[0]["generated_tokens_per_s"] if warm else None},
                "gpu_memory_peak": max(r["gpu_memory"]["torch_max_allocated_gb"] for r in d["rows"]),
                "nvidia_smi_used_mib": max((r["gpu_memory"]["nvidia_smi_used_mib"] or 0) for r in d["rows"]),
                "scheduler": eng["scheduler"],
                "kernel_families": {k: {kk: vv for kk, vv in v.items()} for k, v in (d["kernels"] or {}).get("families", {}).items()} if d["kernels"] else None,
                "run_wall_s": secs.get((model, cfg)),
                "shapes": shapes,
            }
        # ratios vs C0 and vs C1 (time-cost: wall/wall, >1 = slower)
        ratios: dict[str, dict[str, float | None]] = {}
        for cfg in cfgs:
            for sk, s in per_cfg[cfg]["shapes"].items():
                r: dict[str, float | None] = {}
                for base in ("C0", "C1"):
                    b = per_cfg.get(base, {}).get("shapes", {}).get(sk)
                    r[f"vs_{base}"] = (s["wall_s"]["mean"] / b["wall_s"]["mean"]) if b and b["wall_s"]["mean"] else None
                ratios.setdefault(cfg, {})[sk] = r  # type: ignore[assignment]
        timings["models"][model] = {"configs": per_cfg, "time_cost_ratios": ratios}

        # Table 4 rows
        for cfg in cfgs:
            pc = per_cfg[cfg]
            for sk, s in pc["shapes"].items():
                eff = pc["effective"]
                restr = [f"config={cfg}"] + [f"env:{k}={v}" for k, v in pc["env_flags"].items()] + (
                    ["enforce_eager=True"] if pc["engine_kwargs"].get("enforce_eager") else []
                ) + [
                    f"effective:mode={eff.get('mode')},cudagraph={eff.get('cudagraph_mode')},runner={eff.get('model_runner')},"
                    f"attn={eff.get('attention_impl', '').split('.')[-1]},fa_version={eff.get('flash_attn_version')},custom_ops={','.join(eff.get('custom_ops') or [])}",
                    "harness:VLLM_ENABLE_V1_MULTIPROCESSING=0,VLLM_USE_FLASHINFER_SAMPLER=0(required on this pod: no ninja),gpu_memory_utilization="
                    + str(pc["engine_kwargs"].get("gpu_memory_utilization")),
                    f"workload:{s['reps']}x{data[model][cfg]['rows'][-1]['shape']['requests']}req,{sk},greedy,ignore_eos,prefix-cache reset per pass,cached_tokens={s['cached_tokens']}",
                    f"measured:gen_tok_s={s['generated_tokens_per_s']['mean']:.0f},prompt_tok_s={s['prompt_tokens_per_s']['mean']:.0f},"
                    f"e2e_p50_s={s['e2e_p50_s']['mean']:.3f},e2e_p95_s={s['e2e_p95_s']['mean']:.3f},conc_peak={s['concurrency_peak_achieved']},"
                    f"conc_mean={s['concurrency_mean_achieved']['mean']:.1f},cap={s['concurrency_cap']}",
                    "kernels:" + ";".join(kernel_families_short(data[model][cfg]["kernels"])),
                    f"raw=bench/{model}/{cfg}/rows.jsonl",
                ]
                ratio = ratios[cfg][sk]["vs_C0"]
                gpu_h = (pc["run_wall_s"] / 3600.0) if pc["run_wall_s"] else None
                row = R.table4_row(
                    common,
                    partition_id=f"throughput:{model}/{sk}",
                    instrumentation_mode=f"{cfg}:" + ("eager" if pc["engine_kwargs"].get("enforce_eager") else "compiled+cudagraph")
                    + ";flags=" + (",".join(f"{k}={v}" for k, v in pc["env_flags"].items()) or "none"),
                    production_overhead_ratio=ratio,
                    replay_commit_overhead_ratio=None,
                    replay_vs_production_gpu_work=None,
                    transparent_check_cost={"cpu_s": None, "gpu_s": None},
                    peak_replay_memory={"device_gb": pc["gpu_memory_peak"], "host_gb": None, "delta_vs_reference": None},
                    committed_logical_bytes=None,
                    processed_bytes={"hashed": None, "copied": None, "retained": None, "emitted": None},
                    description_size_bytes=None,
                    constructor_inputs_bytes=None,
                    logical_size={"gates": None, "vus": None, "rus": None},
                    lookup_cost_s=None,
                    validation_cost_s=None,
                    moe_padding={"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None},
                    restrictions=restr,
                    spend={"gpu_hours": gpu_h, "usd": (gpu_h * args.usd_per_hour) if gpu_h else None},
                )
                table4.append(row)

        # Markdown
        md.append(f"### {model}")
        md.append("")
        md.append("| shape | config | gen tok/s (mean of reps; min–max) | prompt tok/s | e2e p50 / p95 s | TTFT p50 s | TPOT p50 ms | conc peak / mean (cap) | steps | peak dev GB | time-cost vs C0 | vs C1 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg in cfgs:
            pc = per_cfg[cfg]
            for sk, s in pc["shapes"].items():
                r = ratios[cfg][sk]
                md.append(
                    f"| {sk} | {cfg} | {s['generated_tokens_per_s']['mean']:,.0f} ({s['generated_tokens_per_s']['min']:,.0f}–{s['generated_tokens_per_s']['max']:,.0f}) | "
                    f"{s['prompt_tokens_per_s']['mean']:,.0f} | {s['e2e_p50_s']['mean']:.2f} / {s['e2e_p95_s']['mean']:.2f} | {s['ttft_p50_s']['mean']:.3f} | "
                    f"{s['tpot_p50_ms']['mean']:.2f} | {s['concurrency_peak_achieved']} / {s['concurrency_mean_achieved']['mean']:.0f} ({s['concurrency_cap']}) | "
                    f"{s['engine_steps']['mean']:.0f} | {pc['gpu_memory_peak']:.1f} | "
                    f"{('%.3f' % r['vs_C0']) if r['vs_C0'] else '-'} | {('%.3f' % r['vs_C1']) if r['vs_C1'] else '-'} |"
                )
        md.append("")
        md.append("Effective modes and cold start: " + "; ".join(
            f"**{cfg}** mode={per_cfg[cfg]['effective'].get('mode')}, cudagraph={per_cfg[cfg]['effective'].get('cudagraph_mode')}, "
            f"runner={per_cfg[cfg]['effective'].get('model_runner')}, FA{per_cfg[cfg]['effective'].get('flash_attn_version')}, "
            f"build {per_cfg[cfg]['engine_build_s']:.0f} s (compile {((per_cfg[cfg].get('engine_memory') or {}).get('torch_compile_s'))} s), "
            f"warm-up pass {per_cfg[cfg]['warmup']['wall_s']:.2f} s"
            for cfg in cfgs))
        md.append("")

    # compact cross-model ratio table (time-cost ratios of mean steady-state wall time; >1 = slower)
    md.append("### Ratios (mean steady-state wall time; C1/C0 = cost of the batch-invariant profile, C2/C1 = cost of disabling compilation + CUDA graphs on top of it)")
    md.append("")
    md.append("| model | shape | C0 gen tok/s | C1 gen tok/s | C2 gen tok/s | C1/C0 | C2/C1 | C2/C0 | achieved conc peak / mean (all configs) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for model, tm in timings["models"].items():
        cf = tm["configs"]
        if not all(c in cf for c in ("C0", "C1", "C2")):
            continue
        for sk in cf["C0"]["shapes"]:
            g = {c: cf[c]["shapes"].get(sk) for c in ("C0", "C1", "C2")}
            if any(v is None for v in g.values()):
                continue
            w = {c: g[c]["wall_s"]["mean"] for c in g}
            conc = " / ".join(f"{g[c]['concurrency_peak_achieved']}/{g[c]['concurrency_mean_achieved']['mean']:.0f}" for c in ("C0", "C1", "C2"))
            md.append(f"| {model} | {sk} | {g['C0']['generated_tokens_per_s']['mean']:,.0f} | {g['C1']['generated_tokens_per_s']['mean']:,.0f} | "
                      f"{g['C2']['generated_tokens_per_s']['mean']:,.0f} | **{w['C1'] / w['C0']:.2f}** | **{w['C2'] / w['C1']:.2f}** | {w['C2'] / w['C0']:.2f} | {conc} |")
    md.append("")

    R.write_jsonl(out / "table4.jsonl", table4)
    (out / "timings.json").write_text(json.dumps(timings, indent=1) + "\n")
    (out / "ratios.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(table4)} table4 rows, timings.json, ratios.md -> {out}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
