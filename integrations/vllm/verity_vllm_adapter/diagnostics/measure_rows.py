"""Summarise the throughput campaign (CONTRACT §16.7) into Table 4 rows + a Markdown table.
CPU / JSON only::

    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.measure_rows \
        --results results/measure-4090-20260907T1614Z --adapter-revision <git sha>

Reads ``raw/<model>/<config>/result.json`` written by ``throughput.py`` and writes
``table4.jsonl`` (one row per model x config), ``summary.json`` and ``summary.md``.

Table 4 mapping (the contract's Table 4 is the cost table; these are production-throughput rows):
``partition_id`` = ``<model>``, ``instrumentation_mode`` = ``production-throughput:<config>``,
``production_overhead_ratio`` = C0 steady-state total tokens/s divided by this configuration's
(shape 128/128; 1.0 for C0 itself; >1 is a slowdown), ``peak_replay_memory.device_gb`` = torch peak
allocated during the passes, ``restrictions`` carries the per-shape numbers, achieved concurrency,
warm-up and the kernels the flag switched, ``spend.gpu_hours`` = wall of the run (build + all
passes) / 3600. Everything replay-related is ``None`` (not measured here).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--adapter-revision", required=True)
a = ap.parse_args()
R = Path(a.results)
exp = R.name

runs: dict[tuple[str, str], dict] = {}
for f in sorted(R.glob("raw/*/*/result.json")):
    d = json.load(open(f))
    runs[(d["model"]["repo"].split("/")[-1], d["config"]["id"])] = d

MODELS = ["SmolLM2-135M", "Qwen2.5-1.5B", "Qwen2.5-7B-Instruct"]
CONFIGS = ["C0", "C1", "C2"]


def agg(d: dict, shape: tuple[int, int]) -> dict | None:
    ps = [p for p in d["passes"] if (p["input_len"], p["output_len"]) == shape]
    if not ps:
        return None
    tps = [p["total_tokens_per_s"] for p in ps]
    return {
        "reps": len(ps),
        "total_tokens_per_s": {"mean": st.fmean(tps), "min": min(tps), "max": max(tps)},
        "generated_tokens_per_s": st.fmean(p["generated_tokens_per_s"] for p in ps),
        "requests_per_s": st.fmean(p["requests_per_s"] for p in ps),
        "wall_s": st.fmean(p["wall_s"] for p in ps),
        "latency_e2e_s": {"p50": st.fmean(p["latency_e2e_s"]["p50"] for p in ps), "p95": st.fmean(p["latency_e2e_s"]["p95"] for p in ps)},
        "ttft_s": {"p50": st.fmean(p["ttft_s"]["p50"] for p in ps), "p95": st.fmean(p["ttft_s"]["p95"] for p in ps)},
        "tpot_s": {"p50": st.fmean(p["tpot_s"]["p50"] for p in ps), "p95": st.fmean(p["tpot_s"]["p95"] for p in ps)},
        "concurrency": {
            "running_mean": st.fmean(p["concurrency"]["running_mean"] for p in ps),
            "running_max": max(p["concurrency"]["running_max"] for p in ps),
            "waiting_max": max(p["concurrency"]["waiting_max"] for p in ps),
        },
        "engine_steps": ps[0]["engine_steps"],
        "gpu_memory": ps[0]["gpu_memory"],
        "all_exact_output_len": all(p["all_requests_generated_exactly_output_len"] for p in ps),
    }


def kernel_summary(d: dict) -> dict:
    k = d.get("kernels") or {}
    fam = k.get("families", {})
    out = {}
    for f in ("matmul", "rmsnorm", "attention", "activation", "rope", "inductor_fused", "kv_cache", "sampling"):
        if f in fam:
            out[f] = {"count": fam[f]["count"], "device_time_us": round(fam[f]["device_time_us"]), "kernels": [n[:90] for n in fam[f]["kernels"][:4]]}
    return out


summary = {"experiment_id": exp, "gpu": None, "software": None, "models": {}, "note": "RTX 4090 (24 GB, sm89, no H100 features): relative costs on this device only"}
rows = []
md = [
    "| model | config | flags | mode (effective) | shape | total tok/s (mean of 3; min-max) | gen tok/s | e2e p50 / p95 s | TTFT p50 s | achieved concurrency mean / max (of 256) | steps | warm-up pass s | engine build s | peak mem torch GB / smi MiB | slowdown vs C0 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for m in MODELS:
    base = runs.get((m, "C0"))
    for c in CONFIGS:
        d = runs.get((m, c))
        if d is None:
            continue
        summary["gpu"] = summary["gpu"] or d["gpu"]
        summary["software"] = summary["software"] or d["software"]
        shapes = {}
        for sh in [(s["input_len"], s["output_len"]) for s in d["workload"]["shapes"]]:
            ag = agg(d, sh)
            if ag is None:
                continue
            ratio = None
            if base is not None:
                bg = agg(base, sh)
                if bg:
                    ratio = bg["total_tokens_per_s"]["mean"] / ag["total_tokens_per_s"]["mean"]
            ag["slowdown_vs_C0"] = ratio
            shapes[f"{sh[0]}/{sh[1]}"] = ag
            md.append(
                f"| {m} | {c} | {' '.join(f'{k}={v}' for k, v in d['config']['env'].items() if k != 'VLLM_ENABLE_V1_MULTIPROCESSING') or '-'}{' enforce_eager' if d['config']['enforce_eager'] else ''} | "
                f"mode {d['effective']['compilation']['mode']}, cudagraph {d['effective']['compilation']['cudagraph_mode']}, custom_ops {d['effective']['compilation']['custom_ops']} | {sh[0]}/{sh[1]} | "
                f"{ag['total_tokens_per_s']['mean']:,.0f} ({ag['total_tokens_per_s']['min']:,.0f}-{ag['total_tokens_per_s']['max']:,.0f}) | {ag['generated_tokens_per_s']:,.0f} | "
                f"{ag['latency_e2e_s']['p50']:.3f} / {ag['latency_e2e_s']['p95']:.3f} | {ag['ttft_s']['p50']:.3f} | {ag['concurrency']['running_mean']:.1f} / {ag['concurrency']['running_max']} | {ag['engine_steps']} | "
                f"{d['warmup']['wall_s']:.2f} | {d['effective']['engine_build_s']:.0f} | {ag['gpu_memory']['torch_max_allocated_gb']:.1f} / {ag['gpu_memory']['nvidia_smi_used_peak_mib']} | "
                f"{('%.2fx' % ratio) if ratio else '-'} |"
            )
        ent = {
            "config": d["config"],
            "effective": d["effective"],
            "warmup": {"wall_s": d["warmup"]["wall_s"], "total_tokens_per_s": d["warmup"]["total_tokens_per_s"], "engine_steps": d["warmup"]["engine_steps"]},
            "engine_build_s": d["effective"]["engine_build_s"],
            "shapes": shapes,
            "kernels": kernel_summary(d),
            "model": d["model"],
            "utc": d["utc"],
        }
        summary["models"].setdefault(m, {})[c] = ent
        s128 = shapes.get("128/128")
        s512 = shapes.get("512/64")
        wall_total = d["effective"]["engine_build_s"] + d["warmup"]["wall_s"] + sum(p["wall_s"] for p in d["passes"])
        restrictions = [
            f"device RTX 4090 24GB (sm89); relative costs only",
            f"flags: {json.dumps(d['config']['env'])}, enforce_eager={d['config']['enforce_eager']}",
            f"effective: mode {d['effective']['compilation']['mode']}, cudagraph {d['effective']['compilation']['cudagraph_mode']}, custom_ops {d['effective']['compilation']['custom_ops']}, model_runner {d['effective']['compilation']['model_runner']}",
            f"workload: 256 prompts, greedy, ignore_eos, all added at t=0, max_num_seqs {d['effective']['max_num_seqs']}, max_num_batched_tokens {d['effective']['max_num_batched_tokens']}, kv blocks {d['effective']['kv_cache_blocks_gpu']} x {d['effective']['block_size']}",
            f"engine build (compile/graph capture/alloc) {d['effective']['engine_build_s']:.0f}s; warm-up pass {d['warmup']['wall_s']:.2f}s = {d['warmup']['total_tokens_per_s']:.0f} tok/s (excluded from steady state)",
        ]
        for label, sdat in (("128/128", s128), ("512/64", s512)):
            if sdat:
                restrictions.append(
                    f"shape {label}: total {sdat['total_tokens_per_s']['mean']:.0f} tok/s (min {sdat['total_tokens_per_s']['min']:.0f}, max {sdat['total_tokens_per_s']['max']:.0f}, 3 reps), gen {sdat['generated_tokens_per_s']:.0f} tok/s, "
                    f"e2e p50 {sdat['latency_e2e_s']['p50']:.3f}s p95 {sdat['latency_e2e_s']['p95']:.3f}s, ttft p50 {sdat['ttft_s']['p50']:.3f}s, tpot p50 {sdat['tpot_s']['p50']*1000:.1f}ms, "
                    f"achieved concurrency mean {sdat['concurrency']['running_mean']:.1f} max {sdat['concurrency']['running_max']} (waiting max {sdat['concurrency']['waiting_max']}), {sdat['engine_steps']} steps"
                    + (f", slowdown vs C0 {sdat['slowdown_vs_C0']:.2f}x" if sdat.get("slowdown_vs_C0") else "")
                )
        ks = kernel_summary(d)
        restrictions.append("kernels (profiled short pass): " + "; ".join(f"{f}: {v['kernels'][0]}" + (f" (+{len(v['kernels'])-1})" if len(v["kernels"]) > 1 else "") for f, v in ks.items() if f in ("matmul", "rmsnorm", "attention", "activation", "rope")))
        rows.append(
            {
                "experiment_id": exp,
                "case": "measure",
                "profile_id": f"{c}:{m}",
                "worker": "W4c",
                "adapter_revision": a.adapter_revision,
                "partition_id": m,
                "instrumentation_mode": f"production-throughput:{c}",
                "production_overhead_ratio": (s128 or {}).get("slowdown_vs_C0"),
                "replay_commit_overhead_ratio": None,
                "replay_vs_production_gpu_work": None,
                "transparent_check_cost": {"cpu_s": None, "gpu_s": None},
                "peak_replay_memory": {"device_gb": (s128 or s512 or {}).get("gpu_memory", {}).get("torch_max_allocated_gb"), "host_gb": None, "delta_vs_reference": None},
                "committed_logical_bytes": None,
                "processed_bytes": {"hashed": None, "copied": None, "retained": None, "emitted": None},
                "description_size_bytes": None,
                "constructor_inputs_bytes": None,
                "logical_size": {"gates": None, "vus": None, "rus": None},
                "lookup_cost_s": None,
                "validation_cost_s": None,
                "moe_padding": {"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None},
                "restrictions": restrictions,
                "spend": {"gpu_hours": wall_total / 3600.0, "usd": None},
            }
        )
with open(R / "table4.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
(R / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
(R / "summary.md").write_text("\n".join(md) + "\n")
print(f"wrote {len(rows)} Table 4 rows; {len(summary['models'])} models")
print("\n".join(md))
