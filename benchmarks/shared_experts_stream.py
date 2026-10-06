"""What does ``VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`` cost a served engine?  GLM-4.7-Flash on one GPU, the served engine's
environment (``verity_vllm.engine.env.ENGINE``, copied here as of branch ``cursor/engine-shared-experts-one-stream-a3ac``) and
engine args (``engine_profile.engine_kwargs`` + ``build.engine_kwargs_for``: eager, bf16, max_model_len 4096,
gpu_memory_utilization 0.80, no prefix caching, no FlashInfer autotune), with the stream flag the only difference:

    one   VLLM_DISABLE_SHARED_EXPERTS_STREAM=1   (the shared expert on the main stream, as ENGINE now has it)
    aux   the flag unset                          (vLLM's default: the shared expert on an aux stream at <= 256 tokens a step)

Each arm runs in its own engine process; reps alternate the arms (one, aux, aux, one, one, aux, ...).  Points, in each engine:
batch-1 TPOT, decode throughput at batch 8, 32 and 128 (every decode step <= 256 tokens), and one prefill-heavy point
(16 prompts x 2048 tokens, one output token: steps far above 256 tokens, so neither arm uses the aux stream there).
Every engine also counts the order vLLM chose for its shared experts (``SharedExperts._determine_shared_experts_order``).

    python benchmarks/shared_experts_stream.py run [--reps 3]      # into $RESEARCH_RUN_DIR/out
    python benchmarks/shared_experts_stream.py paired [--pairs 8]  # both arms in one engine, paired (``paired``'s docstring)
    python benchmarks/shared_experts_stream.py summarize OUT
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODEL = "/workspace/hf/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
ENGINE = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
}
FLAG = "VLLM_DISABLE_SHARED_EXPERTS_STREAM"
ARMS = ("one", "aux")
DECODE_BATCHES = (8, 32, 128)
DECODE_IN, DECODE_OUT = 256, 128
B1_IN, B1_OUT, B1_REQS = 512, 128, 3
PREFILL_B, PREFILL_IN = 16, 2048


def _arm_env(arm: str) -> dict[str, str]:
    env = dict(os.environ)
    for k, v in ENGINE.items():
        env.setdefault(k, v)
    if arm == "one":
        env[FLAG] = "1"
    else:
        env.pop(FLAG, None)
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _gpu() -> dict:
    q = "name,uuid,driver_version,memory.total,clocks.max.sm,power.limit"
    sel = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", *([f"--id={sel.split(',')[0]}"] if sel else [])],
                       capture_output=True, text=True)
    return dict(zip(q.split(","), [s.strip() for s in r.stdout.strip().splitlines()[0].split(",")])) if r.returncode == 0 else {}


def arm(args) -> int:
    import numpy as np
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.runner import shared_experts as se

    counts: dict[str, int] = {}
    orig = se.SharedExperts._determine_shared_experts_order

    def counted(self, hidden_states):
        order = orig(self, hidden_states)
        key = f"{order.name}@{'le256' if hidden_states.shape[0] <= 256 else 'gt256'}"
        counts[key] = counts.get(key, 0) + 1
        return order

    se.SharedExperts._determine_shared_experts_order = counted

    t0 = time.perf_counter()
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.80,
              trust_remote_code=False, enforce_eager=True, enable_prefix_caching=False, max_num_seqs=max(DECODE_BATCHES),
              kernel_config={"enable_flashinfer_autotune": False})
    init_s = time.perf_counter() - t0
    vocab = llm.get_tokenizer().vocab_size
    rng = np.random.default_rng(1000 + args.rep)

    def prompts(n: int, length: int) -> list[dict]:
        return [{"prompt_token_ids": rng.integers(1000, vocab - 1000, length).tolist()} for _ in range(n)]

    def timed(ps: list[dict], out_tokens: int) -> float:
        sp = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        outs = llm.generate(ps, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        assert all(len(o.outputs[0].token_ids) == out_tokens for o in outs)
        return dt

    timed(prompts(8, 256), 32)
    timed(prompts(PREFILL_B, PREFILL_IN), 1)
    counts.clear()

    points = {}
    full = [timed(prompts(1, B1_IN), B1_OUT) for _ in range(B1_REQS)]
    one = [timed(prompts(1, B1_IN), 1) for _ in range(B1_REQS)]
    points["b1_tpot_ms"] = (statistics.median(full) - statistics.median(one)) / (B1_OUT - 1) * 1e3
    for b in DECODE_BATCHES:
        t_full = timed(prompts(b, DECODE_IN), DECODE_OUT)
        t_one = timed(prompts(b, DECODE_IN), 1)
        points[f"b{b}_decode_tok_s"] = b * (DECODE_OUT - 1) / (t_full - t_one)
    t_pre = timed(prompts(PREFILL_B, PREFILL_IN), 1)
    points["prefill_tok_s"] = PREFILL_B * PREFILL_IN / t_pre

    res = {"arm": args.arm, "rep": args.rep, "points": points, "init_s": init_s, "shared_expert_orders": counts,
           "env": {k: os.environ.get(k) for k in [*ENGINE, "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "CUDA_VISIBLE_DEVICES"]},
           "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}, "gpu": _gpu(),
           "affinity": sorted(os.sched_getaffinity(0)),
           "engine": {"model": args.model, "enforce_eager": True, "max_model_len": 4096, "gpu_memory_utilization": 0.80,
                      "enable_prefix_caching": False, "max_num_seqs": max(DECODE_BATCHES),
                      "max_num_batched_tokens": llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens},
           "workload": {"b1": [B1_IN, B1_OUT, B1_REQS], "decode": [DECODE_IN, DECODE_OUT, DECODE_BATCHES],
                        "prefill": [PREFILL_B, PREFILL_IN]}}
    out = Path(args.out) / f"arm-r{args.rep}-{args.arm}.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0


def paired(args) -> int:
    """Both arms in one engine (flag unset, so vLLM builds the aux stream), each decode point measured back to back in both,
    the order alternating by pair: ``one`` forces ``NO_OVERLAP``, the order the flag's ``_stream = None`` gives; ``aux`` is
    vLLM's own choice.  Host load that drifts between engines (shared cores) cancels in the paired deltas."""
    os.environ.update({k: v for k, v in _arm_env("aux").items() if k in ENGINE or k == "HF_HUB_OFFLINE"})
    os.environ.pop(FLAG, None)
    import numpy as np
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.runner import shared_experts as se

    mode = {"m": "aux"}
    counts: dict[str, dict[str, int]] = {"one": {}, "aux": {}}
    orig = se.SharedExperts._determine_shared_experts_order

    def switched(self, hidden_states):
        order = orig(self, hidden_states)
        if mode["m"] == "one":
            order = se.SharedExpertsOrder.NO_OVERLAP
        c = counts[mode["m"]]
        key = f"{order.name}@{'le256' if hidden_states.shape[0] <= 256 else 'gt256'}"
        c[key] = c.get(key, 0) + 1
        return order

    se.SharedExperts._determine_shared_experts_order = switched
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.80,
              trust_remote_code=False, enforce_eager=True, enable_prefix_caching=False, max_num_seqs=max(DECODE_BATCHES),
              kernel_config={"enable_flashinfer_autotune": False})
    vocab = llm.get_tokenizer().vocab_size
    rng = np.random.default_rng(2000)
    out_tokens = args.out_tokens

    def timed(ps: list[dict], n: int) -> float:
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        llm.generate(ps, sp, use_tqdm=False)
        return time.perf_counter() - t

    def prompts(n: int, length: int) -> list[dict]:
        return [{"prompt_token_ids": rng.integers(1000, vocab - 1000, length).tolist()} for _ in range(n)]

    for m in ARMS:
        mode["m"] = m
        timed(prompts(8, 256), 32)
    for c in counts.values():
        c.clear()
    shapes = {"b1": (1, B1_IN), **{f"b{b}": (b, DECODE_IN) for b in DECODE_BATCHES}}
    samples: list[dict] = []
    for p in range(args.pairs):
        order = list(ARMS) if p % 2 == 0 else list(reversed(ARMS))
        for name, (b, length) in shapes.items():
            mode["m"] = "aux"
            t1 = timed(prompts(b, length), 1)
            for m in order:
                mode["m"] = m
                dt = timed(prompts(b, length), out_tokens) - t1
                samples.append({"pair": p, "point": name, "arm": m, "decode_s": dt, "step_ms": dt / (out_tokens - 1) * 1e3,
                                "tok_s": b * (out_tokens - 1) / dt})
        print(f"[paired] pair {p} done", flush=True)
    lines = ["| point | aux: median ms/step (min..max) | one: median ms/step (min..max) | paired delta: median (min..max) | pairs |",
             "|---|---|---|---|---|"]
    for name, (b, _) in shapes.items():
        a = [s["step_ms"] for s in samples if s["point"] == name and s["arm"] == "aux"]
        o = [s["step_ms"] for s in samples if s["point"] == name and s["arm"] == "one"]
        d = [(y / x - 1) * 100 for x, y in zip(a, o)]
        lines.append(f"| {name} decode (batch {b}) | {statistics.median(a):.1f} ({min(a):.1f}..{max(a):.1f}) | "
                     f"{statistics.median(o):.1f} ({min(o):.1f}..{max(o):.1f}) | {statistics.median(d):+.2f}% "
                     f"({min(d):+.2f}..{max(d):+.2f}) | {len(d)} |")
    summary = ("\n".join(lines) + "\n\nDelta: one/aux - 1 of the decode step time (positive: one stream slower). Orders chosen: "
               + json.dumps(counts) + "\n")
    ok = (counts["aux"].get("MULTI_STREAM_OVERLAPPED@le256", 0) > 0 and "NO_OVERLAP@le256" not in counts["aux"]
          and counts["one"].get("NO_OVERLAP@le256", 0) > 0 and "MULTI_STREAM_OVERLAPPED@le256" not in counts["one"])
    out = Path(args.out)
    res = {"ok": ok, "samples": samples, "orders": counts, "out_tokens": out_tokens, "pairs": args.pairs, "gpu": _gpu(),
           "versions": {"vllm": vllm.__version__, "torch": torch.__version__}, "affinity": sorted(os.sched_getaffinity(0)),
           "env": {k: os.environ.get(k) for k in [*ENGINE, "CUDA_VISIBLE_DEVICES"]}}
    (out / "paired.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    (out / "summary-paired.md").write_text(summary)
    print(summary)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        result = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1",
                  "workload_fingerprint": {"workload": "shared_experts_stream.paired", "model": args.model, "pairs": args.pairs},
                  "validation": {"status": "passed" if ok else "failed",
                                 "checker": "in-process engine; every <=256-token call overlapped in aux mode and none in one mode",
                                 "detail": json.dumps(counts), "evidence": []},
                  "measurements": [{"name": f"p{s['pair']}.{s['arm']}.{s['point']}.step_ms", "value": s["step_ms"], "unit": "ms"}
                                   for s in samples],
                  "measurement_files": [os.path.relpath(p, run_dir) for p in (out / "paired.json", out / "summary-paired.md")],
                  "artifacts": []}
        (Path(run_dir) / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return 0 if ok else 1


# Each point's delta is extra time with the shared expert on the main stream: one/aux - 1 for TPOT, aux/one - 1 for tok/s.
POINTS = (("batch-1 TPOT (ms)", "b1_tpot_ms", False), ("decode b8 (tok/s)", "b8_decode_tok_s", True),
          ("decode b32 (tok/s)", "b32_decode_tok_s", True), ("decode b128 (tok/s)", "b128_decode_tok_s", True),
          ("prefill 16x2048 (tok/s)", "prefill_tok_s", True))


def summarize(out: Path) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("arm-r*.json"))]
    by = {(r["rep"], r["arm"]): r["points"] for r in rows}
    reps = sorted({r for r, a in by if (r, "one") in by and (r, "aux") in by})
    lines = ["| point | aux (vLLM default): median (min..max) | one stream: median (min..max) | delta of medians | per-rep deltas (min..max) | reps |",
             "|---|---|---|---|---|---|"]
    for name, key, higher_better in POINTS:
        a = [by[r, "aux"][key] for r in reps]
        o = [by[r, "one"][key] for r in reps]
        if not a:
            continue
        d = (lambda x, y: x / y - 1) if higher_better else (lambda x, y: y / x - 1)
        per = [d(x, y) * 100 for x, y in zip(a, o)]
        lines.append(f"| {name} | {statistics.median(a):.2f} ({min(a):.2f}..{max(a):.2f}) | {statistics.median(o):.2f} "
                     f"({min(o):.2f}..{max(o):.2f}) | {d(statistics.median(a), statistics.median(o)) * 100:+.2f}% | "
                     f"{min(per):+.2f}..{max(per):+.2f}% | {len(reps)} |")
    orders = {f"{r['arm']} r{r['rep']}": r["shared_expert_orders"] for r in rows}
    return ("\n".join(lines) + "\n\nDelta: extra time with the shared expert on the main stream (one/aux - 1 for TPOT, "
            "aux/one - 1 for tok/s); positive means one stream is slower.\n\nShared-expert orders chosen in the measured "
            "window (order@tokens per step):\n\n" + "\n".join(f"- {k}: {v}" for k, v in orders.items()) + "\n")


def run(args) -> int:
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rc = 0
    for rep in range(args.reps):
        order = list(ARMS) if rep % 2 == 0 else list(reversed(ARMS))
        for a in order:
            log = out / "logs" / f"arm-r{rep}-{a}.log"
            cmd = [sys.executable, __file__, "arm", "--arm", a, "--rep", str(rep), "--out", str(out), "--model", args.model]
            print(f"[se] rep {rep} arm {a} -> {log}", flush=True)
            t = time.perf_counter()
            with log.open("w") as f:
                r = subprocess.run(cmd, env=_arm_env(a), stdout=f, stderr=subprocess.STDOUT).returncode
            print(f"[se] rc={r} in {time.perf_counter() - t:.0f} s", flush=True)
            rc |= r
    summary = summarize(out)
    (out / "summary.md").write_text(summary)
    print(summary)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        rows = [json.loads(p.read_text()) for p in sorted(out.glob("arm-r*.json"))]
        result = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1",
                  "workload_fingerprint": {"workload": "shared_experts_stream", "model": args.model, "reps": args.reps},
                  "validation": {"status": "passed" if rc == 0 and len(rows) == 2 * args.reps else "failed",
                                 "checker": "every engine process exited 0 and wrote its points", "detail": f"rc={rc}, rows={len(rows)}",
                                 "evidence": []},
                  "measurements": [{"name": f"r{r['rep']}.{r['arm']}.{k}", "value": v, "unit": "ms" if "ms" in k else "tok/s"}
                                   for r in rows for k, v in r["points"].items()],
                  "measurement_files": [os.path.relpath(p, run_dir) for p in [*sorted(out.glob("*.json")), out / "summary.md"]],
                  "artifacts": []}
        (Path(run_dir) / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return rc


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "arm", "paired", "summarize"))
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=str(Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out"))
    args = ap.parse_args(argv)
    if args.mode == "summarize":
        print(summarize(Path(args.dirs[0] if args.dirs else args.out)))
        return 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return {"run": run, "arm": arm, "paired": paired}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
