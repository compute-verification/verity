"""Llama-3.1-8B served by vLLM on one GPU (in process, FP8 online quantization, full CUDA graphs for decode) under a
bursty arrival trace, with the filler top-up (`topup.cu`) beside it: each window's registered reads against R·T, and
what the top-up costs the serving. Writes `serve-<arm>.json` and `serve.log` under `--out`.

Pool sizings (one per process, `--pool`), each with a static filler buffer for the card's rest down to `--headroom-gib`,
the top-up reading the pool's tensors and the buffer:
  large   vLLM's pool at `--mem` of the card, its free pages the filler
  sized   the pool at `--kv-gib` (vLLM's `kv_cache_memory_bytes`), the buffer most of the filler

Top-up policies, each over the same trace in the same engine (`--policies`):
  none      no top-up
  idle      the top-up paused while the engine has work, running at full grid when it has none
  throttle  the idle policy, plus a CTA cap (`--throttle-ctas`) while the engine has work
  full      the top-up at full grid throughout (the interference bound)
  gemm      no top-up thread: while the engine has no work, FP8 GEMMs at M = 16 over the pool and the buffer in 64 MiB
            slices, 4 a sync, until the next arrival (the reader a receipt bound to a PoUW tile needs, without its
            sketch; once a run, like none)

Each policy runs in each receipt mode of `--modes` (read, b3, sk16: `topup.cu`'s). A window's registered bytes are
the steps' reads of registered data inside it (the weights' bytes a step, every linear and the LM head, and each
scheduled request's context of KV, 128 KiB a token: its pages, a lower bound on attention's reads) plus the filler
bytes the top-up read inside it, its chunks on the GPU's clock placed on the host's by one sync; R comes from
`bench.py`'s read (`--R`). The trace is generated from `--seed` (bursts of requests at exponential gaps, each burst's
requests at once), the same for every policy; a step's time is the host's between consecutive scheduler updates, and
each model step's interval on the GPU is recorded beside it (`gpu_steps`: events before its forward and after its
sampling). `--shared-prefix` puts one seeded prefix before every prompt, with prefix caching on; `--async-scheduling`
sets vLLM's, which the record states either way.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import MODES, RUN_SEED, Topup  # noqa: E402

KV_TOKEN_BYTES = 32 * 2 * 8 * 128 * 2  # Llama-3.1-8B, bf16 KV
CAPTURE = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
GEMM_K, GEMM_SLICE, GEMM_GROUP = 4096, 64 << 20, 4


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


LOGF = None


def trace(seed: int, seconds: float, burst_gap_s: float, burst_mean: int, prompts, outputs) -> list[dict]:
    """Bursts at exponential gaps (mean `burst_gap_s`), each of 1 + Poisson(`burst_mean` − 1) requests arriving
    together, prompt and output lengths drawn from the lists."""
    rng = random.Random(seed)
    t, out = rng.expovariate(1 / burst_gap_s) * 0.25, []
    while t < seconds:
        n = 1
        lam = burst_mean - 1
        # Knuth's Poisson
        L, k, p = pow(2.718281828459045, -lam), 0, 1.0
        while True:
            p *= rng.random()
            if p <= L:
                break
            k += 1
        n += k
        for _ in range(n):
            out.append({"id": f"r{len(out)}", "t": t, "prompt": rng.choice(prompts), "output": rng.choice(outputs),
                        "tokens_seed": rng.randrange(1 << 30)})
        t += rng.expovariate(1 / burst_gap_s)
    return out


def windows(steps: list[dict], idle: list[tuple[float, float]], recs, offset_ms: float, t0: float, t1: float,
            width: float, R: float, weight_bytes: int) -> dict:
    """Each window of `width` seconds in [t0, t1): registered bytes (steps' pro rata, filler's pro rata) over R·T."""
    n = int((t1 - t0) // width)
    reg = [0.0] * n
    fil = [0.0] * n
    busy = [0.0] * n

    def spread(a: float, b: float, amount: float, into: list[float]) -> None:
        if b <= a:
            return
        i0, i1 = int((a - t0) // width), int((b - t0) // width)
        for i in range(max(0, i0), min(n - 1, i1) + 1):
            lo, hi = t0 + i * width, t0 + (i + 1) * width
            ov = min(b, hi) - max(a, lo)
            if ov > 0:
                into[i] += amount * ov / (b - a)

    for s in steps:
        spread(s["t0"], s["t1"], weight_bytes + s["kv_tokens"] * KV_TOKEN_BYTES, reg)
        spread(s["t0"], s["t1"], s["t1"] - s["t0"], busy)
    for a, b, nbytes, *_ in recs:
        ha, hb = (a - offset_ms) / 1e3, (b - offset_ms) / 1e3
        spread(ha, hb, nbytes, fil)
    frac = [(reg[i] + fil[i]) / (R * 1e9 * width) for i in range(n)]
    if not frac:
        return {"width_s": width, "windows": 0}
    q = sorted(frac)
    return {"width_s": width, "windows": n, "mean": statistics.fmean(frac), "min": q[0], "p10": q[len(q) // 10],
            "median": statistics.median(frac), "steps_share": statistics.fmean(busy) / width,
            "filler_share": sum(fil) / max(1.0, sum(reg) + sum(fil)), "fractions": [round(v, 4) for v in frac]}


def main() -> int:
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topup-lib", required=True)
    ap.add_argument("--kv-lib", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="/workspace/pouw/mvp-e2e/models/llama-3.1-8b-instruct")
    ap.add_argument("--pool", choices=("large", "sized"), default="large")
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--kv-gib", type=float, default=12.0)
    ap.add_argument("--headroom-gib", type=float, default=6.0)
    ap.add_argument("--policies", type=lambda s: s.split(","), default=["none", "idle", "throttle", "full", "gemm"])
    ap.add_argument("--modes", type=lambda s: s.split(","), default=["sk16"])
    ap.add_argument("--throttle-ctas", type=int, default=32)
    ap.add_argument("--R", type=float, default=1540.0)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--burst-gap-s", type=float, default=3.0)
    ap.add_argument("--burst-mean", type=int, default=12)
    ap.add_argument("--prompts", type=lambda s: [int(v) for v in s.split(",")], default=[256, 1024, 2048])
    ap.add_argument("--outputs", type=lambda s: [int(v) for v in s.split(",")], default=[64, 256, 512])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--async-scheduling", choices=("default", "on", "off"), default="default")
    ap.add_argument("--shared-prefix", type=int, default=0,
                    help="tokens of one seeded prefix before every prompt, with prefix caching on (0: none, caching off)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    LOGF = (a.out / "serve.log").open("a")

    import torch
    import vllm
    from verity_vllm.engine.engine_profile import get_model_runner, get_scheduler
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = {"model": a.model, "quantization": "fp8", "max_num_seqs": max(CAPTURE),
          "enable_prefix_caching": a.shared_prefix > 0, "seed": 0,
          "max_model_len": a.shared_prefix + max(a.prompts) + max(a.outputs) + 64,
          "max_num_batched_tokens": a.max_batched_tokens, "enforce_eager": False,
          "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": CAPTURE}}
    if a.async_scheduling != "default":
        kw["async_scheduling"] = a.async_scheduling == "on"
    if a.pool == "large":
        kw["gpu_memory_utilization"] = a.mem
    else:
        kw["kv_cache_memory_bytes"] = int(a.kv_gib * (1 << 30))
    t = time.perf_counter()
    llm = LLM(**kw)
    log(f"engine up in {time.perf_counter() - t:.1f} s ({a.pool})")
    eng = llm.llm_engine
    runner = get_model_runner(llm)
    sched = get_scheduler(llm)
    cache = eng.vllm_config.cache_config
    weight_bytes = sum(p.numel() * p.element_size() for n, p in runner.model.named_parameters()
                       if "embed_tokens" not in n)
    kv_tensors = []
    for kc in runner.kv_caches:
        for tns in (kc if isinstance(kc, (list, tuple)) else [kc]):
            kv_tensors.append(tns)
    seen, regions = set(), []
    for tns in kv_tensors:
        if tns.data_ptr() not in seen:
            seen.add(tns.data_ptr())
            regions.append((tns.data_ptr(), tns.numel() * tns.element_size()))
    pool_bytes = sum(n for _, n in regions)
    # Both sizings register the card down to `--headroom-gib`: the top-up reads the pool and a static buffer for the rest.
    free, card = torch.cuda.mem_get_info()
    static = max(0, free - int(a.headroom_gib * (1 << 30)))
    filler = torch.zeros(static, dtype=torch.uint8, device="cuda") if static >= (1 << 30) else None
    if filler is not None:
        regions.append((filler.data_ptr(), filler.numel()))
    res = {"format": "verity-kv-filler/serve/v0", "args": {k: v for k, v in vars(a).items() if k != "out"},
           "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
           "card_bytes": card, "weight_bytes_per_step": weight_bytes,
           "weight_bytes_resident": sum(p.numel() * p.element_size() for p in runner.model.parameters()),
           "kv_pool_bytes": pool_bytes, "static_filler_bytes": static if filler is not None else 0,
           "kv_blocks": int(cache.num_gpu_blocks), "block_size": int(cache.block_size),
           "filler_bytes": sum(n for _, n in regions), "topup_regions": len(regions), "runs": [],
           "async_scheduling": getattr(eng.vllm_config.scheduler_config, "async_scheduling", None),
           "batch_queue_size": getattr(getattr(getattr(eng, "engine_core", None), "engine_core", None),
                                       "batch_queue_size", None)}
    res["unregistered_bytes"] = card - res["weight_bytes_resident"] - pool_bytes - res["static_filler_bytes"]
    log(f"weights read a step {weight_bytes / 1e9:.2f} GB; pool {pool_bytes / 1e9:.1f} GB in {len(kv_tensors)} tensors;"
        f" static filler {res['static_filler_bytes'] / 1e9:.1f} GB; top-up over {res['filler_bytes'] / 1e9:.1f} GB;"
        f" card {card / 1e9:.1f} GB, {res['unregistered_bytes'] / 1e9:.1f} GB unregistered; async scheduling "
        f"{res['async_scheduling']}, batch queue {res['batch_queue_size']}")

    out_path = a.out / f"serve-{a.pool}.json"

    def save() -> None:
        out_path.write_text(json.dumps(res, indent=1) + "\n")

    tu = Topup(a.topup_lib)
    tu.so.tu_pause.argtypes = [ctypes.c_int]
    tu.so.tu_throttle.argtypes = [ctypes.c_int]
    least, _ = torch.cuda.Stream.priority_range()
    least = max(least, _)
    vocab = eng.vllm_config.model_config.hf_config.vocab_size

    # the GEMM reader: every contiguous tensor of the top-up's regions in FP8 slices of GEMM_K columns
    gemm_tensors = [t for t in kv_tensors if t.is_contiguous()] + ([filler] if filler is not None else [])
    slices = [(t, off) for t in gemm_tensors for off in range(0, t.numel() * t.element_size() - GEMM_SLICE + 1,
                                                              GEMM_SLICE)]
    gx = torch.zeros(16, GEMM_K, device="cuda", dtype=torch.float8_e4m3fn)
    gone = torch.ones((), device="cuda", dtype=torch.float32)
    gstream = torch.cuda.Stream()
    gpos = [0]

    def gemm_idle(until: float, out: list) -> None:
        while until - time.perf_counter() > 0.002:
            t0 = time.perf_counter()
            with torch.cuda.stream(gstream):
                for _ in range(GEMM_GROUP):
                    t, off = slices[gpos[0] % len(slices)]
                    gpos[0] += 1
                    w = t.view(-1).view(torch.uint8)[off:off + GEMM_SLICE].view(torch.float8_e4m3fn)
                    torch._scaled_mm(gx, w.view(-1, GEMM_K).t(), scale_a=gone, scale_b=gone, out_dtype=torch.bfloat16)
            gstream.synchronize()
            out.append((t0 * 1e3, time.perf_counter() * 1e3, GEMM_GROUP * GEMM_SLICE))

    log(f"GEMM reader: {len(slices)} slices of {GEMM_SLICE >> 20} MiB over {len(gemm_tensors)} tensors")

    # the scheduler's view of each step
    cur: dict = {}
    orig = sched.update_from_output

    def update_from_output(so, mro):
        now = time.perf_counter()
        t0 = max(cur.get("call_t0", now), cur.get("last_t1", 0.0))
        cur["last_t1"] = now
        ns = so.num_scheduled_tokens
        kv_tokens = 0
        prefill = 0
        for rid, n in ns.items():
            r = sched.requests.get(rid)
            if r is None:
                continue
            kv_tokens += int(r.num_computed_tokens)
            # a decode request schedules one token; with the batch queue, num_computed_tokens is a step ahead
            if n > 1:
                prefill += n
        try:
            free = int(sched.kv_cache_manager.block_pool.get_num_free_blocks())
        except AttributeError:
            free = -1
        cur.setdefault("steps", []).append({"t0": t0, "t1": now, "tokens": int(so.total_num_scheduled_tokens), "reqs": len(ns),
                             "prefill_tokens": prefill, "kv_tokens": kv_tokens, "free_blocks": free,
                             "preempted": len(getattr(so, "preempted_req_ids", None) or ())})
        return orig(so, mro)

    sched.update_from_output = update_from_output

    # each model step's interval on the GPU: an event before its forward and one after its sampling, on the runner's
    # stream (execute_model returns None when sample_tokens finishes the step)
    ex_orig, st_orig = runner.execute_model, getattr(runner, "sample_tokens", None)

    def mark() -> "torch.cuda.Event":
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def execute_model(*args, **kwargs):
        e0 = mark()
        out = ex_orig(*args, **kwargs)
        if out is None and st_orig is not None:
            cur["gpu_open"] = e0
        elif "gpu" in cur:
            cur["gpu"].append((e0, mark()))
        return out

    def sample_tokens(*args, **kwargs):
        out = st_orig(*args, **kwargs)
        e0 = cur.pop("gpu_open", None)
        if e0 is not None and "gpu" in cur:
            cur["gpu"].append((e0, mark()))
        return out

    runner.execute_model = execute_model
    if st_orig is not None:
        runner.sample_tokens = sample_tokens

    warm = SamplingParams(max_tokens=8, ignore_eos=True)
    llm.generate([TokensPrompt(prompt_token_ids=[1000 + i for i in range(64)])], warm, use_tqdm=False)
    prefix = [random.Random(a.seed + 7).randrange(1000, vocab - 1000) for _ in range(a.shared_prefix)]

    reqs = trace(a.seed, a.seconds, a.burst_gap_s, a.burst_mean, a.prompts, a.outputs)
    res["trace"] = {"requests": len(reqs), "prompt_tokens": sum(a.shared_prefix + r["prompt"] for r in reqs),
                    "output_tokens": sum(r["output"] for r in reqs), "shared_prefix": a.shared_prefix}
    log(f"trace: {res['trace']}")
    for mode in a.modes:
        for policy in a.policies:
            if policy in ("none", "gemm") and mode != a.modes[0]:
                continue
            cur.clear()
            cur["steps"], cur["gpu"] = [], []
            # a step's interval runs from its engine.step call, or the previous step's update if later, to its update
            torch.cuda.synchronize()
            offset_ms = 0.0
            grecs: list[tuple[float, float, int]] = []
            if policy not in ("none", "gemm"):
                tu.start(regions, MODES[mode], a.chunk_mib << 20, 3, least, RUN_SEED, ctas=0)
                if policy in ("idle", "throttle"):
                    tu.so.tu_pause(1)
                g = tu.now(0)
                offset_ms = g - time.perf_counter() * 1e3
            pending = sorted(reqs, key=lambda r: r["t"])
            k = 0
            torch.cuda.synchronize()
            base = mark()
            base.synchronize()
            t_start = time.perf_counter()
            first: dict[str, float] = {}
            done: dict[str, float] = {}
            idle: list[tuple[float, float]] = []
            while k < len(pending) or eng.has_unfinished_requests():
                now = time.perf_counter() - t_start
                while k < len(pending) and pending[k]["t"] <= now:
                    r = pending[k]
                    rr = random.Random(r["tokens_seed"])
                    ids = prefix + [rr.randrange(1000, vocab - 1000) for _ in range(r["prompt"])]
                    eng.add_request(f"{policy}-{mode}-{r['id']}", TokensPrompt(prompt_token_ids=ids),
                                    SamplingParams(max_tokens=r["output"], ignore_eos=True, temperature=0.0))
                    k += 1
                if eng.has_unfinished_requests():
                    if policy == "idle":
                        tu.so.tu_pause(1)
                    elif policy == "throttle":
                        tu.so.tu_throttle(a.throttle_ctas)
                        tu.so.tu_pause(0)
                    cur["call_t0"] = time.perf_counter()
                    outs = eng.step()
                    s1 = time.perf_counter()
                    for o in outs:
                        if o.outputs and len(o.outputs[0].token_ids) and o.request_id not in first:
                            first[o.request_id] = s1
                        if o.finished:
                            done[o.request_id] = s1
                else:
                    if policy in ("idle", "throttle"):
                        tu.so.tu_throttle(0)
                        tu.so.tu_pause(0)
                    i0 = time.perf_counter()
                    nxt = pending[k]["t"] + t_start if k < len(pending) else i0
                    if policy == "gemm":
                        gemm_idle(min(i0 + 0.05, nxt), grecs)
                    else:
                        time.sleep(max(0.0, min(0.05, nxt - i0)))
                    idle.append((i0, time.perf_counter()))
            t_end = time.perf_counter()
            torch.cuda.synchronize()
            gpu_steps = [[round(base.elapsed_time(e0) / 1e3, 6), round(base.elapsed_time(e1) / 1e3, 6)]
                         for e0, e1 in cur["gpu"]]
            gpu_busy, edge = 0.0, 0.0
            for g0, g1 in gpu_steps:
                gpu_busy += max(0.0, g1 - max(g0, edge))
                edge = max(edge, g1)
            recs = grecs
            if policy not in ("none", "gemm"):
                tu.stop()
                recs = tu.log()
            steps = cur["steps"]
            by_id = {f"{policy}-{mode}-{r['id']}": r for r in reqs}
            ttft = [first[i] - (by_id[i]["t"] + t_start) for i in first]
            tpot = [(done[i] - first[i]) / max(1, by_id[i]["output"] - 1) for i in done if i in first]
            dec = [s["t1"] - s["t0"] for s in steps if s["prefill_tokens"] == 0 and s["reqs"] > 0]
            run = {"policy": policy, "mode": mode, "seconds": t_end - t_start, "steps": len(steps),
                   "requests_done": len(done), "ttft_s_median": statistics.median(ttft) if ttft else None,
                   "ttft_s_p90": sorted(ttft)[int(0.9 * (len(ttft) - 1))] if ttft else None,
                   "tpot_ms_median": statistics.median(tpot) * 1e3 if tpot else None,
                   "decode_step_ms_median": statistics.median(dec) * 1e3 if dec else None,
                   "idle_s": sum(b - a for a, b in idle), "filler_bytes": sum(r[2] for r in recs),
                   "preemptions": sum(s["preempted"] for s in steps),
                   "free_blocks_min": min((s["free_blocks"] for s in steps), default=None),
                   "windows": [windows(steps, idle, recs, offset_ms, t_start, t_end, w, a.R, weight_bytes)
                               for w in (0.25, 1.0, 5.0)],
                   "step_records": [{k2: (round(v - t_start, 5) if k2 in ("t0", "t1") else v) for k2, v in s.items()}
                                    for s in steps],
                   "gpu_busy_s": gpu_busy, "gpu_steps": gpu_steps}
            res["runs"].append(run)
            w1 = run["windows"][1]
            log(f"{policy}/{mode}: {run['seconds']:.1f} s, {len(steps)} steps, TTFT {run['ttft_s_median']:.3f} s, TPOT "
                f"{run['tpot_ms_median']:.2f} ms, decode step {run['decode_step_ms_median']:.2f} ms, idle "
                f"{run['idle_s']:.1f} s, filler {run['filler_bytes'] / 1e9:.0f} GB; 1 s windows mean "
                f"{w1.get('mean', 0):.3f} p10 {w1.get('p10', 0):.3f} min {w1.get('min', 0):.3f} of R; GPU busy "
                f"{gpu_busy:.1f} s in {len(gpu_steps)} model steps")
            save()
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
