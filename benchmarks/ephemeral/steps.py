"""One stock vLLM engine (FP8 online quantization, dummy weights at a checkpoint's shapes, FP8 KV cache) decoding at each
batch and context the KV cache holds, with a profiler window (cudaProfilerStart/Stop) over `--profile-steps` steady decode
steps of each point, for `ncu --profile-from-start off`; each window opens with torch's ``spin_kernel`` so the profile
splits by point.  Without ncu the same windows are harmless and the steps are timed.

``--layers N`` keeps the checkpoint's shapes with N decoder layers (``hf_overrides``), so batch 64 at 32k fits one card;
`attribute.py` reads per-layer and per-step bytes off the kernels and scales to the full depth.  Dummy weights move the
same bytes as real ones (decode GEMMs, attention and greedy sampling don't branch on values).  The record holds what the
design registers (each FP8 linear's weight, the KV cache) and what it doesn't (the BF16 lm_head and embedding table), so
`attribute.py` can split the step's DRAM bytes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch


def log(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def unwrap(m):
    while not isinstance(m, torch.nn.Module) and hasattr(m, "runnable"):
        m = m.runnable
    return m


class Window:
    """Opens after four pure-decode steps of the point's batch and closes `steps` steps later, recording the live tokens
    (the scheduler's computed tokens over its running requests) at both ends."""

    def __init__(self, steps: int, live_tokens) -> None:
        self.steps, self.point, self.run, self.opened, self.on = steps, None, 0, False, False
        self.live_tokens = live_tokens
        self.windows: list[dict] = []

    def begin(self, b: int, ctx: int) -> None:
        self.point, self.run, self.opened = {"batch": b, "context": ctx}, 0, False

    def __call__(self, n: int, r: int) -> None:
        if not self.steps or self.point is None:
            return
        b = self.point["batch"]
        self.run = self.run + 1 if n == r == b else 0
        if not self.opened and self.run == 4:
            torch.cuda.synchronize()
            self.point["tokens_open"] = self.live_tokens()
            torch.cuda.profiler.start()
            torch.cuda._sleep(1000)
            self.opened = self.on = True
        elif self.on and (self.run == 0 or self.run >= 4 + self.steps):
            self.end()

    def end(self) -> None:
        if self.on:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            self.point["tokens_close"] = self.live_tokens()
            self.on = False
            self.windows.append(self.point)
        self.point = None


def steady(rec: list, b: int) -> dict:
    iv = [t1 - t0 for (t0, n0, r0), (t1, n1, r1) in itertools.pairwise(rec) if n0 == r0 == b and n1 == r1 == b]
    if len(iv) < 4:
        return {"steps": len(iv)}
    iv = iv[1:]
    med = statistics.median(iv)
    return {"steps": len(iv), "step_ms": med * 1e3, "mean_ms": statistics.fmean(iv) * 1e3, "tokens_per_s": b / med}


def read_rate(gib: float = 8.0, reps: int = 5) -> float:
    """HBM read bandwidth by a max over a buffer far larger than L2 (bytes / s, best of reps)."""
    x = torch.empty(int(gib * (1 << 30)) // 2, dtype=torch.bfloat16, device="cuda").fill_(1)
    x.max()
    best = math.inf
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        x.max()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) / 1e3)
    del x
    torch.cuda.empty_cache()
    return gib * (1 << 30) / best


def budget(model, cache, hf) -> dict:
    """Bytes the step reads once (weights by kind) and the KV cache's bytes a token, from the engine's own tensors."""
    from vllm.model_executor.layers.linear import LinearBase
    linears, scales, n_lin = 0, 0, 0
    for _, m in model.named_modules():
        w = getattr(m, "weight", None)
        if not isinstance(m, LinearBase) or w is None or w.dtype != torch.float8_e4m3fn:
            continue
        linears += w.numel() * w.element_size()
        n_lin += 1
        for s in ("weight_scale", "input_scale"):
            t = getattr(m, s, None)
            if isinstance(t, torch.Tensor):
                scales += t.numel() * t.element_size()
    lm = model.lm_head.weight
    emb = model.model.embed_tokens.weight
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    kv_elem = 1 if "fp8" in str(cache.cache_dtype) else 2
    return {
        "linear_bytes": linears, "linear_scale_bytes": scales, "linears": n_lin,
        "lm_head_bytes": lm.numel() * lm.element_size(), "lm_head_dtype": str(lm.dtype),
        "embed_bytes": emb.numel() * emb.element_size(), "tied": lm.data_ptr() == emb.data_ptr(),
        "kv_bytes_per_token_layer": 2 * hf.num_key_value_heads * head_dim * kv_elem,
        "layers": hf.num_hidden_layers,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a checkpoint directory (its config and tokenizer; weights are dummy)")
    ap.add_argument("--layers", type=int, default=0, help="decoder layers (0: the checkpoint's)")
    ap.add_argument("--compiled", action="store_true", help="vLLM's default compilation (its fusion passes) instead of mode 0")
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--contexts", default="2048,8192,32768")
    ap.add_argument("--steps", type=int, default=24, help="decode steps a point")
    ap.add_argument("--profile-steps", type=int, default=0)
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--max-batched-tokens", type=int, default=16384,
                    help="the prefill chunk (70B's MLP activation over 32k tokens is 3.5 GiB, past what 0.93 leaves)")
    ap.add_argument("--read-rate", action="store_true", help="measure HBM read bandwidth after the points")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    capture = sorted(set(batches))

    import vllm
    from verity_vllm.engine.engine_profile import get_model_runner, get_scheduler
    from verity_vllm.engine.hooks import Hooks
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cc: dict = {"cudagraph_capture_sizes": capture}
    if not args.compiled:
        cc.update({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    # the warmup's dummy batch (max_num_batched_tokens) must fit max_num_seqs x max_model_len, or FlashInfer's autotune
    # reads past its buffers
    def decode_tokens(b: int, ctx: int) -> int:
        """Tokens a sequence generates: the steps, plus the chunked prefill's steps (the batch's prompts enter a chunk at
        a time, and the first ones decode meanwhile), so the point reaches `steps` pure-decode steps of its batch."""
        return args.steps + -(-b * ctx // args.max_batched_tokens) + 4

    seqs = max(64, *capture)
    max_len = max(c + decode_tokens(max(batches), c) for c in contexts) + 16
    kw = {"model": args.model, "load_format": "dummy", "quantization": "fp8", "max_num_seqs": seqs,
          "enable_prefix_caching": False, "seed": 0, "max_model_len": max_len,
          "max_num_batched_tokens": min(args.max_batched_tokens, seqs * max_len), "gpu_memory_utilization": args.mem,
          "kv_cache_dtype": args.kv_cache_dtype, "enforce_eager": False, "compilation_config": cc}
    if args.layers:
        kw["hf_overrides"] = {"num_hidden_layers": args.layers}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    log(f"engine up in {time.perf_counter() - t0:.1f} s")
    runner = get_model_runner(llm)
    model = unwrap(runner.model)
    vc = llm.llm_engine.vllm_config
    hf, cache = vc.model_config.hf_config, vc.cache_config
    kv_tokens = int(cache.num_gpu_blocks) * int(cache.block_size)
    attn = sorted({type(getattr(m, "attn_backend", None)).__name__ for _, m in model.named_modules()
                   if getattr(m, "attn_backend", None) is not None})
    rec: list = []
    sched = get_scheduler(llm)
    window = Window(args.profile_steps, lambda: sum(int(r.num_computed_tokens) for r in sched.running))
    hooks = Hooks(tag="ra-ephemeral")

    def make(orig):
        def update_from_output(so, mro):
            rec.append((time.perf_counter(), so.total_num_scheduled_tokens, len(so.num_scheduled_tokens)))
            window(so.total_num_scheduled_tokens, len(so.num_scheduled_tokens))
            return orig(so, mro)
        return update_from_output
    hooks.wrap(sched, "update_from_output", make)

    props = torch.cuda.get_device_properties(0)
    pc = getattr(vc.compilation_config, "pass_config", None)
    result = {
        "model": args.model, "layers_override": args.layers, "compiled": args.compiled, "vllm": vllm.__version__,
        "torch": torch.__version__, "gpu": props.name, "sms": props.multi_processor_count,
        "l2_bytes": getattr(props, "L2_cache_size", None), "kv_tokens": kv_tokens, "block_size": int(cache.block_size),
        "kv_cache_dtype": str(cache.cache_dtype), "attention": attn, "compilation": cc,
        "pass_config": str(pc) if pc is not None else None, "async_scheduling": getattr(vc.scheduler_config,
                                                                                         "async_scheduling", None),
        "budget": budget(model, cache, hf),
        "model_shape": {"layers": hf.num_hidden_layers, "hidden": hf.hidden_size, "heads": hf.num_attention_heads,
                        "kv_heads": hf.num_key_value_heads, "intermediate": hf.intermediate_size, "vocab": hf.vocab_size},
        "points": [], "profile_steps": args.profile_steps, "profiled": window.windows,
    }
    out = Path(args.out)

    def save() -> None:
        out.write_text(json.dumps(result, indent=1))
    save()
    log("budget", json.dumps(result["budget"]))

    rng = random.Random(0)
    warm = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(512)])]
    llm.generate(warm, SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, detokenize=False), use_tqdm=False)
    for ctx in contexts:
        for b in batches:
            mt = decode_tokens(b, ctx)
            need = b * (ctx + mt + 16)
            point = {"batch": b, "context": ctx, "kv_needed": need}
            if need > kv_tokens:
                point["skipped"] = f"KV cache holds {kv_tokens} tokens, the point needs {need}"
                result["points"].append(point)
                save()
                continue
            prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(ctx)]) for _ in range(b)]
            rec.clear()
            window.begin(b, ctx)
            t = time.perf_counter()
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True, detokenize=False),
                         use_tqdm=False)
            window.end()
            point.update(steady(rec, b), wall_s=time.perf_counter() - t, max_tokens=mt, updates=len(rec))
            result["points"].append(point)
            save()
            log("point", json.dumps(point))
    if args.read_rate:
        result["read_bytes_per_s"] = read_rate()
        log(f"read rate {result['read_bytes_per_s'] / 1e9:.1f} GB/s")
    hooks.uninstall()
    save()


if __name__ == "__main__":
    main()
