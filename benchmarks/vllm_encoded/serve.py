"""One arm of the encoded-weights decode benchmark: Llama-3.1-8B served by vLLM (in process, FP8 online quantization, full
CUDA graphs for decode), timed at each decode batch and context length.

Arms:
  stock     vLLM's own FP8 linears (CUTLASS on sm_120), same engine settings
  plain     every linear through `ve.cu`'s fused GEMM over plain FP8 fragments (k = 0): the native kernel of this harness
  enc       every linear through the fused GEMM over codec words only (k rounds of Sloth at p = 2^61 - 1); the plaintext
            weight is freed before vLLM profiles memory, so decode runs with no plaintext copy of any linear in HBM

The plain and encoded arms install at the end of the model runner's `load_model` (before memory profiling and graph
capture), through `verity_vllm.engine.hooks`.  Before anything is timed, each linear passes its gates: every GEMM form
gives the same partial sums bit for bit (encoded against plain at every capture size), the plain sums agree with a float
product, the encoded store decodes to the payloads on the device and in `codec.py`, and (encoded arm) no FP8 or plain
fragment tensor of a linear's size is left anywhere in the process.  Each arm then generates fixed prompts greedily
(`gate`), which `compare.py` requires to be bit-identical between plain and encoded arms (token ids and log-probs).

A decode step's time is the interval between two consecutive scheduler updates that both scheduled exactly B decode tokens
for B requests (`update_from_output`), the median over the point's steady decode; prefill is not timed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kernels as K
import layout
import torch

CAPTURE = [1, 4, 8, 16, 32]
SEED = hashlib.sha512(b"ra-vllm-encoded/key/v0").digest()[:32]
SPIN = 20_000_000
GATE_PROMPTS = [
    "The capital of France is",
    "Write a haiku about memory bandwidth.",
    "def fibonacci(n):",
    "Explain why the sky is blue in one sentence.",
]


def log(*a) -> None:
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *a, flush=True)


def unwrap(m):
    while not isinstance(m, torch.nn.Module) and hasattr(m, "runnable"):
        m = m.runnable
    return m


def fp8_linears(model) -> list[tuple[str, torch.nn.Module]]:
    from vllm.model_executor.layers.linear import LinearBase
    out = []
    for name, m in model.named_modules():
        if isinstance(m, LinearBase) and getattr(m, "weight", None) is not None and m.weight.dtype == torch.float8_e4m3fn:
            out.append((name, m))
    return out


def rand_fp8(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    return (torch.randn((m, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


class Install:
    """The plain or encoded arm's linears, built inside the runner's `load_model`."""

    def __init__(self, args) -> None:
        self.args = args
        self.record: dict = {}
        self.linears: list[tuple[str, K.Linear]] = []
        self.hooks = None

    def __call__(self, runner) -> None:
        from verity_vllm.engine.hooks import Hooks
        args = self.args
        t0 = time.perf_counter()
        model = unwrap(runner.model)
        layers = fp8_linears(model)
        if not layers:
            raise RuntimeError("no FP8 linear layers to install over")
        lib = K.Lib(args.lib)
        k = args.k if args.arm == "enc" else 0
        if k:
            lib.set_key(SEED, k)
        torch.cuda.synchronize()
        alloc0 = torch.cuda.memory_allocated()
        gen = torch.Generator(device="cuda")
        cands: dict[tuple[int, int], list[K.Config]] = {}
        gates = {"layers": 0, "configs_checked": 0, "python_words": 0, "max_rel_err": 0.0}
        base, plain_bytes, store_bytes, encode_s = 0, 0, 0, 0.0
        self.hooks = Hooks(tag="ra-vllm-encoded")
        for idx, (name, layer) in enumerate(layers):
            w = layer.weight.t()
            n, kd = w.shape
            if not w.is_contiguous() or (n, kd) != (layer.output_size_per_partition, layer.input_size_per_partition):
                raise RuntimeError(f"{name}: weight {tuple(layer.weight.shape)} is not the (K, N) view of an N x K matrix")
            ws = layer.weight_scale
            if ws.numel() != 1:
                raise RuntimeError(f"{name}: weight scale {tuple(ws.shape)} is not per tensor")
            plain = K.PlainLinear(lib, w, ws)
            if k:
                torch.cuda.synchronize()
                te = time.perf_counter()
                lin = K.EncodedLinear(lib, w, ws, k, base)
                torch.cuda.synchronize()
                encode_s += time.perf_counter() - te
            else:
                lin = plain
            nch = kd // layout.TASK_COLS
            gen.manual_seed(1000 + idx)
            for m in CAPTURE:
                x8 = rand_fp8(m, kd, gen)
                ref = plain.gemm(x8, K.Config(0))
                for kk, op in ((0, plain), (k, lin)) if k else ((0, plain),):
                    for cfg in cands.setdefault((kk, m, nch), K.candidates(lib, kk, m, nch)):
                        if not bits_equal(op.gemm(x8, cfg), ref):
                            raise RuntimeError(f"gate: {name} k={kk} {cfg.name} M={m}: partial sums differ from plain simple")
                        gates["configs_checked"] += 1
                if m == 4:
                    exact = x8.float() @ w.float().t()
                    err = (ref.sum(0) - exact).abs().max().item() / max(exact.abs().max().item(), 1e-30)
                    gates["max_rel_err"] = max(gates["max_rel_err"], err)
                    if err > 1e-3:
                        raise RuntimeError(f"gate: {name}: fused sums are {err:.2e} off the float product")
            if k:
                if not torch.equal(lin.decoded(), lin.payloads):
                    raise RuntimeError(f"gate: {name}: the store does not decode to its payloads")
                if not torch.equal(lin.dense().view(torch.uint8), w.view(torch.uint8)):
                    raise RuntimeError(f"gate: {name}: the decoded weight is not the weight")
                codec = K._codec()
                rng = random.Random(idx)
                store = lin.store.view(torch.int64)
                for j in [rng.randrange(store.numel()) for _ in range(args.python_words)]:
                    if codec.decode_word(int(store[j]), base + j, lib.key, lib.bits) != int(lin.payloads[j]):
                        raise RuntimeError(f"gate: {name}: codec.py decodes word {base + j} otherwise")
                gates["python_words"] += args.python_words
                lin.drop_payloads()
                base += lin.store.numel()
                del plain
            elif not torch.equal(lin.dense().view(torch.uint8), w.view(torch.uint8)):
                raise RuntimeError(f"gate: {name}: the plain fragments are not the weight")
            plain_bytes += w.numel()
            store_bytes += lin.nbytes
            lin.name = name
            self.linears.append((name, lin))
            del w
            layer._parameters["weight"] = torch.nn.Parameter(
                torch.empty(0, dtype=torch.float8_e4m3fn, device=layer.weight_scale.device), requires_grad=False)
            quant = layer.quant_method.fp8_linear.quant_fp8
            self.hooks.wrap(layer.quant_method, "apply", lambda orig, lin=lin, quant=quant:
                            (lambda lyr, x, bias=None: lin(lyr, x, bias, orig, lambda t: quant(t, None, None))))
            gates["layers"] += 1
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc1 = torch.cuda.memory_allocated()
        selection = self.select(lib, k)
        scan = plaintext_scan(min(lin.n * lin.kd for _, lin in self.linears))
        if k and (scan["fp8"] or scan["uint8"]):
            raise RuntimeError(f"gate: plaintext left in the process: {scan}")
        self.record = {
            "arm": args.arm, "k": k, "lib": Path(args.lib).name, "bits": lib.bits, "kmax": lib.kmax, "layers": len(layers),
            "plain_bytes": plain_bytes, "store_bytes": store_bytes, "store_over_plain": store_bytes / plain_bytes,
            "words": base, "encode_s": encode_s, "allocated_before": alloc0, "allocated_after": alloc1,
            "allocated_delta": alloc1 - alloc0, "plaintext_scan": scan, "gates": gates, "selection": selection,
            "fp8_kernel": type(layers[0][1].quant_method.fp8_linear).__name__,
            "activation_quant": str(layers[0][1].quant_method.activation_quant_key),
            "install_s": time.perf_counter() - t0,
        }
        log("installed", json.dumps({k_: v for k_, v in self.record.items() if k_ != "selection"}))

    def select(self, lib: K.Lib, k: int) -> list[dict]:
        """Each shape's GEMM config at each capture size: the fastest candidate over all of the shape's layers in turn (so
        the weights stream from HBM, not L2), queued behind a spin so host launches don't set the time."""
        groups: dict[tuple[int, int], list[K.Linear]] = {}
        for _, lin in self.linears:
            groups.setdefault((lin.n, lin.kd), []).append(lin)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(7)
        out = []
        for (n, kd), lins in groups.items():
            nch = kd // layout.TASK_COLS
            for m in CAPTURE:
                x8 = rand_fp8(m, kd, gen)
                times = {}
                for cfg in K.candidates(lib, k, m, nch):
                    times[cfg] = time_config(lib, lins, x8, cfg, self.args.select_reps)
                best = min(times, key=times.get)
                for lin in lins:
                    lin.configs[m] = best
                out.append({"n": n, "k": kd, "m": m, "layers": len(lins), "best": best.name, "groups": best.groups,
                            "us": {c.name: round(t, 2) for c, t in times.items()}})
                log(f"select n={n} k={kd} m={m}: {best.name} {times[best]:.1f} us", {c.name: round(t, 1) for c, t in times.items()})
        return out


def time_config(lib: K.Lib, lins: list[K.Linear], x8: torch.Tensor, cfg: K.Config, reps: int) -> float:
    for lin in lins:
        lin.gemm(x8, cfg)
    s = K.stream()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lib.call("wc_spin", SPIN, s)
    e0.record()
    for _ in range(reps):
        for lin in lins:
            lin.gemm(x8, cfg)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (reps * len(lins))


def plaintext_scan(min_elems: int) -> dict:
    """CUDA tensors in the process at least one linear's size, by dtype: FP8 (a weight) and uint8 (plain fragments)."""
    gc.collect()
    seen, out = set(), {"fp8": 0, "uint8": 0, "fp8_bytes": 0, "uint8_bytes": 0, "held_by": []}
    for o in gc.get_objects():
        if not isinstance(o, torch.Tensor) or o.is_meta or not o.is_cuda or o.numel() < min_elems:
            continue
        key = (o.untyped_storage().data_ptr(), o.dtype)
        if key in seen or o.dtype not in (torch.float8_e4m3fn, torch.uint8):
            continue
        seen.add(key)
        kind = "fp8" if o.dtype == torch.float8_e4m3fn else "uint8"
        out[kind] += 1
        out[kind + "_bytes"] += o.untyped_storage().nbytes()
        if len(out["held_by"]) < 4:
            out["held_by"].append([kind, list(o.shape), sorted({type(r).__name__ for r in gc.get_referrers(o)})[:6]])
    return out


def hook_runner(hooks, install: Install) -> None:
    import importlib
    for mod in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
        try:
            cls = importlib.import_module(mod).GPUModelRunner
        except (ImportError, AttributeError):
            continue

        def make(orig):
            def load_model(self, *a, **kw):
                r = orig(self, *a, **kw)
                install(self)
                return r
            return load_model
        hooks.wrap(cls, "load_model", make)


def recorder(hooks, sched, rec: list) -> None:
    def make(orig):
        def update_from_output(so, mro):
            rec.append((time.perf_counter(), so.total_num_scheduled_tokens, len(so.num_scheduled_tokens)))
            return orig(so, mro)
        return update_from_output
    hooks.wrap(sched, "update_from_output", make)


def steady(rec: list, b: int) -> dict:
    """The point's decode steps: intervals between consecutive updates that each scheduled b tokens for b requests."""
    iv = [t1 - t0 for (t0, n0, r0), (t1, n1, r1) in itertools.pairwise(rec) if n0 == r0 == b and n1 == r1 == b]
    if len(iv) < 4:
        return {"steps": len(iv)}
    iv = iv[1:]
    med = statistics.median(iv)
    q = statistics.quantiles(iv, n=10)
    return {"steps": len(iv), "step_ms": med * 1e3, "mean_ms": statistics.fmean(iv) * 1e3, "p10_ms": q[0] * 1e3,
            "p90_ms": q[-1] * 1e3, "tokens_per_s": b / med}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("stock", "stock-compiled", "plain", "enc"), required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lib", help="a build of ve.cu (plain and enc arms)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--contexts", default="512,2048,8192")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--mem", type=float, default=0.75)
    ap.add_argument("--max-batched-tokens", type=int, default=16384)
    ap.add_argument("--select-reps", type=int, default=3)
    ap.add_argument("--python-words", type=int, default=16)
    ap.add_argument("--gate-tokens", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    import vllm
    from verity_vllm.engine.engine_profile import get_model_runner, get_scheduler
    from verity_vllm.engine.hooks import Hooks
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    hooks = Hooks(tag="ra-vllm-encoded")
    install = Install(args)
    if args.arm in ("plain", "enc"):
        if not args.lib:
            ap.error("--lib is required for the plain and enc arms")
        hook_runner(hooks, install)
    max_tokens_cap = args.steps + math.ceil(max(batches) * max(contexts) / args.max_batched_tokens) + 8
    cc = {"cudagraph_capture_sizes": CAPTURE}
    if args.arm != "stock-compiled":
        cc.update({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    kw = {"model": args.model, "quantization": "fp8", "max_num_seqs": max(batches), "enable_prefix_caching": False, "seed": 0,
          "max_model_len": max(contexts) + max_tokens_cap + 64, "max_num_batched_tokens": args.max_batched_tokens,
          "gpu_memory_utilization": args.mem, "enforce_eager": False, "compilation_config": cc}
    if args.revision:
        kw["revision"] = args.revision
    t0 = time.perf_counter()
    llm = LLM(**kw)
    startup_s = time.perf_counter() - t0
    log(f"engine up in {startup_s:.1f} s")
    runner = get_model_runner(llm)
    model = unwrap(runner.model)
    hf = llm.llm_engine.vllm_config.model_config.hf_config
    cache = llm.llm_engine.vllm_config.cache_config
    kv_tokens = int(cache.num_gpu_blocks) * int(cache.block_size)
    attn = sorted({type(getattr(m, "attn_backend", None)).__name__ for _, m in model.named_modules()
                   if getattr(m, "attn_backend", None) is not None})
    rec: list = []
    recorder(hooks, get_scheduler(llm), rec)

    props = torch.cuda.get_device_properties(0)
    result = {
        "arm": args.arm, "k": args.k if args.arm == "enc" else 0, "model": args.model, "revision": args.revision,
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": props.name,
        "sms": props.multi_processor_count, "startup_s": startup_s, "kv_tokens": kv_tokens,
        "kv_cache_dtype": str(cache.cache_dtype), "attention": attn, "compilation": cc,
        "max_batched_tokens": args.max_batched_tokens, "install": install.record,
        "model_shape": {"layers": hf.num_hidden_layers, "hidden": hf.hidden_size, "heads": hf.num_attention_heads,
                        "kv_heads": hf.num_key_value_heads, "head_dim": getattr(hf, "head_dim", None) or
                        hf.hidden_size // hf.num_attention_heads, "intermediate": hf.intermediate_size,
                        "vocab": hf.vocab_size, "tied": bool(getattr(hf, "tie_word_embeddings", False))},
        "gate": [], "points": [],
    }
    if args.arm in ("stock", "stock-compiled"):
        layers = fp8_linears(model)
        result["install"] = {"arm": args.arm, "layers": len(layers), "plain_bytes": sum(l.weight.numel() for _, l in layers),
                             "fp8_kernel": type(layers[0][1].quant_method.fp8_linear).__name__,
                             "activation_quant": str(layers[0][1].quant_method.activation_quant_key)}
    out = Path(args.out)

    def save() -> None:
        out.write_text(json.dumps(result, indent=1))

    sp = SamplingParams(temperature=0.0, max_tokens=args.gate_tokens, logprobs=1, seed=0)
    for o in llm.generate(GATE_PROMPTS, sp, use_tqdm=False):
        c = o.outputs[0]
        result["gate"].append({"prompt": o.prompt, "token_ids": list(c.token_ids),
                               "logprobs": [float(lp[t].logprob).hex() for t, lp in zip(c.token_ids, c.logprobs)]})
    save()
    log("gate generation done")

    rng = random.Random(0)
    warm = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(512)])]
    llm.generate(warm, SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, detokenize=False), use_tqdm=False)
    for ctx in contexts:
        for b in batches:
            need = b * (ctx + args.steps + math.ceil(b * ctx / args.max_batched_tokens) + 8)
            point = {"batch": b, "context": ctx, "kv_needed": need}
            if need > kv_tokens:
                point["skipped"] = f"KV cache holds {kv_tokens} tokens, the point needs {need}"
                result["points"].append(point)
                continue
            prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(ctx)]) for _ in range(b)]
            mt = args.steps + math.ceil(b * ctx / args.max_batched_tokens) + 8
            rec.clear()
            t = time.perf_counter()
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True, detokenize=False),
                         use_tqdm=False)
            point.update(steady(rec, b), wall_s=time.perf_counter() - t, max_tokens=mt, updates=len(rec))
            result["points"].append(point)
            save()
            log("point", json.dumps(point))
    hooks.uninstall()
    if install.hooks is not None:
        install.hooks.uninstall()
    save()


if __name__ == "__main__":
    main()
