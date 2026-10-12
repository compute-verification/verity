"""One arm of the encoded-weights decode benchmark: Llama-3.1-8B served by vLLM (in process, FP8 online quantization, full
CUDA graphs for decode), timed at each decode batch and context length.

Arms:
  stock     vLLM's own FP8 linears (CUTLASS on sm_120), same engine settings
  plain     every linear through `ve.cu`'s fused GEMM over plain FP8 fragments (k = 0): the native kernel of this harness
  enc       every linear through the fused GEMM over codec words only (k rounds of Sloth at p = 2^61 - 1); the plaintext
            weight is freed before vLLM profiles memory, so decode runs with no plaintext copy of any linear in HBM
  wide      the same with ra-codec-wide's fused GEMM (`wide.cu`, ``--wide-lib``: Sloth over p = 2^127 - 1, 2^255 - 19 or
            2^521 - 1, k rounds, k = 0 its plain payloads), with ``--tag`` also tagging every block of words it reads
            (ro-blake3/v1, the bandwidth audit's tag) as decode runs

Each arm runs the engine in compilation mode 0 with full CUDA graphs for decode, or with ``--compiled`` as vLLM serves by
default (torch.compile, piecewise and full graphs; the fused linear is the opaque op ``ve_enc::linear``, and vLLM's activation
quantization before it is compiled as in the stock arm).  vLLM's default compile tunes inductor's kernels by timing in
each process, so two processes can compile different reductions; ``--no-autotune`` compiles without that tuning, the
same in every process.

The plain and encoded arms install at the end of the model runner's `load_model` (before memory profiling, compilation and
graph capture), through `verity_vllm.engine.hooks`.  Before anything is timed, each linear passes its gates: every GEMM form
gives the same partial sums bit for bit (encoded against plain at every capture size), the plain sums agree with a float
product, the encoded store decodes to the payloads on the device and in `codec.py` (`wide.py`), sampled tags equal the
bandwidth audit's reference (`verity.core.protocols.bandwidth.tags`), and (encoded arms) no FP8 or plain fragment tensor
of a linear's size is left anywhere in the process.  Each arm then generates fixed prompts greedily
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
TAG_SEED = hashlib.sha256(b"ra-vllm-encoded/tag-seed/v0").digest()
TAG = "ro-blake3/v1"
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
        wide = args.arm == "wide"
        k = args.k if args.arm in ("enc", "wide") else 0
        wlib = K.WideLib(args.wide_lib) if wide else None
        if wide:
            if k:
                wlib.set_key(SEED, k)
            if args.tag:
                wlib.set_tag(TAG_SEED)
                wlib.tag_buffer(max(layout.wide_store_words(m.output_size_per_partition, m.input_size_per_partition,
                                                            wlib.payload) // wlib.lane_words for _, m in layers))
        elif k:
            lib.set_key(SEED, k)
        torch.cuda.synchronize()
        alloc0 = torch.cuda.memory_allocated()
        gen = torch.Generator(device="cuda")
        cands: dict[tuple, list[K.Config]] = {}
        gates = {"layers": 0, "configs_checked": 0, "python_words": 0, "tags_checked": 0, "max_rel_err": 0.0}
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
            if k or wide:
                torch.cuda.synchronize()
                te = time.perf_counter()
                lin = K.WideLinear(lib, wlib, w, ws, k, base) if wide else K.EncodedLinear(lib, w, ws, k, base)
                torch.cuda.synchronize()
                encode_s += time.perf_counter() - te
            else:
                lin = plain
            nch = kd // layout.TASK_COLS
            gen.manual_seed(1000 + idx)
            for m in CAPTURE:
                x8 = rand_fp8(m, kd, gen)
                ref = plain.gemm(x8, K.Config(0))
                for op in (plain, lin) if lin is not plain else (plain,):
                    for cfg in cands.setdefault((type(op).__name__, op.k, m, nch), op.candidates(m)):
                        if not bits_equal(op.gemm(x8, cfg), ref):
                            raise RuntimeError(f"gate: {name} {type(op).__name__} k={op.k} {cfg.name} M={m}: partial sums "
                                               "differ from plain simple")
                        gates["configs_checked"] += 1
                if m == 4:
                    exact = x8.float() @ w.float().t()
                    err = (ref.sum(0) - exact).abs().max().item() / max(exact.abs().max().item(), 1e-30)
                    gates["max_rel_err"] = max(gates["max_rel_err"], err)
                    if err > 1e-3:
                        raise RuntimeError(f"gate: {name}: fused sums are {err:.2e} off the float product")
            if wide and wlib.tag_seed is not None:
                gates["tags_checked"] += check_tags(lin, name, args.python_tags)
            if k or wide:
                if not torch.equal(lin.decoded(), lin.payloads):
                    raise RuntimeError(f"gate: {name}: the store does not decode to its payloads")
                if not torch.equal(lin.dense().view(torch.uint8), w.view(torch.uint8)):
                    raise RuntimeError(f"gate: {name}: the decoded weight is not the weight")
                if k:
                    check_words(lin, name, idx, args.python_words)
                    gates["python_words"] += args.python_words
                lin.drop_payloads()
                base += lin.words if wide else lin.store.numel()
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
            K.LINEARS.append(lin)
            fn = apply(len(K.LINEARS) - 1, layer.quant_method.fp8_linear.quant_fp8)
            self.hooks.wrap(layer.quant_method, "apply", lambda orig, fn=fn: fn)
            gates["layers"] += 1
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc1 = torch.cuda.memory_allocated()
        selection = self.select()
        scan = plaintext_scan(min(lin.n * lin.kd for _, lin in self.linears))
        if k and (scan["fp8"] or scan["uint8"]):
            raise RuntimeError(f"gate: plaintext left in the process: {scan}")
        codec = ({"field": wlib.field.name, "word_bits": wlib.bits, "payload_bits": wlib.payload, "limbs": wlib.limbs,
                  "lane_words": wlib.lane_words, "wide_lib": Path(args.wide_lib).name, "kmax": wlib.kmax,
                  "tag": TAG if wlib.tag_seed is not None else None}
                 if wide else {"field": "p61", "word_bits": 61, "payload_bits": lib.bits, "tag": None})
        self.record = {
            "arm": args.arm, "k": k, "lib": Path(args.lib).name, "bits": codec["payload_bits"], "kmax": lib.kmax,
            "codec": codec, "layers": len(layers),
            "plain_bytes": plain_bytes, "store_bytes": store_bytes, "store_over_plain": store_bytes / plain_bytes,
            "words": base, "encode_s": encode_s, "allocated_before": alloc0, "allocated_after": alloc1,
            "allocated_delta": alloc1 - alloc0, "plaintext_scan": scan, "gates": gates, "selection": selection,
            "fp8_kernel": type(layers[0][1].quant_method.fp8_linear).__name__,
            "activation_quant": str(layers[0][1].quant_method.activation_quant_key),
            "install_s": time.perf_counter() - t0,
        }
        log("installed", json.dumps({k_: v for k_, v in self.record.items() if k_ != "selection"}))

    def select(self) -> list[dict]:
        """Each shape's GEMM config at each capture size: the fastest candidate over all of the shape's layers in turn (so
        the weights stream from HBM, not L2)."""
        groups: dict[tuple[int, int], list[K.Linear]] = {}
        for _, lin in self.linears:
            groups.setdefault((lin.n, lin.kd), []).append(lin)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(7)
        out = []
        for (n, kd), lins in groups.items():
            for m in CAPTURE:
                x8 = rand_fp8(m, kd, gen)
                times = {}
                for cfg in lins[0].candidates(m):
                    times[cfg] = time_config(lins, x8, cfg, self.args.select_trials)
                best = min(times, key=times.get)
                for lin in lins:
                    lin.configs[m] = best
                out.append({"n": n, "k": kd, "m": m, "layers": len(lins), "best": best.name, "groups": best.groups,
                            "us": {c.name: round(t, 2) for c, t in times.items()}})
                log(f"select n={n} k={kd} m={m}: {best.name} {times[best]:.1f} us", {c.name: round(t, 1) for c, t in times.items()})
        torch.cuda.empty_cache()
        return out


def check_words(lin: K.Linear, name: str, idx: int, n: int) -> None:
    """``n`` sampled words of the store decode to their payloads in the Python reference (`codec.py` or `wide.py`)."""
    rng = random.Random(idx)
    if isinstance(lin, K.WideLinear):
        wide, w = K._wide(), lin.wlib
        for j in [rng.randrange(lin.words) for _ in range(n)]:
            y, want = layout.wide_word(lin.store, j, w.limbs), layout.wide_word(lin.payloads, j, w.limbs)
            if wide.decode_word(y, lin.base + j, w.key, w.payload) != want:
                raise RuntimeError(f"gate: {name}: wide.py decodes word {lin.base + j} otherwise")
        return
    codec, store = K._codec(), lin.store.view(torch.int64)
    for j in [rng.randrange(store.numel()) for _ in range(n)]:
        if codec.decode_word(int(store[j]), lin.base + j, lin.lib.key, lin.lib.bits) != int(lin.payloads[j]):
            raise RuntimeError(f"gate: {name}: codec.py decodes word {lin.base + j} otherwise")


def check_tags(lin: K.WideLinear, name: str, n: int) -> int:
    """The last GEMM's tags of ``n`` sampled blocks and the last block equal the bandwidth audit's ro-blake3/v1 of the
    block's encoded words at its index."""
    from verity.core.protocols.bandwidth import tags
    w = lin.wlib
    torch.cuda.synchronize()
    keyed = tags.get(TAG).keyed(w.tag_seed)
    rng = random.Random(lin.base)
    js = sorted({lin.blocks - 1, *(rng.randrange(lin.blocks) for _ in range(n))})
    got = w.tags[:8 * lin.blocks].view(-1, 8)[js].cpu().numpy()
    for j, t in zip(js, got):
        if t.tobytes() != keyed(lin.tag_block_base + j, layout.wide_block(lin.store, j, w.lane_words, w.limbs)):
            raise RuntimeError(f"gate: {name}: block {lin.tag_block_base + j}'s tag is not {TAG}'s")
    return len(js)


def apply(idx: int, quant):
    """A layer's FP8 apply: vLLM's own activation quantization (traced and fused by torch.compile when the engine compiles),
    then the linear through the op."""
    def fn(layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"activations are {x.dtype}, the fused linear writes BF16")
        x8, xs = quant(x.reshape(-1, x.shape[-1]), None, None)
        y = torch.ops.ve_enc.linear(x8, xs, idx)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])
    return fn


def time_config(lins: list[K.Linear], x8: torch.Tensor, cfg: K.Config, trials: int) -> float:
    """Microseconds a layer: the shape's layers in turn, captured in one CUDA graph as decode runs them (one partials
    buffer, as the graph pool reuses a freed one), the fastest of ``trials`` timings of two replays each."""
    part = torch.empty((lins[0].nchunks, x8.shape[0], lins[0].n), dtype=torch.float32, device=x8.device)
    for lin in lins:
        lin.gemm(x8, cfg, part=part)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for lin in lins:
            lin.gemm(x8, cfg, part=part)
    g.replay()
    best = math.inf
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 2)
    del g
    return best * 1e3 / len(lins)


def graph_ms(fn, trials: int) -> float:
    """Milliseconds of ``fn``'s kernels captured in one CUDA graph: the fastest of ``trials`` timings of two replays."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    best = math.inf
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / 2)
    del g
    torch.cuda.empty_cache()
    return best


def native_pass(layers: list[tuple[str, torch.nn.Module]], ms: list[int], trials: int) -> list[dict]:
    """One native pass of the weights at each batch m: every FP8 linear in model order through vLLM's own apply (its
    activation quantization and GEMM, eager) and through ``torch._scaled_mm`` alone on FP8 activations (per-tensor
    scales), each in one CUDA graph.  The rate audit's d is a step's time in such passes."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(11)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    out = []
    for m in ms:
        xs = [torch.randn((m, layer.input_size_per_partition), generator=gen, device="cuda", dtype=torch.bfloat16)
              for _, layer in layers]
        x8 = [x.to(torch.float8_e4m3fn) for x in xs]

        def run_apply(xs=xs):
            for (_, layer), x in zip(layers, xs):
                layer.quant_method.apply(layer, x)

        def run_mm(x8=x8):
            for (_, layer), x in zip(layers, x8):
                torch._scaled_mm(x, layer.weight, scale_a=one, scale_b=layer.weight_scale.reshape(()),
                                 out_dtype=torch.bfloat16)

        point = {"batch": m, "apply_ms": graph_ms(run_apply, trials)}
        try:
            point["scaled_mm_ms"] = graph_ms(run_mm, trials)
        except RuntimeError as e:
            point["scaled_mm_error"] = str(e)[:300]
        point["pass_ms"] = min(point["apply_ms"], point.get("scaled_mm_ms", math.inf))
        out.append(point)
        log("native pass", json.dumps(point))
    return out


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


class Window:
    """cudaProfilerStart/Stop around `steps` steady decode steps of each point (for `ncu --profile-from-start off`), each
    window opened by torch's ``spin_kernel`` (`torch.cuda._sleep`) so the profile splits by point."""

    def __init__(self, steps: int) -> None:
        self.steps, self.point, self.run, self.opened, self.on = steps, None, 0, False, False
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
            torch.cuda.profiler.start()
            torch.cuda._sleep(1000)
            self.opened = self.on = True
        elif self.on and (self.run == 0 or self.run >= 4 + self.steps):
            self.end()

    def end(self) -> None:
        if self.on:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            self.on = False
            self.windows.append(self.point)
        self.point = None


def recorder(hooks, sched, rec: list, window: Window) -> None:
    def make(orig):
        def update_from_output(so, mro):
            rec.append((time.perf_counter(), so.total_num_scheduled_tokens, len(so.num_scheduled_tokens)))
            window(so.total_num_scheduled_tokens, len(so.num_scheduled_tokens))
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
    ap.add_argument("--arm", choices=("stock", "plain", "enc", "wide"), required=True)
    ap.add_argument("--compiled", action="store_true",
                    help="vLLM's default compilation (torch.compile, piecewise and full CUDA graphs) instead of mode 0")
    ap.add_argument("--no-autotune", action="store_true",
                    help="compiled: without inductor's max-autotune and coordinate-descent tuning, which vLLM turns on for "
                         "each capture size and which pick kernels by timing, so the compiled step is the same in every "
                         "process (the compiled gate)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--lib", help="a build of ve.cu (plain, enc and wide arms; the wide arm reduces with it)")
    ap.add_argument("--wide-lib", help="a build of wide.cu (wide arm)")
    ap.add_argument("--tag", action="store_true", help="wide arm: tag every block the GEMMs read (ro-blake3/v1)")
    ap.add_argument("--python-tags", type=int, default=3, help="sampled tags a layer checked against the reference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--contexts", default="512,2048,8192")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--mem", type=float, default=0.75)
    ap.add_argument("--max-batched-tokens", type=int, default=16384)
    ap.add_argument("--select-trials", type=int, default=5)
    ap.add_argument("--native-trials", type=int, default=10,
                    help="stock arm: timings of one native pass of the weights at each batch (0: none)")
    ap.add_argument("--python-words", type=int, default=16)
    ap.add_argument("--gate-tokens", type=int, default=16)
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="open a profiler window (cudaProfilerStart/Stop) over this many steady decode steps a point")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    if args.compiled:
        # the cache key doesn't cover the installed linears: a cached stock graph would bypass them
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        if args.no_autotune:
            os.environ["VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE"] = "0"
            os.environ["VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING"] = "0"
    elif args.no_autotune:
        ap.error("--no-autotune is a compiled mode")

    import vllm
    from verity_vllm.engine.engine_profile import get_model_runner, get_scheduler
    from verity_vllm.engine.hooks import Hooks
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    hooks = Hooks(tag="ra-vllm-encoded")
    install = Install(args)
    if args.arm in ("plain", "enc", "wide"):
        if not args.lib or (args.arm == "wide") != bool(args.wide_lib):
            ap.error("--lib is required for the plain, enc and wide arms, and --wide-lib for the wide arm only")
        hook_runner(hooks, install)
    max_tokens_cap = args.steps + math.ceil(max(batches) * max(contexts) / args.max_batched_tokens) + 8
    cc = {"cudagraph_capture_sizes": CAPTURE}
    if not args.compiled:
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
    window = Window(args.profile_steps)
    recorder(hooks, get_scheduler(llm), rec, window)

    props = torch.cuda.get_device_properties(0)
    result = {
        "arm": args.arm, "compiled": args.compiled, "autotune": args.compiled and not args.no_autotune,
        "k": args.k if args.arm in ("enc", "wide") else 0,
        "codec_tag": TAG if args.arm == "wide" and args.tag else None, "model": args.model,
        "revision": args.revision,
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": props.name,
        "sms": props.multi_processor_count, "startup_s": startup_s, "kv_tokens": kv_tokens,
        "kv_cache_dtype": str(cache.cache_dtype), "attention": attn, "compilation": cc,
        "max_batched_tokens": args.max_batched_tokens, "install": install.record,
        "model_shape": {"layers": hf.num_hidden_layers, "hidden": hf.hidden_size, "heads": hf.num_attention_heads,
                        "kv_heads": hf.num_key_value_heads, "head_dim": getattr(hf, "head_dim", None) or
                        hf.hidden_size // hf.num_attention_heads, "intermediate": hf.intermediate_size,
                        "vocab": hf.vocab_size, "tied": bool(getattr(hf, "tie_word_embeddings", False))},
        "gate": [], "points": [], "profile_steps": args.profile_steps, "profiled": window.windows,
    }
    if args.arm == "stock":
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
            window.begin(b, ctx)
            t = time.perf_counter()
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True, detokenize=False),
                         use_tqdm=False)
            window.end()
            point.update(steady(rec, b), wall_s=time.perf_counter() - t, max_tokens=mt, updates=len(rec))
            result["points"].append(point)
            save()
            log("point", json.dumps(point))
    if args.arm == "stock" and args.native_trials:
        result["native_pass"] = native_pass(fp8_linears(model), batches, args.native_trials)
    hooks.uninstall()
    if install.hooks is not None:
        install.hooks.uninstall()
    save()


if __name__ == "__main__":
    main()
