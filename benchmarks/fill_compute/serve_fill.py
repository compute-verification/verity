"""One arm of the filler under real serving: Llama-3.1-8B in vLLM on one GPU (in process, FP8 online quantization, full CUDA
graphs for decode, as benchmarks/vllm_encoded/serve.py runs it), its decode timed at each batch and context, with or
without a persistent Pearl-C filler on the SMs the step doesn't use.

Arms (``--arm``):
  stock       vLLM as it is: every SM, no filler
  gc:N        the step on a green context of N SMs (fill.cu's `gc_split`), no filler
  gc:N:fill   the same, and the 128 x 128 fold filler (bench.py's best arm) on the rest through every timed point, launched
              in bounded chunks by fill.cu's native pump (`Fill`)
  gc:N:fill:D the same at duty D: each CTA idles its tile time x (1 / D - 1) after each tile, the tile time measured
              before vLLM starts (the filler alone on every SM, `Fill.tile_ns`)
  gc:N:rcpt[:D] the same with the receipts store: each tile writes its sketch-sha512 receipts (receipt.py), not the fold
  pc:A:B:rcpt[:D] the partition chosen per step by the running requests' mean context: at most --narrow-ctx, the step on
              A SMs and the receipt filler on the rest; above it, the step on B < A SMs and the filler on the rest (fill.cu's
              `gc_nest`: the step's two streams nested, the filler's too). The step switches stream between steps, the
              new one waiting on the old, and the pump launches each chunk on the partition the current step leaves

How the step is confined: before vLLM starts, the step's green-context stream becomes torch's current stream, so vLLM
captures its CUDA graphs and launches its eager kernels there (`gc_split`'s `sm_spread` check: a green context's
launches, direct and from a graph, stay on its SMs). Each arm records the stream `execute_model` ran on, which must be
the green one (for `pc`, one of its two; a graph captured on either and replayed on the other stays on the replay
stream's SMs, `nest_check`).

Gates: each arm generates fixed prompts greedily (`gate`: token ids and log-probs), which compare_fill.py requires to be
bit-identical to the stock arm's; a filled arm's fold (or receipts) is checked against `gemm_pearlc_d`'s words (receipt.py's
receipts of them) on the same operands after its last point (every tile it ran, by the queue counter's passes over the
problem).

A decode step's time is serve.py's: the interval between consecutive scheduler updates that both scheduled exactly B
decode tokens for B requests, the median over the point's steady decode.

A context `A+B` is a mixed batch: half its requests at A tokens, half at B (the point's `context` their mean, `mix` both).

  python serve_fill.py --lib libfill.so --model DIR --arm gc:128:fill --out OUT.json [--batches 16,32] [--contexts 512,4096]
"""
from __future__ import annotations

import argparse
import ctypes as C
import itertools
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

import numpy as np
import torch

import bench as B

GATE_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1905, Albert Einstein published",
    "A haiku about the sea:",
]
CAPTURE = [1, 4, 8, 16, 32]


def log(*a) -> None:
    print("[serve_fill]", *a, file=sys.stderr, flush=True)


def steady(rec: list, b: int) -> dict:
    iv = [t1 - t0 for (t0, n0, r0), (t1, n1, r1) in itertools.pairwise(rec) if n0 == r0 == b and n1 == r1 == b]
    if len(iv) < 4:
        return {"steps": len(iv)}
    iv = iv[1:]
    med = statistics.median(iv)
    q = statistics.quantiles(iv, n=10)
    return {"steps": len(iv), "step_ms": med * 1e3, "p10_ms": q[0] * 1e3, "p90_ms": q[-1] * 1e3, "tokens_per_s": b / med}


class Fill:
    """The 128 x 128 filler (store `fold` or `rcpt`) on the filler's green stream through fill.cu's native pump: `depth`
    chunks of `per` tiles a CTA queued at a time, so a device-wide synchronize inside the engine waits for at most `depth`
    chunks (a persistent filler would hold it forever), and every position launched is a tile computed."""

    def __init__(self, lib: B.Lib, stream, fsms: int, sms: int, M: int = 4096, N: int = 4096, K: int = 8192,
                 per: int = 2, depth: int = 2, duty: float = 1.0, store: str = "fold"):
        if not 0 < duty <= 1:
            raise ValueError(f"duty {duty}")
        self.lib, self.stream, self.fsms, self.sms, self.per, self.depth = lib, stream, fsms, sms, per, depth
        self.store = store
        base = B.random_filler(M, N, K, qcap=1)
        self.f = base.with_tile(128, qcap=1)
        self.credit = B.credit_tile(M, N, K, 128)
        self.tile_ns = self.alone()
        self.duty, self.gap_ns = duty, int(self.tile_ns * (1 / duty - 1))

    def alone(self) -> float:
        """One tile's time on one SM: two passes over the problem on every SM, one CTA each, on a stream of the device's."""
        f, s = self.f, torch.cuda.Stream()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for timed in (False, True):
            if timed:
                e0.record(s)
            for pos in range(0, 2 * f.tiles, 4 * self.sms):
                self.lib.fill(0, B.STORES[self.store], self.sms, f.args(pos, min(4 * self.sms, 2 * f.tiles - pos), log=False,
                                                                         stop=False), s, f.minb)
            if timed:
                e1.record(s)
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e6 * self.sms / (2 * f.tiles)
        lib.l.fill_pump_start.argtypes = [C.c_int] * 5 + [C.POINTER(B.Args), C.c_void_p]
        lib.l.fill_pump_start.restype = C.c_int
        lib.l.fill_pump_tiles.restype = C.c_longlong
        lib.l.fill_pump_stop.argtypes = [C.POINTER(C.c_int)]
        lib.l.fill_pump_stop.restype = C.c_longlong
        lib.l.fill_pump_alt.argtypes = [C.c_void_p, C.c_int]
        lib.l.fill_pump_set_wide.argtypes = [C.c_int]
        self.args = None

    def start(self, wide=None) -> None:
        f = self.f
        if wide is not None:
            self.lib.l.fill_pump_alt(C.c_void_p(wide[0].cuda_stream), wide[1])
        f.poison()
        torch.cuda.synchronize()
        self.args = f.args(0, 0, log=False, stop=False, gap_ns=self.gap_ns)
        rc = self.lib.l.fill_pump_start(f.minb, B.STORES[self.store], self.fsms, self.per, self.depth, C.byref(self.args),
                                        C.c_void_p(self.stream.cuda_stream))
        if rc:
            raise RuntimeError(f"fill_pump_start: {rc}")

    def tiles(self) -> int:
        return int(self.lib.l.fill_pump_tiles())

    def stop(self) -> int:
        rc = C.c_int(0)
        n = int(self.lib.l.fill_pump_stop(C.byref(rc)))
        torch.cuda.synchronize()
        if rc.value:
            raise RuntimeError(f"the pump's launch failed: {rc.value}")
        return n

    def check(self, n: int) -> dict:
        """The fold of n queue positions (each tile XORed in once per pass) against `gemm_pearlc_d`'s words, or each tile's
        receipts against receipt.py's of them."""
        f = self.f
        refs = B.references(self.lib, f, self.sms, torch.cuda.current_stream())
        counts = np.full(f.tiles, n // f.tiles, np.int64)
        counts[: n % f.tiles] += 1
        return B.compare(f, B.STORES[self.store], *refs, counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="16,32")
    ap.add_argument("--contexts", default="512,4096")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gate-tokens", type=int, default=32)
    ap.add_argument("--mem", type=float, default=0.80)
    ap.add_argument("--narrow-ctx", type=int, default=512, help="arm pc: the mean context at or below which the step is wide")
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [[int(y) for y in x.split("+")] for x in args.contexts.split(",")]
    parts = args.arm.split(":")
    if parts[0] == "pc" and len(parts) in (4, 5) and parts[3] == "rcpt" and int(parts[2]) < int(parts[1]):
        pass
    elif parts[0] not in ("stock", "gc") or (parts[0] == "gc" and len(parts) not in (2, 3, 4)) or \
            (len(parts) >= 3 and parts[2] not in ("fill", "rcpt")):
        ap.error("--arm stock | gc:N | gc:N:fill[:D] | gc:N:rcpt[:D] | pc:A:B:rcpt[:D]")

    lib = B.Lib(args.lib)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    result: dict = {"arm": args.arm, "sms": sms, "gate": [], "points": []}
    fill = None
    nest = None
    if parts[0] == "pc":
        g = lib.gc_nest(int(parts[2]), int(parts[1]) - int(parts[2]), -5, 0)
        if "error" in g:
            raise SystemExit(f"gc_nest: {g['error']}")
        nest = g
        rem, wide_fill = g["rem_sms"], g["mid_sms"] + g["rem_sms"]
        on = {k: lib.sm_set(g[a], g[b], 4 * sms) for k, a, b in
              (("narrow_on_wide", "step_narrow", "step_wide"), ("wide_on_narrow", "step_wide", "step_narrow"),
               ("wide_on_wide", "step_wide", "step_wide"), ("fill_wide", "fill_wide", "fill_wide"),
               ("fill_narrow", "fill_narrow", "fill_narrow"))}
        result["green"] = {"base_sms": g["base_sms"], "mid_sms": g["mid_sms"], "rem_sms": rem,
                           "nest_check": {k: len(v) for k, v in on.items()},
                           "nest_ok": on["narrow_on_wide"] <= on["wide_on_wide"] and not (on["wide_on_wide"] & on["fill_wide"])
                           and len(on["wide_on_narrow"]) == g["base_sms"] + g["mid_sms"]
                           and not (on["wide_on_narrow"] & on["fill_narrow"])}
        torch.cuda.set_stream(g["step_narrow"])
        result["step_stream"] = [g["step_narrow"].cuda_stream, g["step_wide"].cuda_stream]
        fill = Fill(lib, g["fill_narrow"], rem, sms, duty=float(parts[4]) if len(parts) == 5 else 1.0, store="rcpt")
        result["filler"] = {"tile": 128, "store": fill.store, "grid": [rem, wide_fill], "shape": [4096, 4096, 8192],
                            "credit_w1_per_tile": fill.credit, "duty": fill.duty, "tile_ns": fill.tile_ns,
                            "gap_ns": fill.gap_ns, "narrow_ctx": args.narrow_ctx}
    if parts[0] == "gc":
        g = lib.gc_split(int(parts[1]), -5, 0)
        if "error" in g:
            raise SystemExit(f"gc_split: {g['error']}")
        result["green"] = {k: v for k, v in g.items() if k not in ("step", "fill")}
        result["green"]["spread_step_graph"] = lib.sm_spread(g["step"], 4 * sms, graph=True)
        torch.cuda.set_stream(g["step"])
        result["step_stream"] = g["step"].cuda_stream
        if len(parts) >= 3:
            fill = Fill(lib, g["fill"], g["fill_sms"], sms, duty=float(parts[3]) if len(parts) == 4 else 1.0,
                        store="rcpt" if parts[2] == "rcpt" else "fold")
            result["filler"] = {"tile": 128, "store": fill.store, "grid": g["fill_sms"], "shape": [4096, 4096, 8192],
                                "credit_w1_per_tile": fill.credit, "duty": fill.duty, "tile_ns": fill.tile_ns,
                                "gap_ns": fill.gap_ns}

    import vllm
    from verity_vllm.engine.engine_profile import get_model_runner, get_scheduler
    from verity_vllm.engine.hooks import Hooks
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    hooks = Hooks(tag="ra-fill-compute")
    seen = set()
    want = {"wide": False}
    switches = {"to_wide": 0, "to_narrow": 0}

    def runner_hook(orig):
        def execute_model(self, *a, **kw):
            if nest is not None:
                cur = torch.cuda.current_stream()
                new = nest["step_wide"] if want["wide"] else nest["step_narrow"]
                if new.cuda_stream != cur.cuda_stream:
                    new.wait_stream(cur)
                    torch.cuda.set_stream(new)
                    lib.l.fill_pump_set_wide(int(want["wide"]))
                    switches["to_wide" if want["wide"] else "to_narrow"] += 1
            seen.add(torch.cuda.current_stream().cuda_stream)
            return orig(self, *a, **kw)
        return execute_model

    import importlib
    for mod in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
        try:
            hooks.wrap(importlib.import_module(mod).GPUModelRunner, "execute_model", runner_hook)
        except (ImportError, AttributeError):
            continue

    steps_cap = args.steps + max(batches) * max(map(max, contexts)) // 2048 + 16
    cc = {"cudagraph_capture_sizes": CAPTURE, "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    kw = {"model": args.model, "quantization": "fp8", "max_num_seqs": max(CAPTURE), "enable_prefix_caching": False, "seed": 0,
          "max_model_len": max(map(max, contexts)) + steps_cap + 64, "max_num_batched_tokens": 2048,
          "gpu_memory_utilization": args.mem, "enforce_eager": False, "compilation_config": cc}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    result.update(vllm=vllm.__version__, torch=torch.__version__, startup_s=time.perf_counter() - t0, compilation=cc)
    hf = llm.llm_engine.vllm_config.model_config.hf_config
    result["model_shape"] = {"layers": hf.num_hidden_layers, "hidden": hf.hidden_size, "intermediate": hf.intermediate_size,
                             "heads": hf.num_attention_heads, "kv_heads": hf.num_key_value_heads, "vocab": hf.vocab_size}
    rec: list = []

    sched = get_scheduler(llm)

    def make(orig):
        def update_from_output(so, mro):
            rec.append((time.perf_counter(), so.total_num_scheduled_tokens, len(so.num_scheduled_tokens)))
            out_ = orig(so, mro)
            if nest is not None:
                run = sched.running
                want["wide"] = bool(run) and sum(r.num_computed_tokens for r in run) / len(run) > args.narrow_ctx
            return out_
        return update_from_output

    hooks.wrap(sched, "update_from_output", make)
    out = Path(args.out)

    def save() -> None:
        out.write_text(json.dumps(result, indent=1))

    sp = SamplingParams(temperature=0.0, max_tokens=args.gate_tokens, logprobs=1, seed=0)
    if fill:
        fill.start(None if nest is None else (nest["fill_wide"], nest["mid_sms"] + nest["rem_sms"]))
    for o in llm.generate(GATE_PROMPTS, sp, use_tqdm=False):
        c = o.outputs[0]
        result["gate"].append({"prompt": o.prompt, "token_ids": list(c.token_ids),
                               "logprobs": [float(lp[t].logprob).hex() for t, lp in zip(c.token_ids, c.logprobs)]})
    result["streams_seen"] = sorted(seen)
    save()
    rng = random.Random(0)
    llm.generate([TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(512)])],
                 SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, detokenize=False), use_tqdm=False)
    for mix in contexts:
        ctx = sum(mix) // len(mix)
        for b in batches:
            prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(mix[i * len(mix) // b])])
                       for i in range(b)]
            mt = args.steps + b * ctx // 2048 + 8
            rec.clear()
            n0 = fill.tiles() if fill else 0
            t = time.perf_counter()
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True, detokenize=False),
                         use_tqdm=False)
            wall = time.perf_counter() - t
            point = {"batch": b, "context": ctx, **({"mix": mix} if len(mix) > 1 else {}), **steady(rec, b), "wall_s": wall,
                     "max_tokens": mt}
            if nest is not None:
                point["switches"] = dict(switches)
            if fill:
                n = fill.tiles() - n0
                fl = n * (fill.f.fp8_flops_tile + fill.f.bf16_flops_tile)
                point["filler"] = {"tiles": n, "per": fill.per, "depth": fill.depth, "tflops": fl / wall / 1e12, "credit_w1_per_s": (1 - B.CAP) * fill.credit * n / wall,
                                   "util": n * (fill.f.fp8_flops_tile / B.PEAK_E4M3 + fill.f.bf16_flops_tile / B.PEAK_BF16) / wall}
            result["points"].append(point)
            save()
            log("point", json.dumps(point))
    if fill:
        n = fill.stop()
        result["filler"]["tiles_total"] = n
        result["filler"]["check"] = fill.check(n)
        log("filler check", result["filler"]["check"])
    result["streams_seen"] = sorted(seen)
    if nest is not None:
        result["switches"] = switches
    hooks.uninstall()
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
