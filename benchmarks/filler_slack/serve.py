"""One arm of the receipted INT8 filler beside vLLM decode on one GPU (P1's falsifier: what the filler fills beside serving,
and what it costs serving). The engine is `benchmarks/fill_compute/serve_fill.py`'s: Llama-3.1-8B in vLLM in process, FP8
online quantization, full CUDA graphs for decode; its gate prompts, decode-step timing (`steady`) and per-arm JSON, so
`benchmarks/fill_compute/compare_fill.py` compares the arms.

Arms (``--arm``):
  stock        vLLM as it is: every SM, no filler (on a high-priority stream of its own, as in the filler arms)
  warmup       the same, run first and left out of the stock baseline
  share:FW     the filler on every SM beside the engine: one chunk is `grid` = #SMs blocks of FW filler warps with no
               shared memory (a block takes only registers and warp slots), on a stream of the lowest priority; the
               engine's stream has the highest
  gc:N         the engine on a green context of N SMs (fill_compute's `gc_split`), no filler
  gc:N:FW      the same, and the filler on the rest, one block of FW warps an SM

The filler is libifill.so's pump (serve.cu): `depth` chunks of `--iters` rounds queued at a time, each chunk's operands
salted by its own seed. Before the engine starts, the filler runs alone for `--alone-s` (its rate, power and clock on
an otherwise idle GPU); after the points, again beside the idle engine. Per point: the decode step (serve_fill's
`steady`), and over the point's wall time the filler's warp-MMAs, its rate in MMAs per SM-clock inside its blocks, its
share of the tensor peak (one m16n8k32 an SMSP every 16 cycles, at the NVML SM clock of the window and at the part's
maximum 2,430 MHz), and the board's power and SM clock (NVML every 20 ms).

Gates: the greedy tokens and log-probs bit-identical to the stock arm's (compare_fill.py); the engine's `execute_model`
on its own stream only; the last chunk's receipts and accumulators of `--check-threads` filler threads equal to
twin.py's.

  python serve.py --lib libifill.so --fill-lib libfill.so --model DIR --arm share:16 --out OUT.json
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
FC = HERE.parent / "fill_compute"
sys.path[:0] = [str(HERE), str(FC)]

import torch

import twin
import serve_fill as SF

KSEED = 0xC0FFEE11
PEAK_PER_SM_CLK = 0.25  # warp-MMAs (m16n8k32) an SM a clock: one an SMSP every 16 cycles
MAX_MHZ = 2430.0
OUTW = 24


class Sampler:
    """NVML on the leased GPU every 20 ms: (host monotonic time, SM MHz, board W)."""

    def __init__(self):
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        uuid = os.environ.get("GPU_LEASE_UUID")
        self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid else pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[tuple[float, float, float]] = []
        self.stop = threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.samples.append((time.monotonic(), float(self.nv.nvmlDeviceGetClockInfo(self.h, 1)),
                                     self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(0.02)

    def window(self, t0: float, t1: float) -> dict:
        xs = [s for s in self.samples if t0 <= s[0] <= t1]
        if not xs:
            return {"samples": 0}
        clk, pw = [s[1] for s in xs], [s[2] for s in xs]
        return {"samples": len(xs), "sm_mhz": statistics.fmean(clk), "sm_mhz_min": min(clk), "power_w": statistics.fmean(pw),
                "power_w_max": max(pw)}

    def reasons(self) -> int | None:
        try:
            return int(self.nv.nvmlDeviceGetCurrentClocksEventReasons(self.h))
        except Exception:  # noqa: BLE001
            return None


class Filler:
    def __init__(self, path: str, sms: int, grid: int, fw: int, iters: int, depth: int, smem: int, stream: int):
        self.l = C.CDLL(path)
        i, vp = C.c_int, C.c_void_p
        self.l.ifill_init.argtypes = [C.c_uint32, i, i]
        self.l.ifill_start.argtypes = [i, i, i, i, vp, C.c_uint32]
        self.l.ifill_stats.argtypes = [C.POINTER(C.c_longlong)]
        self.l.ifill_stop.argtypes = [C.POINTER(i)]
        self.l.ifill_stop.restype = C.c_longlong
        self.l.ifill_last.argtypes = [C.POINTER(C.c_uint32), i, C.POINTER(C.c_uint32), C.POINTER(i)]
        self.sms, self.grid, self.fw, self.iters, self.depth, self.smem, self.stream = sms, grid, fw, iters, depth, smem, stream
        self.seed0 = 0x5EED0000
        rc = self.l.ifill_init(KSEED, grid, depth)
        if rc:
            raise RuntimeError(f"ifill_init: {rc}")

    def start(self) -> None:
        rc = self.l.ifill_start(self.grid, self.fw, self.iters, self.smem, C.c_void_p(self.stream), self.seed0)
        if rc:
            raise RuntimeError(f"ifill_start: {rc}")

    def stats(self) -> tuple[float, ...]:
        o = (C.c_longlong * 5)()
        self.l.ifill_stats(o)
        return tuple(o)

    def stop(self) -> int:
        rc = C.c_int(0)
        n = int(self.l.ifill_stop(C.byref(rc)))
        if rc.value:
            raise RuntimeError(f"the pump's launch failed: {rc.value}")
        self.seed0 += n + 1
        return n

    def check(self, n: int) -> dict:
        """The last chunk's first n threads against twin.py (after stop)."""
        buf, seed, iters = (C.c_uint32 * (n * OUTW))(), C.c_uint32(0), C.c_int(0)
        rc = self.l.ifill_last(buf, n, C.byref(seed), C.byref(iters))
        if rc:
            return {"error": rc, "mismatched": None, "tiles_checked": 0}
        bad = []
        step = max(1, n // 64)
        for t in range(0, n, step):
            if list(buf[t * OUTW: t * OUTW + 20]) != twin.thread_words(seed.value, KSEED, iters.value, t):
                bad.append(t)
        return {"seed": seed.value, "iters": iters.value, "tiles_checked": len(range(0, n, step)), "mismatched": len(bad),
                "first_bad": bad[:4]}


def delta(f: Filler, s0: tuple, s1: tuple, t0: float, t1: float, smp: Sampler, sms: int) -> dict:
    """The filler over a window: s0, s1 its stats at the window's ends, t0, t1 host times."""
    chunks, mmas, span = (s1[i] - s0[i] for i in range(3))
    nv = smp.window(t0, t1)
    wall = t1 - t0
    out = {"wall_s": wall, "chunks": chunks, "warp_mmas": mmas, "most_blocks_on_one_sm": s1[4], **nv}
    if not chunks:
        return out
    rate = mmas / wall
    out["mma_per_s"] = rate
    out["mma_per_sm_clk_in_block"] = mmas / span if span else None
    out["int8_tops"] = rate * 2 * 16 * 8 * 32 / 1e12
    mhz = nv.get("sm_mhz")
    if mhz:
        out["share_of_peak"] = rate / (sms * PEAK_PER_SM_CLK * mhz * 1e6)
    out["share_of_peak_at_max_clock"] = rate / (sms * PEAK_PER_SM_CLK * MAX_MHZ * 1e6)
    # compare_fill.py's fields: TFLOPS and tensor utilization against the E4M3 peak at 2,100 MHz (the INT8 m16n8k32 runs
    # at the same rate as the FP8 one)
    out["tflops"] = out["int8_tops"]
    out["util"] = rate * 2 * 16 * 8 * 32 / SF.B.PEAK_E4M3
    return out


def alone(f: Filler, smp: Sampler, seconds: float, sms: int) -> dict:
    f.start()
    time.sleep(0.5)
    s0, t0 = f.stats(), time.monotonic()
    time.sleep(seconds)
    s1, t1 = f.stats(), time.monotonic()
    f.stop()
    return delta(f, s0, s1, t0, t1, smp, sms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--fill-lib", help="fill_compute's libfill.so, for gc arms (gc_split)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--contexts", default="512,4096")
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--gate-tokens", type=int, default=32)
    ap.add_argument("--mem", type=float, default=0.80)
    ap.add_argument("--iters", type=int, default=4096, help="filler rounds a chunk")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--alone-s", type=float, default=5.0)
    ap.add_argument("--check-threads", type=int, default=2048)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [[int(y) for y in x.split("+")] for x in args.contexts.split(",")]
    parts = args.arm.split(":")
    if parts[0] not in ("stock", "warmup", "share", "gc") or (parts[0] == "share" and len(parts) != 2) or \
            (parts[0] == "gc" and len(parts) not in (2, 3)):
        ap.error("--arm stock | warmup | share:FW | gc:N | gc:N:FW")

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    result: dict = {"arm": args.arm, "sms": sms, "gate": [], "points": [], "chunk_iters": args.iters, "depth": args.depth}
    smp = Sampler()
    lib = C.CDLL(args.lib)
    lib.ifill_stream.argtypes = [C.c_int, C.POINTER(C.c_void_p)]
    fill = None
    if parts[0] == "gc":
        if not args.fill_lib:
            ap.error("gc arms need --fill-lib")
        B = SF.B.Lib(args.fill_lib)
        g = B.gc_split(int(parts[1]), -5, 0)
        if "error" in g:
            raise SystemExit(f"gc_split: {g['error']}")
        result["green"] = {k: v for k, v in g.items() if k not in ("step", "fill")}
        result["green"]["spread_step_graph"] = B.sm_spread(g["step"], 4 * sms, graph=True)
        step = g["step"]
        if len(parts) == 3:
            fill = Filler(args.lib, sms, g["fill_sms"], int(parts[2]), args.iters, args.depth,
                          lib.ifill_smem_one(), g["fill"].cuda_stream)
            result["green"]["spread_fill"] = B.sm_spread(g["fill"], 4 * g["fill_sms"], graph=False)
    else:
        step = torch.cuda.Stream(priority=-5)
        if parts[0] == "share":
            s = C.c_void_p()
            if lib.ifill_stream(0, C.byref(s)):
                raise SystemExit("ifill_stream failed")
            fill = Filler(args.lib, sms, sms, int(parts[1]), args.iters, args.depth, 0, s.value)
    torch.cuda.set_stream(step)
    result["step_stream"] = step.cuda_stream
    if fill:
        result["filler"] = {"kernel": "filler.cu base-u1 (FT = 4 m16n8k32 s8, duo-φ receipts at L = 32)", "grid": fill.grid,
                            "fw": fill.fw, "iters": fill.iters, "depth": fill.depth, "smem": fill.smem}
        result["filler"]["alone_before"] = alone(fill, smp, args.alone_s, sms)
        SF.log("filler alone", json.dumps(result["filler"]["alone_before"]))

    import vllm
    from verity_vllm.engine.engine_profile import get_scheduler
    from verity_vllm.engine.hooks import Hooks
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    hooks = Hooks(tag="filler-slack")
    seen = set()

    def runner_hook(orig):
        def execute_model(self, *a, **kw):
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
    cc = {"cudagraph_capture_sizes": SF.CAPTURE, "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    kw = {"model": args.model, "quantization": "fp8", "max_num_seqs": max(SF.CAPTURE), "enable_prefix_caching": False,
          "seed": 0, "max_model_len": max(map(max, contexts)) + steps_cap + 64, "max_num_batched_tokens": 2048,
          "gpu_memory_utilization": args.mem, "enforce_eager": False, "compilation_config": cc}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    result.update(vllm=vllm.__version__, torch=torch.__version__, startup_s=time.perf_counter() - t0, compilation=cc)
    hf = llm.llm_engine.vllm_config.model_config.hf_config
    result["model_shape"] = {"layers": hf.num_hidden_layers, "hidden": hf.hidden_size, "intermediate": hf.intermediate_size,
                             "heads": hf.num_attention_heads, "kv_heads": hf.num_key_value_heads, "vocab": hf.vocab_size}
    rec: list = []

    def make(orig):
        def update_from_output(so, mro):
            rec.append((time.perf_counter(), so.total_num_scheduled_tokens, len(so.num_scheduled_tokens)))
            return orig(so, mro)
        return update_from_output

    hooks.wrap(get_scheduler(llm), "update_from_output", make)
    out = Path(args.out)

    def save() -> None:
        out.write_text(json.dumps(result, indent=1))

    sp = SamplingParams(temperature=0.0, max_tokens=args.gate_tokens, logprobs=1, seed=0)
    if fill:
        fill.start()
    for o in llm.generate(SF.GATE_PROMPTS, sp, use_tqdm=False):
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
            s0 = fill.stats() if fill else None
            t, tm = time.perf_counter(), time.monotonic()
            llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True, detokenize=False),
                         use_tqdm=False)
            wall, tm1 = time.perf_counter() - t, time.monotonic()
            point = {"batch": b, "context": ctx, **({"mix": mix} if len(mix) > 1 else {}), **SF.steady(rec, b), "wall_s": wall,
                     "max_tokens": mt, "nvml": smp.window(tm, tm1)}
            if fill:
                point["filler"] = delta(fill, s0, fill.stats(), tm, tm1, smp, sms)
            result["points"].append(point)
            save()
            SF.log("point", json.dumps(point))
    if fill:
        s0, tm = fill.stats(), time.monotonic()
        time.sleep(args.alone_s)
        result["filler"]["beside_idle_engine"] = delta(fill, s0, fill.stats(), tm, time.monotonic(), smp, sms)
        n = fill.stop()
        result["filler"]["chunks_total"] = n
        result["filler"]["check"] = fill.check(min(args.check_threads, fill.grid * fill.fw * 32))
        SF.log("filler check", result["filler"]["check"])
    else:
        tm = time.monotonic()
        time.sleep(min(args.alone_s, 2.0))
        result["idle_engine"] = smp.window(tm, time.monotonic())
    result["clocks_event_reasons"] = smp.reasons()
    result["streams_seen"] = sorted(seen)
    hooks.uninstall()
    smp.stop.set()
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
