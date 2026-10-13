"""vLLM's decode step of Llama-3.1-8B (the engine `serve.py` serves: FP8 online, full CUDA graphs for decode) at each
batch × context of `--shapes`, its kernels timed by torch's profiler, and the step's own fraction of R split by kernel
class: the GEMMs (the four linears in FP8 and the LM head) against the weights' bytes, attention against the KV it
reads (each request's context, 128 KiB a token), and the rest (norms, rotary, activation quantization, sampling,
copies) against nothing registered. `native` re-times the step with the GEMMs at `--native-R` of R (ra-tag-fast's
native linears: 6.98 GB in 4.71 ms at M = 1, 0.96 R) and the rest as measured; `native_all` puts attention at that
rate too. Writes `step_profile.json` and `step_profile.log` under `--out`. `--stack stack|pstack` profiles the encoded
stack's step instead (`stack_hook.py`): its fused decode + GEMM + receipts and its lm_head's decode are GEMMs against its
stores, paged.cu's attention, KV append and gather are attention, and its receipt tree is among the rest. `--prefill N`
then profiles one prompt of N tokens alone, each of its chunks of `--max-batched-tokens` a step of its own; every step
records its wall time beside the kernels'.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve import CAPTURE, KV_TOKEN_BYTES

# Attention is matched first: FlashAttention's kernels are cutlass templates named `flash_fwd_splitkv`.
CLASSES = (("attention", re.compile(r"flash|attn|attention|fmha|batchdecode|batchprefill|mergestate|merge_state|pg_append|"
                                    r"pg_gather|pg_combine", re.IGNORECASE)),
           ("gemm", re.compile(r"gemm|gemv|cutlass|xmma|cublas|nvjet|matmul|scaled_mm|splitk|wt::|(^|\W)fused_kernel|"
                               r"(^|\W)decode_kernel|pg_reduce", re.IGNORECASE)))


def classify(name: str) -> str:
    return next((c for c, rx in CLASSES if rx.search(name)), "other")


def split(kernels: list[dict], steps: int, weight_bytes: int, kv_bytes: float, R: float, native: float) -> dict:
    """Each class's ms a step from the profiler's kernel events (`name`, `ts`, `dur` in µs) over `steps` model steps,
    and the step's fraction of R measured and with the GEMMs (`native`) or every registered read (`native_all`) at
    `native` of R."""
    ms = {c: sum(e["dur"] for e in kernels if classify(e["name"]) == c) / 1e3 / steps for c in ("gemm", "attention",
                                                                                               "other")}
    kern = sum(ms.values())
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1e3 / steps
    rate = R * 1e6  # bytes a millisecond
    gemm_native = weight_bytes / (native * rate)
    native_ms = gemm_native + ms["attention"] + ms["other"]
    native_all_ms = (weight_bytes + kv_bytes) / (native * rate) + ms["other"]
    busy = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    gaps, end = [], busy[0][1]
    for a, b in busy[1:]:
        if a > end:
            gaps.append(a - end)
        end = max(end, b)
    inside = [g for g in gaps if g < 1000]  # µs; a gap of a millisecond or more is the host between two steps
    by = {}
    for e in kernels:
        by.setdefault(e["name"], 0.0)
        by[e["name"]] += e["dur"] / 1e3 / steps
    return {"ms": ms, "kernel_ms": kern, "span_ms": span, "kernels_a_step": len(kernels) / steps,
            "of_R": (weight_bytes + kv_bytes) / (rate * kern), "of_R_span": (weight_bytes + kv_bytes) / (rate * span),
            "gemm_of_R": weight_bytes / (rate * ms["gemm"]) if ms["gemm"] else None,
            "attention_of_R": kv_bytes / (rate * ms["attention"]) if ms["attention"] else None,
            "other_share": ms["other"] / kern, "native_ms": native_ms,
            "idle_in_step_ms": sum(inside) / 1e3 / steps, "idle_in_step_gaps": len(inside) / steps,
            "idle_between_steps_ms": sum(g for g in gaps if g >= 1000) / 1e3 / steps,
            "native_of_R": (weight_bytes + kv_bytes) / (rate * native_ms), "native_all_ms": native_all_ms,
            "native_all_of_R": (weight_bytes + kv_bytes) / (rate * native_all_ms),
            "kernels": [{"name": n[:160], "class": classify(n), "ms": round(t, 4)}
                        for n, t in sorted(by.items(), key=lambda kv: -kv[1])]}


def traced(prof) -> list[dict]:
    """A torch profile's kernel events (`name`, `ts`, `dur` in µs)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(path))
        return [e for e in json.loads(path.read_text())["traceEvents"] if e.get("cat") == "kernel" and e.get("ph") == "X"]


NCU_METRICS = ("gpu__time_duration.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
               "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum", "dram__cycles_active.sum",
               "dram__cycles_elapsed.sum")


def ncu_rows(text: str) -> list[dict]:
    """`ncu --import … --csv --page raw --print-units base`: one dict a kernel, its name and NCU_METRICS as floats."""
    import csv
    import io
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if not (r.get("ID") or "").strip().isdigit():
            continue  # the units row
        rows.append({"name": r["Kernel Name"], **{m: float(r[m].replace(",", "")) for m in NCU_METRICS}})
    return rows


def ncu_split(rows: list[dict], steps: int, weight_bytes: float, kv_bytes: float, R: float) -> dict:
    """Each class's DRAM bytes a step against the bytes registered for it (the GEMMs: the weights; attention: the KV;
    the rest: nothing), and the bus time inside kernels that their traffic leaves (duration × R minus reads and
    writes)."""
    reg = {"gemm": weight_bytes, "attention": kv_bytes, "other": 0.0}
    out = {}
    for c in ("gemm", "attention", "other", "all"):
        rs = [r for r in rows if c == "all" or classify(r["name"]) == c]
        rd = sum(r["dram__bytes_op_read.sum"] for r in rs) / steps
        wr = sum(r["dram__bytes_op_write.sum"] for r in rs) / steps
        ns = sum(r["gpu__time_duration.sum"] for r in rs) / steps
        registered = sum(reg.values()) if c == "all" else reg[c]
        out[c] = {"kernels": len(rs) / steps, "ms": ns / 1e6, "dram_read_GB": rd / 1e9, "dram_write_GB": wr / 1e9,
                  "l2_read_GB": sum(r["lts__t_sectors_op_read.sum"] for r in rs) * 32 / steps / 1e9,
                  "l2_write_GB": sum(r["lts__t_sectors_op_write.sum"] for r in rs) * 32 / steps / 1e9,
                  "registered_GB": registered / 1e9, "read_over_registered": rd / registered if registered else None,
                  "dram_GBps": (rd + wr) / ns if ns else None,
                  "dram_active": (sum(r["dram__cycles_active.sum"] for r in rs) / sum(r["dram__cycles_elapsed.sum"]
                                                                                  for r in rs)) if rs else None,
                  "bus_left_GB": (ns * 1e-9 * R * 1e9 - rd - wr) / 1e9}
    by: dict = {}
    for r in rows:
        k = by.setdefault(r["name"][:160], {"class": classify(r["name"]), "n": 0, "ms": 0.0, "read_GB": 0.0,
                                            "write_GB": 0.0})
        k["n"] += 1
        k["ms"] += r["gpu__time_duration.sum"] / 1e6 / steps
        k["read_GB"] += r["dram__bytes_op_read.sum"] / 1e9 / steps
        k["write_GB"] += r["dram__bytes_op_write.sum"] / 1e9 / steps
    out["kernels"] = [{"name": n, **v} for n, v in sorted(by.items(), key=lambda kv: -kv[1]["ms"])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="/workspace/pouw/mvp-e2e/models/llama-3.1-8b-instruct")
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--shapes", type=lambda s: [tuple(int(v) for v in x.split("x")) for x in s.split(",")],
                    default=[(1, 2048), (8, 4096), (64, 2048), (128, 1024)])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--prefill", type=int, default=0,
                    help="after the shapes, one prompt of this many tokens alone, each of its prefill chunks "
                         "(--max-batched-tokens) profiled as a step of its own")
    ap.add_argument("--R", type=float, default=1540.0)
    ap.add_argument("--native-R", type=float, default=0.96)
    ap.add_argument("--ncu", action="store_true",
                    help="bracket the measured steps with cudaProfilerStart/Stop for `ncu --profile-from-start off`, "
                         "no torch profiler; writes step_profile-ncu.json")
    ap.add_argument("--from-ncu", type=Path,
                    help="no vLLM: split the kernels of `ncu --import … --csv --page raw --print-units base` against "
                         "the bytes step_profile-ncu.json registers; writes step_profile-ncu-split.json")
    ap.add_argument("--stack", choices=("pstack", "stack"), default=None, help="the encoded stack's step (stack_hook.py)")
    ap.add_argument("--stack-k", type=int, default=3)
    ap.add_argument("--wt-lib")
    ap.add_argument("--reader-lib")
    ap.add_argument("--paged-lib")
    a = ap.parse_args()
    if a.stack and not (a.wt_lib and a.reader_lib and a.paged_lib):
        ap.error("--stack takes --wt-lib, --reader-lib and --paged-lib")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.from_ncu:
        meta = json.loads((a.out / "step_profile-ncu.json").read_text())
        rows = ncu_rows(a.from_ncu.read_text())
        s = meta["shapes"][0]
        res = {**meta, "split": ncu_split(rows, s["steps"], meta["weight_bytes_per_step"], s["kv_bytes_per_step"], a.R)}
        (a.out / "step_profile-ncu-split.json").write_text(json.dumps(res, indent=1) + "\n")
        for c in ("gemm", "attention", "other", "all"):
            v = res["split"][c]
            print(f"{c}: {v['kernels']:.0f} kernels, {v['ms']:.3f} ms; DRAM read {v['dram_read_GB']:.3f} GB against "
                  f"{v['registered_GB']:.3f} registered ({v['read_over_registered'] or 0:.3f}), write "
                  f"{v['dram_write_GB']:.3f} GB; L2 read {v['l2_read_GB']:.3f} GB; {v['dram_GBps'] or 0:.0f} GB/s; "
                  f"DRAM active {v['dram_active'] or 0:.3f}; bus left inside {v['bus_left_GB']:.3f} GB", flush=True)
        return 0
    logf = (a.out / "step_profile.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch
    import vllm
    from torch.profiler import ProfilerActivity, profile
    from verity_vllm.engine.engine_profile import get_model_runner
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    stack, capture = None, CAPTURE
    if a.stack:
        import stack_hook
        stack = stack_hook.install(a.stack, a.stack_k, a.wt_lib, a.reader_lib, a.paged_lib)
        capture = stack[2]
    llm = LLM(model=a.model, quantization="fp8", max_num_seqs=max(CAPTURE), enable_prefix_caching=False, seed=0,
              max_model_len=max([c for _, c in a.shapes] + [a.prefill]) + a.steps + 128,
              max_num_batched_tokens=a.max_batched_tokens,
              enforce_eager=False, gpu_memory_utilization=a.mem,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": capture})
    eng = llm.llm_engine
    runner = get_model_runner(llm)
    weight_bytes = sum(p.numel() * p.element_size() for n, p in runner.model.named_parameters()
                       if "embed_tokens" not in n)
    if stack:
        weight_bytes = stack_hook.weight_bytes(stack[1], runner.model)
    vocab = eng.vllm_config.model_config.hf_config.vocab_size
    count = [0]
    ex_orig = runner.execute_model

    def execute_model(*args, **kwargs):
        count[0] += 1
        return ex_orig(*args, **kwargs)

    runner.execute_model = execute_model
    res = {"format": "verity-kv-filler/step-profile/v0", "args": {k: v for k, v in vars(a).items() if k != "out"},
           "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
           "weight_bytes_per_step": weight_bytes, "stack": a.stack, "cudagraph_capture_sizes": capture,
           "stack_install": {k: v for k, v in stack[1].record.items() if k != "selection"} if stack else None,
           "async_scheduling": getattr(eng.vllm_config.scheduler_config, "async_scheduling", None), "shapes": []}
    log(f"weights read a step {weight_bytes / 1e9:.2f} GB; async scheduling {res['async_scheduling']}")
    rng = random.Random(3)
    for batch, ctx in a.shapes:
        ids = [f"p{batch}x{ctx}-{i}" for i in range(batch)]
        for rid in ids:
            eng.add_request(rid, TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000) for _ in range(ctx)]),
                            SamplingParams(max_tokens=a.steps + 64, ignore_eos=True, temperature=0.0))
        made: dict[str, int] = {}
        while len(made) < batch or min(made.values()) < 4:
            for o in eng.step():
                made[o.request_id] = len(o.outputs[0].token_ids)
        torch.cuda.synchronize()
        count[0] = 0
        if a.ncu:
            torch.cuda.cudart().cudaProfilerStart()
            for _ in range(a.steps):
                eng.step()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            steps = count[0]
            eng.abort_request(ids)
            while eng.has_unfinished_requests():
                eng.step()
            res["shapes"].append({"batch": batch, "ctx": ctx, "steps": steps,
                                  "kv_bytes_per_step": batch * (ctx + 4 + a.steps / 2) * KV_TOKEN_BYTES})
            (a.out / "step_profile-ncu.json").write_text(json.dumps(res, indent=1) + "\n")
            log(f"{batch}x{ctx}: {steps} steps under ncu")
            continue
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.steps):
                eng.step()
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        steps = count[0]
        kernels = traced(prof)
        eng.abort_request(ids)
        while eng.has_unfinished_requests():
            eng.step()
        kv_bytes = batch * (ctx + 4 + a.steps / 2) * KV_TOKEN_BYTES
        s = {"batch": batch, "ctx": ctx, "steps": steps, "kv_bytes_per_step": kv_bytes, "wall_ms": wall * 1e3 / steps,
             **split(kernels, steps, weight_bytes, kv_bytes, a.R, a.native_R)}
        res["shapes"].append(s)
        log(f"{batch}x{ctx}: {steps} steps, {s['kernels_a_step']:.0f} kernels a step, kernels {s['kernel_ms']:.3f} ms "
            f"(span {s['span_ms']:.3f}) a step at {s['of_R']:.3f} R ({s['of_R_span']:.3f} on the span): GEMM "
            f"{s['ms']['gemm']:.3f} ms at {s['gemm_of_R'] or 0:.3f} R, attention {s['ms']['attention']:.3f} ms at "
            f"{s['attention_of_R'] or 0:.3f} R, other {s['ms']['other']:.3f} ms ({s['other_share']:.1%}); native GEMMs "
            f"{s['native_of_R']:.3f} R, every read native {s['native_all_of_R']:.3f} R; idle between kernels in a step "
            f"{s['idle_in_step_ms']:.3f} ms ({s['idle_in_step_gaps']:.0f} gaps), between steps "
            f"{s['idle_between_steps_ms']:.3f} ms")
        (a.out / "step_profile.json").write_text(json.dumps(res, indent=1) + "\n")
    if a.prefill:
        # Each chunk's step: the GEMMs against the weights, attention against the KV of the chunk's context (its own
        # tokens and the earlier chunks', each read once).
        eng.add_request("prefill", TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab - 1000)
                                                                  for _ in range(a.prefill)]),
                        SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0))
        torch.cuda.synchronize()
        res["prefill"], done = [], 0
        while eng.has_unfinished_requests():
            count[0] = 0
            t0 = time.perf_counter()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                eng.step()
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            if not count[0]:
                continue
            chunk = min(a.max_batched_tokens, a.prefill - done)
            done += chunk
            s = {"chunk": chunk, "context": done, "steps": count[0], "wall_ms": wall * 1e3,
                 **split(traced(prof), count[0], weight_bytes, done * KV_TOKEN_BYTES, a.R, a.native_R)}
            res["prefill"].append(s)
            log(f"prefill chunk of {chunk} at context {done}: wall {s['wall_ms']:.2f} ms, kernels {s['kernel_ms']:.3f} "
                f"ms (span {s['span_ms']:.3f}): GEMM {s['ms']['gemm']:.3f}, attention {s['ms']['attention']:.3f}, other "
                f"{s['ms']['other']:.3f}; idle in step {s['idle_in_step_ms']:.3f} ms")
        (a.out / "step_profile.json").write_text(json.dumps(res, indent=1) + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
