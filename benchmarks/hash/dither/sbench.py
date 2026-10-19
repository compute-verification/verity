"""Dither's served standard number on one GPU: each arm a resident vLLM engine in its own worker process on the same GPU,
the parent alternating reps between them (arm order rotated each rep), on int8-exact Llama-3.1-8B (the verity-int8 pin,
CA's served_h1 row) and served_h1's prompts (`tests.engine.int8_exact_gpu.PROMPTS`, cycled).

Measures, per arm and rep:
    tpot_b32_ms      decode at batch 32: 32 requests, greedy, ignore_eos; (t(N2) - t(N1)) / (N2 - N1) over two generate calls
                     of N1 and N2 output tokens, so prefill and the engine's per-call overhead cancel
    tokens_per_s     at saturation: SAT_REQS requests of SAT_TOKENS output tokens each, max_num_seqs MAX_SEQS (requests wait
                     in the scheduler throughout), output tokens over wall time
and the gate: each arm's greedy token ids on the 32 requests equal `stock`'s (an arm that changes no served value must).

Arms:
    stock        native vLLM int8: the pin's weights and scales under a compressed-tensors W8A8 config (int8 channel weights,
                 dynamic per-token int8 activations), so vLLM serves them with its own kernels, compiled with CUDA graphs
                 (vLLM's defaults).  On sm_120 vLLM's first int8 choice, CutlassInt8ScaledMMLinearKernel, refuses at its first
                 call ("Int8 not supported on SM120"), so the arm disables it (VLLM_DISABLED_KERNELS) and vLLM's selector takes
                 its next, TritonInt8ScaledMMLinearKernel (scaled_int8_quant + triton_scaled_mm)
    stock-humming  the same with Triton disabled too: vLLM's third choice, HummingInt8ScaledMMLinearKernel
    exact-eager  served_h1's engine without the option: `engine.build.build_engine` on the int8-exact profile (int8_exact=all),
                 as SERVE-H1 runs it (eager, batch-invariant)
    exact        the int8-exact linears (`verity_int8::row_quant` + `verity_int8::scaled_mm`, torch._int_mm) compiled with
                 CUDA graphs, attention vLLM's: today's served y in a compiled engine (built here, outside `build_engine`,
                 which runs int8_exact eagerly only; both ops are custom ops, opaque to Inductor)
    dither       exact with every linear's y from Dither's fused GEMM (`dither_fused`: z = xq W^T and its dequantization in
                 one kernel that also computes the previous step's receipts; A, admit and z tables, the round, the live
                 coin and the seeds between steps)
    dither-side  dither with the receipts in one `h1r` launch a step on a second stream, after the step's coin, beside the
                 next step (`dither_fused.patch(rc="side")`); the `h1f` launches compute z and y only
    dither-1s    exact with every linear's y from Dither's one stream (`dither_one`, `h1o.cu`, CA's H1V3): y = dequant(O' -
                 sum_d 128^d D_d) from the noised mainloop the call's receipts fold, z that O; the call's A table, seeds
                 (the window's salt and root(A_j), root(B_j): epsilon 'seeded') and units in the forward; admit and z
                 tables, the step trees, the round and the receipt trees between steps.  It serves the dithered product, so
                 its tokens are not exact's (reported, not gated)
    dither-1s-credit  dither-1s on an eager engine with every linear's `apply` inside CA's credit gate (#2684's
                 `credit_gate.Trace`, the file DITHER_CREDIT_GATE names): the gate reports its verdicts (`credit_2684`)

`gate` (--gate, before the reps): exact and dither each generate GATE_TOKENS tokens for the 32 requests; dither's window
checks one B32 decode step (`dither_fused.check`, its transcript buffers 0xA5 before the kernels wrote them) and refuses
its no-write control (a later B32 step whose kernels write none of them), and its tokens must equal exact's (dither-1s:
`dither_one.check` on its gate, control, fault and fault_wb steps, tokens reported), under a CUDA
profiler range and the NVTX range GATE_RANGE/<arm> (nsys --capture-range=cudaProfilerApi; kernels.py counts the dither worker's
kernels inside the range only).  The decode gate compares each arm's
tokens with its reference arm's (dither: exact; any other: the first arm).  A failed gate times nothing (exit 4).

    python sbench.py --out OUT [--arms stock,exact,dither] [--reps 3] [--t 64] [--ops decode] [--gate]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7"
B = 32
N1, N2 = 32, 160
SAT_REQS, SAT_TOKENS, MAX_SEQS, MAX_TOKENS_STEP = 768, 256, 256, 1024
CT_CONFIG = {
    "quant_method": "compressed-tensors", "format": "int-quantized", "quantization_status": "compressed",
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False,
                    "group_size": None, "observer": "minmax", "actorder": None, "block_structure": None},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True,
                              "group_size": None, "observer": None, "actorder": None, "block_structure": None}}},
    "ignore": ["lm_head"], "kv_cache_scheme": None}


def prompts(n: int) -> list[str]:
    sys.path.insert(0, str(ROOT / "integrations" / "vllm"))
    from tests.engine.int8_exact_gpu import PROMPTS
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


def stock_checkpoint(pin: str, scratch: Path) -> Path:
    """The pin's files linked into SCRATCH, its config's quantization_config replaced by compressed-tensors W8A8's."""
    d = scratch / "stock-ckpt"
    d.mkdir(parents=True, exist_ok=True)
    for f in Path(pin).iterdir():
        if f.name != "config.json" and not (d / f.name).exists():
            (d / f.name).symlink_to(f.resolve())
    cfg = json.loads((Path(pin) / "config.json").read_text())
    cfg["quantization_config"] = CT_CONFIG
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    return d


def engine_args(gpu_mem: float) -> dict:
    # FlashInfer's autotune off, as `engine.build` sets it for the int8-exact engine (it tunes no kernel these engines run)
    # vLLM refuses max_num_seqs above max_num_batched_tokens; every arm of a run gets the same cap
    t = int(os.environ.get("SBENCH_T", MAX_TOKENS_STEP))
    return {"max_num_seqs": min(MAX_SEQS, t), "max_num_batched_tokens": t,
            "gpu_memory_utilization": gpu_mem,
            "enable_prefix_caching": False, "max_model_len": 2048, "seed": 0,
            "kernel_config": {"enable_flashinfer_autotune": False}}


DISABLED = {"stock": "CutlassInt8ScaledMMLinearKernel",
            "stock-humming": "CutlassInt8ScaledMMLinearKernel,TritonInt8ScaledMMLinearKernel"}


def kernels_of(llm) -> dict:
    """The linear kernels the engine's layers really run: class name -> layers."""
    from verity_vllm.engine import engine_profile as prof
    seen: dict = {}
    for _, mod in prof.get_model_runner(llm).get_model().named_modules():
        k = getattr(getattr(mod, "scheme", None), "kernel", None)
        q = getattr(mod, "quant_method", None)
        name = type(k).__name__ if k is not None else (type(q).__name__ if q is not None and hasattr(mod, "weight") else None)
        if name:
            seen[name] = seen.get(name, 0) + 1
    return seen


def build(arm: str, scratch: Path, gpu_mem: float):
    from vllm import LLM
    if arm in DISABLED:
        os.environ["VLLM_DISABLED_KERNELS"] = DISABLED[arm]
    if arm in ("stock", "stock-humming"):
        return LLM(model=str(stock_checkpoint(PIN, scratch)), **engine_args(gpu_mem))
    if arm == "exact-eager":
        vllm_dir = ROOT / "integrations" / "vllm"
        sys.path.insert(0, str(vllm_dir))
        from tests.engine import int8_exact_gpu as G
        from verity_vllm.engine.build import build_engine
        from verity_vllm.program.frontend.target_profile import TargetProfile
        target = TargetProfile.from_device().with_overrides(int8_exact="all")
        ns = argparse.Namespace(checkpoints=str(vllm_dir / "manifests" / "checkpoints.json"), role="LLAMA31_8B_INT8",
                                base_role="LLAMA31_8B", repo="verity-int8/Llama-3.1-8B-INT8",
                                hf_home=os.environ.get("HF_HOME"), by="kernels", at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        entry, _ = G.pin(ns)
        ckpts = G.scratch_checkpoints(ns.checkpoints, entry, scratch)
        a = engine_args(gpu_mem)
        eng = build_engine(ckpts, case=ns.role, target=target, max_num_seqs=a.pop("max_num_seqs"), engine_args=a)
        return eng.llm
    if arm in ("exact", "dither", "dither-side"):
        sys.path.insert(0, str(HERE))
        import dither_fused as DF
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
        a = engine_args(gpu_mem)
        if arm != "exact":
            DF.patch(t=a["max_num_batched_tokens"], h1c=os.environ["DITHER_H1C"], h1f=os.environ["DITHER_H1F"], gate=True,
                     rc="side" if arm == "dither-side" else "fused")
        llm = LLM(model=str(DF.int8_checkpoint(PIN, scratch)), **a)
        if arm != "exact":
            win = DF.hook(llm)
            llm.dither_stats = win.stats
            llm.dither_win = win
        return llm
    if arm in ("dither-1s", "dither-1s-credit"):
        sys.path.insert(0, str(HERE))
        import dither_fused as DF
        import dither_one as DO
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
        a = engine_args(gpu_mem)
        credit = os.environ["DITHER_CREDIT_GATE"] if arm == "dither-1s-credit" else None
        if credit:
            a["enforce_eager"] = True
        DO.patch(t=a["max_num_batched_tokens"], h1c=os.environ["DITHER_H1C"], h1o=os.environ["DITHER_H1O"], gate=True,
                 credit=credit)
        llm = LLM(model=str(DF.int8_checkpoint(PIN, scratch)), **a)
        win = DO.hook(llm)
        llm.dither_stats, llm.dither_win, llm.dither_mod = win.stats, win, DO
        return llm
    raise ValueError(arm)


REF_OF = {"dither": "exact", "dither-side": "exact", "dither-1s": "exact", "dither-1s-credit": "exact"}
# an arm that serves another value: its tokens are compared, not required equal
TOKENS_GATED = {"dither-1s": False, "dither-1s-credit": False}
GATE_TOKENS = 16    # enough B32 decode steps for the gate's capture, its receipts, the control's and its receipts
GATE_RANGE = "verity/dither/gate"   # GATE_RANGE/<arm>: kernels.py reads each dither worker's kernels inside its own


def worker(arm: str, conn, scratch: str, gpu_mem: float) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from vllm import SamplingParams
    if arm.startswith("dither"):
        sys.path.insert(0, str(HERE))
        import dither_fused as DF
    t0 = time.perf_counter()
    try:
        llm = build(arm, Path(scratch), gpu_mem)
    except Exception as e:  # noqa: BLE001
        conn.send({"error": f"{type(e).__name__}: {e}"})
        return
    conn.send({"ready": True, "build_s": round(time.perf_counter() - t0, 1), "kernels": kernels_of(llm),
               "disabled": os.environ.get("VLLM_DISABLED_KERNELS")})
    ps = prompts(B)

    def gen(reqs: list[str], n: int):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, seed=0)
        t = time.perf_counter()
        outs = llm.generate(reqs, sp, use_tqdm=False)
        return time.perf_counter() - t, outs

    while True:
        cmd = conn.recv()
        if cmd["op"] == "stop":
            conn.send({"stopped": True})
            return
        if cmd["op"] == "warm":
            gen(ps, 8)
            gen(prompts(64), 16)
            conn.send({"warm": True})
        elif cmd["op"] == "decode":
            t1, _ = gen(ps, N1)
            t2, outs = gen(ps, N2)
            conn.send({"tpot_b32_ms": round(1e3 * (t2 - t1) / (N2 - N1), 4), "t_n1_s": round(t1, 4), "t_n2_s": round(t2, 4),
                       "tokens": [list(o.outputs[0].token_ids) for o in outs]})
        elif cmd["op"] == "gate":
            win = getattr(llm, "dither_win", None)
            mod = getattr(llm, "dither_mod", None) or (DF if win is not None else None)
            if win is not None:
                if mod is DF:
                    win.arm(B)
                    win.price_y = True
                else:
                    win.arm(B, mod.CAPTURES)
            prof = os.environ.get("DITHER_NSYS") == "1" and win is not None
            if prof:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(f"{GATE_RANGE}/{arm}")
            t, outs = gen(ps, GATE_TOKENS)
            if prof:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
            res = {"gate_s": round(t, 3), "tokens": [list(o.outputs[0].token_ids) for o in outs]}
            if win is not None:
                t0 = time.perf_counter()
                if mod is DF:
                    win.price_y = False
                    with torch.inference_mode():
                        res["check"] = DF.check(win, os.environ["DITHER_H1F"], "gate")
                        res["control"] = DF.check(win, os.environ["DITHER_H1F"], "control")
                    res["verdict"] = DF.gate_verdict(res["check"], res["control"])
                else:
                    lib = os.environ[mod.LIB_ENV]
                    with torch.inference_mode():
                        caps = {n: mod.check(win, lib, n) for n in mod.CAPTURES}
                    res["check"] = caps.pop("gate")
                    res.update(caps)
                    res["verdict"] = mod.gate_verdict(res["check"], *(caps[n] for n in mod.CAPTURES[1:]))
                    if getattr(mod, "CG", None) is not None:
                        res["credit_2684"] = mod.credit_doc()
                res["check_s"] = round(time.perf_counter() - t0, 1)
            conn.send(res)
        elif cmd["op"] == "sat":
            t, outs = gen(prompts(SAT_REQS), SAT_TOKENS)
            n = sum(len(o.outputs[0].token_ids) for o in outs)
            conn.send({"tokens_per_s": round(n / t, 2), "sat_s": round(t, 3), "sat_tokens": n})
        if hasattr(llm, "dither_stats"):
            conn.send({"stats": llm.dither_stats()})
        else:
            conn.send({"stats": None})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="stock,exact-eager")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--t", type=int, default=MAX_TOKENS_STEP, help="max_num_batched_tokens: the Dither window's T")
    ap.add_argument("--ops", default="decode,sat")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    os.environ["SBENCH_T"] = str(a.t)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import tempfile
    scratch = Path(a.scratch or tempfile.mkdtemp(prefix="sbench."))
    arms = a.arms.split(",")
    ctx = mp.get_context("spawn")
    conns, procs, info = {}, {}, {}
    for arm in arms:      # one engine at a time comes up, so their memory profiling doesn't race
        here, there = ctx.Pipe()
        p = ctx.Process(target=worker, args=(arm, there, str(scratch), a.gpu_mem), daemon=True)
        p.start()
        msg = here.recv()
        if "error" in msg:
            print(json.dumps({"arm": arm, **msg}), flush=True)
            return 3
        conns[arm], procs[arm], info[arm] = here, p, msg
        print(json.dumps({"arm": arm, **msg}), flush=True)
    for arm in arms:
        conns[arm].send({"op": "warm"})
        conns[arm].recv(), conns[arm].recv()
    reps: list[dict] = []
    ref_tokens: dict = {}
    gate = {}
    gate_doc: dict = {}
    if a.gate:
        for arm in arms:
            conns[arm].send({"op": "gate"})
            res = conns[arm].recv()
            res["stats"] = conns[arm].recv()["stats"]
            gate_doc[arm] = res
        for arm in arms:
            ref = REF_OF.get(arm, arms[0])
            if ref in gate_doc and ref != arm:
                gate_doc[arm]["tokens_equal_" + ref] = gate_doc[arm]["tokens"] == gate_doc[ref]["tokens"]
        for arm in arms:
            print(json.dumps({"gate": arm, **{k: v for k, v in gate_doc[arm].items() if k != "tokens"}}), flush=True)
        (out / "gate.json").write_text(json.dumps(gate_doc, indent=1) + "\n")
        failed = [arm for arm in arms if "verdict" in gate_doc[arm] and not (
            gate_doc[arm]["verdict"]["ok"] and (gate_doc[arm].get("tokens_equal_" + REF_OF.get(arm, arms[0]))
                                                or not TOKENS_GATED.get(arm, True)))]
        if failed:
            print(json.dumps({"gate_failed": failed, "timed": False}), flush=True)
            for arm in arms:
                conns[arm].send({"op": "stop"})
            return 4
    for r in range(a.reps):
        for op in a.ops.split(","):
            for j in range(len(arms)):
                arm = arms[(j + r) % len(arms)]
                conns[arm].send({"op": op})
                res = conns[arm].recv()
                stats = conns[arm].recv()["stats"]
                toks = res.pop("tokens", None)
                if toks is not None:
                    ref_tokens.setdefault((r, arm), toks)
                row = {"rep": r, "op": op, "arm": arm, **res, "stats": stats}
                reps.append(row)
                print(json.dumps(row), flush=True)
                with open(out / "reps.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
        for arm in arms:
            if (r, arm) in ref_tokens:
                ref = REF_OF.get(arm, arms[0])
                gate.setdefault(arm, []).append(None if ref == arm else ref_tokens[(r, arm)] == ref_tokens.get((r, ref)))
    for arm in arms:
        conns[arm].send({"op": "stop"})
    summary = {"arms": arms, "info": info, "gate_tokens_equal_ref_arm": gate, "ref_of": REF_OF, "n1": N1, "n2": N2, "b": B,
               "gate": {arm: {k: v for k, v in d.items() if k != "tokens"} for arm, d in gate_doc.items()},
               "saturation": {"requests": SAT_REQS, "tokens": SAT_TOKENS, "max_num_seqs": min(MAX_SEQS, a.t),
                              "max_num_batched_tokens": a.t}, "per_arm": {}}
    for arm in arms:
        tp = [x["tpot_b32_ms"] for x in reps if x["arm"] == arm and x["op"] == "decode"]
        th = [x["tokens_per_s"] for x in reps if x["arm"] == arm and x["op"] == "sat"]
        summary["per_arm"][arm] = {"tpot_b32_ms": statistics.median(tp) if tp else None, "tpot_all": tp,
                                   "tokens_per_s": statistics.median(th) if th else None, "tokens_per_s_all": th}
    base = summary["per_arm"][arms[0]]
    for arm in arms[1:]:
        p = summary["per_arm"][arm]
        if p["tpot_b32_ms"] and base["tpot_b32_ms"]:
            p["tpot_overhead_pct"] = round(100 * (p["tpot_b32_ms"] / base["tpot_b32_ms"] - 1), 3)
        if p["tokens_per_s"] and base["tokens_per_s"]:
            p["throughput_overhead_pct"] = round(100 * (1 - p["tokens_per_s"] / base["tokens_per_s"]), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["per_arm"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
