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
                 dynamic per-token int8 activations), so vLLM serves them with its own kernels (scaled_int8_quant +
                 cutlass_scaled_mm), compiled with CUDA graphs (vLLM's defaults)
    exact-eager  served_h1's engine without the option: `engine.build.build_engine` on the int8-exact profile (int8_exact=all),
                 as SERVE-H1 runs it (eager, batch-invariant)
    dither       stock with Dither's served H1 + fx1 path (`dither_arm.install`), when given

    python sbench.py --out OUT [--arms stock,dither] [--reps 3]
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
    return {"max_num_seqs": MAX_SEQS, "max_num_batched_tokens": MAX_TOKENS_STEP, "gpu_memory_utilization": gpu_mem,
            "enable_prefix_caching": False, "max_model_len": 2048, "seed": 0}


def build(arm: str, scratch: Path, gpu_mem: float):
    from vllm import LLM
    if arm in ("stock", "dither"):
        llm = LLM(model=str(stock_checkpoint(PIN, scratch)), **engine_args(gpu_mem))
        if arm == "dither":
            sys.path.insert(0, str(HERE))
            import dither_arm
            dither_arm.install(llm)
        return llm
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
    raise ValueError(arm)


def worker(arm: str, conn, scratch: str, gpu_mem: float) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import SamplingParams
    t0 = time.perf_counter()
    try:
        llm = build(arm, Path(scratch), gpu_mem)
    except Exception as e:  # noqa: BLE001
        conn.send({"error": f"{type(e).__name__}: {e}"})
        return
    conn.send({"ready": True, "build_s": round(time.perf_counter() - t0, 1)})
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
    a = ap.parse_args()
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
    ref_tokens = None
    gate = {}
    for r in range(a.reps):
        for op in ("decode", "sat"):
            for j in range(len(arms)):
                arm = arms[(j + r) % len(arms)]
                conns[arm].send({"op": op})
                res = conns[arm].recv()
                stats = conns[arm].recv()["stats"]
                toks = res.pop("tokens", None)
                if toks is not None:
                    if arm == arms[0] and ref_tokens is None:
                        ref_tokens = toks
                    gate.setdefault(arm, []).append(toks == ref_tokens if ref_tokens is not None else None)
                row = {"rep": r, "op": op, "arm": arm, **res, "stats": stats}
                reps.append(row)
                print(json.dumps(row), flush=True)
                with open(out / "reps.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
    for arm in arms:
        conns[arm].send({"op": "stop"})
    summary = {"arms": arms, "info": info, "gate_tokens_equal_first_arm": gate, "n1": N1, "n2": N2, "b": B,
               "saturation": {"requests": SAT_REQS, "tokens": SAT_TOKENS, "max_num_seqs": MAX_SEQS,
                              "max_num_batched_tokens": MAX_TOKENS_STEP}, "per_arm": {}}
    for arm in arms:
        tp = [x["tpot_b32_ms"] for x in reps if x["arm"] == arm and x["op"] == "decode"]
        th = [x["tokens_per_s"] for x in reps if x["arm"] == arm and x["op"] == "sat"]
        summary["per_arm"][arm] = {"tpot_b32_ms": statistics.median(tp), "tpot_all": tp,
                                   "tokens_per_s": statistics.median(th), "tokens_per_s_all": th}
    base = summary["per_arm"][arms[0]]
    for arm in arms[1:]:
        p = summary["per_arm"][arm]
        p["tpot_overhead_pct"] = round(100 * (p["tpot_b32_ms"] / base["tpot_b32_ms"] - 1), 3)
        p["throughput_overhead_pct"] = round(100 * (1 - p["tokens_per_s"] / base["tokens_per_s"]), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["per_arm"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
