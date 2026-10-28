"""The native baseline on one GPU: vLLM serving the int8 Llama-3.1-8B pin (LLAMA31_8B_INT8, verity-int8-channel127/v1), one
arm a process, at decode B32 and at saturation.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python native.py --arm ARM --out OUT [--reps 5]

Arms:
  served-row    compute accounting's served_h1 stock: `engine.build.build_engine` on the pin with
                TargetProfile.from_device() + int8_exact (`--int8-exact`, default all, as serve.py), which runs eagerly only
                (build_engine refuses a compiled int8_exact engine: Inductor would fuse and re-round its quantizer)
  vllm-int8     vLLM's own compiled W8A8 int8 (compressed-tensors: per-channel static weights, dynamic per-token
                activations; CutlassInt8ScaledMMLinearKernel, i.e. cutlass_scaled_mm) on the same int8 weights, through a
                sibling directory whose config.json carries a compressed-tensors quantization_config; default compilation
                and CUDA graphs. A refusal (vLLM's sm_100 build has CUTLASS int8 only for sm_90) is recorded with its text
  vllm-int8-triton   the same with VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel (vLLM falls to its Triton int8)

Measures (fresh random prompts each call, prefix caching off, ignore_eos, greedy):
  decode B32    32 requests of 128 prompt tokens, generate 1 and 256 tokens, interleaved --reps times:
                TPOT = (t256 - t1) / 255 per rep (the 255 decode steps of a 32-request batch), medians
  saturation    1024 requests of 512 prompt tokens and 256 output tokens, max_num_seqs 256: output tokens/s, 2 reps
Writes OUT/native-ARM.json; the engine's selected linear kernel is read from vLLM's log ("Selected ... for ...").
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

from verity_config.settings import of

settings = of(__file__)
ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "integrations" / "vllm"
ROLE = "LLAMA31_8B_INT8"
SEQS, MODEL_LEN, BATCHED = 256, 2048, 8192

CT_CONFIG = {
    "quant_method": "compressed-tensors", "format": "int-quantized", "quantization_status": "compressed",
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False,
                    "group_size": None, "actorder": None, "block_structure": None, "observer": "minmax"},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True,
                              "group_size": None, "observer": None},
        "output_activations": None}},
    "ignore": ["lm_head"], "kv_cache_scheme": None,
}


def pin_entry() -> dict:
    doc = json.loads((VLLM_DIR / "manifests" / "checkpoints-int8.json").read_text())
    return next(c for c in doc["checkpoints"] if c.get("role") == ROLE)


def sibling(pin: Path, scratch: Path) -> Path:
    """The pin's files as symlinks, with config.json's quantization_config compressed-tensors' W8A8 int8."""
    d = scratch / "ct-int8"
    d.mkdir()
    for f in pin.iterdir():
        if f.name != "config.json":
            (d / f.name).symlink_to(f.resolve())
    cfg = json.loads((pin / "config.json").read_text())
    cfg["quantization_config"] = CT_CONFIG
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    return d


def build(arm: str, a, scratch: Path):
    from vllm import LLM
    common = {"max_num_seqs": SEQS, "max_model_len": MODEL_LEN, "max_num_batched_tokens": BATCHED,
              "enable_chunked_prefill": True, "enable_prefix_caching": False, "gpu_memory_utilization": a.gpu_mem}
    entry = pin_entry()
    if arm == "served-row":
        sys.path.insert(0, str(VLLM_DIR))
        from tests.engine import int8_exact_gpu as G
        from verity_vllm.engine.build import build_engine
        from verity_vllm import TargetProfile
        target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact)
        cps = G.scratch_checkpoints(str(VLLM_DIR / "manifests" / "checkpoints.json"), entry, scratch)
        eng = build_engine(cps, case=ROLE, target=target, max_num_seqs=SEQS,
                           engine_args={k: v for k, v in common.items() if k != "max_num_seqs"})
        return eng.llm, {"enforce_eager": eng.kwargs.get("enforce_eager"), "int8_exact": a.int8_exact,
                         "target": target.to_json(), "effective": eng.effective,
                         "engine_kwargs": {k: v for k, v in eng.kwargs.items() if isinstance(v, (int, float, str, bool))}}
    model = sibling(Path(entry["local_path"]), scratch)
    llm = LLM(model=str(model), dtype="bfloat16", seed=0, **common)
    vc = llm.llm_engine.vllm_config
    return llm, {"enforce_eager": vc.model_config.enforce_eager, "compilation": str(vc.compilation_config.mode),
                 "cudagraph_mode": str(vc.compilation_config.cudagraph_mode),
                 "disabled_kernels": settings.VLLM_DISABLED_KERNELS,
                 "flashinfer_sampler": settings.VLLM_USE_FLASHINFER_SAMPLER}


def prompts(rng: random.Random, n: int, length: int) -> list[dict]:
    return [{"prompt_token_ids": [rng.randrange(1000, 120000) for _ in range(length)]} for _ in range(n)]


def timed(llm, ps, n_out: int) -> float:
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=n_out, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(ps, sp, use_tqdm=False)
    t = time.perf_counter() - t0
    got = sum(len(o.outputs[0].token_ids) for o in outs)
    if got != n_out * len(ps):
        raise RuntimeError(f"generated {got} tokens, wanted {n_out * len(ps)}")
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["served-row", "vllm-int8", "vllm-int8-triton"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sat-reps", type=int, default=2)
    ap.add_argument("--sat-requests", type=int, default=1024)
    ap.add_argument("--int8-exact", default="all", choices=["linears", "attention", "all"])
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc: dict = {"schema": "dither-hb/native/v1", "arm": a.arm, "role": ROLE, "pin": pin_entry().get("revision"),
                 "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "numa": settings.DITHER_NUMA_RECORD,
                 "workload": {"decode": {"batch": 32, "prompt": 128, "out": [1, 256]},
                              "saturation": {"requests": a.sat_requests, "prompt": 512, "out": 256, "max_num_seqs": SEQS}}}
    path = out / f"native-{a.arm}.json"
    rng = random.Random(0x5EED)
    with tempfile.TemporaryDirectory() as scratch:
        t0 = time.perf_counter()
        try:
            llm, info = build(a.arm, a, Path(scratch))
        except Exception as e:  # noqa: BLE001 - a refusal is the arm's result
            doc["refusal"] = "".join(traceback.format_exception(e))[-6000:]
            doc["refused_at"] = "engine build (load, profile run or CUDA-graph capture)"
            path.write_text(json.dumps(doc, indent=1))
            print(json.dumps({"arm": a.arm, "refused": doc["refusal"][-600:]}), flush=True)
            return 3
        doc["build_s"] = round(time.perf_counter() - t0, 1)
        doc["engine"] = info
        import torch
        doc["device"] = torch.cuda.get_device_name()
        doc["vllm"] = __import__("vllm").__version__
        timed(llm, prompts(rng, 32, 128), 16)
        t1s, t256s, tpot = [], [], []
        for _ in range(a.reps):
            t1 = timed(llm, prompts(rng, 32, 128), 1)
            t256 = timed(llm, prompts(rng, 32, 128), 256)
            t1s.append(t1)
            t256s.append(t256)
            tpot.append((t256 - t1) / 255)
        doc["decode_b32"] = {"t1_s": [round(v, 5) for v in t1s], "t256_s": [round(v, 5) for v in t256s],
                             "tpot_ms": [round(v * 1e3, 4) for v in tpot], "tpot_ms_median": round(statistics.median(tpot) * 1e3, 4)}
        path.write_text(json.dumps(doc, indent=1))
        sat = []
        for _ in range(a.sat_reps):
            t = timed(llm, prompts(rng, a.sat_requests, 512), 256)
            sat.append(a.sat_requests * 256 / t)
        doc["saturation"] = {"out_tok_s": [round(v, 1) for v in sat], "out_tok_s_median": round(statistics.median(sat), 1)}
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(doc, indent=1))
    print(json.dumps({"arm": a.arm, "tpot_ms": doc["decode_b32"]["tpot_ms_median"],
                      "sat_out_tok_s": doc["saturation"]["out_tok_s_median"], "build_s": doc["build_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
