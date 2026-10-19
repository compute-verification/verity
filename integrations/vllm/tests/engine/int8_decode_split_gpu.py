"""VLLM-49 on one GPU: the exact-int8 engine of `--role` (v1 `LLAMA31_8B_INT8`, its LM head BF16; v2 `LLAMA31_8B_INT8_V2`, its LM
head int8) on the one request of a row's workload (`--workload`, its `prompt_token_ids`): its greedy tokens, and its decode step's
kernel time under torch's profiler, split by the innermost op that launched each kernel:

  int8_linear     `verity_int8::row_quant` and `verity_int8::scaled_mm` (`int8_lm_head` is the scaled_mm whose weight has the vocab's rows)
  int8_attention  every other `verity_int8::` op
  bf16_gemm       `aten::mm`, `aten::addmm`, `aten::matmul`, `aten::linear` (v1's LM head)
  other           the rest (norms, rotary, the activation, embedding, sampling, copies)

A decode step's time is the profiled generation of `--max-tokens` tokens less that of one token, over `--max-tokens` - 1.  Writes
`<out>/int8_decode_split.<role>.json`; `--compare A.json B.json` prints the two runs' greedy-token agreement and splits.

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    $PY tests/engine/int8_decode_split_gpu.py --role LLAMA31_8B_INT8_V2 --workload workloads/ROW.json --out DIR
    $PY tests/engine/int8_decode_split_gpu.py --compare DIR/int8_decode_split.LLAMA31_8B_INT8.json DIR/int8_decode_split.LLAMA31_8B_INT8_V2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "verity-vllm/int8-decode-split/v1"
CLASSES = ("int8_linear", "int8_lm_head", "int8_attention", "bf16_gemm", "other")
LINEAR_OPS = ("verity_int8::row_quant", "verity_int8::scaled_mm")
BF16_GEMMS = ("aten::mm", "aten::addmm", "aten::matmul", "aten::linear")


def classify(evt: Any, vocab: int) -> str:
    e = evt
    while e is not None:
        if e.name == "verity_int8::scaled_mm":
            shapes = getattr(e, "input_shapes", None) or []
            return "int8_lm_head" if len(shapes) > 2 and list(shapes[2])[:1] == [vocab] else "int8_linear"
        if e.name in LINEAR_OPS:
            return "int8_linear"
        if e.name.startswith("verity_int8::"):
            return "int8_attention"
        if e.name in BF16_GEMMS:
            return "bf16_gemm"
        e = e.cpu_parent
    return "other"


def profiled(llm: Any, prompt: list[int], max_tokens: int, vocab: int) -> dict[str, Any]:
    import torch
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    us = dict.fromkeys(CLASSES, 0.0)
    for evt in prof.events():
        for k in getattr(evt, "kernels", ()):
            us[classify(evt, vocab)] += k.duration
    return {"max_tokens": max_tokens, "wall_ms": wall * 1e3, "kernel_ms": {c: v / 1e3 for c, v in us.items()},
            "tokens": list(out[0].outputs[0].token_ids)}


def run(a: argparse.Namespace) -> dict[str, Any]:
    import torch

    from verity_vllm.engine.build import build_engine
    from verity_vllm.program.frontend.target_profile import TargetProfile

    wl = json.loads(Path(a.workload).read_text())
    (req,) = wl["requests"]
    prompt = list(req["prompt_token_ids"])
    target = TargetProfile.from_device().with_overrides(int8_exact="all")
    eng = build_engine(a.checkpoints, case=a.role, target=target, max_num_seqs=1,
                       engine_args={"max_model_len": len(prompt) + a.max_tokens})
    vocab = eng.model.lm_head.weight.shape[0]
    head = {"weight": str(eng.model.lm_head.weight.dtype), "method": type(eng.model.lm_head.quant_method).__name__}
    from vllm import SamplingParams
    greedy = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, ignore_eos=True)
    tokens = list(eng.llm.generate([{"prompt_token_ids": prompt}], greedy, use_tqdm=False)[0].outputs[0].token_ids)
    one, many = profiled(eng.llm, prompt, 1, vocab), profiled(eng.llm, prompt, a.max_tokens, vocab)
    steps = a.max_tokens - 1
    decode = {c: (many["kernel_ms"][c] - one["kernel_ms"][c]) / steps for c in CLASSES}
    total = sum(decode.values())
    return {"schema": SCHEMA, "argv": sys.argv, "role": a.role, "repo": eng.cp.get("repo"), "revision": eng.cp.get("revision"),
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "row": wl.get("sweep", {}).get("row_id"),
            "prompt_len": len(prompt), "lm_head": head, "vocab": vocab, "greedy_tokens": tokens, "profiled": [one, many],
            "decode_step": {"wall_ms": (many["wall_ms"] - one["wall_ms"]) / steps, "kernel_ms": decode, "kernel_total_ms": total,
                            "share": {c: decode[c] / total for c in CLASSES}}}


def compare(paths: list[str]) -> str:
    a, b = (json.loads(Path(p).read_text()) for p in paths)
    ta, tb = a["greedy_tokens"], b["greedy_tokens"]
    n = min(len(ta), len(tb))
    first = next((i for i in range(n) if ta[i] != tb[i]), None)
    lines = [f"greedy tokens {a['role']} vs {b['role']}: {sum(x == y for x, y in zip(ta, tb))} of {n} positions equal, "
             f"identical prefix {n if first is None else first}" + ("" if first is None else f" (first divergence at {first})")]
    for d in (a, b):
        s, k = d["decode_step"]["share"], d["decode_step"]["kernel_ms"]
        lines.append(f"{d['role']} lm_head {d['lm_head']['weight']}: decode step {d['decode_step']['kernel_total_ms']:.3f} ms of kernels "
                     f"({d['decode_step']['wall_ms']:.3f} ms wall); " +
                     ", ".join(f"{c} {k[c]:.3f} ms {100 * s[c]:.2f}%" for c in CLASSES))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--role")
    p.add_argument("--workload")
    p.add_argument("--checkpoints", default="manifests/checkpoints.json")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--out", default=".")
    p.add_argument("--compare", nargs=2, metavar="JSON")
    a = p.parse_args()
    if a.compare:
        print(compare(a.compare))
        return 0
    if not (a.role and a.workload):
        p.error("--role and --workload (or --compare A B)")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    doc = run(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"int8_decode_split.{a.role}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"role": a.role, "lm_head": doc["lm_head"], "decode_step": doc["decode_step"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
