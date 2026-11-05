"""On one GPU: the exact-int8 engine of `--role` (v1 `LLAMA31_8B_INT8`, its LM head BF16; v2 `LLAMA31_8B_INT8_V2`, its LM
head int8) on the one request of a workload (`--workload`, its `prompt_token_ids`): its greedy tokens, and its decode step's
kernel time under torch's profiler, split by where each kernel was launched:

  lm_head         inside the logits processor's forward (`HEAD_SCOPE`): the LM head's product, int8 (v2) or BF16 (v1), and its slice
  int8_linear     elsewhere under `verity_int8::row_quant` or `verity_int8::scaled_mm`
  int8_attention  elsewhere under any other `verity_int8::` op
  bf16_gemm       elsewhere under `aten::mm`, `aten::addmm`, `aten::matmul` or `aten::linear`
  other           the rest the profiler ties to a CPU op (norms, rotary, the activation, embedding, sampling, copies)
  unattributed    device events it ties to none (a Triton launch through the driver API), listed by kernel name

A decode step's time is the profiled generation of `--max-tokens` tokens less that of one token, over `--max-tokens` - 1; its wall
time is also taken unprofiled.  Writes
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

SCHEMA = "verity-vllm/int8-decode-split/v4"
CLASSES = ("int8_linear", "lm_head", "int8_attention", "bf16_gemm", "other", "unattributed")
LINEAR_OPS = ("verity_int8::row_quant", "verity_int8::scaled_mm")
BF16_GEMMS = ("aten::mm", "aten::addmm", "aten::matmul", "aten::linear")
#: the profiler range around the logits processor's forward (the LM head's product and its vocab slice)
HEAD_SCOPE = "verity::logits_processor"
TOP = 12


def classify(evt: Any) -> str:
    """The class of a kernel launched under `evt`: `lm_head` inside the logits processor's range, else by the innermost named op."""
    e, op = evt, None
    while e is not None:
        if e.name == HEAD_SCOPE:
            return "lm_head"
        if op is None:
            op = ("int8_linear" if e.name in LINEAR_OPS else "int8_attention" if e.name.startswith("verity_int8::") else
                  "bf16_gemm" if e.name in BF16_GEMMS else None)
        e = e.cpu_parent
    return op or "other"


def scope(module: Any, name: str) -> None:
    import torch

    opened: list[Any] = []

    def enter(_m: Any, _args: Any) -> None:
        opened.append(torch.profiler.record_function(name))
        opened[-1].__enter__()

    def leave(_m: Any, _args: Any, _out: Any) -> None:
        opened.pop().__exit__(None, None, None)

    module.register_forward_pre_hook(enter)
    module.register_forward_hook(leave)


def profiled(llm: Any, prompt: list[int], max_tokens: int) -> dict[str, Any]:
    """Every device kernel's time by class: kernels the profiler correlates with a CPU op by `classify`, and per kernel name the
    device time beyond them (`unattributed`, e.g. a Triton launch through the driver API)."""
    import torch
    from torch.autograd import DeviceType
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    us = dict.fromkeys(CLASSES, 0.0)
    names: dict[str, dict[str, float]] = {c: {} for c in CLASSES}
    device: dict[str, float] = {}
    for evt in prof.events():
        if evt.device_type == DeviceType.CUDA:
            if evt.name != HEAD_SCOPE:      # the range's own device annotation spans the head's kernels
                device[evt.name] = device.get(evt.name, 0.0) + evt.time_range.elapsed_us()
            continue
        for k in getattr(evt, "kernels", ()):
            c = classify(evt)
            us[c] += k.duration
            names[c][k.name] = names[c].get(k.name, 0.0) + k.duration
    # every kernel is also a device event of its own: what the device events hold beyond the correlated kernels is unattributed
    for n, d in device.items():
        rest = d - sum(names[c].get(n, 0.0) for c in CLASSES if c != "unattributed")
        if rest > 0:
            us["unattributed"] += rest
            names["unattributed"][n] = rest
    top = {c: dict(sorted(((n[:120], v / 1e3) for n, v in d.items()), key=lambda x: -x[1])[:TOP]) for c, d in names.items()}
    return {"max_tokens": max_tokens, "wall_ms": wall * 1e3, "kernel_ms": {c: v / 1e3 for c, v in us.items()}, "top_kernels_ms": top,
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
    scope(eng.model.logits_processor, HEAD_SCOPE)
    from vllm import SamplingParams

    def timed(n: int) -> tuple[float, list[int]]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = eng.llm.generate([{"prompt_token_ids": prompt}], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True),
                               use_tqdm=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3, list(out[0].outputs[0].token_ids)

    timed(a.max_tokens)
    (t1, _), (tn, tokens) = timed(1), timed(a.max_tokens)
    one, many = profiled(eng.llm, prompt, 1), profiled(eng.llm, prompt, a.max_tokens)
    steps = a.max_tokens - 1
    decode = {c: (many["kernel_ms"][c] - one["kernel_ms"][c]) / steps for c in CLASSES}
    total = sum(decode.values())
    return {"schema": SCHEMA, "argv": sys.argv, "role": a.role, "repo": eng.cp.get("repo"), "revision": eng.cp.get("revision"),
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "row": wl.get("sweep", {}).get("row_id"),
            "prompt_len": len(prompt), "lm_head": head, "vocab": vocab, "greedy_tokens": tokens, "profiled": [one, many],
            "decode_step": {"wall_ms_unprofiled": (tn - t1) / steps, "wall_ms_profiled": (many["wall_ms"] - one["wall_ms"]) / steps,
                            "kernel_ms": decode, "kernel_total_ms": total, "share": {c: decode[c] / total for c in CLASSES}}}


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
                     f"({d['decode_step']['wall_ms_unprofiled']:.3f} ms wall unprofiled); " +
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
