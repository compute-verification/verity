"""Resolve the CUDA kernels the *engine* launches for Qwen2.5-1.5B on this GPU under the POC's
declared profile flags (W1's ``verity_vllm_adapter.profile``: ``VLLM_BATCH_INVARIANT=1``,
``VLLM_USE_V2_MODEL_RUNNER=1``, ``VLLM_ENABLE_V1_MULTIPROCESSING=0``), for one prefill + a few
decode steps, under ``torch.profiler``.  Both the production configuration (stock: compile +
cudagraphs) and the replay configuration (``enforce_eager=True``) are of interest; run twice.

    python -m verity_vllm_numerics.resolve_engine_kernels OUT.json [--eager] [--no-batch-invariant]

The output lists every CUDA kernel name with its launch count, plus the GEMM-relevant subset
(cuBLAS/cuBLASLt/CUTLASS/Triton matmul kernels), so the numerical rows can cite the kernels
actually observed rather than the ones a stand-alone ``F.linear`` call happens to pick.
"""

from __future__ import annotations

import collections
import json
import os
import sys

eager = "--eager" in sys.argv
if "--no-batch-invariant" not in sys.argv:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

out_path = sys.argv[1]
kw = dict(model="Qwen/Qwen2.5-1.5B", dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.35, seed=0)
if eager:
    kw["enforce_eager"] = True
llm = LLM(**kw)
sp = SamplingParams(temperature=0.0, max_tokens=8)
prompts = ["The capital of France is", "def fibonacci(n):\n    "]
llm.generate(prompts, sp)  # warm-up (compile, cudagraph capture)
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    outs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
names: collections.Counter[str] = collections.Counter()
for evt in prof.events():
    if getattr(evt, "device_type", None) is not None and str(evt.device_type).endswith("CUDA"):
        names[evt.name] += 1
GEMM_MARKERS = ("gemm", "gemv", "matmul", "cutlass", "splitK", "Kernel2<", "mm_")
vc = llm.llm_engine.vllm_config
cc = vc.compilation_config
res = {
    "eager": eager,
    "vllm": __import__("vllm").__version__,
    "torch": torch.__version__,
    "triton": __import__("triton").__version__,
    "gpu": torch.cuda.get_device_name(0),
    "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
    "compilation": {
        "mode": str(getattr(cc, "mode", None)),
        "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
        "enforce_eager": bool(vc.model_config.enforce_eager),
        "model_runner": "v2" if getattr(vc, "use_v2_model_runner", False) else "v1",
        "custom_ops": list(getattr(cc, "custom_ops", [])),
    },
    "outputs": [o.outputs[0].text for o in outs],
    "gemm_kernels": [{"name": k, "count": v} for k, v in names.most_common() if any(m in k for m in GEMM_MARKERS)],
    "kernels": [{"name": k, "count": v} for k, v in names.most_common()],
}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res["compilation"]))
print(json.dumps(res["gemm_kernels"], indent=1)[:20000])
