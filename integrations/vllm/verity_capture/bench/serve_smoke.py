#!/usr/bin/env python3
"""R10 owner 10: serving smoke for a worker -- the harness's own engine (verity_capture.vllm_adapter.build_engine: profile.apply_env
sets VLLM_BATCH_INVARIANT=1 and the declared env, enforce_eager, prefix caching off) on the pinned SmolLM2-135M (case B0), one
greedy 8-token generate.  Prints SERVE-SMOKE-OK <tokens> or SERVE-SMOKE-FAIL.  Runs inside /workspace/cp/veritor with
PYTHONPATH=. and the cu129 venv; gpu_memory_utilization=0.3 so a co-tenant job on the pod is not evicted."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
try:
    from verity_capture.vllm_adapter import build_engine
    from vllm import SamplingParams

    t0 = time.perf_counter()
    eng = build_engine(max_num_seqs=1, engine_args={"gpu_memory_utilization": float(os.environ.get("SMOKE_GPU_UTIL", "0.3")),
                                                    "max_model_len": 256})
    out = eng.llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=8))
    toks = list(out[0].outputs[0].token_ids)
    import torch, vllm  # noqa: E402
    print("SERVE-SMOKE-OK", "tokens", toks, "text", repr(out[0].outputs[0].text), "build_s", round(eng.build_s, 1), "total_s",
          round(time.perf_counter() - t0, 1), "vllm", vllm.__version__, "torch", torch.__version__,
          "BI", os.environ.get("VLLM_BATCH_INVARIANT"), "eager", eng.kwargs.get("enforce_eager"), "case", eng.case)
except Exception as e:  # noqa: BLE001
    print("SERVE-SMOKE-FAIL", type(e).__name__, str(e)[:300])
    sys.exit(1)
