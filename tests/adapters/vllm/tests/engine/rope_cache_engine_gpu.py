"""On one GPU: an int8 Llama engine on vLLM's FlashAttention (`TargetProfile.int8_exact` "linears", the int8 pin of `--role`),
whose build installs the fused rope and KV-cache write (`engine.rope_cache`), against the same engine with it uninstalled.

One process writes `<out>/rope_cache_engine_gpu.json`:

  install   what `build_engine` installed (`eng.rope_cache`) and each decoder layer's attention impl.
  kernels   the rope and cache kernels a short generation launches in each arm (the profiler's names): `rope_cache_kernel`
            alone fused, vLLM's two unfused.
  equal     greedy generations in three passes, fused, unfused, fused again: PROMPTS batched, then a `--long`-token prompt
            alone.  Each pass's token ids, their logprobs and, by SHA-256, every layer's attention output (o_proj's input) at
            every call are the first pass's.
  timed     at each of `--ts`, a prefill of T tokens as one step (max_tokens 1), the arms interleaved, `--reps` each: ms a
            request between CUDA events on either side of `generate`.

    cd adapters/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    $PY tests/engine/rope_cache_engine_gpu.py --out DIR --at ISO_TIME --by LANE [--role R --base-role B --repo REPO] [--ts 8192,32768]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from tests.engine import int8_pin as X

RECORD = "rope_cache_engine_gpu.json"


def layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "self_attn") and hasattr(m.self_attn, "o_proj")]


class Digests:
    """SHA-256 of each layer's attention output, call by call, while `on`."""

    def __init__(self, model: torch.nn.Module):
        self.on = False
        self.got: list[tuple[str, str]] = []
        self.handles = [m.self_attn.o_proj.register_forward_hook(self._hook(n)) for n, m in layers(model)]

    def _hook(self, name: str):
        def hook(mod, args, out):
            if self.on:
                x = args[0].contiguous()
                self.got.append((name, hashlib.sha256(x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()))
        return hook

    def take(self) -> list[tuple[str, str]]:
        got, self.got = self.got, []
        return got


def generate(llm: Any, prompts: list[Any], max_tokens: int) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0, logprobs=1), use_tqdm=False)
    return [{"token_ids": list(o.outputs[0].token_ids),
             "logprobs": [float(lp[t].logprob) for t, lp in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)]} for o in outs]


def kernels(llm: Any, prompts: list[Any]) -> list[str]:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        generate(llm, prompts, 2)
        torch.cuda.synchronize()
    return sorted({e.name.split("<")[0].split("(")[0].replace("void ", "") for e in prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and ("rope" in e.name or "rotary" in e.name or "reshape_and_cache" in e.name)})






