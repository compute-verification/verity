"""Write a `RuntimeTree` document from a model instantiated on the META device as Build does (`cb_a.vllm_meta`: vLLM's own config +
model registry, shapes only, no weights, no GPU).  Pod only (needs vLLM).

    python -m verity_vllm.acquire.runtime_tree --model <hf repo> --revision <rev> [--dtype bfloat16] [--max-model-len N] [--tp W] --out tree.json

The runner is not built on the meta path; its facts are declared (`runner.source`): vLLM's V1 GPU model runner exposes `sample()` and a
`Sampler` nn.Module attribute -- the same two facts `NativeHostCommitter.install` / `_select_modules` read from a live engine.  A live
engine's tree (m1_capture under ACQUIRE_ENGINE=v2 writes DIR/runtime_tree.json) carries the observed facts instead.
"""
from __future__ import annotations

import argparse
import json
import sys

from verity_vllm.acquire.correspondence import RuntimeTree, runtime_tree_of


def meta_runtime_tree(model: str, revision: str, *, dtype: str = "bfloat16", max_model_len: int = 2048, tensor_parallel_size: int = 1,
                      attention_backend: str | None = None) -> RuntimeTree:
    from verity_capture.experimental.cb_a import vllm_meta as VM

    cfg = VM.make_vllm_config(model=model, revision=revision, dtype=dtype, max_model_len=max_model_len, attention_backend=attention_backend,
                              tensor_parallel_size=tensor_parallel_size)
    VM.ensure_distributed(cfg)
    m = VM.instantiate_meta(cfg)
    rt = runtime_tree_of(m, None, source=f"meta:{model}@{revision}")
    rt.runner = {"sample": True, "sampler_module": True, "sampler_class": ["Sampler"],
                 "source": "declared (meta path builds no runner): vLLM V1 GPUModelRunner.sample() + its Sampler nn.Module, the facts install() reads live"}
    return rt


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rt = meta_runtime_tree(a.model, a.revision, dtype=a.dtype, max_model_len=a.max_model_len, tensor_parallel_size=a.tp, attention_backend=a.attention_backend)
    rt.save(a.out)
    print(json.dumps({"modules": len(rt.modules), "buffers": len(rt.buffers), "out": a.out}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
