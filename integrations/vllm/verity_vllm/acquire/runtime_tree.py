"""Write a `RuntimeTree` document from a model instantiated on the META device as Build does (`cb_a.vllm_meta`: vLLM's own config +
model registry, shapes only, no weights, no GPU).  Pod only (needs vLLM).

    python -m verity_vllm.acquire.runtime_tree --model <hf repo> --revision <rev> [--dtype bfloat16] [--max-model-len N] [--tp W] --out tree.json

The runner is not built on the meta path; its facts are declared (`runner.source`) -- the two facts `NativeHostCommitter.install` /
`_select_modules` read from a live engine: the runner exposes `sample()` (True) and whether its `sampler` attribute is an nn.Module
(`--sampler-module`; default False: the pinned vLLM's live engine shows no such module -- m1_capture's runtime_tree.json under
ACQUIRE_ENGINE=v2 on the l40s b64 reference row, 2026-09-23 -- so v1's `runner.sampler` forward hook is not registered there either).
A live engine's tree carries the observed facts; `--runner-facts <live runtime_tree.json>` copies them.
"""
from __future__ import annotations

import argparse
import json
import sys

from verity_vllm.acquire.correspondence import RuntimeTree, runtime_tree_of


def meta_runtime_tree(model: str, revision: str, *, dtype: str = "bfloat16", max_model_len: int = 2048, tensor_parallel_size: int = 1,
                      attention_backend: str | None = None, sampler_module: bool = False, runner_facts: dict | None = None) -> RuntimeTree:
    from verity_capture.experimental.cb_a import vllm_meta as VM

    cfg = VM.make_vllm_config(model=model, revision=revision, dtype=dtype, max_model_len=max_model_len, attention_backend=attention_backend,
                              tensor_parallel_size=tensor_parallel_size)
    VM.ensure_distributed(cfg)
    m = VM.instantiate_meta(cfg)
    rt = runtime_tree_of(m, None, source=f"meta:{model}@{revision}")
    if runner_facts is not None:
        rt.runner = dict(runner_facts, source=f"copied from a live engine's tree: {runner_facts.get('source')}")
    else:
        rt.runner = {"sample": True, "sampler_module": bool(sampler_module),
                     "source": "declared (meta path builds no runner): the runner's sample() is wrapped by install(); sampler nn.Module attribute " + ("declared present" if sampler_module else "absent on the pinned vLLM's live engine")}
        if sampler_module:
            rt.runner["sampler_class"] = ["Sampler"]
    return rt


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--sampler-module", action="store_true", help="declare the runner's `sampler` attribute an nn.Module (v1 hooks it; never fires)")
    ap.add_argument("--runner-facts", default=None, help="a live engine's runtime_tree.json whose `runner` facts are copied")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    facts = RuntimeTree.load(a.runner_facts).runner if a.runner_facts else None
    rt = meta_runtime_tree(a.model, a.revision, dtype=a.dtype, max_model_len=a.max_model_len, tensor_parallel_size=a.tp, attention_backend=a.attention_backend,
                           sampler_module=a.sampler_module, runner_facts=facts)
    rt.save(a.out)
    print(json.dumps({"modules": len(rt.modules), "buffers": len(rt.buffers), "out": a.out}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
