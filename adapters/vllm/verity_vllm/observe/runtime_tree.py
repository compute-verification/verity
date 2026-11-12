"""The RUNTIME's module tree -- `RuntimeTree` (the record, `verity_vllm.records.runtime_tree`, whose names this module hands on):
the built model's module paths with their class MRO names + buffers + the runner's facts,
taken from a live engine (`runtime_tree_of`) or from a model instantiated on the META device as Build does (`meta_runtime_tree`, pod
only: needs vLLM).  Module-tree facts are NOT correspondence: they say which modules the runtime HAS, not which module a Program Call was
lowered from (that is `verity_vllm.correspondence.runtime`, read by `acquire.correspondence`).  Serialisable so the CPU gate runs off-pod.

    verity-vllm runtime-tree --model <hf repo> --revision <rev> [--dtype bfloat16] [--max-model-len N] [--tp W] --out tree.json

The runner is not built on the meta path; its facts are declared (`runner.source`) -- the two facts `NativeHostCommitter.install` /
`_select_modules` read from a live engine: the runner exposes `sample()` (True) and whether its `sampler` attribute is an nn.Module
(`--sampler-module`; default False: the pinned vLLM's live engine shows no such module -- m1_capture's runtime_tree.json on the l40s
b64 reference row -- so the class tables' `runner.sampler` forward hook is not registered there either).
A live engine's tree carries the observed facts; `--runner-facts <live runtime_tree.json>` copies them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from verity_vllm.config import option
from verity_vllm.engine.profiles import layer_classes as LC
from verity_vllm.records.runtime_tree import (
    RUNNER_PATH,
    RuntimeTree,
    mro_names,
    runtime_tree_from_paths,
    runtime_tree_of,
)
from verity_vllm.records.runtime_tree import SCHEMA as RUNTIME_TREE_SCHEMA

__all__ = ["RUNNER_PATH", "RUNTIME_TREE_SCHEMA", "Options", "RuntimeTree", "main", "meta_runtime_tree", "mro_names", "runtime_tree_from_paths",
           "runtime_tree_of"]


def meta_runtime_tree(model: str, revision: str, *, dtype: str = "bfloat16", max_model_len: int = 2048, tensor_parallel_size: int = 1,
                      attention_backend: str | None = None, sampler_module: bool = False, runner_facts: dict | None = None) -> RuntimeTree:
    from verity_vllm.program.frontend import vllm_meta as VM

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
            rt.runner["sampler_class"] = [LC.SAMPLER]
    return rt


@dataclass(frozen=True, kw_only=True)
class Options:
    model: str = option("--model")
    revision: str = option("--revision")
    dtype: str = option("--dtype", default="bfloat16")
    max_model_len: int = option("--max-model-len", type=int, default=2048)
    tp: int = option("--tp", type=int, default=1)
    attention_backend: str | None = option("--attention-backend", default=None)
    sampler_module: bool = option("--sampler-module", default=False, action="store_true",
                                  help="declare the runner's `sampler` attribute an nn.Module (v1 hooks it; never fires)")
    runner_facts: str | None = option("--runner-facts", default=None,
                                      help="a live engine's runtime_tree.json whose `runner` facts are copied")
    out: str = option("--out")


def main(a: Options) -> int:
    facts = RuntimeTree.load(a.runner_facts).runner if a.runner_facts else None
    rt = meta_runtime_tree(a.model, a.revision, dtype=a.dtype, max_model_len=a.max_model_len, tensor_parallel_size=a.tp, attention_backend=a.attention_backend,
                           sampler_module=a.sampler_module, runner_facts=facts)
    rt.save(a.out)
    print(json.dumps({"modules": len(rt.modules), "buffers": len(rt.buffers), "out": a.out}), flush=True)
    return 0
