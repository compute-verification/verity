"""The RUNTIME's module tree -- `RuntimeTree`: the built model's module paths with their class MRO names + buffers + the runner's facts,
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from verity_vllm.config import option

RUNTIME_TREE_SCHEMA = "verity-vllm/runtime-tree/v1"
RUNNER_PATH = "runner.sampler"          # v1's path of the sampler's Values (the runner's `sample()` return, not a module hook)


@dataclass
class RuntimeTree:
    """The runtime's module tree: `modules[path] = [class names of the MRO]`, `buffers` = named buffers, `runner` facts."""
    modules: dict[str, list[str]] = field(default_factory=dict)
    buffers: list[str] = field(default_factory=list)
    runner: dict[str, Any] = field(default_factory=dict)      # {"sample": bool, "sampler_module": bool, "source": str}
    source: str = ""

    def has_module(self, path: str) -> bool:
        return path in self.modules

    def classes(self, path: str) -> set[str]:
        return set(self.modules.get(path) or ())

    def to_json(self) -> dict[str, Any]:
        return {"schema": RUNTIME_TREE_SCHEMA, "source": self.source, "runner": dict(self.runner), "modules": {k: list(v) for k, v in sorted(self.modules.items())},
                "buffers": sorted(self.buffers)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "RuntimeTree":
        if d.get("schema") not in (None, RUNTIME_TREE_SCHEMA):
            raise ValueError(f"runtime tree schema {d.get('schema')!r} != {RUNTIME_TREE_SCHEMA}")
        return cls(modules={str(k): [str(c) for c in v] for k, v in (d.get("modules") or {}).items()}, buffers=[str(b) for b in (d.get("buffers") or [])],
                   runner=dict(d.get("runner") or {}), source=str(d.get("source") or ""))

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeTree":
        return cls.from_json(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1) + "\n")


def mro_names(m: Any) -> list[str]:
    return [c.__name__ for c in type(m).__mro__]


def runtime_tree_of(model: Any, runner: Any = None, *, source: str = "live") -> RuntimeTree:
    """The tree of a built `torch.nn.Module` (+ the engine's runner).  The root module (`named_modules()` name "") is kept under its
    name "" so a Value produced in the root's forward body has an owning module in the tree."""
    mods = {name: mro_names(m) for name, m in model.named_modules()}
    bufs = [name for name, _b in model.named_buffers()]
    r: dict[str, Any] = {"sample": bool(runner is not None and hasattr(runner, "sample")), "sampler_module": False, "source": "engine runner" if runner is not None else "no runner"}
    if runner is not None:
        smp = getattr(runner, "sampler", None)
        import torch
        r["sampler_module"] = isinstance(smp, torch.nn.Module)
        if r["sampler_module"]:
            r["sampler_class"] = mro_names(smp)
    return RuntimeTree(modules=mods, buffers=bufs, runner=r, source=source)


def runtime_tree_from_paths(paths: Mapping[str, Iterable[str]], *, buffers: Iterable[str] = (), runner: Mapping[str, Any] | None = None, source: str = "synthetic") -> RuntimeTree:
    """A tree from `{module path: [class names]}` (tests, fixtures)."""
    return RuntimeTree(modules={str(k): [str(c) for c in v] for k, v in paths.items()}, buffers=[str(b) for b in buffers], runner=dict(runner or {}), source=source)


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
            rt.runner["sampler_class"] = ["Sampler"]
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
