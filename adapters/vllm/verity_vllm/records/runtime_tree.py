"""The runtime's module tree (`verity-vllm/runtime-tree/v1`, ``runtime_tree.json``): the built model's module paths with their class
MRO names, its named buffers and the runner's facts, as the acquisition plan (`acquire.plan`) and its reader of where a Call lives
at runtime (`correspondence.reader_for_acquire`) read it.

Module-tree facts are not correspondence: they say which modules the runtime has, not which module a Program Call was lowered from
(that is `verity_vllm.correspondence.runtime`).  A tree is taken from a live engine (`runtime_tree_of`), from a model instantiated on
the meta device as the Build does (`verity_vllm.observe.runtime_tree.meta_runtime_tree`, which needs vLLM) or from paths
(`runtime_tree_from_paths`: tests, fixtures).  Its one writer is `RuntimeTree.save`, which writes every field, so `from_json` and
`load` refuse a missing one by name (`fields.MissingField`).  Unlike the package's other records it is not frozen: the meta path
states the runner's facts on a tree it has built.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.records.fields import RecordError, required

__all__ = ["RUNNER_PATH", "RuntimeTree", "SCHEMA", "mro_names", "runtime_tree_from_paths", "runtime_tree_of"]

SCHEMA = "verity-vllm/runtime-tree/v1"
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

    def to_json(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "source": self.source, "runner": dict(self.runner), "modules": {k: list(v) for k, v in sorted(self.modules.items())},
                "buffers": sorted(self.buffers)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any], where: str = "the runtime tree") -> RuntimeTree:
        """The tree as `to_json` wrote it: a field it writes and `d` lacks is refused by name, and so is another schema."""
        schema = required(d, "schema", where, str)
        if schema != SCHEMA:
            raise RecordError(where, "schema", f"is {schema!r}, not {SCHEMA!r}")
        return cls(modules={str(k): [str(c) for c in v] for k, v in required(d, "modules", where, Mapping).items()},
                   buffers=[str(b) for b in required(d, "buffers", where, (list, tuple))],
                   runner=dict(required(d, "runner", where, Mapping)), source=required(d, "source", where, str))

    @classmethod
    def load(cls, path: str | Path) -> RuntimeTree:
        return cls.from_json(json.loads(Path(path).read_text()), str(path))

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
