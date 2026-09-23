"""Runtime correspondence for the acquisition plan: a TINY adapter over today's annotations.

Today a Program Call's module path and fx node are recovered from its instance name (`required_manifest.split_name(name)[0]`, then
`collector_path` under the export wrapper prefix) and the manifest identity row already carries them as `op_path` / `output_member`.
The parallel correspondence lane adds a `correspondence` namespace with exactly the three functions below; this module has the same
three names and signatures so the swap is one import (`from verity_vllm.correspondence import ...` for `AnnotationCorrespondence`).

    owning_module(call) -> str            the runtime module path (`named_modules()` name) that owns the Call
    return_slot(call, port) -> ReturnSlot  where the Call's output `port` is on the module's return: the whole return (`out`), tuple slot k
                                          (`0`, `1`), an attribute of a returned object (`sampled_token_ids`), or None (the port is no return)
    arg_slot(call, i) -> ArgSlot           the i-th positional argument of the module's forward

`call` is a required-value manifest identity row (dict: `op_path`, `output_member`, ...) or a Program instance row (dict: `name`, `spec`).
The RUNTIME side is `RuntimeTree`: the built model's module paths with their class MRO names + buffers + the runner's facts, taken from a
live engine (`runtime_tree_of`) or from a model instantiated on the meta device as Build does; serialisable so the CPU gate can run off-pod.
No model or family name appears here: every fact comes from the tree or the manifest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from verity_capture.commit.required_manifest import collector_path, split_name

RUNTIME_TREE_SCHEMA = "verity-vllm/runtime-tree/v1"
RUNNER_PATH = "runner.sampler"          # v1's path of the sampler's Values (the runner's `sample()` return, not a module hook)


@dataclass(frozen=True)
class ReturnSlot:
    """Where a Value sits on a module's return.  `index=None, attr=None` = the whole return (a single tensor); `index=k` = tuple slot k;
    `attr` = an attribute of the returned object (the runner's `SamplerOutput.sampled_token_ids`)."""
    index: int | None = None
    attr: str | None = None

    def label(self) -> str:
        if self.attr is not None:
            return f".{self.attr}"
        return "out" if self.index is None else str(self.index)


@dataclass(frozen=True)
class ArgSlot:
    index: int

    def label(self) -> str:
        return f"in[{self.index}]"


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

    def class_name(self, path: str) -> str | None:
        m = self.modules.get(path)
        return m[0] if m else None

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
        try:
            import torch
            r["sampler_module"] = isinstance(smp, torch.nn.Module)
        except ImportError:                                                        # pragma: no cover
            r["sampler_module"] = smp is not None
        if r["sampler_module"]:
            r["sampler_class"] = mro_names(smp)
    return RuntimeTree(modules=mods, buffers=bufs, runner=r, source=source)


def runtime_tree_from_paths(paths: Mapping[str, Iterable[str]], *, buffers: Iterable[str] = (), runner: Mapping[str, Any] | None = None, source: str = "synthetic") -> RuntimeTree:
    """A tree from `{module path: [class names]}` (tests, fixtures)."""
    return RuntimeTree(modules={str(k): [str(c) for c in v] for k, v in paths.items()}, buffers=[str(b) for b in buffers], runner=dict(runner or {}), source=source)


class AnnotationCorrespondence:
    """`owning_module` / `return_slot` / `arg_slot` over today's annotations, plus the runtime tree and the (optional) in-place facts.

    `inplace_facts`: {module path: {"mutates_input": bool, "output_mutated_after": bool}} -- alias / in-place facts the annotations may
    carry one day; today none do, so a consumer-input site's lifetime is `unknown` unless a fact says the input is left unmutated."""

    def __init__(self, runtime: RuntimeTree, *, wrapper_prefix: str = "", aliases: Mapping[str, Any] | None = None,
                 inplace_facts: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.runtime = runtime
        self.wrapper_prefix = wrapper_prefix
        self.aliases = dict(aliases or {})                      # program op_path -> [canonical op_path, invocation] (`op_path_aliases`)
        self.inplace_facts = {str(k): dict(v) for k, v in (inplace_facts or {}).items()}

    # -- the three functions of the correspondence namespace ------------------------------------------------------------------

    def owning_module(self, call: Mapping[str, Any]) -> str:
        if "op_path" in call:
            op = str(call["op_path"])
        elif "name" in call:
            op = collector_path(split_name(str(call["name"]))[0], self.wrapper_prefix)
        else:
            raise KeyError("call carries neither op_path nor name")
        alias = self.aliases.get(op)
        return str(alias[0]) if alias else op

    def return_slot(self, call: Mapping[str, Any], port: str | None = None) -> ReturnSlot | None:
        port = str(call.get("output_member", "out") if port is None else port)
        if port == "in":
            return None                                          # an input of the module: no return slot
        if port == "out":
            return ReturnSlot()
        if port.isdigit():
            return ReturnSlot(index=int(port))
        if self.owning_module(call) == RUNNER_PATH:
            return ReturnSlot(attr=port)                         # `sampled_token_ids` / `seed` / `splits` of the runner's sample()
        return ReturnSlot()                                      # a hidden-stream geometry name, a weight, a tap plane: the whole return of its site

    def arg_slot(self, call: Mapping[str, Any], i: int = 0) -> ArgSlot:
        return ArgSlot(index=int(i))

    # -- lifetime facts -------------------------------------------------------------------------------------------------------------

    def input_lifetime(self, consumer: str) -> str:
        f = self.inplace_facts.get(consumer)
        if f is None:
            return "unknown"
        return "unmutated" if not f.get("mutates_input", True) else "mutated-by-consumer"

    def output_lifetime(self, producer: str) -> str:
        f = self.inplace_facts.get(producer)
        if f is None:
            return "unknown"
        return "unmutated" if not f.get("output_mutated_after", True) else "mutated-after-hook"
