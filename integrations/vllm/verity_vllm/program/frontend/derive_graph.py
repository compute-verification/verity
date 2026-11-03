"""Derivation from a functional ATen graph, the one `torch.func` and `make_fx` produce (a joint forward-backward graph: a training
step), which `derive` would not re-export (`derive` exports pre-dispatch itself).

    derive_graph(graph, example_inputs, *, mode, profile, rules, name, input_names, theta) -> DeriveReport

The graph's placeholders are the Program's inputs, in order, named `input_names` (default: the placeholders' own names); its output
node's tuple is the Program's outputs. No module owns a tensor of the graph, so there is no `weights` record: `theta` names the
inputs that are the model's weights (`Program.theta`, the root's `theta` input nodes). Otherwise the derivation is
`derive.derive_report`'s.
"""

from __future__ import annotations

import keyword
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.report import DeriveReport
from verity_vllm.program.frontend.rules.base import RuleSet


def _renamed(graph: torch.fx.GraphModule, names: list[str]) -> torch.fx.GraphModule:
    """`graph` with its placeholders named `names` (a derived Program's parameters are named after its graph's placeholders)."""
    g = torch.fx.Graph()
    env: dict = {}
    it = iter(names)
    for n in graph.graph.nodes:
        if n.op == "placeholder":
            env[n] = g.placeholder(next(it))
            env[n].meta = dict(n.meta)
        else:
            env[n] = g.node_copy(n, lambda a: env[a])
    return torch.fx.GraphModule(graph, g)


def derive_graph(graph: torch.fx.GraphModule, example_inputs: Sequence[Any], *, mode: str, profile: str | None = None,
                 rules: RuleSet | None = None, name: str | None = None, input_names: Sequence[str] | None = None,
                 theta: Sequence[str] = (), check_registry: bool = True) -> DeriveReport:
    """The derivation of `graph` (module docstring) on `example_inputs` (fake or meta tensors: no value is read)."""
    phs = [n.name for n in graph.graph.nodes if n.op == "placeholder"]
    names = list(input_names) if input_names is not None else phs
    if len(names) != len(phs) or len(set(names)) != len(names):
        raise ValueError(f"input_names must name the graph's {len(phs)} placeholders once each, got {len(names)} names")
    if bad := [n for n in names if not n.isidentifier() or keyword.iskeyword(n)]:
        raise ValueError(f"input names must be Python identifiers (a derived Program's parameters are its placeholders): {bad[:4]}")
    if not set(theta) <= set(names):
        raise ValueError(f"theta names inputs the graph does not have: {sorted(set(theta) - set(names))}")
    out = next(n for n in graph.graph.nodes if n.op == "output")
    if not isinstance(out.args[0], (tuple, list)):
        raise ValueError("the graph must return a tuple of tensors")
    gm = graph if names == phs else _renamed(graph, names)
    got = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    if got != names:
        raise ValueError(f"torch.fx renamed inputs: {[(a, b) for a, b in zip(names, got) if a != b][:4]}")
    sig = SimpleNamespace(user_inputs=got, inputs_to_parameters={}, inputs_to_buffers={}, inputs_to_lifted_tensor_constants={},
                          inputs_to_lifted_custom_objs={},
                          output_specs=[SimpleNamespace(kind=SimpleNamespace(name="USER_OUTPUT"), arg=None)] * len(out.args[0]))
    ep = SimpleNamespace(graph=gm.graph, graph_module=gm, graph_signature=sig, constants={}, theta=tuple(theta))

    def source(*inputs):
        return gm(*inputs)

    source.__name__ = source.__qualname__ = name or "Graph"
    return derive_report(source, example_inputs, mode=mode, profile=profile, rules=rules, name=name, check_registry=check_registry,
                         exported=ep)


__all__ = ["derive_graph"]
