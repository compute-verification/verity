"""The Program names a node by the producing module's path in the source module's own `named_modules()` vocabulary.

vLLM 0.28 moved the MoE router call into the runner (`MoERunner.gate = gate`, called in its forward): the exported `nn_module_stack` of
the router Gemm ends in a bare `gate`, so the Qwen3-30B-A3B Program named 48 routers `gate/...` while the collector binds
`model.layers.<L>.mlp.gate/0`; the manifest merged them into ONE identity (element_range x48) and Commit failed identity coverage.
`Ctx.stack_path` resolves an unrooted entry relative to its parent and canonicalises a duplicate registration to the first one.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.program.frontend.derive import Ctx, module_registrations  # noqa: E402


class _Runner(torch.nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate                     # the block's router, re-registered by its consumer (vLLM MoERunner)
        self.w13 = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.w13(self.gate(x))


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 4)    # registered FIRST under mlp.gate
        self.experts = _Runner(self.gate)    # ... and again under mlp.experts.gate

    def forward(self, x):
        return self.experts(x)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = torch.nn.Identity()
        self.layers = torch.nn.ModuleList([_Block(), _Block()])
        self.layers[1].rope = self.rope      # a shared instance under two layer paths (vLLM get_rope)
        self.layers[0].rope = self.rope


def _ctx(model) -> Ctx:
    ctx = Ctx.__new__(Ctx)
    ctx._module_paths, ctx._canonical_path = module_registrations(model)
    ctx.unresolved_module_paths = {}
    return ctx


def _node(*stack: tuple[str, type], name: str = "mm_3"):
    return SimpleNamespace(meta={"nn_module_stack": {f"k{i}": (p, c) for i, (p, c) in enumerate(stack)}}, name=name)


def test_module_registrations_first_path_is_canonical():
    paths, canon = module_registrations(_Model())
    assert {"layers.0.gate", "layers.0.experts.gate", "layers.0.experts", "layers.1.experts.w13", "rope", "layers.0.rope"} <= paths
    assert canon["layers.0.experts.gate"] == "layers.0.gate" and canon["layers.1.experts.gate"] == "layers.1.gate"
    assert canon["layers.0.rope"] == "rope" and canon["layers.1.rope"] == "rope"     # first registration = `named_modules()`'s name
    assert "layers.0.gate" not in canon
    assert module_registrations(object()) == (set(), {})


def test_unrooted_router_entry_resolves_to_the_collectors_path():
    ctx = _ctx(_Model())
    n = _node(("", _Model), ("layers.0", _Block), ("layers.0.experts", _Runner), ("gate", torch.nn.Linear))
    assert ctx.stack_path(n) == "layers.0.gate"                       # resolved under its parent, then canonical (not experts.gate)
    assert ctx.hint_name(n) == "layers.0.gate/mm_3"
    n1 = _node(("", _Model), ("layers.1", _Block), ("layers.1.experts", _Runner), ("gate", torch.nn.Linear))
    assert ctx.stack_path(n1) == "layers.1.gate"                      # 48 layers -> 48 names, not one bare `gate`


def test_rooted_entries_are_kept_verbatim_including_shared_rope():
    ctx = _ctx(_Model())
    n = _node(("", _Model), ("layers.1", _Block), ("layers.1.rope", torch.nn.Identity))
    assert ctx.stack_path(n) == "layers.1.rope"                       # a rooted duplicate stays per layer (manifest alias rule)
    n2 = _node(("", _Model), ("layers.1", _Block), ("layers.1.experts", _Runner), ("layers.1.experts.w13", torch.nn.Linear))
    assert ctx.stack_path(n2) == "layers.1.experts.w13"
    unknown = _node(("", _Model), ("layers.1", _Block), ("nowhere", torch.nn.Linear))
    assert ctx.stack_path(unknown) == "nowhere"                       # unknown to the module: what the trace said
    assert ctx.hint_name(SimpleNamespace(meta={}, name="add_1")) == "add_1"


def test_rooted_duplicate_stays_under_the_duplicate_name_fail_closed_at_coverage():
    # an export that yields the full duplicate path itself is NOT rewritten -- the collector binds `layers.1.gate`,
    # so a manifest built from this name fails C1 coverage as missing (the pre-fix symptom, named), never a silent merge
    ctx = _ctx(_Model())
    n = _node(("", _Model), ("layers.1", _Block), ("layers.1.experts", _Runner), ("layers.1.experts.gate", torch.nn.Linear))
    assert ctx.stack_path(n) == "layers.1.experts.gate"
    assert ctx.unresolved_module_paths == {}


def test_unresolved_unrooted_entries_are_counted_for_the_provenance():
    # an unrooted entry under no registered path keeps its bare name (as before) and is COUNTED, so the
    # 48-layers-under-one-name failure mode is visible at derive (`annotations.derive.unresolved_module_paths`), not first at Commit
    ctx = _ctx(_Model())
    for k in range(3):
        assert ctx.stack_path(_node(("", _Model), (f"layers.{k % 2}", _Block), ("orphan", torch.nn.Linear))) == "orphan"
    assert ctx.stack_path(_node(("", _Model), ("layers.0", _Block), ("layers.0.experts", _Runner), ("gate", torch.nn.Linear))) == "layers.0.gate"
    assert ctx.unresolved_module_paths == {"orphan": 3}               # the resolved router is not counted


class _SharedExperts(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self._layer = layer                  # the block's shared expert MLP (vLLM SharedExperts)


class _Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj, self.act_fn, self.down_proj = torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(4, 4)


class _GlmMoe(torch.nn.Module):
    """GLM-5.3's `DeepseekV2MoE`: the shared expert registered first, then the runner holding it as `_shared_experts._layer`."""

    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 4)
        self.shared_experts = _Mlp()
        self.experts = _Runner(self.gate)
        self.experts._shared_experts = _SharedExperts(self.shared_experts)


class _GlmDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _GlmMoe()


class _GlmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_GlmDecoder(), _GlmDecoder()])


@pytest.mark.parametrize("container", [True, False], ids=["with-SharedExperts-entry", "without"])
def test_glm53_shared_expert_entries_resolve_per_layer(container):
    """The runner's `_forward_impl` calls the shared expert as `_shared_experts._layer`: every entry below it is qualified from the
    runner, so the manifest of record named all 75 MoE layers' shared-expert Calls `_shared_experts._layer.act_fn` (one identity).
    Each resolves to the layer's `mlp.shared_experts.<leaf>`, the capture map's `shared.*` taps."""
    ctx = _ctx(_GlmModel())
    for k in (0, 1):
        head = (("", _GlmModel), (f"layers.{k}", _GlmDecoder), (f"layers.{k}.mlp", _GlmMoe), (f"layers.{k}.mlp.experts", _Runner))
        mid = (("_shared_experts", _SharedExperts),) if container else ()
        for leaf in ("gate_up_proj", "act_fn", "down_proj"):
            n = _node(*head, *mid, ("_shared_experts._layer", _Mlp), (f"_shared_experts._layer.{leaf}", torch.nn.Module))
            assert ctx.stack_path(n) == f"layers.{k}.mlp.shared_experts.{leaf}"
        assert ctx.stack_path(_node(*head, *mid, ("_shared_experts._layer", _Mlp))) == f"layers.{k}.mlp.shared_experts"
        assert ctx.stack_path(_node(*head, ("gate", torch.nn.Linear))) == f"layers.{k}.mlp.gate"
    assert ctx.unresolved_module_paths == {}


def test_without_a_module_the_trace_path_is_used():
    ctx = Ctx.__new__(Ctx)
    ctx._module_paths, ctx._canonical_path = module_registrations(None)
    ctx.unresolved_module_paths = {}
    assert ctx.stack_path(_node(("a.b", object), ("gate", object))) == "gate"
    assert ctx.unresolved_module_paths == {}                          # no module vocabulary -> nothing to resolve against, nothing counted
