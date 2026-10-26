"""`verity_vllm.protocols.linear.api` on stand-ins, with neither vLLM nor torch: the stack refuses what can't compose by the declared
attributes, computes inside a pass with the source's weight, fails closed outside one, and always ends a forward at the
source."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from verity_vllm.errors import Refused
from verity_vllm.protocols.linear.api import Arm, Control, Forward, LinearSite, PassSpec, Stack

SITES = (LinearSite(0, 0, "model.layers.0.self_attn.qkv_proj", 6144, 4096, "bf16"),
         LinearSite(1, 1, "model.layers.0.mlp.down_proj", 4096, 14336, "bf16", has_bias=True))


@dataclass
class Pass:
    spec: PassSpec
    calls: list = field(default_factory=list)
    ended: int = 0

    def call(self, site, x, weight, bias):
        self.calls.append((site.weight_id, x, weight, bias))
        return ("y", site.weight_id, x, weight)

    def end(self):
        self.ended += 1
        return {"pass": self.spec.index, "units": len(self.calls)}


@dataclass
class Executor:
    name: str = "pouw:pearl-c-sm120-v1-h3"
    keeps_weight_state: bool = True
    accepts: frozenset = frozenset({"bf16"})
    controls: frozenset = frozenset({"poison", "omit"})
    schedules: frozenset = frozenset({"serial", "deferred"})
    opened: list = field(default_factory=list)

    def refusals(self, sites):
        return [f"linear {s.weight_id}: k = {s.k}" for s in sites if s.k % 1024]

    def begin(self, spec):
        self.opened.append(Pass(spec))
        return self.opened[-1]


@dataclass
class Source:
    name: str = "pous:band-chain/d12/v1"
    forbids_weight_state: bool = True
    controls: frozenset = frozenset({"drop"})
    ended: list = field(default_factory=list)

    def weight(self, site, fwd):
        return ("decoded", site.weight_id, None if fwd is None else fwd.index)

    def forward_end(self, fwd):
        self.ended.append(fwd.index)


class Layer:
    weight = "plaintext"
    quant_config = "the layer's own"


def original(layer, x, bias):
    return ("stock", layer.weight, layer.quant_config, x, bias)


PEARLC = Arm("pearlc", "bf16", "pouw:pearl-c-sm120-v1-h3")


def test_a_source_that_forbids_weight_state_refuses_an_executor_that_keeps_it():
    (why,) = Stack(SITES, Executor(), Source()).refusals()
    assert "keeps weight-derived state" in why and "pous:band-chain/d12/v1 forbids it" in why
    assert Stack(SITES, Executor(keeps_weight_state=False), Source()).refusals() == []
    assert Stack(SITES, Executor()).refusals() == []


def test_a_weight_format_or_shape_the_executor_does_not_take_is_refused():
    sites = (*SITES, LinearSite(2, 2, "model.layers.0.mlp.gate_up_proj", 28672, 3584, "fp16"))
    got = Stack(sites, Executor()).refusals()
    assert "reads bf16 weights, not fp16" in got[0] and got[1] == "linear 2: k = 3584"


def test_a_control_or_schedule_nothing_implements_is_refused_by_name():
    stack = Stack(SITES, Executor(keeps_weight_state=False), Source())
    assert stack.refusals(PassSpec(controls=(Control("poison", 0xA5), Control("drop", 0.01)), schedule="deferred")) == []
    got = stack.refusals(PassSpec(controls=(Control("tamper", "tile"),), schedule="h3o"))
    assert got == ["control 'tamper': nothing in this stack implements it",
                   "schedule 'h3o': pouw:pearl-c-sm120-v1-h3 runs deferred, serial"]
    with pytest.raises(Refused, match="control 'tamper'"):
        stack.activate(Arm("tampered", "bf16", "pouw:pearl-c-sm120-v1-h3", spec=PassSpec(controls=(Control("tamper"),))))


def test_an_inactive_stack_is_vllms_own_apply_with_the_sources_weight():
    assert Stack(SITES, Executor()).apply(SITES[0], Layer(), "x", None, original) == ("stock", "plaintext", "the layer's own", "x", None)
    stack = Stack(SITES, None, Source())
    with stack.forward():
        got = stack.apply(SITES[1], Layer(), "x", "b", original)
    assert got == ("stock", ("decoded", 1, 0), "the layer's own", "x", "b")


def test_the_executor_computes_inside_a_pass_with_the_sources_weight():
    ex, src = Executor(keeps_weight_state=False), Source()
    stack = Stack(SITES, ex, src)
    stack.activate(PEARLC)
    with stack.pass_(PassSpec(index=7)) as p:
        for f in range(2):
            with stack.forward() as fwd:
                assert fwd == Forward(f, 7)
                for s in SITES:
                    stack.apply(s, Layer(), f"x{f}", None, original)
    assert [c[:3] for c in p.calls] == [(0, "x0", ("decoded", 0, 0)), (1, "x0", ("decoded", 1, 0)),
                                        (0, "x1", ("decoded", 0, 1)), (1, "x1", ("decoded", 1, 1))]
    assert stack.passes == [{"pass": 7, "units": 4}] and src.ended == [0, 1]


def test_an_active_executor_outside_a_pass_fails_closed():
    stack = Stack(SITES, Executor())
    stack.activate(PEARLC)
    with pytest.raises(RuntimeError, match="active outside a pass"):
        stack.apply(SITES[0], Layer(), "x", None, original)


def test_a_forward_that_raises_still_ends_at_the_source_and_its_pass_is_ended_unrecorded():
    ex, src = Executor(keeps_weight_state=False), Source()
    stack = Stack(SITES, ex, src)
    stack.activate(PEARLC)
    with pytest.raises(ZeroDivisionError), stack.pass_() as p, stack.forward():
        stack.apply(SITES[0], Layer(), "x", None, original)
        1 / 0
    assert src.ended == [0] and p.ended == 1 and stack.passes == []


def test_arms_switch_only_between_passes_and_name_the_stacks_executor():
    stack = Stack(SITES, Executor())
    stack.activate(PEARLC)
    with stack.pass_():
        with pytest.raises(RuntimeError, match="arms switch between passes"):
            stack.activate(None)
    stack.activate(Arm("bf16", "bf16"))
    assert stack.apply(SITES[0], Layer(), "x", None, original)[0] == "stock"
    with pytest.raises(ValueError, match="this stack's executor is pouw:pearl-c-sm120-v1-h3"):
        stack.activate(Arm("ncp", "bf16", "pouw:ncp-v2"))


def test_the_interface_imports_neither_vllm_nor_torch():
    path = Path(__file__).resolve().parents[3] / "verity_vllm" / "protocols" / "linear" / "api.py"
    mods = {a.name for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom) and n.module}
    assert not {m for m in mods if m.split(".")[0] in ("vllm", "torch", "triton", "numpy")}
