"""`MoEFusedSharedExpertsFp8`'s router and gate steps called by themselves, with the arguments `match` gives them for a whole scope
(the synthetic `vllm.moe_forward_shared` events of `tests/engine/test_gen_shared_moe_fp8.py`).  `match` refuses a scaling factor
that is no number and a router without a bias view before it calls either step, so no scope reaches these; each step by itself
still fails as its own reads say: the router on the number, the gate on the bias it looks up."""

from __future__ import annotations

import dataclasses

import pytest

from tests.engine import test_gen_shared_moe_fp8 as S
from verity_vllm.observe.events import StepStart
from verity_vllm.observe.fold.profiles.gen_shared_moe_patterns import (
    MoEFusedSharedExpertsFp8,
)
from verity_vllm.observe.fold.resolver import Group, resolve_step
from verity_vllm.observe.fold.tree import step_events

M = 4
BIAS_AT = 6  # `bias` is the seventh parameter of the router's schema, `scoring_func` the eighth


def _step_arguments(monkeypatch, name: str) -> tuple:
    """The pattern and the arguments `match` gives its step `name` on the whole scope."""
    got: list[tuple] = []
    step = getattr(MoEFusedSharedExpertsFp8, name)

    def record(self, *args):
        got.append((self, *args))
        return step(self, *args)

    start = StepStart(0, [("r0", p) for p in range(M)], list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0, 1, 2, 3]}, 16,
                      list(range(M)), [("r0", M - 1)], [M - 1], 1, M, M, None, None, "prefill")
    with monkeypatch.context() as patched:
        patched.setattr(MoEFusedSharedExpertsFp8, name, record)
        resolved = resolve_step(step_events([start, *S.scope_events(M)]), S._ctx(start))
    assert isinstance(resolved[0], Group) and len(got) == 1
    return got[0]


@pytest.mark.parametrize("factor", [True, "2.5", None])
def test_the_router_step_alone_refuses_a_scaling_factor_that_is_no_number(monkeypatch, factor):
    pattern, node, ctx, sc, hidden, outs, hint = _step_arguments(monkeypatch, "_router")
    alone = dataclasses.replace(pattern, config={**pattern.config, "ROUTED_SCALING_FACTOR": factor})
    with pytest.raises(TypeError, match=r"^float\(\) argument must be a string or a real number, not 'NoneType'$"):
        alone._router(node, ctx, sc, hidden, outs, hint)


def test_the_gate_step_alone_looks_up_the_bias_a_router_does_not_bind_as_none(monkeypatch):
    pattern, node, ctx, sc, hidden, hint = _step_arguments(monkeypatch, "_scope_gate")
    enter = sc.router.enter
    unbound = dataclasses.replace(enter, args=list(enter.args[:BIAS_AT]), kwargs={**enter.kwargs, "scoring_func": enter.args[BIAS_AT + 1]})
    router = dataclasses.replace(sc.router, enter=unbound)
    with pytest.raises(AttributeError, match=r"^'NoneType' object has no attribute 'alloc'$"):
        pattern._scope_gate(node, ctx, dataclasses.replace(sc, router=router), hidden, hint)
