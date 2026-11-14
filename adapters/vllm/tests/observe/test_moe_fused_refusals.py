"""Two refusals of `MoEFusedExperts.match` hand on the reason their child gave: a `silu_and_mul` that does not resolve, and a
descendant that a scope pattern refuses.  Each returns an `Unsupported` of the scope with the child's reason in its detail."""

from __future__ import annotations

import dataclasses

from tests.engine import test_gen_ov_moe as plain
from tests.engine import test_gen_ov_moe_qwen3 as rank
from verity_vllm.observe.events import DispatchEnter, Event
from verity_vllm.observe.fold.resolver import Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events


def _as_float32(events: list[Event], op: str, at: int) -> list[Event]:
    """`events` with argument `at` of the one dispatcher call of `op` recorded as float32."""
    out: list[Event] = []
    changed = 0
    for e in events:
        if isinstance(e, DispatchEnter) and e.op == op:
            args = list(e.args)
            args[at] = dataclasses.replace(args[at], dtype="float32")
            e = dataclasses.replace(e, args=args)
            changed += 1
        out.append(e)
    assert changed == 1, changed
    return out


def test_a_silu_and_mul_that_does_not_resolve_refuses_the_scope_with_its_reason():
    start = plain._start(3)
    events = _as_float32(plain.moe_scope_events(3), "_C.silu_and_mul.default", 1)
    res = resolve_step(step_events([start, *events]), plain._ctx(start))
    assert len(res) == 1 and isinstance(res[0], Unsupported), res
    assert res[0].op == "vllm.moe_forward.default"
    assert res[0].reason == "silu_and_mul inside the MoE scope did not resolve as SiluMul"
    assert res[0].detail["child_reason"] == "SiluMul is bf16"
    assert res[0].detail["detail"]["dtypes"] == ["float32", "bfloat16"]


def test_a_descendant_that_a_scope_pattern_refuses_refuses_the_scope_with_its_reason():
    start = rank._start(2)
    events = _as_float32(rank.moe_scope_events(2, all_reduce="inside"), "vllm.all_reduce.default", 0)
    reduce_id = next(e.id for e in events if isinstance(e, DispatchEnter) and e.op == "vllm.all_reduce.default")
    res = resolve_step(step_events([start, *events]), rank._ctx(start))
    assert len(res) == 1 and isinstance(res[0], Unsupported), res
    assert res[0].op == "vllm.moe_forward.default"
    assert res[0].reason == "scope pattern tp.collective refused all_reduce inside moe_forward"
    detail = res[0].detail
    assert (detail["child_id"], detail["child"]) == (reduce_id, "vllm.all_reduce.default")
    assert detail["child_reason"] == "all_reduce: partial is not a row-major bf16 [M,N] view"
    assert "float32" in detail["child_detail"]["partial"]
