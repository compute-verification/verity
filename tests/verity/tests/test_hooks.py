"""The served run's events and requests (`verity.hooks`)."""

from __future__ import annotations

from verity import hooks as H


def test_hook_types_hold_no_tensor_type() -> None:
    store = object()
    assert {e.hook for e in H.EVENTS} == set(H.HOOKS)
    assert H.RangesDeclared(4, 16, 128, 2, (store,)).stores == (store,)
    assert H.RangeFreed((7,), step=3).ranges == (7,)
    assert H.StepBegan(3, ("in-1",), 16).inputs == ("in-1",)

    def kernel(due: H.BlockDue, original):
        return original(None)
    due = H.BlockDue("layers.0.mlp", 0, {"n": 8, "k": 4}, "int8", {"x": store, "weight": store})
    assert kernel(due, lambda inputs: ("served", inputs)) == ("served", None)
