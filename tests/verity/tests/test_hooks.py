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


def test_unmet_names_each_request_the_properties_do_not_meet() -> None:
    eager, one = H.RequireProperty("eager", True, "e"), H.RequireProperty("world", 1, "w")
    bf16, kernels = H.RequireProperty("dtype", "bf16", "d"), H.RequireProperty("kernels", "profile", "k", fixes=True)
    asked = [("pouw", eager), ("pouw", one), ("pous", bf16), ("spot", kernels), ("spot", H.ReuseOldestFreed())]
    assert H.unmet(asked, {"eager": True, "world": 1, "dtype": "bf16", "kernels": "profile"}) == []
    assert H.unmet(asked, {"eager": False, "world": 1, "dtype": "fp8"}) == [("pouw", eager), ("pous", bf16), ("spot", kernels)]
